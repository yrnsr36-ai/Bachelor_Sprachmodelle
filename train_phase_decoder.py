import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
# NEU: Importiere confusion_matrix und classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, \
    classification_report
from peft import get_peft_model, LoraConfig, TaskType
import time
from collections import Counter
import numpy as np
import random
from typing import Optional

# -------------------------------
# Device & Seed
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
torch.backends.cudnn.benchmark = True


# -------------------------------
# Custom Loss Function: Focal Loss
# -------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss für unausgewogene Klassifikationsaufgaben.
    Reduziert den Gewichtungsbeitrag von gut klassifizierten Beispielen.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        # Berechne Cross Entropy Loss
        logpt = F.log_softmax(input, dim=-1)
        pt = torch.exp(logpt)

        # Führt die Klassen-spezifische Gewichtung durch (Alpha)
        if self.alpha is not None:
            # Stelle sicher, dass alpha auf dem korrekten Device/DType ist
            if not isinstance(self.alpha, torch.Tensor):
                # Wenn alpha nur ein Tensor mit Gewichten ist (Class Weights)
                self.alpha = self.alpha.to(input.device, dtype=input.dtype)

            # Maske für das Extrahieren von pt für die korrekten Klassen
            alpha_t = self.alpha.gather(0, target)
            logpt_for_target = logpt.gather(1, target.unsqueeze(-1)).squeeze(-1)  # Logit für die Zielklasse
            logpt = logpt_for_target * alpha_t  # Wende alpha an

        else:
            logpt = logpt.gather(1, target.unsqueeze(-1)).squeeze(-1)

        # Die Focal Loss Modifikation: (1 - pt)**gamma
        loss = -1 * (1 - pt.gather(1, target.unsqueeze(-1)).squeeze(-1)) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# -------------------------------
# Data Augmentation (Random Word Swap)
# -------------------------------
def random_word_swap_augmentation(text: str, swap_rate: float = 0.1) -> str:
    """
    Führt Data Augmentation durch: Tauscht zufällig Wörter im Text (max. 10% der Wörter).
    """
    words = text.split()
    if len(words) < 2:
        return text

    num_swaps = min(max(1, int(len(words) * swap_rate)), len(words) // 2)

    new_words = list(words)
    for _ in range(num_swaps):
        # Wähle zwei zufällige, unterschiedliche Indizes
        idx1, idx2 = random.sample(range(len(words)), 2)

        # Tausche die Wörter
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

    return " ".join(new_words)


# -------------------------------
# Label Cleaning
# -------------------------------
def clean_label(label):
    """Reinigt Labels und stellt sicher, dass Multi-Label-Annotationen (falls vorhanden) zu einem einzigen Label werden."""
    if isinstance(label, list):
        label = label[0] if label else None
    if not label or label in ["*", "_", "", "null", None]:
        return "_none_"
    label = re.sub(r"\[\d+\]", "", label)
    return label.strip().lower()


# -------------------------------
# Position-Aware Klassifikationsmodell
# -------------------------------
class PositionAwareClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels, loss_type="cross_entropy", class_weights=None,
                 focal_gamma=2.0, focal_alpha=None, compute_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.compute_dtype = compute_dtype if compute_dtype is not None else torch.float32
        base_device = next(base_model.parameters()).device

        # Definiere die Loss-Funktion
        if loss_type == "focal_loss":
            print(f"Verwende Focal Loss (Gamma={focal_gamma})")
            # FocalLoss nutzt class_weights über den alpha-Parameter
            self.loss_fn = FocalLoss(
                gamma=focal_gamma,
                alpha=class_weights.to(device=base_device,
                                       dtype=self.compute_dtype) if class_weights is not None else None,
                reduction='mean'
            )
        else:
            print("Verwende gewichtete Cross-Entropy Loss")
            if class_weights is not None:
                self.loss_fn = nn.CrossEntropyLoss(
                    weight=class_weights.to(device=base_device, dtype=self.compute_dtype))
            else:
                self.loss_fn = nn.CrossEntropyLoss()

        # Klassifikator nimmt hidden_size + 1 (für position_ratio)
        self.classifier = nn.Linear(hidden_size + 1, num_labels).to(device=base_device, dtype=self.compute_dtype)

    # Hilfsfunktion für Mean Pooling
    def _mean_pooling(self, hidden_states, attention_mask):
        """Berechnet das Mean Pooling über die Nicht-Padding-Tokens."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, position_ratio=None, labels=None,
                **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_state = outputs.hidden_states[-1]

        # Mean Pooling über alle nicht-Padding-Tokens
        pooled_state = self._mean_pooling(hidden_state, attention_mask)

        # Concatenation des Position-Ratios mit dem gepoolten State
        if position_ratio is not None:
            position_ratio_feature = position_ratio.unsqueeze(-1).to(pooled_state.dtype).to(pooled_state.device)
            combined_state = torch.cat((pooled_state, position_ratio_feature), dim=-1)
        else:
            # Fallback, wenn position_ratio fehlt (sollte nicht passieren)
            zeros = torch.zeros((pooled_state.size(0), 1), device=pooled_state.device, dtype=pooled_state.dtype)
            combined_state = torch.cat((pooled_state, zeros), dim=-1)

        logits = self.classifier(combined_state)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.long())

        return {"loss": loss, "logits": logits}


# -------------------------------
# Custom Data Collator
# -------------------------------
class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Stellt sicher, dass die 'position_ratio' und die Classification-Labels korrekt
    als 1D-Tensor an den Trainer übergeben werden.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, features, return_tensors=None):
        # 1. Standard-Collating für input_ids, attention_mask und labels
        # Extrahieren und Entfernen von labels und position_ratio für das nachfolgende Token-Padding
        labels = [feature.pop("labels") for feature in features]

        # FIX: Extrahieren des position_ratio als Liste von Skalaren
        position_ratios = []
        for feature in features:
            ratio = feature.pop("position_ratio")  # pop es heraus, um es nicht zu überschreiben

            # KORREKTUR für RuntimeError:
            if isinstance(ratio, torch.Tensor):
                # Wir nehmen das erste Element des flachen Tensors und konvertieren es zum Skalar.
                ratio = ratio.flatten()[0].item()

            position_ratios.append(ratio)

        # 2. Token-Padding durch die Basisklasse (oder manuell)
        batch = super().__call__(features, return_tensors)

        # 3. Hinzufügen der Labels und position_ratio zum Batch

        # Labels als LongTensor hinzufügen (für die Klassifikation)
        batch["labels"] = torch.tensor(labels, dtype=torch.long).to(batch["input_ids"].device)

        # Zeile 224: Hinzufügen der position_ratios zum Batch
        batch["position_ratio"] = torch.tensor(position_ratios, dtype=torch.float32).to(batch["input_ids"].device)

        return batch


# -------------------------------
# LoRA Zielmodule (Unverändert)
# -------------------------------
def find_lora_targets(model):
    """Identifiziert die Zielmodule für LoRA basierend auf gängigen LLM-Architekturen."""
    candidate_keywords = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                          "project_in", "project_out", "query", "key", "value"]
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for kw in candidate_keywords:
                if kw in name:
                    if "q_proj" in name: found.add("q_proj")
                    if "v_proj" in name: found.add("v_proj")
                    if "k_proj" in name: found.add("k_proj")
                    if "o_proj" in name: found.add("o_proj")
                    if "gate_proj" in name: found.add("gate_proj")
                    if "up_proj" in name: found.add("up_proj")
                    if "down_proj" in name: found.add("down_proj")
    if not found:
        print("Keine Standardmodule gefunden – verwende Fallback 'query, key, value'.")
        return ["query", "key", "value"]
    print(f"Erkannte LoRA-Zielmodule: {sorted(list(found))}")
    return sorted(list(found))


# -------------------------------
# Dataset Vorbereitung (mit Augmentation)
# -------------------------------
def prepare_dataset(file_paths, tokenizer, max_length=128, num_proc=1, augment_train=False):
    """
    Lädt und verarbeitet den Datensatz, entfernt '_none_' und berechnet Label IDs.
    Erweitert den Trainingsdatensatz optional mit Augmentation.
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenizer.padding_side = 'left'

    dataset = load_dataset("json", data_files=file_paths)

    # Filtern aller '_none_' Instanzen
    print(f"Ursprüngliche Größe des Trainingssets: {len(dataset['train'])}")
    for split_name in dataset.keys():
        dataset[split_name] = dataset[split_name].filter(
            lambda ex: clean_label(ex.get("phase")) != "_none_",
            num_proc=num_proc
        )
    print(f"Größe des Trainingssets nach Entfernen von '_none_': {len(dataset['train'])}")

    # Labels nur aus den gefilterten Daten sammeln
    all_labels_set = {clean_label(ex.get("phase")) for split in dataset.keys() for ex in dataset[split]}
    all_labels = sorted(list(all_labels_set))

    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(all_labels)

    print(f"{num_labels} Labels gefunden: {all_labels}")

    # Stellen Sie sicher, dass num_labels mindestens 2 ist
    if num_labels < 2:
        raise ValueError(
            f"Zu wenige relevante Labels ({num_labels}) gefunden. Überprüfen Sie die Daten und den Filter.")

    # --- DATEN-AUGMENTATION (NUR FÜR TRAININGSSET) ---
    if augment_train and "train" in dataset:
        print(">>> Wende Data Augmentation (Random Word Swap) auf Trainingsdaten an...")

        # Funktion zum Duplizieren und Augmentieren
        def augment_example(example):
            augmented_text = random_word_swap_augmentation(example["text"])

            # Gebe das Original und die augmentierte Version zurück
            # Flatten wird die Listen in den Spalten zu einzelnen Beispielen machen.
            return {
                "text": [example["text"], augmented_text],
                "phase": [example["phase"], example["phase"]],
                "position": [example["position"], example["position"]],
            }

        original_train_len = len(dataset["train"])

        # Dupliziere den Datensatz und wende Augmentation an
        dataset["train"] = dataset["train"].map(
            augment_example,
            batched=False,
            # Das Entfernen der Spalten ist notwendig, da map Listen zurückgibt
            remove_columns=dataset["train"].column_names,
            num_proc=num_proc
        ).flatten()  # Führt die Listen in den Spalten zusammen

        print(
            f"Trainingsdatensatz von {original_train_len} auf {len(dataset['train'])} Beispiele erweitert (1x Duplizierung).")

    def preprocess_batch(batch):
        texts = batch["text"]
        labels = batch.get("phase", [None] * len(texts))  # Hier ist None, da wir zuvor gefiltert haben
        position_ratios = batch["position"]

        encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        # Hier werden die Labels in Integer-IDs umgewandelt
        encoded["labels"] = [label2id[clean_label(p)] for p in labels]
        encoded["position_ratio"] = position_ratios
        return encoded

    for split_name in dataset.keys():
        dataset[split_name] = dataset[split_name].map(
            preprocess_batch,
            batched=True,
            num_proc=num_proc
        )

    # Hier werden die Spalten für das Training gesetzt, aber erst *nach* der Baseline-Berechnung

    return dataset, label2id, id2label, num_labels


# -------------------------------
# Modell laden + LoRA
# -------------------------------
def load_model_and_tokenizer(model_name, num_labels, loss_type, class_weights, custom_config):
    """Lädt das Basemodell, wendet LoRA an und initialisiert den Klassifikator mit der gewählten Loss-Funktion."""
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    compute_dtype = torch.float32
    optim_type = "adamw_torch"
    train_bs, grad_acc = 8, 4

    if custom_config.get("quantize", False):
        bnb_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        print(f">>> Lade 4-Bit Quantisierung (BitsAndBytesConfig). Verwende {bnb_compute_dtype}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bnb_compute_dtype
        )
        base_model = AutoModelForCausalLM.from_pretrained(model_name, config=config, quantization_config=bnb_config,
                                                          device_map={"": 0}, attn_implementation='eager',
                                                          trust_remote_code=True)
        compute_dtype = bnb_compute_dtype
        optim_type = "paged_adamw_32bit"
        train_bs, grad_acc = 4, 8
    else:
        print(">>> Lade in voller Präzision (float32)")
        base_model = AutoModelForCausalLM.from_pretrained(model_name, config=config, dtype=torch.float32,
                                                          device_map={"": 0}, attn_implementation='eager')
        compute_dtype = torch.float32

    if hasattr(base_model, "gradient_checkpointing_enable"):
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable()

    target_modules = find_lora_targets(base_model)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=custom_config.get("lora_r", 16),
        lora_alpha=custom_config.get("lora_alpha", 32),
        lora_dropout=custom_config.get("lora_dropout", 0.1),
        target_modules=target_modules
    )

    for param in base_model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model_peft = get_peft_model(base_model, lora_config)
    hidden_size = config.hidden_size

    # Instanziiere den benutzerdefinierten Klassifikator mit Loss-Parametern
    model = PositionAwareClassifier(
        model_peft,
        hidden_size,
        num_labels,
        loss_type=loss_type,
        class_weights=class_weights,
        focal_gamma=custom_config.get("focal_gamma", 2.0),
        focal_alpha=custom_config.get("focal_alpha", None),  # Wird ignoriert, da wir class_weights verwenden
        compute_dtype=compute_dtype
    )

    model.classifier.requires_grad_(True)
    model.to(dtype=compute_dtype)
    return model, tokenizer, train_bs, grad_acc, compute_dtype, optim_type


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    model_configs = {
        "gemma": {
            "name": "google/gemma-3-270m-it",
            "quantize": False,
            "epochs": 15,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lr": 3e-5,
            "loss_type": "focal_loss",  # 'cross_entropy' oder 'focal_loss'
            "focal_gamma": 2.0,  # Gamma-Wert für Focal Loss
            "augment_train": True  # Data Augmentation aktivieren
        },
        "mistral": {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "quantize": True,
            "epochs": 10,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "lr": 2e-4,
            "loss_type": "cross_entropy",
            "focal_gamma": 2.0,
            "augment_train": False
        },
        "llamalein": {
            "name": "LSX-UniWue/LLaMmlein_1B",
            "quantize": False,
            "epochs": 15,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lr": 5e-5,
            "loss_type": "cross_entropy",
            "focal_gamma": 2.0,
            "augment_train": False
        }
    }

    # WICHTIG: Hier können Sie das Modell und die Loss-Funktion wählen
    chosen_model_key = "gemma"
    current_config = model_configs.get(chosen_model_key)
    if not current_config:
        raise ValueError(f"Ungültiger Modellschlüssel: {chosen_model_key}")

    model_name = current_config["name"]
    num_train_epochs = current_config["epochs"]
    learning_rate = current_config["lr"]
    loss_type = current_config["loss_type"]
    augment_train = current_config["augment_train"]

    file_paths = {"train": "phase_train.jsonl", "validation": "phase_val.jsonl", "test": "phase_test.jsonl"}
    sanitized_model_name = re.sub(r'[/.]', '', model_name).lower()
    output_dir_name = f"phase_model_{sanitized_model_name}_enhanced"

    # 1. Datensatzvorbereitung
    tokenizer_for_setup = AutoTokenizer.from_pretrained(model_name)
    if tokenizer_for_setup.pad_token is None:
        tokenizer_for_setup.pad_token = tokenizer_for_setup.eos_token if tokenizer_for_setup.eos_token else "[PAD]"
    tokenizer_for_setup.padding_side = 'left'

    # Übergabe des Augmentation-Flags
    dataset, label2id, id2label, num_labels = prepare_dataset(
        file_paths,
        tokenizer_for_setup,
        augment_train=augment_train
    )

    # -------------------------------
    # Baseline Berechnung (Daten sind hier noch Python-Ints!)
    # -------------------------------
    # train_label_ids enthält die Label-IDs (Python-Integers)
    train_label_ids = [example["labels"] for example in dataset["train"]]
    label_counts = Counter(train_label_ids)

    # Sicherstellen, dass alle Labels in der Zählung sind
    full_counts = [label_counts.get(i, 0) for i in range(num_labels)]

    most_common_id = label_counts.most_common(1)[0][0]
    most_common_label_name = id2label[most_common_id]

    # KORRIGIERTE ZEILE: Entfernt .item(), da example["labels"] ein Python-Int ist.
    test_label_ids = np.array([example["labels"] for example in dataset["test"]])
    baseline_preds = np.full_like(test_label_ids, most_common_id)

    baseline_accuracy = accuracy_score(test_label_ids, baseline_preds)
    baseline_f1_macro = f1_score(test_label_ids, baseline_preds, average="macro", zero_division=0)

    print("\n*** E R M I T T E L T E   P H A S E - B A S E L I N E ***")
    print(f"Häufigste Klasse: {most_common_label_name}")
    print(f"Baseline Accuracy (Test-Set): {baseline_accuracy:.4f}")
    print(f"Baseline Macro F1 (Test-Set): {baseline_f1_macro:.4f}")
    print("-" * 50)

    # -------------------------------
    # Class Weights berechnen
    # -------------------------------
    counts_tensor = torch.tensor(full_counts, dtype=torch.float32)

    # Class Weights basierend auf Effective Number of Samples (ENS)
    beta = 0.999
    effective_num = 1.0 - np.power(beta, counts_tensor.numpy())
    effective_num[effective_num == 0] = 1e-8
    weights_np = (1.0 - beta) / effective_num

    # Normalisieren (damit die Summe der Gewichte num_classes ergibt)
    weights = torch.tensor(weights_np, dtype=torch.float32)
    weights = weights / weights.sum() * num_labels
    class_weights = weights.to(device)

    print("\nBerechnete Klassen-Gewichte (Index entspricht Label ID):")
    for i, (weight, count) in enumerate(zip(class_weights.tolist(), counts_tensor.tolist())):
        print(f"  ID {i} ({id2label.get(i, 'unbekannt')}): Gewicht={weight:.4f}, Count={int(count)}")
    print("-" * 50)

    # -------------------------------
    # Dataset auf PyTorch umstellen (Muss VOR dem Modell-Laden passieren)
    # -------------------------------
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "position_ratio"])

    # Modell laden und initialisieren
    model, tokenizer, train_bs, grad_acc, compute_dtype, optim_type = load_model_and_tokenizer(
        model_name, num_labels, loss_type, class_weights, current_config
    )


    # Metrik-Funktion erstellen (Unverändert)
    def wrapped_compute_metrics(eval_pred):
        """Metrik-Funktion, die dynamisch num_labels verwendet."""
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        relevant_labels = list(range(num_labels))

        return {
            "eval_precision": precision_score(labels, preds, average="macro", labels=relevant_labels, zero_division=0),
            "eval_recall": recall_score(labels, preds, average="macro", labels=relevant_labels, zero_division=0),
            "eval_f1": f1_score(labels, preds, average="macro", labels=relevant_labels, zero_division=0),
            "eval_accuracy": accuracy_score(labels, preds)
        }


    # Training Arguments
    bf16_status = current_config.get("quantize", False) and compute_dtype == torch.bfloat16

    training_args = TrainingArguments(
        output_dir=f"./phase_results_{chosen_model_key}",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_bs,
        gradient_accumulation_steps=grad_acc,
        fp16=False,
        bf16=bf16_status,
        logging_steps=50,
        optim=optim_type,  # Verwende den optim_type aus der Modell-Ladefunktion
        report_to="none",
        fp16_full_eval=False,
        num_train_epochs=num_train_epochs,
        max_grad_norm=1.0,
        weight_decay=0.01
    )

    print(f"Trainer BF16 Status: {bf16_status}")

    start_time = time.time()

    # Trainer initialisieren und Training starten
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=wrapped_compute_metrics,
        data_collator=CustomDataCollator(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # Modell-Speicherung
    model.base_model.save_pretrained(output_dir_name)
    torch.save(model.classifier.state_dict(), os.path.join(output_dir_name, "classifier.pt"))
    tokenizer.save_pretrained(output_dir_name)

    # Metriken auf Test-Set ausgeben
    print("\n*** Evaluierung des Test-Datensatzes (LLM-Modell) ***")

    # 1. Prediction durchführen
    predictions = trainer.predict(dataset["test"])
    preds = predictions.predictions.argmax(axis=-1)
    true_labels = predictions.label_ids
    label_names = [id2label[i] for i in range(num_labels)]

    # 2. Detaillierter Klassifikationsbericht (um die F1-Score-Schwächen zu identifizieren)
    print("\n*** Detaillierter Klassifikationsbericht (Per-Klasse Metriken) ***")
    report = classification_report(true_labels, preds, target_names=label_names, zero_division=0)
    print(report)

    # 3. Konfusionsmatrix
    print("\n*** Konfusionsmatrix (Zeilen=Wahr, Spalten=Vorhersage) ***")
    cm = confusion_matrix(true_labels, preds)
    print(f"Labels (Index): {list(id2label.keys())}")
    print(f"Labels (Name): {label_names}")
    print(cm)
    print("-" * 50)

    # Optional: Ausgabe der Makro-Metriken aus dem predict-Result (sind dieselben wie im report)
    # results = trainer.evaluate(dataset["test"])
    # print(f"\nPrecision: {results['eval_precision']:.4f}")
    # print(f"Recall: {results['eval_recall']:.4f}")
    # print(f"F1-Score: {results['eval_f1']:.4f}")
    # print(f"Accuracy: {results['eval_accuracy']:.4f}")

    end_time = time.time()
    total_time_seconds = end_time - start_time
    print(f"Ausführungsdauer: {total_time_seconds:.4f}")
