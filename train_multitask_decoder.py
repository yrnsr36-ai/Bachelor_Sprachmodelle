import re
import os
import time
import argparse
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput

# -------------------------------
# KONFIGURATION & SETUP
# -------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    """Setzt den Seed für NumPy und PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
torch.backends.cudnn.benchmark = True


# -------------------------------
# LABEL & MODELL-HILFSFUNKTIONEN
# -------------------------------
def clean_label(label):
    """Reinigt Labels (beide Skripte)."""
    if isinstance(label, list):
        label = label[0] if label else None
    if not label or label in ["*", "_", "", "null", None]:
        return "_none_"
    label = re.sub(r"\[\d+\]", "", label)
    return label.strip().lower()


def find_lora_targets(model):
    """Identifiziert die Zielmodule für LoRA (konsolidiert)."""
    candidate_keywords = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "query", "key", "value"
    ]
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for kw in candidate_keywords:
                if kw in name:
                    # Füge nur den Teil hinzu, der das Modul beschreibt
                    kw_stripped = name.split('.')[-1]
                    if any(c in kw_stripped for c in
                           ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "query", "key",
                            "value"]):
                        found.add(kw_stripped)
                    break

    if not found:
        print("Keine Standardmodule gefunden – verwende Fallback ['classifier'] oder ['query', 'key', 'value'].")
        return ["classifier"] if not hasattr(model, 'classifier') else ["query", "key", "value"]

        # Filtere Duplikate und gib sortiert zurück
    unique_targets = sorted(list(set([kw for kw in found if any(c in kw for c in candidate_keywords)])))
    print(f"Erkannte LoRA-Zielmodule: {unique_targets}")
    return unique_targets


# -------------------------------
# KLASSEN-DEFINITIONEN
# -------------------------------

# --- 1. Technik-Klassifikator (Ende des Tokens) ---
class LMForSequenceClassification(nn.Module):
    """Klassifikationskopf für Decoder-Modelle mit End-of-Sequence Pooling (Skript 1)."""

    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.config.num_labels = num_labels
        self.config.problem_type = "single_label_classification"

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # letztes Token repräsentiert Sequenz bei Left-Padding/Decoder-LMs
        hidden = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels.long().to(logits.device))
        return SequenceClassifierOutput(loss=loss, logits=logits)


# --- 2. Phase-Klassifikator (Mean-Pooling + Position) ---
class PositionAwareClassifier(nn.Module):
    """Klassifikationskopf mit Mean-Pooling und Position-Ratio Feature (Skript 2)."""

    def __init__(self, base_model, hidden_size, num_labels, class_weights=None):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.config = base_model.config
        self.config.num_labels = num_labels
        self.config.problem_type = "single_label_classification"


        # Bestimme Device/DType von den Basis-Parametern
        base_param = next(base_model.parameters())
        base_device = base_param.device
        base_dtype = base_param.dtype

        # Loss-Funktion mit Klassen-Gewichten
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device=base_device, dtype=base_dtype))
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Klassifikator: hidden_size + 1 (für position_ratio)
        self.classifier = nn.Linear(hidden_size + 1, num_labels).to(device=base_device, dtype=base_dtype)

    def __getattr__(self, name):
        """
        Leitet Anfragen nach 'config' oder anderen Attributen, die der Trainer/PeftModel
        erwartet, an das eingekapselte Basismodell weiter.
        """
        # Versuche, das Attribut vom Standard-Modul abzurufen
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Wenn nicht gefunden, versuche es im Basismodell
            if name == "base_model" or name == "config" or name == "dtype":
                return getattr(self.base_model, name)
            raise

    def _mean_pooling(self, hidden_states, attention_mask):
        """Berechnet das Mean Pooling über die Nicht-Padding-Tokens."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

        # train_multitask_decoder.py, Zeile ~160 (Korrektur)

        def forward(self, *args, **kwargs):

            # Extrahieren der benötigten Argumente aus kwargs
            input_ids = kwargs.pop('input_ids', None)
            attention_mask = kwargs.pop('attention_mask', None)
            labels = kwargs.pop('labels', None)
            position_ratio = kwargs.pop('position_ratio', None)  # Ihr benutzerdefiniertes Feature

            # 1. Sicherstellen, dass das Basismodell die benötigten Argumente erhält
            #    (und keine Duplikate entstehen, da wir sie aus kwargs entfernt haben)
            kwargs.pop("output_hidden_states", None)
            kwargs.pop("return_dict", None)

            # 2. Das Basismodell aufrufen (nur die zwingend erforderlichen Argumente)
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs  # Füge alle verbleibenden (hoffentlich kompatiblen) Argumente hinzu
            )

            # 3. Rest des Codes zur Klassifikation
            hidden_state = outputs.hidden_states[-1]

            # Führen Sie Mean-Pooling durch
            pooled_output = self._mean_pooling(hidden_state, attention_mask)

            # Fügen Sie das position_ratio Feature hinzu
            if position_ratio is not None:
                # Stellen Sie sicher, dass position_ratio die korrekte Form hat
                if position_ratio.dim() == 1:
                    position_ratio = position_ratio.unsqueeze(1)

                # Konkatenation (angenommen, das ist Ihre Logik)
                combined_output = torch.cat((pooled_output, position_ratio), dim=-1)
            else:
                combined_output = pooled_output

            # Klassifikations-Layer
            logits = self.classifier(combined_output)

            # Loss-Berechnung (falls labels vorhanden)
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(
                    weight=self.loss_fn.weight if hasattr(self.loss_fn, 'weight') else None).to(logits.device)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).to(logits.device))

            # Rückgabe der Ergebnisse
            return SequenceClassifierOutput(loss=loss, logits=logits)

# --- 3. Custom Data Collator für Phase-Modell ---
class CustomDataCollator(DataCollatorForLanguageModeling):
    """Fügt position_ratio und Classification-Labels dem Batch hinzu (Skript 2)."""

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Labels (IDs) extrahieren
        if "labels" in features[0]:
            labels = [f.pop("labels") for f in features]
        else:
            labels = None

        # position_ratio-Werte extrahieren
        position_ratios = [f.pop("position_ratio") for f in features]

        # Standard-Collator für tokenisierte Inputs
        batch = super().__call__(features, return_tensors=return_tensors)

        # position_ratio und Labels hinzufügen
        # Stelle sicher, dass position_ratio auf das richtige Device/DType kommt
        dtype = torch.float32  # position_ratio ist float
        batch["position_ratio"] = torch.tensor(position_ratios, dtype=dtype).to(batch["input_ids"].device)

        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long).to(batch["input_ids"].device)

        return batch


# --- 4. Custom Trainer für Technik-Modell ---
class TechnikTrainer(Trainer):
    """Trainer mit expliziter CrossEntropyLoss-Berechnung auf Long-Tensor-Labels (Skript 1)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").long().to(device)
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


# -------------------------------
# METRICS FUNKTION
# -------------------------------
def compute_metrics(pred):
    """Berechnet F1-Macro, Precision, Recall und Accuracy."""
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    labels = labels.astype(int)  # Sicherstellen, dass Labels int für sklearn sind

    return {
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds)
    }


# -------------------------------
# DATEN LADEN & VORBEREITEN
# -------------------------------

def load_and_prepare_data(task, tokenizer, max_length=128):
    """Lädt und bereitet den Datensatz für die ausgewählte Aufgabe vor."""

    # Task-spezifische Konfiguration
    if task == "technik":
        file_paths = {"train": "technik_train.jsonl", "validation": "technik_val.jsonl", "test": "technik_test.jsonl"}
        label_col = "technik"
        filter_fn = lambda ex: ex.get("speaker") == "L" and clean_label(ex.get("technik")) != "_none_"
        extra_cols = []
    elif task == "phase":
        file_paths = {"train": "phase_train.jsonl", "validation": "phase_val.jsonl", "test": "phase_test.jsonl"}
        label_col = "phase"
        filter_fn = lambda ex: clean_label(ex.get("phase")) != "_none_"
        extra_cols = ["position_ratio"]  # Die Spalte "position" wird im map-Schritt zu "position_ratio"
    else:
        # Dieser Fehler sollte nach der Entfernung von argparse nicht mehr auftreten
        raise ValueError("Unbekannte Aufgabe. Wähle 'technik' oder 'phase'.")

    # Lade Rohdaten
    dataset = load_dataset("json", data_files=file_paths)

    # Filter und sammle Labels
    for split in dataset.keys():
        dataset[split] = dataset[split].filter(filter_fn)

    all_labels = sorted({
        clean_label(ex.get(label_col))
        for split in dataset.keys()
        for ex in dataset[split]
    })
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(all_labels)

    if num_labels < 2:
        raise ValueError(f"Zu wenige Labels ({num_labels}) gefunden. Überprüfen Sie die Daten und den Filter.")

    print(f"{num_labels} {task.capitalize()}-Labels gefunden: {all_labels}")

    # Tokenisierung und Label-Encoding
    def preprocess_batch(batch):
        texts = batch["text"]
        labels_raw = batch.get(label_col, ["_none_"] * len(texts))

        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        encoded["labels"] = [label2id[clean_label(p)] for p in labels_raw]

        if task == "phase":
            # Füge 'position_ratio' hinzu, welches in den Rohdaten 'position' heißt
            encoded["position_ratio"] = batch["position"]

        return encoded

    dataset = dataset.map(preprocess_batch, batched=True)

    # Baseline Berechnung
    train_label_ids = [example["labels"] for example in dataset["train"]]
    label_counts = Counter(train_label_ids)
    most_common_id = label_counts.most_common(1)[0][0]
    most_common_label_name = id2label[most_common_id]
    test_label_ids = np.array([example["labels"] for example in dataset["test"]])
    baseline_preds = np.full_like(test_label_ids, most_common_id)

    baseline_accuracy = accuracy_score(test_label_ids, baseline_preds)
    baseline_f1_macro = f1_score(test_label_ids, baseline_preds, average="macro", zero_division=0)

    print(f"\n*** E R M I T T E L T E  {task.upper()} - B A S E L I N E ***")
    print(f"Häufigste Klasse: {most_common_label_name}")
    print(f"Baseline Accuracy (Test-Set): {baseline_accuracy:.4f}")
    print(f"Baseline Macro F1 (Test-Set): {baseline_f1_macro:.4f}")
    print("-" * 50)

    # Klassengewichte (nur für 'phase' verwendet, aber immer berechnet)
    counts_tensor = torch.tensor([label_counts.get(i, 0) for i in range(num_labels)], dtype=torch.float32)
    weights = 1.0 / (counts_tensor + 1e-6)
    class_weights = weights / weights.sum() * num_labels

    return dataset, num_labels, label2id, id2label, class_weights


# -------------------------------
# TRAININGS FUNKTIONEN
# -------------------------------

def setup_and_train(task, model_name, dataset, num_labels, class_weights=None):
    """Hauptlogik für das Einrichten und Trainieren des Modells."""

    # Task-spezifische Konfiguration (Konsolidiert)
    if "7b" in model_name.lower():
        train_bs, grad_acc = 4, 8
        quantize = True
    elif "1b" in model_name.lower() or "270m" in model_name.lower():
        train_bs, grad_acc = 8, 4
        quantize = False
    else:
        train_bs, grad_acc = 16, 1
        quantize = False

    # Standard-Hyperparameter
    learning_rate = 3e-4 if task == "technik" else 2e-4  # Angepasst nach Skripten
    num_train_epochs = 10 if task == "technik" else 15 if "gemma" in model_name.lower() or "llamlein" in model_name.lower() else 10
    lora_r = 16 if task == "technik" else 32 if "7b" in model_name.lower() else 16
    lora_alpha = 32 if task == "technik" else 64 if "7b" in model_name.lower() else 32
    lora_dropout = 0.1 if task == "technik" else 0.05 if "7b" in model_name.lower() else 0.1

    # DType-Bestimmung (identisch in beiden Skripten)
    if quantize:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        fp16_arg, bf16_arg = False, (compute_dtype == torch.bfloat16)
    else:
        compute_dtype = torch.float16 if "1b" in model_name.lower() or "270m" in model_name.lower() else torch.float32
        fp16_arg, bf16_arg = (compute_dtype == torch.float16), False

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    tokenizer.padding_side = 'left'  # Wichtig für Decoder-LMs

    model_class = AutoModel
    # 1. Basismodell laden (mit/ohne Quantisierung)
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype
        )
        # Für Phase (Skript 2) wird AutoModelForCausalLM verwendet, für Technik (Skript 1) AutoModel

        base_model = model_class.from_pretrained(
            model_name, config=config, quantization_config=bnb_config,
            device_map={"": 0}, trust_remote_code=True,
            attn_implementation='eager' if task == "phase" else None
        )
    else:

        base_model = model_class.from_pretrained(
            model_name, config=config, dtype=compute_dtype,
            device_map={"": 0}, trust_remote_code=True,
            attn_implementation='eager' if task == "phase" else None
        )

    # Enable Gradient Checkpointing
    if hasattr(base_model, "gradient_checkpointing_enable"):
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable()

    # 2. Custom Wrapper erstellen
    hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 4096))
    if task == "technik":
        model = LMForSequenceClassification(base_model, hidden_size, num_labels)
        # Stelle sicher, dass der neue Klassifikator trainierbar ist
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
    else:  # task == "phase"
        model = PositionAwareClassifier(base_model, hidden_size, num_labels, class_weights=class_weights)
        model.classifier.requires_grad_(True)
        # Bei Phase-Modell LoRA auf das base_model anwenden, welches in PositionAwareClassifier gewrappt ist
        for param in base_model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)  # LayerNorm etc. in float32 halten

    # 3. LoRA anwenden
    target_modules = find_lora_targets(base_model if task == "phase" else model.base_model)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    # 4. Finales Setup
    if task == "technik":
        # Zusätzliche Konvertierung für korrekten dtype der Classification Head
        model.classifier.to(compute_dtype)
        # Dataset-Format für Technik: input_ids, attention_mask, labels
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        # Trainer: TechnikTrainer
        Trainer_Class = TechnikTrainer
        data_collator = None  # Standard Data Collator wird verwendet
    else:  # task == "phase"
        # Stellen Sie sicher, dass der gesamte model-Wrapper den compute_dtype hat
        model.to(dtype=compute_dtype)
        # Dataset-Format für Phase: input_ids, attention_mask, labels, position_ratio
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "position_ratio"])
        # Trainer: Standard Trainer
        Trainer_Class = Trainer
        data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

    model.print_trainable_parameters()

    # 5. Training Arguments
    sanitized_model_name = re.sub(r'[/.]', '-', model_name).lower()
    output_dir_name = f"{task}_model_{sanitized_model_name}"

    training_args = TrainingArguments(
        output_dir=f"./{task}_results",
        eval_strategy="epoch",
        # Speichere für Technik, nicht für Phase (Skript 2)
        save_strategy="epoch" if task == "technik" else "no",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=train_bs,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=(task == "technik"),  # Nur für Technik
        fp16=fp16_arg,
        bf16=bf16_arg,
        gradient_accumulation_steps=grad_acc,
        logging_steps=20 if task == "technik" else 50,
        report_to="none",
        optim="adamw_torch",  # Für Phase explizit angegeben
        max_grad_norm=1.0,  # Für Phase angegeben
        remove_unused_columns=False,
    )
    print(f"Trainer BF16 Status: {bf16_arg}")

    start_time = time.time()

    # 6. Trainer initialisieren und starten
    trainer = Trainer_Class(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,

    )

    trainer.train()

    # 7. Speichern
    final_output_dir = output_dir_name + ("_optimized" if task == "phase" else "")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # 8. Test-Metriken ausgeben
    print(f"\n*** Evaluierung des Test-Datensatzes (LLM-Modell für {task.upper()}) ***")
    results = trainer.evaluate(dataset["test"])

    print(f"Precision: {results.get('eval_precision', 'N/A'):.4f}")
    print(f"Recall: {results.get('eval_recall', 'N/A'):.4f}")
    print(f"F1-Score: {results.get('eval_f1', 'N/A'):.4f}")
    print(f"Accuracy: {results.get('eval_accuracy', 'N/A'):.4f}")

    end_time = time.time()
    total_time_seconds = end_time - start_time
    print(f"Ausführungsdauer: {total_time_seconds:.4f} Sekunden")


# -------------------------------
# HAUPTPROGRAMM-LOGIK (ANGEPASST)
# -------------------------------

if __name__ == "__main__":

    # ----------------------------------------------------
    # ⚙️ KONFIGURATION HIER FESTLEGEN ⚙️
    # ----------------------------------------------------

    # Wähle die zu trainierende Aufgabe: 'technik' oder 'phase'
    TRAINING_TASK = "phase"  # Setze auf "technik" oder "phase"

    # Wähle das Basis-LLM (Key aus model_choices)
    CHOSEN_MODEL_KEY = "gemma"  # Setze auf "mistral", "gemma" oder "llamalein"

    # ----------------------------------------------------

    model_choices = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "gemma": "google/gemma-3-270m-it",
        "llamalein": "LSX-UniWue/LLaMmlein_1B"
    }

    chosen_model_name = model_choices.get(CHOSEN_MODEL_KEY)

    if chosen_model_name is None:
        raise ValueError(f"Ungültiger Modell-Key '{CHOSEN_MODEL_KEY}'. Erlaubt sind: {list(model_choices.keys())}")

    if TRAINING_TASK not in ["technik", "phase"]:
        raise ValueError(f"Ungültige Aufgabe '{TRAINING_TASK}'. Erlaubt sind: 'technik' oder 'phase'.")

    print(f"🔥 Starte Training für Task: {TRAINING_TASK.upper()} mit Modell: {chosen_model_name}")

    # Schritt 1: Tokenizer initialisieren (für prepare_data)
    tokenizer_for_setup = AutoTokenizer.from_pretrained(chosen_model_name)
    if tokenizer_for_setup.pad_token is None:
        tokenizer_for_setup.pad_token = tokenizer_for_setup.eos_token if tokenizer_for_setup.eos_token else "[PAD]"
    tokenizer_for_setup.padding_side = 'left'

    # Schritt 2: Daten laden und vorverarbeiten
    dataset, num_labels, label2id, id2label, class_weights = load_and_prepare_data(
        TRAINING_TASK,
        tokenizer_for_setup
    )

    # Schritt 3: Training starten
    # Klassengewichte nur für 'phase' an die Setup-Funktion übergeben, da nur dort verwendet
    weights_to_pass = class_weights if TRAINING_TASK == "phase" else None

    setup_and_train(
        TRAINING_TASK,
        chosen_model_name,
        dataset,
        num_labels,
        class_weights=weights_to_pass
    )