import re
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from peft import get_peft_model, LoraConfig, TaskType
import os
import time
from collections import Counter
import numpy as np

# -------------------------------
# Device
# -------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


# -------------------------------
# Label Cleaning
# -------------------------------
def clean_label(label):
    if isinstance(label, list):
        label = label[0] if label else None
    if not label or label in ["*", "_", "", "null", None]:
        return "_none_"
    label = re.sub(r"\[\d+\]", "", label)
    return label.strip().lower()


# -------------------------------
# Modell-Wrapper für Decoder-LMs
# -------------------------------
class LMForSequenceClassification(nn.Module):
    """
    Universeller Klassifikationskopf für Decoder-Modelle (Gemma, Mistral, LLaMmlein)
    """

    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Konfigurationsinfos ergänzen
        self.config.num_labels = num_labels
        self.config.problem_type = "single_label_classification"

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # letztes Token repräsentiert Sequenz
        hidden = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            # CrossEntropyLoss wendet implizit Softmax an
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# -------------------------------
# LoRA-Zielerkennung
# -------------------------------
def find_lora_targets(model):
    """
    Findet automatisch sinnvolle Zielmodule für LoRA
    """
    candidate_keywords = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  # Nur lineare Schichten prüfen
            for kw in candidate_keywords:
                if kw in name:
                    found.add(kw)
    if not found:
        print("Keine Standardmodule gefunden – verwende Fallback.")
        return ["classifier"]  # Minimalziel, falls kein Standardlayer
    print(f"Erkannte LoRA-Zielmodule: {sorted(list(found))}")
    return sorted(list(found))


# -------------------------------
# Hauptprogramm
# -------------------------------
if __name__ == "__main__":
    # -------------------------------
    # Dataset laden
    # -------------------------------
    dataset = load_dataset("json", data_files={
        "train": "technik_train.jsonl",
        "validation": "technik_val.jsonl",
        "test": "technik_test.jsonl"
    })


    def filter_technik(example):
        # Filtern nach Sprecher L und nicht-leeren Labels
        return example.get("speaker") == "L" and clean_label(example.get("technik")) != "_none_"


    for split in ["train", "validation", "test"]:
        dataset[split] = dataset[split].filter(filter_technik)

    # -------------------------------
    # Labels vorbereiten
    # -------------------------------
    all_labels = sorted({
        clean_label(ex.get("technik"))
        for split in ["train", "validation", "test"]
        for ex in dataset[split]
    })
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(all_labels)
    print(f"{num_labels} Technik-Labels gefunden: {all_labels}")


    def encode_labels(example):
        t_clean = clean_label(example.get("technik"))
        example["labels"] = label2id[t_clean]
        return example


    dataset = dataset.map(encode_labels)

    # ----------------------------------------------------
    # *** BASELINE BERECHNUNG (Häufigste Klasse) ***
    # ----------------------------------------------------

    # 1. Bestimme die Häufigste-Klasse-Information aus dem Trainings-Set
    # Die Labels sind bereits als IDs im Dataset gespeichert
    train_label_ids = [example["labels"] for example in dataset["train"]]
    label_counts = Counter(train_label_ids)

    # Finde die ID des häufigsten Labels
    # label_counts.most_common(1) gibt [(id, count)] zurück
    most_common_id = label_counts.most_common(1)[0][0]
    most_common_label_name = id2label[most_common_id]

    # 2. Test-Labels extrahieren (als NumPy-Array)
    test_label_ids = np.array([example["labels"] for example in dataset["test"]])

    # 3. Baseline-Vorhersagen erstellen (immer die häufigste Klasse vorhersagen)
    baseline_preds = np.full_like(test_label_ids, most_common_id)

    # 4. Metriken berechnen
    baseline_accuracy = accuracy_score(test_label_ids, baseline_preds)
    # Wichtig: Verwende 'macro' Durchschnitt und 'zero_division=0' für fairen Vergleich
    baseline_f1_macro = f1_score(test_label_ids, baseline_preds, average="macro", zero_division=0)

    # 5. Ausgabe der Baseline-Ergebnisse
    print("\n*** E R M I T T E L T E  T E C H N I K - B A S E L I N E ***")
    print(f"Basis: Häufigste Klasse im Training.")
    print(f"Häufigste Klasse: {most_common_label_name}")
    print(f"Baseline Accuracy (Test-Set): {baseline_accuracy:.4f}")
    print(f"Baseline Macro F1 (Test-Set): {baseline_f1_macro:.4f}")
    print("-" * 50)
    # ----------------------------------------------------

    # -------------------------------
    # Modellwahl
    # -------------------------------
    #model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = "google/gemma-3-270m-it"
    # model_name = "LSX-UniWue/LLaMmlein_1B"

    # -------------------------------
    # Pfad für die Speicherung erstellen
    # -------------------------------
    sanitized_model_name = re.sub(r'[/.]', '-', model_name).lower()
    output_dir_name = f"technik_model_{sanitized_model_name}"

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )


    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # -------------------------------
    # Dynamische TrainingArguments
    # -------------------------------
    if "7b" in model_name.lower():
        train_bs, grad_acc = 4, 8
        compute_dtype = torch.bfloat16
        fp16_arg = False
        bf16_arg = True
        quantize = True
    elif "1b" in model_name.lower() or "270m" in model_name.lower():  # Kleine Modelle
        train_bs, grad_acc = 8, 4
        # FIX: Setze dtype beim Laden auf float32, und lasse fp16=True die AMP-Logik steuern.
        # Dies behebt den "Attempting to unscale FP16 gradients." Fehler.
        compute_dtype = torch.float32
        fp16_arg = True
        bf16_arg = False
        quantize = False
    else:
        train_bs, grad_acc = 16, 1
        compute_dtype = torch.float32
        fp16_arg = False
        bf16_arg = False
        quantize = False

    # -------------------------------
    # Modell laden (mit oder ohne Quantisierung)
    # -------------------------------

    if quantize:
        # BitsAndBytesConfig definieren
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype
        )
        # Modell mit Quantisierung laden
        base_model = AutoModel.from_pretrained(
            model_name,
            config=config,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True
        )
    else:
        # Modell ohne Quantisierung laden
        base_model = AutoModel.from_pretrained(
            model_name,
            config=config,
            dtype=compute_dtype,  # Jetzt torch.float32 für 270m-Modell (FIx)
            device_map={"": 0},
            trust_remote_code=True
        )

    # -------------------------------
    # Hidden Size und Custom Wrapper erstellen
    # -------------------------------
    hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 4096))
    model = LMForSequenceClassification(base_model, hidden_size, num_labels)

    # Gradient Checkpointing aktivieren, wenn möglich
    if hasattr(base_model, "gradient_checkpointing_enable"):
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable()

    # -------------------------------
    # LoRA mit automatischer Zielauswahl
    # -------------------------------

    target_modules = find_lora_targets(model.base_model)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
    )

    # Sicherstellen, dass der neue Klassifikator trainierbar ist, auch wenn LoRA aktiv ist
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True

    model = get_peft_model(model, lora_config)

    # Zusätzliche Konvertierung für korrekten dtype der Classification Head, wenn nötig
    # Für float32/fp16-Training sollte der Classifier ebenfalls float32 sein.
    model.classifier.to(compute_dtype)

    model.print_trainable_parameters()
    print(
        f"Model config: hidden_size={getattr(model.config, 'hidden_size', 'N/A')}, num_labels={model.config.num_labels}")


    # -------------------------------
    # Custom Trainer
    # -------------------------------
    class TechnikTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Labels werden als Long-Tensor benötigt
            # HINWEIS: .to(device) ist unnötig, da Trainer dies im Training übernimmt.
            labels = inputs.pop("labels").long() #.to(device)
            outputs = model(**inputs)
            logits = outputs["logits"]
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            return (loss, outputs) if return_outputs else loss


    # -------------------------------
    # Metrics
    # -------------------------------
    def compute_metrics(pred):
        logits, labels = pred
        preds = logits.argmax(axis=-1)
        # Labels müssen für sklearn int sein
        labels = labels.astype(int)
        return {
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "f1": f1_score(labels, preds, average="macro", zero_division=0),
            "accuracy": accuracy_score(labels, preds)
        }


    training_args = TrainingArguments(
        output_dir="./technik_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=train_bs,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        fp16=fp16_arg,
        bf16=bf16_arg,
        gradient_accumulation_steps=grad_acc,
        logging_steps=20,
        report_to="none"
    )

    # Zeitmessung starten
    start_time = time.time()

    trainer = TechnikTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,  # processing_class durch tokenizer ersetzt
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Modell-Speicherung mit dynamischen Pfad
    model.save_pretrained(output_dir_name)
    tokenizer.save_pretrained(output_dir_name)

    # -------------------------------
    # Metriken auf Test-Set ausgeben
    # -------------------------------
    print("\n*** Evaluierung des Test-Datensatzes (LLM-Modell) ***")
    results = trainer.evaluate(dataset["test"])

    # Ausgabe der gewünschten Metriken
    print(f"Precision: {results.get('eval_precision', 'N/A'):.4f}")
    print(f"Recall: {results.get('eval_recall', 'N/A'):.4f}")
    print(f"F1-Score: {results.get('eval_f1', 'N/A'):.4f}")
    print(f"Accuracy: {results.get('eval_accuracy', 'N/A'):.4f}")

    # Zeitmessung beenden und Ergebnis ausgeben
    end_time = time.time()
    total_time_seconds = end_time - start_time
    print(f"Ausführungsdauer: {total_time_seconds:.4f}")