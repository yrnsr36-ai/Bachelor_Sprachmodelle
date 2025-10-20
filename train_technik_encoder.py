import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from peft import get_peft_model, LoraConfig, TaskType
import os
import time

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

if __name__ == "__main__":
    # -------------------------------
    # Dataset laden
    # -------------------------------
    dataset = load_dataset("json", data_files={
        "train": "technik_train.jsonl",
        "validation": "technik_val.jsonl",
        "test": "technik_test.jsonl"
    })

    # -------------------------------
    # Nur Lehrer-Sätze mit echter Technik behalten (_none_ entfernen)
    # -------------------------------

    def filter_technik(example):
        return example.get("speaker") == "L" and clean_label(example.get("technik")) != "_none_"


    for split in ["train", "validation", "test"]:
        dataset[split] = dataset[split].filter(filter_technik)

    # -------------------------------
    # Alle Labels sammeln
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

    # -------------------------------
    # Labels encodieren
    # -------------------------------
    def encode_labels(example):
        t_clean = clean_label(example.get("technik"))
        example["labels"] = label2id[t_clean]
        return example

    dataset = dataset.map(encode_labels)

    # -------------------------------
    # Tokenizer
    # -------------------------------
    model_name = "bert-base-german-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # -------------------------------
    # Pfad für die Speicherung erstellen
    # -------------------------------
    sanitized_model_name = re.sub(r'[/.]', '-', model_name).lower()
    output_dir_name = f"technik_model_{sanitized_model_name}"

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
    # Modell + LoRA
    # -------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    # -------------------------------
    # Custom Trainer
    # -------------------------------
    class TechnikTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels").long().to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # -------------------------------
    # Metrics
    # -------------------------------
    def compute_metrics(pred):
        logits, labels = pred
        preds = logits.argmax(axis=-1)
        labels = labels.astype(int)
        return {
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "f1": f1_score(labels, preds, average="macro", zero_division=0),
            "accuracy": accuracy_score(labels, preds)
        }

    # -------------------------------
    # TrainingArguments
    # -------------------------------
    training_args = TrainingArguments(
        output_dir="./technik_results_bert-base-german-cased",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        fp16=True,
        dataloader_num_workers=0,
        gradient_accumulation_steps=1,
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
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Modell-Speicherung mit dynamischen Pfad
    model.save_pretrained(output_dir_name)
    tokenizer.save_pretrained(output_dir_name)

    results = trainer.evaluate(dataset["test"])

    print(f"Precision: {results.get('eval_precision', 'N/A'):.4f}")
    print(f"Recall:    {results.get('eval_recall', 'N/A'):.4f}")
    print(f"F1-Score:  {results.get('eval_f1', 'N/A'):.4f}")
    print(f"Accuracy:  {results.get('eval_accuracy', 'N/A'):.4f}")

    # Zeitmessung beenden und Ergebnis ausgeben
    end_time = time.time()
    total_time_seconds = end_time - start_time
    print(f"Ausführungsdauer: {total_time_seconds:.4f}")