import torch
import copy
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn
from multiprocessing import freeze_support
from transformers.modeling_outputs import SequenceClassifierOutput
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Einstellungen
# -------------------------------
MODEL_NAME = "bert-base-german-cased"
TRAIN_FILES = {
    "train": "phase_train.jsonl",
    "validation": "phase_val.jsonl",
    "test": "phase_test.jsonl"
}
OUTPUT_DIR = "./phase_results_encoder"
MAX_LEN = 128
WINDOW_STRIDE = 64
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 5

MODEL_REQUIRED_COLUMNS = ["input_ids", "attention_mask", "position_ratio", "labels"]


# ----------------------------------------------------
# Dataset vorbereiten
# ----------------------------------------------------
def prepare_dataset(model_name, train_files):
    dataset = load_dataset("json", data_files=train_files)
    all_labels = sorted({ex["phase"] for split in ["train", "validation", "test"] for ex in dataset[split]})
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(all_labels)

    def encode_labels(example):
        example["labels"] = label2id[example["phase"]]
        return example

    dataset = dataset.map(encode_labels)

    def add_position_ratio(example, idx, split_len):
        example["position_ratio"] = float(idx / max(split_len - 1, 1))
        return example

    for split_name in ["train", "validation", "test"]:
        split_len = len(dataset[split_name])
        dataset[split_name] = dataset[split_name].map(
            lambda ex, idx: add_position_ratio(ex, idx, split_len),
            with_indices=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(example):
        tokenized_output = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LEN,
            stride=WINDOW_STRIDE,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            padding="max_length"
        )
        if tokenized_output.get("overflow_to_sample_mapping", None) is not None:
            sample_index = tokenized_output["overflow_to_sample_mapping"]
            tokenized_output["labels"] = [example["labels"][i] for i in sample_index]
            tokenized_output["position_ratio"] = [example["position_ratio"][i] for i in sample_index]
            del tokenized_output["overflow_to_sample_mapping"]

            if "token_type_ids" in tokenized_output:
                del tokenized_output["token_type_ids"]

        return tokenized_output

    remove_cols = ["text", "phase"]
    for split in dataset.keys():
        for col in dataset[split].column_names:
            if col not in MODEL_REQUIRED_COLUMNS and col not in remove_cols:
                remove_cols.append(col)

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=remove_cols)
    dataset.set_format(type="torch", columns=[c for c in MODEL_REQUIRED_COLUMNS if c in dataset["train"].column_names])

    return dataset, tokenizer, num_labels, id2label, label2id


# ----------------------------------------------------
# Modell + LoRA + position_ratio Kopf
# ----------------------------------------------------
def build_model(model_name, num_labels):
    base_model = AutoModel.from_pretrained(model_name)
    hidden_size = base_model.config.hidden_size

    target_modules = list(set([name.split('.')[-1] for name, module in base_model.named_modules() if
                               isinstance(module, nn.Linear) and name.split('.')[-1] in ["query", "key", "value",
                                                                                         "q_proj", "k_proj",
                                                                                         "v_proj"]]))

    print("LoRA target modules:", target_modules)

    if target_modules:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1, inference_mode=False,
            target_modules=target_modules
        )
        base_model = get_peft_model(base_model, lora_config)

    class PositionAwareClassifier(nn.Module):
        def __init__(self, base_model, hidden_size, num_labels):
            super().__init__()
            self.base_model = base_model
            self.num_labels = num_labels
            self.classifier = nn.Linear(hidden_size, num_labels)
            self.position_dense = nn.Linear(1, num_labels)
            self.position_scale = nn.Parameter(torch.ones(1) * 2.0)

        def forward(self, input_ids=None, attention_mask=None, position_ratio=None, labels=None):

            # Argumente nur für das Basismodell zusammenstellen
            base_model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            # Syntax korrigiert: {k: v for k, v in ...}
            base_model_inputs = {k: v for k, v in base_model_inputs.items() if v is not None}

            outputs = self.base_model(**base_model_inputs)

            hidden_state = outputs.last_hidden_state
            # Encoder-spezifisch: Verwende das CLS-Token (Index 0)
            cls_state = hidden_state[:, 0, :]

            logits = self.classifier(cls_state)
            if position_ratio is not None:
                position_ratio = position_ratio.to(logits.device).unsqueeze(-1).float()
                logits += self.position_dense(position_ratio) * self.position_scale

            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

            return SequenceClassifierOutput(loss=loss, logits=logits)

    return PositionAwareClassifier(base_model, hidden_size, num_labels)


# ----------------------------------------------------
# Custom Trainer
# ----------------------------------------------------
class PhaseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        labels = inputs.pop("labels").long().to(next(model.parameters()).device)
        position_ratio = inputs.pop("position_ratio", None)

        # Unerwünschte Tokenizer-Outputs entfernen
        inputs.pop("token_type_ids", None)

        # Explizites model_inputs-Dictionary erstellen, um sicherzustellen, dass keine
        # unerwarteten Argumente des Trainers (wie z.B. num_items_in_batch) weitergeleitet werden.
        model_inputs = {
            "input_ids": inputs.pop("input_ids", None),
            "attention_mask": inputs.pop("attention_mask", None),
            "labels": labels,  # Muss an unser benutzerdefiniertes Modell übergeben werden
            "position_ratio": position_ratio  # Muss an unser benutzerdefiniertes Modell übergeben werden
        }

        # Entferne None-Werte
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}

        # Aufruf des PositionAwareClassifier
        outputs = model(**model_inputs)

        return (outputs.loss, outputs) if return_outputs else outputs.loss


# ----------------------------------------------------
# Metrics
# ----------------------------------------------------
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


# ----------------------------------------------------
# Main
# ----------------------------------------------------
if __name__ == "__main__":
    freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset + Tokenizer
    dataset, tokenizer, num_labels, id2label, label2id = prepare_dataset(MODEL_NAME, TRAIN_FILES)
    model = build_model(MODEL_NAME, num_labels)

    model.base_model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, eval_strategy="epoch", save_strategy="epoch", learning_rate=3e-4,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS, weight_decay=0.01, load_best_model_at_end=True,

        # Optimierungen für RTX 4090
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        # ---------------------------------

        logging_steps=20,
        report_to="none", remove_unused_columns=False
    )

    trainer = PhaseTrainer(model=model, args=training_args, train_dataset=dataset["train"],
                           eval_dataset=dataset["validation"], processing_class=tokenizer, compute_metrics=compute_metrics)

    print("Starte Training...")
    trainer.train()

    model_save_dir = "phase_model_bert-base-german-cased"
    if hasattr(model.base_model, 'save_pretrained'):
        model.base_model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"Modell gespeichert in: {model_save_dir}")

    print("\n*** Evaluierung des Test-Datensatzes ***")
    results = trainer.evaluate(dataset["test"])

    # Ausgabe der gewünschten Metriken
    print(f"Precision: {results.get('eval_precision', 'N/A'):.4f}")
    print(f"Recall:    {results.get('eval_recall', 'N/A'):.4f}")
    print(f"F1-Score:  {results.get('eval_f1', 'N/A'):.4f}")
    print(f"Accuracy:  {results.get('eval_accuracy', 'N/A'):.4f}")

    # Zeitmessung beenden und Ergebnis ausgeben
    end_time = time.time()
    total_time_seconds = end_time - start_time
    print(f"Ausführungsdauer: {total_time_seconds:.4f}")