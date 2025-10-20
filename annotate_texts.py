import os
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F  # Füge F hinzu, da wir es für Mean Pooling benötigen
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
import spacy
import re
from datasets import load_dataset
from typing import Optional  # Füge Optional hinzu, um Typ-Hints konsistent zu halten
import numpy as np  # Wird für Mean Pooling benötigt

# -----------------------------
# Einstellungen
# -----------------------------

# ANNAHME: Die Pfade MÜSSEN zu den tatsächlichen Speicherorten passen,
# die Ihre Trainingsskripte erzeugen (output_dir_name + Lora Adapter Pfad)
# Beispiel: Wenn Ihr Training "gemma" verwendet hat, passen Sie es an:
phase_model_base = "google/gemma-3-270m-it"  # Beispiel: Gemma wurde in Skript 1 verwendet
phase_adapter_path = "phase_model_google/gemma-3-270m-it/lora"  # Korrigiere den Pfad basierend auf Skript 1
phase_classifier_path = "phase_model_google/gemma-3-270m-it/classifier.pt"  # Korrigiere den Pfad basierend auf Skript 1

technik_model_base = "mistralai/Mistral-7B-Instruct-v0.2"  # Beispiel: Mistral wurde in Skript 2 verwendet
technik_adapter_path = "technik_model_mistralai-mistral-7b-instruct-v0-2/lora"  # Korrigiere den Pfad basierend auf Skript 2
technik_classifier_path = "technik_model_mistralai-mistral-7b-instruct-v0-2/classifier.pt"  # Korrigiere den Pfad basierend auf Skript 2

input_folder = "New_Text"
output_folder = "Annotated"
# Nutze den korrekten Device-Typ
device = "cuda" if torch.cuda.is_available() else "cpu"

# Threshold für Technik-Vorhersagen
TECHNIK_THRESHOLD = 0.6

# -----------------------------
# SpaCy für Satzsegmentierung
# -----------------------------
try:
    nlp = spacy.load("de_core_news_sm")
except:
    import spacy.cli

    spacy.cli.download("de_core_news_sm")
    nlp = spacy.load("de_core_news_sm")


# -----------------------------
# Label Cleaning
# -----------------------------
def clean_label(label):
    if isinstance(label, list):
        label = label[0] if label else ""
    if not label or label in ["*", "_", "", "null", None]:
        return "_none_"
    label = re.sub(r"\[\d+\]", "", label)
    return label.strip().lower()


# -----------------------------
# Modell-Wrapper für Phase (PositionAwareClassifier-Logik)
# -----------------------------
class PhaseInferenceModel(nn.Module):
    """
    Nachbildung des PositionAwareClassifier (Skript 1) für die Inferenz,
    unter Verwendung von Mean Pooling und Konkatenation des Position-Ratios.
    """

    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size + 1, num_labels)  # +1 für position_ratio
        self.config = base_model.config

    def _mean_pooling(self, hidden_states, attention_mask):
        """Berechnet das Mean Pooling über die Nicht-Padding-Tokens (wie in Skript 1)."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask=None, position_ratio=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_state = outputs.hidden_states[-1]

        # Mean Pooling über alle nicht-Padding-Tokens
        pooled_state = self._mean_pooling(hidden_state, attention_mask)

        # Konkatenation des Position-Ratios
        if position_ratio is not None:
            # Stelle sicher, dass position_ratio auf die korrekte Form/Dtype gebracht wird
            position_ratio_feature = position_ratio.unsqueeze(-1).to(pooled_state.dtype).to(pooled_state.device)
            combined_state = torch.cat((pooled_state, position_ratio_feature), dim=-1)
        else:
            # Fallback wie im Trainingsskript
            zeros = torch.zeros((pooled_state.size(0), 1), device=pooled_state.device, dtype=pooled_state.dtype)
            combined_state = torch.cat((pooled_state, zeros), dim=-1)

        logits = self.classifier(combined_state)

        return F.softmax(logits, dim=-1)


# -----------------------------
# Modell-Wrapper für Technik (LMForSequenceClassification-Logik)
# -----------------------------
class TechnikInferenceModel(nn.Module):
    """
    Nachbildung des LMForSequenceClassification (Skript 2) für die Inferenz.
    Verwendet das Hidden State des letzten Tokens.
    """

    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.config = base_model.config

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # letztes Token repräsentiert Sequenz (standard Vorgehen für Decoder-LMs bei Klassifikation)
        hidden = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(hidden)
        return F.softmax(logits, dim=-1)


# -----------------------------
# Modell laden (LoRA + korrekter Kopf)
# -----------------------------
def load_full_model(base_model_name, adapter_path, classifier_path, num_labels, id2label, use_position_aware: bool):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Basismodell laden (float16/bfloat16 für große Modelle auf GPU, float32 auf CPU)
    config = AutoConfig.from_pretrained(base_model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32  # Verwende float16 für Speicher-Effizienz
    base_model = AutoModel.from_pretrained(base_model_name, config=config, torch_dtype=dtype)

    hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 4096))

    if use_position_aware:
        # 2. Wrapper erstellen (Phase)
        model = PhaseInferenceModel(base_model, hidden_size, num_labels)
        # Für Phase müssen wir den Classifier-Zustand des gesamt-Moduls laden, da es mehr als nur "classifier" gibt.
        state_dict_key = 'classifier.pt'
    else:
        # 2. Wrapper erstellen (Technik)
        model = TechnikInferenceModel(base_model, hidden_size, num_labels)
        # Für Technik haben wir nur den "classifier"-Head im Zustand
        state_dict_key = 'classifier.pt'

    # 3. LoRA-Adapter laden
    try:
        model.base_model = PeftModel.from_pretrained(model.base_model, adapter_path)
        # Optional: Zusammenführen der LoRA-Gewichte für schnellere Inferenz
        model.base_model = model.base_model.merge_and_unload()
        print(f"LoRA-Adapter von {adapter_path} erfolgreich geladen und entladen.")
    except Exception as e:
        print(f"WARNUNG: Konnte LoRA-Adapter nicht laden von {adapter_path}. Ist der Pfad korrekt?")

    # 4. Klassifikationskopf laden
    try:
        classifier_state = torch.load(classifier_path, map_location=device)

        # Für das Phase-Modell (PositionAwareClassifier) laden wir den kompletten state_dict des Kopfes
        if use_position_aware:
            # Die Logik muss alle Gewichte des PhaseInferenceModel Head abdecken.
            # Im Trainingsskript wird nur `model.classifier.state_dict()` gespeichert.
            # Da das PhaseInferenceModel hier nur die Linearschicht `self.classifier` hat (ohne die position_dense/scale),
            # passen wir die Logik an die ursprüngliche Absicht des Trainingsskripts (Skript 1) an,
            # welches nur `model.classifier.state_dict()` speichert.
            model.classifier.load_state_dict(classifier_state)

        # Für das Technik-Modell (LMForSequenceClassification) laden wir den kompletten state_dict des Kopfes
        else:
            # Das Technik-Skript (Skript 2) speichert ebenfalls nur `model.classifier.state_dict()`.
            model.classifier.load_state_dict(classifier_state)

        print(f"Klassifikationskopf von {classifier_path} erfolgreich geladen.")

    except Exception as e:
        print(
            f"FEHLER: Konnte Klassifikationskopf nicht laden von {classifier_path}. Prüfe den Pfad und den Inhalt der Datei!")
        print(e)  # Zeige den Fehler

    # Stelle sicher, dass der Model-Dtype korrekt ist, auch für den Classifier
    model.to(device)
    model.eval()
    return tokenizer, model


# -----------------------------
# Phase Labels aus Training (mit _none_)
# -----------------------------
# Die Logik zum Laden der Labels aus der Trainingsdatei ist KORREKT
phase_files = {"train": "phase_train.jsonl"}
if not os.path.exists("phase_train.jsonl"):
    print("FEHLER: 'phase_train.jsonl' nicht gefunden. Passe den Pfad an.")
    exit()

phase_train = load_dataset("json", data_files=phase_files)["train"]
phase_labels = sorted({clean_label(ex["phase"]) for ex in phase_train})
phase_label2id = {l: i for i, l in enumerate(phase_labels)}
phase_id2label = {i: l for l, i in phase_label2id.items()}
phase_num_labels = len(phase_labels)

print(f"Phase Labels ({phase_num_labels}): {phase_labels}")

tokenizer_phase, model_phase = load_full_model(
    phase_model_base,
    phase_adapter_path,
    phase_classifier_path,
    num_labels=phase_num_labels,
    id2label=phase_id2label,
    use_position_aware=True  # Position Aware ist notwendig
)

# -----------------------------
# Technik Labels aus Training (ohne _none_)
# -----------------------------
# Die Logik zum Laden der Labels aus der Trainingsdatei ist KORREKT
technik_files = {"train": "technik_train.jsonl"}
if not os.path.exists("technik_train.jsonl"):
    print("FEHLER: 'technik_train.jsonl' nicht gefunden. Passe den Pfad an.")
    exit()

technik_train = load_dataset("json", data_files=technik_files)["train"]


def filter_technik(example):
    # Sprecher L und nicht-leere Labels, wie im Training
    return example.get("speaker") == "L" and clean_label(example.get("technik")) != "_none_"


technik_train_filtered = technik_train.filter(filter_technik)
technik_labels = sorted({clean_label(ex["technik"]) for ex in technik_train_filtered})

technik_label2id = {l: i for i, l in enumerate(technik_labels)}
technik_id2label = {i: l for l, i in technik_label2id.items()}
technik_num_labels = len(technik_labels)

print(f"Technik Labels ({technik_num_labels}): {technik_labels}")

tokenizer_technik, model_technik = load_full_model(
    technik_model_base,
    technik_adapter_path,
    technik_classifier_path,
    num_labels=technik_num_labels,
    id2label=technik_id2label,
    use_position_aware=False  # Standard-Klassifikator
)


# -----------------------------
# Vorhersagefunktionen (sind korrekt für die neuen Model-Klassen)
# -----------------------------
def predict_phase(text, position_ratio, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    # position_ratio muss ein 1D Tensor sein, wie er im Collator des Trainingsskripts erzeugt wurde
    ratio_tensor = torch.tensor([position_ratio], dtype=torch.float32).to(device)

    with torch.no_grad():
        # Das Modell gibt direkt die Softmax-Wahrscheinlichkeiten zurück
        probs = model(**inputs, position_ratio=ratio_tensor)

    max_idx = torch.argmax(probs, dim=-1).item()
    return model.config.id2label[max_idx]


def predict_technik(text, tokenizer, model, threshold=TECHNIK_THRESHOLD):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        # Das Modell gibt direkt die Softmax-Wahrscheinlichkeiten der positiven Klassen zurück
        probs = model(**inputs)
        max_prob, max_idx = torch.max(probs, dim=-1)

    max_prob_item = max_prob.item()

    if max_prob_item < threshold:
        return ""
    else:
        best_label = model.config.id2label[max_idx.item()]
        return best_label


# -----------------------------
# Rest des Skripts (Dateihandling und Annotation)
# -----------------------------
os.makedirs(output_folder, exist_ok=True)


def get_model_shortname(model_name: str) -> str:
    model_name = model_name.lower()
    if "gemma" in model_name:
        return "Gemma3"
    elif "mistral" in model_name:
        return "Mistral7B"
    elif "llammlein" in model_name:
        return "LLaMmlein1B"
    else:
        return "UnknownModel"


model_shortname = get_model_shortname(phase_model_base)

for file in glob.glob(os.path.join(input_folder, "*.txt")):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    doc = nlp(text)

    # ... (Satzsegmentierung und Sprechererkennung bleiben gleich) ...
    sentences_raw = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    parsed_sentences = []
    last_speaker = None

    for sent_text in sentences_raw:
        speaker = None
        clean_text = sent_text

        if sent_text.startswith("L:"):
            speaker = "L"
            clean_text = sent_text.split(":", 1)[1].strip()
        elif sent_text.startswith("E:"):
            speaker = "E"
            clean_text = sent_text.split(":", 1)[1].strip()

        if speaker:
            last_speaker = speaker
        else:
            # Wenn kein Speaker-Präfix, verwende den letzten Speaker
            speaker = last_speaker if last_speaker else "-"

        if clean_text:
            parsed_sentences.append({"text": clean_text, "speaker": speaker})

    total_sentences = len(parsed_sentences)
    annotated_sentences = []

    for i, sent_data in enumerate(parsed_sentences):
        clean_text = sent_data["text"]
        speaker = sent_data["speaker"]

        # 2. Berechnung des Position-Ratios
        position_ratio = i / max(total_sentences - 1, 1) if total_sentences > 1 else 0.0

        # 3. Vorhersagen
        phase = predict_phase(clean_text, position_ratio, tokenizer_phase, model_phase)

        technik = ""
        if speaker == "L":
            technik = predict_technik(clean_text, tokenizer_technik, model_technik, threshold=TECHNIK_THRESHOLD)

            # Sollte nicht passieren, aber zur Sicherheit
            if technik == "_none_":
                technik = ""

        annotated_sentences.append({
            "text": clean_text,
            "speaker": speaker,
            "phase": phase,
            "technik": technik
        })

    # -----------------------------
    # Ausgabe mit Modellnamen im Dateinamen
    # -----------------------------
    base_name = os.path.splitext(os.path.basename(file))[0]
    output_file = os.path.join(output_folder, f"{base_name}_annotated_{model_shortname}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in annotated_sentences:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Annotiert: {file} → {output_file}")