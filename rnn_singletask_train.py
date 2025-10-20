import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import json
import os
import glob
import time
import re
import numpy as np


# ======================================================
# Dataset-Klasse
# ======================================================
class SentenceDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, label_encoder, max_len=50):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.label_encoder = label_encoder
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = text.split()
        x = [self.word_to_idx.get(t, self.word_to_idx['<UNK>']) for t in tokens]
        x = x + [0] * (self.max_len - len(x))
        x = x[:self.max_len]
        # Label in numerischen Wert umwandeln
        y = self.label_encoder.transform([label])[0]
        return torch.tensor(x), torch.tensor(y)


# ======================================================
# BiLSTM-Klassifikator
# ======================================================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Bidirektional bedeutet hidden_dim * 2 in der Ausgabe
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, (h, _) = self.lstm(emb)
        # Konkateniere den letzten Forward- und Backward-Hidden State
        hidden = torch.cat((h[-2], h[-1]), dim=1)
        logits = self.fc(hidden)
        return logits


# ======================================================
# Hilfsfunktionen
# ======================================================
def load_jsonl(file_path, label_key="phase"):
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            texts.append(entry["text"])
            labels.append(entry[label_key])
    return texts, labels


def build_vocab(texts):
    all_tokens = [tok for sent in texts for tok in sent.split()]
    word_to_idx = {w: i + 1 for i, w in enumerate(set(all_tokens))}
    word_to_idx["<PAD>"] = 0
    word_to_idx["<UNK>"] = len(word_to_idx)
    return word_to_idx


def calculate_baseline(labels, task_name):
    """Berechnet die Baseline-Genauigkeit (Mehrheitsklasse)."""
    counts = Counter(labels)
    majority_class, count = counts.most_common(1)[0]
    total = len(labels)
    baseline_accuracy = count / total
    print(f"--- Baseline ({task_name}) ---")
    print(f"  Gesamt-Samples: {total}")
    print(f"  Mehrheitsklasse: '{majority_class}' ({count} Instanzen)")
    print(f"  Baseline-Accuracy: {baseline_accuracy:.4f}")
    print("-" * (20 + len(task_name)))
    return baseline_accuracy


def calculate_metrics(model, data_loader, label_encoder):
    """Evaluiert das Modell und berechnet F1, Accuracy, Precision und Recall."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    # Berechne Metriken
    accuracy = accuracy_score(all_labels, all_preds)
    # Average='weighted' ist gut für unausgewogene Datensätze (wie Technik)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train_model(train_file, val_file, label_key="phase",
                embedding_dim=100, hidden_dim=128, max_len=50,
                batch_size=16, epochs=5):
    train_texts, train_labels = load_jsonl(train_file, label_key)
    val_texts, val_labels = load_jsonl(val_file, label_key)

    # 1. BASELINE BERECHNUNG HINZUFÜGEN
    calculate_baseline(train_labels, label_key.capitalize())

    word_to_idx = build_vocab(train_texts)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    num_labels = len(label_encoder.classes_)

    train_dataset = SentenceDataset(train_texts, train_labels, word_to_idx, label_encoder, max_len)
    val_dataset = SentenceDataset(val_texts, val_labels, word_to_idx, label_encoder, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(vocab_size=len(word_to_idx),
                             embedding_dim=embedding_dim,
                             hidden_dim=hidden_dim,
                             output_dim=num_labels).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch + 1}/{epochs} (Task: {label_key.capitalize()}), Loss: {total_loss / len(train_loader):.4f}")
    end_time = time.time()
    print(f"Training für {label_key.capitalize()} abgeschlossen in {end_time - start_time:.2f} Sekunden")

    # 2. METRIKEN BERECHNEN
    val_metrics = calculate_metrics(model, val_loader, label_encoder)

    return model, word_to_idx, label_encoder, val_metrics


# ======================================================
# Satz-Segmentierung
# ======================================================
def split_into_sentences(text):
    sentence_endings = re.compile(r'([.!?])')
    parts = sentence_endings.split(text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i].strip() + parts[i + 1]
        if sentence:
            sentences.append(sentence.strip())
    if len(parts) % 2 != 0 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return sentences


# ======================================================
# Vorhersage / Annotation
# ======================================================
def predict_from_transcript(phase_model, phase_word2idx, phase_enc,
                            tech_model, tech_word2idx, tech_enc,
                            infile, outfile,
                            max_len=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    results = []

    start_time = time.time()
    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Speaker extrahieren
            if line.startswith("E:"):
                speaker = "E"
                text = line[2:].strip()
            elif line.startswith("L:"):
                speaker = "L"
                text = line[2:].strip()
            else:
                speaker = ""
                text = line

            # Text in Sätze splitten
            sentences = split_into_sentences(text)

            for sent in sentences:
                tokens = sent.split()

                # Phase Vorhersage
                x_phase = [phase_word2idx.get(t, phase_word2idx["<UNK>"]) for t in tokens]
                x_phase = x_phase + [0] * (max_len - len(x_phase))
                x_phase = torch.tensor(x_phase[:max_len]).unsqueeze(0).to(device)
                phase_model.eval()
                with torch.no_grad():
                    logits = phase_model(x_phase)
                    pred_phase = torch.argmax(logits, dim=1).item()
                    phase_label = phase_enc.inverse_transform([pred_phase])[0]

                # Technik nur für Lehrer
                tech_label = ""
                if speaker == "L":
                    x_tech = [tech_word2idx.get(t, tech_word2idx["<UNK>"]) for t in tokens]
                    x_tech = x_tech + [0] * (max_len - len(x_tech))
                    x_tech = torch.tensor(x_tech[:max_len]).unsqueeze(0).to(device)
                    tech_model.eval()
                    with torch.no_grad():
                        logits = tech_model(x_tech)
                        pred_tech = torch.argmax(logits, dim=1).item()
                        tech_label = tech_enc.inverse_transform([pred_tech])[0]
                        # Die 'O'-Klasse (Other/Keine Technik) soll im Output leer sein
                        if tech_label.lower() == "o":
                            tech_label = ""

                results.append({
                    "text": sent,
                    "speaker": speaker,
                    "phase": phase_label,
                    "technik": tech_label
                })

    # Ergebnisse speichern
    with open(outfile, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    end_time = time.time()
    print(f"Datei {infile} annotiert und gespeichert in {outfile}")
    print(f"Annotation abgeschlossen in {end_time - start_time:.2f} Sekunden")


# ======================================================
# Hauptprogramm
# ======================================================
if __name__ == "__main__":
    # Modelle trainieren
    phase_model, phase_word2idx, phase_enc, phase_metrics = train_model(
        "phase_train.jsonl", "phase_val.jsonl", label_key="phase"
    )

    print("\n" + "=" * 50)
    print("--- ENDERGEBNISSE PHASE KLASSIFIKATION (VALIDIERUNG) ---")
    print(f"F1 Score:    {phase_metrics['f1']:.4f}")
    print(f"Accuracy:    {phase_metrics['accuracy']:.4f}")
    print(f"Precision:   {phase_metrics['precision']:.4f}")
    print(f"Recall:      {phase_metrics['recall']:.4f}")
    print("=" * 50 + "\n")

    tech_model, tech_word2idx, tech_enc, tech_metrics = train_model(
        "technik_train.jsonl", "technik_val.jsonl", label_key="technik"
    )

    print("\n" + "=" * 50)
    print("--- ENDERGEBNISSE TECHNIK KLASSIFIKATION (VALIDIERUNG) ---")
    print(f"F1 Score:    {tech_metrics['f1']:.4f}")
    print(f"Accuracy:    {tech_metrics['accuracy']:.4f}")
    print(f"Precision:   {tech_metrics['precision']:.4f}")
    print(f"Recall:      {tech_metrics['recall']:.4f}")
    print("=" * 50 + "\n")

    # Alle .txt-Dateien im Ordner New_Text annotieren
    # (Dieser Teil setzt voraus, dass die Ordnerstruktur existiert)
    print("\n" + "=" * 50)
    print("STARTE ANNOTATION NEUER TEXTDATEIEN")
    print("=" * 50)
    for infile in glob.glob("New_Text/*.txt"):
        filename = os.path.basename(infile).replace(".txt", "")
        outfile = os.path.join("Annotated", f"{filename}_RNN.jsonl")
        predict_from_transcript(
            phase_model, phase_word2idx, phase_enc,
            tech_model, tech_word2idx, tech_enc,
            infile, outfile
        )
