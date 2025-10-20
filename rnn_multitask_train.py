import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import Counter
import json
import os
import glob
import time
import re
import numpy as np


# ======================================================
# Dataset-Klasse für Multi-Task
# ======================================================
class MultiTaskDataset(Dataset):
    def __init__(self, texts, phase_labels, tech_labels, word_to_idx, phase_enc, tech_enc, max_len=50):
        self.texts = texts
        self.phase_labels = phase_labels
        self.tech_labels = tech_labels
        self.word_to_idx = word_to_idx
        self.phase_enc = phase_enc
        self.tech_enc = tech_enc
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        phase_label = self.phase_labels[idx]
        tech_label = self.tech_labels[idx]

        tokens = text.split()
        x = [self.word_to_idx.get(t, self.word_to_idx['<UNK>']) for t in tokens]
        x = x + [0] * (self.max_len - len(x))
        x = x[:self.max_len]

        y_phase = self.phase_enc.transform([phase_label])[0]
        y_tech = self.tech_enc.transform([tech_label])[0]

        return torch.tensor(x), torch.tensor(y_phase), torch.tensor(y_tech)


# ======================================================
# Multi-Task BiLSTM (mit vorinitialisierten Embeddings & Dropout)
# ======================================================
class MultiTaskBiLSTM(nn.Module):
    def __init__(self, weights_matrix, embedding_dim, hidden_dim, phase_classes, tech_classes, dropout_rate=0.5):
        super(MultiTaskBiLSTM, self).__init__()
        # Embedding Layer initialisiert mit vortrainierten Vektoren
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.tensor(weights_matrix, dtype=torch.float),
            freeze=False  # Erlaubt Fine-Tuning der Embeddings während des Trainings
        )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        # Separater Layer für Phase
        self.fc_phase = nn.Linear(hidden_dim * 2, phase_classes)
        # Separater Layer für Technik
        self.fc_tech = nn.Linear(hidden_dim * 2, tech_classes)

    def forward(self, x):
        emb = self.embedding(x)
        # Gemeinsamer LSTM-Encoder
        _, (h, _) = self.lstm(emb)
        hidden = torch.cat((h[-2], h[-1]), dim=1)
        hidden = self.dropout(hidden)
        # Task-spezifische Ausgaben
        phase_logits = self.fc_phase(hidden)
        tech_logits = self.fc_tech(hidden)
        return phase_logits, tech_logits


# ======================================================
# Hilfsfunktionen
# ======================================================
def load_jsonl_multitask(file_path):
    texts, phase_labels, tech_labels, speakers = [], [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            texts.append(entry["text"])
            phase_labels.append(entry["phase"])
            speakers.append(entry.get("speaker", ""))

            # Technik Label: 'O' für Eltern oder wenn nicht annotiert
            if entry.get("speaker", "") == "E":
                tech_labels.append("O")
            else:
                tech_labels.append(entry.get("technik", "O"))
    return texts, phase_labels, tech_labels, speakers


def build_vocab(texts):
    all_tokens = [tok for sent in texts for tok in sent.split()]
    word_to_idx = {w: i + 1 for i, w in enumerate(set(all_tokens))}
    word_to_idx["<PAD>"] = 0
    word_to_idx["<UNK>"] = len(word_to_idx)
    return word_to_idx


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


def load_pretrained_embeddings(word_to_idx, embedding_dim):
    """
    Simuliert das Laden vortrainierter Embeddings (z.B. Word2Vec/fastText).
    Erstellt eine Matrix, die die Vektoren für alle Wörter im Vokabular enthält.
    In einem realen Szenario würde hier eine große Datei eingelesen werden.
    """
    vocab_size = len(word_to_idx)
    # Erstellt zufällige Vektoren (Simulation)
    weights_matrix = np.random.uniform(-0.5, 0.5, size=(vocab_size, embedding_dim))

    # Der Index 0 (<PAD>) sollte auf Nullvektor gesetzt werden
    weights_matrix[word_to_idx["<PAD>"]] = np.zeros(embedding_dim)

    print(f"  [Embedding] Simulierte Embedding-Matrix der Größe {vocab_size}x{embedding_dim} erstellt.")
    return weights_matrix


# ======================================================
# Metrik-Funktionen
# ======================================================
def calculate_baseline_multitask(phase_labels, tech_labels):
    """Berechnet die Baseline-Genauigkeit (Mehrheitsklasse) für beide Tasks."""
    print("\n" + "=" * 50)
    print("--- BASELINES (TRAININGS-DATENSATZ) ---")

    # Phase Baseline
    phase_counts = Counter(phase_labels)
    majority_phase, phase_count = phase_counts.most_common(1)[0]
    total_samples = len(phase_labels)
    phase_baseline_accuracy = phase_count / total_samples
    print(f"  [Phase] Gesamt-Samples: {total_samples}")
    print(f"  [Phase] Mehrheitsklasse: '{majority_phase}' ({phase_count} Instanzen)")
    print(f"  [Phase] Baseline-Accuracy: {phase_baseline_accuracy:.4f}")

    # Technik Baseline
    tech_counts = Counter(tech_labels)
    majority_tech, tech_count = tech_counts.most_common(1)[0]
    tech_baseline_accuracy = tech_count / total_samples
    print(f"  [Technik] Gesamt-Samples: {total_samples}")
    print(f"  [Technik] Mehrheitsklasse: '{majority_tech}' ({tech_count} Instanzen)")
    print(f"  [Technik] Baseline-Accuracy: {tech_baseline_accuracy:.4f}")

    print("=" * 50)


def calculate_metrics_multitask(model, data_loader):
    """Evaluiert das Modell und berechnet Metriken für beide Tasks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_phase_preds, all_phase_labels = [], []
    all_tech_preds, all_tech_labels = [], []

    with torch.no_grad():
        for x_batch, y_phase, y_tech in data_loader:
            x_batch, y_phase, y_tech = x_batch.to(device), y_phase.to(device), y_tech.to(device)

            phase_logits, tech_logits = model(x_batch)

            phase_preds = torch.argmax(phase_logits, dim=1).cpu().numpy()
            tech_preds = torch.argmax(tech_logits, dim=1).cpu().numpy()

            all_phase_preds.extend(phase_preds)
            all_phase_labels.extend(y_phase.cpu().numpy())

            all_tech_preds.extend(tech_preds)
            all_tech_labels.extend(y_tech.cpu().numpy())

    # Metrikberechnung für PHASE (Multi-Class)
    metrics_phase = {
        "accuracy": accuracy_score(all_phase_labels, all_phase_preds),
        "precision": precision_score(all_phase_labels, all_phase_preds, average='weighted', zero_division=0),
        "recall": recall_score(all_phase_labels, all_phase_preds, average='weighted', zero_division=0),
        "f1": f1_score(all_phase_labels, all_phase_preds, average='weighted', zero_division=0)
    }

    # Metrikberechnung für TECHNIK (Multi-Class, inklusive 'O' Klasse)
    metrics_tech = {
        "accuracy": accuracy_score(all_tech_labels, all_tech_preds),
        "precision": precision_score(all_tech_labels, all_tech_preds, average='weighted', zero_division=0),
        "recall": recall_score(all_tech_labels, all_tech_preds, average='weighted', zero_division=0),
        "f1": f1_score(all_tech_labels, all_tech_preds, average='weighted', zero_division=0)
    }

    return metrics_phase, metrics_tech


# ======================================================
# Multi-Task Training angepasst für "O"-Label
# ======================================================
def train_multitask(train_file, val_file, embedding_dim=100, hidden_dim=128, max_len=50, batch_size=16, epochs=10,
                    dropout_rate=0.5):
    train_texts, train_phase, train_tech, train_speakers = load_jsonl_multitask(train_file)
    val_texts, val_phase, val_tech, val_speakers = load_jsonl_multitask(val_file)

    # BASELINE BERECHNUNG
    calculate_baseline_multitask(train_phase, train_tech)

    # Vokabular
    word_to_idx = build_vocab(train_texts)

    # VORINITIALISIERUNG DER EMBEDDINGS
    weights_matrix = load_pretrained_embeddings(word_to_idx, embedding_dim)

    # LabelEncoder Phase
    phase_enc = LabelEncoder()
    phase_enc.fit(train_phase)

    # Technik LabelEncoder nur für Lehrer + "O"
    train_texts_tech = [t for t, s in zip(train_texts, train_speakers) if s == "L"]
    train_tech_tech = [tech for tech, s in zip(train_tech, train_speakers) if s == "L"]
    # Sicherstellen, dass "O" im Encoder ist, da es in load_jsonl_multitask hinzugefügt wird
    unique_tech_labels = list(set(train_tech + ["O"]))
    tech_enc = LabelEncoder()
    tech_enc.fit(unique_tech_labels)

    # Datasets
    train_dataset = MultiTaskDataset(train_texts, train_phase, train_tech, word_to_idx, phase_enc, tech_enc)
    val_dataset = MultiTaskDataset(val_texts, val_phase, val_tech, word_to_idx, phase_enc, tech_enc)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modell initialisiert mit vorinitialisierter Matrix
    model = MultiTaskBiLSTM(weights_matrix, embedding_dim, hidden_dim, len(phase_enc.classes_), len(tech_enc.classes_),
                            dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_phase, y_tech in train_loader:
            x_batch, y_phase, y_tech = x_batch.to(device), y_phase.to(device), y_tech.to(device)
            optimizer.zero_grad()
            phase_logits, tech_logits = model(x_batch)

            # Loss für Phase und Technik summieren
            loss_phase = criterion(phase_logits, y_phase)
            loss_tech = criterion(tech_logits, y_tech)
            loss = loss_phase + loss_tech

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
    end_time = time.time()
    print(f"Multi-Task Training abgeschlossen in {end_time - start_time:.2f} Sekunden")

    # METRIKEN BERECHNEN
    metrics_phase, metrics_tech = calculate_metrics_multitask(model, val_loader)

    return model, word_to_idx, phase_enc, tech_enc, metrics_phase, metrics_tech


# ======================================================
# Annotation / Vorhersage
# ======================================================
def annotate_multitask(model, word_to_idx, phase_enc, tech_enc, infile, outfile, max_len=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    results = []

    start_time = time.time()
    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("E:"):
                speaker = "E"
                text = line[2:].strip()
            elif line.startswith("L:"):
                speaker = "L"
                text = line[2:].strip()
            else:
                speaker = ""
                text = line

            sentences = split_into_sentences(text)
            for sent in sentences:
                tokens = sent.split()
                x = [word_to_idx.get(t, word_to_idx["<UNK>"]) for t in tokens]
                x = x + [0] * (max_len - len(x))
                x_tensor = torch.tensor(x[:max_len]).unsqueeze(0).to(device)

                model.eval()
                with torch.no_grad():
                    phase_logits, tech_logits = model(x_tensor)

                    # Phase Vorhersage
                    phase_pred = phase_enc.inverse_transform([torch.argmax(phase_logits, dim=1).item()])[0]

                    # Technik Vorhersage (nur für Lehrer)
                    if speaker == "L":
                        tech_pred_raw = tech_enc.inverse_transform([torch.argmax(tech_logits, dim=1).item()])[0]
                        # 'O' = Keine Technik -> Leerer String im Output
                        tech_pred = "" if tech_pred_raw.lower() == "o" else tech_pred_raw
                    else:
                        tech_pred = ""  # Technik ist für Schüler/Eltern irrelevant

                results.append({
                    "text": sent,
                    "speaker": speaker,
                    "phase": phase_pred,
                    "technik": tech_pred
                })

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
    # Multi-Task Training mit vorinitialisierten Embeddings, erhöhter Epochenzahl und Dropout
    model, word_to_idx, phase_enc, tech_enc, phase_metrics, tech_metrics = train_multitask(
        "multitask_train.jsonl",
        "multitask_val.jsonl",
        epochs=15,  # Erhöht auf 15, um die Vorteile des geringeren Dropouts besser zu nutzen
        dropout_rate=0.3
    )

    # Ausgabe der Ergebnisse
    print("\n" + "=" * 50)
    print("--- ENDERGEBNISSE PHASE KLASSIFIKATION (VALIDIERUNG) ---")
    print(f"F1 Score:    {phase_metrics['f1']:.4f}")
    print(f"Accuracy:    {phase_metrics['accuracy']:.4f}")
    print(f"Precision:   {phase_metrics['precision']:.4f}")
    print(f"Recall:      {phase_metrics['recall']:.4f}")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("--- ENDERGEBNISSE TECHNIK KLASSIFIKATION (VALIDIERUNG) ---")
    print(f"F1 Score:    {tech_metrics['f1']:.4f}")
    print(f"Accuracy:    {tech_metrics['accuracy']:.4f}")
    print(f"Precision:   {tech_metrics['precision']:.4f}")
    print(f"Recall:      {tech_metrics['recall']:.4f}")
    print("=" * 50 + "\n")

    # Annotation für alle txt-Dateien
    input_folder = "New_Text"
    output_folder = "Annotated"
    os.makedirs(output_folder, exist_ok=True)

    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
    if not txt_files:
        print(
            f"Keine .txt-Dateien im Ordner {input_folder} gefunden! (Bitte erstellen Sie den Ordner und fügen Sie Testdaten hinzu)")
    else:
        print("\n" + "=" * 50)
        print("STARTE ANNOTATION NEUER TEXTDATEIEN")
        print("=" * 50)
        for infile in txt_files:
            filename = os.path.basename(infile).replace(".txt", "")
            outfile = os.path.join(output_folder, f"{filename}_RNN_multitask.jsonl")
            annotate_multitask(model, word_to_idx, phase_enc, tech_enc, infile, outfile)
