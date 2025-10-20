import os
import glob
import json
from collections import Counter
from sklearn.model_selection import train_test_split
import re

# -----------------------------
# Einstellungen
# -----------------------------
input_folder = "INCEpTION Data"

phase_train_file = "phase_train.jsonl"
phase_val_file = "phase_val.jsonl"
phase_test_file = "phase_test.jsonl"

technik_train_file = "technik_train.jsonl"
technik_val_file = "technik_val.jsonl"
technik_test_file = "technik_test.jsonl"

train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1


# -----------------------------
# Label Cleaning
# -----------------------------
def clean_label(label):
    """Entfernt Marker wie *[17] oder [17], gibt Kleinbuchstaben zurück"""
    # Diese Funktion wird nur im Konvertierungsskript verwendet
    # um die rohen TSV-Labels zu säubern.
    if not label or label.startswith("*") or label == "_":
        return None
    label = re.sub(r"\[\d+\]", "", label)
    return label.strip().lower()


# -----------------------------
# TSV-Dateien einlesen
# -----------------------------
def parse_webanno_tsv(file_path):
    sentences = []
    current_tokens, current_phases, current_techniken = [], [], []
    current_speaker = None
    last_speaker = None

    phase_keywords = [
        "beschlussphase", "rahmen", "begrüßung", "inhalt", "zeitlich", "sonstiges",
        "informationsphase", "smalltalk", "wertschätzende reflexion",
        "terminvereinbarung", "verabschiedung", "argumentationsphase"
    ]

    # Weil in den alten Datensätzen der Leitfaden andere Phasen beeinhaltet
    label_mapping = {
        "zusammenfassung": "wertschätzende reflexion",
        "vorwärtsgewandte strukturierung": "strukturieren",
        "rückwärtsgewandte strukturierung": "strukturieren",
        "wiederholung mit eigenen worten": "paraphrasieren",
        "zusammenfassende wiederholung": "paraphrasieren"
    }

    with open(file_path, "r", encoding="utf-8-sig") as f:
        all_lines = [line.rstrip("\n") for line in f]

    total_lines = len(all_lines)

    for idx, line in enumerate(all_lines):
        line_strip = line.strip()

        # Satzgrenze oder Kommentar
        if not line_strip or line_strip.startswith("#"):
            if current_tokens:
                tokens_cleaned = [
                    t for t in current_tokens
                    if t.lower() not in ["glz"] and t not in ["(", ")"]
                ]
                if tokens_cleaned and tokens_cleaned[0] in [":", "."]:
                    tokens_cleaned = tokens_cleaned[1:]

                text = " ".join(tokens_cleaned).strip()
                text = re.sub(r"\(\s*glz\s*\.\s*\)", "", text, flags=re.IGNORECASE)
                text = re.sub(r"^[:\.]\s*", "", text)

                speaker = current_speaker if current_speaker else last_speaker
                position = idx / total_lines  # relative Position im Dokument

                sentences.append({
                    "text": text,
                    "phase": list({p for p in current_phases if p}),  # Liste der Phase-Labels (kann leer sein!)
                    "technik": list({t for t in current_techniken if t}),
                    "speaker": speaker,
                    "position": position
                })

                current_tokens, current_phases, current_techniken = [], [], []
                if current_speaker:
                    last_speaker = current_speaker
                current_speaker = None
            continue

        parts = line_strip.split("\t")
        if len(parts) < 3:
            continue

        token = None
        for col in parts[2:]:
            col = col.strip()
            if not col or col.startswith("*") or re.match(r"\[\d+\]", col):
                continue
            if re.match(r"\d+-\d+", col):
                continue
            token = col
            break
        if not token:
            continue

        # Sprecher erkennen (L,T = Lehrer, E,P = Eltern)
        if token.lower() in ["l", "e", "t", "p"]:
            current_speaker = "E" if token.lower() in ["e", "p"] else "L"
            continue

        current_tokens.append(token)

        for ann in parts[3:]:
            if ann and ann != "_":
                label_clean = clean_label(ann)
                if not label_clean:
                    continue

                if label_clean in label_mapping:
                    label_clean = label_mapping[label_clean]

                if "sonstiges" in label_clean:
                    # Hier wird die Phase basierend auf der Position im Gespräch genauer aufgeschlüsselt
                    if idx < 25:
                        current_phases.append("anfangsphase_sonstiges")
                    elif idx >= total_lines - 25:
                        current_phases.append("abschlussphase_sonstiges")
                    else:
                        current_phases.append("sonstiges")
                elif any(keyword in label_clean for keyword in phase_keywords):
                    current_phases.append(label_clean)
                else:
                    current_techniken.append(label_clean)

    # Letzter Satz
    if current_tokens:
        tokens_cleaned = [
            t for t in current_tokens
            if t.lower() not in ["glz"] and t not in ["(", ")"]
        ]
        if tokens_cleaned and tokens_cleaned[0] in [":", "."]:
            tokens_cleaned = tokens_cleaned[1:]

        text = " ".join(tokens_cleaned).strip()
        text = re.sub(r"\(\s*glz\s*\.\s*\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^[:\.]\s*", "", text)

        speaker = current_speaker if current_speaker else last_speaker
        position = total_lines / total_lines  # letzter Satz → Position = 1
        sentences.append({
            "text": text,
            "phase": list({p for p in current_phases if p}),
            "technik": list({t for t in current_techniken if t}),
            "speaker": speaker,
            "position": position
        })

    return sentences


# -----------------------------
# JSONL Funktionen
# -----------------------------
def to_phase_jsonl(data, filename):
    """Phase Single-Label (String) - Fehlende Phasen erhalten '_none_'"""
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            # Wenn eine Phase annotiert wurde, nehmen Sie die erste
            if item["phase"]:
                phase_label = item["phase"][0]
            # Wenn KEINE Phase annotiert wurde, weisen Sie "_none_" zu
            else:
                phase_label = "_none_"

            # Sätze mit phase = "_none_" überspringen
            if phase_label == "_none_":
                continue

            # NUR Sätze schreiben, die nicht komplett leer sind
            if not item["text"].strip():
                continue

            f.write(json.dumps({
                "text": item["text"],
                "phase": phase_label,
                "speaker": item["speaker"],
                "position": item["position"]
            }, ensure_ascii=False) + "\n")


def to_technik_jsonl(data, filename):
    """
    Technik Single-Label (String).
    Enthält jetzt ALLE Lehrer-Sätze. Fehlende Technik erhalten einen leeren String ("").
    """
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            # Nur Lehrer-Sätze ('L')
            if item["speaker"] != "L":
                continue

            if not item["text"].strip():
                continue

            # Bestimme das Technik-Label: Erste Technik, oder '' wenn keine annotiert
            if item["technik"]:
                tech_label = item["technik"][0].strip().lower()
            else:
                # <-- ANPASSUNG: Zuweisung eines leeren Strings für nicht annotierte Lehrer-Sätze
                tech_label = ""

            f.write(json.dumps({
                "text": item["text"],
                "technik": tech_label,
                "speaker": item["speaker"],
                "position": item["position"]
            }, ensure_ascii=False) + "\n")


def to_multitask_jsonl(data, filename):
    """
    Multitask: Phase + Technik in einer Zeile.
    Phase: Single-Label (ohne _none_).
    Technik: Single-Label für 'L' (ohne Label: ""), sonst leer.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            # Phase Label bestimmen (Single-Label, _none_ wenn keine)
            phase_label = item["phase"][0] if item["phase"] else "_none_"

            # Sätze mit phase = "_none_" überspringen (Phase-Task-Filter)
            if phase_label == "_none_":
                continue

            # Technik-Logik: Nur für Sprecher 'L'
            tech_label = ""  # Standard: Kein Label für Nicht-Lehrer

            if item["speaker"] == "L":
                if item["technik"]:
                    tech_label = item["technik"][0].strip().lower()
                else:
                    # <-- ANPASSUNG: Zuweisung eines leeren Strings für nicht annotierte Lehrer-Sätze
                    tech_label = ""

            if not item["text"].strip():
                continue

            f.write(json.dumps({
                "text": item["text"],
                "phase": phase_label,
                "technik": tech_label,
                "speaker": item["speaker"],
                "position": item["position"]
            }, ensure_ascii=False) + "\n")


# -----------------------------
# Hauptausführung
# -----------------------------
all_sentences = []
print("--- Starte TSV-Einlesevorgang ---")
for file in glob.glob(os.path.join(input_folder, "*.tsv")):
    sents = parse_webanno_tsv(file)
    print(f"Eingelesen: {file}: {len(sents)} Sätze")
    all_sentences.extend(sents)

# -----------------------------
# Train/Val/Test Split
# -----------------------------
print(f"Gesamtzahl aller Sätze: {len(all_sentences)}")
train_val, test = train_test_split(all_sentences, test_size=test_ratio, random_state=42)
train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

# -----------------------------
# JSONL-Dateien schreiben
# -----------------------------
print("\n--- Schreibe JSONL-Dateien ---")
to_phase_jsonl(train, phase_train_file)
to_phase_jsonl(val, phase_val_file)
to_phase_jsonl(test, phase_test_file)

to_technik_jsonl(train, technik_train_file)
to_technik_jsonl(val, technik_val_file)
to_technik_jsonl(test, technik_test_file)

to_multitask_jsonl(train, "multitask_train.jsonl")
to_multitask_jsonl(val, "multitask_val.jsonl")
to_multitask_jsonl(test, "multitask_test.jsonl")

# -----------------------------
# Statistiken
# -----------------------------
phases_with_none = Counter()
for s in all_sentences:
    label = s["phase"][0] if s["phase"] else "_none_"
    phases_with_none[label] += 1

techniken = Counter(t for s in all_sentences for t in s["technik"] if t)

print("\n--- Statistik ---")
print(f"Anzahl aller Sätze: {len(all_sentences)}")
print(f"Phase-Labels: {len(phases_with_none)} → {dict(phases_with_none)}")
print(f"️Technik-Labels (nur annotierte): {len(techniken)} → {dict(techniken)}")

