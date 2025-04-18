import os
import json
from collections import defaultdict
from pathlib import Path

# Soft disagreement config.
# Most of the labels will differ. If that difference is small, then take the higher or lower agreement, or a mean between the two.
config = {
    "tolerance": 1,              # Max distance between labels for soft agreement
    "prefer": "higher",          # Options: "higher", "lower", or "mean"
    "apply_soft_agreement": True
}

def load_unique_text_rows(filepath):
    seen_texts = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text")
            if text is not None and text not in seen_texts:
                seen_texts[text] = row
    return seen_texts

def resolve_soft_agreement(labels, config):
    unique_labels = list(set(labels))
    if len(unique_labels) == 2 and config["apply_soft_agreement"]:
        a, b = unique_labels
        if abs(a - b) <= config["tolerance"]:
            if config["prefer"] == "higher":
                return max(a, b)
            elif config["prefer"] == "lower":
                return min(a, b)
            elif config["prefer"] == "mean":
                return round((a + b) / 2)
    return None

def compare_labels_in_directory(directory):
    directory = Path(directory)
    jsonl_files = list(directory.glob("*.jsonl"))
    if len(jsonl_files) < 2:
        raise ValueError("At least two .jsonl files are required for comparison.")

    text_file_rows = defaultdict(dict)
    for file in jsonl_files:
        rows = load_unique_text_rows(file)
        for text, row in rows.items():
            text_file_rows[text][file.name] = row

    agreed = []
    disagreed = []

    for text, file_rows in text_file_rows.items():
        labels = [row["label"] for row in file_rows.values()]
        unique_labels = list(set(labels))
        first_row = next(iter(file_rows.values())).copy()

        if len(unique_labels) == 1:
            agreed.append(first_row)
        else:
            resolved = resolve_soft_agreement(labels, config)
            if resolved is not None:
                first_row["label"] = resolved
                agreed.append(first_row)
            else:
                entry = {k: v for k, v in first_row.items() if k != "label"}
                for i, (source, row) in enumerate(file_rows.items(), 1):
                    entry[f"label{i}"] = row["label"]
                    entry[f"source{i}"] = source
                disagreed.append(entry)

    with open(directory / "agreed.jsonl", 'w', encoding='utf-8') as f:
        for entry in agreed:
            f.write(json.dumps(entry) + "\n")

    with open(directory / "disagreed.jsonl", 'w', encoding='utf-8') as f:
        for entry in disagreed:
            f.write(json.dumps(entry) + "\n")

    print(f"Agreed: {len(agreed)}, Disagreed: {len(disagreed)}")
    print(f"Config: {config}")

# Example usage
if __name__ == "__main__":
    compare_labels_in_directory("./dataset_v2_labelled")
