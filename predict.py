"""
Classify a new comment across all 6 toxicity labels using the trained
BiLSTM model and validation-tuned per-label thresholds.

Usage:
    python predict.py "Some comment text here"
    python predict.py --interactive
"""

import argparse
import json

import torch

from data_utils import LABEL_COLS, encode, load_vocab
from model import ToxicMultiLabelClassifier


def load_model():
    with open("artifacts/bilstm_config.json") as f:
        config = json.load(f)
    vocab = load_vocab("artifacts/bilstm_vocab.json")

    model = ToxicMultiLabelClassifier(
        vocab_size=len(vocab),
        num_labels=config["num_labels"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pad_idx=vocab["<pad>"],
    )
    model.load_state_dict(torch.load("artifacts/bilstm_model.pt", map_location="cpu"))
    model.eval()

    try:
        with open("artifacts/tuned_threshold_results.json") as f:
            thresholds = json.load(f)["bilstm"]["thresholds"]
    except FileNotFoundError:
        thresholds = {label: 0.5 for label in LABEL_COLS}

    return model, vocab, config["max_len"], thresholds


def predict(text, model, vocab, max_len, thresholds):
    ids = encode(text, vocab, max_len)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        proba = torch.sigmoid(logits).squeeze(0).tolist()

    labels_triggered = [
        label for label, p in zip(LABEL_COLS, proba) if p >= thresholds[label]
    ]
    return {
        "comment": text,
        "predicted_labels": labels_triggered if labels_triggered else ["clean"],
        "probabilities": {label: round(p, 4) for label, p in zip(LABEL_COLS, proba)},
        "thresholds_used": {label: round(thresholds[label], 3) for label in LABEL_COLS},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classify a comment's toxicity (multi-label)."
    )
    parser.add_argument("text", nargs="?", help="Comment text to classify")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    model, vocab, max_len, thresholds = load_model()

    if args.interactive or not args.text:
        print("Enter comments to classify (empty line to quit):")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                break
            print(
                json.dumps(predict(text, model, vocab, max_len, thresholds), indent=2)
            )
    else:
        print(
            json.dumps(predict(args.text, model, vocab, max_len, thresholds), indent=2)
        )


if __name__ == "__main__":
    main()
