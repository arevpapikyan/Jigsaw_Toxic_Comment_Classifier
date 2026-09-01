"""
Data loading, tokenization, and Dataset utilities for multi-label toxic
comment classification on the Jigsaw dataset.
"""

import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def simple_tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"([.,!?;:'\"()\n])", r" \1 ", text)
    return text.split()


def build_vocab(texts, min_freq=3, max_vocab_size=30000):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.most_common(max_vocab_size):
        if freq < min_freq:
            continue
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len):
    tokens = simple_tokenize(text)
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


class ToxicMultiLabelDataset(Dataset):
    """texts: list[str], labels: (N, num_labels) array of 0/1."""

    def __init__(self, texts, labels, vocab, max_len):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def load_official_test_set(
    test_csv="data/test.csv", test_labels_csv="data/test_labels.csv"
):
    """
    The official Kaggle test set marks rows that were NOT used for scoring
    with -1 across all label columns. These must be filtered out, or you'll
    silently corrupt evaluation metrics.
    """
    test = pd.read_csv(test_csv)
    labels = pd.read_csv(test_labels_csv)
    merged = test.merge(labels, on="id")
    used_mask = (merged[LABEL_COLS] != -1).all(axis=1)
    merged = merged.loc[used_mask].reset_index(drop=True)
    return merged


def save_vocab(vocab, path):
    with open(path, "w") as f:
        json.dump(vocab, f)


def load_vocab(path):
    with open(path) as f:
        return json.load(f)
