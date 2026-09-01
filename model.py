"""
Multi-label neural network for toxic comment classification.

Architecture: Embedding -> Bidirectional LSTM -> mean+max pooling -> Dense
-> sigmoid outputs (one independent probability per label, since a comment
can belong to multiple toxicity categories at once).
"""

import torch
from torch import nn


class ToxicMultiLabelClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        embed_dim=100,
        hidden_dim=96,
        num_layers=1,
        dropout=0.3,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                128, num_labels
            ),  # raw logits; sigmoid applied via loss/inference
        )

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()

        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        lstm_out_masked = lstm_out * mask
        sum_hidden = lstm_out_masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-6)
        mean_pool = sum_hidden / lengths

        lstm_out_for_max = lstm_out.masked_fill(mask == 0, float("-inf"))
        max_pool, _ = lstm_out_for_max.max(dim=1)

        pooled = torch.cat([mean_pool, max_pool], dim=1)
        logits = self.classifier(pooled)
        return logits
