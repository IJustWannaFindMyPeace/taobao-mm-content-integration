"""Minimal ID attention baseline for a training-only engineering probe."""
import torch
from torch import nn

PAD, OOV = 0, 1


def build_vocab(values):
    # Signed raw IDs are opaque identifiers; negative IDs are valid.
    return {value: index + 2 for index, value in enumerate(sorted(set(values) - {0}))}


def encode(value, vocab):
    return PAD if value == 0 else vocab.get(value, OOV)


class IDProbe(nn.Module):
    def __init__(self, vocabulary_size, width=16):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width, padding_idx=PAD)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.tower = nn.Sequential(nn.Linear(width * 2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, target, history, valid):
        target = self.embedding(target)
        history = self.embedding(history)
        scores = (self.query(target).unsqueeze(1) * self.key(history)).sum(-1)
        weights = scores.masked_fill(~valid, torch.finfo(scores.dtype).min).softmax(-1)
        weights = weights * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        interest = (weights.unsqueeze(-1) * history).sum(1)
        return self.tower(torch.cat([target, interest], dim=-1)).squeeze(-1)
