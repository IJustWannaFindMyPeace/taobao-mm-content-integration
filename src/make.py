"""Content-only MAKE precursor with an explicitly adapted DIN-style aggregator."""
import torch
from torch import nn
from torch.nn import functional as F


class ContentMAKE(nn.Module):
    """Fixed 128D inputs; local activation pooling and a four-linear-layer head.

    Local activation uses [q, k, q-k, q*k]. Unnormalized scalar weights
    preserve interest intensity. Widths and ReLU are public-data adaptations.
    """
    def __init__(self, width=128):
        super().__init__()
        self.activation = nn.Sequential(nn.Linear(width*4, 80), nn.ReLU(),
                                        nn.Linear(80, 40), nn.ReLU(), nn.Linear(40, 1))
        self.hidden = nn.Sequential(nn.Linear(width, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.output = nn.Linear(32, 1)

    def extract(self, target, history, valid, target_available, history_available):
        # Inputs are features, never trainable encoder outputs in this experiment.
        target = F.normalize(target.detach(), dim=-1)
        history = F.normalize(history.detach(), dim=-1)
        mask = valid & history_available & target_available[:, None]
        history = history * mask[..., None]
        query = target[:, None, :].expand_as(history)
        scores = self.activation(torch.cat([query, history, query-history, query*history], -1)).squeeze(-1)
        interest = (history * (scores * mask)[..., None]).sum(1)
        hidden = self.hidden(interest)
        return interest, hidden

    def forward(self, *inputs):
        return self.output(self.extract(*inputs)[1]).squeeze(-1)
