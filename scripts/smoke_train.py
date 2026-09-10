"""Synthetic CUDA plumbing check, not C0 or a paper result."""
import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn


class SmokeRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(513, 16, padding_idx=0)
        self.query = nn.Linear(16, 16)
        self.key = nn.Linear(16, 16)
        self.tower = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, target, history, valid):
        target = self.embedding(target)
        history = self.embedding(history)
        scores = (self.key(history) * self.query(target).unsqueeze(1)).sum(-1)
        # Finite sentinel plus explicit renormalization supports empty histories.
        weights = scores.masked_fill(~valid, -1e9).softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        interest = (history * weights.unsqueeze(-1)).sum(1)
        return self.tower(torch.cat([target, interest], dim=-1)).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently substitute CPU')
    torch.manual_seed(42)
    device = torch.device('cuda')
    model = SmokeRanker().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    target = torch.randint(1, 513, (32,), device=device)
    history = torch.randint(1, 513, (32, 50), device=device)
    valid = torch.ones_like(history, dtype=torch.bool)
    valid[0] = False
    valid[1, :20] = False
    history[~valid] = 0
    labels = (torch.arange(32, device=device) % 2).float()
    before = model.embedding.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    losses = []
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(target, history, valid), labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix('.pt')
    torch.save(model.state_dict(), checkpoint)
    restored = SmokeRanker().to(device)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        difference = (model(target, history, valid) - restored(target, history, valid)).abs().max().item()
    updated = not torch.equal(before, model.embedding.weight)
    assert updated and difference == 0 and losses[-1] < losses[0]
    assert torch.count_nonzero(model.embedding.weight[0]).item() == 0
    result = dict(kind='synthetic_plumbing_only_not_paper_result', python=platform.python_version(),
                  torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), steps=40, batch=32, history_length=50,
                  first_loss=losses[0], final_loss=losses[-1], parameters_updated=updated,
                  reload_max_abs_error=difference, empty_history_finite=True,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, seconds=elapsed)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
