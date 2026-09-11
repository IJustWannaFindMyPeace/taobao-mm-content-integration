"""Train a minimal ID baseline on a bounded training shard; no evaluation claim."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from id_probe import IDProbe, build_vocab, encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--history-length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.history_length) < 1:
        raise ValueError('Training dimensions must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this resource probe')
    pf = pq.ParquetFile(args.input)
    if not 0 < pf.metadata.num_rows <= 4096:
        raise ValueError('Only a nonempty engineering probe of at most 4096 rows is supported')
    rows = pf.read(columns=['205', '150_2_180', 'label_0']).to_pylist()
    for row in rows:
        if row['label_0'] not in ([1, 0], [0, 1]):
            raise ValueError('Unexpected click label')
        if row['205'] is None or row['205'] == 0:
            raise ValueError('Target ID must be present and non-padding')
        if any(v is None for v in row['150_2_180']):
            raise ValueError('Null history ID')
    # Match upstream suffix selection; temporal semantics remain unverified.
    histories = [r['150_2_180'][-args.history_length:] for r in rows]
    vocab = build_vocab([r['205'] for r in rows] + [v for h in histories for v in h])
    target = torch.tensor([encode(r['205'], vocab) for r in rows])
    history = torch.tensor([[0] * (args.history_length - len(h)) + [encode(v, vocab) for v in h] for h in histories])
    valid = history != 0
    labels = torch.tensor([r['label_0'][1] for r in rows], dtype=torch.float32)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    dataset = TensorDataset(target, history, valid, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                        generator=torch.Generator().manual_seed(args.seed))
    evaluation = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    model = IDProbe(len(vocab) + 2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    initial = model.embedding.weight.detach().clone()

    def predict(net):
        net.eval()
        with torch.no_grad():
            return torch.cat([net(t.cuda(), h.cuda(), m.cuda()).cpu() for t, h, m, _ in evaluation])

    initial_loss = nn.functional.binary_cross_entropy_with_logits(predict(model), labels).item()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    curves = []
    for epoch in range(args.epochs):
        model.train()
        total, seen = 0.0, 0
        for t, h, m, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(t.cuda(), h.cuda(), m.cuda()), y.cuda())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            total += loss.item() * len(y)
            seen += len(y)
        assert seen == len(rows)
        curves.append(dict(epoch=epoch + 1, rows=seen, online_train_logloss=total / seen))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    logits = predict(model)
    assert len(logits) == len(rows) and torch.isfinite(logits).all()
    updated = not torch.equal(initial, model.embedding.weight)
    assert updated and torch.count_nonzero(model.embedding.weight[0]) == 0
    with torch.no_grad():
        q = target[:2].cuda()
        empty = torch.zeros((2, args.history_length), dtype=torch.long, device='cuda')
        # Masked values must not affect predictions, including an entirely empty history.
        mask = torch.zeros_like(empty, dtype=torch.bool)
        empty_logits = model(q, empty, mask)
        masked_error = (empty_logits - model(q, torch.full_like(empty, 2), mask)).abs().max().item()
        assert torch.isfinite(empty_logits).all() and masked_error == 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.output_dir / 'vocab.json'
    vocab_path.write_text(json.dumps(vocab, sort_keys=True), encoding='utf-8')
    checkpoint = args.output_dir / 'model.pt'
    torch.save(dict(state_dict=model.state_dict(), vocabulary_size=len(vocab) + 2, width=16), checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    restored = IDProbe(saved['vocabulary_size'], saved['width']).cuda()
    restored.load_state_dict(saved['state_dict'])
    reload_error = (logits - predict(restored)).abs().max().item()
    assert reload_error == 0
    torch.save(dict(row_index=torch.arange(len(rows)), logits=logits, labels=labels), args.output_dir / 'train_predictions.pt')
    repo = Path(__file__).resolve().parents[1]
    report = dict(kind='training_only_id_engineering_probe_not_formal_C0',
        source_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        implementation_sha256={str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(), repo / 'src/id_probe.py']},
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        vocab_sha256=hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        seed=args.seed, rows=len(rows), clicks=int(labels.sum()), vocab_size=len(vocab) + 2,
        batch_size=args.batch_size, history_length=args.history_length, epochs=args.epochs,
        optimizer='Adam', learning_rate=0.001, width=16, epoch_curve=curves,
        initial_train_logloss=initial_loss,
        final_train_logloss=nn.functional.binary_cross_entropy_with_logits(logits, labels).item(),
        parameters_updated=updated, prediction_rows=len(logits), reload_max_abs_error=reload_error,
        empty_history_finite=True, masked_value_max_abs_error=masked_error,
        training_seconds=elapsed, training_peak_allocated_mib=peak, training_peak_reserved_mib=reserved,
        limitations=['training-only size-selected shard; no held-out evaluation',
            'item-only attention prototype, not final shared C0 architecture',
            'suffix history order follows reference code, not independently verified timestamps',
            'vocabulary built only from this probe targets and selected histories',
            'memory includes diagnostic parameter snapshot; not a full-data estimate'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
