"""Measure a complete candidate vocabulary with bounded training, without evaluation."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--model', choices=['c6', 'r4'], required=True)
    parser.add_argument('--muse-repo', type=Path)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 200 or args.batch_size < 1:
        raise ValueError('At least 200 steps and a positive batch size required')
    if args.report.exists():
        raise ValueError('Use a new report path')
    root = Path(__file__).resolve().parents[1]
    if args.model == 'r4':
        if args.muse_repo is None:
            raise ValueError('MUSE repository required for R4')
        root = args.muse_repo.resolve()
    sys.path.insert(0, str(root / 'src'))
    from feature_store import FeatureStore
    if args.model == 'c6':
        from make_fusion import C6
        from simtier import histogram
    else:
        from muse_model import MUSEModel
    sources = {p.name: digest(p) for p in sorted((root / 'src').glob('*.py'))}
    sources['probe_scale.py'] = digest(Path(__file__))
    store = FeatureStore(args.store, 'train')
    if args.steps * args.batch_size > len(store):
        raise ValueError('Insufficient training rows for distinct probe batches')
    torch.manual_seed(42)
    model = (C6(store.manifest['vocab_sizes']) if args.model == 'c6'
             else MUSEModel(store.manifest['vocab_sizes'])).cuda().train()
    sparse = list(model.embeddings.parameters())
    sparse_ids = {id(p) for p in sparse}
    dense = [p for p in model.parameters() if id(p) not in sparse_ids and p.requires_grad]
    optimizers = [torch.optim.SparseAdam(sparse, lr=.001),
                  torch.optim.AdamW(dense, lr=.001, weight_decay=0)]
    length = 200 if args.model == 'c6' else 1000
    rows = np.random.default_rng(42).permutation(len(store))[:args.steps * args.batch_size]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    warmup_end = None
    initial = model.tower[-1].weight.detach().clone()
    for step in range(args.steps):
        batch = store.batch(rows[step*args.batch_size:(step+1)*args.batch_size], length, with_content=True)
        keys = ['features', 'history_item', 'history_category', 'history_valid',
                'target_content', 'history_content', 'target_content_available', 'history_content_available']
        inputs = [torch.as_tensor(batch[k], device='cuda', dtype=torch.long if k in keys[:3] else None) for k in keys]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        if args.model == 'c6':
            with torch.no_grad():
                counts = histogram(inputs[4], inputs[5], inputs[3], inputs[6], inputs[7])
            logits = model(*inputs[:4], counts, *inputs[4:])
        else:
            logits = model(*inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.as_tensor(batch['labels'], device='cuda'))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad = parameter.grad.coalesce().values() if parameter.grad.is_sparse else parameter.grad
                if not torch.isfinite(grad).all():
                    raise RuntimeError('Nonfinite gradient')
        for optimizer in optimizers:
            optimizer.step()
        if step == 19:
            torch.cuda.synchronize()
            warmup_end = time.perf_counter()
        if (step + 1) % 50 == 0:
            print(f'{args.model}: {step+1}/{args.steps} steps', flush=True)
    torch.cuda.synchronize()
    ended = time.perf_counter()
    allocated = torch.cuda.max_memory_allocated() / 2**20
    ram = psutil.Process().memory_info().peak_wset / 2**20
    updated = not torch.equal(initial, model.tower[-1].weight)
    if not updated:
        raise RuntimeError('No parameter update')
    report = dict(scope='resource_probe_only', model=args.model, seed=42,
        store_manifest_sha256=digest(args.store/'manifest.json'), source_sha256=sources,
        vocabulary_sizes=store.manifest['vocab_sizes'], steps=args.steps, batch_size=args.batch_size,
        history_length=length, k=50 if args.model == 'r4' else None,
        seconds=ended-started, post_warmup_rows_per_second=(args.steps-20)*args.batch_size/(ended-warmup_end),
        peak_allocated_mib=allocated, peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        process_peak_working_set_mib=ram, parameters_updated=updated,
        gate=dict(allocated_limit_mib=9216, ram_limit_mib=24576, passed=allocated<=9216 and ram<=24576),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, evaluation_performed=False,
        limitations=['Bounded random-row probe; not full-epoch memory or throughput assurance.',
            'C6 uses random frozen content weights and online histogram construction; no effectiveness claim or transferred checkpoint.'
            if args.model == 'c6' else 'Full candidate vocabulary, but only 200 or more sampled training steps.'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
