"""Measure exact train-history item vocabulary lower bounds before cohort expansion.

No labels, validation metrics, test files, or training configuration are changed.
The bound excludes target-only items and all other embedding fields.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from user_split import mix64
from fetch_source import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Fresh output directory required')
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    entries = {entry['path']: entry for entry in manifest['files']}
    buckets = [75, 100]
    process = psutil.Process()

    def status(phase, **extra):
        payload = dict(phase=phase, seconds=time.perf_counter()-started,
                       process_rss_gib=process.memory_info().rss/2**30, **extra)
        temp = args.output_dir/'status.tmp'
        temp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        temp.replace(args.output_dir/'status.json')
        print(json.dumps(payload), flush=True)
        if process.memory_info().rss > 20*2**30 or psutil.virtual_memory().available < 4*2**30:
            raise RuntimeError('Capacity scan memory guard reached')

    sources = {}
    for name in ['train_samples.parquet', 'train_user_features.parquet']:
        path = args.source_dir/'raw'/name
        status('verify_source', file=name)
        digest = sha256(path)
        if digest != entries['raw/'+name]['lfs']['oid']:
            raise ValueError('Source hash mismatch: '+name)
        sources[name] = digest

    users = {b: [] for b in buckets}
    exposures = {b: 0 for b in buckets}
    pf = pq.ParquetFile(args.source_dir/'raw/train_samples.parquet')
    scanned = 0
    for batch in pf.iter_batches(batch_size=131072, columns=['129_1']):
        ids = batch.column(0).to_numpy()
        bucket = mix64(ids, 42) % 1000
        train = mix64(ids, 2026) % 5 != 0
        for b in buckets:
            selected = ids[train & (bucket < b)]
            exposures[b] += len(selected)
            users[b].append(np.unique(selected))
        scanned += len(ids)
    if scanned != pf.metadata.num_rows:
        raise ValueError('Incomplete exposure scan')
    users = {b: np.unique(np.concatenate(parts)) for b, parts in users.items()}
    status('history_scan', scanned_users=0,
           training_users={b: len(v) for b, v in users.items()})
    keys = {b: np.empty(0, dtype=np.int64) for b in buckets}
    pending = {b: [] for b in buckets}
    found = {b: [] for b in buckets}
    pf = pq.ParquetFile(args.source_dir/'raw/train_user_features.parquet')
    scanned = 0
    for step, batch in enumerate(pf.iter_batches(batch_size=4096,
              columns=['129_1', '150_2_180'], use_threads=False), 1):
        table = pa.Table.from_batches([batch])
        ids = table['129_1'].to_numpy()
        for b in buckets:
            positions = np.searchsorted(users[b], ids)
            mask = (positions < len(users[b])) & (users[b][np.minimum(positions, len(users[b])-1)] == ids)
            found[b].append(ids[mask])
            column = table['150_2_180'].filter(pa.array(mask))
            if column.null_count or pc.list_flatten(column).null_count:
                raise ValueError('Null history')
            values = np.unique(pc.list_flatten(column).to_numpy())
            pending[b].append(values[values != 0])
        scanned += len(ids)
        if step % 128 == 0:
            for b in buckets:
                keys[b] = np.unique(np.concatenate([keys[b]] + pending[b]))
                pending[b].clear()
            status('history_scan', scanned_users=scanned, total_users=pf.metadata.num_rows,
                   item_counts={b: len(v) for b, v in keys.items()})
    if scanned != pf.metadata.num_rows:
        raise ValueError('Incomplete history scan')
    results = {}
    for b in buckets:
        seen = np.concatenate(found[b])
        if len(seen) != len(users[b]) or not np.array_equal(np.sort(seen), users[b]):
            raise ValueError('Missing or duplicate selected user history')
        keys[b] = np.unique(np.concatenate([keys[b]] + pending[b]))
        count = len(keys[b])
        results[b] = dict(training_users=len(users[b]), training_exposures=exposures[b],
                          unique_nonpadding_history_items=count,
                          item_weight_and_two_adam_states_gib=(count+2)*16*4*3/2**30)
    result = dict(scope='exact_train_history_item_vocabulary_lower_bound',
                  revision=manifest['revision'], source_sha256=sources,
                  script_sha256=sha256(Path(__file__)), candidates=results,
                  seconds=time.perf_counter()-started,
                  test_read=False, labels_read=False,
                  limitations=['excludes target-only items and other embedding fields',
                               'excludes activations, gradients, CUDA workspace and allocator overhead',
                               'capacity evidence only; no model-quality or final scale decision'])
    (args.output_dir/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    status('complete', candidates=results)


if __name__ == '__main__':
    main()
