"""Download one pinned training shard and validate a diagnostic subset.

This size-selected shard is NOT a representative train/validation/test split.
No test data or content vectors are downloaded by this script.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--shard', default='train/train-shard-000159.parquet')
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text(encoding='utf-8'))
    entry = next(x for x in source['files'] if x['path'] == args.shard)
    if not args.shard.startswith('train/'):
        raise ValueError('Probe must use training data only')
    if entry['size'] > 64 * 1024 * 1024:
        raise ValueError('This diagnostic downloader is limited to 64 MiB shards')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / Path(args.shard).name
    expected = entry['lfs']['oid']
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{args.shard}"
        temporary = path.with_suffix('.part')
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
            received = 0
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > entry['size']:
                    raise ValueError('Downloaded bytes exceed manifest size')
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise ValueError('SHA256 mismatch; partial file not promoted')
        temporary.replace(path)
    pf = pq.ParquetFile(path)
    required = {'label_0', '129_1', '205', '206', '150_2_180', '151_2_180'}
    if not required.issubset(pf.schema_arrow.names):
        raise ValueError(f'Missing columns: {required-set(pf.schema_arrow.names)}')
    rows = []
    for batch in pf.iter_batches(batch_size=256):
        rows.extend(batch.to_pylist())
        if len(rows) >= 4096:
            rows = rows[:4096]
            break
    for row in rows:
        assert row['label_0'] in ([1, 0], [0, 1])
        assert row['205'] is not None and row['206'] is not None
        assert len(row['150_2_180']) == len(row['151_2_180'])
        assert all(i is not None for i in row['150_2_180'])
    pq.write_table(pa.Table.from_pylist(rows, schema=pf.schema_arrow), args.output_dir/'probe.parquet')
    users = {}
    for row in rows:
        users.setdefault(row['129_1'], set()).add(row['label_0'][1])
    report = dict(purpose='engineering_probe_not_formal_evaluation', revision=source['revision'],
                  source_file=args.shard, source_sha256=expected, source_rows=pf.metadata.num_rows,
                  probe_rows=len(rows), columns=pf.schema_arrow.names,
                  clicks=sum(x['label_0'][1] for x in rows),
                  users=len(users), users_with_both_labels=sum(len(v)==2 for v in users.values()),
                  history_min=min(len(x['150_2_180']) for x in rows),
                  history_max=max(len(x['150_2_180']) for x in rows),
                  target_items=len({x['205'] for x in rows}),
                  content_vectors_available=False,
                  limitations=['size-selected training shard; selection bias',
                               'user records may exist outside probe',
                               'no validation/test split yet; no reported AUC',
                               'time semantics not yet independently validated'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
