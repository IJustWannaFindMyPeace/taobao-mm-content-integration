"""Audit independently sampled official demo tables without fabricating joins."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tables = {p.stem: pq.read_table(p).to_pylist() for p in args.data.glob('*.parquet')}
    summary = {'files': {}, 'splits': {}}
    for path in args.data.glob('*.parquet'):
        summary['files'][path.name] = dict(rows=len(tables[path.stem]),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    item_ids = {x['205'] for x in tables['item_features']}
    content_ids = {x['205'] for x in tables['scl_embedding_int8_p90']}
    for split in ['train', 'test']:
        samples = tables[f'{split}_samples']
        users = {x['129_1']: x for x in tables[f'{split}_user_features']}
        labels = [x['label_0'] for x in samples]
        assert all(x in ([1, 0], [0, 1]) for x in labels)
        summary['splits'][split] = dict(samples=len(samples), clicks=sum(x[1] for x in labels),
            user_join_hits=sum(x['129_1'] in users for x in samples),
            item_join_hits=sum(x['205'] in item_ids for x in samples),
            content_join_hits=sum(x['205'] in content_ids for x in samples),
            history_lengths=sorted({len(x['150_2_180']) for x in users.values()}),
            aligned_item_category_lengths=all(len(x['150_2_180']) == len(x['151_2_180']) for x in users.values()))
    summary['content_dimensions'] = sorted({len(x['205_c']) for x in tables['scl_embedding_int8_p90']})
    summary['decision'] = 'schema_fixture_only; require coherently joined real subset before C0 training'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
