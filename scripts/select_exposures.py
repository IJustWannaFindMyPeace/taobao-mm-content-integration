"""Scan complete raw exposure tables into deterministic user cohorts."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from user_split import user_masks
from fetch_source import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--buckets', type=int, choices=[1, 5, 10], default=1)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    files = {e['path']: e for e in manifest['files']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats, scans = {}, {}
    for original in ['train', 'test']:
        name = f'raw/{original}_samples.parquet'
        path = args.source_dir / name
        if sha256(path) != files[name]['lfs']['oid']:
            raise ValueError('Source hash mismatch')
        pf = pq.ParquetFile(path)
        print(f'{original}: {pf.metadata.num_rows} rows; columns={pf.schema_arrow.names}', flush=True)
        # Ignore labels when selecting users. Test excludes the validation-user cohort.
        outputs = ['train', 'validation'] if original == 'train' else ['test']
        writers, totals, users = {}, {k: 0 for k in outputs}, {k: set() for k in outputs}
        offset = 0
        try:
            for batch in pf.iter_batches(batch_size=131072):
                table = pa.Table.from_batches([batch])
                ids_column = table['129_1']
                if ids_column.null_count:
                    raise ValueError('Null user IDs')
                ids = ids_column.to_numpy()
                training, validation = user_masks(ids, args.buckets)
                table = table.append_column('source_row', pa.array(np.arange(offset, offset + len(table), dtype=np.int64)))
                for split in outputs:
                    mask = validation if split == 'validation' else training
                    selected = table.filter(pa.array(mask))
                    if split not in writers:
                        writers[split] = pq.ParquetWriter(args.output_dir / (split + '.parquet.part'), table.schema, compression='zstd')
                    if len(selected):
                        writers[split].write_table(selected)
                        totals[split] += len(selected)
                        users[split].update(ids[mask].tolist())
                offset += len(batch)
            if offset != pf.metadata.num_rows:
                raise ValueError('Incomplete source scan')
        finally:
            for writer in writers.values(): writer.close()
        scans[original] = dict(rows=offset, sha256=files[name]['lfs']['oid'])
        for split in outputs:
            target = args.output_dir / (split + '.parquet')
            target.with_suffix('.parquet.part').replace(target)
            stats[split] = dict(rows=totals[split], users=len(users[split]), sha256=sha256(target))
        print(f'{original}: completed {offset} rows', flush=True)
    report = dict(protocol='user-cohort-v1-exposures', revision=manifest['revision'],
        selection=f'splitmix64(user_id_bits XOR 42) modulo 1000 less than {args.buckets}',
        buckets=args.buckets,
        validation='independent splitmix64 seed 2026 modulo 5 equals 0',
        test='official test boundary; selected non-validation cohort; labels not used for selection',
        scans=scans, subsets=stats, history_join_complete=False, formal_training_ready=False,
        limitations=['user-disjoint validation is not a temporal holdout',
            'test cohort may contain users absent from training',
            'history and item joins, coverage and duplicate checks remain pending'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
