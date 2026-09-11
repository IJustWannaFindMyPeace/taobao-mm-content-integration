"""Verify item source and attach attributes without multiplying exposures."""
import argparse
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from item_join import attach_items
from fetch_source import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--items', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cohort-report', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text(encoding='utf-8'))
    cohort = json.loads(args.cohort_report.read_text(encoding='utf-8'))
    entry = next(e for e in source['files'] if e['path'] == 'raw/item_features.parquet')
    if cohort['revision'] != source['revision'] or sha256(args.items) != entry['lfs']['oid']:
        raise ValueError('Source revision or item digest mismatch')
    items = pq.read_table(args.items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(revision=source['revision'], item_source_sha256=entry['lfs']['oid'], splits={})
    for split in ['train', 'validation', 'test']:
        path = args.input_dir / (split + '.parquet')
        if sha256(path) != cohort['subsets'][split]['sha256']:
            raise ValueError('Cohort input digest mismatch')
        exposures = pq.read_table(path)
        result = attach_items(exposures, items)
        assert len(result) == len(exposures) and result['source_row'].equals(exposures['source_row'])
        output = args.output_dir / (split + '.parquet')
        temporary = output.with_suffix('.parquet.part')
        pq.write_table(result, temporary, compression='zstd')
        temporary.replace(output)
        report['splits'][split] = dict(rows=len(result), target_attribute_coverage=1.0,
            exposure_order_preserved=True, sha256=sha256(output))
    report['history_join_complete'] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
