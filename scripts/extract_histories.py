"""Stream verified raw history tables and retain selected users only."""
import argparse
import json
import time
import psutil
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fetch_source import sha256


def extract(source, requested, output):
    started = time.perf_counter()
    with pq.ParquetFile(source, pre_buffer=False) as pf:
        wanted = pa.array(sorted(requested), type=pa.int64())
        found = set()
        scanned = 0
        saved = 0
        temporary = output.with_suffix('.parquet.part')
        with pq.ParquetWriter(temporary, pf.schema_arrow, compression='zstd') as writer:
            for batch in pf.iter_batches(batch_size=4096, use_threads=False):
                table = pa.Table.from_batches([batch])
                scanned += len(table)
                if scanned % 262144 == 0:
                    print(f"Scanned {scanned}/{pf.metadata.num_rows} users", flush=True)
                selected = table.filter(pc.is_in(table['129_1'], value_set=wanted))
                if not len(selected): continue
                ids = selected['129_1'].to_pylist()
                if len(set(ids)) != len(ids) or found.intersection(ids):
                    raise ValueError('Multiple history records for one selected user')
                for name in ['150_2_180', '151_2_180']:
                    if selected[name].null_count or pc.list_flatten(selected[name]).null_count:
                        raise ValueError('Null history or history element')
                if pc.list_value_length(selected['150_2_180']).to_pylist() != pc.list_value_length(selected['151_2_180']).to_pylist():
                    raise ValueError('History item/category length mismatch')
                found.update(ids)
                saved += len(selected)
                writer.write_table(selected)
        if scanned != pf.metadata.num_rows or found != requested:
            raise ValueError(f'Incomplete history coverage: {len(found)}/{len(requested)}')
        temporary.replace(output)
        return dict(source_rows=scanned, selected_users=saved, coverage=1.0, sha256=sha256(output),
                    seconds=time.perf_counter()-started,
                    process_peak_working_set_mib=psutil.Process().memory_info().peak_wset/2**20)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--cohort-dir', type=Path, required=True)
    parser.add_argument('--cohort-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    cohort = json.loads(args.cohort_report.read_text(encoding='utf-8'))
    if manifest['revision'] != cohort['revision']: raise ValueError('Revision mismatch')
    entries = {e['path']:e for e in manifest['files']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(revision=manifest['revision'], splits={})
    for mode, splits in [('train',['train','validation']), ('test',['test'])]:
        requested = set()
        for split in splits:
            path = args.cohort_dir / (split + '.parquet')
            if sha256(path) != cohort['subsets'][split]['sha256']: raise ValueError('Cohort digest mismatch')
            requested.update(pq.read_table(path, columns=['129_1'])['129_1'].to_pylist())
        name = f'raw/{mode}_user_features.parquet'
        source = args.source_dir / name
        if sha256(source) != entries[name]['lfs']['oid']: raise ValueError('History digest mismatch')
        result['splits'][mode] = extract(source, requested, args.output_dir / (mode + '_users.parquet'))
        print(mode + ' history extraction complete', flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
