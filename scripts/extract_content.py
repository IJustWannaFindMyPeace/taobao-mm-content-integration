"""Retain fixed int8 content for cohort targets and full selected histories."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fetch_source import sha256


def positions(keys, requested):
    indices = np.searchsorted(requested, keys)
    valid = indices < len(requested)
    valid[valid] &= requested[indices[valid]] == keys[valid]
    return indices, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cohort-dir', type=Path, required=True)
    parser.add_argument('--cohort-report', type=Path, required=True)
    parser.add_argument('--history-dir', type=Path, required=True)
    parser.add_argument('--history-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    cohort = json.loads(args.cohort_report.read_text(encoding='utf-8'))
    history = json.loads(args.history_report.read_text(encoding='utf-8'))
    if not manifest['revision'] == cohort['revision'] == history['revision']:
        raise ValueError('Revision mismatch')
    entry = next(e for e in manifest['files'] if e['path']=='raw/scl_embedding_int8_p90.parquet')
    if sha256(args.source) != entry['lfs']['oid']: raise ValueError('Content digest mismatch')
    parts, targets = [], {}
    for split in ['train','validation','test']:
        path = args.cohort_dir / (split + '.parquet')
        if sha256(path) != cohort['subsets'][split]['sha256']: raise ValueError('Cohort digest mismatch')
        values = pq.read_table(path, columns=['205'])['205'].to_numpy()
        targets[split] = values
        parts.append(np.unique(values))
    for split in ['train','test']:
        path = args.history_dir / (split + '_users.parquet')
        if sha256(path) != history['splits'][split]['sha256']: raise ValueError('History digest mismatch')
        with pq.ParquetFile(path) as pf:
            for batch in pf.iter_batches(batch_size=4096, columns=['150_2_180']):
                parts.append(np.unique(pc.list_flatten(batch.column(0)).to_numpy()))
    requested = np.unique(np.concatenate(parts))
    del parts
    requested = requested[requested != 0]
    if not len(requested): raise ValueError('No requested items')
    seen = np.zeros(len(requested), dtype=bool)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'content_int8.parquet'
    temporary = output.with_suffix('.parquet.part')
    scanned, saved, zero_vectors, dimension = 0, 0, 0, None
    with pq.ParquetFile(args.source, pre_buffer=False) as pf, pq.ParquetWriter(temporary, pq.read_schema(args.source), compression='zstd') as writer:
        for batch in pf.iter_batches(batch_size=131072, use_threads=False):
            table = pa.Table.from_batches([batch])
            keys = table['205'].to_numpy()
            indices, valid = positions(keys, requested)
            selected = table.filter(pa.array(valid))
            matched = indices[valid]
            if seen[matched].any() or len(np.unique(matched)) != len(matched):
                raise ValueError('Duplicate requested content keys')
            if len(selected):
                col = selected['205_c']
                if col.null_count or pc.list_flatten(col).null_count: raise ValueError('Null content')
                lengths = pc.list_value_length(col).to_numpy()
                if dimension is None: dimension = int(lengths[0])
                if dimension < 1 or not np.all(lengths == dimension): raise ValueError('Content dimensions differ')
                values = pc.list_flatten(col).to_numpy().reshape(-1, dimension)
                zero_vectors += int(np.all(values == 0, axis=1).sum())
                writer.write_table(selected)
                seen[matched] = True
                saved += len(selected)
            scanned += len(table)
            if scanned % 4194304 == 0: print(f'Content scanned {scanned}/{pf.metadata.num_rows}', flush=True)
        if scanned != pf.metadata.num_rows: raise ValueError('Incomplete content scan')
    temporary.replace(output)
    target_coverage = {}
    for split, keys in targets.items():
        idx, valid = positions(keys, requested)
        found = np.zeros(len(keys), dtype=bool)
        found[valid] = seen[idx[valid]]
        target_coverage[split] = dict(exposures=len(keys), matched=int(found.sum()), coverage=float(found.mean()))
    result = dict(revision=manifest['revision'], source_sha256=entry['lfs']['oid'], output_sha256=sha256(output),
        source_rows=scanned, requested_unique_items=len(requested), matched_unique_items=saved,
        missing_unique_items=int((~seen).sum()), unique_coverage=float(seen.mean()),
        dimension=dimension, storage='int8', reference_dequantization_scale=1/127,
        zero_vectors=zero_vectors, target_coverage=target_coverage,
        process_peak_working_set_mib=psutil.Process().memory_info().peak_wset/2**20,
        seconds=time.perf_counter()-started,
        limitations=['missing content is not padding; downstream fallback remains to be specified',
            'requested content union includes evaluation items but does not define trainable ID vocabulary',
            'unique item coverage is not exposure-weighted history coverage'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
