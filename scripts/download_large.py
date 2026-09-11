"""Sequential bounded-range downloader with verified resume and atomic completion."""
import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from fetch_source import sha256

ALLOWED = ['raw/train_user_features.parquet', 'raw/test_user_features.parquet', 'raw/scl_embedding_int8_p90.parquet']


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def download(source, entry, root, max_chunks=None):
    target = root / entry['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = entry['lfs']['oid']
    partial = target.with_suffix('.parquet.part')
    journal = target.with_suffix('.download.json')
    status = target.with_suffix('.status.json')
    if target.exists():
        if target.stat().st_size != entry['size'] or sha256(target) != expected:
            raise ValueError('Existing final file failed checksum')
        return
    identity = dict(revision=source['revision'], path=entry['path'], size=entry['size'], sha256=expected)
    state = dict(identity=identity, chunks=[])
    if journal.exists():
        state = json.loads(journal.read_text(encoding='utf-8'))
        if state['identity'] != identity:
            raise ValueError('Resume source changed')
    if not partial.exists():
        if state['chunks']:
            raise ValueError('Resume data missing')
        partial.touch()
    # Recheck every durable chunk before trusting resume state.
    offset = 0
    with partial.open('r+b') as stream:
        for chunk in state['chunks']:
            data = stream.read(chunk['size'])
            if len(data) != chunk['size'] or hashlib.sha256(data).hexdigest() != chunk['sha256']:
                raise ValueError('Resume chunk checksum mismatch')
            offset += chunk['size']
        stream.truncate(offset)
    if shutil.disk_usage(root).free < entry['size'] - offset + 2 * 1024**3:
        raise RuntimeError('Insufficient disk space')
    chunk_size = 64 * 1024**2
    count = 0
    base = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{entry['path']}"
    while offset < entry['size']:
        if max_chunks is not None and count >= max_chunks:
            write_json(status, dict(state='paused_at_chunk_limit', downloaded=offset, total=entry['size']))
            return
        end = min(offset + chunk_size, entry['size']) - 1
        for attempt in range(4):
            try:
                # Distinct query key prevents intermediaries reusing another byte range.
                req = urllib.request.Request(base + f'?download=true&range_start={offset}&range_end={end}',
                    headers={'Range': f'bytes={offset}-{end}', 'Accept-Encoding': 'identity'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{entry["size"]}':
                        raise ValueError('Server did not honor exact range')
                    digest = hashlib.sha256()
                    received = 0
                    with partial.open('r+b') as stream:
                        stream.seek(offset)
                        while block := response.read(min(4 * 1024**2, end - offset + 2 - received)):
                            received += len(block)
                            if received > end - offset + 1:
                                raise ValueError('Oversized range response')
                            stream.write(block)
                            digest.update(block)
                        if received != end - offset + 1:
                            raise ValueError('Incomplete range response')
                        stream.flush()
                        os.fsync(stream.fileno())
                break
            except Exception as exc:
                with partial.open('r+b') as stream: stream.truncate(offset)
                write_json(status, dict(state='retrying', downloaded=offset, total=entry['size'], attempt=attempt+1, error=type(exc).__name__))
                if attempt == 3: raise
                time.sleep(2 ** attempt)
        state['chunks'].append(dict(size=received, sha256=digest.hexdigest()))
        write_json(journal, state)
        offset += received
        count += 1
        write_json(status, dict(state='downloading', downloaded=offset, total=entry['size']))
        print(f"{entry['path']}: {offset}/{entry['size']} ({offset/entry['size']:.2%})", flush=True)
    write_json(status, dict(state='verifying_full_sha256', downloaded=offset, total=entry['size']))
    if sha256(partial) != expected:
        raise ValueError('Full source SHA256 mismatch')
    partial.replace(target)
    write_json(status, dict(state='complete_verified', downloaded=offset, total=entry['size'], sha256=expected))
    print('Verified ' + entry['path'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--files', nargs='+', choices=ALLOWED, required=True)
    parser.add_argument('--max-chunks', type=int)
    args = parser.parse_args()
    if args.max_chunks is not None and args.max_chunks < 1: raise ValueError('Positive chunk limit required')
    source = json.loads(args.manifest.read_text(encoding='utf-8'))
    entries = {x['path']: x for x in source['files']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        download(source, entries[name], args.output_dir, args.max_chunks)


if __name__ == '__main__': main()
