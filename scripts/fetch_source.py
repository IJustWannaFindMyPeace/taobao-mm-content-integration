"""Download explicitly named pinned files with streaming SHA256 verification."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--files', nargs='+', required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text(encoding='utf-8'))
    entries = {x['path']: x for x in source['files']}
    for name in args.files:
        entry = entries[name]
        if name not in ['raw/train_samples.parquet', 'raw/test_samples.parquet', 'raw/item_features.parquet']:
            raise ValueError('Only the bounded raw exposure and item tables are supported')
        if entry['size'] > 1024**3:
            raise ValueError('File exceeds 1 GiB limit')
        target = args.output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = entry['lfs']['oid']
        if target.exists() and target.stat().st_size == entry['size'] and sha256(target) == expected:
            print('Verified cached ' + name, flush=True)
            continue
        temporary = target.with_suffix('.parquet.part')
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{name}"
        print('Downloading ' + name, flush=True)
        received, next_progress = 0, 64 * 1024**2
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
            while block := response.read(4 * 1024**2):
                received += len(block)
                if received > entry['size']:
                    raise ValueError('Response exceeds declared size')
                output.write(block)
                if received >= next_progress:
                    print(f'{name}: {received / 1024**2:.0f} MiB', flush=True)
                    next_progress += 64 * 1024**2
        if received != entry['size'] or sha256(temporary) != expected:
            raise ValueError('Size or SHA256 mismatch')
        temporary.replace(target)
        print('Verified ' + name, flush=True)


if __name__ == '__main__':
    main()
