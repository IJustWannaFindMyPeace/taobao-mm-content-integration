"""Run six downstream controls serially using a validation-selected content checkpoint."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from fetch_source import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--content-run', type=Path, required=True)
    parser.add_argument('--hist-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Fresh output directory required')
    root = Path(__file__).resolve().parents[1]
    manifest_hash = sha256(args.store / 'manifest.json')
    manifest = json.loads((args.store / 'manifest.json').read_text())
    content_report = args.content_run / 'report.json'
    content = json.loads(content_report.read_text())
    cache = json.loads((args.hist_cache / 'manifest.json').read_text())
    if content['store_manifest_sha256'] != manifest_hash or cache['store_manifest_sha256'] != manifest_hash:
        raise ValueError('Pretraining/cache must use the same store')
    if content['test_evaluated'] or len(content['epochs']) != content['config']['epochs']:
        raise ValueError('Require complete validation-only pretraining')
    for epoch in content['epochs']:
        if epoch['rows'] != manifest['splits']['train']['rows'] or epoch['reload_max_abs_error'] != 0:
            raise ValueError('Incomplete or invalid pretraining epoch')
    best = max(content['epochs'], key=lambda e: (e['validation']['gauc'], -e['epoch']))
    checkpoint = args.content_run / f"epoch-{best['epoch']}.pt"
    if sha256(checkpoint) != best['checkpoint_sha256']:
        raise ValueError('Selected checkpoint digest mismatch')
    names = ['c0', 'c2', 'c3', 'c5', 'c6', 'c7']
    configs = {}
    for name in names:
        cfg = json.loads((root / 'configs' / (name + '.json')).read_text())
        cfg['seed'] = content['config']['seed']
        if cfg['history_length'] != content['config']['history_length']:
            raise ValueError('Pretraining history length mismatch')
        if cfg.get('simtier_bins') and (cache['bins'] != cfg['simtier_bins'] or cache['history_length'] != cfg['history_length']):
            raise ValueError('Histogram contract mismatch')
        if cfg.get('pretrained_content'):
            cfg['content_checkpoint_sha256'] = best['checkpoint_sha256']
        configs[name] = cfg
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.output_dir / 'configs'
    config_dir.mkdir()
    for name, cfg in configs.items():
        (config_dir / (name + '.json')).write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    state = dict(planned=names, completed=[], active=None, failed=None,
        store_manifest_sha256=manifest_hash, content_report_sha256=sha256(content_report),
        selected_epoch=best['epoch'], selection='maximum validation GAUC; earliest epoch on ties',
        checkpoint_sha256=best['checkpoint_sha256'], started=time.time())

    def save():
        temporary = args.output_dir / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(args.output_dir / 'status.json')

    save()
    for name in names:
        state['active'] = name
        save()
        command = [sys.executable, '-u', str(root / 'scripts/train_c0.py'),
            '--store', str(args.store.resolve()), '--config', str((config_dir / (name + '.json')).resolve()),
            '--output-dir', str((args.output_dir / name).resolve())]
        if configs[name].get('simtier_bins'):
            command.extend(['--hist-cache', str(args.hist_cache.resolve())])
        if configs[name].get('pretrained_content'):
            command.extend(['--content-checkpoint', str(checkpoint.resolve())])
        try:
            with (args.output_dir / (name + '.log')).open('w', encoding='utf-8') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((args.output_dir / name / 'report.json').read_text())
            if (report['store_manifest_sha256'] != manifest_hash or report['test_evaluated']
                or report['reload_max_abs_error'] != 0
                or report['validation']['rows'] != manifest['splits']['validation']['rows']
                or sum(e['rows'] for e in report['epochs']) != manifest['splits']['train']['rows'] * configs[name]['epochs']):
                raise ValueError('Completed run failed integrity checks')
        except Exception as exc:
            state.update(active=None, failed=name, error=str(exc))
            save()
            raise
        state['completed'].append(name)
        state['active'] = None
        save()
    state['finished'] = time.time()
    save()


if __name__ == '__main__':
    main()
