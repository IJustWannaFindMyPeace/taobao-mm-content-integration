"""Compare random joint MAKE with frozen pretrained endpoints; validation only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from audit_pretraining_endpoints import digest
from metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    for name in ('store', 'joint-zero', 'p1', 'p3', 'p5', 'strategy-audit',
                 'endpoint-audit', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Fresh output directory required')
    strategy = json.loads(args.strategy_audit.read_text())
    endpoint = json.loads(args.endpoint_audit.read_text())
    if strategy['test_evaluated'] or endpoint['test_evaluated']:
        raise ValueError('Expected validation-only provenance')
    paths = {'p0_joint': args.joint_zero, 'p1': args.p1, 'p3': args.p3, 'p5': args.p5}
    manifest_hash = digest(args.store / 'manifest.json')
    manifest = json.loads((args.store / 'manifest.json').read_text())
    expected = {k: np.load(args.store / f'validation.{k}.npy', mmap_mode='r')
                for k in ('labels', 'source_rows', 'user_rows')}
    checks = {}
    reference = None
    implementation = None
    for name, path in paths.items():
        report = json.loads((path / 'report.json').read_text())
        report_hash = digest(path / 'report.json')
        known = (strategy['report_sha256']['c4'] if name == 'p0_joint'
                 else endpoint['runs'][name]['report_sha256'])
        if report_hash != known or report['store_manifest_sha256'] != manifest_hash:
            raise ValueError('Provenance changed: ' + name)
        cfg = report['config']
        if (report['test_evaluated'] or cfg['epochs'] != 1
                or len(report['epochs']) != 1
                or report['epochs'][0]['rows'] != manifest['splits']['train']['rows']
                or report['reload_max_abs_error'] != 0 or not report['parameters_updated']):
            raise ValueError('Run integrity failed: ' + name)
        if not cfg['make_fusion'] or cfg['pretrained_content'] != (name != 'p0_joint') or cfg['freeze_content'] != (name != 'p0_joint'):
            raise ValueError('Strategy mismatch: ' + name)
        common = {k: v for k, v in cfg.items() if k not in
                  ('experiment', 'pretrained_content', 'freeze_content', 'content_checkpoint_sha256')}
        if reference is None:
            reference, implementation = common, report['implementation_sha256']
        if common != reference or report['implementation_sha256'] != implementation:
            raise ValueError('Uncontrolled downstream change: ' + name)
        with np.load(path / 'validation_predictions.npz') as saved:
            for key, values in expected.items():
                np.testing.assert_array_equal(saved[key], values)
            metrics = evaluate(expected['labels'], saved['logits'], expected['user_rows'])
        if metrics != report['validation']:
            raise ValueError('Metric mismatch: ' + name)
        checks[name] = {'report_sha256': report_hash,
                        'prediction_sha256': digest(path / 'validation_predictions.npz'),
                        'model_sha256': digest(path / 'model.pt'),
                        'metrics': metrics, 'config': cfg,
                        'all_rows_and_recomputed_metrics_equal': True}
        print('Audited ' + name, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    for name in ('p1', 'p3', 'p5'):
        output = args.output_dir / (name + '-vs-p0-joint.json')
        subprocess.run([sys.executable, str(Path(__file__).with_name('compare_validation.py')),
                        '--baseline', str(args.joint_zero / 'validation_predictions.npz'),
                        '--candidate', str(paths[name] / 'validation_predictions.npz'),
                        '--report', str(output)], check=True, stdout=subprocess.DEVNULL)
        comparisons[name + '-vs-p0-joint'] = json.loads(output.read_text())
        print('Compared ' + name, flush=True)
    result = {'scope': 'paper_style_zero_reference_adapted_single_seed_validation',
              'source': 'https://arxiv.org/html/2407.19467v1#S6.F6',
              'store_manifest_sha256': manifest_hash,
              'audit_source_sha256': digest(Path(__file__)),
              'strategy_audit_sha256': digest(args.strategy_audit),
              'endpoint_audit_sha256': digest(args.endpoint_audit),
              'runs': checks, 'comparisons': comparisons, 'test_evaluated': False,
              'limitations': ['post-hoc validation analysis; one training seed',
                             'joint zero vs frozen positive epochs changes both initialization and freezing',
                             'unequal total compute; adapted architecture, dataset and epoch grid',
                             '2000 user bootstrap samples per comparison; no multiplicity correction',
                             'not original paper numerical reproduction or evidence of monotonic improvement']}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
