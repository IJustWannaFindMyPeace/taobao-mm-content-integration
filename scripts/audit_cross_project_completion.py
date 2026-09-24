"""Audit common protocol, checkpoint coverage and independently recompute test metrics."""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

from supplementary_test_contract import digest, validate_freeze


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def independent_metrics(labels, scores, users):
    """Vectorized average ranks, separate from production per-user metric loops."""
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    assert np.isin(y, [0, 1]).all() and np.isfinite(s).all()
    _, inverse, counts = np.unique(users, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=y)
    negatives = counts - positives
    valid = (positives > 0) & (negatives > 0)
    order = np.lexsort((s, inverse))
    sorted_groups, sorted_scores = inverse[order], s[order]
    starts = np.r_[0, np.flatnonzero((np.diff(sorted_groups) != 0) | (np.diff(sorted_scores) != 0)) + 1]
    ends = np.r_[starts[1:], len(y)]
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    rank = np.repeat((starts + ends + 1) / 2, ends - starts) - offsets[sorted_groups]
    rank_sum = np.bincount(sorted_groups, weights=y[order] * rank)
    user_auc = (rank_sum[valid] - positives[valid] * (positives[valid] + 1) / 2) / (positives[valid] * negatives[valid])
    total_positive = int(y.sum())
    global_order = np.argsort(s)
    tie_starts = np.r_[0, np.flatnonzero(np.diff(s[global_order]) != 0) + 1]
    tie_ends = np.r_[tie_starts[1:], len(y)]
    global_ranks = np.repeat((tie_starts + tie_ends + 1) / 2, tie_ends - tie_starts)
    global_auc = (np.sum(global_ranks[y[global_order] == 1]) - total_positive * (total_positive + 1) / 2) / (total_positive * (len(y) - total_positive))
    valid_rows = int(counts[valid].sum())
    return dict(rows=len(y), positives=total_positive, auc=float(global_auc),
                gauc=float(np.sum(user_auc * counts[valid]) / valid_rows),
                logloss=float(np.mean(np.logaddexp(0, s) - y * s)),
                users=len(counts), valid_users=int(valid.sum()), valid_rows=valid_rows,
                excluded_users=int((~valid).sum()), excluded_rows=len(y)-valid_rows,
                valid_row_fraction=valid_rows/len(y),
                gauc_weighting='exposure_count_among_users_with_both_labels')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    w = args.work.resolve()
    content = Path(__file__).resolve().parents[1]
    roots = {'content': content, 'muse': content.parent/'taobao-mm-muse-repro'}
    protocol = read(content/'configs/shared-downstream-protocol-v1.json')
    assert (content/'configs/shared-downstream-protocol-v1.json').read_bytes() == (roots['muse']/'configs/shared-downstream-protocol-v1.json').read_bytes()
    freeze = read(w/'final-test-freeze-v1.json')
    inventory = read(w/'final-evaluation-inventory-v1.json')
    assert digest(w/'final-evaluation-inventory-v1.json') == freeze['inventory_sha256']
    supplemental = validate_freeze(w/'supplementary-heldout-test-v1/freeze.json',
                                  w/'supplementary-test-candidates-v1.json', w/'shared-feature-store-050', content)
    si = read(w/'supplementary-test-candidates-v1.json')
    state = read(w/'supplementary-heldout-test-v1/status.json')
    assert state['phase'] == 'completed' and not state['active'] and not state['failed']
    for project, entries in freeze['source_sha256'].items():
        for name, expected in entries.items():
            assert digest(roots[project]/'src'/name) == expected
    for name, expected in freeze['test_arrays_sha256'].items():
        assert digest(w/'shared-feature-store-050'/name) == expected
    # Tie handling, single-class users, and unequal group weights in a small fixture.
    sys.path.insert(0, str(content/'src'))
    from metrics import evaluate
    fixture = ([0, 1, 1, 0, 1, 0, 0, 1], [0, 0, 2, 1, 0, 0, 3, 2], [0, 0, 1, 1, 1, 2, 2, 3])
    aa, bb = independent_metrics(*fixture), evaluate(*fixture)
    for k in aa:
        assert abs(aa[k]-bb[k]) < 1e-12 if isinstance(aa[k], float) else aa[k] == bb[k]
    canonical = {k: np.load(w/'shared-feature-store-050'/('test.'+k+'.npy'), mmap_mode='r')
                 for k in ('labels', 'source_rows', 'user_rows')}
    formal = []
    for e in inventory['models']:
        formal.append(('main', e['id'], Path(e['run']), freeze['models'][e['id']], w/'final-heldout-test-v1'/e['id']))
    for key, entry in si['models'].items():
        formal.append(('supplement', key, Path(entry['run']), entry, w/'supplementary-heldout-test-v1/models'/key))
    known = {str(run.resolve()).lower() for _, _, run, _, _ in formal}
    results = []
    for i, (phase, key, run, entry, output) in enumerate(formal):
        train = read(run/'report.json')
        assert digest(run/'report.json') == entry['training_report_sha256']
        assert train['config'] == entry['config']
        assert train['store_manifest_sha256'] == protocol['store_manifest_sha256']
        c = train['config']
        for field in ('batch_size', 'embedding_width', 'dense_learning_rate', 'sparse_learning_rate', 'weight_decay', 'resource_limits'):
            assert c[field] == protocol[field], (key, field)
        assert c['seed'] in protocol['training_seeds']
        assert train['steps'] == 11924*c['epochs']
        assert c['epochs'] == (3 if key.endswith('--c0-epochs3') else 1)
        report = read(output/'report.json')
        assert report['test_evaluated'] and not report['optimizer_created']
        assert report['model_sha256'] == entry['model_sha256']
        assert digest(output/'test_predictions.npz') == report['test_predictions_sha256']
        with np.load(output/'test_predictions.npz') as predictions:
            for field in canonical:
                np.testing.assert_array_equal(predictions[field], canonical[field])
            measured = independent_metrics(predictions['labels'], predictions['logits'], predictions['user_rows'])
        errors = {}
        for field, value in measured.items():
            if isinstance(value, float):
                errors[field] = abs(value - report['metrics'][field])
                assert errors[field] < 1e-12, (key, field, errors[field])
            else:
                assert value == report['metrics'][field], (key, field)
        results.append(dict(phase=phase, model_id=key, report_sha256=digest(output/'report.json'),
                            test_predictions_sha256=report['test_predictions_sha256'],
                            metric_absolute_errors=errors, rows=measured['rows']))
        if i % 10 == 0:
            print(f'Independent metric audit: {i+1}/{len(formal)}', flush=True)
    excluded = []
    for path in sorted(w.rglob('*.pt')):
        run = path.parent
        if path.name == 'model.pt' and str(run.resolve()).lower() in known:
            continue
        relative = path.relative_to(w).as_posix()
        if path.name.startswith('epoch-'):
            reason = 'upstream_content_pretraining_checkpoint'
        elif path.name != 'model.pt':
            reason = 'environment_or_prediction_artifact'
        elif 'probe' in relative:
            reason = 'resource_or_smoke_probe'
        elif relative.startswith(('batch-quality-', 'batch-confirmation-')):
            reason = 'validation_only_protocol_selection'
        elif 'scorer-seed' in relative:
            reason = 'upstream_retrieval_scorer'
        else:
            train = read(run/'report.json')
            assert train['config']['batch_size'] == 32, relative
            reason = 'superseded_exploration_batch32'
        excluded.append(dict(path=relative, reason=reason))
    common = ('final-heldout-test-summary.json','final-results-summary.json','supplementary-test-summary.json',
              'supplementary-test-freeze-public.json','supplementary-test-models.csv')
    for name in common:
        assert digest(content/'reports'/name) == digest(roots['muse']/'reports'/name)
    result = dict(scope='cross_project_protocol_coverage_and_independent_metrics_audit', date='2026-09-25',
                  all_passed=True, formal_model_entries=len(formal), independent_metric_checks=results,
                  enumerated_pt_files=len(formal)+len(excluded), excluded_artifacts=excluded,
                  exclusion_counts=dict(collections.Counter(e['reason'] for e in excluded)),
                  common_protocol_verified=True, shared_reports_identical=True,
                  max_metric_absolute_error=max(v for r in results for v in r['metric_absolute_errors'].values()),
                  limitations=['Not an exhaustive proof of all original-paper implementations.',
                               'Checkpoint weight hashes were verified during evaluation; this audit does not reread all weight bytes.',
                               'Does not rerun training, GPU inference, or all bootstrap draws.',
                               'Explicit pretraining/epoch/architecture budgets remain different research axes.'])
    for root in roots.values():
        (root/'reports/cross-project-completion-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('independent_metric_checks','excluded_artifacts')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
