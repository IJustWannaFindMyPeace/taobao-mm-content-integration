"""Audit saved core predictions and bootstrap paired user-level GAUC differences."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from metrics import evaluate, auc
from fetch_source import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.store/'manifest.json').read_text())
    store_hash = sha256(args.store/'manifest.json')
    labels = np.load(args.store/'validation.labels.npy')
    users = np.load(args.store/'validation.user_rows.npy')
    source_rows = np.load(args.store/'validation.source_rows.npy')
    names = ['c0','c2','c3','c5','c6','c7']
    reports, logits, checks = {}, {}, {}
    for name in names:
        path = args.runs/name/'report.json'
        r = json.loads(path.read_text())
        if r['store_manifest_sha256'] != store_hash or r['test_evaluated'] or r['reload_max_abs_error'] != 0 or not r['parameters_updated']:
            raise ValueError('Invalid run contract: '+name)
        if len(r['epochs']) != 1 or r['epochs'][0]['rows'] != manifest['splits']['train']['rows']:
            raise ValueError('Incomplete training: '+name)
        if name != 'c0':
            for key in ['seed','embedding_width','history_length','short_length','batch_size','epochs','dense_learning_rate','sparse_learning_rate','weight_decay']:
                if r['config'][key] != reports['c0']['config'][key]:
                    raise ValueError('Common configuration mismatch: '+key)
            if r['implementation_sha256'] != reports['c0']['implementation_sha256']:
                raise ValueError('Implementation changed across runs')
        prediction = args.runs/name/'validation_predictions.npz'
        with np.load(prediction) as saved:
            for key, expected in [('labels',labels),('user_rows',users),('source_rows',source_rows)]:
                np.testing.assert_array_equal(saved[key], expected)
            logits[name] = saved['logits']
        if not np.isfinite(logits[name]).all():
            raise ValueError('Nonfinite predictions')
        measured = evaluate(labels, logits[name], users)
        for key in ['auc','gauc','logloss','rows','valid_users','valid_rows']:
            if measured[key] != r['validation'][key]:
                raise ValueError('Metric mismatch: '+name+'/'+key)
        reports[name] = r
        checks[name] = dict(metrics=measured, report_sha256=sha256(path), prediction_sha256=sha256(prediction),
            training_seconds=r['training_seconds'], peak_allocated_mib=r['training_peak_allocated_mib'],
            saved_rows_and_metrics_verified=True)
        print('Audited '+name, flush=True)
    if reports['c5']['content_checkpoint_sha256'] != reports['c6']['content_checkpoint_sha256']:
        raise ValueError('MAKE checkpoint mismatch')
    if reports['c3']['histogram_cache_sha256'] != reports['c6']['histogram_cache_sha256']:
        raise ValueError('Histogram cache mismatch')
    groups = {}
    for i, user in enumerate(users):
        groups.setdefault(int(user), []).append(i)
    weights = np.zeros(len(groups))
    user_auc = {name:np.zeros(len(groups)) for name in names}
    for group, indices in enumerate(groups.values()):
        if labels[indices].min() == labels[indices].max():
            continue
        weights[group] = len(indices)
        for name in names:
            user_auc[name][group] = auc(labels[indices], logits[name][indices])
    pairs = [('c2','c0'),('c3','c0'),('c5','c0'),('c5','c3'),('c6','c3'),('c6','c5'),('c7','c3'),('c7','c6')]
    deltas = {(a,b):weights*(user_auc[a]-user_auc[b]) for a,b in pairs}
    samples = {pair:[] for pair in pairs}
    rng = np.random.default_rng(2026)
    for _ in range(20):
        draw = rng.integers(0,len(weights),size=(100,len(weights)))
        denominator = weights[draw].sum(1)
        if np.any(denominator == 0):
            raise ValueError('No effective bootstrap users')
        for pair in pairs:
            samples[pair].extend((deltas[pair][draw].sum(1)/denominator).tolist())
    comparisons = {}
    for a,b in pairs:
        comparisons[a+'-minus-'+b] = dict(
            delta={key:checks[a]['metrics'][key]-checks[b]['metrics'][key] for key in ['auc','gauc','logloss']},
            gauc_delta_user_bootstrap_95_percentile=np.quantile(samples[(a,b)],[.025,.975]).tolist())
    result = dict(scope='single_seed_validation_core_audit', store_manifest_sha256=store_hash,
        source_sha256=sha256(Path(__file__)), seed=reports['c0']['config']['seed'], models=checks,
        comparisons=comparisons, bootstrap_replicates=2000, bootstrap_seed=2026,
        bootstrap_unit='all validation users; single-label users have zero GAUC weight',
        limitations=['conditional on fixed trained models and selected cohort',
            'does not quantify training-seed or cohort-selection variability',
            'multiple comparison intervals are not multiplicity adjusted',
            'validation evidence only; no held-out test evaluation'])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(comparisons,indent=2),flush=True)


if __name__ == '__main__':
    main()
