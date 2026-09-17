"""Descriptive validation diagnostics by available history length; no training."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from feature_store import FeatureStore
from metrics import auc
from fetch_source import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    store = FeatureStore(args.store, 'validation')
    users, inverse = np.unique(store.user_rows, return_inverse=True)
    counts = np.zeros(len(users), dtype=np.int64)
    for start in range(0, len(users), 2048):
        stop = min(start + 2048, len(users))
        counts[start:stop] = (store.hist_item[users[start:stop]] != 0).sum(axis=1)
    masks = {'0': counts == 0, '1-50': (counts > 0) & (counts <= 50),
             '51-200': (counts > 50) & (counts <= 200),
             '201-500': (counts > 200) & (counts <= 500),
             '501-1000': counts > 500}
    if not np.all(np.sum(list(masks.values()), axis=0) == 1):
        raise ValueError('History groups must partition users')
    groups = {name: mask[inverse] for name, mask in masks.items()}
    report = {'scope': 'supplied_models_validation_history_length_diagnostic',
              'store_manifest_sha256': sha256(args.store / 'manifest.json'),
              'implementation_sha256': sha256(Path(__file__)),
              'group_definition': 'non-padding records in supplied 1000-position history; OOV and repeated behaviors count; pre-retrieval',
              'test_evaluated': False, 'models': {}, 'prediction_sha256': {},
              'limitations': ['descriptive AUC and LogLoss; no subgroup confidence intervals',
                             'available history is capped at 1000, not true lifetime length',
                             'groups differ in users and labels; not causal effects of history length',
                             'small history groups may be too small for reliable inference']}
    for name, path in json.loads(args.models.read_text(encoding='utf-8')).items():
        with np.load(path) as predictions:
            for key in ['labels', 'source_rows', 'user_rows']:
                np.testing.assert_array_equal(predictions[key], getattr(store, key))
            logits = predictions['logits'].astype(np.float64)
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite predictions')
        values = {}
        for group, mask in groups.items():
            labels = store.labels[mask]
            scores = logits[mask]
            values[group] = {'rows': int(mask.sum()), 'users': int(len(np.unique(store.user_rows[mask]))),
                             'positives': int(labels.sum()),
                             'auc': auc(labels, scores) if len(labels) else None,
                             'logloss': float(np.mean(np.logaddexp(0, scores) - labels * scores)) if len(labels) else None}
        report['models'][name] = values
        report['prediction_sha256'][name] = sha256(Path(path))
        print('Audited ' + name, flush=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
