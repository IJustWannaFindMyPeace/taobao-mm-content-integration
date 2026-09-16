"""Descriptive validation diagnostics by content availability; no model training."""
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
    groups = {'target_available': store.target_content != 0,
              'target_missing': store.target_content == 0}
    for length in [200, 1000]:
        valid_count = np.zeros(len(users), dtype=np.int64)
        covered_count = np.zeros(len(users), dtype=np.int64)
        for start in range(0, len(users), 2048):
            stop = min(start + 2048, len(users))
            selected = users[start:stop]
            valid = store.hist_item[selected, -length:] != 0
            covered = store.hist_content[selected, -length:] != 0
            valid_count[start:stop] = valid.sum(axis=1)
            covered_count[start:stop] = (valid & covered).sum(axis=1)
        ratio = np.divide(covered_count, valid_count,
                          out=np.zeros(len(users), dtype=float), where=valid_count > 0)
        masks = {'empty': valid_count == 0,
                 '0': (valid_count > 0) & (covered_count == 0),
                 '(0,0.5]': (ratio > 0) & (ratio <= .5),
                 '(0.5,0.9]': (ratio > .5) & (ratio <= .9),
                 '(0.9,1]': ratio > .9}
        assert np.all(np.sum(list(masks.values()), axis=0) == 1)
        groups.update({f'history_L{length}_{name}': mask[inverse]
                       for name, mask in masks.items()})
    report = {'scope': 'supplied_models_validation_content_coverage_diagnostic',
              'store_manifest_sha256': sha256(args.store / 'manifest.json'),
              'implementation_sha256': sha256(Path(__file__)),
              'group_definition': 'non-padding history records with content / non-padding records; repeated behaviors counted',
              'test_evaluated': False, 'models': {}, 'prediction_sha256': {},
              'limitations': ['descriptive AUC and LogLoss; no subgroup confidence intervals',
                             'history coverage uses fixed pre-retrieval windows, not selected top-K',
                             'groups differ in users, labels and history lengths; not causal effects of coverage',
                             'target-missing subgroup may be too small for reliable inference']}
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
