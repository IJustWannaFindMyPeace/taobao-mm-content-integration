"""Check all frozen pair identities and Git checkout bytes without rerunning models."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from supplementary_test_contract import digest


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    w = a.work.resolve()
    content = Path(__file__).resolve().parents[1]
    roots = {'content': content, 'muse': content.parent/'taobao-mm-muse-repro'}
    main_freeze = read(w/'final-test-freeze-v1.json')
    supplement = read(w/'supplementary-heldout-test-v1/freeze.json')
    dependencies = dict(supplement['dependency_sha256'])
    for project, files in main_freeze['source_sha256'].items():
        for name, value in files.items():
            dependencies[project+'/src/'+name] = value
    for name, value in main_freeze['evaluation_support_sha256'].items():
        dependencies['content/scripts/'+name] = value
    dependencies['content/scripts/evaluate_checkpoint.py'] = main_freeze['evaluator_sha256']
    for key, expected in dependencies.items():
        project, relative = key.split('/', 1)
        assert digest(roots[project]/relative) == expected
        # The index is also valid before the metadata-fix commit is created.
        data = subprocess.check_output(['git', '-C', str(roots[project]), 'cat-file', '--filters',
                                        '--path='+relative, ':'+relative])
        assert hashlib.sha256(data).hexdigest() == expected, key
    checked = []
    for phase, frozen in [('main', main_freeze), ('supplement', supplement)]:
        for index, pair in enumerate(frozen['comparisons']):
            if phase == 'main':
                path = w/'final-heldout-test-v1/comparisons'/(pair['candidate']+'-vs-'+pair['baseline']+'.json')
            else:
                path = w/'supplementary-heldout-test-v1/comparisons'/f'pair-{index:03d}.json'
            comparison = read(path)
            assert comparison['rows'] == 923211 and comparison['bootstrap_replicates'] == 2000
            assert comparison['bootstrap_seed'] == 2026
            for role in ('baseline', 'candidate'):
                key = pair[role]
                if phase == 'main' or key.startswith('main:'):
                    folder = w/'final-heldout-test-v1'/key.removeprefix('main:')
                else:
                    folder = w/'supplementary-heldout-test-v1/models'/key
                report = read(folder/'report.json')
                assert comparison[role] == report['metrics']
                assert comparison[role+'_sha256'] == report['test_predictions_sha256']
            for metric in ('auc', 'gauc', 'logloss'):
                assert comparison['candidate_minus_baseline'][metric] == comparison['candidate'][metric]-comparison['baseline'][metric]
            lo, hi = comparison['gauc_delta_user_bootstrap_95_percentile']
            assert -1 <= lo <= hi <= 1
            checked.append(dict(phase=phase, pair_index=index, report_sha256=digest(path)))
    assert len(checked) == 159
    result = dict(scope='all_frozen_comparisons_and_checkout_bytes', date='2026-09-25', all_passed=True,
                  frozen_dependency_checkout_count=len(dependencies), dependency_sha256=dependencies,
                  comparisons=checked, comparison_count=len(checked),
                  limitations=['Verifies comparison provenance and arithmetic; does not regenerate bootstrap draws.',
                               'Git filters use the staged index and declared attributes; original model code and local freeze bytes are unchanged.'])
    for root in roots.values():
        (root/'reports/execution-contract-audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(f'PASS: {len(dependencies)} frozen checkout files; {len(checked)} comparison identities and deltas.')


if __name__ == '__main__':
    main()
