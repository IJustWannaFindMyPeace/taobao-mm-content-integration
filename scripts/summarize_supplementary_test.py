"""Verify completed supplemental artifacts and publish portable result summaries."""
import argparse
import csv
import io
import json
from pathlib import Path

from supplementary_test_contract import digest, validate_freeze


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def direction(report):
    lo, hi = report['gauc_delta_user_bootstrap_95_percentile']
    return 'positive' if lo > 0 else 'negative' if hi < 0 else 'crosses_zero'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    content = Path(__file__).resolve().parents[1]
    roots = {'content': content, 'muse': content.parent / 'taobao-mm-muse-repro'}
    run = work / 'supplementary-heldout-test-v1'
    state = read(run / 'status.json')
    inventory_path = work / 'supplementary-test-candidates-v1.json'
    inventory = read(inventory_path)
    frozen = validate_freeze(run / 'freeze.json', inventory_path, work / 'shared-feature-store-050', content)
    freeze_hash = digest(run / 'freeze.json')
    assert state['phase'] == 'completed' and state['failed'] is None
    assert set(state['completed_models']) == set(inventory['models']) == set(frozen['models'])
    assert sorted(state['completed_comparisons']) == list(range(93))
    assert state['freeze_sha256'] == freeze_hash
    assert len(inventory['models']) == 73
    records = {}
    for key, entry in inventory['models'].items():
        folder = run / 'models' / key
        report = read(folder / 'report.json')
        assert digest(folder / 'report.json') == state['model_report_sha256'][key]
        assert report['freeze_sha256'] == freeze_hash
        assert report['test_evaluated'] and not report['optimizer_created']
        assert report['model_sha256'] == entry['model_sha256']
        assert report['training_report_sha256'] == entry['training_report_sha256']
        assert report['rows'] == report['metrics']['rows'] == 923211
        assert digest(folder / 'test_predictions.npz') == report['test_predictions_sha256']
        replay_path = work / 'supplementary-validation-replays-v1' / key / 'report.json'
        replay = read(replay_path)
        assert digest(replay_path) == frozen['replay_report_sha256'][key]
        assert replay['rows'] == 768773 and replay['max_abs_logit_error'] == 0
        records[key] = dict(family=entry['family'], config=entry['config'],
                            histogram_mode=entry['histogram_mode'],
                            validation_metrics=replay['metrics'], test_metrics=report['metrics'],
                            model_sha256=report['model_sha256'],
                            report_sha256=state['model_report_sha256'][key],
                            predictions_sha256=report['test_predictions_sha256'])
    reference_reports = {}
    for key, entry in frozen['references'].items():
        folder = Path(entry['run'])
        assert digest(folder / 'report.json') == entry['report_sha256']
        assert digest(folder / 'test_predictions.npz') == entry['predictions_sha256']
        reference_reports[key] = read(folder / 'report.json')
    comparisons = []
    for index, pair in enumerate(frozen['comparisons']):
        name = f'pair-{index:03d}'
        path = run / 'comparisons' / (name + '.json')
        assert digest(path) == state['comparison_report_sha256'][name]
        report = read(path)
        assert report['pair'] == pair and report['pair_index'] == index
        assert report['freeze_sha256'] == freeze_hash and report['test_evaluated']
        assert report['rows'] == 923211 and report['bootstrap_replicates'] == 2000
        for role in ('baseline', 'candidate'):
            key = pair[role]
            source = reference_reports[key]['metrics'] if key.startswith('main:') else records[key]['test_metrics']
            assert report[role] == source
        report['test_gauc_direction'] = direction(report)
        source = pair['source']
        if source.startswith(('content/', 'muse/')):
            project, relative = source.split('/', 1)
            filename, tail = relative.split('.json/', 1)
            validation = read(roots[project] / 'reports' / (filename + '.json'))
            for part in tail.split('/'):
                validation = validation[part]
            report['validation_comparison'] = dict(
                candidate_minus_baseline=validation['candidate_minus_baseline'],
                gauc_delta_user_bootstrap_95_percentile=validation['gauc_delta_user_bootstrap_95_percentile'],
                gauc_direction=direction(validation))
        comparisons.append(report)
    assert sum('validation_comparison' in r for r in comparisons) == 84
    summary = dict(scope='completed_supplementary_test_evaluation',
                   original_freeze_sha256=freeze_hash, models=records, comparisons=comparisons,
                   main_references={k: dict(report_sha256=v['report_sha256'],
                                           predictions_sha256=v['predictions_sha256'])
                                    for k, v in frozen['references'].items()},
                   checks=dict(validation_exact_replays=73, verified_test_models=73,
                               verified_test_comparisons=93, matched_validation_comparisons=84,
                               prediction_file_hashes_verified=True, frozen_dependencies_verified=True,
                               frozen_test_arrays_verified=True),
                   limitations=[
                       'Additional test evaluation after main test was inspected; not a new blind test.',
                       'Same 5% user cohort and three seeds; no new cohort replication.',
                       'Pointwise 2000-replicate user bootstrap intervals; no multiplicity correction.',
                       'ID epoch2 checkpoint was not retained; only its validation diagnostics remain.',
                       'Durations are excluded from performance comparisons due to concurrent external load.',
                       'Local downstream adaptations; no reconstruction of unavailable original representations or industrial systems.'])
    # Public projection preserves identifiers and hashes, but not machine-specific paths.
    public_freeze = json.loads(json.dumps(frozen))
    for section in ('models', 'references'):
        for entry in public_freeze[section].values():
            entry['run'] = Path(entry['run']).relative_to(work).as_posix()
    public_freeze['publication_note'] = 'Portable projection; path fields are work-relative. Its file hash differs from the original execution freeze.'
    public_freeze['original_execution_freeze_sha256'] = freeze_hash
    csv_buffer = io.StringIO(newline='')
    writer = csv.writer(csv_buffer)
    writer.writerow(['model_id', 'family', 'seed', 'validation_auc', 'validation_gauc', 'validation_logloss',
                     'test_auc', 'test_gauc', 'test_logloss'])
    for key, r in records.items():
        writer.writerow([key, r['family'], r['config']['seed']] +
                        [r[split][m] for split in ('validation_metrics', 'test_metrics')
                         for m in ('auc', 'gauc', 'logloss')])
    table = ['# 追加测试：完整配对结果', '',
             '差值均为候选减基线。正向/负向表示单项 95% 用户 bootstrap 区间完全位于零的一侧；跨零不等于两者等效。93 项比较未作多重校正。', '',
             '模型标识与完整配置见 [结构化报告](../reports/supplementary-test-summary.json)。前 84 项附原验证区间方向；最后 9 项按预先冻结契约执行。', '',
             '|编号|基线 → 候选|测试 ΔGAUC|95% 区间|验证 → 测试方向|', '|---|---|---:|---|---|']
    labels = {'positive': '正向', 'negative': '负向', 'crosses_zero': '跨零'}
    for r in comparisons:
        lo, hi = r['gauc_delta_user_bootstrap_95_percentile']
        before = labels[r['validation_comparison']['gauc_direction']] if 'validation_comparison' in r else '未在此映射'
        table.append(f"|{r['pair_index']:03d}|`{r['pair']['baseline']}` → `{r['pair']['candidate']}`|{r['candidate_minus_baseline']['gauc']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|{before} → {labels[r['test_gauc_direction']]}|")
    for root in roots.values():
        (root / 'reports/supplementary-test-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (root / 'reports/supplementary-test-freeze-public.json').write_text(json.dumps(public_freeze, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (root / 'reports/supplementary-test-models.csv').write_text(csv_buffer.getvalue(), encoding='utf-8')
        (root / 'docs/supplementary-test-comparisons.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    print(json.dumps(summary['checks']))
    print('Validation-to-test interval-direction changes:')
    for r in comparisons:
        if 'validation_comparison' in r and r['test_gauc_direction'] != r['validation_comparison']['gauc_direction']:
            print(r['pair_index'], r['pair']['source'], r['validation_comparison']['gauc_direction'], '->', r['test_gauc_direction'])


if __name__ == '__main__':
    main()
