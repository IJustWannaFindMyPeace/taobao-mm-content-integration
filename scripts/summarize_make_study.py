"""Audit a completed study and produce paired comparisons without test access."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from fetch_source import sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--original-c0',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--frozen-pretrained',type=Path,help='Reuse a previously completed C5 run')
    parser.add_argument('--store',type=Path,help='Verify saved epoch predictions against the source store')
    args=parser.parse_args()
    status=json.loads((args.study/'status.json').read_text())
    if status['active'] or status['failed'] or set(status['completed'])!=set(status['planned']):
        raise ValueError('The entire study must complete successfully before comparison')
    names=['c0-epochs3','c4','c4-frozen','c5','c5-joint']
    paths={name:args.study/name for name in names}
    if args.frozen_pretrained:paths['c5']=args.frozen_pretrained
    reports={name:json.loads((paths[name]/'report.json').read_text()) for name in names}
    original=json.loads((args.original_c0/'report.json').read_text())
    common=['seed','embedding_width','history_length','short_length','batch_size',
            'dense_learning_rate','sparse_learning_rate','weight_decay']
    expected_rows=original['epochs'][0]['rows']
    checks={}
    if args.store:
        if sha256(args.store/'manifest.json')!=original['store_manifest_sha256']:raise ValueError('Source store mismatch')
        expected={key:np.load(args.store/('validation.'+key+'.npy')) for key in ['labels','source_rows','user_rows']}
    for name,report in reports.items():
        if report['scope']!='one_seed_validation_run' or report['test_evaluated']:
            raise ValueError(f'Invalid evaluation scope: {name}')
        if report['store_manifest_sha256']!=original['store_manifest_sha256']:
            raise ValueError(f'Store mismatch: {name}')
        if report['implementation_sha256']!=original['implementation_sha256']:raise ValueError('Training implementation mismatch')
        if any(report['config'][key]!=original['config'][key] for key in common):
            raise ValueError(f'Common training configuration mismatch: {name}')
        count=3 if name=='c0-epochs3' else 1
        if len(report['epochs'])!=count or any(e['rows']!=expected_rows for e in report['epochs']):
            raise ValueError(f'Incomplete training epochs: {name}')
        if not report['parameters_updated'] or report['reload_max_abs_error']!=0:
            raise ValueError(f'Update/reload verification failed: {name}')
        if args.store:
            checks[name]=[]
            for epoch in report['epochs']:
                path=paths[name]/f"validation-epoch-{epoch['epoch']}.npz"
                with np.load(path) as saved:
                    for key,value in expected.items():np.testing.assert_array_equal(saved[key],value)
                    measured=evaluate(expected['labels'],saved['logits'],expected['user_rows'])
                if measured!=epoch['validation']:raise ValueError('Saved epoch metrics mismatch')
                checks[name].append({'epoch':epoch['epoch'],'prediction_sha256':sha256(path),'metrics_recomputed_equal':True})
            with np.load(paths[name]/'validation_predictions.npz') as final,np.load(paths[name]/f"validation-epoch-{count}.npz") as last:
                for key in ['labels','source_rows','user_rows','logits']:np.testing.assert_array_equal(final[key],last[key])
        print('Audited '+name,flush=True)
    for name,pretrained,frozen in [('c4',False,False),('c4-frozen',False,True),('c5',True,True),('c5-joint',True,False)]:
        cfg=reports[name]['config']
        if not cfg['make_fusion'] or cfg['pretrained_content']!=pretrained or cfg['freeze_content']!=frozen:raise ValueError('Factorial switches mismatch')
    if reports['c5']['content_checkpoint_sha256']!=reports['c5-joint']['content_checkpoint_sha256']:raise ValueError('Pretraining checkpoint mismatch')
    with np.load(args.original_c0/'validation_predictions.npz') as old, np.load(args.study/'c0-epochs3/validation-epoch-1.npz') as new:
        for key in ['labels','source_rows','user_rows']:
            np.testing.assert_array_equal(old[key],new[key])
        error=float(np.max(np.abs(old['logits']-new['logits'])))
        np.testing.assert_allclose(old['logits'],new['logits'],rtol=0,atol=1e-6)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh output directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    predictions={name:paths[name]/'validation_predictions.npz' for name in names[1:]}
    predictions['c0']=args.study/'c0-epochs3/validation-epoch-1.npz'
    pairs=[('c4','c5-joint'),('c4-frozen','c5'),('c4-frozen','c4'),('c5','c5-joint')]
    pairs += [('c0',name) for name in names[1:]]
    comparisons={}
    for baseline,candidate in pairs:
        name=candidate+'-vs-'+baseline; path=args.output_dir/(name+'.json')
        command=[sys.executable,str(Path(__file__).with_name('compare_validation.py')),
                 '--baseline',str(predictions[baseline]),'--candidate',str(predictions[candidate]),'--report',str(path)]
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
        comparisons[name]=json.loads(path.read_text())
        print('Compared '+name,flush=True)
    result={'scope':'completed_single_seed_validation_study','original_c0_epoch1_max_abs_error':error,
            'report_sha256':{name:sha256(paths[name]/'report.json') for name in names},
            'saved_prediction_checks':checks,'audit_source_sha256':sha256(Path(__file__)),
            'c0_epochs':reports['c0-epochs3']['epochs'],
            'comparisons':comparisons,'test_evaluated':False,
            'limitations':['exploratory multiple comparisons; intervals are not multiplicity-adjusted',
                           'bootstrap conditions on trained models and selected cohort',
                           'pretrained conditions use additional training compute']}
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Study audited; paired reports saved.',flush=True)


if __name__=='__main__':main()
