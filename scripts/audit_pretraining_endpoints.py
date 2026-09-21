"""Audit frozen MAKE endpoints with controlled pretraining duration."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    for name in ['p0','p1','p3','p5','pretraining','prefix-audit']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Fresh output directory required')
    manifest=json.loads((args.store/'manifest.json').read_text())
    expected={key:np.load(args.store/('validation.'+key+'.npy')) for key in ['labels','source_rows','user_rows']}
    paths={name:getattr(args,name) for name in ['p0','p1','p3','p5']}
    reports={}
    checks={}
    for name,path in paths.items():
        r=json.loads((path/'report.json').read_text());cfg=r['config']
        if r['store_manifest_sha256']!=digest(args.store/'manifest.json') or r['test_evaluated'] or r['reload_max_abs_error']!=0 or not r['parameters_updated']:
            raise ValueError('Run integrity failed: '+name)
        if cfg['epochs']!=1 or len(r['epochs'])!=1 or r['epochs'][0]['rows']!=manifest['splits']['train']['rows']:
            raise ValueError('Incomplete training: '+name)
        if r['steps']!=(manifest['splits']['train']['rows']+cfg['batch_size']-1)//cfg['batch_size']:
            raise ValueError('Incomplete training steps')
        with np.load(path/'validation_predictions.npz') as saved:
            for key,value in expected.items():np.testing.assert_array_equal(saved[key],value)
            logits=saved['logits']
        measured=evaluate(expected['labels'],logits,expected['user_rows'])
        if measured!=r['validation']:raise ValueError('Metric mismatch: '+name)
        reports[name]=r
        checks[name]={'report_sha256':digest(path/'report.json'),'prediction_sha256':digest(path/'validation_predictions.npz'),
            'all_rows_and_recomputed_metrics_equal':True,'metrics':measured,
            'training_seconds':r['training_seconds'],'peak_allocated_mib':r['training_peak_allocated_mib']}
        print('Audited '+name,flush=True)
    prefix=json.loads(args.prefix_audit.read_text())
    pretrain=json.loads((args.pretraining/'report.json').read_text())
    if prefix['test_evaluated'] or prefix['extended_report_sha256']!=digest(args.pretraining/'report.json'):raise ValueError('Prefix provenance mismatch')
    if len(prefix['prefix'])!=3 or any(x['max_prediction_error']!=0 or not x['all_weights_exact'] for x in prefix['prefix']):raise ValueError('Prefix not identical')
    normalize=lambda cfg:{k:v for k,v in cfg.items() if k not in ['experiment','pretrained_content','content_checkpoint_sha256']}
    for name,r in reports.items():
        c=r['config']
        if not c['make_fusion'] or not c['freeze_content'] or c['pretrained_content']!=(name!='p0'):raise ValueError('Endpoint switches mismatch')
        if normalize(c)!=normalize(reports['p0']['config']) or r['implementation_sha256']!=reports['p0']['implementation_sha256']:raise ValueError('Uncontrolled downstream change')
        if not (paths[name]/'model.pt').exists():raise ValueError('Missing weights')
        if name!='p0':
            epoch=int(name[1:]);weight=args.pretraining/f'epoch-{epoch}.pt'
            if digest(weight)!=c['content_checkpoint_sha256'] or digest(weight)!=pretrain['epochs'][epoch-1]['checkpoint_sha256']:raise ValueError('Endpoint weight mismatch')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    comparisons={}
    for baseline,candidate in [('p0','p1'),('p1','p3'),('p3','p5')]:
        name=candidate+'-vs-'+baseline
        output=args.output_dir/(name+'.json')
        subprocess.run([sys.executable,str(Path(__file__).with_name('compare_validation.py')),
            '--baseline',str(paths[baseline]/'validation_predictions.npz'),
            '--candidate',str(paths[candidate]/'validation_predictions.npz'),'--report',str(output)],check=True,stdout=subprocess.DEVNULL)
        comparisons[name]=json.loads(output.read_text())
        print('Compared '+name,flush=True)
    result={'scope':'single_seed_frozen_pretraining_endpoint_audit','store_manifest_sha256':digest(args.store/'manifest.json'),
        'audit_source_sha256':digest(Path(__file__)),'runs':checks,'comparisons':comparisons,'test_evaluated':False,
        'limitations':['fixed trained models and cohort; not training-seed variability',
            'three intervals without multiplicity correction','pretraining duration changes total compute; one seed; fixed P0/P1/P3/P5']}
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
