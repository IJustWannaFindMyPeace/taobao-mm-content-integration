"""Serial inference and predefined comparisons for a frozen 54-model inventory."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from evaluate_checkpoint import digest


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser()
    for name in ('inventory','freeze','store','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    lock=read(args.freeze);lock_hash=digest(args.freeze)
    inventory=read(args.inventory);scripts=Path(__file__).parent
    if digest(args.inventory)!=lock['inventory_sha256']:raise ValueError('Inventory hash changed')
    if set(lock['models'])!={e['id'] for e in inventory['models']}:raise ValueError('Model set changed')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh output required; no silent test reruns')
    for name,value in lock['test_arrays_sha256'].items():
        if digest(args.store/name)!=value:raise ValueError('Frozen test input changed: '+name)
    for name,value in lock['evaluation_support_sha256'].items():
        if digest(scripts/name)!=value:raise ValueError('Frozen evaluation support changed')
    args.output_dir.mkdir(parents=True)
    state={'phase':'inference','planned':list(lock['models']),'completed':[],'active':None,'failed':None,
           'freeze_sha256':lock_hash,'comparisons_completed':[],'test_evaluation_started':True}
    def save():
        tmp=args.output_dir/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(args.output_dir/'status.json')
    save();reports={};comparisons={}
    try:
        for entry in inventory['models']:
            name=entry['id'];state['active']=name;save()
            if digest(args.freeze)!=lock_hash:raise ValueError('Freeze file mutated')
            with (args.output_dir/(name+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(scripts/'evaluate_checkpoint.py'),
                    '--project',entry['project'],'--family',entry['family'],'--run',entry['run'],
                    '--store',str(args.store.resolve()),'--split','test','--freeze',str(args.freeze.resolve()),
                    '--model-id',name,'--output-dir',str((args.output_dir/name).resolve())],
                    check=True,stdout=log,stderr=subprocess.STDOUT)
            r=read(args.output_dir/name/'report.json')
            if (r['freeze_sha256']!=lock_hash or r['rows']!=lock['test_rows'] or not r['test_evaluated']
                or r['optimizer_created'] or r['model_sha256']!=lock['models'][name]['model_sha256']):
                raise ValueError('Test report gate failed')
            reports[name]={'metrics':r['metrics'],'report_sha256':digest(args.output_dir/name/'report.json'),
                           'predictions_sha256':r['test_predictions_sha256'],'seconds':r['seconds']}
            state['completed'].append(name);state['active']=None;save()
            print('Evaluated '+name,flush=True)
        state['phase']='paired_comparisons';save()
        folder=args.output_dir/'comparisons';folder.mkdir()
        for pair in lock['comparisons']:
            name=pair['candidate']+'-vs-'+pair['baseline'];state['active']=name;save()
            for filename,value in lock['evaluation_support_sha256'].items():
                if digest(scripts/filename)!=value:raise ValueError('Comparison support changed')
            target=folder/(name+'.json')
            subprocess.run([sys.executable,str(scripts/'compare_test.py'),
                '--baseline',str(args.output_dir/pair['baseline']/'test_predictions.npz'),
                '--candidate',str(args.output_dir/pair['candidate']/'test_predictions.npz'),
                '--report',str(target)],check=True,stdout=subprocess.DEVNULL)
            comparison=read(target)
            for side in ('candidate','baseline'):
                if comparison[side+'_sha256']!=reports[pair[side]]['predictions_sha256']:
                    raise ValueError('Comparison prediction changed')
                if comparison[side]!=reports[pair[side]]['metrics']:raise ValueError('Comparison metrics changed')
            comparisons[name]=comparison
            state['comparisons_completed'].append(name);state['active']=None;save()
        (args.output_dir/'summary.json').write_text(json.dumps({'scope':'frozen_54_model_heldout_test',
            'freeze_sha256':lock_hash,'runs':reports,'comparisons':comparisons,'test_evaluated':True,
            'limitations':['fixed 5% user cohort; not full released dataset',
                'adapted downstream reproduction; no online evidence',
                '66 user-bootstrap intervals without multiplicity correction']},indent=2))
        state['phase']='complete';save()
    except Exception as exc:
        state['failed']={'active':state['active'],'error':str(exc)};save();raise


if __name__=='__main__':main()
