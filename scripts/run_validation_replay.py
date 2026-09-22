"""Replay a fixed inventory serially; never open test data or train models."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from evaluate_checkpoint import digest


def main():
    p=argparse.ArgumentParser()
    for name in ('inventory','store','hist-cache','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    inventory=json.loads(args.inventory.read_text())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh queue directory required')
    if len({e['id'] for e in inventory['models']})!=len(inventory['models']):raise ValueError('Duplicate model id')
    for e in inventory['models']:
        if Path(e['id']).name!=e['id']:raise ValueError('Invalid model id')
        if digest(Path(e['run'])/'report.json')!=e['report_sha256']:raise ValueError('Training report changed')
    args.output_dir.mkdir(parents=True)
    evaluator=Path(__file__).with_name('evaluate_checkpoint.py')
    code_hash=digest(evaluator)
    state={'planned':[e['id'] for e in inventory['models']],'completed':[],'active':None,'failed':None,'test_evaluated':False}
    def save():
        t=args.output_dir/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(args.output_dir/'status.json')
    save();results={}
    for e in inventory['models']:
        state['active']=e['id'];save()
        try:
            if digest(evaluator)!=code_hash:raise ValueError('Evaluator changed during queue')
            output=args.output_dir/e['id']
            if e.get('reuse_replay'):
                output=Path(e['reuse_replay'])
            else:
                with (args.output_dir/(e['id']+'.log')).open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,str(evaluator),'--project',e['project'],'--family',e['family'],
                        '--run',e['run'],'--store',str(args.store.resolve()),'--hist-cache',str(args.hist_cache.resolve()),
                        '--output-dir',str(output.resolve())],check=True,stdout=log,stderr=subprocess.STDOUT)
            r=json.loads((output/'report.json').read_text())
            if (r['evaluator_sha256']!=code_hash or r['training_report_sha256']!=e['report_sha256']
                or r['max_abs_logit_error']!=0 or r['test_evaluated'] or r['optimizer_created']):
                raise ValueError('Replay audit failed')
            results[e['id']]={'report_sha256':digest(output/'report.json'),'replay_dir':str(output.resolve()),
                               'model_sha256':r['model_sha256'],'training_report_sha256':e['report_sha256']}
            state['completed'].append(e['id']);state['active']=None;save()
            print('Replayed '+e['id'],flush=True)
        except Exception as exc:
            state['failed']={'model':e['id'],'error':str(exc)};state['active']=None;save();raise
    (args.output_dir/'summary.json').write_text(json.dumps({'scope':'full_validation_replay_inventory',
        'inventory_sha256':digest(args.inventory),'evaluator_sha256':code_hash,'models':results,'test_evaluated':False},indent=2))


if __name__=='__main__':main()
