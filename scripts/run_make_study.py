"""Run the epoch diagnostic and MAKE factorial study serially on one GPU."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--pretraining-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh study directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    jobs=[('c4-probe','c4',20),('c5-probe','c5',20),('c0-epochs3','c0-epochs3',None),
          ('c4','c4',None),('c4-frozen','c4-frozen',None),('c5','c5',None),('c5-joint','c5-joint',None)]
    status={'completed':[],'active':None,'failed':None,'planned':[job[0] for job in jobs]}
    def save():
        temporary=args.output_dir/'status.tmp'
        temporary.write_text(json.dumps(status,indent=2),encoding='utf-8')
        temporary.replace(args.output_dir/'status.json')
    for name,config,limit in jobs:
        status['active']=name;save()
        cfg=json.loads((root/'configs'/f'{config}.json').read_text())
        command=[sys.executable,str(root/'scripts/train_c0.py'),'--store',str(args.store.resolve()),
                 '--config',str(root/'configs'/f'{config}.json'),'--output-dir',str((args.output_dir/name).resolve())]
        if cfg.get('pretrained_content'):
            command+=['--content-checkpoint',str((args.pretraining_run/'epoch-2.pt').resolve())]
        if limit:command+=['--max-steps',str(limit)]
        print(f'Starting {name}',flush=True)
        with (args.output_dir/f'{name}.log').open('w',encoding='utf-8') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            status['failed']={'job':name,'exit_code':code};status['active']=None;save()
            raise RuntimeError(f'{name} failed; inspect its log. Later jobs were not started.')
        report=json.loads((args.output_dir/name/'report.json').read_text())
        if not report['parameters_updated'] or report['reload_max_abs_error']!=0:
            status['failed']={'job':name,'reason':'verification failed'};status['active']=None;save()
            raise RuntimeError('Parameter update or reload verification failed')
        status['completed'].append(name);status['active']=None;save()
        print(f'Completed {name}',flush=True)


if __name__=='__main__':main()
