"""Run the epoch diagnostic and MAKE factorial study serially on one GPU."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from fetch_source import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--pretraining-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,help='Explicit common downstream protocol; diagnostic epochs remain unchanged')
    parser.add_argument('--supplement-only',action='store_true',help='Reuse the existing frozen pretrained control; run only missing controls and epoch diagnostic')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh study directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    jobs=[('c4-probe','c4',20),('c5-probe','c5',20),('c0-epochs3','c0-epochs3',None),
          ('c4','c4',None),('c4-frozen','c4-frozen',None),('c5','c5',None),('c5-joint','c5-joint',None)]
    if args.supplement_only:
        jobs=[('c0-epochs3','c0-epochs3',None),('c4','c4',None),('c4-frozen','c4-frozen',None),('c5-joint','c5-joint',None)]
    pretraining=json.loads((args.pretraining_run/'report.json').read_text())
    protocol=json.loads(args.protocol.read_text()) if args.protocol else None
    if protocol and (protocol['store_manifest_sha256']!=sha256(args.store/'manifest.json') or pretraining['config']['seed'] not in protocol['training_seeds']):
        raise ValueError('Common protocol data or seed mismatch')
    if pretraining['store_manifest_sha256']!=sha256(args.store/'manifest.json') or pretraining['test_evaluated'] or len(pretraining['epochs'])!=pretraining['config']['epochs']:
        raise ValueError('Require complete same-store validation-only pretraining')
    best=max(pretraining['epochs'],key=lambda e:(e['validation']['gauc'],-e['epoch']))
    checkpoint=args.pretraining_run/f"epoch-{best['epoch']}.pt"
    if sha256(checkpoint)!=best['checkpoint_sha256']:raise ValueError('Pretraining checkpoint digest mismatch')
    snapshots=args.output_dir/'configs';snapshots.mkdir()
    for config in {job[1] for job in jobs}:
        cfg=json.loads((root/'configs'/f'{config}.json').read_text())
        if protocol:
            for key in ['batch_size','dense_learning_rate','sparse_learning_rate','weight_decay','embedding_width','resource_limits']:
                cfg[key]=protocol[key]
        cfg['seed']=pretraining['config']['seed']
        if cfg['history_length']!=pretraining['config']['history_length']:raise ValueError('History mismatch')
        if cfg.get('pretrained_content'):cfg['content_checkpoint_sha256']=best['checkpoint_sha256']
        (snapshots/f'{config}.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    status={'completed':[],'active':None,'failed':None,'planned':[job[0] for job in jobs]}
    if protocol:
        status['protocol_sha256']=sha256(args.protocol)
        (args.output_dir/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    def save():
        temporary=args.output_dir/'status.tmp'
        temporary.write_text(json.dumps(status,indent=2),encoding='utf-8')
        temporary.replace(args.output_dir/'status.json')
    for name,config,limit in jobs:
        status['active']=name;save()
        cfg=json.loads((snapshots/f'{config}.json').read_text())
        command=[sys.executable,str(root/'scripts/train_c0.py'),'--store',str(args.store.resolve()),
                 '--config',str((snapshots/f'{config}.json').resolve()),'--output-dir',str((args.output_dir/name).resolve())]
        if cfg.get('pretrained_content'):
            command+=['--content-checkpoint',str(checkpoint.resolve())]
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
