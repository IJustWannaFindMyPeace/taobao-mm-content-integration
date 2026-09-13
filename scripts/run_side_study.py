"""Serial complete-epoch Vector and SimScore validation runs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    state={'planned':['c2','c7'],'completed':[],'active':None,'failed':None}
    def save():
        tmp=args.output_dir/'status.tmp'
        tmp.write_text(json.dumps(state,indent=2));tmp.replace(args.output_dir/'status.json')
    for name in state['planned']:
        state['active']=name;save();print('Starting '+name,flush=True)
        with (args.output_dir/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(root/'scripts/train_c0.py'),
                '--store',str(args.store.resolve()),'--config',str(root/'configs'/(name+'.json')),
                '--output-dir',str((args.output_dir/name).resolve())],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state['failed']=name;state['active']=None;save()
            raise RuntimeError(name+' failed; later runs were not started')
        state['completed'].append(name);state['active']=None;save()
        print('Completed '+name,flush=True)


if __name__=='__main__':main()
