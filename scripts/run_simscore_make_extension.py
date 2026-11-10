"""Train all fixed seeds, freeze, evaluate and compare the additional combination."""
import sys,json,time,subprocess,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORK=Path(sys.argv[1]).resolve()
STORE=Path(sys.argv[2]).resolve()
BASE=WORK.parent
SEEDS=[42,2026,3407]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
state={'started':time.time(),'completed':[],'active':None}
def save():
 p=WORK/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(WORK/'status.json')
def run(name,cmd):
 state['active']=name;save()
 with (WORK/(name+'.log')).open('w',encoding='utf-8') as log:
  subprocess.run([sys.executable,'-u',*map(str,cmd)],stdout=log,stderr=subprocess.STDOUT,check=True)
 state['completed'].append(name);save()
try:
 run('unit-audit',[ROOT/'scripts/audit_simscore_make.py',WORK/'unit-audit.json'])
 for seed in SEEDS:
  pre=BASE/f'c1-050-seed{seed}-epochs3';p=json.loads((pre/'report.json').read_text());best=max(p['epochs'],key=lambda x:(x['validation']['gauc'],-x['epoch']))
  run(f'train-{seed}',[ROOT/'scripts/train_simscore_make.py','--store',STORE,'--config',WORK/'configs'/f'{seed}.json','--output-dir',WORK/f'seed{seed}','--content-checkpoint',pre/f"epoch-{best['epoch']}.pt"])
  report=json.loads((WORK/f'seed{seed}'/'report.json').read_text());assert report['epochs'][0]['rows']==3052475 and report['reload_max_abs_error']==0
 frozen={str(seed):{'model_sha256':sha(WORK/f'seed{seed}'/'model.pt'),'report_sha256':sha(WORK/f'seed{seed}'/'report.json')} for seed in SEEDS}
 (WORK/'frozen.json').write_text(json.dumps({'time':time.time(),'models':frozen,'protocol_sha256':sha(WORK/'protocol.json')},indent=2))
 for seed in SEEDS:
  run(f'test-{seed}',[ROOT/'scripts/evaluate_simscore_make.py','--store',STORE,'--run',WORK/f'seed{seed}'])
  assert sha(WORK/f'seed{seed}'/'model.pt')==frozen[str(seed)]['model_sha256']
  for ref in ['c7','c5','c6','c0']:
   run(f'compare-{ref}-{seed}',[ROOT/'scripts/compare_test.py','--baseline',BASE/'final-heldout-test-v1'/f'{ref}-seed{seed}'/'test_predictions.npz','--candidate',WORK/f'seed{seed}'/'test_predictions.npz','--report',WORK/f'comparison-{ref}-{seed}.json'])
 state['active']=None;state['finished']=time.time();save()
except Exception as e:
 state['error']=repr(e);save();raise
