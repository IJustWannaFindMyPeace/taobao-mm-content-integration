"""Run exact full validation replays before freezing supplementary test evaluation."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 for key in ('inventory','store','output-dir'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--resume',action='store_true')
 a=p.parse_args();out=a.output_dir.resolve()
 if out.exists() and not a.resume:raise ValueError('Fresh output required')
 out.mkdir(parents=True,exist_ok=True)
 state=json.loads((out/'status.json').read_text()) if a.resume else dict(phase='waiting_inventory',completed=[],active=None,failed=None,test_evaluated=False)
 if a.resume:
  stamp=str(time.time_ns());(out/f'status-before-resume-{stamp}.json').write_text(json.dumps(state,indent=2))
  inventory_hash=hashlib.sha256(a.inventory.read_bytes()).hexdigest()
  evaluator_hash=hashlib.sha256(Path(__file__).with_name('replay_supplementary.py').read_bytes()).hexdigest()
  for key in state['completed']:
   r=json.loads((out/key/'report.json').read_text())
   assert r['inventory_sha256']==inventory_hash and r['evaluator_sha256']==evaluator_hash and r['max_abs_logit_error']==0 and not r['test_evaluated']
  interrupted=state.get('active')
  if interrupted and interrupted not in state['completed']:
   source=(out/interrupted).resolve();target=(out/(interrupted+'.interrupted-'+stamp)).resolve()
   assert source.parent==out and target.parent==out
   if source.exists():source.rename(target)
   log=out/f'{interrupted}.log'
   if log.exists():log.rename(out/f'{interrupted}.interrupted-{stamp}.log')
  state.update(active=None,failed=None)
 def save():
  t=out/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(out/'status.json')
 save();deadline=time.time()+1800
 try:
  while not a.inventory.exists():
   if time.time()>deadline:raise TimeoutError('Inventory not ready within 30 minutes')
   time.sleep(10)
  inv=json.loads(a.inventory.read_text());state['planned']=list(inv['models']);state['phase']='validation_replay';save()
  for key in state['planned']:
   if key in state['completed']:continue
   state['active']=key;save()
   with (out/f'{key}.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,str(Path(__file__).with_name('replay_supplementary.py')),'--inventory',str(a.inventory.resolve()),'--store',str(a.store.resolve()),'--model-id',key,'--output-dir',str((out/key).resolve())],stdout=log,stderr=subprocess.STDOUT,check=True)
   r=json.loads((out/key/'report.json').read_text());assert r['max_abs_logit_error']==0 and not r['test_evaluated']
   state['completed'].append(key);state['active']=None;save()
  state['phase']='completed';save()
 except Exception as e:
  state.update(phase='failed',failed=dict(job=state['active'],error=repr(e)));save();raise
if __name__=='__main__':main()
