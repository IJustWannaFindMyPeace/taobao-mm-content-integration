"""Run exact full validation replays before freezing supplementary test evaluation."""
import argparse,json,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 for key in ('inventory','store','output-dir'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();out=a.output_dir
 if out.exists():raise ValueError('Fresh output required')
 out.mkdir(parents=True);state=dict(phase='waiting_inventory',completed=[],active=None,failed=None,test_evaluated=False)
 def save():
  t=out/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(out/'status.json')
 save();deadline=time.time()+1800
 try:
  while not a.inventory.exists():
   if time.time()>deadline:raise TimeoutError('Inventory not ready within 30 minutes')
   time.sleep(10)
  inv=json.loads(a.inventory.read_text());state['planned']=list(inv['models']);state['phase']='validation_replay';save()
  for key in state['planned']:
   state['active']=key;save()
   with (out/f'{key}.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,str(Path(__file__).with_name('replay_supplementary.py')),'--inventory',str(a.inventory.resolve()),'--store',str(a.store.resolve()),'--model-id',key,'--output-dir',str((out/key).resolve())],stdout=log,stderr=subprocess.STDOUT,check=True)
   r=json.loads((out/key/'report.json').read_text());assert r['max_abs_logit_error']==0 and not r['test_evaluated']
   state['completed'].append(key);state['active']=None;save()
  state['phase']='completed';save()
 except Exception as e:
  state.update(phase='failed',failed=dict(job=state['active'],error=repr(e)));save();raise
if __name__=='__main__':main()
