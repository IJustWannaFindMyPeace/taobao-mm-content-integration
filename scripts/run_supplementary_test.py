"""Serialize and safely resume supplemental test evaluation after replay acceptance."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from supplementary_test_contract import digest,single_writer,verify_dependencies

def read(p):return json.loads(p.read_text())
def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 w=a.work.resolve();out=a.output_dir.resolve();root=Path(__file__).resolve().parents[1];out.mkdir(parents=True,exist_ok=True)
 with single_writer(out/'.run.lock'):
  if (out/'status.json').exists() and not a.resume:raise ValueError('Existing run requires --resume')
  state=read(out/'status.json') if a.resume else dict(phase='waiting_validation',completed_models=[],completed_comparisons=[],active=None,failed=None,test_evaluated=False)
  if state.get('failed'):
   atomic(out/f'failure-history-{time.time_ns()}.json',state);state['failed']=None
  def save():atomic(out/'status.json',state)
  def command(name,args):
   state['active']=name;save()
   with (out/f'{name}.log').open('a',encoding='utf-8') as log:subprocess.run([sys.executable,*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
   state['active']=None;save()
  def archive_partial(folder):
   resolved=folder.resolve();assert out in resolved.parents
   if folder.exists():
    dest=folder.with_name(folder.name+'.interrupted-'+str(time.time_ns())).resolve();assert dest.parent==resolved.parent;folder.rename(dest)
  save()
  try:
   deadline=time.time()+12*3600
   while True:
    replay=read(w/'supplementary-validation-replays-v1/status.json')
    if replay['phase']=='failed':raise RuntimeError('Validation replay failed; test cannot start')
    if replay['phase']=='completed':break
    if time.time()>deadline:raise TimeoutError('Validation gate waiting exceeded 12 hours')
    time.sleep(10)
   inventory=w/'supplementary-test-candidates-v1.json';inv=read(inventory);contract=read(w/'supplementary-comparison-contract-v1.json');verify_dependencies(contract,root)
   # Confirm the entire construction and forward path is identical to original replay.
   old=(root/'scripts/replay_supplementary.py').read_text(encoding='utf-8-sig');new=(root/'scripts/evaluate_supplementary.py').read_text(encoding='utf-8-sig')
   left=old[old.index(' store=FeatureStore'):old.index(" with np.load(run/'validation_predictions.npz') as old:")]
   right=new[new.index(' store=FeatureStore'):new.index(' error=None')].replace('FeatureStore(a.store,a.split)',"FeatureStore(a.store,'validation')")
   if left!=right:raise ValueError('Evaluator forward path differs from verified replay')
   acceptance=out/'adapter-validation';state['phase']='adapter_acceptance';save()
   if not (acceptance/'report.json').exists():
    archive_partial(acceptance)
    command('adapter-acceptance',[root/'scripts/evaluate_supplementary.py','--inventory',inventory,'--store',w/'shared-feature-store-050','--model-id',next(iter(inv['models'])),'--output-dir',acceptance])
   r=read(acceptance/'report.json');assert r['max_abs_logit_error']==0 and not r['test_evaluated'] and r['evaluator_sha256']==digest(root/'scripts/evaluate_supplementary.py')
   frozenpath=out/'freeze.json'
   if not frozenpath.exists():command('freeze',[root/'scripts/freeze_supplementary_test.py','--work',w,'--acceptance',acceptance,'--output',frozenpath])
   frozen=read(frozenpath);freezehash=digest(frozenpath);verify_dependencies(frozen,root)
   state.update(phase='test_evaluation',freeze_sha256=freezehash);save()
   (out/'models').mkdir(exist_ok=True);(out/'comparisons').mkdir(exist_ok=True)
   state.setdefault('model_report_sha256',{});state.setdefault('comparison_report_sha256',{})
   for key,entry in inv['models'].items():
    verify_dependencies(frozen,root);folder=out/'models'/key;report=folder/'report.json'
    if not report.exists():
     if key in state['completed_models']:raise ValueError('Completed model report missing')
     archive_partial(folder)
     command('test-'+key,[root/'scripts/evaluate_supplementary.py','--inventory',inventory,'--store',w/'shared-feature-store-050','--model-id',key,'--split','test','--freeze',frozenpath,'--output-dir',folder])
    r=read(report)
    assert r['test_evaluated'] and r['rows']==923211 and r['model_id']==key and r['model_sha256']==entry['model_sha256'] and r['freeze_sha256']==freezehash
    assert r['config']==entry['config'] if 'config' in r else r['training_report_sha256']==entry['training_report_sha256']
    assert digest(folder/'test_predictions.npz')==r['test_predictions_sha256']
    if key in state['model_report_sha256']:assert digest(report)==state['model_report_sha256'][key]
    state['model_report_sha256'][key]=digest(report)
    if key not in state['completed_models']:state['completed_models'].append(key)
    state['test_evaluated']=True;save()
   state['phase']='paired_test_statistics';save()
   def path_for(key):return Path(frozen['references'][key]['run']) if key.startswith('main:') else out/'models'/key
   for index,pair in enumerate(frozen['comparisons']):
    name=f'pair-{index:03d}';report=out/'comparisons'/f'{name}.json';verify_dependencies(frozen,root)
    if not report.exists():
     if index in state['completed_comparisons']:raise ValueError('Completed comparison missing')
     args=[root/'scripts/compare_supplementary_test.py','--freeze',frozenpath,'--pair-index',index,'--report',report]
     for role in ('baseline','candidate'):
      folder=path_for(pair[role]);args+=['--'+role,folder/'test_predictions.npz','--'+role+'-report',folder/'report.json']
     command(name,args)
    r=read(report);assert r['test_evaluated'] and r['freeze_sha256']==freezehash and r['pair_index']==index and r['pair']==pair
    if name in state['comparison_report_sha256']:assert digest(report)==state['comparison_report_sha256'][name]
    state['comparison_report_sha256'][name]=digest(report)
    if index not in state['completed_comparisons']:state['completed_comparisons'].append(index)
    save()
   state.update(phase='completed',active=None);save()
  except Exception as e:
   state.update(phase='failed',failed=dict(job=state['active'],error=repr(e)));save();raise
if __name__=='__main__':main()
