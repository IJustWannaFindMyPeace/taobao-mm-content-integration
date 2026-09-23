"""Run fixed C3/C6 SimTier boundary sensitivity, validation only."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from fetch_source import sha256


def read(p):return json.loads(p.read_text())


def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args();w=a.work.resolve();out=a.output_dir.resolve();root=Path(__file__).resolve().parents[1];store=w/'shared-feature-store-050'
 if out.exists():raise ValueError('Fresh output required')
 out.mkdir();(out/'configs').mkdir()
 inventory=read(w/'final-evaluation-inventory-v1.json');refs={x['id']:Path(x['run']) for x in inventory['models'] if x['id'].startswith(('c3-','c6-'))}
 sources=[root/'scripts/train_c0.py',root/'scripts/cache_simtier_boundaries.py',*sorted((root/'src').glob('*.py'))]
 hashes={p.name:sha256(p) for p in sources};manifest=sha256(store/'manifest.json')
 state=dict(phase='preflight',completed=[],active=None,failed=None,test_evaluated=False)
 def save():
  tmp=out/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'status.json')
 def execute(name,command):
  assert all(sha256(p)==hashes[p.name] for p in sources)
  state['active']=name;save();started=time.time()
  with (out/f'{name}.log').open('w',encoding='utf-8') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  state['completed'].append({'job':name,'seconds':time.time()-started});state['active']=None;save()
 try:
  reference_info={};configs={};checkpoints={}
  for key,path in refs.items():
   r=read(path/'report.json');assert r['store_manifest_sha256']==manifest and not r['test_evaluated']
   assert r['steps']==11924 and r['parameters_updated'] and r['reload_max_abs_error']==0
   assert r['histogram_cache_sha256']==sha256(w/'simtier-cache-050/manifest.json')
   assert all(hashes[k]==v for k,v in r['implementation_sha256'].items())
   configs[key]=r['config'];reference_info[key]=dict(report_sha256=sha256(path/'report.json'),predictions_sha256=sha256(path/'validation_predictions.npz'),run=str(path))
   if key.startswith('c6'):
    seed=r['config']['seed'];pre=w/f'c1-050-seed{seed}-epochs3'
    matches=[pre/f"epoch-{e['epoch']}.pt" for e in read(pre/'report.json')['epochs'] if e['checkpoint_sha256']==r['content_checkpoint_sha256']]
    assert len(matches)==1 and sha256(matches[0])==r['content_checkpoint_sha256'];checkpoints[key]=str(matches[0])
  assert len(refs)==6
  estimate=sum(read(path/'report.json')['training_seconds'] for path in refs.values())*2/3600
  assert estimate<6,'Review estimated training cost'
  (out/'plan.json').write_text(json.dumps(dict(reference=reference_info,configs=configs,source_hashes=hashes,manifest_sha256=manifest,checkpoints=checkpoints,modes=['paper_literal','right_closed'],test_evaluated=False,estimated_training_hours_excluding_validation=estimate),indent=2))
  state['phase']='cache';save()
  execute('cache',[sys.executable,str(root/'scripts/cache_simtier_boundaries.py'),'--store',str(store),'--reference-cache',str(w/'simtier-cache-050'),'--output-dir',str(out/'cache')])
  state['phase']='training';save()
  for key in sorted(refs):
   for mode in ('paper_literal','right_closed'):
    name=key+'-'+mode;config=out/'configs'/f'{name}.json';config.write_text(json.dumps(configs[key],indent=2))
    cmd=[sys.executable,str(root/'scripts/train_c0.py'),'--store',str(store),'--config',str(config),'--output-dir',str(out/name),'--hist-cache',str(out/'cache'/mode)]
    if key in checkpoints:cmd+=['--content-checkpoint',checkpoints[key]]
    execute(name,cmd)
    r=read(out/name/'report.json');assert r['config']==configs[key] and r['steps']==11924 and r['epochs'][0]['rows']==3052475
    assert r['parameters_updated'] and r['reload_max_abs_error']==0 and not r['test_evaluated'] and r['store_manifest_sha256']==manifest
    assert r['histogram_cache_sha256']==sha256(out/'cache'/mode/'manifest.json')
    assert r['implementation_sha256']==read(refs[key]/'report.json')['implementation_sha256']
    with np.load(out/name/'validation_predictions.npz') as pred:
     assert len(pred['logits'])==768773 and np.isfinite(pred['logits']).all()
     for k in ('labels','source_rows','user_rows'):np.testing.assert_array_equal(pred[k],np.load(store/f'validation.{k}.npy',mmap_mode='r'))
  state['phase']='comparisons';save()
  for key in sorted(refs):
   baseline=refs[key]/'validation_predictions.npz';assert sha256(baseline)==reference_info[key]['predictions_sha256']
   for mode in ('paper_literal','right_closed'):
    name=key+'-'+mode
    execute('compare-'+name,[sys.executable,str(root/'scripts/compare_validation.py'),'--baseline',str(baseline),'--candidate',str(out/name/'validation_predictions.npz'),'--report',str(out/f'compare-{name}.json')])
  state['phase']='completed';save()
 except Exception as e:
  state.update(phase='failed',failed={'job':state['active'],'error':repr(e)});save();raise
if __name__=='__main__':main()
