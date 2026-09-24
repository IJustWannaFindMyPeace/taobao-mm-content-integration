"""Inventory all completed supplementary downstream configurations before test."""
import argparse,json,hashlib
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();w=a.work.resolve()
 if a.output.exists():raise ValueError('Fresh inventory required')
 entries={};missing=[]
 def add(folder,family,mode=None):
  folder=folder.resolve();r=read(folder/'report.json');key=folder.relative_to(w).as_posix().replace('/','--')
  if key in entries:return
  assert r['parameters_updated'] and r['reload_max_abs_error']==0 and not r['test_evaluated']
  assert r['steps']==r['config']['epochs']*11924
  assert all(e['rows']==3052475 for e in r['epochs'])
  assert (folder/'model.pt').exists() and (folder/'validation_predictions.npz').exists()
  entries[key]=dict(run=str(folder),family=family,histogram_mode=mode,config=r['config'],model_sha256=sha(folder/'model.pt'),training_report_sha256=sha(folder/'report.json'),validation_predictions_sha256=sha(folder/'validation_predictions.npz'),store_manifest_sha256=r['store_manifest_sha256'])
  print('Inventoried',key,flush=True)
 for stage,family in [('post-main-short-interest-v1','din'),('post-main-twin-extension-v1','twin'),('post-main-miss-extension-v1','miss'),('post-main-g3-extension-v1','g3')]:
  for folder in sorted((w/stage).iterdir()):
   if folder.is_dir() and not folder.name.startswith('probe') and (folder/'model.pt').exists():add(folder,family)
 for seed in (42,2026,3407):
  add(w/f'post-main-g1-extension-v1/downstream-seed{seed}','g1')
  strategy=w/('shared-make-strategy-v1-seed42' if seed==42 else f'post-main-make-seeds-v1/seed{seed}')
  for variant in ('c4','c4-frozen','c5-joint','c0-epochs3'):add(strategy/variant,'content')
  missing.append(dict(seed=seed,item='ID-only epoch2',reason='only final model.pt saved; epoch2 validation predictions are not a checkpoint; no replacement retraining'))
 for folder in sorted((w/'post-main-simtier-boundaries-v1').iterdir()):
  if folder.is_dir() and (folder/'model.pt').exists():add(folder,'content','paper_literal' if folder.name.endswith('paper_literal') else 'right_closed')
 for folder in sorted((w/'shared-pretraining-endpoints-v1-seed42').iterdir()):
  if folder.is_dir() and (folder/'model.pt').exists():add(folder,'content')
 for seed in (2026,3407):
  for endpoint in ('p1','p5'):add(w/f'post-main-make-epochs-v1/endpoints-seed{seed}/{endpoint}','content')
 add(w/'post-main-make-epochs-v1/p3-seed2026','content')
 for folder in sorted((w/'shared-length-sensitivity-v1-seed42').iterdir()):
  if folder.is_dir() and (folder/'model.pt').exists():add(folder,'muse')
 for folder in sorted((w/'post-main-length-seeds-v1').iterdir()):
  if folder.is_dir() and (folder/'model.pt').exists():add(folder,'muse')
 result=dict(scope='supplementary_downstream_test_candidate_inventory',models=entries,missing_checkpoints=missing,excluded=['resource probes','auxiliary content/scorer pretraining curves, not final downstream recommendation models','anonymous cases','posthoc grouping, already separate'],policy='all completed configurations and seeds retained; no selection by scores; validation replay required before separate test freeze',test_evaluated=False)
 a.output.write_text(json.dumps(result,indent=2));print('Total',len(entries))
if __name__=='__main__':main()
