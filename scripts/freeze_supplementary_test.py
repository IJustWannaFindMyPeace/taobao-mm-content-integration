"""Freeze supplementary test only after complete exact replay and adapter acceptance."""
import argparse,json,time
from pathlib import Path
from supplementary_test_contract import digest,verify_dependencies

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for name in ('work','acceptance','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();w=a.work.resolve();root=Path(__file__).resolve().parents[1]
 if a.output.exists():raise ValueError('Do not replace an existing freeze')
 inventory=w/'supplementary-test-candidates-v1.json';inv=read(inventory);contract=read(w/'supplementary-comparison-contract-v1.json');state=read(w/'supplementary-validation-replays-v1/status.json')
 if state['phase']!='completed' or state['failed'] or set(state['completed'])!=set(inv['models']):raise ValueError('All 73 exact replays required')
 assert len(inv['models'])==73 and len(contract['comparisons'])==93 and contract['candidate_inventory_sha256']==digest(inventory)
 verify_dependencies(contract,root)
 replay_hashes={}
 for key,entry in inv['models'].items():
  path=w/'supplementary-validation-replays-v1'/key/'report.json';r=read(path)
  assert r['max_abs_logit_error']==0 and r['rows']==768773 and not r['test_evaluated'] and not r['optimizer_created']
  assert r['model_id']==key and r['model_sha256']==entry['model_sha256'] and r['inventory_sha256']==digest(inventory)
  assert r['evaluator_sha256']==digest(root/'scripts/replay_supplementary.py')
  project='content' if entry['family']=='content' else 'muse'
  for name,value in r['source_sha256'].items():
   matches=[digest for path,digest in contract['dependency_sha256'].items() if path.startswith(project+'/') and path.split('/')[-1]==name]
   assert len(matches)==1 and matches[0]==value,'Replay source provenance mismatch: '+name
  run=Path(entry['run']);assert digest(run/'report.json')==entry['training_report_sha256'] and digest(run/'model.pt')==entry['model_sha256']
  replay_hashes[key]=digest(path)
 acceptance=read(a.acceptance/'report.json')
 assert acceptance['scope']=='supplementary_validation_evaluation' and acceptance['max_abs_logit_error']==0 and not acceptance['test_evaluated']
 assert acceptance['evaluator_sha256']==digest(root/'scripts/evaluate_supplementary.py') and acceptance['inventory_sha256']==digest(inventory)
 for name,value in contract['test_arrays_sha256'].items():assert digest(w/'shared-feature-store-050'/name)==value
 references={};oldfreeze=read(w/'final-test-freeze-v1.json')
 for pair in contract['comparisons']:
  for key in (pair['baseline'],pair['candidate']):
   if not key.startswith('main:') or key in references:continue
   model=key.removeprefix('main:');folder=w/'final-heldout-test-v1'/model;r=read(folder/'report.json')
   assert r['test_evaluated'] and r['rows']==923211 and r['model_sha256']==oldfreeze['models'][model]['model_sha256']
   assert r['freeze_sha256']==digest(w/'final-test-freeze-v1.json') and digest(folder/'test_predictions.npz')==r['test_predictions_sha256']
   references[key]=dict(run=str(folder),report_sha256=digest(folder/'report.json'),predictions_sha256=r['test_predictions_sha256'])
 dependencies=dict(contract['dependency_sha256'])
 for name in ('evaluate_supplementary.py','supplementary_test_contract.py','compare_supplementary_test.py','freeze_supplementary_test.py','run_supplementary_test.py'):
  dependencies['content/scripts/'+name]=digest(root/'scripts'/name)
 result=dict(scope='frozen_supplementary_test_v1',frozen_at_unix=time.time(),all_validation_replays_exact=True,candidate_inventory_sha256=digest(inventory),models=inv['models'],comparisons=contract['comparisons'],comparison_contract_sha256=digest(w/'supplementary-comparison-contract-v1.json'),replay_report_sha256=replay_hashes,acceptance_report_sha256=digest(a.acceptance/'report.json'),references=references,dependency_sha256=dependencies,test_arrays_sha256=contract['test_arrays_sha256'],store_manifest_sha256=digest(w/'shared-feature-store-050/manifest.json'),policy='All fixed models and comparisons; no tuning or selection from supplementary test outcomes; prior main test already inspected',test_evaluated_at_freeze=False)
 with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print('Frozen 73 models and 93 comparisons',flush=True)
if __name__=='__main__':main()
