"""Audit candidate coverage, pair contracts and test data without test inference."""
import argparse,json,hashlib,collections
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def check_pair(a,b):
 if a['store_manifest_sha256']!=b['store_manifest_sha256']:raise ValueError('Different data stores')
 for key in ('seed','batch_size','embedding_width','dense_learning_rate','sparse_learning_rate','weight_decay'):
  if a['config'][key]!=b['config'][key]:raise ValueError('Uncontrolled common setting: '+key)

def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();w=a.work.resolve()
 content=Path(__file__).resolve().parents[1];muse=content.parent/'taobao-mm-muse-repro';projects={'content':content,'muse':muse}
 inventory=read(w/'supplementary-test-candidates-v1.json');original=read(w/'final-evaluation-inventory-v1.json');models=dict(inventory['models']);lookup={v['validation_predictions_sha256']:k for k,v in models.items()};checks=[]
 for e in original['models']:
  folder=Path(e['run']);assert sha(folder/'report.json')==e['report_sha256'];r=read(folder/'report.json');predsha=sha(folder/'validation_predictions.npz');key='main:'+e['id']
  models[key]=dict(run=e['run'],config=r['config'],store_manifest_sha256=r['store_manifest_sha256'],validation_predictions_sha256=predsha);lookup[predsha]=key
 sources={'content':['post-main-make-three-seed-summary.json','post-main-make-epochs-validation.json','post-main-simtier-boundaries-validation.json','shared-pretraining-endpoints-v1-seed42-audit.json'],'muse':['post-main-din-validation.json','post-main-twin-validation.json','post-main-miss-validation.json','post-main-g1-validation.json','post-main-g3-validation.json','post-main-length-seeds-validation.json']}
 pairs=[];provenance={}
 def walk(x,path):
  if isinstance(x,dict):
   if 'baseline_sha256' in x and 'candidate_sha256' in x:
    b=lookup.get(x['baseline_sha256']);c=lookup.get(x['candidate_sha256'])
    if b is None or c is None:raise ValueError('Unmapped comparison '+path)
    check_pair(models[b],models[c]);pairs.append(dict(baseline=b,candidate=c,source=path))
   for key,value in x.items():walk(value,path+'/'+str(key))
  elif isinstance(x,list):
   for i,value in enumerate(x):walk(value,path+'/'+str(i))
 for project,names in sources.items():
  for name in names:
   file=projects[project]/'reports'/name;provenance[project+'/'+name]=sha(file);walk(read(file),project+'/'+name)
 assert len(pairs)==84
 for seed in (2026,3407):
  path=w/f'post-main-make-epochs-v1/endpoints-audit-seed{seed}/summary.json';provenance[f'work/endpoints-audit-seed{seed}']=sha(path);walk(read(path),f'work/endpoints-audit-seed{seed}')
 for seed in (42,2026,3407):
  candidates=[key for key,v in inventory['models'].items() if key.endswith('--c0-epochs3') and v['config']['seed']==seed];assert len(candidates)==1
  b=f'main:c0-seed{seed}';c=candidates[0];check_pair(models[b],models[c]);pairs.append(dict(baseline=b,candidate=c,source='predefined final ID epoch3 minus main ID epoch1; epoch2 checkpoint unavailable'))
 assert len(pairs)==93 and len({(p['baseline'],p['candidate']) for p in pairs})==93
 assert len(models)==127
 for key,v in inventory['models'].items():
  run=Path(v['run']);r=read(run/'report.json');assert sha(run/'report.json')==v['training_report_sha256'] and r['config']==v['config']
  assert (run/'model.pt').is_file() and (run/'model.pt').stat().st_size>0
  assert r['config']['batch_size']==256 and r['config']['seed'] in (42,2026,3407)
  assert r['steps']==11924*r['config']['epochs'] and not r['test_evaluated']
  assert r['store_manifest_sha256']==sha(w/'shared-feature-store-050/manifest.json')
  assert v['histogram_mode'] in (None,'paper_literal','right_closed')
  if v['histogram_mode'] is not None:assert r['config']['simtier_bins']==20 and r['config']['history_length']==200
 helpers={}
 for project,root in projects.items():
  for folder in ('src','extensions'):
   for file in sorted((root/folder).glob('*.py')):helpers[f'{project}/{folder}/{file.name}']=sha(file)
 for name in ('replay_supplementary.py','cache_simtier_boundaries.py','compare_validation.py'):
  helpers['content/scripts/'+name]=sha(content/'scripts'/name)
 state=read(w/'supplementary-validation-replays-v1/status.json');replayed=[]
 for key in state['completed']:
  r=read(w/'supplementary-validation-replays-v1'/key/'report.json')
  assert r['model_id']==key and r['rows']==768773 and r['max_abs_logit_error']==0 and not r['test_evaluated'] and not r['optimizer_created']
  assert r['inventory_sha256']==sha(w/'supplementary-test-candidates-v1.json') and r['model_sha256']==inventory['models'][key]['model_sha256']
  assert r['evaluator_sha256']==helpers['content/scripts/replay_supplementary.py'];replayed.append(key)
 freeze=read(w/'final-test-freeze-v1.json');data={}
 for name,value in freeze['test_arrays_sha256'].items():
  measured=sha(w/'shared-feature-store-050'/name);assert measured==value,'Test data changed: '+name;data[name]=measured
 print('Test input fingerprints verified',flush=True)
 plan=dict(scope='proposed_supplementary_test_comparison_contract',candidate_inventory_sha256=sha(w/'supplementary-test-candidates-v1.json'),comparisons=pairs,source_reports_sha256=provenance,test_arrays_sha256=data,dependency_sha256=helpers,policy='93 fixed pairs, all candidates; no selection from test outcomes',frozen_for_test=False)
 (w/'supplementary-comparison-contract-v1.json').write_text(json.dumps(plan,indent=2))
 result=dict(scope='pre_test_design_and_artifact_audit',candidate_models=73,original_reference_models=54,mapped_existing_pairs=90,additional_final_id_epoch_pairs=3,total_pairs=93,unmapped_pairs=0,test_input_files_verified=len(data),replays_verified_at_audit=len(replayed),models_retrained=False,new_test_inference_performed=False,ready_for_test=False,blocking_gates=['all 73 exact validation replays','test evaluator using the same replayed forward path','freeze test evaluator plus all helper dependency hashes','explicit test-labelled paired statistics','crash-safe and single-writer orchestration; verify completed output provenance before skipping'],findings=[dict(id='A1',issue='73 candidate records were not a complete comparison contract',resolution='93 pair definitions mapped and fixed in pre-test contract'),dict(id='A2',issue='replay report does not fingerprint all script helpers',resolution='helper snapshot recorded; final test freeze must enforce it'),dict(id='A3',issue='existing comparison script always labels output as validation',resolution='do not directly publish its output as test evidence; test adapter required'),dict(id='A4',issue='ID epoch2 weights were not saved',resolution='explicitly excluded, retain validation-only diagnosis; epoch3 vs epoch1 test pair included'),dict(id='A5',issue='current resume runner has no single-writer lock',resolution='avoid concurrent launches; test orchestration must add lock before execution')])
 for root in projects.values():
  (root/'reports/supplementary-pretest-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
  (root/'reports/supplementary-comparison-contract-v1.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
