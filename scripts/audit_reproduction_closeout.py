"""Bounded artifact integrity closeout; no training or metric rerun."""
import argparse,json,hashlib
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();w=a.work
 content=Path(__file__).resolve().parents[1];muse=content.parent/'taobao-mm-muse-repro';projects={'content':content,'muse':muse}
 checks=[]
 def record(name,ok,detail):
  checks.append(dict(name=name,passed=bool(ok),detail=detail))
 freeze=read(w/'final-test-freeze-v1.json')
 record('shared_protocol_identical',(content/'configs/shared-downstream-protocol-v1.json').read_bytes()==(muse/'configs/shared-downstream-protocol-v1.json').read_bytes(),'Byte-identical shared downstream protocol')
 for name,root in projects.items():
  record(name+'_freeze_identical',sha(root/'reports/final-test-freeze-v1.json')==sha(w/'final-test-freeze-v1.json'),'Repository freeze matches local frozen specification')
  for file,digest in freeze['source_sha256'][name].items():record(name+'_source_'+file,sha(root/'src'/file)==digest,'Frozen source SHA-256')
 inventory=read(w/'final-evaluation-inventory-v1.json')
 record('inventory_unchanged',sha(w/'final-evaluation-inventory-v1.json')==freeze['inventory_sha256'],'Original 54-model inventory digest')
 for model in inventory['models']:record('main_report_'+model['id'],sha(Path(model['run'])/'report.json')==model['report_sha256'],'Original training report unchanged')
 for name in ('final-heldout-test-summary.json','final-results-summary.json'):
  record('shared_'+name,sha(content/'reports'/name)==sha(muse/'reports'/name),'Both projects publish identical primary summary')
 stages=['post-main-g1-extension-v1','post-main-g3-extension-v1','post-main-make-epochs-v1','post-main-simtier-boundaries-v1','post-main-length-seeds-v1']
 for stage in stages:
  s=read(w/stage/'status.json');record(stage,s.get('phase')=='completed' and not s.get('failed') and not s.get('active') and not s.get('test_evaluated'),'Completed, no active job, validation only')
 artifacts={'content':['post-main-make-epochs-validation.json','post-main-simtier-boundaries-validation.json','posthoc-paper-groups-v1.json'],'muse':['post-main-din-validation.json','post-main-twin-validation.json','post-main-miss-validation.json','post-main-g1-validation.json','post-main-g3-validation.json','post-main-length-seeds-validation.json','adapted-baseline-comparison.json','anonymous-retrieval-cases.json']}
 digests={}
 for project,names in artifacts.items():
  for name in names:
   file=projects[project]/'reports'/name;data=read(file);digests[project+'/'+name]=sha(file)
   record(project+'_artifact_'+name,isinstance(data,dict),'Machine-readable report exists and parses; digest recorded')
 result=dict(scope='bounded_closeout_integrity_audit',date='2026-09-24',checks=checks,all_passed=all(x['passed'] for x in checks),artifact_sha256=digests,limitations=['not a new exhaustive paper review','does not rerun every metric or rehash all model/data arrays','does not certify full original-paper reproduction','posthoc test group analysis remains separate from validation extensions'])
 for root in projects.values():(root/'reports/reproduction-closeout-integrity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({'checks':len(checks),'all_passed':result['all_passed'],'failed':[x for x in checks if not x['passed']]},indent=2))
 if not result['all_passed']:raise RuntimeError('Integrity audit failed')
if __name__=='__main__':main()
