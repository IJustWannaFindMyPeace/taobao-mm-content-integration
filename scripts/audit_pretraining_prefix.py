"""Verify a five-epoch trajectory and its three-epoch reference prefix."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from fetch_source import sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import evaluate

def main():
    p=argparse.ArgumentParser()
    for name in ['store','reference','extended','report']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.report.exists():raise ValueError('Fresh report required')
    old=json.loads((a.reference/'report.json').read_text());new=json.loads((a.extended/'report.json').read_text())
    normalize=lambda c:{k:v for k,v in c.items() if k!='epochs'}
    if normalize(old['config'])!=normalize(new['config']) or old['implementation_sha256']!=new['implementation_sha256']:raise ValueError('Trajectory specification changed')
    manifest=json.loads((a.store/'manifest.json').read_text());store_hash=sha256(a.store/'manifest.json')
    expected={k:np.load(a.store/('validation.'+k+'.npy')) for k in ['labels','source_rows','user_rows']}
    checks=[]
    for run,folder,count in [(old,a.reference,3),(new,a.extended,5)]:
        if run['test_evaluated'] or run['store_manifest_sha256']!=store_hash or len(run['epochs'])!=count or run['config']['epochs']!=count:raise ValueError('Run scope mismatch')
        if run['steps']!=count*((manifest['splits']['train']['rows']+run['config']['batch_size']-1)//run['config']['batch_size']):raise ValueError('Step mismatch')
        for e in run['epochs']:
            epoch=e['epoch'];checkpoint=folder/f'epoch-{epoch}.pt'
            if e['rows']!=manifest['splits']['train']['rows'] or e['reload_max_abs_error']!=0 or sha256(checkpoint)!=e['checkpoint_sha256']:raise ValueError('Epoch integrity failed')
            predictions=folder/f'validation-epoch-{epoch}.npz'
            with np.load(predictions) as saved:
                for k,v in expected.items():np.testing.assert_array_equal(saved[k],v)
                if evaluate(expected['labels'],saved['logits'],expected['user_rows'])!=e['validation']:raise ValueError('Metric mismatch')
            checks.append(dict(run='reference' if count==3 else 'extended',epoch=epoch,prediction_sha256=sha256(predictions),checkpoint_sha256=e['checkpoint_sha256'],metrics=e['validation']))
            print('Audited',count,epoch,flush=True)
    prefix=[]
    for epoch in [1,2,3]:
        with np.load(a.reference/f'validation-epoch-{epoch}.npz') as x,np.load(a.extended/f'validation-epoch-{epoch}.npz') as y:
            error=float(np.max(np.abs(x['logits']-y['logits'])))
            np.testing.assert_array_equal(x['logits'],y['logits'])
        x=torch.load(a.reference/f'epoch-{epoch}.pt',map_location='cpu',weights_only=True)
        y=torch.load(a.extended/f'epoch-{epoch}.pt',map_location='cpu',weights_only=True)
        if x.keys()!=y.keys() or any(not torch.equal(x[k],y[k]) for k in x):raise ValueError('Prefix weights differ')
        prefix.append(dict(epoch=epoch,max_prediction_error=error,all_weights_exact=True))
    result=dict(scope='pretraining_trajectory_prefix_audit',store_manifest_sha256=store_hash,prefix=prefix,epochs=checks,reference_report_sha256=sha256(a.reference/'report.json'),extended_report_sha256=sha256(a.extended/'report.json'),script_sha256=sha256(Path(__file__)),test_evaluated=False)
    a.report.write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
