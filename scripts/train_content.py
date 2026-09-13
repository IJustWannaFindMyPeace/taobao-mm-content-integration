"""Train content-only epoch curves; reserve official test for final evaluation."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from make import ContentMAKE
from feature_store import FeatureStore
from metrics import evaluate
from fetch_source import sha256


def inputs(batch):
    return [torch.as_tensor(batch[key],device='cuda') for key in
            ['target_content','history_content','history_valid','target_content_available','history_content_available']]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-steps',type=int)
    args=parser.parse_args()
    if args.max_steps is not None and args.max_steps<1:raise ValueError('Positive step limit required')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh output directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(args.config.read_text())
    implementation_hashes={p.name:sha256(p) for p in [Path(__file__),Path(__file__).parents[1]/'src/make.py']}
    torch.manual_seed(cfg['seed']); rng=np.random.default_rng(cfg['seed'])
    train=FeatureStore(args.store,'train'); val=FeatureStore(args.store,'validation')
    model=ContentMAKE().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    curves=[]; steps=0
    initial=model.output.weight.detach().clone()
    for epoch in range(1,cfg['epochs']+1):
        model.train(); seen=0; total=0.
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started=time.perf_counter()
        order=rng.permutation(len(train))
        for start in range(0,len(order),cfg['batch_size']):
            batch=train.batch(order[start:start+cfg['batch_size']],cfg['history_length'],with_content=True)
            optimizer.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(model(*inputs(batch)),torch.as_tensor(batch['labels'],device='cuda'))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradient')
            optimizer.step(); steps+=1; seen+=len(batch['labels']); total+=loss.item()*len(batch['labels'])
            if steps%500==0:
                progress={'epoch':epoch,'steps':steps,'rows':seen,'online_train_logloss':total/seen}
                print(json.dumps(progress),flush=True)
                (args.output_dir/'progress.json').write_text(json.dumps(progress))
            if args.max_steps and steps>=args.max_steps:break
        torch.cuda.synchronize()
        record={'epoch':epoch,'rows':seen,'training_seconds':time.perf_counter()-started,
                'training_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'training_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'online_train_logloss':total/seen}
        model.eval(); predictions=[]
        with torch.no_grad():
            for start in range(0,len(val),cfg['batch_size']):
                batch=val.batch(np.arange(start,min(start+cfg['batch_size'],len(val))),cfg['history_length'],with_content=True)
                predictions.append(model(*inputs(batch)).cpu().numpy())
        logits=np.concatenate(predictions)
        record['validation']=evaluate(val.labels,logits,val.user_rows)
        checkpoint=args.output_dir/f'epoch-{epoch}.pt'
        torch.save(model.state_dict(),checkpoint)
        record['checkpoint_sha256']=sha256(checkpoint)
        probe=inputs(val.batch(np.arange(4),cfg['history_length'],with_content=True))
        with torch.no_grad():before=model(*probe).cpu()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        with torch.no_grad():record['reload_max_abs_error']=(before-model(*probe).cpu()).abs().max().item()
        assert record['reload_max_abs_error']==0
        np.savez(args.output_dir/f'validation-epoch-{epoch}.npz',logits=logits,labels=val.labels,source_rows=val.source_rows,user_rows=val.user_rows)
        curves.append(record)
        result={'scope':'partial_training_resource_probe' if args.max_steps else 'single_seed_content_learning_curve',
                'config':cfg,'store_manifest_sha256':sha256(args.store/'manifest.json'),
                'implementation_sha256':implementation_hashes,
                'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'steps':steps,'epochs':curves,
                'parameters_updated':not torch.equal(initial,model.output.weight),'test_evaluated':False,
                'limitations':['single seed and one cohort','adapted DIN activation and layer widths','model-only checkpoints; no optimizer resume']}
        (args.output_dir/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(record),flush=True)
        if args.max_steps and steps>=args.max_steps:break


if __name__=='__main__':main()
