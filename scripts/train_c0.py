"""Train C0 against the fixed store; validation only, no test evaluation."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from c0 import C0
from simtier import C3
from feature_store import FeatureStore
from metrics import evaluate
from fetch_source import sha256


def tensors(batch):
    result = [torch.as_tensor(batch[k],device='cuda',dtype=torch.bool if k=='history_valid' else torch.long)
            for k in ['features','history_item','history_category','history_valid']]
    if 'histogram' in batch: result.append(torch.as_tensor(batch['histogram'],device='cuda'))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--hist-cache',type=Path)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    cfg=json.loads(args.config.read_text())
    if args.max_steps is not None and args.max_steps<1:raise ValueError('Positive step limit required')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh run directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    train=FeatureStore(args.store,'train');validation=FeatureStore(args.store,'validation')
    histograms={}
    if cfg.get('simtier_bins'):
        if args.hist_cache is None:raise ValueError('Histogram cache required')
        cache=json.loads((args.hist_cache/'manifest.json').read_text())
        if cache['store_manifest_sha256']!=sha256(args.store/'manifest.json') or cache['history_length']!=cfg['history_length'] or cache['bins']!=cfg['simtier_bins']:
            raise ValueError('Histogram cache contract mismatch')
        for split,store in [('train',train),('validation',validation)]:
            path=args.hist_cache/(split+'.npy')
            if sha256(path)!=cache['splits'][split]['sha256']:raise ValueError('Histogram digest mismatch')
            histograms[split]=np.load(path,mmap_mode='r')
            if histograms[split].shape!=(len(store),cfg['simtier_bins']):raise ValueError('Histogram shape mismatch')
        model=C3(train.manifest['vocab_sizes'],cfg['embedding_width'],cfg['short_length'],cfg['simtier_bins']).cuda()
    else:
        model=C0(train.manifest['vocab_sizes'],cfg['embedding_width'],cfg['short_length']).cuda()
    def get_batch(store,split,indices):
        batch=store.batch(indices,cfg['history_length'])
        if histograms:batch['histogram']=histograms[split][indices]
        return batch
    sparse=list(model.embeddings.parameters());sparse_ids={id(p) for p in sparse}
    dense=[p for p in model.parameters() if id(p) not in sparse_ids]
    optimizers=[torch.optim.SparseAdam(sparse,lr=cfg['sparse_learning_rate']),
        torch.optim.AdamW(dense,lr=cfg['dense_learning_rate'],weight_decay=cfg['weight_decay'])]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    steps=0;curves=[];update_verified=False
    for epoch in range(cfg['epochs']):
        model.train();seen=0;loss_sum=0.
        order=rng.permutation(len(train))
        for start in range(0,len(order),cfg['batch_size']):
            batch=get_batch(train,'train',order[start:start+cfg['batch_size']])
            for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
            inputs=tensors(batch)
            before=model.tower[-1].weight.detach().clone() if steps==0 else None
            logits=model(*inputs)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(batch['labels'],device='cuda'))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    grad=param.grad.coalesce().values() if param.grad.is_sparse else param.grad
                    if not torch.isfinite(grad).all():raise RuntimeError('Nonfinite gradient')
            for optimizer in optimizers:optimizer.step()
            if before is not None:update_verified=not torch.equal(before,model.tower[-1].weight)
            seen+=len(batch['labels']);loss_sum+=loss.item()*len(batch['labels']);steps+=1
            if steps%100==0:
                print(f'epoch={epoch+1} steps={steps} rows={seen} logloss={loss_sum/seen:.6f}',flush=True)
                (args.output_dir/'progress.json').write_text(json.dumps({'epoch':epoch+1,'steps':steps,'rows':seen}))
            if args.max_steps and steps>=args.max_steps:break
        curves.append({'epoch':epoch+1,'rows':seen,'online_train_logloss':loss_sum/seen})
        if args.max_steps and steps>=args.max_steps:break
    torch.cuda.synchronize()
    training_seconds=time.perf_counter()-started
    peak=torch.cuda.max_memory_allocated()/2**20
    reserved=torch.cuda.max_memory_reserved()/2**20
    assert update_verified
    # Record train-only memory before validation. No test dataset is instantiated.
    model.eval();predictions=[]
    with torch.no_grad():
        for start in range(0,len(validation),cfg['batch_size']):
            b=get_batch(validation,'validation',np.arange(start,min(start+cfg['batch_size'],len(validation))))
            predictions.append(model(*tensors(b)).cpu().numpy())
    logits=np.concatenate(predictions)
    metrics=evaluate(validation.labels,logits,validation.user_rows)
    np.savez(args.output_dir/'validation_predictions.npz',logits=logits,labels=validation.labels,source_rows=validation.source_rows,user_rows=validation.user_rows)
    torch.save(model.state_dict(),args.output_dir/'model.pt')
    # Reload into the existing model to avoid duplicating the large vocabulary on GPU.
    probe=get_batch(validation,'validation',np.arange(4))
    with torch.no_grad():before=model(*tensors(probe)).cpu()
    model.load_state_dict(torch.load(args.output_dir/'model.pt',map_location='cpu',weights_only=True))
    with torch.no_grad():error=(before-model(*tensors(probe)).cpu()).abs().max().item()
    assert error==0
    result={'scope':'partial_training_resource_probe' if args.max_steps else 'one_seed_validation_run',
        'config':cfg,'store_manifest_sha256':sha256(args.store/'manifest.json'),
        'implementation_sha256':{str(p.name):sha256(p) for p in [Path(__file__),Path(__file__).parents[1]/'src/c0.py',Path(__file__).parents[1]/'src/simtier.py']},
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'steps':steps,'epochs':curves,
        'training_seconds':training_seconds,'training_peak_allocated_mib':peak,'training_peak_reserved_mib':reserved,
        'parameters_updated':update_verified,'reload_max_abs_error':error,'validation':metrics,
        'test_evaluated':False,'histogram_cache_sha256':sha256(args.hist_cache/'manifest.json') if histograms else None,'limitations':['single seed; no comparative model result','user-disjoint validation','local architecture and optimizer adaptation']}
    (args.output_dir/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
