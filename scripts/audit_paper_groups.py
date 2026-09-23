"""Post-hoc balanced-entity groups; reuse and authenticate frozen predictions."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import evaluate


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def balanced_groups(ids,values,n):
    """Primary numeric value; SHA256-ID tie break fixed without labels/scores."""
    tie=np.array([hashlib.sha256(str(int(x)).encode()).hexdigest() for x in ids])
    order=np.lexsort((tie,values));group=np.empty(len(ids),dtype=np.int16)
    for g,indices in enumerate(np.array_split(order,n)):group[indices]=g
    return group


def main():
    p=argparse.ArgumentParser();p.add_argument('--store',type=Path,required=True)
    p.add_argument('--items-dir',type=Path,required=True);p.add_argument('--predictions-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise ValueError('Fresh output required')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.store/'manifest.json').read_text())
    for split in ('train','test'):
        assert sha(a.items_dir/f'{split}.parquet')==manifest['input_sha256'][split]
    train=pq.read_table(a.items_dir/'train.parquet',columns=['205'])['205'].to_numpy()
    table=pq.read_table(a.items_dir/'test.parquet',columns=['205','source_row'])
    targets=table['205'].to_numpy();source=table['source_row'].to_numpy()
    labels=np.load(a.store/'test.labels.npy',mmap_mode='r');users=np.load(a.store/'test.user_rows.npy',mmap_mode='r')
    np.testing.assert_array_equal(source,np.load(a.store/'test.source_rows.npy',mmap_mode='r'))
    keys,counts=np.unique(train,return_counts=True)
    items,item_inverse=np.unique(targets,return_inverse=True)
    index=np.searchsorted(keys,items);found=index<len(keys);found[found]&=keys[index[found]]==items[found]
    frequency=np.zeros(len(items),dtype=np.int64);frequency[found]=counts[index[found]]
    user_keys,user_inverse=np.unique(users,return_inverse=True)
    history=np.load(a.store/'test_users.item.npy',mmap_mode='r')
    lengths=np.empty(len(user_keys),dtype=np.int64)
    for start in range(0,len(user_keys),1024):
        lengths[start:start+1024]=(history[user_keys[start:start+1024]]!=0).sum(1)
    schemes={}
    for name,n,ids,values,inverse,models in [
        ('cikm_item_8',8,items,frequency,item_inverse,('c0','c6')),
        ('muse_item_9',9,items,frequency,item_inverse,('r1-l1000','r4-l1000')),
        ('muse_user_9',9,user_keys,lengths,user_inverse,('r1-l1000','r4-l1000'))]:
        entity_groups=balanced_groups(ids,values,n);row_groups=entity_groups[inverse]
        assert len(row_groups)==len(labels)
        metadata=[]
        for g in range(n):
            mask=entity_groups==g;v=values[mask]
            metadata.append({'group':g+1,'entities':int(mask.sum()),'rows':int((row_groups==g).sum()),
                'value_min':int(v.min()),'value_max':int(v.max()),'zero_value_entities':int((v==0).sum())})
        assert sum(x['rows'] for x in metadata)==len(labels)
        assert max(x['entities'] for x in metadata)-min(x['entities'] for x in metadata)<=1
        np.savez(a.output_dir/f'{name}-membership.npz',entity_ids=ids,entity_values=values,entity_groups=entity_groups)
        schemes[name]={'groups':metadata,'models':{},'membership_sha256':sha(a.output_dir/f'{name}-membership.npz')}
        for seed in (42,2026,3407):
            for model in models:
                job=f'{model}-seed{seed}';folder=a.predictions_dir/job;path=folder/'test_predictions.npz'
                report=json.loads((folder/'report.json').read_text());digest=sha(path)
                assert report['test_predictions_sha256']==digest
                assert report['store_manifest_sha256']==sha(a.store/'manifest.json')
                with np.load(path) as pred:
                    for key,expected in [('labels',labels),('source_rows',source),('user_rows',users)]:np.testing.assert_array_equal(pred[key],expected)
                    metrics=[evaluate(labels[row_groups==g],pred['logits'][row_groups==g],users[row_groups==g]) for g in range(n)]
                schemes[name]['models'][job]={'prediction_sha256':digest,'metrics':metrics}
                (a.output_dir/'progress.json').write_text(json.dumps({'scheme':name,'model':job}))
                print(name,job,flush=True)
    result={'scope':'posthoc_paper_group_count_alignment','post_hoc':True,'test_evaluated':True,
        'test_rows':len(labels),'store_manifest_sha256':sha(a.store/'manifest.json'),
        'implementation_sha256':sha(Path(__file__)),'schemes':schemes,
        'rules':{'entity_universe':'distinct evaluated test targets/users; not equal exposure groups',
                 'item_score':'training target exposure count only; unseen target frequency zero',
                 'user_score':'nonpadding positions in supplied capped-1000 history',
                 'ties':'SHA256 of decimal entity ID ascending; split ties for balanced entity counts',
                 'membership_uses':'entity identities and input covariates, never labels or predictions'},
        'limitations':['group-count alignment, not exact unpublished original membership',
                       'test entity universe defines rank groups; boundaries are not training-only quantiles',
                       'same-frequency or capped-length ties may span adjacent groups',
                       'item-group GAUC uses only within-group user exposures; user-group GAUC retains whole users',
                       'descriptive analysis without subgroup confidence intervals; no retraining or selection']}
    (a.output_dir/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
