"""Validate all exposure labels and sampled raw-history reconstruction."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from feature_store import FeatureStore,encode_ids
from build_feature_store import dense_history
from fetch_source import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--items-dir',type=Path,required=True)
    parser.add_argument('--history-dir',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.store/'manifest.json').read_text(encoding='utf-8'))
    for name,entry in manifest['arrays'].items():
        if sha256(args.store/name)!=entry['sha256']:raise ValueError('Array checksum mismatch')
    checks={}
    vocab=np.load(args.store/'vocab.205.npy')
    for split in ['train','validation','test']:
        store=FeatureStore(args.store,split)
        table=pq.read_table(args.items_dir/(split+'.parquet'),columns=['129_1','205','label_0','source_row'])
        labels=pc.list_flatten(table['label_0']).to_numpy().reshape(-1,2)[:,1]
        np.testing.assert_array_equal(store.labels,labels)
        np.testing.assert_array_equal(store.source_rows,table['source_row'].to_numpy())
        np.testing.assert_array_equal(store.features[:,6],encode_ids(table['205'].to_numpy(),vocab))
        indices=np.unique(np.r_[np.arange(min(4,len(store))),len(store)-1])
        group='test' if split=='test' else 'train'
        history=pq.read_table(args.history_dir/(group+'_users.parquet'),columns=['129_1','150_2_180'])
        ids=history['129_1'].to_numpy();order=np.argsort(ids)
        raw_users=table['129_1'].to_numpy()[indices]
        expected_rows=order[np.searchsorted(ids[order],raw_users)]
        raw=dense_history(history['150_2_180'].take(expected_rows))
        for length in [50,200,1000]:
            batch=store.batch(indices,length,with_content=True)
            np.testing.assert_array_equal(batch['history_item'],encode_ids(raw[:,-length:],vocab))
            np.testing.assert_array_equal(batch['history_valid'],raw[:,-length:]!=0)
            assert np.isfinite(batch['history_content']).all()
            assert np.all(batch['history_content'][~batch['history_content_available']]==0)
            assert np.all(batch['target_content'][~batch['target_content_available']]==0)
        # Missing target content must remain an exposure in the store.
        missing=np.flatnonzero(store.target_content==0)
        assert len(missing)==manifest['splits'][split]['content_missing_exposures']
        if len(missing):assert np.all(store.batch(missing)['features'][:,6]!=0)
        checks[split]={'rows':len(store),'all_labels_and_source_rows_equal':True,'all_target_id_mappings_equal':True,
            'sampled_history_lengths':[50,200,1000],'missing_target_exposures_retained':len(missing)}
    args.report.write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
