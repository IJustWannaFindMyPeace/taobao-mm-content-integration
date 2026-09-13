"""Train-target exposure frequency groups; validation labels never define bins."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import auc
from fetch_source import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--items-dir',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True,help='JSON mapping names to validation npz paths')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.store/'manifest.json').read_text())
    for split in ['train','validation']:
        if sha256(args.items_dir/(split+'.parquet'))!=manifest['input_sha256'][split]:
            raise ValueError('Joined exposure source mismatch')
    train=pq.read_table(args.items_dir/'train.parquet',columns=['205'])['205'].to_numpy()
    val=pq.read_table(args.items_dir/'validation.parquet',columns=['205','source_row'])
    target=val['205'].to_numpy();source=val['source_row'].to_numpy()
    labels=np.load(args.store/'validation.labels.npy',mmap_mode='r')
    np.testing.assert_array_equal(source,np.load(args.store/'validation.source_rows.npy',mmap_mode='r'))
    keys,counts=np.unique(train,return_counts=True)
    index=np.searchsorted(keys,target);found=index<len(keys)
    found[found] &= keys[index[found]]==target[found]
    frequencies=np.zeros(len(target),dtype=np.int64);frequencies[found]=counts[index[found]]
    group=np.searchsorted([0,1,5,20,100],frequencies,side='left')
    names=['0','1','2-5','6-20','21-100','101+']
    model_paths=json.loads(args.models.read_text())
    results={};digests={}
    for name,path in model_paths.items():
        with np.load(path) as predictions:
            np.testing.assert_array_equal(predictions['source_rows'],source)
            np.testing.assert_array_equal(predictions['labels'],labels)
            logits=predictions['logits']
            if not np.isfinite(logits).all():raise ValueError('Nonfinite predictions')
            values=[]
            for i,group_name in enumerate(names):
                mask=group==i;y=labels[mask];s=logits[mask].astype(np.float64)
                values.append({'training_target_exposure_frequency':group_name,'validation_rows':int(mask.sum()),
                    'distinct_validation_targets':int(len(np.unique(target[mask]))),'positives':int(y.sum()),
                    'auc':auc(y,s) if len(y) else None,
                    'logloss':float(np.mean(np.logaddexp(0,s)-y*s)) if len(y) else None})
            results[name]=values
        digests[name]=sha256(Path(path))
    report={'scope':'single_seed_validation_frequency_diagnostic','store_manifest_sha256':sha256(args.store/'manifest.json'),
            'implementation_sha256':sha256(Path(__file__)),'prediction_sha256':digests,
            'frequency_source':'raw target item IDs in selected training exposures only',
            'training_rows':len(train),'validation_rows':len(labels),'models':results,'test_evaluated':False,
            'limitations':['zero train-target frequency is not proof of a newly created item or no history interactions',
                           'fixed frequency bins; not original-paper quantile groups',
                           'descriptive AUC/logloss only; no subgroup confidence intervals or seed variability']}
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Frequency report completed; all prediction rows and labels aligned.')


if __name__=='__main__':main()
