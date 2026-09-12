"""Compare cohort coverage against the complete source key table."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc


def contains(keys, query):
    pos=np.searchsorted(keys,query)
    valid=pos<len(keys)
    valid[valid] &= keys[pos[valid]]==query[valid]
    return valid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--cohort-dir',type=Path,required=True)
    parser.add_argument('--history-dir',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    keys=np.sort(pq.read_table(args.source,columns=['205'])['205'].to_numpy())
    result={'scope':'1_percent_cohort_direct_full_source_key_audit','source_key_rows':len(keys),'splits':{}}
    for split in ['train','validation','test']:
        exposure=pq.read_table(args.cohort_dir/(split+'.parquet'),columns=['129_1','205'])
        freq=Counter(exposure['129_1'].to_pylist())
        target=exposure['205'].to_numpy()
        matched=contains(keys,target)
        valid_count=hit_count=weighted_valid=weighted_hit=0
        path=args.history_dir/('test_users.parquet' if split=='test' else 'train_users.parquet')
        with pq.ParquetFile(path) as pf:
            for batch in pf.iter_batches(batch_size=4096,columns=['129_1','150_2_180']):
                ids=batch.column(0).to_pylist()
                lengths=pc.list_value_length(batch.column(1)).to_numpy()
                items=pc.list_flatten(batch.column(1)).to_numpy()
                valid=items!=0
                found=contains(keys,items)&valid
                weights=np.repeat(np.array([freq.get(u,0) for u in ids],dtype=np.int64),lengths)
                selected=weights>0
                valid_count+=int(np.sum(valid&selected));hit_count+=int(np.sum(found&selected))
                weighted_valid+=int(weights[valid].sum());weighted_hit+=int(weights[found].sum())
        result['splits'][split]={'target_missing_exposures':int((~matched).sum()),'target_missing_unique':len(np.unique(target[~matched])),
            'nonpadding_history_occurrences':valid_count,'matched_history_occurrences':hit_count,
            'user_history_occurrence_coverage':hit_count/valid_count,
            'exposure_weighted_history_coverage':weighted_hit/weighted_valid}
        print(split,result['splits'][split],flush=True)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
