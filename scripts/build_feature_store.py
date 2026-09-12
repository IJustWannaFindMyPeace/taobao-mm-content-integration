"""Build a versioned, memory-mapped feature store without expanding user histories."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from feature_store import FIELDS, encode_ids, content_rows
from fetch_source import sha256


def dense_history(column):
    lengths=pc.list_value_length(column).to_numpy()
    if np.any(lengths>1000): raise ValueError('History exceeds supported length')
    if np.all(lengths==1000): return pc.list_flatten(column).to_numpy().reshape(-1,1000)
    result=np.zeros((len(column),1000),dtype=np.int64)
    for i, values in enumerate(column.to_pylist()):
        if values: result[i,-len(values):]=values
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--items-dir',type=Path,required=True)
    parser.add_argument('--items-report',type=Path,required=True)
    parser.add_argument('--history-dir',type=Path,required=True)
    parser.add_argument('--history-report',type=Path,required=True)
    parser.add_argument('--content-dir',type=Path,required=True)
    parser.add_argument('--content-report',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    output=args.output_dir
    if output.exists() and any(output.iterdir()): raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    reports=[json.loads(p.read_text(encoding='utf-8')) for p in [args.items_report,args.history_report,args.content_report]]
    if len({r['revision'] for r in reports})!=1: raise ValueError('Revision mismatch')
    exposures={}
    inputs={}
    for split in ['train','validation','test']:
        path=args.items_dir/(split+'.parquet')
        digest=sha256(path)
        if digest!=reports[0]['splits'][split]['sha256']: raise ValueError('Item join checksum mismatch')
        inputs[split]=digest
        exposures[split]=pq.read_table(path)
    histories={}
    for group in ['train','test']:
        path=args.history_dir/(group+'_users.parquet')
        digest=sha256(path)
        if digest!=reports[1]['splits'][group]['sha256']: raise ValueError('History checksum mismatch')
        inputs[group+'_users']=digest
        histories[group]=pq.read_table(path,columns=FIELDS[:6])
    train_users=np.unique(exposures['train']['129_1'].to_numpy())
    train_user_mask=np.isin(histories['train']['129_1'].to_numpy(),train_users)
    pieces={f:[] for f in FIELDS}
    for field in FIELDS[:6]: pieces[field].append(histories['train'][field].to_numpy()[train_user_mask])
    for field in FIELDS[6:]: pieces[field].append(exposures['train'][field].to_numpy())
    with pq.ParquetFile(args.history_dir/'train_users.parquet') as pf:
        for batch in pf.iter_batches(batch_size=4096,columns=['129_1','150_2_180','151_2_180']):
            table=pa.Table.from_batches([batch])
            selected=table.filter(pc.is_in(table['129_1'],value_set=pa.array(train_users)))
            for field,seq in [('205','150_2_180'),('206','151_2_180')]:
                pieces[field].append(np.unique(pc.list_flatten(selected[seq]).to_numpy()))
    vocab={}
    for field in FIELDS:
        keys=np.unique(np.concatenate(pieces.pop(field)))
        keys=keys[keys!=0]
        if len(keys)+2>=np.iinfo(np.int32).max: raise ValueError('Vocabulary overflow')
        vocab[field]=keys
        np.save(output/('vocab.'+field+'.npy'),keys)
    print('Training vocabularies built', {k:len(v)+2 for k,v in vocab.items()},flush=True)
    content_path=args.content_dir/'content_int8.parquet'
    digest=sha256(content_path)
    if digest!=reports[2]['output_sha256']: raise ValueError('Content checksum mismatch')
    inputs['content']=digest
    content_keys=pq.read_table(content_path,columns=['205'])['205'].to_numpy()
    order=np.argsort(content_keys)
    sorted_keys=content_keys[order]
    vectors=np.lib.format.open_memmap(output/'content_int8.npy',mode='w+',dtype=np.int8,shape=(len(order)+1,128))
    vectors[0]=0
    offset=1
    with pq.ParquetFile(content_path) as pf:
        for batch in pf.iter_batches(batch_size=131072,columns=['205_c']):
            values=pc.list_flatten(batch.column(0)).to_numpy().reshape(-1,128)
            vectors[offset:offset+len(values)]=values;offset+=len(values)
    vectors.flush();del vectors
    for group in ['train','test']:
        count=len(histories[group])
        arrays={name:np.lib.format.open_memmap(output/(group+'_users.'+name+'.npy'),mode='w+',dtype=np.int32,shape=(count,1000)) for name in ['item','category','content']}
        offset=0
        with pq.ParquetFile(args.history_dir/(group+'_users.parquet')) as pf:
            for batch in pf.iter_batches(batch_size=4096,columns=['150_2_180','151_2_180']):
                item=dense_history(batch.column(0));category=dense_history(batch.column(1))
                end=offset+len(item)
                arrays['item'][offset:end]=encode_ids(item,vocab['205'])
                arrays['category'][offset:end]=encode_ids(category,vocab['206'])
                arrays['content'][offset:end]=content_rows(item,sorted_keys,order)
                offset=end
        if offset!=count: raise ValueError('History output row mismatch')
        for array in arrays.values(): array.flush()
        del arrays
    statistics={}
    for split,table in exposures.items():
        group='test' if split=='test' else 'train'
        users=histories[group]['129_1'].to_numpy()
        user_order=np.argsort(users)
        raw_user=table['129_1'].to_numpy()
        positions=np.searchsorted(users[user_order],raw_user)
        if np.any(positions>=len(users)) or not np.array_equal(users[user_order[positions]],raw_user): raise ValueError('Unmatched exposure user')
        rows=user_order[positions]
        features=np.empty((len(table),len(FIELDS)),dtype=np.int32)
        for col,field in enumerate(FIELDS):
            raw=histories[group][field].to_numpy()[rows] if field in FIELDS[:6] else table[field].to_numpy()
            features[:,col]=encode_ids(raw,vocab[field])
        labels=pc.list_flatten(table['label_0']).to_numpy().reshape(-1,2)
        if not np.all((labels==0)|(labels==1)) or not np.all(labels.sum(1)==1): raise ValueError('Invalid labels')
        content=content_rows(table['205'].to_numpy(),sorted_keys,order)
        outputs={'features':features,'labels':labels[:,1].astype(np.float32),'user_rows':rows.astype(np.int32),'content':content,'source_rows':table['source_row'].to_numpy()}
        for name,value in outputs.items():np.save(output/(split+'.'+name+'.npy'),value)
        statistics[split]={'rows':len(table),'target_id_oov_exposures':int((features[:,6]==1).sum()),'user_id_oov_exposures':int((features[:,0]==1).sum()),'content_missing_exposures':int((content==0).sum())}
    artifacts={path.name:{'bytes':path.stat().st_size,'sha256':sha256(path)} for path in sorted(output.glob('*.npy'))}
    manifest={'format':'shared-feature-store-v1','revision':reports[0]['revision'],'fields':FIELDS,
        'vocab_sizes':{f:len(v)+2 for f,v in vocab.items()},'input_sha256':inputs,'arrays':artifacts,
        'splits':statistics,'history_length':1000,'padding_index':0,'oov_index':1,'content_missing_index':0,
        'vocabulary_scope':'train targets and train-user histories only','seconds':time.perf_counter()-started,
        'process_peak_working_set_mib':psutil.Process().memory_info().peak_wset/2**20}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'vocab_sizes':manifest['vocab_sizes'],'splits':statistics,'seconds':manifest['seconds']},indent=2))


if __name__=='__main__':main()
