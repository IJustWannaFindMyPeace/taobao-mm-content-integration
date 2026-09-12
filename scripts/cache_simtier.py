"""Cache frozen count features for training and validation only."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from feature_store import FeatureStore
from simtier import histogram
from fetch_source import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--history-length',type=int,default=200)
    parser.add_argument('--bins',type=int,default=20)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh output required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    result={'version':'cosine_count_histogram_v1','bins':args.bins,'history_length':args.history_length,
        'store_manifest_sha256':sha256(args.store/'manifest.json'),'boundary':'floor((cosine+1)*N/2), clamped to [0,N-1]',
        'missing_policy':'exclude unavailable pairs; all-zero histogram for unavailable target','splits':{}}
    for split in ['train','validation']:
        store=FeatureStore(args.store,split)
        output=np.lib.format.open_memmap(args.output_dir/(split+'.npy'),mode='w+',dtype=np.float32,shape=(len(store),args.bins))
        matched_pairs=0
        for start in range(0,len(store),256):
            b=store.batch(np.arange(start,min(start+256,len(store))),args.history_length,with_content=True)
            with torch.no_grad():
                counts=histogram(*[torch.as_tensor(b[k],device='cuda') for k in ['target_content','history_content','history_valid','target_content_available','history_content_available']],bins=args.bins)
            array=counts.cpu().numpy();output[start:start+len(array)]=array;matched_pairs+=int(array.sum())
            if start%65536==0:print(split,start,len(store),flush=True)
        output.flush();del output
        result['splits'][split]={'rows':len(store),'matched_pairs':matched_pairs,'sha256':sha256(args.output_dir/(split+'.npy'))}
    result['seconds']=time.perf_counter()-started
    result['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    (args.output_dir/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
