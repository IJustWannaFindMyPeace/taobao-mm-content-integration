import sys
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from feature_store import FeatureStore,encode_ids,content_rows


class FeatureStoreContracts(unittest.TestCase):
    def test_oov_can_have_content(self):
        raw=np.array([0,-8,99,2])
        ids=encode_ids(raw,np.array([-8,2]))
        content=content_rows(raw,np.array([-8,99]),np.array([1,0]))
        self.assertEqual(ids.tolist(),[0,2,1,3])
        self.assertEqual(content.tolist(),[0,2,1,0])
        self.assertTrue(ids[2]!=0 and content[2]!=0)
        self.assertTrue(ids[3]!=0 and content[3]==0)

    def test_empty_training_vocabulary(self):
        self.assertEqual(encode_ids(np.array([0,-1,9]),np.array([],dtype=np.int64)).tolist(),[0,1,1])

    def test_reader_retains_missing_and_oov(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'manifest.json').write_text(json.dumps({'format':'shared-feature-store-v1'}))
            arrays={'train.features':np.array([[2]*10,[1]*10]),'train.labels':np.array([1.,0.]),
                'train.user_rows':np.array([0,0]),'train.content':np.array([0,1]),
                'train.source_rows':np.array([5,9]),'train_users.item':np.array([[0,1,2]]),
                'train_users.category':np.array([[0,1,2]]),'train_users.content':np.array([[0,1,0]]),
                'content_int8':np.array([[0]*128,[127]*128],dtype=np.int8)}
            for name,array in arrays.items():np.save(root/(name+'.npy'),array)
            store=FeatureStore(root,'train')
            batch=store.batch([1,0],3,with_content=True)
            self.assertEqual(batch['history_valid'].tolist(),[[False,True,True]]*2)
            self.assertEqual(batch['target_content_available'].tolist(),[True,False])
            self.assertEqual(batch['source_rows'].tolist(),[9,5])
            self.assertTrue(np.all(batch['target_content'][0]==1))
            with self.assertRaises(ValueError):store.batch([-1])
            # Explicitly close maps before Windows temporary-directory cleanup.
            for value in vars(store).values():
                if isinstance(value,np.memmap):value._mmap.close()


if __name__=='__main__':unittest.main()
