"""Shared ID, padding and content lookup contracts for CIKM and MUSE."""
import json
from pathlib import Path
import numpy as np

FIELDS = ['129_1','130_1','130_2','130_3','130_4','130_5','205','206','213','214']


def encode_ids(raw, sorted_train_keys):
    raw = np.asarray(raw)
    index = np.searchsorted(sorted_train_keys, raw)
    found = index < len(sorted_train_keys)
    found[found] &= sorted_train_keys[index[found]] == raw[found]
    encoded = np.ones(raw.shape, dtype=np.int32)
    encoded[found] = index[found] + 2
    encoded[raw == 0] = 0
    return encoded


def content_rows(raw, sorted_keys, rows_in_source):
    index = np.searchsorted(sorted_keys, raw)
    found = index < len(sorted_keys)
    found[found] &= sorted_keys[index[found]] == raw[found]
    found &= raw != 0
    result = np.zeros(raw.shape, dtype=np.int32)
    result[found] = rows_in_source[index[found]] + 1
    return result


class FeatureStore:
    """CPU memmap reader. Materialize only selected exposures and history positions."""
    def __init__(self, root, split):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if split not in ['train','validation','test']: raise ValueError('Invalid split')
        load = lambda name: np.load(self.root/(name+'.npy'), mmap_mode='r')
        self.features = load(split+'.features')
        self.labels = load(split+'.labels')
        self.user_rows = load(split+'.user_rows')
        self.target_content = load(split+'.content')
        self.source_rows = load(split+'.source_rows')
        group = 'test_users' if split == 'test' else 'train_users'
        self.hist_item = load(group+'.item')
        self.hist_category = load(group+'.category')
        self.hist_content = load(group+'.content')
        self.content = load('content_int8')

    def __len__(self): return len(self.labels)

    def batch(self, indices, history_length=1000, with_content=False):
        if not 1 <= history_length <= 1000: raise ValueError('History length must be 1..1000')
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices<0) or np.any(indices>=len(self)):
            raise ValueError('Invalid exposure indices')
        user = self.user_rows[indices]
        item = self.hist_item[user, -history_length:]
        history_content = self.hist_content[user, -history_length:]
        target_content = self.target_content[indices]
        result = dict(features=self.features[indices], labels=self.labels[indices],
            history_item=item, history_category=self.hist_category[user,-history_length:],
            history_valid=item!=0, history_content_available=history_content!=0,
            target_content_available=target_content!=0, source_rows=self.source_rows[indices])
        if with_content:
            result['history_content'] = self.content[history_content].astype(np.float32)/127
            result['target_content'] = self.content[target_content].astype(np.float32)/127
        return result
