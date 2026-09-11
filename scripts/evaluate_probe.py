"""Evaluate saved training probe predictions; no test-set claim."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--training-report', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.training_report.read_text(encoding='utf-8'))
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if digest != source['input_sha256']:
        raise ValueError('Prediction source does not match training input')
    rows = pq.read_table(args.input, columns=['129_1', 'label_0']).to_pylist()
    saved = torch.load(args.predictions, weights_only=True, map_location='cpu')
    if not torch.equal(saved['row_index'], torch.arange(len(rows))):
        raise ValueError('Missing, duplicate or unordered prediction rows')
    labels = torch.tensor([r['label_0'][1] for r in rows], dtype=torch.float32)
    if not torch.equal(labels, saved['labels']):
        raise ValueError('Prediction labels do not match source rows')
    result = evaluate(labels.numpy(), saved['logits'].numpy(), [r['129_1'] for r in rows])
    result.update(scope='training_probe_only_not_validation_or_test', input_sha256=digest,
        predictions_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        limitations=['GAUC groups complete only within this probe', 'training fit is not evidence of generalization'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
