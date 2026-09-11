"""Whole-population binary ranking metrics. Undefined metrics return None."""
import numpy as np


def _inputs(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not len(y):
        raise ValueError('Expected nonempty equal-length one-dimensional arrays')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError('Labels must be binary and scores finite')
    return y.astype(np.int64), s


def auc(labels, scores):
    y, s = _inputs(labels, scores)
    positive = int(y.sum())
    negative = len(y) - positive
    if not positive or not negative:
        return None
    order = np.argsort(s, kind='stable')
    sorted_scores = s[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, len(y)]
    rank_sum = 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        # One-based average rank gives half credit to tied positive/negative pairs.
        rank_sum += y[order[start:end]].sum() * (start + 1 + end) / 2
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def evaluate(labels, logits, user_ids):
    y, s = _inputs(labels, logits)
    users = np.asarray(user_ids)
    if users.ndim != 1 or len(users) != len(y):
        raise ValueError('One user ID required per row')
    if any(v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)) for v in users):
        raise ValueError('User IDs must be present and finite')
    groups = {}
    for index, user in enumerate(users):
        groups.setdefault(user, []).append(index)
    numerator, valid_rows, valid_users = 0.0, 0, 0
    for indices in groups.values():
        value = auc(y[indices], s[indices])
        if value is not None:
            numerator += len(indices) * value
            valid_rows += len(indices)
            valid_users += 1
    return dict(rows=len(y), positives=int(y.sum()), auc=auc(y, s),
        logloss=float(np.mean(np.logaddexp(0, s) - y * s)),
        gauc=numerator / valid_rows if valid_rows else None,
        gauc_weighting='exposure_count_among_users_with_both_labels',
        users=len(groups), valid_users=valid_users, valid_rows=valid_rows,
        excluded_users=len(groups) - valid_users, excluded_rows=len(y) - valid_rows,
        valid_row_fraction=valid_rows / len(y))
