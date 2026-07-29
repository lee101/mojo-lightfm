"""Ranking metrics with signatures matching lightfm.evaluation."""

from __future__ import annotations

import numpy as np


def precision_at_k(
    model,
    test_interactions,
    train_interactions=None,
    k=10,
    user_features=None,
    item_features=None,
    preserve_rows=False,
    num_threads=1,
    check_intersections=True,
):
    ranks = model.predict_rank(
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features,
        item_features=item_features,
        num_threads=num_threads,
        check_intersections=check_intersections,
    )
    ranks.data = np.less(ranks.data, k).astype(np.float32)
    values = np.asarray(ranks.sum(axis=1)).ravel() / k
    if not preserve_rows:
        values = values[test_interactions.getnnz(axis=1) > 0]
    return values


def recall_at_k(
    model,
    test_interactions,
    train_interactions=None,
    k=10,
    user_features=None,
    item_features=None,
    preserve_rows=False,
    num_threads=1,
    check_intersections=True,
):
    ranks = model.predict_rank(
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features,
        item_features=item_features,
        num_threads=num_threads,
        check_intersections=check_intersections,
    )
    ranks.data = np.less(ranks.data, k).astype(np.float32)
    retrieved = np.asarray(test_interactions.getnnz(axis=1)).ravel()
    hits = np.asarray(ranks.sum(axis=1)).ravel()
    mask = retrieved > 0
    if not preserve_rows:
        return hits[mask] / retrieved[mask]
    result = np.zeros_like(hits, dtype=np.float64)
    result[mask] = hits[mask] / retrieved[mask]
    return result


def auc_score(
    model,
    test_interactions,
    train_interactions=None,
    user_features=None,
    item_features=None,
    preserve_rows=False,
    num_threads=1,
    check_intersections=True,
):
    ranks = model.predict_rank(
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features,
        item_features=item_features,
        num_threads=num_threads,
        check_intersections=check_intersections,
    )
    train_counts = (
        np.zeros(ranks.shape[0], dtype=np.int32)
        if train_interactions is None
        else np.asarray(train_interactions.getnnz(axis=1)).ravel()
    )
    result = np.full(ranks.shape[0], 0.5, dtype=np.float32)
    for user in range(ranks.shape[0]):
        row = np.sort(ranks.data[ranks.indptr[user] : ranks.indptr[user + 1]])
        positives = len(row)
        negatives = ranks.shape[1] - positives - train_counts[user]
        if positives and negatives:
            adjusted = np.maximum(row - np.arange(positives), 0)
            result[user] = np.mean(1.0 - adjusted / negatives)
    if not preserve_rows:
        result = result[test_interactions.getnnz(axis=1) > 0]
    return result


def reciprocal_rank(
    model,
    test_interactions,
    train_interactions=None,
    user_features=None,
    item_features=None,
    preserve_rows=False,
    num_threads=1,
    check_intersections=True,
):
    ranks = model.predict_rank(
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features,
        item_features=item_features,
        num_threads=num_threads,
        check_intersections=check_intersections,
    )
    result = np.zeros(ranks.shape[0], dtype=np.float32)
    for user in range(ranks.shape[0]):
        row = ranks.data[ranks.indptr[user] : ranks.indptr[user + 1]]
        if len(row):
            result[user] = 1.0 / (row.min() + 1.0)
    if not preserve_rows:
        result = result[test_interactions.getnnz(axis=1) > 0]
    return result
