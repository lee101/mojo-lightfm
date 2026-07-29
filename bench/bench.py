"""Benchmarks against LightFM 1.17; run only through `pixi run bench`."""

from __future__ import annotations

import math
import os
import platform
import time

import numpy as np
import scipy.sparse as sp
from lightfm import LightFM as UpstreamLightFM

from mojo_lightfm import LightFM as MojoLightFM


def elapsed(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def interaction_matrix(n_users, n_items, per_user, seed=0):
    rng = np.random.RandomState(seed)
    rows = np.repeat(np.arange(n_users, dtype=np.int32), per_user)
    cols = np.concatenate(
        [rng.choice(n_items, per_user, replace=False) for _ in range(n_users)]
    ).astype(np.int32)
    return sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    )


def feature_matrix(rows, categories, identity=False):
    row = np.arange(rows, dtype=np.int32)
    category = row % categories
    values = np.ones(rows, dtype=np.float32)
    metadata = sp.csr_matrix((values, (row, category)), shape=(rows, categories))
    if identity:
        return sp.hstack(
            [sp.identity(rows, dtype=np.float32, format="csr"), metadata],
            format="csr",
        )
    return metadata


def prediction_case(num_threads=1):
    n_users, n_items, pairs = 5_000, 8_000, 1_000_000
    matrix = interaction_matrix(n_users, n_items, 2)
    ours = MojoLightFM(no_components=32, random_state=1).fit(matrix, epochs=0)
    theirs = UpstreamLightFM(no_components=32, random_state=1).fit(matrix, epochs=0)
    rng = np.random.RandomState(2)
    users = rng.randint(0, n_users, size=pairs).astype(np.int32)
    items = rng.randint(0, n_items, size=pairs).astype(np.int32)
    return (
        lambda: ours.predict(users, items, num_threads=num_threads),
        lambda: theirs.predict(users, items, num_threads=num_threads),
    )


def hybrid_prediction_case():
    n_users, n_items, pairs = 4_000, 6_000, 300_000
    matrix = interaction_matrix(n_users, n_items, 2)
    user_features = feature_matrix(n_users, 64, identity=True)
    item_features = feature_matrix(n_items, 96, identity=True)
    ours = MojoLightFM(no_components=32, random_state=1).fit(
        matrix,
        user_features=user_features,
        item_features=item_features,
        epochs=0,
    )
    theirs = UpstreamLightFM(no_components=32, random_state=1).fit(
        matrix,
        user_features=user_features,
        item_features=item_features,
        epochs=0,
    )
    rng = np.random.RandomState(3)
    users = rng.randint(0, n_users, size=pairs).astype(np.int32)
    items = rng.randint(0, n_items, size=pairs).astype(np.int32)
    return (
        lambda: ours.predict(
            users,
            items,
            user_features=user_features,
            item_features=item_features,
            num_threads=1,
        ),
        lambda: theirs.predict(
            users,
            items,
            user_features=user_features,
            item_features=item_features,
            num_threads=1,
        ),
    )


def fit_case(loss):
    matrix = interaction_matrix(2_000, 3_000, 10, seed=4)
    return (
        lambda: MojoLightFM(
            no_components=32, loss=loss, random_state=5
        ).fit(matrix, epochs=1, num_threads=1),
        lambda: UpstreamLightFM(
            no_components=32, loss=loss, random_state=5
        ).fit(matrix, epochs=1, num_threads=1),
    )


def logistic_case():
    positives = interaction_matrix(2_000, 3_000, 10, seed=6)
    rng = np.random.RandomState(7)
    rows = rng.randint(0, 2_000, size=20_000)
    cols = rng.randint(0, 3_000, size=20_000)
    negatives = sp.coo_matrix(
        (-np.ones(20_000, dtype=np.float32), (rows, cols)),
        shape=positives.shape,
    )
    matrix = (positives + negatives).tocoo()
    return (
        lambda: MojoLightFM(
            no_components=32, loss="logistic", random_state=8
        ).fit(matrix, epochs=1, num_threads=1),
        lambda: UpstreamLightFM(
            no_components=32, loss="logistic", random_state=8
        ).fit(matrix, epochs=1, num_threads=1),
    )


def rank_case():
    n_users, n_items = 500, 2_000
    combined = interaction_matrix(n_users, n_items, 13, seed=9).tocsr()
    rows = np.repeat(np.arange(n_users), 13)
    train_mask = np.tile(np.arange(13) < 10, n_users)
    train = sp.coo_matrix(
        (
            combined.data[train_mask],
            (rows[train_mask], combined.indices[train_mask]),
        ),
        shape=combined.shape,
    ).tocsr()
    test = sp.coo_matrix(
        (
            combined.data[~train_mask],
            (rows[~train_mask], combined.indices[~train_mask]),
        ),
        shape=combined.shape,
    ).tocsr()
    ours = MojoLightFM(no_components=32, loss="warp", random_state=10).fit(
        train, epochs=5
    )
    theirs = UpstreamLightFM(no_components=32, loss="warp", random_state=10).fit(
        train, epochs=5
    )
    return (
        lambda: ours.predict_rank(test, train_interactions=train, num_threads=1),
        lambda: theirs.predict_rank(test, train_interactions=train, num_threads=1),
    )


CASES = [
    ("predict, identity (1M pairs, 32 factors)", prediction_case),
    (
        "predict, identity (1M pairs, 32 factors, 4 threads)",
        lambda: prediction_case(4),
    ),
    ("predict, hybrid (300k pairs, 32 factors)", hybrid_prediction_case),
    ("logistic fit (40k interactions, 32 factors)", logistic_case),
    ("BPR fit (20k interactions, 32 factors)", lambda: fit_case("bpr")),
    ("WARP fit (20k interactions, 32 factors)", lambda: fit_case("warp")),
    ("predict_rank (500 users x 2k items)", rank_case),
]


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| workload | mojo-lightfm | lightfm 1.17 | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, factory in CASES:
        ours, theirs = factory()
        ours()
        theirs()
        mojo_time = elapsed(ours)
        upstream_time = elapsed(theirs)
        ratio = upstream_time / mojo_time
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
