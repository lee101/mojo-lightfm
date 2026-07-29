"""Numerical and behavioural parity against lightfm 1.17."""

import numpy as np
import pytest
import scipy.sparse as sp

upstream = pytest.importorskip("lightfm")
upstream_evaluation = pytest.importorskip("lightfm.evaluation")

from mojo_lightfm import LightFM
from mojo_lightfm import evaluation


def interactions(n_users=8, n_items=20, positives=4, seed=3):
    rng = np.random.RandomState(seed)
    rows = np.repeat(np.arange(n_users), positives)
    cols = np.concatenate(
        [rng.choice(n_items, positives, replace=False) for _ in range(n_users)]
    )
    return sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    )


def initialized_pair(matrix, **kwargs):
    ours = LightFM(random_state=7, **kwargs).fit(matrix, epochs=0)
    theirs = upstream.LightFM(random_state=7, **kwargs).fit(matrix, epochs=0)
    return ours, theirs


def test_initialization_matches_upstream():
    ours, theirs = initialized_pair(interactions(), no_components=7)
    assert np.array_equal(ours.item_embeddings, theirs.item_embeddings)
    assert np.array_equal(ours.user_embeddings, theirs.user_embeddings)
    assert np.array_equal(ours.item_embedding_gradients, theirs.item_embedding_gradients)
    assert np.array_equal(ours.user_embedding_gradients, theirs.user_embedding_gradients)


def test_identity_prediction_matches_upstream():
    matrix = interactions()
    ours, theirs = initialized_pair(matrix, no_components=7)
    user_ids = np.repeat(np.arange(matrix.shape[0], dtype=np.int32), matrix.shape[1])
    item_ids = np.tile(np.arange(matrix.shape[1], dtype=np.int32), matrix.shape[0])
    actual = ours.predict(user_ids, item_ids)
    expected = theirs.predict(user_ids, item_ids)
    assert actual.dtype == np.float32
    assert np.allclose(actual, expected, atol=2e-8)


def test_simd_tail_prediction_matches_upstream():
    matrix = interactions()
    ours, theirs = initialized_pair(matrix, no_components=13)
    users = np.repeat(np.arange(matrix.shape[0], dtype=np.int32), matrix.shape[1])
    items = np.tile(np.arange(matrix.shape[1], dtype=np.int32), matrix.shape[0])
    assert np.allclose(
        ours.predict(users, items),
        theirs.predict(users, items),
        atol=2e-8,
    )


@pytest.mark.parametrize("pairs", [32767, 32768])
def test_parallel_prediction_threshold_matches_serial(pairs):
    matrix = interactions()
    model, _ = initialized_pair(matrix, no_components=13)
    users = np.arange(pairs, dtype=np.int32) % matrix.shape[0]
    items = np.arange(pairs, dtype=np.int32)[::-1] % matrix.shape[1]
    serial = model.predict(users, items, num_threads=1)
    parallel = model.predict(users, items, num_threads=2)
    assert np.array_equal(parallel, serial)


def test_hybrid_prediction_and_representations_match_upstream():
    matrix = interactions(n_users=6, n_items=9)
    user_features = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.2],
                [1.0, 0.0, 0.3],
                [0.0, 1.0, 0.7],
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.9],
            ],
            dtype=np.float32,
        )
    )
    item_features = sp.csr_matrix(
        np.eye(9, 4, dtype=np.float32) + 0.1,
        dtype=np.float32,
    )
    ours = LightFM(no_components=5, random_state=7).fit(
        matrix,
        user_features=user_features,
        item_features=item_features,
        epochs=0,
    )
    theirs = upstream.LightFM(no_components=5, random_state=7).fit(
        matrix,
        user_features=user_features,
        item_features=item_features,
        epochs=0,
    )
    users = np.repeat(np.arange(6, dtype=np.int32), 9)
    items = np.tile(np.arange(9, dtype=np.int32), 6)
    actual = ours.predict(
        users, items, user_features=user_features, item_features=item_features
    )
    expected = theirs.predict(
        users, items, user_features=user_features, item_features=item_features
    )
    assert np.allclose(actual, expected, atol=3e-8)
    for ours_repr, theirs_repr in (
        (
            ours.get_user_representations(user_features),
            theirs.get_user_representations(user_features),
        ),
        (
            ours.get_item_representations(item_features),
            theirs.get_item_representations(item_features),
        ),
    ):
        assert np.allclose(ours_repr[0], theirs_repr[0], atol=2e-8)
        assert np.allclose(ours_repr[1], theirs_repr[1], atol=2e-8)


@pytest.mark.parametrize("schedule", ["adagrad", "adadelta"])
def test_logistic_epoch_matches_upstream(schedule):
    matrix = interactions()
    ours = LightFM(
        no_components=6,
        loss="logistic",
        learning_schedule=schedule,
        random_state=7,
    ).fit(matrix, epochs=1)
    theirs = upstream.LightFM(
        no_components=6,
        loss="logistic",
        learning_schedule=schedule,
        random_state=7,
    ).fit(matrix, epochs=1)
    users = np.repeat(np.arange(8, dtype=np.int32), 20)
    items = np.tile(np.arange(20, dtype=np.int32), 8)
    assert np.allclose(
        ours.predict(users, items),
        theirs.predict(users, items),
        atol=2e-3,
        rtol=2e-3,
    )


def disjoint_train_test():
    matrix = interactions(n_users=10, n_items=30, positives=6, seed=9).tocsr()
    train = matrix.copy()
    test = matrix.copy()
    train.data[:] = 1.0
    test.data[:] = 1.0
    train_mask = np.tile([True, True, True, True, False, False], 10)
    test_mask = ~train_mask
    train = sp.coo_matrix(
        (
            train.data[train_mask],
            (
                np.repeat(np.arange(10), 6)[train_mask],
                train.indices[train_mask],
            ),
        ),
        shape=matrix.shape,
    ).tocsr()
    test = sp.coo_matrix(
        (
            test.data[test_mask],
            (
                np.repeat(np.arange(10), 6)[test_mask],
                test.indices[test_mask],
            ),
        ),
        shape=matrix.shape,
    ).tocsr()
    return train, test


def test_predict_rank_matches_upstream():
    train, test = disjoint_train_test()
    ours, theirs = initialized_pair(train, no_components=8)
    actual = ours.predict_rank(test, train_interactions=train)
    expected = theirs.predict_rank(test, train_interactions=train)
    assert np.array_equal(actual.indices, expected.indices)
    assert np.array_equal(actual.indptr, expected.indptr)
    assert np.array_equal(actual.data, expected.data)


@pytest.mark.parametrize(
    ("ours_metric", "upstream_metric", "kwargs"),
    [
        (evaluation.precision_at_k, upstream_evaluation.precision_at_k, {"k": 5}),
        (evaluation.recall_at_k, upstream_evaluation.recall_at_k, {"k": 5}),
        (evaluation.auc_score, upstream_evaluation.auc_score, {}),
        (evaluation.reciprocal_rank, upstream_evaluation.reciprocal_rank, {}),
    ],
)
def test_evaluation_matches_upstream(ours_metric, upstream_metric, kwargs):
    train, test = disjoint_train_test()
    ours, theirs = initialized_pair(train, no_components=8)
    actual = ours_metric(
        ours, test, train_interactions=train, preserve_rows=True, **kwargs
    )
    expected = upstream_metric(
        theirs, test, train_interactions=train, preserve_rows=True, **kwargs
    )
    assert np.allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize("loss", ["bpr", "warp"])
def test_pairwise_training_quality_matches_upstream(loss):
    rng = np.random.RandomState(4)
    n_users, n_items, latent = 50, 80, 6
    preference = rng.normal(size=(n_users, latent)) @ rng.normal(
        size=(n_items, latent)
    ).T
    rows, cols = [], []
    for user in range(n_users):
        positives = np.argsort(preference[user])[-8:]
        rows.extend([user] * len(positives))
        cols.extend(positives)
    matrix = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    )
    ours = LightFM(
        no_components=10, loss=loss, random_state=2
    ).fit(matrix, epochs=15)
    theirs = upstream.LightFM(
        no_components=10, loss=loss, random_state=2
    ).fit(matrix, epochs=15)
    actual = evaluation.auc_score(ours, matrix.tocsr()).mean()
    expected = upstream_evaluation.auc_score(theirs, matrix.tocsr()).mean()
    assert actual > 0.75
    assert actual == pytest.approx(expected, abs=0.025)


def test_hybrid_feature_training_matches_upstream():
    n_users, n_items, groups = 30, 45, 3
    user_features = sp.csr_matrix(
        (
            np.ones(n_users, dtype=np.float32),
            (np.arange(n_users), np.arange(n_users) % groups),
        ),
        shape=(n_users, groups),
    )
    item_features = sp.csr_matrix(
        (
            np.ones(n_items, dtype=np.float32),
            (np.arange(n_items), np.arange(n_items) % groups),
        ),
        shape=(n_items, groups),
    )
    rows, cols = [], []
    for user in range(n_users):
        positive = np.flatnonzero(np.arange(n_items) % groups == user % groups)[:8]
        rows.extend([user] * len(positive))
        cols.extend(positive)
    matrix = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    )
    ours = LightFM(
        no_components=6, loss="warp", random_state=3
    ).fit(
        matrix,
        user_features=user_features,
        item_features=item_features,
        epochs=15,
    )
    theirs = upstream.LightFM(
        no_components=6, loss="warp", random_state=3
    ).fit(
        matrix,
        user_features=user_features,
        item_features=item_features,
        epochs=15,
    )
    actual = evaluation.auc_score(
        ours,
        matrix.tocsr(),
        user_features=user_features,
        item_features=item_features,
    ).mean()
    expected = upstream_evaluation.auc_score(
        theirs,
        matrix.tocsr(),
        user_features=user_features,
        item_features=item_features,
    ).mean()
    assert actual > 0.7
    assert actual == pytest.approx(expected, abs=0.02)


def test_fit_partial_continues_training():
    matrix = interactions()
    one_call = LightFM(loss="warp", random_state=11).fit(matrix, epochs=3)
    partial = LightFM(loss="warp", random_state=11)
    partial.fit_partial(matrix, epochs=1)
    partial.fit_partial(matrix, epochs=2)
    assert np.array_equal(one_call.item_embeddings, partial.item_embeddings)
    assert np.array_equal(one_call.user_embeddings, partial.user_embeddings)


def test_sample_weights_have_upstream_effect():
    matrix = interactions()
    weights = matrix.copy()
    weights.data = np.linspace(0.2, 2.0, matrix.nnz, dtype=np.float32)
    ours = LightFM(loss="logistic", random_state=5).fit(
        matrix, sample_weight=weights, epochs=2
    )
    theirs = upstream.LightFM(loss="logistic", random_state=5).fit(
        matrix, sample_weight=weights, epochs=2
    )
    users = np.repeat(np.arange(8, dtype=np.int32), 20)
    items = np.tile(np.arange(20, dtype=np.int32), 8)
    assert np.allclose(
        ours.predict(users, items),
        theirs.predict(users, items),
        atol=4e-3,
        rtol=4e-3,
    )


def test_regularized_logistic_tracks_upstream():
    matrix = interactions()
    ours = LightFM(
        loss="logistic",
        item_alpha=1e-3,
        user_alpha=1e-3,
        random_state=7,
    ).fit(matrix, epochs=3)
    theirs = upstream.LightFM(
        loss="logistic",
        item_alpha=1e-3,
        user_alpha=1e-3,
        random_state=7,
    ).fit(matrix, epochs=3)
    users = np.repeat(np.arange(8, dtype=np.int32), 20)
    items = np.tile(np.arange(20, dtype=np.int32), 8)
    assert np.allclose(
        ours.predict(users, items),
        theirs.predict(users, items),
        atol=3e-3,
        rtol=3e-3,
    )


def test_validation_and_unsupported_loss():
    matrix = interactions()
    with pytest.raises(NotImplementedError, match="warp-kos"):
        LightFM(loss="warp-kos").fit(matrix)
    with pytest.raises(ValueError, match="threads"):
        LightFM().fit(matrix, num_threads=0)
    with pytest.raises(ValueError, match="fit the model"):
        LightFM().predict([0], [0])


def test_set_params_matches_estimator_convention():
    model = LightFM().set_params(loss="warp", learning_rate=0.01)
    assert model.loss == "warp"
    assert model.get_params()["learning_rate"] == 0.01
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(not_a_parameter=True)


def test_native_boundary_rejects_narrowing_and_invalid_model_buffers():
    model = LightFM(random_state=1).fit(interactions(), epochs=0)
    with pytest.raises(ValueError, match="int32"):
        model.predict([2**32], [0])
    with pytest.raises(TypeError, match="integers"):
        model.predict([0.0], [0])

    model.item_embeddings = model.item_embeddings[:, ::-1]
    with pytest.raises(ValueError, match="C-contiguous"):
        model.predict([0], [0])


def test_empty_training_and_prediction_do_not_pass_empty_buffers_to_native():
    matrix = sp.coo_matrix((0, 0), dtype=np.float32)
    model = LightFM(random_state=1).fit(matrix, epochs=2)
    assert model.predict([], []).shape == (0,)
    biases, embeddings = model.get_item_representations(
        sp.csr_matrix((0, 0), dtype=np.float32)
    )
    assert biases.shape == (0,)
    assert embeddings.shape == (0, model.no_components)


def test_training_rejects_fractional_epochs_and_non_finite_abi_parameters():
    matrix = interactions()
    with pytest.raises(ValueError, match="epochs"):
        LightFM().fit(matrix, epochs=1.5)
    with pytest.raises(ValueError, match="learning_rate"):
        LightFM(learning_rate=np.inf).fit(matrix)
