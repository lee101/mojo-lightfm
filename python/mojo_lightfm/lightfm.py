"""Python-compatible LightFM estimator backed by Mojo compute kernels."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._lib import addr, lib

CYTHON_DTYPE = np.float32
_INT32_MAX = np.iinfo(np.int32).max


def _check_sparse_int32_range(matrix) -> None:
    if any(dimension > _INT32_MAX for dimension in matrix.shape):
        raise ValueError("sparse matrix dimensions exceed the int32 ABI")
    for values in (getattr(matrix, "row", None), getattr(matrix, "col", None),
                   getattr(matrix, "indices", None), getattr(matrix, "indptr", None)):
        if values is not None and values.size:
            if values.min() < 0 or values.max() > _INT32_MAX:
                raise ValueError("sparse matrix indices exceed the int32 ABI")


def _csr(matrix, shape=None) -> sp.csr_matrix:
    source = sp.csr_matrix(matrix, shape=shape)
    _check_sparse_int32_range(source)
    result = source.astype(np.float32, copy=False)
    result.sum_duplicates()
    result.sort_indices()
    if result.indices.dtype != np.int32 or result.indptr.dtype != np.int32:
        result = sp.csr_matrix(
            (
                np.ascontiguousarray(result.data, dtype=np.float32),
                np.ascontiguousarray(result.indices, dtype=np.int32),
                np.ascontiguousarray(result.indptr, dtype=np.int32),
            ),
            shape=result.shape,
        )
    else:
        result.data = np.ascontiguousarray(result.data, dtype=np.float32)
        result.indices = np.ascontiguousarray(result.indices, dtype=np.int32)
        result.indptr = np.ascontiguousarray(result.indptr, dtype=np.int32)
    return result


def _coo(matrix) -> sp.coo_matrix:
    source = sp.coo_matrix(matrix)
    _check_sparse_int32_range(source)
    result = source.astype(np.float32, copy=False)
    result.row = np.ascontiguousarray(result.row, dtype=np.int32)
    result.col = np.ascontiguousarray(result.col, dtype=np.int32)
    result.data = np.ascontiguousarray(result.data, dtype=np.float32)
    return result


def _ids(values, name):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.size == 0:
        return np.empty(0, dtype=np.int32)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    if array.size and (array.min() < 0 or array.max() > _INT32_MAX):
        raise ValueError(f"{name} values must fit in int32 and be non-negative")
    return np.ascontiguousarray(array, dtype=np.int32)


class LightFM:
    """Hybrid latent-factor recommender with the upstream LightFM API."""

    def __init__(
        self,
        no_components=10,
        k=5,
        n=10,
        learning_schedule="adagrad",
        loss="logistic",
        learning_rate=0.05,
        rho=0.95,
        epsilon=1e-6,
        item_alpha=0.0,
        user_alpha=0.0,
        max_sampled=10,
        random_state=None,
    ):
        for name, value in (
            ("no_components", no_components),
            ("k", k),
            ("n", n),
            ("max_sampled", max_sampled),
        ):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if item_alpha < 0.0 or user_alpha < 0.0:
            raise ValueError("item_alpha and user_alpha must be non-negative")
        if not 0 < rho < 1:
            raise ValueError("rho must be between zero and one")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if learning_schedule not in ("adagrad", "adadelta"):
            raise ValueError("unknown learning schedule")
        if loss not in ("logistic", "warp", "bpr", "warp-kos"):
            raise ValueError("unknown loss")

        self.loss = loss
        self.learning_schedule = learning_schedule
        self.no_components = int(no_components)
        self.learning_rate = learning_rate
        self.k = int(k)
        self.n = int(n)
        self.rho = rho
        self.epsilon = epsilon
        self.max_sampled = int(max_sampled)
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)
        self._reset_state()

    def _reset_state(self):
        self.item_embeddings = None
        self.item_embedding_gradients = None
        self.item_embedding_momentum = None
        self.item_biases = None
        self.item_bias_gradients = None
        self.item_bias_momentum = None
        self.user_embeddings = None
        self.user_embedding_gradients = None
        self.user_embedding_momentum = None
        self.user_biases = None
        self.user_bias_gradients = None
        self.user_bias_momentum = None

    def _check_initialized(self):
        if self.item_embeddings is None or self.user_embeddings is None:
            raise ValueError(
                "You must fit the model before trying to obtain predictions."
            )

    def _check_model_buffers(self):
        self._check_initialized()
        c = self.no_components
        for name, array in (
            ("item_embeddings", self.item_embeddings),
            ("user_embeddings", self.user_embeddings),
        ):
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float32
                or array.ndim != 2
                or array.shape[1] != c
            ):
                raise ValueError(
                    f"{name} must be a two-dimensional float32 array "
                    f"with {c} columns"
                )
        specifications = (
            ("item_embeddings", self.item_embeddings, (None, c)),
            ("item_embedding_gradients", self.item_embedding_gradients, self.item_embeddings.shape),
            ("item_embedding_momentum", self.item_embedding_momentum, self.item_embeddings.shape),
            ("item_biases", self.item_biases, (self.item_embeddings.shape[0],)),
            ("item_bias_gradients", self.item_bias_gradients, (self.item_embeddings.shape[0],)),
            ("item_bias_momentum", self.item_bias_momentum, (self.item_embeddings.shape[0],)),
            ("user_embeddings", self.user_embeddings, (None, c)),
            ("user_embedding_gradients", self.user_embedding_gradients, self.user_embeddings.shape),
            ("user_embedding_momentum", self.user_embedding_momentum, self.user_embeddings.shape),
            ("user_biases", self.user_biases, (self.user_embeddings.shape[0],)),
            ("user_bias_gradients", self.user_bias_gradients, (self.user_embeddings.shape[0],)),
            ("user_bias_momentum", self.user_bias_momentum, (self.user_embeddings.shape[0],)),
        )
        for name, array, shape in specifications:
            if not isinstance(array, np.ndarray) or array.dtype != np.float32:
                raise ValueError(f"{name} must be a float32 NumPy array")
            if not array.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
            if not array.flags.writeable:
                raise ValueError(f"{name} must be writable")
            if len(shape) != array.ndim or any(
                expected is not None and actual != expected
                for actual, expected in zip(array.shape, shape)
            ):
                raise ValueError(f"{name} has an invalid shape")

    def _initialize(self, no_item_features, no_user_features):
        c = self.no_components
        self.item_embeddings = (
            (self.random_state.rand(no_item_features, c) - 0.5) / c
        ).astype(np.float32)
        self.user_embeddings = (
            (self.random_state.rand(no_user_features, c) - 0.5) / c
        ).astype(np.float32)
        self.item_biases = np.zeros(no_item_features, dtype=np.float32)
        self.user_biases = np.zeros(no_user_features, dtype=np.float32)
        self.item_embedding_gradients = np.zeros_like(self.item_embeddings)
        self.user_embedding_gradients = np.zeros_like(self.user_embeddings)
        self.item_bias_gradients = np.zeros_like(self.item_biases)
        self.user_bias_gradients = np.zeros_like(self.user_biases)
        self.item_embedding_momentum = np.zeros_like(self.item_embeddings)
        self.user_embedding_momentum = np.zeros_like(self.user_embeddings)
        self.item_bias_momentum = np.zeros_like(self.item_biases)
        self.user_bias_momentum = np.zeros_like(self.user_biases)
        if self.learning_schedule == "adagrad":
            self.item_embedding_gradients += 1.0
            self.user_embedding_gradients += 1.0
            self.item_bias_gradients += 1.0
            self.user_bias_gradients += 1.0

    def _construct_feature_matrices(
        self, n_users, n_items, user_features, item_features
    ):
        if user_features is None:
            user_features = sp.identity(
                n_users, dtype=np.float32, format="csr"
            )
        else:
            user_features = _csr(user_features)
        if item_features is None:
            item_features = sp.identity(
                n_items, dtype=np.float32, format="csr"
            )
        else:
            item_features = _csr(item_features)
        user_features = _csr(user_features)
        item_features = _csr(item_features)
        if n_users > user_features.shape[0]:
            raise Exception(
                "Number of user feature rows does not equal the number of users"
            )
        if n_items > item_features.shape[0]:
            raise Exception(
                "Number of item feature rows does not equal the number of items"
            )
        if (
            self.user_embeddings is not None
            and self.user_embeddings.shape[0] < user_features.shape[1]
        ):
            raise ValueError(
                "The user feature matrix specifies more features than there are "
                "estimated feature embeddings"
            )
        if (
            self.item_embeddings is not None
            and self.item_embeddings.shape[0] < item_features.shape[1]
        ):
            raise ValueError(
                "The item feature matrix specifies more features than there are "
                "estimated feature embeddings"
            )
        return user_features, item_features

    def _process_sample_weight(self, interactions, sample_weight):
        if sample_weight is None:
            if np.array_equiv(interactions.data, 1.0):
                return interactions.data
            return np.ones_like(interactions.data, dtype=np.float32)
        if self.loss == "warp-kos":
            raise NotImplementedError(
                "k-OS loss with sample weights not implemented."
            )
        if not isinstance(sample_weight, sp.coo_matrix):
            raise ValueError("Sample_weight must be a COO matrix.")
        if sample_weight.shape != interactions.shape:
            raise ValueError(
                "Sample weight and interactions matrices must be the same shape"
            )
        if not (
            np.array_equal(interactions.row, sample_weight.row)
            and np.array_equal(interactions.col, sample_weight.col)
        ):
            raise ValueError(
                "Sample weight and interaction matrix entries must be in the same order"
            )
        return np.ascontiguousarray(sample_weight.data, dtype=np.float32)

    def fit(
        self,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        epochs=1,
        num_threads=1,
        verbose=False,
    ):
        self._reset_state()
        return self.fit_partial(
            interactions,
            user_features=user_features,
            item_features=item_features,
            sample_weight=sample_weight,
            epochs=epochs,
            num_threads=num_threads,
            verbose=verbose,
        )

    def fit_partial(
        self,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        epochs=1,
        num_threads=1,
        verbose=False,
    ):
        if self.loss == "warp-kos":
            raise NotImplementedError("The warp-kos loss is not covered")
        if num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")
        if not isinstance(epochs, (int, np.integer)) or epochs < 0:
            raise ValueError("epochs must be a non-negative integer")
        float32_limit = np.finfo(np.float32).max
        for name in (
            "learning_rate", "item_alpha", "user_alpha", "rho", "epsilon"
        ):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if abs(value) > float32_limit:
                raise ValueError(f"{name} exceeds the float32 ABI")
        interactions = _coo(interactions)
        weights = self._process_sample_weight(interactions, sample_weight)
        n_users, n_items = interactions.shape
        user_features, item_features = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )
        for values in (
            user_features.data,
            item_features.data,
            interactions.data,
            weights,
        ):
            if not np.isfinite(values).all():
                raise ValueError("Not all input values are finite.")
        if self.item_embeddings is None:
            self._initialize(item_features.shape[1], user_features.shape[1])
        self._check_model_buffers()
        if item_features.shape[1] != self.item_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in item_features")
        if user_features.shape[1] != self.user_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in user_features")

        positives = _csr(interactions)
        order = np.arange(interactions.nnz, dtype=np.int32)
        user_work = np.empty(self.no_components + 1, dtype=np.float32)
        positive_work = np.empty_like(user_work)
        negative_work = np.empty_like(user_work)
        loss_kind = {"logistic": 0, "bpr": 1, "warp": 2}[self.loss]
        schedule = int(self.learning_schedule == "adadelta")
        function = lib().mlfm_fit_epoch
        epoch_range = range(epochs)
        if verbose:
            try:
                from tqdm import trange

                epoch_range = trange(int(epochs), desc="Epoch")
            except ImportError:
                pass
        for _ in epoch_range:
            if interactions.nnz == 0:
                continue
            order[:] = np.arange(interactions.nnz, dtype=np.int32)
            self.random_state.shuffle(order)
            state = int(self.random_state.randint(1, np.iinfo(np.int32).max))
            function(
                loss_kind,
                schedule,
                addr(item_features.indptr),
                addr(item_features.indices),
                addr(item_features.data),
                addr(user_features.indptr),
                addr(user_features.indices),
                addr(user_features.data),
                addr(positives.indptr),
                addr(positives.indices),
                addr(interactions.row),
                addr(interactions.col),
                addr(interactions.data),
                addr(weights),
                addr(order),
                interactions.nnz,
                addr(self.item_embeddings),
                addr(self.item_embedding_gradients),
                addr(self.item_embedding_momentum),
                addr(self.item_biases),
                addr(self.item_bias_gradients),
                addr(self.item_bias_momentum),
                addr(self.user_embeddings),
                addr(self.user_embedding_gradients),
                addr(self.user_embedding_momentum),
                addr(self.user_biases),
                addr(self.user_bias_gradients),
                addr(self.user_bias_momentum),
                n_items,
                self.no_components,
                self.learning_rate,
                self.item_alpha,
                self.user_alpha,
                self.rho,
                self.epsilon,
                self.max_sampled,
                state,
                addr(user_work),
                addr(positive_work),
                addr(negative_work),
            )
            if not (
                np.isfinite(self.item_embeddings).all()
                and np.isfinite(self.user_embeddings).all()
                and np.isfinite(self.item_biases).all()
                and np.isfinite(self.user_biases).all()
            ):
                raise ValueError(
                    "Not all estimated parameters are finite; try a lower learning rate"
                )
        return self

    def predict(
        self, user_ids, item_ids, item_features=None, user_features=None, num_threads=1
    ):
        self._check_model_buffers()
        if num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")
        item_ids = _ids(item_ids, "item_ids")
        if isinstance(user_ids, (int, np.integer)):
            user_ids = np.repeat(user_ids, len(item_ids))
        user_ids = _ids(user_ids, "user_ids")
        if len(user_ids) != len(item_ids):
            raise ValueError(
                f"Expected the number of user IDs ({len(user_ids)}) to equal the "
                f"number of item IDs ({len(item_ids)})"
            )
        if not len(user_ids):
            return np.empty(0, dtype=np.float32)
        if user_ids.min() < 0 or item_ids.min() < 0:
            raise ValueError("User or item ids cannot be negative.")
        max_user = int(user_ids.max())
        max_item = int(item_ids.max())
        if user_features is None:
            if max_user >= len(self.user_biases):
                raise ValueError("User feature matrix specifies more features")
            user_biases = self.user_biases
            user_embeddings = self.user_embeddings
        else:
            user_features = _csr(user_features)
            if max_user >= user_features.shape[0]:
                raise Exception(
                    "Number of user feature rows does not equal the number of users"
                )
            user_biases, user_embeddings = self._representations(
                user_features, self.user_embeddings, self.user_biases
            )
        if item_features is None:
            if max_item >= len(self.item_biases):
                raise ValueError("Item feature matrix specifies more features")
            item_biases = self.item_biases
            item_embeddings = self.item_embeddings
        else:
            item_features = _csr(item_features)
            if max_item >= item_features.shape[0]:
                raise Exception(
                    "Number of item feature rows does not equal the number of items"
                )
            item_biases, item_embeddings = self._representations(
                item_features, self.item_embeddings, self.item_biases
            )
        result = np.empty(len(user_ids), dtype=np.float32)
        lib().mlfm_predict_dense(
            addr(user_embeddings),
            addr(user_biases),
            addr(item_embeddings),
            addr(item_biases),
            addr(user_ids),
            addr(item_ids),
            len(user_ids),
            self.no_components,
            addr(result),
            int(num_threads),
        )
        return result

    def _representations(self, features, embeddings, biases):
        self._check_model_buffers()
        features = _csr(features)
        if features.shape[1] != embeddings.shape[0]:
            raise ValueError("Incorrect number of feature columns")
        result_embeddings = np.empty(
            (features.shape[0], self.no_components), dtype=np.float32
        )
        result_biases = np.empty(features.shape[0], dtype=np.float32)
        if features.shape[0] == 0:
            return result_biases, result_embeddings
        lib().mlfm_representations(
            addr(features.indptr),
            addr(features.indices),
            addr(features.data),
            addr(embeddings),
            addr(biases),
            features.shape[0],
            self.no_components,
            addr(result_embeddings),
            addr(result_biases),
        )
        return result_biases, result_embeddings

    def get_item_representations(self, features=None):
        self._check_model_buffers()
        if features is None:
            return self.item_biases, self.item_embeddings
        return self._representations(features, self.item_embeddings, self.item_biases)

    def get_user_representations(self, features=None):
        self._check_model_buffers()
        if features is None:
            return self.user_biases, self.user_embeddings
        return self._representations(features, self.user_embeddings, self.user_biases)

    def _score_all(self, user_ids, user_features, item_features):
        user_biases, user_embeddings = self.get_user_representations(user_features)
        item_biases, item_embeddings = self.get_item_representations(item_features)
        users = _ids(user_ids, "user_ids")
        result = np.empty((len(users), len(item_biases)), dtype=np.float32)
        if len(users) == 0 or len(item_biases) == 0:
            return result
        lib().mlfm_score_matrix(
            addr(user_embeddings),
            addr(user_biases),
            addr(item_embeddings),
            addr(item_biases),
            addr(users),
            len(users),
            len(item_biases),
            self.no_components,
            addr(result),
        )
        return result

    def predict_rank(
        self,
        test_interactions,
        train_interactions=None,
        item_features=None,
        user_features=None,
        num_threads=1,
        check_intersections=True,
    ):
        self._check_model_buffers()
        if num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")
        test = _csr(test_interactions)
        n_users, n_items = test.shape
        if train_interactions is None:
            train = _csr(sp.csr_matrix(test.shape, dtype=np.float32))
        else:
            train = _csr(train_interactions)
            if train.shape != test.shape:
                raise ValueError("Train and test interactions must have the same shape")
            if check_intersections:
                intersections = test.multiply(train).nnz
                if intersections:
                    raise ValueError(
                        "Test interactions matrix and train interactions matrix share "
                        f"{intersections} interactions."
                    )
        user_features, item_features = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )
        ranks_data = np.zeros(test.nnz, dtype=np.float32)
        active_users = np.flatnonzero(np.diff(test.indptr)).astype(np.int32)
        for batch_start in range(0, len(active_users), 256):
            batch_users = active_users[batch_start : batch_start + 256]
            scores = self._score_all(batch_users, user_features, item_features)
            for local, user in enumerate(batch_users):
                start, stop = test.indptr[user : user + 2]
                train_start, train_stop = train.indptr[user : user + 2]
                excluded = train.indices[train_start:train_stop]
                row_scores = scores[local]
                for position in range(start, stop):
                    item = test.indices[position]
                    candidate = row_scores >= row_scores[item]
                    candidate[item] = False
                    candidate[excluded] = False
                    ranks_data[position] = np.count_nonzero(candidate)
        return sp.csr_matrix(
            (ranks_data, test.indices.copy(), test.indptr.copy()), shape=test.shape
        )

    def get_params(self, deep=True):
        return {
            "loss": self.loss,
            "learning_schedule": self.learning_schedule,
            "no_components": self.no_components,
            "learning_rate": self.learning_rate,
            "k": self.k,
            "n": self.n,
            "rho": self.rho,
            "epsilon": self.epsilon,
            "max_sampled": self.max_sampled,
            "item_alpha": self.item_alpha,
            "user_alpha": self.user_alpha,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key} for estimator LightFM.")
            setattr(self, key, value)
        return self
