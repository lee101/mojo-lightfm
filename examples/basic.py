"""Minimal training and ranking example."""

import numpy as np
from scipy import sparse

from mojo_lightfm import LightFM
from mojo_lightfm.evaluation import precision_at_k


interactions = sparse.coo_matrix(
    (
        np.ones(6, dtype=np.float32),
        ([0, 0, 0, 1, 1, 1], [0, 1, 2, 2, 3, 4]),
    ),
    shape=(2, 5),
)
model = LightFM(no_components=8, loss="warp", random_state=0).fit(
    interactions, epochs=20
)
scores = model.predict(0, np.arange(5, dtype=np.int32))
print(np.argsort(-scores))
print(precision_at_k(model, interactions.tocsr(), k=3))
