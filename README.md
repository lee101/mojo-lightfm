# mojo-lightfm

`mojo-lightfm` is a standalone Mojo port of the compute-heavy core of
[LightFM](https://github.com/lyst/lightfm), the hybrid matrix-factorization
recommender. It learns feature embeddings from sparse user-item interactions
and sparse user/item metadata, then scores and ranks recommendations through a
compiled Mojo shared library.

The Python entry point is `mojo_lightfm.LightFM`. Its covered methods and
signatures mirror LightFM 1.17, so code for the supported subset generally only
needs an import change:

```python
from mojo_lightfm import LightFM
```

## Coverage

Covered:

- `LightFM` construction and learned attributes
- `fit` and `fit_partial`
- logistic, BPR, and WARP losses
- AdaGrad and AdaDelta learning schedules
- CSR user and item features, including weighted hybrid features
- COO interaction weights through `sample_weight`
- `predict`, `predict_rank`, `get_user_representations`, and
  `get_item_representations`
- `get_params` and `set_params`
- `precision_at_k`, `recall_at_k`, `auc_score`, and `reciprocal_rank` in
  `mojo_lightfm.evaluation`

Not covered:

- the `warp-kos` loss
- parallel/Hogwild fitting; `num_threads` is validated for compatibility but
  fitting remains serial
- `lightfm.data.Dataset`, dataset download helpers, and cross-validation helpers
- GPU execution

Pair prediction honors `num_threads`. It stays serial below 32,768 pairs to
avoid thread-launch overhead and partitions larger independent scoring jobs
across the requested workers.

No GPU path is included. Prediction performs about 0.25 FLOP per byte of
embedding data read, well below the roughly 2 FLOP/byte level that can justify
device transfer and launch costs. Sparse stochastic training also has ordered
model-update dependencies, so this port keeps buffers on the CPU.

BPR and WARP use stochastic negative sampling. Given the same seed, their
learned arrays are not byte-identical to LightFM's thread-local C RNG, but the
tests require ranking-quality parity on the same data. Logistic inference,
hybrid representation composition, ranks, and evaluation metrics are checked
numerically against the real LightFM 1.17 package.

## Install and run

The environment deliberately uses Python 3.11 because that is the newest
CPython build for which conda-forge supplies LightFM 1.17.

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` compiles `src/lightfm.mojo` into
`dist/libmojo-lightfm.so`.

## Example

The complete example is in `examples/basic.py`. Run it inside the project with:

```bash
pixi run python examples/basic.py
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic. Times are the best of three warmed runs. Both
implementations use one thread except for the explicitly labeled four-thread
row. “upstream / Mojo” above 1 means Mojo is faster.

| workload | mojo-lightfm | lightfm 1.17 | upstream / Mojo |
|---|---:|---:|---:|
| predict, identity (1M pairs, 32 factors) | 26.76 ms | 136.51 ms | 5.10x |
| predict, identity (1M pairs, 32 factors, 4 threads) | 21.87 ms | 42.01 ms | 1.92x |
| predict, hybrid (300k pairs, 32 factors) | 9.10 ms | 48.06 ms | 5.28x |
| logistic fit (40k interactions, 32 factors) | 20.36 ms | 56.31 ms | 2.77x |
| BPR fit (20k interactions, 32 factors) | 15.20 ms | 39.09 ms | 2.57x |
| WARP fit (20k interactions, 32 factors) | 15.28 ms | 38.08 ms | 2.49x |
| predict_rank (500 users x 2k items) | 32.11 ms | 84.88 ms | 2.64x |

Re-run the benchmark rather than treating these numbers as portable:

```bash
pixi run bench
```

That task holds a machine-wide lock to avoid concurrent benchmark jobs.

## How it works

SciPy owns all sparse matrices and NumPy owns every model and scratch buffer.
The Python layer normalizes sparse structures to CSR/COO with `int32` indices
and contiguous `float32` values, matching LightFM's storage. Buffers cross the
C ABI as integer addresses through `ctypes`; Mojo reconstructs typed pointers
inside non-parametric `@export` functions. No allocation or ownership crosses
the FFI boundary. Calls are synchronous, and Python locals retain every NumPy
and SciPy owner until the native call returns. Before each call, the bridge
checks dtypes, contiguity, shapes, index ranges, and non-null addresses.

The shared library is one Mojo compilation unit. Prediction scores identity
features directly from the model's NumPy arrays. For hybrid inputs it composes
each sparse row once, then reuses those dense representations across all
requested pairs. Representation accumulation and dot products use native-width
Float32 SIMD loads and stores with scalar remainder loops. Pair scoring
partitions batches of at least 32,768 pairs into independent chunks scheduled
across the requested CPU workers.

Training performs sparse feature updates in place using AdaGrad or AdaDelta.
Pairwise BPR and WARP updates share the same representation and gradient path,
while WARP repeatedly samples until it finds a margin-violating negative. Dense
all-item scoring is also used by rank prediction.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The parity suite installs and exercises LightFM 1.17 directly; it does not use
mock benchmark results or a simplified Python oracle.
