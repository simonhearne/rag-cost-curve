"""WS3 dimensionality-reduction transforms: MRL truncation and PCA.

Two ways to get from 1024-d mxbai vectors to a shorter vector:

* **MRL truncation** — keep the first `d` coordinates, renormalize. Valid
  only because mxbai-embed-large-v1 is Matryoshka-trained; whether that
  claim survives contact with retrieval recall is what WS3 measures.
* **PCA** — project onto the top `d` principal directions of the corpus,
  renormalize. The "already committed to a non-MRL model" story.

Both are an orthonormal projection followed by an L2 renormalization, which
is what makes them comparable: the only difference is *which* d-dimensional
subspace is kept. PCA additionally re-centers on the corpus mean (the
standard definition, and what faiss/sklearn do). Corpus vectors and query
vectors always go through the identical function — a transform applied to
only one side would silently destroy the metric.

The PCA rotation is fitted on a corpus-only sample (never queries) and
cached to disk with a pinned sign convention so its sha256 is stable.

Pure functions, unit-tested in tests/test_dimensionality.py.
"""

import io
import zipfile
from pathlib import Path

import numpy as np

# Streaming block size for corpus transforms: 50k x 1024 fp32 = 200 MB.
TRANSFORM_BATCH = 50_000
# float64 accumulation block for the PCA scatter matrix.
_FIT_BLOCK = 50_000


def _as_2d(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-d array of row vectors, got ndim={a.ndim}")
    return a


def _renormalize(y: np.ndarray) -> np.ndarray:
    """L2-normalize rows in float32 (IP == cosine everywhere downstream)."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    n = np.linalg.norm(y, axis=1, keepdims=True)
    return y / n


def mrl_truncate(x, d: int) -> np.ndarray:
    """Matryoshka truncation: first `d` coordinates, renormalized."""
    a = _as_2d(x)
    if d > a.shape[1]:
        raise ValueError(f"target dim {d} exceeds input dim {a.shape[1]}")
    return _renormalize(a[:, :d])


def sample_rows(n_total: int, n_sample: int, seed: int) -> np.ndarray:
    """Deterministic sorted row indices for a without-replacement sample.

    Sorted because the caller reads them out of a memmap — random access to
    ascending offsets is what keeps the read sequential-ish.
    """
    if n_sample > n_total:
        raise ValueError(f"cannot sample {n_sample} rows from {n_total}")
    idx = np.random.default_rng(seed).choice(n_total, n_sample, replace=False)
    return np.sort(idx)


def _pin_signs(components: np.ndarray) -> np.ndarray:
    """Make each component's largest-magnitude entry positive.

    LAPACK's eigenvector signs are arbitrary; pinning them makes the cached
    rotation matrix byte-reproducible, which is what lets its sha256 go into
    data/MANIFEST.json as a meaningful check.
    """
    lead = components[np.arange(components.shape[0]),
                      np.argmax(np.abs(components), axis=1)]
    return components * np.where(lead < 0, -1.0, 1.0)[:, None]


def fit_pca(sample, center: bool = True) -> dict:
    """Fit PCA on a corpus sample. Never pass queries to this function.

    Returns {mean, components (d_full, d_full) rows = components ordered by
    descending explained variance, explained_variance, n_sample, dim,
    centered}. The scatter matrix is accumulated in float64 over blocks so a
    500k x 1024 fp32 sample never needs a float64 copy of itself in RAM.

    `center=False` fits the *uncentered* second moment — i.e. a truncated SVD
    of the data matrix. Prefer it for cosine/IP retrieval: with no mean
    subtraction the projection is an orthogonal map, so keeping every
    component preserves inner products exactly and the ranking is untouched.
    Centering followed by the L2 renormalization every arm applies is NOT
    orthogonal on the sphere, and it reorders cosine neighbours: measured
    against full-dim ground truth on the WS3 corpus it cost 0.17 recall@10 at
    896d, where only 128 of 1024 directions were discarded, and produced a
    flat ~0.83 ceiling from 896d all the way down to 512d. Text-embedding
    vectors sit in a narrow cone with a large shared component, so removing
    that component is a big geometric change, not the harmless preprocessing
    step the sklearn/faiss default suggests.
    """
    a = _as_2d(sample)
    n, dim = a.shape
    if n < 2:
        raise ValueError("PCA needs at least 2 sample rows")

    if center:
        total = np.zeros(dim, dtype=np.float64)
        for s in range(0, n, _FIT_BLOCK):
            total += np.asarray(a[s:s + _FIT_BLOCK],
                                dtype=np.float64).sum(axis=0)
        mean = total / n
    else:
        mean = np.zeros(dim, dtype=np.float64)

    scatter = np.zeros((dim, dim), dtype=np.float64)
    for s in range(0, n, _FIT_BLOCK):
        block = np.asarray(a[s:s + _FIT_BLOCK], dtype=np.float64) - mean
        scatter += block.T @ block
    cov = scatter / (n - 1)

    eigvals, eigvecs = np.linalg.eigh(cov)          # ascending, orthonormal
    order = np.argsort(eigvals)[::-1]
    components = _pin_signs(np.ascontiguousarray(eigvecs[:, order].T))
    return {
        "mean": mean,
        "components": components,
        "explained_variance": np.ascontiguousarray(eigvals[order]),
        "n_sample": int(n),
        "dim": int(dim),
        "centered": bool(center),
    }


def pca_project(x, model: dict, d: int) -> np.ndarray:
    """Center on the fitted mean, project onto the top `d` components,
    renormalize. Identical call for corpus rows and query rows."""
    a = _as_2d(x)
    if a.shape[1] != model["dim"]:
        raise ValueError(
            f"input dim {a.shape[1]} does not match model dim {model['dim']}")
    if d > model["dim"]:
        raise ValueError(f"target dim {d} exceeds model dim {model['dim']}")
    centered = np.asarray(a, dtype=np.float64) - model["mean"]
    return _renormalize(centered @ model["components"][:d].T)


# ------------------------------------------------------------ persistence ---
# Written as a ZIP_STORED npz with a fixed member timestamp: np.savez stamps
# the current time into the zip directory, which would change the file's
# sha256 on every save and make the manifest entry meaningless.

_PCA_MEMBERS = ("components", "mean", "explained_variance", "n_sample", "dim",
                "centered")


def save_pca(model: dict, path) -> Path:
    """Write the fitted rotation to a byte-reproducible .npz."""
    path = Path(path)
    arrays = {
        "components": np.asarray(model["components"], dtype=np.float64),
        "mean": np.asarray(model["mean"], dtype=np.float64),
        "explained_variance": np.asarray(model["explained_variance"],
                                         dtype=np.float64),
        "n_sample": np.asarray(model["n_sample"], dtype=np.int64),
        "dim": np.asarray(model["dim"], dtype=np.int64),
        "centered": np.asarray(int(model.get("centered", True)),
                               dtype=np.int64),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name in _PCA_MEMBERS:
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            z.writestr(info, buf.getvalue())
    return path


def load_pca(path) -> dict:
    with np.load(Path(path)) as z:
        return {
            "components": z["components"],
            "mean": z["mean"],
            "explained_variance": z["explained_variance"],
            "n_sample": int(z["n_sample"]),
            "dim": int(z["dim"]),
            "centered": bool(z["centered"]),
        }


def transform_corpus(src, out_path, fn, batch: int = TRANSFORM_BATCH) -> Path:
    """Stream `fn` over `src` row-blocks into a new .npy memmap.

    `fn` takes a (rows, d_in) block and returns a (rows, d_out) block; the
    output dim is probed from the first row, so the same helper serves MRL
    and PCA. Blocking never changes the result — both transforms are row-wise.
    """
    out_path = Path(out_path)
    n = src.shape[0]
    probe = fn(np.asarray(src[:1]))
    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(n, probe.shape[1]))
    for s in range(0, n, batch):
        out[s:s + batch] = fn(np.asarray(src[s:s + batch]))
    out.flush()
    del out
    return out_path
