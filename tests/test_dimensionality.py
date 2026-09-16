"""Unit tests for benchlib.dimensionality dimensionality-reduction transforms.

Run: PYTHONPATH=. .venv/bin/python -m unittest tests.test_dimensionality -v
Stdlib unittest on purpose — no new pinned dependencies.

These transforms decide the WS3 verdict, so they are tested before they are
written: MRL truncation and PCA must be exact, deterministic (the rotation
matrix is checksummed into data/MANIFEST.json), and must treat corpus and
query vectors through the identical code path.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from benchlib import dimensionality
from benchlib.dimensionality import (
    fit_pca,
    load_pca,
    mrl_truncate,
    pca_project,
    sample_rows,
    save_pca,
    transform_corpus,
)


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestMrlTruncate(unittest.TestCase):
    def test_keeps_first_d_columns_then_renormalizes(self):
        x = np.array([[3.0, 4.0, 100.0, 100.0]], dtype=np.float32)
        out = mrl_truncate(x, 2)
        np.testing.assert_allclose(out, [[0.6, 0.8]], rtol=1e-6)

    def test_rows_are_unit_norm(self):
        rng = np.random.default_rng(0)
        x = _unit(rng.normal(size=(50, 32)))
        out = mrl_truncate(x, 8)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)

    def test_returns_contiguous_float32(self):
        x = np.asfortranarray(np.ones((4, 6), dtype=np.float64))
        out = mrl_truncate(x, 3)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(out.flags["C_CONTIGUOUS"])

    def test_full_dim_is_identity_for_normalized_input(self):
        rng = np.random.default_rng(1)
        x = _unit(rng.normal(size=(20, 16))).astype(np.float32)
        np.testing.assert_allclose(mrl_truncate(x, 16), x, atol=1e-6)

    def test_batch_invariant(self):
        # corpus is transformed in blocks, queries in one shot: identical math
        rng = np.random.default_rng(2)
        x = _unit(rng.normal(size=(10, 12))).astype(np.float32)
        whole = mrl_truncate(x, 5)
        blocks = np.vstack([mrl_truncate(x[:4], 5), mrl_truncate(x[4:], 5)])
        np.testing.assert_array_equal(whole, blocks)

    def test_target_dim_larger_than_input_raises(self):
        with self.assertRaises(ValueError):
            mrl_truncate(np.ones((2, 4), dtype=np.float32), 8)

    def test_one_dimensional_input_raises(self):
        with self.assertRaises(ValueError):
            mrl_truncate(np.ones(4, dtype=np.float32), 2)


class TestSampleRows(unittest.TestCase):
    def test_deterministic_for_same_seed(self):
        a = sample_rows(1000, 50, seed=42)
        b = sample_rows(1000, 50, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_sample(self):
        a = sample_rows(1000, 50, seed=42)
        b = sample_rows(1000, 50, seed=43)
        self.assertFalse(np.array_equal(a, b))

    def test_sorted_unique_and_in_range(self):
        idx = sample_rows(1000, 200, seed=42)
        self.assertEqual(len(idx), 200)
        self.assertEqual(len(np.unique(idx)), 200)
        np.testing.assert_array_equal(idx, np.sort(idx))
        self.assertGreaterEqual(idx.min(), 0)
        self.assertLess(idx.max(), 1000)

    def test_sample_larger_than_population_raises(self):
        with self.assertRaises(ValueError):
            sample_rows(10, 20, seed=42)


class TestFitPca(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = _unit(rng.normal(size=(400, 16))).astype(np.float32)

    def test_components_are_orthonormal(self):
        m = fit_pca(self.x)
        c = m["components"]
        np.testing.assert_allclose(c @ c.T, np.eye(16), atol=1e-5)

    def test_shapes_and_sample_size_recorded(self):
        m = fit_pca(self.x)
        self.assertEqual(m["components"].shape, (16, 16))
        self.assertEqual(m["mean"].shape, (16,))
        self.assertEqual(m["explained_variance"].shape, (16,))
        self.assertEqual(m["n_sample"], 400)
        self.assertEqual(m["dim"], 16)

    def test_explained_variance_is_descending(self):
        v = fit_pca(self.x)["explained_variance"]
        self.assertTrue(np.all(np.diff(v) <= 1e-9))

    def test_mean_is_the_sample_mean(self):
        m = fit_pca(self.x)
        np.testing.assert_allclose(m["mean"], self.x.mean(axis=0), atol=1e-6)

    def test_deterministic_across_calls(self):
        a, b = fit_pca(self.x), fit_pca(self.x)
        np.testing.assert_array_equal(a["components"], b["components"])
        np.testing.assert_array_equal(a["mean"], b["mean"])

    def test_component_sign_convention_is_pinned(self):
        # eigenvector signs are arbitrary in LAPACK; pin them so the cached
        # rotation matrix has a stable sha256 across machines.
        c = fit_pca(self.x)["components"]
        lead = c[np.arange(c.shape[0]), np.argmax(np.abs(c), axis=1)]
        self.assertTrue(np.all(lead > 0))

    def test_block_size_does_not_change_the_fit(self):
        # the real fit streams a 500k x 1024 sample in float64 blocks; a bug
        # in that accumulation would be invisible at test-array sizes
        big = _unit(np.random.default_rng(14).normal(size=(500, 16)))
        big = big.astype(np.float32)
        one_shot = fit_pca(big)
        with mock.patch.object(dimensionality, "_FIT_BLOCK", 37):
            chunked = fit_pca(big)
        np.testing.assert_allclose(chunked["mean"], one_shot["mean"], atol=1e-12)
        np.testing.assert_allclose(chunked["components"],
                                   one_shot["components"], atol=1e-9)
        np.testing.assert_allclose(chunked["explained_variance"],
                                   one_shot["explained_variance"], atol=1e-12)

    def test_rank_deficient_data_puts_all_variance_in_first_k(self):
        # data living in a 3-d subspace of 16-d: components 4.. carry nothing
        rng = np.random.default_rng(8)
        basis = np.linalg.qr(rng.normal(size=(16, 3)))[0]
        coords = rng.normal(size=(300, 3))
        x = (coords @ basis.T).astype(np.float32)
        v = fit_pca(x)["explained_variance"]
        self.assertGreater(v[2], 1e-3)
        self.assertLess(v[3], 1e-6)


class TestUncenteredPca(unittest.TestCase):
    """Uncentered PCA (truncated SVD) keeps the projection an orthogonal map
    on the sphere. Centering then renormalizing is NOT orthogonal, and
    measured against full-dim neighbour identity it cost 0.17 recall at 896d
    — where only 128 of 1024 directions were discarded."""

    def setUp(self):
        rng = np.random.default_rng(20)
        # embeddings live in a cone: a large shared component + variation,
        # which is exactly the regime where centering changes the ranking
        base = np.abs(rng.normal(size=(1, 16)))
        self.x = _unit(base + 0.35 * rng.normal(size=(300, 16)))
        self.x = self.x.astype(np.float32)

    def test_mean_is_zero_when_centering_disabled(self):
        m = fit_pca(self.x, center=False)
        np.testing.assert_array_equal(m["mean"], np.zeros(16))
        self.assertFalse(m["centered"])

    def test_centered_flag_defaults_to_true_for_backwards_compatibility(self):
        self.assertTrue(fit_pca(self.x)["centered"])

    def test_components_still_orthonormal(self):
        c = fit_pca(self.x, center=False)["components"]
        np.testing.assert_allclose(c @ c.T, np.eye(16), atol=1e-5)

    def test_full_rank_projection_is_an_exact_rotation(self):
        # THE correctness property centering breaks: keeping every component
        # must preserve the original inner products exactly, so recall against
        # full-dim ground truth is 1.0 by construction.
        m = fit_pca(self.x, center=False)
        got = pca_project(self.x, m, 16)
        np.testing.assert_allclose(got @ got.T, self.x @ self.x.T, atol=1e-4)

    def test_centered_full_rank_projection_does_NOT_preserve_inner_products(self):
        # the documented trap, pinned so nobody "simplifies" it back
        m = fit_pca(self.x, center=True)
        got = pca_project(self.x, m, 16)
        self.assertGreater(np.abs(got @ got.T - self.x @ self.x.T).max(), 0.01)

    def test_round_trip_preserves_the_centered_flag(self):
        m = fit_pca(self.x, center=False)
        with tempfile.TemporaryDirectory() as tmp:
            back = load_pca(save_pca(m, Path(tmp) / "p.npz"))
        self.assertFalse(back["centered"])
        np.testing.assert_array_equal(back["mean"], m["mean"])


class TestPcaProject(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.x = _unit(rng.normal(size=(200, 16))).astype(np.float32)
        self.model = fit_pca(self.x)

    def test_shape_dtype_and_unit_norm(self):
        out = pca_project(self.x, self.model, 4)
        self.assertEqual(out.shape, (200, 4))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_batch_invariant(self):
        # the corpus streams in blocks, the query set goes in one call
        whole = pca_project(self.x, self.model, 6)
        blocks = np.vstack([pca_project(self.x[:37], self.model, 6),
                            pca_project(self.x[37:], self.model, 6)])
        np.testing.assert_array_equal(whole, blocks)

    def test_full_rank_projection_preserves_centered_ranking(self):
        # keeping every component is a rotation of the centered data, so the
        # cosine ranking must be identical to the centered full-dim ranking
        c = self.x - self.model["mean"]
        ref = _unit(c) @ _unit(c).T
        got = pca_project(self.x, self.model, 16)
        np.testing.assert_allclose(got @ got.T, ref, atol=1e-4)

    def test_rank_deficient_data_projects_losslessly(self):
        rng = np.random.default_rng(10)
        basis = np.linalg.qr(rng.normal(size=(16, 3)))[0]
        x = (rng.normal(size=(120, 3)) @ basis.T).astype(np.float32)
        model = fit_pca(x)
        c = x - model["mean"]
        ref = _unit(c) @ _unit(c).T
        got = pca_project(x, model, 3)
        np.testing.assert_allclose(got @ got.T, ref, atol=1e-4)

    def test_target_dim_larger_than_model_raises(self):
        with self.assertRaises(ValueError):
            pca_project(self.x, self.model, 32)

    def test_dim_mismatch_with_model_raises(self):
        with self.assertRaises(ValueError):
            pca_project(np.ones((3, 8), dtype=np.float32), self.model, 4)


class TestSaveLoadPca(unittest.TestCase):
    def test_round_trip_is_exact(self):
        rng = np.random.default_rng(11)
        model = fit_pca(_unit(rng.normal(size=(100, 8))).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            p = save_pca(model, Path(tmp) / "pca.npz")
            back = load_pca(p)
        np.testing.assert_array_equal(back["components"], model["components"])
        np.testing.assert_array_equal(back["mean"], model["mean"])
        self.assertEqual(back["n_sample"], model["n_sample"])
        self.assertEqual(back["dim"], model["dim"])

    def test_saved_bytes_are_reproducible(self):
        # the rotation matrix is checksummed into MANIFEST.json — two saves of
        # the same model must produce byte-identical files
        rng = np.random.default_rng(12)
        model = fit_pca(_unit(rng.normal(size=(100, 8))).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            a = save_pca(model, Path(tmp) / "a.npz").read_bytes()
            b = save_pca(model, Path(tmp) / "b.npz").read_bytes()
        self.assertEqual(a, b)


class TestTransformCorpus(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.src = _unit(rng.normal(size=(97, 16))).astype(np.float32)

    def test_streamed_output_matches_whole_array_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = transform_corpus(self.src, Path(tmp) / "o.npy",
                                   lambda b: mrl_truncate(b, 4), batch=10)
            got = np.load(out, mmap_mode="r")
            np.testing.assert_array_equal(np.asarray(got),
                                          mrl_truncate(self.src, 4))
            self.assertEqual(got.shape, (97, 4))
            self.assertEqual(got.dtype, np.float32)

    def test_batch_size_does_not_change_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = np.load(transform_corpus(self.src, Path(tmp) / "a.npy",
                                         lambda b: mrl_truncate(b, 5), batch=7))
            b = np.load(transform_corpus(self.src, Path(tmp) / "b.npy",
                                         lambda b: mrl_truncate(b, 5), batch=97))
        np.testing.assert_array_equal(a, b)

    def test_works_on_a_memmap_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_p = Path(tmp) / "src.npy"
            np.save(src_p, self.src)
            src = np.load(src_p, mmap_mode="r")
            out = transform_corpus(src, Path(tmp) / "o.npy",
                                   lambda b: mrl_truncate(b, 3), batch=16)
            np.testing.assert_array_equal(np.load(out),
                                          mrl_truncate(self.src, 3))


if __name__ == "__main__":
    unittest.main()
