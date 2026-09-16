"""Pins the WS2 footprint decomposition against the committed measurements.

The decomposition is what lets `results/index-footprint.md` say where the
RaBitQ arm's bytes go (codes 42 %, IVF centroids 24–27 %, RaBitQ estimator
factors ~10 %, segment heap ~19 %). Two things have to stay true for that
attribution to mean anything, and both are checked here against the
committed `data/ws2_cache/` JSON rather than synthetic data:

  1. the model's error bar on non-RaBitQ IVF arms is ~0 B/vector, so the
     ~29 B/vector residual on the RaBitQ arms is a real component and not
     modelling slop; and
  2. at 1M — where segment sizes are unrecorded and the centroid term is
     taken as the residual — the implied centroid count per segment stays
     at or below the configured nlist=4096. If it ever exceeds it, the
     decomposition is over-attributing to centroids and the doc is wrong.

Run: PYTHONPATH=. .venv/bin/python -m unittest tests.test_footprint_decomposition -v
"""

import importlib.util
import unittest
from pathlib import Path

from benchlib.config import REPO_ROOT, SLICES

SCRIPT = (REPO_ROOT / "scripts" / "index-footprint"
          / "ws2_footprint_decomposition.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("ws2_decomp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFootprintDecomposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_module()
        cls.arms = {s: cls.m.load_arms(s) for s in ("1m", "10m")}
        cls.factor, cls.diag = cls.m.identify_rabitq_factor_bytes(
            cls.arms["10m"], SLICES["10m"])

    def test_cache_present(self):
        self.assertTrue(Path(SCRIPT).exists())
        self.assertIn("rabitq", self.arms["1m"])
        self.assertIn("rabitq", self.arms["10m"])

    def test_model_error_bar_is_small_on_non_rabitq_arms(self):
        # If this grows, the RaBitQ factor identification below is not safe.
        self.assertLess(self.diag["non_rabitq_worst_b_per_vector"], 2.0)

    def test_rabitq_factor_cost_is_the_odd_one_out(self):
        resid = self.diag["residual_b_per_vector"]
        rabitq = [v for a, v in resid.items() if "rabitq" in a]
        other = [v for a, v in resid.items() if "rabitq" not in a]
        self.assertGreater(min(rabitq), 10 * max(abs(v) for v in other))
        # ~29 B/vector on top of the 128 B of codes: the "1-bit" payload is
        # really ~157 B/vector (26x, not 32x) before any IVF overhead.
        self.assertTrue(25.0 <= self.factor <= 35.0, self.factor)

    def test_components_sum_to_measured_loaded_bytes(self):
        for scale in ("1m", "10m"):
            rows = self.m.decompose(scale, self.arms[scale], self.factor)
            by_arm = {}
            for r in rows:
                by_arm.setdefault(r["arm"], []).append(r)
            for arm, rs in by_arm.items():
                with self.subTest(scale=scale, arm=arm):
                    self.assertLessEqual(
                        abs(sum(r["bytes"] for r in rs)
                            - rs[0]["loaded_bytes"]), 2)

    def test_1m_inferred_centroids_stay_under_configured_nlist(self):
        rows = self.m.decompose("1m", self.arms["1m"], self.factor)
        checked = 0
        for r in rows:
            if r["component"] != "ivf_coarse_centroids":
                continue
            implied = (r["bytes"] / r["n_segments"]
                       / (self.m.D * self.m.FP32))
            with self.subTest(arm=r["arm"]):
                self.assertGreater(implied, 0)
                self.assertLessEqual(round(implied), self.m.NLIST)
            checked += 1
        self.assertGreater(checked, 0)

    def test_centroid_term_dominates_the_rabitq_gap(self):
        # The claim in results/index-footprint.md: for the no-refine RaBitQ
        # arm, the un-quantized fp32 centroid table is the largest single
        # component after the codes themselves, at both scales.
        for scale in ("1m", "10m"):
            rows = [r for r in self.m.decompose(
                scale, self.arms[scale], self.factor) if r["arm"] == "rabitq"]
            ranked = sorted(rows, key=lambda r: -r["bytes"])
            with self.subTest(scale=scale):
                self.assertEqual(ranked[0]["component"], "quantized_codes")
                self.assertIn(ranked[1]["component"],
                              ("ivf_coarse_centroids", "segment_heap"))
                cent = next(r for r in rows
                            if r["component"] == "ivf_coarse_centroids")
                self.assertGreater(cent["share_of_loaded"], 0.20)


if __name__ == "__main__":
    unittest.main()
