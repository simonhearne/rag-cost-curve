"""Unit tests for benchlib.quantization pure measurement functions.

Run: PYTHONPATH=. .venv/bin/python -m unittest tests.test_quantization -v
Stdlib unittest on purpose — no new pinned dependencies.
"""

import subprocess
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from benchlib.quantization import (
    RSS_UNAVAILABLE,
    docker_exec,
    latency_summary,
    milvus_rss_bytes,
    payload_compression,
    pick_best_sweep_points,
    recall_at_k,
)


class TestRecallAtK(unittest.TestCase):
    def test_perfect_match_is_one(self):
        gt = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
        pred = gt.copy()
        self.assertEqual(recall_at_k(pred, gt, 4), 1.0)

    def test_order_within_top_k_does_not_matter(self):
        gt = np.array([[1, 2, 3, 4]])
        pred = np.array([[4, 3, 2, 1]])
        self.assertEqual(recall_at_k(pred, gt, 4), 1.0)

    def test_half_overlap_is_half(self):
        gt = np.array([[1, 2, 3, 4]])
        pred = np.array([[1, 2, 99, 98]])
        self.assertEqual(recall_at_k(pred, gt, 4), 0.5)

    def test_uses_only_first_k_columns(self):
        # pred/gt wider than k: only the first k of each count.
        gt = np.array([[1, 2, 3, 4, 5]])
        pred = np.array([[1, 2, 9, 8, 3]])  # 3 is in gt top-5 but not pred top-2… k=2
        self.assertEqual(recall_at_k(pred, gt, 2), 1.0)
        self.assertEqual(recall_at_k(pred, gt, 4), 0.5)

    def test_averages_over_queries(self):
        gt = np.array([[1, 2], [3, 4]])
        pred = np.array([[1, 2], [99, 98]])
        self.assertEqual(recall_at_k(pred, gt, 2), 0.5)

    def test_mismatched_query_count_raises(self):
        with self.assertRaises(ValueError):
            recall_at_k(np.array([[1]]), np.array([[1], [2]]), 1)


class TestLatencySummary(unittest.TestCase):
    def test_percentiles_of_known_distribution(self):
        samples = list(range(1, 101))  # 1..100 ms
        s = latency_summary(samples)
        self.assertAlmostEqual(s["p50_ms"], 50.5)
        self.assertAlmostEqual(s["p99_ms"], 99.01)

    def test_constant_samples(self):
        s = latency_summary([5.0] * 10)
        self.assertEqual(s["p50_ms"], 5.0)
        self.assertEqual(s["p99_ms"], 5.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            latency_summary([])


class TestPayloadCompression(unittest.TestCase):
    def test_fp32_baseline_is_one(self):
        self.assertEqual(payload_compression(4096, 1024), 1.0)

    def test_one_bit_rabitq_is_32x(self):
        self.assertEqual(payload_compression(128, 1024), 32.0)

    def test_sq8_is_4x(self):
        self.assertEqual(payload_compression(1024, 1024), 4.0)


def _ok(stdout):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


class TestDockerExecRetry(unittest.TestCase):
    """Docker Desktop intermittently kills `docker exec` under load (observed
    as CalledProcessError(-5) twice during the WS3 matrix). These reads are
    deterministic, so retrying is safe — but unlike RSS they produce the
    actual measured footprint, so exhausting the retries must still raise
    rather than report a default."""

    def test_returns_stdout_on_first_success(self):
        with mock.patch("subprocess.run", side_effect=[_ok("hello\n")]):
            self.assertEqual(docker_exec("c", "echo hi", retry_wait_s=0),
                             "hello\n")

    def test_retries_a_transient_signal_kill_then_succeeds(self):
        outs = [subprocess.CalledProcessError(-5, "docker"), _ok("second\n")]
        with mock.patch("subprocess.run", side_effect=outs) as run:
            got = docker_exec("c", "echo hi", attempts=3, retry_wait_s=0)
        self.assertEqual(got, "second\n")
        self.assertEqual(run.call_count, 2)

    def test_raises_after_exhausting_attempts(self):
        err = subprocess.CalledProcessError(-5, "docker")
        with mock.patch("subprocess.run", side_effect=err):
            with self.assertRaises(subprocess.CalledProcessError):
                docker_exec("c", "echo hi", attempts=2, retry_wait_s=0)


class TestMilvusRssBytes(unittest.TestCase):
    """RSS is an audit-only field: a transient `docker exec` failure must
    never take down a benchmark arm that has already run its workload.
    Observed for real during WS3: docker exec killed by signal 5 mid-matrix,
    losing a completed 20-minute arm."""

    def test_returns_median_of_samples_in_bytes(self):
        outs = [_ok("VmRSS:\t   1024 kB\n"), _ok("VmRSS:\t   4096 kB\n"),
                _ok("VmRSS:\t   2048 kB\n")]
        with mock.patch("subprocess.run", side_effect=outs):
            got = milvus_rss_bytes(samples=3, interval_s=0)
        self.assertEqual(got, 2048 * 1024)

    def test_retries_a_transient_failure_then_succeeds(self):
        outs = [subprocess.CalledProcessError(-5, "docker"),
                _ok("VmRSS:\t   512 kB\n")]
        with mock.patch("subprocess.run", side_effect=outs):
            got = milvus_rss_bytes(samples=1, interval_s=0, attempts=3)
        self.assertEqual(got, 512 * 1024)

    def test_returns_sentinel_instead_of_raising_when_docker_keeps_failing(self):
        err = subprocess.CalledProcessError(-5, "docker")
        with mock.patch("subprocess.run", side_effect=err):
            got = milvus_rss_bytes(samples=1, interval_s=0, attempts=2)
        self.assertEqual(got, RSS_UNAVAILABLE)

    def test_returns_sentinel_when_output_has_no_vmrss_line(self):
        with mock.patch("subprocess.run", side_effect=[_ok(""), _ok("")]):
            got = milvus_rss_bytes(samples=1, interval_s=0, attempts=2)
        self.assertEqual(got, RSS_UNAVAILABLE)


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["arm", "sweep_param", "sweep_value", "recall_at_10", "qps"],
    )


class TestPickBestSweepPoints(unittest.TestCase):
    def setUp(self):
        # arm_a reaches 0.99; arm_b tops out at 0.97.
        self.df = _frame([
            ("arm_a", "nprobe", 1, 0.85, 5000.0),
            ("arm_a", "nprobe", 8, 0.95, 3000.0),
            ("arm_a", "nprobe", 64, 0.992, 800.0),
            ("arm_b", "nprobe", 1, 0.90, 6000.0),
            ("arm_b", "nprobe", 8, 0.94, 4000.0),
            ("arm_b", "nprobe", 64, 0.97, 900.0),
        ])

    def test_picks_max_qps_row_meeting_target(self):
        out = pick_best_sweep_points(self.df, [0.90])
        row_a = out[(out.arm == "arm_a") & (out.target == 0.90)].iloc[0]
        # both nprobe=8 (0.95, 3000qps) and nprobe=64 qualify; 8 has higher qps
        self.assertEqual(row_a.sweep_value, 8)
        self.assertTrue(row_a.met_target)
        row_b = out[(out.arm == "arm_b") & (out.target == 0.90)].iloc[0]
        self.assertEqual(row_b.sweep_value, 1)  # 0.90 qualifies at 6000 qps

    def test_unmet_target_flagged_with_best_recall_row(self):
        out = pick_best_sweep_points(self.df, [0.99])
        row_b = out[(out.arm == "arm_b") & (out.target == 0.99)].iloc[0]
        self.assertFalse(row_b.met_target)
        self.assertEqual(row_b.sweep_value, 64)  # its max-recall point
        self.assertAlmostEqual(row_b.recall_at_10, 0.97)

    def test_one_row_per_arm_target(self):
        out = pick_best_sweep_points(self.df, [0.90, 0.95, 0.99])
        self.assertEqual(len(out), 6)


if __name__ == "__main__":
    unittest.main()
