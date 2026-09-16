"""Task 10(b): a regression test for scripts/recall-quality/ws4e_phase1_run.py, which had
none before this. The single line `if replicate: continue` in
write_depth_points is all that stops the Amendment 1 replicate cell's 264
sq8_np512/k=1 rows from folding into the primary k=1 aggregate and turning
it into a 528-row average.

Design: results/recall-vs-quality.md
"""

import importlib.util

import pandas as pd
import pytest


def _load_ws4e_phase1_run():
    """scripts/ is deliberately NOT a package (see test_ws4e.py's
    test_no_changepoint_is_fitted_on_four_arms) -- load it by file path
    rather than making scripts/ importable, which would change how every
    other script resolves its own imports."""
    spec = importlib.util.spec_from_file_location(
        "ws4e_phase1_run", "scripts/recall-quality/ws4e_phase1_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cells_marks_exactly_one_cell_as_replicate():
    mod = _load_ws4e_phase1_run()
    replicate_flags = [replicate for (_, _, _, replicate) in mod.cells()]
    assert replicate_flags.count(True) == 1


def test_the_replicate_cell_is_the_amendment_1_arm_and_depth():
    from benchlib.config import WS4E_REPLICATE_ARM, WS4E_REPLICATE_K
    mod = _load_ws4e_phase1_run()
    reps = [(k, arms) for (k, _, arms, replicate) in mod.cells() if replicate]
    assert len(reps) == 1
    k, arms = reps[0]
    assert k == WS4E_REPLICATE_K
    assert arms == (WS4E_REPLICATE_ARM,)


def test_depth_points_never_folds_the_replicate_into_a_primary_group():
    """Regression for the single `if replicate: continue` line in
    write_depth_points: without it, sq8_np512/k=1 would report
    n_questions=528 (264 primary + 264 replicate) instead of 264."""
    from benchlib.config import ws4e_dir
    path = ws4e_dir() / "depth_points.csv"
    if not path.exists():
        pytest.skip("results/ws4e/depth_points.csv not built yet")
    df = pd.read_csv(path)
    assert (df["n_questions"] <= 264).all()
    row = df[(df.arm == "sq8_np512") & (df.k == 1)]
    assert len(row) == 1
    assert row.iloc[0]["n_questions"] == 264
