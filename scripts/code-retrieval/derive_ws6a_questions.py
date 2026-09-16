#!/usr/bin/env python
"""Derive and validate the computed columns on results/ws6a/questions.csv.

Fills in_stuffed_prefix and expected_files_last_modified, then validates. The
derived columns are computed AFTER blind authoring so prefix membership cannot
influence question selection -- the authored set is committed first, before
the prefix manifest exists on disk, so the ordering is provable from history.

Run:  python scripts/code-retrieval/derive_ws6a_questions.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.code_retrieval import bench as ws6a
from benchlib.config import (
    DATA_DIR, WS6A_DIFFICULTY_MIX, WS6A_N_QUESTIONS, WS6A_N_SCALING_QUESTIONS,
    ws6a_checkout, ws6a_dir,
)


def main() -> None:
    root = ws6a_checkout()
    qpath = ws6a_dir() / "questions.csv"
    df = pd.read_csv(qpath, dtype=str, keep_default_na=False)

    repo_files = set(ws6a.git_ls_files(root))
    prefix = json.loads((DATA_DIR / "ws6a_prefix_manifest.json").read_text())
    included = prefix["files"]

    problems = ws6a.validate_questions(df.to_dict("records"), repo_files)
    if problems:
        for p in problems:
            print(f"PROBLEM: {p}")
        raise SystemExit(f"{len(problems)} problem(s); fix questions.csv first")

    if len(df) != WS6A_N_QUESTIONS:
        raise SystemExit(f"expected {WS6A_N_QUESTIONS} questions, got {len(df)}")
    n_scaling = (df["scaling_subset"].str.lower() == "true").sum()
    if n_scaling != WS6A_N_SCALING_QUESTIONS:
        raise SystemExit(
            f"expected {WS6A_N_SCALING_QUESTIONS} scaling_subset questions, "
            f"got {n_scaling}")

    mix = df["difficulty"].value_counts().to_dict()
    if mix != WS6A_DIFFICULTY_MIX:
        raise SystemExit(f"difficulty mix must be {WS6A_DIFFICULTY_MIX}, got {mix}")

    df["in_stuffed_prefix"] = [
        ws6a.classify_prefix_membership(ws6a.split_field(f), included)
        for f in df["expected_files"]
    ]
    df["expected_files_last_modified"] = [
        max(ws6a.git_last_modified(root, p) for p in ws6a.split_field(f))
        for f in df["expected_files"]
    ]

    df.to_csv(qpath, index=False)
    manifest.record(qpath)

    print(f"validated {len(df)} questions")
    print("difficulty:", mix)
    print("in_stuffed_prefix:", df["in_stuffed_prefix"].value_counts().to_dict())
    print("oldest expected file:", df["expected_files_last_modified"].min())
    print("newest expected file:", df["expected_files_last_modified"].max())


if __name__ == "__main__":
    main()
