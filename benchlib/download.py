"""Pinned-revision downloads for the corpus shards and NQ-Open.

All downloads go through hf_hub_download at a pinned git revision, are
hard-linked into data/ so the tree is self-contained, and are sha256-recorded
in data/MANIFEST.json.
"""

from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from . import manifest
from .config import (
    CORPUS_N_SHARDS,
    CORPUS_REPO,
    CORPUS_REVISION,
    CORPUS_SHARD_TEMPLATE,
    DATA_DIR,
    NQ_REPO,
    NQ_REVISION,
    NQ_TRAIN_FILE,
    NQ_VAL_FILE,
)

SHARD_DIR = DATA_DIR / "corpus_shards"
NQ_DIR = DATA_DIR / "nq"


def _link_out(cached: str, local: Path) -> Path:
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        try:
            local.hardlink_to(cached)
        except OSError:
            import shutil

            shutil.copy2(cached, local)
    return local


def shard_local_path(i: int) -> Path:
    return SHARD_DIR / f"train-{i:05d}-of-{CORPUS_N_SHARDS:05d}.parquet"


def download_shard(i: int) -> Path:
    local = shard_local_path(i)
    if local.exists():
        return local
    cached = hf_hub_download(
        repo_id=CORPUS_REPO,
        filename=CORPUS_SHARD_TEMPLATE.format(i=i),
        repo_type="dataset",
        revision=CORPUS_REVISION,
    )
    return _link_out(cached, local)


def ensure_rows(target_rows: int, record_manifest: bool = True) -> list[dict]:
    """Download shards in order until cumulative rows >= target_rows.

    Deterministic prefix: the 1M slice is a strict prefix of the 10M slice.
    Returns [{shard, path, rows, cum_rows}].
    """
    out, cum = [], 0
    bar = tqdm(total=target_rows, unit="rows", desc="corpus rows")
    for i in range(CORPUS_N_SHARDS):
        p = download_shard(i)
        rows = pq.ParquetFile(p).metadata.num_rows
        if record_manifest:
            manifest.record(p, extra={"rows": rows, "shard": i,
                                      "repo": CORPUS_REPO,
                                      "revision": CORPUS_REVISION})
        cum += rows
        out.append({"shard": i, "path": str(p), "rows": rows, "cum_rows": cum})
        bar.update(min(rows, max(0, target_rows - (cum - rows))))
        if cum >= target_rows:
            break
    bar.close()
    if cum < target_rows:
        raise RuntimeError(f"corpus exhausted at {cum:,} rows < {target_rows:,}")
    return out


def fetch_nq() -> dict[str, Path]:
    """Download NQ-Open train + validation parquet at the pinned revision."""
    out = {}
    for split, filename in (("train", NQ_TRAIN_FILE), ("validation", NQ_VAL_FILE)):
        cached = hf_hub_download(repo_id=NQ_REPO, filename=filename,
                                 repo_type="dataset", revision=NQ_REVISION)
        local = _link_out(cached, NQ_DIR / Path(filename).name)
        manifest.record(local, extra={"repo": NQ_REPO, "revision": NQ_REVISION,
                                      "split": split})
        out[split] = local
    return out
