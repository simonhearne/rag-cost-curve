"""Checksum manifest: every file the notebooks produce or download gets a
sha256 recorded in data/MANIFEST.json. `verify` re-hashes and compares.
"""

import hashlib
import json
import time
from pathlib import Path

from .config import DATA_DIR, REPO_ROOT

MANIFEST_PATH = DATA_DIR / "MANIFEST.json"


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"files": {}, "meta": {}}


def _manifest_key(path: Path) -> str:
    """Manifest key for `path`.

    A path under DATA_DIR keeps the DATA_DIR-relative key it has always had,
    so every entry already committed in data/MANIFEST.json (WS2's included)
    keeps resolving. A path outside DATA_DIR (e.g. a results/ CSV) is keyed
    relative to REPO_ROOT instead of raising: `relative_to(DATA_DIR)` used to
    raise ValueError for any results/ path, which hit
    scripts/code-retrieval/build_ws6a_corpus.py AFTER a ~2,900-call paid run had already
    completed (I-1).
    """
    path = Path(path).resolve()
    try:
        return str(path.relative_to(DATA_DIR.resolve()))
    except ValueError:
        return str(path.relative_to(REPO_ROOT.resolve()))


def _resolve_manifest_key(rel: str) -> Path:
    """Inverse of `_manifest_key`: the path a manifest key refers to."""
    candidate = DATA_DIR / rel
    if not candidate.exists():
        outside = REPO_ROOT / rel
        if outside.exists():
            return outside
    return candidate


def record(path: Path, extra: dict | None = None) -> str:
    """Hash `path` and record it in the manifest. Returns the digest."""
    m = load_manifest()
    digest = sha256_file(path)
    rel = _manifest_key(path)
    m["files"][rel] = {
        "sha256": digest,
        "bytes": Path(path).stat().st_size,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(extra or {}),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2, sort_keys=True))
    return digest


def record_meta(key: str, value) -> None:
    m = load_manifest()
    m["meta"][key] = value
    MANIFEST_PATH.write_text(json.dumps(m, indent=2, sort_keys=True))


def verify(paths: list[Path] | None = None) -> list[str]:
    """Re-hash files against the manifest. Returns list of mismatches."""
    m = load_manifest()
    bad = []
    entries = m["files"]
    for rel, info in entries.items():
        p = _resolve_manifest_key(rel)
        if paths is not None and p.resolve() not in [q.resolve() for q in paths]:
            continue
        if not p.exists():
            bad.append(f"MISSING {rel}")
        elif sha256_file(p) != info["sha256"]:
            bad.append(f"HASH MISMATCH {rel}")
    return bad
