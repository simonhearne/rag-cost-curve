"""Pytest fixtures for WS6a tests."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def stub_ripgrep(tmp_path, monkeypatch):
    """Provide a stub ripgrep executable on PATH for testing grep/glob logic.

    This fixture creates a fake `rg` command that emits ripgrep-shaped output
    (path:lineno:text lines, exit 0 with matches, exit 1 with none). It does
    NOT exercise real ripgrep behavior — only our parsing and truncation logic.
    Integration tests use the real ripgrep when it exists (guarded by
    shutil.which("rg")).
    """
    # Create a bin directory and add it to PATH
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    # Create the stub rg executable
    stub_script = bin_dir / "rg"
    stub_script.write_text("""\
#!/usr/bin/env python3
import sys
import re

# Simple stub that handles our test patterns
# Invoked as: rg [--line-number] [--no-heading] [--color=never] [--glob GLOB] PATTERN TARGET

args = sys.argv[1:]

# Respond to the guard's functionality probe (`rg --version`) the way real
# ripgrep does: exit 0 with a version string. Without this, the stricter
# make_agentic_tools() guard (I-7) would reject this stub too.
if args == ["--version"]:
    print("ripgrep-stub 0.0.0 (test double)")
    sys.exit(0)

# Skip known flags
known_flags = {"--line-number", "--no-heading", "--color=never"}
i = 0
pattern = None
target = None
glob_pattern = None

while i < len(args):
    if args[i] == "--glob":
        i += 2
        continue
    if args[i] in known_flags:
        i += 1
        continue
    if pattern is None:
        pattern = args[i]
    else:
        target = args[i]
    i += 1

# Simulate different behaviors based on pattern and target
if not pattern or not target:
    sys.exit(1)

# Generate fake matches
try:
    target_path = sys.argv[-1]
    if "many_matches" in target_path:
        # Emit 150 matches for testing truncation
        for i in range(150):
            print(f"{target_path}:{i+1}:match_{i}")
        sys.exit(0)
    elif "test.txt" in target_path and pattern == "hello":
        print(f"{target_path}:1:hello world")
        sys.exit(0)
    elif "single_file" in target_path:
        # Exactly 100 matches (boundary case)
        for i in range(100):
            print(f"{target_path}:{i+1}:pattern match {i}")
        sys.exit(0)
    elif "error_pattern" in pattern:
        # Simulate an error condition (non-zero, non-1 exit code)
        print("Error: something went wrong", file=sys.stderr)
        sys.exit(2)
    else:
        sys.exit(1)  # No matches
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(2)
""")
    stub_script.chmod(0o755)

    # Prepend the bin directory to PATH
    old_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", f"{bin_dir}:{old_path}")

    return stub_script
