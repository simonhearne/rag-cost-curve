"""Shared agent harness: client construction, single-turn and arm execution,
the judge, checkpointing and token counting.

Top-level rather than under a benchmark-specific package because 18 of its 33
import sites are outside the code-search benchmark -- the recall-quality,
memory and top-k runners all use its checkpointing and judge helpers. It was
named ws6a_runner.py historically.

WS6a live half: token counting, agentic tools, arm runners, judge,
orchestration. Everything here touches an API or the filesystem.

Pure logic lives in benchlib/code_retrieval/bench.py and is unit-tested;
this module was exercised by a pre-flight smoke script against live services
before any benchmark run (since removed; see git history for
scripts/smoke_ws6a_arms.py).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR, WS6A_AGENT_MODEL
from .code_retrieval import bench as _pure


# I-6. `messages.count_tokens` prices a whole REQUEST -- role framing and
# message structure -- not a bare string. That constant rides along on every
# call, so a corpus counted file-by-file (~2,900 calls) carries it ~2,900
# times while the same corpus counted in one call carries it once. The two
# "repo token totals" then disagree, and the per-file inflation is largest on
# the smallest files, which shifts the greedy prefix boundary. Measured once
# per run and subtracted from every batch, both totals describe the same
# string. See measure_request_overhead().


def measure_request_overhead(client, model: str) -> int:
    """Tokens a count_tokens request adds beyond the message text itself.

    For texts A and B whose concatenation introduces no token merge at the
    seam (both end in a newline here, so it cannot):

        c(A) = F + t(A) ; c(B) = F + t(B) ; c(AB) = F + t(A) + t(B)
        =>  F = c(A) + c(B) - c(AB)

    Three unbilled calls. Recorded in repo_stats.csv and the prefix manifest so
    the correction is auditable rather than an unexplained constant.
    """
    a = "alpha beta gamma delta epsilon zeta eta theta\n"
    b = "one two three four five six seven eight nine\n"

    def c(text: str) -> int:
        return client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}],
        ).input_tokens

    return c(a) + c(b) - c(a + b)


def count_tokens_batched(client, model: str, texts, max_chars: int = 2_000_000,
                         overhead: int = 0) -> int:
    """Exact token count of `texts` concatenated, via messages.count_tokens.

    Counting the concatenation is the right measure: it is exactly what arm (c)
    puts in the context window. Batched because a single request must stay
    under the API's size limit and the model's context window. Boundary effects
    between batches are on the order of one token per batch and are documented
    rather than corrected for.

    `overhead` is the per-request framing constant from
    measure_request_overhead(); it is subtracted once per batch. Pass 0 (the
    default) for the raw API figure.
    """
    total = 0
    buf: list[str] = []
    buf_chars = 0

    def flush() -> int:
        if not buf:
            return 0
        resp = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": "".join(buf)}],
        )
        return resp.input_tokens - overhead

    for t in texts:
        if buf_chars + len(t) > max_chars and buf:
            total += flush()
            buf, buf_chars = [], 0
        buf.append(t)
        buf_chars += len(t)
    total += flush()
    return total


# I-5. The prefix that reaches the context window is not the bare file bodies:
# every file carries a "===== path =====" header and the units are joined by a
# blank line. Selecting and reporting on bodies alone describes a DIFFERENT
# string from the one that is sent and paid for -- roughly 20k tokens adrift
# over a ~1,100-file prefix. prefix_unit_text() is the single definition of
# "what one file contributes", and load_prefix_text() is built from it, so the
# measured unit and the sent string cannot drift apart.

PREFIX_SEPARATOR = "\n\n"


def prefix_unit_text(root, path: str) -> str:
    """Exactly what `path` contributes to the stuffed prefix, separator included.

    "".join(prefix_unit_text(root, p) for p in files) reproduces
    load_prefix_text() plus one trailing separator -- so a sum over units
    over-counts the real string by that single trailing "\n\n" (~1 token),
    which is recorded rather than corrected.
    """
    body = (Path(root) / path).read_text(encoding="utf-8", errors="replace")
    return f"===== {path} =====\n" + body + PREFIX_SEPARATOR


def load_prefix_text(root, manifest_path=DATA_DIR / "ws6a_prefix_manifest.json") -> str:
    """Materialise arm (c)'s stuffed prefix as one string.

    Defined once and reused by every caller: the prefix must be byte-identical
    across all 40 questions or the prompt cache never hits, and rebuilding it
    ad hoc in each script is how that drift happens.
    """
    root = Path(root)
    manifest_data = json.loads(Path(manifest_path).read_text())
    units = [prefix_unit_text(root, p) for p in manifest_data["files"]]
    return "".join(units)[:-len(PREFIX_SEPARATOR)] if units else ""


# ---------------------------------------------------------- agentic tools


def resolve_in_root(root, path: str) -> Path:
    """Resolve `path` against `root`, refusing anything that escapes it.

    Tool arguments are untrusted model output. Resolve to canonical form and
    confine to the repository -- `..`, absolute paths, and symlink escapes all
    rejected.
    """
    root = Path(root).resolve()
    candidate = (root / path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path {path!r} resolves outside the repository root")
    return candidate


GREP_MAX_MATCHES = 100
READ_MAX_LINES = 400


def _ripgrep_available() -> bool:
    """True only if `rg` is on PATH AND runs `--version` successfully.

    shutil.which("rg") alone is satisfied by any non-functional file named
    `rg` (a broken stub, a name collision) -- I-7. If such a thing reaches
    PATH, every grep call returns an error string and the whole agentic arm
    silently produces uniformly-wrong, low-token rows: the most likely live
    trigger for C-2's fabricated 0%-accuracy arm.
    """
    if not shutil.which("rg"):
        return False
    try:
        proc = subprocess.run(["rg", "--version"], capture_output=True,
                              text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def make_agentic_tools(root):
    """Build the arm (b) tool set, bound to one repository root.

    Modelled on Claude Code's grep/glob/read semantics. These are OUR
    reimplementation, not Claude Code's tools -- the schemas are committed so
    any claim about arm (b) is auditable as a claim about these tools.

    Raises RuntimeError if ripgrep is not found or not functional on PATH,
    failing loudly at tool-construction time rather than opaquely mid-run
    after spending time.
    """
    from anthropic import beta_tool

    root = Path(root).resolve()

    # Fail loudly at tool-construction time if ripgrep is missing or
    # non-functional, rather than failing opaquely mid-run after spending
    # benchmark time and money.
    if not _ripgrep_available():
        raise RuntimeError(
            "ripgrep not found or not functional on PATH (`rg --version` "
            "failed); install it with `brew install ripgrep`"
        )

    @beta_tool
    def grep(pattern: str, path: str = ".", glob: str = "") -> str:
        """Search file contents with a regular expression.

        Returns at most 100 matches across all matched files. Use the glob
        argument to filter by filename pattern.

        Args:
            pattern: Regular expression to search for.
            path: Directory or file to search, relative to the repository root.
            glob: Optional filename filter, e.g. "*.py".
        """
        try:
            target = resolve_in_root(root, path)
        except ValueError as exc:
            return f"Error: {exc}"
        cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
        if glob:
            cmd += ["--glob", glob]
        cmd += [pattern, str(target)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            return "Error: ripgrep not found on PATH"
        if proc.returncode not in (0, 1):
            return f"Error: ripgrep failed: {proc.stderr.strip()}"
        out = proc.stdout.strip()
        if not out:
            return "No matches."
        lines = out.splitlines()
        # Truncate globally in Python for an accurate truncation signal.
        # We read ripgrep's full output into memory rather than bounding it
        # with --max-count, because for ~3k-file corpus the memory cost is
        # trivial and an accurate truncation marker matters more — an agent
        # that wrongly believes it saw every match will answer from partial data.
        if len(lines) > GREP_MAX_MATCHES:
            body = "\n".join(l.removeprefix(f"{root}/") for l in lines[:GREP_MAX_MATCHES])
            body += f"\n... truncated at {GREP_MAX_MATCHES} matches"
        else:
            body = "\n".join(l.removeprefix(f"{root}/") for l in lines)
        return body

    @beta_tool
    def glob(pattern: str) -> str:
        """List repository files matching a glob pattern.

        Returns at most 100 files. Patterns must be relative (no / prefix) and
        may not contain .. segments.

        Args:
            pattern: Glob pattern relative to the repository root, e.g.
                "fastapi/**/*.py".
        """
        # Reject absolute patterns and patterns containing .. traversal.
        if pattern.startswith("/"):
            return "Error: glob pattern must be relative, not absolute."
        if ".." in pattern.split("/"):
            return "Error: glob pattern must not contain .. segments."
        try:
            hits_all = root.glob(pattern)
        except (NotImplementedError, ValueError) as exc:
            return f"Error: invalid glob pattern: {exc}"
        # Filter results to paths genuinely inside the root, reusing resolve_in_root.
        hits = []
        for p in hits_all:
            if p.is_file():
                try:
                    resolve_in_root(root, str(p.relative_to(root)))
                    hits.append(str(p.relative_to(root)))
                except ValueError:
                    # Skip hits outside root (shouldn't happen after pattern check)
                    pass
        hits = sorted(hits)
        if not hits:
            return "No files matched."
        if len(hits) > GREP_MAX_MATCHES:
            body = "\n".join(hits[:GREP_MAX_MATCHES])
            body += f"\n... truncated at {GREP_MAX_MATCHES} of {len(hits)} files"
        else:
            body = "\n".join(hits)
        return body

    @beta_tool
    def read(path: str, offset: int = 1, limit: int = READ_MAX_LINES) -> str:
        """Read a file's contents with line numbers.

        Returns at most 400 lines. Limit is clamped to this maximum. Offsets
        are 1-indexed (first line is 1, not 0).

        Args:
            path: File path relative to the repository root.
            offset: 1-indexed first line to return.
            limit: Maximum number of lines to return.
        """
        try:
            target = resolve_in_root(root, path)
        except ValueError as exc:
            return f"Error: {exc}"
        if not target.is_file():
            return f"Error: {path} is not a file."
        limit = min(limit, READ_MAX_LINES)
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        chunk = lines[max(offset - 1, 0): max(offset - 1, 0) + limit]
        numbered = "\n".join(f"{offset + i:6d}\t{l}" for i, l in enumerate(chunk))
        if offset - 1 + limit < len(lines):
            numbered += f"\n... {len(lines) - (offset - 1 + limit)} more lines"
        return numbered or "(empty file)"

    return [grep, glob, read]


# --------------------------------------------------------------- arm runners

# ONE system prompt for all four arms, byte-identical. It deliberately names no
# tool and no retrieval strategy: the arms must differ in exactly one variable,
# the `tools` list. Changing this string invalidates every prior run.
SYSTEM_PROMPT = (
    "You are answering questions about a Python codebase.\n\n"
    "Answer the question directly and concisely. Always cite the specific "
    "repository-relative file path or paths your answer relies on (for example "
    "`package/module.py`), and name the relevant functions or classes.\n\n"
    "If you cannot determine the answer, say so plainly rather than guessing."
)

MAX_TOKENS = 4096

# Per-turn transcript: tool_result content is truncated to this many chars in
# the transcript (the untruncated length is recorded alongside it), so a
# giant grep dump does not blow up transcript size. I-4.
TRANSCRIPT_TOOL_RESULT_MAX_CHARS = 2_000


@dataclass
class RunResult:
    qid: str
    arm: str
    answer: str = ""
    turns: int = 0
    hit_turn_cap: bool = False
    stop_reason: str = ""
    wall_clock_s: float = 0.0
    started_at: str = ""
    error: str = ""
    usage: "object" = field(default_factory=lambda: _pure.Usage())
    # One entry per message: role, stop_reason, per-turn usage (dict, or None
    # for the synthesized tool-result turn), and a compact content
    # representation. The audit trail behind every aggregate row -- I-4.
    turns_detail: list = field(default_factory=list)


def _compact_blocks(blocks) -> list:
    """Compact JSONL-safe representation of a message's content blocks.

    Handles both SDK content-block objects (text/tool_use, attribute access)
    and the plain dicts the tool runner uses for synthesized tool_result
    turns (dict access). tool_result content is truncated to
    TRANSCRIPT_TOOL_RESULT_MAX_CHARS with the untruncated length recorded
    alongside it -- truncating loses no signal about whether the answer had
    the full result available, since the length is preserved.
    """
    out = []
    for b in blocks or []:
        btype = getattr(b, "type", None)
        if btype is None and isinstance(b, dict):
            btype = b.get("type")

        if btype == "text":
            text = b.text if hasattr(b, "text") else b.get("text", "")
            out.append({"type": "text", "text": text})
        elif btype == "tool_use":
            name = b.name if hasattr(b, "name") else b.get("name")
            inp = b.input if hasattr(b, "input") else b.get("input")
            out.append({"type": "tool_use", "name": name, "input": inp})
        elif btype == "tool_result":
            content = b.get("content") if isinstance(b, dict) else getattr(b, "content", "")
            content_str = content if isinstance(content, str) else json.dumps(content)
            is_error = b.get("is_error") if isinstance(b, dict) else getattr(b, "is_error", False)
            tool_use_id = (b.get("tool_use_id") if isinstance(b, dict)
                          else getattr(b, "tool_use_id", ""))
            out.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content_str[:TRANSCRIPT_TOOL_RESULT_MAX_CHARS],
                "content_length": len(content_str),
                "is_error": bool(is_error),
            })
        else:
            out.append({"type": btype or "unknown"})
    return out


def _single_turn(client, system_blocks, question: str, model: str,
                 max_tokens: int = MAX_TOKENS):
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_blocks,
        messages=[{"role": "user", "content": question}],
    )


def run_arm(client, arm: str, qid: str, question: str, *, root=None,
            prefix_text: str = "", mcp_tools=None,
            model: str = WS6A_AGENT_MODEL, kind: str | None = None,
            system_prompt: str | None = None,
            max_tokens: int | None = None) -> RunResult:
    """Run one (question, arm) pair and return measured results.

    Every token comes from an API `usage` field, accumulated across turns.
    Tool-using arms are capped at WS6A_TURN_CAP -- the SAME cap for both, so the
    arms never differ in turn budget as well as tool availability.

    `arm` is the published LABEL and `kind` the behaviour it dispatches on;
    they are the same string unless a caller says otherwise, so every WS6a
    call site is unchanged. WS6c needs them separable: `indexed_topk3` runs
    the `indexed` code path with a different tool binding, and recording it
    under the label `indexed` would collide with the default-top-k arm in the
    same checkpoint -- the two arms would overwrite each other's rows.

    `system_prompt` defaults to the module-level SYSTEM_PROMPT (WS6a's
    Python-codebase framing), so every frozen WS6a/WS6b/WS6c call site that
    does not pass it is byte-identical to before. A caller on a different
    corpus (WS8's GitHub issue threads, not a codebase) passes its own
    string here instead -- see benchlib.config.WS8_SYSTEM_PROMPT. Whatever
    is passed is used for EVERY kind branch below, so a caller that changes
    it changes exactly one variable (the framing text), never the tools list
    as well.

    `max_tokens` defaults to the module-level MAX_TOKENS, same pattern as
    `system_prompt`: every frozen call site that does not pass it is
    byte-identical to before. A caller re-running a single truncated
    question (stop_reason="max_tokens", the whole budget consumed by an
    unbilled `thinking` block with no text emitted) passes a larger value
    here instead of re-running the whole arm under a raised default, which
    would silently change every other row's comparability too.
    """
    from .config import WS6A_TURN_CAP

    kind = arm if kind is None else kind
    system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    started = time.time()
    result = RunResult(qid=qid, arm=arm,
                       started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime(started)))
    total = _pure.Usage()

    try:
        if kind == "parametric":
            resp = _single_turn(client, system_prompt, question, model,
                               max_tokens)
            total.add(_pure.Usage.from_api(resp.usage))
            result.answer = _text_of(resp)
            result.turns = 1
            result.stop_reason = resp.stop_reason or ""
            result.turns_detail.append({
                "role": "assistant", "stop_reason": result.stop_reason,
                "usage": _pure.Usage.from_api(resp.usage).as_dict(),
                "content": _compact_blocks(resp.content),
            })

        elif kind == "stuffed":
            # Stable cached prefix, volatile question after it: the
            # shared-prefix/varying-suffix pattern. 1h TTL because the run is
            # interleaved over hours.
            system_blocks = [
                {"type": "text", "text": system_prompt},
                {"type": "text",
                 "text": "Repository contents follow.\n\n" + prefix_text,
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ]
            resp = _single_turn(client, system_blocks, question, model,
                               max_tokens)
            total.add(_pure.Usage.from_api(resp.usage))
            result.answer = _text_of(resp)
            result.turns = 1
            result.stop_reason = resp.stop_reason or ""
            result.turns_detail.append({
                "role": "assistant", "stop_reason": result.stop_reason,
                "usage": _pure.Usage.from_api(resp.usage).as_dict(),
                "content": _compact_blocks(resp.content),
            })

        elif kind in ("agentic", "indexed"):
            tools = (make_agentic_tools(root) if kind == "agentic"
                     else list(mcp_tools or []))
            if not tools:
                raise ValueError(f"arm {arm!r} requires a non-empty tool list")
            runner = client.beta.messages.tool_runner(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=tools,
                messages=[{"role": "user", "content": question}],
            )
            last = None
            for message in runner:
                last = message
                total.add(_pure.Usage.from_api(message.usage))
                result.turns += 1
                result.turns_detail.append({
                    "role": message.role, "stop_reason": message.stop_reason or "",
                    "usage": _pure.Usage.from_api(message.usage).as_dict(),
                    "content": _compact_blocks(message.content),
                })
                # Public, cached: the same tool-result content the runner
                # will consume internally on its next iteration. Capturing
                # it here (rather than only the assistant messages the
                # iterator yields) is what makes tool_result blocks -- the
                # only place retrieved chunk payloads live -- visible in the
                # transcript. Cached, so this never calls a tool twice.
                tool_response = runner.generate_tool_call_response()
                if tool_response is not None:
                    result.turns_detail.append({
                        "role": tool_response.get("role", "user"),
                        "stop_reason": "", "usage": None,
                        "content": _compact_blocks(tool_response.get("content")),
                    })
                if result.turns >= WS6A_TURN_CAP:
                    # Only a genuine cutoff -- the model still wanted to call
                    # a tool -- counts as capped. A run that happens to give
                    # its final answer exactly on the last allowed turn is
                    # NOT capped; gating on the count alone false-positives
                    # that case and inflates the published turn_cap_rate. I-3.
                    if message.stop_reason == "tool_use":
                        result.hit_turn_cap = True
                    break
            if last is not None:
                result.answer = _text_of(last)
                result.stop_reason = last.stop_reason or ""
        else:
            raise ValueError(f"unknown arm {arm!r} (kind {kind!r})")

    except Exception as exc:  # recorded, never silently swallowed
        result.error = f"{type(exc).__name__}: {exc}"

    result.usage = total
    result.wall_clock_s = round(time.time() - started, 3)
    return result


def _text_of(message) -> str:
    return "".join(b.text for b in message.content if b.type == "text")


# --------------------------------------------------------------------- judge

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}


# Transient API failures (429, 500, 529 Overloaded) are ordinary weather on a
# multi-hour run, not bugs. The SDK's default max_retries=2 is sized for
# interactive use; a 529 exhausted it and killed the run 32 pairs in. Retries
# use the SDK's own exponential backoff and honour retry-after.
WS6A_API_MAX_RETRIES = 8


def make_client(**kwargs):
    """Anthropic client tuned for a long unattended benchmark run.

    Every script that spends money builds its client here, so retry behaviour
    is one decision in one place rather than a per-script default.
    """
    import anthropic

    kwargs.setdefault("max_retries", WS6A_API_MAX_RETRIES)
    return anthropic.Anthropic(**kwargs)


# The judge is claude-opus-5, which runs ADAPTIVE THINKING BY DEFAULT, and
# max_tokens caps thinking plus output together. At the original 512 the judge
# occasionally spent the whole budget thinking and returned a thinking block
# with NO text block -- json.loads("") then raised and killed a multi-hour run
# 15 pairs in. Observed, not theorised. The schedule escalates rather than
# fixing one large value so the common case stays cheap.
JUDGE_MAX_TOKENS_SCHEDULE = (4096, 16384)


class JudgeError(RuntimeError):
    """The judge returned nothing parseable after every retry.

    Raised rather than defaulted to judge_correct=False: recording a judge
    failure as a wrong answer fabricates a result, which is the C-2 lesson.
    execute_run treats this like an arm error -- the pair is NOT checkpointed,
    so a resumed run retries it.
    """


def load_judge_prompt(path) -> str:
    return Path(path).read_text()


def judge_answer(client, prompt_template: str, question: str, expected_files,
                 expected_symbols, answer: str, model=None) -> dict:
    """Blind LLM judge. Returns {judge_correct, judge_reason, usage}.

    The judge never learns which arm produced the answer -- only the question,
    the expected values, and the answer text.
    """
    from .config import WS6A_JUDGE_MODEL

    model = model or WS6A_JUDGE_MODEL
    if not answer.strip():
        return {"judge_correct": False, "judge_reason": "empty answer",
                "usage": _pure.Usage()}

    filled = prompt_template.format(
        question=question,
        expected_files="; ".join(expected_files),
        expected_symbols="; ".join(expected_symbols) or "(none specified)",
        answer=answer,
    )

    total = _pure.Usage()
    last = ""
    for attempt, budget in enumerate(JUDGE_MAX_TOKENS_SCHEDULE, start=1):
        resp = client.messages.create(
            model=model,
            max_tokens=budget,
            messages=[{"role": "user", "content": filled}],
            output_config={"format": {"type": "json_schema",
                                      "schema": JUDGE_SCHEMA}},
        )
        # Every attempt is billed, so every attempt's usage is accumulated --
        # a retried judgement must not look cheaper than it was.
        total.add(_pure.Usage.from_api(resp.usage))
        last = _text_of(resp)
        if last.strip():
            try:
                payload = json.loads(last)
            except json.JSONDecodeError:
                continue
            return {
                "judge_correct": bool(payload["correct"]),
                "judge_reason": payload["reason"],
                "usage": total,
            }
        # Empty text means the budget went entirely to thinking; retry bigger.

    raise JudgeError(
        f"judge produced no parseable verdict after "
        f"{len(JUDGE_MAX_TOKENS_SCHEDULE)} attempts "
        f"(last stop_reason={resp.stop_reason!r}, text={last[:120]!r})")


# -------------------------------------------------------------- orchestration


def append_checkpoint(path, row: dict) -> None:
    """Append one completed (qid, arm) result as a JSONL line, flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_checkpoint(path) -> dict:
    """{(qid, arm): row} for every result already recorded.

    Tolerates truncated final lines (from a kill mid-write). Malformed lines
    are skipped with a warning rather than losing the entire checkpoint.
    """
    path = Path(path)
    if not path.exists():
        return {}
    done = {}
    with open(path) as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done[(row["qid"], row["arm"])] = row
            except json.JSONDecodeError as exc:
                print(f"Warning: {path}:{line_num} truncated or malformed, skipped: {exc}",
                      file=sys.stderr)
    return done


def checkpoint_spent(path) -> float:
    """Total billed spend already recorded -- the resume point for the governor.

    Sums cost_total_billed (agent + judge, C-1) -- the same quantity
    governor.charge() is actually charged, so a resumed run's governor starts
    from the true cumulative spend rather than an agent-only undercount.
    """
    return round(sum(r.get("cost_total_billed", 0.0)
                     for r in load_checkpoint(path).values()), 6)


# Every checkpoint glob (relative to DATA_DIR) that total_spent_across_checkpoints
# sums over. A later workstream that wants its own spend folded into the same
# cumulative governor adds one pattern here -- see that function's docstring.
SPEND_CHECKPOINT_GLOBS = (
    "ws6a_*.jsonl",
    "ws6a_cache/*.jsonl",
    "ws6c_cache/*.jsonl",
    "ws9_cache/*.jsonl",
)

# WS9 is a SEPARATELY AUTHORISED budget ($55, spec section 7), not a later
# phase of WS6a's $75 or WS6c's $45. Seeding its governor from the tuple above
# would start it at ~$70 already spent and trip it on the first call, so WS9
# passes these narrower globs instead. They still cover every WS9 checkpoint
# shape, so WS9 remains cumulative ACROSS ITS OWN PHASES -- the per-script
# fresh budget that the I-2 defect describes is not being reintroduced.
WS9_SPEND_CHECKPOINT_GLOBS = ("ws9_cache/*.jsonl",)


def total_spent_across_checkpoints(data_dir, globs=SPEND_CHECKPOINT_GLOBS) -> float:
    """Cumulative billed spend across checkpoint files matching `globs`.

    The governor covers all phases of a workstream cumulatively (spec section
    8). Each script (smoke run, full run, each scaling-sweep subset, WS6c's
    topk run) writes its own checkpoint file; seeding CostGovernor(spent=...)
    from just one file authorises each phase its own fresh budget -- roughly
    4x the spec's cap (I-2). Callers MUST seed from this, not from a single
    checkpoint_spent() call.

    The DEFAULT deliberately fails CLOSED across workstreams: a WS6a script
    also sees WS6c's and WS9's spend, and vice versa. Over-counting a shared
    cap is the safe direction -- it can only make a governor trip EARLIER than
    the true remaining budget, never authorise more than intended.

    `globs` exists for the one case where that is wrong: a workstream with its
    OWN authorised cap, which must not inherit a spent balance from budgets it
    is not drawing on. WS9 passes WS9_SPEND_CHECKPOINT_GLOBS for exactly that
    reason. Narrowing `globs` is a budget decision and belongs in a spec, never
    in a script written on the day.
    """
    data_dir = Path(data_dir)
    paths = sorted(set().union(
        *(data_dir.glob(pattern) for pattern in globs)))
    return round(sum(checkpoint_spent(p) for p in paths), 6)


def execute_run(client, questions_df, arms, *, root, prefix_text, mcp_tools,
                prices, checkpoint_path, judge_prompt, governor,
                transcript_dir=None, seed=None, arm_kinds=None,
                system_prompt=None, max_tokens=None):
    """Run every (question, arm) pair interleaved, checkpointing as it goes.

    Skips pairs already in the checkpoint, so an aborted run resumes rather than
    restarts. A BudgetExceeded stops the run cleanly with everything so far
    preserved.

    `arm_kinds` maps an arm LABEL to the behaviour it dispatches on, for
    callers that run one code path twice under different labels (WS6c's
    `indexed` and `indexed_topk3`). Unmapped labels dispatch on themselves,
    which is every WS6a arm.

    `system_prompt` and `max_tokens` are forwarded to every run_arm() call
    unchanged, same default-None, resolve-inside-run_arm pattern as run_arm
    itself (see its docstring for R17/R19) -- every WS6a/WS6c call site that
    does not pass them is byte-identical to before. WS8 passes both: its own
    WS8_SYSTEM_PROMPT (R17 -- reusing WS6a's Python-codebase framing here
    would give WS8's two tool-using arms a wrong prompt while Phase 1's
    reused parametric rows ran under the corrected one, differing in two
    variables instead of one) and an explicit max_tokens with headroom over
    the 4096 default (R19 -- extended thinking is on by default for
    claude-sonnet-5 and can consume a whole small budget with no text
    emitted, as happened to Phase 1's ws8_q065).
    """
    import pandas as pd

    from .config import SEED, WS6A_AGENT_MODEL, WS6A_JUDGE_MODEL

    seed = SEED if seed is None else seed
    qrows = {r["qid"]: r for r in questions_df.to_dict("records")}
    active = [q for q, r in qrows.items()
              if str(r.get("excluded", "false")).lower() != "true"]
    order = _pure.interleaved_order(active, arms, seed)

    done = load_checkpoint(checkpoint_path)
    # Counted against THIS call's order, not against the whole checkpoint: a
    # checkpoint shared by two execute_run calls with different arm sets
    # (WS6c's two MCP sessions) otherwise prints a negative remainder.
    todo = sum(1 for pair in order if pair not in done)
    print(f"{len(done)} pair(s) already in the checkpoint; {todo} of "
          f"{len(order)} to run")

    errors: list[tuple[str, str, str]] = []

    for qid, arm in order:
        if (qid, arm) in done:
            continue
        row = qrows[qid]
        expected_files = _pure.split_field(row["expected_files"])
        expected_symbols = _pure.split_field(row["expected_symbols"])

        res = run_arm(client, arm, qid, row["question"], root=root,
                      prefix_text=prefix_text, mcp_tools=mcp_tools,
                      kind=(arm_kinds or {}).get(arm),
                      system_prompt=system_prompt, max_tokens=max_tokens)

        # Truncation handling (Task 6, standing): a max_tokens stop with an
        # EMPTY answer means the whole budget went to an unbilled thinking
        # block with no text emitted (Phase 1 lost ws8_q065 to exactly this).
        # judge_answer short-circuits an empty answer to judge_correct=False
        # WITHOUT calling the judge, so left unguarded this checkpoints an
        # unmeasured question as a free, confident zero -- the same C-2
        # shape as an arm error, just via a different mechanism. Treated
        # identically: skip the checkpoint entirely so a resumed run retries
        # the pair (matches gate_ws8.py's own truncated_empty handling).
        truncated_empty = (res.stop_reason == "max_tokens"
                           and not (res.answer or "").strip())

        if res.error or truncated_empty:
            # C-2: an arm error must NEVER be recorded as a confidently-wrong
            # answer -- that checkpoints a fabricated 0%-accuracy,
            # near-zero-cost row and resume skips it forever. Skip the
            # checkpoint entirely so a resumed run retries this pair.
            reason = res.error or (
                "truncated: max_tokens hit with an empty answer (thinking "
                "consumed the whole budget) -- not a genuine decline")
            errors.append((qid, arm, reason))
            print(f"  WARNING: {arm:11s} {qid}  run_arm failed, NOT checkpointed "
                  f"(will retry on resume): {reason}", file=sys.stderr)
            continue

        agent_price = prices[WS6A_AGENT_MODEL]
        billed = _pure.cost_billed(res.usage, agent_price)
        listed = _pure.cost_uncached_list(res.usage, agent_price)

        scored = _pure.score_programmatic(res.answer, expected_files,
                                          expected_symbols)
        try:
            verdict = judge_answer(client, judge_prompt, row["question"],
                                   expected_files, expected_symbols, res.answer)
        except Exception as exc:
            # Same rule as an arm error (C-2) for the PAIR: skip the
            # checkpoint entirely so a resumed run retries the pair, rather
            # than recording a fabricated verdict or aborting the whole run
            # over one judgement.
            #
            # Deliberately broad. run_arm already swallows its own exceptions,
            # so before this the judge was the ONLY unguarded API call in the
            # loop -- and a single transient 529 Overloaded killed a run 32
            # pairs in.
            #
            # Judge-exception spend pattern (established by gate_ws8.py's own
            # hand-rolled loop, generalised here): the AGENT call above
            # already happened and was billed (`billed`, computed just above
            # this try block). Charging the governor and appending a
            # spend-only checkpoint line under a distinct arm label means
            # spent_from_checkpoint sums it -- so cumulative spend stays
            # accurate across a resumed run -- while load_checkpoint's
            # resume-skip logic (keyed on (qid, arm)) never mistakes it for a
            # completed pair. Never discard an already-billed agent call.
            spend_only_arm = f"{arm}_judge_error"
            append_checkpoint(checkpoint_path, {
                "qid": qid, "arm": spend_only_arm,
                "cost_total_billed": billed,
                "error": f"{type(exc).__name__}: {exc}",
                "note": "agent call billed and charged; judge raised, pair "
                        "not recorded as done, resume will retry it",
            })
            governor.charge(billed)  # raises BudgetExceeded to stop cleanly; already checkpointed
            errors.append((qid, arm, f"{type(exc).__name__}: {exc}"))
            print(f"  WARNING: {arm:11s} {qid}  judge failed, agent cost "
                  f"${billed:.6f} charged and checkpointed as spend-only "
                  f"(will retry on resume): {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        judge_price = prices[WS6A_JUDGE_MODEL]
        judge_cost = _pure.cost_billed(verdict["usage"], judge_price)

        # I-3: a capped run is scored incorrect but still reported -- the
        # answer text and judge verdict stay visible so the divergence
        # between "judge liked it" and "forced incorrect because capped"
        # is auditable, rather than silently smoothed into the hit rate.
        any_file_hit = scored["any_file_hit"] and not res.hit_turn_cap
        all_files_hit = scored["all_files_hit"] and not res.hit_turn_cap
        symbol_hit = scored["symbol_hit"] and not res.hit_turn_cap

        record = {
            "qid": qid, "arm": arm,
            **res.usage.as_dict(),
            "turns": res.turns, "hit_turn_cap": res.hit_turn_cap,
            "stop_reason": res.stop_reason, "error": res.error,
            "wall_clock_s": res.wall_clock_s, "started_at": res.started_at,
            "answer": res.answer,
            "any_file_hit": any_file_hit,
            "all_files_hit": all_files_hit,
            "symbol_hit": symbol_hit,
            "matched_files": ";".join(scored["matched_files"]),
            "judge_correct": verdict["judge_correct"],
            "judge_reason": verdict["judge_reason"],
            # C-1: cost_agent_billed and cost_uncached_list are the matched
            # pair -- both priced over the agent-only token set, so their
            # ratio is the real caching story. cost_total_billed (agent +
            # judge) is the governor's charged quantity; it must never be
            # compared against cost_uncached_list, which was the old
            # (wrongly named) "cost_billed" bug: for the no-cache parametric
            # arm that comparison reported caching as making the run 160%
            # MORE expensive.
            "cost_total_billed": round(billed + judge_cost, 6),
            "cost_agent_billed": billed,
            "cost_judge_billed": judge_cost,
            "cost_uncached_list": listed,
            "in_stuffed_prefix": row.get("in_stuffed_prefix", ""),
            "difficulty": row.get("difficulty", ""),
            "scaling_subset": row.get("scaling_subset", ""),
        }

        append_checkpoint(checkpoint_path, record)

        if transcript_dir:
            try:
                tdir = Path(transcript_dir) / arm
                tdir.mkdir(parents=True, exist_ok=True)
                # I-4: one JSONL line per turn (the audit trail), not a
                # byte-duplicate of the aggregate checkpoint row.
                with open(tdir / f"{qid}.jsonl", "w") as fh:
                    for turn in res.turns_detail:
                        fh.write(json.dumps(turn) + "\n")
            except Exception as exc:
                print(f"Warning: transcript write failed for {arm:11s} {qid}: {exc}",
                      file=sys.stderr)

        governor.charge(record["cost_total_billed"])  # raises BudgetExceeded to stop
        print(f"  {arm:11s} {qid}  ${governor.spent:6.2f} spent  "
              f"turns={res.turns:2d}  judge={verdict['judge_correct']}")

    if errors:
        print(f"\n=== {len(errors)} pair(s) FAILED and were NOT recorded "
              f"(retry by re-running -- resume will pick them up) ===",
              file=sys.stderr)
        for qid, arm, err in errors:
            print(f"  {arm:11s} {qid}: {err}", file=sys.stderr)

    return pd.DataFrame(list(load_checkpoint(checkpoint_path).values()))


# ------------------------------------------------------------- claude-context


def mcp_env(codebase_path, collection_prefix: str = "ws6a") -> dict:
    """Environment for the claude-context MCP server.

    Credentials come from the ambient environment and are never written to the
    repo. Raises if any required variable is missing rather than silently
    indexing into the wrong place.

    Two settings are deliberate, and both were established by reading
    @zilliz/claude-context-mcp@0.1.15's own dist/:

    * CODE_CHUNKS_COLLECTION_NAME_OVERRIDE -- the ONLY collection-naming
      variable the pinned package recognises. The plan specified
      COLLECTION_NAME_PREFIX, which the package ignores entirely; under that
      name every scope would land in an auto-named
      `hybrid_code_chunks_<pathHash>` collection and the S/M/L sweep would have
      no way to tell its own indexes apart.
    * background sync and the filesystem trigger watcher are turned OFF. Both
      default to on and would re-embed the corpus mid-benchmark -- spending
      money outside the governor and moving the index under a run that is
      supposed to be measuring a fixed one.
    """
    from .config import WS6A_EMBED_MODEL, WS6A_EMBED_PROVIDER

    required = ("MILVUS_ADDRESS", "MILVUS_TOKEN", "OPENAI_API_KEY")
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f"missing required env vars: {', '.join(missing)}")

    env = dict(os.environ)
    env.update({
        "EMBEDDING_PROVIDER": WS6A_EMBED_PROVIDER,
        "EMBEDDING_MODEL": WS6A_EMBED_MODEL,
        "CODE_CHUNKS_COLLECTION_NAME_OVERRIDE": collection_prefix,
        "CLAUDE_CONTEXT_BACKGROUND_SYNC": "false",
        "CLAUDE_CONTEXT_TRIGGER_WATCHER": "false",
    })
    return env


# Only search_code is exposed to the agent. The other three MCP tools are
# administrative: index_codebase and clear_index MUTATE the index (clear_index
# would destroy the collection mid-benchmark), and get_indexing_status answers
# a question the agent has no reason to ask. A claude-context user querying an
# already-indexed repository uses search_code; exposing the admin surface adds
# tools the model can waste turns on and one it can do real damage with.
WS6A_MCP_AGENT_TOOLS = ("search_code",)


def _mcp_result_text(result) -> str:
    """MCP CallToolResult -> plain text for the tool_runner."""
    parts = [t for t in (getattr(b, "text", None) for b in result.content)
             if t is not None]
    return "\n".join(parts) if parts else "(no result)"


def claude_context_tools(session, raw_tools, codebase_path, topk=None):
    """MCP tool definitions -> SYNC beta_tool callables with the path bound.

    Two things happen here, and both were established by watching the arm fail.

    1. tool_runner receives RUNNABLE tools, never raw MCP definitions -- a raw
       definition has no callable behind it, so every call returns "Tool not
       found" and the arm looks uniformly, plausibly bad rather than broken.

    2. `path` is bound to the indexed codebase and REMOVED from the schema the
       model sees. claude-context's search_code requires an "ABSOLUTE path to
       the codebase directory", and the byte-identical system prompt (which
       cannot name a tool, let alone a filesystem path) never tells the model
       what that path is. Observed consequence: the model called no tool at
       all, answered both smoke questions from parametric memory, and produced
       a plausible-looking 1-turn indexed row. That is a HARNESS artifact, not
       a property of indexed search, and it would have understated arm (a) for
       the entire run.

       Binding the path is the CONTROLLED fix rather than naming the path in
       the system prompt: arm (b)'s grep/glob/read already have their root
       bound in a closure and never ask the model for it, so binding here makes
       the two tool-using arms differ in retrieval strategy and nothing else.
       It also matches deployment reality -- a claude-context client such as
       Claude Code supplies the workspace path from its own context rather than
       relying on the model to guess it.

    Synchronous by design (run_arm and execute_run are sync, so the arms differ
    only in the tools list); the wrapper bridges back to the MCP session's event
    loop via anyio.from_thread.run, which requires execute_run to run inside
    anyio.to_thread.run_sync while that loop owns the main thread.

    3. `topk`, when set, binds search_code's `limit` and removes it from the
       schema, exactly as `path` is bound. Measured from the WS6a transcripts:
       the model supplied `limit` on 1 of 110 calls, so the frozen run is a
       measurement of the server DEFAULT. Leaving the parameter visible while
       binding it would let the model set a value that is then silently
       overridden -- the arms would differ in what the model believes, not
       only in what it gets. topk=None reproduces the frozen behaviour.
    """
    import anyio.from_thread
    from anthropic import beta_tool

    tools = []
    for t in raw_tools:
        if t.name not in WS6A_MCP_AGENT_TOOLS:
            continue
        schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema")
        schema = json.loads(json.dumps(schema))  # deep copy; never mutate the server's
        bound = {"path": str(codebase_path)}
        schema.get("properties", {}).pop("path", None)
        if topk is not None:
            bound["limit"] = int(topk)
            schema.get("properties", {}).pop("limit", None)
        if "required" in schema:
            schema["required"] = [r for r in schema["required"] if r not in bound]

        def _make(tool_name, bound_args):
            def call(**kwargs):
                kwargs.update(bound_args)
                return _mcp_result_text(
                    anyio.from_thread.run(session.call_tool, tool_name, kwargs))
            return call

        tools.append(beta_tool(
            _make(t.name, dict(bound)), name=t.name, description=t.description,
            input_schema=schema,
        ))
    return tools


@asynccontextmanager
async def claude_context_session(codebase_path, collection_prefix="ws6a",
                                 stderr_log=None, topk=None):
    """Async CONTEXT MANAGER yielding (session, sync_tools, raw_tool_defs).

    Use as `async with claude_context_session(...) as (session, tools, raw):`.

    Deliberately not a bare async generator driven by __anext__()/aclose(), as
    the plan specified: stdio_client and ClientSession each open an anyio task
    group, and anyio requires a task group to be exited from the same task that
    entered it. aclose() unwinds from the generator's finalisation context
    instead, which raises a BaseExceptionGroup wrapping GeneratorExit on every
    teardown -- observed, not theorised. @asynccontextmanager keeps entry and
    exit in the caller's task.

    The server is chatty on stderr (it logs its whole configuration, including
    which embedding model it resolved and whether background sync is off).
    That output is the provenance for what was actually indexed, so it is
    captured to a file rather than discarded.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from .config import WS6A_MCP_PACKAGE

    params = StdioServerParameters(
        command="npx",
        args=["-y", WS6A_MCP_PACKAGE],
        env=mcp_env(codebase_path, collection_prefix),
    )
    errlog = open(stderr_log, "a") if stderr_log else sys.stderr
    try:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                yield (session,
                       claude_context_tools(
                           session, listed.tools, codebase_path, topk=topk),
                       listed.tools)
    finally:
        if stderr_log:
            errlog.close()
