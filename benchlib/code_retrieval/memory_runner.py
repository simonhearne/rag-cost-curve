"""WS6b live half: memsearch tool wrappers, arm runners, judge.

Everything here touches an API, a subprocess or the filesystem. Pure logic
lives in memory.py and is unit-tested; this module was exercised by a
pre-flight smoke script against live services before any full run (no
longer present in scripts/).

benchlib/agent_harness.py is imported, never modified: WS6a's results are
published and its regression tests pin that module's behaviour.

Design: results/code-retrieval.md
"""

import json
import subprocess
import time
from pathlib import Path

from ..config import (WS6B_AGENT_MODEL, WS6B_JUDGE_MODEL, WS6B_EMBED_MODEL,
                     WS6B_EMBED_PROVIDER, WS6B_MEMSEARCH_ROOT, WS6B_MILVUS_URI,
                     WS6B_TURN_CAP)
from . import bench as _pure
from ..agent_harness import (JUDGE_MAX_TOKENS_SCHEDULE, JUDGE_SCHEMA, JudgeError,
                          RunResult, _compact_blocks, _text_of, resolve_in_root)

MAX_TOKENS = 4096
TRANSCRIPT_TOOL_RESULT_MAX_CHARS = 2_000
WINDOW_EXCEEDED = "window_exceeded"

# ONE system prompt for all three arms, byte-identical. It names no tool and
# no retrieval strategy, so the arms differ in exactly one variable: what the
# model is given. Changing this string invalidates every prior run.
WS6B_SYSTEM_PROMPT = (
    "You are answering questions about the history of a software project.\n\n"
    "Answer the question directly and concisely, stating the specific value, "
    "name or decision the question asks for.\n\n"
    "If the history records more than one answer at different times, give the "
    "most recent one.\n\n"
    "If you cannot determine the answer, say so plainly rather than guessing."
)


# ----------------------------------------------------------------- judge


def judge_ws6b(client, prompt_template: str, question: str,
               expected_answer: str, stale_answer: str, answer: str,
               model=None) -> dict:
    """Blind judge. Returns {judge_correct, judge_reason, usage}.

    The judge sees the question, the expected value and the candidate answer.
    It never learns which arm produced the answer. It IS told the superseded
    value for superseded probes -- that is a property of the question, not of
    the arm, so blindness holds.
    """
    model = model or WS6B_JUDGE_MODEL
    if not (answer or "").strip():
        return {"judge_correct": False, "judge_reason": "empty answer",
                "usage": _pure.Usage()}

    filled = prompt_template.format(
        question=question, expected_answer=expected_answer,
        stale_answer=stale_answer or "(none)", answer=answer)

    total = _pure.Usage()
    last = ""
    for budget in JUDGE_MAX_TOKENS_SCHEDULE:
        resp = client.messages.create(
            model=model, max_tokens=budget,
            messages=[{"role": "user", "content": filled}],
            output_config={"format": {"type": "json_schema",
                                      "schema": JUDGE_SCHEMA}})
        # Every attempt is billed, so every attempt's usage accumulates: a
        # retried judgement must not look cheaper than it was.
        total.add(_pure.Usage.from_api(resp.usage))
        last = _text_of(resp)
        if last.strip():
            try:
                payload = json.loads(last)
            except json.JSONDecodeError:
                continue
            return {"judge_correct": bool(payload["correct"]),
                    "judge_reason": payload["reason"], "usage": total}
    raise JudgeError(
        f"judge produced no parseable verdict after "
        f"{len(JUDGE_MAX_TOKENS_SCHEDULE)} attempts (last={last[:120]!r})")


# ------------------------------------------------------------- the gate


def gate_verdict(accuracy: float) -> str:
    """Spec 5.5. 'investigate' is a real outcome, not a soft pass.

    An 'investigate' verdict requires inspecting every correct item by name
    in results/code-retrieval.md; it does not authorise proceeding on its own.
    """
    from ..config import WS6B_GATE_FAIL, WS6B_GATE_PASS
    if accuracy <= WS6B_GATE_PASS:
        return "pass"
    if accuracy <= WS6B_GATE_FAIL:
        return "investigate"
    return "fail"


# --------------------------------------------------------- memsearch tools


def memsearch_cli(args: list[str], *, timeout: int = 300):
    """Invoke the SHIPPED memsearch CLI at the pinned commit.

    `uv run` from the plugin checkout, so the commit that executes is the
    commit that is pinned, rather than whatever `pip install memsearch`
    happens to resolve to today.
    """
    return subprocess.run(
        ["uv", "run", "--quiet", "memsearch", *args],
        cwd=str(WS6B_MEMSEARCH_ROOT), capture_output=True, text=True,
        timeout=timeout)


def memsearch_available() -> bool:
    """True only if memsearch EXECUTES successfully.

    WS6a I-7: `rg` was on PATH as a shell function while shutil.which
    returned None, and the arm was silently unrunnable for an entire session.
    A path check is not a capability check. Fail loudly at tool-construction
    time rather than opaquely mid-run after spending money.
    """
    try:
        return memsearch_cli(["--version"], timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _ms_common(collection: str) -> list[str]:
    return ["--collection", collection, "--milvus-uri", WS6B_MILVUS_URI,
            "--provider", WS6B_EMBED_PROVIDER, "--model", WS6B_EMBED_MODEL]


def make_memsearch_tools(collection: str, corpus_root, *, check: bool = True):
    """Arm (a)'s three tools, mirroring the shipped memory-recall skill.

    L1 search -> L2 expand -> L3 read the source markdown. The synthetic
    corpus has no .jsonl transcripts, so the source markdown IS the deepest
    layer -- which is also memsearch's own position that markdown is the
    source of truth and Milvus a rebuildable shadow index.

    No memsearch parameter is tuned: defaults are what C1's low-effort claim
    is about, so arm (a)'s numbers are a floor, not a ceiling.
    """
    from anthropic import beta_tool

    root = Path(corpus_root).resolve()
    if check and not memsearch_available():
        raise RuntimeError(
            "memsearch is not runnable (`uv run memsearch --version` failed); "
            f"checked in {WS6B_MEMSEARCH_ROOT}")

    @beta_tool
    def memory_search(query: str, top_k: int = 5) -> str:
        """Search the project's conversation history for relevant passages.

        Returns ranked excerpts, each with a chunk hash and its source file.

        Args:
            query: What to look for, in natural language.
            top_k: How many results to return.
        """
        try:
            proc = memsearch_cli(["search", query, "--top-k", str(top_k),
                                  "--json-output", *_ms_common(collection)])
        except subprocess.TimeoutExpired:
            return "Error: search timed out."
        if proc.returncode != 0:
            return f"Error: search failed: {proc.stderr.strip()[:500]}"
        return proc.stdout.strip() or "No results."

    @beta_tool
    def memory_expand(chunk_hash: str) -> str:
        """Show the full section of history surrounding a search result.

        Args:
            chunk_hash: The chunk hash from a memory_search result.
        """
        try:
            proc = memsearch_cli(["expand", chunk_hash, "--json-output",
                                  *_ms_common(collection)])
        except subprocess.TimeoutExpired:
            return "Error: expand timed out."
        if proc.returncode != 0:
            return f"Error: expand failed: {proc.stderr.strip()[:500]}"
        return proc.stdout.strip() or "Not found."

    @beta_tool
    def memory_read_session(source_path: str) -> str:
        """Read one day of the raw conversation history in full.

        Args:
            source_path: A day file name such as "2025-03-14.md".
        """
        try:
            target = resolve_in_root(root, Path(source_path).name)
        except ValueError as exc:
            return f"Error: {exc}"
        if not target.exists():
            return f"Error: no such session file: {source_path}"
        return target.read_text()

    return [memory_search, memory_expand, memory_read_session]


# ------------------------------------------------------------ replay arm


def load_corpus_text(corpus_root, first_day: int, last_day: int) -> str:
    """Concatenate day files [first_day, last_day) in chronological order.

    Sorted by filename, which is an ISO date, so chronological order is
    lexicographic order. Never a directory-listing order, which is not
    stable across filesystems.
    """
    files = sorted(Path(corpus_root).glob("*.md"))[first_day:last_day]
    return "\n\n".join(f.read_text() for f in files)


def replay_system_blocks(corpus_text: str) -> list[dict]:
    """Stable cached prefix, volatile question after it.

    The history is cached at the 1h TTL because probes run contiguously per
    (arm, size) cell (spec 5.6): one write, then n-1 reads. The tiny system
    prompt is NOT cached -- caching a ~60-token block costs more in write
    premium than it could ever save.
    """
    return [
        {"type": "text", "text": WS6B_SYSTEM_PROMPT},
        {"type": "text",
         "text": "The project's conversation history follows.\n\n" + corpus_text,
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]


# ------------------------------------------------------------ arm runner


def is_window_error(exc) -> bool:
    """True when an exception is the context-window rejection.

    The API's message is `prompt is too long: N tokens > M maximum`. It does
    NOT contain the word "context", which is what an earlier version of this
    check looked for.
    """
    low = str(exc).lower()
    if "prompt is too long" in low:
        return True
    return ("token" in low and "maximum" in low
            and ("too long" in low or "exceed" in low))


def run_arm_ws6b(client, arm: str, probe_id: str, question: str, *,
                 corpus_text: str = "", tools=None,
                 model: str = WS6B_AGENT_MODEL) -> RunResult:
    """Run one (probe, arm) pair and return measured results.

    Every token comes from an API `usage` field, accumulated across turns.
    Arm 'memsearch' is capped at WS6B_TURN_CAP; the other arms are
    single-turn by construction, so no cap can differ between them.
    """
    started = time.time()
    result = RunResult(qid=probe_id, arm=arm,
                       started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime(started)))
    total = _pure.Usage()

    try:
        if arm == "parametric":
            resp = client.messages.create(
                model=model, max_tokens=MAX_TOKENS, system=WS6B_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": question}])
            total.add(_pure.Usage.from_api(resp.usage))
            result.answer = _text_of(resp)
            result.turns = 1
            result.stop_reason = resp.stop_reason or ""
            result.turns_detail.append({
                "role": "assistant", "stop_reason": result.stop_reason,
                "usage": _pure.Usage.from_api(resp.usage).as_dict(),
                "content": _compact_blocks(resp.content)})

        elif arm in ("replay", "replay_trunc"):
            if not corpus_text:
                raise ValueError(f"arm {arm!r} requires corpus_text")
            resp = client.messages.create(
                model=model, max_tokens=MAX_TOKENS,
                system=replay_system_blocks(corpus_text),
                messages=[{"role": "user", "content": question}])
            total.add(_pure.Usage.from_api(resp.usage))
            result.answer = _text_of(resp)
            result.turns = 1
            result.stop_reason = resp.stop_reason or ""
            result.turns_detail.append({
                "role": "assistant", "stop_reason": result.stop_reason,
                "usage": _pure.Usage.from_api(resp.usage).as_dict(),
                "content": _compact_blocks(resp.content)})

        elif arm == "memsearch":
            if not tools:
                raise ValueError("arm 'memsearch' requires a non-empty tool list")
            runner = client.beta.messages.tool_runner(
                model=model, max_tokens=MAX_TOKENS, system=WS6B_SYSTEM_PROMPT,
                tools=list(tools),
                messages=[{"role": "user", "content": question}])
            last = None
            for message in runner:
                last = message
                total.add(_pure.Usage.from_api(message.usage))
                result.turns += 1
                result.turns_detail.append({
                    "role": message.role,
                    "stop_reason": message.stop_reason or "",
                    "usage": _pure.Usage.from_api(message.usage).as_dict(),
                    "content": _compact_blocks(message.content)})
                # Public and cached: the same tool-result content the runner
                # consumes internally on its next iteration. Capturing it here
                # is what makes tool_result blocks -- the only place retrieved
                # chunk payloads live -- visible in the transcript.
                tool_response = runner.generate_tool_call_response()
                if tool_response is not None:
                    result.turns_detail.append({
                        "role": tool_response.get("role", "user"),
                        "stop_reason": "", "usage": None,
                        "content": _compact_blocks(tool_response.get("content"))})
                if result.turns >= WS6B_TURN_CAP:
                    # Only a genuine cutoff -- the model still wanted a tool --
                    # counts as capped. A run that gives its final answer
                    # exactly on the last allowed turn is NOT capped; gating on
                    # the count alone false-positives that and inflates the
                    # published turn_cap_rate.
                    if message.stop_reason == "tool_use":
                        result.hit_turn_cap = True
                    break
            if last is not None:
                result.answer = _text_of(last)
                result.stop_reason = last.stop_reason or ""
        else:
            raise ValueError(f"unknown arm {arm!r}")

    except Exception as exc:  # recorded, never silently swallowed
        msg = f"{type(exc).__name__}: {exc}"
        # Spec 8.4: exceeding the window is a hard capability boundary with a
        # null cost, never an extrapolated one.
        #
        # Matched against the API's ACTUAL message, captured verbatim:
        #   "prompt is too long: 1221932 tokens > 1000000 maximum"
        # An earlier version required the word "context" and therefore missed
        # it, so the cliff was recorded as a generic error and the row was
        # dropped instead of published. The word "context" does not appear.
        if is_window_error(exc):
            result.error = WINDOW_EXCEEDED
            result.stop_reason = msg[:400]
        else:
            result.error = msg

    result.usage = total
    result.wall_clock_s = round(time.time() - started, 3)
    return result
