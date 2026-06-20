"""Execute every ``python`` code block in the docs and DIFF its printed
output against the ``output`` block shown next to it.

The companion ``dev/check_docs.py`` only catches *exceptions*; it never checks
that a printed number still matches what the docs claim. That gap let the
gmdb-gmab chapter's TVOG silently drift (3,433,960 -> 1,638) without any test
failing. This suite closes it: a doc number that drifts now fails CI.

Matching convention (already used throughout the cookbook):

    ```python
    print(f"BEL = {m.bel[0]:.2f}")
    ```

    출력:

    ```
    BEL = 39.11
    ```

A ``python`` block's expected output is the *next fenced block* when that block
is bare ```` ``` ```` or ```` ```text ```` and only blank lines / a short
"출력:" marker sit between them. Blocks with no such following block (setup-only
code, or code followed by another ``python``/``{admonition}`` block) are still
executed -- to build up the shared namespace -- but their output is not diffed.

Blocks are run in one shared namespace per file, in document order, so a later
block may use names bound by an earlier one (the cookbook chapters rely on this).
"""
from __future__ import annotations

import difflib
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # no display for any plotting block

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    ROOT / "README.md",
    ROOT / "docs" / "getting-started.md",
    *sorted((ROOT / "docs" / "tutorial").glob("*.md")),
    *sorted((ROOT / "docs" / "solvency").glob("*.md")),
    *sorted((ROOT / "docs" / "cookbook").rglob("*.md")),
]

# A fenced block: ```<info>\n<body>```  -- capture the info string and the body.
_FENCE = re.compile(r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```[ \t]*$",
                    re.DOTALL | re.MULTILINE)

# Code blocks that are inherently non-deterministic or side-effect only: execute
# them (for namespace state) but never diff their stdout. Keyed by substrings of
# the code. Plotting blocks emit no stable stdout; an unseeded RNG is unstable.
_NO_DIFF_CODE = ("matplotlib", "plt.", ".savefig(", ".show()",
                 "generate_images", "tempfile", "mkdtemp", "TemporaryDirectory")


def _is_nondeterministic(code: str) -> bool:
    """True if the block uses randomness without a fixed seed."""
    if "random" not in code:
        return False
    # default_rng(<arg>) / seed(<arg>) / RandomState(<arg>) with an explicit
    # non-empty seed argument is deterministic; a bare default_rng() is not.
    seeded = re.search(r"(default_rng|RandomState|seed)\(\s*\w", code)
    return seeded is None


def _blocks(text: str):
    """Yield (info, body) for each fenced block, in document order."""
    for m in _FENCE.finditer(text):
        yield m.group("info").strip(), m.group("body")


def _python_cases(text: str):
    """Walk a doc's fenced blocks and yield (code, expected_or_None) for each
    python block, where expected is the adjacent output block's body or None."""
    blocks = list(_blocks(text))
    cases = []
    for i, (info, body) in enumerate(blocks):
        if info not in ("python", "{code-block} python"):
            continue
        expected = None
        if i + 1 < len(blocks):
            nxt_info, nxt_body = blocks[i + 1]
            # The next fence is the output iff it is bare or ```text. Anything
            # else (python, {admonition}, {list-table}, ...) means no output.
            if nxt_info in ("", "text"):
                expected = nxt_body
        cases.append((body, expected))
    return cases


def _norm(s: str) -> list[str]:
    """Normalize a block body for value-level comparison: collapse each line's
    internal whitespace runs to a single space and strip, then drop trailing
    blank lines. Collapsing whitespace lets the comparison catch *number* drift
    while ignoring pure spacing differences -- numpy array reprs space their
    elements differently across versions (``[   0.   4500.]`` vs
    ``[   0. 4500.]``), which is not a doc-correctness change."""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in s.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _subseq_missing(want: list[str], got: list[str]) -> str | None:
    """Return the first ``want`` line that is not found (in order) in ``got``,
    or None if every ``want`` line appears as an ordered subsequence of ``got``.
    Blank ``want`` lines are skipped (they carry no value)."""
    it = iter(got)
    for w in want:
        if w == "":
            continue
        if not any(w == g for g in it):
            return w
    return None


_DOC_FILES = [p for p in TARGETS if p.exists()]


@pytest.mark.parametrize("doc", _DOC_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_doc_python_block_output_matches(doc: Path):
    """Run a doc's python blocks in order; assert every diffable block's stdout
    matches the output shown beside it."""
    cases = _python_cases(doc.read_text())
    if not cases:
        pytest.skip("no python blocks")

    ns = {"__name__": "__doc_exec__"}
    mismatches: list[str] = []
    diffed = 0
    for idx, (code, expected) in enumerate(cases):
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compile(code, f"{doc.name}#block{idx}", "exec"), ns)
        except Exception as exc:  # a doc block must at least run
            mismatches.append(f"block {idx}: raised {type(exc).__name__}: {exc}")
            continue
        if expected is None:
            continue
        if any(k in code for k in _NO_DIFF_CODE) or _is_nondeterministic(code):
            continue
        got = _norm(buf.getvalue())
        want = _norm(expected)
        if not got:
            # The block produced no stdout, so the adjacent fenced block is an
            # illustration (a REPL-style repr, a schematic, an input echo), not
            # captured output -- nothing to diff against.
            continue
        # The shown output must appear in the actual stdout as an ordered
        # subsequence. This lets a doc display an *elided* view (a few lines of a
        # long tree) while still failing if any shown line -- e.g. a number --
        # drifts, since a changed line no longer matches.
        missing = _subseq_missing(want, got)
        if missing:
            diff = "\n".join(difflib.unified_diff(
                want, got, fromfile=f"block{idx} expected (in doc)",
                tofile=f"block{idx} actual (printed)", lineterm=""))
            mismatches.append(
                f"block {idx} output drift (shown line not in actual: "
                f"{missing!r}):\n{diff}")
        else:
            diffed += 1

    assert not mismatches, (
        f"{doc.relative_to(ROOT)}: {len(mismatches)} doc block(s) drifted "
        f"({diffed} matched)\n\n" + "\n\n".join(mismatches))
