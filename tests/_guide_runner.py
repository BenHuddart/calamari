"""Execute a Markdown user guide and compare every marked output with what the code prints.

The user guides in ``docs/`` promise that every example runs as written, in order, from an
empty directory, and that every output shown is the real one. Nothing enforced that, and the
guides rotted silently: examples that raised, outputs that had drifted from their own code.
This module is the check. ``tests/test_guides.py`` runs every guide through it, one fresh
interpreter per guide, so CI's ordinary ``pytest tests/`` job fails the moment a guide stops
telling the truth.

The rules, which are the whole contract between a guide's author and this check:

Shared snippets
    Option-free whole-file MyST includes are expanded before execution. Included
    setup/output markers retain their meaning; diagnostic line numbers refer to
    the expanded text. Include cycles and unsupported options are errors.

Which blocks run
    Every fenced block whose info string is ``python`` runs, in document order, in one
    namespace per guide (so a later block sees an earlier block's names, as a reader's
    session would), from a fresh empty working directory, under matplotlib's Agg backend.
    Figures are closed after every block, so memory stays bounded however many plots a guide
    draws; a block that needs a figure made earlier must hold on to it (``ax = ...``), which
    the guides do anyway. A block that raises fails the check -- unless its marked output
    shows that exception (below).

Setup that lives at the end of a guide
    A guide may keep the code that writes its input files at the end, where it reads best.
    Mark each such block with ``<!-- run-first -->`` on the line immediately before its
    opening fence: marked blocks run before all others (in document order among
    themselves), then the rest run in document order. No heading text is interpreted.

Marking literal output
    A plain fenced block that is the literal output of the nearest preceding ``python``
    block is marked with ``<!-- output -->`` on the line immediately before its opening
    fence; prose may sit between the two blocks. The marker renders as nothing: GitHub hides
    HTML comments, and MyST passes them into the site's HTML as comments. Plain blocks
    *without* the marker are never compared: annotated summaries (a table of selected
    values, a file listing with commentary) and illustrative text not produced by the
    preceding block stay free-form. So marking is a claim -- "this is exactly what that code
    prints" -- and the check holds the guide to it. A python block has at most one marked
    output; an HTML comment in front of a fence that is not one of the two markers is an
    error, so a misspelt marker cannot silently disable a comparison.

What counts as a block's output (notebook semantics)
    Everything the block writes to stdout while it runs, then -- if the block's final
    top-level statement is an expression -- ``repr()`` of its value, as Jupyter shows it.
    The value is not shown when it is ``None`` or a matplotlib object (an ``Axes``, a
    ``Legend``, or a list/tuple/array holding only matplotlib objects): those reprs carry
    memory addresses, and nobody documents them. If the block raises, its output is whatever
    it wrote before the raise followed by ``f"{type(exc).__name__}: {exc}"``.

Warnings
    Every warning a block lets escape is written into its output, interleaved in the order
    it was emitted, as a ``f"{category.__name__}: {message}"`` line. The filter is
    ``"always"``, so repeats are shown and so are categories Python's default filters hide
    (``DeprecationWarning``, ``ResourceWarning``): the result does not depend on which
    warnings an earlier block happened to trigger first. A block with a marked output must
    therefore show its warnings -- a new warning is drift, like a changed number. A warning
    escaping a block with *no* marked output fails nothing, but is returned as a note, which
    the test reports in pytest's warnings summary.

Comparison
    The actual output and the marked block are compared after collapsing every run of
    whitespace to one space and stripping both ends. That tolerates hand-wrapping a long
    error message across lines and nothing else: any changed character fails. A value that
    is genuinely platform-sensitive in its last digits is handled in the guide (print it
    rounded), never by loosening this comparison.

Indented fences
    MyST (like CommonMark) allows a fence inside a list item. Following CommonMark, a fence
    indented by N spaces has up to N spaces of indentation removed from each content line,
    so such a block runs exactly as it renders.

Run as a script (``python _guide_runner.py GUIDE.md RESULT.json``), it executes the guide in
the current directory and writes the report as JSON; the test runs it that way, in a
subprocess, so no guide can leak state (a registered loader, a journal style, ``rcParams``)
into the rest of the suite, or benefit from state the suite left behind.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import difflib
import io
import json
import linecache
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

#: The two markers. Each is an HTML comment alone on the line immediately before a fence.
OUTPUT_MARKER = "output"
RUN_FIRST_MARKER = "run-first"
_MARKERS = (OUTPUT_MARKER, RUN_FIRST_MARKER)

# An opening fence: three or more backticks or tildes, optionally indented (a fence inside a
# list item). A backtick fence's info string may not contain a backtick (CommonMark 4.5).
_FENCE_OPEN = re.compile(r"^(?P<indent> *)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
# An HTML comment alone on a line: the only place a marker may appear.
_COMMENT = re.compile(r"^\s*<!--\s*(?P<body>.*?)\s*-->\s*$")


class GuideSyntaxError(ValueError):
    """A guide's markers are malformed (misspelt, misplaced, or an unclosed fence)."""


@dataclass
class Fence:
    """One fenced block of a guide, as the reader sees it rendered."""

    line: int  # 1-based line of the opening fence in the .md file
    info: str  # the info string, e.g. "python" or ""
    body: str  # content with the fence's own indentation removed
    marker: str | None  # "output", "run-first", or None

    @property
    def is_python(self) -> bool:
        return self.info.split()[:1] == ["python"]


@dataclass
class BlockResult:
    """What one python block did, and what the guide claims it did."""

    index: int  # 1-based count among the guide's python blocks
    line: int
    run_first: bool
    actual: str
    expected: str | None = None
    expected_line: int | None = None
    error: str | None = None  # formatted traceback if the block raised
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class GuideReport:
    """The outcome of running one guide: per-block results, failures, and notes."""

    guide: str
    blocks: list[BlockResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def compared(self) -> int:
        return sum(b.expected is not None for b in self.blocks)


def read_guide(path: Path, _stack: tuple[Path, ...] = ()) -> str:
    """Expand whole-file MyST includes so shared snippets are executed too.

    Only option-free, three-backtick include directives are supported. Refuse
    other include forms rather than silently checking different code from Sphinx.
    Error line numbers subsequently refer to the expanded guide.
    """
    path = path.resolve()
    if path in _stack:
        raise GuideSyntaxError(f"cyclic include: {path.name}")
    text = path.read_text(encoding="utf-8")
    includes = [f for f in parse_fences(text) if f.info.startswith("{include}")]
    pattern = re.compile(r"(?m)^```\{include\} ([^\n]+)\n```[ \t]*$")
    if any(f.body.strip() or f.marker for f in includes) or len(list(pattern.finditer(text))) != len(includes):
        raise GuideSyntaxError(f"{path.name}: use option-free whole-file includes")
    return pattern.sub(
        lambda m: read_guide(path.parent / m[1].strip(), (*_stack, path)), text
    )


def parse_fences(text: str) -> list[Fence]:
    """Split a Markdown document into its fenced blocks, attaching any marker to each."""
    lines = text.splitlines()
    fences: list[Fence] = []
    i = 0
    while i < len(lines):
        m = _FENCE_OPEN.match(lines[i])
        if m is None or (m["fence"][0] == "`" and "`" in m["info"]):
            i += 1
            continue
        indent, fence, info = len(m["indent"]), m["fence"], m["info"].strip()
        closing = re.compile(rf"^ *{re.escape(fence[0])}{{{len(fence)},}} *$")
        start = i
        body: list[str] = []
        i += 1
        while i < len(lines) and not closing.match(lines[i]):
            # CommonMark 4.5: up to N spaces of indentation are removed from each content
            # line, where N is the opening fence's indentation.
            line = lines[i]
            strip = len(line) - len(line.lstrip(" "))
            body.append(line[min(strip, indent):])
            i += 1
        if i == len(lines):
            raise GuideSyntaxError(f"line {start + 1}: fence {fence!r} is never closed")
        i += 1
        marker = None
        if start > 0 and (c := _COMMENT.match(lines[start - 1])) is not None:
            marker = c["body"]
            if marker not in _MARKERS:
                raise GuideSyntaxError(
                    f"line {start}: {lines[start - 1].strip()!r} directly before a fence is "
                    f"not a marker; use <!-- {OUTPUT_MARKER} --> or <!-- {RUN_FIRST_MARKER} -->, "
                    "or put a blank line between a real comment and the fence"
                )
        fences.append(Fence(start + 1, info, "\n".join(body) + "\n", marker))
    return fences


def pair_blocks(fences: list[Fence]) -> list[tuple[Fence, Fence | None]]:
    """Pair every python block with its marked output block, if it has one."""
    pairs: list[tuple[Fence, Fence | None]] = []
    for f in fences:
        if f.is_python:
            if f.marker == OUTPUT_MARKER:
                raise GuideSyntaxError(
                    f"line {f.line}: <!-- {OUTPUT_MARKER} --> marks a python block; it belongs "
                    "on the plain block showing that code's output"
                )
            pairs.append((f, None))
        elif f.marker == RUN_FIRST_MARKER:
            raise GuideSyntaxError(
                f"line {f.line}: <!-- {RUN_FIRST_MARKER} --> marks a block that is not python"
            )
        elif f.marker == OUTPUT_MARKER:
            if not pairs:
                raise GuideSyntaxError(
                    f"line {f.line}: marked output has no python block before it"
                )
            code, previous = pairs[-1]
            if previous is not None:
                raise GuideSyntaxError(
                    f"line {f.line}: a second marked output for the python block at line "
                    f"{code.line} (the first is at line {previous.line})"
                )
            pairs[-1] = (code, f)
    return pairs


def normalise(text: str) -> str:
    """Collapse every run of whitespace to one space and strip the ends."""
    return " ".join(text.split())


def _is_matplotlib(value: object) -> bool:
    """True for a matplotlib object, or a non-empty container holding only such objects."""
    if type(value).__module__.split(".")[0] == "matplotlib":
        return True
    if isinstance(value, (list, tuple)) and value:
        return all(_is_matplotlib(v) for v in value)
    # An object array, e.g. the axes grid from plt.subplots(2, 2); checked by name so this
    # module need not import numpy.
    if type(value).__name__ == "ndarray" and value.dtype.kind == "O":
        return value.size > 0 and all(_is_matplotlib(v) for v in value.flat)
    return False


def _close_figures() -> None:
    # Only if the guide imported pyplot: importing it here would itself pick a backend.
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        plt.close("all")


def _run_block(fence: Fence, filename: str, namespace: dict) -> tuple[str, str | None, list]:
    """Execute one block with notebook semantics; return (output, traceback, warnings)."""
    buffer = io.StringIO()
    emitted: list[str] = []

    def show(message, category, filename_, lineno, file=None, line=None):
        text = f"{category.__name__}: {message}"
        emitted.append(text)
        buffer.write(text + "\n")

    error = None
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = show
        with contextlib.redirect_stdout(buffer):
            try:
                tree = ast.parse(fence.body, filename)
                # Line numbers in tracebacks point into the .md file, not the block.
                ast.increment_lineno(tree, fence.line)
                last = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = ast.Expression(tree.body.pop().value)
                exec(compile(tree, filename, "exec"), namespace)
                if last is not None:
                    value = eval(compile(last, filename, "eval"), namespace)
                    if value is not None and not _is_matplotlib(value):
                        buffer.write(repr(value) + "\n")
            except BaseException as exc:  # SystemExit from a guide is a failure too
                if isinstance(exc, KeyboardInterrupt):
                    raise
                buffer.write(f"{type(exc).__name__}: {exc}\n")
                error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return buffer.getvalue(), error, emitted


def _diff(expected: str, actual: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            expected.strip("\n").splitlines(),
            actual.strip("\n").splitlines(),
            "shown in the guide",
            "produced by the code",
            lineterm="",
        )
    )


def run_guide(text: str, guide: str) -> GuideReport:
    """Run every python block of a guide, in the current directory, and compare its outputs.

    ``guide`` names the guide in messages and tracebacks. Failures are collected rather than
    raised, so one run reports every problem in the guide at once.
    """
    report = GuideReport(guide)
    start = time.perf_counter()
    try:
        pairs = pair_blocks(parse_fences(text))
    except GuideSyntaxError as exc:
        report.failures.append(f"{guide}: {exc}")
        return report
    # Tracebacks quote the guide's own lines.
    linecache.cache[guide] = (len(text), None, text.splitlines(True), guide)
    namespace: dict = {"__name__": "__main__"}
    numbered = [(n, code, out) for n, (code, out) in enumerate(pairs, start=1)]
    order = [p for p in numbered if p[1].marker == RUN_FIRST_MARKER]
    order += [p for p in numbered if p[1].marker != RUN_FIRST_MARKER]
    results: dict[int, BlockResult] = {}
    for n, code, out in order:
        tick = time.perf_counter()
        actual, error, emitted = _run_block(code, guide, namespace)
        _close_figures()
        result = BlockResult(n, code.line, code.marker == RUN_FIRST_MARKER, actual,
                             error=error, warnings=emitted, seconds=time.perf_counter() - tick)
        if out is not None:
            result.expected, result.expected_line = out.body, out.line
        results[n] = result
    report.blocks = [results[n] for n in sorted(results)]

    for b in report.blocks:
        where = f"{guide}:{b.line} (python block {b.index})"
        if b.expected is not None and normalise(b.expected) != normalise(b.actual):
            message = (f"{where}: output differs from the marked block at line "
                       f"{b.expected_line}\n{_diff(b.expected, b.actual)}")
            if b.error is not None:
                message += f"\n\nThe block raised:\n{b.error}"
            report.failures.append(message)
        elif b.expected is None and b.error is not None:
            report.failures.append(f"{where}: raised, and no marked output shows it\n{b.error}")
        elif b.expected is None:
            report.notes.extend(f"{where}: {w}" for w in b.warnings)
    report.seconds = time.perf_counter() - start
    return report


def main(argv: list[str]) -> int:
    guide_path, result_path = Path(argv[0]), Path(argv[1])
    report = run_guide(read_guide(guide_path), guide_path.name)
    result_path.write_text(json.dumps(dataclasses.asdict(report)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
