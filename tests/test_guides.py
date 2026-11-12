"""Every user guide in ``docs/`` runs as written and prints what it says it prints.

The guides promise that every example runs, in order, from an empty directory, and that every
output shown is real. Before this check nothing held them to it, and they rotted without
anyone noticing: most of one guide's examples raised, and every example of another. The rules
-- which blocks run, how literal output is marked (``<!-- output -->``), how end-of-guide setup
is run first (``<!-- run-first -->``), and exactly what is compared -- are in the docstring of
``tests/_guide_runner.py``.

Each guide runs in its own interpreter (``python _guide_runner.py``) from a fresh empty
directory. In-process, a guide would inherit whatever the rest of the suite left behind -- a
registered loader, a journal style, ``rcParams`` -- and could pass on the strength of it, or
leave its own registrations behind to break a later test. A reader's session starts clean;
so does this one.

Maintained Markdown pages under ``docs/`` are checked recursively, excluding generated
trees. The basic guide receives invented file fixtures before execution.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from _guide_runner import (
    GuideSyntaxError,
    normalise,
    pair_blocks,
    parse_fences,
    read_guide,
    run_guide,
)

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
RUNNER = Path(__file__).resolve().parent / "_guide_runner.py"

#: Seconds allowed for one guide. The slowest takes a few seconds; the limit only stops a
#: hung guide from hanging CI.
_TIMEOUT_S = 600


def _guides(root: Path = DOCS) -> list[Path]:
    """Find maintained pages outside generated API/build trees."""
    generated = {"_build", "api"}
    return sorted(
        p for p in root.rglob("*.md")
        if p.relative_to(root).parts[0] not in generated
    )


class GuideWarning(UserWarning):
    """A warning escaped a guide block whose output the guide does not show."""


@pytest.mark.parametrize("guide", _guides(), ids=lambda p: p.relative_to(DOCS).as_posix())
def test_guide(guide: Path, tmp_path: Path):
    """Run one guide end to end; fail on any raising block or any drifted marked output."""
    try:
        text = read_guide(guide)
        fences = parse_fences(text)
        pair_blocks(fences)
    except GuideSyntaxError as exc:
        pytest.fail(f"{guide.name}: {exc}")
    n_python = sum(f.is_python for f in fences)
    if n_python == 0:
        return  # nothing to run; the markers (if any) were checked above

    work = tmp_path / "work"
    work.mkdir()
    if guide.name == "USER_GUIDE.md":
        from _workflow import write_inputs

        write_inputs(work)
    result = tmp_path / "result.json"
    # Agg: headless, and set before the guide can import pyplot. A fixed hash seed keeps the
    # iteration order of any set a guide prints the same from run to run.
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"}
    proc = subprocess.run(
        [sys.executable, str(RUNNER), str(guide), str(result)],
        cwd=work, env=env, capture_output=True, text=True, timeout=_TIMEOUT_S,
    )
    if proc.returncode != 0 or not result.exists():
        pytest.fail(f"{guide.name}: the runner itself failed (exit {proc.returncode})\n"
                    f"{proc.stderr[-4000:]}")
    report = json.loads(result.read_text(encoding="utf-8"))
    assert len(report["blocks"]) == n_python, "the runner skipped python blocks"
    for note in report["notes"]:
        warnings.warn(note, GuideWarning, stacklevel=1)
    if report["failures"]:
        pytest.fail(f"{len(report['failures'])} problem(s) in {guide.name}:\n\n"
                    + "\n\n".join(report["failures"]), pytrace=False)


# ---------------------------------------------------------------------------
# The runner itself: each rule on a tiny synthetic guide, so a broken comparator fails here
# rather than passing every real guide vacuously.
# ---------------------------------------------------------------------------

FENCE = "```"


def _md(*parts: str) -> str:
    return "\n".join(parts) + "\n"


def _py(code: str, indent: str = "") -> str:
    lines = [FENCE + "python", *code.splitlines(), FENCE]
    return "\n".join(indent + ln if ln else ln for ln in lines)


def _out(text: str, marker: bool = True) -> str:
    return ("<!-- output -->\n" if marker else "") + "\n".join([FENCE, text, FENCE])


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return lambda text: run_guide(text, "demo.md")


def test_marked_output_that_matches_passes(run):
    report = run(_md("Some prose.", _py("print(6 * 7)"), "", _out("42")))
    assert report.failures == []
    assert report.compared == 1


def test_marked_output_that_differs_fails_with_line_and_diff(run):
    text = _md("# Demo", "", _py("print(6 * 7)"), "", _out("43"))
    report = run(text)
    assert len(report.failures) == 1
    message = report.failures[0]
    # Names the guide and the python block's line in the .md file, and shows both values.
    assert message.startswith("demo.md:3 (python block 1)")
    assert "marked block at line 8" in message
    assert "-43" in message and "+42" in message


def test_comparison_collapses_whitespace_only(run):
    wrapped = _md(_py("print('a long message that the guide wraps')"), "",
                  _out("a long message\n   that the guide wraps  "))
    assert run(wrapped).failures == []
    changed = _md(_py("print('a long message')"), "", _out("a long massage"))
    assert len(run(changed).failures) == 1
    assert normalise("  a\n\tb  c \n") == "a b c"


def test_final_expression_is_shown_as_repr(run):
    # repr, not str: the quotes are part of the output.
    assert run(_md(_py("'MyRig'"), _out("'MyRig'"))).failures == []
    assert len(run(_md(_py("'MyRig'"), _out("MyRig"))).failures) == 1
    # stdout first, then the value.
    assert run(_md(_py("print('first')\n1 + 1"), _out("first\n2"))).failures == []


def test_none_and_matplotlib_values_are_not_shown(run):
    assert run(_md(_py("print('only this')\nNone"), _out("only this"))).failures == []
    code = ("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\nprint('drawn')\nax.plot([0, 1], [0, 1])")
    assert run(_md(_py(code), _out("drawn"))).failures == []


def test_only_the_final_statement_is_displayed(run):
    # A bare expression mid-block shows nothing, as in a notebook; an assignment last
    # shows nothing either.
    report = run(_md(_py("1 + 1\nx = 3"), _out("")))
    assert report.failures == []
    assert report.blocks[0].actual == ""


def test_marked_exception_is_the_expected_output(run):
    code = "print('before')\nraise ValueError('bad input')"
    assert run(_md(_py(code), _out("before\nValueError: bad input"))).failures == []
    wrong = run(_md(_py(code), _out("before\nValueError: other input")))
    assert len(wrong.failures) == 1
    assert "The block raised" in wrong.failures[0]


def test_unmarked_exception_fails(run):
    report = run(_md(_py("1 / 0")))
    assert len(report.failures) == 1
    assert "raised, and no marked output shows it" in report.failures[0]
    assert "ZeroDivisionError" in report.failures[0]


def test_unmarked_plain_block_is_not_compared(run):
    report = run(_md(_py("print(1)"), "", _out("an annotated summary, not output", marker=False)))
    assert report.failures == []
    assert report.compared == 0


def test_blocks_share_one_namespace_in_order(run):
    report = run(_md(_py("x = 2"), "", _py("print(x * 21)"), "", _out("42")))
    assert report.failures == []


def test_run_first_blocks_run_before_the_rest(run, tmp_path, monkeypatch):
    text = _md(
        _py("from pathlib import Path\nprint(Path('input.txt').read_text())"), "",
        _out("written first"), "",
        "## Generating the input", "",
        "<!-- run-first -->", _py("from pathlib import Path\n"
                                  "Path('input.txt').write_text('written first')"),
    )
    report = run(text)
    assert report.failures == []
    # Results are reported in document order, whatever order they ran in.
    assert [b.run_first for b in report.blocks] == [False, True]
    # Without the marker the same guide, in a fresh directory, fails: the file does not exist
    # when the first block reads it.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    monkeypatch.chdir(fresh)
    unmarked = run(text.replace("<!-- run-first -->\n", ""))
    assert len(unmarked.failures) == 1
    assert "FileNotFoundError" in unmarked.failures[0]


def test_indented_fence_in_a_list_item_is_dedented(run):
    text = _md(
        "- A list item whose example is part of the item:", "",
        _py("if True:\n    print('inside')", indent="  "), "",
        "  <!-- output -->", f"  {FENCE}", "  inside", f"  {FENCE}",
    )
    fences = parse_fences(text)
    assert fences[0].body == "if True:\n    print('inside')\n"
    assert run(text).failures == []


def test_warning_is_part_of_marked_output_in_emission_order(run):
    code = ("import warnings\nprint('a')\nwarnings.warn('careful')\nprint('b')\n"
            "warnings.warn('again', RuntimeWarning)")
    shown = _out("a\nUserWarning: careful\nb\nRuntimeWarning: again")
    assert run(_md(_py(code), shown)).failures == []
    # A marked output that does not show the warning is drift, like a changed number.
    assert len(run(_md(_py(code), _out("a\nb"))).failures) == 1


def test_warning_from_unmarked_block_is_a_note_not_a_failure(run):
    code = "import warnings\nwarnings.warn('careful')"
    report = run(_md(_py(code)))
    assert report.failures == []
    assert report.notes == ["demo.md:1 (python block 1): UserWarning: careful"]


def test_repeated_warnings_are_all_shown(run):
    # The "always" filter: the output does not depend on what warned earlier.
    code = "import warnings\nfor _ in range(2):\n    warnings.warn('twice')"
    first = run(_md(_py(code), _out("UserWarning: twice\nUserWarning: twice")))
    assert first.failures == []


@pytest.mark.parametrize(
    "text, message",
    [
        (_md(_py("print(1)"), "<!-- ouput -->", FENCE, "1", FENCE), "is not a marker"),
        (_md("<!-- output -->", FENCE, "1", FENCE), "no python block before it"),
        (_md(_py("print(1)"), _out("1"), _out("1")), "a second marked output"),
        (_md("<!-- output -->", _py("print(1)")), "marks a python block"),
        (_md("<!-- run-first -->", FENCE, "x", FENCE), "not python"),
        (_md(FENCE + "python", "print(1)"), "never closed"),
    ],
    ids=["misspelt", "orphan-output", "two-outputs", "output-on-python", "run-first-on-plain",
         "unclosed"],
)
def test_malformed_markers_fail_the_guide(run, text, message):
    report = run(text)
    assert len(report.failures) == 1
    assert message in report.failures[0]


def test_ordinary_comment_separated_from_a_fence_is_allowed(run):
    text = _md("<!-- an editor's note -->", "", _py("print(1)"), _out("1"))
    assert run(text).failures == []


def test_runner_script_isolates_the_guide(tmp_path):
    # The path the real guides take: a subprocess from an empty directory, report as JSON.
    guide = tmp_path / "demo.md"
    guide.write_text(_md(_py("import os\nprint(sorted(os.listdir('.')))"), _out("[]")))
    work = tmp_path / "work"
    work.mkdir()
    result = tmp_path / "result.json"
    subprocess.run([sys.executable, str(RUNNER), str(guide), str(result)], cwd=work,
                   check=True, timeout=_TIMEOUT_S)
    report = json.loads(result.read_text())
    assert report["failures"] == []
    assert report["blocks"][0]["actual"] == "[]\n"


def test_guide_discovery_includes_nested_maintained_pages(tmp_path):
    for name in ("start.md", "workflows/fit.md",
                 "api/generated.md", "_build/html/page.md"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Page\n")
    assert [p.relative_to(tmp_path).as_posix() for p in _guides(tmp_path)] == [
        "start.md", "workflows/fit.md",
    ]


def test_included_setup_and_output_are_executed(tmp_path, monkeypatch):
    snippet = tmp_path / "snippets" / "setup.md"
    snippet.parent.mkdir()
    snippet.write_text("<!-- run-first -->\n```python\nx = 4\n```\n")
    guide = tmp_path / "guide.md"
    guide.write_text(
        "```python\nprint(x + 1)\n```\n<!-- output -->\n```\n5\n```\n"
        "```{include} snippets/setup.md\n```\n"
    )
    monkeypatch.chdir(tmp_path)
    report = run_guide(read_guide(guide), "included.md")
    assert not report.failures
    assert report.compared == 1
    snippet.write_text(snippet.read_text().replace("x = 4", "x = 7"))
    assert run_guide(read_guide(guide), "included.md").failures


@pytest.mark.parametrize("body", [
    "```{include} guide.md\n```\n",
    "```{include} other.md\n:start-line: 2\n```\n",
])
def test_includes_refuse_cycles_and_unsupported_options(tmp_path, body):
    guide = tmp_path / "guide.md"
    guide.write_text(body)
    with pytest.raises(GuideSyntaxError):
        read_guide(guide)
