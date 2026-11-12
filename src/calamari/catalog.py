"""Discovery and cataloguing of magnetometry measurement files.

Provides :func:`find_files` to walk a directory tree and identify files
recognised by any registered :mod:`calamari.io` loader, :func:`summarise`
to build a one-row inventory summary of a single file, :func:`logbook_line`
/ :func:`logbook` to render those summaries as human-readable logbook
entries, and :func:`build_catalog` to assemble the whole tree's inventory
as a :class:`Catalog` -- the rows plus the root they resolve against, with
chainable selection (``where``/``of_type``/``at``/``drop``/``sort_by``), a
repr that is the logbook, and iteration as file paths so the plot helpers
consume one directly. :func:`deduplicated` and
:func:`select_representative` (also exposed as ``Catalog`` methods) reduce
an inventory by removing verified byte-identical cross-group copies and
choosing one representative sweep per nominal setpoint respectively.

This module makes no assumption about *why* a tree is organised into
subfolders -- a real project typically mirrors several data sources
(a collaborator's beamtime export, a working copy of the same data, runs
grouped by sample revision, ...) under one root, and wants an inventory
that reflects that structure. Rather than hardcoding folder-name markers,
grouping and duplicate detection are exposed as caller-supplied
parameters (``group_fn`` and ``dedup_between``) -- see :func:`build_catalog`.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from calamari.constants import OE_PER_TESLA
from calamari.exceptions import CatalogSelectionError, InputKeyError, InputTypeError
from calamari.io import load, sniff_format
from calamari.measurement import SCAN_H_TOL_OE, SCAN_T_TOL_K

#: Rows shown by :meth:`Catalog.__repr__` before truncating. Large enough
#: that a real campaign directory usually shows whole, small enough that an
#: accidental repr of a thousand-file tree stays readable in a notebook;
#: ``.logbook()`` always renders every row.
_CATALOG_REPR_MAX_ROWS = 30

#: Read comparisons in 64 KiB blocks to bound memory independently of export size.
_DUP_BLOCK_BYTES = 64 * 1024


def _default_group_fn(rel_path: Path) -> str:
    """Default ``group_fn``: the first path component of a file's path
    relative to the catalog root (e.g. its top-level subfolder name, or
    the filename itself for a file directly under the root).

    Deterministic and instrument-/project-agnostic. Pass a custom
    ``group_fn`` to :func:`build_catalog` (or :func:`summarise`) for
    project-specific grouping -- e.g. by data source, collaborator, or
    dataset revision -- by matching path fragments, a lookup table, or
    any other rule over the relative path.
    """
    return rel_path.parts[0] if rel_path.parts else str(rel_path)


def _is_relative_to(path: Path, root: Path) -> bool:
    """Backport-safe check for ``Path.is_relative_to`` (kept defensive
    against odd path combinations, e.g. on Windows or across drives)."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def find_files(
    root: Path | str,
    *,
    skip_dirs: tuple[str, ...] = (".git",),
    path_filter: Callable[[Path], bool] | None = None,
) -> list[Path]:
    """Walk ``root`` recursively and return files recognised by a
    registered :mod:`calamari.io` loader, in sorted path order.

    The signature check (:func:`~calamari.io.sniff_format`) is cheap
    (reads only the first ~40 lines) so this is safe to run over large
    trees; it also never raises on an unreadable file (permission or
    encoding errors are treated as "not a match" by the loader registry).

    Open the root before walking it, so a missing or unreadable root raises
    the operating system's :class:`OSError` rather than resembling an empty
    directory. Filesystem exceptions retain their original types (IO-5).

    A *subdirectory* that cannot be listed does not abort the walk: one
    locked folder deep in a shared tree should not cost the inventory of
    everything else. It is not skipped silently either -- files inside it
    are missing from the result, and a catalog quietly short of a whole
    run folder is exactly the failure nobody notices -- so a single
    :class:`UserWarning` after the walk names every such directory and the
    ``skip_dirs=`` that leaves it out knowingly.

    Parameters
    ----------
    root : Path or str
        Root directory to scan. Relative paths resolve against the current
        working directory, as for any file operation.
    skip_dirs : tuple[str, ...], optional
        Directory names pruned from the walk wherever encountered (not
        just at the top level) -- e.g. version-control metadata
        directories that may themselves contain look-alike files.
        Defaults to ``('.git',)``. A pruned directory is never listed, so
        naming an unreadable one here also silences its warning.
    path_filter : callable, optional
        Predicate receiving each file's path relative to ``root``. False
        excludes the file before any content probe or read. Use only path
        information for cheap filtering; a filename rule cannot establish
        scientific file role. Directories are still listed (use
        ``skip_dirs`` to prune them). Exceptions from the predicate
        propagate to the caller. Default None: probe every candidate.

    Returns
    -------
    list[Path]
        Paths of files whose content matches a registered loader's
        signature, sorted by path component (each component compared as a
        plain, case-sensitive string). The order is a property of the
        names alone, so it is the same on every machine: :func:`os.walk`
        yields directory-listing order, which the filesystem chooses and
        which is not alphabetical even on one machine, and a result indexed
        by position -- ``paths[0]`` -- would otherwise mean a different
        file on each. Comparing component by component keeps each
        directory's contents together (``run/b.dat`` sorts before
        ``run-2/a.dat``, although ``'-' < '/'`` in a flat string sort). The
        sort is deliberately not "natural": ``chi_10K.dat`` sorts before
        ``chi_2K.dat``, because reading physics into a filename is a guess.
        Sort the measurements themselves (the ``*_family`` plot helpers
        order curves by their measured fixed value) when order matters.

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist.
    NotADirectoryError
        If ``root`` exists but is not a directory (a single file is not a
        tree to search -- pass it to :func:`calamari.io.load` directly).
    PermissionError
        If ``root`` exists but cannot be listed.

    Warns
    -----
    UserWarning
        If one or more subdirectories below ``root`` cannot be listed; they
        are left out and the walk continues.
    """
    root = Path(root)
    # os.walk reports a failure to list its top directory only through
    # onerror, and ignores it by default. Opening the root here lets that
    # failure propagate as the OS's own error -- FileNotFoundError,
    # NotADirectoryError or PermissionError -- with the path in its message.
    os.scandir(root).close()

    skip = set(skip_dirs)
    matches: list[Path] = []
    unreadable: list[OSError] = []

    for dirpath, dirnames, filenames in os.walk(root, onerror=unreadable.append):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for filename in filenames:
            candidate = Path(dirpath) / filename
            if path_filter is not None and not path_filter(candidate.relative_to(root)):
                continue
            if sniff_format(candidate) is not None:
                matches.append(candidate)

    if unreadable:
        _warn_unreadable_dirs(unreadable, skip_dirs)
    # Path.parts rather than Path ordering: on Windows, Path comparison is
    # case-insensitive, and the order must not depend on the platform.
    return sorted(matches, key=lambda path: path.parts)


def _warn_unreadable_dirs(errors: list[OSError], skip_dirs: tuple[str, ...]) -> None:
    """The warning :func:`find_files` gives for subdirectories it could not list.

    One warning for the whole walk, listing every directory: a tree with a
    dozen locked folders should not produce a dozen warnings, and the
    ``skip_dirs=`` fix is easier to write with all the names in view. The
    suggested ``skip_dirs`` keeps the caller's own entries, since passing
    the keyword replaces the default rather than adding to it.
    """
    lines = [f"  {error.filename}: {error.strerror or error}" for error in errors]
    names = sorted({Path(error.filename).name for error in errors if error.filename})
    suggested = tuple(skip_dirs) + tuple(name for name in names if name not in skip_dirs)
    n = len(errors)
    warnings.warn(
        f"find_files: {n} director{'ies' if n > 1 else 'y'} could not be listed, so any "
        "measurement files inside are missing from the result:\n" + "\n".join(lines) + "\n"
        f"Fix the permissions to include them, or pass skip_dirs={suggested!r} to leave "
        "them out without this warning.",
        stacklevel=_external_stacklevel(),
    )


def _external_stacklevel() -> int:
    """``stacklevel`` that points a warning at the first frame outside this module.

    :func:`find_files` is called both directly and from :func:`build_catalog`
    and :func:`logbook`, so a fixed ``stacklevel`` would blame this module's
    own line on the indirect routes. Counting frames until the stack leaves
    ``calamari.catalog`` names the caller's line either way. It is the same
    rule as ``calamari.plot``'s helper of this name, which cannot be reused
    here because it counts frames of *its own* module.
    """
    frame = sys._getframe(1)  # the function about to warn
    level = 1
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


def summarise(
    path: Path | str,
    root: Path | str | None = None,
    *,
    group_fn: Callable[[Path], str] | None = None,
) -> dict:
    """Build a one-row inventory summary dict for a single measurement file.

    Parameters
    ----------
    path : Path or str
        Path to the measurement file to summarise.
    root : Path or str, optional
        Root directory used to compute ``relative_path`` (and, if
        ``group_fn`` is given, the value passed to it). Defaults to
        ``path``'s parent if not given.
    group_fn : callable, optional
        A function mapping the file's path relative to ``root`` (or the
        absolute path, if ``path`` is not under ``root``) to a group
        label string. If not given, no ``'group'`` key is included in
        the returned dict at all -- this is the standalone-call default;
        :func:`build_catalog` always supplies one (see there) so that its
        output DataFrame is consistently groupable/sortable.

    Returns
    -------
    dict
        Summary row with keys: ``relative_path``, ``group`` (only if
        ``group_fn`` was given), ``filename``, ``data_format``,
        ``measurement_type``, ``file_role``, ``sample_name``,
        ``sample_mass_mg``, ``date``, ``n_points``, ``n_measurements``,
        ``T_min``, ``T_max``, ``H_min_Oe``, ``H_max_Oe``, ``H_min_T``,
        ``H_max_T``, ``ac_frequencies_Hz``, ``scan_kind``,
        ``comment``.

    Raises
    ------
    UnsupportedFileError
        If ``path`` is not a recognised magnetometry file -- propagated
        from :func:`calamari.io.load` (IO-5). A caller summarising an
        arbitrary path should gate on
        :func:`calamari.io.is_supported_file` first. :func:`build_catalog`
        does not need to: it only ever passes paths that
        :func:`find_files` already vetted, and catches any residual
        per-file failure into ``df.attrs['parse_errors']``.
    OSError
        If ``path`` cannot be read.
    """
    path = Path(path)
    root = Path(root) if root is not None else path.parent

    measurement = load(path)
    rel_path = path.relative_to(root) if _is_relative_to(path, root) else path

    row: dict = {"relative_path": str(rel_path)}
    if group_fn is not None:
        row["group"] = group_fn(rel_path)
    row["filename"] = path.name
    # Everything else is Measurement.summary() -- the per-file half of a
    # catalog row. Building on it rather than re-deriving keeps the two from
    # drifting, and means scan_kind, the T/H spans and the AC-frequency
    # formatting all have exactly one home.
    row.update(measurement.summary())
    return row


def _format_span(
    lo: float | None, hi: float | None, symbol: str, unit: str, tol: float
) -> str:
    """Render a quantity's span as logbook text: held, swept, or absent.

    A held quantity reads ``'T = 5 K'`` and a swept one ``'2 < T < 40 K'``,
    so the same cell is honest for an isotherm and for a scan without the
    caller having to branch on ``scan_kind``. An absent quantity is an
    empty string -- a blank cell keeps the row, where a placeholder number
    would invent one.

    "Held" means within ``tol`` -- the same tolerance
    :meth:`~calamari.measurement.Measurement.classify_scan` uses -- never
    exact equality. A held value displays as
    the midpoint; swept bounds are rounded to ~1/100 of the span, so the
    0.1 Oe of noise on a nominal-zero endpoint cannot print as
    ``1.2e-05 T``. This rounds only the *display*: the untouched numbers
    stay in ``summary()`` and the data.
    """
    if lo is None or hi is None:
        return ""
    if hi - lo <= tol:
        return f"{symbol} = {0.5 * (lo + hi):.4g} {unit}"
    scale = 10.0 ** (math.floor(math.log10(hi - lo)) - 2)
    return f"{round(lo / scale) * scale:g} < {symbol} < {round(hi / scale) * scale:g} {unit}"


def _none_if_missing(value):
    """Collapse the spellings of "absent" a summary value can arrive in.

    A :meth:`~calamari.measurement.Measurement.summary` dict reports an
    absent quantity as ``None``, but the same value read back out of a
    catalog DataFrame arrives as ``NaN`` -- pandas' storage for a missing
    float -- or, from a nullable-dtype frame, as ``pd.NA``. The logbook
    renderer accepts rows from both sources, so every spelling must read
    as "absent" rather than printing as a value; ``pd.isna`` is the one
    check that recognises them all.
    """
    return None if pd.isna(value) else value


def _logbook_cells_from_row(row, name: str) -> list[str]:
    """The logbook column strings for one summary row (a dict or frame row).

    The frame-facing core of the logbook: everything it reads is a
    :meth:`~calamari.measurement.Measurement.summary` key, so it renders a
    catalog row **without touching the file** -- which is what lets
    :class:`Catalog`'s repr describe hundreds of files instantly (and
    still describe them after the tree has moved).
    """
    mass = _none_if_missing(row["sample_mass_mg"])
    frequencies = _none_if_missing(row["ac_frequencies_Hz"])
    return [
        name,
        _none_if_missing(row["date"]) or "",
        _none_if_missing(row["sample_name"]) or "",
        "" if mass is None else f"{mass:g} mg",
        row["measurement_type"],
        f"{frequencies} Hz" if frequencies else "",
        row["scan_kind"],
        _format_span(
            _none_if_missing(row["T_min"]), _none_if_missing(row["T_max"]),
            "T", "K", SCAN_T_TOL_K,
        ),
        # The H cell displays Tesla, so the Oe tolerance converts (10 Oe = 1 mT).
        _format_span(
            _none_if_missing(row["H_min_T"]), _none_if_missing(row["H_max_T"]),
            "H", "T", SCAN_H_TOL_OE / OE_PER_TESLA,
        ),
    ]


def _logbook_cells(measurement, *, root: Path | str | None = None) -> list[str]:
    """The column strings :func:`logbook_line` joins and :func:`logbook` aligns."""
    path = measurement.path
    if root is not None and _is_relative_to(path, Path(root)):
        name = str(path.relative_to(Path(root)))
    else:
        name = path.name
    return _logbook_cells_from_row(measurement.summary(), name)


def _align_logbook_rows(rows: list[list[str]], delimiter: str | None) -> list[str]:
    """Join cell rows into logbook lines: delimiter-joined, or space-padded
    into aligned columns (the default), with trailing padding stripped."""
    if delimiter is not None:
        return [delimiter.join(cells) for cells in rows]
    if not rows:
        return []
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    ]


def logbook_line(measurement, *, root: Path | str | None = None) -> str:
    """One tab-separated logbook entry for a loaded measurement.

    The line a lab logbook wants for "what is this file?": name, date,
    sample, mass, measurement type, AC frequencies, scan kind, and the
    temperature and field windows -- each as a held value (``H = 0.1 T``)
    or a span (``2 < T < 40 K``) as the data dictates. "Held" is judged
    with the same tolerances as
    :meth:`~calamari.measurement.Measurement.classify_scan` (0.5 K,
    10 Oe), so a setpoint recorded with real instrument jitter reads
    ``T = 9 K``, not a millikelvin sweep, and the two cells can never
    contradict the ``scan_kind`` cell between them.

    Every value comes from :meth:`~calamari.measurement.Measurement.summary`
    rather than the raw accessors, which buys three things at once: spans
    exclude housekeeping rows (a trailing locate scan cannot stretch an
    isotherm's apparent temperature range); a quantity the file lacks is a
    blank cell rather than a crash (a DC file simply has no AC frequency);
    and a file an accessor would *raise* on -- an unrecognised unit, an
    ambiguous moment -- still gets a line, with the unanswerable cells
    blank. A logbook must describe the problem files too; they are the
    ones you go looking for.

    The AC cell lists the distinct frequencies (``'97; 997 Hz'``), never an
    average: the mean of a two-frequency interleave describes neither
    measurement.

    Parameters
    ----------
    measurement : Measurement
        A loaded measurement (or any subclass, e.g. an
        :class:`~calamari.repeats.AggregatedMeasurement`).
    root : Path or str, optional
        If given and the file lies under it, the name cell is the path
        relative to ``root`` (so subfolder structure survives into the
        logbook); otherwise just the filename.

    Returns
    -------
    str
        Tab-separated cells, in the column order above. Empty cells stay
        as empty fields, so the column structure is stable across a mixed
        directory and the output pastes into a spreadsheet.
    """
    return "\t".join(_logbook_cells(measurement, root=root))


def logbook(
    root: Path | str,
    *,
    paths: Iterable[Path | str] | None = None,
    delimiter: str | None = None,
) -> list[str]:
    """Logbook lines for every recognised file under ``root``, sorted by date.

    The one-call form of :func:`logbook_line` over a directory tree::

        print("\\n".join(calamari.logbook(data_dir)))

    Files are discovered with :func:`find_files` (by content, never by
    extension, so stray non-measurement ``.dat`` files are skipped) and
    sorted by date then name -- the order a paper logbook would be in. A
    file with no readable date sorts first rather than being dropped.

    The walk is only a discovery convenience: when a catalog DataFrame has
    already done the discovering -- and been filtered down to the rows that
    matter -- re-walking the tree would silently resurrect every file the
    filtering dropped. Pass the surviving ``relative_path`` values as
    ``paths`` instead, and the logbook renders exactly those files, keeping
    the column alignment a per-file :func:`logbook_line` loop cannot::

        kept = df[df["filename"] != "aborted.dat"]
        print("\\n".join(calamari.logbook(root, paths=kept["relative_path"])))

    Parameters
    ----------
    root : Path or str
        Directory tree to inventory (and the base that relative ``paths``
        entries and the name cell are resolved against).
    paths : iterable of Path or str, or a single Path or str, optional
        Exactly which files to render, instead of walking ``root``.
        Relative entries are resolved against ``root`` (absolute ones are
        used as-is), so a catalog's ``relative_path`` column can be passed
        directly. A single bare path renders that one file -- it is not
        iterated as characters. Unlike the walk, an entry that is not a recognised
        measurement file raises (:class:`~calamari.exceptions.UnsupportedFileError`)
        rather than being skipped: an explicit list is a claim about what
        should be there, and a silently missing line would hide its failure.
    delimiter : str, optional
        ``None`` (default) pads the columns with spaces so the logbook
        reads aligned in a terminal or a text file. Pass ``'\\t'`` for
        tab-separated lines to paste into a spreadsheet.

    Returns
    -------
    list[str]
        One line per recognised file. Empty when the tree holds none.

    Raises
    ------
    OSError
        If ``root`` is missing, is not a directory, or cannot be listed
        (walk mode; see :func:`find_files`), or if a file
        :func:`find_files` matched cannot be read when parsed -- e.g.
        deleted mid-walk. Files that fail the signature check never
        raise; they are simply absent, like any other unrecognised file
        (``paths`` mode raises instead, as above).
    UnsupportedFileError
        If a ``paths`` entry is not a recognised measurement file.
    """
    root = Path(root)
    if isinstance(paths, (str, Path)):
        # A bare string is itself an iterable -- of characters -- so without
        # this, logbook(root, paths="run.dat") would try to load root/'r',
        # root/'u', ... and fail confusingly. Treat it as the one path meant.
        paths = [paths]
    files = find_files(root) if paths is None else [root / Path(p) for p in paths]
    rows = [_logbook_cells(load(path), root=root) for path in files]
    rows.sort(key=lambda cells: (cells[1], cells[0]))
    return _align_logbook_rows(rows, delimiter)


def _content_key(path: Path) -> bytes:
    """Stream a SHA-256 digest to shortlist byte-identical exports.

    The digest alone never establishes equality; :func:`_same_content`
    verifies every shortlisted pair without parsing or rounding values.
    Filesystem read errors propagate rather than labelling an unread file.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_DUP_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.digest()


def _same_content(first: Path, second: Path) -> bool:
    """Verify byte equality with bounded memory and no cached filesystem verdict."""
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            block = left.read(_DUP_BLOCK_BYTES)
            if block != right.read(_DUP_BLOCK_BYTES):
                return False
            if not block:
                return True


def _detect_duplicates(
    df: pd.DataFrame, root: Path, dedup_between: tuple[str, str]
) -> pd.DataFrame:
    """Detect duplicate measurements between two catalog groups in-place.

    For every filename appearing in both groups named by ``dedup_between``,
    shortlists with :func:`_content_key` then verifies byte-for-byte equality
    to classify the pair as identical or merely name-colliding.
    Only ``file_role == 'summary'`` rows
    participate (companion raw/diagnostic files are not cross-referenced).

    ``dedup_between = (redundant_group, canonical_group)``: the first
    group is treated as the potentially-redundant one, the second as
    preferred/canonical when content matches -- generalising the common
    case of one data source being a working copy or superset of another.

    Adds three columns to ``df``:

    - ``is_duplicate`` (bool): True for the redundant-group copy of a
      pair verified identical to a canonical-group copy.
    - ``canonical_path`` (str): the ``relative_path`` of the copy chosen
      as canonical (the canonical-group copy for identical pairs; the
      row's own ``relative_path`` otherwise).
    - ``dup_status`` (str): ``'identical'``, ``'name_collision_differs'``,
      or ``''`` (no same-named counterpart in the other group).

    When a filename has more than one file in a group (e.g. the same
    measurement filed under two different subfolders), every same-named
    file within that group is compared against every same-named file in
    the other group. Each row is ``identical`` if it has any byte-identical
    counterpart; unmatched rows are ``name_collision_differs``, including
    in mixed cases with other verified copies of the same name. Formatting,
    line endings and all header metadata participate: a different export is
    retained even if it parses to equivalent values. Files must remain
    unchanged while the catalog is built; flags describe that comparison,
    not a later revalidation when calling ``deduplicated()``.

    Parameters
    ----------
    df : pandas.DataFrame
        Catalog rows (as produced by :func:`summarise`), already
        containing ``relative_path``, ``group``, ``filename``, and
        ``file_role``.
    root : Path
        Root directory the catalog was built from (used to resolve
        ``relative_path`` back to an absolute path for byte comparison).
    dedup_between : tuple[str, str]
        ``(redundant_group, canonical_group)`` group labels to compare --
        see :func:`build_catalog`.

    Returns
    -------
    pandas.DataFrame
        ``df`` with the three columns above added (and, as an attribute,
        ``df.attrs['name_collision_differs']``: a sorted list of
        filenames flagged as genuine name collisions).
    """
    group_a_name, group_b_name = dedup_between

    df = df.copy()
    df["is_duplicate"] = False
    df["canonical_path"] = df["relative_path"]
    df["dup_status"] = ""

    summary_mask = df["file_role"] == "summary"
    group_a = df[summary_mask & (df["group"] == group_a_name)]
    group_b = df[summary_mask & (df["group"] == group_b_name)]

    shared_names = sorted(set(group_a["filename"]) & set(group_b["filename"]))

    key_cache: dict[str, bytes] = {}

    def _key_for(rel_path: str) -> bytes:
        if rel_path not in key_cache:
            key_cache[rel_path] = _content_key(root / rel_path)
        return key_cache[rel_path]

    collisions: list[str] = []

    for name in shared_names:
        a_rows = group_a[group_a["filename"] == name]
        b_rows = group_b[group_b["filename"] == name]

        a_keys = {idx: _key_for(row["relative_path"]) for idx, row in a_rows.iterrows()}
        b_keys = {idx: _key_for(row["relative_path"]) for idx, row in b_rows.iterrows()}

        for b_idx, b_key in b_keys.items():
            matching_a = [
                a_idx for a_idx, a_key in a_keys.items()
                if a_key == b_key and _same_content(
                    root / df.at[a_idx, "relative_path"],
                    root / df.at[b_idx, "relative_path"],
                )
            ]
            if matching_a:
                canonical_path = df.at[b_idx, "relative_path"]
                for a_idx in matching_a:
                    df.at[a_idx, "is_duplicate"] = True
                    df.at[a_idx, "canonical_path"] = canonical_path
                    df.at[a_idx, "dup_status"] = "identical"
                df.at[b_idx, "dup_status"] = "identical"

        unmatched = [
            idx for idx in list(a_keys) + list(b_keys) if df.at[idx, "dup_status"] == ""
        ]
        if unmatched:
            collisions.append(name)
            for idx in unmatched:
                df.at[idx, "dup_status"] = "name_collision_differs"

    df.attrs["name_collision_differs"] = collisions
    return df


def deduplicated(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Return a de-duplicated view of a catalog DataFrame.

    Drops rows flagged ``is_duplicate == True`` (i.e. the redundant-group
    copy verified byte-identical to a canonical-group copy when the catalog
    was built -- see :func:`build_catalog`). No files are rechecked here.
    Canonical-group entries and copies outside the requested comparison
    remain; this does not merge all exports of a physical measurement.

    Rows flagged ``dup_status == 'name_collision_differs'`` (same
    filename in both groups but genuinely different content) are NOT
    dropped. Different bytes may mean different measurements, metadata or
    formatting; all are retained conservatively.

    Parameters
    ----------
    catalog_df : pandas.DataFrame
        A catalog DataFrame produced by :func:`build_catalog` (must
        contain an ``is_duplicate`` column).

    Returns
    -------
    pandas.DataFrame
        A copy of ``catalog_df`` containing only rows with
        ``is_duplicate == False``, with the index reset.

    Raises
    ------
    KeyError
        If ``catalog_df`` has no ``is_duplicate`` column (i.e. it was
        not produced by :func:`build_catalog`).
    """
    if "is_duplicate" not in catalog_df.columns:
        raise InputKeyError(
            "catalog_df has no 'is_duplicate' column; pass a DataFrame "
            "produced by calamari.catalog.build_catalog()"
        )
    return catalog_df.loc[~catalog_df["is_duplicate"]].reset_index(drop=True)


def build_catalog(
    root: Path | str,
    *,
    group_fn: Callable[[Path], str] | None = None,
    dedup_between: tuple[str, str] | None = None,
    path_filter: Callable[[Path], bool] | None = None,
) -> Catalog:
    """Build a :class:`Catalog` -- the inventory of every recognised
    measurement file under ``root``, plus the root it resolves against.

    Parameters
    ----------
    root : Path or str
        Root directory to scan.
    group_fn : callable, optional
        Maps a file's path (relative to ``root``) to a group label
        string -- see :func:`summarise`. Unlike a direct :func:`summarise`
        call, ``build_catalog`` always applies *some* ``group_fn``: if
        none is given, :func:`_default_group_fn` (the first relative path
        component) is used, so the returned DataFrame always has a
        ``'group'`` column to sort and (optionally) deduplicate by.
    dedup_between : tuple[str, str], optional
        ``(redundant_group, canonical_group)`` group labels (as produced
        by ``group_fn``) to cross-check same-named summary files for byte
        equality (including headers and formatting) -- see
        :func:`_detect_duplicates`. Default None: no duplicate detection
        is attempted, and every row gets ``is_duplicate=False``,
        ``dup_status=''`` (the columns are still present, for a
        consistent schema regardless of whether dedup was requested).
    path_filter : callable, optional
        Predicate over root-relative file paths, passed to :func:`find_files`
        before content probing. Excluded files are never opened by calamari,
        including during duplicate detection. Included files still undergo
        normal content recognition and metadata classification. For an
        explicit selection, use e.g. ``lambda path: path in selected`` with
        a set of root-relative Paths. Default None: include every candidate.
        Predicate exceptions propagate; they are not file parse errors.

    Returns
    -------
    Catalog
        One row per recognised file (see :func:`summarise` for columns,
        plus ``is_duplicate``, ``canonical_path``, and ``dup_status`` --
        see :func:`_detect_duplicates`), sorted by ``group`` then
        ``filename`` then ``relative_path``, wrapped with ``root`` as a
        :class:`Catalog` (the plain frame is its ``.df``). Files that
        fail to parse are skipped and reported on ``.parse_errors`` (a
        list of ``(path, error_message)`` tuples); filenames genuinely
        colliding between the two ``dedup_between`` groups (same name,
        different content) are listed on ``.name_collision_differs``; the
        total number of included, recognised files (before parse failures) is
        ``.n_scanned``. The same three values are mirrored on
        ``.df.attrs`` under those names.

    Raises
    ------
    OSError
        If ``root`` is missing (:class:`FileNotFoundError`), is not a
        directory (:class:`NotADirectoryError`), or cannot be listed
        (:class:`PermissionError`) -- propagated from :func:`find_files`,
        so a mistyped root is never reported as an empty catalog.
    """
    root = Path(root)
    effective_group_fn = group_fn if group_fn is not None else _default_group_fn
    files = find_files(root, path_filter=path_filter)

    rows = []
    parse_errors: list[tuple[str, str]] = []

    for file_path in files:
        try:
            rows.append(summarise(file_path, root=root, group_fn=effective_group_fn))
        except Exception as exc:  # noqa: BLE001 - want to catch & report all
            parse_errors.append((str(file_path), f"{type(exc).__name__}: {exc}"))

    df = pd.DataFrame(rows)
    if not df.empty:
        # relative_path is unique, so it breaks (group, filename) ties
        # deterministically -- os.walk order varies between filesystems,
        # and regenerated catalogs must be byte-identical.
        df = df.sort_values(["group", "filename", "relative_path"]).reset_index(drop=True)
        if dedup_between is not None:
            df = _detect_duplicates(df, root, dedup_between)
        else:
            df["is_duplicate"] = False
            df["canonical_path"] = df["relative_path"]
            df["dup_status"] = ""
            df.attrs["name_collision_differs"] = []
    else:
        df["is_duplicate"] = pd.Series(dtype=bool)
        df["canonical_path"] = pd.Series(dtype=str)
        df["dup_status"] = pd.Series(dtype=str)
        df.attrs["name_collision_differs"] = []
    df.attrs["parse_errors"] = parse_errors
    df.attrs["n_scanned"] = len(files)
    return Catalog(df, root)


class Catalog:
    """A directory's recognised measurement files: the inventory rows plus
    the root they resolve against.

    The class exists because a bare DataFrame kept losing its ``root``:
    every trip back from rows to files -- printing a logbook of a filtered
    inventory, loading the survivors for a family plot -- made the caller
    re-supply the root and re-join the paths by hand. A ``Catalog``
    remembers it, so the rows and the files stay one object.

    Selection methods (:meth:`where`, :meth:`of_type`, :meth:`at`,
    :meth:`drop`, :meth:`sort_by`, :meth:`select_representative`,
    :meth:`deduplicated`, and boolean-mask indexing) each return a **new**
    ``Catalog`` -- the same bargain as
    :meth:`~calamari.measurement.Measurement.restrict` -- so they chain,
    and the original is never mutated. Only what pandas is clumsy at gets
    a named method; for everything else the frame itself is one attribute
    away (``cat.df``), ``cat["col"]`` returns the column, and
    ``cat[mask]`` filters, so any pandas expression still works.

    Iterating a ``Catalog`` yields its files' **absolute paths**, which is
    what lets the plot helpers consume one directly: every family and map
    helper accepts paths and loads them itself, so
    ``plot_chi_vs_t_family(cat.of_type("AC"))`` needs no intermediate
    list. :meth:`load` is the explicit spelling of the same bridge.
    What crosses it is a **whole file**: selection works on rows of the
    inventory, one per file, so a combined VSM+AC run chosen by
    ``of_type("AC")`` still loads with its VSM rows -- and a chi plot's
    ``source='auto'`` refuses it, since it offers both an AC chi and a DC
    moment. Pass ``source='ac'`` on a tree that may hold combined runs.

    The repr **is** the logbook -- see :meth:`__repr__`.

    Parameters
    ----------
    df : pandas.DataFrame
        Inventory rows as :func:`build_catalog` produces them (one
        :func:`summarise` dict per row).
    root : Path or str
        Directory the ``relative_path`` column resolves against.
    """

    def __init__(self, df: pd.DataFrame, root: Path | str):
        self._df = df
        # Normalised so iteration really does yield absolute paths -- the
        # advertised contract, and the one the plot helpers rely on after
        # the caller has chdir'd away. absolute() rather than resolve():
        # the anchor is fixed without rewriting symlinks into paths the
        # caller never typed, and without requiring the directory to exist.
        self._root = Path(root).absolute()
        # Scan metadata travels on df.attrs (set by build_catalog); reading
        # it here rather than taking parameters means a Catalog built from
        # any catalog-shaped frame -- including a pandas-filtered .df --
        # degrades to empty metadata instead of failing.
        self._parse_errors = list(df.attrs.get("parse_errors", []))
        self._n_scanned = int(df.attrs.get("n_scanned", len(df)))
        self._name_collision_differs = list(df.attrs.get("name_collision_differs", []))

    # -- the two halves ----------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """The inventory as a plain DataFrame -- the pandas escape hatch.

        This is the live frame, not a copy: cheap to read, and any pandas
        expression works on it. Catalog methods never mutate it; treat it
        the same way.
        """
        return self._df

    @property
    def root(self) -> Path:
        """The directory the ``relative_path`` column resolves against."""
        return self._root

    @property
    def paths(self) -> list[Path]:
        """Every row's file as an absolute path, in row order."""
        if "relative_path" not in self._df.columns:  # an empty build
            return []
        return [self._root / p for p in self._df["relative_path"]]

    @property
    def parse_errors(self) -> list[tuple[str, str]]:
        """``(path, error message)`` for each file the build failed to parse."""
        return self._parse_errors

    @property
    def n_scanned(self) -> int:
        """Files the build's walk found, before any parse failures."""
        return self._n_scanned

    @property
    def name_collision_differs(self) -> list[str]:
        """Filenames in both ``dedup_between`` groups with differing content."""
        return self._name_collision_differs

    # -- container protocol ------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        """Iterate as absolute file paths (see the class docstring)."""
        return iter(self.paths)

    def __getitem__(self, key):
        """``cat["col"]`` returns the column; ``cat[mask]`` filters.

        The DataFrame idioms the documentation leans on, kept working on
        the Catalog itself so the common filter --
        ``cat[cat["T_max"] >= 100]`` -- reads exactly as it did on the
        frame. Selection follows pandas: a column name gives the Series, a
        list of column names the DataFrame projection (a display view, not
        an inventory, so it is not re-wrapped). Only a boolean mask -- the
        one selection that still means "these files" -- comes back as a
        Catalog. Anything fancier belongs on ``.df``.
        """
        if isinstance(key, str):
            return self._df[key]
        if isinstance(key, list) and key and all(isinstance(item, str) for item in key):
            return self._df[key]
        # Only a genuinely boolean mask returns a Catalog. Anything else
        # list-like (integer positions, an ndarray of column names) would
        # be silently reinterpreted by pandas -- as column selection, or a
        # confusing KeyError -- so it is refused by name instead.
        if isinstance(key, pd.Series) and pd.api.types.is_bool_dtype(key):
            return self._derive(self._df[key])
        if isinstance(key, np.ndarray) and key.dtype == np.bool_:
            return self._derive(self._df[key])
        if isinstance(key, list) and key and all(isinstance(item, (bool, np.bool_)) for item in key):
            return self._derive(self._df[key])
        raise InputTypeError(
            f"Catalog[...] takes a column name (str), a list of column names, "
            f"or a boolean mask, got {type(key).__name__}"
            f"{'' if not isinstance(key, (pd.Series, np.ndarray, list)) else ' of non-bools'}. "
            "For positional or label indexing use the DataFrame directly: "
            "cat.df.iloc[...] / cat.df.loc[...]."
        )

    def _derive(self, df: pd.DataFrame) -> Catalog:
        """A new Catalog over ``df``, keeping root and scan metadata.

        Every selection method funnels through here, so "returns a new
        Catalog, root and metadata intact, index reset" is one fact in one
        place. The reset matters: derived catalogs are new inventories, and
        leaking the parent's row labels would make otherwise-identical
        catalogs compare differently depending on their history.
        """
        derived = df.reset_index(drop=True)
        derived.attrs["parse_errors"] = self._parse_errors
        derived.attrs["n_scanned"] = self._n_scanned
        derived.attrs["name_collision_differs"] = self._name_collision_differs
        return type(self)(derived, self._root)

    # -- selection ---------------------------------------------------------

    def where(self, **criteria) -> Catalog:
        """Rows matching every ``column=value`` equality, as a new Catalog.

        The keyword spelling of the boolean-mask filter, for the common
        case: ``cat.where(group="mt_family", scan_kind="T-scan")``.
        ``value=None`` matches rows where the column is missing (pandas
        ``NaN``/``None``), so ``where(sample_mass_mg=None)`` finds the
        files whose operator left the mass blank. Selecting nothing is an
        ordinary answer and returns an empty Catalog; a column that does
        not exist raises ``KeyError`` naming the columns that do, because
        that is a typo, not a selection.
        """
        if not criteria:
            raise CatalogSelectionError("where() with no criteria selects everything; drop the call")
        mask = pd.Series(True, index=self._df.index)
        for column, value in criteria.items():
            if column not in self._df.columns:
                raise InputKeyError(
                    f"catalog has no column {column!r}; columns are "
                    f"{sorted(self._df.columns)}"
                )
            if value is None:
                mask &= self._df[column].isna()
            else:
                mask &= self._df[column] == value
        return self._derive(self._df[mask])

    def of_type(self, measurement_type: str) -> Catalog:
        """Rows whose file contains the given measurement family.

        Composite-aware, which is the reason it exists: a combined run's
        ``measurement_type`` column reads ``'VSM+AC'``, so the equality
        filter ``where(measurement_type="AC")`` silently misses it -- the
        same trap as ``m.measurement_type == 'AC'`` on a loaded file, and
        the same fix. ``of_type("AC")`` matches ``'AC'`` and ``'VSM+AC'``
        alike, using the vocabulary of
        :attr:`~calamari.measurement.Measurement.measurement_types`.
        An empty result is an ordinary answer (a directory may hold no AC
        runs), matching :meth:`where` rather than the loaded-file
        ``of_type``, which raises on a family the *file* lacks.

        The two share a name but not a grain: this one keeps *files*, the
        loaded-file one keeps *rows*. A combined run selected here comes
        back from :meth:`load` still ``'VSM+AC'``; take the family from
        each loaded file
        (``[m.of_type("AC") for m in cat.of_type("AC").load()]``) or name
        the plot's ``source=`` when the rows must be one family.
        """
        if self._df.empty:
            return self._derive(self._df.copy())
        mask = self._df["measurement_type"].map(
            lambda label: measurement_type in str(label).split("+")
        )
        return self._derive(self._df[mask])

    def at(
        self,
        T_K: float | None = None,
        H_Oe: float | None = None,
        *,
        t_tol_K: float = SCAN_T_TOL_K,
        h_tol_Oe: float = SCAN_H_TOL_OE,
    ) -> Catalog:
        """Rows whose sweep **held** a setpoint at the given value.

        The tolerance filter pandas equality cannot express: a "10000 Oe"
        isotherm is recorded as 9998.7 Oe, so ``where(H_min_Oe=10000)``
        finds nothing. A row matches when the quantity's whole span
        ``[min, max]`` lies within ``tol`` of the value -- which also
        excludes sweeps that merely *pass through* it, since their span
        breaks the window. The default tolerances are the held-quantity
        tolerances used everywhere else (0.5 K, 10 Oe), so "held" here
        agrees with ``scan_kind`` and the logbook's ``H = 1 T`` cells.
        Give ``T_K``, ``H_Oe``, or both (both must hold).
        """
        if T_K is None and H_Oe is None:
            raise CatalogSelectionError("at() needs T_K=, H_Oe=, or both")
        if self._df.empty:
            return self._derive(self._df.copy())
        mask = pd.Series(True, index=self._df.index)
        if T_K is not None:
            mask &= (self._df["T_min"] >= T_K - t_tol_K) & (self._df["T_max"] <= T_K + t_tol_K)
        if H_Oe is not None:
            mask &= (self._df["H_min_Oe"] >= H_Oe - h_tol_Oe) & (
                self._df["H_max_Oe"] <= H_Oe + h_tol_Oe
            )
        return self._derive(self._df[mask])

    def drop(self, *filenames: str) -> Catalog:
        """Everything except the named files, as a new Catalog.

        A filename that matches no row raises
        :class:`~calamari.exceptions.CatalogSelectionError` rather than
        being ignored: naming a file is a claim that it is in the
        inventory, and a typo that silently "dropped" nothing would leave
        the unwanted file in every downstream plot. A name appearing in
        several rows (the same filename under two subfolders) drops all
        of them -- use a mask on ``relative_path`` to distinguish copies.
        """
        if not filenames:
            raise CatalogSelectionError("drop() with no filenames drops nothing; drop the call")
        present = set(self._df["filename"]) if "filename" in self._df.columns else set()
        missing = sorted(set(filenames) - present)
        if missing:
            raise CatalogSelectionError(
                f"drop() names {missing} but the catalog holds no such file(s) -- "
                f"a typo would otherwise silently drop nothing. The catalog's "
                f"{len(self._df)} files are listed by repr(catalog)."
            )
        return self._derive(self._df[~self._df["filename"].isin(filenames)])

    def sort_by(self, *columns: str, ascending: bool = True) -> Catalog:
        """The same rows reordered by the given column(s), as a new Catalog.

        Row order is presentation order -- the repr and :meth:`logbook`
        render it as-is -- so this is how the logbook gets a physics
        ordering: ``cat.sort_by("H_min_Oe")`` lists a field family by
        field, where the build's default (group, then filename) would
        string-sort ``'500Oe'`` after ``'4000Oe'``. The sort is stable, so
        earlier orderings survive as tie-breaks in later ones.
        """
        if not columns:
            raise CatalogSelectionError("sort_by() needs at least one column name")
        missing = sorted(set(columns) - set(self._df.columns))
        if missing:
            raise InputKeyError(
                f"catalog has no column(s) {missing}; columns are "
                f"{sorted(self._df.columns)}"
            )
        return self._derive(
            self._df.sort_values(list(columns), ascending=ascending, kind="stable")
        )

    def select_representative(
        self,
        *,
        by: str,
        prefer_span: tuple[str, str] | None = None,
        tol: float = 0.0,
        keep: str = "widest",
    ) -> Catalog:
        """One representative row per nominal setpoint, as a new Catalog.

        The method form of :func:`select_representative` (see there for
        the policies and their blind spots), so it chains with the other
        selections: ``cat.of_type("AC").select_representative(...)``.
        """
        return self._derive(
            select_representative(
                self._df, by=by, prefer_span=prefer_span, tol=tol, keep=keep
            )
        )

    def deduplicated(self) -> Catalog:
        """The rows not flagged as verified duplicates, as a new Catalog.

        The method form of :func:`deduplicated`. Byte-equality flags are
        recorded by :func:`build_catalog` with ``dedup_between=``; without
        that comparison all rows remain. Files are not rechecked here.
        """
        return self._derive(deduplicated(self._df))

    # -- back to files -----------------------------------------------------

    def load(self) -> list:
        """Every row's file as a loaded Measurement, in row order.

        The explicit bridge back to the single-file API -- and to anything
        a Catalog cannot be handed directly. (The plot helpers do not need
        it: they accept the Catalog itself, loading its paths as they
        draw.) Loading is eager and unguarded on purpose: these paths came
        from a walk that already vetted them, so a failure here means the
        tree changed underneath the catalog, which deserves its raise.
        """
        return [load(path) for path in self.paths]

    def load_one(self):
        """The single remaining file, loaded -- or a refusal naming the count.

        The handoff from "select the run" to "inspect the run":
        ``cat.where(filename="mh_5K.dat").load_one()``. Raises
        :class:`~calamari.exceptions.CatalogSelectionError` unless exactly
        one row remains, because a filter quietly matching two files and
        handing back either would be the wrong file half the time, with
        nothing to show it.
        """
        if len(self._df) == 1:
            return load(self.paths[0])
        if self._df.empty:
            raise CatalogSelectionError(
                "load_one() on an empty catalog: the selection matched no files."
            )
        shown = ", ".join(self._df["filename"].head(5))
        more = "" if len(self._df) <= 5 else f", ... ({len(self._df)} in all)"
        raise CatalogSelectionError(
            f"load_one() needs exactly one remaining file, but the selection "
            f"holds {len(self._df)}: {shown}{more}. Narrow it (where, drop, a "
            f"mask), or use .load() for all of them."
        )

    # -- rendering ---------------------------------------------------------

    def logbook(self, *, delimiter: str | None = None) -> list[str]:
        """This catalog as aligned logbook lines, one per row, in row order.

        Rendered from the rows alone -- no file is re-read -- so it is
        instant and works even if the tree has since moved. Row order is
        the catalog's own: :meth:`sort_by` is the knob, where the
        walk-based :func:`logbook` always sorts by date. Pass
        ``delimiter='\\t'`` for spreadsheet-ready lines.
        """
        return self._render_rows(self._df, delimiter)

    @staticmethod
    def _render_rows(frame: pd.DataFrame, delimiter: str | None) -> list[str]:
        """Logbook lines for exactly the given rows.

        Shared by :meth:`logbook` (the whole frame) and :meth:`__repr__`
        (its head), so the repr can stay cheap on a huge catalog by never
        building cells for rows it will not show.
        """
        if frame.empty:
            return []
        rows = [
            _logbook_cells_from_row(row, row["relative_path"])
            for _, row in frame.iterrows()
        ]
        return _align_logbook_rows(rows, delimiter)

    def __repr__(self) -> str:
        """The logbook, because a directory *is* a table.

        A Measurement's repr is the one-line answer to "what is this
        file?"; a Catalog's is the one-glance answer to "what did I
        measure?" -- so looking at the object in a notebook replaces
        calling anything. Truncated past
        :data:`_CATALOG_REPR_MAX_ROWS` rows with a count of the rest
        (:meth:`logbook` always renders all); rendered without file I/O,
        so it is safe on any size of catalog.
        """
        if self._df.empty:
            return f"<Catalog: 0 recognised files under {self._root}>"
        # Render only the head: cells are never built for hidden rows, so
        # the repr stays cheap however large the catalog. (Column widths
        # therefore align over the shown rows -- the ones being read.)
        lines = self._render_rows(self._df.head(_CATALOG_REPR_MAX_ROWS), None)
        hidden = len(self._df) - _CATALOG_REPR_MAX_ROWS
        if hidden > 0:
            lines.append(f"... {hidden} more file(s) -- .logbook() renders all")
        return "\n".join(lines)


#: ``keep`` policies implemented by :func:`select_representative`. The
#: parameter is validated eagerly (rather than silently falling back to a
#: default) so that a caller anticipating a future policy gets a clear error
#: instead of a quietly-wrong selection.
_SELECT_REPRESENTATIVE_KEEP_POLICIES = ("widest", "most_points")


def select_representative(
    df: pd.DataFrame,
    *,
    by: str,
    prefer_span: tuple[str, str] | None = None,
    tol: float = 0.0,
    keep: str = "widest",
) -> pd.DataFrame:
    """Thin a catalog DataFrame to one representative row per nominal setpoint.

    A catalog built over a real instrument tree routinely contains more than
    one sweep recorded at what is physically the same setpoint: a run
    aborted partway through and re-taken, an operator repeating a sweep,
    successive files that agree to the setpoint's rounding precision but not
    bit-for-bit. Keeping every one of them double-counts boundary points
    derived downstream (a truncated re-run masquerading as an independent
    sweep, or the same transition counted twice) -- this function reduces
    each cluster of same-setpoint rows to a single representative before
    that happens. The originating case is a per-temperature field-sweep
    catalog where a partial re-run needs to lose to the sweep that actually
    covers the full field range; this generalises that policy to any
    setpoint column and any pair of span-defining columns.

    ``keep="widest"`` has a blind spot the summary columns cannot expose: a
    span is only the two endpoints. A superseded first attempt whose sweep
    reached both ends but skipped or lost a stretch in the middle -- an
    interior hole -- reports exactly the same ``lo``/``hi`` as the complete
    re-run, so on span it *ties* with the sweep that replaced it, and the
    lexicographic tie-break can keep the holey original. This has happened
    on a real campaign: temperature scans at fixed fields where abandoned
    first attempts matched the re-runs' summary span while missing a large
    interior temperature interval. Point count sees what the span cannot --
    the hole is exactly the missing rows -- so ``keep="most_points"``
    selects, within each setpoint group, the row with the largest
    ``n_measurements`` (the housekeeping-excluded count from
    :meth:`~calamari.measurement.Measurement.summary`; ``n_points`` is used
    when the frame has no ``n_measurements`` column). This assumes the
    candidates sample comparably -- a densely-sampled narrow sweep out-counts
    a coarse complete one -- so prefer it when the competing rows are
    attempts at the *same* sweep, and ``"widest"`` when they genuinely
    differ in range.

    Parameters
    ----------
    df : pandas.DataFrame
        Catalog rows, e.g. as produced by :func:`build_catalog`. Must
        contain ``by`` and ``relative_path``; both ``prefer_span`` columns
        when ``prefer_span`` is given; and a point-count column
        (``n_measurements`` or ``n_points``) when ``keep="most_points"``.
    by : str
        Name of the setpoint column to group rows by (e.g. a temperature or
        field column). Rows are grouped by proximity in this column, not
        by exact equality -- see ``tol``.
    prefer_span : tuple[str, str], optional
        ``(lo_col, hi_col)`` column names whose difference (``hi - lo``)
        measures how much of a range each row's measurement spans (e.g.
        ``("H_min_Oe", "H_max_Oe")``). Required for ``keep="widest"``,
        where the row with the largest span is kept. Optional for
        ``keep="most_points"``: if given there, span is the secondary
        criterion among rows with equal point counts.
    tol : float, optional
        Rows are chained into the same group when consecutive ``by`` values
        (sorted ascending) differ by no more than ``tol``; a gap larger than
        ``tol`` starts a new group. Default ``0.0``: only exactly equal
        setpoints group together. A physically meaningful choice is the
        instrument's setpoint precision (e.g. ``0.5`` K for a temperature
        column recorded to the nearest half-kelvin).
    keep : str, optional
        Selection policy within a group. ``"widest"`` (default) keeps the
        row with the largest ``prefer_span`` range; ``"most_points"`` keeps
        the row with the most measured points (see above for when each is
        the right choice). Any other value raises ``ValueError``.

    Returns
    -------
    pandas.DataFrame
        One row per setpoint group, in ``by``-ascending order, with the
        original columns and column order preserved (no helper columns
        leak into the result) and the index reset. Ties within a group
        (after every criterion of the ``keep`` policy) are broken
        deterministically by ``relative_path`` (lexicographically smallest
        wins), so the result is reproducible across runs regardless of
        input row order.

    Raises
    ------
    ValueError
        If ``keep`` is not a supported policy, or ``keep="widest"`` is
        requested without ``prefer_span``.
    KeyError
        If any column the chosen policy needs (see ``df`` above) is
        missing from ``df``.
    """
    if keep not in _SELECT_REPRESENTATIVE_KEEP_POLICIES:
        raise CatalogSelectionError(
            f"keep={keep!r} is not a supported policy; "
            f"supported policies are {_SELECT_REPRESENTATIVE_KEEP_POLICIES!r}"
        )
    if keep == "widest" and prefer_span is None:
        raise CatalogSelectionError("keep='widest' selects by span; pass prefer_span=(lo_col, hi_col)")

    required = {by, "relative_path"}
    if prefer_span is not None:
        lo_col, hi_col = prefer_span
        required |= {lo_col, hi_col}
    if keep == "most_points":
        count_col = "n_measurements" if "n_measurements" in df.columns else "n_points"
        required.add(count_col)
    missing = required - set(df.columns)
    if missing:
        raise InputKeyError(f"df is missing required column(s) for select_representative: {missing!r}")

    if df.empty:
        return df.copy()

    # Sort by the setpoint column so that within-tol grouping can be done by
    # chaining consecutive gaps (relative_path as a secondary key only
    # affects row order among exact setpoint ties, not the grouping itself).
    working = df.sort_values([by, "relative_path"]).reset_index(drop=True)

    # Chain-link grouping: a new group starts wherever the gap to the
    # previous (sorted) setpoint exceeds tol. This groups tight clusters of
    # near-duplicate setpoints even when no two rows in the cluster are
    # individually within tol of the *first* row -- e.g. successive re-runs
    # that each drift a little from the last.
    working["_group"] = (working[by].diff() > tol).cumsum()

    # The policy's criteria, best-first: point count for "most_points" (with
    # span as the secondary criterion when available), span alone for
    # "widest". relative_path last as the deterministic tie-break;
    # drop_duplicates(keep="first") then picks exactly one row per group.
    score_cols: list[str] = []
    if keep == "most_points":
        working["_count"] = working[count_col]
        score_cols.append("_count")
    if prefer_span is not None:
        working["_span"] = working[hi_col] - working[lo_col]
        score_cols.append("_span")

    selected = (
        working.sort_values(
            ["_group", *score_cols, "relative_path"],
            ascending=[True, *([False] * len(score_cols)), True],
        )
        .drop_duplicates(subset="_group", keep="first")
        .sort_values(by)
        .drop(columns=["_group", *score_cols])
        .reset_index(drop=True)
    )
    return selected
