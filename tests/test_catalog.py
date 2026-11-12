"""Tests for calamari.catalog: file discovery, per-file summaries, full
inventory assembly (with optional cross-group duplicate detection), and
the deduplicated view.

Per the project's no-real-data-files policy, every test tree is built
from :mod:`calamari.simulate` synthetic files written into ``tmp_path``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import calamari
from calamari import catalog, simulate


def _write_mh_file(path, *, m_sat_offset=0.0):
    """A small synthetic MPMS3 M(H) file (fixed T, swept H)."""
    field_oe = np.linspace(-1000.0, 1000.0, 11)
    temperature_k = np.full(11, 5.0)
    moment_emu = simulate.synthetic_mh_curve(
        field_oe, m_sat_emu=1.0e-3 + m_sat_offset, h_knee_oe=200.0
    )
    return simulate.write_mpms3_file(
        path,
        app="SQUID DC",
        sample_material="SyntheticSample",
        sample_mass_mg=10.0,
        temperature_K=temperature_k,
        field_oe=field_oe,
        vsm_moment_emu=moment_emu,
    )


@pytest.fixture
def sample_tree(tmp_path):
    """A two-group tree: one byte-identical duplicate pair (``common.dat``),
    one same-name-different-content pair (``collide.dat``), and one
    non-MPMS decoy file that must never appear in the catalog."""
    group_a = tmp_path / "GroupA"
    group_b = tmp_path / "GroupB"
    group_a.mkdir()
    group_b.mkdir()

    # Identical content in both groups (written with identical arguments).
    _write_mh_file(group_a / "common.dat")
    _write_mh_file(group_b / "common.dat")

    # Same filename, genuinely different content.
    _write_mh_file(group_a / "collide.dat", m_sat_offset=0.0)
    _write_mh_file(group_b / "collide.dat", m_sat_offset=5.0e-4)

    (tmp_path / "decoy.dat").write_text("just some unrelated text\nnot a measurement file\n")

    return tmp_path


def test_find_files_excludes_decoy(sample_tree):
    files = catalog.find_files(sample_tree)
    names = {p.name for p in files}
    assert names == {"common.dat", "collide.dat"}
    assert len(files) == 4


def test_find_files_skips_default_git_dir(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    _write_mh_file(git_dir / "hidden.dat")
    _write_mh_file(tmp_path / "visible.dat")

    files = catalog.find_files(tmp_path)
    assert {p.name for p in files} == {"visible.dat"}


def test_find_files_custom_skip_dirs(tmp_path):
    skip_dir = tmp_path / "skipme"
    skip_dir.mkdir()
    _write_mh_file(skip_dir / "a.dat")
    _write_mh_file(tmp_path / "b.dat")

    files = catalog.find_files(tmp_path, skip_dirs=("skipme",))
    assert {p.name for p in files} == {"b.dat"}


def test_find_files_missing_root_raises_file_not_found(tmp_path):
    """os.walk yields nothing for a missing top directory, so a typo'd root
    used to come back as [] -- the same answer as a real empty directory --
    and surfaced later as 'requires at least one measurement'."""
    missing = tmp_path / "no_such_dir"
    with pytest.raises(FileNotFoundError, match="no_such_dir") as excinfo:
        catalog.find_files(missing)
    # IO-5: the filesystem's own error, not a calamari judgement about content.
    assert not isinstance(excinfo.value, calamari.CalamariError)
    assert excinfo.value.filename == str(missing)


def test_find_files_file_as_root_raises_not_a_directory(tmp_path):
    """A single file is not a tree to search; returning [] (or [that file])
    would hide that the caller meant load(), not find_files()."""
    single = _write_mh_file(tmp_path / "one.dat")
    with pytest.raises(NotADirectoryError, match="one.dat"):
        catalog.find_files(single)


def test_build_catalog_and_logbook_missing_root_raise(tmp_path):
    """The walk's callers inherit the error rather than reporting an empty tree."""
    missing = tmp_path / "no_such_dir"
    with pytest.raises(FileNotFoundError):
        catalog.build_catalog(missing)
    with pytest.raises(FileNotFoundError):
        catalog.logbook(missing)


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions that bind the current user (root ignores them)",
)
def test_find_files_unreadable_root_raises_permission_error(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        with pytest.raises(PermissionError, match="locked"):
            catalog.find_files(locked)
    finally:
        locked.chmod(0o755)


def test_find_files_returns_sorted_paths_whatever_the_creation_order(tmp_path):
    """Directory-listing order is the filesystem's choice (not alphabetical,
    even on one machine), so the result is sorted, component by component.
    Files are created scrambled, across subdirectories, so creation order
    cannot pass for sorted order by accident."""
    for rel in ("m.dat", "sub/z.dat", "z.dat", "run-2/a.dat", "b.dat", "run/b.dat",
                "sub/a.dat", "A.dat", "run/a.dat", "chi_2K.dat", "chi_10K.dat"):
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        _write_mh_file(tmp_path / rel)
    (tmp_path / "sub" / "notes.dat").write_text("not a measurement\n")

    files = catalog.find_files(tmp_path)
    rel = [p.relative_to(tmp_path).as_posix() for p in files]
    assert rel == [
        "A.dat",                      # case-sensitive: upper case before lower
        "b.dat",
        "chi_10K.dat",                # plain string order, never a numeric guess
        "chi_2K.dat",
        "m.dat",
        "run/a.dat",                  # a directory's contents stay together ...
        "run/b.dat",
        "run-2/a.dat",                # ... ahead of 'run-2', though '-' < '/'
        "sub/a.dat",
        "sub/z.dat",
        "z.dat",
    ]


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions that bind the current user (root ignores them)",
)
def test_find_files_unreadable_file_is_tolerated_silently(tmp_path, recwarn):
    """An unreadable *file* is 'not a match', exactly as before: the signature
    probe never raises, and it is no reason to abort or warn about a scan."""
    _write_mh_file(tmp_path / "good.dat")
    locked = tmp_path / "locked.dat"
    _write_mh_file(locked)
    locked.chmod(0)
    try:
        files = catalog.find_files(tmp_path)
    finally:
        locked.chmod(0o644)
    assert [p.name for p in files] == ["good.dat"]
    assert not [w for w in recwarn if "find_files" in str(w.message)]


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions that bind the current user (root ignores them)",
)
def test_find_files_unreadable_subdirectory_warns_and_continues(tmp_path):
    """One locked folder must not cost the rest of the tree, but the files it
    hides must not vanish silently either: one warning names it and the fix."""
    _write_mh_file(tmp_path / "good.dat")
    locked = tmp_path / "locked"
    locked.mkdir()
    _write_mh_file(locked / "hidden.dat")
    locked.chmod(0)
    try:
        with pytest.warns(UserWarning, match="find_files: 1 directory could not be listed") as rec:
            files = catalog.find_files(tmp_path)
        # Through build_catalog too, and still blaming the caller's line.
        with pytest.warns(UserWarning, match="could not be listed") as via_catalog:
            cat = catalog.build_catalog(tmp_path)
        # Pruning it by name is the documented way to leave it out knowingly.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pruned = catalog.find_files(tmp_path, skip_dirs=(".git", "locked"))
    finally:
        locked.chmod(0o755)
    assert [p.name for p in files] == ["good.dat"]
    assert [p.name for p in pruned] == ["good.dat"]
    message = str(rec[0].message)
    assert str(locked) in message
    assert "skip_dirs=('.git', 'locked')" in message
    # Blames the caller's line, not calamari's own.
    assert rec[0].filename == __file__
    assert via_catalog[0].filename == __file__
    assert cat.paths == [tmp_path / "good.dat"]


def test_summarise_without_group_fn_has_no_group_key(sample_tree):
    row = catalog.summarise(sample_tree / "GroupA" / "common.dat")
    assert "group" not in row
    assert row["filename"] == "common.dat"
    assert row["data_format"] == "MPMS3"
    assert row["scan_kind"] == "H-scan"
    assert row["T_min"] == pytest.approx(5.0)
    assert row["T_max"] == pytest.approx(5.0)
    assert row["H_min_Oe"] == pytest.approx(-1000.0)
    assert row["H_max_Oe"] == pytest.approx(1000.0)
    assert row["n_points"] == 11


def test_summarise_with_group_fn(sample_tree):
    path = sample_tree / "GroupA" / "common.dat"
    row = catalog.summarise(path, root=sample_tree, group_fn=lambda rel: rel.parts[0])
    assert row["group"] == "GroupA"
    assert row["relative_path"] == "GroupA/common.dat"


def test_build_catalog_annotation_names_what_it_returns():
    """The signature advertised pd.DataFrame long after the function started
    returning a Catalog, so a reader (or a type checker) trusting it would
    reach for DataFrame methods a Catalog does not have."""
    import typing

    hints = typing.get_type_hints(catalog.build_catalog)
    assert hints["return"] is catalog.Catalog


def test_build_catalog_default_group_fn_and_counts(sample_tree):
    cat = catalog.build_catalog(sample_tree)
    assert isinstance(cat, catalog.Catalog)
    assert len(cat) == 4
    assert set(cat["group"]) == {"GroupA", "GroupB"}
    assert cat.n_scanned == 4
    assert cat.parse_errors == []
    # No dedup_between was given: schema columns still present, all False/empty.
    assert not cat["is_duplicate"].any()
    assert (cat["dup_status"] == "").all()
    assert (cat["canonical_path"] == cat["relative_path"]).all()
    assert cat.name_collision_differs == []
    # The scan metadata is mirrored on the plain frame for pandas-side use.
    assert cat.df.attrs["n_scanned"] == 4


def test_build_catalog_custom_group_fn(sample_tree):
    cat = catalog.build_catalog(sample_tree, group_fn=lambda rel: rel.parts[0].lower())
    assert set(cat["group"]) == {"groupa", "groupb"}


def test_build_catalog_dedup_flags_and_canonical_preference(sample_tree):
    cat = catalog.build_catalog(sample_tree, dedup_between=("GroupA", "GroupB"))

    common_rows = cat.where(filename="common.dat")
    a_common = common_rows.where(group="GroupA").df.iloc[0]
    b_common = common_rows.where(group="GroupB").df.iloc[0]

    assert a_common["is_duplicate"]
    assert a_common["dup_status"] == "identical"
    assert a_common["canonical_path"] == b_common["relative_path"]
    assert not b_common["is_duplicate"]
    assert b_common["dup_status"] == "identical"

    collide_rows = cat.where(filename="collide.dat")
    assert set(collide_rows["dup_status"]) == {"name_collision_differs"}
    assert not collide_rows["is_duplicate"].any()

    assert cat.name_collision_differs == ["collide.dat"]


def test_deduplicated_drops_exactly_the_redundant_copy(sample_tree):
    cat = catalog.build_catalog(sample_tree, dedup_between=("GroupA", "GroupB"))
    deduped = cat.deduplicated()

    assert len(deduped) == len(cat) - 1
    common_rows = deduped.where(filename="common.dat")
    assert len(common_rows) == 1
    assert common_rows["group"].tolist() == ["GroupB"]
    # The name-collision pair is not deduplicated -- both rows survive.
    assert len(deduped.where(filename="collide.dat")) == 2
    # The function form still works on the plain frame, and agrees.
    assert catalog.deduplicated(cat.df)["relative_path"].tolist() == (
        deduped["relative_path"].tolist()
    )


@pytest.mark.parametrize(
    "difference",
    ["small_cgs", "si_signal", "frequency", "row_order", "errors", "drive",
     "populated_column", "units", "sample_mass", "sample_name", "formatting", "line_endings"],
)
def test_dedup_retains_distinct_exports(tmp_path, difference):
    """No signal, settings, metadata or formatting distinction may be discarded."""
    temperature = np.linspace(2.0, 10.0, 9)
    moment = np.linspace(1e-7, 2e-7, 9)
    first = dict(
        app="SQUID VSM", temperature_K=temperature, field_oe=np.full(9, 1000.0),
        vsm_moment_emu=moment,
    )
    second = first.copy()
    if difference == "small_cgs":
        second["vsm_moment_emu"] = 2 * moment
    elif difference == "si_signal":
        first["unit_system"] = second["unit_system"] = "Am2"
        second["vsm_moment_emu"] = 100 * moment
    elif difference in {"frequency", "drive"}:
        for options in (first, second):
            options.update(app="SQUID AC", chi_real=moment,
                           ac_frequency_hz=np.full(9, 97.0), ac_drive_oe=np.ones(9))
        second["ac_frequency_hz" if difference == "frequency" else "ac_drive_oe"] = (
            np.full(9, 997.0 if difference == "frequency" else 2.0)
        )
    elif difference == "row_order":
        second.update(temperature_K=temperature[::-1], vsm_moment_emu=moment[::-1])
    elif difference == "errors":
        first["vsm_moment_err_emu"] = np.full(9, 1e-9)
        second["vsm_moment_err_emu"] = np.full(9, 2e-9)
    elif difference == "populated_column":
        first["extra_columns"] = {"Auxiliary (V)": np.full(9, np.nan)}
        second["extra_columns"] = {"Auxiliary (V)": np.ones(9)}
    elif difference == "units":
        second["unit_system"] = "Am2"
    elif difference == "sample_mass":
        second["sample_mass_mg"] = 7.0
    elif difference == "sample_name":
        second["sample_material"] = "OtherSyntheticSample"
    for group, options in (("A", first), ("B", second)):
        (tmp_path / group).mkdir()
        simulate.write_mpms3_file(tmp_path / group / "scan.dat", **options)
    path = tmp_path / "B/scan.dat"
    if difference == "formatting":
        path.write_bytes(path.read_bytes().replace(b"[Data]", b"\n[Data]"))
    elif difference == "line_endings":
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    cat = catalog.build_catalog(tmp_path, dedup_between=("A", "B"))
    assert cat.parse_errors == []
    assert cat["dup_status"].tolist() == ["name_collision_differs"] * 2
    assert cat["canonical_path"].tolist() == ["A/scan.dat", "B/scan.dat"]
    assert cat.name_collision_differs == ["scan.dat"]
    assert len(cat.deduplicated()) == 2


def test_dedup_verifies_bytes_even_when_digests_collide(sample_tree, monkeypatch):
    """A shortlist hash collision must never remove a distinct export."""
    monkeypatch.setattr(catalog, "_content_key", lambda path: b"same digest")
    cat = catalog.build_catalog(sample_tree, dedup_between=("GroupA", "GroupB"))
    assert len(cat.deduplicated()) == 3
    assert cat.where(filename="collide.dat")["dup_status"].tolist() == (
        ["name_collision_differs"] * 2
    )


@pytest.mark.parametrize("tail", ["identical", "changed", "extra_row"])
def test_dedup_checks_full_exports_beyond_first_block(tmp_path, monkeypatch, tail):
    """Hash collisions cannot conceal changed suffixes or different lengths."""
    temperature = np.linspace(2.0, 10.0, 5000)
    moment = np.linspace(1e-7, 2e-7, 5000)
    for group in ("A", "B"):
        (tmp_path / group).mkdir()
        simulate.write_mpms3_file(
            tmp_path / group / "scan.dat", app="SQUID VSM",
            temperature_K=temperature, field_oe=np.full(5000, 1000.0),
            vsm_moment_emu=moment,
        )
    path = tmp_path / "B/scan.dat"
    original = path.read_bytes()
    assert len(original) > 64 * 1024
    lines = original.splitlines(keepends=True)
    if tail == "changed":
        lines[-1] = lines[-1].replace(b"2e-07", b"3e-07")
        assert lines[-1] != original.splitlines(keepends=True)[-1]
    elif tail == "extra_row":
        lines.append(lines[-1])
    path.write_bytes(b"".join(lines))
    assert path.read_bytes()[:64 * 1024] == original[:64 * 1024]
    monkeypatch.setattr(catalog, "_content_key", lambda path: b"same digest")
    cat = catalog.build_catalog(tmp_path, dedup_between=("A", "B"))
    assert cat.parse_errors == []
    assert len(cat.deduplicated()) == (1 if tail == "identical" else 2)


def test_dedup_mixed_same_name_files_are_auditable_and_deterministic(tmp_path):
    for rel, offset in (("B/z", 0.0), ("A/different", 0.001), ("B/different", 0.002),
                        ("A/copy", 0.0), ("B/a", 0.0), ("C/other", 0.0)):
        folder = tmp_path / rel
        folder.mkdir(parents=True)
        _write_mh_file(folder / "scan.dat", m_sat_offset=offset)
    cat = catalog.build_catalog(tmp_path, dedup_between=("A", "B"))
    assert cat["relative_path"].tolist() == [
        "A/copy/scan.dat", "A/different/scan.dat", "B/a/scan.dat",
        "B/different/scan.dat", "B/z/scan.dat", "C/other/scan.dat",
    ]
    assert cat["dup_status"].tolist() == [
        "identical", "name_collision_differs", "identical",
        "name_collision_differs", "identical", "",
    ]
    assert cat["is_duplicate"].tolist() == [True, False, False, False, False, False]
    assert cat.df.iloc[0]["canonical_path"] == "B/z/scan.dat"
    assert cat.name_collision_differs == ["scan.dat"]
    assert len(cat.deduplicated()) == 5


def test_dedup_leaves_identical_raw_files_out_of_scope(tmp_path):
    for group in ("A", "B"):
        (tmp_path / group).mkdir()
        simulate.write_mpms3_raw_file(
            tmp_path / group / "scan.rw.dat",
            raw_position_mm=[-1.0, 0.0, 1.0], raw_voltage_v=[0.01, 0.02, 0.01],
        )
    cat = catalog.build_catalog(tmp_path, dedup_between=("A", "B"))
    assert cat["file_role"].tolist() == ["raw", "raw"]
    assert cat["dup_status"].tolist() == ["", ""]
    assert cat.name_collision_differs == []
    assert len(cat.deduplicated()) == 2


def test_deduplicated_raises_keyerror_without_flag_column():
    bare = pd.DataFrame({"relative_path": ["a.dat"], "filename": ["a.dat"]})
    with pytest.raises(KeyError):
        catalog.deduplicated(bare)


def test_build_catalog_empty_root(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    cat = catalog.build_catalog(empty_dir)

    assert len(cat) == 0
    assert cat.n_scanned == 0
    assert cat.parse_errors == []
    assert cat.name_collision_differs == []
    assert "is_duplicate" in cat.df.columns
    assert cat.paths == []
    assert cat.logbook() == []
    assert "0 recognised files" in repr(cat)


@pytest.mark.parametrize("relative_root", [False, True])
def test_path_filter_excludes_files_before_probe_or_read(sample_tree, monkeypatch, relative_root):
    """Filtering must save discovery reads as well as full parses and dedup reads."""
    import builtins
    import io

    simulate.write_mpms3_raw_file(
        sample_tree / "scan.rw.dat",
        raw_position_mm=[-1.0, 0.0, 1.0], raw_voltage_v=[0.01, 0.02, 0.01],
    )
    selected = {Path("GroupA/common.dat"), Path("GroupB/common.dat")}
    expected = catalog.build_catalog(sample_tree, dedup_between=("GroupA", "GroupB"))
    expected = expected.where(filename="common.dat")
    excluded = {
        p.absolute() for p in sample_tree.rglob("*")
        if p.is_file() and p.relative_to(sample_tree) not in selected
    }

    def guard_open(original):
        def checked(path, *args, **kwargs):
            if isinstance(path, (str, os.PathLike)):
                assert Path(path).absolute() not in excluded, f"opened excluded file: {path}"
            return original(path, *args, **kwargs)
        return checked

    monkeypatch.setattr(builtins, "open", guard_open(builtins.open))
    monkeypatch.setattr(io, "open", guard_open(io.open))
    probed = []
    original_probe = catalog.sniff_format

    def probe(path):
        assert path.absolute() not in excluded
        probed.append(path.relative_to(root))
        return original_probe(path)

    monkeypatch.setattr(catalog, "sniff_format", probe)
    root = sample_tree
    if relative_root:
        monkeypatch.chdir(sample_tree.parent)
        root = Path(sample_tree.name)
    seen = []

    def include(path):
        assert not path.is_absolute()
        seen.append(path)
        return path in selected

    actual = catalog.build_catalog(root, path_filter=include, dedup_between=("GroupA", "GroupB"))
    # The full tree's raw row has absent quality counts, promoting its
    # integer columns to floats. Compare values, not that pandas inference.
    pd.testing.assert_frame_equal(actual.df, expected.df.reset_index(drop=True), check_dtype=False)
    assert set(probed) == selected
    assert len(probed) == 2
    assert len(seen) == 6
    assert actual.n_scanned == 2
    assert actual.parse_errors == []
    assert actual.name_collision_differs == []
    assert actual.paths == expected.paths
    pd.testing.assert_frame_equal(
        actual.where(group="GroupB").load_one().data,
        expected.where(group="GroupB").load_one().data,
    )
    assert len(actual.deduplicated()) == 1


def test_path_filter_empty_selection_still_validates_root(tmp_path, monkeypatch):
    _write_mh_file(tmp_path / "excluded.dat")

    def unexpected_probe(path):
        pytest.fail(f"probed excluded path: {path}")

    monkeypatch.setattr(catalog, "sniff_format", unexpected_probe)
    cat = catalog.build_catalog(tmp_path, path_filter=lambda path: False)
    assert len(cat) == cat.n_scanned == 0
    assert cat.parse_errors == []
    assert cat.paths == []
    with pytest.raises(FileNotFoundError):
        catalog.build_catalog(tmp_path / "missing", path_filter=lambda path: False)
    with pytest.raises(NotADirectoryError):
        catalog.build_catalog(tmp_path / "excluded.dat", path_filter=lambda path: False)


def test_path_filter_keeps_content_recognition_and_parse_error_counts(sample_tree, monkeypatch):
    """An included decoy is still unrecognised; failures after recognition are recorded."""
    original_load = catalog.load

    def load(path):
        if path.name == "common.dat":
            raise PermissionError("synthetic read failure after recognition")
        if path.parent.name == "GroupB":
            raise ValueError("synthetic parse failure")
        return original_load(path)

    monkeypatch.setattr(catalog, "load", load)
    cat = catalog.build_catalog(sample_tree, path_filter=lambda path: True)
    assert cat.n_scanned == 4
    assert len(cat) == 1
    assert cat["relative_path"].tolist() == ["GroupA/collide.dat"]
    assert cat.parse_errors == [
        (str(sample_tree / "GroupA/common.dat"),
         "PermissionError: synthetic read failure after recognition"),
        (str(sample_tree / "GroupB/collide.dat"), "ValueError: synthetic parse failure"),
        (str(sample_tree / "GroupB/common.dat"),
         "PermissionError: synthetic read failure after recognition"),
    ]
    assert cat.df.attrs["parse_errors"] == cat.parse_errors
    assert cat.df.attrs["n_scanned"] == 4


def test_path_filter_exceptions_propagate(sample_tree):
    def broken_filter(path):
        raise RuntimeError("caller predicate failed")

    with pytest.raises(RuntimeError, match="caller predicate failed"):
        catalog.build_catalog(sample_tree, path_filter=broken_filter)


def test_find_files_path_filter_preserves_order_and_skip_dirs(sample_tree):
    selected = {Path("GroupB/common.dat"), Path("GroupA/common.dat")}
    assert catalog.find_files(sample_tree, path_filter=lambda path: path in selected) == [
        sample_tree / path for path in sorted(selected)
    ]
    assert catalog.find_files(
        sample_tree, skip_dirs=("GroupA",), path_filter=lambda path: path in selected
    ) == [sample_tree / "GroupB/common.dat"]


def test_path_filter_does_not_classify_role_and_preserves_custom_groups(tmp_path):
    """Even an included file with a summary-like name gets its role from content."""
    simulate.write_mpms3_raw_file(
        tmp_path / "scan.dat",
        raw_position_mm=[-1.0, 0.0, 1.0], raw_voltage_v=[0.01, 0.02, 0.01],
    )
    cat = catalog.build_catalog(
        tmp_path, path_filter=lambda path: path.name == "scan.dat",
        group_fn=lambda path: path.stem.upper(),
    )
    assert cat["file_role"].tolist() == ["raw"]
    assert cat["group"].tolist() == ["SCAN"]
    assert cat.n_scanned == 1


# ---------------------------------------------------------------------------
# select_representative
# ---------------------------------------------------------------------------


def _setpoint_frame():
    """Synthetic catalog-shaped frame: two exact-duplicate setpoints
    (T=5.0, rows 'a.dat'/'b.dat' of differing span), one near-duplicate
    just outside a tight tolerance ('c.dat', T=5.02), and two lone
    setpoints ('d.dat', 'e.dat')."""
    return pd.DataFrame(
        {
            "relative_path": ["b.dat", "a.dat", "c.dat", "d.dat", "e.dat"],
            "T_K": [5.0, 5.0, 5.02, 10.0, 20.0],
            "H_min_Oe": [-500.0, -1000.0, -800.0, -2000.0, -100.0],
            "H_max_Oe": [500.0, 1000.0, 800.0, 2000.0, 100.0],
        }
    )


def test_select_representative_exact_duplicate_keeps_widest():
    df = _setpoint_frame()
    result = catalog.select_representative(
        df, by="T_K", prefer_span=("H_min_Oe", "H_max_Oe")
    )

    # tol=0.0: only the two exact T=5.0 rows group; 5.02 is its own group.
    assert sorted(result["relative_path"]) == ["a.dat", "c.dat", "d.dat", "e.dat"]
    row_5k = result[result["T_K"] == 5.0].iloc[0]
    assert row_5k["relative_path"] == "a.dat"  # span 2000 Oe beats b.dat's 1000 Oe
    # Original columns/order preserved, no helper columns leaked.
    assert list(result.columns) == list(df.columns)


def test_select_representative_groups_within_tol():
    df = _setpoint_frame()
    result = catalog.select_representative(
        df, by="T_K", prefer_span=("H_min_Oe", "H_max_Oe"), tol=0.05
    )

    # With tol=0.05, 5.0 and 5.02 chain into one group; a.dat (span 2000)
    # still wins over both b.dat (1000) and c.dat (1600).
    assert sorted(result["relative_path"]) == ["a.dat", "d.dat", "e.dat"]
    assert len(result) == 3


def test_select_representative_ties_broken_by_relative_path():
    df = pd.DataFrame(
        {
            "relative_path": ["z_run.dat", "m_run.dat"],
            "T_K": [15.0, 15.0],
            "H_min_Oe": [-1000.0, -1000.0],
            "H_max_Oe": [1000.0, 1000.0],
        }
    )
    result = catalog.select_representative(df, by="T_K", prefer_span=("H_min_Oe", "H_max_Oe"))

    assert len(result) == 1
    assert result.iloc[0]["relative_path"] == "m_run.dat"  # lexicographically smallest


def test_select_representative_does_not_mutate_input():
    df = _setpoint_frame()
    original = df.copy(deep=True)
    catalog.select_representative(df, by="T_K", prefer_span=("H_min_Oe", "H_max_Oe"))
    pd.testing.assert_frame_equal(df, original)


def test_select_representative_empty_frame():
    df = pd.DataFrame(columns=["relative_path", "T_K", "H_min_Oe", "H_max_Oe"])
    result = catalog.select_representative(df, by="T_K", prefer_span=("H_min_Oe", "H_max_Oe"))
    assert result.empty
    assert list(result.columns) == list(df.columns)


def test_select_representative_bad_keep_raises():
    df = _setpoint_frame()
    with pytest.raises(ValueError):
        catalog.select_representative(
            df, by="T_K", prefer_span=("H_min_Oe", "H_max_Oe"), keep="most_recent"
        )


def test_select_representative_widest_requires_prefer_span():
    df = _setpoint_frame()
    with pytest.raises(ValueError):
        catalog.select_representative(df, by="T_K", keep="widest")


def _holey_rerun_frame():
    """Two attempts at the same T-scan at one field setpoint: an abandoned
    first attempt with a large interior hole and a complete re-run. Their
    summary spans are identical -- the span columns cannot see the hole --
    so only point count separates them. A lone third setpoint checks the
    grouping is untouched."""
    return pd.DataFrame(
        {
            "relative_path": ["attempt1.dat", "rerun.dat", "other.dat"],
            "H_Oe": [1000.0, 1000.0, 5000.0],
            "T_min": [2.0, 2.0, 2.0],
            "T_max": [40.0, 40.0, 40.0],
            "n_measurements": [45, 77, 77],
            "n_points": [46, 78, 78],
        }
    )


def test_select_representative_most_points_beats_holey_equal_span_row():
    # keep="widest" ties on span and keeps the holey first attempt purely
    # because "attempt1.dat" < "rerun.dat" lexicographically -- the failure
    # mode most_points exists to fix.
    df = _holey_rerun_frame()
    widest = catalog.select_representative(df, by="H_Oe", prefer_span=("T_min", "T_max"))
    assert widest[widest["H_Oe"] == 1000.0].iloc[0]["relative_path"] == "attempt1.dat"

    result = catalog.select_representative(df, by="H_Oe", keep="most_points")
    assert sorted(result["relative_path"]) == ["other.dat", "rerun.dat"]
    assert result[result["H_Oe"] == 1000.0].iloc[0]["relative_path"] == "rerun.dat"
    # No helper columns leak; prefer_span was never needed.
    assert list(result.columns) == list(df.columns)


def test_select_representative_most_points_falls_back_to_n_points():
    df = _holey_rerun_frame().drop(columns=["n_measurements"])
    result = catalog.select_representative(df, by="H_Oe", keep="most_points")
    assert result[result["H_Oe"] == 1000.0].iloc[0]["relative_path"] == "rerun.dat"


def test_select_representative_most_points_span_breaks_count_ties():
    df = pd.DataFrame(
        {
            "relative_path": ["narrow.dat", "wide.dat"],
            "T_K": [5.0, 5.0],
            "H_min_Oe": [-500.0, -1000.0],
            "H_max_Oe": [500.0, 1000.0],
            "n_measurements": [30, 30],
        }
    )
    result = catalog.select_representative(
        df, by="T_K", prefer_span=("H_min_Oe", "H_max_Oe"), keep="most_points"
    )
    assert len(result) == 1
    assert result.iloc[0]["relative_path"] == "wide.dat"


def test_select_representative_most_points_missing_count_column_raises():
    df = _setpoint_frame()  # has neither n_measurements nor n_points
    with pytest.raises(KeyError):
        catalog.select_representative(df, by="T_K", keep="most_points")


def test_select_representative_missing_column_raises_keyerror():
    df = _setpoint_frame().drop(columns=["H_max_Oe"])
    with pytest.raises(KeyError):
        catalog.select_representative(df, by="T_K", prefer_span=("H_min_Oe", "H_max_Oe"))


# ---------------------------------------------------------------------------
# logbook_line / logbook
# ---------------------------------------------------------------------------


def _load(path):
    return calamari.load(path)


def test_logbook_line_mh_isotherm(tmp_path):
    """Held T reads 'T = ...'; swept H reads a signed span in Tesla."""
    path = _write_mh_file(tmp_path / "mh.dat")
    cells = catalog.logbook_line(_load(path)).split("\t")
    assert cells[0] == "mh.dat"
    assert cells[1] == "2020-01-29"
    assert cells[2] == "SyntheticSample"
    assert cells[3] == "10 mg"
    assert cells[7] == "T = 5 K"
    assert cells[8] == "-0.1 < H < 0.1 T"  # signed: -1000..1000 Oe


def test_logbook_line_lists_distinct_frequencies_not_a_mean(tmp_path):
    """A two-frequency interleave lists both; a mean would describe neither."""
    t = np.linspace(2.0, 40.0, 40)
    path = simulate.write_mpms3_file(
        tmp_path / "chi.dat",
        app="SQUID AC",
        temperature_K=t,
        field_oe=np.full(40, 100.0),
        chi_real=0.05 / (t + 1.0),
        chi_imag=0.001 / (t + 1.0),
        ac_frequency_hz=np.where(np.arange(40) % 2, 997.0, 97.0),
        ac_drive_oe=np.full(40, 3.0),
    )
    cells = catalog.logbook_line(_load(path)).split("\t")
    assert cells[5] == "97; 997 Hz"
    assert cells[6] == "T-scan"
    assert cells[7] == "2 < T < 40 K"


def test_logbook_line_missing_values_are_blank_cells(tmp_path):
    """No mass and no AC data must give blank cells, not a crash."""
    path = simulate.write_mpms3_file(
        tmp_path / "mt.dat",
        app="SQUID VSM",
        sample_mass_mg=None,
        temperature_K=np.linspace(2.0, 40.0, 20),
        field_oe=np.full(20, 1000.0),
        vsm_moment_emu=np.full(20, 1.0e-3),
    )
    cells = catalog.logbook_line(_load(path)).split("\t")
    assert cells[3] == ""  # mass unrecorded
    assert cells[5] == ""  # no AC frequency in a DC/VSM file
    assert len(cells) == 9  # column structure intact regardless


def test_logbook_line_excludes_housekeeping_rows_from_spans(tmp_path):
    """A trailing locate scan far off-setpoint must not stretch the T cell."""
    path = simulate.write_mpms3_file(
        tmp_path / "hk.dat",
        app="SQUID VSM",
        temperature_K=np.array([2.0, 2.0, 2.0, 2.0, 8.5]),
        field_oe=np.array([0.0, 100.0, 200.0, 300.0, 0.0]),
        vsm_moment_emu=np.array([1e-3] * 4 + [np.nan]),
        transport_action=np.array([1.0] * 4 + [3.0]),
    )
    cells = catalog.logbook_line(_load(path)).split("\t")
    assert cells[7] == "T = 2 K"


def test_logbook_line_survives_unknown_unit(tmp_path):
    """A file whose temperature accessor raises still gets a line."""
    path = simulate.write_mpms3_file(
        tmp_path / "bad.dat",
        app="SQUID DC",
        temperature_K=np.linspace(2.0, 40.0, 10),
        field_oe=np.full(10, 500.0),
        dc_moment_emu=np.full(10, 1.0e-3),
    )
    path.write_text(path.read_text().replace("Temperature (K)", "Temperature (degC)"))
    cells = catalog.logbook_line(_load(path)).split("\t")
    assert cells[6] == "unknown"
    assert cells[7] == ""  # unanswerable, not invented
    assert cells[8] == "H = 0.05 T"  # the readable cells survive


def test_logbook_line_root_keeps_subfolder(tmp_path):
    sub = tmp_path / "GroupA"
    sub.mkdir()
    path = _write_mh_file(sub / "mh.dat")
    line = catalog.logbook_line(_load(path), root=tmp_path)
    assert line.split("\t")[0] == str(Path("GroupA") / "mh.dat")


def test_logbook_walks_sorts_and_skips_decoys(tmp_path):
    _write_mh_file(tmp_path / "later.dat")  # default date 1/29/2020
    early = simulate.write_mpms3_file(
        tmp_path / "early.dat",
        date="1/5/2020",
        temperature_K=np.linspace(2.0, 40.0, 10),
        field_oe=np.full(10, 100.0),
        dc_moment_emu=np.full(10, 1.0e-3),
    )
    assert early.exists()
    (tmp_path / "notes.dat").write_text("not a measurement file\n")

    lines = catalog.logbook(tmp_path, delimiter="\t")
    names = [line.split("\t")[0] for line in lines]
    assert names == ["early.dat", "later.dat"]  # date order, decoy absent


def test_logbook_aligned_output_pads_columns(tmp_path):
    _write_mh_file(tmp_path / "a_very_long_filename.dat")
    _write_mh_file(tmp_path / "b.dat")
    lines = catalog.logbook(tmp_path)
    assert len(lines) == 2
    # Both dates start at the same column, so the columns are aligned.
    assert lines[0].index("2020-01-29") == lines[1].index("2020-01-29")
    assert "\t" not in lines[0]


def test_logbook_empty_tree(tmp_path):
    assert catalog.logbook(tmp_path) == []
    assert catalog.logbook(tmp_path, delimiter="\t") == []


def test_logbook_paths_renders_exactly_the_given_files(tmp_path):
    """paths= is the catalog bridge: a filtered catalog's relative_path
    column renders just those files, still date-sorted and aligned."""
    _write_mh_file(tmp_path / "keep_a.dat")
    _write_mh_file(tmp_path / "keep_b.dat")
    _write_mh_file(tmp_path / "drop.dat")

    df = catalog.build_catalog(tmp_path)
    kept = df[df["filename"] != "drop.dat"]

    lines = catalog.logbook(tmp_path, paths=kept["relative_path"])
    names = [line.split()[0] for line in lines]
    assert names == ["keep_a.dat", "keep_b.dat"]
    # Alignment survives: both dates start at the same column.
    assert lines[0].index("2020-01-29") == lines[1].index("2020-01-29")


def test_logbook_paths_accepts_absolute_paths(tmp_path):
    path = _write_mh_file(tmp_path / "abs.dat")
    lines = catalog.logbook(tmp_path, paths=[path], delimiter="\t")
    assert len(lines) == 1
    assert lines[0].split("\t")[0] == "abs.dat"


def test_logbook_paths_accepts_a_single_bare_string(tmp_path):
    """A bare str is an iterable of characters; it must be read as one
    path, not walked character by character."""
    _write_mh_file(tmp_path / "solo.dat")
    lines = catalog.logbook(tmp_path, paths="solo.dat", delimiter="\t")
    assert len(lines) == 1
    assert lines[0].split("\t")[0] == "solo.dat"


def test_logbook_paths_unrecognised_file_raises(tmp_path):
    """An explicit paths entry that is not a measurement file raises --
    unlike the walk, which silently skips non-matching files."""
    decoy = tmp_path / "notes.dat"
    decoy.write_text("not a measurement file\n")
    with pytest.raises(calamari.UnsupportedFileError):
        catalog.logbook(tmp_path, paths=["notes.dat"])


def test_logbook_line_setpoint_jitter_reads_as_held(tmp_path):
    """A real held setpoint carries jitter; the cell must not read as a sweep.

    Regression: with exact-equality "held" detection, an H-scan at a nominal
    9 K rendered ``8.9961 < T < 9.00107 K`` -- contradicting the ``H-scan``
    cell on its own line -- and full-precision bounds printed the noise on a
    nominal-zero field endpoint as ``1.20094e-05 T``.
    """
    n = 25
    rng_t = 9.0 + np.linspace(-0.0039, 0.00107, n)  # ~5 mK of jitter
    field = np.linspace(0.0, 60000.0, n)
    field[0] = 0.120094  # 0.12 Oe of noise on the zero endpoint
    path = simulate.write_mpms3_file(
        tmp_path / "jitter.dat",
        app="SQUID VSM",
        temperature_K=rng_t,
        field_oe=field,
        vsm_moment_emu=np.full(n, 1.0e-3),
    )
    cells = catalog.logbook_line(_load(path)).split("\t")
    assert cells[6] == "H-scan"
    assert cells[7] == "T = 8.999 K"  # held within tolerance: midpoint, not a span
    assert cells[8] == "0 < H < 6 T"  # bounds rounded to ~1/100 of the span


# ---------------------------------------------------------------------------
# Catalog: the class
# ---------------------------------------------------------------------------


def _catalog_frame(n=3):
    """A catalog-shaped frame built directly, no files involved.

    Constructing the frame by hand is the point: it proves (and lets tests
    exercise) that Catalog's selection and rendering run on the rows alone.
    Rows are H-scan isotherms at 5, 10, 15, ... K with distinct field
    spans, and deliberately unsorted filenames.
    """
    names = [f"run_{n - i:02d}.dat" for i in range(n)]
    return pd.DataFrame(
        {
            "relative_path": names,
            "group": names,
            "filename": names,
            "data_format": ["MPMS3"] * n,
            "measurement_type": ["DC"] * n,
            "file_role": ["summary"] * n,
            "sample_name": ["SyntheticSample"] * n,
            "sample_mass_mg": [10.0] * n,
            "date": ["2020-01-29"] * n,
            "n_points": [11] * n,
            "n_measurements": [11] * n,
            "T_min": [5.0 * (i + 1) for i in range(n)],
            "T_max": [5.0 * (i + 1) for i in range(n)],
            "H_min_Oe": [-1000.0 * (i + 1) for i in range(n)],
            "H_max_Oe": [1000.0 * (i + 1) for i in range(n)],
            "H_min_T": [-0.1 * (i + 1) for i in range(n)],
            "H_max_T": [0.1 * (i + 1) for i in range(n)],
            "ac_frequencies_Hz": [""] * n,
            "scan_kind": ["H-scan"] * n,
            "max_repeats": [1] * n,
            "comment": [None] * n,
            "is_duplicate": [False] * n,
            "canonical_path": names,
            "dup_status": [""] * n,
        }
    )


def test_catalog_repr_is_the_logbook_and_reads_no_files(tmp_path):
    """The repr must describe the catalog from its rows alone -- proven by
    deleting the tree after the build and rendering anyway."""
    _write_mh_file(tmp_path / "a.dat")
    _write_mh_file(tmp_path / "b.dat")
    cat = catalog.build_catalog(tmp_path)

    for path in cat.paths:
        path.unlink()

    rendered = repr(cat)
    assert rendered == "\n".join(cat.logbook())
    assert "a.dat" in rendered and "b.dat" in rendered
    assert "2020-01-29" in rendered
    # Same cells as the walk-based logbook would have produced (checked
    # against a fresh tree, since this one is gone).
    _write_mh_file(tmp_path / "a.dat")
    _write_mh_file(tmp_path / "b.dat")
    assert catalog.logbook(tmp_path) == cat.logbook()


def test_catalog_repr_truncates_past_the_cap_and_renders_only_the_head():
    n = catalog._CATALOG_REPR_MAX_ROWS + 5
    frame = _catalog_frame(n)
    # Poison a row past the cap: rendering it would raise (a string has no
    # arithmetic span), so a repr that works proves hidden rows are never
    # rendered -- the property that keeps repr cheap on any size of catalog.
    frame["T_min"] = frame["T_min"].astype(object)
    frame.loc[n - 1, "T_min"] = "poison"
    cat = catalog.Catalog(frame, Path("."))

    lines = repr(cat).splitlines()
    assert len(lines) == catalog._CATALOG_REPR_MAX_ROWS + 1
    assert "5 more file(s)" in lines[-1]
    with pytest.raises(TypeError):
        cat.logbook()  # ...whereas logbook() renders all, poison included


def test_catalog_column_access_and_mask_filtering():
    cat = catalog.Catalog(_catalog_frame(3), Path("."))

    assert isinstance(cat["T_min"], pd.Series)
    kept = cat[cat["T_min"] > 7.0]
    assert isinstance(kept, catalog.Catalog)
    assert len(kept) == 2
    assert len(cat) == 3  # original untouched
    assert kept.df.index.tolist() == [0, 1]  # derived catalogs re-index
    assert kept.root == cat.root

    # A column list follows pandas: a DataFrame projection, not a Catalog.
    projection = cat[["filename", "T_min"]]
    assert isinstance(projection, pd.DataFrame)
    assert list(projection.columns) == ["filename", "T_min"]

    # A numpy boolean mask filters like a Series mask does.
    np_kept = cat[(cat["T_min"] > 7.0).to_numpy()]
    assert isinstance(np_kept, catalog.Catalog) and len(np_kept) == 2

    # Non-boolean list-likes are refused by name, never handed to pandas
    # to reinterpret as column selection or a confusing KeyError.
    with pytest.raises(TypeError, match="cat.df.iloc"):
        cat[0]
    with pytest.raises(TypeError, match="non-bools"):
        cat[[0, 1]]
    with pytest.raises(TypeError, match="non-bools"):
        cat[np.array(["filename", "T_min"])]
    with pytest.raises(TypeError, match="non-bools"):
        cat[cat["T_min"]]  # a float Series is not a mask


def test_relative_root_still_yields_absolute_paths(tmp_path, monkeypatch):
    """The iteration contract is absolute paths; a catalog built with a
    relative root must honour it even after the caller chdirs away."""
    _write_mh_file(tmp_path / "here.dat")
    monkeypatch.chdir(tmp_path)
    cat = catalog.build_catalog(".")

    assert cat.root.is_absolute()
    assert all(p.is_absolute() for p in cat)
    monkeypatch.chdir(tmp_path.parent)
    assert all(p.exists() for p in cat.paths)  # still resolvable from elsewhere


def test_logbook_renders_missing_values_blank_not_nan():
    """A NaN mass (pandas' storage for an absent float) must render as a
    blank cell, never the string 'nan'."""
    frame = _catalog_frame(1)
    frame.loc[0, "sample_mass_mg"] = np.nan
    line = catalog.Catalog(frame, Path(".")).logbook()[0]
    assert "nan" not in line
    assert "mg" not in line  # the whole cell is blank, not just the number


def test_where_filters_by_equality_and_none():
    frame = _catalog_frame(3)
    frame.loc[1, "sample_mass_mg"] = np.nan
    cat = catalog.Catalog(frame, Path("."))

    assert len(cat.where(scan_kind="H-scan")) == 3
    assert cat.where(filename="run_02.dat")["T_min"].tolist() == [10.0]
    # None matches the missing-value rows, the "operator left it blank" ask.
    assert cat.where(sample_mass_mg=None)["filename"].tolist() == ["run_02.dat"]
    # Criteria AND together.
    assert len(cat.where(scan_kind="H-scan", filename="run_01.dat")) == 1

    with pytest.raises(KeyError, match="no column 'colour'"):
        cat.where(colour="red")
    with pytest.raises(ValueError, match="no criteria"):
        cat.where()


def test_of_type_matches_composites(tmp_path):
    """The reason of_type exists: where(measurement_type='AC') silently
    misses a combined run labelled 'VSM+AC'."""
    frame = _catalog_frame(3)
    frame.loc[1, "measurement_type"] = "VSM+AC"
    frame.loc[2, "measurement_type"] = "AC"
    cat = catalog.Catalog(frame, Path("."))

    assert cat.of_type("AC")["filename"].tolist() == ["run_02.dat", "run_01.dat"]
    assert cat.of_type("VSM")["filename"].tolist() == ["run_02.dat"]
    assert len(cat.where(measurement_type="AC")) == 1  # the trap, for contrast
    assert len(cat.of_type("SQUID")) == 0  # an ordinary empty answer


def test_of_type_selects_files_not_rows(tmp_path):
    """For combined runs, of_type keeps the
    whole file, so the chi helpers' source='auto' refuses it (AC chi and a
    DC moment both on offer) and source='ac' is the documented resolution --
    it plots only the AC rows, the VSM rows being blank in chi'."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from calamari import plot

    nan = np.nan
    is_ac = np.arange(8) % 2 == 1
    for field in (500.0, 1000.0):
        simulate.write_mpms3_file(
            tmp_path / f"combined_{field:.0f}Oe.dat",
            app="SQUID VSM",
            temperature_K=np.repeat([2.0, 4.0, 6.0, 8.0], 2),
            field_oe=np.full(8, field),
            vsm_moment_emu=np.where(is_ac, nan, 1e-3),
            chi_real=np.where(is_ac, 1e-4, nan),
            ac_frequency_hz=np.where(is_ac, 997.0, nan),
            transport_action=np.where(is_ac, 5.0, 1.0),
        )
    fam = catalog.build_catalog(tmp_path).of_type("AC")

    assert len(fam) == 2
    assert [m.measurement_type for m in fam.load()] == ["VSM+AC", "VSM+AC"]
    try:
        with pytest.raises(ValueError, match="offers both an AC susceptibility"):
            plot.plot_chi_vs_t_family(fam)
        ax = plot.plot_chi_vs_t_family(fam, source="ac")
        assert [len(line.get_xdata()) for line in ax.lines] == [4, 4]
    finally:
        plt.close("all")


def test_at_selects_held_setpoints_not_sweeps_through():
    frame = _catalog_frame(3)  # isotherms at T = 5, 10, 15 K
    # Turn row 0 into a T-scan sweeping 2-20 K: it passes through 10 K but
    # never holds it, so at(T_K=10) must exclude it.
    frame.loc[0, ["T_min", "T_max", "scan_kind"]] = [2.0, 20.0, "T-scan"]
    cat = catalog.Catalog(frame, Path("."))

    assert cat.at(T_K=10.0)["filename"].tolist() == ["run_02.dat"]
    # Tolerance is a real window: a nearby setpoint within tol matches.
    assert cat.at(T_K=9.8, t_tol_K=0.5)["filename"].tolist() == ["run_02.dat"]
    assert len(cat.at(T_K=9.0, t_tol_K=0.5)) == 0
    # Field works the same way: a +-1000 Oe sweep is centred on zero but
    # never *holds* zero, so at the default 10 Oe tolerance it is excluded.
    held = catalog.Catalog(_catalog_frame(1), Path("."))
    assert len(held.at(H_Oe=0.0)) == 0
    with pytest.raises(ValueError, match="needs T_K=, H_Oe="):
        cat.at()


def test_drop_removes_named_files_and_refuses_unknown_names():
    cat = catalog.Catalog(_catalog_frame(3), Path("."))

    kept = cat.drop("run_02.dat")
    assert kept["filename"].tolist() == ["run_03.dat", "run_01.dat"]
    assert len(cat) == 3

    from calamari.exceptions import CatalogSelectionError

    with pytest.raises(CatalogSelectionError, match="nope.dat"):
        cat.drop("nope.dat")
    with pytest.raises(ValueError, match="drops nothing"):
        cat.drop()


def test_sort_by_reorders_rows_and_therefore_the_logbook():
    cat = catalog.Catalog(_catalog_frame(3), Path("."))

    by_t = cat.sort_by("T_min")
    assert by_t["T_min"].tolist() == [5.0, 10.0, 15.0]
    assert by_t.logbook()[0].startswith("run_03.dat")  # row order IS render order
    desc = cat.sort_by("T_min", ascending=False)
    assert desc["T_min"].tolist() == [15.0, 10.0, 5.0]
    assert cat["filename"].tolist() == ["run_03.dat", "run_02.dat", "run_01.dat"]

    with pytest.raises(KeyError, match="no column"):
        cat.sort_by("colour")
    with pytest.raises(ValueError, match="at least one column"):
        cat.sort_by()


def test_iteration_and_load_bridge_back_to_files(tmp_path):
    _write_mh_file(tmp_path / "one.dat")
    _write_mh_file(tmp_path / "two.dat")
    cat = catalog.build_catalog(tmp_path)

    paths = list(cat)
    assert paths == cat.paths
    assert all(p.is_absolute() and p.exists() for p in paths)

    loaded = cat.load()
    assert [m.path.name for m in loaded] == [p.name for p in cat.paths]
    assert all(m.data_format == "MPMS3" for m in loaded)


def test_load_one_is_the_single_file_handoff(tmp_path):
    from calamari.exceptions import CatalogSelectionError

    _write_mh_file(tmp_path / "one.dat")
    _write_mh_file(tmp_path / "two.dat")
    cat = catalog.build_catalog(tmp_path)

    m = cat.where(filename="one.dat").load_one()
    assert m.path.name == "one.dat"

    with pytest.raises(CatalogSelectionError, match="holds 2"):
        cat.load_one()
    with pytest.raises(CatalogSelectionError, match="matched no files"):
        cat.where(filename="gone.dat").load_one()


def test_select_representative_method_matches_the_function():
    frame = _setpoint_frame()
    # _setpoint_frame is selection-shaped, not logbook-shaped; that is fine
    # for everything but rendering.
    cat = catalog.Catalog(frame, Path("."))
    picked = cat.select_representative(
        by="T_K", prefer_span=("H_min_Oe", "H_max_Oe"), tol=0.05
    )
    expected = catalog.select_representative(
        frame, by="T_K", prefer_span=("H_min_Oe", "H_max_Oe"), tol=0.05
    )
    assert isinstance(picked, catalog.Catalog)
    assert picked["relative_path"].tolist() == expected["relative_path"].tolist()


def test_catalog_logbook_delimiter_passthrough():
    cat = catalog.Catalog(_catalog_frame(2), Path("."))
    lines = cat.logbook(delimiter="\t")
    assert len(lines) == 2
    assert lines[0].split("\t")[0] == "run_02.dat"


def test_selection_methods_chain(tmp_path):
    """The workflow the class exists for, end to end on real files."""
    scans = tmp_path / "scans"
    scans.mkdir()
    _write_mh_file(scans / "keep.dat")
    _write_mh_file(scans / "unwanted.dat")
    _write_mh_file(tmp_path / "other.dat")

    cat = catalog.build_catalog(tmp_path)
    picked = cat.where(group="scans").drop("unwanted.dat").sort_by("filename")
    assert picked["filename"].tolist() == ["keep.dat"]
    assert picked.load_one().path.name == "keep.dat"
