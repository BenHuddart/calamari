"""Tests for calamari.scales: the smoothing-window scan and its feature tracks.

All data is synthetic (:mod:`calamari.simulate`, invented numbers), with a
planted slope change whose location is known, so the tests assert how a
feature behaves across windows rather than recorded reference numbers.
Each test or campaign fixture uses its own explicit seeded generator.
"""

from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import calamari
from calamari import scales, simulate, transitions
from calamari.exceptions import SmoothingSensitivityError
from calamari.io import load
from calamari.plot import _detection as plot_detection
from calamari.scales import MEMBER_COLUMNS, TRACK_COLUMNS, SmoothingScanResult, scan_smoothing

#: The declared windows most tests scan, in K: 5, 7, 11 and 15 points on the
#: 0.2 K grid of :func:`_tscan`.
WINDOWS = (1.0, 1.5, 2.0, 3.0)

#: The swept temperatures of every T-scan fixture: 141 points over 2-30 K,
#: a 0.2 K step.
T_GRID = np.linspace(2.0, 30.0, 141)

#: Per-point noise of the fixtures, in the (invented) arbitrary units x 1e-6.
NOISE = 0.004

#: Planted kinks: (held field in T, slope-change temperature in K).
KINKS = ((0.5, 16.0), (1.0, 15.0), (1.5, 14.0))

DETECT = {"criterion": "kink", "methods": ("peak", "derivative", "kink")}


def _kink(t, t_k, lo=0.4, hi=0.1, round_k=0.3):
    """A one-sided slope change ('down'): slope ``lo`` below ``t_k``, ``hi``
    above, the corner rounded over ``round_k`` by a softplus."""
    return lo * (t - t_k) + (hi - lo) * round_k * np.logaddexp(0, (t - t_k) / round_k)


def _tscan(path: Path, y: np.ndarray, h_t: float, t=T_GRID) -> Path:
    return simulate.write_mpms3_file(
        path, app="SQUID AC", temperature_K=t, field_oe=np.full_like(t, h_t * 1e4),
        chi_real=y * 1e-6, ac_frequency_hz=np.full_like(t, 997.0),
    )


def _kink_family(tmp_path: Path, seed: int = 20260928) -> list:
    rng = np.random.default_rng(seed)
    sweeps = []
    for i, (h_t, t_k) in enumerate(KINKS):
        y = 5.0 + _kink(T_GRID, t_k) + rng.normal(scale=NOISE, size=T_GRID.shape)
        sweeps.append(load(_tscan(tmp_path / f"kink_{i}.dat", y, h_t)))
    return sweeps


def _noise_family(tmp_path: Path, n: int = 6, seed: int = 20260929) -> list:
    rng = np.random.default_rng(seed)
    sweeps = []
    for i in range(n):
        y = 5.0 + 0.1 * (T_GRID - 2.0) + rng.normal(scale=NOISE, size=T_GRID.shape)
        sweeps.append(load(_tscan(tmp_path / f"noise_{i}.dat", y, 0.25 * i)))
    return sweeps


def _hscan(path: Path, rng, h_c_t: float = 2.0, fixed_t_k: float = 5.0) -> Path:
    h_oe = np.linspace(0.0, 50_000.0, 141)
    mom = 0.5 * (1 + np.tanh((h_oe / 1e4 - h_c_t) / 0.15)) + rng.normal(
        scale=0.005, size=h_oe.shape
    )
    return simulate.write_mpms3_file(
        path, app="SQUID", temperature_K=np.full_like(h_oe, fixed_t_k), field_oe=h_oe,
        dc_moment_emu=mom,
    )


@pytest.fixture(scope="module")
def kink_family(tmp_path_factory):
    return _kink_family(tmp_path_factory.mktemp("kinks"))


@pytest.fixture(scope="module")
def kink_scan(kink_family):
    return scan_smoothing(kink_family, windows=WINDOWS, track_tol_k=0.5, **DETECT)


def _planted_tracks(scan: SmoothingScanResult) -> pd.DataFrame:
    """Per sweep, the kink track whose median sits nearest the planted slope change."""
    tracks = scan.tracks[scan.tracks["method"] == "kink"]
    rows = []
    for sweep_id, (_, t_k) in enumerate(KINKS):
        own = tracks[tracks["sweep_id"] == sweep_id]
        rows.append(own.loc[(own["T_K_median"] - t_k).abs().idxmin()])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# What the scan finds.
# ---------------------------------------------------------------------------


def test_planted_kink_is_tracked_across_every_window(kink_scan):
    """The slope change is found at every declared window, in one place and
    with one sign. Measured on this fixture: spread 0.11-0.22 K on a 0.2 K
    grid, per-window significance 1.2-1.6 at 1 K rising to 11-13 at 3 K."""
    planted = _planted_tracks(kink_scan)
    for (_, t_k), (_, track) in zip(KINKS, planted.iterrows(), strict=True):
        assert track["n_windows"] == len(WINDOWS)
        assert track["n_present"] == len(WINDOWS)
        assert track["persistence"] == 1.0
        assert abs(track["T_K_median"] - t_k) < 0.3
        assert track["T_K_spread"] < 0.5  # 2.5 grid steps
        assert track["sign_mode"] == "down"
        assert track["sign_consistency"] == 1.0
        assert not track["edge_any"]
        # The range is reported, and it is wide: exactly why no single
        # window's value may be quoted as "the" significance.
        assert track["significance_min"] < 3.0 < 5.0 < track["significance_max"]
        assert track["significance_min"] <= track["significance_median"]
        assert track["significance_median"] <= track["significance_max"]
        # The H-coordinated columns are NaN on a T-swept track, and the
        # held field places the sweep.
        assert math.isnan(track["T_K"])
        assert track["H_T"] == pytest.approx(KINKS[int(track["sweep_id"])][0])
        assert all(math.isnan(track[f"H_T_{s}"]) for s in ("median", "min", "max", "spread"))


def test_members_address_each_windows_result(kink_scan):
    """Every member row points at a real candidate of its own window's result,
    with that candidate's location, sign and significance."""
    members = kink_scan.members
    assert list(members.columns) == MEMBER_COLUMNS
    assert list(kink_scan.tracks.columns) == TRACK_COLUMNS
    for _, row in members.iterrows():
        cands = kink_scan.at(row["window"]).candidates
        cand = cands[cands["candidate_id"] == row["candidate_id"]].iloc[0]
        assert cand["T_K"] == row["T_K"]
        assert cand["sign"] == row["sign"]
        assert cand["significance"] == row["significance"]
        assert cand["sweep_id"] == row["sweep_id"]
    # Every candidate of every window is in exactly one track.
    n_candidates = sum(len(r.candidates) for r in kink_scan.results.values())
    assert len(members) == n_candidates
    assert not members.duplicated(["window", "candidate_id"]).any()
    # A track holds at most one member per window.
    assert not members.duplicated(["track_id", "window"]).any()
    sizes = members.groupby("track_id").size()
    assert (sizes.sort_index().to_numpy() == kink_scan.tracks["n_present"].to_numpy()).all()


def test_noise_only_sweeps_persist_too_and_stay_insignificant(tmp_path):
    """Persistence is not evidence: pure noise makes persistent tracks.

    Measured on this fixture (6 sloping noise sweeps, windows 1-3 K, 0.5 K
    tolerance): 1117 tracks, median persistence 0.5, mean 0.53, and 249
    tracks (22 %) present at all four windows -- 76 of them with one sign
    throughout -- while no track exceeds significance 1.6 at any window.
    On 40 such sweeps, 14 % of kink tracks and 30 % of derivative tracks
    were present at every window, and the fraction grew with the tolerance
    (4 % and 11 % at 0.2 K, 26 % and 38 % at 1 K). A noise excursion is a
    feature of the data, and every window smooths the same data, so it is
    found again. What separates the planted kink from these is its
    significance, not its persistence. The bounds below are generous
    against the measured values.
    """
    noise = scan_smoothing(_noise_family(tmp_path), windows=WINDOWS, track_tol_k=0.5, **DETECT)
    tracks = noise.tracks
    persistent = tracks["persistence"] == 1.0
    assert tracks["persistence"].mean() < 0.7
    assert persistent.mean() < 0.4
    assert persistent.sum() > 0  # the trap, pinned: noise persists too
    assert tracks["significance_max"].max() < 5.0


def test_no_combined_statistic_and_no_best_window(kink_scan):
    """The design rule, pinned: the scan never combines per-window
    significances and never names a favoured window."""
    for column in kink_scan.tracks.columns:
        assert "best" not in column
        assert "stouffer" not in column
        assert column != "chain_significance"
    stats = [c for c in kink_scan.tracks.columns if c.startswith("significance")]
    assert stats == ["significance_min", "significance_median", "significance_max"]


# ---------------------------------------------------------------------------
# Windows that resolve to one analysis.
# ---------------------------------------------------------------------------


def _short_sweep(tmp_path: Path) -> list:
    """21 points, 1 K apart: every window under 6 K resolves to 5 points."""
    t = np.linspace(2.0, 22.0, 21)
    rng = np.random.default_rng(20260930)
    y = 5.0 + _kink(t, 12.0, round_k=0.5) + rng.normal(scale=NOISE, size=t.shape)
    return [load(_tscan(tmp_path / "short.dat", y, 0.5, t=t))]


def test_windows_that_clamp_together_warn_and_count_once(tmp_path):
    """2, 4 and 5 K all resolve to 5 points on a 1 K grid (the floor, and
    the odd-count rounding); 30 K is clamped to the sweep's 21. Without the
    correction a feature present at all four would report persistence 4/2."""
    with pytest.warns(UserWarning) as caught:
        scan = scan_smoothing(_short_sweep(tmp_path), windows=(2.0, 4.0, 5.0, 30.0),
                              track_tol_k=1.5, **DETECT)
    messages = [str(w.message) for w in caught if "resolve to" in str(w.message)]
    assert len(messages) == 1  # once per affected sweep
    (message,) = messages
    assert "sweep_id=0" in message
    assert "[2.0, 4.0, 5.0]" in message
    assert "5 points" in message
    assert "short.dat" in message

    (entry,) = scan.params["smooth_points"]
    assert entry["smooth_points"] == {2.0: 5, 4.0: 5, 5.0: 5, 30.0: 21}
    assert entry["n_windows"] == 2
    (group,) = scan.params["resolved_together"]
    assert group["windows"] == (2.0, 4.0, 5.0)
    assert group["smooth_points"] == 5

    tracks = scan.tracks
    assert (tracks["n_windows"] == 2).all()
    assert (tracks["persistence"] <= 1.0).all()
    assert (tracks["n_present"] <= 2).all()
    # The clamped windows' identical detections share a track, so the
    # members table keeps every hit while the track counts the analysis once.
    at_5 = scan.members[scan.members["smooth_points"] == 5]
    per_track = at_5.groupby("track_id")["window"].apply(tuple)
    assert (per_track == (2.0, 4.0, 5.0)).all()
    assert "resolved together" in repr(scan)


def test_distinct_windows_do_not_warn(kink_family):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scan = scan_smoothing(kink_family[:1], windows=WINDOWS, track_tol_k=0.5, **DETECT)
    assert scan.params["resolved_together"] == []
    (entry,) = scan.params["smooth_points"]
    assert entry["smooth_points"] == {1.0: 5, 1.5: 7, 2.0: 11, 3.0: 15}
    assert entry["n_windows"] == 4


def test_detection_records_the_resolved_point_count(tmp_path):
    """The per-signal record carries ``smooth_points`` beside ``smooth_window``;
    the scan reads its clamp check from it."""
    sweeps = _kink_family(tmp_path)
    single = transitions.detect_transitions(sweeps[0], smooth_window=2.0)
    record = single.params["per_signal"]["chi_real"]
    assert record["smooth_points"] == 11
    assert record["smooth_window"] == pytest.approx(11 * 0.2)
    batch = transitions.detect_transitions_all(sweeps, smooth_window=1.5)
    assert [e["per_signal"]["chi_real"]["smooth_points"] for e in batch.params["per_sweep"]] == [
        7, 7, 7,
    ]


# ---------------------------------------------------------------------------
# Refusals, each naming the fix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"smooth_window": 2.0}, r"smooth_window is set by the scan.*windows="),
        ({"windows": (1.0, 2.0)}, r"at least 3.*windows=\(1.0, 1.5, 2.0\)"),
        ({"windows": 2.0}, r"must be a sequence.*windows=\(1.0, 1.5, 2.0\)"),
        ({"windows": "123"}, r"must be a sequence"),
        ({"windows": (1.0, 0.0, 2.0)}, r"positive and finite.*0.0"),
        ({"windows": (1.0, -1.0, 2.0)}, r"positive and finite.*-1.0"),
        ({"windows": (1.0, float("nan"), 2.0)}, r"positive and finite.*nan"),
        ({"windows": (1.0, float("inf"), 2.0)}, r"positive and finite.*inf"),
        ({"windows": (1.0, True, 2.0)}, r"must be a number"),
        ({"windows": (1.0, 2.0, 2)}, r"repeats \[2.0\].*pass each window once"),
        ({"track_tol_k": None}, r"track_tol_k is None.*no default.*track_tol_k= in K"),
        ({"track_tol_k": 0.0}, r"track_tol_k must be positive"),
        ({"track_tol_k": float("nan")}, r"track_tol_k must be positive"),
        ({"track_tol_t": -1.0}, r"track_tol_t must be positive"),
        ({"link": 0.8}, r"link must be a dict.*t_tol_k"),
    ],
)
def test_refusals_name_the_fix(kink_family, kwargs, match):
    call = {"windows": WINDOWS, "track_tol_k": 0.5, **DETECT, **kwargs}
    with pytest.raises(SmoothingSensitivityError, match=match):
        scan_smoothing(kink_family[:1], **call)


def test_h_swept_scan_needs_its_own_tolerance_and_reports_in_tesla(tmp_path):
    """An isotherm's feature is located in Tesla: the T-tolerance does not
    cover it, and its track summarises the H_T columns."""
    rng = np.random.default_rng(20261001)
    sweeps = [load(_hscan(tmp_path / "mh.dat", rng))]
    windows = (0.2, 0.3, 0.5)
    with pytest.raises(SmoothingSensitivityError, match=r"swept along H.*track_tol_t is None.*in T"):
        scan_smoothing(sweeps, windows=windows, track_tol_k=0.5)
    scan = scan_smoothing(sweeps, windows=windows, track_tol_t=0.1)
    step = scan.tracks.loc[scan.tracks["significance_median"].idxmax()]
    assert step["swept_axis"] == "H"
    assert step["method"] == "derivative"
    assert step["persistence"] == 1.0
    assert step["H_T_median"] == pytest.approx(2.0, abs=0.1)
    assert step["T_K"] == pytest.approx(5.0)
    assert math.isnan(step["H_T"])
    assert math.isnan(step["T_K_median"])


# ---------------------------------------------------------------------------
# Determinism and the recipe.
# ---------------------------------------------------------------------------


def test_window_order_does_not_change_the_tables(kink_family):
    reference = scan_smoothing(kink_family, windows=WINDOWS, track_tol_k=0.5, **DETECT)
    for order in list(itertools.permutations(WINDOWS))[1::7]:
        scan = scan_smoothing(kink_family, windows=order, track_tol_k=0.5, **DETECT)
        pd.testing.assert_frame_equal(scan.tracks, reference.tracks)
        pd.testing.assert_frame_equal(scan.members, reference.members)
        assert list(scan.results) == list(WINDOWS)
        assert scan.params["windows"] == order
        assert scan.params["windows_sorted"] == WINDOWS


def test_params_record_the_recipe(kink_scan):
    params = kink_scan.params
    assert params["windows"] == WINDOWS
    assert params["windows_sorted"] == WINDOWS
    assert params["track_tol_k"] == 0.5
    assert params["track_tol_t"] is None
    assert params["track_by"] == ("sweep_id", "signal", "method")
    assert params["detect_kwargs"] == DETECT
    assert params["link"] is None
    assert params["calamari_version"]
    assert [e["sweep_id"] for e in params["smooth_points"]] == [0, 1, 2]
    for window, result in kink_scan.results.items():
        assert result.params["detect_kwargs"]["smooth_window"] == window
        assert result.params["detect_kwargs"]["criterion"] == "kink"


def test_input_is_read_once_and_a_single_measurement_is_a_batch(kink_family):
    from_generator = scan_smoothing(
        (m for m in kink_family), windows=WINDOWS, track_tol_k=0.5, **DETECT
    )
    listed = scan_smoothing(kink_family, windows=WINDOWS, track_tol_k=0.5, **DETECT)
    pd.testing.assert_frame_equal(from_generator.tracks, listed.tracks)
    single = scan_smoothing(kink_family[0], windows=WINDOWS, track_tol_k=0.5, **DETECT)
    assert set(single.tracks["sweep_id"]) == {0}
    pd.testing.assert_frame_equal(
        single.tracks.drop(columns="track_id").reset_index(drop=True),
        listed.tracks[listed.tracks["sweep_id"] == 0]
        .drop(columns="track_id")
        .reset_index(drop=True),
    )


def test_link_carries_each_windows_chains(kink_family):
    link = {"t_tol_k": 1.5}
    scan = scan_smoothing(kink_family, windows=WINDOWS, track_tol_k=0.5, link=link, **DETECT)
    assert scan.params["link"] == link
    assert list(scan.members.columns) == MEMBER_COLUMNS + ["chain_id", "chain_significance"]
    assert list(scan.tracks.columns) == TRACK_COLUMNS + [
        "chain_significance_min", "chain_significance_median",
    ]
    assert scan.members["chain_id"].notna().all()
    for window, result in scan.results.items():
        assert result.params["link"]["t_tol_k"] == 1.5
        chained = result.candidates.set_index("candidate_id")
        mine = scan.members[scan.members["window"] == window]
        assert (mine["chain_id"].to_numpy()
                == chained.loc[mine["candidate_id"], "chain_id"].to_numpy()).all()
        assert (mine["chain_significance"].to_numpy()
                == chained.loc[mine["candidate_id"], "chain_significance"].to_numpy()).all()
    # The planted boundary chains across the three sweeps at every window;
    # the track reports the range of that chain statistic, never a sum.
    planted = _planted_tracks(scan)
    for _, track in planted.iterrows():
        rows = scan.members[scan.members["track_id"] == track["track_id"]]
        assert track["chain_significance_min"] == rows["chain_significance"].min()
        assert track["chain_significance_median"] == rows["chain_significance"].median()
    # Unlinked, the chain columns are absent rather than NaN.
    unlinked = scan_smoothing(kink_family, windows=WINDOWS, track_tol_k=0.5, **DETECT)
    assert "chain_id" not in unlinked.members.columns


def test_robust_forwarded_filters_what_is_tracked(kink_family):
    """A robust= in the detection keywords filters every window first; the
    tracks then describe what the filter kept, and say so by their counts."""
    scan = scan_smoothing(kink_family, windows=WINDOWS, track_tol_k=0.5,
                          robust={"min_significance": 5.0}, **DETECT)
    for window, result in scan.results.items():
        assert result.params["robust"]["min_significance"] == 5.0
        assert (scan.members.loc[scan.members["window"] == window, "significance"] >= 5).all()
    planted = _planted_tracks(scan)
    # Below 5 at the narrow windows, so the kink is absent there.
    assert (planted["n_present"] < len(WINDOWS)).all()


def test_at_is_an_exact_lookup(kink_scan):
    assert kink_scan.at(2.0) is kink_scan.results[2.0]
    assert kink_scan.at(2) is kink_scan.results[2.0]
    for bad in (2.5, 2.0000001, "2.0x", None):
        with pytest.raises(SmoothingSensitivityError, match=r"ran windows \[1.0, 1.5, 2.0, 3.0\].*exactly"):
            kink_scan.at(bad)


@pytest.mark.parametrize("axis", ["T", "H"])
def test_sensitivity_summary_counts_representations_and_describes_intervals(axis):
    """A duplicate representation cannot outvote a supported polarity change."""
    members = pd.DataFrame({
        "window": [1.0, 1.1, 2.0, 3.0], "smooth_points": [5, 5, 9, 15],
        "T_K": [10.0, 10.0, 10.2, 10.4], "H_T": [1.0, 1.0, 1.2, 1.4],
        "T_err_K": [0.3] * 4, "H_err_T": [0.3] * 4,
        "sign": ["up", "up", "down", "up"], "edge": [False, False, False, True],
    })
    summary = scales._sensitivity_summary(members, axis, 4)
    assert summary["n_present"] == 3
    assert summary["persistence"] == 0.75
    assert summary["sign_consistency"] == pytest.approx(2 / 3)
    assert summary["T_K_spread" if axis == "T" else "H_T_spread"] == pytest.approx(0.4)
    assert summary["interval_overlap"] is True
    assert summary["edge_any"] is True
    members.loc[3, "T_K" if axis == "T" else "H_T"] += 1.0
    assert scales._sensitivity_summary(members, axis, 4)["interval_overlap"] is False
    empty = scales._sensitivity_summary(members.iloc[:0], axis, 4)
    assert empty["n_present"] == 0
    assert math.isnan(empty["sign_consistency"])
    assert math.isnan(empty["T_K_spread" if axis == "T" else "H_T_spread"])
    assert empty["interval_overlap"] is None


def test_repr_states_the_windows_and_counts(kink_scan):
    text = repr(kink_scan)
    assert text.startswith("SmoothingScanResult(windows [1, 1.5, 2, 3], ")
    assert f"{len(kink_scan.tracks)} tracks" in text
    assert f"{len(kink_scan.members)} members" in text


def test_reexported_from_calamari():
    assert calamari.scan_smoothing is scan_smoothing
    assert calamari.SmoothingScanResult is SmoothingScanResult
    assert "scan_smoothing" in calamari.__all__


# ---------------------------------------------------------------------------
# The shared chain walk: link() is unchanged by the refactor.
# ---------------------------------------------------------------------------


def _reference_link_candidates(cands, by, t_tol_k, h_tol_t):
    """``_link_candidates`` exactly as it stood before the walk was factored
    out for the scan (main at 7135f24), frozen here so the refactor is held
    to byte-identical output. Do not edit it to make a test pass."""
    from calamari.transitions._results import _stouffer as stouffer
    from calamari.transitions._results import _unique_or_none as unique_or_none
    geometry = {"T": ("H_T", "T_K"), "H": ("T_K", "H_T")}
    n_rows = len(cands)
    if n_rows == 0:
        return np.array([], dtype=int), pd.DataFrame(columns=transitions.CHAIN_COLUMNS)
    axis_col = cands["swept_axis"].to_numpy()
    sig = cands["significance"].to_numpy(dtype=float)
    t_vals = cands["T_K"].to_numpy(dtype=float)
    h_vals = cands["H_T"].to_numpy(dtype=float)
    columns = {"T_K": t_vals, "H_T": h_vals}
    tolerances = {"T": t_tol_k, "H": h_tol_t}
    fam = np.full(n_rows, np.nan)
    loc = np.full(n_rows, np.nan)
    for axis, (fam_col, loc_col) in geometry.items():
        on_axis = axis_col == axis
        fam[on_axis] = columns[fam_col][on_axis]
        loc[on_axis] = columns[loc_col][on_axis]
    sweep_index = np.full(n_rows, -1)
    n_sweeps = {}
    for axis in geometry:
        on_axis = (axis_col == axis) & np.isfinite(fam) & np.isfinite(loc)
        values = sorted({float(v) for v in fam[on_axis]})
        n_sweeps[axis] = len(values)
        order = {v: i for i, v in enumerate(values)}
        sweep_index[on_axis] = [order[float(v)] for v in fam[on_axis]]
    if by:
        by_columns = [cands[c].astype(str).to_numpy() for c in by]
        group_keys = [tuple(col[i] for col in by_columns) for i in range(n_rows)]
    else:
        group_keys = [() for _ in range(n_rows)]
    groups = {}
    unplaceable = []
    for i in range(n_rows):
        if sweep_index[i] < 0:
            unplaceable.append(i)
            continue
        groups.setdefault((*group_keys[i], str(axis_col[i])), []).append(i)

    def seed_key(i):
        s = sig[i]
        return (math.inf if math.isnan(s) else -s, fam[i], loc[i], i)

    chains = [[i] for i in unplaceable]
    for key, members in sorted(groups.items(), key=lambda kv: kv[0]):
        axis = key[-1]
        tol = float(tolerances[axis])
        by_sweep = {}
        for i in members:
            by_sweep.setdefault(int(sweep_index[i]), []).append(i)
        claimed = set()
        for seed in sorted(members, key=seed_key):
            if seed in claimed:
                continue
            claimed.add(seed)
            chain = [seed]
            for step in (1, -1):
                ref, fi = loc[seed], int(sweep_index[seed])
                while True:
                    fi += step
                    if not (0 <= fi < n_sweeps[axis]):
                        break
                    pool = [i for i in by_sweep.get(fi, ()) if i not in claimed]
                    if not pool:
                        break
                    best = min(pool, key=lambda i: (abs(loc[i] - ref), loc[i], i))
                    if abs(loc[best] - ref) > tol:
                        break
                    claimed.add(best)
                    chain.append(best)
                    ref = loc[best]
            chains.append(sorted(chain, key=lambda i: (fam[i], loc[i], i)))

    def finite_or_last(value):
        return math.inf if math.isnan(value) else value

    def chain_key(members):
        combined = stouffer(sig[members])
        start = members[0]
        return (
            math.inf if math.isnan(combined) else -combined,
            finite_or_last(fam[start]),
            finite_or_last(loc[start]),
            start,
        )

    def summarise(values, how):
        present = values[~np.isnan(values)]
        return float(how(present)) if present.size else float("nan")

    chains.sort(key=chain_key)
    chain_ids = np.full(n_rows, -1, dtype=int)
    rows = []
    for cid, members in enumerate(chains):
        chain_ids[members] = cid
        member_sig = sig[members]
        rows.append(
            {
                "chain_id": cid,
                "n": len(members),
                "signal": unique_or_none(cands["signal"].to_numpy()[members])
                if "signal" in cands.columns
                else None,
                "method": unique_or_none(cands["method"].to_numpy()[members]),
                "sign": unique_or_none(cands["sign"].to_numpy()[members]),
                "swept_axis": unique_or_none(axis_col[members]),
                "T_K_min": summarise(t_vals[members], np.min),
                "T_K_max": summarise(t_vals[members], np.max),
                "H_T_min": summarise(h_vals[members], np.min),
                "H_T_max": summarise(h_vals[members], np.max),
                "significance_median": summarise(member_sig, np.median),
                "significance_max": summarise(member_sig, np.max),
                "chain_significance": stouffer(member_sig),
            }
        )
    return chain_ids, pd.DataFrame(rows, columns=transitions.CHAIN_COLUMNS)


def _random_candidates(seed: int, n: int = 400) -> pd.DataFrame:
    """A dense, awkward candidates table: both axes, coarse locations (so
    distances tie), repeated significances, NaN locations and significances."""
    rng = np.random.default_rng(seed)
    axis = rng.choice(["T", "H"], size=n)
    held = rng.choice([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], size=n)
    swept = np.round(rng.uniform(2.0, 12.0, size=n), 1)
    sig = np.round(rng.exponential(2.0, size=n), 1)
    sig[rng.random(n) < 0.03] = np.nan
    swept[rng.random(n) < 0.03] = np.nan
    return pd.DataFrame(
        {
            "T_K": np.where(axis == "T", swept, held * 4),
            "H_T": np.where(axis == "T", held, swept / 4),
            "swept_axis": axis,
            "signal": rng.choice(["chi_real", "chi_imag"], size=n),
            "method": rng.choice(["peak", "derivative", "kink"], size=n),
            "sign": rng.choice(["max", "min"], size=n),
            "significance": sig,
        }
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize(
    "by", [("signal", "method", "sign"), ("signal", "method"), ()], ids=["default", "nosign", "none"]
)
def test_refactored_walk_matches_the_frozen_link(seed, by):
    from calamari.transitions._results import _link_candidates

    cands = _random_candidates(seed)
    for t_tol, h_tol in ((0.3, 0.1), (1.0, 0.4)):
        ids, table = _link_candidates(cands, by, t_tol, h_tol)
        ref_ids, ref_table = _reference_link_candidates(cands, by, t_tol, h_tol)
        np.testing.assert_array_equal(ids, ref_ids)
        pd.testing.assert_frame_equal(table, ref_table, check_exact=True)


def test_link_on_a_detected_family_matches_the_frozen_link(tmp_path):
    """link() end to end on a detected family, against the frozen walk."""
    sweeps = _kink_family(tmp_path) + _noise_family(tmp_path, n=3)
    for window in (1.0, 2.0):
        result = transitions.detect_transitions_all(sweeps, smooth_window=window, **DETECT)
        linked = result.link(t_tol_k=0.8)
        ref_ids, ref_table = _reference_link_candidates(
            result.candidates, ("signal", "method", "sign"), 0.8, None
        )
        np.testing.assert_array_equal(linked.candidates["chain_id"].to_numpy(), ref_ids)
        pd.testing.assert_frame_equal(linked.chains, ref_table, check_exact=True)


def test_reference_preserves_native_identity_and_complete_recipe(kink_scan):
    import json

    ref = kink_scan.reference(2.0, min_member_significance=3.0)
    original = kink_scan.at(2.0)
    pd.testing.assert_frame_equal(ref.candidates[original.candidates.columns], original.candidates)
    pd.testing.assert_frame_equal(ref.detections, original.detections)
    for row in ref.candidates.itertuples():
        member = kink_scan.members.query('window == 2.0 and candidate_id == @row.candidate_id')
        assert len(member) == 1
        assert row.smoothing_track_id == member.iloc[0].track_id
    recipe = ref.params['smoothing']
    assert recipe['reference_window'] == 2.0
    assert recipe['min_member_significance'] == 3.0
    assert recipe['smooth_points'] == kink_scan.params['smooth_points']
    pd.testing.assert_frame_equal(pd.DataFrame(recipe['members']), kink_scan.members)
    assert len(recipe['results']) == len(WINDOWS)
    assert json.loads(json.dumps(recipe))['reference_window'] == 2.0
    assert ref.sensitivity_audit.retained.all()
    assert ref.sensitivity_audit.sensitivity_pass.isna().all()
    # The snapshot owns its recipe and does not mutate the scan.
    recipe['windows'] = ()
    assert kink_scan.params['windows'] == WINDOWS


def test_reference_filters_restore_and_compose_with_selection_and_link(kink_scan):
    ref = kink_scan.reference(2.0)
    assert (ref.candidates.smoothing_n_present < 4).any()  # transient detections
    gates = dict(min_significance=0, exclude_edge=False, min_smoothing_n=4,
                 max_smoothing_spread_k=0.5, min_smoothing_sign_consistency=1.0)
    tight = ref.robust(**{**gates, 'min_smoothing_n': 100})
    assert tight.candidates.empty
    restored = tight.robust(**gates)
    assert len(restored.candidates)
    pd.testing.assert_frame_equal(restored.candidates, ref.robust(**gates).candidates)
    pd.testing.assert_frame_equal(tight.all().candidates, ref.candidates)
    assert 'robust' not in tight.all().params
    assert tight.all().params['smoothing'] == ref.params['smoothing']
    selection = dict(swept_axis='T', method='kink', sign='down', t_window_k=(2, 30))
    a = ref.select(**selection).robust(**gates).link(t_tol_k=1.5)
    b = ref.link(t_tol_k=1.5).robust(**gates).select(**selection)
    pd.testing.assert_frame_equal(a.candidates, b.candidates)
    pd.testing.assert_frame_equal(a.sensitivity_audit, b.sensitivity_audit)
    assert len(a.all().candidates) == len(ref.candidates)
    audit = restored.sensitivity_audit
    assert len(audit) == len(ref.candidates)
    assert (~audit.retained).any()
    assert audit.loc[~audit.sensitivity_pass.fillna(False), 'sensitivity_reasons'].str.len().gt(0).all()
    assert audit.to_json() == restored.sensitivity_audit.to_json()


def test_qualification_separates_weak_and_supported_polarity_changes(kink_scan):
    from dataclasses import replace

    members = kink_scan.members.copy()
    track_id = int(_planted_tracks(kink_scan).iloc[0].track_id)
    own = members.track_id.eq(track_id)
    members.loc[own, 'sign'] = 'down'
    members.loc[own, 'significance'] = 10.0
    opposite = own & members.window.eq(1.0)
    members.loc[opposite, 'sign'] = 'up'
    members.loc[opposite, 'significance'] = 1.0
    scan = replace(kink_scan, members=members)
    ref = scan.reference(2.0, min_member_significance=5.0)
    row = ref.candidates.query('smoothing_track_id == @track_id').iloc[0]
    assert row.smoothing_raw_sign_consistency == 0.75
    assert row.smoothing_sign_consistency == 1.0
    assert row.smoothing_n_present == 3
    cid = row.candidate_id
    gates = dict(min_significance=0, exclude_edge=False, min_smoothing_sign_consistency=1)
    assert cid in ref.robust(**gates).candidates.candidate_id.values
    members.loc[opposite, 'significance'] = 10.0
    supported = replace(scan, members=members).reference(2.0, min_member_significance=5)
    assert cid not in supported.robust(**gates).candidates.candidate_id.values
    empty = scan.reference(2.0, min_member_significance=1e100)
    assert set(empty.candidates.smoothing_status) == {'no_qualified_members'}
    filtered = empty.robust(**gates)
    assert filtered.candidates.empty
    assert filtered.sensitivity_audit.sensitivity_pass.isna().all()
    assert set(filtered.sensitivity_audit.sensitivity_reasons) == {'no_qualified_members'}


def test_reference_duplicates_insufficient_windows_and_missing_tracks(tmp_path, kink_scan):
    from dataclasses import replace

    with pytest.warns(UserWarning):
        scan = scan_smoothing(_short_sweep(tmp_path), windows=(2, 4, 5, 30),
                              track_tol_k=1.5, **DETECT)
    ref = scan.reference(4)
    assert len(ref.candidates)
    assert (ref.candidates.smoothing_n_windows == 2).all()
    assert (ref.candidates.smoothing_n_present <= 2).all()
    assert set(ref.candidates.smoothing_status) == {'insufficient_windows'}
    assert ref.robust(min_significance=0, exclude_edge=False, min_smoothing_n=1).candidates.empty
    # A scan made with robust= can leave full reference candidates untracked.
    missing = replace(kink_scan, members=kink_scan.members.iloc[:0]).reference(2)
    assert set(missing.candidates.smoothing_status) == {'missing_track'}
    assert missing.robust(min_smoothing_persistence=0).candidates.empty
    assert set(missing.robust(min_smoothing_n=1).sensitivity_audit.sensitivity_reasons) == {
        'missing_track'
    }
    # No reference detection is invented from tracks at another window.
    result = kink_scan.at(2)
    empty = replace(result, candidates=result.candidates.iloc[:0],
                    detections=result.detections.iloc[:0])
    no_reference = replace(kink_scan, results={**kink_scan.results, 2.0: empty}).reference(2)
    assert no_reference.candidates.empty
    assert no_reference.sensitivity_audit.empty
    assert no_reference.robust(min_smoothing_n=2).candidates.empty


@pytest.mark.parametrize('axis', ['T', 'H'])
def test_sensitivity_location_and_interval_gates(kink_scan, axis):
    from dataclasses import replace

    # Start with native scan identities; control the diagnostic coordinates only,
    # so drift and interval compatibility can be tested independently of a detector.
    members = kink_scan.members.copy()
    track_id = int(_planted_tracks(kink_scan).iloc[0].track_id)
    own = members.track_id.eq(track_id)
    members.loc[own, 'T_K'] = members.loc[own, 'window']
    members.loc[own, 'H_T'] = members.loc[own, 'window']
    members.loc[own, ['T_err_K', 'H_err_T']] = 0.01
    results = {}
    for window, result in kink_scan.results.items():
        cands = result.candidates.copy()
        cands['swept_axis'] = axis
        results[window] = replace(result, candidates=cands)
    ref = replace(kink_scan, members=members, results=results).reference(2)
    row = ref.candidates.query('smoothing_track_id == @track_id').iloc[0]
    gates = dict(min_significance=0, exclude_edge=False)
    name = 'max_smoothing_spread_k' if axis == 'T' else 'max_smoothing_spread_t'
    assert row.candidate_id not in ref.robust(**gates, **{name: 0.1}).candidates.candidate_id.values
    assert row.candidate_id in ref.robust(**gates, **{name: 3}).candidates.candidate_id.values
    assert row.candidate_id not in ref.robust(
        **gates, require_smoothing_interval_overlap=True).candidates.candidate_id.values
    assert row.candidate_id in ref.robust(
        **gates, require_smoothing_interval_overlap=False).candidates.candidate_id.values
    # An axis-specific threshold leaves the other swept axis alone.
    other = 'max_smoothing_spread_t' if axis == 'T' else 'max_smoothing_spread_k'
    assert row.candidate_id in ref.robust(**gates, **{other: 0}).candidates.candidate_id.values


@pytest.mark.parametrize('kwargs', [
    {'min_smoothing_n': -1}, {'min_smoothing_n': 1.5}, {'min_smoothing_n': True},
    {'min_smoothing_persistence': 1.1}, {'min_smoothing_sign_consistency': float('nan')},
    {'max_smoothing_spread_k': -1}, {'require_smoothing_interval_overlap': 1},
])
def test_sensitivity_rejects_invalid_thresholds(kink_scan, kwargs):
    from calamari.exceptions import CalamariError

    with pytest.raises(CalamariError, match='robust.*invalid'):
        kink_scan.reference(2).robust(**kwargs)


def test_sensitivity_requires_reference_and_valid_qualification(kink_scan):
    from calamari.exceptions import CalamariError

    with pytest.raises(CalamariError, match='reference'):
        kink_scan.at(2).robust(min_smoothing_n=2)
    with pytest.raises(CalamariError, match='reference'):
        _ = kink_scan.at(2).sensitivity_audit
    for bad in (True, -1, float('inf'), float('nan'), '5'):
        with pytest.raises(CalamariError, match='min_member_significance'):
            kink_scan.reference(2, min_member_significance=bad)


def test_sensitivity_select_can_promote_eligible_secondary(kink_scan):
    from dataclasses import replace

    ref = kink_scan.reference(2)
    cands = ref.candidates.copy()
    own = cands.query("sweep_id == 0 and method == 'kink' and sign == 'down'")
    assert len(own) >= 2
    first, second = own.index[:2]
    cands.loc[:, 'smoothing_status'] = 'assessed'
    cands.loc[:, 'smoothing_T_K_spread'] = 0.0
    cands.loc[:, 'significance'] = 0.0
    cands.loc[first, 'significance'] = 100.0
    cands.loc[second, 'significance'] = 50.0
    cands.loc[first, 'smoothing_T_K_spread'] = 10.0
    controlled = replace(ref, candidates=cands)
    selection = dict(swept_axis='T', method='kink', sign='down', t_window_k=(2, 30))
    gates = dict(min_significance=0, exclude_edge=False, max_smoothing_spread_k=1)
    before = controlled.select(**selection)
    assert before.candidates.iloc[0].candidate_id == cands.loc[first, 'candidate_id']
    after = before.robust(**gates)
    assert after.candidates.iloc[0].candidate_id == cands.loc[second, 'candidate_id']
    pd.testing.assert_frame_equal(after.candidates,
                                  controlled.robust(**gates).select(**selection).candidates)


def test_reference_multiple_signals_native_mapping_and_plot_exports(tmp_path):
    import json

    from calamari import plot

    t = np.linspace(2, 12, 201)
    chi = simulate.synthetic_chi_peak(t, centre=7, width=0.4, height=0.02,
                                      noise=1e-5, rng=np.random.default_rng(73))
    path = simulate.write_mpms3_file(
        tmp_path / 'multiple.dat', app='SQUID AC', temperature_K=t,
        field_oe=np.full_like(t, 1000), chi_real=chi, chi_imag=0.1 * chi,
        ac_frequency_hz=np.full_like(t, 997),
    )
    scan = scan_smoothing(load(path), windows=(0.3, 0.5, 0.7), track_tol_k=0.3,
                          signal='all', methods=('peak', 'derivative', 'kink'))
    ref = scan.reference(0.5, min_member_significance=3)
    assert {'chi_real', 'chi_imag'} <= set(ref.candidates.signal)
    for row in ref.candidates.itertuples():
        track = scan.tracks.query('track_id == @row.smoothing_track_id').iloc[0]
        assert track.signal == row.signal
        assert track.method == row.method
    kept = ref.robust(min_significance=0, min_smoothing_n=3,
                       min_smoothing_persistence=0.5, max_smoothing_spread_k=1)
    assert len(kept.candidates)
    assert (kept.candidates.smoothing_n_present == 3).all()
    frame, thresholds = plot_detection._transitions_frame(kept)
    pd.testing.assert_frame_equal(frame, kept.candidates)
    title = plot_detection._overlay_legend_title('method', None, thresholds)
    assert 'reference window 0.5' in title
    assert 'member significance ≥ 3' in title
    assert 'smoothing support ≥ 3' in title
    ax = plot.plot_transitions(kept)
    ax.figure.savefig(tmp_path / 'sensitivity.png')
    assert (tmp_path / 'sensitivity.png').stat().st_size > 0
    kept.sensitivity_audit.to_csv(tmp_path / 'audit.csv', index=False)
    audit = pd.read_csv(tmp_path / 'audit.csv')
    assert len(audit) == len(ref.candidates)
    recipe = json.loads(json.dumps(kept.params))
    assert recipe['smoothing']['reference_window'] == 0.5
    assert recipe['robust']['min_smoothing_n'] == 3
    assert len(recipe['smoothing']['members']) == len(scan.members)


def test_native_field_reference_coordinates_and_duplicate_windows(tmp_path):
    from dataclasses import replace

    path = _hscan(tmp_path / 'field.dat', np.random.default_rng(73))
    scan = scan_smoothing(load(path), windows=(0.2, 0.3, 0.5), track_tol_t=0.2)
    ref = scan.reference(0.3)
    native = scan.at(0.3)
    pd.testing.assert_frame_equal(ref.candidates[native.candidates.columns], native.candidates)
    assert ref.candidates.swept_axis.eq('H').all()
    assert ref.candidates.smoothing_T_K_spread.isna().all()
    assert ref.candidates.smoothing_H_T_spread.notna().all()
    kept = ref.robust(min_significance=0, exclude_edge=False, max_smoothing_spread_t=0.5)
    assert len(kept.candidates)
    # Unknown interval errors do not silently become interval agreement.
    members = scan.members.copy()
    members['H_err_T'] = np.nan
    unknown = replace(scan, members=members).reference(0.3)
    assert unknown.candidates.smoothing_interval_overlap.isna().all()
    assert unknown.robust(require_smoothing_interval_overlap=True).candidates.empty
    with pytest.warns(UserWarning):
        duplicate = scan_smoothing(_short_sweep(tmp_path), windows=(2, 4, 7, 9),
                                   track_tol_k=2, **DETECT)
    ref = duplicate.reference(4)
    assert ref.candidates.smoothing_n_windows.eq(3).all()
    assert ref.candidates.smoothing_n_present.le(3).all()
    assert ref.candidates.smoothing_raw_n_present.le(3).all()


def test_scan_propagates_unexpected_upstream_failure(monkeypatch):
    failure = OSError("synthetic upstream failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(scales, "detect_transitions_all", fail)
    with pytest.raises(OSError) as caught:
        scales.scan_smoothing([], windows=(1.0, 2.0, 3.0))
    assert caught.value is failure
