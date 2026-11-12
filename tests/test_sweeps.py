"""Tests for calamari.sweeps: plateau segmentation, isotherm splitting, and
M(H)-loop folding.

All data is synthetic (see :mod:`calamari.simulate` and the project's
no-real-data-files policy) with planted, known structure, so tests assert
recovery of that structure rather than against recorded reference numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calamari import simulate, sweeps
from calamari.exceptions import SweepInputError
from calamari.io import load
from calamari.measurement import Measurement

# ---------------------------------------------------------------------------
# segment_by_constant
# ---------------------------------------------------------------------------


def test_segment_by_constant_counts_plateaus():
    """A clean staircase splits into one segment per plateau, each holding
    exactly that plateau's rows."""
    x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
    segments = sweeps.segment_by_constant(x, tol=0.5)

    assert len(segments) == 3
    assert [len(s) for s in segments] == [3, 2, 4]
    # Segments partition the input in order.
    assert np.array_equal(np.concatenate(segments), np.arange(len(x)))


def test_segment_by_constant_tol_sensitivity():
    """tol sits between the small within-group change and the big
    inter-group step: a 0.2 wobble is kept together, a 1.0 jump splits."""
    x = np.array([0.0, 0.2, 0.1, 5.0, 5.1, 4.9])
    two = sweeps.segment_by_constant(x, tol=0.5)
    assert len(two) == 2
    assert [len(s) for s in two] == [3, 3]

    # A tol below even the small wobble splits far more aggressively.
    many = sweeps.segment_by_constant(x, tol=0.05)
    assert len(many) > 2


def test_segment_by_constant_tolerates_slow_drift():
    """A plateau that drifts gradually (each step under tol) stays one
    segment even though its total span exceeds tol -- the consecutive-diff
    rule tolerates settle drift."""
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # span 0.6 > tol
    segments = sweeps.segment_by_constant(x, tol=0.2)
    assert len(segments) == 1
    assert len(segments[0]) == len(x)


def test_segment_by_constant_min_points_drops_short_runs():
    """min_points removes plateaus with too few rows."""
    x = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0])  # plateaus of 3, 1, 3
    segments = sweeps.segment_by_constant(x, tol=0.5, min_points=2)
    assert [len(s) for s in segments] == [3, 3]


def test_segment_by_constant_empty_and_validation():
    """Empty input yields no segments; bad parameters raise."""
    assert sweeps.segment_by_constant(np.array([]), tol=0.5) == []
    with pytest.raises(SweepInputError, match="tol"):
        sweeps.segment_by_constant([1.0, 2.0], tol=-1.0)
    with pytest.raises(SweepInputError, match="min_points"):
        sweeps.segment_by_constant([1.0, 2.0], tol=0.5, min_points=0)


# ---------------------------------------------------------------------------
# split_isotherms
# ---------------------------------------------------------------------------


def _write_stacked_mh_file(path, temps_k, n_per_isotherm=15):
    """Write a stacked multi-isotherm M(H) file by concatenating per-isotherm
    ramps -- demonstrating that the existing simulate writers already express
    a stacked file with no dedicated helper (see split_isotherms' docstring)."""
    field_oe = np.linspace(0.0, 25_000.0, n_per_isotherm)
    temperature_K = np.concatenate([np.full(n_per_isotherm, t) for t in temps_k])
    field_stacked = np.tile(field_oe, len(temps_k))
    moment_emu = np.concatenate(
        [simulate.synthetic_mh_curve(field_oe, m_sat_emu=1e-3, h_knee_oe=8000.0)
         for _ in temps_k]
    )
    return simulate.write_mpms3_file(
        path,
        app="SQUID DC",
        temperature_K=temperature_K,
        field_oe=field_stacked,
        vsm_moment_emu=moment_emu,
    )


def test_split_isotherms_recovers_stack(tmp_path):
    """A stacked file splits into one real sub-Measurement per temperature,
    with the right temperatures and per-isotherm row counts."""
    temps = [2.0, 3.0, 4.0]
    path = _write_stacked_mh_file(tmp_path / "stacked.dat", temps, n_per_isotherm=15)

    pairs = sweeps.split_isotherms(load(path))

    assert len(pairs) == 3
    recovered_t = [t for t, _ in pairs]
    assert recovered_t == pytest.approx(temps)
    for _, sub in pairs:
        assert isinstance(sub, Measurement)
        assert len(sub.data) == 15


def test_split_isotherms_single_file_is_one_block(tmp_path):
    """An ordinary single-temperature M(H) file yields exactly one isotherm
    at its mean temperature (the whole-span guard)."""
    field_oe = np.linspace(0.0, 25_000.0, 21)
    temperature_K = np.full(21, 5.0)
    moment_emu = simulate.synthetic_mh_curve(field_oe, m_sat_emu=1e-3, h_knee_oe=8000.0)
    path = simulate.write_mpms3_file(
        tmp_path / "single.dat",
        app="SQUID DC",
        temperature_K=temperature_K,
        field_oe=field_oe,
        vsm_moment_emu=moment_emu,
    )

    pairs = sweeps.split_isotherms(load(path))
    assert len(pairs) == 1
    t_k, sub = pairs[0]
    assert t_k == pytest.approx(5.0)
    assert len(sub.data) == 21


def test_split_isotherms_requires_temperature():
    """Splitting by temperature is meaningless with no temperature column."""
    m = Measurement(
        path="nofile.dat",
        data_format="MPMS3",
        measurement_type="DC",
        file_role="summary",
        title=None,
        header={},
        data=pd.DataFrame({"Field (Oe)": [0.0, 1.0, 2.0]}),
    )
    with pytest.raises(SweepInputError, match="temperature column"):
        sweeps.split_isotherms(m)


# ---------------------------------------------------------------------------
# increasing_field_branch
# ---------------------------------------------------------------------------


def test_increasing_field_branch_folds_loop():
    """A 0 -> +H_max -> 0 loop folds to its monotone up-leg: output is
    non-decreasing, roughly half the points, and the up-leg feature
    survives."""
    n = 51
    up = np.linspace(0.0, 10.0, n)
    down = np.linspace(10.0, 0.0, n)[1:]  # avoid duplicating the turning point
    field = np.concatenate([up, down])
    # A step feature at H=5 planted identically on both legs.
    moment = 0.5 * (1.0 + np.tanh((np.abs(field) - 5.0) / 0.3))

    h_inc, m_inc = sweeps.increasing_field_branch(field, moment)

    assert np.all(np.diff(h_inc) >= 0.0)  # monotone non-decreasing
    assert len(h_inc) == n  # exactly the leading up-leg
    assert len(h_inc) < len(field)  # genuinely folded
    # The retained branch still spans 0 -> 10 through the H=5 feature.
    assert h_inc[0] == pytest.approx(0.0)
    assert h_inc[-1] == pytest.approx(10.0)
    assert m_inc[h_inc >= 5.0].mean() > m_inc[h_inc < 5.0].mean()


def test_increasing_field_branch_is_unit_agnostic():
    """The helper only uses |field| and monotonicity, so a Tesla-scaled loop
    folds to a Tesla-scaled branch -- the output unit follows the input."""
    up_oe = np.linspace(0.0, 20_000.0, 40)
    down_oe = np.linspace(20_000.0, 0.0, 40)[1:]
    field_oe = np.concatenate([up_oe, down_oe])
    moment = np.linspace(0.0, 1.0, len(field_oe))

    h_inc, _ = sweeps.increasing_field_branch(field_oe, moment)
    assert h_inc.max() == pytest.approx(20_000.0)


def test_increasing_field_branch_fallback_sorts_decreasing_start():
    """A sweep that opens on a decreasing leg (no usable leading up-run)
    falls back to the whole sweep sorted by |field|."""
    field = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 0.0])
    moment = np.arange(6, dtype=float)

    h_inc, m_inc = sweeps.increasing_field_branch(field, moment)
    assert np.all(np.diff(h_inc) >= 0.0)
    assert len(h_inc) == len(field)  # nothing dropped, just reordered


def test_increasing_field_branch_length_mismatch_raises():
    with pytest.raises(SweepInputError, match="same shape"):
        sweeps.increasing_field_branch(np.arange(5.0), np.arange(4.0))
