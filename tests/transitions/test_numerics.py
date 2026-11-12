"""Array-level transition detection, noise, grouping and scientific properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import erf

from calamari import simulate, transitions
from calamari.transitions import (
    METHODS,
    find_transitions,
)

from ._helpers import (
    FISHER_NOISE,
    _cusp,
    _kinked,
    _slope_change_above,
    _toy_afm_chi,
    _toy_afm_t,
)

# Method registry.


def test_registry_every_method_is_fully_cited():
    """Each method carries a marker and at least one citation on each of
    the statistical and magnetometry sides -- the dual-citation contract."""
    for key, info in METHODS.items():
        assert info.key == key
        assert info.marker
        kinds = {c.kind for c in info.citations}
        assert "statistical" in kinds, key
        assert "magnetometry" in kinds, key
        for c in info.citations:
            assert c.role in ("origin", "review", "exemplar", "implementation")


def test_peak_recovered_and_flanks_grouped(rng):
    """A cusp yields ONE candidate: the peak, with its derivative flanks
    grouped in (n_methods == 2) rather than reported as separate rows --
    the flanks are geometric companions, not independent transitions.
    The planted cusp is the only candidate above the default threshold.
    """
    x = np.linspace(5.0, 20.0, 151)
    result = find_transitions(
        x, _cusp(x, rng=rng), methods=("peak", "derivative"), x_name="T", x_unit="K"
    )
    kept = result.robust()
    assert len(kept.candidates) == 1
    row = kept.candidates.iloc[0]
    assert row["T_K"] == pytest.approx(12.5, abs=0.2)
    assert row["method"] == "peak"
    assert row["n_methods"] == 2
    assert row["sign"] == "max"
    assert row["significance"] > 20
    assert (result.candidates["significance"] > transitions.DEFAULT_MIN_SIGNIFICANCE).sum() == 1

    dets = kept.detections
    flanks = dets[dets["method"] == "derivative"]
    assert set(flanks["sign"]) == {"rise", "fall"}
    # rise below the peak, fall above it
    assert flanks[flanks["sign"] == "rise"]["T_K"].iloc[0] < row["T_K"]
    assert flanks[flanks["sign"] == "fall"]["T_K"].iloc[0] > row["T_K"]


def test_dip_detected_as_minimum():
    x = np.linspace(0.0, 10.0, 121)
    y = 1.0 - simulate.synthetic_chi_peak(x, centre=4.0, width=0.5, height=0.8)
    result = find_transitions(x, y, methods="peak", x_name="H", x_unit="T")
    row = result.robust().candidates.iloc[0]
    assert row["H_T"] == pytest.approx(4.0, abs=0.2)
    assert row["sign"] == "min"


def test_derivative_recovers_step_midpoint(rng):
    x = np.linspace(0.0, 5.0, 141)
    y = 0.5 * (1 + np.tanh((x - 2.0) / 0.15)) + rng.normal(scale=0.005, size=x.shape)
    result = find_transitions(x, y, methods="derivative", x_name="H", x_unit="T")
    row = result.robust().candidates.iloc[0]
    assert row["H_T"] == pytest.approx(2.0, abs=0.1)
    assert row["method"] == "derivative"
    assert row["feature"] == "step"
    assert row["sign"] == "rise"
    assert row["H_T_err"] > 0


def test_onset_two_tangent_sits_before_midpoint_and_outranks_derivative(rng):
    """The two-tangent onset lands on the baseline side of the step's
    midpoint, grouped with its parent derivative hit, and -- sitting above
    'derivative' in the criterion priority -- reports the coordinate.

    The upper branch carries a slope (as a normal-state tail does) so the
    flat lower side is unambiguously the baseline: for a perfectly
    symmetric step the flatter-side choice is noise-driven, which is the
    data's ambiguity, not the method's.
    """
    x = np.linspace(5.0, 20.0, 151)
    y = 0.5 * (1 + np.tanh((x - 12.5) / 0.4)) + 0.04 * np.maximum(x - 12.5, 0.0)
    y = y + rng.normal(scale=0.01, size=x.shape)
    result = find_transitions(
        x, y, methods=("derivative", "onset"), x_name="T", x_unit="K"
    )
    kept = result.robust()
    assert len(kept.candidates) == 1
    row = kept.candidates.iloc[0]
    assert row["method"] == "onset"
    assert row["n_methods"] == 2
    assert row["T_K"] < 12.5  # onset precedes the midpoint for a rising step
    assert row["T_K"] == pytest.approx(12.1, abs=0.4)

    # criterion= overrides the priority: the midpoint is then reported.
    mid = find_transitions(
        x, y, methods=("derivative", "onset"), x_name="T", x_unit="K",
        criterion="derivative",
    ).robust()
    assert mid.candidates.iloc[0]["method"] == "derivative"
    assert mid.candidates.iloc[0]["T_K"] == pytest.approx(12.5, abs=0.15)


def test_onset_alone_reports_onsets_without_their_parent_rows(rng):
    """methods='onset' is a legitimate call and must not raise.

    Onsets are built from derivative extrema, so without 'derivative' in
    ``methods`` the parent hit is not among the reported rows -- grouping
    used to index the missing parent and raise ``KeyError: 0``, turning a
    documented method combination into a crash. An orphaned onset now
    starts its own group, and the row is still the two-tangent
    coordinate.
    """
    x = np.linspace(5.0, 20.0, 151)
    y = 0.5 * (1 + np.tanh((x - 12.5) / 0.4)) + rng.normal(scale=0.005, size=x.shape)
    result = find_transitions(x, y, methods="onset", x_name="T", x_unit="K")
    assert result.params["methods"] == ("onset",)
    kept = result.robust()
    assert len(kept.candidates) == 1
    row = kept.candidates.iloc[0]
    assert row["method"] == "onset"
    assert row["feature"] == "onset"
    assert row["n_methods"] == 1  # no parent derivative row to group with
    assert row["T_K"] == pytest.approx(12.5, abs=1.0)
    assert set(kept.detections["method"]) == {"onset"}


def test_instrument_errors_preferred_over_inferred_noise(rng):
    x = np.linspace(0.0, 10.0, 101)
    y = _cusp(x, centre=5.0, width=0.5, rng=rng)
    with_err = find_transitions(
        x, y, methods="peak", y_err=np.full_like(x, 0.02), x_name="T", x_unit="K"
    )
    assert with_err.params["noise_source"] == "instrument"
    assert with_err.params["noise_sigma"] == pytest.approx(0.02)
    without = find_transitions(x, y, methods="peak", x_name="T", x_unit="K")
    assert without.params["noise_source"] == "scatter"


def test_edge_candidates_flagged_and_excluded_by_default():
    """A feature supported only by a terminal smoothing window is flagged
    edge=True and dropped by robust() unless exclude_edge=False."""
    # Centre inside the terminal HALF-window (the savgol boundary-refit
    # zone): the default 11-point window on this grid flags i >= 96.
    x = np.linspace(0.0, 10.0, 101)
    y = simulate.synthetic_chi_peak(x, centre=9.8, width=0.4, height=1.0)
    result = find_transitions(x, y, methods="peak", x_name="T", x_unit="K")
    edge_rows = result.candidates[result.candidates["edge"]]
    assert len(edge_rows) >= 1
    kept = result.robust()
    assert not kept.candidates["edge"].any()
    assert result.robust(exclude_edge=False).candidates["edge"].any()


def test_hysteresis_loop_refused_by_name():
    x = np.linspace(0.0, 5.0, 80)
    y = np.tanh(x - 2.5)
    with pytest.raises(ValueError, match="not monotonic"):
        find_transitions(
            np.concatenate([x, x[::-1]]), np.concatenate([y, y[::-1]]),
            methods="derivative",
        )


def test_repeated_setpoints_refused_by_name():
    x = np.repeat(np.linspace(0.0, 5.0, 40), 3)
    y = np.tanh(x - 2.5)
    with pytest.raises(ValueError, match="repeat"):
        find_transitions(x, y, methods="derivative")


def test_isolated_duplicates_dropped_and_counted():
    """Below the systematic-repeats line, duplicate setpoints are dropped
    keep-first but COUNTED in params -- tolerated, never silent (caught in
    review: the drop used to leave no record)."""
    x = np.linspace(0.0, 10.0, 100)
    x[20], x[45], x[70] = x[19], x[44], x[69]  # 3% duplicates: tolerated
    y = simulate.synthetic_chi_peak(x, centre=5.0, width=0.5, height=1.0, background=0.1)
    result = find_transitions(x, y, methods="peak")
    assert result.params["n_duplicates_dropped"] == 3
    assert result.params["n_points"] == 97
    assert result.robust().candidates.iloc[0]["x"] == pytest.approx(5.0, abs=0.2)


def test_flat_signal_yields_no_candidates():
    x = np.linspace(0.0, 10.0, 60)
    result = find_transitions(x, np.full_like(x, 3.0), methods=("peak", "derivative"))
    assert result.candidates.empty


def test_invalid_methods_and_criterion_refused():
    x = np.linspace(0.0, 10.0, 60)
    y = np.tanh(x - 5.0)
    with pytest.raises(ValueError, match="methods"):
        find_transitions(x, y, methods="wavelet")
    with pytest.raises(ValueError, match="criterion"):
        find_transitions(x, y, methods="peak", criterion="vibes")


def test_params_record_the_resolved_recipe(rng):
    x = np.linspace(0.0, 10.0, 101)
    result = find_transitions(
        x, _cusp(x, centre=5.0, rng=rng), methods="peak", smooth_window=1.0,
        x_name="T", x_unit="K",
    )
    p = result.params
    assert p["methods"] == ("peak",)
    assert p["smooth_points"] % 2 == 1
    assert p["smooth_window"] == pytest.approx(p["smooth_points"] * p["grid_dx"])
    assert p["noise_source"] in ("instrument", "scatter")
    assert p["criterion_priority"][0] == "peak"

# Noise source override (Phase A)


def test_noise_auto_prefers_instrument_and_records_both_estimates(rng):
    x = np.linspace(0.0, 10.0, 101)
    y = _cusp(x, centre=5.0, width=0.5, rng=rng)
    result = find_transitions(
        x, y, methods="peak", y_err=np.full_like(x, 0.02), x_name="T", x_unit="K"
    )
    p = result.params
    assert p["noise"] == "auto"
    assert p["noise_source"] == "instrument"
    assert p["noise_sigma"] == pytest.approx(0.02)
    assert p["noise_sigma_instrument"] == pytest.approx(0.02)
    assert p["noise_sigma_scatter"] > 0
    assert p["noise_ratio"] == pytest.approx(
        p["noise_sigma_instrument"] / p["noise_sigma_scatter"]
    )


def test_noise_scatter_always_uses_second_difference_estimate(rng):
    """noise='scatter' ignores a usable error column entirely, even one
    that noise='auto' would have accepted without complaint."""
    x = np.linspace(0.0, 10.0, 101)
    y = _cusp(x, centre=5.0, width=0.5, rng=rng)
    result = find_transitions(
        x, y, methods="peak", y_err=np.full_like(x, 0.02), noise="scatter",
        x_name="T", x_unit="K",
    )
    assert result.params["noise_source"] == "scatter"
    assert result.params["noise_sigma"] == pytest.approx(result.params["noise_sigma_scatter"])
    assert result.params["noise_sigma"] != pytest.approx(0.02)


def test_noise_instrument_requires_usable_errors_and_names_the_fix(rng):
    x = np.linspace(0.0, 10.0, 101)
    y = _cusp(x, centre=5.0, width=0.5, rng=rng)
    with pytest.raises(ValueError, match="no usable per-point errors") as exc_info:
        find_transitions(x, y, methods="peak", noise="instrument", label="my-sweep")
    assert "my-sweep" in str(exc_info.value)
    assert "noise='scatter'" in str(exc_info.value)

    # With a usable column, 'instrument' succeeds and matches 'auto'.
    result = find_transitions(
        x, y, methods="peak", y_err=np.full_like(x, 0.02), noise="instrument"
    )
    assert result.params["noise_source"] == "instrument"
    assert result.params["noise_sigma"] == pytest.approx(0.02)


def test_noise_caller_float_is_recorded_as_caller(rng):
    x = np.linspace(0.0, 10.0, 101)
    y = _cusp(x, centre=5.0, width=0.5, rng=rng)
    result = find_transitions(x, y, methods="peak", noise=0.05)
    assert result.params["noise"] == 0.05
    assert result.params["noise_source"] == "caller"
    assert result.params["noise_sigma"] == pytest.approx(0.05)
    # The caller-supplied value does not require or consult an error column.
    assert result.params["noise_sigma_instrument"] is None


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_noise_rejects_non_positive_or_non_finite_float(bad, rng):
    x = np.linspace(0.0, 10.0, 101)
    y = _cusp(x, centre=5.0, width=0.5, rng=rng)
    with pytest.raises(ValueError, match="positive"):
        find_transitions(x, y, methods="peak", noise=bad)


def test_noise_rejects_unknown_string(rng):
    x = np.linspace(0.0, 10.0, 101)
    y = _cusp(x, centre=5.0, width=0.5, rng=rng)
    with pytest.raises(ValueError, match="noise must be"):
        find_transitions(x, y, methods="peak", noise="vibes")


def test_noise_params_keys_present(rng):
    x = np.linspace(0.0, 10.0, 101)
    y = _cusp(x, centre=5.0, width=0.5, rng=rng)
    result = find_transitions(x, y, methods="peak", y_err=np.full_like(x, 0.02))
    for key in (
        "noise", "noise_source", "noise_sigma", "noise_sigma_instrument",
        "noise_sigma_scatter", "noise_ratio",
    ):
        assert key in result.params


def test_kink_recovers_slope_change_with_its_sign(rng):
    """The curvature extremum sits at the slope change itself, and its
    sign says which way the slope went: 'down' where the slope decreases
    through the kink, 'up' where it increases."""
    x = np.linspace(5.0, 20.0, 151)
    for slope_change, expected in ((-0.40, "down"), (+0.40, "up")):
        result = find_transitions(
            x, _kinked(x, slope_change=slope_change, rng=rng), methods="kink",
            x_name="T", x_unit="K",
        )
        kept = result.robust()
        assert len(kept.candidates) == 1
        row = kept.candidates.iloc[0]
        assert row["T_K"] == pytest.approx(12.5, abs=0.2)  # 2 grid steps
        assert row["method"] == "kink"
        assert row["feature"] == "kink"
        assert row["sign"] == expected
        assert row["T_K_err"] > 0


def test_one_sided_kink_reported_once_and_never_hosts_a_derivative_hit(rng):
    """A one-sided slope change is reported ONCE, by the kink method, at
    the planted position -- and it hosts nothing.

    dy/dx steps through a sigmoid at a one-sided kink, so it has no
    genuine extremum there; any derivative hit nearby is noise or a
    curvature artefact and belongs in its own row. Grouping used to let
    such a hit join the kink (via the kink's prominence base interval),
    which both overstated ``n_methods`` and -- once 'derivative' outranked
    'kink' -- moved the coordinate off the slope change.
    """
    x = np.linspace(5.0, 20.0, 151)
    y = _kinked(x, rng=rng)
    kept = find_transitions(
        x, y, methods=("derivative", "kink"), x_name="T", x_unit="K"
    ).robust()
    kinks = kept.candidates[kept.candidates["method"] == "kink"]
    assert len(kinks) == 1
    row = kinks.iloc[0]
    assert row["T_K"] == pytest.approx(12.5, abs=0.2)  # 2 grid steps
    assert row["sign"] == "down"  # the planted slope change is a decrease
    assert row["n_methods"] == 1

    members = kept.detections[kept.detections["candidate_id"] == row["candidate_id"]]
    assert list(members["method"]) == ["kink"]
    # Any derivative hits present are separate candidates, not companions.
    others = kept.candidates[kept.candidates["candidate_id"] != row["candidate_id"]]
    assert set(others["method"]) <= {"derivative"}


def test_rounded_step_reported_at_its_midpoint_with_both_shoulders(rng):
    """A rounded step is ONE candidate, reported at its midpoint.

    The derivative extremum sits at the midpoint and the curvature
    extrema are the step's two shoulders, one of each sign either side of
    it. Grouping keeps all three together and -- 'derivative' outranking
    'kink' -- reports the midpoint; the earlier priority reported
    whichever shoulder happened to be stronger, about a shoulder's
    distance away and with a sign that flipped along a family.
    """
    x = np.linspace(5.0, 20.0, 201)
    y = 0.5 * (1 + np.tanh((x - 12.5) / 0.5)) + rng.normal(scale=0.005, size=x.shape)
    kept = find_transitions(
        x, y, methods=("derivative", "kink"), x_name="T", x_unit="K"
    ).robust()
    assert len(kept.candidates) == 1
    row = kept.candidates.iloc[0]
    assert row["method"] == "derivative"
    assert row["sign"] == "rise"
    assert row["n_methods"] == 2
    assert row["T_K"] == pytest.approx(12.5, abs=0.15)  # 2 grid steps

    dets = kept.detections
    assert list(dets[dets["method"] == "derivative"]["sign"]) == ["rise"]
    shoulders = dets[dets["method"] == "kink"]
    assert len(shoulders) == 2
    below = shoulders[shoulders["T_K"] < row["T_K"]]
    above = shoulders[shoulders["T_K"] > row["T_K"]]
    assert list(below["sign"]) == ["up"]  # the rise curves up out of the baseline
    assert list(above["sign"]) == ["down"]  # and back down onto the upper one

    # Nor does criterion='kink' promote a shoulder: a step has no slope
    # change of its own, so it still reports its midpoint.
    promoted = find_transitions(
        x, y, methods=("derivative", "kink"), x_name="T", x_unit="K",
        criterion="kink",
    ).robust().candidates.iloc[0]
    assert promoted["method"] == "derivative"
    assert promoted["T_K"] == row["T_K"]


@pytest.mark.parametrize("width", [0.5, 1.5])
def test_clean_rounded_step_keeps_both_shoulders(width):
    """A noise-free rounded step, narrow or broad against the window, is one
    candidate: its midpoint plus one shoulder of each sign either side.
    The guard the step-shoulder rule must never break."""
    x = np.linspace(0.0, 40.0, 401)
    result = find_transitions(
        x, np.tanh((x - 20.0) / width), methods=("derivative", "kink"),
        x_name="T", x_unit="K", smooth_window=1.0, noise=1e-3,
    )
    assert len(result.candidates) == 1
    row = result.candidates.iloc[0]
    assert (row["method"], row["sign"], row["n_methods"]) == ("derivative", "rise", 2)
    assert row["T_K"] == pytest.approx(20.0, abs=0.1)
    kinks = result.detections[result.detections["method"] == "kink"]
    assert list(kinks.sort_values("T_K")["sign"]) == ["up", "down"]


def test_two_nearby_steps_each_keep_their_own_shoulders():
    """Two rises 2.5 K apart, each about three windows wide: two step
    candidates, each with its own shoulder pair, and the dip in dy/dx
    between them standing alone. Neither step borrows the other's
    shoulder, although both lie inside each other's base interval."""
    x = np.linspace(0.0, 40.0, 401)
    y = np.tanh((x - 18.0) / 0.8) + 0.6 * np.tanh((x - 20.5) / 0.8)
    result = find_transitions(
        x, y, methods=("derivative", "kink"), x_name="T", x_unit="K",
        smooth_window=1.0, noise=1e-3,
    )
    cands = result.candidates
    steps = cands[cands["sign"] == "rise"]
    assert list(steps["T_K"].round(1)) == [18.0, 20.5]
    assert list(steps["n_methods"]) == [2, 2]
    for _, step in steps.iterrows():
        members = result.detections[result.detections["candidate_id"] == step["candidate_id"]]
        kinks = members[members["method"] == "kink"].sort_values("T_K")
        assert list(kinks["sign"]) == ["up", "down"]
        assert (kinks["T_K"].iloc[0] < step["T_K"] < kinks["T_K"].iloc[1])
    dip = cands[cands["sign"] == "fall"]
    assert list(dip["n_methods"]) == [1]


def _kink_beside_a_weak_bump(x, bump_height=0.02, bump_at=6.0, kink_at=7.5):
    """A one-sided slope change with a weak dy/dx bump 1.5 K below it.

    The slope falls from 0.2 to 0.1 through ``kink_at`` (a softplus corner
    rounded over 0.3 K), and dy/dx carries a Gaussian bump of height
    ``bump_height`` and width 0.5 K at ``bump_at``. The bump's own upper
    curvature extremum merges into the slope change's, so the nearest
    detected 'down' curvature extremum above the bump is the slope change
    itself. Noise-free.
    """
    ramp = 0.1 * x - 0.03 * np.logaddexp(0.0, (kink_at - x) / 0.3)
    bump = bump_height * np.sqrt(np.pi / 2) * 0.5 * erf((x - bump_at) / (np.sqrt(2) * 0.5))
    return ramp + bump


def test_weak_derivative_bump_does_not_claim_a_distant_slope_change():
    """A slope change is not the shoulder of a weak dy/dx bump beside it.

    Rule 3 used to let a free derivative extremum claim, on each side, the
    nearest matching-sign kink inside its prominence base interval. A weak
    bump on a slope plateau has a base interval running on to wherever
    dy/dx next drops -- here past the real slope change -- so the change
    was taken for the bump's upper shoulder and vanished from the
    candidates under every criterion. A shoulder must now lie within the
    bump's footprint, where dy/dx stands above the level its prominence is
    measured from. The slope change is out there, and stands alone.
    """
    x = np.linspace(2.0, 30.0, 141)
    y = _kink_beside_a_weak_bump(x)
    for criterion in (None, "kink"):
        cands = find_transitions(
            x, y, methods=("peak", "derivative", "kink"), x_name="T", x_unit="K",
            smooth_window=2.0, noise=1e-3, criterion=criterion,
        ).candidates
        change = cands[(cands["method"] == "kink") & (cands["sign"] == "down")]
        assert len(change) == 1
        assert change.iloc[0]["T_K"] == pytest.approx(7.5, abs=0.2)  # 1 grid step
        assert change.iloc[0]["n_methods"] == 1
        bump = cands[cands["method"] == "derivative"]
        assert list(bump["sign"]) == ["rise"]
        assert bump.iloc[0]["T_K"] == pytest.approx(6.0, abs=0.2)
        assert bump.iloc[0]["n_methods"] == 1  # one shoulder is not a step


def test_step_shoulder_must_be_adjacent_to_its_midpoint():
    """A shoulder is the curvature extremum next to the midpoint.

    dy/dx here steps from 0.1 up to 0.2 at 8 K, dips narrowly at 14 K,
    swells broadly at 20 K and drops to 0.12 at 26 K. The broad swell is a
    derivative extremum whose own curvature extrema are too weak to
    detect, and whose footprint runs from the 8 K step to the 26 K drop,
    because dy/dx stays above 0.12 all that way. Footprint alone would make
    both real slope changes the swell's "shoulders", 12 K and 6 K away. But
    the dip and its own curvature extrema lie between the swell and the
    8 K change, so that change is not adjacent and cannot be a shoulder,
    and with one shoulder the swell is not a step. Noise-free.
    """
    x = np.linspace(0.0, 40.0, 401)

    def sigmoid(z):
        return 0.5 * (1 + np.tanh(z / 2))

    def gauss(c, s):
        return np.exp(-0.5 * ((x - c) / s) ** 2)

    dydx = (0.1 + 0.1 * sigmoid((x - 8.0) / 0.15) - 0.01 * gauss(14.0, 0.3)
            + 0.01 * gauss(20.0, 2.0) - 0.08 * sigmoid((x - 26.0) / 0.15))
    y = np.concatenate([[0.0], np.cumsum(0.5 * (dydx[1:] + dydx[:-1]) * np.diff(x))])
    cands = find_transitions(
        x, y, methods=("peak", "derivative", "kink"), x_name="T", x_unit="K",
        smooth_window=1.0, noise=1e-4,
    ).candidates
    changes = cands[cands["method"] == "kink"]
    assert list(changes["sign"]) == ["up", "down"]
    assert list(changes["T_K"].round(1)) == [8.0, 26.0]
    assert list(changes["n_methods"]) == [1, 1]
    swell = cands[(cands["method"] == "derivative") & (cands["T_K"].round(1) == 20.0)]
    assert list(swell["n_methods"]) == [1]


def test_distant_kink_not_swallowed_by_a_cusp(rng):
    """A slope change well away from a cusp stays its own candidate.

    A large cusp's prominence base interval can span most of a sweep
    (here it reaches the start, past the planted kink), so "any kink
    inside the peak's base interval is a companion" absorbed a genuine
    second boundary and deleted it from the output. Only the cusp's OWN
    curvature companions -- those between its derivative flanks, plus the
    first kink beyond each flank -- belong to it; the next feature out
    does not.
    """
    x = np.linspace(5.0, 20.0, 151)
    y = simulate.synthetic_chi_peak(
        x, centre=15.0, width=2.5, height=2.0, background=0.2, noise=0.001, rng=rng
    ) + _slope_change_above(x, 8.0, 0.3)
    kept = find_transitions(
        x, y, methods=("peak", "derivative", "kink"), x_name="T", x_unit="K"
    ).robust()
    assert len(kept.candidates) == 2
    far, cusp = kept.candidates.iloc[0], kept.candidates.iloc[1]

    assert far["method"] == "kink"
    assert far["T_K"] == pytest.approx(8.0, abs=0.2)  # 2 grid steps
    assert far["sign"] == "up"
    assert far["n_methods"] == 1

    # The cusp keeps its own companions -- the whole point of the method
    # staying usable alongside 'peak'.
    assert cusp["method"] == "peak"
    assert cusp["n_methods"] == 3
    assert cusp["T_K"] > 12.0  # the linear ramp displaces the maximum upwards
    cusp_dets = kept.detections[kept.detections["candidate_id"] == cusp["candidate_id"]]
    assert (cusp_dets["T_K"] > far["T_K"] + 1.0).all()


def test_cusp_absorbs_its_curvature_companions(rng):
    """A cusp's own curvature minimum sits at the cusp and its flanking
    curvature maxima sit inside the cusp's prominence base interval, so
    adding 'kink' to a genuine cusp still yields ONE candidate -- reported
    by the peak, with three methods grouped in."""
    x = np.linspace(5.0, 20.0, 151)
    kept = find_transitions(
        x, _cusp(x, rng=rng), methods=("peak", "derivative", "kink"), x_name="T", x_unit="K"
    ).robust()
    assert len(kept.candidates) == 1
    row = kept.candidates.iloc[0]
    assert row["method"] == "peak"
    assert row["n_methods"] == 3
    assert row["T_K"] == pytest.approx(12.5, abs=0.2)
    assert "kink" in set(kept.detections["method"])


def _sharp_footed_cusp(x, x0=20.0, height=5.0, slope_lo=0.5, slope_hi=2.0, rounding=4.0):
    """A cusp whose feet are sharper than its top, noise-free.

    Two lines, rising at ``slope_lo`` below ``x0`` and falling at
    ``slope_hi`` above, joined by a hyperbolic rounding of the top (so the
    central curvature extremum sits exactly where the lines meet, at
    ``x0``), and clipped at zero, so each foot is a sharp corner. After
    smoothing, the steep side's foot is a much stronger curvature extremum
    than the rounded top: the shape on which the most significant kink in
    a cusp's group is a flank, not the cusp.
    """
    rise = height + slope_lo * (x - x0)
    fall = height - slope_hi * (x - x0)
    top = 0.5 * ((rise + fall) - np.sqrt((rise - fall) ** 2 + rounding**2))
    return np.maximum(0.0, top)


def test_kink_criterion_reports_a_cusps_own_slope_change_not_its_flank():
    """``criterion='kink'`` on a cusp reports the central curvature extremum.

    A cusp's group holds its central curvature extremum AND the flanking
    ones of the opposite sign. Ranked by significance, the strongest kink
    reported, which on a cusp with sharp feet is a flank: several kelvin
    off the cusp and with the wrong sign. Only the kink of the central sign
    between the derivative flanks is the cusp's own slope change. Mirrored
    for a minimum. Noise-free, so no draw can decide it.
    """
    x = np.linspace(0.0, 40.0, 401)
    for y, peak_sign, central, flank in (
        (_sharp_footed_cusp(x), "max", "down", "up"),
        (-_sharp_footed_cusp(x), "min", "up", "down"),
    ):
        kw = dict(
            methods=("peak", "derivative", "kink"), x_name="T", x_unit="K",
            smooth_window=1.0, noise=1e-3,
        )
        default = find_transitions(x, y, **kw)
        # The trap is real: a flank outscores the cusp's own curvature extremum.
        dets = default.detections
        kinks = dets[dets["method"] == "kink"]
        strongest = kinks.loc[kinks["significance"].idxmax()]
        assert strongest["sign"] == flank
        assert strongest["T_K"] > 22.0
        assert len(default.candidates) == 1
        assert default.candidates.iloc[0]["method"] == "peak"
        assert default.candidates.iloc[0]["sign"] == peak_sign

        promoted = find_transitions(x, y, criterion="kink", **kw)
        assert len(promoted.candidates) == 1
        row = promoted.candidates.iloc[0]
        assert row["method"] == "kink"
        assert row["sign"] == central
        assert row["T_K"] == pytest.approx(20.0, abs=0.1)  # 1 grid step
        assert row["n_methods"] == 3
        # Only the reporter changed: the same detections, grouped the same way.
        pd.testing.assert_frame_equal(promoted.detections, default.detections)


def test_kink_criterion_several_central_kinks_most_significant_reports():
    """Among a cusp's own kinks, the most significant reports, not the nearest.

    A concave polyline whose top bends three times: a strong corner at
    14 K (slope 1 -> 0.1), a weak one at 18 K right beside the maximum
    (0.1 -> -0.2) and a middling one at 22 K (-0.2 -> -0.9). All three are
    'down' and between the derivative flanks, so all are eligible. Picking
    the nearest to the peak would report the weakest bend and pull the
    kink onto the maximum; the strongest reports. Noise-free.
    """
    x = np.linspace(0.0, 40.0, 401)
    y = np.interp(x, [0.0, 4.0, 14.0, 18.0, 22.0, 22.0 + 9.6 / 0.9, 40.0],
                  [0.0, 0.0, 10.0, 10.4, 9.6, 0.0, 0.0])
    kw = dict(methods=("peak", "derivative", "kink"), x_name="T", x_unit="K",
              smooth_window=1.0, noise=1e-3)
    default = find_transitions(x, y, **kw)
    assert len(default.candidates) == 1
    peak_t = default.candidates.iloc[0]["T_K"]
    dets = default.detections
    central = dets[(dets["method"] == "kink") & (dets["sign"] == "down")]
    assert sorted(central["T_K"].round(3)) == [14.0, 18.0, 22.0]
    by_t = central.set_index(central["T_K"].round(3))
    # The trap: the weakest bend is the one nearest the maximum.
    assert abs(18.0 - peak_t) < abs(14.0 - peak_t) and abs(18.0 - peak_t) < abs(22.0 - peak_t)
    assert by_t.loc[18.0, "significance"] < by_t.loc[22.0, "significance"]
    assert by_t.loc[22.0, "significance"] < by_t.loc[14.0, "significance"]

    row = find_transitions(x, y, criterion="kink", **kw).candidates.iloc[0]
    assert (row["method"], row["sign"]) == ("kink", "down")
    assert row["T_K"] == pytest.approx(14.0, abs=0.1)  # 1 grid step
    assert row["significance"] == pytest.approx(by_t.loc[14.0, "significance"])


def test_kink_criterion_step_without_a_slope_change_of_its_own_reports_as_default():
    """A rounded step under ``criterion='kink'`` reports its midpoint.

    A step's only kinks are its two shoulders, about half a window either
    side of the midpoint. Promoting the kink used to move the coordinate
    onto one of them; neither is the step, so the group falls through to
    what the default order reports -- the derivative midpoint, or its onset
    when 'onset' was searched. Noise-free: the shoulders tie exactly.
    """
    x = np.linspace(0.0, 40.0, 401)
    y = np.tanh((x - 20.0) / 1.5)
    for methods in (("derivative", "kink"), ("derivative", "kink", "onset")):
        kw = dict(methods=methods, x_name="T", x_unit="K", smooth_window=1.0, noise=1e-3)
        default = find_transitions(x, y, **kw)
        promoted = find_transitions(x, y, criterion="kink", **kw)
        assert set(default.detections["method"]) == set(methods)  # shoulders were found
        pd.testing.assert_frame_equal(promoted.candidates, default.candidates)
    row = find_transitions(
        x, y, methods=("derivative", "kink"), x_name="T", x_unit="K",
        smooth_window=1.0, noise=1e-3, criterion="kink",
    ).candidates.iloc[0]
    assert row["method"] == "derivative"
    assert row["sign"] == "rise"
    assert row["T_K"] == pytest.approx(20.0, abs=0.1)


def test_kink_criterion_lone_kink_and_flankless_cusp_report_their_kink():
    """Where the kink IS the group's slope change, ``criterion='kink'`` keeps it.

    A one-sided slope change is a lone kink (no peak, no derivative in its
    group) and reports itself under either order. A cusp too gentle for
    derivative flanks -- slope +0.2 turning to -0.1 -- has no flanks to
    bound it, so any central-sign kink in its group is its own; promoting
    'kink' reports that one, not the peak.
    """
    x = np.linspace(0.0, 40.0, 401)
    kw = dict(methods=("peak", "derivative", "kink"), x_name="T", x_unit="K",
              smooth_window=1.0, noise=1e-3)

    lone_y = 0.2 * x + _slope_change_above(x, 20.0, 0.3, width=0.3)
    default = find_transitions(x, lone_y, **kw)
    promoted = find_transitions(x, lone_y, criterion="kink", **kw)
    pd.testing.assert_frame_equal(promoted.candidates, default.candidates)
    row = promoted.candidates.iloc[0]
    assert (row["method"], row["sign"], row["n_methods"]) == ("kink", "up", 1)
    assert row["T_K"] == pytest.approx(20.0, abs=0.1)

    cusp_y = 0.2 * x + _slope_change_above(x, 20.0, -0.3, width=0.3)
    dets = find_transitions(x, cusp_y, **kw).detections
    assert set(dets["method"]) == {"peak", "kink"}  # no derivative flanks at all
    row = find_transitions(x, cusp_y, criterion="kink", **kw).candidates.iloc[0]
    assert (row["method"], row["sign"], row["n_methods"]) == ("kink", "down", 2)
    assert row["T_K"] == pytest.approx(20.0, abs=0.1)


def test_curvature_curve_always_recorded(rng):
    """curves[] carries d2ydx2 whether or not 'kink' ran: it is how a
    caller checks that a derivative extremum is really a slope change."""
    x = np.linspace(5.0, 20.0, 151)
    result = find_transitions(x, _kinked(x, rng=rng), methods="derivative")
    curve = result.curves[("", "")]
    assert set(curve) == {"x", "y_smooth", "dydx", "d2ydx2"}
    assert curve["d2ydx2"].shape == curve["x"].shape
    # The planted slope change is the curve's largest |curvature|.
    assert curve["x"][np.argmax(np.abs(curve["d2ydx2"]))] == pytest.approx(12.5, abs=0.2)


def test_kink_raw_strength_unit_is_per_x_squared(rng):
    """A curvature prominence is in signal-per-x-squared; the unit column
    keeps that explicit rather than mixing three units in one column."""
    # A rounded step, so that one candidate carries rows of both kinds:
    # its midpoint (a derivative extremum) and its curvature shoulders.
    x = np.linspace(5.0, 20.0, 201)
    y = 0.5 * (1 + np.tanh((x - 12.5) / 0.5)) + rng.normal(scale=0.005, size=x.shape)
    dets = find_transitions(
        x, y, methods=("derivative", "kink"), x_name="T", x_unit="K"
    ).robust().detections
    kink_units = set(dets[dets["method"] == "kink"]["raw_strength_unit"])
    assert kink_units == {"y/K²"}
    assert all(u.endswith("²") for u in kink_units)
    assert set(dets[dets["method"] == "derivative"]["raw_strength_unit"]) == {"y/K"}


def test_kink_registry_entry_and_provenance(rng):
    """The method is registered with its own marker and its citations join
    through provenance() like every other method's."""
    info = METHODS["kink"]
    assert info.marker == "D"
    assert {c.kind for c in info.citations} >= {"statistical", "magnetometry", "transport"}
    x = np.linspace(5.0, 20.0, 151)
    prov = find_transitions(x, _kinked(x, rng=rng), methods="kink").robust().provenance()
    assert prov["method_name"].iloc[0] == "curvature extremum (kink criterion)"
    assert "Asti" in prov["citations"].iloc[0]
    assert "Semeniuk" in prov["citations"].iloc[0]


def test_fisher_registry_entry_cites_fisher_1962():
    info = METHODS["fisher"]
    assert info.name == "Fisher specific-heat criterion"
    markers = [m.marker for m in METHODS.values()]
    assert len(set(markers)) == len(markers)  # every criterion keeps its own shape
    texts = [c.text for c in info.citations]
    assert any(t.startswith("M. E. Fisher, Philos. Mag. 7, 1731 (1962)") for t in texts)
    origins = {(c.kind, c.role) for c in info.citations}
    assert origins == {("statistical", "origin"), ("magnetometry", "origin")}
    # The derivative criterion no longer claims Fisher lives outside this module.
    assert "Fisher" not in "; ".join(c.text for c in METHODS["derivative"].citations)


def test_fisher_hits_are_never_grouped_even_beside_derivative_flanks():
    """With 'derivative' running as well, the cusp keeps its flanks and the
    Fisher maximum still stands alone."""
    t = _toy_afm_t()
    chi = _toy_afm_chi(t) + np.random.default_rng(3).normal(scale=FISHER_NOISE, size=t.shape)
    result = find_transitions(
        t, chi, methods=("peak", "derivative", "fisher"), x_name="T", x_unit="K"
    )
    cands, dets = result.candidates, result.detections
    fisher_ids = set(cands.loc[cands["method"] == "fisher", "candidate_id"])
    assert fisher_ids
    assert (cands.loc[cands["method"] == "fisher", "n_methods"] == 1).all()
    for cid, members in dets.groupby("candidate_id"):
        methods = set(members["method"])
        assert ("fisher" in methods) == (cid in fisher_ids)
        if "fisher" in methods:
            assert methods == {"fisher"}
    cusp = cands[(cands["method"] == "peak") & (cands["sign"] == "max")]
    assert int(cusp["n_methods"].iloc[0]) == 2  # peak + its derivative flanks


@pytest.mark.parametrize("shape", ["cusp", "afm"])
def test_adding_fisher_leaves_the_other_methods_untouched(shape):
    """Fisher rows are appended; every other row, id, detection and noise
    record is byte-for-byte what the same call gives without 'fisher'."""
    x = np.linspace(5.0, 20.0, 151) if shape == "cusp" else _toy_afm_t()
    rng = np.random.default_rng(17)
    if shape == "cusp":
        y = simulate.synthetic_chi_peak(
            x, centre=12.5, width=1.0, height=1.0, background=0.2, noise=0.02, rng=rng
        )
    else:
        y = _toy_afm_chi(x) + rng.normal(scale=FISHER_NOISE, size=x.shape)
    kw = dict(x_name="T", x_unit="K")
    base = find_transitions(x, y, methods=("peak", "derivative"), **kw)
    more = find_transitions(x, y, methods=("peak", "derivative", "fisher"), **kw)

    others = more.candidates[more.candidates["method"] != "fisher"].reset_index(drop=True)
    pd.testing.assert_frame_equal(others, base.candidates)
    other_dets = more.detections[more.detections["method"] != "fisher"].reset_index(drop=True)
    pd.testing.assert_frame_equal(other_dets, base.detections)
    # Fisher rows come after the grouped ones, in order of location.
    n = len(base.candidates)
    tail = more.candidates.iloc[n:]
    assert (tail["method"] == "fisher").all()
    assert tail["T_K"].is_monotonic_increasing
    assert list(more.candidates["candidate_id"]) == list(range(len(more.candidates)))
    for key, value in base.params.items():
        if key != "methods":
            assert more.params[key] == value, key
    for name, curve in base.curves[("", "")].items():
        np.testing.assert_array_equal(more.curves[("", "")][name], curve)


def test_fisher_refuses_an_axis_that_is_not_absolute_temperature():
    x = np.linspace(0.5, 10.0, 60)
    y = np.exp(-x)
    with pytest.raises(ValueError, match="absolute temperature.*x_unit is 'T'"):
        find_transitions(x, y, methods="fisher", x_name="H", x_unit="T")
    with pytest.raises(ValueError, match="absolute temperature.*x_unit is ''"):
        find_transitions(x, y, methods="fisher")
    with pytest.raises(ValueError, match="every x must be positive.*smallest is -1"):
        find_transitions(
            np.linspace(-1.0, 10.0, 60), y, methods=("peak", "fisher"), x_name="T", x_unit="K"
        )
    with pytest.raises(ValueError, match="smallest is 0"):
        find_transitions(
            np.linspace(0.0, 10.0, 60), y, methods="fisher", x_name="T", x_unit="K"
        )


def test_criterion_fisher_is_refused():
    t = _toy_afm_t()
    with pytest.raises(ValueError, match="criterion='fisher' has nothing to rank"):
        find_transitions(
            t, _toy_afm_chi(t), methods=("peak", "fisher"), x_name="T", x_unit="K",
            criterion="fisher",
        )


def test_fisher_noise_is_resolved_separately_for_y_times_t():
    """params['fisher_noise'] describes z = y*T: the instrument estimate is
    median(y_err * T) and the scatter estimate is z's own; the signal's
    record is what it would be without 'fisher'."""
    t = _toy_afm_t()
    rng = np.random.default_rng(29)
    chi = _toy_afm_chi(t) + rng.normal(scale=FISHER_NOISE, size=t.shape)
    err = np.full_like(t, FISHER_NOISE) * (1.0 + 0.5 * rng.random(t.shape))
    kw = dict(x_name="T", x_unit="K", y_err=err)
    with_f = find_transitions(t, chi, methods=("peak", "fisher"), **kw)
    without = find_transitions(t, chi, methods="peak", **kw)

    rec = with_f.params["fisher_noise"]
    z = t * chi
    scatter_z = float(np.median(np.abs(np.diff(z, n=2)))) / (0.6745 * np.sqrt(6.0))
    assert rec["noise"] == "auto"
    assert rec["noise_source"] == "instrument"
    assert rec["noise_sigma_instrument"] == pytest.approx(float(np.median(err * t)), rel=1e-12)
    assert rec["noise_sigma"] == rec["noise_sigma_instrument"]
    assert rec["noise_sigma_scatter"] == pytest.approx(scatter_z, rel=1e-12)
    assert rec["noise_ratio"] == pytest.approx(
        rec["noise_sigma_instrument"] / rec["noise_sigma_scatter"], rel=1e-12
    )
    for key in ("noise", "noise_source", "noise_sigma", "noise_sigma_instrument",
                "noise_sigma_scatter", "noise_ratio"):
        assert with_f.params[key] == without.params[key], key
        # Plain Python values, so the record reads the same on every numpy.
        assert type(rec[key]) in (str, float, type(None)), key
    assert "fisher_noise" not in without.params

    # 'scatter' pins z's own second-difference estimate.
    pinned = find_transitions(t, chi, methods="fisher", noise="scatter", **kw)
    assert pinned.params["fisher_noise"]["noise_sigma"] == pytest.approx(scatter_z, rel=1e-12)


def test_fisher_noise_caller_float_is_scaled_by_the_median_temperature():
    """A float noise= is a sigma in y's units; for y*T it becomes
    noise * median(T), never the same number read as a sigma in y*K."""
    t = _toy_afm_t()
    chi = _toy_afm_chi(t)
    result = find_transitions(
        t, chi, methods=("peak", "fisher"), x_name="T", x_unit="K", noise=FISHER_NOISE
    )
    rec = result.params["fisher_noise"]
    assert rec["noise"] == FISHER_NOISE
    assert rec["noise_source"] == "caller"
    assert rec["noise_sigma"] == pytest.approx(FISHER_NOISE * float(np.median(t)), rel=1e-12)
    assert result.params["noise_sigma"] == FISHER_NOISE
