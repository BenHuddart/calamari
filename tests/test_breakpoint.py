"""Tests for the segmented_linear and bent_cable models and fit_breakpoint.

All data are synthetic straight lines and corners with invented numbers,
each test seeding its own ``np.random.default_rng`` -- no instrument files,
matching the project testing policy.
"""

from __future__ import annotations

import dataclasses
import re
import warnings

import matplotlib

matplotlib.use("Agg")  # headless; must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import calamari  # noqa: E402
from calamari.fitting import (  # noqa: E402
    MODELS,
    BreakpointResult,
    bent_cable,
    fit,
    fit_breakpoint,
    fit_windows,
    segmented_linear,
)
from calamari.plot import plot_fit  # noqa: E402

# A corner in the middle of a 51-point range, slopes +0.2 then 0, noise 0.004:
# invented numbers on the scale of the planning simulation.
X = np.linspace(4.0, 14.0, 51)
X_BREAK, Y_BREAK, SLOPE_BELOW, SLOPE_CHANGE, NOISE = 8.2, 1.0, 0.2, -0.2, 0.004


def _kink(rng: np.random.Generator, x: np.ndarray = X, x_break: float = X_BREAK) -> np.ndarray:
    clean = segmented_linear(x, x_break, Y_BREAK, SLOPE_BELOW, SLOPE_CHANGE)
    return clean + rng.normal(scale=NOISE, size=x.size)


def _line(rng: np.random.Generator) -> np.ndarray:
    return Y_BREAK + SLOPE_BELOW * (X - X_BREAK) + rng.normal(scale=NOISE, size=X.size)


# --------------------------------------------------------------------------
# The model and its guess
# --------------------------------------------------------------------------


def test_segmented_linear_matches_formula():
    x = np.linspace(-3.0, 5.0, 17)
    got = segmented_linear(x, 1.5, 2.0, 0.5, -1.25)
    below = 2.0 + 0.5 * (x - 1.5)
    above = 2.0 + (0.5 - 1.25) * (x - 1.5)
    assert np.allclose(got, np.where(x < 1.5, below, above))


def test_segmented_linear_is_registered_with_a_guess():
    defn = MODELS["segmented_linear"]
    assert defn.param_names == ("x_break", "y_break", "slope_below", "slope_change")
    assert defn.guess is not None


def test_guess_finds_a_planted_corner_on_the_sample_grid_exactly():
    y = segmented_linear(X, X[20], Y_BREAK, SLOPE_BELOW, SLOPE_CHANGE)  # corner at a sample
    seed = MODELS["segmented_linear"].guess(X, y)
    assert seed["x_break"] == X[20]
    assert seed["slope_change"] == pytest.approx(SLOPE_CHANGE, rel=1e-9)


def test_guess_is_deterministic():
    y = _kink(np.random.default_rng(21))
    guess = MODELS["segmented_linear"].guess
    assert guess(X, y) == guess(X.copy(), y.copy())


def test_guess_keeps_three_points_on_each_side():
    # A corner planted outside the eligible grid is seeded at the grid's end.
    y = segmented_linear(X, X[1], Y_BREAK, SLOPE_BELOW, SLOPE_CHANGE)
    seed = MODELS["segmented_linear"].guess(X, y)
    assert seed["x_break"] >= X[3]


def test_guess_falls_back_rather_than_raising_on_too_few_points():
    # fit() calls the guess before its own dof check, so a raise here would
    # turn FT-1's success=False into an exception.
    seed = MODELS["segmented_linear"].guess(X[:5], X[:5] * 2.0)
    assert np.isfinite(list(seed.values())).all()
    assert fit("segmented_linear", X[:4], X[:4] * 2.0).success is False


def test_plain_fit_recovers_the_corner_through_the_registry():
    y = _kink(np.random.default_rng(22))
    result = fit("segmented_linear", X, y)
    assert result.success
    assert result.parameters["x_break"] == pytest.approx(X_BREAK, abs=0.05)


# --------------------------------------------------------------------------
# fit_breakpoint on a real corner
# --------------------------------------------------------------------------


def test_sharp_corner_hesse_covers_truth_at_about_the_nominal_rate():
    # 50 seeded draws of a sharp corner. HESSE is calibrated here (66 % over
    # 300 draws when this was written), so the 1-sigma interval should cover
    # the truth ~68 % of the time. The binomial sd of a 50-draw rate near
    # 0.68 is ~0.066, so [0.50, 0.86] is about +/-2.7 sd: wide enough not to
    # flake on a seed change, and it fails for an error bar off by a factor
    # of 2 either way (expected coverage 0.95 or 0.38). This seed gives 0.74.
    rng = np.random.default_rng(2026)
    found, errors = [], []
    for _ in range(50):
        bp = fit_breakpoint(X, _kink(rng))
        assert bp.identified
        found.append(bp.x_break)
        errors.append(bp.x_break_err)
    found, errors = np.array(found), np.array(errors)
    coverage = np.mean(np.abs(found - X_BREAK) <= errors)
    assert 0.50 <= coverage <= 0.86
    # Unbiased: the mean is within 3 standard errors of the truth.
    assert abs(found.mean() - X_BREAK) <= 3 * found.std() / np.sqrt(found.size)
    # And the typical error bar matches the actual spread. A 50-draw sd is
    # itself uncertain by ~10 %, so a band of +/-30 % is about 3 of those;
    # this seed gives 0.93.
    assert 0.7 <= np.median(errors) / found.std() <= 1.3


def test_result_reports_slopes_counts_and_no_warnings():
    y = _kink(np.random.default_rng(23))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bp = fit_breakpoint(X, y)
    assert bp.identified and bp.warnings == ()
    assert bp.slope_below == pytest.approx(SLOPE_BELOW, abs=5 * bp.slope_below_err)
    assert bp.slope_change == pytest.approx(SLOPE_CHANGE, abs=5 * bp.slope_change_err)
    assert bp.slope_above == pytest.approx(SLOPE_BELOW + SLOPE_CHANGE, abs=5 * bp.slope_above_err)
    assert bp.n_below == int(np.sum(X < bp.x_break))
    assert bp.n_above == int(np.sum(X > bp.x_break))
    assert bp.n_points == X.size
    assert bp.x_break == bp.segmented.parameters["x_break"]


def test_slope_above_error_matches_the_covariance_by_hand():
    bp = fit_breakpoint(X, _kink(np.random.default_rng(24)))
    names = bp.segmented.covariance_parameters
    cov = bp.segmented.covariance
    i, j = names.index("slope_below"), names.index("slope_change")
    by_hand = np.sqrt(cov[i, i] + cov[j, j] + 2.0 * cov[i, j])
    assert bp.slope_above_err == pytest.approx(by_hand, rel=1e-6)
    # And the correlation matters: ignoring it would give a different answer.
    assert not np.isclose(by_hand, np.hypot(bp.slope_below_err, bp.slope_change_err), rtol=0.05)


def test_information_criteria_follow_the_rss_formula_and_ignore_yerr():
    y = _kink(np.random.default_rng(25))
    bp = fit_breakpoint(X, y)
    n = X.size
    slope, intercept = np.polyfit(X, y, 1)
    assert bp.rss_linear == pytest.approx(np.sum((y - slope * X - intercept) ** 2), rel=1e-12)
    gain = n * np.log(bp.rss_linear / bp.rss_segmented)
    assert bp.delta_aic == pytest.approx(gain - 4.0, rel=1e-12)
    assert bp.delta_bic == pytest.approx(gain - 2.0 * np.log(n), rel=1e-12)
    # With yerr=None the fit's own RSS is the exact minimum the criteria use.
    assert bp.segmented.chi_squared == pytest.approx(bp.rss_segmented, rel=1e-6)
    # An error column 300x the real scatter changes the parameter errors (it
    # is taken at face value) but not the comparison.
    inflated = fit_breakpoint(X, y, yerr=300 * NOISE)
    assert inflated.delta_bic == bp.delta_bic
    assert inflated.x_break_err > 50 * bp.x_break_err
    assert inflated.params["yerr"] == "scalar"


def test_profile_is_a_table_over_the_sample_grid_with_a_display_interval():
    y = _kink(np.random.default_rng(26))
    bp = fit_breakpoint(X, y)
    assert isinstance(bp.profile, pd.DataFrame)
    assert list(bp.profile.columns) == ["x_break", "rss"]
    # The grid is every sample with three points strictly on each side.
    assert np.array_equal(bp.profile["x_break"].to_numpy(), X[3:-3])
    grid_best = bp.profile["x_break"][bp.profile["rss"].idxmin()]
    assert grid_best == MODELS["segmented_linear"].guess(X, y)["x_break"]
    assert bp.profile_threshold == pytest.approx(bp.rss_segmented * np.exp(1.0 / X.size))
    lower, upper = bp.profile_interval
    assert lower < bp.x_break < upper
    # The ends are where the exact profile crosses the threshold.
    from calamari.fitting.models import _segmented_solve

    for end in (lower, upper):
        assert _segmented_solve(X, y, end)[1] == pytest.approx(bp.profile_threshold, rel=1e-6)
    # For a sharp corner it is comparable to the HESSE interval.
    assert 0.5 < (upper - lower) / (2 * bp.x_break_err) < 2.0


@pytest.mark.parametrize("slope_change", [-0.2, -0.02, 0.0])
def test_rss_segmented_is_the_exact_minimum_over_the_continuous_range(slope_change):
    # The minimum is found from knots and each interval's two-line solution,
    # not by searching; a dense brute-force grid must never beat it.
    from calamari.fitting.models import _segmented_profile

    rng = np.random.default_rng(40)
    y = segmented_linear(X, 9.13, Y_BREAK, SLOPE_BELOW, slope_change)
    y = y + rng.normal(scale=NOISE, size=X.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the null case warns
        bp = fit_breakpoint(X, y)
    dense = np.linspace(X[2], X[-3], 20001)
    assert bp.rss_segmented <= _segmented_profile(X, y, dense).min() * (1 + 1e-12)


def test_tiny_magnitude_y_converges_to_the_same_corner():
    # Convergence must be invariant under rescaling to emu/Oe-sized data.
    y = _kink(np.random.default_rng(27))
    unit = fit_breakpoint(X, y)
    tiny = fit_breakpoint(X, y * 1e-6)
    assert tiny.segmented.success and tiny.identified
    assert tiny.x_break == pytest.approx(unit.x_break, rel=1e-8)
    assert tiny.x_break_err == pytest.approx(unit.x_break_err, rel=1e-4)
    assert tiny.slope_change == pytest.approx(unit.slope_change * 1e-6, rel=1e-6)
    assert tiny.delta_bic == pytest.approx(unit.delta_bic, rel=1e-9)
    # The plain registry path too, seeded away from the answer.
    seeded = fit("segmented_linear", X, y * 1e-6, initial={"x_break": 9.0})
    assert seeded.success
    assert seeded.parameters["x_break"] == pytest.approx(unit.x_break, rel=1e-6)


def test_same_input_same_output_and_order_does_not_matter():
    y = _kink(np.random.default_rng(28))
    first, second = fit_breakpoint(X, y), fit_breakpoint(X, y)
    assert first.x_break == second.x_break
    assert first.x_break_err == second.x_break_err
    assert first.profile_interval == second.profile_interval
    order = np.random.default_rng(1).permutation(X.size)
    shuffled = fit_breakpoint(X[order], y[order])
    assert shuffled.x_break == pytest.approx(first.x_break, rel=1e-10)
    assert shuffled.delta_bic == pytest.approx(first.delta_bic, rel=1e-10)


def test_noise_free_corner_is_exact_and_noise_free_line_is_not_a_break():
    exact = fit_breakpoint(X, segmented_linear(X, X_BREAK, Y_BREAK, SLOPE_BELOW, SLOPE_CHANGE))
    assert exact.identified
    assert exact.x_break == pytest.approx(X_BREAK, abs=1e-9)
    # Both sums of squares are round-off here; their ratio must not read as
    # evidence for a break.
    with pytest.warns(UserWarning, match="delta_bic"):
        line = fit_breakpoint(X, 1.0 + 0.2 * X)
    assert not line.identified


# --------------------------------------------------------------------------
# Warnings: no break, and a corner held by a limit
# --------------------------------------------------------------------------


def test_no_break_warns_and_is_not_identified():
    with pytest.warns(UserWarning, match=r"delta_bic = .* <= 0.*not identified"):
        bp = fit_breakpoint(X, _line(np.random.default_rng(3)))
    assert bp.identified is False
    assert bp.delta_bic <= 0
    assert any("delta_bic" in message for message in bp.warnings)
    # The result is still returned in full (warn, don't fix).
    assert np.isfinite(bp.x_break) and np.isfinite(bp.x_break_err)


def test_corner_held_at_a_given_limit_warns():
    y = _kink(np.random.default_rng(29))
    with pytest.warns(UserWarning, match=r"of the limit 9 .*limits="):
        bp = fit_breakpoint(X, y, limits=(9.0, 12.0))
    assert bp.identified is False
    assert bp.delta_bic > 0  # a break is preferred; the limit is the problem
    assert bp.params["limits"] == (9.0, 12.0) and bp.params["limits_given"] is True


def test_corner_beyond_the_default_range_warns_at_the_edge():
    y = _kink(np.random.default_rng(30), x_break=4.3)  # below the 3rd point, 4.4
    with pytest.warns(UserWarning, match=r"of the limit 4\.4"):
        bp = fit_breakpoint(X, y)
    assert bp.identified is False
    assert bp.params["limits"] == (X[2], X[-3]) and bp.params["limits_given"] is False


def test_a_failed_segmented_fit_warns_and_reports_nan(monkeypatch):
    import calamari.fitting.breakpoint as module

    real_fit = module.fit

    def failing(model, *args, **kwargs):
        result = real_fit(model, *args, **kwargs)
        if model == "segmented_linear":
            result = dataclasses.replace(result, success=False)
        return result

    monkeypatch.setattr(module, "fit", failing)
    with pytest.warns(UserWarning, match="did not converge"):
        bp = fit_breakpoint(X, _kink(np.random.default_rng(31)))
    assert bp.identified is False
    assert np.isnan(bp.x_break) and np.isnan(bp.slope_above)
    assert bp.n_below == bp.n_above == 0


# --------------------------------------------------------------------------
# Registry integration and plotting
# --------------------------------------------------------------------------


def test_fit_windows_runs_segmented_linear_over_three_windows():
    y = _kink(np.random.default_rng(32))
    scan = fit_windows("segmented_linear", X, y, windows=[(4.0, 14.0), (5.0, 12.0), (6.0, 11.0)])
    assert scan.table["success"].all()
    assert list(scan.table.columns[7:9]) == ["x_break", "x_break_err"]
    assert np.allclose(scan.table["x_break"], X_BREAK, atol=0.05)


def test_plot_fit_overlays_the_segmented_result_unchanged():
    y = _kink(np.random.default_rng(33))
    bp = fit_breakpoint(X, y)
    ax = plot_fit(X, bp.segmented)
    line = ax.get_lines()[-1]
    assert np.allclose(line.get_ydata(), bp.segmented.curve(line.get_xdata()))
    plt.close(ax.figure)


def test_exported_and_frozen():
    assert calamari.fit_breakpoint is fit_breakpoint
    assert calamari.BreakpointResult is BreakpointResult
    bp = fit_breakpoint(X, _kink(np.random.default_rng(34)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        bp.x_break = 0.0  # type: ignore[misc]
    assert "calamari_version" in bp.params


# --------------------------------------------------------------------------
# Refusals: each names the fix
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"x": X[:6], "y": X[:6]}, r"at least 7 points.*Widen the range"),
        ({"x": X, "y": X[:-1]}, r"same length.*Pass one sweep"),
        ({"x": X.reshape(3, 17), "y": X.reshape(3, 17)}, r"1-D"),
        (
            {"x": X, "y": np.where(X > 10, np.nan, X)},
            r"must be finite.*keep = np.isfinite",
        ),
        ({"x": np.repeat(X[:2], 4), "y": np.repeat(X[:2], 4)}, r"distinct.*Widen the range"),
        ({"yerr": np.ones(5)}, r"yerr must be a scalar or a 1-D array.*yerr=None"),
        ({"yerr": np.ones((51, 1))}, r"yerr must be a scalar or a 1-D array"),
        ({"yerr": 0.0}, r"finite and positive.*yerr=None"),
        ({"yerr": np.where(X > 10, np.nan, 0.01)}, r"finite and positive"),
        ({"limits": (3.0, 12.0)}, r"reach outside \(4\.4, 13\.6\).*Narrow the limits"),
        ({"limits": (5.0, 14.0)}, r"reach outside"),
        ({"limits": (12.0, 9.0)}, r"lower < upper"),
        ({"limits": (9.0, np.inf)}, r"finite"),
        ({"limits": 9.0}, r"\(lower, upper\) pair.*limits=\(8\.0, 12\.0\)"),
        ({"limits": (True, 12.0)}, r"\(lower, upper\) pair"),
        ({"limits": (8.21, 8.29)}, r"hold no sample position.*Widen the limits"),
    ],
)
def test_refusals_name_the_fix(kwargs, match):
    call = {"x": X, "y": _kink(np.random.default_rng(35))}
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        fit_breakpoint(call.pop("x"), call.pop("y"), **call)


# --------------------------------------------------------------------------
# The bent cable: a corner rounded by a parabolic bend of half-width gamma
# --------------------------------------------------------------------------

#: The rounded corner the bent-cable tests plant: a softplus of width 0.4 on
#: the corner above, whose straight asymptotes meet at X_BREAK. A softplus,
#: not a bent cable, so the model is not handed its own shape.
ROUNDING = 0.4


def _rounded(rng: np.random.Generator, x: np.ndarray = X, width: float = ROUNDING) -> np.ndarray:
    clean = (
        Y_BREAK
        + SLOPE_BELOW * (x - X_BREAK)
        + SLOPE_CHANGE * width * np.logaddexp(0.0, (x - X_BREAK) / width)
    )
    return clean + rng.normal(scale=NOISE, size=x.size)


def _grid(lo: float, hi: float) -> np.ndarray:
    """Samples every 0.2 from lo to hi inclusive, rounded so the ends are exact."""
    return np.round(np.arange(lo, hi + 1e-9, 0.2), 10)


def test_bent_cable_matches_the_papers_basis_and_is_smooth():
    # Chiu, Lockhart & Routledge: q = (x - tau + gamma)^2 / (4 gamma) on
    # |x - tau| <= gamma, x - tau above, 0 below; y = b0 + b1 x + b2 q.
    x = np.linspace(-3.0, 5.0, 801)
    tau, gamma, y_break, below, change = 1.5, 0.7, 2.0, 0.5, -1.25
    offset = x - tau
    q = np.where(
        np.abs(offset) <= gamma,
        (offset + gamma) ** 2 / (4 * gamma),
        np.where(offset > gamma, offset, 0.0),
    )
    got = bent_cable(x, tau, y_break, below, change, gamma)
    assert np.allclose(got, y_break + below * offset + change * q, rtol=0, atol=1e-12)
    # Value and slope are continuous at both ends of the bend.
    for edge in (tau - gamma, tau + gamma):
        h = 1e-6
        left, mid, right = bent_cable(
            np.array([edge - h, edge, edge + h]), tau, y_break, below, change, gamma
        )
        assert right - mid == pytest.approx(mid - left, rel=1e-4)
    # At the centre the curve sits change * gamma / 4 off the arms' meeting point.
    centre = bent_cable(np.array([tau]), tau, y_break, below, change, gamma)[0]
    assert centre == pytest.approx(y_break + change * gamma / 4)
    # gamma -> 0 is the sharp corner, with no division by zero; only |gamma| counts.
    with np.errstate(all="raise"):
        sharp = bent_cable(x, tau, y_break, below, change, 0.0)
    assert np.array_equal(sharp, segmented_linear(x, tau, y_break, below, change))
    assert np.array_equal(bent_cable(x, tau, y_break, below, change, -gamma), got)


def test_bent_cable_is_registered_with_a_deterministic_guess():
    defn = MODELS["bent_cable"]
    assert defn.param_names == ("x_break", "y_break", "slope_below", "slope_change", "gamma")
    y = _rounded(np.random.default_rng(41))
    seed = defn.guess(X, y)
    assert seed == defn.guess(X.copy(), y.copy())
    # The corner and slopes are the sharp corner's profile guess ...
    sharp = MODELS["segmented_linear"].guess(X, y)
    assert {k: seed[k] for k in sharp} == sharp
    # ... and the bend starts at two median sample spacings.
    assert seed["gamma"] == pytest.approx(2 * (X[1] - X[0]))
    # Too few distinct x for a spacing: no gamma seed, but no raise either.
    assert "gamma" not in defn.guess(np.ones(5), np.arange(5.0))


def test_bent_cable_holds_a_rounded_corner_still_where_the_sharp_one_moves():
    # A symmetric rounded corner, three lopsided fitting ranges, 30 seeded
    # draws each. The sharp corner (carried on the same result as
    # `segmented`) moves with the range; the bent cable holds. Over 300 draws
    # per range the mean offsets were +0.056, +0.167, -0.080 for the sharp
    # corner and +0.008, +0.037, -0.014 for the bent cable (sd 0.017 to
    # 0.035): the bent cable's parabola is not the planted softplus, so a
    # little bias remains. 0.06 is the largest of those plus about three
    # standard errors of a 30-draw mean (0.035 / sqrt(30) = 0.006). This seed
    # gives +0.061, +0.165, -0.082 and +0.011, +0.032, -0.021.
    rng = np.random.default_rng(2026)
    sharp_means, bent_means = [], []
    for lo, hi in [(4.0, 14.0), (6.0, 16.0), (4.0, 11.0)]:
        x = _grid(lo, hi)
        sharp, bent = [], []
        for _ in range(30):
            bp = fit_breakpoint(x, _rounded(rng, x), model="bent_cable")
            assert bp.identified and bp.warnings == ()
            assert bp.delta_bic_vs_segmented > 0  # the rounding is resolved
            sharp.append(bp.segmented.parameters["x_break"] - X_BREAK)
            bent.append(bp.x_break - X_BREAK)
        sharp_means.append(np.mean(sharp))
        bent_means.append(np.mean(bent))
    sharp_means, bent_means = np.array(sharp_means), np.array(bent_means)
    # The problem is present: the sharp corner moves by far more than its scatter.
    assert np.ptp(sharp_means) > 0.15
    # The bent cable stays near the truth in every range, and moves less than half as much.
    assert np.all(np.abs(bent_means) < 0.06)
    assert np.all(np.abs(bent_means) < np.abs(sharp_means) / 2)
    assert np.ptp(bent_means) < np.ptp(sharp_means) / 2


def test_sharp_corner_puts_gamma_at_its_lower_limit_and_warns():
    spacing = X[1] - X[0]
    exact = segmented_linear(X, X_BREAK, Y_BREAK, SLOPE_BELOW, SLOPE_CHANGE)
    for y in (exact, _kink(np.random.default_rng(42))):
        with pytest.warns(
            UserWarning,
            match=r"gamma = .* lower limit .*not resolved.*model='segmented_linear'",
        ):
            bp = fit_breakpoint(X, y, model="bent_cable")
        assert bp.params["gamma_limits"] == pytest.approx((spacing / 2, X[-1] - X[0]))
        assert bp.gamma - spacing / 2 <= bp.gamma_err
        # The corner is still located, so this warns without unsetting identified.
        assert bp.identified
        assert bp.x_break == pytest.approx(X_BREAK, abs=0.05)
        assert bp.delta_bic_vs_segmented < 0  # the sharp corner is preferred
        assert sum("lower limit" in message for message in bp.warnings) == 1


def test_default_model_is_the_sharp_corner_unchanged():
    # Every field the result had before bent_cable existed is identical with
    # and without model=, and the new fields keep their defaults. (Checked
    # against main when this was written: all pre-existing fields, warnings,
    # profiles and both FitResults compared equal over 24 cases.)
    y = _kink(np.random.default_rng(43))
    default = fit_breakpoint(X, y)
    explicit = fit_breakpoint(X, y, model="segmented_linear")
    names = [f.name for f in dataclasses.fields(BreakpointResult)]
    for name in names[: names.index("params")]:
        a, b = getattr(default, name), getattr(explicit, name)
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b)
        elif name in ("segmented", "linear"):
            assert a.parameters == b.parameters and a.uncertainties == b.uncertainties
        else:
            assert a == b, name
    assert list(default.profile.columns) == ["x_break", "rss"]
    assert default.model == "segmented_linear" and default.params["model"] == "segmented_linear"
    assert default.bent_cable is None
    assert "gamma_limits" not in default.params
    for name in (
        "gamma",
        "gamma_err",
        "rss_bent_cable",
        "delta_aic_vs_segmented",
        "delta_bic_vs_segmented",
    ):
        assert np.isnan(getattr(default, name)), name


def test_bent_cable_result_carries_both_comparisons():
    y = _rounded(np.random.default_rng(44))
    bp = fit_breakpoint(X, y, model="bent_cable")
    n = X.size
    assert bp.model == "bent_cable" and bp.params["model"] == "bent_cable"
    assert isinstance(bp.bent_cable, calamari.FitResult) and bp.bent_cable.success
    assert bp.segmented.model == "segmented_linear"
    assert bp.x_break == bp.bent_cable.parameters["x_break"]
    assert bp.gamma == bp.bent_cable.parameters["gamma"]
    assert bp.gamma_err == bp.bent_cable.uncertainties["gamma"]
    # The sharp corner's own numbers are those of the default call.
    sharp = fit_breakpoint(X, y)
    assert bp.rss_segmented == sharp.rss_segmented and bp.rss_linear == sharp.rss_linear
    # From RSS, k = 5 against 2 (line) and against 4 (sharp corner).
    gain_line = n * np.log(bp.rss_linear / bp.rss_bent_cable)
    assert bp.delta_aic == pytest.approx(gain_line - 2 * 3, rel=1e-12)
    assert bp.delta_bic == pytest.approx(gain_line - 3 * np.log(n), rel=1e-12)
    gain_sharp = n * np.log(bp.rss_segmented / bp.rss_bent_cable)
    assert bp.delta_aic_vs_segmented == pytest.approx(gain_sharp - 2, rel=1e-12)
    assert bp.delta_bic_vs_segmented == pytest.approx(gain_sharp - np.log(n), rel=1e-12)
    assert bp.delta_bic_vs_segmented > 0
    # With yerr=None the fitted curve reaches the RSS the criteria use.
    assert bp.bent_cable.chi_squared == pytest.approx(bp.rss_bent_cable, rel=1e-6)
    # The profile is over the same grid, with the bend profiled out, and its
    # display interval brackets the centre.
    assert list(bp.profile.columns) == ["x_break", "rss", "gamma"]
    assert np.array_equal(bp.profile["x_break"].to_numpy(), X[3:-3])
    assert bp.profile["rss"].min() >= bp.rss_bent_cable * (1 - 1e-9)
    lower, upper = bp.profile_interval
    assert lower < bp.x_break < upper
    assert bp.profile_threshold == pytest.approx(bp.rss_bent_cable * np.exp(1.0 / n))
    # Slopes and the derived slope above, from the bent cable.
    assert bp.slope_above == pytest.approx(bp.slope_below + bp.slope_change)
    assert bp.slope_change == pytest.approx(SLOPE_CHANGE, abs=5 * bp.slope_change_err)


def test_bent_cable_criteria_ignore_yerr():
    y = _rounded(np.random.default_rng(45))
    plain = fit_breakpoint(X, y, model="bent_cable")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # errors this large reach the limits
        inflated = fit_breakpoint(X, y, model="bent_cable", yerr=300 * NOISE)
    assert inflated.params["yerr"] == "scalar"
    # MIGRAD can refine the numerical profile by a few floating-point ulps;
    # changing error scale must preserve the scores to numerical precision.
    assert inflated.delta_bic == pytest.approx(plain.delta_bic, rel=1e-12, abs=1e-12)
    assert inflated.delta_bic_vs_segmented == pytest.approx(
        plain.delta_bic_vs_segmented, rel=1e-12, abs=1e-12
    )
    assert inflated.x_break_err > 50 * plain.x_break_err


def test_a_bend_that_leaves_an_arm_unestablished_warns_and_is_not_identified():
    # A wide rounding seen through a narrow range: the bend reaches past the
    # low end, so there is no straight arm below it to meet.
    x = _grid(7.0, 9.4)
    y = _rounded(np.random.default_rng(4), x, width=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bp = fit_breakpoint(x, y, model="bent_cable")
    arm = r"the bend spans .* leaving 0 point\(s\) below.*Pass a wider range"
    assert any(re.search(arm, str(w.message)) for w in caught)
    assert bp.identified is False
    assert bp.x_break - bp.gamma < x[3]


def test_no_break_warns_for_the_bent_cable_too():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bp = fit_breakpoint(X, _line(np.random.default_rng(3)), model="bent_cable")
    assert re.search(r"delta_bic = .* <= 0.*not identified", str(caught[0].message))
    assert bp.identified is False and bp.delta_bic <= 0
    # Every warning raised is also kept on the result.
    assert bp.warnings == tuple(str(w.message) for w in caught)


def test_a_failed_bent_cable_fit_warns_and_reports_nan(monkeypatch):
    import calamari.fitting.breakpoint as module

    real_fit = module.fit

    def failing(model, *args, **kwargs):
        result = real_fit(model, *args, **kwargs)
        if model == "bent_cable":
            result = dataclasses.replace(result, success=False)
        return result

    monkeypatch.setattr(module, "fit", failing)
    with pytest.warns(UserWarning, match=r"the bent_cable fit did not converge"):
        bp = fit_breakpoint(X, _rounded(np.random.default_rng(46)), model="bent_cable")
    assert bp.identified is False
    assert np.isnan(bp.x_break) and np.isnan(bp.gamma)
    assert np.isfinite(bp.rss_bent_cable)  # the profiled minimum still stands


def test_tiny_magnitude_y_converges_for_the_bent_cable():
    # Include a fifth parameter on the scale of x when rescaling the response.
    y = _rounded(np.random.default_rng(47))
    unit = fit_breakpoint(X, y, model="bent_cable")
    tiny = fit_breakpoint(X, y * 1e-6, model="bent_cable")
    assert tiny.bent_cable.success and tiny.identified
    assert tiny.x_break == pytest.approx(unit.x_break, rel=1e-8)
    assert tiny.gamma == pytest.approx(unit.gamma, rel=1e-6)
    assert tiny.gamma_err == pytest.approx(unit.gamma_err, rel=1e-4)
    assert tiny.delta_bic == pytest.approx(unit.delta_bic, rel=1e-9)
    assert tiny.delta_bic_vs_segmented == pytest.approx(unit.delta_bic_vs_segmented, rel=1e-9)
    # The plain registry path too, from the model's own guess.
    plain = fit("bent_cable", X, y * 1e-6)
    assert plain.success
    assert plain.parameters["x_break"] == pytest.approx(unit.x_break, rel=1e-6)


def test_bent_cable_is_deterministic():
    y = _rounded(np.random.default_rng(48))
    first = fit_breakpoint(X, y, model="bent_cable")
    second = fit_breakpoint(X, y, model="bent_cable")
    for name in (
        "x_break",
        "x_break_err",
        "gamma",
        "gamma_err",
        "rss_bent_cable",
        "delta_bic_vs_segmented",
        "profile_interval",
    ):
        assert getattr(first, name) == getattr(second, name), name
    pd.testing.assert_frame_equal(first.profile, second.profile)


def test_fit_windows_runs_bent_cable_over_three_windows():
    y = _rounded(np.random.default_rng(49))
    scan = fit_windows("bent_cable", X, y, windows=[(4.0, 14.0), (6.0, 14.0), (4.0, 11.0)])
    assert scan.table["success"].all()
    assert list(scan.table.columns[-2:]) == ["gamma", "gamma_err"]
    assert np.allclose(scan.table["x_break"], X_BREAK, atol=0.1)


def test_plot_fit_overlays_the_bent_cable_unchanged():
    bp = fit_breakpoint(X, _rounded(np.random.default_rng(50)), model="bent_cable")
    ax = plot_fit(X, bp.bent_cable)
    line = ax.get_lines()[-1]
    assert np.allclose(line.get_ydata(), bp.bent_cable.curve(line.get_xdata()))
    plt.close(ax.figure)


@pytest.mark.parametrize("model", ["hinge", "linear", "Bent_Cable", None, 3])
def test_an_unknown_model_is_refused_naming_the_choices(model):
    with pytest.raises(
        ValueError,
        match=r"model must be one of \['segmented_linear', 'bent_cable'\].*model='bent_cable'",
    ):
        fit_breakpoint(X, _kink(np.random.default_rng(51)), model=model)
