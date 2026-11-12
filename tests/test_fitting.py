"""Tests for calamari.fitting: model functions, registry, and the fit driver.

All data are synthetic (calamari.simulate signal helpers), seeded for exact
reproducibility -- no instrument files, matching the project testing policy.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from calamari import FitConstraintError
from calamari.constants import K_B_ERG_PER_K, MU_B_EMU
from calamari.exceptions import ModelRegistrationError
from calamari.fitting import (
    MODELS,
    FitResult,
    ModelDefinition,
    brillouin,
    curie_weiss,
    fit,
    fit_windows,
    inverse_curie_weiss,
    linear,
    register_model,
)
from calamari.sample import mu_eff_from_curie_constant
from calamari.simulate import curie_weiss_chi

# --------------------------------------------------------------------------
# Model functions
# --------------------------------------------------------------------------


def test_curie_weiss_matches_formula():
    T = np.linspace(5.0, 300.0, 40)
    got = curie_weiss(T, C=2.0, theta=-10.0, chi0=0.01)
    assert np.allclose(got, 2.0 / (T + 10.0) + 0.01)


def test_inverse_curie_weiss_is_reciprocal():
    T = np.linspace(20.0, 300.0, 30)
    C, theta = 3.0, -12.0
    assert np.allclose(inverse_curie_weiss(T, C, theta), 1.0 / curie_weiss(T, C, theta, 0.0))


def test_linear():
    x = np.linspace(-5, 5, 11)
    assert np.allclose(linear(x, 2.5, -1.0), 2.5 * x - 1.0)


def test_brillouin_saturates_to_Ms():
    # Large field / low temperature drives B_J -> 1, so M -> M_s.
    H = np.array([5.0e7])  # very large field in Oe
    m = brillouin(H, M_s=4.2, g=2.0, J=3.5, T_K=2.0)
    assert m[0] == pytest.approx(4.2, rel=1e-6)


def test_brillouin_is_odd_in_field():
    H = np.linspace(-2.0e4, 2.0e4, 21)
    m = brillouin(H, M_s=1.3, g=2.0, J=3.5, T_K=4.0)
    assert np.allclose(m, -m[::-1], atol=1e-12)
    # And zero at zero field (the coth-cancellation branch).
    assert brillouin(np.array([0.0]), 1.3, 2.0, 3.5, 4.0)[0] == 0.0


def test_brillouin_small_field_curie_slope():
    # At small y, B_J(y) ~ (J+1)/(3J) * y, so with y = g J mu_B H / (k_B T) the
    # J cancels: M/H ~ M_s (J+1) g mu_B / (3 k_B T) (the Curie-law slope). Use a
    # modest field (not H -> 0) to stay clear of the coth cancellation at y = 0.
    M_s, g, J, T = 1.0, 2.0, 3.5, 10.0
    H = np.array([10.0])
    expected_slope = M_s * (J + 1.0) * g * MU_B_EMU / (3.0 * K_B_ERG_PER_K * T)
    got = brillouin(H, M_s, g, J, T)[0] / H[0]
    assert got == pytest.approx(expected_slope, rel=1e-3)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_builtin_models_registered():
    assert {"curie_weiss", "inverse_curie_weiss", "linear", "brillouin"} <= set(MODELS)


def test_register_model_rejects_duplicate():
    with pytest.raises(ModelRegistrationError, match="already registered"):
        register_model(MODELS["linear"])


def test_register_and_use_custom_model():
    name = "test_quadratic_tmp"
    if name in MODELS:  # keep the test idempotent across reruns in one process
        del MODELS[name]
    defn = register_model(
        ModelDefinition(
            name=name,
            function=lambda x, a: a * np.asarray(x, dtype=float) ** 2,
            param_names=("a",),
            param_defaults={"a": 1.0},
            description="test-only quadratic",
        )
    )
    try:
        x = np.linspace(0, 3, 20)
        res = fit(defn, x, 2.0 * x**2)
        assert res.success
        assert res.parameters["a"] == pytest.approx(2.0, rel=1e-6)
    finally:
        del MODELS[name]


def test_curie_weiss_guess_is_close():
    T = np.linspace(10.0, 300.0, 50)
    chi = curie_weiss_chi(T, C=2.5, theta_K=-15.0, chi0=0.0)
    guess = MODELS["curie_weiss"].guess(T, chi)
    assert guess["C"] == pytest.approx(2.5, rel=0.05)
    assert guess["theta"] == pytest.approx(-15.0, abs=2.0)


# --------------------------------------------------------------------------
# fit(): parameter recovery and error conventions
# --------------------------------------------------------------------------


def _synthetic_cw(rng, *, C=2.5, theta=-18.0, chi0=0.002, noise=None):
    T = np.linspace(10.0, 300.0, 120)
    chi = curie_weiss_chi(T, C, theta, chi0)
    if noise is None:
        noise = 0.02 * chi.mean()
    y = chi + rng.normal(scale=noise, size=T.shape)
    return T, y, noise


def test_recovers_curie_weiss_within_uncertainty():
    rng = np.random.default_rng(12345)
    C, theta, chi0 = 2.5, -18.0, 0.002
    T, y, noise = _synthetic_cw(rng, C=C, theta=theta, chi0=chi0)
    res = fit("curie_weiss", T, y, yerr=np.full_like(T, noise))
    assert res.success
    # Truth within 3 sigma of the reported (absolute) uncertainties.
    for name, truth in (("C", C), ("theta", theta), ("chi0", chi0)):
        pull = abs(res.parameters[name] - truth) / res.uncertainties[name]
        assert pull < 3.0, (name, pull)
    # reduced chi^2 near 1 when the true per-point error is supplied.
    assert 0.5 < res.reduced_chi_squared < 2.0


def test_yerr_none_rescales_uncertainties():
    rng = np.random.default_rng(7)
    T, y, noise = _synthetic_cw(rng)
    with_err = fit("curie_weiss", T, y, yerr=np.full_like(T, noise))
    assumed = fit("curie_weiss", T, y)  # yerr=None -> rescaled by sqrt(chi2/dof)
    assert with_err.yerr_assumed is False
    assert assumed.yerr_assumed is True
    # A constant error scale does not move the least-squares minimum, and both
    # runs now converge to it: the assumed-error fit minimises against the
    # residual scatter, so neither stops short. MIGRAD's convergence goal
    # (EDM < 2e-4) puts each run within ~0.02 sigma of the minimum; allow
    # 0.01 sigma between them, which the fix meets with a wide margin. Before
    # the fix the two differed by 2% in C (~5 sigma).
    for name in ("C", "theta", "chi0"):
        assert assumed.parameters[name] == pytest.approx(
            with_err.parameters[name], abs=1e-2 * with_err.uncertainties[name]
        ), name
    # Same minimum, so the rescaled errors are the absolute ones times
    # sqrt(chi2/dof) of the supplied-error fit -- the definition of
    # absolute_sigma=False -- to the precision HESSE is evaluated at.
    for name in ("C", "theta", "chi0"):
        expected = with_err.uncertainties[name] * np.sqrt(with_err.reduced_chi_squared)
        assert assumed.uncertainties[name] == pytest.approx(expected, rel=1e-2), name


# --------------------------------------------------------------------------
# Assumed errors at any data scale
#
# With yerr=None the fit used to run against unit errors. For data of small
# magnitude -- a real chi in emu/Oe is 1e-5 to 1e-7 -- the least-squares cost
# is then already below MIGRAD's absolute convergence goal at the starting
# point, so the fit stopped where it began and reported success=True with
# the wrong parameters. These tests pin the fix: a constant sigma on the data's
# own scale, so nothing about a fit depends on the units y happens to be in.
# --------------------------------------------------------------------------

_CW_T = np.linspace(50.0, 300.0, 120)
_CW_TRUTH = {"C": 0.5, "theta": -20.0, "chi0": 2e-4}


@pytest.mark.parametrize("scale", [1e2, 1.0, 1e-2, 1e-4, 1e-6])
def test_yerr_none_recovers_curie_weiss_at_any_scale(scale):
    # Noise-free, chi0 free. Before the fix every scale from 1 down gave
    # C = 0.575, theta = -29.7, chi0 = -2.6e-5, all with success=True.
    y = scale * curie_weiss(_CW_T, **_CW_TRUTH)
    res = fit("curie_weiss", _CW_T, y)
    assert res.success
    assert res.yerr_assumed is True
    assert res.parameters["C"] == pytest.approx(scale * _CW_TRUTH["C"], rel=1e-6)
    assert res.parameters["theta"] == pytest.approx(_CW_TRUTH["theta"], rel=1e-6)
    assert res.parameters["chi0"] == pytest.approx(scale * _CW_TRUTH["chi0"], rel=1e-6)


@pytest.mark.parametrize("k", [1e3, 1e-3, 1e-5, 1e-7])
def test_yerr_none_curie_weiss_is_scale_equivariant(k):
    # Multiplying the data by k must multiply C, chi0 and their errors by k
    # and leave theta alone: the physics cannot depend on the units of y.
    # Noisy data, so the minimum is not an exact fit.
    rng = np.random.default_rng(31)
    y = curie_weiss(_CW_T, **_CW_TRUTH)
    y = y + rng.normal(scale=0.01 * y.mean(), size=y.shape)
    ref = fit("curie_weiss", _CW_T, y)
    got = fit("curie_weiss", _CW_T, k * y)
    assert ref.success and got.success
    for name, power in (("C", 1), ("theta", 0), ("chi0", 1)):
        factor = k**power
        # Both runs sit within MIGRAD's goal (~0.02 sigma) of the same scaled
        # minimum; 0.05 sigma between them is a real equivariance check.
        tol = 5e-2 * ref.uncertainties[name] * factor
        assert got.parameters[name] == pytest.approx(ref.parameters[name] * factor, abs=tol)
        assert got.uncertainties[name] == pytest.approx(
            ref.uncertainties[name] * factor, rel=1e-2
        )
    assert got.chi_squared == pytest.approx(ref.chi_squared * k**2, rel=1e-6)
    assert got.reduced_chi_squared == pytest.approx(ref.reduced_chi_squared * k**2, rel=1e-6)


@pytest.mark.parametrize("k", [1e2, 1e-2, 1e-4])
def test_yerr_none_brillouin_is_scale_equivariant(k):
    # A paramagnetic M(H) with M_s ~ 3e-4 emu -- a realistic magnitude, and
    # small enough to stall a unit-error fit -- fitting M_s and T_K together.
    rng = np.random.default_rng(37)
    H = np.linspace(-5.0e4, 5.0e4, 81)
    y = brillouin(H, M_s=3.3e-4, g=2.0, J=3.5, T_K=5.0)
    y = y + rng.normal(scale=1e-6, size=H.shape)
    kwargs = {"fixed": {"g": 2.0, "J": 3.5}}
    ref = fit("brillouin", H, y, initial={"M_s": 1e-4, "T_K": 4.0}, **kwargs)
    got = fit("brillouin", H, k * y, initial={"M_s": 1e-4 * k, "T_K": 4.0}, **kwargs)
    assert ref.success and got.success
    # The reference itself is right: truth within 3 sigma.
    assert abs(ref.parameters["M_s"] - 3.3e-4) < 3 * ref.uncertainties["M_s"]
    assert abs(ref.parameters["T_K"] - 5.0) < 3 * ref.uncertainties["T_K"]
    for name, power in (("M_s", 1), ("T_K", 0)):
        factor = k**power
        tol = 5e-2 * ref.uncertainties[name] * factor
        assert got.parameters[name] == pytest.approx(ref.parameters[name] * factor, abs=tol)
        assert got.uncertainties[name] == pytest.approx(
            ref.uncertainties[name] * factor, rel=1e-2
        )


def test_yerr_none_chi_squared_is_raw_sum_of_squares():
    # With assumed errors chi_squared is documented as the raw sum of squared
    # residuals, whatever sigma the fit was minimised against internally.
    # Checked on susceptibility-sized data (~1e-5) with noise.
    rng = np.random.default_rng(41)
    truth = {"C": 5e-3, "theta": -20.0, "chi0": 2e-6}
    y = curie_weiss(_CW_T, **truth) + rng.normal(scale=1e-8, size=_CW_T.shape)
    res = fit("curie_weiss", _CW_T, y)
    assert res.success
    p = res.parameters
    independent = float(np.sum((y - curie_weiss(_CW_T, p["C"], p["theta"], p["chi0"])) ** 2))
    assert res.chi_squared == pytest.approx(independent, rel=1e-9)
    assert res.reduced_chi_squared == pytest.approx(independent / res.dof, rel=1e-9)
    # And it is a minimum: no higher than the sum of squares at the
    # generating parameters. A fit that stops short fails this.
    at_truth = float(np.sum((y - curie_weiss(_CW_T, **truth)) ** 2))
    assert res.chi_squared <= at_truth


def test_yerr_none_fixed_parameter_small_scale():
    # A fixed parameter does not consume a degree of freedom, and the refined
    # sigma must use the same dof: the reported errors then match the
    # supplied-error fit's times sqrt(chi2/dof), as for a fully free fit.
    rng = np.random.default_rng(43)
    noise = 1e-8
    y = curie_weiss(_CW_T, 5e-3, -20.0, 0.0) + rng.normal(scale=noise, size=_CW_T.shape)
    assumed = fit("curie_weiss", _CW_T, y, fixed={"chi0": 0.0})
    with_err = fit("curie_weiss", _CW_T, y, yerr=noise, fixed={"chi0": 0.0})
    assert assumed.success and with_err.success
    assert assumed.dof == _CW_T.size - 2
    assert assumed.parameters["chi0"] == 0.0
    assert assumed.uncertainties["chi0"] == 0.0
    for name in ("C", "theta"):
        assert assumed.parameters[name] == pytest.approx(
            with_err.parameters[name], abs=1e-2 * with_err.uncertainties[name]
        )
        expected = with_err.uncertainties[name] * np.sqrt(with_err.reduced_chi_squared)
        assert assumed.uncertainties[name] == pytest.approx(expected, rel=1e-2)


def test_yerr_none_exact_fit_succeeds():
    # Residuals at floating-point round-off leave no scatter to estimate sigma
    # from. Refining against it would make the cost pure round-off noise and
    # MIGRAD would call a perfect fit invalid; the first pass must stand.
    x = np.linspace(0.0, 10.0, 25)
    res = fit("linear", x, 3e-6 * x - 2e-7)
    assert res.success
    assert res.parameters["slope"] == pytest.approx(3e-6, rel=1e-9)
    assert res.parameters["intercept"] == pytest.approx(-2e-7, rel=1e-9)
    assert res.chi_squared == pytest.approx(0.0, abs=1e-30)


def test_yerr_none_constant_data_fits():
    # std(y) = 0 cannot seed sigma; the fit falls back to 1 and still works.
    x = np.linspace(0.0, 10.0, 25)
    res = fit("linear", x, np.full_like(x, 4.0))
    assert res.success
    assert res.parameters["slope"] == pytest.approx(0.0, abs=1e-9)
    assert res.parameters["intercept"] == pytest.approx(4.0, rel=1e-9)


def test_fixed_parameter_honoured_and_excluded_from_dof():
    rng = np.random.default_rng(3)
    T, y, noise = _synthetic_cw(rng, chi0=0.0)
    free = fit("curie_weiss", T, y, yerr=np.full_like(T, noise))
    held = fit("curie_weiss", T, y, yerr=np.full_like(T, noise), fixed={"chi0": 0.0})
    assert held.parameters["chi0"] == 0.0
    assert held.fixed == {"chi0": 0.0}
    assert "chi0" not in held.covariance_parameters
    # One fewer free parameter -> one more degree of freedom.
    assert held.dof == free.dof + 1
    assert held.uncertainties["chi0"] == 0.0


def test_limits_respected():
    rng = np.random.default_rng(9)
    T, y, noise = _synthetic_cw(rng, C=2.5)
    # Cap C below its true value; the fit must not exceed the bound.
    res = fit("curie_weiss", T, y, yerr=np.full_like(T, noise), limits={"C": (0.0, 2.0)})
    assert res.parameters["C"] <= 2.0 + 1e-9


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_observations_raise(bad):
    from calamari import FitInputError

    with pytest.raises(FitInputError, match="y must be finite"):
        fit("linear", [1, 2, 3, 4], [0, bad, 1, 2])


@pytest.mark.parametrize("n_points", [0, 1, 2, 3])
@pytest.mark.parametrize("yerr", [None, 1e-6])
def test_underdetermined_fit_is_a_failure_with_honest_dof(n_points, yerr):
    # With no more points than free parameters the points fix the parameters
    # exactly (or not at all): there is no fit to judge and no scatter for
    # HESSE. fit() used to clamp dof at 1, interpolate, and report
    # success=True with uncertainties. FT-1: a failure, returned not raised.
    T = np.linspace(50.0, 300.0, n_points)
    chi = curie_weiss(T, **_CW_TRUTH)
    seeds = {"C": 1.0, "theta": 0.0, "chi0": 0.0}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # refused quietly, even when empty
        res = fit(
            "curie_weiss", T, chi,
            yerr=None if yerr is None else np.full_like(T, yerr),
            initial=seeds,
        )
    assert res.success is False
    assert res.dof == n_points - 3  # not clamped: 0 or negative
    assert res.n_points == n_points
    assert res.uncertainties == {}
    assert res.covariance is None
    assert res.minos_errors is None
    assert np.isnan(res.chi_squared) and np.isnan(res.reduced_chi_squared)
    assert res.parameters == seeds  # not minimised: the starting values
    assert res.yerr_assumed is (yerr is None)
    assert "FAILED" in res.summary()


def test_underdetermined_counts_only_free_parameters():
    # Fixing chi0 leaves two free parameters: two points are still too few,
    # three are enough for a one-dof fit.
    T = np.array([100.0, 200.0, 300.0])
    chi = curie_weiss(T, **_CW_TRUTH)
    held = {"chi0": _CW_TRUTH["chi0"]}
    two = fit("curie_weiss", T[:2], chi[:2], fixed=held)
    assert two.success is False
    assert two.dof == 0
    assert two.uncertainties == {}
    assert two.parameters["chi0"] == _CW_TRUTH["chi0"]
    three = fit("curie_weiss", T, chi, fixed=held)
    assert three.success is True
    assert three.dof == 1
    assert three.parameters["theta"] == pytest.approx(_CW_TRUTH["theta"], rel=1e-3)


def test_one_degree_of_freedom_still_fits():
    # The boundary: n_free + 1 points is the smallest fittable problem.
    T = np.linspace(50.0, 300.0, 4)
    res = fit("curie_weiss", T, curie_weiss(T, **_CW_TRUTH))
    assert res.success is True
    assert res.dof == 1
    assert set(res.uncertainties) == {"C", "theta", "chi0"}


def test_unknown_names_raise_even_when_underdetermined():
    # The early refusal must not swallow a caller error that fit() raises for
    # well-determined data: the same call raises whatever the point count.
    T = np.array([100.0])
    chi = curie_weiss(T, **_CW_TRUTH)
    with pytest.raises(KeyError, match="chi_0"):
        fit("curie_weiss", T, chi, fixed={"chi_0": 0.0})
    with pytest.raises(KeyError, match="thta"):
        fit("curie_weiss", T, chi, initial={"thta": 0.0})
    with pytest.raises(KeyError, match="thetaa"):
        fit("curie_weiss", T, chi, limits={"thetaa": (-100.0, 0.0)})


def test_unknown_limits_name_raises_rather_than_failing_the_fit():
    # limits used to be checked where they were applied, inside the FT-1
    # guard, so a misspelt name came back as success=False on data that fits
    # perfectly with the name spelt right: a caller error disguised as a fit
    # the data could not support.
    T = np.linspace(50.0, 300.0, 20)
    chi = curie_weiss(T, **_CW_TRUTH)
    assert fit("curie_weiss", T, chi, limits={"theta": (-100.0, 0.0)}).success is True
    with pytest.raises(KeyError, match=r"limits contains unknown parameter\(s\) \['thetaa'\]"):
        fit("curie_weiss", T, chi, limits={"thetaa": (-100.0, 0.0)})


def test_model_that_raises_returns_success_false_not_raise():
    # The other FT-1 route: an exception inside the minimisation.
    def explode(x, a):
        raise FloatingPointError("model cannot be evaluated")

    defn = ModelDefinition(
        name="test_explode_tmp",
        function=explode,
        param_names=("a",),
        param_defaults={"a": 1.0},
        description="test-only model that always raises",
    )
    res = fit(defn, np.linspace(1.0, 5.0, 10), np.linspace(1.0, 5.0, 10))
    assert res.success is False
    assert res.uncertainties == {}


def test_run_minos_populates_display_only_band():
    rng = np.random.default_rng(11)
    T, y, noise = _synthetic_cw(rng)
    res = fit("curie_weiss", T, y, yerr=np.full_like(T, noise), run_minos=True)
    assert res.success
    assert res.minos_errors is not None
    assert "C" in res.minos_errors
    lo, hi = res.minos_errors["C"]
    assert lo < 0 < hi  # bracketing offsets


def test_summary_contains_every_free_parameter():
    rng = np.random.default_rng(5)
    T, y, noise = _synthetic_cw(rng)
    res = fit("curie_weiss", T, y, yerr=np.full_like(T, noise))
    text = res.summary()
    for name in ("C", "theta", "chi0"):
        assert name in text


def test_curve_evaluates_best_fit():
    rng = np.random.default_rng(13)
    T, y, noise = _synthetic_cw(rng)
    res = fit("curie_weiss", T, y, yerr=np.full_like(T, noise))
    manual = curie_weiss(
        T, res.parameters["C"], res.parameters["theta"], res.parameters["chi0"]
    )
    assert np.allclose(res.curve(T), manual)


def test_brillouin_fit_recovers_Ms_with_fixed_g_J_T():
    # Fit only M_s (g, J, T fixed by the caller, as intended).
    rng = np.random.default_rng(21)
    H = np.linspace(-6.0e4, 6.0e4, 80)
    truth = brillouin(H, M_s=3.3, g=2.0, J=3.5, T_K=2.0)
    y = truth + rng.normal(scale=0.01, size=H.shape)
    res = fit(
        "brillouin", H, y,
        yerr=np.full_like(H, 0.01),
        fixed={"g": 2.0, "J": 3.5, "T_K": 2.0},
        initial={"M_s": 1.0},
    )
    assert res.success
    assert res.parameters["M_s"] == pytest.approx(3.3, rel=0.02)
    assert res.covariance_parameters == ("M_s",)


# --------------------------------------------------------------------------
# fit_windows: one model, many fitting ranges
# --------------------------------------------------------------------------

_FW_T = np.linspace(10.0, 300.0, 59)  # 5 K steps
_FW_CW = {"C": 0.5, "theta": -20.0, "chi0": 2e-4}
_FW_WINDOWS = [(20.0, 300.0), (50.0, 300.0), (100.0, 300.0), (20.0, 200.0)]


def _fw_data(model):
    """Noise-free data for *model* with the planted truth, and that truth."""
    if model == "curie_weiss":
        return curie_weiss(_FW_T, **_FW_CW), dict(_FW_CW)
    truth = {"C": 0.5, "theta": -20.0}
    return inverse_curie_weiss(_FW_T, **truth), truth


@pytest.mark.parametrize("model", ["curie_weiss", "inverse_curie_weiss"])
def test_fit_windows_recovers_planted_truth_in_every_window(model):
    y, truth = _fw_data(model)
    scan = fit_windows(model, _FW_T, y, windows=_FW_WINDOWS)
    assert list(scan.table.columns[:7]) == [
        "x_min", "x_max", "n_points", "success", "dof", "reduced_chi_squared", "yerr_assumed",
    ]
    assert list(scan.table.columns[7:]) == [c for p in truth for c in (p, f"{p}_err")]
    assert len(scan.table) == len(_FW_WINDOWS)
    assert scan.table["success"].all()
    assert scan.table["yerr_assumed"].all()
    for _, row in scan.table.iterrows():
        for name, value in truth.items():
            assert row[name] == pytest.approx(value, rel=1e-4), (row["x_min"], name)


def test_fit_windows_rows_align_with_results():
    y, _ = _fw_data("curie_weiss")
    windows = [(20.0, 300.0), (10.0, 15.0), (100.0, 300.0)]  # a failure in the middle
    scan = fit_windows("curie_weiss", _FW_T, y, windows=windows)
    assert len(scan.results) == len(scan.table)
    for i, result in enumerate(scan.results):
        assert isinstance(result, FitResult)
        row = scan.table.iloc[i]
        assert row["success"] == result.success
        assert row["n_points"] == result.n_points
        if result.success:
            assert row["dof"] == result.dof
            assert row["reduced_chi_squared"] == result.reduced_chi_squared
            for name in ("C", "theta", "chi0"):
                assert row[name] == result.parameters[name]
                assert row[f"{name}_err"] == result.uncertainties[name]
    assert list(scan.table["success"]) == [True, False, True]


def test_fit_windows_underdetermined_window_is_a_failure_row():
    # A window with no more points than free parameters is a failure row, not
    # an interpolation relayed as a fit. fit() refuses it itself; the scan
    # reports fit()'s own result and its unclamped dof.
    y, _ = _fw_data("curie_weiss")

    windows = [(400.0, 500.0), (10.0, 10.0 + 1e-9), (10.0, 15.0), (10.0, 20.0), (10.0, 25.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an empty window must not warn either
        scan = fit_windows("curie_weiss", _FW_T, y, windows=windows)
    table = scan.table
    assert list(table["n_points"]) == [0, 1, 2, 3, 4]
    assert list(table["dof"]) == [-3, -2, -1, 0, 1]
    assert list(table["success"]) == [False, False, False, False, True]
    failed = table[~table["success"]]
    param_columns = ["C", "C_err", "theta", "theta_err", "chi0", "chi0_err"]
    assert failed[param_columns + ["reduced_chi_squared"]].isna().all().all()
    for i, result in enumerate(scan.results[:4]):
        assert result.success is False
        assert result.uncertainties == {}
        assert result.dof == table.loc[i, "dof"]

    # Fixing a parameter frees a degree of freedom: 3 points now determine 2.
    held = fit_windows(
        "curie_weiss", _FW_T, y, windows=[(10.0, 15.0), (10.0, 20.0)],
        fixed={"chi0": _FW_CW["chi0"]},
    )
    assert list(held.table["success"]) == [False, True]
    assert list(held.table["dof"]) == [0, 1]
    assert held.results[0].parameters["chi0"] == _FW_CW["chi0"]


def test_fit_windows_invalid_input_propagates():
    from calamari import FitInputError

    y, _ = _fw_data("curie_weiss")
    y = y.copy()
    y[_FW_T == 150.0] = np.nan
    with pytest.raises(FitInputError):
        fit_windows("curie_weiss", _FW_T, y, windows=[(20.0, 100.0), (100.0, 200.0)])


def test_fit_windows_numerical_failure_is_a_row():
    def limited_domain(x, a):
        if np.any(x > 100):
            raise FloatingPointError("model evaluation failed")
        return a*x

    definition = ModelDefinition(name="limited", function=limited_domain,
                                 param_names=("a",), param_defaults={"a": 1.0},
                                 description="synthetic model evaluation failure")
    scan = fit_windows(definition, _FW_T, 2*_FW_T, windows=[(20., 100.), (100., 200.)])
    assert list(scan.table["success"]) == [True, False]
    assert np.isnan(scan.table.loc[1, "a"])
    assert scan.results[1].success is False



def test_fit_windows_fixed_parameter_reports_zero_error():
    y, _ = _fw_data("curie_weiss")
    scan = fit_windows(
        "curie_weiss", _FW_T, y, windows=_FW_WINDOWS, fixed={"chi0": _FW_CW["chi0"]}
    )
    assert scan.table["success"].all()
    assert (scan.table["chi0"] == _FW_CW["chi0"]).all()
    assert (scan.table["chi0_err"] == 0.0).all()
    assert (scan.table["theta_err"] > 0.0).all()
    # One parameter fewer to estimate: one more degree of freedom per window.
    free = fit_windows("curie_weiss", _FW_T, y, windows=_FW_WINDOWS)
    assert list(scan.table["dof"]) == list(free.table["dof"] + 1)


def test_fit_windows_preserves_window_order():
    rng = np.random.default_rng(31)
    y, _ = _fw_data("curie_weiss")
    y = y * (1 + 0.01 * rng.standard_normal(y.size))
    windows = [(100.0, 300.0), (20.0, 300.0), (50.0, 150.0), (20.0, 300.0)]
    scan = fit_windows("curie_weiss", _FW_T, y, windows=windows)
    assert list(zip(scan.table["x_min"], scan.table["x_max"])) == windows
    for (lo, hi), result in zip(windows, scan.results):
        inside = (_FW_T >= lo) & (_FW_T <= hi)  # inclusive at both ends
        direct = fit("curie_weiss", _FW_T[inside], y[inside])
        assert result.n_points == np.count_nonzero(inside)
        assert result.parameters == direct.parameters


def test_fit_windows_slices_array_yerr_with_the_window():
    # Errors that vary strongly along x: pairing a window's points with the
    # wrong errors changes both the weights and the chi-squared.
    rng = np.random.default_rng(37)
    y, _ = _fw_data("curie_weiss")
    yerr = 1e-5 * (1.0 + _FW_T / 10.0)
    y = y + yerr * rng.standard_normal(y.size)
    windows = [(150.0, 300.0), (20.0, 120.0)]
    scan = fit_windows("curie_weiss", _FW_T, y, windows=windows, yerr=yerr)
    assert not scan.table["yerr_assumed"].any()
    for (lo, hi), result in zip(windows, scan.results):
        inside = (_FW_T >= lo) & (_FW_T <= hi)
        direct = fit("curie_weiss", _FW_T[inside], y[inside], yerr=yerr[inside])
        assert result.parameters == direct.parameters
        assert result.chi_squared == direct.chi_squared
        assert result.uncertainties == direct.uncertainties

    # A scalar broadcasts to every point, exactly like a constant array.
    scalar = fit_windows("curie_weiss", _FW_T, y, windows=windows, yerr=2e-5)
    constant = fit_windows(
        "curie_weiss", _FW_T, y, windows=windows, yerr=np.full_like(_FW_T, 2e-5)
    )
    pd.testing.assert_frame_equal(scalar.table, constant.table)

    with pytest.raises(ValueError, match="yerr must be a scalar or the same length"):
        fit_windows("curie_weiss", _FW_T, y, windows=windows, yerr=yerr[:-1])


@pytest.mark.parametrize(
    "windows",
    [
        [(300.0, 50.0)],               # reversed
        [(50.0, 50.0)],                # zero width
        [(np.nan, 300.0)],             # non-finite
        [(50.0, np.inf)],
        (50.0, 300.0),                 # one window without the outer brackets
        [(50.0,)],                     # not a pair
        [(50.0, 100.0, 300.0)],
        [("50", 300.0)],               # not numbers
        [(True, 300.0)],               # a bool is not a bound
        [],                            # nothing to scan
        "50-300",
        None,
    ],
)
def test_fit_windows_refuses_malformed_windows(windows):
    y, _ = _fw_data("curie_weiss")
    with pytest.raises(ValueError, match="window"):
        fit_windows("curie_weiss", _FW_T, y, windows=windows)


def test_fit_windows_accepts_an_array_of_windows():
    y, _ = _fw_data("curie_weiss")
    scan = fit_windows("curie_weiss", _FW_T, y, windows=np.array(_FW_WINDOWS))
    assert scan.params["windows"] == _FW_WINDOWS


def test_fit_windows_refuses_caller_errors_before_fitting():
    y, _ = _fw_data("curie_weiss")
    with pytest.raises(ValueError, match="same length"):
        fit_windows("curie_weiss", _FW_T, y[:-1], windows=_FW_WINDOWS)
    with pytest.raises(TypeError, match="fixd"):
        fit_windows("curie_weiss", _FW_T, y, windows=_FW_WINDOWS, fixd={"chi0": 0.0})
    with pytest.raises(KeyError, match="no_such_model"):
        fit_windows("no_such_model", _FW_T, y, windows=_FW_WINDOWS)
    # Refused even when every window is too sparse to fit, so the same call
    # cannot raise or not depending on the data.
    with pytest.raises(KeyError, match="chi_0"):
        fit_windows("curie_weiss", _FW_T, y, windows=[(400.0, 500.0)], fixed={"chi_0": 0.0})


def test_fit_windows_records_params():
    y, _ = _fw_data("curie_weiss")
    initial = {"C": 0.4, "theta": -10.0}
    limits = {"C": (0.0, None)}
    scan = fit_windows(
        MODELS["curie_weiss"], _FW_T, y, windows=[(20, 300), (100.0, 300.0)],
        initial=initial, limits=limits, run_minos=True,
    )
    assert scan.params["model"] == "curie_weiss"
    assert scan.params["windows"] == [(20.0, 300.0), (100.0, 300.0)]
    assert all(isinstance(v, float) for w in scan.params["windows"] for v in w)
    assert scan.params["fit_kwargs"] == {
        "initial": initial, "limits": limits, "run_minos": True,
    }
    assert isinstance(scan.params["calamari_version"], str)
    # Forwarded, not just recorded: MINOS ran in every window.
    assert all(r.minos_errors for r in scan.results)
    # A copy: editing the caller's dict afterwards does not rewrite history.
    initial["C"] = 99.0
    assert scan.params["fit_kwargs"]["initial"]["C"] == 0.4


def test_fit_windows_result_is_frozen_and_exported():
    import calamari
    from calamari.fitting import FitWindowsResult

    assert calamari.fit_windows is fit_windows
    assert calamari.FitWindowsResult is FitWindowsResult
    y, _ = _fw_data("curie_weiss")
    scan = fit_windows("curie_weiss", _FW_T, y, windows=_FW_WINDOWS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        scan.table = None


# --------------------------------------------------------------------------
# FitResult.derive: a quantity built from the parameters, with its error
# --------------------------------------------------------------------------


def _mu_eff(p):
    """mu_eff from a per-ion molar Curie constant, as a caller would write it."""
    return mu_eff_from_curie_constant(p["C"])


def _chi_at_50K(p):
    """chi at 50 K: non-linear in, and correlated across, all three parameters."""
    return p["C"] / (50.0 - p["theta"]) + p["chi0"]


def _noisy_cw_fit(seed=5, noise=0.01, **fit_kwargs):
    """A curie_weiss fit to 1 %-noisy data, so every error is non-trivial."""
    rng = np.random.default_rng(seed)
    y = curie_weiss(_FW_T, **_FW_CW) * (1 + noise * rng.standard_normal(_FW_T.size))
    result = fit("curie_weiss", _FW_T, y, **fit_kwargs)
    assert result.success and result.covariance is not None
    return result


def test_derive_linear_function_is_the_exact_propagation():
    # For a linear f the delta method is exact: sigma_f^2 = J C J^T with
    # J = df/dp, including the strong C-theta anticorrelation. The weight on
    # theta makes the two terms comparable, so adding the errors in
    # quadrature instead would be wrong by a factor of about three.
    result = _noisy_cw_fit()
    value, err = result.derive(lambda p: p["C"] + 0.01 * p["theta"])
    assert result.covariance_parameters == ("C", "theta", "chi0")
    jacobian = np.array([1.0, 0.01, 0.0])
    expected = np.sqrt(jacobian @ result.covariance @ jacobian)
    assert value == pytest.approx(result.parameters["C"] + 0.01 * result.parameters["theta"])
    assert err == pytest.approx(expected, rel=1e-9)
    uncorrelated = np.hypot(result.uncertainties["C"], 0.01 * result.uncertainties["theta"])
    assert err < 0.5 * uncorrelated  # the correlation matters here


def test_derive_mu_eff_matches_the_analytic_error():
    # mu = k sqrt(C), so sigma_mu = mu sigma_C / (2 C) exactly to first order.
    result = _noisy_cw_fit()
    mu, mu_err = result.derive(_mu_eff)
    C, sigma_C = result.parameters["C"], result.uncertainties["C"]
    assert mu == mu_eff_from_curie_constant(C)
    assert mu_err == pytest.approx(mu * sigma_C / (2.0 * C), rel=1e-7)


def test_derive_agrees_with_monte_carlo_over_the_covariance():
    # An independent check of the whole machinery on a non-linear function of
    # all three correlated parameters. The relative errors are ~1 %, where
    # first order holds, so the Monte Carlo spread must agree within its own
    # sampling error (~0.5 % for 20000 draws).
    result = _noisy_cw_fit()
    value, err = result.derive(_chi_at_50K)
    rng = np.random.default_rng(11)
    names = result.covariance_parameters
    centre = np.array([result.parameters[n] for n in names])
    draws = rng.multivariate_normal(centre, result.covariance, size=20000)
    spread = np.std(_chi_at_50K(dict(zip(names, draws.T))), ddof=1)
    assert value == pytest.approx(_chi_at_50K(result.parameters))
    assert err == pytest.approx(spread, rel=0.03)


def test_derive_fixed_parameter_contributes_no_variance():
    result = _noisy_cw_fit(fixed={"chi0": _FW_CW["chi0"]})
    assert result.covariance_parameters == ("C", "theta")
    # chi0 is available to func, exactly as held, and adds nothing to the error.
    value, err = result.derive(lambda p: p["C"] + 1e3 * p["chi0"])
    assert value == pytest.approx(result.parameters["C"] + 1e3 * _FW_CW["chi0"])
    assert err == pytest.approx(result.uncertainties["C"], rel=1e-9)
    assert result.derive(lambda p: p["chi0"]) == (_FW_CW["chi0"], 0.0)


def test_derive_zero_sigma_free_parameter_is_not_stepped():
    # A free parameter HESSE pins exactly carries no variance whatever the
    # slope; it must not be differentiated with a zero step (0/0).
    result = _noisy_cw_fit()
    covariance = result.covariance.copy()
    covariance[2, :] = 0.0
    covariance[:, 2] = 0.0
    pinned = dataclasses.replace(result, covariance=covariance)
    probed = []

    def spy(p):
        probed.append(p["chi0"])
        return p["C"] + p["chi0"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value, err = pinned.derive(spy)
    assert set(probed) == {result.parameters["chi0"]}
    assert err == pytest.approx(result.uncertainties["C"], rel=1e-9)

    # Likewise a sigma so far below the value that the step rounds away:
    # there is no step to divide by, and nothing representable to carry.
    covariance[2, 2] = (1e-20 * result.parameters["chi0"]) ** 2
    lost = dataclasses.replace(result, covariance=covariance)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value, err = lost.derive(spy)
    assert err == pytest.approx(result.uncertainties["C"], rel=1e-9)


def test_derive_without_covariance_reports_value_and_nan_error():
    result = dataclasses.replace(_noisy_cw_fit(), covariance=None, covariance_parameters=())
    value, err = result.derive(_mu_eff)
    assert value == mu_eff_from_curie_constant(result.parameters["C"])
    assert np.isnan(err)


def test_derive_on_a_failed_fit_is_nan_without_calling_func():
    failed = fit("curie_weiss", _FW_T[:3], curie_weiss(_FW_T[:3], **_FW_CW))
    assert failed.success is False

    def must_not_run(p):
        raise AssertionError("func called on a failed fit")

    value, err = failed.derive(must_not_run)
    assert np.isnan(value) and np.isnan(err)


def test_derive_non_finite_value_is_nan_and_numpy_says_why():
    result = _noisy_cw_fit()
    # The caller's own function is invalid at the best fit: NaN for both, and
    # numpy's warning is let through because it describes the caller's call.
    with pytest.warns(RuntimeWarning, match="invalid value"):
        value, err = result.derive(lambda p: np.sqrt(-p["C"]))
    assert np.isnan(value) and np.isnan(err)


def test_derive_non_finite_at_a_probe_point_is_a_nan_error_silently():
    # Finite at the best fit, undefined a hair below it: no slope to
    # propagate. The probe points are derive()'s choice, so their warnings
    # are not passed on.
    result = _noisy_cw_fit()
    C_best = result.parameters["C"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value, err = result.derive(lambda p: np.sqrt(p["C"] - C_best))
    assert value == 0.0
    assert np.isnan(err)


def test_derive_func_errors_propagate_unchanged():
    result = _noisy_cw_fit()
    with pytest.raises(KeyError, match="chi_0"):
        result.derive(lambda p: p["chi_0"])
    with pytest.raises(TypeError, match="single number"):
        result.derive(lambda p: np.array([p["C"], p["theta"]]))


def test_derive_hands_func_a_copy_of_every_parameter():
    result = _noisy_cw_fit(fixed={"chi0": _FW_CW["chi0"]})
    before = dict(result.parameters)

    def vandal(p):
        assert set(p) == {"C", "theta", "chi0"}
        p["C"] = 0.0
        return 1.0

    assert result.derive(vandal)[0] == 1.0
    assert result.parameters == before


# --------------------------------------------------------------------------
# fit_windows(derived=...)
# --------------------------------------------------------------------------


def test_fit_windows_tabulates_derived_quantities_after_the_parameters():
    rng = np.random.default_rng(7)
    y = curie_weiss(_FW_T, **_FW_CW) * (1 + 0.01 * rng.standard_normal(_FW_T.size))
    windows = [(20.0, 300.0), (10.0, 15.0), (100.0, 300.0)]  # a failure in the middle
    scan = fit_windows(
        "curie_weiss", _FW_T, y, windows=windows,
        derived={"mu_eff": _mu_eff, "chi_50K": _chi_at_50K},
    )
    assert list(scan.table.columns[7:]) == [
        "C", "C_err", "theta", "theta_err", "chi0", "chi0_err",
        "mu_eff", "mu_eff_err", "chi_50K", "chi_50K_err",
    ]
    assert list(scan.table["success"]) == [True, False, True]
    derived_columns = ["mu_eff", "mu_eff_err", "chi_50K", "chi_50K_err"]
    for i, result in enumerate(scan.results):
        row = scan.table.iloc[i]
        if result.success:
            assert (row["mu_eff"], row["mu_eff_err"]) == result.derive(_mu_eff)
            assert (row["chi_50K"], row["chi_50K_err"]) == result.derive(_chi_at_50K)
            assert row["mu_eff_err"] > 0.0
        else:
            assert row[derived_columns].isna().all()
    # Recorded as descriptions, not the callables themselves.
    assert scan.params["derived"] == {
        "mu_eff": f"{__name__}._mu_eff",
        "chi_50K": f"{__name__}._chi_at_50K",
    }
    plain = fit_windows("curie_weiss", _FW_T, y, windows=windows)
    assert plain.params["derived"] == {}
    pd.testing.assert_frame_equal(plain.table, scan.table[plain.table.columns])


def test_fit_windows_derived_error_names_the_quantity_and_window():
    y, _ = _fw_data("curie_weiss")

    def fussy(p):
        raise ZeroDivisionError("planted")

    windows = [(10.0, 15.0), (100.0, 300.0)]  # window 0 fails, so func is not called there
    with pytest.raises(ValueError, match=r"'boom'.*fussy.*window 1 \(100, 300\)") as info:
        fit_windows("curie_weiss", _FW_T, y, windows=windows, derived={"boom": fussy})
    assert isinstance(info.value.__cause__, ZeroDivisionError)


@pytest.mark.parametrize(
    "derived, error, match",
    [
        ([("mu_eff", _mu_eff)], TypeError, "mapping"),
        (_mu_eff, TypeError, "mapping"),
        ({"mu_eff": 2.83}, TypeError, "not callable"),
        ({1: _mu_eff}, TypeError, "non-empty strings"),
        ({"": _mu_eff}, TypeError, "non-empty strings"),
        ({"C": _mu_eff}, ValueError, r"\['C', 'C_err'\]"),
        ({"theta_err": _mu_eff}, ValueError, "ends in '_err'"),
        ({"mu_err": _mu_eff}, ValueError, "ends in '_err'"),
        ({"success": _mu_eff}, ValueError, r"\['success'\], which"),
        ({"x_max": _mu_eff}, ValueError, r"\['x_max'\], which"),
    ],
)
def test_fit_windows_refuses_bad_derived_before_fitting(monkeypatch, derived, error, match):
    import calamari.fitting.windows as windows_module

    def no_fitting(*args, **kwargs):
        raise AssertionError("fit() called before derived= was validated")

    monkeypatch.setattr(windows_module, "fit", no_fitting)
    y, _ = _fw_data("curie_weiss")
    with pytest.raises(error, match=match):
        fit_windows("curie_weiss", _FW_T, y, windows=_FW_WINDOWS, derived=derived)


def test_fit_windows_derived_is_not_a_fit_keyword():
    # derived= is fit_windows' own: not forwarded, not recorded as a fit
    # keyword, and unknown keywords are still refused beside it.
    from calamari.fitting.windows import _FORWARDED_FIT_KWARGS

    assert "derived" not in _FORWARDED_FIT_KWARGS
    y, _ = _fw_data("curie_weiss")
    scan = fit_windows(
        "curie_weiss", _FW_T, y, windows=_FW_WINDOWS, derived={"mu_eff": _mu_eff}
    )
    assert scan.params["fit_kwargs"] == {}
    with pytest.raises(TypeError, match="fixd"):
        fit_windows(
            "curie_weiss", _FW_T, y, windows=_FW_WINDOWS,
            derived={"mu_eff": _mu_eff}, fixd={"chi0": 0.0},
        )


@pytest.mark.parametrize("yerr", [None, 0.2])
@pytest.mark.parametrize("n_points", [0, 1, 10])
@pytest.mark.parametrize("value", [-1.0, 5.0])
def test_fixed_outside_limits_is_a_caller_error(monkeypatch, yerr, n_points, value):
    import iminuit

    from calamari import CalamariError

    def unexpected_minimiser(*args, **kwargs):
        pytest.fail("contradictory constraints reached Minuit")

    monkeypatch.setattr(iminuit, "Minuit", unexpected_minimiser)
    x = np.arange(float(n_points))
    with pytest.raises(FitConstraintError) as caught:
        fit("linear", x, 2 * x + 1, yerr=yerr,
            fixed={"slope": value}, limits={"slope": (0.0, 1.0)})
    assert isinstance(caught.value, CalamariError)
    assert issubclass(FitConstraintError, ValueError)
    message = str(caught.value)
    for detail in ("linear", "slope", str(value), "(0.0, 1.0)", "correct"):
        assert detail in message


@pytest.mark.parametrize("yerr", [None, 0.2])
@pytest.mark.parametrize("bounds", [
    (0.0, 4.0), (2.0, 4.0), (0.0, 2.0), (2.0, 2.0),
    (None, 2.0), (2.0, None), (None, None), (-np.inf, np.inf),
])
def test_compatible_fixed_limits_preserve_fit_and_metadata(yerr, bounds):
    x = np.arange(10.0)
    y = 2 * x + 1 + 0.1 * np.cos(x)
    reference = fit("linear", x, y, yerr=yerr, fixed={"slope": 2.0})
    result = fit("linear", x, y, yerr=yerr,
                 fixed={"slope": 2.0}, limits={"slope": bounds})
    assert result.success and reference.success
    assert result.parameters["slope"] == result.fixed["slope"] == 2.0
    assert result.uncertainties["slope"] == 0.0
    assert result.dof == 9
    assert result.covariance_parameters == ("intercept",)
    assert result.yerr_assumed == (yerr is None)
    assert result.parameters == pytest.approx(reference.parameters)
    assert result.uncertainties == pytest.approx(reference.uncertainties)
    np.testing.assert_allclose(result.covariance, reference.covariance)


@pytest.mark.parametrize("windows", [[(0.0, 9.0)], [(20.0, 30.0)]])
def test_fit_windows_propagates_fixed_limit_conflict(windows):
    x = np.arange(10.0)
    with pytest.raises(FitConstraintError, match="slope"):
        fit_windows("linear", x, 2 * x + 1, windows=windows,
                    fixed={"slope": 5.0}, limits={"slope": (0.0, 1.0)})


@pytest.mark.parametrize("n_points", [0, 10])
@pytest.mark.parametrize("value,bounds", [
    (2.0, (4.0, 0.0)), (2.0, (np.nan, 4.0)), (2.0, (0.0, np.nan)),
    (np.nan, (None, None)), (np.inf, (None, None)), (-np.inf, (-np.inf, np.inf)),
    (2.0, (np.inf, None)), (2.0, (None, -np.inf)),
])
def test_invalid_fixed_interval_is_a_caller_error(n_points, value, bounds):
    x = np.arange(float(n_points))
    with pytest.raises(FitConstraintError, match="slope"):
        fit("linear", x, 2 * x + 1,
            fixed={"slope": value}, limits={"slope": bounds})


@pytest.mark.parametrize('x,y,yerr', [
    (1, [1], None), ([[1, 2]], [1, 2], None), ([1], [1, 2], None),
    ([np.nan], [1], None), ([1], [np.inf], None),
    ([1], [1], 0), ([1], [1], -1), ([1], [1], np.nan),
    ([1], [1], np.inf), ([1], [1], [1, 2]), ([1], [1], [[1]]),
    ([], [], -1), ([1], ['bad'], None),
])
def test_invalid_fit_inputs_raise_before_sparse_return(x, y, yerr):
    from calamari import FitInputError

    with pytest.raises(FitInputError):
        fit('linear', x, y, yerr=yerr)


def test_scalar_and_vector_uncertainties_are_equivalent():
    x = np.arange(6.)
    y = 2*x + 1 + np.array([0.1, -0.2, 0.1, -0.1, 0.2, -0.1])
    scalar = fit('linear', x, y, yerr=0.2)
    vector = fit('linear', x, y, yerr=np.full(6, 0.2))
    assert scalar.success and vector.success
    assert scalar.parameters == pytest.approx(vector.parameters)
    assert scalar.uncertainties == pytest.approx(vector.uncertainties)


@pytest.mark.parametrize('name', ['x', 'y', 'yerr'])
@pytest.mark.parametrize('value', [np.array([1+2j]*4), [10**1000]*4])
def test_unrepresentable_fit_inputs_raise(name, value):
    from calamari import FitInputError

    arguments = dict(x=np.arange(4.), y=np.arange(4.), yerr=1.)
    arguments[name] = value
    with pytest.raises(FitInputError):
        fit('linear', **arguments)


@pytest.mark.parametrize("operation", [fit, fit_windows])
def test_model_programming_error_propagates(operation):
    error = NameError("invented programming error")
    def broken(x, a):
        raise error
    model = ModelDefinition("invented", broken, ("a",), {"a": 1.}, "invented")
    kwargs = {"windows": [(0., 10.)]} if operation is fit_windows else {}
    with pytest.raises(NameError) as caught:
        operation(model, np.arange(10.), np.arange(10.), **kwargs)
    assert caught.value is error


@pytest.mark.parametrize("error", [FloatingPointError("invented"), NameError("invented")])
def test_minos_failure_preserves_hesse_or_propagates(monkeypatch, error):
    from iminuit import Minuit
    def broken(self, *args, **kwargs):
        raise error
    monkeypatch.setattr(Minuit, "minos", broken)
    x = np.arange(10.)
    if isinstance(error, NameError):
        with pytest.raises(NameError) as caught:
            fit("linear", x, 2*x + 1, yerr=0.2, run_minos=True)
        assert caught.value is error
    else:
        result = fit("linear", x, 2*x + 1, yerr=0.2, run_minos=True)
        assert result.success and result.minos_errors is None
        assert result.uncertainties["slope"] > 0
