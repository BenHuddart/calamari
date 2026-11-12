"""Tests for calamari.boundary: a stated model through one criterion's points, one per sweep.

All data is synthetic, with invented numbers and a planted boundary whose form and
parameters are known, so the tests assert recovery against the truth rather than recorded
reference numbers. Each test or campaign fixture uses its own explicit seeded generator.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import calamari
from calamari import simulate
from calamari.boundary import (
    BAND_POINTS,
    LOW_CHI2_TAIL,
    POINT_COLUMNS,
    SWEEP_COLUMNS,
    BoundaryFit,
    fit_boundary,
)
from calamari.fitting import ModelDefinition
from calamari.io import load
from calamari.transitions import detect_transitions_all

#: Planted boundary: T(H) = T0 sqrt(1 - (H/Hc)^2), invented values in K and T.
T0, HC = 14.0, 2.2

#: Held fields of the synthetic family, in T: ten sweeps short of Hc, so the
#: curvature is strong at the top of the range and the form is identified.
H_FAMILY = np.linspace(0.0, 1.8, 10)


def _dome(h_t, T0_K, Hc_T):
    return T0_K * np.sqrt(np.clip(1.0 - (np.asarray(h_t, dtype=float) / Hc_T) ** 2, 0.0, None))


DOME = ModelDefinition(
    name="dome", function=_dome, param_names=("T0_K", "Hc_T"),
    param_defaults={"T0_K": 15.0, "Hc_T": 3.0},
    description="T0 sqrt(1 - (H/Hc)^2), the planted form",
)


def _points(rng, *, sigma=0.1, err=None, criterion="peak/max", shift=None) -> pd.DataFrame:
    """One point per sweep on the planted dome, scattered by ``sigma``, errors ``err``."""
    t = _dome(H_FAMILY, T0, HC) + rng.normal(scale=sigma, size=H_FAMILY.size)
    if shift is not None:
        sweep, by = shift
        t[sweep] += by
    return pd.DataFrame({
        "sweep_id": np.arange(H_FAMILY.size),
        "H_T": H_FAMILY,
        "T_K": t,
        "T_err_K": sigma if err is None else err,
        "criterion": criterion,
    })


def _quiet(*args, **kwargs) -> BoundaryFit:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fit_boundary(*args, **kwargs)


# ---------------------------------------------------------------------------
# Recovery and calibration against planted truth.
# ---------------------------------------------------------------------------


def test_recovers_a_planted_boundary_within_its_hesse_errors():
    bf = _quiet(_points(np.random.default_rng(101)), DOME, swept_axis="T")
    assert bf.fit.success
    for name, truth in (("T0_K", T0), ("Hc_T", HC)):
        pull = (bf.fit.parameters[name] - truth) / bf.fit.uncertainties[name]
        assert abs(pull) < 3.0, name
    estimates = bf.estimates.set_index("parameter")
    assert estimates.loc["T0_K", "hesse_err"] == pytest.approx(bf.fit.uncertainties["T0_K"])
    assert np.all(np.isfinite(estimates["jackknife_se"]))
    assert (estimates["criterion"] == "peak/max").all()


def test_band_and_parameter_errors_cover_the_truth_at_the_nominal_rate():
    """A small fixed-seed Monte Carlo: 200 families of ten points with honest errors.

    The band is the delta-method propagation of the HESSE covariance, which is exact for
    a form linear in its parameters and first-order otherwise; the dome is mildly
    non-linear over this range, so the band should cover the planted curve at about the
    nominal 68.3 % per point. One draw's coverage fraction is strongly correlated along
    the grid (its standard deviation across draws is about 0.32), so the mean over 200
    draws has a standard error of about 0.023; the tolerance is +/- 3.5 of those. At 2000
    draws (not run here) the mean was 0.683. The parameter pulls are held to the same
    nominal rate, and their spread to 1 within 15 %.
    """
    rng = np.random.default_rng(20260928)
    coverage, pulls, flagged = [], [], 0
    for _ in range(200):
        bf = _quiet(_points(rng), DOME, swept_axis="T")
        assert bf.fit.success
        band = bf.band
        truth = _dome(band["H_T"], T0, HC)
        coverage.append(np.mean(np.abs(band["model_K"] - truth) <= band["model_err_K"]))
        pulls.append([(bf.fit.parameters[n] - v) / bf.fit.uncertainties[n]
                      for n, v in (("T0_K", T0), ("Hc_T", HC))])
        flagged += bf.scatter_below_errors
    pulls = np.asarray(pulls)
    assert 0.60 <= np.mean(coverage) <= 0.76
    assert np.all((np.mean(np.abs(pulls) <= 1.0, axis=0) >= 0.60)
                  & (np.mean(np.abs(pulls) <= 1.0, axis=0) <= 0.76))
    assert np.all(np.abs(pulls.std(axis=0) - 1.0) <= 0.15)
    # Honest errors trip the low-scatter flag at about LOW_CHI2_TAIL (0.135 %).
    assert flagged <= 2


def test_band_spans_the_used_points_and_band_at_extrapolates_on_request():
    bf = _quiet(_points(np.random.default_rng(102)), DOME, swept_axis="T")
    band = bf.band
    assert len(band) == BAND_POINTS
    assert band["H_T"].min() == pytest.approx(H_FAMILY.min())
    assert band["H_T"].max() == pytest.approx(H_FAMILY.max())
    assert list(band.columns) == ["H_T", "model_K", "model_err_K", "lower_K", "upper_K",
                                  "criterion"]
    np.testing.assert_allclose(band["upper_K"] - band["lower_K"], 2 * band["model_err_K"])
    # The band is FitResult.derive at each point.
    value, err = bf.fit.derive(lambda p: _dome(1.0, p["T0_K"], p["Hc_T"]))
    at = bf.band_at([1.0])
    assert at["model_K"].iloc[0] == pytest.approx(value)
    assert at["model_err_K"].iloc[0] == pytest.approx(err)
    # Beyond the points the band widens: the form is speaking there.
    far = bf.band_at([2.1])
    assert far["model_err_K"].iloc[0] > band["model_err_K"].max()


# ---------------------------------------------------------------------------
# Refusals.
# ---------------------------------------------------------------------------


def test_mixed_criteria_are_refused_with_the_fix():
    points = _points(np.random.default_rng(103))
    points.loc[5:, "criterion"] = "kink/down"
    with pytest.raises(ValueError, match=r"mix criterion.*Fit each criterion separately"):
        fit_boundary(points, DOME, swept_axis="T")


def test_mixed_method_or_sign_columns_are_refused():
    points = _points(np.random.default_rng(104), criterion="c")
    points["method"] = "peak"
    points["sign"] = ["max"] * 5 + ["min"] * 5
    with pytest.raises(ValueError, match=r"mix sign values \['max', 'min'\]"):
        fit_boundary(points, DOME, swept_axis="T")


def test_duplicate_sweeps_are_refused_windows_are_not_rows():
    points = _points(np.random.default_rng(105))
    points.loc[3, "sweep_id"] = 2
    with pytest.raises(ValueError, match=r"sweep_id\(s\) \[2\] appear more than once.*yerr_sys"):
        fit_boundary(points, DOME, swept_axis="T")


def test_missing_columns_are_named():
    points = _points(np.random.default_rng(106)).drop(columns=["T_err_K", "criterion"])
    with pytest.raises(ValueError, match=r"missing column\(s\) \['T_err_K', 'criterion'\]"):
        fit_boundary(points, DOME, swept_axis="T")
    # An H-swept frame needs the other roles.
    with pytest.raises(ValueError, match=r"missing column\(s\) \['H_err_T'\]"):
        fit_boundary(_points(np.random.default_rng(106)), DOME, swept_axis="H")
    assert POINT_COLUMNS["H"] == ("sweep_id", "T_K", "H_T", "H_err_T", "criterion")


def test_a_candidates_table_is_pointed_at_the_result():
    frame = _points(np.random.default_rng(107)).drop(columns="criterion")
    frame["method"], frame["sign"] = "peak", "max"
    with pytest.raises(ValueError, match=r"pass the select\(\)-ed TransitionResult itself"):
        fit_boundary(frame, DOME, swept_axis="T")


def test_a_frame_needs_its_swept_axis():
    with pytest.raises(ValueError, match=r"needs swept_axis='T'"):
        fit_boundary(_points(np.random.default_rng(108)), DOME)
    with pytest.raises(ValueError, match=r"swept_axis must be 'T' or 'H'"):
        fit_boundary(_points(np.random.default_rng(108)), DOME, swept_axis="x")


def test_unusable_errors_and_unknown_ids_are_refused():
    points = _points(np.random.default_rng(109))
    bad = points.copy()
    bad.loc[4, "T_err_K"] = 0.0
    with pytest.raises(ValueError, match=r"sweep\(s\) \[4\].*not finite and positive"):
        fit_boundary(bad, DOME, swept_axis="T")
    with pytest.raises(ValueError, match=r"exclude names sweep id\(s\) \[42\]"):
        fit_boundary(points, DOME, swept_axis="T", exclude=[42])
    with pytest.raises(ValueError, match=r"exclude must be a collection"):
        fit_boundary(points, DOME, swept_axis="T", exclude=3)
    with pytest.raises(ValueError, match=r"yerr_sys names sweep id\(s\) \[42\]"):
        fit_boundary(points, DOME, swept_axis="T", yerr_sys={42: 0.1})
    with pytest.raises(ValueError, match=r"sweep\(s\) \[1, 2, 3.*no finite, non-negative"):
        fit_boundary(points, DOME, swept_axis="T", yerr_sys={0: 0.1})
    with pytest.raises(KeyError, match="Unknown model"):
        fit_boundary(points, DOME, swept_axis="T", compare=["no_such_model"])
    with pytest.raises(ValueError, match="must be a select"):
        fit_boundary(points.to_dict(), DOME, swept_axis="T")


# ---------------------------------------------------------------------------
# A detected family: the sweeps table accounts for every sweep.
# ---------------------------------------------------------------------------

#: The T-scan grid of the detected family, in K.
T_SCAN = np.linspace(4.0, 30.0, 131)


def _cusp_file(path: Path, rng, *, h_t: float, t_n: float, frequencies=(511.0,)) -> Path:
    chi = simulate.synthetic_chi_peak(
        T_SCAN, centre=t_n, width=1.0, height=2.0, background=0.1, noise=0.02, rng=rng,
    )
    reps = len(frequencies)
    return simulate.write_mpms3_file(
        path, app="SQUID AC",
        temperature_K=np.repeat(T_SCAN, reps),
        field_oe=np.full(reps * T_SCAN.size, h_t * 1.0e4),
        chi_real=np.repeat(chi, reps),
        ac_frequency_hz=np.tile(np.asarray(frequencies, dtype=float), T_SCAN.size),
    )


def _mh_file(path: Path, rng, *, t_k: float) -> Path:
    h_oe = np.linspace(0.0, 30_000.0, 61)
    moment = 5.0e-4 * (1.0 + np.tanh((h_oe - 20_000.0) / 1500.0))
    moment = moment + rng.normal(scale=5.0e-6, size=h_oe.size)
    return simulate.write_mpms3_file(
        path, app="SQUID DC", temperature_K=np.full_like(h_oe, t_k), field_oe=h_oe,
        dc_moment_emu=moment,
    )


def _line_file(path: Path) -> Path:
    """A noiseless straight chi'(T): no extremum anywhere, so no candidate at all."""
    return simulate.write_mpms3_file(
        path, app="SQUID AC", temperature_K=T_SCAN, field_oe=np.full(T_SCAN.size, 1.6e4),
        chi_real=0.1 + 1.0e-3 * T_SCAN, ac_frequency_hz=np.full(T_SCAN.size, 511.0),
    )


@pytest.fixture(scope="module")
def family(tmp_path_factory):
    """Eight sweeps: five T-scans on the dome (sweeps 0, 2, 3, 4, 6), a
    multi-frequency T-scan detection refuses without ``frequency=`` (sweep 1,
    skipped), an M(H) isotherm (sweep 5), which belongs to the other family, and
    a featureless line (sweep 7), which yields no candidate at all. Sweep 6's
    cusp is planted at 25 K, outside the select window."""
    root = tmp_path_factory.mktemp("boundary_family")
    rng = np.random.default_rng(20260929)
    fields = {0: 0.0, 2: 0.6, 3: 1.0, 4: 1.4}
    paths = {}
    for sid, h_t in fields.items():
        paths[sid] = _cusp_file(root / f"t{sid}.dat", rng, h_t=h_t,
                                t_n=float(_dome(h_t, T0, HC)))
    paths[1] = _cusp_file(root / "mf.dat", rng, h_t=0.3, t_n=float(_dome(0.3, T0, HC)),
                          frequencies=(7.0, 511.0))
    paths[5] = _mh_file(root / "mh.dat", rng, t_k=5.0)
    paths[6] = _cusp_file(root / "t6.dat", rng, h_t=1.7, t_n=25.0)
    paths[7] = _line_file(root / "line.dat")
    sweeps = [load(paths[i]) for i in range(8)]
    result = detect_transitions_all(sweeps, outliers="ignore").robust()
    return result.select(swept_axis="T", method="peak", sign="max", t_window_k=(6.0, 18.0))


def test_the_sweeps_table_lists_every_sweep_of_the_family(family):
    before = family.candidates.copy()
    bf = _quiet(family, DOME, exclude=[3])
    sweeps = bf.sweeps
    assert list(sweeps.columns) == SWEEP_COLUMNS["T"]
    status = dict(zip(sweeps["sweep_id"], sweeps["status"], strict=True))
    assert status[0] == status[2] == status[4] == "used"
    assert status[3] == "excluded by caller"
    assert status[6] == "no candidate in window"
    assert status[1].startswith("skipped: ") and "frequenc" in status[1]
    assert status[7] == "no candidate detected"
    assert 5 not in status  # the isotherm belongs to the H-swept family ...
    assert bf.params["other_axis_sweep_ids"] == [5]  # ... and is recorded, not lost
    assert bf.params["exclude"] == [3]
    assert bf.params["n_used"] == 3

    rows = sweeps.set_index("sweep_id")
    # No imputation: the empty sweep has a prediction and no location.
    assert np.isnan(rows.loc[6, "T_K"]) and np.isnan(rows.loc[6, "T_err_K"])
    assert rows.loc[6, "H_T"] == pytest.approx(1.7)
    assert rows.loc[6, "predicted_K"] == pytest.approx(_dome(1.7, *bf.fit.parameters.values()))
    assert np.isnan(rows.loc[1, "H_T"]) and np.isnan(rows.loc[1, "predicted_K"])
    # A sweep with no candidate carries no held coordinate, so nothing is predicted.
    assert np.isnan(rows.loc[7, "H_T"]) and np.isnan(rows.loc[7, "predicted_K"])
    # predicted stays its own column, never the location.
    used = rows[rows["status"] == "used"]
    chosen = family.candidates.set_index("sweep_id")
    np.testing.assert_array_equal(used["T_K"], chosen.loc[used.index, "T_K"])
    np.testing.assert_allclose(used["residual_K"], used["T_K"] - used["predicted_K"])
    np.testing.assert_allclose(used["pull"], used["residual_K"] / used["T_err_fit_K"])
    np.testing.assert_array_equal(used["significance"], chosen.loc[used.index, "significance"])
    # The excluded point keeps its residual, from the fit made without it.
    assert np.isfinite(rows.loc[3, "residual_K"])
    assert rows.loc[3, "predicted_K"] == pytest.approx(
        float(bf.fit.curve([rows.loc[3, "H_T"]])[0]))
    assert bf.fit.n_points == 3
    assert (sweeps["criterion"] == "peak/max").all()
    # DM-1: the result it was built from is untouched.
    pd.testing.assert_frame_equal(family.candidates, before)


def test_a_result_needs_a_selection_on_the_axis(family):
    with pytest.raises(ValueError, match=r"has not been select\(\)-ed.*result.select\("):
        fit_boundary(family.all(), DOME)
    with pytest.raises(ValueError, match=r"no selection on swept_axis='H'.*h_window_t"):
        fit_boundary(family, DOME, swept_axis="H")


def test_a_result_selected_on_both_axes_needs_swept_axis(family):
    both = family.select(swept_axis="H", method="derivative", sign="rise",
                         h_window_t=(1.0, 2.9))
    with pytest.raises(ValueError, match=r"selected on \['H', 'T'\].*Pass swept_axis"):
        fit_boundary(both, DOME)
    bf = _quiet(both, DOME, swept_axis="T")
    assert bf.params["swept_axis"] == "T"
    assert bf.params["select"]["method"] == "peak"


def test_params_record_the_recipe(family):
    bf = _quiet(family, DOME, compare=["linear"], yerr_sys=0.05)
    p = bf.params
    assert p["model"] == "dome" and p["criterion"] == "peak/max" and p["swept_axis"] == "T"
    assert p["signal"] == "chi_real" and p["source"] == "TransitionResult"
    assert p["select"]["window"] == (6.0, 18.0)
    assert p["errors"] == "stat" and bf.total.params["errors"] == "stat+sys"
    assert p["yerr_sys"]["given"] and p["yerr_sys"]["source"] == "scalar"
    assert p["compare"] == ["linear"]
    assert p["low_chi2_tail"] == LOW_CHI2_TAIL
    assert p["calamari_version"] == calamari.__version__
    assert "significance" in p["weighting"]
    text = repr(bf)
    assert "'dome' model of T_K against H_T, criterion peak/max" in text
    assert "4 used, 0 excluded, 3 without a point" in text
    assert "phase" not in text.lower()


def test_no_output_says_phase(family):
    bf = _quiet(family, DOME, compare=["linear"], yerr_sys=0.05)
    for fit in (bf, bf.total):
        for table in (fit.sweeps, fit.estimates, fit.band, fit.jackknife, fit.comparison):
            assert not any("phase" in str(c).lower() for c in table.columns)
            text = table.astype(str).to_string().lower()
            assert "phase" not in text
        assert not any("phase" in w.lower() for w in fit.warnings)


# ---------------------------------------------------------------------------
# Systematic errors: two fits.
# ---------------------------------------------------------------------------


def test_yerr_sys_reports_both_fits_with_larger_errors_for_the_total():
    points = _points(np.random.default_rng(110))
    sys_err = {sid: 0.05 + 0.02 * sid for sid in points["sweep_id"]}
    bf = _quiet(points, DOME, swept_axis="T", yerr_sys=sys_err)
    assert bf.total is not None and bf.total.total is None
    for name in ("T0_K", "Hc_T"):
        assert bf.total.fit.uncertainties[name] > bf.fit.uncertainties[name]
    total = bf.total.sweeps.set_index("sweep_id")
    np.testing.assert_allclose(total["T_err_sys_K"], [sys_err[s] for s in total.index])
    np.testing.assert_allclose(total["T_err_fit_K"],
                               np.hypot(total["T_err_K"], total["T_err_sys_K"]))
    np.testing.assert_allclose(bf.sweeps["T_err_fit_K"], bf.sweeps["T_err_K"])
    assert (bf.total.band["model_err_K"] > bf.band["model_err_K"]).all()
    assert bf.params["yerr_sys"]["values"] == pytest.approx(sys_err)
    # Without yerr_sys there is one fit, and nothing is derived.
    plain = _quiet(points, DOME, swept_axis="T")
    assert plain.total is None and plain.sweeps["T_err_sys_K"].isna().all()


def test_a_systematic_column_is_the_alternative_to_yerr_sys():
    points = _points(np.random.default_rng(111))
    points["T_err_sys_K"] = 0.08
    bf = _quiet(points, DOME, swept_axis="T")
    assert bf.total is not None
    assert bf.params["yerr_sys"]["source"] == "column T_err_sys_K"
    with pytest.raises(ValueError, match=r"'T_err_sys_K' column and yerr_sys="):
        fit_boundary(points, DOME, swept_axis="T", yerr_sys=0.1)


# ---------------------------------------------------------------------------
# Diagnostics.
# ---------------------------------------------------------------------------


def test_the_jackknife_finds_an_outlier_sweep():
    points = _points(np.random.default_rng(112), shift=(6, 1.0))  # a 10-sigma point
    bf = _quiet(points, DOME, swept_axis="T")
    jack = bf.jackknife
    assert len(jack) == H_FAMILY.size and (jack["refit"] == "ok").all()
    largest = jack[["T0_K_shift_sigma", "Hc_T_shift_sigma"]].abs().max(axis=1)
    assert int(jack.loc[largest.idxmax(), "sweep_id"]) == 6
    estimates = bf.estimates.set_index("parameter")
    assert 6 in set(estimates["most_influential_sweep_id"])
    assert estimates["max_abs_shift_sigma"].max() == pytest.approx(largest.max())
    # The jackknife SE is reported beside HESSE, from the formula.
    n = H_FAMILY.size
    leave_out = jack["Hc_T"].to_numpy()
    expected = np.sqrt((n - 1) / n * np.sum((leave_out - leave_out.mean()) ** 2))
    assert estimates.loc["Hc_T", "jackknife_se"] == pytest.approx(expected)
    # The shift is in units of the full fit's HESSE error.
    row = jack.set_index("sweep_id").loc[6]
    assert row["Hc_T_shift_sigma"] == pytest.approx(
        (row["Hc_T"] - bf.fit.parameters["Hc_T"]) / bf.fit.uncertainties["Hc_T"])


def test_an_underdetermined_refit_is_reported_not_dropped():
    points = _points(np.random.default_rng(113)).iloc[:3]
    with pytest.warns(UserWarning, match=r"did not succeed.*underdetermined: dof 0"):
        bf = fit_boundary(points, DOME, swept_axis="T")
    assert bf.fit.success and bf.fit.dof == 1
    assert list(bf.jackknife["refit"]) == ["underdetermined: dof 0"] * 3
    assert bf.jackknife["T0_K"].isna().all()
    assert bf.estimates["jackknife_se"].isna().all()


def test_a_failed_fit_is_returned_not_raised():
    """FT-1: two points cannot fix two parameters and leave a residual."""
    points = _points(np.random.default_rng(114)).iloc[:2]
    with pytest.warns(UserWarning, match=r"did not converge \(success=False; 2 points, dof 0\)"):
        bf = fit_boundary(points, DOME, swept_axis="T")
    assert not bf.fit.success
    assert bf.sweeps["predicted_K"].isna().all()
    assert bf.band["model_K"].isna().all()
    assert len(bf.jackknife) == 0
    assert np.isnan(bf.chi_squared_quantile) and not bf.scatter_below_errors
    assert "FAILED" in repr(bf)


def test_compare_prefers_the_curved_form_on_curved_data():
    points = _points(np.random.default_rng(115))
    bf = _quiet(points, DOME, swept_axis="T", compare=["linear"])
    table = bf.comparison.set_index("model")
    assert table.loc["dome", "role"] == "stated" and table.loc["dome", "delta_bic"] == 0.0
    assert table.loc["linear", "success"] and table.loc["linear", "n_free"] == 2
    assert table.loc["linear", "delta_bic"] > 0  # positive favours the stated model
    assert table.loc["linear", "delta_aic"] > 0
    # The criteria are chi-squared based with the supplied errors.
    n = H_FAMILY.size
    assert table.loc["linear", "bic"] == pytest.approx(
        table.loc["linear", "chi_squared"] + 2 * np.log(n))
    # Stated the other way round, the sign flips.
    line_first = _quiet(points, "linear", swept_axis="T", compare=[DOME])
    assert line_first.comparison.set_index("model").loc["dome", "delta_bic"] < 0


def test_grid_floor_errors_are_flagged_as_scatter_below_errors():
    """Errors of 0.2 K on points scattered by 0.004 K, as a grid-floored location
    error gives on clean sweeps: chi2/dof comes out near 0.0004."""
    points = _points(np.random.default_rng(116), sigma=0.004, err=0.2)
    with pytest.warns(UserWarning, match=r"scatter far less than their errors allow"):
        bf = fit_boundary(points, DOME, swept_axis="T")
    assert bf.scatter_below_errors
    assert bf.chi_squared_quantile < LOW_CHI2_TAIL
    assert "scatter below errors" in repr(bf)
    estimates = bf.estimates.set_index("parameter")
    # The jackknife reads the scatter, HESSE the stated errors.
    assert (estimates["jackknife_se"] < 0.2 * estimates["hesse_err"]).all()
    honest = _quiet(_points(np.random.default_rng(116)), DOME, swept_axis="T")
    assert not honest.scatter_below_errors


def test_the_flag_criterion_is_the_chi2_distribution_not_chi2_per_dof():
    """With one degree of freedom a tiny chi2/dof is not implausible: P(chi2_1 <= 1e-4)
    is about 0.008, above the 3-sigma tail, so the same chi2/dof that is flagged on
    many dof is not flagged here."""
    points = _points(np.random.default_rng(117), sigma=0.004, err=0.2).iloc[[0, 5, 9]]
    bf = _quiet(points, DOME, swept_axis="T")
    assert bf.fit.dof == 1 and bf.fit.reduced_chi_squared < 0.01
    assert not bf.scatter_below_errors
    assert LOW_CHI2_TAIL == pytest.approx(0.0013499, rel=1e-4)


def test_an_h_swept_family_uses_the_other_roles():
    rng = np.random.default_rng(118)
    t_held = np.linspace(2.0, 12.0, 8)
    h = HC * np.sqrt(1.0 - (t_held / T0) ** 2) + rng.normal(scale=0.01, size=t_held.size)
    frame = pd.DataFrame({"sweep_id": range(8), "T_K": t_held, "H_T": h, "H_err_T": 0.01,
                          "criterion": "derivative/rise"})
    ellipse = ModelDefinition(
        name="ellipse", function=lambda t, H0_T, Tc_K: _dome(t, H0_T, Tc_K),
        param_names=("H0_T", "Tc_K"), param_defaults={"H0_T": 2.0, "Tc_K": 15.0},
        description="H0 sqrt(1 - (T/Tc)^2)")
    bf = _quiet(frame, ellipse, swept_axis="H")
    assert list(bf.sweeps.columns) == SWEEP_COLUMNS["H"]
    assert "predicted_T" in bf.sweeps.columns and "residual_T" in bf.sweeps.columns
    assert list(bf.band.columns[:2]) == ["T_K", "model_T"]
    assert abs(bf.fit.parameters["H0_T"] - HC) < 3 * bf.fit.uncertainties["H0_T"]
    assert "H_T against T_K" in repr(bf)


def test_a_frame_row_without_a_location_is_listed_not_imputed():
    points = _points(np.random.default_rng(119))
    points.loc[4, ["T_K", "T_err_K", "criterion"]] = [np.nan, np.nan, None]
    before = points.copy()
    bf = _quiet(points, DOME, swept_axis="T")
    row = bf.sweeps.set_index("sweep_id").loc[4]
    assert row["status"] == "no point given"
    assert np.isnan(row["T_K"]) and np.isfinite(row["predicted_K"])
    assert bf.fit.n_points == 9
    pd.testing.assert_frame_equal(points, before)  # DM-1


def test_significance_is_shown_not_weighted():
    """Doubling one point's significance changes nothing in the fit."""
    points = _points(np.random.default_rng(120))
    points["significance"] = 10.0
    heavier = points.copy()
    heavier.loc[3, "significance"] = 1000.0
    a = _quiet(points, DOME, swept_axis="T")
    b = _quiet(heavier, DOME, swept_axis="T")
    assert a.fit.parameters == b.fit.parameters
    assert b.sweeps.set_index("sweep_id").loc[3, "significance"] == 1000.0


def test_reexported_from_calamari():
    assert calamari.fit_boundary is fit_boundary
    assert calamari.BoundaryFit is BoundaryFit
    assert "fit_boundary" in calamari.__all__ and "BoundaryFit" in calamari.__all__


def test_fit_boundary_propagates_fixed_limit_conflict():
    points = _points(np.random.default_rng(123))
    with pytest.raises(calamari.FitConstraintError, match="T0_K"):
        fit_boundary(points, DOME, swept_axis="T",
                     fixed={"T0_K": 14.0}, limits={"T0_K": (0.0, 10.0)})
