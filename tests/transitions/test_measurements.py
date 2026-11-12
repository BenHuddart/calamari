"""Measurement preparation, screening and batch detection workflows."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calamari import simulate, sweeps, transitions
from calamari.io import load
from calamari.transitions import (
    CANDIDATE_COLUMNS,
    DETECTION_COLUMNS,
    detect_transitions,
    detect_transitions_all,
    find_transitions,
)

from ._helpers import (
    FISHER_NOISE,
    FISHER_T_N,
    _ac_tscan_file,
    _curve_rng,
    _toy_afm_ac_file,
    _toy_afm_chi,
    _toy_afm_dc_file,
    _toy_afm_t,
)


def _dc_hscan_file(
    path, *, fixed_t_k=5.0, h_c_t=2.0, with_errors=True, loop=False, err_scale=0.005
, rng):
    """One M(H) metamagnetic step at ``h_c_t`` Tesla.

    ``err_scale`` is the ``dc_moment_err_emu`` value written to the file;
    the actual point-to-point scatter is always ``0.005`` (the generated noise
    below), so passing a larger ``err_scale`` reproduces an instrument
    error column that overstates the real scatter -- see the noise=
    override tests.
    """
    h_oe = np.linspace(0.0, 50_000.0, 141)
    mom = 0.5 * (1 + np.tanh((h_oe / 1e4 - h_c_t) / 0.15))
    mom = mom + rng.normal(scale=0.005, size=h_oe.shape)
    if loop:
        h_oe = np.concatenate([h_oe, h_oe[::-1]])
        mom = np.concatenate([mom, mom[::-1]])
    kwargs = {"dc_moment_emu": mom}
    if with_errors:
        kwargs["dc_moment_err_emu"] = np.full_like(h_oe, err_scale)
    return simulate.write_mpms3_file(
        path,
        app="SQUID",
        temperature_K=np.full_like(h_oe, fixed_t_k),
        field_oe=h_oe,
        **kwargs,
    )


def test_detect_ac_tscan_places_cusp_at_fixed_field(tmp_path, rng):
    m = load(_ac_tscan_file(tmp_path / "ac.dat", rng=rng))
    result = detect_transitions(m).robust()
    assert list(result.candidates.columns) == CANDIDATE_COLUMNS
    assert list(result.detections.columns) == DETECTION_COLUMNS
    row = result.candidates.iloc[0]
    assert row["T_K"] == pytest.approx(14.0, abs=0.2)
    assert row["H_T"] == pytest.approx(0.01, abs=1e-6)
    assert row["T_err_K"] > 0
    assert np.isnan(row["H_err_T"])  # the fixed axis has a span, not an error
    assert row["H_span_T"] >= 0
    assert row["swept_axis"] == "T"
    assert row["signal"] == "chi_real"
    assert row["frequency_hz"] == pytest.approx(511.0)
    assert row["method"] == "peak"
    assert row["measurement_file"] == str(m.path)


def test_detect_dc_hscan_uses_instrument_errors(tmp_path, rng):
    m = load(_dc_hscan_file(tmp_path / "dc.dat", rng=rng))
    result = detect_transitions(m)
    assert result.params["per_signal"]["moment_emu"]["noise_source"] == "instrument"
    assert result.params["per_signal"]["moment_emu"]["n_duplicates_dropped"] == 0
    row = result.robust().candidates.iloc[0]
    assert row["H_T"] == pytest.approx(2.0, abs=0.05)
    assert row["T_K"] == pytest.approx(5.0, abs=1e-6)
    assert row["swept_axis"] == "H"
    assert row["method"] == "derivative"
    # T-scan default would run peak too; an M(H) sweep runs derivative only.
    assert result.params["per_signal"]["moment_emu"]["methods"] == ("derivative",)


def test_detect_multifrequency_refused_then_selected(tmp_path, rng):
    path = _ac_tscan_file(tmp_path / "mf.dat", frequencies=(7.0, 511.0), rng=rng)
    m = load(path)
    with pytest.raises(ValueError, match="frequencies"):
        detect_transitions(m)
    result = detect_transitions(m, frequency=511.0)
    assert result.params["frequency_hz"] == pytest.approx(511.0)
    row = result.robust().candidates.iloc[0]
    assert row["T_K"] == pytest.approx(14.0, abs=0.2)
    assert row["frequency_hz"] == pytest.approx(511.0)


def test_detect_mixed_scan_refused_towards_split_isotherms(tmp_path):
    temps_hc = {2.0: 1.5, 3.0: 1.2}
    path = _write_stacked_mh_file(tmp_path / "stacked.dat", temps_hc)
    with pytest.raises(ValueError, match="split_isotherms"):
        detect_transitions(load(path))


def test_detect_hysteresis_needs_branch(tmp_path, rng):
    m = load(_dc_hscan_file(tmp_path / "loop.dat", loop=True, rng=rng))
    with pytest.raises(ValueError, match="branch"):
        detect_transitions(m)
    result = detect_transitions(m, branch="increasing").robust()
    assert result.candidates.iloc[0]["H_T"] == pytest.approx(2.0, abs=0.05)


def test_detect_signal_all_tags_channels(tmp_path, rng):
    t = np.linspace(4.0, 30.0, 131)
    chi = simulate.synthetic_chi_peak(
        t, centre=14.0, width=1.0, height=2.0, background=0.1, noise=0.02, rng=rng
    )
    chi_im = simulate.synthetic_chi_peak(
        t, centre=13.5, width=1.0, height=0.5, background=0.02, noise=0.005, rng=rng
    )
    path = simulate.write_mpms3_file(
        tmp_path / "both.dat", app="SQUID AC",
        temperature_K=t, field_oe=np.full_like(t, 100.0),
        chi_real=chi, chi_imag=chi_im, ac_frequency_hz=np.full_like(t, 511.0),
    )
    m = load(path)
    # Default: ONE primary channel, so one transition is reported once.
    default = detect_transitions(m).robust()
    assert set(default.candidates["signal"]) == {"chi_real"}
    # signal='all' searches both, tagged -- nothing merges across signals.
    both = detect_transitions(m, signal="all").robust()
    assert set(both.candidates["signal"]) == {"chi_real", "chi_imag"}


def test_detect_top_caps_visibly(tmp_path, rng):
    m = load(_ac_tscan_file(tmp_path / "ac.dat", rng=rng))
    result = detect_transitions(m, top=1)
    assert result.params["top"] == 1
    per_group = result.candidates.groupby(["signal", "method"]).size()
    assert (per_group <= 1).all()


def test_detect_t_window_restricts(tmp_path, rng):
    m = load(_ac_tscan_file(tmp_path / "ac.dat", rng=rng))
    result = detect_transitions(m, t_window_k=(10.0, 18.0)).robust()
    assert ((result.candidates["T_K"] >= 10.0) & (result.candidates["T_K"] <= 18.0)).all()
    assert result.params["t_window_k"] == (10.0, 18.0)

# Quality screening before detection (outliers=).

#: Where the planted glitch sits: row 90 of the 131-point grid below, on the
#: smooth high-temperature tail, well clear of the 14 K cusp.
_GLITCH_ROW, _GLITCH_T = 90, 22.0


def _glitch_tscan_file(path, *, glitch=True, frequencies=(511.0,), glitch_frequency=None):
    """A chi'(T) cusp at 14 K with one bad reading at 22 K.

    Its own seeded generator, so the planted scatter -- and hence the
    glitch's score and the spurious candidates it causes -- does not depend
    on which tests ran first. With several ``frequencies`` the rows
    interleave, and the glitch lands on ``glitch_frequency`` only.
    """
    rng = np.random.default_rng(20260927)
    t = np.linspace(4.0, 30.0, 131)
    chi = simulate.synthetic_chi_peak(
        t, centre=14.0, width=1.0, height=2.0, background=0.1, noise=0.02, rng=rng
    )
    reps = len(frequencies)
    freq = np.tile(np.asarray(frequencies, dtype=float), len(t))
    chi_rows = np.repeat(chi, reps)
    if glitch:
        target = frequencies[0] if glitch_frequency is None else glitch_frequency
        chi_rows[_GLITCH_ROW * reps + list(frequencies).index(target)] += 1.0
    return simulate.write_mpms3_file(
        path,
        app="SQUID AC",
        temperature_K=np.repeat(t, reps),
        field_oe=np.full(reps * len(t), 100.0),
        chi_real=chi_rows,
        ac_frequency_hz=freq,
    )


def _unscreenable_tscan_file(path):
    """The glitch-free sweep plus ``Min./Max. Temperature`` columns in degC.

    quality() reads those columns for its temperature-spread evidence and
    refuses the unrecognised unit (UnknownUnitError); detection never reads
    them. So screening fails on a sweep detection handles perfectly well --
    the shape that made the default 'warn' a regression until screening
    failure stopped blocking detection.
    """
    rng = np.random.default_rng(20260927)
    t = np.linspace(4.0, 30.0, 131)
    chi = simulate.synthetic_chi_peak(
        t, centre=14.0, width=1.0, height=2.0, background=0.1, noise=0.02, rng=rng
    )
    return simulate.write_mpms3_file(
        path,
        app="SQUID AC",
        temperature_K=t,
        field_oe=np.full_like(t, 100.0),
        chi_real=chi,
        ac_frequency_hz=np.full_like(t, 511.0),
        extra_columns={
            "Min. Temperature (degC)": t - 273.15,
            "Max. Temperature (degC)": t - 273.15,
        },
    )


def _near_glitch(candidates) -> pd.DataFrame:
    return candidates[(candidates["T_K"] - _GLITCH_T).abs() < 2.5]


def test_glitch_is_flagged_and_the_cusp_is_not(tmp_path):
    """The premise of every test below: quality() flags exactly the planted
    point, and not the 14 K cusp detection is meant to find."""
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    assert list(m.quality().flagged.index) == [_GLITCH_ROW]


def test_outliers_ignore_keeps_the_glitch_and_its_spurious_candidates(tmp_path):
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 'ignore' means no warning at all
        result = detect_transitions(m, outliers="ignore")
    assert len(_near_glitch(result.candidates)) >= 2
    # The glitch's candidates survive the default publication filter too.
    assert len(_near_glitch(result.robust().candidates)) >= 1
    assert result.params["outliers"]["n_dropped"] == 0


def test_outliers_drop_removes_the_spurious_candidates_and_keeps_the_cusp(tmp_path):
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    result = detect_transitions(m, outliers="drop")
    assert _near_glitch(result.candidates).empty
    kept = result.robust().candidates
    assert len(kept) == 1
    assert kept.iloc[0]["T_K"] == pytest.approx(14.0, abs=0.2)
    assert result.params["outliers"]["n_dropped"] == 1


def test_outliers_warn_is_the_default_and_warns_once_naming_the_point(tmp_path):
    """The default keeps the point (the candidates are those of 'ignore')
    and says so once, naming file, coordinate, value, score and the fixes."""
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = detect_transitions(m)
    assert len(record) == 1
    message = str(record[0].message)
    assert record[0].category is UserWarning
    assert "g.dat" in message
    assert f"at T = {_GLITCH_T:g} K (row {_GLITCH_ROW})" in message
    assert f"chi_real {m.chi_real[_GLITCH_ROW]:.4g}" in message
    for fix in ("outliers='drop'", "outliers='ignore'", "outlier_threshold="):
        assert fix in message
    assert result.params["outliers"]["mode"] == "warn"
    ignored = detect_transitions(m, outliers="ignore")
    pd.testing.assert_frame_equal(result.candidates, ignored.candidates)


def test_flagged_point_outside_the_window_does_not_warn(tmp_path):
    """Only points detection will search are worth a warning. The glitch is
    still scored (on the whole file), just not counted as searched."""
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = detect_transitions(m, t_window_k=(4.0, 18.0))
    record = result.params["outliers"]
    assert record["n_flagged_in_search"] == 0
    assert record["flagged"] == []
    assert detect_transitions(m, t_window_k=(18.0, 30.0), outliers="ignore").params[
        "outliers"
    ]["n_flagged_in_search"] == 1


def test_scored_on_the_measurement_as_passed_not_the_window(tmp_path):
    """A window whose edge falls on the glitch must not change its score:
    scoring happens before the window, against the file's own neighbours."""
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    full = detect_transitions(m, outliers="ignore").params["outliers"]["flagged"]
    edge = detect_transitions(m, t_window_k=(_GLITCH_T, 30.0), outliers="ignore").params[
        "outliers"
    ]["flagged"]
    assert edge == full


def test_flagged_point_at_another_frequency_does_not_warn(tmp_path):
    """A glitch at a frequency that is not searched cannot move a candidate;
    warning about it would send the caller to the wrong data."""
    path = _glitch_tscan_file(
        tmp_path / "mf.dat", frequencies=(7.0, 511.0), glitch_frequency=7.0
    )
    m = load(path)
    assert len(m.quality().flagged) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clean = detect_transitions(m, frequency=511.0)
    assert clean.params["outliers"]["n_flagged_in_search"] == 0
    with pytest.warns(UserWarning, match="outliers='drop'"):
        detect_transitions(m, frequency=7.0)


def test_outliers_params_record_the_screening(tmp_path):
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    record = detect_transitions(m, outliers="drop").params["outliers"]
    assert set(record) == {
        "mode", "threshold", "screened", "error", "n_flagged_in_search", "n_dropped",
        "flagged",
    }
    assert record["mode"] == "drop"
    assert record["screened"] is True and record["error"] is None
    assert record["threshold"] == 20.0
    assert record["n_flagged_in_search"] == 1
    (point,) = record["flagged"]
    assert set(point) == {"row", "T_K", "value", "score"}
    assert point["row"] == _GLITCH_ROW
    assert point["T_K"] == pytest.approx(_GLITCH_T)
    assert point["value"] == m.chi_real[_GLITCH_ROW]
    assert point["score"] > 20.0
    # outlier_threshold moves the line, and the line is recorded.
    lax = detect_transitions(m, outliers="drop", outlier_threshold=1e6).params["outliers"]
    assert lax["threshold"] == 1e6
    assert lax["n_dropped"] == 0 and lax["flagged"] == []


def test_quality_report_is_an_explicit_drop(tmp_path):
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    via_report = detect_transitions(m, outliers=m.quality())
    via_drop = detect_transitions(m, outliers="drop")
    assert via_report.params["outliers"]["mode"] == "report"
    assert via_report.params["outliers"]["n_dropped"] == 1
    pd.testing.assert_frame_equal(via_report.candidates, via_drop.candidates)


def test_quality_report_taken_before_restrict_is_refused(tmp_path):
    """restrict() renumbers rows from 0, so a report from before it names
    labels that no longer exist -- or, the other way round, labels that now
    hold different points. Both must be refused, not silently misapplied."""
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    sub = m.restrict(t_range=(6.0, 30.0))
    with pytest.raises(ValueError, match=r"restrict\(\)"):
        detect_transitions(sub, outliers=m.quality())
    with pytest.raises(ValueError, match=r"chi_real at row 0 .*restrict\(\)"):
        detect_transitions(m, outliers=sub.quality())


def test_quality_report_from_another_file_is_refused(tmp_path):
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    other = load(_glitch_tscan_file(tmp_path / "other.dat"))
    with pytest.raises(ValueError, match="other.dat.*another file"):
        detect_transitions(m, outliers=other.quality())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"outliers": "bogus"}, "outliers must be"),
        ({"outliers": None}, "outliers must be"),
        ({"outliers": True}, "outliers must be"),
        ({"outlier_threshold": 0.0}, "outlier_threshold must be"),
        ({"outlier_threshold": -5.0}, "outlier_threshold must be"),
        ({"outlier_threshold": float("nan")}, "outlier_threshold must be"),
        ({"outlier_threshold": True}, "outlier_threshold must be"),
        ({"outlier_threshold": "20"}, "outlier_threshold must be"),
    ],
)
def test_outlier_keywords_validated(tmp_path, kwargs, match):
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    with pytest.raises(ValueError, match=match):
        detect_transitions(m, **kwargs)


def test_quality_report_with_a_threshold_is_refused(tmp_path):
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    with pytest.raises(ValueError, match="its own threshold"):
        detect_transitions(m, outliers=m.quality(), outlier_threshold=5.0)


def test_unscreenable_premise_quality_raises_on_a_column_detection_ignores(tmp_path):
    from calamari.exceptions import UnknownUnitError

    m = load(_unscreenable_tscan_file(tmp_path / "degc.dat"))
    with pytest.raises(UnknownUnitError, match="degC"):
        m.quality()


def _same_sweep_screenable(tmp_path):
    """The unscreenable sweep's candidates as detection found them before
    screening existed: the identical data without the degC columns, with
    screening switched off."""
    return detect_transitions(
        load(_glitch_tscan_file(tmp_path / "plain.dat", glitch=False)),
        smooth_window=0.8, outliers="ignore",
    )


def test_unscreenable_sweep_warn_detects_unscreened_and_warns_once(tmp_path):
    m = load(_unscreenable_tscan_file(tmp_path / "degc.dat"))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = detect_transitions(m, smooth_window=0.8)
    assert len(record) == 1
    message = str(record[0].message)
    assert record[0].category is UserWarning
    assert "degc.dat was not screened" in message
    assert "UnknownUnitError" in message
    assert "outliers='ignore'" in message
    expected = _same_sweep_screenable(tmp_path)
    pd.testing.assert_frame_equal(
        result.candidates.drop(columns="measurement_file"),
        expected.candidates.drop(columns="measurement_file"),
    )
    assert len(result.robust().candidates) == 1


def test_unscreenable_sweep_ignore_detects_unscreened_silently(tmp_path):
    m = load(_unscreenable_tscan_file(tmp_path / "degc.dat"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = detect_transitions(m, smooth_window=0.8, outliers="ignore")
    assert result.robust().candidates.iloc[0]["T_K"] == pytest.approx(14.0, abs=0.2)
    assert result.params["outliers"]["screened"] is False


def test_unscreenable_sweep_drop_raises_chaining_the_cause(tmp_path):
    """A removal that cannot be honoured must not pass for a cleaned sweep."""
    from calamari.exceptions import UnknownUnitError

    m = load(_unscreenable_tscan_file(tmp_path / "degc.dat"))
    with pytest.raises(ValueError, match="could not be screened") as info:
        detect_transitions(m, smooth_window=0.8, outliers="drop")
    message = str(info.value)
    assert "outliers='ignore'" in message and "restrict(mask=" in message
    assert isinstance(info.value.__cause__, UnknownUnitError)


def test_unscreenable_sweep_record(tmp_path):
    m = load(_unscreenable_tscan_file(tmp_path / "degc.dat"))
    record = detect_transitions(
        m, smooth_window=0.8, outliers="ignore", outlier_threshold=50.0
    ).params["outliers"]
    assert record["screened"] is False
    assert record["error"].startswith("UnknownUnitError: ")
    assert "degC" in record["error"]
    assert record["mode"] == "ignore"
    assert record["threshold"] == 50.0
    assert record["n_flagged_in_search"] == 0
    assert record["n_dropped"] == 0
    assert record["flagged"] == []


def test_unscreenable_sweep_passed_report_is_unaffected(tmp_path):
    """A report needs no scoring, so a report from the plain twin is refused
    for the path, as any foreign report is -- not for screening failure."""
    m = load(_unscreenable_tscan_file(tmp_path / "degc.dat"))
    other = load(_glitch_tscan_file(tmp_path / "plain.dat", glitch=False))
    with pytest.raises(ValueError, match="another file"):
        detect_transitions(m, outliers=other.quality())


def test_batch_shows_which_sweeps_went_unscreened(tmp_path):
    screened = load(_glitch_tscan_file(tmp_path / "g.dat"))
    unscreened = load(_unscreenable_tscan_file(tmp_path / "degc.dat"))
    clean = load(_glitch_tscan_file(tmp_path / "clean.dat", glitch=False))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        batch = detect_transitions_all([screened, unscreened, clean], smooth_window=0.8)
    assert batch.skipped == ()
    per_sweep = batch.params["per_sweep"]
    assert [e["outliers"]["screened"] for e in per_sweep] == [True, False, True]
    assert per_sweep[1]["outliers"]["error"].startswith("UnknownUnitError: ")
    assert [e["outliers"]["n_flagged_in_search"] for e in per_sweep] == [1, 0, 0]
    messages = [str(r.message) for r in record]
    assert sum("was not screened" in s for s in messages) == 1
    assert sum("in the searched rows" in s for s in messages) == 1
    # Under 'drop' the unscreenable sweep is refused, visibly, not searched.
    dropped = detect_transitions_all([screened, unscreened, clean], outliers="drop")
    assert [Path(p).name for p, _ in dropped.skipped] == ["degc.dat"]
    assert "could not be screened" in dropped.skipped[0][1]


def test_batch_refuses_a_malformed_outliers_value_instead_of_skipping(tmp_path):
    """A typo is the caller's error, not every sweep's failure."""
    m = load(_glitch_tscan_file(tmp_path / "g.dat"))
    with pytest.raises(ValueError, match="outliers must be"):
        detect_transitions_all([m], outliers="dorp")


def test_batch_records_each_sweeps_screening(tmp_path):
    glitched = load(_glitch_tscan_file(tmp_path / "g.dat"))
    clean = load(_glitch_tscan_file(tmp_path / "clean.dat", glitch=False))
    batch = detect_transitions_all([glitched, clean], outliers="drop")
    per_sweep = batch.params["per_sweep"]
    assert [e["outliers"]["n_dropped"] for e in per_sweep] == [1, 0]
    for entry, m in zip(per_sweep, (glitched, clean), strict=True):
        assert entry["outliers"] == detect_transitions(m, outliers="drop").params["outliers"]
    assert _near_glitch(batch.candidates).empty
    assert batch.params["detect_kwargs"]["outliers"] == "drop"

# Batch detection.


def _mh_step_moment(field_tesla, h_c, width=0.08):
    return 0.5 * (1.0 + np.tanh((field_tesla - h_c) / width))


def _write_stacked_mh_file(path, temps_hc, n_per_isotherm=60):
    """A stacked M(H) file, one 0 -> 2.5 T ramp per temperature, each with
    a metamagnetic step at that isotherm's own critical field."""
    field_oe = np.linspace(0.0, 25_000.0, n_per_isotherm)
    field_tesla = field_oe / 1e4
    temps = list(temps_hc)
    temperature_K = np.concatenate([np.full(n_per_isotherm, t) for t in temps])
    field_stacked = np.tile(field_oe, len(temps))
    moment_emu = np.concatenate(
        [_mh_step_moment(field_tesla, temps_hc[t]) for t in temps]
    )
    return simulate.write_mpms3_file(
        path,
        app="SQUID DC",
        temperature_K=temperature_K,
        field_oe=field_stacked,
        vsm_moment_emu=moment_emu,
    )


def test_batch_collects_failures_instead_of_hiding_them(tmp_path, rng):
    good = load(_ac_tscan_file(tmp_path / "good.dat", rng=rng))
    bad = load(_ac_tscan_file(tmp_path / "bad.dat", frequencies=(7.0, 511.0), rng=rng))
    result = detect_transitions_all([good, bad])
    assert len(result.skipped) == 1
    assert result.skipped[0][0] == str(bad.path)
    assert "frequencies" in result.skipped[0][1]
    assert result.params["n_sweeps"] == 2
    assert result.params["n_searched"] == 1
    assert set(result.candidates["probe"]) == {"AC"}


def test_batch_accepts_split_isotherm_pairs(tmp_path):
    temps_hc = {2.0: 1.5, 3.0: 1.2, 4.0: 0.9}
    path = _write_stacked_mh_file(tmp_path / "stacked.dat", temps_hc)
    pairs = sweeps.split_isotherms(load(path))
    result = detect_transitions_all(pairs).robust()
    assert set(result.candidates["probe"]) == {"DC"}
    for t_k, h_c in temps_hc.items():
        rows = result.candidates[np.isclose(result.candidates["T_K"], t_k, atol=0.01)]
        assert not rows.empty, f"no candidate at T = {t_k} K"
        best = rows.loc[rows["significance"].idxmax()]
        assert best["H_T"] == pytest.approx(h_c, abs=0.15)


def test_batch_shared_path_pairs_keep_detections_linked(tmp_path):
    """split_isotherms pairs are sub-Measurements of ONE file, so every
    sweep shares a path while restarting candidate_id at 0. Re-keying must
    stay positional: a path-keyed map silently attached every detection to
    the last sweep's candidate (caught in review)."""
    temps_hc = {2.0: 1.5, 3.0: 1.2, 4.0: 0.9}
    path = _write_stacked_mh_file(tmp_path / "stacked.dat", temps_hc)
    result = detect_transitions_all(sweeps.split_isotherms(load(path)))
    assert result.candidates["candidate_id"].is_unique
    merged = result.detections.merge(
        result.candidates, on="candidate_id", suffixes=("_det", "_cand")
    )
    assert len(merged) == len(result.detections)
    # Each detection's own coordinates must match the candidate it links to.
    assert (merged["T_K_det"] == merged["T_K_cand"]).all()
    assert np.allclose(merged["H_T_det"], merged["H_T_cand"], atol=0.2)


def test_detect_methods_accepts_any_string_sequence(tmp_path, rng):
    """Only None means defaults: an ndarray of method names is the
    caller's explicit choice, not silently replaced (caught in review)."""
    m = load(_dc_hscan_file(tmp_path / "dc.dat", rng=rng))
    result = detect_transitions(m, methods=np.array(["derivative"]))
    assert result.params["per_signal"]["moment_emu"]["methods"] == ("derivative",)
    with pytest.raises(ValueError, match="methods"):
        detect_transitions(m, methods=np.array(["wavelet"]))


def test_batch_robust_forwarding_records_thresholds(tmp_path, rng):
    m = load(_ac_tscan_file(tmp_path / "ac.dat", rng=rng))
    result = detect_transitions_all([m], robust={"min_significance": 10.0})
    assert result.params["robust"]["min_significance"] == 10.0
    assert (result.candidates["significance"] >= 10.0).all()


def test_batch_candidate_ids_link_detections(tmp_path, rng):
    files = [
        _ac_tscan_file(tmp_path / "a.dat", fixed_h_oe=100.0, t_n=14.0, rng=rng),
        _ac_tscan_file(tmp_path / "b.dat", fixed_h_oe=2000.0, t_n=12.0, rng=rng),
    ]
    result = detect_transitions_all([load(f) for f in files])
    assert result.candidates["candidate_id"].is_unique
    assert set(result.detections["candidate_id"]) <= set(result.candidates["candidate_id"])
    # Every detection's file agrees with its candidate's file.
    merged = result.detections.merge(
        result.candidates, on="candidate_id", suffixes=("_det", "_cand")
    )
    assert (merged["measurement_file_det"] == merged["measurement_file_cand"]).all()


def test_batch_curves_survive_sweeps_that_share_a_path(tmp_path):
    """Two restrict() slices of one file share its path. Keyed by path, the
    second slice's curves overwrote the first's, so a batch searched two
    sweeps and could show the curve behind only one of them."""
    m = load(_ac_tscan_file(tmp_path / "ac.dat", rng=_curve_rng()))
    slices = [m.restrict(t_range=(4, 16)), m.restrict(t_range=(12, 30))]
    result = detect_transitions_all(slices, outliers="ignore")
    assert result.params["n_searched"] == 2
    assert len(result.curves) == 2
    for sweep_id, (lo, hi) in enumerate([(4, 16), (12, 30)]):
        x = result.curve(sweep_id)["x"]
        assert x.min() == pytest.approx(lo, abs=0.25)
        assert x.max() == pytest.approx(hi, abs=0.25)
    # Every candidate row addresses the curve it was found in.
    for row in result.candidates.itertuples():
        x = result.curve(row.sweep_id, row.signal)["x"]
        assert x.min() <= row.T_K <= x.max()


def test_batch_curves_keep_every_split_isotherm(tmp_path):
    """The isotherms split_isotherms cuts from one file all carry its path,
    so they collided too: each must keep the curve of its own step."""
    temps_hc = {2.0: 1.5, 3.0: 1.2, 4.0: 0.9}
    path = _write_stacked_mh_file(tmp_path / "stacked.dat", temps_hc)
    pairs = sweeps.split_isotherms(load(path))
    result = detect_transitions_all(pairs, outliers="ignore")
    assert len(result.curves) == len(temps_hc)
    for sweep_id, (t_k, h_c) in enumerate(temps_hc.items()):
        assert pairs[sweep_id][0] == pytest.approx(t_k, abs=0.01)
        curve = result.curve(sweep_id, "moment_emu")
        steepest = curve["x"][np.argmax(np.abs(curve["dydx"]))]
        assert steepest == pytest.approx(h_c, abs=0.1)

# Per-sweep smooth_window in the batch (Phase B)


def _ac_tscan_file_n(path, n_points, *, t_n=14.0, fixed_h_oe=100.0, rng):
    """Same one-cusp chi'(T) shape as :func:`_ac_tscan_file`, with a
    caller-chosen point count -- the batch smooth_window tests need two
    sweeps of the same family at different densities, the way a campaign
    can contain after measurements at different sampling densities."""
    t = np.linspace(4.0, 30.0, n_points)
    chi = simulate.synthetic_chi_peak(
        t, centre=t_n, width=1.0, height=2.0, background=0.1, noise=0.02, rng=rng
    )
    return simulate.write_mpms3_file(
        path,
        app="SQUID AC",
        temperature_K=t,
        field_oe=np.full(n_points, fixed_h_oe),
        chi_real=chi,
        ac_frequency_hz=np.full(n_points, 511.0),
    )


def test_batch_smooth_window_callable_resolves_per_sweep_density(tmp_path, rng):
    """A campaign whose sweeps do not share a point density -- a dense
    271-point run and a sparse 137-point run of the same family -- needs a
    different window for each; one physical-units value for the whole
    batch is wrong for one of them by construction (the motivating defect
    for this parameter)."""
    dense = load(_ac_tscan_file_n(tmp_path / "dense.dat", 271, rng=rng))
    sparse = load(_ac_tscan_file_n(tmp_path / "sparse.dat", 137, rng=rng))

    def window_by_density(m):
        return 1.0 if len(m.temperature.dropna()) > 200 else 3.0

    result = detect_transitions_all([dense, sparse], smooth_window=window_by_density)
    assert result.skipped == ()
    per_sweep = result.params["per_sweep"]
    assert [row["measurement_file"] for row in per_sweep] == [
        str(dense.path),
        str(sparse.path),
    ]
    # The value actually passed to detect_transitions for each sweep.
    assert per_sweep[0]["smooth_window"] == pytest.approx(1.0)
    assert per_sweep[1]["smooth_window"] == pytest.approx(3.0)
    dense_resolved = per_sweep[0]["per_signal"]["chi_real"]["smooth_window"]
    sparse_resolved = per_sweep[1]["per_signal"]["chi_real"]["smooth_window"]
    # The resolved value is the requested window snapped to an odd point
    # count times the sweep's own grid spacing, so compare by ordering
    # rather than exact equality to the requested 1.0/3.0 K.
    assert dense_resolved < sparse_resolved
    # And it matches what detect_transitions on the same sweep would record.
    direct = detect_transitions(dense, smooth_window=1.0)
    assert dense_resolved == pytest.approx(direct.params["per_signal"]["chi_real"]["smooth_window"])


def test_batch_smooth_window_float_still_forwarded_and_recorded(tmp_path, rng):
    """A plain float keeps working exactly as before, and is recorded in
    per_sweep and detect_kwargs like any other resolved value."""
    m = load(_ac_tscan_file_n(tmp_path / "ac.dat", 131, rng=rng))
    result = detect_transitions_all([m], smooth_window=2.5)
    per_sweep = result.params["per_sweep"]
    assert len(per_sweep) == 1
    assert per_sweep[0]["measurement_file"] == str(m.path)
    assert per_sweep[0]["smooth_window"] == pytest.approx(2.5)
    assert result.params["detect_kwargs"]["smooth_window"] == pytest.approx(2.5)


def test_batch_smooth_window_callable_raising_lands_in_skipped(tmp_path, rng):
    m = load(_ac_tscan_file_n(tmp_path / "ac.dat", 131, rng=rng))

    def broken_window(measurement):
        raise RuntimeError("no window for you")

    result = detect_transitions_all([m], smooth_window=broken_window)
    assert result.params["per_sweep"] == []
    assert len(result.skipped) == 1
    path, reason = result.skipped[0]
    assert path == str(m.path)
    assert "smooth_window" in reason
    assert "broken_window" in reason


def test_batch_smooth_window_callable_invalid_return_lands_in_skipped(tmp_path, rng):
    m = load(_ac_tscan_file_n(tmp_path / "ac.dat", 131, rng=rng))

    def negative_window(measurement):
        return -1.0

    result = detect_transitions_all([m], smooth_window=negative_window)
    assert result.params["per_sweep"] == []
    assert len(result.skipped) == 1
    assert "smooth_window" in result.skipped[0][1]


def test_batch_smooth_window_callable_name_recorded_in_detect_kwargs(tmp_path, rng):
    """detect_kwargs must stay printable: the callable object itself never
    appears there, only its name."""
    m = load(_ac_tscan_file_n(tmp_path / "ac.dat", 131, rng=rng))

    def window_by_density(measurement):
        return 2.0

    result = detect_transitions_all([m], smooth_window=window_by_density)
    assert result.params["detect_kwargs"]["smooth_window"] == (
        f"{__name__}.{window_by_density.__qualname__}"
    )


def test_noise_auto_warns_on_inflated_instrument_errors(tmp_path, rng):
    """An error column ~100x the actual scatter (the observed real-file
    defect) makes noise='auto' warn, naming the fix."""
    m = load(_dc_hscan_file(tmp_path / "dc.dat", err_scale=0.5, rng=rng))  # scatter is ~0.005
    with pytest.warns(UserWarning, match="noise='scatter'") as record:
        result = detect_transitions(m)
    assert result.params["per_signal"]["moment_emu"]["noise_source"] == "instrument"
    assert result.params["per_signal"]["moment_emu"]["noise_ratio"] > transitions.NOISE_RATIO_WARN
    assert any("noise" in str(w.message) for w in record)


def test_noise_auto_does_not_warn_on_honest_errors(tmp_path, rng):
    """An error column close to the real scatter (the default fixture,
    ratio ~1) must not warn -- simplefilter('error') turns any stray
    warning into a test failure."""
    m = load(_dc_hscan_file(tmp_path / "dc.dat", rng=rng))  # default err_scale == scatter scale
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = detect_transitions(m)
    assert result.params["per_signal"]["moment_emu"]["noise_source"] == "instrument"


def test_noise_batch_forwards_override(tmp_path, rng):
    """detect_transitions_all forwards noise= through **kwargs to
    detect_transitions -- checked by its effect (the noise='auto' warning
    disappearing under noise='scatter'), not just by the recorded kwarg."""
    path = _dc_hscan_file(tmp_path / "dc.dat", err_scale=0.5, rng=rng)
    m = load(path)

    with pytest.warns(UserWarning, match="noise='scatter'"):
        auto_result = detect_transitions_all([m])
    assert "noise" not in auto_result.params["detect_kwargs"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scatter_result = detect_transitions_all([m], noise="scatter")
    assert scatter_result.params["detect_kwargs"]["noise"] == "scatter"
    assert not scatter_result.candidates.empty


def test_detect_measurement_accepts_kink_opt_in(tmp_path, rng):
    """The measurement-level path: 'kink' is opt-in (no scan-kind default
    changes) and its rows carry feature == 'kink'."""
    m = load(_ac_tscan_file(tmp_path / "ac.dat", rng=rng))
    default = detect_transitions(m)
    assert "kink" not in default.params["per_signal"]["chi_real"]["methods"]

    result = detect_transitions(m, methods=("peak", "derivative", "kink"))
    assert result.params["per_signal"]["chi_real"]["methods"] == (
        "peak", "derivative", "kink",
    )
    kept = result.robust()
    row = kept.candidates.iloc[0]
    assert row["T_K"] == pytest.approx(14.0, abs=0.2)
    assert row["method"] == "peak"  # the cusp still outranks its companions
    kinks = kept.detections[kept.detections["method"] == "kink"]
    assert not kinks.empty
    assert set(kinks["feature"]) == {"kink"}
    assert set(kinks["sign"]) <= {"up", "down"}
    assert list(kept.detections.columns) == DETECTION_COLUMNS
    assert (str(m.path), "chi_real") in kept.curves
    assert "d2ydx2" in kept.curves[(str(m.path), "chi_real")]


def test_batch_link_and_robust_both_recorded(tmp_path, rng):
    """detect_transitions_all applies link then robust in one call, and
    records both -- the order matters, since a chain gate needs chains."""
    files = [
        _ac_tscan_file(tmp_path / "a.dat", fixed_h_oe=100.0, t_n=14.0, rng=rng),
        _ac_tscan_file(tmp_path / "b.dat", fixed_h_oe=2000.0, t_n=13.8, rng=rng),
    ]
    result = detect_transitions_all(
        [load(f) for f in files],
        link={"t_tol_k": 0.5},
        robust={"min_significance": 5.0, "min_chain_n": 2},
    )
    assert result.params["link"]["t_tol_k"] == 0.5
    assert result.params["link"]["statistic"] == "stouffer"
    assert result.params["robust"]["min_chain_n"] == 2
    assert len(result.candidates)
    assert (result.candidates["chain_n"] >= 2).all()


def test_single_sweep_links_to_singletons(tmp_path, rng):
    """One sweep is a family of one: linking is allowed (it is not a
    refusal case) and every candidate is a chain of one."""
    m = load(_ac_tscan_file(tmp_path / "one.dat", rng=rng))
    linked = detect_transitions(m).link(t_tol_k=0.5)
    assert (linked.candidates["chain_n"] == 1).all()
    assert set(linked.chains["n"]) == {1}


def test_detect_transitions_stamps_sweep_id_zero(tmp_path, rng):
    result = detect_transitions(load(_ac_tscan_file(tmp_path / "ac.dat", rng=rng)))
    assert CANDIDATE_COLUMNS[-1] == "sweep_id"
    assert DETECTION_COLUMNS[-1] == "sweep_id"
    assert set(result.candidates["sweep_id"]) == {0}
    assert set(result.detections["sweep_id"]) == {0}


def test_sweep_id_is_stable_when_a_sweep_is_skipped(tmp_path, rng):
    """Ids are input positions over every entry, so a failed sweep leaves a
    gap rather than renumbering the sweeps after it."""
    first = load(_ac_tscan_file(tmp_path / "a.dat", fixed_h_oe=100.0, rng=rng))
    bad = load(_ac_tscan_file(tmp_path / "bad.dat", frequencies=(7.0, 511.0), rng=rng))
    third = load(_ac_tscan_file(tmp_path / "c.dat", fixed_h_oe=2000.0, rng=rng))
    result = detect_transitions_all([first, bad, third])
    assert result.params["skipped_sweep_ids"] == [1]
    assert result.skipped[0][0] == str(bad.path)  # (path, reason) unchanged
    assert len(result.skipped[0]) == 2
    assert [e["sweep_id"] for e in result.params["per_sweep"]] == [0, 2]
    assert set(result.candidates["sweep_id"]) == {0, 2}
    by_file = result.candidates.groupby("measurement_file")["sweep_id"].unique()
    assert list(by_file[str(third.path)]) == [2]
    assert set(result.detections["sweep_id"]) == {0, 2}


def test_fisher_is_never_a_default(tmp_path):
    m = load(_toy_afm_ac_file(tmp_path / "afm.dat"))
    result = detect_transitions(m)
    assert "fisher" not in result.params["per_signal"]["chi_real"]["methods"]
    assert "fisher" not in set(result.candidates["method"])
    assert "fisher_noise" not in result.params["per_signal"]["chi_real"]
    assert "yT" not in result.curves[(str(m.path), "chi_real")]


def test_fisher_sits_below_the_cusp_and_both_are_reported(tmp_path):
    """The toy AFM's cusp at T_N and its Fisher maximum are two candidates,
    not one: the Fisher maximum lies below the cusp, visibly, because the
    two are never grouped."""
    m = load(_toy_afm_ac_file(tmp_path / "afm.dat"))
    result = detect_transitions(m, methods=("peak", "fisher"))
    # Sub-threshold noise wiggles of d(chi T)/dT are candidates too (this
    # sweep has one at significance ~1 just above T_N); robust() is what
    # says which features are real, so count there.
    kept = result.robust().candidates
    cusps = kept[(kept["method"] == "peak") & (kept["sign"] == "max")]
    fishers = kept[(kept["method"] == "fisher") & (kept["sign"] == "max")]
    assert len(cusps) == 1
    assert len(fishers) == 1  # exactly one Fisher maximum, and it is near T_N
    cusp, fisher = cusps.iloc[0], fishers.iloc[0]
    assert cusp["T_K"] == pytest.approx(FISHER_T_N, abs=0.2)
    # Below the cusp by more than the grid step: an offset, not jitter.
    assert FISHER_T_N - 2.0 < fisher["T_K"] < cusp["T_K"] - 0.4
    assert cusp["n_methods"] == 1 and fisher["n_methods"] == 1
    assert fisher["feature"] == "fisher"
    assert fisher["candidate_id"] != cusp["candidate_id"]

    # Each Fisher candidate is made of its own detection and nothing else.
    cands, dets = result.candidates, result.detections
    for cid in cands.loc[cands["method"] == "fisher", "candidate_id"]:
        members = dets[dets["candidate_id"] == cid]
        assert list(members["method"]) == ["fisher"]
        assert set(members["raw_strength_unit"]) == {"y·K/K"}


def test_fisher_refused_on_an_h_scan(tmp_path, rng):
    m = load(_dc_hscan_file(tmp_path / "mh.dat", rng=rng))
    with pytest.raises(ValueError, match="'fisher' needs a temperature sweep.*H-scan"):
        detect_transitions(m, methods=("derivative", "fisher"))


def test_fisher_refused_on_chi_imag(tmp_path):
    m = load(_toy_afm_ac_file(tmp_path / "afm.dat", chi_imag=True))
    with pytest.raises(ValueError, match="'fisher' was asked for on chi_imag"):
        detect_transitions(m, signal="chi_imag", methods=("peak", "fisher"))
    with pytest.raises(ValueError, match="chi_imag"):
        detect_transitions(m, signal="all", methods=("peak", "fisher"))
    # chi' of the same file is fine.
    ok = detect_transitions(m, signal="chi_real", methods=("peak", "fisher"))
    assert "fisher" in set(ok.candidates["method"])


@pytest.mark.parametrize("field_oe", [0.0, 5.0, -10.0])
def test_fisher_refused_on_the_moment_at_zero_held_field(tmp_path, field_oe):
    m = load(_toy_afm_dc_file(tmp_path / "mt.dat", field_oe=field_oe))
    with pytest.raises(ValueError, match="needs a held, non-zero field"):
        detect_transitions(m, methods=("peak", "fisher"))
    # The refusal is Fisher's alone: the same sweep searches without it.
    detect_transitions(m, methods="peak")


def test_fisher_on_a_dc_moment_at_nonzero_field(tmp_path):
    """M*T = H*(chi*T) at a held field, so the moment finds the same Fisher
    temperature as chi' does: a constant positive scale changes nothing."""
    field_oe = 1000.0
    dc = detect_transitions(
        load(_toy_afm_dc_file(tmp_path / "mt.dat", field_oe=field_oe)),
        methods=("peak", "fisher"),
    )
    t = _toy_afm_t()
    chi = _toy_afm_chi(t) + np.random.default_rng(13).normal(scale=FISHER_NOISE, size=t.shape)
    ref = find_transitions(t, chi, methods=("peak", "fisher"), x_name="T", x_unit="K")
    dc_f = dc.candidates[dc.candidates["method"] == "fisher"].reset_index(drop=True)
    ref_f = ref.candidates[ref.candidates["method"] == "fisher"].reset_index(drop=True)
    assert dc_f["sign"].tolist() == ref_f["sign"].tolist()
    np.testing.assert_allclose(dc_f["T_K"], ref_f["T_K"], rtol=1e-9)
    np.testing.assert_allclose(dc_f["significance"], ref_f["significance"], rtol=1e-6)
    assert dc_f["signal"].eq("moment_emu").all()
    assert dc.params["per_signal"]["moment_emu"]["fisher_noise"]["noise_sigma"] == pytest.approx(
        ref.params["fisher_noise"]["noise_sigma"] * field_oe, rel=1e-6
    )


def test_fisher_provenance_cites_fisher(tmp_path):
    m = load(_toy_afm_ac_file(tmp_path / "afm.dat"))
    prov = detect_transitions(m, methods=("peak", "fisher")).provenance()
    fisher = prov[prov["method"] == "fisher"]
    assert not fisher.empty
    assert (fisher["method_name"] == "Fisher specific-heat criterion").all()
    assert fisher["citations"].str.contains("M. E. Fisher, Philos. Mag. 7, 1731").all()
    assert fisher["citations"].str.contains("Savitzky").all()
    assert not prov.loc[prov["method"] == "peak", "citations"].str.contains("Fisher").any()


def test_fisher_curves_hold_y_times_t_and_its_derivative(tmp_path):
    m = load(_toy_afm_ac_file(tmp_path / "afm.dat"))
    result = detect_transitions(m, methods=("peak", "fisher"))
    curve = result.curves[(str(m.path), "chi_real")]
    t, chi = m.temperature.to_numpy(dtype=float), m.chi_real.to_numpy(dtype=float)
    window = result.params["per_signal"]["chi_real"]["smooth_window"]
    x, z_smooth, dz = transitions.smooth_derivative(t, t * chi, smooth_window=window)
    np.testing.assert_allclose(curve["x"], x)
    np.testing.assert_allclose(curve["yT"], z_smooth, rtol=1e-12)
    np.testing.assert_allclose(curve["dyT_dT"], dz, rtol=1e-12)
    # The Fisher maximum is that curve's peak near T_N -- within half a
    # window, since the location is a local quadratic's vertex and this peak
    # is lopsided (steep on the T_N side), which pulls the vertex below it.
    fisher = result.robust().candidates.query("method == 'fisher' and sign == 'max'")
    near = (x > FISHER_T_N - 3) & (x < FISHER_T_N + 3)
    t_max = x[near][np.argmax(dz[near])]
    assert abs(float(fisher["T_K"].iloc[0]) - t_max) <= window / 2


def test_fisher_sign_vocabulary_in_select(tmp_path):
    m = load(_toy_afm_ac_file(tmp_path / "afm.dat"))
    result = detect_transitions(m, methods=("peak", "fisher"))
    with pytest.raises(ValueError, match=r"sign 'rise' is not in the 'fisher' vocabulary"):
        result.select(swept_axis="T", method="fisher", sign="rise")


@pytest.mark.parametrize("options", [
    {"methods": ("typo",)}, {"methods": ()}, {"criterion": "fisher"},
    {"noise": "typo"}, {"noise": 0}, {"noise": np.nan},
    {"smooth_window": -1}, {"smooth_window": np.inf}, {"smooth_window": 1+0j},
    {"polyorder": -1}, {"polyorder": 1.5}, {"prominence_fraction": np.nan},
    {"signal": "typo"}, {"branch": "typo"}, {"top": -1},
    {"frequency": np.nan}, {"frequency": 1+0j}, {"prominence_fraction": 1+0j},
    {"t_window_k": (5, 2)}, {"h_window_oe": (0,)},
])
@pytest.mark.parametrize("populated", [False, True])
def test_batch_invalid_shared_settings_raise_before_consuming_inputs(tmp_path, options, populated, rng):
    from calamari import CalamariError

    measurements = [load(_ac_tscan_file(tmp_path / "ac.dat", rng=rng))] if populated else []
    consumed = []

    def inputs():
        consumed.append(True)
        yield from measurements

    with pytest.raises(CalamariError) as raised:
        detect_transitions_all(inputs(), outliers="ignore", **options)
    assert isinstance(raised.value, ValueError)
    assert consumed == []
    if populated:
        with pytest.raises(type(raised.value)):
            detect_transitions(measurements[0], outliers="ignore", **options)


def test_batch_data_and_callable_failures_keep_successful_sweeps(tmp_path, rng):
    good = load(_ac_tscan_file(tmp_path / "good.dat", rng=rng))
    bad = load(_ac_tscan_file(tmp_path / "bad.dat", frequencies=(7.0, 511.0), rng=rng))
    callback_calls = []

    def window(measurement):
        callback_calls.append(measurement)
        if len(callback_calls) == 3:
            raise RuntimeError("no window for this sweep")
        return 1.0

    result = detect_transitions_all([good, bad, good], smooth_window=window, outliers="ignore")
    assert callback_calls == [good, bad, good]
    assert result.params["n_searched"] == 1
    assert result.params["skipped_sweep_ids"] == [1, 2]
    assert "frequencies" in result.skipped[0][1]
    assert "smooth_window callable" in result.skipped[1][1]
    assert set(result.candidates["sweep_id"]) == {0}


def test_batch_instrument_noise_unavailable_remains_a_sweep_failure(tmp_path, rng):
    good = load(_dc_hscan_file(tmp_path / "good.dat", with_errors=True, rng=rng))
    bad = load(_dc_hscan_file(tmp_path / "bad.dat", with_errors=False, rng=rng))
    result = detect_transitions_all([bad, good], noise="instrument", outliers="ignore")
    assert result.params["n_searched"] == 1
    assert result.params["skipped_sweep_ids"] == [0]
    assert "no usable per-point errors" in result.skipped[0][1]
    assert set(result.candidates["sweep_id"]) == {1}


def test_empty_batch_rejects_unknown_keyword():
    with pytest.raises(TypeError, match="unexpected keyword"):
        detect_transitions_all([], method="peak")


def test_detection_accepts_channel_sequences_and_open_ended_windows(tmp_path, rng):
    m = load(_ac_tscan_file(tmp_path / "ac.dat", rng=rng))
    direct = detect_transitions(m, signal=np.array(["chi_real"]), outliers="ignore")
    batch = detect_transitions_all(
        [m], signal=np.array(["chi_real"]), t_window_k=(-np.inf, np.inf), outliers="ignore"
    )
    pd.testing.assert_frame_equal(direct.candidates, batch.candidates.drop(columns="probe"))
