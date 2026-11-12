"""Candidate views, linking, selection, reconciliation and result provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calamari import simulate, sweeps, transitions
from calamari.io import load
from calamari.transitions import (
    CANDIDATE_COLUMNS,
    DETECTION_COLUMNS,
    METHODS,
    TransitionResult,
    detect_transitions,
    detect_transitions_all,
    find_transitions,
)

from ._helpers import (
    _ac_tscan_file,
    _curve_rng,
    _cusp,
    _kinked,
    _toy_afm_ac_file,
    _toy_afm_dc_file,
)


def test_robust_records_thresholds_and_all_restores(rng):
    x = np.linspace(5.0, 20.0, 151)
    result = find_transitions(
        x, _cusp(x, rng=rng), methods=("peak", "derivative"), x_name="T", x_unit="K"
    )
    tight = result.robust(min_significance=10.0)
    assert tight.params["robust"]["min_significance"] == 10.0
    # Loosening after tightening works: robust() always filters from the
    # full detection, and .all() recovers it.
    loose = tight.robust(min_significance=0.0, exclude_edge=False)
    assert len(loose.candidates) == len(result.candidates)
    assert len(tight.all().candidates) == len(result.candidates)
    assert "robust" not in tight.all().params


def test_robust_refuses_threshold_on_all_nan_statistic():
    """A threshold whose statistic is NaN on every row would silently drop
    everything; it raises instead."""
    cands = pd.DataFrame(
        [{"candidate_id": 0, "significance": np.nan, "n_methods": 1, "edge": False,
          "sweep_range_over_noise": np.nan}]
    )
    dets = pd.DataFrame([{"candidate_id": 0}])
    result = TransitionResult(candidates=cands, detections=dets, params={})
    with pytest.raises(ValueError, match="NaN"):
        result.robust()


def test_provenance_joins_citations(rng):
    x = np.linspace(0.0, 10.0, 101)
    result = find_transitions(x, _cusp(x, centre=5.0, rng=rng), methods="peak")
    prov = result.provenance()
    assert "Savitzky" in prov["citations"].iloc[0]


def test_curve_on_single_sweep_and_array_results(tmp_path, rng):
    """One accessor whatever produced the result: a single sweep is sweep 0,
    and its (measurement_file, signal) keys are unchanged."""
    m = load(_ac_tscan_file(tmp_path / "ac.dat", rng=_curve_rng()))
    result = detect_transitions(m, outliers="ignore")
    stored = result.curves[(str(m.path), "chi_real")]
    for got in (result.curve(), result.curve(0, "chi_real"), result.curve(np.int64(0))):
        assert got.keys() == stored.keys()
        for name in stored:
            assert got[name] is stored[name]
    with pytest.raises(ValueError, match=r"single-sweep result.*sweep_id 0"):
        result.curve(1)

    x = np.linspace(5.0, 20.0, 151)
    array = find_transitions(x, _kinked(x, rng=rng), methods="derivative")
    np.testing.assert_array_equal(array.curve()["d2ydx2"], array.curves[("", "")]["d2ydx2"])


def test_curve_survives_filters_and_link(tmp_path):
    """robust(), select(), link() and all() rebuild the result; each must
    carry the curves, and curve() must still find every sweep."""
    files = [
        _ac_tscan_file(tmp_path / "a.dat", fixed_h_oe=100.0, t_n=14.0, rng=_curve_rng()),
        _ac_tscan_file(tmp_path / "b.dat", fixed_h_oe=2000.0, t_n=12.0, rng=_curve_rng()),
    ]
    result = detect_transitions_all([load(f) for f in files], outliers="ignore")
    views = {
        "robust": result.robust(),
        "select": result.select(swept_axis="T", method="peak", sign="max", t_window_k=(8, 18)),
        "link": result.link(t_tol_k=3.0),
        "link+robust": result.link(t_tol_k=3.0).robust(min_chain_n=2),
        "all": result.robust().select(
            swept_axis="T", method="peak", sign="max", t_window_k=(8, 18)
        ).all(),
    }
    for name, view in views.items():
        for sweep_id in (0, 1):
            np.testing.assert_array_equal(
                view.curve(sweep_id)["y_smooth"],
                result.curve(sweep_id)["y_smooth"],
                err_msg=name,
            )


def test_curve_names_the_choices_when_sweep_or_signal_is_wrong(tmp_path):
    good = load(_ac_tscan_file(tmp_path / "good.dat", rng=_curve_rng()))
    bad = load(_ac_tscan_file(tmp_path / "bad.dat", frequencies=(7.0, 511.0), rng=_curve_rng()))
    result = detect_transitions_all([good, bad, good], outliers="ignore")
    assert result.params["skipped_sweep_ids"] == [1]

    with pytest.raises(ValueError, match=r"sweep_id=1 was skipped.*frequencies") as info:
        result.curve(1)
    assert "[0, 2]" in str(info.value)
    with pytest.raises(ValueError, match=r"no sweep_id=5.*\[0, 2\].*skipped: \[1\]"):
        result.curve(5)
    with pytest.raises(ValueError, match=r"signal='chi_imag'.*\['chi_real'\]"):
        result.curve(0, "chi_imag")
    with pytest.raises(ValueError, match="sweep_id must be an integer"):
        result.curve("0")

    # Two signals in one sweep: signal=None cannot choose between them.
    both = load(_toy_afm_ac_file(tmp_path / "both.dat", chi_imag=True))
    single = detect_transitions(both, signal="all", outliers="ignore")
    batch = detect_transitions_all([both], signal="all", outliers="ignore")
    for ambiguous in (single, batch):
        with pytest.raises(ValueError, match=r"\['chi_real', 'chi_imag'\].*signal="):
            ambiguous.curve()
        assert ambiguous.curve(0, "chi_imag").keys() >= {"x", "y_smooth", "dydx", "d2ydx2"}

# Cross-sweep linking

#: The synthetic campaign: ten chi'(T) sweeps at fixed fields from 0.01 T
#: to 1 T, each carrying a strong cusp at a fixed temperature plus a weak
#: feature whose centre drifts linearly with the field. The weak feature's
#: height is tuned so its per-sweep significance lands around 2-3 -- below
#: robust()'s default floor of 5 in every sweep, which is exactly the
#: defect linking exists to answer.
LINK_FIELDS_OE = np.linspace(100.0, 10_000.0, 10)

LINK_STRONG_T_K = 8.0

LINK_WEAK_HEIGHT = 0.045

#: Larger than the 0.33 K the weak feature drifts between adjacent sweeps,
#: small against the sweep's 26 K span.
LINK_TOL_K = 0.8


def _link_weak_centre_k(field_oe):
    """Planted centre of the weak feature at a given fixed field."""
    return 22.0 - 3.0e-4 * np.asarray(field_oe, dtype=float)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """The unfiltered, unlinked result for the synthetic campaign."""
    rng = np.random.default_rng(20260828)
    tmp_path = tmp_path_factory.mktemp("linking")
    t = np.linspace(4.0, 30.0, 261)
    measurements = []
    for i, h_oe in enumerate(LINK_FIELDS_OE):
        chi = (
            0.1
            + simulate.synthetic_chi_peak(t, centre=LINK_STRONG_T_K, width=1.0, height=1.0)
            + simulate.synthetic_chi_peak(
                t, centre=_link_weak_centre_k(h_oe), width=0.8, height=LINK_WEAK_HEIGHT
            )
            + rng.normal(scale=0.008, size=t.shape)
        )
        path = simulate.write_mpms3_file(
            tmp_path / f"link_{i}.dat",
            app="SQUID AC",
            temperature_K=t,
            field_oe=np.full_like(t, h_oe),
            chi_real=chi,
            ac_frequency_hz=np.full_like(t, 511.0),
        )
        measurements.append(load(path))
    return detect_transitions_all(measurements)


def _near_planted(row) -> bool:
    """Is this candidate one of the two planted features in its sweep?"""
    weak = _link_weak_centre_k(row["H_T"] * 1e4)
    return (
        abs(row["T_K"] - LINK_STRONG_T_K) <= LINK_TOL_K
        or abs(row["T_K"] - weak) <= LINK_TOL_K
    )


def test_weak_feature_is_planted_below_the_per_sweep_floor(campaign):
    """The premise: the weak feature is found in every sweep, at 2-4
    sigma, and so is lost by robust() in most of them -- a per-sweep
    statistic cannot see that it recurs."""
    cands = campaign.candidates
    for h_oe in LINK_FIELDS_OE:
        sweep = cands[np.isclose(cands["H_T"], h_oe / 1e4)]
        near = sweep[(sweep["T_K"] - _link_weak_centre_k(h_oe)).abs() <= 0.5]
        assert len(near) == 1, h_oe
        assert 2.0 <= float(near["significance"].iloc[0]) <= 4.0

    kept = campaign.robust().candidates
    survived = sum(
        bool(
            len(
                kept[
                    np.isclose(kept["H_T"], h_oe / 1e4)
                    & ((kept["T_K"] - _link_weak_centre_k(h_oe)).abs() <= 0.5)
                ]
            )
        )
        for h_oe in LINK_FIELDS_OE
    )
    assert survived <= 2  # lost in at least 80% of the sweeps


def test_linked_chain_recovers_the_weak_boundary(campaign):
    """Combining the recurrence publishes the weak boundary: one chain
    covers at least 80% of the sweeps, its members sit on the planted
    positions, and the strong cusp chains across every sweep."""
    linked = campaign.link(t_tol_k=LINK_TOL_K)
    kept = linked.robust(min_significance=1.0, min_chain_significance=8.0)

    weak_rows = kept.candidates[
        (kept.candidates["T_K"] - _link_weak_centre_k(kept.candidates["H_T"] * 1e4)).abs()
        <= LINK_TOL_K
    ]
    assert weak_rows["chain_id"].nunique() == 1
    assert len(weak_rows) >= 0.8 * len(LINK_FIELDS_OE)
    for _, row in weak_rows.iterrows():
        planted = _link_weak_centre_k(row["H_T"] * 1e4)
        assert abs(row["T_K"] - planted) <= LINK_TOL_K

    strong_rows = kept.candidates[
        (kept.candidates["T_K"] - LINK_STRONG_T_K).abs() <= LINK_TOL_K
    ]
    assert strong_rows["chain_id"].nunique() == 1
    assert set(strong_rows["chain_n"]) == {len(LINK_FIELDS_OE)}


def test_no_chance_chain_survives_the_chain_threshold(campaign):
    """The statistic, not the length, does the discriminating: this family
    is dense with noise candidates that chain by chance, and none of those
    chains clears the threshold the planted ones do."""
    linked = campaign.link(t_tol_k=LINK_TOL_K)
    kept = linked.robust(min_significance=1.0, min_chain_significance=8.0)
    assert len(kept.candidates)
    assert kept.candidates.apply(_near_planted, axis=1).all()
    # And chance chains do exist to be rejected, so the assertion above is
    # not passing vacuously.
    chance = linked.chains[
        (linked.chains["n"] >= 2) & (linked.chains["chain_significance"] < 8.0)
    ]
    assert len(chance) >= 1


def test_link_is_deterministic(campaign):
    """TR-1: two link() calls on the same result agree exactly."""
    first = campaign.link(t_tol_k=LINK_TOL_K)
    second = campaign.link(t_tol_k=LINK_TOL_K)
    pd.testing.assert_frame_equal(first.candidates, second.candidates)
    pd.testing.assert_frame_equal(first.chains, second.chains)


def test_singletons_carry_their_own_significance(campaign):
    """A candidate that links to nothing is a chain of one, and Stouffer's
    combination of one value is that value -- so the chain columns never
    flatter an isolated candidate."""
    linked = campaign.link(t_tol_k=LINK_TOL_K)
    singles = linked.candidates[linked.candidates["chain_n"] == 1]
    assert len(singles)
    assert singles["chain_significance"].to_numpy() == pytest.approx(
        singles["significance"].to_numpy()
    )
    assert (linked.candidates["chain_id"] >= 0).all()
    assert linked.candidates["chain_id"].nunique() == len(linked.chains)


def test_chain_significance_grows_like_root_n(campaign):
    """Stouffer's Z on the weak chain: n members of roughly equal
    significance combine to about sqrt(n) times one of them."""
    linked = campaign.link(t_tol_k=LINK_TOL_K)
    weak = linked.candidates[
        (linked.candidates["T_K"] - _link_weak_centre_k(linked.candidates["H_T"] * 1e4)).abs()
        <= 0.5
    ]
    chain = linked.chains[linked.chains["chain_id"] == weak["chain_id"].iloc[0]].iloc[0]
    expected = weak["significance"].sum() / np.sqrt(len(weak))
    assert chain["chain_significance"] == pytest.approx(expected)
    assert chain["chain_significance"] > np.sqrt(len(weak)) * 0.9 * weak["significance"].min()


def test_missing_tolerance_named_in_the_refusal(campaign):
    with pytest.raises(ValueError, match="t_tol_k"):
        campaign.link()
    with pytest.raises(ValueError, match="t_tol_k must be positive"):
        campaign.link(t_tol_k=0.0)


def test_link_refuses_unknown_by_column(campaign):
    with pytest.raises(ValueError, match="banana"):
        campaign.link(t_tol_k=LINK_TOL_K, by=("signal", "banana"))


def test_chain_threshold_on_unlinked_result_points_at_link(campaign):
    with pytest.raises(ValueError, match=r"link\("):
        campaign.robust(min_chain_significance=5.0)
    with pytest.raises(ValueError, match=r"link\("):
        campaign.robust(min_chain_n=2)


def test_link_after_robust_reapplies_thresholds_and_all_keeps_chains(campaign):
    """Linking and loosening commute: link() works from the full detection
    and puts the previous thresholds back, and all() recovers everything
    WITH its chains (link is a description, not a filter)."""
    filtered = campaign.robust(min_significance=10.0)
    linked = filtered.link(t_tol_k=LINK_TOL_K)
    assert linked.params["robust"]["min_significance"] == 10.0
    assert (linked.candidates["significance"] >= 10.0).all()
    # The chains were built from every candidate, not just the survivors:
    # linking a filtered result gives each survivor exactly the chain it
    # has when the whole detection is linked.
    reference = campaign.link(t_tol_k=LINK_TOL_K)
    merged = linked.candidates.merge(
        reference.candidates[["candidate_id", "chain_id", "chain_n", "chain_significance"]],
        on="candidate_id",
        suffixes=("", "_ref"),
    )
    assert len(merged) == len(linked.candidates)
    for column in ("chain_id", "chain_n", "chain_significance"):
        assert (merged[column] == merged[f"{column}_ref"]).all()

    restored = linked.all()
    assert "robust" not in restored.params
    assert len(restored.candidates) == len(campaign.candidates)
    for column in ("chain_id", "chain_n", "chain_significance"):
        assert column in restored.candidates.columns
    assert len(restored.chains) > len(linked.chains)


def test_link_records_its_settings(campaign):
    linked = campaign.link(t_tol_k=LINK_TOL_K)
    assert linked.params["link"] == {
        "t_tol_k": LINK_TOL_K,
        "h_tol_t": None,
        "by": ("signal", "method", "sign"),
        "statistic": "stouffer",
    }


def test_chains_table_columns_and_filtered_restriction(campaign):
    linked = campaign.link(t_tol_k=LINK_TOL_K)
    assert list(linked.chains.columns) == transitions.CHAIN_COLUMNS
    assert linked.chains["chain_id"].is_monotonic_increasing
    assert linked.chains["chain_significance"].is_monotonic_decreasing

    kept = linked.robust(min_significance=1.0, min_chain_significance=8.0)
    assert set(kept.chains["chain_id"]) == set(kept.candidates["chain_id"])
    assert len(kept.chains) < len(linked.chains)
    # The statistics are the link-time ones: they are what justified
    # keeping these rows, so filtering must not recompute them.
    merged = kept.chains.merge(linked.chains, on="chain_id", suffixes=("_kept", "_all"))
    assert (merged["chain_significance_kept"] == merged["chain_significance_all"]).all()


def test_link_refuses_array_level_result(rng):
    """An array-level find_transitions result has no (T, H) coordinates to
    order a family with, and says so rather than linking nothing."""
    x = np.linspace(5.0, 20.0, 151)
    result = find_transitions(x, _cusp(x, rng=rng), methods="peak", x_name="T", x_unit="K")
    with pytest.raises(ValueError, match="detect_transitions"):
        result.link(t_tol_k=0.5)


def test_link_of_empty_result_keeps_the_columns():
    """An empty result links to an empty result -- with the columns
    present, so downstream code does not branch on emptiness."""
    empty = TransitionResult(
        candidates=pd.DataFrame(columns=CANDIDATE_COLUMNS),
        detections=pd.DataFrame(columns=DETECTION_COLUMNS),
        params={},
    )
    linked = empty.link(t_tol_k=0.5)
    assert len(linked.candidates) == 0
    for column in ("chain_id", "chain_n", "chain_significance"):
        assert column in linked.candidates.columns
    assert list(linked.chains.columns) == transitions.CHAIN_COLUMNS
    assert len(linked.chains) == 0

# sweep_id, and one candidate per sweep inside a physical window (select).

#: Held fields of the T-scan family (Tesla). Its planted boundary drifts
#: 1.6 K between adjacent sweeps -- far enough that one narrow fixed window
#: cannot follow it.
SELECT_FIELDS_T = (0.2, 0.6, 1.0, 1.4)

#: A stronger cusp every T-scan also carries, outside the target window: the
#: highest-significance candidate in each sweep, and not the boundary.
SELECT_DECOY_T_K = 22.0

#: Isotherm temperatures (K) of the stacked M(H) file.
SELECT_TEMPS_K = (2.0, 4.0, 6.0)

#: A larger metamagnetic step every isotherm also carries, below the target
#: window -- again the strongest derivative candidate, and not the boundary.
SELECT_DECOY_H_T = 0.8

SELECT_T_WINDOW = (6.0, 16.0)

SELECT_H_WINDOW = (1.4, 3.2)

SELECT_T = {"swept_axis": "T", "method": "peak", "sign": "max", "t_window_k": SELECT_T_WINDOW}

SELECT_H = {
    "swept_axis": "H", "method": "derivative", "sign": "rise", "h_window_t": SELECT_H_WINDOW,
}


def _select_t_b(h_t):
    """Planted T-scan boundary (K) at held field ``h_t`` (Tesla)."""
    return 14.0 - 4.0 * h_t


def _select_h_c(t_k):
    """Planted isotherm critical field (Tesla) at ``t_k`` (K)."""
    return 3.0 - 0.2 * t_k


@pytest.fixture(scope="module")
def select_campaign(tmp_path_factory):
    """Two families, each sweep with the target and a stronger decoy.

    Four fixed-field M(T) scans (sweep ids 0-3), then one stacked M(H)
    file split into three isotherms (ids 4-6) that share its path.
    """
    tmp_path = tmp_path_factory.mktemp("select")
    rng = np.random.default_rng(5)
    t = np.linspace(4.0, 30.0, 131)
    tscans = []
    for i, h_t in enumerate(SELECT_FIELDS_T):
        moment = (
            0.1
            + simulate.synthetic_chi_peak(t, centre=_select_t_b(h_t), width=0.8, height=1.0)
            + simulate.synthetic_chi_peak(t, centre=SELECT_DECOY_T_K, width=0.8, height=3.0)
            + rng.normal(scale=0.01, size=t.shape)
        )
        path = simulate.write_mpms3_file(
            tmp_path / f"mt_{i}.dat", app="SQUID DC", temperature_K=t,
            field_oe=np.full_like(t, h_t * 1e4), dc_moment_emu=moment,
        )
        tscans.append(load(path))
    field_oe = np.linspace(0.0, 50_000.0, 101)
    temps, fields, moments = [], [], []
    for t_k in SELECT_TEMPS_K:
        h = field_oe / 1e4
        moment = (
            0.5 * (1 + np.tanh((h - _select_h_c(t_k)) / 0.1))
            + 1.5 * (1 + np.tanh((h - SELECT_DECOY_H_T) / 0.1))
            + rng.normal(scale=0.01, size=h.shape)
        )
        temps.append(np.full_like(field_oe, t_k))
        fields.append(field_oe)
        moments.append(moment)
    stacked = simulate.write_mpms3_file(
        tmp_path / "mh.dat", app="SQUID DC", temperature_K=np.concatenate(temps),
        field_oe=np.concatenate(fields), vsm_moment_emu=np.concatenate(moments),
    )
    isotherms = sweeps.split_isotherms(load(stacked))
    return detect_transitions_all(tscans + isotherms)


def test_campaign_premise_decoy_outranks_target_in_every_sweep(select_campaign):
    """Without select the strongest candidate of every sweep is the decoy,
    so picking by significance alone would draw the wrong boundary."""
    cands = select_campaign.candidates
    for sid, sweep in cands.groupby("sweep_id"):
        strongest = sweep.loc[sweep["significance"].idxmax()]
        if strongest["swept_axis"] == "T":
            assert strongest["T_K"] == pytest.approx(SELECT_DECOY_T_K, abs=0.2), sid
        else:
            assert strongest["H_T"] == pytest.approx(SELECT_DECOY_H_T, abs=0.1), sid


def test_select_picks_the_target_in_every_sweep(select_campaign):
    chosen = select_campaign.select(**SELECT_T).select(**SELECT_H)
    cands = chosen.candidates
    assert list(cands["sweep_id"]) == list(range(7))
    for _, row in cands.iterrows():
        if row["swept_axis"] == "T":
            assert row["method"] == "peak" and row["sign"] == "max"
            assert row["T_K"] == pytest.approx(_select_t_b(row["H_T"]), abs=0.2)
        else:
            assert row["method"] == "derivative" and row["sign"] == "rise"
            assert row["H_T"] == pytest.approx(_select_h_c(row["T_K"]), abs=0.1)
    # Detections follow their candidates, exactly as under robust().
    assert set(chosen.detections["candidate_id"]) == set(cands["candidate_id"])


def test_isotherms_sharing_one_path_stay_distinct_by_sweep_id(select_campaign):
    """split_isotherms sub-measurements share their file's path, so only
    sweep_id tells them apart -- and select keeps one per isotherm, not one
    per file."""
    h_rows = select_campaign.candidates[select_campaign.candidates["swept_axis"] == "H"]
    assert h_rows["measurement_file"].nunique() == 1
    assert sorted(h_rows["sweep_id"].unique()) == [4, 5, 6]
    chosen = select_campaign.select(**SELECT_H).candidates
    chosen_h = chosen[chosen["swept_axis"] == "H"]
    assert len(chosen_h) == len(SELECT_TEMPS_K)
    assert sorted(chosen_h["sweep_id"]) == [4, 5, 6]
    assert chosen_h["measurement_file"].nunique() == 1


def test_select_passes_the_other_axis_through(select_campaign):
    chosen = select_campaign.select(**SELECT_T)
    full = select_campaign.candidates
    pd.testing.assert_frame_equal(
        chosen.candidates[chosen.candidates["swept_axis"] == "H"].reset_index(drop=True),
        full[full["swept_axis"] == "H"].reset_index(drop=True),
    )
    assert (chosen.candidates["swept_axis"] == "T").sum() == len(SELECT_FIELDS_T)


def test_empty_sweeps_are_recorded_not_dropped(select_campaign):
    """A window that misses the boundary in three sweeps keeps one row and
    names the other three -- never a silent gap."""
    chosen = select_campaign.select(
        swept_axis="T", method="peak", sign="max", t_window_k=(12.5, 14.0)
    )
    record = chosen.params["select"]["T"]
    assert record["empty_sweeps"] == [1, 2, 3]
    assert sorted(record["windows"]) == [0, 1, 2, 3]
    t_rows = chosen.candidates[chosen.candidates["swept_axis"] == "T"]
    assert list(t_rows["sweep_id"]) == [0]


def test_callable_window_follows_a_drifting_boundary(select_campaign):
    """The boundary moves 1.6 K per sweep; a window centred on where you
    expect it, as a function of the held field, finds it in every sweep and
    records what it resolved to."""

    def around_boundary(h_t):
        centre = 14.0 - 4.0 * h_t
        return (centre - 1.0, centre + 1.0)

    chosen = select_campaign.select(
        swept_axis="T", method="peak", sign="max", t_window_k=around_boundary
    )
    record = chosen.params["select"]["T"]
    assert record["window"] == f"{__name__}.{around_boundary.__qualname__}"
    assert record["empty_sweeps"] == []
    for sid, h_t in enumerate(SELECT_FIELDS_T):
        lo, hi = record["windows"][sid]
        assert (lo + hi) / 2 == pytest.approx(_select_t_b(h_t))
    t_rows = chosen.candidates[chosen.candidates["swept_axis"] == "T"]
    assert len(t_rows) == len(SELECT_FIELDS_T)
    for _, row in t_rows.iterrows():
        assert row["T_K"] == pytest.approx(_select_t_b(row["H_T"]), abs=0.2)


def test_callable_window_receives_temperature_for_h_sweeps(select_campaign):
    seen = []

    def window(t_k):
        seen.append(t_k)
        return (_select_h_c(t_k) - 0.2, _select_h_c(t_k) + 0.2)

    chosen = select_campaign.select(
        swept_axis="H", method="derivative", sign="rise", h_window_t=window
    )
    assert seen == pytest.approx(list(SELECT_TEMPS_K), abs=0.01)
    assert chosen.params["select"]["H"]["empty_sweeps"] == []


def test_select_record_holds_plain_python_values(select_campaign):
    """No numpy scalars in params: their reprs differ across the CI matrix."""
    record = select_campaign.select(**SELECT_T).select(**SELECT_H).params["select"]
    for axis_record in record.values():
        for sid, window in axis_record["windows"].items():
            assert type(sid) is int
            assert all(type(b) is float for b in window)
        assert all(type(s) is int for s in axis_record["empty_sweeps"])
        assert all(type(b) is float for b in axis_record["window"])


def test_select_breaks_ties_by_location_then_id():
    """TR-1: equal significance picks the lower location, whatever the row
    order, and the choice is repeatable."""
    rows = [
        {"candidate_id": 0, "T_K": 9.0, "H_T": 0.5, "significance": 7.0},
        {"candidate_id": 1, "T_K": 8.0, "H_T": 0.5, "significance": 7.0},
        {"candidate_id": 2, "T_K": 8.5, "H_T": 0.5, "significance": 6.0},
    ]
    frame = pd.DataFrame(rows).assign(
        swept_axis="T", signal="chi_real", method="peak", sign="max", sweep_id=0,
        edge=False,
    ).reindex(columns=CANDIDATE_COLUMNS)
    dets = pd.DataFrame(columns=DETECTION_COLUMNS)
    for order in (frame, frame.iloc[::-1].reset_index(drop=True)):
        chosen = TransitionResult(candidates=order, detections=dets, params={}).select(
            swept_axis="T", method="peak", sign="max"
        )
        assert list(chosen.candidates["candidate_id"]) == [1]


def test_select_refuses_an_ambiguous_signal_by_name(tmp_path, rng):
    t = np.linspace(4.0, 30.0, 131)
    chi = simulate.synthetic_chi_peak(
        t, centre=14.0, width=1.0, height=2.0, background=0.1, noise=0.02, rng=rng
    )
    path = simulate.write_mpms3_file(
        tmp_path / "both.dat", app="SQUID AC", temperature_K=t,
        field_oe=np.full_like(t, 100.0), chi_real=chi, chi_imag=0.2 * chi,
        ac_frequency_hz=np.full_like(t, 511.0),
    )
    both = detect_transitions(load(path), signal="all")
    with pytest.raises(ValueError, match=r"'chi_imag', 'chi_real'.*signal="):
        both.select(swept_axis="T", method="peak", sign="max")
    chosen = both.select(swept_axis="T", method="peak", sign="max", signal="chi_real")
    assert set(chosen.candidates["signal"]) == {"chi_real"}
    assert chosen.params["select"]["T"]["signal"] == "chi_real"
    with pytest.raises(ValueError, match="moment_emu"):
        both.select(swept_axis="T", method="peak", sign="max", signal="moment_emu")


def test_select_refuses_the_other_axis_window(select_campaign):
    with pytest.raises(ValueError, match="h_window_t was given for swept_axis='T'"):
        select_campaign.select(swept_axis="T", method="peak", sign="max", h_window_t=(1, 2))
    with pytest.raises(ValueError, match="t_window_k was given for swept_axis='H'"):
        select_campaign.select(
            swept_axis="H", method="derivative", sign="rise", t_window_k=(1, 2)
        )


def test_select_requires_method_and_sign(select_campaign):
    with pytest.raises(TypeError):
        select_campaign.select(swept_axis="T", sign="max")
    with pytest.raises(TypeError):
        select_campaign.select(swept_axis="T", method="peak")
    with pytest.raises(ValueError, match="method and sign are required"):
        select_campaign.select(swept_axis="T", method=None, sign="max")
    with pytest.raises(ValueError, match="method and sign are required"):
        select_campaign.select(swept_axis="T", method="peak", sign=None)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"swept_axis": "X", "method": "peak", "sign": "max"}, "swept_axis must be"),
        ({"swept_axis": "T", "method": "wavelet", "sign": "max"}, "unknown method"),
        ({"swept_axis": "T", "method": "derivative", "sign": "max"}, "vocabulary"),
        ({**SELECT_T, "t_window_k": (16.0, 6.0)}, "lo < hi"),
        ({**SELECT_T, "t_window_k": (6.0, np.inf)}, "finite"),
        ({**SELECT_T, "t_window_k": 12.0}, r"not a \(lo, hi\) pair"),
        ({**SELECT_T, "t_window_k": lambda h: (1.0, 2.0, 3.0)}, "returned, for sweep 0"),
        ({**SELECT_T, "t_window_k": lambda h: 1 / 0}, "raised ZeroDivisionError"),
    ],
)
def test_select_validates_its_arguments(select_campaign, kwargs, match):
    with pytest.raises(ValueError, match=match):
        select_campaign.select(**kwargs)


def test_select_refuses_an_axis_with_no_candidates(tmp_path, rng):
    result = detect_transitions(load(_ac_tscan_file(tmp_path / "ac.dat", rng=rng)))
    with pytest.raises(ValueError, match=r"swept along H.*\['T'\]"):
        result.select(swept_axis="H", method="derivative", sign="rise")


def test_select_refuses_array_level_result(rng):
    x = np.linspace(5.0, 20.0, 151)
    result = find_transitions(x, _cusp(x, rng=rng), methods="peak", x_name="T", x_unit="K")
    with pytest.raises(ValueError, match="detect_transitions"):
        result.select(swept_axis="T", method="peak", sign="max")


def test_robust_then_select_equals_select_then_robust(select_campaign):
    """The thresholds decide which candidates exist; the selection picks
    among them -- in either call order. A threshold that removes sweep 3's
    target empties that sweep, and loosening it again refills it."""
    first = select_campaign.robust(min_significance=35.0).select(**SELECT_T)
    second = select_campaign.select(**SELECT_T).robust(min_significance=35.0)
    pd.testing.assert_frame_equal(first.candidates, second.candidates)
    pd.testing.assert_frame_equal(first.detections, second.detections)
    assert first.params["select"] == second.params["select"]
    assert first.params["robust"] == second.params["robust"]
    assert first.params["select"]["T"]["empty_sweeps"] == [3]

    loosened = first.robust()
    assert loosened.params["select"]["T"]["empty_sweeps"] == []
    t_rows = loosened.candidates[loosened.candidates["swept_axis"] == "T"]
    assert sorted(t_rows["sweep_id"]) == [0, 1, 2, 3]


def test_select_works_on_the_robust_view(select_campaign):
    """After robust(), select never resurrects a candidate the thresholds
    removed: the H rows it keeps all pass the threshold."""
    chosen = select_campaign.robust(min_significance=15.0).select(**SELECT_H)
    h_rows = chosen.candidates[chosen.candidates["swept_axis"] == "H"]
    assert (h_rows["significance"] >= 15.0).all()
    assert chosen.params["select"]["H"]["empty_sweeps"] == [4]


def test_link_then_select_equals_select_then_link(select_campaign):
    tolerances = {"t_tol_k": 2.0, "h_tol_t": 0.5}
    first = select_campaign.link(**tolerances).select(**SELECT_T)
    second = select_campaign.select(**SELECT_T).link(**tolerances)
    pd.testing.assert_frame_equal(first.candidates, second.candidates)
    pd.testing.assert_frame_equal(first.chains, second.chains)
    assert first.params["select"] == second.params["select"]
    assert first.params["link"] == second.params["link"]
    # A selected candidate keeps the chain it has when everything is linked,
    # and the chain table follows the survivors, statistics as at link time.
    reference = select_campaign.link(**tolerances)
    merged = first.candidates.merge(
        reference.candidates[["candidate_id", "chain_id", "chain_significance"]],
        on="candidate_id", suffixes=("", "_ref"),
    )
    assert (merged["chain_id"] == merged["chain_id_ref"]).all()
    assert set(first.chains["chain_id"]) == set(first.candidates["chain_id"])
    kept = first.chains.merge(reference.chains, on="chain_id", suffixes=("", "_all"))
    assert (kept["chain_significance"] == kept["chain_significance_all"]).all()


def test_all_undoes_robust_and_select_but_keeps_link(select_campaign):
    filtered = (
        select_campaign.link(t_tol_k=2.0, h_tol_t=0.5)
        .robust()
        .select(**SELECT_T)
        .select(**SELECT_H)
    )
    restored = filtered.all()
    assert "robust" not in restored.params
    assert "select" not in restored.params
    assert "link" in restored.params
    assert len(restored.candidates) == len(select_campaign.candidates)
    assert "chain_id" in restored.candidates.columns
    assert len(restored.chains) > len(filtered.chains)
    assert "selected" not in repr(restored)


def test_chained_selects_are_both_recorded(select_campaign):
    chosen = select_campaign.select(**SELECT_T).select(**SELECT_H)
    assert list(chosen.params["select"]) == ["T", "H"]
    assert chosen.params["select"]["T"]["window"] == SELECT_T_WINDOW
    assert chosen.params["select"]["H"]["window"] == SELECT_H_WINDOW
    assert repr(chosen) == "TransitionResult(7 candidates, 15 detections (selected on T and H))"
    both = chosen.robust()
    assert repr(both).endswith("(robust-filtered, selected on T and H))")
    # robust() re-applied both selections, not just the last one.
    assert list(both.params["select"]) == ["T", "H"]
    assert len(both.candidates) == 7


def test_selecting_an_axis_again_replaces_its_choice(select_campaign):
    """Like robust(), a second select on one axis starts from the
    unselected view, so a window can be widened after being narrowed."""
    narrow = select_campaign.select(
        swept_axis="T", method="peak", sign="max", t_window_k=(12.5, 14.0)
    )
    widened = narrow.select(**SELECT_T)
    direct = select_campaign.select(**SELECT_T)
    pd.testing.assert_frame_equal(widened.candidates, direct.candidates)
    assert widened.params["select"] == direct.params["select"]


def test_select_method_matches_the_reported_criterion(select_campaign):
    """A cusp's derivative flanks are grouped into its 'peak' row, so they
    are not offered to method='derivative': on these T-scans no candidate
    reports that criterion, and every sweep is recorded empty."""
    chosen = select_campaign.select(swept_axis="T", method="derivative", sign="rise")
    assert chosen.params["select"]["T"]["empty_sweeps"] == [0, 1, 2, 3]
    assert not (chosen.candidates["swept_axis"] == "T").any()

# One criterion across a cusp-to-kink crossover

#: Thirteen chi'(T) sweeps, 0 to 3 T in 0.25 T steps, each a continuous
#: piecewise-linear curve whose slope changes at a planted T_k(H) (rounded over
#: 0.3 K by a softplus). The slope below is +0.3 per K; the slope above runs from
#: -0.38 per K at 0 T and rises by 0.16 per K per T, so it passes through zero at
#: 0.38 / 0.16 = 2.375 T, between the 2.25 T and 2.5 T sweeps. Below that field
#: the curve has a maximum (a cusp); above it only the slope change is left.
CROSSOVER_FIELDS_T = np.arange(0.0, 3.01, 0.25)

CROSSOVER_H_T = 0.38 / 0.16

CROSSOVER_KW = {
    "methods": ("peak", "derivative", "kink"),
    "t_window_k": (3.0, 24.0),
    "smooth_window": 2.0,
    # The boundary moves by at most 0.5 * (3^2 - 2.75^2) = 0.72 K between
    # adjacent sweeps; 0.8 K admits that and is small against the 21 K window.
    "link": {"t_tol_k": 0.8},
}


def _crossover_t_k(h_t):
    """Planted slope change of the crossover family, in K."""
    return 13.0 - 0.5 * np.asarray(h_t, dtype=float) ** 2


@pytest.fixture(scope="module")
def crossover_family(tmp_path_factory):
    rng = np.random.default_rng(20260928)
    directory = tmp_path_factory.mktemp("crossover")
    t = np.linspace(2.0, 30.0, 141)
    measurements = []
    for i, h_t in enumerate(CROSSOVER_FIELDS_T):
        t_k = float(_crossover_t_k(h_t))
        above = -0.38 + 0.16 * h_t
        chi = (
            1.0
            + 0.3 * (t - t_k)
            + (above - 0.3) * 0.3 * np.logaddexp(0.0, (t - t_k) / 0.3)
            + rng.normal(scale=0.006, size=t.shape)
        )
        path = simulate.write_mpms3_file(
            directory / f"crossover_{i}.dat",
            app="SQUID AC",
            temperature_K=t,
            field_oe=np.full_like(t, h_t * 1.0e4),
            chi_real=chi,
            ac_frequency_hz=np.full_like(t, 511.0),
        )
        measurements.append(load(path))
    return measurements


def test_one_kink_chain_spans_a_cusp_to_kink_crossover(crossover_family):
    """Along a boundary where a cusp flattens into a one-sided kink, the peak
    criterion drifts above the slope change and then vanishes, so its chain
    stops at the crossover; the curvature criterion marks the slope change on
    both sides, so criterion='kink' traces the boundary as one chain.

    This pins current behaviour (it passes on the code it was written
    against): it guards the documented recipe in PHASE_DIAGRAMS.md, "One
    criterion across a crossover", against a grouping or linking change that
    would break it.

    The bounds are generous by construction. Over 40 seeds of this family the
    peak chain was always 0-2.25 T with its last member 0.97-1.20 K above the
    slope change; the kink chain always started at 0 T and ended at 2.5 T or
    beyond (11-13 members), its median |offset| was at most 0.043 K, its chain
    significance 22-26, and every other chain's at most 5.4.
    """
    by_peak = detect_transitions_all(crossover_family, **CROSSOVER_KW)
    cands = by_peak.candidates
    on_boundary = cands[(cands["T_K"] - _crossover_t_k(cands["H_T"])).abs() < 2.0]
    peaks = on_boundary[on_boundary["method"] == "peak"]
    # One peak chain on the boundary, from 0 T, ending below the crossover.
    assert peaks["chain_id"].nunique() == 1
    peak_chain = by_peak.chains.set_index("chain_id").loc[peaks["chain_id"].iloc[0]]
    assert peak_chain["H_T_min"] == 0.0
    assert peak_chain["H_T_max"] < CROSSOVER_H_T
    assert (peaks["H_T"] < CROSSOVER_H_T).all()
    # ... and before it ends the peak has drifted well off the slope change.
    last = peaks.sort_values("H_T").iloc[-1]
    assert last["T_K"] - _crossover_t_k(last["H_T"]) > 0.5

    by_kink = detect_transitions_all(crossover_family, criterion="kink", **CROSSOVER_KW)
    kinks = by_kink.chains[(by_kink.chains["method"] == "kink")
                           & (by_kink.chains["sign"] == "down")]
    chain = kinks.sort_values("n").iloc[-1]
    # One kink chain, from 0 T to beyond the crossover.
    assert chain["H_T_min"] == 0.0
    assert chain["H_T_max"] > CROSSOVER_H_T
    assert chain["n"] >= 11
    members = by_kink.candidates[by_kink.candidates["chain_id"] == chain["chain_id"]]
    assert (members["T_K"] - _crossover_t_k(members["H_T"])).abs().median() < 0.15
    # The documented chain gate keeps that chain (every member over the
    # per-sweep floor of 1) and nothing else.
    kept = by_kink.robust(min_significance=1.0, min_chain_significance=8.0)
    assert set(kept.candidates["chain_id"]) == {chain["chain_id"]}
    assert len(kept.candidates) == (members["significance"] >= 1.0).sum() >= 11

#: Held fields (Oe) and planted ordering temperatures (K) of the small
#: Fisher family: T_N falls by 0.5 K per Tesla, as a field suppresses order.
FISHER_FAMILY = ((2_000.0, 9.9), (10_000.0, 9.5), (20_000.0, 9.0))


def test_select_picks_the_fisher_maximum_per_sweep(tmp_path):
    """The three-call workflow's middle step: one Fisher maximum per sweep,
    inside the window, each below that sweep's own cusp."""
    ms = [
        load(_toy_afm_dc_file(tmp_path / f"mt_{i}.dat", field_oe=h, t_n=t_n, seed=40 + i))
        for i, (h, t_n) in enumerate(FISHER_FAMILY)
    ]
    batch = detect_transitions_all(ms, methods=("peak", "fisher"))
    assert batch.skipped == ()
    for entry in batch.params["per_sweep"]:  # the batch forwards the record
        assert entry["per_signal"]["moment_emu"]["fisher_noise"]["noise_source"] == "scatter"

    chosen = batch.select(swept_axis="T", method="fisher", sign="max", t_window_k=(7.0, 11.0))
    rows = chosen.candidates.sort_values("sweep_id").reset_index(drop=True)
    assert rows["sweep_id"].tolist() == [0, 1, 2]
    assert (rows["method"] == "fisher").all() and (rows["sign"] == "max").all()
    assert chosen.params["select"]["T"]["empty_sweeps"] == []

    cusps = batch.select(swept_axis="T", method="peak", sign="max", t_window_k=(7.0, 11.0))
    cusp_rows = cusps.candidates.sort_values("sweep_id").reset_index(drop=True)
    for (_, t_n), fisher_t, cusp_t in zip(
        FISHER_FAMILY, rows["T_K"], cusp_rows["T_K"], strict=True
    ):
        assert cusp_t == pytest.approx(t_n, abs=0.2)
        assert t_n - 2.0 < fisher_t < cusp_t - 0.4

# Reconciling the T-swept and H-swept families (reconcile)

#: Held fields (Tesla) of the reconcile campaign's T-scans. The planted
#: boundary puts the first two above the isotherms' temperature range, so
#: they must come out uncovered; the last is weak (see RECONCILE_WEAK_H_T).
RECONCILE_FIELDS_T = (0.05, 0.5, 1.0, 1.5, 2.0)

#: Isotherm temperatures (K). The 4 K one's critical field lies above the
#: highest T-scan field, so it must come out uncovered.
RECONCILE_TEMPS_K = (4.0, 7.0, 9.0, 11.0)

#: The T-scan drawn with a small cusp. Measured on this campaign, its
#: selected cusp scores 8.99, the isotherms' steps 19.0-24.8 and the other
#: T-scans' cusps 77.9-89.6, so RECONCILE_WEAK_GATE removes this one point
#: and nothing else: robust() can take out the T family's top node and
#: shrink what it covers.
RECONCILE_WEAK_H_T = 2.0

#: A min_significance between the weak cusp (8.99) and the weakest isotherm
#: step (19.0), with room on both sides.
RECONCILE_WEAK_GATE = 14.0

#: How far the shifted campaign moves the 11 K isotherm's step (Tesla).
RECONCILE_SHIFT_T = 0.25

RECONCILE_TOLS = {"t_tol_k": 0.4, "h_tol_t": 0.05}

RECONCILE_SELECT_T = {"swept_axis": "T", "method": "peak", "sign": "max"}

RECONCILE_SELECT_H = {"swept_axis": "H", "method": "derivative", "sign": "rise"}


def _reconcile_t_b(h_t):
    """The planted boundary as T_b(H) in K. Linear, so a piecewise-linear
    interpolant through points on it is exact, and a residual measures only
    detection error or a planted shift."""
    return 14.0 - 4.0 * h_t


def _reconcile_h_b(t_k):
    """The same line as H_b(T) in Tesla."""
    return (14.0 - t_k) / 4.0


def _reconcile_campaign(directory, *, shift_11k=0.0):
    """T-scans and split isotherms tracing one line, optionally with the
    11 K isotherm's step moved by ``shift_11k`` Tesla."""
    directory.mkdir()
    rng = np.random.default_rng(3)
    t = np.linspace(2.0, 20.0, 181)
    tscans = []
    for i, h_t in enumerate(RECONCILE_FIELDS_T):
        height = 0.1 if h_t == RECONCILE_WEAK_H_T else 1.0
        moment = (
            0.1
            + simulate.synthetic_chi_peak(t, centre=_reconcile_t_b(h_t), width=0.8, height=height)
            + rng.normal(scale=0.005, size=t.shape)
        )
        path = simulate.write_mpms3_file(
            directory / f"mt_{i}.dat", app="SQUID DC", temperature_K=t,
            field_oe=np.full_like(t, h_t * 1e4), dc_moment_emu=moment,
        )
        tscans.append(load(path))
    field_oe = np.linspace(0.0, 30_000.0, 121)
    temps, fields, moments = [], [], []
    for t_k in RECONCILE_TEMPS_K:
        h_c = _reconcile_h_b(t_k) + (shift_11k if t_k == 11.0 else 0.0)
        moment = 0.5 * (1 + np.tanh((field_oe / 1e4 - h_c) / 0.1))
        moments.append(moment + rng.normal(scale=0.005, size=field_oe.shape))
        temps.append(np.full_like(field_oe, t_k))
        fields.append(field_oe)
    stacked = simulate.write_mpms3_file(
        directory / "mh.dat", app="SQUID DC", temperature_K=np.concatenate(temps),
        field_oe=np.concatenate(fields), vsm_moment_emu=np.concatenate(moments),
    )
    return detect_transitions_all(tscans + sweeps.split_isotherms(load(stacked)))


@pytest.fixture(scope="module")
def reconcile_campaign(tmp_path_factory):
    """Sweep ids 0-4 are the T-scans, 5-8 the isotherms, all on one line."""
    return _reconcile_campaign(tmp_path_factory.mktemp("reconcile") / "line")


@pytest.fixture(scope="module")
def reconcile_shifted(tmp_path_factory):
    """The same campaign with the 11 K isotherm's step moved up in field."""
    return _reconcile_campaign(
        tmp_path_factory.mktemp("reconcile") / "shifted", shift_11k=RECONCILE_SHIFT_T
    )


def _selected(result):
    return result.select(**RECONCILE_SELECT_T).select(**RECONCILE_SELECT_H)


def _by_sweep(result) -> pd.DataFrame:
    return result.candidates.set_index("sweep_id").sort_index()


def test_one_planted_line_reconciles_everywhere_it_is_covered(reconcile_campaign):
    rec = _selected(reconcile_campaign).reconcile(**RECONCILE_TOLS)
    rows = _by_sweep(rec)
    assert list(rows.index) == list(range(9))  # nothing dropped
    covered = {0: False, 1: False, 2: True, 3: True, 4: True,
               5: False, 6: True, 7: True, 8: True}
    assert rows["cross_covered"].to_dict() == covered
    inside = rows[rows["cross_covered"]]
    assert inside["cross_concordant"].eq(True).all()
    # Residuals near zero relative to their own uncertainty, not merely
    # inside a generous tolerance.
    assert (inside["cross_residual_sigma"].abs() < 0.5).all()
    assert (inside["cross_residual_K"].abs().fillna(0.0) < 0.05).all()
    assert (inside["cross_residual_T"].abs().fillna(0.0) < 0.01).all()


def test_each_residual_is_nan_on_the_other_axis(reconcile_campaign):
    rows = _selected(reconcile_campaign).reconcile(**RECONCILE_TOLS).candidates
    on_t = rows["swept_axis"] == "T"
    assert rows.loc[on_t, "cross_residual_K"].isna().all()
    assert rows.loc[~on_t, "cross_residual_T"].isna().all()
    covered = rows["cross_covered"]
    assert rows.loc[on_t & covered, "cross_residual_T"].notna().all()
    assert rows.loc[~on_t & covered, "cross_residual_K"].notna().all()


def test_uncovered_points_are_never_extrapolated(reconcile_campaign):
    """The 0.05 T and 0.5 T scans order above the isotherms' 4-11 K; the
    4 K isotherm's 2.5 T lies above the highest T-scan. None gets a number,
    and concordance is undefined (<NA>), not False."""
    rows = _by_sweep(_selected(reconcile_campaign).reconcile(**RECONCILE_TOLS))
    for sid in (0, 1, 5):
        row = rows.loc[sid]
        assert not row["cross_covered"]
        assert np.isnan(row["cross_residual_K"]) and np.isnan(row["cross_residual_T"])
        assert np.isnan(row["cross_residual_sigma"])
        assert row["cross_concordant"] is pd.NA
    assert str(rows["cross_concordant"].dtype) == "boolean"
    assert rows["cross_covered"].dtype == bool


def test_a_shifted_isotherm_breaks_exactly_the_comparisons_it_enters(reconcile_shifted):
    """Moving the 11 K step up by 0.25 T touches two comparisons, both
    predictable from the planted line: the isotherm itself now sits at
    T_b(1.0 T) = 10 K, so T0 - T_b = +1.0 K; and the 1.0 T scan (T_N = 10 K)
    is read off the H family between 9 K (1.25 T) and 11 K (now 1.0 T), at
    1.125 T, so H0 - H_b = -0.125 T. Everything else stays concordant."""
    rec = _selected(reconcile_shifted).reconcile(**RECONCILE_TOLS)
    rows = _by_sweep(rec)
    broken = rows.index[rows["cross_concordant"].eq(False).fillna(False)].tolist()
    assert broken == [2, 8]
    assert rows.loc[8, "cross_residual_K"] == pytest.approx(4.0 * RECONCILE_SHIFT_T, abs=0.05)
    assert rows.loc[2, "cross_residual_T"] == pytest.approx(-RECONCILE_SHIFT_T / 2, abs=0.01)
    assert rows.loc[8, "cross_residual_sigma"] > 3.0
    assert rows.loc[2, "cross_residual_sigma"] < -3.0
    assert rows.loc[[3, 4, 6, 7], "cross_concordant"].eq(True).all()
    assert rec.params["reconcile"]["T"]["n_concordant"] == 2
    assert rec.params["reconcile"]["H"]["n_concordant"] == 2


def _reconcile_table(t_nodes, h_points, *, h_nodes=((5.0, 2.0), (6.5, 1.6))):
    """A hand-built, already-selected result: T-scans ``(H_T, T_K,
    T_err_K)``, then the isotherms ``h_nodes`` ``(T_K, H_T)`` with a 0.05 T
    error, then ``h_points`` ``(T_K, H_T, H_err_T)``. One candidate per
    sweep, sweep ids in that order."""
    rows = []
    for h_t, t_k, err in t_nodes:
        rows.append({"T_K": t_k, "H_T": h_t, "T_err_K": err, "swept_axis": "T",
                     "method": "peak", "sign": "max"})
    for t_k, h_t, err in [*[(t, h, 0.05) for t, h in h_nodes], *h_points]:
        rows.append({"T_K": t_k, "H_T": h_t, "H_err_T": err, "swept_axis": "H",
                     "method": "derivative", "sign": "rise"})
    frame = pd.DataFrame(rows)
    frame["candidate_id"] = range(len(frame))
    frame["sweep_id"] = range(len(frame))
    frame = frame.assign(signal="moment_emu", significance=10.0, edge=False)
    frame = frame.reindex(columns=CANDIDATE_COLUMNS)
    result = TransitionResult(
        candidates=frame, detections=pd.DataFrame(columns=DETECTION_COLUMNS), params={}
    )
    return _selected(result)


def test_combined_uncertainty_follows_the_documented_derivation():
    """T family nodes (0 T, 10 K, +-0.1 K) and (1 T, 6 K, +-0.3 K): at an
    isotherm's H_c = 0.5 T the boundary is 8 K, its error the nodes' errors
    interpolated linearly (0.2 K), and the slope -4 K/T times the point's
    own 0.05 T error adds 0.2 K in quadrature."""
    chosen = _reconcile_table([(0.0, 10.0, 0.1), (1.0, 6.0, 0.3)], [(8.5, 0.5, 0.05)])
    rows = _by_sweep(chosen.reconcile(t_tol_k=0.4, h_tol_t=0.1))
    point = rows.loc[4]
    assert point["cross_covered"]
    assert point["cross_residual_K"] == pytest.approx(0.5)
    assert point["cross_residual_sigma"] == pytest.approx(0.5 / np.hypot(0.2, 4.0 * 0.05))
    assert not point["cross_concordant"]  # 0.5 K > t_tol_k


def test_zero_combined_uncertainty_gives_nan_not_inf():
    chosen = _reconcile_table([(0.0, 10.0, 0.0), (1.0, 6.0, 0.0)], [(8.5, 0.5, 0.0)])
    rows = _by_sweep(chosen.reconcile(t_tol_k=1.0, h_tol_t=0.1))
    assert rows.loc[4, "cross_residual_K"] == pytest.approx(0.5)
    assert np.isnan(rows.loc[4, "cross_residual_sigma"])
    assert rows.loc[4, "cross_concordant"]
    assert not np.isinf(rows["cross_residual_sigma"]).any()


def test_a_family_of_one_covers_nothing():
    """A single T-scan is not a boundary: the isotherms are all uncovered,
    even the one whose H_c equals that scan's held field."""
    chosen = _reconcile_table([(0.5, 8.0, 0.1)], [(8.0, 0.5, 0.05)])
    rec = chosen.reconcile(**RECONCILE_TOLS)
    on_h = rec.candidates[rec.candidates["swept_axis"] == "H"]
    assert not on_h["cross_covered"].any()
    assert on_h["cross_residual_K"].isna().all()
    assert rec.params["reconcile"]["H"]["n_covered"] == 0


@pytest.mark.parametrize(
    "select_axes, missing",
    [((), "'T' and 'H'"), (("T",), "'H'"), (("H",), "'T'")],
)
def test_reconcile_requires_a_selection_on_both_axes(reconcile_campaign, select_axes, missing):
    partial = reconcile_campaign
    for axis in select_axes:
        partial = partial.select(**(RECONCILE_SELECT_T if axis == "T" else RECONCILE_SELECT_H))
    with pytest.raises(ValueError, match=rf"meaningless.*select\(swept_axis=\.\.\.\) for {missing}"):
        partial.reconcile(**RECONCILE_TOLS)


@pytest.mark.parametrize("family", ["T", "H"])
def test_duplicate_held_values_are_refused_by_sweep_id(family):
    if family == "T":  # two T-scans 5 Oe apart, inside SCAN_H_TOL_OE
        chosen = _reconcile_table([(0.0, 10.0, 0.1), (1.0, 6.0, 0.1), (1.0005, 6.1, 0.1)], [])
        match = r"SCAN_H_TOL_OE\): sweep_ids \[1, 2\] at H_T = 1, 1\.0005 T"
    else:  # two isotherms 0.3 K apart, inside SCAN_T_TOL_K
        chosen = _reconcile_table(
            [(0.0, 10.0, 0.1), (1.0, 6.0, 0.1)], [], h_nodes=((7.0, 1.8), (7.3, 1.7), (9.0, 1.2))
        )
        match = r"SCAN_T_TOL_K\): sweep_ids \[2, 3\] at T_K = 7, 7\.3 K"
    with pytest.raises(ValueError, match=match):
        chosen.reconcile(**RECONCILE_TOLS)


def test_held_values_just_outside_the_scan_tolerance_are_accepted():
    chosen = _reconcile_table(
        [(0.0, 10.0, 0.1), (1.0, 6.0, 0.1)], [], h_nodes=((7.0, 1.8), (7.6, 1.7))
    )
    assert chosen.reconcile(**RECONCILE_TOLS).params["reconcile"]["H"]["n_points"] == 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"t_tol_k": None, "h_tol_t": 0.05}, "t_tol_k is None.*no default"),
        ({"t_tol_k": 0.4, "h_tol_t": None}, "h_tol_t is None.*no default"),
        ({"t_tol_k": 0.0, "h_tol_t": 0.05}, "t_tol_k must be positive and finite"),
        ({"t_tol_k": -1.0, "h_tol_t": 0.05}, "t_tol_k must be positive and finite"),
        ({"t_tol_k": 0.4, "h_tol_t": float("nan")}, "h_tol_t must be positive and finite"),
        ({"t_tol_k": float("inf"), "h_tol_t": 0.05}, "t_tol_k must be positive and finite"),
        ({"t_tol_k": "0.4", "h_tol_t": 0.05}, "t_tol_k must be a number"),
        ({"t_tol_k": True, "h_tol_t": 0.05}, "t_tol_k must be a number"),
    ],
)
def test_reconcile_validates_its_tolerances(reconcile_campaign, kwargs, match):
    chosen = _selected(reconcile_campaign)
    with pytest.raises(ValueError, match=match):
        chosen.reconcile(**kwargs)


def test_reconcile_tolerances_have_no_defaults(reconcile_campaign):
    chosen = _selected(reconcile_campaign)
    with pytest.raises(TypeError):
        chosen.reconcile(t_tol_k=0.4)
    with pytest.raises(TypeError):
        chosen.reconcile(0.4, 0.05)  # keyword-only


def test_robust_after_reconcile_recomputes_on_the_new_view(reconcile_campaign):
    """A threshold that removes the weak 2.0 T scan removes the T family's
    top node, so the 7 K isotherm (H_c = 1.75 T) loses its coverage: the
    columns are recomputed, not carried. Either call order gives the same."""
    rec = _selected(reconcile_campaign).reconcile(**RECONCILE_TOLS)
    later = rec.robust(min_significance=RECONCILE_WEAK_GATE)
    earlier = _selected(reconcile_campaign.robust(min_significance=RECONCILE_WEAK_GATE)).reconcile(
        **RECONCILE_TOLS
    )
    pd.testing.assert_frame_equal(later.candidates, earlier.candidates)
    assert later.params["reconcile"] == earlier.params["reconcile"]
    assert later.params["select"]["T"]["empty_sweeps"] == [4]
    rows = _by_sweep(later)
    assert 4 not in rows.index
    assert not rows.loc[6, "cross_covered"]
    assert later.params["reconcile"]["T"] == {
        "residual": "cross_residual_T", "n_points": 4, "n_covered": 2, "n_concordant": 2,
    }
    assert later.params["reconcile"]["H"]["n_covered"] == 2
    assert list(later.params)[-1] == "reconcile"
    # Loosening again restores the node and the coverage.
    restored = later.robust(min_significance=5.0)
    assert _by_sweep(restored).loc[6, "cross_covered"]


def test_select_after_reconcile_recomputes(reconcile_campaign):
    """Re-selecting the T axis with a 9-15 K window changes the T family:
    the 1.5 T scan (cusp at 8 K, nothing else) is left empty, and the weak
    2.0 T scan's choice moves from its 6 K cusp to its strongest noise
    maximum inside the window, at 11.8 K -- select keeps the best candidate
    in a window, real or not. The reconcile columns follow, and the bad
    node shows: the 7 K isotherm is now read off a segment bent up to
    11.8 K, and is no longer concordant."""
    rec = _selected(reconcile_campaign).reconcile(**RECONCILE_TOLS)
    narrowed = rec.select(**RECONCILE_SELECT_T, t_window_k=(9.0, 15.0))
    direct = (
        reconcile_campaign.select(**RECONCILE_SELECT_T, t_window_k=(9.0, 15.0))
        .select(**RECONCILE_SELECT_H)
        .reconcile(**RECONCILE_TOLS)
    )
    pd.testing.assert_frame_equal(narrowed.candidates, direct.candidates)
    assert narrowed.params["reconcile"] == direct.params["reconcile"]
    assert narrowed.params["select"]["T"]["empty_sweeps"] == [3]
    assert narrowed.params["reconcile"]["T"]["n_points"] == 4
    rows = _by_sweep(narrowed)
    assert rows.loc[4, "T_K"] == pytest.approx(11.8, abs=0.1)
    assert _by_sweep(rec).loc[6, "cross_concordant"]
    assert not rows.loc[6, "cross_concordant"]
    assert rows.loc[6, "cross_residual_K"] < -1.0
    assert narrowed.params["reconcile"] != rec.params["reconcile"]


def test_link_after_reconcile_recomputes_and_commutes(reconcile_campaign):
    tolerances = {"t_tol_k": 2.5, "h_tol_t": 0.6}
    first = _selected(reconcile_campaign).reconcile(**RECONCILE_TOLS).link(**tolerances)
    second = _selected(reconcile_campaign).link(**tolerances).reconcile(**RECONCILE_TOLS)
    pd.testing.assert_frame_equal(first.candidates, second.candidates)
    assert first.params == second.params
    assert {"chain_id", "cross_residual_K", "cross_concordant"} <= set(first.candidates)
    assert list(first.params)[-1] == "reconcile"


def test_all_drops_reconcile_with_the_selection_it_depends_on(reconcile_campaign):
    rec = _selected(reconcile_campaign).link(t_tol_k=2.5, h_tol_t=0.6).reconcile(
        **RECONCILE_TOLS
    )
    restored = rec.all()
    assert "reconcile" not in restored.params
    assert "link" in restored.params
    assert not any(c.startswith("cross_") for c in restored.candidates.columns)
    assert "chain_id" in restored.candidates.columns
    assert "reconciled" not in repr(restored)


def test_reconcile_record_holds_plain_python_values(reconcile_campaign):
    rec = _selected(reconcile_campaign).reconcile(t_tol_k=np.float64(0.4), h_tol_t=1)
    record = rec.params["reconcile"]
    assert record == {
        "t_tol_k": 0.4,
        "h_tol_t": 1.0,
        "T": {"residual": "cross_residual_T", "n_points": 5, "n_covered": 3, "n_concordant": 3},
        "H": {"residual": "cross_residual_K", "n_points": 4, "n_covered": 3, "n_concordant": 3},
    }
    assert type(record["t_tol_k"]) is float and type(record["h_tol_t"]) is float
    for axis in ("T", "H"):
        assert all(type(v) is int for k, v in record[axis].items() if k != "residual")


def test_reconcile_repr(reconcile_shifted):
    rec = _selected(reconcile_shifted).reconcile(**RECONCILE_TOLS)
    assert repr(rec) == (
        "TransitionResult(9 candidates, 24 detections "
        "(selected on T and H, reconciled: 6 of 9 covered, 4 concordant))"
    )


def test_public_result_and_method_metadata_preserve_pickle_paths():
    import pickle

    t = np.linspace(2., 30., 101)
    result = find_transitions(t, np.exp(-((t - 12.) / 2.)**2), methods="peak")
    for value in (result, METHODS["peak"], METHODS["peak"].citations[0]):
        assert type(value).__module__ == "calamari.transitions"
        restored = pickle.loads(pickle.dumps(value))
        assert type(restored) is type(value)
        if isinstance(value, TransitionResult):
            pd.testing.assert_frame_equal(restored.candidates, value.candidates)
            pd.testing.assert_frame_equal(restored.detections, value.detections)
            assert restored.params == value.params
            assert restored.curve().keys() == value.curve().keys()
            for key in value.curve():
                np.testing.assert_array_equal(restored.curve()[key], value.curve()[key])
        else:
            assert restored == value


def test_public_result_annotations_remain_resolvable():
    from typing import get_type_hints

    hints = get_type_hints(TransitionResult)
    assert hints["candidates"] is pd.DataFrame
    assert hints["detections"] is pd.DataFrame
    assert hints["chains"] == pd.DataFrame | None
