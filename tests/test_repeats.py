"""Tests for calamari.repeats: grouping repeat measurements and combining them.

Every file here is synthetic (PK-2), but the two shapes that matter are
reproductions of real MPMS3 exports read during design:

* an AC temperature scan whose ``Measurement Number`` alternates 1, 2, 1, 2 --
  each setpoint measured twice;
* a combined VSM + AC field scan with **no repeats at all**, whose duplicated
  field values are an interleave of VSM (``Transport Action`` 1) and AC
  (``Transport Action`` 5) rows, one of each per setpoint.

The second is the important one: it is what a naive "consecutive rows at the
same setpoint are repeats" rule gets catastrophically wrong, and
``test_interleaved_vsm_and_ac_is_not_repeats`` is the regression test for it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from calamari import load, simulate
from calamari.exceptions import (
    MeasurementFamilyError,
    RepeatGroupError,
    RepeatWeightError,
)
from calamari.repeats import (
    AggregatedMeasurement,
    aggregate_repeats,
    combine,
    group_consecutive,
    repeat_groups,
)

# ---------------------------------------------------------------------------
# combine: the array-first statistics
# ---------------------------------------------------------------------------


def test_weighted_mean_and_birge_match_hand_calculation():
    """Two points 2 sigma apart: the weighted mean is the midpoint, the
    internal error is sigma/sqrt(2), and S = |z| because nu = 1."""
    point = combine([1.00, 1.02], [0.01, 0.01])

    assert point.n == 2
    assert point.weighted
    np.testing.assert_allclose(point.value, 1.01)
    np.testing.assert_allclose(point.internal, 0.01 / np.sqrt(2))
    np.testing.assert_allclose(point.birge, np.sqrt(2))
    # error = internal * max(1, S) = (0.01/sqrt2) * sqrt2 = 0.01
    np.testing.assert_allclose(point.error, 0.01)


def test_birge_does_not_shrink_an_error_when_repeats_agree():
    """max(1, S) is one-sided on purpose: repeats that agree better than
    chance do not license a smaller error bar than the instrument's own."""
    point = combine([1.000, 1.000], [0.01, 0.01])

    assert point.birge == pytest.approx(0.0)
    np.testing.assert_allclose(point.error, point.internal)


def test_external_error_reduces_to_the_standard_error_of_the_mean():
    """external = sigma_int * S, which for equal input errors is exactly
    s/sqrt(n) -- the identity that makes the generalisation trustworthy."""
    values = [1.0, 1.1, 1.3, 1.2]
    point = combine(values, [0.05] * 4, error="external")

    expected = np.std(values, ddof=1) / np.sqrt(len(values))
    np.testing.assert_allclose(point.error, expected)
    np.testing.assert_allclose(point.external, expected)


def test_internal_error_is_the_propagated_one():
    point = combine([1.0, 2.0], [0.01, 0.02], error="internal")
    np.testing.assert_allclose(point.error, 1 / np.sqrt(1 / 0.01**2 + 1 / 0.02**2))


def test_a_single_point_keeps_its_own_error_and_has_no_birge():
    """No scatter estimate exists for one point, and inventing one would be
    worse than saying so."""
    point = combine([1.0], [0.02])

    assert point.n == 1
    assert point.value == 1.0
    assert point.error == 0.02
    assert np.isnan(point.birge)


def test_an_all_blank_group_is_n_zero_not_an_error():
    """The normal case for one quantity in an interleaved AC/DC file, so it
    must not raise."""
    point = combine([np.nan, np.nan], [0.01, 0.01])

    assert point.n == 0
    assert np.isnan(point.value)


def test_all_zero_errors_fall_back_to_unweighted():
    """These instruments write a zero error to mean 'one scan averaged'. A
    group of them carries no weights, so the scatter is all there is."""
    point = combine([1.0, 1.1], [0.0, 0.0])

    assert not point.weighted
    np.testing.assert_allclose(point.value, 1.05)
    np.testing.assert_allclose(point.error, np.std([1.0, 1.1], ddof=1) / np.sqrt(2))


def test_mixing_zero_and_real_errors_refuses_to_guess():
    """A zero is an infinite weight: weighting would silently return whichever
    row carried the zero and discard the rest."""
    with pytest.raises(RepeatWeightError, match="mixes unusable standard errors"):
        combine([1.0, 1.1], [0.0, 0.01])


def test_external_is_a_working_escape_hatch_for_mixed_errors():
    """The resolution the exception names must actually resolve it.

    'external' asks for the observed scatter and no weighting, which is
    exactly what a group with unusable weights can still deliver -- so it
    combines rather than raising. Raising anyway would have made the message
    on RepeatWeightError name a call that does not work.
    """
    point = combine([1.0, 1.1], [0.0, 0.01], error="external")

    assert not point.weighted
    np.testing.assert_allclose(point.value, 1.05)
    np.testing.assert_allclose(point.error, np.std([1.0, 1.1], ddof=1) / np.sqrt(2))

    # ...and the other two rules still refuse, because they need weights.
    for rule in ("birge", "internal"):
        with pytest.raises(RepeatWeightError):
            combine([1.0, 1.1], [0.0, 0.01], error=rule)


def test_aggregate_repeats_honours_the_external_escape_hatch(tmp_path):
    """aggregate_repeats always forwards the file's error column into combine,
    so the escape hatch has to survive that path too -- a file with one
    single-scan zero beside real errors is the case it exists for."""
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "mixedzero.dat",
            app="SQUID DC",
            temperature_K=np.repeat([2.0, 2.4], 2),
            field_oe=np.zeros(4),
            vsm_moment_emu=[1.0, 1.1, 2.0, 2.1],
            vsm_moment_err_emu=[0.0, 0.01, 0.01, 0.01],
            extra_columns={
                "Measurement Number": [1, 2, 1, 2],
            },
        )
    )
    with pytest.raises(RepeatWeightError, match="mixes unusable standard errors"):
        aggregate_repeats(m)

    aggregated = aggregate_repeats(m, error="external")
    assert len(aggregated.data) == 2
    np.testing.assert_allclose(aggregated.moment_emu.to_numpy(), [1.05, 2.05])


def test_an_unknown_error_rule_is_rejected():
    with pytest.raises(ValueError, match="error must be"):
        combine([1.0, 1.1], [0.01, 0.01], error="birgee")


# ---------------------------------------------------------------------------
# group_consecutive: the array-first grouping
# ---------------------------------------------------------------------------


def test_grouping_bounds_the_total_span_not_the_consecutive_difference():
    """The difference from sweeps.segment_by_constant, which deliberately
    tolerates unbounded cumulative drift. Here a run of small steps must NOT
    chain into one group."""
    x = np.array([0.0, 0.4, 0.8, 1.2, 1.6])
    groups = group_consecutive([x], [1.0])

    assert [g.tolist() for g in groups] == [[0, 1, 2], [3, 4]]


def test_a_partition_is_never_grouped_across():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    kind = np.array([1, 5, 5, 1])
    groups = group_consecutive([x], [0.01], partition=kind)

    assert sorted(g.tolist() for g in groups) == [[0, 3], [1, 2]]


def test_a_group_over_max_repeats_raises():
    x = np.zeros(10)
    with pytest.raises(RepeatGroupError, match="over max_repeats"):
        group_consecutive([x], [0.01], max_repeats=4)


# ---------------------------------------------------------------------------
# The two real-file shapes
# ---------------------------------------------------------------------------


def _repeat_ac_file(tmp_path, name="repeats.dat"):
    """An AC T-scan measuring every setpoint twice, counter 1, 2, 1, 2.

    Reproduces the observed real-file shape: a tiny within-pair temperature
    difference against a much larger setpoint step, one frequency, one field.
    """
    temperature = np.repeat([2.0, 2.2, 2.4, 2.6], 2) + np.tile([0.0, 1e-4], 4)
    chi = np.array([1.00, 1.02, 2.00, 2.10, 3.00, 3.01, 4.00, 3.90])
    return load(
        simulate.write_mpms3_file(
            tmp_path / name,
            app="SQUID AC",
            temperature_K=temperature,
            field_oe=np.full(8, 60_000.0),
            chi_real=chi,
            ac_frequency_hz=np.full(8, 999.0),
            ac_drive_oe=np.full(8, 1.0),
            transport_action=np.full(8, 5.0),
            extra_columns={
                "AC X' Std Err. (emu/Oe)": np.full(8, 0.01),
                "Measurement Number": [1, 2, 1, 2, 1, 2, 1, 2],
            },
        )
    )


def _interleaved_file(tmp_path, name="interleaved.dat"):
    """A combined VSM + AC field scan with NO repeats.

    Each field setpoint is measured once by VSM (Transport Action 1) and once
    by AC (5), written as two rows carrying the same field. The interleave
    order flips between setpoints, exactly as the real file does.
    """
    action = np.array([1.0, 5.0, 5.0, 1.0, 1.0, 5.0])
    moment = np.where(action == 1.0, [1e-3, 0, 0, 2e-3, 3e-3, 0], np.nan)
    chi = np.where(action == 5.0, [0, 1.0, 2.0, 0, 0, 3.0], np.nan)
    return load(
        simulate.write_mpms3_file(
            tmp_path / name,
            app="SQUID AC",
            temperature_K=np.full(6, 30.0),
            field_oe=np.repeat([0.0, 200.0, 400.0], 2),
            vsm_moment_emu=moment,
            chi_real=chi,
            vsm_moment_err_emu=np.where(action == 1.0, 1e-5, np.nan),
            transport_action=action,
            extra_columns={"Measurement Number": np.ones(6)},
        )
    )


def test_the_instrument_counter_groups_the_repeats(tmp_path):
    m = _repeat_ac_file(tmp_path)
    assert [g.tolist() for g in repeat_groups(m)] == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_aggregating_combines_values_errors_and_setpoints(tmp_path):
    m = _repeat_ac_file(tmp_path)
    aggregated = aggregate_repeats(m)

    assert isinstance(aggregated, AggregatedMeasurement)
    assert len(aggregated.data) == 4
    assert aggregated.grouped_by == "measurement-number"

    # Values are the weighted means; with equal errors that is the midpoint.
    np.testing.assert_allclose(
        aggregated.chi_real.to_numpy(), [1.01, 2.05, 3.005, 3.95]
    )
    # Setpoints are averaged, not taken from one row.
    np.testing.assert_allclose(aggregated.temperature.to_numpy()[0], 2.00005)
    # The pair 2.00/2.10 is 10 sigma apart, so its error is inflated well past
    # the propagated 0.01/sqrt(2).
    errors = aggregated.chi_real_err.to_numpy()
    assert errors[1] > 5 * 0.01 / np.sqrt(2)
    assert errors[2] == pytest.approx(0.01 / np.sqrt(2))


def test_interleaved_vsm_and_ac_is_not_repeats(tmp_path):
    """The regression test this module exists for.

    A real combined run writes one VSM row and one AC row per field setpoint,
    carrying the *same* field value. Grouping on the setpoint alone pairs them
    and reports n = 2 for measurements that were never repeated. Partitioning
    on Transport Action is what prevents it, so aggregating this file must
    change nothing at all.
    """
    m = _interleaved_file(tmp_path)

    assert [g.tolist() for g in repeat_groups(m)] == [[0], [1], [2], [3], [4], [5]]

    aggregated = aggregate_repeats(m)
    assert len(aggregated.data) == len(m.data)
    np.testing.assert_allclose(
        aggregated.data["Moment (emu)"].to_numpy(),
        m.data["Moment (emu)"].to_numpy(),
        equal_nan=True,
    )
    # n is per quantity, not per group: the AC rows contribute no moment.
    assert aggregated.data["Moment [n]"].tolist() == [1, 0, 0, 1, 1, 0]


# ---------------------------------------------------------------------------
# The guard: neither mechanism is trusted alone
#
# Every case below silently corrupted the data before the per-group setpoint
# verification was added, and each was found by running the counter rule
# against a pattern no observed file happens to use.
# ---------------------------------------------------------------------------


def test_a_1122_counter_does_not_average_two_frequencies(tmp_path):
    """A counter running 1,1,2,2 rather than 1,2,1,2 pairs repeat 1 of one
    setpoint with repeat 2 of the previous. Here that would average a 100 Hz
    measurement with a 10 Hz one."""
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "c1122.dat",
            app="SQUID AC",
            temperature_K=np.full(4, 2.0),
            field_oe=np.zeros(4),
            chi_real=[1.0, 2.0, 1.1, 2.1],
            ac_frequency_hz=[100.0, 10.0, 100.0, 10.0],
            ac_drive_oe=np.ones(4),
            extra_columns={"Measurement Number": [1, 1, 2, 2]},
        )
    )
    with pytest.raises(RepeatGroupError, match="not one setpoint"):
        repeat_groups(m)


def test_a_global_counter_does_not_collapse_the_file(tmp_path):
    """A 'Measurement Number' that is really a global index runs 1..N and
    never resets, proposing one group containing everything."""
    n = 6
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "cglobal.dat",
            app="SQUID AC",
            temperature_K=np.linspace(2.0, 7.0, n),
            field_oe=np.zeros(n),
            chi_real=np.ones(n),
            extra_columns={"Measurement Number": np.arange(1, n + 1)},
        )
    )
    with pytest.raises(RepeatGroupError, match="not one setpoint"):
        repeat_groups(m)


def test_a_slow_ramp_is_not_mistaken_for_repeats(tmp_path):
    """Measure-while-sweeping is a real mode. With a 0.002 K step under a
    0.025 K tolerance, bounding each group's span still chops the ramp into
    fixed-width chunks -- turning a sweep into a coarser sweep. Only the
    separation-of-scales test catches it."""
    n = 100
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "ramp.dat",
            app="SQUID AC",
            temperature_K=2.0 + np.arange(n) * 0.002,
            field_oe=np.zeros(n),
            chi_real=np.ones(n),
        )
    )
    with pytest.raises(RepeatGroupError, match="no better separated"):
        repeat_groups(m)


def test_a_held_axis_that_wanders_does_not_block_grouping(tmp_path):
    """Only the swept axis has step structure; a held one must not be judged.

    A field held nominally constant across a temperature scan still drifts a
    couple of Oe, which exceeds the tight REPEAT_H_TOL_OE and so is not
    skipped as constant -- but it has no steps for a gap-to-width ratio to
    find. Requiring separation on *every* varying axis rejected this file,
    even though `classify_scan` calls the same field constant and the
    temperature axis is cleanly stepped.
    """
    setpoints = 2.0 + 0.2 * np.arange(6)
    temperature = np.repeat(setpoints, 2) + np.tile([0.0, 1e-4], 6)
    # Held at ~1000 Oe: a 2.5 Oe drift across the file, well inside
    # SCAN_H_TOL_OE (10 Oe) but well outside REPEAT_H_TOL_OE (1 Oe).
    field = 1000.0 + np.linspace(0.0, 2.5, temperature.size)

    m = load(
        simulate.write_mpms3_file(
            tmp_path / "heldfield.dat",
            app="SQUID DC",
            temperature_K=temperature,
            field_oe=field,
            dc_moment_emu=np.arange(temperature.size, dtype=float),
        )
    )
    assert m.scan_kind == "T-scan"  # the field is "constant" by any usual test

    groups = repeat_groups(m)
    assert [g.tolist() for g in groups] == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]
    assert len(aggregate_repeats(m).data) == 6


def test_a_ramp_with_a_wandering_held_axis_still_raises(tmp_path):
    """Relaxing to 'at least one axis' must not let the ramp through: a ramp
    is separated on no axis, however many of them wander."""
    n = 100
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "rampjitter.dat",
            app="SQUID AC",
            temperature_K=2.0 + np.arange(n) * 0.002,
            field_oe=1000.0 + np.linspace(0.0, 2.5, n),
            chi_real=np.ones(n),
        )
    )
    with pytest.raises(RepeatGroupError, match="no better separated"):
        repeat_groups(m)


def test_a_piecewise_constant_axis_cannot_rubber_stamp_a_ramp(tmp_path):
    """One axis's vote must be earned by measuring the gap, not assumed.

    A field held at one value for half a file and another for the rest varies
    by more than REPEAT_H_TOL_OE across the file, so it is eligible for the
    separation test -- yet it is bit-identical inside every group, and almost
    every neighbouring pair of groups shares its value. Treating
    "bit-identical within groups" as infinitely good separation without
    checking the between-group gap let that axis vote pass and accept any
    grouping at all, including the measure-while-sweeping ramp the whole
    check exists to catch.
    """
    n = 100
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "rampstep.dat",
            app="SQUID AC",
            temperature_K=2.0 + np.arange(n) * 0.002,
            field_oe=np.where(np.arange(n) < n // 2, 1000.0, 1005.0),
            chi_real=np.ones(n),
        )
    )
    with pytest.raises(RepeatGroupError, match="no better separated"):
        repeat_groups(m)


def test_a_bit_identical_but_stepped_axis_does_count_as_separated(tmp_path):
    """...and the abstention above must not become a blanket skip.

    A stacked M(H) run repeats each field setpoint, so the field is
    bit-identical inside every group *and* differs between them. That is
    infinitely good separation and has to keep counting, otherwise a file
    whose setpoints are recorded exactly -- the tidiest case there is --
    would stop aggregating.
    """
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "stackedmh.dat",
            app="SQUID DC",
            temperature_K=np.full(8, 2.0),
            field_oe=np.repeat([0.0, 200.0, 400.0, 600.0], 2),
            vsm_moment_emu=np.arange(8, dtype=float),
        )
    )
    assert [g.tolist() for g in repeat_groups(m)] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert len(aggregate_repeats(m).data) == 4


def test_an_interleaved_ramp_is_still_caught(tmp_path):
    """Boundaries join groups of the same partition, not file-adjacent rows.

    In an interleaved VSM/AC file no two rows of one kind are ever adjacent,
    so comparing file-adjacent rows skips every boundary and leaves nothing to
    judge -- a measure-while-sweeping ramp recorded in interleaved mode would
    pass with no evidence examined at all. This is the guard's target failure
    in the file shape this module exists for, so it gets its own test.
    """
    n = 80
    action = np.tile([1.0, 5.0], n // 2)
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "interramp.dat",
            app="SQUID AC",
            temperature_K=2.0 + np.arange(n) * 0.002,
            field_oe=np.zeros(n),
            vsm_moment_emu=np.where(action == 1.0, 1e-3, np.nan),
            chi_real=np.where(action == 5.0, 1.0, np.nan),
            transport_action=action,
        )
    )
    with pytest.raises(RepeatGroupError, match="no better separated"):
        repeat_groups(m)


def test_interleaved_repeats_still_aggregate(tmp_path):
    """...and the partition rule must not make interleaved files unaggregatable.

    Two VSM rows and two AC rows per field setpoint: genuine repeats within
    each mode, which must combine to one row per mode per setpoint.
    """
    action = np.tile([1.0, 1.0, 5.0, 5.0], 4)
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "interrep.dat",
            app="SQUID AC",
            temperature_K=np.full(16, 30.0),
            field_oe=np.repeat([0.0, 200.0, 400.0, 600.0], 4),
            vsm_moment_emu=np.where(action == 1.0, 1e-3, np.nan),
            chi_real=np.where(action == 5.0, 1.0, np.nan),
            transport_action=action,
        )
    )
    groups = repeat_groups(m)

    assert all(len(g) == 2 for g in groups)
    assert len(groups) == 8  # 4 setpoints x 2 modes
    assert len(aggregate_repeats(m).data) == 8


def test_use_counter_false_is_the_escape_hatch(tmp_path):
    """When a file's counter means something else, the setpoints can still
    group it."""
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "escape.dat",
            app="SQUID AC",
            temperature_K=np.repeat([2.0, 2.4], 2) + np.tile([0.0, 1e-4], 2),
            field_oe=np.zeros(4),
            chi_real=[1.0, 1.1, 2.0, 2.1],
            extra_columns={"Measurement Number": [1, 2, 3, 4]},
        )
    )
    with pytest.raises(RepeatGroupError):
        repeat_groups(m)

    groups = repeat_groups(m, use_counter=False)
    assert [g.tolist() for g in groups] == [[0, 1], [2, 3]]


def test_setpoint_grouping_works_without_any_instrument_columns(tmp_path):
    """Missing counters and transport codes leave setpoint-based grouping available."""
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "without_counters.dat",
            app="SQUID DC",
            temperature_K=np.repeat([2.0, 2.4, 2.8], 2) + np.tile([0.0, 1e-4], 3),
            field_oe=np.zeros(6),
            dc_moment_emu=[1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        )
    )
    aggregated = aggregate_repeats(m)

    assert aggregated.grouped_by == "setpoint-tolerance"
    assert len(aggregated.data) == 3
    np.testing.assert_allclose(aggregated.moment_emu.to_numpy(), [1.05, 2.05, 3.05])


# ---------------------------------------------------------------------------
# Measurement families are separate populations
#
# Every case below needed the caller to run m.of_type(...) first before
# planning was done per family: the partition came from the Transport Action
# column rather than from row_families, and the mechanism choice and the
# separation check were made once for the whole file.
# ---------------------------------------------------------------------------


def _mixed_file_without_transport_action(tmp_path, name="mixed.dat", **extra):
    """A combined VSM + AC run that states nothing about its transport.

    One VSM row and one AC row per field setpoint, each setpoint measured
    twice, and no ``Transport Action`` column at all -- so the families can
    only be told apart by the columns each row populates.
    """
    kind = np.tile([1, 5], 4)
    return load(
        simulate.write_mpms3_file(
            tmp_path / name,
            app="SQUID AC",
            temperature_K=np.full(8, 30.0),
            field_oe=np.repeat([0.0, 0.0, 200.0, 200.0], 2),
            vsm_moment_emu=np.where(kind == 1, 1e-3, np.nan),
            chi_real=np.where(kind == 5, 1.0, np.nan),
            **extra,
        )
    )


def test_families_are_separated_with_no_transport_action_column(tmp_path):
    """The rows decide when the instrument does not state it.

    Keyed on the Transport Action column, this file had no partition at all,
    so a VSM row and an AC row at one setpoint landed in one group and were
    written out as a SINGLE row carrying both a moment and a susceptibility --
    a measurement the instrument never made. Each quantity's [n] said 1, so
    nothing about the output looked wrong.
    """
    m = _mixed_file_without_transport_action(tmp_path)
    assert m.measurement_types == ("VSM", "AC")

    assert [g.tolist() for g in repeat_groups(m)] == [[0, 2], [1, 3], [4, 6], [5, 7]]

    aggregated = aggregate_repeats(m)
    assert len(aggregated.data) == 4
    # One family per row, and the aggregate still holds both.
    assert aggregated.data["Moment [n]"].tolist() == [2, 0, 2, 0]
    assert aggregated.data["AC X' [n]"].tolist() == [0, 2, 0, 2]
    assert aggregated.measurement_types == ("VSM", "AC")


def test_per_family_counters_do_not_fuse_across_families(tmp_path):
    """The counter path fuses too, and more quietly.

    Two families each counting their own repeats 1, 2 interleave in file order
    as 1, 1, 2, 2 -- the very pattern this module warns about, arrived at
    innocently. Unpartitioned, the counter proposed groups spanning the family
    boundary and the setpoint check passed them, because they ARE one setpoint.
    """
    counter = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0])
    m = _mixed_file_without_transport_action(
        tmp_path, name="counters.dat", extra_columns={"Measurement Number": counter}
    )

    groups = repeat_groups(m)
    assert [g.tolist() for g in groups] == [[0, 2], [1, 3], [4, 6], [5, 7]]

    aggregated = aggregate_repeats(m)
    assert aggregated.grouped_by == "measurement-number"
    assert len(aggregated.data) == 4
    # No row carries two families' quantities.
    both = aggregated.data["Moment (emu)"].notna() & aggregated.data["AC X' (emu/Oe)"].notna()
    assert not both.any()


def _one_family_ramping_beside_another_repeating(tmp_path, name="veto.dat"):
    """AC repeats at 0.4 K spacing interleaved with a VSM ramp at 0.06 K.

    The VSM steps clear the 0.025 K repeat tolerance, so they are not
    repeats; the AC pairs sit 3e-3 K apart, which is the within-pair scatter
    a real repeat file shows.
    """
    ac_temperature = np.repeat([2.0, 2.4, 2.8, 3.2], 2) + np.tile([0.0, 3e-3], 4)
    vsm_temperature = 30.0 + 0.06 * np.arange(4)
    order = np.argsort(np.concatenate([np.arange(8) * 1.0, np.arange(4) * 2.0 + 0.5]))
    temperature = np.concatenate([ac_temperature, vsm_temperature])[order]
    action = np.concatenate([np.full(8, 5.0), np.full(4, 1.0)])[order]
    return load(
        simulate.write_mpms3_file(
            tmp_path / name,
            app="SQUID AC",
            temperature_K=temperature,
            field_oe=np.zeros(12),
            vsm_moment_emu=np.where(action == 1.0, 1e-3, np.nan),
            chi_real=np.where(action == 5.0, 1.0, np.nan),
            transport_action=action,
        )
    )


def test_one_familys_ramp_does_not_veto_anothers_repeats(tmp_path):
    """The separation check is one family's question, asked of that family.

    Judged over the whole file it was `min(between) / max(within)`, pointwise
    no larger than any single family's ratio -- so the VSM ramp's steps stood
    in for the AC pairs' and the whole file was refused, though nothing about
    the AC measurement was in doubt. (It could only ever veto, never vouch,
    which is why per-family checking is a relaxation and not a loosening.)
    """
    m = _one_family_ramping_beside_another_repeating(tmp_path)

    groups = [g.tolist() for g in repeat_groups(m)]
    assert groups == [[0, 2], [1], [3, 5], [4], [6, 8], [7], [9, 11], [10]]
    # The AC pairs combined; every VSM row survived as itself.
    assert len(aggregate_repeats(m).data) == 8


def test_a_counter_on_one_family_does_not_decide_for_the_other(tmp_path):
    """Mechanism per family, and the file says which was used where.

    'Measurement Number' populated on the AC rows and blank on the VSM ones is
    evidence about AC repeats and none at all about VSM ones. File-wide it put
    everything on the counter path, where a blank breaks the group at every
    row: the VSM repeats came back as singletons, with no error raised.
    """
    action = np.tile([1.0, 1.0, 5.0, 5.0], 3)
    counter = np.where(action == 5.0, np.tile([np.nan, np.nan, 1.0, 2.0], 3), np.nan)
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "onecounter.dat",
            app="SQUID AC",
            temperature_K=np.full(12, 30.0),
            field_oe=np.repeat([0.0, 200.0, 400.0], 4),
            vsm_moment_emu=np.where(action == 1.0, [1e-3, 1.1e-3] * 6, np.nan),
            chi_real=np.where(action == 5.0, 1.0, np.nan),
            transport_action=action,
            extra_columns={"Measurement Number": counter},
        )
    )

    groups = repeat_groups(m)
    assert [g.tolist() for g in groups] == [
        [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
    ]

    aggregated = aggregate_repeats(m)
    assert aggregated.grouped_by_family == {
        "VSM": "setpoint-tolerance",
        "AC": "measurement-number",
    }
    # The composite string follows measurement_type: FAMILY_ORDER, not file order.
    assert aggregated.grouped_by == "VSM:setpoint-tolerance+AC:measurement-number"
    assert aggregated.summary()["grouped_by_family"]["AC"] == "measurement-number"


def test_a_multi_family_aggregate_still_reproduces_its_groups(tmp_path):
    """The .source + .params recoverability contract has to survive planning
    per family, which is what stops aggregation and repeat_groups drifting
    apart -- they must answer the same question the same way."""
    m = _one_family_ramping_beside_another_repeating(tmp_path, name="recover.dat")
    aggregated = aggregate_repeats(m)

    recovered = repeat_groups(aggregated.source, **aggregated.params)
    assert [g.tolist() for g in recovered] == [g.tolist() for g in repeat_groups(m)]


def test_a_row_belonging_to_no_family_is_refused_where_families_could_fuse(tmp_path):
    """A row populating two families' columns is the row that could be fused.

    calamari cannot tell whose repeats it belongs with, and grouping it by
    setpoint alone would combine two different measurements into one point --
    so it says so rather than picking.
    """
    kind = np.array([1, 5, 1, 5])
    moment = np.where(kind == 1, 1e-3, np.nan)
    chi = np.where(kind == 5, 1.0, np.nan)
    moment[3] = 2e-3  # row 3 now carries a VSM moment AND an AC susceptibility
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "unplaceable.dat",
            app="SQUID AC",
            temperature_K=np.full(4, 30.0),
            field_oe=np.repeat([0.0, 200.0], 2),
            vsm_moment_emu=moment,
            chi_real=chi,
        )
    )

    with pytest.raises(MeasurementFamilyError, match="cannot be assigned"):
        repeat_groups(m)
    with pytest.raises(MeasurementFamilyError, match="of_type"):
        aggregate_repeats(m)


def test_a_row_belonging_to_no_family_is_fine_when_there_is_only_one(tmp_path):
    """...and the refusal must not spread to files with nothing to fuse.

    A blank moment on one row of an ordinary DC file makes that row
    unplaceable too, by exactly the same rule. With one family present there
    is nothing it could be confused with, so it groups as it always did --
    refusing here would reject a file over the row that says least.
    """
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "blankrow.dat",
            app="SQUID DC",
            temperature_K=np.repeat([2.0, 2.4, 2.8], 2) + np.tile([0.0, 1e-4], 3),
            field_oe=np.zeros(6),
            dc_moment_emu=[1.0, 1.1, 2.0, np.nan, 3.0, 3.1],
        )
    )

    assert [g.tolist() for g in repeat_groups(m)] == [[0, 1], [2, 3], [4, 5]]
    assert aggregate_repeats(m).grouped_by == "setpoint-tolerance"


def test_a_setpoint_no_family_measured_aggregates_quietly(tmp_path):
    """Channels absent throughout a family's repeat group stay NaN without warnings."""
    action = np.tile([6.0, 6.0, 5.0, 5.0], 3)
    is_ac = action == 5.0
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "dcandac.dat",
            app="SQUID AC",
            temperature_K=np.repeat([2.0, 2.4, 2.8], 4),
            field_oe=np.zeros(12),
            dc_moment_emu=np.where(is_ac, np.nan, 1e-3),
            chi_real=np.where(is_ac, 1.0, np.nan),
            ac_frequency_hz=np.where(is_ac, 999.0, np.nan),
            ac_drive_oe=np.where(is_ac, 1.0, np.nan),
            transport_action=action,
        )
    )
    assert m.measurement_types == ("DC", "AC")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        aggregated = aggregate_repeats(m)

    assert len(aggregated.data) == 6
    families = aggregated.row_families.to_numpy()
    drive = aggregated.data["AC Drive (Oe)"].to_numpy()
    assert np.isnan(drive[families == "DC"]).all()   # blank in, blank out
    np.testing.assert_allclose(drive[families == "AC"], 1.0)


def test_a_code_contradicting_its_own_columns_is_refused_by_grouping(tmp_path):
    """Grouping now asks every file which family each row is, so a file whose
    answer is self-contradictory can no longer be aggregated as it stands.

    This is a deliberate narrowing: such a file used to group on its Transport
    Action codes without the columns getting a say. The message names
    restrict() rather than of_type(), because of_type asks the same question
    and would fail the same way.
    """
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "contradiction.dat",
            app="SQUID AC",
            temperature_K=np.full(4, 30.0),
            field_oe=np.repeat([0.0, 200.0], 2),
            chi_real=np.full(4, 1.0),
            transport_action=np.full(4, 1.0),  # coded VSM, populating AC columns
        )
    )

    with pytest.raises(MeasurementFamilyError, match="restrict"):
        repeat_groups(m)


# ---------------------------------------------------------------------------
# The type
# ---------------------------------------------------------------------------


def test_restrict_preserves_the_type_and_its_provenance(tmp_path):
    """dataclasses.replace constructs type(self), so the derivation helpers
    need no override -- and the [n] diagnostics are columns, so restrict
    subsets them for free."""
    aggregated = aggregate_repeats(_repeat_ac_file(tmp_path))
    part = aggregated.restrict(t_range=(0.0, 2.3))

    assert isinstance(part, AggregatedMeasurement)
    assert part.grouped_by == aggregated.grouped_by
    assert part.error_rule == aggregated.error_rule
    assert len(part.data) == 2
    assert part.data["AC X' [n]"].tolist() == [2, 2]


def test_the_repr_reports_the_reduction_and_survives_restrict(tmp_path):
    """The median n is computed from the [n] columns rather than stored: a
    stored 'was 8 rows' would keep claiming a reduction the surviving rows no
    longer show."""
    aggregated = aggregate_repeats(_repeat_ac_file(tmp_path))

    assert repr(aggregated).startswith("<AggregatedMeasurement ")
    assert "combined n=2" in repr(aggregated)
    assert "4 pts" in repr(aggregated)
    assert "2 pts" in repr(aggregated.restrict(t_range=(0.0, 2.3)))


def test_the_repr_inherits_the_ac_frequency_bit(tmp_path):
    """_repr_bits is extended, not replaced, so the parent's AC frequency
    appears in an aggregate's repr without this class doing anything."""
    aggregated = aggregate_repeats(_repeat_ac_file(tmp_path))
    assert "999 Hz" in repr(aggregated)


def test_summary_reports_how_the_aggregation_was_done(tmp_path):
    aggregated = aggregate_repeats(_repeat_ac_file(tmp_path))
    summary = aggregated.summary()

    assert summary["grouped_by"] == "measurement-number"
    assert summary["error_rule"] == "birge"
    assert summary["median_n"] == 2.0
    assert summary["max_birge"] > 1.0
    assert summary["n_points"] == 4  # groups, not the file's rows


def test_the_source_measurement_reproduces_the_groups(tmp_path):
    """No row-membership bookkeeping is stored, because grouping is
    deterministic: .source plus .params is enough to get the rows back."""
    m = _repeat_ac_file(tmp_path)
    aggregated = aggregate_repeats(m)

    recovered = repeat_groups(aggregated.source, **aggregated.params)
    assert [g.tolist() for g in recovered] == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_aggregating_twice_is_refused(tmp_path):
    """After one pass the counter column holds first-values, so a second pass
    would quietly fall through to tolerance grouping."""
    aggregated = aggregate_repeats(_repeat_ac_file(tmp_path))

    with pytest.raises(TypeError, match="already aggregated"):
        aggregate_repeats(aggregated)


def test_a_provenance_column_collision_raises_rather_than_overwrites(tmp_path):
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "collide.dat",
            app="SQUID AC",
            temperature_K=np.repeat([2.0, 2.4], 2) + np.tile([0.0, 1e-4], 2),
            field_oe=np.zeros(4),
            chi_real=[1.0, 1.1, 2.0, 2.1],
            extra_columns={"AC X' [n]": np.arange(4.0)},
        )
    )
    with pytest.raises(RepeatGroupError, match="provenance column"):
        aggregate_repeats(m)


def test_housekeeping_rows_are_carried_through_not_dropped(tmp_path):
    """Transport Action 2/3/4 are touchdowns and locate scans rather than
    measurements. They are never grouped, but losing rows silently would be
    worse than carrying one that means nothing to us."""
    action = np.array([5.0, 5.0, 2.0, 5.0, 5.0])
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "housekeeping.dat",
            app="SQUID AC",
            temperature_K=[2.0, 2.0001, 2.0, 2.4, 2.4001],
            field_oe=np.zeros(5),
            chi_real=[1.0, 1.1, np.nan, 2.0, 2.1],
            transport_action=action,
            extra_columns={"Measurement Number": [1, 2, 1, 1, 2]},
        )
    )
    groups = repeat_groups(m)

    assert [g.tolist() for g in groups] == [[0, 1], [2], [3, 4]]
    assert len(aggregate_repeats(m).data) == 3


# ---------------------------------------------------------------------------
# Downstream: an AggregatedMeasurement is an ordinary Measurement
# ---------------------------------------------------------------------------


def test_aggregating_with_internal_errors_leaves_a_fit_unchanged(tmp_path):
    """The claim the module docstring makes, pinned. Weighted least squares
    over n repeats and over their combined point differ by a term independent
    of the parameters, so MIGRAD and HESSE land in exactly the same place --
    which is why aggregation is not applied by default.

    The repeats sit at *identical* temperatures because that is the condition
    the identity requires; with realistic setpoint jitter the combined point
    lands at the mean x instead and the agreement loosens slightly.

    Tolerance note: solved analytically (a two-parameter weighted normal
    equation) the two covariance matrices agree to 3e-16. Through iminuit they
    agree to ~1e-4, because HESSE differentiates numerically with step sizes
    that track the absolute function scale -- and chi2_raw = chi2_agg + const
    is precisely a shift in that scale. The looser bound here is iminuit's
    numerics, not the identity.
    """
    from calamari import fit

    rng = np.random.default_rng(20260814)
    setpoints = np.arange(2.0, 12.0, 1.0)
    temperature = np.repeat(setpoints, 2)
    truth = 3.0 * np.repeat(setpoints, 2) + 1.0
    moment = truth + rng.normal(0.0, 0.05, temperature.size)

    m = load(
        simulate.write_mpms3_file(
            tmp_path / "fitme.dat",
            app="SQUID DC",
            temperature_K=temperature,
            field_oe=np.zeros(temperature.size),
            vsm_moment_emu=moment,
            vsm_moment_err_emu=np.full(temperature.size, 0.05),
            extra_columns={
                "Measurement Number": np.tile([1, 2], len(setpoints)),
            },
        )
    )
    aggregated = aggregate_repeats(m, error="internal")

    raw_fit = fit("linear", m.temperature, m.moment_emu, yerr=m.moment_err_emu)
    agg_fit = fit(
        "linear",
        aggregated.temperature,
        aggregated.moment_emu,
        yerr=aggregated.moment_err_emu,
    )

    assert raw_fit.success and agg_fit.success
    for name in raw_fit.parameters:
        assert raw_fit.parameters[name] == pytest.approx(
            agg_fit.parameters[name], rel=1e-6
        )
        assert raw_fit.uncertainties[name] == pytest.approx(
            agg_fit.uncertainties[name], rel=1e-4
        )


def test_an_aggregate_can_be_split_into_isotherms(tmp_path):
    """sweeps.split_isotherms builds on restrict, so it preserves the type."""
    from calamari.sweeps import split_isotherms

    field = np.linspace(0.0, 1000.0, 4)
    temperature = np.concatenate([np.full(4, 2.0), np.full(4, 6.0)])
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "stacked.dat",
            app="SQUID DC",
            temperature_K=np.repeat(temperature, 2),
            field_oe=np.repeat(np.tile(field, 2), 2),
            vsm_moment_emu=np.repeat(np.arange(8.0), 2) + np.tile([0.0, 0.1], 8),
            extra_columns={"Measurement Number": np.tile([1, 2], 8)},
        )
    )
    aggregated = aggregate_repeats(m)
    isotherms = split_isotherms(aggregated)

    assert len(isotherms) == 2
    for _, sub in isotherms:
        assert isinstance(sub, AggregatedMeasurement)


# ---------------------------------------------------------------------------
# Detection is free
# ---------------------------------------------------------------------------


def test_a_file_with_repeats_announces_them(tmp_path):
    """Counter-only, deliberately: summary() runs on every file in a catalog
    build, so a tolerance-based detector would announce phantom repeats
    tree-wide."""
    m = _repeat_ac_file(tmp_path)

    assert m.max_repeats() == 2
    assert "repeats x2" in repr(m)
    assert m.summary()["max_repeats"] == 2


def test_a_file_without_repeats_says_nothing(tmp_path):
    m = _interleaved_file(tmp_path)

    assert m.max_repeats() == 1
    assert "repeats" not in repr(m)


def test_a_settling_excursion_within_a_pair_still_groups(tmp_path):
    """Invented jitter exceeds the former 0.01 K tolerance but fits the current
    tolerance; counter pairs group while well-separated setpoints remain distinct."""
    temperature = np.array([12.0, 12.0, 12.65, 12.664, 13.4, 13.4])
    m = load(
        simulate.write_mpms3_file(
            tmp_path / "settling.dat",
            app="SQUID AC",
            temperature_K=temperature,
            field_oe=np.full(6, 250.0),
            chi_real=np.ones(6),
            extra_columns={"Measurement Number": [1, 2, 1, 2, 1, 2]},
        )
    )
    groups = [g.tolist() for g in repeat_groups(m)]
    assert groups == [[0, 1], [2, 3], [4, 5]]
    assert len(aggregate_repeats(m).data) == 3
