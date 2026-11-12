"""Explicit frequency selection has one acceptance rule across analysis and maps."""

import numpy as np
import pandas as pd
import pytest

import calamari
from calamari import plot, simulate
from calamari.exceptions import DetectionInputError, PlotInputError


@pytest.fixture
def sweeps(tmp_path):
    t = np.linspace(2., 20., 61)
    return [calamari.load(simulate.write_mpms3_file(
        tmp_path / f"invented_{i}.dat", app="SQUID AC", temperature_K=t,
        field_oe=np.full_like(t, field), chi_real=np.exp(-((t-10.)/2.)**2),
        ac_frequency_hz=np.full_like(t, 100.),
    )) for i, field in enumerate([500., 1000.])]


def test_close_frequency_request_preserves_data_and_resolved_frequency(sweeps):
    exact = calamari.detect_transitions(sweeps[0], frequency=100., outliers='ignore')
    near = calamari.detect_transitions(sweeps[0], frequency=100.05, outliers='ignore')
    pd.testing.assert_frame_equal(exact.candidates, near.candidates)
    assert near.params['frequency_hz'] == 100.
    exact_grid = plot.chi_map_grid(sweeps, frequency=100.)
    near_grid = plot.chi_map_grid(sweeps, frequency=100.05)
    assert near_grid.frequency_hz == exact_grid.frequency_hz == 100.
    np.testing.assert_equal(near_grid.values, exact_grid.values)


@pytest.mark.parametrize('frequency', [100.5, 0., -1., np.nan, np.inf, True, 1+2j])
def test_invalid_or_unmatched_requests_raise_in_both_paths(sweeps, frequency):
    with pytest.raises(DetectionInputError):
        calamari.detect_transitions(sweeps[0], frequency=frequency)
    with pytest.raises(PlotInputError):
        plot.chi_map_grid(sweeps, frequency=frequency)


def test_explicit_frequency_requires_available_ac_data(tmp_path):
    t = np.linspace(2., 20., 21)
    m = calamari.load(simulate.write_mpms3_file(
        tmp_path / 'invented_dc.dat', app='SQUID DC', temperature_K=t,
        field_oe=np.full_like(t, 500.), dc_moment_emu=np.ones_like(t),
    ))
    with pytest.raises(DetectionInputError, match='AC signal'):
        calamari.detect_transitions(m, frequency=100.)


def test_ambiguous_and_boundary_matching():
    from calamari._sweep_rules import select_frequency
    assert select_frequency([100.], 100.1, rtol=.001, error_type=ValueError) == 100.
    with pytest.raises(ValueError, match='matches none'):
        select_frequency([100.], 100.101, rtol=.001, error_type=ValueError)
    with pytest.raises(ValueError, match='ambiguously'):
        select_frequency([100., 100.15], 100.075, rtol=.001, error_type=ValueError)
