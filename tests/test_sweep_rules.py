"""Independent numerical checks for common sweep preparation rules."""

import numpy as np
import pytest

from calamari._sweep_rules import frequency_clusters, sweep_backtracking


def test_frequency_groups_do_not_chain_tolerance_drift():
    values = np.array([100.0, 100.0, 100.5, 101.0, 101.5])
    assert frequency_clusters(values, rtol=0.01) == [100.0, 101.5]
    assert frequency_clusters(values[::-1], rtol=0.01) == [100.0, 101.5]


@pytest.mark.parametrize('values,expected', [
    ([0., 2., 1.5, 4.], (4., 0.5)),
    ([4., 1.5, 2., 0.], (4., 0.5)),
    ([2., 2., 2.], (0., 0.)),
    ([0., 1., 1., 2.], (2., 0.)),
])
def test_backtracking_is_travel_against_overall_direction(values, expected):
    assert sweep_backtracking(np.array(values)) == expected


def test_detection_rounding_preserves_existing_extreme_frequency_behavior(tmp_path):
    from calamari import load, simulate
    from calamari.transitions._measurement import _select_frequency

    path = simulate.write_mpms3_file(
        tmp_path / 'synthetic.dat', temperature_K=[5., 6.], field_oe=[100., 100.],
        chi_real=[1e-4, 1.1e-4], ac_frequency_hz=[1e307, 1e307],
    )
    # Preserve the old round-overflow outcome rather than changing policy during extraction.
    with np.errstate(over='ignore', invalid='ignore'):
        mask, selected = _select_frequency(load(path), None)
    assert selected == np.inf
    np.testing.assert_array_equal(mask, [True, True])


def test_quality_still_excludes_missing_and_infinite_frequencies():
    from calamari.quality import _frequency_clusters

    assert _frequency_clusters(np.array([np.nan, np.inf, 100., 100.])) == [100.]
