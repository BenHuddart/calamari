"""Regression checks for deliberate public failures outside the common hierarchy."""

import numpy as np
import pandas as pd
import pytest

import calamari
from calamari.exceptions import CalamariError
from calamari.fitting import fit_windows
from calamari.repeats import combine
from calamari.sample import curie_constant_from_inverse_slope


@pytest.mark.parametrize(('call', 'builtin'), [
    (lambda: fit_windows('linear', [1., 2., 3.], [2., 4., 6.], windows=[]), ValueError),
    (lambda: fit_windows('linear', [1., 2., 3.], [2., 4., 6.], windows=[(1, 3)], typo=True), TypeError),
    (lambda: curie_constant_from_inverse_slope(0), ValueError),
    (lambda: combine([1, 2], error='invented'), ValueError),
    (lambda: calamari.Catalog(pd.DataFrame(), '.').where(missing=1), KeyError),
])
def test_public_failures_have_both_bases(call, builtin):
    with pytest.raises(builtin) as caught:
        call()
    assert isinstance(caught.value, CalamariError)


def test_static_sweep_refusal_is_catchable(tmp_path):
    path = calamari.simulate.write_mpms3_file(
        tmp_path / 'invented.dat', app='SQUID DC', temperature_K=np.full(3, 10.),
        field_oe=np.full(3, 500.), dc_moment_emu=np.full(3, .001),
    )
    with pytest.raises(ValueError) as caught:
        calamari.detect_transitions(calamari.load(path), outliers='ignore')
    assert isinstance(caught.value, CalamariError)
