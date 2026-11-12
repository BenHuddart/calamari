"""Integration checks for file-based analysis workflows."""

from importlib.metadata import version

import numpy as np
import pytest
from PIL import Image

from _workflow import run_workflow


def test_file_analysis_and_export(tmp_path):
    record = run_workflow(tmp_path)
    assert record['success'] and not record['yerr_assumed']
    assert record['mass_source'] == 'user'
    assert record['version'] == record['provenance_version'] == version('calamari-magnetometry')
    # The supplied file contains an exact straight line, independent of the fit model.
    assert record['parameters']['slope'] == pytest.approx(1e-4, rel=1e-6)
    assert record['parameters']['intercept'] == pytest.approx(2e-3, rel=1e-6)
    assert all(np.isfinite(e) and e > 0 for e in record['uncertainties'].values())
    # Eight mg at 180 g/mol, two ions/f.u.; moment is in emu, mu_B in erg/G.
    expected = 0.003 / ((0.008 / 180) * 6.02214076e23 * 2 * 9.2740100783e-21)
    assert record['normalised_moment'] == pytest.approx(expected, rel=1e-8)
    np.testing.assert_allclose(record['candidate_temperature_K'], [30.0], atol=0.5, rtol=0)
    assert (tmp_path / 'measurement.pdf').read_bytes().startswith(b'%PDF-')
    with Image.open(tmp_path / 'measurement.png') as image:
        assert image.format == 'PNG' and min(image.size) > 100
        assert any(low < high for low, high in image.convert('RGB').getextrema())
