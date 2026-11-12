"""Synthetic regressions for required export structure and field preservation."""

import numpy as np
import pytest

from calamari import FileParseError, is_supported_file, load, simulate, sniff_format


@pytest.fixture
def export_lines():
    arguments = dict(temperature_K=[5., 6.], field_oe=[100., 100.])
    return simulate.mpms3_lines(**arguments, vsm_moment_emu=[0.01, 0.02])


@pytest.mark.parametrize('damage', ['header', 'data', 'titles', 'order', 'duplicate', 'extra'])
def test_malformed_recognised_export_raises(tmp_path, export_lines, damage):
    lines = export_lines.copy()
    data_index = lines.index('[Data]')
    if damage == 'header':
        lines.remove('[Header]')
    elif damage == 'data':
        lines.remove('[Data]')
    elif damage == 'titles':
        lines = lines[:data_index+1]
    elif damage == 'order':
        lines.remove('[Data]')
        lines.insert(0, '[Data]')
    elif damage == 'duplicate':
        lines.insert(data_index, '[Data]')
    else:
        lines[-1] += ',123'
    path = tmp_path / 'synthetic.dat'
    path.write_text('\n'.join(lines))
    assert sniff_format(path) == 'MPMS3'
    with pytest.raises(FileParseError, match='synthetic.dat'):
        load(path)
    assert not is_supported_file(path)


def test_valid_empty_export_preserves_columns(tmp_path, export_lines):
    path = tmp_path / 'synthetic.dat'
    index = export_lines.index('[Data]')
    path.write_text('\n'.join(export_lines[:index+2]))
    measurement = load(path)
    assert measurement.data.empty
    assert 'Temperature (K)' in measurement.data.columns
    assert is_supported_file(path)


def test_surplus_empty_fields_preserve_values(tmp_path, export_lines):
    path = tmp_path / 'synthetic.dat'
    path.write_text('\n'.join(export_lines))
    expected = load(path)
    index = export_lines.index('[Data]')
    for row in range(index+2, len(export_lines)):
        export_lines[row] += ',, '
    path.write_text('\n'.join(export_lines))
    actual = load(path)
    assert actual.data.equals(expected.data)
    np.testing.assert_array_equal(actual.temperature, [5., 6.])
    np.testing.assert_array_equal(actual.moment_emu, [0.01, 0.02])
    assert is_supported_file(path)


@pytest.mark.parametrize('position', ['middle', 'trailing'])
def test_populated_untitled_fields_are_rejected(tmp_path, export_lines, position):
    index = export_lines.index('[Data]') + 1
    if position == 'trailing':
        export_lines[index] += ','
        export_lines[-1] += ',9'
    else:
        for number in range(index, len(export_lines)):
            fields = export_lines[number].split(',')
            fields.insert(1, '' if number == index else '9')
            export_lines[number] = ','.join(fields)
    path = tmp_path / 'synthetic.dat'
    path.write_text('\n'.join(export_lines))
    with pytest.raises(FileParseError):
        load(path)


def test_catalogue_retains_malformed_candidates(tmp_path, export_lines):
    from calamari import build_catalog
    from calamari.catalog import find_files

    bad = tmp_path / 'synthetic.dat'
    bad.write_text('\n'.join(export_lines[:export_lines.index('[Data]')]))
    (tmp_path / 'unrelated.txt').write_text('unrelated')
    assert find_files(tmp_path) == [bad]
    catalog = build_catalog(tmp_path)
    assert catalog.df.empty
    assert catalog.n_scanned == 1
    assert 'FileParseError' in str(catalog.parse_errors)
