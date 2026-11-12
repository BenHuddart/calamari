"""Stable descriptive labels and informative software-identity failures."""

from functools import partial
from importlib import metadata

import pytest

from calamari._provenance import calamari_version, describe_callable


class Window:
    def __get__(self, instance, owner):
        return self

    def __call__(self, value):
        return 2.0

    def __repr__(self):
        raise AssertionError('repr must not be evaluated')


def test_callable_labels_are_stable_and_descriptive():
    expected = f'{__name__}.Window'
    assert describe_callable(Window()) == describe_callable(Window()) == expected
    assert describe_callable(Window().__call__) == expected + '.__call__'
    assert describe_callable(partial(Window(), 1)) == f'partial({expected})'
    assert describe_callable(Window) == expected


def test_qualified_names_distinguish_functions():
    def first(value):
        return value
    def second(value):
        return value
    assert describe_callable(first) != describe_callable(second)
    assert describe_callable(first).endswith('.<locals>.first')


def test_version_fallback_is_only_for_missing_distribution(monkeypatch):
    assert calamari_version() == metadata.version('calamari-magnetometry')
    def missing(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(metadata, 'version', missing)
    assert calamari_version() == 'unknown'
    def broken(name):
        raise RuntimeError('invented failure')
    monkeypatch.setattr(metadata, 'version', broken)
    with pytest.raises(RuntimeError, match='invented failure'):
        calamari_version()


def test_equivalent_batch_callables_preserve_results_and_resolved_settings(tmp_path):
    import pandas as pd

    import calamari
    from _workflow import write_inputs

    write_inputs(tmp_path)
    measurement = calamari.load(tmp_path / 'transition.dat')
    results = [calamari.detect_transitions_all(
        [measurement], smooth_window=Window(), methods='peak', signal='moment_emu',
        noise=2e-5, outliers='ignore',
    ) for _ in range(2)]
    pd.testing.assert_frame_equal(results[0].candidates, results[1].candidates)
    assert results[0].params == results[1].params
    assert results[0].params['per_sweep'][0]['smooth_window'] == 2.0
    assert describe_callable(str.upper) == 'builtins.str.upper'
    assert describe_callable([].append) == 'builtins.list.append'
