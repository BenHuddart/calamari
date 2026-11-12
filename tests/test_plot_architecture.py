"""Presentation boundaries and the stable public paths of prepared results."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import pickle
from pathlib import Path

import numpy as np

import calamari
from calamari import _plot_data, plot


def test_presentation_import_boundaries():
    root = Path(calamari.__file__).parent
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        presentation = relative.parts[0] == "plot" or relative == Path("style.py")
        reexports = relative == Path("__init__.py")
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parent = ".".join(("calamari", *relative.parts[:-1]))
                    module = importlib.util.resolve_name("." * node.level + module, parent)
                names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
            else:
                continue
            for name in names:
                if name == "matplotlib" or name.startswith("matplotlib."):
                    assert presentation, f"{relative} imports {name}"
                if name in {"calamari.plot", "calamari.style"} or name.startswith("calamari.plot."):
                    assert presentation or reexports, f"{relative} imports {name}"


def test_public_plot_paths_and_preparation_identity():
    for name in plot.__all__:
        value = getattr(plot, name)
        if inspect.isfunction(value) or inspect.isclass(value):
            assert value.__module__ == "calamari.plot"
            assert pickle.loads(pickle.dumps(value)) is value
    for name in ("PlotSeries", "PlotGrid", "MapOutlier", "chi_series", "chi_map_grid"):
        assert getattr(plot, name) is getattr(_plot_data, name)


def test_prepared_series_pickle_preserves_arrays_and_labels():
    series = plot.PlotSeries(
        np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([0.1, 0.2]), "T (K)", "M (emu)", None
    )
    restored = pickle.loads(pickle.dumps(series))
    assert type(restored) is plot.PlotSeries
    for field in ("x", "y", "yerr"):
        np.testing.assert_array_equal(getattr(restored, field), getattr(series, field))
    assert (restored.xlabel, restored.ylabel, restored.label) == ("T (K)", "M (emu)", None)
