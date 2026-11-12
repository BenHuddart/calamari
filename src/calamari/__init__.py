"""Calamari -- Python tools for analysing SQUID magnetometry data.

A short tour
------------
Calamari turns a Quantum Design MPMS 3 file (sniffed by
content, not extension) into a format-agnostic
:class:`~calamari.measurement.Measurement`
(:mod:`calamari.io`); pairs it with a physical
:class:`~calamari.sample.Sample`
(:mod:`calamari.sample`) to convert raw emu readings into Bohr magnetons
per ion/formula-unit or molar/mass susceptibility; indexes a directory of
files into a de-duplicated inventory (:mod:`calamari.catalog`); finds
candidate phase-transition features in a swept sequence
(:mod:`calamari.transitions`), and describes how they hold up across
smoothing windows (:mod:`calamari.scales`); fits a model with iminuit
(:mod:`calamari.fitting`), including a stated model through one
criterion's points across a family of sweeps (:mod:`calamari.boundary`);
and plots the result on a matplotlib ``Axes``
(:mod:`calamari.plot`). :mod:`calamari.simulate`
generates synthetic MPMS3 files and physics-shaped signals for
testing and examples -- no real instrument file is required to try any of
this.

Quickstart
----------
>>> import calamari
>>> m = calamari.load("run.dat")                            # doctest: +SKIP
>>> sample = calamari.Sample(name="my compound", mass_mg=10.0)
>>> result = calamari.fit("curie_weiss", m.temperature, m.chi_real)  # doctest: +SKIP
>>> print(result.summary())                                 # doctest: +SKIP
"""

from __future__ import annotations

from calamari import constants, exceptions, plot, quality, repeats, simulate, style, sweeps
from calamari._version import __version__
from calamari.boundary import BoundaryFit, fit_boundary
from calamari.catalog import (
    Catalog,
    build_catalog,
    deduplicated,
    find_files,
    logbook,
    logbook_line,
    select_representative,
    summarise,
)
from calamari.exceptions import (
    CalamariError,
    CatalogSelectionError,
    CurveInputError,
    DetectionInputError,
    FileParseError,
    FitConstraintError,
    FitInputError,
    FitParameterError,
    MeasurementFamilyError,
    PlotInputError,
    PlotTypeError,
    QualityInputError,
    RepeatGroupError,
    RepeatWeightError,
    SampleValidationError,
    UnknownFormatError,
    UnknownModelError,
    UnknownStyleError,
    UnsupportedFileError,
)
from calamari.fitting import (
    MODELS,
    BreakpointResult,
    FitResult,
    FitWindowsResult,
    fit,
    fit_breakpoint,
    fit_windows,
    register_model,
)
from calamari.io import Measurement, is_supported_file, load, sniff_format
from calamari.quality import QualityReport
from calamari.repeats import (
    AggregatedMeasurement,
    aggregate_repeats,
    combine,
    repeat_groups,
)
from calamari.sample import Sample, mu_eff_from_curie_constant
from calamari.scales import SmoothingScanResult, scan_smoothing
from calamari.transitions import (
    CANDIDATE_COLUMNS,
    CHAIN_COLUMNS,
    DETECTION_COLUMNS,
    METHODS,
    TransitionResult,
    detect_transitions,
    detect_transitions_all,
    find_transitions,
    smooth_derivative,
)

__all__ = [
    "__version__",
    "load",
    "sniff_format",
    "is_supported_file",
    "CalamariError",
    "CurveInputError",
    "DetectionInputError",
    "PlotInputError",
    "PlotTypeError",
    "UnknownModelError",
    "FitParameterError",
    "QualityInputError",

    "CatalogSelectionError",
    "FitConstraintError",
    "FileParseError",
    "FitInputError",
    "SampleValidationError",
    "UnsupportedFileError",
    "UnknownFormatError",
    "UnknownStyleError",
    "RepeatGroupError",
    "RepeatWeightError",
    "MeasurementFamilyError",
    "Measurement",
    "Catalog",
    "AggregatedMeasurement",
    "QualityReport",
    "aggregate_repeats",
    "repeat_groups",
    "combine",
    "Sample",
    "mu_eff_from_curie_constant",
    "find_files",
    "summarise",
    "logbook",
    "logbook_line",
    "build_catalog",
    "deduplicated",
    "select_representative",
    "find_transitions",
    "smooth_derivative",
    "detect_transitions",
    "detect_transitions_all",
    "TransitionResult",
    "METHODS",
    "CANDIDATE_COLUMNS",
    "DETECTION_COLUMNS",
    "CHAIN_COLUMNS",
    "scan_smoothing",
    "SmoothingScanResult",
    "fit",
    "FitResult",
    "fit_windows",
    "FitWindowsResult",
    "fit_breakpoint",
    "BreakpointResult",
    "fit_boundary",
    "BoundaryFit",
    "MODELS",
    "register_model",
    "constants",
    "exceptions",
    "simulate",
    "sweeps",
    "repeats",
    "quality",
    "plot",
    "style",
]
