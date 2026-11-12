"""Least-squares fitting with iminuit for Calamari.

Re-exports the model registry, the built-in model functions, the
:func:`~calamari.fitting.engine.fit` entry point with its
:class:`~calamari.fitting.engine.FitResult`, and
:func:`~calamari.fitting.windows.fit_windows`, which repeats one fit over
several x-ranges to show whether a parameter depends on the range chosen,
and :func:`~calamari.fitting.breakpoint.fit_breakpoint`, which locates a
change of slope and compares it with no change at all.
See :mod:`calamari.fitting.models` for the model catalogue and
:mod:`calamari.fitting.engine` for the fit driver and error conventions.

The ``curie_weiss`` model includes a constant susceptibility background:

.. math::

    \\chi(T) = \\chi_0 + \\frac{C}{T-\\theta}.

The ``inverse_curie_weiss`` model fits :math:`1/\\chi=(T-\\theta)/C` and has no
background term. An omitted constant background can make its fitted Weiss
temperature depend on the fitting range. Fitting ``curie_weiss`` with ``chi0``
free accounts for that contribution; other model inadequacies can still cause
range dependence. :func:`~calamari.fitting.windows.fit_windows` records that sensitivity.

Fixed parameters and limits
---------------------------
A parameter named in both ``fixed`` and ``limits`` must satisfy the limits
exactly, with inclusive endpoints (including equal bounds). ``None`` or an
outward infinity leaves that side open. For parameters in both mappings,
fixed values must be finite and bounds ordered without NaN. An incompatible request raises
:class:`~calamari.exceptions.FitConstraintError` (also a ``ValueError``)
before minimisation, even for empty or underdetermined data. Correct the
fixed value or the bounds. ``fit_windows`` propagates this caller error;
numerical fit failures remain ordinary ``success=False`` results.

Model and parameter selection
-----------------------------
An unknown model raises :class:`~calamari.exceptions.UnknownModelError`; an
unknown parameter in ``initial``, ``fixed`` or ``limits`` raises
:class:`~calamari.exceptions.FitParameterError`. Both are also ``KeyError``
and :class:`~calamari.exceptions.CalamariError`. These caller errors remain
distinct from ordinary unsuccessful numerical fits.

Quickstart
----------
>>> import numpy as np
>>> from calamari.fitting import fit
>>> T = np.linspace(5, 300, 60)
>>> chi = 2.5 / (T + 8.0)                      # Curie-Weiss, theta = -8 K
>>> result = fit("curie_weiss", T, chi)
>>> round(result.parameters["theta"], 1)
-8.0
"""

from __future__ import annotations

from calamari.fitting.breakpoint import BreakpointResult, fit_breakpoint
from calamari.fitting.engine import FitResult, fit
from calamari.fitting.models import (
    MODELS,
    ModelDefinition,
    bent_cable,
    brillouin,
    curie_weiss,
    inverse_curie_weiss,
    linear,
    register_model,
    segmented_linear,
)
from calamari.fitting.windows import FitWindowsResult, fit_windows

__all__ = [
    "MODELS",
    "BreakpointResult",
    "FitResult",
    "FitWindowsResult",
    "ModelDefinition",
    "bent_cable",
    "brillouin",
    "curie_weiss",
    "fit",
    "fit_breakpoint",
    "fit_windows",
    "inverse_curie_weiss",
    "linear",
    "register_model",
    "segmented_linear",
]
