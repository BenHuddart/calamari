"""Exception types raised deliberately by calamari.

Every error calamari raises *on purpose* -- as opposed to one propagated
from numpy, pandas, or the filesystem -- derives from :class:`CalamariError`.
That gives a caller one honest way to wrap an analysis step::

    try:
        m = calamari.load(path)
    except calamari.CalamariError:
        ...  # calamari rejected this file; a bare `except Exception` here
             # would also have swallowed a typo in the caller's own code

Each concrete type *additionally* derives from the built-in exception a
caller would naturally reach for -- :class:`UnsupportedFileError` and
:class:`UnknownFormatError` are both :class:`ValueError` subclasses, the
standard-library pattern (``json.JSONDecodeError``, ``UnicodeDecodeError``)
for "the input you handed me is not the kind of thing I parse". Existing
``except ValueError`` handlers therefore keep working, and the calamari-
specific type is available to anyone who wants to be precise.

Note what is deliberately *not* here: a missing, unreadable, or
permission-denied file raises the ordinary :class:`OSError` subclass the
filesystem produced (``FileNotFoundError``, ``IsADirectoryError``,
``PermissionError``). Those are not calamari's judgement about the file's
content -- wrapping them would only hide which of the two very different
problems actually occurred (see :func:`calamari.io.load`).
"""

from __future__ import annotations

__all__ = [
    "CalamariError",
    "InputValueError",
    "InputTypeError",
    "InputKeyError",
    "ModelRegistrationError",
    "SweepInputError",
    "PlotInputError",
    "PlotTypeError",
    "UnknownModelError",
    "FitParameterError",
    "QualityInputError",
    "CurveInputError",
    "DetectionInputError",
    "FitConstraintError",
    "FileParseError",
    "FitInputError",
    "SampleValidationError",
    "SmoothingSensitivityError",
    "CatalogSelectionError",
    "UnsupportedFileError",
    "UnknownFormatError",
    "UnknownUnitError",
    "AmbiguousColumnError",
    "RepeatGroupError",
    "RepeatWeightError",
    "MeasurementFamilyError",
    "UnknownStyleError",
    "UnknownFigureFormatError",
]


class CalamariError(Exception):
    """Base class for every error calamari raises deliberately.

    Never raised directly -- catch it to mean "calamari rejected something",
    and raise one of the concrete subclasses below.
    """


class InputValueError(CalamariError, ValueError):
    """An invalid input value; also a :class:`CalamariError` and :class:`ValueError`."""


class InputTypeError(CalamariError, TypeError):
    """An unsupported input type/keyword; also :class:`CalamariError` and :class:`TypeError`."""


class InputKeyError(CalamariError, KeyError):
    """An unavailable input key/column; also :class:`CalamariError` and :class:`KeyError`."""


class ModelRegistrationError(CalamariError, ValueError):
    """A requested model registration conflicts with an existing name."""


class SweepInputError(CalamariError, ValueError):
    """Sweep segmentation or branch selection received invalid inputs."""


class PlotInputError(CalamariError, ValueError):
    """A plotting selection, unit request or presentation setting is invalid."""


class PlotTypeError(CalamariError, TypeError):
    """A plotting operation received an unsupported object or keyword."""


class UnknownModelError(CalamariError, KeyError):
    """A fit requested a model name absent from the registry."""


class FitParameterError(CalamariError, KeyError):
    """A fit setting names a parameter absent from the selected model."""


class QualityInputError(CalamariError, ValueError):
    """An outlier policy or supplied quality report is invalid for its measurement."""


class DetectionInputError(CalamariError, ValueError):
    """A detection request or sweep violates the detection input contract."""


class CurveInputError(CalamariError, ValueError):
    """A requested curve is unavailable or its sweep cannot be prepared for smoothing."""


class SampleValidationError(CalamariError, ValueError):
    """A sample normalisation quantity is invalid or required but missing."""


class FitInputError(CalamariError, ValueError):
    """Fit observations or supplied uncertainties violate the input contract."""


class FileParseError(CalamariError, ValueError):
    """A recognised instrument export has malformed required structure."""


class FitConstraintError(CalamariError, ValueError):
    """A requested fixed fit value is incompatible with its parameter limits.

    Covers parameters named in both ``fixed`` and ``limits``: a value outside
    its inclusive bounds, a non-finite fixed value, or reversed/NaN bounds.
    Raised before minimisation, including for empty or underdetermined data.
    Correct the fixed value or the bounds; this is a caller configuration
    error rather than an ordinary unsuccessful fit.
    """


class SmoothingSensitivityError(CalamariError, ValueError):
    """A sensitivity recipe cannot be assessed; correct the named setting."""


class UnsupportedFileError(CalamariError, ValueError):
    """A file was read successfully but no registered loader recognised it.

    Raised by :func:`calamari.io.load` when content sniffing (see IO-1 in
    ``docs/ARCHITECTURE.md``) finds no loader whose
    :meth:`~calamari.io.base.BaseLoader.sniff` claims the file. This means
    the bytes were readable and simply are not a magnetometry file calamari
    knows how to parse -- distinct from an :class:`OSError`, which means the
    file could not be read at all.

    Use :func:`calamari.io.is_supported_file` to test for this case without
    raising (e.g. when walking a directory of mixed content).
    """


class UnknownFormatError(CalamariError, ValueError):
    """``load(..., fmt=...)`` named a format that is not registered.

    A caller error rather than a data error: the file was never even
    consulted, because the requested ``format_name`` matches no loader in
    :class:`~calamari.io.base.LoaderRegistry`. The message lists the
    formats that *are* registered.
    """


class AmbiguousColumnError(CalamariError, ValueError):
    """Several columns could satisfy one accessor, and they are not the same
    measurement.

    Some accessors span alternative *spellings* of a single quantity --
    MPMS3's ``Magnetic Field`` can be recorded in different units for one quantity
    written two ways, so whichever a file uses is simply the answer. The
    moment is different: ``Moment`` (VSM mode) and ``DC Moment Free Ctr``
    (DC scan) are alternative *measurements*, made by different techniques
    with different systematics, and a sequence can switch between them or
    alternate row by row.

    When more than one such column carries data, picking one by priority
    would return part of the file and leave the rest as ``NaN`` --
    indistinguishable from points the instrument never recorded, so the
    loss is invisible. This is raised instead, and the message names the
    candidates and how to choose between them (see
    :meth:`calamari.measurement.Measurement.with_moment`).
    """


class UnknownUnitError(CalamariError, ValueError):
    """A data column was found, but its unit suffix is not recognised.

    Quantum Design instruments record the operator's chosen unit system in
    the data-column name itself -- a moment column is ``Moment (emu)`` or
    ``Moment (Am2)`` depending on a menu setting, and susceptibility and
    drive-field columns vary likewise.
    A :class:`~calamari.measurement.Measurement` accessor therefore has to
    read the suffix to know what it is returning.

    This is raised when the column calamari was looking for is present but
    carries a unit token it cannot convert. The alternative -- returning the
    numbers unscaled, as though they were already in the accessor's declared
    unit -- would be a silent unit error of exactly the kind DM-2 exists to
    prevent, and could be wrong by orders of magnitude (A m^2 and emu differ
    by 1000x). Returning ``None`` instead would be almost as bad: it is
    indistinguishable from "this file has no moment column at all", so a
    real measurement would vanish without explanation.

    Encountering this means the unit vocabulary needs extending, not that
    the file is bad. The message names the column, the token, and the
    tokens that are recognised.
    """


class RepeatGroupError(CalamariError, ValueError):
    """A proposed group of repeat measurements is not one setpoint.

    Repeat aggregation collapses several rows into one, so every row in a
    group has to be a measurement of the *same thing*. Two mechanisms propose
    groups -- the instrument's own ``Measurement Number`` counter, and a
    setpoint tolerance -- and neither is trusted on its own: every group is
    checked afterwards against the setpoint columns, and this is raised when
    one fails.

    That check is not defensive decoration. A ``Measurement Number`` that runs
    ``1,1,2,2`` rather than ``1,2,1,2`` pairs repeat 1 of one setpoint with
    repeat 2 of the previous one, and a counter that is really a global
    measurement index (``1,2,3,...,N``) proposes a single group containing the
    whole file. Both would otherwise average unrelated measurements together
    silently -- in a multi-frequency AC file, two different frequencies.

    Also raised when a group exceeds ``max_repeats``, which means the
    tolerance is wrong rather than that the file holds that many repeats: the
    consecutive-difference chaining underneath is documented in
    :func:`calamari.sweeps.segment_by_constant` as *deliberately* tolerating
    unbounded cumulative drift, which is right for plateau finding and wrong
    here -- a slow measure-while-sweeping ramp would otherwise chain into one
    group and average an entire sweep to a point.

    The message names the offending rows and the observed span, so the fix is
    either a tolerance that matches the file or a look at the file itself.
    """


class RepeatWeightError(CalamariError, ValueError):
    """A repeat group mixes zero and non-zero standard errors.

    A zero error is not "exact": these instruments write a **zero** standard
    error when only one scan was averaged, so a zero and a real error inside
    one group are two different conventions in one column (see the warning on
    :attr:`calamari.measurement.Measurement.moment_err_emu`). Inverse-variance
    weighting cannot span them -- a zero is an infinite weight, so the
    non-zero points would contribute nothing at all and the "weighted mean"
    would silently be whichever row happened to carry the zero.

    Falling back to an unweighted mean would be a guess about which
    convention the caller meant. The message names the group and both
    resolutions: drop the offending rows with
    :meth:`~calamari.measurement.Measurement.restrict`, or pass
    ``error='external'`` to weight nothing and take the observed scatter.
    """


class MeasurementFamilyError(CalamariError, ValueError):
    """A question about a file's measurement families could not be answered.

    A single MPMS3 file can hold more than one *family* of measurement -- a
    combined run writes one VSM row and one AC row per setpoint, and another
    interleaves AC and DC scans. This is
    raised when:

    * a caller asks :meth:`~calamari.measurement.Measurement.of_type` for a
      family the file does not contain, which is a caller error in the same
      sense as naming a column that is not there;
    * the ``Transport Action`` column carries a code calamari has no family
      for. Codes 1-5 are vendor-documented and 6 is observed; a wider set means
      the table needs extending against a real file, not guessing;
    * a row's ``Transport Action`` code and its populated columns disagree
      about which family it belongs to -- an AC-coded row carrying only a DC
      moment, say. Silence is not disagreement: a row with nothing populated
      contradicts nothing and is left alone.

    Deliberately raised only when the family question is *asked*, never during
    :func:`calamari.io.load`. A file whose codes calamari cannot read still
    parses, and its temperature, field and moment are still there to be had --
    refusing the whole file over a question the caller may never ask would be
    the wrong trade (IO-5 keeps loading strict about *dialects*, not about
    every column's vocabulary).
    """


class CatalogSelectionError(CalamariError, ValueError):
    """A catalog selection did not describe the rows it claimed to.

    Raised by :class:`calamari.catalog.Catalog` when a selection is a
    *claim* and the claim fails: :meth:`~calamari.catalog.Catalog.drop`
    was given a filename matching no row, or
    :meth:`~calamari.catalog.Catalog.load_one` matched zero or several
    files. In both cases silently returning the surviving rows would hide
    the failure -- a mistyped filename would "drop" nothing, and a sloppy
    filter would hand back an arbitrary file -- so the mismatch is
    reported by name instead, with the matching candidates listed where
    there are any.

    Not raised for a filter that merely selects nothing: an empty
    :meth:`~calamari.catalog.Catalog.where` result is an ordinary answer
    to an ordinary question, and only the methods above assert more than
    "keep whatever matches".
    """


class UnknownStyleError(CalamariError, ValueError):
    """A journal style was requested that :mod:`calamari.style` does not know.

    Raised for an unregistered journal or look name, a width that is not
    ``'single'``/``'double'``/a positive number of cm, or an alias clash on
    registration. A misspelt journal must not fall through to matplotlib's
    6.4 x 4.8 inch default: a figure that *looks* styled but is the wrong
    width for the column is exactly the failure the module exists to
    prevent, and it is invisible until the proofs come back. The message
    lists the registered names.
    """


class UnknownFigureFormatError(CalamariError, ValueError):
    """A figure was asked to save in a format matplotlib's canvas cannot write.

    Raised by :func:`calamari.plot.save` *before* any file is written. The
    check runs over every requested format up front so that
    ``save(ax, "fig1", formats=("pdf", "pgn"))`` leaves nothing behind
    rather than a PDF beside a traceback -- a half-written set of sibling
    files is the confusing outcome a multi-format save exists to avoid.
    The message names the path, the offending format, and the formats the
    current backend supports.
    """
