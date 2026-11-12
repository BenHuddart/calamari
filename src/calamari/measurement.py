"""Measurements with explicit quantities, units and scientific choices.

Loaders preserve the parsed table and metadata. Accessors resolve exact column
base names and unit suffixes for MPMS3 DC, VSM and AC measurements. Selection
and moment choice derive new views; caller-owned tables remain accessible.
See docs/USER_GUIDE.md for supported exports and docs/METHODS.md for conversions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from calamari.constants import (
    EMU_PER_A_M2,
    EMU_PER_OE_PER_A_M2_PER_TESLA,
    OE_PER_TESLA,
)
from calamari.exceptions import (
    AmbiguousColumnError,
    CalamariError,
    InputValueError,
    MeasurementFamilyError,
    UnknownUnitError,
)

#: Split the trailing unit suffix, accepting optional whitespace before it.
_UNIT_SUFFIX_RE = re.compile(r"^(?P<base>.*?)\s*\((?P<unit>[^()]*)\)$")


def _split_unit_suffix(column: str) -> tuple[str, str | None]:
    """Split a column name into ``(base, unit)``; unit is None if absent."""
    match = _UNIT_SUFFIX_RE.match(column.strip())
    if match is None:
        return column.strip(), None
    return match.group("base"), match.group("unit")


def _columns_with_base(data: pd.DataFrame, bases: Sequence[str]) -> list:
    """Columns of ``data`` whose base name matches one of ``bases``, in file order.

    The one matching rule behind every "which column is X?" question in this
    module: unit suffix stripped, base compared exactly (case-insensitively),
    never by substring -- see :meth:`Measurement._resolve_column` for why
    substring matching is unsafe against the real column vocabulary. Returns
    the column labels as they appear in ``data.columns``.
    """
    wanted = {b.lower() for b in bases}
    return [c for c in data.columns if _split_unit_suffix(str(c))[0].lower() in wanted]


# ---------------------------------------------------------------------------
# Column vocabulary.
#
# Bases are matched EXACTLY (case-insensitively) -- see
# Measurement._resolve_column for why substring matching is unsafe against
# the real column set. Unit tables map a lower-cased unit token to the
# factor converting it to calamari's canonical CGS-Gaussian unit.
#
# Add spellings only with documentary or observed-export evidence and a
# synthetic regression; retain the detailed source mapping with the maintainer.
# ---------------------------------------------------------------------------

#: Temperature: only ever Kelvin on these instruments.
_TEMPERATURE_BASES = ("Temperature",)
_TEMPERATURE_UNITS = {"k": 1.0}

#: Applied field recorded by MPMS3.
_FIELD_BASES = ("Magnetic Field",)
#: Oe is canonical; the operator may instead select Tesla.
_FIELD_UNITS = {"oe": 1.0, "t": OE_PER_TESLA}

# Populated VSM and free-centre DC moments require explicit choice if both exist.
_MOMENT_BASES = ("Moment", "DC Moment Free Ctr")
#: emu is canonical; the operator may instead select A m^2.
_MOMENT_UNITS = {"emu": 1.0, "am2": EMU_PER_A_M2}

# Recorded in-phase and out-of-phase susceptibility components.
_CHI_REAL_SUSCEPTIBILITY_BASES = ("AC X'",)
_CHI_IMAG_SUSCEPTIBILITY_BASES = ("AC X''",)
#: emu/Oe is canonical; the operator may instead select A m^2 / T.
_SUSCEPTIBILITY_UNITS = {
    "emu/oe": 1.0,
    "am2/t": EMU_PER_OE_PER_A_M2_PER_TESLA,
}

# Per-point standard errors paired explicitly with their source moment.
_MOMENT_ERROR_BASES = {
    "Moment": "M. Std. Err.",
    "DC Moment Free Ctr": "DC Moment Err Free Ctr",
    "DC Moment Fixed Ctr": "DC Moment Err Fixed Ctr",
}

#: Errors paired with the recorded susceptibility components.
_CHI_REAL_ERROR_BASES = {"AC X'": "AC X' Std Err."}
_CHI_IMAG_ERROR_BASES = {"AC X''": "AC X'' Std Err."}

#: AC drive frequency. NOT a bare ``Frequency`` -- see
#: :attr:`Measurement.ac_frequency` for why that column is a different
#: physical quantity.
_AC_FREQUENCY_BASES = ("AC Frequency",)
_FREQUENCY_UNITS = {"hz": 1.0}

#: AC drive-field amplitude; shares the field unit table.
_AC_DRIVE_BASES = ("AC Drive",)

#: Temperature span (K) above which a run is treated as a temperature
#: sweep rather than a nominally isothermal measurement. 0.5 K is a
#: little larger than typical MPMS setpoint stability, so a run held "at
#: temperature" stays below it while a genuine T-scan clears it easily.
#: Generic default -- tune per instrument via ``classify_scan``.
SCAN_T_TOL_K = 0.5

#: Field span (Oe) above which a run is treated as a field sweep rather
#: than a nominally constant-field measurement. 10 Oe comfortably
#: absorbs magnet setpoint jitter while remaining tiny next to a real
#: H-scan. Generic default -- tune per instrument via ``classify_scan``.
SCAN_H_TOL_OE = 10.0

#: MPMS3's per-row record of the transport activity. The mapping below includes
#: measurement and housekeeping codes; DC code 6 is supported by observed exports.
#:
#: :data:`_TRANSPORT_ACTION_FAMILIES` below is the one home for the codes:
#: ``calamari.repeats`` classifies rows through :func:`row_families` rather than
#: reading this column itself, so the two modules cannot disagree about which
#: rows are measurements.
_TRANSPORT_ACTION_BASES = ("Transport Action",)

#: Codes that are instrument housekeeping rather than measurements: the transport
#: positioning itself, not measuring the sample. Such a row carries setpoint values
#: (a temperature, a field) but no moment.
#:
#: Summary exports can contain off-setpoint housekeeping rows. Preserve them
#: in data, but exclude them from measurement spans and scan classification.
_HOUSEKEEPING_TRANSPORT_ACTIONS = (2.0, 3.0, 4.0)

#: Label for a row that is housekeeping rather than a measurement. Not a
#: measurement family: it never appears in :attr:`Measurement.measurement_types`
#: and :meth:`Measurement.split_by_type` does not return a bucket for it.
HOUSEKEEPING = "housekeeping"

#: ``Transport Action`` code -> measurement family. The strongest evidence a file
#: offers about which kind of measurement a row is, because the instrument states
#: it per row rather than leaving it to be inferred.
_TRANSPORT_ACTION_FAMILIES = {
    1.0: "VSM",
    2.0: HOUSEKEEPING,
    3.0: HOUSEKEEPING,
    4.0: HOUSEKEEPING,
    5.0: "AC",
    6.0: "DC",
}

#: Precedence for the composite ``measurement_type`` label of a file holding more
#: than one family: ``'VSM+AC'``, never ``'AC+VSM'``.
#:
#: Fixed rather than file order, and that is the whole reason it exists. A
#: combined run's interleave order flips between setpoints (VSM,AC | AC,VSM | ...),
#: so file order would label two identical runs differently -- and a catalog
#: grouping by measurement type would split one experiment into two groups for no
#: physical reason.
FAMILY_ORDER = ("DC", "VSM", "AC")

#: Populated signal columns classify rows where no transport code is supplied.
_VSM_ROW_BASES = ("Moment", "M. Quad. Signal")
_DC_ROW_BASES = ("DC Moment Free Ctr", "DC Moment Fixed Ctr")
_AC_ROW_BASES = (
    "AC X'",
    "AC X''",
    "AC Moment",
    "AC Susceptibility",
)


def _bases_populated(data: pd.DataFrame, bases: Sequence[str]) -> pd.Series:
    """Boolean per row: does any column with one of these bases carry a value?"""
    columns = _columns_with_base(data, bases)
    if not columns:
        return pd.Series(False, index=data.index)
    return data[columns].notna().any(axis=1)


def row_families(data: pd.DataFrame, *, path: object = None) -> pd.Series:
    """Classify each row of ``data`` as ``'VSM'``, ``'DC'``, ``'AC'``,
    :data:`HOUSEKEEPING`, or ``'unknown'``.

    The rule is IO-4's, applied one level down from measurement-type inference:
    **prefer the evidence the file states outright, fall back to what the data
    shows, and let each check the other.**

    ``Transport Action`` decides where the instrument writes it, because that is
    the file's per-row statement of what the transport was doing.
    Where it is absent, the populated columns decide
    instead, and the same holds per row for a *blank* cell, which states
    nothing. Where a code is present, the columns still get a say: a row whose
    code and columns name *different* families raises, because one of the two
    is wrong and calamari cannot tell which. A row that populates nothing
    contradicts nothing and is left as its code says.

    Returns a Series aligned to ``data``, so it can be used directly as a mask.

    Raises
    ------
    MeasurementFamilyError
        For an unrecognised ``Transport Action`` code, or a row whose code and
        columns disagree.
    """
    where = f"{path}: " if path is not None else ""
    families = {
        "VSM": _bases_populated(data, _VSM_ROW_BASES),
        "DC": _bases_populated(data, _DC_ROW_BASES),
        "AC": _bases_populated(data, _AC_ROW_BASES),
    }
    # A row is only *claimed* by the columns when exactly one family is
    # populated. Two at once is a row we cannot speak for -- and is exactly what
    # a caller sees on the rare file that writes both moments on one row.
    claim = pd.Series("unknown", index=data.index, dtype=object)
    total = sum(f.astype(int) for f in families.values())
    for name, populated in families.items():
        claim[populated & (total == 1)] = name

    action_columns = _columns_with_base(data, _TRANSPORT_ACTION_BASES)
    if not action_columns:
        return claim.rename("family")
    action_column = action_columns[0]

    codes = data[action_column].to_numpy(dtype=float)
    unknown = sorted(
        {c for c in codes if np.isfinite(c) and c not in _TRANSPORT_ACTION_FAMILIES}
    )
    if unknown:
        known = ", ".join(
            f"{code:g} = {family}"
            for code, family in sorted(_TRANSPORT_ACTION_FAMILIES.items())
        )
        raise MeasurementFamilyError(
            f"{where}{action_column!r} carries the code(s) "
            f"{', '.join(f'{c:g}' for c in unknown)}, which calamari has no "
            f"measurement family for. Known codes: {known}. "
            "Inspect .data; classification requires an evidenced mapping for "
            "every populated code. Report an unsupported export using a "
            "minimal synthetic example."
        )

    stated = pd.Series(
        [_TRANSPORT_ACTION_FAMILIES.get(c, "unknown") for c in codes],
        index=data.index,
        dtype=object,
    ).rename("family")

    # A BLANK code states nothing, so it cannot decide -- and cannot disagree
    # either. Such a row falls back to what its columns claim, which is the
    # same rule as a file with no Transport Action column at all, applied per
    # row. (Treating the blank as a family of its own made a row with a real
    # moment beside an empty code look like a contradiction, and refused a
    # file over the one cell that said least.)
    blank = stated == "unknown"
    if blank.any():
        stated = stated.mask(blank, claim)

    # The cross-check. Only a row that claims a DIFFERENT family disagrees;
    # a row claiming nothing is silent, not contradictory, and a housekeeping
    # row is expected to populate nothing.
    contested = (
        (claim != "unknown")
        & (stated != "unknown")
        & (stated != HOUSEKEEPING)
        & (claim != stated)
    )
    if contested.any():
        first = int(np.flatnonzero(contested.to_numpy())[0])
        raise MeasurementFamilyError(
            f"{where}row {first} is coded {codes[first]:g} in {action_column!r}, "
            f"which is a {stated.iloc[first]} measurement, but the columns it "
            f"populates are {claim.iloc[first]} columns "
            f"({int(contested.sum())} row(s) disagree this way). One of the two "
            "is wrong and calamari cannot tell which, so it refuses rather than "
            "silently picking the source that happens to be checked first. "
            "Inspect .data for these rows and check the transport codes against "
            "the populated quantities. Report an unsupported column mapping "
            "using a minimal synthetic example."
        )
    return stated


def _compose_type(families: Sequence[str]) -> str | None:
    """Join families into a composite label in :data:`FAMILY_ORDER`, or None."""
    present = {f for f in families if f not in (HOUSEKEEPING, "unknown")}
    ordered = [f for f in FAMILY_ORDER if f in present]
    return "+".join(ordered) if ordered else None


# ``eq=False``: the generated ``__eq__`` compares fields as a tuple, and two
# pandas DataFrames compare ELEMENTWISE, so ``m.restrict() == m.restrict()``
# (and ``m in [...]``) raised ``ValueError`` whenever the frames were not the
# identical object. Identity is also the honest semantics -- "are these the
# same measurement?" already has a considered answer in
# ``catalog._content_key`` -- and ``eq=False`` restores ``__hash__``, so a
# ``Measurement`` can live in a set or dict again.
@dataclass(repr=False, eq=False)
class Measurement:
    """A single parsed magnetometry measurement file.

    Attributes
    ----------
    path : Path
        Source file path.
    data_format : str
        Loader dialect that produced this object: the ``format_name`` of
        the registered loader that claimed the file, normally ``'MPMS3'``. Always a real dialect --
        :func:`calamari.io.load` raises
        :class:`~calamari.exceptions.UnsupportedFileError` for a file no
        loader recognises rather than returning a partly-parsed object
        (IO-5), so downstream code never has to test for an "unknown"
        format. (``measurement_type`` below is a separate question and
        *can* still be ``'unknown'``: the dialect is known, but which kind
        of measurement it holds may not be.)
    measurement_type : str
        One of ``'DC'``, ``'AC'``, ``'VSM'``, or ``'unknown'``.
    file_role : str
        One of ``'summary'``, ``'raw'``, ``'diagnostic'``, ``'log'``.
    title : str or None
        Descriptive run name from TITLE; it does not identify the measurement family.
    header : dict
        Parsed header metadata (sample name/mass/comment/date and any other ``INFO`` fields, plus ``raw_lines`` and
        the ``info`` sub-dict).
    data : pandas.DataFrame
        Raw data table from the ``[Data]`` section, numeric where
        possible, with column names taken verbatim from the file.
    """

    path: Path
    data_format: str
    measurement_type: str
    file_role: str
    title: str | None
    header: dict = field(default_factory=dict)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    moment_base: str | None = None

    def _resolve_column(
        self, bases: Sequence[str], units: dict[str, float], *, quantity: str
    ) -> pd.Series | None:
        """Return a data column rescaled to ``quantity``'s canonical unit.

        Every accessor below routes through here. A column name is split
        into a **base name** and a parenthesised **unit token**
        (``"Moment (emu)"`` -> ``("Moment", "emu")``), the base is matched
        *exactly* against ``bases`` in priority order, and the value is
        scaled by whatever factor ``units`` gives for the token.

        Two deliberate properties:

        **Bases match exactly, never by substring.** ``"Temperature (K)"`` is a
        substring of ``"Min. Temperature (K)"``, ``"Field (Oe)"`` of
        ``"Max. Field (Oe)"``, and ``"AC X'"`` of both ``"AC X'' (emu/Oe)"``
        and ``"AC X' Std Err. (emu/Oe)"``. These columns can appear beside
        the quantities being resolved. Matching the base exactly makes
        every one of those collisions impossible rather than
        order-dependent.

        **An unrecognised unit raises rather than guessing.** See
        :class:`~calamari.exceptions.UnknownUnitError`.

        Parameters
        ----------
        bases : sequence of str
            Column base names to try, in priority order, case-insensitive.
        units : dict
            Maps a lower-cased unit token to the factor converting a value
            in that unit to the canonical one. A column carrying **no**
            parenthesised token is assumed to be canonical already (real
            files always write one; this only keeps a malformed or
            hand-edited header usable).
        quantity : str
            Human-readable quantity name, used only in the error message.

        Returns
        -------
        pandas.Series or None
            The rescaled column, or None if no base matched. When the
            factor is 1.0 the original column object is returned unscaled,
            so the common case allocates nothing.

        Raises
        ------
        UnknownUnitError
            If a base matched but its unit token is not in ``units``.
        """
        matches = [
            column
            for base in bases
            for column in self.data.columns
            if _split_unit_suffix(str(column))[0].lower() == base.lower()
        ]
        if not matches:
            return None

        # A matching column only counts if it carries values: an instrument
        # writes the column set for every installed application, so files
        # routinely hold empty columns for unused modes (IO-4: trust the
        # values, not the names). And when NO match carries values the
        # answer is None, not an all-NaN series -- None is the documented
        # "does this file have X?" answer (docs/USER_GUIDE.md), and an all-NaN
        # series says "yes" while containing nothing. The absent-vs-empty
        # distinction stays available through `data.columns`.
        column = next(
            (c for c in matches if self.data[c].notna().any()),
            None,
        )
        if column is None:
            return None

        col_unit = _split_unit_suffix(str(column))[1]
        if col_unit is None:
            return self.data[column]
        factor = units.get(col_unit.strip().lower())
        if factor is None:
            raise UnknownUnitError(
                f"{self.path}: column {column!r} holds the {quantity} "
                f"but its unit {col_unit.strip()!r} is not recognised. "
                f"Known units for {quantity}: "
                f"{', '.join(sorted(units))}. Inspect the column suffix and "
                "the accessor's supported units; report an unsupported unit "
                "using a minimal synthetic example."
            )
        if factor == 1.0:
            return self.data[column]
        return self.data[column] * factor

    @property
    def temperature(self) -> pd.Series | None:
        """Temperature column in Kelvin, or None if absent."""
        return self._resolve_column(
            _TEMPERATURE_BASES, _TEMPERATURE_UNITS, quantity="temperature"
        )

    @property
    def field_oe(self) -> pd.Series | None:
        """Applied field column in Oe, or None if absent.

        Converts automatically if the file records field in Tesla (the
        instrument's alternate unit selection) -- the accessor's declared
        unit is what you get, per DM-2.
        """
        return self._resolve_column(_FIELD_BASES, _FIELD_UNITS, quantity="field")

    @property
    def field_tesla(self) -> pd.Series | None:
        """Applied field column converted to Tesla, or None if absent.

        Uses the exact CGS-Gaussian factor ``OE_PER_TESLA`` (1 T = 1e4
        Oe) from :mod:`calamari.constants`; unit conversions never
        hard-code the factor at the call site.
        """
        oe = self.field_oe
        if oe is None:
            return None
        return oe / OE_PER_TESLA

    @property
    def moment_emu(self) -> pd.Series | None:
        """DC moment column in emu, or None if absent.

        Matches MPMS3 ``Moment`` / ``DC Moment Free Ctr`` bases. A file recorded in A m^2 (the
        instrument's alternate moment unit) is converted to emu here, so
        the accessor's name stays true whatever the operator selected.

        Of MPMS3's two DC-scan moments this reads the **free-centre** one
        (``DC Moment Free Ctr``), never ``DC Moment Fixed Ctr``. The
        free-centre fit leaves both sample position and moment free, which
        is the right default for a well-centred sample with a decent
        signal; for a weak moment that fit is poorly conditioned and the
        AutoTracking-constrained fixed-centre value can be the more
        trustworthy number. Read ``data['DC Moment Fixed Ctr (emu)']``
        directly for that case, and ``data['DC Free Fit']`` (0-1) to judge
        which to believe.

        If more than one moment column carries data -- a sequence that
        switched measurement mode partway, or alternated row by row --
        this raises :class:`~calamari.exceptions.AmbiguousColumnError`
        rather than picking one. Choose with :meth:`with_moment`; see
        :attr:`moment_columns` for what the file offers.
        """
        if self.moment_base is not None:
            return self._resolve_column(
                (self.moment_base,), _MOMENT_UNITS, quantity="moment"
            )

        populated = self.moment_columns
        if len(populated) > 1:
            counts = ", ".join(
                f"{name!r} ({int(self.data[name].notna().sum())} points)"
                for name in populated
            )
            raise AmbiguousColumnError(
                f"{self.path}: more than one moment column carries data -- "
                f"{counts}. These are different measurements (VSM mode vs DC "
                "scan, or free- vs fixed-centre fits), not one quantity spelled "
                "two ways, so returning either would silently leave the other's "
                "rows as NaN -- indistinguishable from points never measured. "
                "Pick one explicitly with m.with_moment(...), e.g. "
                f"m.with_moment({populated[0]!r}); m.moment_columns lists them "
                "in priority order."
            )
        return self._resolve_column(_MOMENT_BASES, _MOMENT_UNITS, quantity="moment")

    @property
    def moment_err_emu(self) -> pd.Series | None:
        """Per-point standard error on :attr:`moment_emu`, in emu, or None.

        The instrument measures this and writes it alongside every moment;
        it is the ``yerr``
        :func:`calamari.fitting.fit <calamari.fitting.engine.fit>` is built
        around, and
        supplying it is the difference between reported uncertainties that
        are absolute and ones rescaled from the residual scatter (see the
        error convention in :mod:`calamari.fitting.engine`)::

            result = calamari.fit("linear", m.field_oe, m.moment_emu,
                                  yerr=m.moment_err_emu)

        Follows :attr:`moment_emu`: whichever moment column is in use names
        its own error column, so a :meth:`with_moment` choice carries over
        and the two can never come from different measurements. Converted
        with the same unit factor as the moment, so the ratio is preserved.

        None when the file records no error for the moment in use.

        .. warning::

           A **zero** error is not the same as no error, and is common: an
           instrument reporting scan-to-scan scatter writes 0 when only one
           scan was averaged, which covers most single-shot measurements.
           Passing an all-zero ``yerr`` to a least-squares fit gives every
           point infinite weight and the result will not be meaningful.
           Check ``(err > 0).any()`` before using it as a weight.
        """
        return self._paired_error(_MOMENT_ERROR_BASES, _MOMENT_UNITS, "moment")

    @property
    def chi_real_err(self) -> pd.Series | None:
        """Recorded standard error on chi_real in emu/Oe, or None if absent.

        Resolved from the error column paired with the populated susceptibility
        component and converted using its own unit suffix. A recorded zero error
        must not be treated as an exact observation when choosing fit weights."""
        return self._ac_component_error(
            _CHI_REAL_SUSCEPTIBILITY_BASES,
            _CHI_REAL_ERROR_BASES,
        )

    @property
    def chi_imag_err(self) -> pd.Series | None:
        """Per-point standard error on :attr:`chi_imag`, in emu/Oe, or None.

        The out-of-phase counterpart of :attr:`chi_real_err`; see there.
        """
        return self._ac_component_error(
            _CHI_IMAG_SUSCEPTIBILITY_BASES,
            _CHI_IMAG_ERROR_BASES,
        )

    def _column_in_use(self, bases: Sequence[str]) -> str | None:
        """Base name of the first column from ``bases`` carrying values.

        Returned spelled as the file spells it (columns sharing a base keep
        the last spelling in file order): callers feed the answer to error
        tables keyed on the spelling.
        """
        for base in bases:
            populated = [
                c
                for c in _columns_with_base(self.data, (base,))
                if self.data[c].notna().any()
            ]
            if populated:
                return _split_unit_suffix(str(populated[-1]))[0]
        return None

    def _paired_error(
        self, error_bases: dict[str, str], units: dict[str, float], quantity: str
    ) -> pd.Series | None:
        """Error column paired with whichever moment column is in use."""
        in_use = self.moment_base or self._column_in_use(_MOMENT_BASES)
        if in_use is None:
            return None
        error_base = next(
            (v for k, v in error_bases.items() if k.lower() == in_use.lower()),
            None,
        )
        if error_base is None:
            return None
        return self._resolve_column(
            (error_base,), units, quantity=f"{quantity} standard error"
        )

    def _ac_component_error(
        self, susceptibility_bases: Sequence[str], error_bases: dict[str, str]
    ) -> pd.Series | None:
        """Resolve the error paired with a populated susceptibility component."""
        in_use = self._column_in_use(susceptibility_bases)
        if in_use is None:
            return None
        base = error_bases.get(in_use)
        if base is None:
            return None
        return self._resolve_column(
            (base,), _SUSCEPTIBILITY_UNITS, quantity="susceptibility error"
        )

    @property
    def moment_columns(self) -> tuple[str, ...]:
        """Moment columns present in ``data`` *and* carrying values.

        The candidates :attr:`moment_emu` chooses between, in priority
        order, named exactly as the file spells them. Empty when the file
        records no moment at all; length 1 for almost every file, since an
        instrument writes the column set for every application it has
        installed and the unused modes' columns sit empty.

        Length > 1 means the file holds more than one kind of moment
        measurement -- see :meth:`with_moment`.
        """
        by_base: dict[str, list[str]] = {}
        for column in self.data.columns:
            if not self.data[column].notna().any():
                continue
            by_base.setdefault(
                _split_unit_suffix(str(column))[0].lower(), []
            ).append(str(column))
        # Iterate the bases, not the file's columns: the order promised here
        # is _MOMENT_BASES priority, and a file is free to write its columns
        # in any order at all.
        return tuple(
            column
            for base in _MOMENT_BASES
            for column in by_base.get(base.lower(), [])
        )

    def with_moment(self, column: str) -> Measurement:
        """Return a NEW Measurement whose :attr:`moment_emu` reads ``column``.

        For a file recording more than one kind of moment, this is how you
        say which you want -- and, because it is a cheap derivation rather
        than a re-parse, how you get both from one load::

            m = calamari.load(path)
            vsm = m.with_moment("Moment")
            dc  = m.with_moment("DC Moment Free Ctr")

        Both are ordinary ``Measurement`` objects sharing the parsed data,
        so either can be handed to plotting, fitting or transition
        detection. It is also the way to reach MPMS3's fixed-centre moment,
        which no accessor returns by default::

            fixed = m.with_moment("DC Moment Fixed Ctr")

        Parameters
        ----------
        column : str
            Column name, with or without its unit suffix --
            ``"DC Moment Free Ctr"`` and ``"DC Moment Free Ctr (emu)"`` are
            both accepted, so a name copied from :attr:`moment_columns` or
            from ``data.columns`` works either way. The unit is still read
            from the file's own suffix and converted as usual, so this
            selects *which measurement*, never what unit it is in.

        Returns
        -------
        Measurement
            A new object; ``self`` is unchanged (DM-1). The choice survives
            :meth:`restrict`.

        Raises
        ------
        ValueError
            If no column with that base name exists in ``data``. Naming a
            column the file does not have is a caller error, and silently
            falling back to the default priority would hide it.
        """
        present = _columns_with_base(self.data, (_split_unit_suffix(column)[0],))
        if not present:
            available = ", ".join(repr(str(c)) for c in self.data.columns) or "(none)"
            raise InputValueError(
                f"{self.path}: no column named {column!r}. Available columns: "
                f"{available}."
            )
        return replace(self, moment_base=_split_unit_suffix(str(present[0]))[0])


    @property
    def chi_real(self) -> pd.Series | None:
        """Recorded AC in-phase susceptibility chi' in emu/Oe, or None if absent.

        Reads MPMS3's AC X' column and converts its unit suffix. No moment/drive
        re-derivation is performed. The recorded error is exposed separately.
        """
        return self._resolve_column(
            _CHI_REAL_SUSCEPTIBILITY_BASES, _SUSCEPTIBILITY_UNITS,
            quantity="in-phase susceptibility",
        )

    @property
    def chi_imag(self) -> pd.Series | None:
        """Recorded AC out-of-phase susceptibility chi'' in emu/Oe, or None."""
        return self._resolve_column(
            _CHI_IMAG_SUSCEPTIBILITY_BASES, _SUSCEPTIBILITY_UNITS,
            quantity="out-of-phase susceptibility",
        )


    @property
    def ac_frequency(self) -> pd.Series | None:
        """Recorded AC drive frequency in Hz, or None if absent.

        Reads AC Frequency. The bare Frequency column describes mechanical VSM
        oscillation and is deliberately excluded."""
        return self._resolve_column(
            _AC_FREQUENCY_BASES, _FREQUENCY_UNITS, quantity="AC frequency"
        )

    @property
    def ac_drive_oe(self) -> pd.Series | None:
        """Achieved peak AC drive field in Oe, or None if absent.

        Reads AC Drive, including modulation and nulling-coil contributions.
        A recorded Tesla unit is converted to Oe."""
        return self._resolve_column(
            _AC_DRIVE_BASES, _FIELD_UNITS, quantity="AC drive amplitude"
        )

    @property
    def sample_name(self) -> str | None:
        """Sample name/material from header INFO, or None."""
        return self.header.get("sample_name")

    @property
    def sample_mass_mg(self) -> float | None:
        """Operator-entered SAMPLE_MASS in milligrams, or None if unrecorded.

        The header cannot corroborate the specimen mass. Sample.from_measurement
        records file/user provenance and lets an explicit caller mass take precedence."""
        return self.header.get("sample_mass")

    @property
    def sample_molar_mass_g_per_mol(self) -> float | None:
        """Molar mass from the header (MPMS3 ``SAMPLE_MOLECULAR_WEIGHT``).

        Exposed so the value is not silently dropped, but deliberately
        **not** used by :meth:`calamari.sample.Sample.from_measurement`,
        which is where the mass *is* adopted automatically. The two are
        not as symmetric as they look:

        - A sample mass is a property of *this specimen*. Only weighing
          it can tell you, so the header may be the only record there
          is -- reading it fills a gap nothing else can.
        - A molar mass is a property of the *compound*: chemistry,
          computable from the formula, the same for every sample of the
          material. The caller always has an authoritative source, so
          preferring a number typed into an instrument wizard would be
          adopting a convenience copy over a better-known fact.

        Opting in is one visible line::

            Sample.from_measurement(
                m, molar_mass_g_per_mol=m.sample_molar_mass_g_per_mol
            )

        The ``g/mol`` in the name is **inferred**, not verified: "molecular
        weight" is numerically the molar mass in g/mol and there is no
        plausible alternative, but no real file carrying a populated value
        has been examined -- the field is routinely left blank.
        """
        return self.header.get("sample_molecular_weight")

    @property
    def comment(self) -> str | None:
        """Sample comment from header INFO, or None."""
        return self.header.get("sample_comment")


    @property
    def date_iso(self) -> str | None:
        """Recorded file creation date as ISO ``YYYY-MM-DD``.

        Parsed from the human-readable ``M/D/YYYY`` token in the
        FILEOPENTIME line. Numeric timestamps are preserved without conversion;
        the empirically supported origin does not establish timezone handling.
        """
        return self.header.get("date_iso")

    @property
    def date_epoch(self) -> float | None:
        """Recorded numeric FILEOPENTIME token, or None.

        Inspected native MPMS3 exports support seconds from 1899-12-30.
        Timezone and clock handling remain unspecified; no conversion is applied.
        Use :attr:`date_iso` for the recorded calendar date."""
        return self.header.get("fileopentime_epoch")

    @property
    def row_families(self) -> pd.Series:
        """Per-row measurement family, aligned to :attr:`data`.

        Computed on access rather than stored, so it stays true through
        :meth:`restrict`: a stored answer would go on describing the rows a
        subset no longer contains. See :func:`row_families` for the rule.
        """
        return row_families(self.data, path=self.path)

    @property
    def measurement_types(self) -> tuple[str, ...]:
        """Which measurement families this file contains, in :data:`FAMILY_ORDER`.

        ``('VSM',)`` for an ordinary VSM run; ``('VSM', 'AC')`` for a combined
        run that measured both at every setpoint. This is the structured form of
        :attr:`measurement_type`, and the one to branch on::

            if 'AC' in m.measurement_types:
                ...

        rather than ``m.measurement_type == 'AC'``, which is False for a file
        holding AC *and* something else.

        Housekeeping rows contribute nothing: they are not a measurement family.
        """
        return tuple(
            f
            for f in FAMILY_ORDER
            if f in set(self.row_families) - {HOUSEKEEPING, "unknown"}
        )

    def of_type(self, family: str) -> Measurement:
        """Return a NEW Measurement holding only one measurement family's rows.

        The way to read a combined run one measurement at a time::

            m = calamari.load(path)          # measurement_type 'VSM+AC'
            vsm = m.of_type("VSM")           # selected VSM moment
            ac = m.of_type("AC")             # selected AC response

        Both are ordinary ``Measurement`` objects sharing the parsed data, so
        either can be handed to plotting, fitting or transition detection --
        the same bargain :meth:`with_moment` offers, one level up. It is also
        how a combined VSM+DC file's :attr:`moment_emu` becomes answerable: the
        ambiguity that accessor refuses to guess at is exactly the choice this
        makes explicit.

        Parameters
        ----------
        family : str
            ``'VSM'``, ``'DC'`` or ``'AC'``. :attr:`measurement_types` lists
            what this file offers.

        Returns
        -------
        Measurement
            A new object; ``self`` is unchanged (DM-1). Its
            :attr:`measurement_type` is restamped to the single family, so it
            no longer claims to hold rows it does not. Asking for the only
            family a file has returns a full copy rather than ``self``, so the
            result is always safe to treat as a derived object.

        Raises
        ------
        MeasurementFamilyError
            If this file holds no rows of that family. Naming a family the file
            does not have is a caller error, and returning an empty Measurement
            would hide it -- an empty sweep looks like a sweep that measured
            nothing, not like a question that was wrong.
        """
        available = self.measurement_types
        if family not in available:
            offered = ", ".join(repr(f) for f in available) or "(none)"
            raise MeasurementFamilyError(
                f"{self.path}: no {family!r} rows in this file. It holds "
                f"{offered}. m.measurement_types lists the families present, "
                "and m.split_by_type() returns all of them at once."
            )
        return self.restrict(mask=(self.row_families == family).to_numpy())

    def split_by_type(self) -> dict[str, Measurement]:
        """Every measurement family in the file, as its own Measurement.

        Keyed in :data:`FAMILY_ORDER`, so iteration order is stable across
        files::

            for family, part in m.split_by_type().items():
                print(family, len(part.data))

        A file with one family returns one entry -- the honest answer, and it
        means a caller written for combined runs also works on ordinary ones
        with no special case. Housekeeping rows are in no bucket: they are not
        a measurement, and every row of :attr:`data` remains reachable from the
        original object.
        """
        return {family: self.of_type(family) for family in self.measurement_types}

    @property
    def moment_method(self) -> str | None:
        """How the moment in :attr:`moment_emu` was measured: ``'VSM'``,
        ``'DC'``, or None if the file records no moment.

        Both a VSM measurement and a DC scan yield a moment in emu, and
        :attr:`moment_emu` names its unit but not its provenance. This names
        the provenance -- the counterpart of
        :attr:`calamari.sample.Sample.mass_source`, and the answer to "how was
        the number I am plotting actually obtained?".

        ``'DC'`` means a DC scan (the instrument translating the sample through
        the pickup coils and fitting the response); ``'VSM'`` means the sample
        vibrating at fixed position. They are different measurements of the
        same quantity, with different noise and different systematics.

        The vocabulary is deliberately the same as
        :attr:`measurement_types`, so the two compose::

            m.of_type(m.moment_method)

        Follows :meth:`with_moment`: whichever moment column is in use decides
        the answer. If more than one moment column carries data and no choice
        has been made, this raises
        :class:`~calamari.exceptions.AmbiguousColumnError` exactly as
        :attr:`moment_emu` does -- if it answered anyway, the two accessors
        could describe different measurements.
        """
        base = self.moment_base
        if base is None:
            populated = self.moment_columns
            if not populated:
                return None
            if len(populated) > 1:
                # Same question, same refusal, same resolution as moment_emu.
                self.moment_emu
            base = _split_unit_suffix(populated[0])[0]
        lowered = base.lower()
        if lowered in {b.lower() for b in _DC_ROW_BASES}:
            return "DC"
        if lowered in {b.lower() for b in _VSM_ROW_BASES}:
            return "VSM"
        return None

    def _measurement_mask(self) -> pd.Series:
        """Boolean mask of the rows that are measurements, not housekeeping.

        True everywhere when no Transport Action column is supplied.

        Deliberately **not** applied to :attr:`data` or to any per-row accessor.
        Two reasons, and both were checked rather than assumed:

        * Dropping rows would break length alignment that other modules depend on
          structurally -- :func:`calamari.repeats.repeat_groups` builds its setpoint
          keys from the accessors and indexes them by positions derived from
          ``len(data)``, and :meth:`restrict`'s ``mask`` is documented against the
          same length.
        * Blanking the setpoints instead would make the accessor lie about what the
          file contains, and :meth:`restrict` drops NaN-temperature rows by its
          documented semantics -- so a housekeeping row would silently vanish
          through an unrelated ``t_range`` filter, which is exactly the invisible
          row loss :mod:`calamari.repeats` refuses on principle.

        The quantity columns need no masking anyway: the instrument leaves the
        moment blank on these rows, so they are already absent from
        :attr:`moment_emu` and friends. The only leak is the *setpoint* columns
        into aggregate descriptions, which is what this exists to plug.
        """
        columns = _columns_with_base(self.data, _TRANSPORT_ACTION_BASES)
        if not columns:
            return pd.Series(True, index=self.data.index)
        actions = self.data[columns[0]].to_numpy(dtype=float)
        return pd.Series(
            ~np.isin(actions, _HOUSEKEEPING_TRANSPORT_ACTIONS), index=self.data.index
        )

    def _measured(self, series: pd.Series | None) -> pd.Series | None:
        """``series`` restricted to measurement rows; None stays None.

        The one call every aggregate description goes through, so a housekeeping
        row cannot reach a span, a scan classification or a repeat count.
        """
        if series is None:
            return None
        return series[self._measurement_mask()]

    def _safe(self, getter, default=None):
        """Evaluate an accessor, swallowing the errors it may raise.

        Used only by :meth:`__repr__` and :meth:`summary`, which describe a
        file and must never be the thing that fails. An accessor can raise
        legitimately -- ``UnknownUnitError`` for an unrecognised unit token,
        ``AmbiguousColumnError`` when several moment columns are populated --
        and a repr that blew up on exactly the odd file you were trying to
        inspect would be worse than useless.
        """
        try:
            value = getter()
        except (CalamariError, KeyError, ValueError, TypeError):
            return default
        return default if value is None else value

    def max_repeats(self) -> int:
        """Largest number of repeat measurements of one setpoint, or 1.

        Read from the instrument's own ``Measurement Number`` column, which
        MPMS3 populates as ``1, 2, 1, 2, ...`` when each setpoint was measured
        twice. A new repetition begins wherever the counter fails to increase.

        Deliberately **counter-only**: it never falls back to guessing from
        setpoint tolerances. This is reported by :meth:`__repr__` and
        :meth:`summary`, and :func:`calamari.catalog.summarise` builds on the
        latter, so a tolerance-based detector would run over every file in a
        directory tree and announce phantom repeats wherever the tolerance
        happened not to suit. Detection that appears unbidden has to be the
        kind that cannot be wrong.

        Returns 1 for a file with no such column, a
        constant one, or genuinely no repeats. Use
        :func:`calamari.repeats.repeat_groups` for the real answer, which
        groups on the setpoints when the counter cannot help.

        Housekeeping rows are excluded before the counter is read
        (:meth:`_measured`). A locate scan sitting between two measurements is not
        a repetition of either, and one carrying a counter value that continues the
        run's sequence would otherwise extend a genuine pair into a phantom triple.
        """
        columns = _columns_with_base(self.data, ("Measurement Number",))
        if not columns:
            return 1
        values = self.data.loc[self._measurement_mask(), columns[0]].to_numpy(dtype=float)
        best = run = 0
        previous = None
        for value in values:
            if previous is not None and np.isfinite(value) and value > previous:
                run += 1
            else:
                run = 1
            previous = value if np.isfinite(value) else None
            best = max(best, run)
        return max(best, 1)

    def _ac_frequency_summary(self) -> str:
        """The distinct AC frequencies as display text, or '' when there are none.

        One formatting site shared by :meth:`__repr__` and :meth:`summary` --
        and, through the latter, :func:`calamari.catalog.logbook` -- so the
        three renderings of "what frequency was this run?" cannot drift
        apart. Lists every distinct frequency (``'97; 997'`` for a
        two-frequency interleave), never an average, which would describe
        neither measurement. Housekeeping rows are excluded first
        (:meth:`_measured`), exactly as for the T/H spans: a locate scan's
        row can carry a leftover value in the frequency column, and a
        frequency nothing was measured at does not belong in a description
        of the measurement. Never raises: a file whose frequency column
        cannot be read reports '' here, because every caller is a
        description of the file.
        """
        frequency = self._safe(lambda: self._measured(self.ac_frequency))
        if frequency is None or not frequency.notna().any():
            return ""
        unique = sorted(frequency.dropna().unique().tolist())
        return "; ".join(f"{f:.4g}" for f in unique)

    def _repr_bits(self) -> list[str]:
        """The fragments :meth:`__repr__` joins, so a subclass can extend them.

        Split out rather than inlined so
        :class:`calamari.repeats.AggregatedMeasurement` can append its own
        without either duplicating this logic or parsing the parent's string
        back apart.
        """
        bits = [f"{self.data_format}/{self.measurement_type}"]
        if self.sample_name:
            bits.append(repr(self.sample_name))
        bits.append(f"{len(self.data)} pts")

        temperature = self._safe(lambda: self._measured(self.temperature))
        if temperature is not None and temperature.notna().any():
            bits.append(f"T {temperature.min():.4g}-{temperature.max():.4g} K")
        field = self._safe(lambda: self._measured(self.field_oe))
        if field is not None and field.notna().any():
            lo, hi = field.min(), field.max()
            bits.append(
                f"H {lo:.6g} Oe" if lo == hi else f"H {lo:.6g}-{hi:.6g} Oe"
            )
        # After the setpoints (T, H), before the classification fields: the
        # frequency is part of *where* an AC run was measured. Omitted, not
        # blank, for files without one, so DC/VSM reprs are unchanged.
        frequencies = self._ac_frequency_summary()
        if frequencies:
            bits.append(f"{frequencies} Hz")
        bits.append(self._safe(lambda: self.scan_kind, "?"))
        if self.date_iso:
            bits.append(self.date_iso)
        repeats = self._safe(lambda: self.max_repeats(), 1)
        if repeats > 1:
            bits.append(f"repeats x{repeats}")
        return bits

    def __repr__(self) -> str:
        """One-line summary: what this file is, not what is in it.

        The dataclass default would print the entire header dict (including
        a verbatim copy of every header line) and the whole DataFrame --
        thousands of characters, as the first thing anyone sees in a
        notebook. This says the useful part instead; ``data`` and ``header``
        are there when wanted.
        """
        return f"<{type(self).__name__} {' | '.join(self._repr_bits())}>"

    def summary(self) -> dict:
        """A one-glance description of this measurement, as a plain dict.

        The "what am I holding?" answer: dialect, measurement type, sample,
        date, point count, temperature and field spans, AC frequencies and
        scan classification. Every value is derived from this object, so it
        costs no re-parse.

        This is the per-file half of a catalog row --
        :func:`calamari.catalog.summarise` builds on it, adding only the
        path-derived fields it needs a root directory to compute, so the two
        cannot drift apart.

        Accessors that would raise (an unrecognised unit, or a file holding
        more than one kind of moment) are reported as ``None`` rather than
        propagating: this method describes a file, including a problematic
        one.

        Returns
        -------
        dict
            Keys: ``data_format``, ``measurement_type``, ``file_role``,
            ``sample_name``, ``sample_mass_mg``, ``date``, ``n_points``,
            ``n_measurements``, ``T_min``, ``T_max``, ``H_min_Oe``, ``H_max_Oe``,
            ``H_min_T``, ``H_max_T``, ``ac_frequencies_Hz``, ``scan_kind``,
            ``max_repeats``, ``comment``, and from
            :meth:`quality` ``n_flagged``, ``max_continuity_score`` and
            ``n_status_departures`` (None when no point could be scored).

        Notes
        -----
        ``n_points`` counts **rows**, and keeps that meaning: a row the instrument
        wrote is a row, and silently redefining a published column would be the
        kind of changed-number-that-looks-the-same this library exists to avoid.
        ``n_measurements`` is the count excluding housekeeping rows (see
        :meth:`_measurement_mask`); the two are equal for every file that has no
        such rows, which is most of them. The T/H spans and ``scan_kind``, by
        contrast, describe *the measurement* and so exclude housekeeping rows --
        they were simply wrong before, not differently defined. The AC
        frequency list excludes them for the same reason (see
        :meth:`_ac_frequency_summary`).
        """

        def span(series) -> tuple[float | None, float | None]:
            if series is None or series.notna().sum() == 0:
                return None, None
            return float(series.min()), float(series.max())

        t_min, t_max = span(self._safe(lambda: self._measured(self.temperature)))
        h_min_oe, h_max_oe = span(self._safe(lambda: self._measured(self.field_oe)))
        h_min_t, h_max_t = span(self._safe(lambda: self._measured(self.field_tesla)))

        return {
            "data_format": self.data_format,
            "measurement_type": self.measurement_type,
            "file_role": self.file_role,
            "sample_name": self.sample_name,
            "sample_mass_mg": self.sample_mass_mg,
            "date": self.date_iso,
            "n_points": int(len(self.data)),
            "n_measurements": int(self._measurement_mask().sum()),
            "T_min": t_min,
            "T_max": t_max,
            "H_min_Oe": h_min_oe,
            "H_max_Oe": h_max_oe,
            "H_min_T": h_min_t,
            "H_max_T": h_max_t,
            "ac_frequencies_Hz": self._ac_frequency_summary(),
            "scan_kind": self._safe(lambda: self.scan_kind, "unknown"),
            "max_repeats": self._safe(lambda: self.max_repeats(), 1),
            "comment": self.comment,
            **self._quality_summary(),
        }

    def _quality_summary(self) -> dict:
        """The three :meth:`quality` counts a catalog row carries.

        All None when nothing could be scored -- a raw or log file, or a
        file whose accessors refuse -- because 0 would claim the file was
        screened and found clean.
        """
        keys = ("n_flagged", "max_continuity_score", "n_status_departures")
        report = self._safe(self.quality)
        if report is None:
            return dict.fromkeys(keys)
        counts = report.summary()
        if counts["n_scored"] == 0:
            return dict.fromkeys(keys)
        return {key: counts[key] for key in keys}

    def quality(self, *, threshold: float | None = None):
        """Screen every measured point: which deserve a second look, and why.

        Returns a :class:`~calamari.quality.QualityReport`: one row per
        measured point, scored within its own measurement family (and AC
        frequency) in file order. A point is **flagged** when its value
        breaks the continuity of its neighbours -- its departure from a
        straight line through them (fitted against the swept field or
        temperature), over how far they stray from that line themselves,
        exceeds ``threshold``. A transition step or a cusp scores near 1, because
        its neighbours disagree too; an isolated glitch scores in the
        hundreds.

        Beside the flag the report carries the *evidence* for reading it --
        the point's uncertainty relative to its neighbours', its within-point
        temperature and field spread, the instrument's fit quality for the
        moment in use, status codes that depart from this file's usual
        value, SQUID range changes. None of these is a gate, because on real
        files each fires on innocent points alone (see
        :mod:`calamari.quality`)::

            report = m.quality()
            report                     # one line: how many flagged, max score
            report.flagged             # the rows, with their evidence
            m.data.loc[report.flagged.index]   # the raw rows themselves

        Nothing is removed from the Measurement (DM-1); the report only
        describes.

        Parameters
        ----------
        threshold : float, optional
            Continuity score above which a point is flagged. Default
            :data:`calamari.quality.CONTINUITY_THRESHOLD` (20), an empirical
            diagnostic default, not a calibrated false-alarm probability.

        Returns
        -------
        calamari.quality.QualityReport

        Raises
        ------
        ValueError
            If ``threshold`` is not positive.
        """
        from calamari.quality import CONTINUITY_THRESHOLD, assess

        return assess(self, threshold=CONTINUITY_THRESHOLD if threshold is None else threshold)

    def classify_scan(
        self, t_tol_K: float = SCAN_T_TOL_K, h_tol_Oe: float = SCAN_H_TOL_OE
    ) -> str:
        """Classify the sweep as a temperature scan, field scan, or other.

        A quantity is considered "~constant" when its span (max - min)
        stays below the given tolerance, i.e. within instrument setpoint
        jitter. Only quantities with at least two non-null points are
        eligible to be judged "varying".

        Housekeeping rows are excluded first (:meth:`_measured`). They are not
        measurements, and they sit wherever the transport happened to be: a real
        isothermal M(H) ending in a locate scan several kelvin away was classified
        ``'mixed'`` on the strength of that one row.

        Parameters
        ----------
        t_tol_K : float, optional
            Temperature span (K) above which temperature is treated as
            varying. Default :data:`SCAN_T_TOL_K` (0.5 K), a generic
            choice tuned on MPMS3 data; raise it for noisier thermometry.
        h_tol_Oe : float, optional
            Field span (Oe) above which field is treated as varying.
            Default :data:`SCAN_H_TOL_OE` (10 Oe).

        Returns
        -------
        str
            ``'T-scan'`` (T varies, H constant), ``'H-scan'`` (H varies,
            T constant), ``'mixed'`` (both vary), or ``'static/other'``
            (neither varies).
        """
        temperature = self._measured(self.temperature)
        field_oe = self._measured(self.field_oe)

        t_varies = False
        h_varies = False

        if temperature is not None and temperature.notna().sum() >= 2:
            t_range = temperature.max() - temperature.min()
            t_varies = bool(t_range > t_tol_K)

        if field_oe is not None and field_oe.notna().sum() >= 2:
            h_range = field_oe.max() - field_oe.min()
            h_varies = bool(h_range > h_tol_Oe)

        if t_varies and not h_varies:
            return "T-scan"
        if h_varies and not t_varies:
            return "H-scan"
        if not t_varies and not h_varies:
            return "static/other"
        return "mixed"

    @property
    def scan_kind(self) -> str:
        """Sweep classification using the default tolerances.

        Convenience wrapper over :meth:`classify_scan`; see there for the
        thresholds and their rationale. One of ``'T-scan'``, ``'H-scan'``,
        ``'mixed'``, ``'static/other'``.
        """
        return self.classify_scan()

    def restrict(
        self,
        *,
        t_range: tuple[float, float] | None = None,
        h_range: tuple[float, float] | None = None,
        mask: object | None = None,
    ) -> Measurement:
        """Return a NEW Measurement keeping only rows that pass every filter.

        Generic row-restriction primitive: this is the library equivalent
        of a project-side "clip this file to the T<=30 K region before
        searching for transitions" wrapper (a `restrict`-shaped need shows
        up any time a caller wants a sub-window of one file without
        re-parsing it -- e.g. dropping a featureless high-T tail before
        transition detection, isolating one field range, or applying an
        arbitrary hand-built boolean selection). Per the immutability
        convention (see module docstring / DM-1), this never mutates
        ``self``: it returns a new ``Measurement`` with a shallow copy of
        ``header`` and a re-indexed copy of ``data``; the original object
        (and any other reference to it) is untouched.

        Parameters
        ----------
        t_range : (float, float), optional
            Inclusive ``(lo, hi)`` bounds (Kelvin) applied to
            :attr:`temperature`. Raises ``ValueError`` if given and this
            Measurement has no temperature column.
        h_range : (float, float), optional
            Inclusive ``(lo, hi)`` bounds (Oe) applied to :attr:`field_oe`
            -- NOT :attr:`field_tesla`, matching the raw column's native
            unit. Raises ``ValueError`` if given and this Measurement has
            no field column.
        mask : array-like of bool, optional
            A boolean mask aligned positionally to ``data`` (i.e. its
            length must equal ``len(self.data)``). Combined with
            ``t_range``/``h_range`` by logical AND when more than one
            filter is given.

        Returns
        -------
        Measurement
            A new object whose ``data`` contains only the rows that pass
            every given filter, with a reset ``RangeIndex``. ``header`` is
            a shallow copy (``dict(self.header)``); all other fields
            (``path``, ``data_format``, ``measurement_type``,
            ``file_role``, ``title``) are shared as-is, since they
            describe the source file and are unaffected by row selection.

        Notes
        -----
        A row whose filter column is NaN is dropped only by the filter
        that reads that column: a NaN :attr:`temperature` fails
        ``t_range`` (so that row is excluded whenever ``t_range`` is
        given) but is untouched by ``h_range`` or ``mask``, and vice
        versa -- this falls out naturally from pandas' NaN-comparisons
        (``NaN >= lo`` and ``NaN <= hi`` are both False), no special-casing
        needed. Giving no filters at all (`t_range=h_range=mask=None`)
        returns a full copy. A filter combination that keeps no rows
        returns a Measurement with an empty (but correctly columned)
        ``data`` DataFrame, not an error.
        """
        keep = pd.Series(True, index=self.data.index)

        if t_range is not None:
            temperature = self.temperature
            if temperature is None:
                raise InputValueError(
                    "restrict(t_range=...) requires a temperature column, "
                    "but this Measurement has none"
                )
            lo, hi = t_range
            keep &= (temperature >= lo) & (temperature <= hi)

        if h_range is not None:
            field_oe = self.field_oe
            if field_oe is None:
                raise InputValueError(
                    "restrict(h_range=...) requires a field column, "
                    "but this Measurement has none"
                )
            lo, hi = h_range
            keep &= (field_oe >= lo) & (field_oe <= hi)

        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape[0] != len(self.data):
                raise InputValueError(
                    f"restrict(mask=...) length {mask_arr.shape[0]} does not "
                    f"match data length {len(self.data)}"
                )
            keep &= pd.Series(mask_arr, index=self.data.index)

        new_data = self.data.loc[keep].reset_index(drop=True)
        # Restamp the composite type from the rows that survived. Without this
        # a subset would go on advertising families it no longer holds --
        # `measurement_types` is derived and would correct itself, leaving the
        # two accessors disagreeing, which is worse than either answer alone.
        # A subset whose rows say nothing (all housekeeping, or a file with no
        # family evidence at all) keeps the label the loader gave the file:
        # silence is not a reason to overwrite what was known.
        try:
            derived = _compose_type(row_families(new_data, path=self.path))
        except MeasurementFamilyError:
            # `restrict` is general row-selection, not a family question. A file
            # whose Transport Action codes calamari cannot read still loads, and
            # must still be filterable by temperature or field -- refusing here
            # would break ordinary subsetting over a question the caller never
            # asked. The label then stays as it was, which for such a file is
            # already the honest 'unknown'.
            derived = None
        return replace(
            self,
            header=dict(self.header),
            data=new_data,
            measurement_type=derived or self.measurement_type,
        )
