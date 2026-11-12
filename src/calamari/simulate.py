"""Generate synthetic MPMS3 inputs and physics-based signals.

Files are generated in temporary directories for tests and examples; no real
instrument exports are committed. mpms3_lines and write_mpms3_file use invented
metadata and explicit DC/VSM/AC channels. Their timestamp offset checks that date
parsing is independent of the numeric epoch, without modelling an instrument clock."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from calamari.exceptions import InputValueError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

#: Seconds between the LabVIEW epoch (1904-01-01 00:00:00 UTC) and the Unix
#: epoch (1970-01-01 00:00:00 UTC): 66 years spanning 17 leap days
#: (1904, 1908, ..., 1968) = 66*365 + 17 days. Used only to derive a
#: synthetic FILEOPENTIME that differs from a Unix timestamp for the same
#: date/time. This deliberate alternative-epoch fixture does not model the MPMS3 clock;
#: it checks that calendar dates do not depend on the numeric timestamp.
_LABVIEW_UNIX_OFFSET_S = 2_082_844_800.0


def _epoch_seconds(date: str, time_str: str) -> float:
    """Convert an ``M/D/YYYY`` + ``H:MM AM/PM`` pair to a Unix epoch (UTC).

    MPMS3 exports write this human-readable pair alongside
    their numeric FILEOPENTIME, which is exactly what
    :func:`calamari.io.quantum_design._extract_date_iso` relies on instead
    of the numeric value. This helper is the inverse: given the same
    human-readable pair, produce a self-consistent numeric epoch to embed
    in synthetic files.
    """
    # Parsed by hand rather than with strptime: %p is locale-dependent,
    # and in locales whose AM/PM designators are empty (en_GB among
    # them) strptime rejects a literal "AM" outright. Hosts can
    # change the process locale, so this helper must not assume the C locale.
    month, day, year = (int(part) for part in date.split("/"))
    clock, meridiem = time_str.split()
    hour, minute = (int(part) for part in clock.split(":"))
    meridiem = meridiem.upper()
    if meridiem not in ("AM", "PM"):
        raise InputValueError(f"time_str {time_str!r}: expected 'H:MM AM' or 'H:MM PM'")
    # strptime's %I accepted only 1-12; keep that contract rather than
    # letting "0:15 AM" slip through the % 12 arithmetic as midnight.
    if not 1 <= hour <= 12 or not 0 <= minute <= 59:
        raise InputValueError(f"time_str {time_str!r}: hour must be 1-12 and minute 00-59")
    hour = hour % 12 + (12 if meridiem == "PM" else 0)
    dt = _dt.datetime(year, month, day, hour, minute, tzinfo=_dt.timezone.utc)
    return dt.timestamp()


def _validate_lengths(n: int, **arrays: np.ndarray) -> None:
    """Raise ValueError naming the offending array if its length != n."""
    for name, arr in arrays.items():
        if arr.shape[0] != n:
            raise InputValueError(
                f"{name} has length {arr.shape[0]}, expected {n} (to match "
                "temperature_K/field_oe)"
            )


def _format_row(values: list[float]) -> str:
    """Format one CSV data row at a precision that survives a float64
    round-trip through ``pandas.read_csv`` well within
    ``numpy.testing.assert_allclose`` default tolerances."""
    return ",".join(f"{v:.10g}" for v in values)


#: Synthetic column suffixes for exercising both supported unit systems.
#: Each suffix identifies the recorded quantity's unit for accessor conversion.
_MPMS3_UNIT_SUFFIXES = {
    "emu": {"field": "Oe", "moment": "emu", "chi": "emu/Oe", "drive": "Oe"},
    "Am2": {"field": "T", "moment": "Am2", "chi": "Am2/T", "drive": "T"},
}

def mpms3_lines(
    *,
    title: str = "synthetic",
    app: str = "SQUID AC",
    sample_material: str | None = "SyntheticSample",
    sample_mass_mg: float | None = 10.0,
    sample_shape: str | None = None,
    sample_comment: str | None = None,
    date: str = "1/29/2020",
    time_str: str = "10:00 AM",
    unit_system: str = "emu",
    temperature_K: ArrayLike,
    field_oe: ArrayLike,
    vsm_moment_emu: ArrayLike | None = None,
    vsm_moment_err_emu: ArrayLike | None = None,
    dc_moment_emu: ArrayLike | None = None,
    dc_moment_err_emu: ArrayLike | None = None,
    chi_real: ArrayLike | None = None,
    chi_imag: ArrayLike | None = None,
    ac_frequency_hz: ArrayLike | None = None,
    ac_drive_oe: ArrayLike | None = None,
    transport_action: ArrayLike | None = None,
    extra_info: dict[str, str] | None = None,
    extra_columns: dict[str, ArrayLike] | None = None,
) -> list[str]:
    """Build the lines of a synthetic MPMS3-dialect ``[Header]``/``[Data]`` file.

    Reproduces the MPMS3 signature (the ``; MPMS3 Data File`` comment and a
    ``BYAPP,SQUID ...`` line -- see :func:`calamari.io.quantum_design.detect_format`)
    and, critically, the **reversed** ``INFO,<value>,<KEY>`` sample-metadata
    ordering (value first, KEY last) that distinguishes this dialect from
    key-before-value ordering.

    Parameters
    ----------
    title : str, optional
        Free-form ``TITLE`` value. MPMS3's TITLE is a
        descriptive run name, NOT the measurement type -- that comes from
        ``app`` instead.
    app : str, optional
        ``BYAPP`` value, e.g. ``'SQUID AC'``/``'SQUID DC'``/``'SQUID VSM'``.
        Note MPMS3's BYAPP is known to say "SQUID AC" even for DC runs in
        real files; pair ``app='SQUID AC'`` with only ``dc_moment_emu`` (no
        ``chi_real``/``chi_imag``) to reproduce that trap in a test.
    sample_material, sample_mass_mg, sample_shape, sample_comment : optional
        Written as reversed ``INFO`` lines when not None (``sample_mass_mg``
        as ``SAMPLE_MASS``, ``sample_material`` as ``SAMPLE_MATERIAL``,
        etc.) and omitted entirely when None, so a test can also exercise
        the "field absent from header" path.
    date, time_str : str, optional
        Human-readable ``M/D/YYYY`` / ``H:MM AM/PM`` tokens embedded in the
        FILEOPENTIME line. The numeric epoch written alongside them is
        derived using a deliberate 1904-origin fixture offset (see
        :data:`_LABVIEW_UNIX_OFFSET_S`). It tests that date parsing uses the
        readable token; it does not reproduce the empirically supported
        1899-12-30 origin or instrument timezone handling.
    unit_system : {'emu', 'Am2'}, optional
        Which unit selection to reproduce in the **column-name suffixes**
        (see :data:`_MPMS3_UNIT_SUFFIXES`). Default ``'emu'``: field in
        ``(Oe)``, moment in ``(emu)``, susceptibility in ``(emu/Oe)``,
        drive in ``(Oe)``. With ``'Am2'`` the same quantities are labelled
        ``(T)``, ``(Am2)``, ``(Am2/T)`` and ``(T)``.

        Note this changes **only the labels**: the arrays you pass are
        written verbatim. That is deliberate -- it lets a test supply
        known numbers, label them A m^2, and assert that
        :attr:`~calamari.measurement.Measurement.moment_emu` returns them
        scaled to emu. The parameter names below keep their canonical-unit
        spellings for that reason.
    extra_info : dict, optional
        Additional ``{KEY: value}`` header lines, written in the MPMS3
        reversed form ``INFO,<value>,<KEY>``. Values are written verbatim,
        so ``{"SAMPLE_MASS": ""}`` reproduces the blank-value line real
        files carry for every sample property the operator left unset, and
        ``{"MOTOR_SERIAL_NUMBER": "SYNTHETIC-MODULE"}`` reproduces an instrument key
        outside calamari's known-key set.
    extra_columns : dict, optional
        Additional ``{column_name: values}`` written verbatim after the
        standard columns. For exercising parser behaviour against columns
        calamari does not read but real files carry -- sibling columns like
        ``AC X' Std Err. (emu/Oe)``, near-miss names like
        ``Min. Temperature (K)``, an unused mode's empty column alongside a
        populated one, or a column with an unrecognised unit.
    temperature_K, field_oe : array_like
        Required. Always written as the first two ``[Data]`` columns
        (``Temperature (K)``, ``Magnetic Field (Oe)``).
    vsm_moment_emu, vsm_moment_err_emu : array_like, optional
        A **VSM** measurement: written as ``Moment (emu)`` and
        ``M. Std. Err. (emu)``, the VSM application's columns.
    dc_moment_emu, dc_moment_err_emu : array_like, optional
        A **DC scan**: written as ``DC Moment Free Ctr (emu)`` and
        ``DC Moment Err Free Ctr (emu)``, the DC-scan application's columns.

        These are different measurements; choose the argument matching the
        instrument mode. Both are exposed by
        :attr:`~calamari.measurement.Measurement.moment_emu`, with explicit
        selection when a file contains both. For a combined run, supply
        both arrays with ``NaN`` where that mode has no measurement.
    chi_real, chi_imag, ac_frequency_hz, ac_drive_oe : array_like, optional
        Written as additional ``[Data]`` columns (``AC X' (emu/Oe)``,
        ``AC X'' (emu/Oe)``, ``AC Frequency (Hz)``, ``AC Drive (Oe)``) only
        when supplied. Real exports can also contain unused empty channels;
        use ``extra_columns`` to reproduce those shapes.
        ``AC Drive (Oe)`` is the MPMS3 spelling of the AC drive-field
        amplitude -- pass
        e.g. ``ac_drive_oe=0.3`` alongside ``chi_real``/``chi_imag`` so a test
        can round-trip :attr:`calamari.measurement.Measurement.ac_drive_oe`.
    transport_action : array_like, optional
        The per-row ``Transport Action`` code: ``1`` VSM, ``5`` AC, ``6`` DC
        scan, and ``2``/``3``/``4`` instrument housekeeping. Real
        summary files carry housekeeping rows, so a fixture that needs one --
        a trailing locate scan, say -- writes it here rather than through
        ``extra_columns``.
    Returns
    -------
    list[str]
        File lines, ready for ``"\\n".join(...)`` (see :func:`write_mpms3_file`).
    """
    if unit_system not in _MPMS3_UNIT_SUFFIXES:
        raise InputValueError(
            f"unit_system must be one of {sorted(_MPMS3_UNIT_SUFFIXES)}, "
            f"got {unit_system!r}"
        )
    suffix = _MPMS3_UNIT_SUFFIXES[unit_system]

    temperature_K = np.asarray(temperature_K, dtype=float)
    field_oe = np.asarray(field_oe, dtype=float)
    n = temperature_K.shape[0]
    _validate_lengths(n, field_oe=field_oe)

    epoch = _epoch_seconds(date, time_str) + _LABVIEW_UNIX_OFFSET_S

    lines = [
        "[Header]",
        "; MPMS3 Data File",
        f"TITLE,{title}",
        f"BYAPP,{app}",
        f"FILEOPENTIME,{epoch} {date} {time_str}",
    ]
    if sample_mass_mg is not None:
        lines.append(f"INFO,{sample_mass_mg},SAMPLE_MASS")
    if sample_material is not None:
        lines.append(f"INFO,{sample_material},SAMPLE_MATERIAL")
    if sample_shape is not None:
        lines.append(f"INFO,{sample_shape},SAMPLE_SHAPE")
    if sample_comment is not None:
        lines.append(f"INFO,{sample_comment},SAMPLE_COMMENT")

    for info_key, info_value in (extra_info or {}).items():
        lines.append(f"INFO,{info_value},{info_key}")

    columns = ["Temperature (K)", f"Magnetic Field ({suffix['field']})"]
    arrays = [temperature_K, field_oe]

    # The VSM and DC-scan moments are written under DIFFERENT column names,
    # because on this instrument they are different measurements: `Moment` is
    # the VSM application's and `DC Moment Free Ctr` the DC-scan application's.
    # Each names its own error column,
    # exactly as Measurement._paired_error expects to find it.
    optional = [
        (f"Moment ({suffix['moment']})", vsm_moment_emu),
        (f"M. Std. Err. ({suffix['moment']})", vsm_moment_err_emu),
        (f"DC Moment Free Ctr ({suffix['moment']})", dc_moment_emu),
        (f"DC Moment Err Free Ctr ({suffix['moment']})", dc_moment_err_emu),
        (f"AC X' ({suffix['chi']})", chi_real),
        (f"AC X'' ({suffix['chi']})", chi_imag),
        ("AC Frequency (Hz)", ac_frequency_hz),
        (f"AC Drive ({suffix['drive']})", ac_drive_oe),
        ("Transport Action", transport_action),
    ]
    for name, values in optional:
        if values is None:
            continue
        arr = np.asarray(values, dtype=float)
        _validate_lengths(n, **{name: arr})
        columns.append(name)
        arrays.append(arr)

    duplicated = sorted(set(extra_columns or ()) & set(columns))
    if duplicated:
        raise InputValueError(
            f"extra_columns would write {duplicated} a second time -- these "
            "already have their own parameter, and two columns of one name in "
            "the [Data] header is a file no instrument writes. Pass the value "
            "through the named parameter instead."
        )

    for name, values in (extra_columns or {}).items():
        arr = np.asarray(values, dtype=float)
        _validate_lengths(n, **{name: arr})
        columns.append(name)
        arrays.append(arr)

    lines.append("[Data]")
    lines.append(",".join(columns))
    for row in zip(*arrays, strict=True):
        lines.append(_format_row(list(row)))

    return lines


def mpms3_raw_lines(
    *,
    raw_position_mm: ArrayLike,
    raw_voltage_v: ArrayLike,
    processed_voltage_v: ArrayLike | None = None,
    fixed_fit_v: ArrayLike | None = None,
    free_fit_v: ArrayLike | None = None,
    time_stamp_s: ArrayLike | None = None,
    sample_material: str | None = "SyntheticSample",
    date: str = "1/29/2020",
    time_str: str = "10:00 AM",
) -> list[str]:
    """Build the lines of a synthetic MPMS3 DC **raw-data** file.

    The per-scan file an MPMS3 DC measurement writes alongside its summary
    (``BaseName.rw.dat``): SQUID voltage
    against sample position for every scan, not one row per measured point.
    Reproduces an observed header shape, which differs from the summary's
    in the two places role classification depends on:

    * The first comment reads ``; DC Raw Data File (default extension
      .raw)`` -- there is **no** ``; MPMS3 Data File`` line, so the file is
      recognised as MPMS3 by its ``BYAPP`` prefix alone.
    * ``BYAPP`` is ``MPMS3 DC RAW DATA,1.0,1.1``: the application name
      itself states the role, which is what lets a raw file be told from a
      summary without trusting its name.

    Parameters
    ----------
    raw_position_mm, raw_voltage_v : array_like
        Required. Written as ``Raw Position (mm)`` and ``Raw Voltage (V)``.
    processed_voltage_v, fixed_fit_v, free_fit_v : array_like, optional
        ``Processed Voltage (V)``, ``Fixed C Fitted (V)`` and
        ``Free C Fitted (V)``. Real files always carry all three, so they
        are always written; each defaults to a copy of ``raw_voltage_v``.
    time_stamp_s : array_like, optional
        ``Time Stamp (sec)``; defaults to one second per row.
    sample_material : str, optional
        Written as a reversed ``INFO`` line when not None.
    date, time_str : str, optional
        The ``FILEOPENTIME`` tokens, as for :func:`mpms3_lines`.

    Returns
    -------
    list[str]
        File lines, ready for ``"\\n".join(...)`` (see
        :func:`write_mpms3_raw_file`).
    """
    position = np.asarray(raw_position_mm, dtype=float)
    voltage = np.asarray(raw_voltage_v, dtype=float)
    n = position.shape[0]
    columns = {
        "Time Stamp (sec)": np.arange(n, dtype=float) if time_stamp_s is None else time_stamp_s,
        "Raw Position (mm)": position,
        "Raw Voltage (V)": voltage,
        "Processed Voltage (V)": voltage if processed_voltage_v is None else processed_voltage_v,
        "Fixed C Fitted (V)": voltage if fixed_fit_v is None else fixed_fit_v,
        "Free C Fitted (V)": voltage if free_fit_v is None else free_fit_v,
    }
    arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    for name, arr in arrays.items():
        if arr.shape[0] != n:
            raise InputValueError(
                f"{name} has length {arr.shape[0]}, expected {n} (to match raw_position_mm)"
            )

    epoch = _epoch_seconds(date, time_str) + _LABVIEW_UNIX_OFFSET_S
    lines = [
        "[Header]",
        "; DC Raw Data File (default extension .raw)",
        "TITLE,",
        f"FILEOPENTIME,{epoch} {date} {time_str}",
        "BYAPP,MPMS3 DC RAW DATA,1.0,1.1",
    ]
    if sample_material is not None:
        lines.append(f"INFO,{sample_material},SAMPLE_MATERIAL")
    lines.append(f"RECORDS,FROM_TO_RECORDS,1,{n}")
    lines.append("[Data]")
    # A leading, always-empty Comment column, as the real file writes it.
    lines.append(",".join(["Comment", *arrays]))
    for row in zip(*arrays.values(), strict=True):
        lines.append("," + _format_row(list(row)))
    return lines


def write_mpms3_raw_file(path: Path | str, **kwargs) -> Path:
    """Write :func:`mpms3_raw_lines` output to ``path`` and return it.

    Thin wrapper -- all keyword arguments are forwarded to
    :func:`mpms3_raw_lines`. The name is the caller's: pass one ending in
    ``.rw.dat`` to match a real export, or any other to test that the
    header alone identifies the file.
    """
    path = Path(path)
    lines = mpms3_raw_lines(**kwargs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_mpms3_file(path: Path | str, **kwargs) -> Path:
    """Write :func:`mpms3_lines` output to ``path`` and return it.

    Thin wrapper -- all keyword arguments are forwarded to
    :func:`mpms3_lines`. Writes UTF-8 with a trailing newline, matching how
    real MPMS3 exports look to :func:`calamari.io.base._read_text_lines`.
    """
    path = Path(path)
    lines = mpms3_lines(**kwargs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def curie_weiss_chi(
    T_K: ArrayLike, C: float, theta_K: float, chi0: float = 0.0
) -> np.ndarray:
    """Curie-Weiss susceptibility chi(T) = C / (T - theta) + chi0.

    Standard paramagnetic susceptibility law; ``theta_K`` is the Weiss
    temperature (positive for ferromagnetic correlations, negative for
    antiferromagnetic), ``C`` the Curie constant, and ``chi0`` an optional
    temperature-independent background (e.g. diamagnetic core/Van Vleck
    contribution). Used both to build synthetic ``chi(T)`` data for
    :mod:`calamari.fitting` model-recovery tests and as example data for
    users prototyping a Curie-Weiss fit.

    Parameters
    ----------
    T_K : array_like
        Temperature in Kelvin.
    C : float
        Curie constant, in whatever units chi is wanted (e.g. emu K / Oe
        for a molar susceptibility).
    theta_K : float
        Weiss temperature in Kelvin.
    chi0 : float, optional
        Temperature-independent offset. Default 0.

    Returns
    -------
    numpy.ndarray
        chi(T) evaluated at each ``T_K``. Diverges at ``T_K == theta_K``
        (not guarded -- callers sweeping through theta should avoid the
        pole, as with any real Curie-Weiss fit).
    """
    T_K = np.asarray(T_K, dtype=float)
    return C / (T_K - theta_K) + chi0


def synthetic_mh_curve(
    field_oe: ArrayLike,
    m_sat_emu: float,
    h_knee_oe: float,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A saturating M(H) curve: ``m_sat_emu * tanh(H / h_knee_oe)``.

    This reversible toy curve exercises fitting and unit conversion; it
    models neither hysteresis nor a particular material. Convert its
    saturation moment with ``Sample.moment_emu_to_muB_per_ion``; this is
    distinct from the effective moment inferred from a Curie constant.
    ``h_knee_oe`` sets the characteristic saturation field.

    Parameters
    ----------
    field_oe : array_like
        Applied field in Oe (may be negative).
    m_sat_emu : float
        Saturation moment in emu.
    h_knee_oe : float
        Characteristic field (Oe) at which the curve is within
        ``tanh(1) ~ 0.76`` of saturation.
    noise : float, optional
        Standard deviation (emu) of additive Gaussian noise. Default 0
        (no noise, and ``rng`` may then be omitted).
    rng : numpy.random.Generator, optional
        Required when ``noise > 0`` -- there is no implicit unseeded
        fallback, so every noisy call is reproducible by construction.

    Returns
    -------
    numpy.ndarray
        M(H) in emu, same shape as ``field_oe``.

    Raises
    ------
    ValueError
        If ``noise > 0`` and ``rng`` is None.
    """
    field_oe = np.asarray(field_oe, dtype=float)
    m = m_sat_emu * np.tanh(field_oe / h_knee_oe)
    if noise:
        if rng is None:
            raise InputValueError(
                "synthetic_mh_curve(noise > 0) requires an explicit rng= "
                "(numpy.random.Generator) so the noise is reproducible"
            )
        m = m + rng.normal(scale=noise, size=m.shape)
    return m


def synthetic_chi_peak(
    x: ArrayLike,
    centre: float,
    width: float,
    height: float,
    background: float = 0.0,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A Gaussian peak on a flat background, for transition-detection tests.

    Stands in for a localised feature in a swept quantity -- e.g. a chi'(H)
    peak at a metamagnetic transition, or a chi(T) peak at an ordering
    temperature -- used to plant a known feature location for
    :func:`calamari.transitions.find_transitions` to recover.

    Parameters
    ----------
    x : array_like
        Independent variable (field or temperature, in the caller's units).
    centre : float
        Peak centre, same units as ``x``.
    width : float
        Gaussian standard deviation, same units as ``x``.
    height : float
        Peak height above ``background``.
    background : float, optional
        Flat background level. Default 0.
    noise : float, optional
        Standard deviation of additive Gaussian noise, same units as
        ``height``. Default 0 (no noise, and ``rng`` may then be omitted).
    rng : numpy.random.Generator, optional
        Required when ``noise > 0``; see :func:`synthetic_mh_curve`.

    Returns
    -------
    numpy.ndarray
        y(x), same shape as ``x``.

    Raises
    ------
    ValueError
        If ``noise > 0`` and ``rng`` is None.
    """
    x = np.asarray(x, dtype=float)
    y = height * np.exp(-0.5 * ((x - centre) / width) ** 2) + background
    if noise:
        if rng is None:
            raise InputValueError(
                "synthetic_chi_peak(noise > 0) requires an explicit rng= "
                "(numpy.random.Generator) so the noise is reproducible"
            )
        y = y + rng.normal(scale=noise, size=y.shape)
    return y
