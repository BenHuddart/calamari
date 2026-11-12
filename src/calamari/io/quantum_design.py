"""Parse Quantum Design MPMS3 measurement exports.

Headers use value-before-key INFO entries. TITLE and BYAPP describe the run
and application; populated rows determine its DC/VSM/AC measurement families.
Column titles and unit suffixes are read dynamically. Unknown header keys and
missing quantities are preserved. See docs/USER_GUIDE.md for supported exports.

Inspected native exports support seconds from 1899-12-30, with timezone and
clock handling unspecified. Preserve numeric timestamps without conversion;
calendar dates come from the human-readable token."""

from __future__ import annotations

import csv
import io as _io
import re
from pathlib import Path

import pandas as pd

from calamari.exceptions import FileParseError, MeasurementFamilyError
from calamari.io.base import BaseLoader, LoaderRegistry, _read_text_lines
from calamari.measurement import (
    Measurement,
    _compose_type,
    row_families,
)

#: Regex to pull an ``M/D/YYYY`` date token out of a FILEOPENTIME line.
_DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")

#: Known MPMS3 INFO KEY tokens (upper-cased). The dialect decides the
#: ``INFO`` ordering (see :func:`_handle_info_line`); this set is only the
#: per-line override for a line that contradicts its own dialect.
_MPMS3_INFO_KEYS = {
    "APPNAME",
    "MOMENT_UNITS",
    "SAMPLE_MATERIAL",
    "SAMPLE_COMMENT",
    "SAMPLE_MASS",
    "SAMPLE_VOLUME",
    "SAMPLE_MOLECULAR_WEIGHT",
    "SAMPLE_SIZE",
    "SAMPLE_SHAPE",
    "SAMPLE_HOLDER",
    "SAMPLE_HOLDER_DETAIL",
    "SAMPLE_OFFSET",
}


def detect_format(lines: list[str]) -> str | None:
    """Recognise MPMS3 from its header comment or observed application labels."""
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if stripped.startswith(";") and "MPMS3 DATA FILE" in upper:
            return "MPMS3"
        if upper.startswith("BYAPP,"):
            application = stripped.split(",", 1)[1].strip().upper()
            if application.startswith(("SQUID", "MPMS3")):
                return "MPMS3"
    return None


def _parse_header_line(line: str) -> tuple[str, list[str]]:
    """Split a comma-separated header line into a key and value tokens."""
    parts = line.split(",")
    key = parts[0].strip()
    values = [p.strip() for p in parts[1:]]
    return key, values


def _extract_date_iso(text: str) -> str | None:
    """Extract an ``M/D/YYYY`` date from a FILEOPENTIME string and return
    it as ISO ``YYYY-MM-DD``, or None if not found.

    Reading the human date preserves the recorded calendar date without
    imposing a timezone or assuming identical clock handling across exports.
    """
    match = _DATE_RE.search(text)
    if not match:
        return None
    month, day, year = match.group(1).split("/")
    try:
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    except ValueError:
        return None


def _parse_header(lines: list[str]) -> dict:
    """Parse MPMS3 header metadata while preserving unknown INFO keys and raw lines."""
    header: dict = {
        "title": None,
        "byapp": None,
        "fileopentime_epoch": None,
        "fileopentime_str": None,
        "date_iso": None,
        "info": {},
        "sample_name": None,
        "sample_mass": None,
        "sample_shape": None,
        "sample_comment": None,
        "sample_molecular_weight": None,
        "raw_lines": list(lines),
    }

    # Known specimen keys; unknown INFO entries remain in header["info"].
    key_field_map = {
        "SAMPLE_MATERIAL": "sample_name",
        "SAMPLE_MASS": "sample_mass",
        "SAMPLE_COMMENT": "sample_comment",
        "SAMPLE_SHAPE": "sample_shape",
        "SAMPLE_MOLECULAR_WEIGHT": "sample_molecular_weight",
    }

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        key, values = _parse_header_line(stripped)
        key_upper = key.upper()

        if key_upper == "TITLE":
            header["title"] = ",".join(values).strip()
        elif key_upper == "BYAPP":
            header["byapp"] = ",".join(values).strip()
        elif key_upper == "FILEOPENTIME":
            if values:
                first_field = values[0].strip()
                epoch_str = first_field.split()[0] if first_field else ""
                try:
                    header["fileopentime_epoch"] = float(epoch_str)
                except ValueError:
                    header["fileopentime_epoch"] = None
                full_line = ",".join(values)
                header["fileopentime_str"] = full_line.strip()
                header["date_iso"] = _extract_date_iso(full_line)
        elif key_upper == "INFO":
            _handle_info_line(header, values, key_field_map)

    # Blank INFO values mean "not set", and must become None rather than ''.
    # Real MPMS3 files write a line for every sample property whether or not
    # the operator filled it in ("INFO,,SAMPLE_MASS"), so '' is the normal
    # representation of a missing mass -- and an empty string propagates into
    # arithmetic as a str, turning a clear "no mass recorded" into a
    # TypeError deep inside a unit conversion.
    for text_field in set(key_field_map.values()) | {
        "title",
        "byapp",
    }:
        value = header.get(text_field)
        if isinstance(value, str) and not value.strip():
            header[text_field] = None

    # Coerce numeric sample fields where possible.
    for numeric_field in ("sample_mass", "sample_molecular_weight"):
        val = header.get(numeric_field)
        if val is not None:
            try:
                header[numeric_field] = float(val)
            except (ValueError, TypeError):
                pass

    return header


def _handle_info_line(header, values, key_field_map):
    """Parse value-before-key INFO, retaining the known-key ordering override."""
    if not values:
        return
    first_upper = values[0].strip().upper()
    last_upper = values[-1].strip().upper()
    key_first = first_upper in _MPMS3_INFO_KEYS and last_upper not in _MPMS3_INFO_KEYS
    if key_first:
        info_key = first_upper
        info_val = ",".join(values[1:]).strip()
    else:
        info_key = last_upper
        info_val = ",".join(values[:-1]).strip()
    header["info"][info_key.lower()] = info_val
    if info_key in key_field_map:
        header[key_field_map[info_key]] = info_val


def _split_sections(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split file lines into header-section and data-section lines.

    Returns
    -------
    (header_lines, data_lines) : tuple[list[str], list[str]]
        ``header_lines`` excludes the ``[Header]`` marker itself.
        ``data_lines`` starts immediately after the ``[Data]`` marker (its
        first non-blank line is the CSV column-header row).
    """
    headers = [i for i, line in enumerate(lines) if line.strip() == "[Header]"]
    sections = [i for i, line in enumerate(lines) if line.strip() == "[Data]"]
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if len(headers) != 1 or len(sections) != 1 or headers[0] != first or headers[0] >= sections[0]:
        raise FileParseError(
            "Expected one [Header] at the start followed by one [Data] section; "
            "re-export the complete instrument file."
        )
    header_idx, data_idx = headers[0], sections[0]

    header_lines = lines[header_idx + 1 : data_idx]
    data_lines = lines[data_idx + 1 :]
    return header_lines, data_lines


def _parse_data_section(data_lines: list[str]) -> pd.DataFrame:
    """Parse the ``[Data]`` section into a DataFrame.

    The first non-empty line is the CSV column-header row; all subsequent
    lines are numeric (or empty) CSV rows. Column names are read
    dynamically -- no fixed column set is assumed. Trailing commas
    (producing a trailing empty field) are handled naturally by
    ``pandas.read_csv``. Empty fields become NaN.

    Parameters
    ----------
    data_lines : list[str]
        Lines following the ``[Data]`` marker (including the CSV header
        row).

    Returns
    -------
    pandas.DataFrame
        Parsed data with numeric columns coerced to float where possible;
        non-numeric columns (e.g. ``Comment``) are left as strings/NaN.
    """
    non_blank = [ln for ln in data_lines if ln.strip() != ""]
    if not non_blank:
        raise FileParseError("Missing column-title row after [Data]; re-export the complete file.")

    csv_text = "\n".join(non_blank)
    try:
        rows = list(csv.reader(_io.StringIO(csv_text), strict=True))
    except csv.Error as exc:
        raise FileParseError(f"Malformed CSV: {exc}; re-export the complete file.") from exc
    columns = rows[0]
    while columns and not columns[-1].strip():
        columns.pop()
    names = [name.strip() for name in columns]
    if not names or not all(names) or len(set(names)) != len(names):
        raise FileParseError("Expected distinct non-empty column titles; check the export header.")
    width = len(columns)
    for number, row in enumerate(rows[1:], start=2):
        if any(field.strip() for field in row[width:]):
            raise FileParseError(
                f"Data CSV row {number} has populated fields beyond its {width} column titles: "
                f"{row[width:]!r}; re-export the complete file."
            )
    # Remove only surplus empty fields; pandas otherwise warns and may truncate real data.
    buffer = _io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows([row[:width] for row in rows])
    csv_text = buffer.getvalue()
    df = pd.read_csv(
        _io.StringIO(csv_text),
        sep=",",
        header=0,
        index_col=False,
        skip_blank_lines=True,
        na_values=["", " "],
        keep_default_na=True,
    )

    # Drop columns that are purely trailing-comma artefacts (unnamed AND
    # entirely NaN).
    drop_cols = [
        c for c in df.columns if str(c).startswith("Unnamed") and df[c].isna().all()
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Coerce object columns that look numeric to numeric.
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            non_null_original = df[col].notna().sum()
            non_null_coerced = coerced.notna().sum()
            if non_null_original == 0 or non_null_coerced >= non_null_original:
                df[col] = coerced

    return df.reset_index(drop=True)


def classify_file_role(path: Path, header: dict) -> str:
    """Classify MPMS3 summary, raw or log data using header evidence then filename."""
    title = (header.get("title") or "").upper()
    if "MPMS LOG DATA" in title or path.suffix.lower() == ".env":
        return "log"
    byapp = header.get("byapp") or ""
    application = next((token.strip().upper() for token in byapp.split(",") if token.strip()), "")
    if application.endswith("RAW DATA") or path.name.lower().endswith(".rw.dat"):
        return "raw"
    return "summary"


def _infer_measurement_type(header: dict, data: pd.DataFrame) -> str:
    """Classify populated row families, consulting BYAPP only when no family is measured.

    Unrecognised or contradictory transport codes yield unknown at load time;
    requesting row-family classification exposes the specific error."""
    try:
        families = _compose_type(row_families(data))
    except MeasurementFamilyError:
        # An unreadable Transport Action code, or one contradicting its own
        # columns, must not stop the file loading (IO-5 is strict about
        # DIALECTS, not about every column's vocabulary); the same call raises
        # properly when a caller actually asks, via measurement_types/of_type.
        #
        # But the answer here is 'unknown', NOT the header's guess. Falling
        # through to BYAPP would let a file whose rows plainly carry a VSM
        # moment be catalogued 'AC' because its header names the AC
        # application -- the exact substitution IO-4 exists to prevent. A
        # classification that FAILED is not the same as one that found
        # nothing, and only the latter earns the hint.
        return "unknown"
    if families is not None:
        return families

    byapp = (header.get("byapp") or "").upper()
    if "VSM" in byapp:
        return "VSM"
    if " AC" in byapp or byapp.endswith("AC"):
        return "AC"
    if " DC" in byapp or byapp.endswith("DC"):
        return "DC"
    return "unknown"


def _parse(path: Path | str) -> Measurement:
    """Parse MPMS3 sections, metadata and data into a Measurement.

    The registered loader supplies the format identity. Files with malformed
    required sections raise FileParseError; unreadable inputs retain OSError."""
    path = Path(path)
    lines = _read_text_lines(path)

    try:
        header_lines, data_lines = _split_sections(lines)
        data = _parse_data_section(data_lines)
    except (FileParseError, pd.errors.ParserError) as exc:
        raise FileParseError(f"{path.name}: {exc}") from exc
    header = _parse_header(header_lines)

    measurement_type = _infer_measurement_type(header, data)
    file_role = classify_file_role(path, header)

    return Measurement(
        path=path,
        data_format="MPMS3",
        measurement_type=measurement_type,
        file_role=file_role,
        title=header.get("title"),
        header=header,
        data=data,
    )


@LoaderRegistry.register
class Mpms3Loader(BaseLoader):
    """Loader for the newer MPMS3 (SQUID) ASCII dialect."""

    format_name = "MPMS3"

    @classmethod
    def sniff(cls, lines: list[str]) -> bool:
        """Claim files whose header carries an MPMS3 signature."""
        return detect_format(lines) == "MPMS3"

    @classmethod
    def load(cls, path: Path) -> Measurement:
        """Parse ``path`` as an MPMS3 file."""
        return _parse(path)
