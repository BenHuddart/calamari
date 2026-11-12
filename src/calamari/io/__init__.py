"""File input for calamari: content-sniffing loader dispatch.

Public entry points:

* :func:`load` -- parse a file into a :class:`~calamari.measurement.Measurement`,
  auto-detecting the dialect by content (or forced with ``fmt=``). Raises
  rather than guessing when nothing recognises the file.
* :func:`sniff_format` -- report the detected dialect without parsing.
* :func:`is_supported_file` -- boolean probe that parses and validates the export.

The built-in MPMS3 loader registers
themselves when :mod:`calamari.io.quantum_design` is imported, which
happens here at package import, so they are available as soon as
``calamari.io`` is imported. New instrument formats register the same
way; nothing here is MPMS-specific beyond that import.

Strictness (IO-5)
-----------------
:func:`load` returns a :class:`~calamari.measurement.Measurement` only for a
file some registered loader actually claims. There is no best-effort
"parse it anyway" fallback and no ``data_format='unknown'`` outcome: a file
calamari does not recognise is an error, not a half-parsed object that
downstream code has to keep re-checking. The four ways a file can fail to
load are kept distinct, because they call for different fixes:

``OSError`` (``FileNotFoundError``, ``IsADirectoryError``, ``PermissionError``)
    The file could not be read at all -- wrong path, a directory, no
    permission.
:class:`~calamari.exceptions.UnsupportedFileError`
    Read fine; no loader's signature matched. Register a loader for the
    format, or fix the file.
:class:`~calamari.exceptions.UnknownFormatError`
    ``fmt=`` named a loader that is not registered. A caller typo; the
    file was never read.

:class:`~calamari.exceptions.FileParseError`
    A recognised export has malformed structure. Re-export the complete file.

To *test* for supportedness instead of handling an exception -- the right
shape for a directory walk over mixed content -- use
:func:`is_supported_file`, which never raises. Catalogue discovery uses signature
recognition so that malformed recognised exports can be recorded as parse errors.
"""

from __future__ import annotations

from pathlib import Path

from calamari.exceptions import CalamariError, UnknownFormatError, UnsupportedFileError

# Importing the Quantum Design module registers Mpms3Loader
# with LoaderRegistry as a side effect (it carries the
# @LoaderRegistry.register decorator). Nothing below references the module
# directly -- it is imported purely for that registration, hence the noqa.
from calamari.io import quantum_design as _quantum_design  # noqa: F401
from calamari.io.base import BaseLoader, LoaderRegistry, read_signature_lines
from calamari.measurement import Measurement

__all__ = [
    "load",
    "sniff_format",
    "is_supported_file",
    "Measurement",
    "BaseLoader",
    "LoaderRegistry",
]

#: Maximum characters of the offending file's first non-blank line quoted
#: back in an :class:`~calamari.exceptions.UnsupportedFileError` message.
#: Long enough to recognise a header line at a glance, short enough that a
#: single enormous CSV row cannot flood a traceback.
_SNIPPET_CHARS = 80


def _registered_format_names() -> str:
    """Comma-joined ``format_name`` of every registered loader, for messages."""
    names = [loader.format_name for loader in LoaderRegistry.loaders()]
    return ", ".join(names) if names else "(none registered)"


def _unsupported_message(path: Path, lines: list[str]) -> str:
    """Build the UnsupportedFileError message for a file nothing claimed.

    Quotes the first non-blank line back at the caller: format detection
    keys off the header region, so that one line is almost always enough to
    see *why* nothing matched (MPMS3 require a
    leading ``[Header]``), without the user having to open the file.
    """
    first = next((line.strip() for line in lines if line.strip()), "")
    if not first:
        detail = "the file is empty or entirely blank"
    else:
        snippet = first if len(first) <= _SNIPPET_CHARS else first[:_SNIPPET_CHARS] + "..."
        detail = f"first non-blank line is {snippet!r}"
    return (
        f"{path}: not a recognised magnetometry file. calamari identifies files "
        f"by content, never by extension, and no registered loader's signature "
        f"matched ({detail}). Registered formats: {_registered_format_names()}. "
        f"MPMS3 measurement exports start with a '[Header]' line; a genuinely "
        f"new format needs its own BaseLoader registered with LoaderRegistry "
        f"(see calamari.io.base)."
    )


def load(path: Path | str, fmt: str | None = None) -> Measurement:
    """Parse a magnetometry file into a :class:`~calamari.measurement.Measurement`.

    The dialect is detected from file **content** (never the extension:
    every Quantum Design export is ``.dat``). A file that no registered
    loader claims is an error -- see the module docstring's strictness
    table. Use :func:`is_supported_file` to probe without raising.

    Parameters
    ----------
    path : Path or str
        File to parse.
    fmt : str, optional
        Force a dialect, bypassing content sniffing. Accepts a registered
        loader's ``format_name`` (e.g. ``'MPMS3'``,
        case-insensitive). Useful for a file whose header is damaged enough
        to defeat sniffing but whose dialect the caller knows.

    Returns
    -------
    Measurement
        Always carrying a real dialect in ``data_format`` -- the
        ``format_name`` of whichever loader parsed it.

    Raises
    ------
    UnsupportedFileError
        If the file was read but no registered loader recognises its
        content (auto-detection only -- an explicit ``fmt`` skips sniffing).
    UnknownFormatError
        If ``fmt`` is given but names no registered loader. The file is not
        read in this case.
    FileParseError
        If a recognised export has malformed required sections or CSV structure.
    OSError
        If the file cannot be read (missing, a directory, no permission).
    """
    path = Path(path)

    if fmt is not None:
        loader = LoaderRegistry.by_name(fmt)
        if loader is None:
            raise UnknownFormatError(
                f"No loader registered for format {fmt!r}. "
                f"Registered formats: {_registered_format_names()}."
            )
        return loader.load(path)

    # Read the signature region here rather than through LoaderRegistry.detect:
    # detect() deliberately swallows read errors to stay usable as a probe, but
    # load() must keep "cannot read this file" (OSError, propagated untouched)
    # distinct from "read it fine, but it is not a magnetometry file"
    # (UnsupportedFileError). Collapsing the two would report a typo'd path as
    # an unsupported format.
    lines = read_signature_lines(path)
    loader = LoaderRegistry.detect_lines(lines)
    if loader is None:
        raise UnsupportedFileError(_unsupported_message(path, lines))
    return loader.load(path)


def sniff_format(path: Path | str) -> str | None:
    """Return the detected dialect name for ``path``, or None.

    Reads only the file's signature region (not the data), returning the
    matching loader's ``format_name`` (e.g. ``'MPMS3'``)
    or None if no loader claims it. Never raises -- an unreadable file is
    reported as None, same as an unrecognised one; call :func:`load` when
    you need those two cases distinguished.
    """
    loader = LoaderRegistry.detect(path)
    return loader.format_name if loader is not None else None


def is_supported_file(path: Path | str) -> bool:
    """Test whether an export can be loaded, including structural validation.

    This parses the file rather than inspecting only its signature. Malformed,
    unsupported and unreadable files return False. Use :func:`sniff_format` for
    cheap format identification, or :func:`load` for a diagnostic exception.
    """
    try:
        load(path)
    except (CalamariError, OSError):
        return False
    return True
