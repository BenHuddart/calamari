"""Loader abstract base class and content-sniffing registry.

Unlike the extension-keyed registry in the asymmetry toolkit (from which
this design is adapted), calamari dispatches by **file content, not file
extension**. Quantum Design instruments write everything as ``.dat``
regardless of dialect, instrument, or even whether the file is a
measurement at all (log/environment files share the extension too), so
an extension-based registry cannot distinguish MPMS3 from an
unrelated ``.dat``. Instead every loader implements a cheap
:meth:`BaseLoader.sniff` over the first few dozen header lines, and
:meth:`LoaderRegistry.detect` reads those lines once and offers them to
each registered loader in turn.

This keeps the door open for other vendors: a new instrument format is
added by writing a loader with its own ``sniff`` and registering it; no
existing code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from calamari.measurement import Measurement

#: Number of leading lines read when cheaply sniffing a file's format
#: signature. The header region of MPMS3 (the
#: ``[Header]`` marker plus TITLE/BYAPP/INFO lines) comfortably fits in
#: the first few dozen lines, so 40 is a safe upper bound that avoids
#: reading large data sections just to identify a file.
_SIGNATURE_SCAN_LINES = 40


def _read_text_lines(path: Path, max_lines: int | None = None) -> list[str]:
    """Read a text file robustly, tolerating the encoding and line-ending
    quirks typical of instrument-exported ASCII files.

    Uses ``utf-8-sig`` (transparently strips a UTF-8 BOM that some
    exporters prepend), ``errors='replace'`` (a stray non-UTF-8 byte must
    not abort a parse), and ``newline=''`` with an explicit CR/LF strip
    (files are frequently CRLF-terminated regardless of platform).

    Parameters
    ----------
    path : Path
        File to read.
    max_lines : int, optional
        If given, stop after this many lines -- used for cheap format
        sniffing so the whole file need not be read to identify it.

    Returns
    -------
    list[str]
        Lines with trailing newline/carriage-return characters stripped.
    """
    lines: list[str] = []
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        for raw_line in fh:
            lines.append(raw_line.rstrip("\r\n"))
            if max_lines is not None and len(lines) >= max_lines:
                break
    return lines


def read_signature_lines(path: Path | str) -> list[str]:
    """Read exactly the leading lines a loader's ``sniff`` inspects.

    The counterpart to :meth:`LoaderRegistry.detect`, which swallows read
    errors to stay usable as a directory-walk probe: this one lets them
    propagate, so a caller that must distinguish "unreadable file" from
    "readable but unrecognised file" can. :func:`calamari.io.load` is
    exactly that caller -- it reads here (letting ``FileNotFoundError`` and
    friends through untouched) and only then asks
    :meth:`LoaderRegistry.detect_lines` whether any loader claims the
    content.

    Parameters
    ----------
    path : Path or str
        File to read.

    Returns
    -------
    list[str]
        Up to :data:`_SIGNATURE_SCAN_LINES` leading lines.

    Raises
    ------
    OSError
        If the file does not exist, is a directory, or cannot be read.
    """
    return _read_text_lines(Path(path), max_lines=_SIGNATURE_SCAN_LINES)


class BaseLoader(ABC):
    """Interface every file-format loader must implement.

    A loader is a stateless class (its two methods are class methods):
    :meth:`sniff` inspects header lines to claim a file, and :meth:`load`
    parses it into a :class:`~calamari.measurement.Measurement`.
    """

    #: Short human-readable / canonical format name, e.g. ``'MPMS3'``.
    #: Also the token accepted by ``load(..., fmt=...)`` to force this
    #: loader.
    format_name: str = ""

    @classmethod
    @abstractmethod
    def sniff(cls, lines: list[str]) -> bool:
        """Return True if this loader recognises the file.

        Parameters
        ----------
        lines : list[str]
            The first lines of the file (at least the header region --
            see :data:`_SIGNATURE_SCAN_LINES`).
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Measurement:
        """Parse ``path`` into a :class:`~calamari.measurement.Measurement`."""
        ...


class LoaderRegistry:
    """Central registry of available content-sniffing loaders."""

    _loaders: list[type[BaseLoader]] = []

    @classmethod
    def register(cls, loader: type[BaseLoader]) -> type[BaseLoader]:
        """Register ``loader``; usable directly or as a class decorator.

        Returns the loader unchanged so it can be applied as
        ``@LoaderRegistry.register`` above a loader class definition.
        Registration is idempotent (registering the same class twice is a
        no-op), which keeps re-imports from duplicating entries.
        """
        if loader not in cls._loaders:
            cls._loaders.append(loader)
        return loader

    @classmethod
    def detect_lines(cls, lines: list[str]) -> type[BaseLoader] | None:
        """Return the loader whose :meth:`BaseLoader.sniff` claims ``lines``.

        The pure matching half of :meth:`detect`, split out so a caller that
        has already read the signature region (or that needs read errors to
        propagate rather than be swallowed -- see
        :func:`read_signature_lines`) does not have to read the file twice
        or re-implement the dispatch order.

        Loaders are offered the lines in registration order and the first
        match wins; ``None`` means no registered loader recognises the
        content.
        """
        for loader in cls._loaders:
            if loader.sniff(lines):
                return loader
        return None

    @classmethod
    def detect(cls, path: Path | str) -> type[BaseLoader] | None:
        """Return the loader that claims ``path``, or None.

        Reads the first :data:`_SIGNATURE_SCAN_LINES` lines **once** and
        offers them to :meth:`detect_lines`. Unreadable files
        (permission/encoding errors) yield None rather than raising, so
        this doubles as a safe "is this a supported file?" probe over a
        directory of mixed content -- which is precisely why
        :func:`calamari.io.load` does *not* use it: a caller loading one
        named file needs "could not read" and "not a magnetometry file"
        to be different outcomes.
        """
        try:
            lines = read_signature_lines(path)
        except (OSError, UnicodeError):
            return None
        return cls.detect_lines(lines)

    @classmethod
    def loaders(cls) -> list[type[BaseLoader]]:
        """Return the registered loaders in registration order (a copy)."""
        return list(cls._loaders)

    @classmethod
    def by_name(cls, format_name: str) -> type[BaseLoader] | None:
        """Return the registered loader whose ``format_name`` matches
        (case-insensitively), or None. Used to honour an explicit
        ``fmt=`` override that bypasses sniffing."""
        target = format_name.strip().lower()
        for loader in cls._loaders:
            if loader.format_name.lower() == target:
                return loader
        return None
