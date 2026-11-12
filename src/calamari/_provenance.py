"""Shared software identity and descriptive callable labels, without import cycles."""

from functools import partial
from importlib import metadata


def calamari_version() -> str:
    """Read installed metadata; only a missing installation returns ``unknown``."""
    try:
        return metadata.version('calamari-magnetometry')
    except metadata.PackageNotFoundError:
        return 'unknown'


def describe_callable(func) -> str:
    """Stable descriptive label, not serialised state or a unique instance identity."""
    if isinstance(func, partial):
        return f'partial({describe_callable(func.func)})'
    target = func if isinstance(getattr(func, '__qualname__', None), str) else type(func)
    module = getattr(target, '__module__', None)
    if not isinstance(module, str):
        module = type(target).__module__
    return f'{module}.{target.__qualname__}'
