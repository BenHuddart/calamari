"""Reject non-site artifacts and local data paths before publishing Sphinx output."""

import sys
from pathlib import Path


def check_site(root: Path) -> None:
    """Check the exact deployment tree, including downloadable source pages."""
    if not (root / 'index.html').is_file():
        raise ValueError('Documentation output must contain index.html')
    forbidden = {'.dat', '.raw', '.diag', '.env', '.lastscan', '.nul', '.drv', '.doctree'}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Symlink in documentation output: {path.relative_to(root)}')
        if path.is_file():
            if path.suffix.lower() in forbidden or '.doctrees' in path.relative_to(root).parts:
                raise ValueError(f'Non-site file in documentation output: {path.relative_to(root)}')
            content = path.read_bytes()
            if any(marker in content for marker in (b'/home/', b'/Users/')):
                raise ValueError(f'Local home path in documentation output: {path.relative_to(root)}')


if __name__ == '__main__':
    check_site(Path(sys.argv[1]))
