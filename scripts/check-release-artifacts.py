"""Validate the two pure-Python artifacts selected for a manually authorised release."""

from __future__ import annotations

import argparse
import email
import re
import tarfile
import zipfile
from pathlib import Path

NAME = 'calamari-magnetometry'
STEM = 'calamari_magnetometry'


def release_version(tag: str, target: str) -> str:
    """Require stable PyPI tags and distinct release-candidate tags on TestPyPI."""
    if target not in {'pypi', 'testpypi'}:
        raise ValueError('Target must be pypi or testpypi')
    number = r'(?:0|[1-9][0-9]*)'
    suffix = r'rc[1-9][0-9]*' if target == 'testpypi' else ''
    if not re.fullmatch(rf'v{number}\.{number}\.{number}{suffix}', tag):
        raise ValueError(f'{target} requires a vMAJOR.MINOR.PATCH{suffix and "rcN"} tag')
    return tag[1:]


def validate(directory: Path, tag: str, target: str) -> str:
    """Reject extra artifacts, wrong identities and unreviewed data/build products."""
    version = release_version(tag, target)
    wheel = directory / f'{STEM}-{version}-py3-none-any.whl'
    source = directory / f'{STEM}-{version}.tar.gz'
    if set(directory.iterdir()) != {wheel, source}:
        raise ValueError('Expected exactly one named pure-Python wheel and one source archive')
    with zipfile.ZipFile(wheel) as archive:
        files = archive.namelist()
        records = [name for name in files if name.endswith('.dist-info/METADATA')]
        if len(records) != 1:
            raise ValueError('Wheel must contain exactly one METADATA record')
        wheel_metadata = archive.read(records[0])
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        prefix = f'{STEM}-{version}/'
        records = [member for member in members if member.name == prefix + 'PKG-INFO']
        if len(records) != 1:
            raise ValueError('Source archive must contain its root PKG-INFO')
        stream = archive.extractfile(records[0])
        if stream is None:
            raise ValueError('Source metadata must be a regular file')
        source_metadata = stream.read()
        files += [member.name for member in members]
    for raw in (wheel_metadata, source_metadata):
        metadata = email.message_from_bytes(raw)
        if metadata['Name'] != NAME or metadata['Version'] != version:
            raise ValueError('Artifact name/version does not match the selected release')
    prohibited = {'.dat', '.raw', '.diag', '.env', '.lastscan', '.nul', '.drv', '.png', '.pdf'}
    for name in files:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.suffix.lower() in prohibited:
            raise ValueError(f'Unexpected artifact member: {name}')
    return version


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('tag')
    parser.add_argument('target', choices=['pypi', 'testpypi'])
    args = parser.parse_args()
    print(validate(args.directory, args.tag, args.target))
