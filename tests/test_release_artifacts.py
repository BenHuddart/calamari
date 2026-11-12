"""Release gates reject mismatched identities before artifacts can be uploaded."""

import io
import runpy
import tarfile
import zipfile
from pathlib import Path

import pytest

CHECKS = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/check-release-artifacts.py'))


def artifacts(root, version='0.5.0', name='calamari-magnetometry', extra=None):
    metadata = f'Name: {name}\nVersion: {version}\n'.encode()
    stem = f'calamari_magnetometry-{version}'
    with zipfile.ZipFile(root / f'{stem}-py3-none-any.whl', 'w') as archive:
        archive.writestr(f'{stem}.dist-info/METADATA', metadata)
        if extra:
            archive.writestr(extra, b'invented')
    with tarfile.open(root / f'{stem}.tar.gz', 'w:gz') as archive:
        member = tarfile.TarInfo(f'{stem}/PKG-INFO')
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))


@pytest.mark.parametrize(('version', 'target'), [('0.5.0', 'pypi'), ('0.5.0rc1', 'testpypi')])
def test_matching_release_artifacts(tmp_path, version, target):
    artifacts(tmp_path, version)
    assert CHECKS['validate'](tmp_path, f'v{version}', target) == version


@pytest.mark.parametrize(('tag', 'target'), [
    ('v0.5.0', 'testpypi'), ('v0.5.0rc1', 'pypi'), ('v0.5.0.dev1', 'pypi'),
    ('v0.5.0+local', 'pypi'), ('v0.5.0rc0', 'testpypi'), ('--help', 'pypi'),
])
def test_wrong_release_kind(tag, target):
    with pytest.raises(ValueError):
        CHECKS['release_version'](tag, target)


def test_wrong_distribution_identity(tmp_path):
    artifacts(tmp_path, name='unrelated-package')
    with pytest.raises(ValueError, match='name/version'):
        CHECKS['validate'](tmp_path, 'v0.5.0', 'pypi')


def test_mismatched_tag_or_extra_artifact(tmp_path):
    artifacts(tmp_path)
    with pytest.raises(ValueError, match='exactly one'):
        CHECKS['validate'](tmp_path, 'v0.5.1', 'pypi')
    (tmp_path / 'unexpected.txt').touch()
    with pytest.raises(ValueError, match='exactly one'):
        CHECKS['validate'](tmp_path, 'v0.5.0', 'pypi')


@pytest.mark.parametrize('extra', ['invented.dat', 'docs/_build/figure.png', '../outside.txt'])
def test_forbidden_archive_members(tmp_path, extra):
    artifacts(tmp_path, extra=extra)
    with pytest.raises(ValueError, match='Unexpected artifact member'):
        CHECKS['validate'](tmp_path, 'v0.5.0', 'pypi')
