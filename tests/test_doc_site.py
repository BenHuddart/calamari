"""The deployment audit rejects artifacts that a successful Sphinx build could leave."""

import runpy
from pathlib import Path

import pytest

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/check-doc-site.py'))


@pytest.mark.parametrize('name,content', [
    ('_sources/guide.md.txt', b'An invented /' + b'home/example/path'),
    ('invented.dat', b'invented'),
    ('.doctrees/environment.pickle', b'invented'),
])
def test_rejects_non_site_output(tmp_path, name, content):
    (tmp_path / 'index.html').write_text('<h1>Calamari</h1>')
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    with pytest.raises(ValueError):
        CHECK['check_site'](tmp_path)


def test_accepts_site_and_downloadable_sources(tmp_path):
    (tmp_path / 'index.html').write_text('<h1>Calamari</h1>')
    (tmp_path / 'guide.md.txt').write_text('Invented guide source')
    CHECK['check_site'](tmp_path)
    (tmp_path / 'link').symlink_to(tmp_path / 'index.html')
    with pytest.raises(ValueError, match='Symlink'):
        CHECK['check_site'](tmp_path)
