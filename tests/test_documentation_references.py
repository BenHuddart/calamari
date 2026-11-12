"""Check document-local citation numbers, links and Physical Review article layout."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_page_local_references(tmp_path):
    pytest.importorskip('sphinxcontrib.bibtex')
    root = Path(__file__).resolve().parents[1]
    for source in ['docs/references.bib', 'docs/_ext/aps_references.py']:
        shutil.copy(root / source, tmp_path / Path(source).name)
    (tmp_path / 'conf.py').write_text(
        'import sys\nfrom pathlib import Path\n'
        'sys.path.insert(0,str(Path(__file__).parent))\n'
        'extensions=["myst_parser","sphinxcontrib.bibtex","aps_references"]\n'
        'bibtex_bibfiles=["references.bib"]\nbibtex_default_style="aps"\n'
    )
    (tmp_path / 'index.md').write_text('# Index\n\n```{toctree}\na\nb\n```\n')
    orders = [('fisher1962', 'savitzky1964'), ('savitzky1964', 'fisher1962')]
    for page, keys in zip(['a', 'b'], orders, strict=True):
        (tmp_path / f'{page}.md').write_text(
            '# Page\n\n' + '\n\n'.join(f'Claim {{footcite:p}}`{key}`.' for key in keys)
            + '\n\n## References\n\n```{footbibliography}\n```\n'
        )
    output = tmp_path / '_html'
    subprocess.run([sys.executable, '-m', 'sphinx', '-n', '-W', '-b', 'html',
                    str(tmp_path), str(output)], check=True, capture_output=True, text=True)
    for page, keys in zip(['a', 'b'], orders, strict=True):
        html = (output / f'{page}.html').read_text()
        for number, key in enumerate(keys, 1):
            assert f'href="#footcite-{key}"' in html
            assert f'{number}<span class="fn-bracket">]</span>' in html
            assert f'id="footcite-{key}"' in html
        assert 'Philos. Mag. <strong>7</strong>, 1731 (1962)' in html
        assert 'Anal. Chem. <strong>36</strong>, 1627 (1964)' in html
        assert 'href="#id1"' in html  # Backlink from the local reference.
