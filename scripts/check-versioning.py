"""Exercise the active SCM backend in disposable repositories without changing real tags."""

from __future__ import annotations

import email
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path


def run(*args: str, cwd: Path, success: bool = True) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('SETUPTOOLS_SCM_', 'GIT_')) and k != 'PYTHONPATH'}
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    if success and result.returncode:
        raise RuntimeError(f'{args!r}\n{result.stdout}\n{result.stderr}')
    return result


def build(source: Path, destination: Path) -> tuple[Path, str]:
    run('uv', 'build', '--wheel', '--out-dir', str(destination), str(source), cwd=source)
    wheel, = destination.glob('*.whl')
    with zipfile.ZipFile(wheel) as archive:
        metadata, = (n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
        version = email.message_from_bytes(archive.read(metadata))['Version']
    return wheel, version


def verify_install(wheel: Path, version: str, env: Path, cwd: Path) -> Path:
    run('uv', 'venv', '--python', '3.13', str(env), cwd=cwd)
    python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    run('uv', 'pip', 'install', '--python', str(python), str(wheel), cwd=cwd)
    run(str(python), '-I', '-c',
        'import calamari, importlib.metadata as m, sys; '
        'assert calamari.__version__ == m.version("calamari-magnetometry") == sys.argv[1]',
        version, cwd=cwd)
    return python


def verify_editable(source: Path, version: str) -> None:
    """uv run must refresh cached metadata when a tag or commit changes."""
    run('uv', 'run', '--no-default-groups', '--python', '3.13', 'python', '-c',
        'import calamari, importlib.metadata as m, sys; '
        'assert calamari.__version__ == m.version("calamari-magnetometry") == sys.argv[1]',
        version, cwd=source)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    refs = run('git', 'show-ref', '--tags', cwd=root, success=False).stdout
    with tempfile.TemporaryDirectory(prefix='calamari-versioning-') as directory:
        work = Path(directory)
        source = work / 'candidate'
        source.mkdir()
        files = run('git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z', cwd=root)
        for name in files.stdout.split('\0'):
            if name:
                destination = source / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, destination)
        run('git', 'init', '-q', cwd=source)
        run('git', 'config', 'user.name', 'Synthetic version test', cwd=source)
        run('git', 'config', 'user.email', 'tests@example.invalid', cwd=source)
        run('git', 'config', 'commit.gpgsign', 'false', cwd=source)
        run('git', 'add', '.', cwd=source)
        run('git', 'commit', '-qm', 'Synthetic candidate snapshot', cwd=source)
        wheel, version = build(source, work / 'untagged')
        assert '.dev' in version and version != '0.5.0', version
        verify_install(wheel, version, work / 'untagged-env', work)
        verify_editable(source, version)
        print('untagged:', version, flush=True)

        run('git', 'tag', 'v0.5.0', cwd=source)
        wheel, version = build(source, work / 'tagged')
        assert version == '0.5.0', version
        verify_install(wheel, version, work / 'tagged-env', work)
        verify_editable(source, version)
        print('tagged:', version, flush=True)

        run('uv', 'build', '--sdist', '--out-dir', str(work / 'sdist'), str(source), cwd=work)
        sdist, = (work / 'sdist').glob('*.tar.gz')
        extracted = work / 'unpacked'
        extracted.mkdir()
        with tarfile.open(sdist) as archive:
            archive.extractall(extracted, filter='data')
            assert not any(n.endswith(('.png', '.pdf'))
                           for n in archive.getnames())
        unpacked, = extracted.iterdir()
        wheel, version = build(unpacked, work / 'rebuilt')
        assert version == '0.5.0', version
        python = verify_install(wheel, version, work / 'archive-env', work)
        print('sdist without Git:', version, flush=True)
        release_dist = work / 'release-dist'
        release_dist.mkdir()
        shutil.copy2(wheel, release_dist / wheel.name)
        shutil.copy2(sdist, release_dist / sdist.name)
        run(str(python), str(unpacked / 'scripts/check-release-artifacts.py'),
            str(release_dist), 'v0.5.0', 'pypi', cwd=work)
        print('release artifact gate accepts the built candidate pair', flush=True)
        run('uv', 'pip', 'install', '--python', str(python),
            '--group', f'{unpacked}/pyproject.toml:dev',
            '--group', f'{unpacked}/pyproject.toml:docs', cwd=work)
        lint = python.parent / ('ruff.exe' if os.name == 'nt' else 'ruff')
        run(str(lint), 'check', 'src', 'tests', cwd=source)
        tests = run(str(python), '-m', 'pytest', 'tests/', cwd=unpacked)
        print(tests.stdout.strip().splitlines()[-1], flush=True)
        package = run(str(python), '-I', '-c',
                      'import calamari; print(calamari.__path__[0])', cwd=work).stdout.strip()
        run(str(python), '-m', 'pytest', '--doctest-plus', '--doctest-modules', package,
            cwd=unpacked)
        sphinx = python.parent / ('sphinx-build.exe' if os.name == 'nt' else 'sphinx-build')
        run(str(sphinx), '-E', '-a', '-n', '-W', '--keep-going', '-b', 'html',
            'docs', str(work / 'html'), cwd=unpacked)
        workflow = run(str(python), str(unpacked / 'tests/_workflow.py'),
                       str(work / 'workflow'), cwd=work)
        record = json.loads(workflow.stdout)
        assert record['version'] == record['provenance_version'] == version
        assert version in (work / 'html/index.html').read_text()
        print('versioned archive: full tests, doctests, docs and production provenance agree', flush=True)

        archive_path = work / 'git-source.tar'
        run('git', 'archive', '--format=tar', '-o', str(archive_path), 'v0.5.0', cwd=source)
        exported = work / 'git-export'
        exported.mkdir()
        with tarfile.open(archive_path) as archive:
            archive.extractall(exported, filter='data')
        wheel, version = build(exported, work / 'git-export-wheel')
        assert version == '0.5.0', version
        verify_install(wheel, version, work / 'git-export-env', work)
        print('Git archive without Git:', version, flush=True)

        (source / 'CHANGELOG.md').write_text('Synthetic development change.\n')
        run('git', 'add', 'CHANGELOG.md', cwd=source)
        run('git', 'commit', '-qm', 'Synthetic development commit', cwd=source)
        wheel, version = build(source, work / 'development')
        assert version.startswith('0.5.1.dev1+g'), version
        verify_install(wheel, version, work / 'development-env', work)
        verify_editable(source, version)
        print('development:', version, flush=True)
        with (source / 'README.md').open('a') as stream:
            stream.write('\nSynthetic dirty change.\n')
        wheel, version = build(source, work / 'dirty')
        assert version.startswith('0.5.1.dev1+g') and '.d' in version.split('+')[1], version
        verify_install(wheel, version, work / 'dirty-env', work)
        print('dirty:', version, flush=True)

        shallow = work / 'shallow'
        run('git', 'clone', '-q', '--depth', '1', source.as_uri(), str(shallow), cwd=work)
        result = run('uv', 'build', '--wheel', str(shallow), cwd=work, success=False)
        assert result.returncode != 0 and 'is shallow' in result.stderr.lower()
        print('shallow history rejected', flush=True)
        raw = work / 'raw-copy'
        shutil.copytree(exported, raw, ignore=shutil.ignore_patterns('*.egg-info', '_version.py'))
        (raw / '.git_archival.txt').unlink()
        result = run('uv', 'build', '--wheel', str(raw), cwd=work, success=False)
        assert result.returncode != 0 and 'unable to detect version' in result.stderr.lower()
        print('unidentified raw copy rejected without a release-version fallback', flush=True)
    assert refs == run('git', 'show-ref', '--tags', cwd=root, success=False).stdout
    print('Real repository tags unchanged; versioning checks remain isolated.')


if __name__ == '__main__':
    main()
