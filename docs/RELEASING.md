# Release verification

This is a maintained verification procedure, not a release announcement.
Builds derive their version from Git using setuptools-scm. The intended
first stable release is 0.5.0, after further polish and explicit maintainer
approval. Citation authorship and the GitHub/Zenodo/PyPI route are recorded in
[CITATION.cff](https://github.com/BenHuddart/calamari/blob/main/CITATION.cff) and this procedure.
The intended PyPI distribution is `calamari-magnetometry`; the Python import remains
`calamari`. No package has been published to production PyPI. The
[0.5.0rc1 candidate on TestPyPI](https://test.pypi.org/project/calamari-magnetometry/0.5.0rc1/)
is available for publication rehearsal.
The source repository is public, and documentation is hosted on
[GitHub Pages](https://benhuddart.github.io/calamari/). Keep README and documentation status
statements aligned with verified hosting and package availability.

## Distribution identity

Built metadata uses `calamari-magnetometry` (normalised to `calamari_magnetometry` in
archive filenames). Runtime versions, analysis records and Sphinx read that distribution's
metadata; users continue to write `import calamari`. Installation instructions currently
use a checkout. After verified publication, update them to
`python -m pip install calamari-magnetometry`. Do not advertise an unpublished package
as installable.

The `distributions` check rebuilds a wheel from the source archive, checks both with
`twine check --strict`, and exercises the installed package outside the checkout.
README links point to the canonical repository. Its logo URLs use the public documentation
site: a white SVG for GitHub's dark theme and the generated white-backed SVG favicon as a
fallback. PyPI's README renderer removes the theme-specific source; the fallback remains
visible on either background. Check the actual TestPyPI rendering before stable publication.

## Verify SCM versioning

```bash
uv run --python 3.13 --no-project python scripts/check-versioning.py
```

The check copies maintained sources into disposable repositories and exercises the active
configuration. It never changes configuration or creates release tags in the real repository:

- `setuptools>=80` and `setuptools-scm>=9.0.3` as build requirements. The latter
  supports the explicit shallow-history rejection used here.
- Dynamic version metadata and a generated `calamari._version` module,
  ignored by Git and imported by `calamari.__version__`. No runtime SCM
  dependency or hard-coded release-version fallback.
- Stable Git archive substitution metadata, with unexpanded templates
  excluded from built source distributions. Built source archives retain
  their version through distribution metadata.

These choices follow the upstream [usage](https://setuptools-scm.readthedocs.io/en/stable/usage/)
and [configuration](https://setuptools-scm.readthedocs.io/v9.0.3/config/) references.

The executable checks cover:

| Source state | Required result |
|---|---|
| Fresh snapshot, no tags | An honest development version, not the intended release |
| Clean disposable `v0.5.0` tag | Exactly `0.5.0` |
| Commit after the tag | Next patch development version with revision identity |
| Dirty development checkout | Development version retaining its dirty marker |
| Wheel installation | Runtime version equals installed distribution metadata |
| Built source distribution without Git | Rebuilt wheel preserves the tagged version |
| Expanded tagged Git archive without Git | Tagged version recovered from archival metadata |
| Shallow repository | Explicit rejection rather than unreliable inference |
| Raw copy without identifying metadata | Explicit rejection, without pretending to be a release |

The versioned source archive also runs the full tests, doctests and fresh
strict Sphinx build. Source lint, Sphinx version output and a small installed-package
workflow's runtime and detection-provenance versions are checked against the candidate. Generated
figures remain excluded. The existing required `distributions` CI job runs
this versioning check alongside ordinary distribution verification. The check also reuses
an editable environment across tag/commit changes to verify uv's metadata cache refresh.

## Source and checkout requirements

Build/install CI jobs check out complete history and tags (`fetch-depth: 0`). The optional
Read the Docs configuration fetches full history and tags before installation. Do not hide
dirty states or override an unknown version with a claimed release. After changing a
checkout's revision, reinstall to refresh the generated version module and distribution
metadata. uv watches commits and tags as cache inputs so ordinary `uv run` commands
refresh metadata after those changes. After editing tracked files without committing,
use `uv run --reinstall-package calamari-magnetometry ...` to refresh the dirty version.
A source checkout must be installed before importing Calamari.

GitHub source downloads are supported only for release tags. An untagged archive can
build as `0.0` because substituted commit/date metadata alone cannot recover its development
version. Use a complete Git checkout or a built source distribution for development.
Shallow checkouts and unidentified raw copies fail explicitly; built source distributions
preserve their version without Git.

Before stable publication, build the actual GitHub archive for the authorised candidate
tag and require its version to match that tag. Repeat this check for each candidate;
the disposable check demonstrates only local `git archive` behaviour.

A real tag must name the reviewed release candidate commit and be created only when the
maintainer authorises it. The disposable checks create tags only in temporary repositories;
passing them does not create a release or authorise publication.

## Prepare PyPI publication

The manually dispatched `publish.yml` workflow is bound to the audited public repository's
numeric ID, `1455528470`. It can run only from `main` in that public repository; forks and
private copies cannot publish. The PyPA publisher action is pinned to the reviewed
v1.14.3 commit. Activation does not authorise a dispatch, tag, upload or release.
Each rehearsal and stable publication still requires explicit maintainer approval.

The `pypi` and `testpypi` environments require approval from `BenHuddart`, allow the sole
maintainer to approve their own dispatch, and prohibit administrator bypass. Both restrict
deployments to the `main` branch. Recheck these settings before each publication.

The workflow accepts a reviewed tag and a destination. It runs from `main`, requires a
public repository, checks that the tag belongs to `main` history, and verifies the latest
nine required checks on its exact commit, supplied by GitHub Actions. It then checks out
the tag, verifies SCM versioning is active, builds once and validates the source archive
and wheel. Installed versions and analysis provenance must agree with the selected tag.
Artifact checks reject unexpected identities, extra packages, instrument-data extensions
and generated PNG/PDF output. The build uses `uv build --no-create-gitignore` so its output
directory contains only the wheel and source archive. These guards supplement the publication
audit; they cannot establish that every committed value is invented or cleared for publication.

Only the separate publishing job receives an OIDC token. It downloads the validated
artifacts without checking out source or rebuilding, then uses the selected index's
Trusted Publisher. The workflow prints SHA-256 hashes and retains artifacts for seven days.
It does not create a GitHub release or Zenodo record.

### Configure publishing

After the exact snapshot and hosted content pass the publication audit:

1. Confirm the canonical public URLs, licence, authorship, README rendering and package
   name availability. Both index names are `calamari-magnetometry`; imports stay `calamari`.
2. Verify SCM versioning and complete Git history in CI and documentation builds using
   the checks above. Confirm runtime, provenance, Sphinx and artifact versions agree.
3. Verify the PyPI and TestPyPI accounts have two-factor authentication and recovery
   methods. Configure separate `pypi` and `testpypi` GitHub environments, with required
   maintainer approval and deployment branches restricted to `main`. Allow the sole
   maintainer to approve their own dispatch; enable prevention of self-review only if
   a distinct approver is available. Verify these settings through read-only inspection.
4. Configure a pending Trusted Publisher on each index using the final repository owner,
   name, `publish.yml` filename and matching environment. No long-lived API token is needed.
   A pending publisher does not reserve a name; recheck before publication.
5. Review the repository-ID lock change and publisher action revision before enabling the
   workflow. Confirm the active workflow is the one reviewed on `main`. Do not dispatch
   either destination until its publication has been explicitly authorised.

See the upstream [Trusted Publisher setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [GitHub Actions publication guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).

### Zenodo integration

The public repository is connected to Zenodo through its release webhook. Zenodo reads
`CITATION.cff` for authorship, affiliation, licence, keywords and the abstract, including
the development disclosure. No `.zenodo.json` overrides that metadata. Inspect the actual
record after publication; connecting the repository alone does not validate the imported
metadata or create a DOI.

For the TestPyPI rehearsal, create only the authorised candidate tag, without publishing
a GitHub release. The stable GitHub release is a separate, authorised step that triggers
Zenodo preservation. Do not use a published GitHub release to test the connection.

### Rehearse, then publish

TestPyPI is public: its rehearsal requires publication clearance and separate authorisation.
Use a reviewed prerelease tag such as `v0.5.0rc1` and dispatch the workflow from `main` with
`testpypi`. The workflow requires `rcN` tags there and stable `vX.Y.Z` tags for PyPI, preventing
an accidental rehearsal under the intended final version. Review the build hashes and
environment approval before uploading.

Inspect the TestPyPI project description and download both artifacts. In a fresh environment,
install runtime dependencies from ordinary PyPI, then install the exact TestPyPI wheel with
`--no-deps --index-url https://test.pypi.org/simple/`. Do not combine both indexes with
`--extra-index-url`. Verify `import calamari`, installed metadata, a synthetic analysis and
a rebuilt source archive. Download GitHub's source archive for the same authorised tag,
build it without Git, and require the resulting version to match the tag. The existing
distribution checks demonstrate the local build and installation path. Repeat the real
index and archive checks for each candidate; local checks do not exercise an upload.

After candidate verification and final release approval, tag the reviewed commit as `v0.5.0`,
ensure its required checks are green, and dispatch with `pypi`. Review and approve the
`pypi` environment deployment. Verify both published artifacts and their hashes, installation
in a clean environment, rendered description, version metadata and public links. Then update
user installation instructions to the verified PyPI command. Coordinate the matching GitHub
release and Zenodo archive; publication across these services is not atomic.

### Recover from an interrupted upload

Inspect the index and workflow logs before retrying; a failed job may have uploaded one
artifact. Preserve the validated artifacts and hashes. Do not rebuild changed content under
the same version, delete a release to reuse its filenames, or use `skip-existing` to hide a
mismatch. If only one artifact is missing, compare published hashes and obtain maintainer
authorisation to upload only the missing original artifact through a reviewed recovery
procedure. The normal workflow deliberately fails on duplicates. If the original artifacts
are unavailable or a correction is needed, prepare a new version; consider yanking a faulty
release with explicit maintainer approval. Do not assume rerunning the whole workflow is safe.

## Verify the AI-use disclosure

Before each publication:

- Check that the [README development disclosure](https://github.com/BenHuddart/calamari/blob/main/README.md#development-disclosure)
  accurately names the tools, purposes and affected parts of the release. It currently
  includes OpenAI Codex and Anthropic's Claude Code. Update it when substantive use changes.
- Keep the capability summary, development disclosure and independent-project statement
  in the `CITATION.cff` abstract consistent with the README. Keep citation keywords aligned
  with `pyproject.toml`. Zenodo imports the abstract as its record description; inspect the
  resulting record to confirm that the disclosure is present. Do not rely solely on the
  archived README. Follow the current
  [Zenodo AI-use guidance](https://support.zenodo.org/help/en-gb/13-policies/227-what-is-your-usage-policy-for-generative-ai-for-depositors)
  and any applicable publication-venue requirements.
- Retain human authorship and accountability in citation and deposit metadata. Do not list
  AI tools as record creators/contributors, claim universal human review, or describe
  AI-assisted code review as human review.

The package README carries the same disclosure to PyPI. These steps prepare the statement;
editing this procedure does not publish a record or connect an account.

## Before any public publication

Audit the exact candidate, reachable history and intended public issues, discussions, logs,
artifacts and hosted documentation sources. Resolve any private-data or credential concerns
before publication. Retain verification evidence without including research inputs or paths.

Build and install the wheel and source archive in clean environments. Exercise representative
synthetic workflows with known expectations and inspect saved figures. Verify supported
Python and minimum-dependency environments, package contents, versions and citation metadata.

Confirm canonical repository/documentation URLs, authorship, licence and release notes.
Add a version/date/DOI only when those facts are established; `Unreleased` must not be
presented as a published release. Cite the specific archived release when referring to a
version. Account connections, visibility changes, public hosting, tags and releases require
explicit maintainer authorisation.

## After verified publication

- Replace checkout-first installation in the README and user guide with the verified
  `python -m pip install calamari-magnetometry` command. Retain concise development guidance.
- Remove pre-release notices from the docs landing page and update the release-guide status.
- Record the actual release date, version and version-specific DOI in citation metadata
  and release notes. Do not move the release tag for these documentation updates.
- Check the live README, documentation, archive and index pages. Editing the README does
  not change the description embedded in distributions already uploaded to PyPI.

## Documentation hosting

The CI workflow builds Sphinx strictly on every PR. On pushes to main, it audits and
uploads only the generated HTML tree, then deploys through the `github-pages` environment
after the complete required-check set succeeds. Only the deployment job has Pages/OIDC
write permissions. Restrict that environment to the main branch. The repository can remain
private while its generated documentation is public; downloadable documentation sources
are part of the published site. Source history is not deployed.

Before advertising the site, inspect navigation, assets, mathematical rendering and local
reference links at its actual Pages URL. A successful local build is insufficient. Read the
Docs remains an optional prepared configuration, rather than the active host.

After a hosting or repository-settings change, verify the Pages environment, workflow
permissions, actual URL, navigation, logos, mathematical rendering and citation links.
Update source-access statements in the README, user guide, docs landing page and this
procedure when repository visibility changes. Retain pre-release notices until stable
installation is verified. Publishing requires a separately authorised manual dispatch and
environment approval.
