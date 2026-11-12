# Calamari — working agreement

Conventions for anyone, human or agent, working on this repository.
Tool-agnostic; `CLAUDE.md` is a compatibility pointer to this agreement.

## Aim and scope

Calamari helps researchers analyse magnetometry measurements through repeatable
workflows, from loading and unit conversion to selection, reduction, detection and fitting.
Figures present the analysis: quantities and uncertainties must be defensible, and the
analysis recipe must be runnable again.

The parser is well evidenced. Build on its public contracts, regression tests and
maintainer-held source evidence; change parsing rules to address demonstrated defects or
new evidence. A parser rewrite is not a prerequisite for improving plotting, workflows or
documentation.

## Start here

1. Read `docs/ARCHITECTURE.md` for the binding design requirements and their rationale.
   Discuss changes to a requirement before implementing them.
2. Before changing `src/calamari/io/` or `Measurement` accessors, read the loading and
   units guides and relevant regressions. Preserve observed behaviour unless contrary
   evidence justifies a correction. Distinguish documentary support, empirical observations
   and unconfirmed hypotheses; synthetic tests cannot establish new instrument facts.
   Ask the maintainer for source evidence when the public contracts and tests cannot settle
   a format question. Detailed manual and observation records remain outside the repository.
3. Read the affected documentation: `docs/USER_GUIDE.md` covers basic workflows and
   `docs/METHODS.md` defines quantities, method assumptions and uncertainty conventions.
   `docs/api.md` is the API reference and `docs/index.md` the Sphinx entry point.

## Repeatable publication figures

- Provide a script or runnable notebook that takes declared inputs through to saved figures.
  Record data selection, moment-column choice, sample facts and units, repeat aggregation,
  screening, detection windows and thresholds, model and weighting choices, style and
  export settings where they affect the result. Record the Calamari revision or release
  and the environment needed to reproduce the analysis. Keep research inputs external.
- Separate measurement, analysis decisions and presentation. Preserve caller-supplied
  `ax=` composition and the existing plotting/style interfaces. Drawing a result must not
  silently redo detection, choose a scientific interpretation or change the data.
- Label quantities, units and normalisation correctly. Distinguish measured scatter,
  statistical errors, systematic uncertainty and sensitivity to analysis choices. A
  transition candidate is not automatically a phase boundary, and an uncalibrated score
  is not a probability or a number of Gaussian sigma.
- Render affected figures and inspect them at their intended physical size. Check readable
  lettering, unclipped labels and legends, distinguishable curves, uncertainty displays,
  panel layout and the saved output. Use journal styles and suitable export formats;
  successful execution alone does not establish publication quality. Report what was
  inspected. A change to a figure must be reproducible without unrecorded manual edits.
- Before release, exercise representative synthetic workflows through installed packages,
  compare numerical results with independent expectations and inspect saved figures.
  Explain differences; do not change thresholds merely to recover an expected result.

## Research data

Real instrument data is essential for development and must remain outside the repository.
Use authorised files locally, in place, when they are needed to settle a question the
existing evidence cannot answer. Manuals document vocabulary and meaning; real exports
establish which fields are present, populated and spelled differently in practice.
If that evidence cannot settle a format question and no suitable authorised file is
available, ask for a representative export rather than inventing its shape.

- Commit no instrument data file, in any format, however small. Generate synthetic test
  inputs with `calamari.simulate` in temporary directories.
- Keep real data paths and identifying research information out of code, comments, docs,
  tests, examples, changelogs, commit messages, issues, PRs and shared diagnostic output.
  This covers the research project, sample, compound, collaborators, site and institution;
  it does not prohibit public authorship or cited vendor/literature sources.
- Invent test values; do not copy a real sample's mass, molar mass or ion count into a test.
- When a file reveals a new shape, describe it generically, reproduce it synthetically
  and add a regression test. Record the non-identifying finding and rationale in its issue
  and relevant public guide; retain the detailed evidence mapping with the maintainer
  outside the repository. For measured comparisons, publish the method and an appropriately
  non-identifying comparison, not the underlying data. Internal consistency checks do not
  establish instrument-wide accuracy or uncertainty calibration.

Working-tree searches are useful guards, not proof of publication clearance. Before making
the repository public or archiving a release, audit reachable history and repository-hosted
discussions, logs and artifacts as well as current files. Report concerns to the maintainer;
do not rewrite history or delete remote records without explicit authorisation.

Any downstream analysis repository, and `asymmetry` / `gleplot`, are read-only references.
Never modify them from this repository's task.

## Evidence and decisions

Verify claims about code by running it and report the command and relevant result. Separate
observed execution, documentary evidence and inference. Synthetic tests check an
implementation against their assumptions; they cannot establish new instrument facts or
establish a physical interpretation on their own.

Prefer informative failures to guesses or partial results. Preserve documented absent-data
`None` / `NaN` values and unsuccessful fit results; do not use them to hide parsing or
validation failures. Errors should identify the relevant input, offending value, expectation
and available remedy. Cross-check metadata against data where possible; otherwise retain
its provenance so the caller can audit it.

Use `git log -S` and the evidence references to understand disagreements. Correct stale
comments and demonstrated defects, with regression tests where behaviour changes. Discuss
changes to public API shape, scientific definitions, defaults affecting physics or the
meaning of published numbers before implementation. Once agreed, proceed within that scope
without repeatedly asking for the same decision. Make routine choices and state assumptions.
Correct earlier errors plainly and keep unresolved questions visible.
Verify external review findings before acting on them or presenting them as established.

## Documentation

Sphinx sources already live in `docs/`, with a concise MyST user guide, method conventions
and an API reference generated from docstrings. Maintain this documentation with
the code. A local build or CI build does not establish that a public site is deployed;
verify hosting separately before publishing a site link. `.readthedocs.yaml` contains the
prepared hosting configuration.

- Keep release documentation focused on the basic user guide and API reference. Use short
  practical snippets with reader-supplied files; CI supplies invented fixtures externally.
  Retain essential scientific assumptions, units, uncertainty conventions and limitations.
  Substantive tutorials and physics exposition are deferred to maintainer-authored work.
- Document new behaviour and changed contracts in the same PR. Follow-up documentation
  PRs may improve navigation, coverage, examples and public hosting; do not defer essential
  units, defaults, uncertainty meanings or breaking-change guidance.
- Use NumPy-style docstrings to explain purpose and scientific meaning; add rationale where
  a reader needs it. Public APIs document units, assumptions and relevant limitations;
  simple private helpers usually need one sentence. Add maintained source pages to the
  Sphinx navigation. Generated API stubs and build output are build products
  and must not be edited or committed.
- Run examples and capture their actual outputs before publishing; do not hand-edit expected
  values to conceal a mismatch. `tests/test_guides.py` executes Python
  fences in maintained Markdown pages under `docs/`, in isolated sessions; the user guide receives file fixtures from `tests/_workflow.py`.
  Generated trees are not standalone guides.
  Exclusions live in that test. `<!-- output -->` marks literal output, including displayed expressions,
  warnings and expected exceptions. `<!-- run-first -->` marks setup run before the guide.
  Unmarked plain blocks are illustrative and not compared. The exact contract lives in
  `tests/_guide_runner.py`; do not loosen it to conceal drift.
- Build with `sphinx-build -n -W --keep-going`. Inspect affected rendered pages and figures;
  a warning-free build does not establish that the explanation or layout is correct. Keep
  reference suppressions narrow and justified. External link checking is a separate
  scheduled/on-demand workflow; distinguish remote outages from broken local references.

## Scientific code rules

1. **Separate calculations from side effects.** Numerical functions take explicit inputs
   and return results. Keep file access, plotting and environment configuration outside
   them; avoid hidden state and input mutation.
2. **Validate at public boundaries.** Check required shapes, alignment, units, ordering and
   physical domains. Define missing/non-finite-value handling; share validation rather than
   repeating competing checks in internal helpers. Add useful type annotations to public APIs.
3. **Make scientific choices explicit.** Identify source quantities, units, normalisation,
   weighting, filtering and model assumptions. Defaults need a scientific rationale;
   dataset-specific choices belong in the caller's recorded workflow.
4. **Preserve information and expose failures.** Keep full numerical precision until
   presentation. Dropping, clipping, rounding, averaging or substituting values requires an
   explicit, documented operation. Catch exceptions narrowly; keep invalid configuration
   distinct from documented unsuccessful results.
5. **Give each scientific rule one production implementation.** Share column resolution,
   conversions, uncertainty and selection rules. Calculations and provenance use the same
   resolved choices; tests may use independent reference implementations.
6. **Test scientific properties and public behaviour.** Use synthetic inputs, analytic
   answers, limiting cases, scaling relations and independent reference calculations where
   appropriate. A generator sharing the implementation's assumptions is not a sole oracle.
   New or corrected behaviour needs a regression that would fail before the change. Cover
   newly observed file shapes with synthetic tests, including shapes already supported.
   Do not add tests that merely mirror implementation or prose.
7. **Justify numerical tolerances.** Choose absolute and relative tolerances using the
   quantity's scale, numerical error and scientific requirement. Check expected shapes and
   finite/missing-value behaviour explicitly; do not loosen tolerances to conceal a defect.
8. **Match uncertainty claims to the method.** State error meanings and assumptions; separate
   statistical, systematic and analysis-choice effects. Probability or coverage claims need
   appropriate calibration, not just a successful fit or a large score.
9. **Prefer small, direct implementations.** Give functions a clear responsibility and
   consistent return types. Add abstractions for demonstrated shared concepts. Use established
   numerical libraries and profile representative workloads before optimising.
10. **Keep results reproducible and explanations executable.** Record resolved settings and
    software versions, pass explicit random generators where needed, and maintain runnable
    examples with known outcomes. Follow the publication-figure and documentation rules above.

## Test maintenance and coverage

Use explicit local or function-scoped seeded generators; tests must not depend on a shared
random stream or execution order. When changing random inputs or test organisation, check
affected tests individually and in a different order as well as in the full suite.

Each test protects a distinct behaviour or scientific property. Consolidate equivalent
cases and setup; identify the retained protection when removing a test. Preserve independent
references and evidenced format regressions. Matching coverage does not establish redundancy.

CI enforces 90% combined source line/branch coverage on Python 3.13, using tests and appended
doctests. Run `coverage run -m pytest tests/`, then `coverage run --append -m pytest
--doctest-plus --doctest-modules src/calamari` and `coverage report` through `uv run --group dev`.
Inspect per-module gaps and explain coverage changes; do not add exclusions or weaken
assertions merely to meet the floor. This measurement excludes child-process execution of
guide snippets; their checks still run. Coverage measures execution, not scientific validity.

## Concision and style

Use the simplest code and shortest explanation that preserve scientific meaning, correctness
and maintainability.

- Comments explain scientific assumptions, numerical traps and non-obvious decisions;
  avoid narrating assignments, loops or names. Document non-obvious constants and their
  provenance; self-explanatory aliases need no commentary.
- Add configuration options and compatibility layers for demonstrated needs. Avoid unnecessary
  wrappers, pass-through helpers, intermediate objects and repeated validation.
- Give explanations one home: detailed derivations in documentation, development history in
  issues, and essential rationale beside the code. Prefer descriptive names to explanatory
  prose while preserving standard physics notation.
- Before finishing, remove code or commentary that adds no behaviour, scientific meaning or
  necessary context. Use judgement rather than arbitrary function-length or comment-count gates.
- Use British spelling and a line length of 100. Preserve physics notation (`C`, `J`, `T`,
  `chi`, `mu_B`). Put necessary lint waivers in the relevant per-file `pyproject.toml` entry,
  with a comment naming the symbols; never blanket-disable a rule to accommodate them.

## Planning and handoffs

Implementation plans live in GitHub issues. When work needs a plan, find an existing issue
or create one before implementation. The issue is the durable record for motivation,
verified evidence, the proposed approach, scope, acceptance criteria, validation and open
decisions. Use `.github/ISSUE_TEMPLATE/work_item.md` as a starting point and scale the detail
to the work. A small correction that needs no standalone plan does not need a planning issue.

Read the issue and its discussion when starting or resuming work in another session. Keep
its scope and plan current as evidence changes; record agreed decisions, material deviations
and remaining work there. A suggested approach is not approval of an unresolved public API
or scientific decision. Resolve those with the maintainer before implementing them.

Do not add new repository-local planning documents or rely on a chat as the only handoff.
Architecture, format evidence, user guides and API reference belong in the repository
and must still be updated with the implementation; an issue does not replace maintained
documentation.

Link implementation PRs to their planning issue. Use closing keywords only when merging
the PR will complete that issue; use a related-issue link for partial or preparatory work.
The PR describes the resulting change and its validation, while the issue retains the plan,
decisions and unresolved work. `.github/pull_request_template.md` provides a concise outline.

## Branches, pull requests and protections

Never commit to `main`. For an authorised repository change, use one coherent branch,
push it and open a PR. Existing prefixes are `feature/` for capabilities, `fix/` for
corrections and `rebuild/` for a specifically agreed module rebuild.

Commit messages and PR descriptions explain the problem, evidence and resulting behaviour.
Breaking changes say so explicitly and are recorded in `CHANGELOG.md` under `Unreleased`.
A read-only review ends with findings; creating issues, changing settings, merging and
publishing follow the user's agreed scope and sequence.

Preserve PR-based protection of `main`, including administrator enforcement and prohibitions
on force pushes and deletion. Inspect current settings through read-only queries rather
than assuming they still match this document. Never verify a guard by performing the real
action it forbids: use inspection, an appropriate dry run, or an isolated test. A dry run
alone is not evidence that every remote protection is enforced. Check who any exemption
actually leaves subject to the rule. Changing protections requires explicit authorisation.

Merging to `main` requires up-to-date, successful GitHub Actions checks: `test (3.10)`,
`test (3.11)`, `test (3.12)`, `test (3.13)`, `lint`, `docs`, `data policy`, `distributions`
and `minimum dependencies`. Keep required job names stable and bind requirements to the
verified GitHub Actions application. Required jobs must run on every PR and report failed
prerequisites as failures. Verify check names, application identity and protection settings
through read-only API queries; merge through the protected PR process without bypasses.

## Verification before finishing

Use `uv run --group dev` / `--group docs` for tooling; do not assume an existing environment
contains it. Choose checks by the change and report their results and limitations:

- **Read-only review:** report inspected evidence and reproductions. Do not rerun unchanged
  checks without a reason, or turn the review into implementation work.
- **Guidance/prose:** check consistency, links and data policy. Run guide tests for affected
  executable examples and a strict Sphinx build for affected site content; inspect rendering.
- **Implementation:** run the full tests, doctests and lint commands below, plus the strict
  docs build when documented behaviour or rendered content is affected. Inspect affected
  figures. Do not repeat passing checks unless changes or unresolved concerns justify it.
- **Release:** also build and test wheel/source distributions, verify supported dependency
  environments and version/citation metadata, perform the publication audit, and exercise
  installed-package synthetic workflows. Release preparation is a separate task.

```bash
uv run --group dev python -m pytest tests/
uv run --group dev python -m pytest --doctest-plus --doctest-modules src/calamari
uv run --group dev ruff check src tests
uv run --group docs sphinx-build -n -W --keep-going -b html docs docs/_build/html
```

Check the changed/tracked files for real data, identifying information and local data paths.
CI's `data policy` job supplies fast working-tree guards; it does not inspect history or
prove that content is synthetic. Widen checks when the change warrants it. If a check fails
or cannot run, state why and whether it affects confidence in the change; do not call it
passed or weaken validation to obtain a green result.

Use heredocs or temporary directories for ad hoc checks and scratch files. Keep real-data
inputs outside the working tree. `uv.lock` is intentionally gitignored; do not commit it
incidentally. Dependency/environment changes should be deliberate and explained.
