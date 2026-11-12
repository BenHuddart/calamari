# Contributing to Calamari

Bug reports, documentation improvements and scientific corrections are welcome.
For substantial changes, open an issue first to agree the problem, approach and validation.
Small corrections can go straight to a pull request. Plans and implementation decisions
live in GitHub issues; the repository already provides issue and PR templates.

## Report a problem

Describe the expected and observed behaviour, the Calamari version or commit, Python and
relevant dependency versions, and a minimal runnable example. Use invented inputs or
`calamari.simulate` so the problem can be reproduced without your measurements.
Keep research files, identifying details and local research paths out of issues and PRs.

## AI-assisted contributions

Disclose substantive use of generative AI in the PR: name the tools, identify the affected
code, tests or prose, and explain how the contribution was checked. Contributors must
understand and review their submissions, verify scientific claims and citations, and
remain responsible for correctness and appropriate reuse of content. Follow the research-data
policy regardless of the tool used. Describe AI-assisted review accurately; do not present
it as human review. AI tools are not credited as software authors or contributors.

## Set up a development checkout

From a checkout, use Python 3.10–3.13 and uv. Run tooling through the declared groups:

```bash
uv run --group dev python -m pytest tests/
uv run --group dev python -m pytest --doctest-plus --doctest-modules src/calamari
uv run --group dev ruff check src tests
uv run --group docs sphinx-build -n -W --keep-going -b html docs docs/_build/html
```

`uv run` prepares the project environment. Start with tests relevant to your change;
implementation changes require the full tests, doctests and lint checks above. Build the
docs when behaviour or site content changes, and inspect affected pages and saved figures.
The required CI checks must pass before merging. The generated site is in
`docs/_build/html`; figures, notebooks, build output and the local `uv.lock` stay out of Git.

## Prepare a focused change

Explain the problem and resulting behaviour in the PR, link its issue when applicable,
and report validation and any limitations. Update essential documentation with the change.
Scientific corrections need explicit quantities, units, assumptions and justified tolerances,
with independent expectations or regression evidence. Keep instrument data external and
create synthetic regressions for observed file shapes.

Read the [working agreement](https://github.com/BenHuddart/calamari/blob/main/AGENTS.md) and [architecture contract](https://github.com/BenHuddart/calamari/blob/main/docs/ARCHITECTURE.md)
for maintained scientific, data and verification rules. Before changing parsing or
measurement accessors, read the
[loading guide](https://github.com/BenHuddart/calamari/blob/main/docs/USER_GUIDE.md)
and [unit conventions](https://github.com/BenHuddart/calamari/blob/main/docs/METHODS.md).
Ask the maintainer for source evidence when proposing a format change that the public
contracts and regression tests cannot settle. Keep detailed manual and observation records external.

Contributions are provided under the project's [MIT licence](https://github.com/BenHuddart/calamari/blob/main/LICENSE).
