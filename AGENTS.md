# Repository Development Conventions

## English Comments

- When adding or modifying Python modules, the module, public classes, and important functions must include accurate English docstrings.
- Key control flow, security boundaries, reproducibility design, and non-obvious implementations must include English inline comments that focus on explaining *why* the code is designed this way.
- Do not add line-by-line code translations, comments that repeat variable names, or mechanically obvious comments; update related comments whenever the code changes.
- Test functions should state the behavior they protect or the regression they prevent; complex test setup steps need an English explanation.
- Important parameter groups in YAML and other experiment configs should use English comments to describe their purpose, but must not record unverified experimental results in comments.
- Identifiers, commands, file formats, third-party API names, and necessary technical terms stay in English to match the code and official documentation.

## Post-change Verification

- After Python changes, run the full test suite: `python -m pytest -q`.
- After experiment config changes, run both config validations and confirm the fingerprint changes match expectations.
- Before committing, run `git diff --check` to avoid introducing meaningless whitespace errors.
