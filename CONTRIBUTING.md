# Contributing

Keep changes small, attributable, and mathematically testable.

## Local gate

Use Python 3.9 or newer, then run:

    python3 -m pip install '.[test]'
    python3 -m pytest -q
    python3 -m compileall -q agent_representation_theory.py test_representation_theory.py
    python3 agent_representation_theory.py

New formulas should include a cited derivation in the pull-request description and tests for group laws, dimensions, and character identities. Bug fixes should include a failing regression test. Do not commit credentials, generated caches, local virtual environments, or build artifacts.

## Pull requests

Explain the behavior changed, the exact verification commands run, and any remaining limitation. A passing test suite supports only the tested claims; it does not establish empirical validity for agent behavior.
