# Agent Representation Theory

A small, tested Python reference for finite-group representations and agent-symmetry experiments.

The project turns abstract symmetry into matrices that can be validated, decomposed, and inspected. It supports cyclic groups and symmetric groups S2 through S8; the built-in complete character table is limited to S3. The agent layer builds a deterministic synthetic representation from a complete set of labeled observations; it does not claim to infer a physical or learned representation from text.

## What is implemented

- finite-dimensional complex representations with shape and finiteness checks
- character evaluation, faithfulness checks, unitarity checks, and homomorphism validation
- regular and irreducible representations of cyclic groups
- trivial, sign, permutation, and true `(n - 1)`-dimensional standard representations of symmetric groups
- character-inner-product decomposition
- deterministic synthetic coordination representations for cyclic groups
- selection rules evaluated over the complete group

For a finite group `G`, multiplicity is computed as:

`m_i = (1 / |G|) sum_g conjugate(chi_i(g)) chi(g)`

A transition from initial mode `i` to final mode `f` through operator mode `o` is allowed when:

`(1 / |G|) sum_g chi_i(g) chi_o(g) conjugate(chi_f(g))`

is a positive integer.

## Install

Python 3.9 or newer is required.

    python3 -m pip install '.[test]'

Runtime dependency: NumPy. SciPy is not required.

## Verify

    python3 -m pytest -q
    python3 -m compileall -q agent_representation_theory.py test_representation_theory.py
    python3 agent_representation_theory.py

## Minimal example

    from agent_representation_theory import CyclicGroup

    group = CyclicGroup(4)
    regular = group.regular_representation()
    irreps = group.irreducible_representations()

    assert regular.is_representation(group.elements, group.multiply, group.identity())
    assert regular.decompose(irreps, group.elements) == {
        "χ0": 1,
        "χ1": 1,
        "χ2": 1,
        "χ3": 1,
    }

## Scope and limits

- Symmetric-group character tables are implemented only for S3.
- Symmetric groups are capped at S8 because this compact implementation materializes all `n!` elements.
- The observation builder is deterministic and mathematically valid, but synthetic. Descriptions select a reproducible mixture of cyclic irreducible modes; they are not embedded or learned.
- Inputs are validated for the supported finite groups. This is not a symbolic algebra system.
- Floating-point comparisons use configurable tolerances.

## Agent Math Series

This repository is one part of a series exploring mathematical structures for agent systems:

- agent-category-theory — composition algebra
- agent-lie-groups — continuous symmetry
- agent-knot-theory — entanglement invariants
- agent-tqft — topological quantum field theory
- agent-operad-theory — multi-input composition
- agent-homotopy-type-theory — identity types
- agent-topos-theory — logic and geometry
- agent-sheaf-theory — local-to-global data
- agent-representation-theory — symmetry representations
- agent-information-geometry — Fisher information
- agent-wasserstein-geometry — optimal transport
- agent-memory-topology — persistent homology

## Contributing and security

See `CONTRIBUTING.md` for the local verification gate and `SECURITY.md` for private vulnerability reporting guidance.

## License

MIT. See `LICENSE`.
