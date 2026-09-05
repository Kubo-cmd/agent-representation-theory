"""Tests for finite-group representation tools."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from agent_representation_theory import (
    AgentSymmetryAnalyzer,
    CyclicGroup,
    Representation,
    SymmetricGroup,
)


class TestRepresentation:
    def test_rejects_invalid_dimension_and_shape(self):
        with pytest.raises(ValueError, match="positive"):
            Representation("bad", 0, {})
        with pytest.raises(ValueError, match="shape"):
            Representation("bad", 2, {"e": np.eye(3)})

    def test_character_and_missing_element(self):
        rep = Representation(
            "test", 2, {"e": np.eye(2), "g": np.array([[0, 1], [1, 0]])}
        )
        assert np.isclose(rep.character("e"), 2.0)
        assert np.isclose(rep.character("g"), 0.0)
        with pytest.raises(ValueError, match="not in representation"):
            rep.character("missing")

    def test_faithfulness_requires_identity_matrix(self):
        faithful = Representation(
            "faithful", 2, {"e": np.eye(2), "g": np.array([[0, 1], [1, 0]])}
        )
        assert faithful.is_faithful()

        kernel = Representation("kernel", 2, {"e": np.eye(2), "g": np.eye(2)})
        assert not kernel.is_faithful()

        bad_identity = Representation("bad", 1, {"e": np.array([[2.0]])})
        assert not bad_identity.is_faithful()

    def test_faithfulness_can_infer_nonstandard_identity_label(self):
        group = CyclicGroup(3)
        assert group.regular_representation().is_faithful()

    def test_rejects_invalid_tolerances(self):
        rep = Representation("trivial", 1, {"e": np.eye(1)})
        for value in (float("inf"), float("nan"), -1.0, True):
            with pytest.raises(ValueError, match="atol"):
                rep.is_unitary(atol=value)

    def test_rejects_duplicate_or_missing_group_elements(self):
        rep = Representation("trivial", 1, {"e": np.ones((1, 1))})
        with pytest.raises(ValueError, match="duplicates"):
            rep.is_irreducible(["e", "e"])
        with pytest.raises(ValueError, match="missing"):
            rep.is_irreducible(["e", "g"])


class TestCyclicGroup:
    def test_rejects_bad_order_and_elements(self):
        for value in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="positive integer"):
                CyclicGroup(value)
        group = CyclicGroup(4)
        for value in ("x1", "r4", "r-1", 1):
            with pytest.raises(ValueError, match="valid element"):
                group.inverse(value)

    def test_group_laws(self):
        group = CyclicGroup(4)
        for first in group.elements:
            assert group.multiply(group.identity(), first) == first
            assert group.multiply(first, group.identity()) == first
            assert group.multiply(first, group.inverse(first)) == group.identity()
            for second in group.elements:
                for third in group.elements:
                    assert group.multiply(group.multiply(first, second), third) == group.multiply(
                        first, group.multiply(second, third)
                    )

    def test_regular_representation_is_valid_and_faithful(self):
        group = CyclicGroup(3)
        regular = group.regular_representation()
        assert regular.dim == 3
        assert regular.is_faithful(group.identity())
        assert regular.is_representation(group.elements, group.multiply, group.identity())
        assert regular.is_unitary()

    def test_irreducibles_and_regular_decomposition(self):
        group = CyclicGroup(5)
        irreps = group.irreducible_representations()
        assert len(irreps) == 5
        assert all(rep.dim == 1 for rep in irreps)
        assert all(rep.is_irreducible(group.elements) for rep in irreps)
        assert group.regular_representation().decompose(irreps, group.elements) == {
            f"χ{k}": 1 for k in range(5)
        }


class TestSymmetricGroup:
    def test_rejects_bad_order_and_permutations(self):
        for value in (1, 0, True, 2.5):
            with pytest.raises(ValueError, match="at least 2"):
                SymmetricGroup(value)
        group = SymmetricGroup(3)
        for value in (
            (0, 1),
            (0, 0, 2),
            (0, 1, 3),
            (0.0, 1.0, 2.0),
            (False, True, 2),
            [0, 1, 2],
        ):
            with pytest.raises(ValueError, match="valid permutation"):
                group.inverse(value)

    def test_multiply_uses_standard_composition(self):
        group = SymmetricGroup(3)
        first = (1, 0, 2)
        second = (0, 2, 1)
        assert group.multiply(first, second) == (1, 2, 0)

    def test_group_laws(self):
        group = SymmetricGroup(3)
        for first in group.elements:
            assert group.multiply(group.identity(), first) == first
            assert group.multiply(first, group.identity()) == first
            assert group.multiply(first, group.inverse(first)) == group.identity()
            for second in group.elements:
                for third in group.elements:
                    assert group.multiply(group.multiply(first, second), third) == group.multiply(
                        first, group.multiply(second, third)
                    )

    def test_sign_is_multiplicative(self):
        group = SymmetricGroup(3)
        for first in group.elements:
            for second in group.elements:
                product_sign = group.sign(group.multiply(first, second))
                assert product_sign == group.sign(first) * group.sign(second)

    def test_named_representations_are_valid(self):
        group = SymmetricGroup(3)
        for rep in (
            group.trivial_representation(),
            group.sign_representation(),
            group.permutation_representation(),
            group.standard_representation(),
        ):
            assert rep.is_representation(group.elements, group.multiply, group.identity())
            assert rep.is_unitary()

    def test_standard_representation_is_two_dimensional_irrep_for_s3(self):
        group = SymmetricGroup(3)
        standard = group.standard_representation()
        assert standard.dim == 2
        assert standard.is_irreducible(group.elements)
        identity_character = standard.character(group.identity())
        transposition_character = standard.character((1, 0, 2))
        cycle_character = standard.character((1, 2, 0))
        assert np.isclose(identity_character, 2)
        assert np.isclose(transposition_character, 0)
        assert np.isclose(cycle_character, -1)

    def test_character_table_has_complete_s3_irreps(self):
        group = SymmetricGroup(3)
        table = group.character_table()
        assert set(table) == {"Trivial", "Sign", "Standard"}
        assert all(len(chars) == len(group.elements) for chars in table.values())
        irreps = [
            group.trivial_representation(),
            group.sign_representation(),
            group.standard_representation(),
        ]
        assert sum(rep.dim**2 for rep in irreps) == len(group.elements)


class TestAgentSymmetryAnalyzer:
    def _observed(self, order: int = 4, dim: int = 3):
        group = CyclicGroup(order)
        analyzer = AgentSymmetryAnalyzer(group)
        for element in group.elements:
            analyzer.observe_coordination(element, f"pattern-{element}")
        return group, analyzer, analyzer.build_observed_representation(dim)

    def test_observations_validate_and_replace_by_element(self):
        group = CyclicGroup(3)
        analyzer = AgentSymmetryAnalyzer(group)
        with pytest.raises(ValueError, match="group element"):
            analyzer.observe_coordination("r3", "bad")
        with pytest.raises(ValueError, match="non-empty"):
            analyzer.observe_coordination("r0", "")
        analyzer.observe_coordination("r0", "first")
        analyzer.observe_coordination("r0", "second")
        assert analyzer.observations == [{"element": "r0", "description": "second"}]

    def test_observed_representation_requires_full_coverage(self):
        group = CyclicGroup(3)
        analyzer = AgentSymmetryAnalyzer(group)
        analyzer.observe_coordination("r0", "only one")
        with pytest.raises(ValueError, match="one observation"):
            analyzer.build_observed_representation(2)

    def test_observed_representation_is_deterministic_and_valid(self):
        group, analyzer, first = self._observed()
        second = analyzer.build_observed_representation(3)
        assert first.is_representation(group.elements, group.multiply, group.identity())
        assert first.is_unitary()
        for element in group.elements:
            assert np.allclose(first.matrices[element], second.matrices[element])

    def test_observed_representation_is_stable_across_processes(self):
        script = """
import json
from agent_representation_theory import AgentSymmetryAnalyzer, CyclicGroup
g = CyclicGroup(4)
a = AgentSymmetryAnalyzer(g)
for e in g.elements:
    a.observe_coordination(e, f'pattern-{e}')
r = a.build_observed_representation(3)
encoded = {e: [[str(v) for v in row] for row in m] for e, m in r.matrices.items()}
print(json.dumps(encoded, sort_keys=True))
"""
        first = subprocess.check_output([sys.executable, "-c", script], text=True)
        second = subprocess.check_output([sys.executable, "-c", script], text=True)
        assert json.loads(first) == json.loads(second)

    def test_decomposition_matches_dimension(self):
        _, analyzer, observed = self._observed(order=4, dim=6)
        decomposition = analyzer.decompose_coordination(observed)
        assert sum(decomposition.values()) == observed.dim

    def test_detect_nontrivial_actions(self):
        group = CyclicGroup(4)
        analyzer = AgentSymmetryAnalyzer(group)
        symmetric = Representation("symmetric", 2, {g: np.eye(2) for g in group.elements})
        assert analyzer.detect_symmetry_breaking(symmetric) == []
        regular = group.regular_representation()
        assert set(analyzer.detect_symmetry_breaking(regular)) == {"r1", "r2", "r3"}

    def test_selection_rules_use_full_character_inner_product(self):
        group = CyclicGroup(3)
        analyzer = AgentSymmetryAnalyzer(group)
        irreps = group.irreducible_representations()
        assert analyzer.compute_selection_rules("χ0", "χ1", "χ1", irreps)
        assert not analyzer.compute_selection_rules("χ0", "χ2", "χ1", irreps)
        with pytest.raises(ValueError, match="Unknown irrep"):
            analyzer.compute_selection_rules("unknown", "χ1", "χ1", irreps)


def test_demo_runs(capsys):
    from agent_representation_theory import demo

    demo()
    output = capsys.readouterr().out
    assert "DEMONSTRATION COMPLETE" in output
    assert "Observed representation valid: True" in output
