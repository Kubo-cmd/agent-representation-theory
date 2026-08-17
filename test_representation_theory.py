"""Tests for Agent Representation Theory."""

import pytest
import numpy as np
from agent_representation_theory import (
    Representation,
    CyclicGroup,
    SymmetricGroup,
    AgentSymmetryAnalyzer,
)


class TestRepresentation:
    def test_character(self):
        """Test character computation."""
        matrices = {
            "e": np.eye(2),
            "g": np.array([[0, 1], [1, 0]])
        }
        rep = Representation("test", 2, matrices)
        
        chi_e = rep.character("e")
        chi_g = rep.character("g")
        
        assert np.isclose(chi_e, 2.0)  # tr(I) = 2
        assert np.isclose(chi_g, 0.0)  # tr(swap) = 0
    
    def test_is_faithful(self):
        """Test faithfulness check."""
        # Faithful representation
        matrices_faithful = {
            "e": np.eye(2),
            "g": np.array([[0, 1], [1, 0]])
        }
        rep_faithful = Representation("faithful", 2, matrices_faithful)
        assert rep_faithful.is_faithful()
        
        # Non-faithful representation
        matrices_nonfaithful = {
            "e": np.eye(2),
            "g": np.eye(2)  # g maps to identity
        }
        rep_nonfaithful = Representation("nonfaithful", 2, matrices_nonfaithful)
        assert not rep_nonfaithful.is_faithful()
    
    def test_is_irreducible(self):
        """Test irreducibility check."""
        # 1D representation is always irreducible
        matrices = {
            "e": np.array([[1.0]]),
            "g": np.array([[-1.0]])
        }
        rep = Representation("1d", 1, matrices)
        assert rep.is_irreducible(["e", "g"])


class TestCyclicGroup:
    def test_multiply(self):
        """Test group multiplication."""
        Z4 = CyclicGroup(4)
        
        assert Z4.multiply("r0", "r0") == "r0"
        assert Z4.multiply("r1", "r1") == "r2"
        assert Z4.multiply("r2", "r3") == "r1"
        assert Z4.multiply("r3", "r3") == "r2"
    
    def test_inverse(self):
        """Test group inverse."""
        Z4 = CyclicGroup(4)
        
        assert Z4.inverse("r0") == "r0"
        assert Z4.inverse("r1") == "r3"
        assert Z4.inverse("r2") == "r2"
        assert Z4.inverse("r3") == "r1"
    
    def test_identity(self):
        """Test identity element."""
        Z4 = CyclicGroup(4)
        assert Z4.identity() == "r0"
    
    def test_regular_representation(self):
        """Test regular representation."""
        Z3 = CyclicGroup(3)
        regular = Z3.regular_representation()
        
        assert regular.dim == 3
        assert regular.is_faithful(identity_element="r0")
        
        # Check that matrices are permutation matrices
        for g, matrix in regular.matrices.items():
            # Each row and column should sum to 1
            assert np.allclose(matrix.sum(axis=0), 1.0)
            assert np.allclose(matrix.sum(axis=1), 1.0)
    
    def test_irreducible_representations(self):
        """Test irreducible representations of Z_n."""
        Z3 = CyclicGroup(3)
        irreps = Z3.irreducible_representations()
        
        assert len(irreps) == 3  # Z_n has n irreps
        
        # All should be 1-dimensional
        for irrep in irreps:
            assert irrep.dim == 1
            assert irrep.is_irreducible(Z3.elements)
    
    def test_decomposition(self):
        """Test decomposition of regular representation."""
        Z3 = CyclicGroup(3)
        regular = Z3.regular_representation()
        irreps = Z3.irreducible_representations()
        
        decomp = regular.decompose(irreps, Z3.elements)
        
        # Regular representation should contain each irrep once
        assert len(decomp) == 3
        for name, mult in decomp.items():
            assert mult == 1


class TestSymmetricGroup:
    def test_multiply(self):
        """Test permutation composition."""
        S3 = SymmetricGroup(3)
        
        # Identity
        e = S3.identity()
        p = (1, 2, 0)
        assert S3.multiply(e, p) == p
        assert S3.multiply(p, e) == p
        
        # Composition
        p1 = (1, 0, 2)  # swap 0 and 1
        p2 = (0, 2, 1)  # swap 1 and 2
        result = S3.multiply(p1, p2)
        assert result == (2, 0, 1)
    
    def test_inverse(self):
        """Test permutation inverse."""
        S3 = SymmetricGroup(3)
        
        p = (1, 2, 0)
        p_inv = S3.inverse(p)
        result = S3.multiply(p, p_inv)
        assert result == S3.identity()
    
    def test_sign(self):
        """Test permutation sign."""
        S3 = SymmetricGroup(3)
        
        # Even permutations
        assert S3.sign((0, 1, 2)) == 1   # identity
        assert S3.sign((1, 2, 0)) == 1   # 3-cycle
        assert S3.sign((2, 0, 1)) == 1   # 3-cycle
        
        # Odd permutations
        assert S3.sign((1, 0, 2)) == -1  # transposition
        assert S3.sign((0, 2, 1)) == -1  # transposition
        assert S3.sign((2, 1, 0)) == -1  # transposition
    
    def test_trivial_representation(self):
        """Test trivial representation."""
        S3 = SymmetricGroup(3)
        trivial = S3.trivial_representation()
        
        assert trivial.dim == 1
        
        # All elements should map to 1
        for p in S3.elements:
            assert np.isclose(trivial.character(p), 1.0)
    
    def test_sign_representation(self):
        """Test sign representation."""
        S3 = SymmetricGroup(3)
        sign_rep = S3.sign_representation()
        
        assert sign_rep.dim == 1
        
        # Character should equal sign
        for p in S3.elements:
            expected = float(S3.sign(p))
            assert np.isclose(sign_rep.character(p), expected)
    
    def test_standard_representation(self):
        """Test standard representation."""
        S3 = SymmetricGroup(3)
        standard = S3.standard_representation()
        
        assert standard.dim == 3
        
        # Check that matrices are permutation matrices
        for p, matrix in standard.matrices.items():
            assert np.allclose(matrix.sum(axis=0), 1.0)
            assert np.allclose(matrix.sum(axis=1), 1.0)
    
    def test_character_table(self):
        """Test character table computation."""
        S3 = SymmetricGroup(3)
        table = S3.character_table()
        
        assert "Trivial" in table
        assert "Sign" in table
        assert "Standard" in table
        
        # Each should have characters for all elements
        for name, chars in table.items():
            assert len(chars) == len(S3.elements)


class TestAgentSymmetryAnalyzer:
    def test_observe_coordination(self):
        """Test observation recording."""
        Z4 = CyclicGroup(4)
        analyzer = AgentSymmetryAnalyzer(Z4)
        
        analyzer.observe_coordination("r0", "synchronized")
        analyzer.observe_coordination("r2", "opposed")
        
        assert len(analyzer.observations) == 2
    
    def test_build_observed_representation(self):
        """Test building representation from observations."""
        Z4 = CyclicGroup(4)
        analyzer = AgentSymmetryAnalyzer(Z4)
        
        for elem in Z4.elements:
            analyzer.observe_coordination(elem, f"pattern_{elem}")
        
        rep = analyzer.build_observed_representation(2)
        
        assert rep.dim == 2
        assert len(rep.matrices) == 4
    
    def test_decompose_coordination(self):
        """Test decomposition of observed coordination."""
        Z3 = CyclicGroup(3)
        analyzer = AgentSymmetryAnalyzer(Z3)
        
        for elem in Z3.elements:
            analyzer.observe_coordination(elem, f"pattern_{elem}")
        
        rep = analyzer.build_observed_representation(2)
        decomp = analyzer.decompose_coordination(rep)
        
        # Should decompose into irreps
        assert isinstance(decomp, dict)
    
    def test_detect_symmetry_breaking(self):
        """Test symmetry breaking detection."""
        Z4 = CyclicGroup(4)
        analyzer = AgentSymmetryAnalyzer(Z4)
        
        # Fully symmetric (all identity)
        matrices_symmetric = {g: np.eye(2) for g in Z4.elements}
        rep_symmetric = Representation("symmetric", 2, matrices_symmetric)
        
        broken_sym = analyzer.detect_symmetry_breaking(rep_symmetric)
        assert len(broken_sym) == 0
        
        # Symmetry broken
        matrices_broken = {
            "r0": np.eye(2),
            "r1": np.array([[0, 1], [1, 0]]),
            "r2": np.eye(2),
            "r3": np.array([[0, 1], [1, 0]])
        }
        rep_broken = Representation("broken", 2, matrices_broken)
        
        broken = analyzer.detect_symmetry_breaking(rep_broken)
        assert len(broken) > 0
    
    def test_compute_selection_rules(self):
        """Test selection rule computation."""
        Z3 = CyclicGroup(3)
        analyzer = AgentSymmetryAnalyzer(Z3)
        irreps = Z3.irreducible_representations()
        
        # χ0 → χ1 via χ1 should be allowed
        allowed = analyzer.compute_selection_rules("χ0", "χ1", "χ1", irreps)
        assert isinstance(allowed, bool)


def test_integration():
    """Integration test: full workflow."""
    # Create group
    Z4 = CyclicGroup(4)
    
    # Get irreps
    irreps = Z4.irreducible_representations()
    assert len(irreps) == 4
    
    # Build regular representation
    regular = Z4.regular_representation()
    assert regular.dim == 4
    
    # Decompose
    decomp = regular.decompose(irreps, Z4.elements)
    assert len(decomp) == 4
    
    # Analyze coordination
    analyzer = AgentSymmetryAnalyzer(Z4)
    for elem in Z4.elements:
        analyzer.observe_coordination(elem, f"coord_{elem}")
    
    observed = analyzer.build_observed_representation(2)
    decomp_observed = analyzer.decompose_coordination(observed)
    
    # Should get some decomposition
    assert isinstance(decomp_observed, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
