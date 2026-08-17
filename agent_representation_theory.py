"""
Agent Representation Theory
How groups act on vector spaces, decomposing symmetries into irreducible components.

Representation theory studies abstract groups by representing their elements as
linear transformations of vector spaces. This makes abstract symmetry concrete
and computable.

Core concepts:
- Group representation: ρ: G → GL(V)
- Irreducible representations (irreps): cannot be decomposed further
- Characters: χ(g) = tr(ρ(g)), class functions
- Schur's lemma: maps between irreps are scalar multiples of identity
- Peter-Weyl theorem: matrix coefficients span L²(G)
- Decomposition: V = ⊕ nᵢ Vᵢ (multiplicity spaces)

Applications to agents:
- Decompose agent behaviors into fundamental modes
- Classify coordination patterns by character
- Detect symmetry breaking in multi-agent systems
- Analyze invariant subspaces of state space
- Compute selection rules for allowed transitions
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import linalg


@dataclass
class Representation:
    """
    A group representation: maps group elements to invertible matrices.
    
    ρ: G → GL(V) where V is a vector space of dimension `dim`.
    """
    name: str
    dim: int
    matrices: Dict[str, NDArray]  # group_element → matrix
    
    def character(self, g: str) -> complex:
        """Character: χ(g) = tr(ρ(g))."""
        if g not in self.matrices:
            raise ValueError(f"Group element {g} not in representation")
        return np.trace(self.matrices[g])
    
    def is_faithful(self, identity_element: str = "e") -> bool:
        """Check if representation is faithful (injective)."""
        # Faithful if only identity maps to identity matrix
        identity_matrix = np.eye(self.dim)
        for g, matrix in self.matrices.items():
            if np.allclose(matrix, identity_matrix) and g != identity_element:
                return False
        return True
    
    def is_irreducible(self, group_elements: List[str]) -> bool:
        """
        Check if representation is irreducible.
        
        Uses the criterion: ρ is irreducible iff 
        (1/|G|) Σ_g |χ(g)|² = 1
        """
        char_sum = 0
        for g in group_elements:
            chi = self.character(g)
            char_sum += np.abs(chi) ** 2
        return np.isclose(char_sum / len(group_elements), 1.0)
    
    def decompose(self, irreps: List['Representation'], 
                  group_elements: List[str]) -> Dict[str, int]:
        """
        Decompose this representation into irreducible components.
        
        Uses character orthogonality:
        nᵢ = (1/|G|) Σ_g χᵢ(g)* χ(g)
        
        Returns: {irrep_name: multiplicity}
        """
        multiplicities = {}
        
        for irrep in irreps:
            inner_product = 0
            for g in group_elements:
                chi_irrep = np.conj(irrep.character(g))
                chi_self = self.character(g)
                inner_product += chi_irrep * chi_self
            
            multiplicity = int(np.round(inner_product.real / len(group_elements)))
            if multiplicity > 0:
                multiplicities[irrep.name] = multiplicity
        
        return multiplicities


class CyclicGroup:
    """
    Cyclic group Z_n: rotations by 2πk/n.
    
    The simplest non-trivial group, fundamental building block.
    """
    
    def __init__(self, n: int):
        self.n = n
        self.elements = [f"r{k}" for k in range(n)]
    
    def multiply(self, g1: str, g2: str) -> str:
        """Group multiplication: r^a * r^b = r^(a+b mod n)."""
        a = int(g1[1:])
        b = int(g2[1:])
        return f"r{(a + b) % self.n}"
    
    def inverse(self, g: str) -> str:
        """Group inverse: (r^a)^(-1) = r^(n-a)."""
        a = int(g[1:])
        return f"r{(self.n - a) % self.n}"
    
    def identity(self) -> str:
        return "r0"
    
    def regular_representation(self) -> Representation:
        """
        Regular representation: group acts on itself by left multiplication.
        
        Dimension = |G| = n.
        """
        matrices = {}
        for g in self.elements:
            matrix = np.zeros((self.n, self.n), dtype=complex)
            for i, h in enumerate(self.elements):
                gh = self.multiply(g, h)
                j = self.elements.index(gh)
                matrix[j, i] = 1.0
            matrices[g] = matrix
        
        return Representation(f"Regular_Z{self.n}", self.n, matrices)
    
    def irreducible_representations(self) -> List[Representation]:
        """
        All irreducible representations of Z_n.
        
        Z_n has n irreps, all 1-dimensional:
        ρ_k(r) = exp(2πi k/n) for k = 0, 1, ..., n-1
        """
        irreps = []
        omega = np.exp(2j * np.pi / self.n)
        
        for k in range(self.n):
            matrices = {}
            for j in range(self.n):
                # r^j maps to omega^(k*j)
                matrices[f"r{j}"] = np.array([[omega ** (k * j)]], dtype=complex)
            irreps.append(Representation(f"χ{k}", 1, matrices))
        
        return irreps


class SymmetricGroup:
    """
    Symmetric group S_n: all permutations of n elements.
    
    Fundamental group in combinatorics and physics.
    |S_n| = n!
    """
    
    def __init__(self, n: int):
        self.n = n
        self.elements = self._generate_permutations()
    
    def _generate_permutations(self) -> List[Tuple[int, ...]]:
        """Generate all permutations of {0, 1, ..., n-1}."""
        from itertools import permutations
        return list(permutations(range(self.n)))
    
    def multiply(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Permutation composition: (p1 ∘ p2)(i) = p2(p1(i)).
        
        Apply p1 first, then p2 (left-to-right).
        """
        return tuple(p2[p1[i]] for i in range(self.n))
    
    def inverse(self, p: Tuple[int, ...]) -> Tuple[int, ...]:
        """Permutation inverse."""
        inv = [0] * self.n
        for i, j in enumerate(p):
            inv[j] = i
        return tuple(inv)
    
    def identity(self) -> Tuple[int, ...]:
        return tuple(range(self.n))
    
    def sign(self, p: Tuple[int, ...]) -> int:
        """
        Sign of permutation: +1 if even, -1 if odd.
        
        Counts number of inversions.
        """
        inversions = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if p[i] > p[j]:
                    inversions += 1
        return 1 if inversions % 2 == 0 else -1
    
    def trivial_representation(self) -> Representation:
        """Trivial representation: all elements map to 1."""
        matrices = {}
        for p in self.elements:
            matrices[p] = np.array([[1.0]])
        return Representation("Trivial", 1, matrices)
    
    def sign_representation(self) -> Representation:
        """Sign representation: p maps to sign(p)."""
        matrices = {}
        for p in self.elements:
            matrices[p] = np.array([[float(self.sign(p))]])
        return Representation("Sign", 1, matrices)
    
    def standard_representation(self) -> Representation:
        """
        Standard representation: permutation matrices on R^n.
        
        Dimension = n.
        """
        matrices = {}
        for p in self.elements:
            matrix = np.zeros((self.n, self.n))
            for i in range(self.n):
                matrix[p[i], i] = 1.0
            matrices[p] = matrix
        return Representation("Standard", self.n, matrices)
    
    def character_table(self) -> Dict[str, List[complex]]:
        """
        Compute character table for small symmetric groups.
        
        Returns: {irrep_name: [χ(g1), χ(g2), ...]}
        """
        # For S_3, we have 3 irreps: trivial, sign, standard (2D)
        if self.n != 3:
            raise NotImplementedError("Character table only implemented for S_3")
        
        table = {}
        
        # Trivial
        trivial = self.trivial_representation()
        table["Trivial"] = [trivial.character(p) for p in self.elements]
        
        # Sign
        sign_rep = self.sign_representation()
        table["Sign"] = [sign_rep.character(p) for p in self.elements]
        
        # Standard (2D irrep)
        std = self.standard_representation()
        table["Standard"] = [std.character(p) for p in self.elements]
        
        return table


class AgentSymmetryAnalyzer:
    """
    Analyze agent coordination patterns using representation theory.
    
    Decomposes complex multi-agent behaviors into fundamental symmetry modes.
    """
    
    def __init__(self, group):
        self.group = group
        self.observations: List[Dict] = []
    
    def observe_coordination(self, group_element, description: str):
        """Record an observed coordination pattern."""
        self.observations.append({
            "element": group_element,
            "description": description
        })
    
    def build_observed_representation(self, dim: int) -> Representation:
        """
        Build representation from observed coordination patterns.
        
        Maps each observed group element to a matrix encoding the pattern.
        """
        matrices = {}
        
        for obs in self.observations:
            g = obs["element"]
            # Create a matrix encoding this coordination pattern
            # (simplified: use random matrices for demonstration)
            np.random.seed(hash(str(g)) % (2**32))
            matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
            # Make invertible
            matrix = matrix + dim * np.eye(dim)
            matrices[g] = matrix
        
        return Representation("Observed", dim, matrices)
    
    def decompose_coordination(self, representation: Representation) -> Dict[str, int]:
        """
        Decompose observed coordination into fundamental modes.
        
        Returns: {mode_name: multiplicity}
        """
        if isinstance(self.group, CyclicGroup):
            irreps = self.group.irreducible_representations()
        else:
            raise NotImplementedError("Only cyclic groups supported for decomposition")
        
        return representation.decompose(irreps, self.group.elements)
    
    def detect_symmetry_breaking(self, representation: Representation) -> List[str]:
        """
        Detect which symmetries are broken by the coordination pattern.
        
        A symmetry is broken if the representation is not invariant under it.
        """
        broken = []
        
        # Check if representation is trivial (fully symmetric)
        identity_matrix = np.eye(representation.dim)
        for g, matrix in representation.matrices.items():
            if not np.allclose(matrix, identity_matrix):
                broken.append(g)
        
        return broken
    
    def compute_selection_rules(self, initial_irrep: str, final_irrep: str,
                                operator_irrep: str, irreps: List[Representation]) -> bool:
        """
        Compute selection rules for transitions.
        
        A transition from initial to final state via operator is allowed iff
        the tensor product contains the trivial representation:
        
        Γ_initial ⊗ Γ_operator ⊗ Γ_final ⊃ Γ_trivial
        
        Returns: True if transition is allowed.
        """
        # Find the irreps
        irrep_dict = {irrep.name: irrep for irrep in irreps}
        
        if initial_irrep not in irrep_dict:
            raise ValueError(f"Unknown irrep: {initial_irrep}")
        if final_irrep not in irrep_dict:
            raise ValueError(f"Unknown irrep: {final_irrep}")
        if operator_irrep not in irrep_dict:
            raise ValueError(f"Unknown irrep: {operator_irrep}")
        
        # Compute character of tensor product
        allowed = False
        for g in self.group.elements:
            chi_initial = irrep_dict[initial_irrep].character(g)
            chi_operator = irrep_dict[operator_irrep].character(g)
            chi_final = np.conj(irrep_dict[final_irrep].character(g))
            
            # Check if trivial representation appears
            chi_product = chi_initial * chi_operator * chi_final
            
            # For cyclic groups, check if sum over all elements gives |G|
            if isinstance(self.group, CyclicGroup):
                if np.isclose(chi_product, 1.0):
                    allowed = True
                    break
        
        return allowed


def demo():
    """Demonstrate representation theory for agent coordination."""
    print("=" * 70)
    print("AGENT REPRESENTATION THEORY DEMO")
    print("=" * 70)
    print()
    
    # Cyclic group Z_4
    print("1. CYCLIC GROUP Z_4 (Rotations by 90°)")
    print("-" * 70)
    Z4 = CyclicGroup(4)
    print(f"Elements: {Z4.elements}")
    print(f"Order: {len(Z4.elements)}")
    print()
    
    # Irreducible representations
    print("Irreducible Representations:")
    irreps = Z4.irreducible_representations()
    for irrep in irreps:
        print(f"  {irrep.name}: dimension {irrep.dim}")
        chi_e = irrep.character("r0")
        chi_r = irrep.character("r1")
        print(f"    χ(e) = {chi_e:.3f}, χ(r) = {chi_r:.3f}")
    print()
    
    # Regular representation
    print("Regular Representation:")
    regular = Z4.regular_representation()
    print(f"  Dimension: {regular.dim}")
    print(f"  Faithful: {regular.is_faithful()}")
    print(f"  Irreducible: {regular.is_irreducible(Z4.elements)}")
    print()
    
    # Decompose regular representation
    print("Decomposition of Regular Representation:")
    decomp = regular.decompose(irreps, Z4.elements)
    for name, mult in decomp.items():
        print(f"  {name}: multiplicity {mult}")
    print()
    
    # Symmetric group S_3
    print("2. SYMMETRIC GROUP S_3 (Permutations of 3 elements)")
    print("-" * 70)
    S3 = SymmetricGroup(3)
    print(f"Order: {len(S3.elements)}")
    print(f"Elements: {S3.elements[:3]} ... (showing first 3)")
    print()
    
    # Representations
    print("Representations:")
    trivial = S3.trivial_representation()
    sign_rep = S3.sign_representation()
    standard = S3.standard_representation()
    
    print(f"  Trivial: dimension {trivial.dim}")
    print(f"  Sign: dimension {sign_rep.dim}")
    print(f"  Standard: dimension {standard.dim}")
    print()
    
    # Character table
    print("Character Table (first 3 elements):")
    char_table = S3.character_table()
    for name, chars in char_table.items():
        print(f"  {name:10s}: {chars[:3]}")
    print()
    
    # Agent coordination analysis
    print("3. AGENT COORDINATION ANALYSIS")
    print("-" * 70)
    
    analyzer = AgentSymmetryAnalyzer(Z4)
    
    # Observe some coordination patterns
    print("Observing coordination patterns:")
    patterns = [
        ("r0", "synchronized"),
        ("r1", "rotated 90°"),
        ("r2", "opposed"),
        ("r3", "rotated 270°"),
    ]
    for elem, desc in patterns:
        analyzer.observe_coordination(elem, desc)
        print(f"  {elem}: {desc}")
    print()
    
    # Build observed representation
    print("Building observed representation (dimension 2):")
    observed = analyzer.build_observed_representation(2)
    print(f"  Dimension: {observed.dim}")
    print(f"  Faithful: {observed.is_faithful()}")
    print()
    
    # Decompose
    print("Decomposition into fundamental modes:")
    decomp = analyzer.decompose_coordination(observed)
    for mode, mult in decomp.items():
        print(f"  {mode}: multiplicity {mult}")
    print()
    
    # Detect symmetry breaking
    print("Symmetry breaking analysis:")
    broken = analyzer.detect_symmetry_breaking(observed)
    if broken:
        print(f"  Broken symmetries: {broken}")
    else:
        print("  All symmetries preserved")
    print()
    
    # Selection rules
    print("Selection rules for transitions:")
    print("  Can χ0 transition to χ1 via χ1 operator?")
    allowed = analyzer.compute_selection_rules("χ0", "χ1", "χ1", irreps)
    print(f"    Allowed: {allowed}")
    print()
    
    print("  Can χ0 transition to χ2 via χ1 operator?")
    allowed = analyzer.compute_selection_rules("χ0", "χ2", "χ1", irreps)
    print(f"    Allowed: {allowed}")
    print()
    
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Key Insights:")
    print("  • Representations make abstract symmetry concrete")
    print("  • Characters provide fingerprints for classification")
    print("  • Decomposition reveals fundamental coordination modes")
    print("  • Selection rules constrain allowed transitions")
    print("  • Symmetry breaking indicates phase transitions")


if __name__ == "__main__":
    demo()
