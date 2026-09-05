"""Finite-group representation tools for agent-symmetry experiments.

The module implements exact group actions for cyclic groups and small symmetric
groups. The observation adapter is explicitly synthetic: labels choose a stable
mixture of cyclic irreducible modes, rather than pretending to learn a group
action from text.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import permutations
from numbers import Real
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

Element = Hashable
Multiply = Callable[[Element, Element], Element]


def _tolerance(value: float) -> float:
    """Return a finite, non-negative numerical tolerance."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("atol must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("atol must be a finite non-negative number")
    return result


def _group_elements(elements: Sequence[Element]) -> Tuple[Element, ...]:
    """Return a validated, non-empty tuple of unique group elements."""
    normalized = tuple(elements)
    if not normalized:
        raise ValueError("group_elements must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("group_elements contains duplicates")
    return normalized


@dataclass
class Representation:
    """A finite-dimensional complex representation ``rho: G -> GL(V)``."""

    name: str
    dim: int
    matrices: Mapping[Element, NDArray]

    def __post_init__(self) -> None:
        if isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim < 1:
            raise ValueError("dim must be a positive integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

        checked: Dict[Element, NDArray] = {}
        for element, value in self.matrices.items():
            matrix = np.asarray(value, dtype=complex)
            if matrix.shape != (self.dim, self.dim):
                raise ValueError(
                    f"matrix for {element!r} has shape {matrix.shape}; "
                    f"expected {(self.dim, self.dim)}"
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"matrix for {element!r} contains non-finite values")
            checked[element] = matrix.copy()
        self.matrices = checked

    def _require_elements(self, group_elements: Sequence[Element]) -> Tuple[Element, ...]:
        elements = _group_elements(group_elements)
        missing = [element for element in elements if element not in self.matrices]
        if missing:
            raise ValueError(f"representation is missing group elements: {missing!r}")
        return elements

    def character(self, element: Element) -> complex:
        """Return the character ``chi(element) = trace(rho(element))``."""
        if element not in self.matrices:
            raise ValueError(f"Group element {element!r} not in representation")
        return complex(np.trace(self.matrices[element]))

    def is_faithful(
        self,
        identity_element: Optional[Element] = None,
        atol: float = 1e-9,
    ) -> bool:
        """Return whether only the identity maps to the identity matrix.

        When no identity label is supplied, it is inferred only if exactly one
        represented element maps to the identity matrix.
        """
        atol = _tolerance(atol)
        identity = np.eye(self.dim, dtype=complex)
        if identity_element is None:
            candidates = [
                element
                for element, matrix in self.matrices.items()
                if np.allclose(matrix, identity, atol=atol, rtol=0)
            ]
            if len(candidates) != 1:
                return False
            identity_element = candidates[0]
        if identity_element not in self.matrices:
            return False
        if not np.allclose(self.matrices[identity_element], identity, atol=atol, rtol=0):
            return False
        return all(
            element == identity_element
            or not np.allclose(matrix, identity, atol=atol, rtol=0)
            for element, matrix in self.matrices.items()
        )

    def is_unitary(self, atol: float = 1e-9) -> bool:
        """Return whether every matrix is unitary."""
        atol = _tolerance(atol)
        identity = np.eye(self.dim, dtype=complex)
        return all(
            np.allclose(matrix.conj().T @ matrix, identity, atol=atol, rtol=0)
            for matrix in self.matrices.values()
        )

    def is_representation(
        self,
        group_elements: Sequence[Element],
        multiply: Multiply,
        identity_element: Element,
        atol: float = 1e-9,
    ) -> bool:
        """Check identity, invertibility, closure, and the homomorphism law."""
        atol = _tolerance(atol)
        elements = self._require_elements(group_elements)
        if identity_element not in elements:
            raise ValueError("identity_element is not in group_elements")
        identity = np.eye(self.dim, dtype=complex)
        if not np.allclose(self.matrices[identity_element], identity, atol=atol, rtol=0):
            return False
        if any(abs(np.linalg.det(self.matrices[element])) <= atol for element in elements):
            return False
        element_set = set(elements)
        for first in elements:
            for second in elements:
                product = multiply(first, second)
                if product not in element_set:
                    return False
                if not np.allclose(
                    self.matrices[product],
                    self.matrices[first] @ self.matrices[second],
                    atol=atol,
                    rtol=0,
                ):
                    return False
        return True

    def character_inner_product(
        self,
        other: "Representation",
        group_elements: Sequence[Element],
    ) -> complex:
        """Return ``<self, other>`` using the finite-group character product."""
        elements = self._require_elements(group_elements)
        other._require_elements(elements)
        return sum(
            np.conj(self.character(element)) * other.character(element)
            for element in elements
        ) / len(elements)

    def is_irreducible(
        self, group_elements: Sequence[Element], atol: float = 1e-8
    ) -> bool:
        """Apply the character norm criterion to an already-validated action.

        Call :meth:`is_representation` first when matrices come from an
        untrusted or external source.
        """
        atol = _tolerance(atol)
        norm = self.character_inner_product(self, group_elements)
        return bool(np.isclose(norm, 1.0, atol=atol, rtol=0))

    def decompose(
        self,
        irreps: Sequence["Representation"],
        group_elements: Sequence[Element],
        atol: float = 1e-8,
    ) -> Dict[str, int]:
        """Decompose an already-validated representation by characters."""
        atol = _tolerance(atol)
        elements = self._require_elements(group_elements)
        names = [irrep.name for irrep in irreps]
        if len(set(names)) != len(names):
            raise ValueError("irrep names must be unique")

        multiplicities: Dict[str, int] = {}
        for irrep in irreps:
            value = irrep.character_inner_product(self, elements)
            rounded = int(round(value.real))
            if abs(value.imag) > atol or rounded < 0 or not np.isclose(
                value.real, rounded, atol=atol, rtol=0
            ):
                raise ValueError(
                    f"character inner product for {irrep.name!r} is not a "
                    f"non-negative integer: {value}"
                )
            if rounded:
                multiplicities[irrep.name] = rounded
        return multiplicities


class CyclicGroup:
    """The cyclic group ``Z_n`` with elements ``r0`` through ``r(n-1)``."""

    def __init__(self, n: int):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        self.n = n
        self.elements = [f"r{index}" for index in range(n)]

    def _index(self, element: object) -> int:
        if not isinstance(element, str) or element not in self.elements:
            raise ValueError(f"{element!r} is not a valid element of Z_{self.n}")
        return int(element[1:])

    def multiply(self, first: Element, second: Element) -> str:
        """Return ``r^a r^b = r^(a+b mod n)``."""
        return f"r{(self._index(first) + self._index(second)) % self.n}"

    def inverse(self, element: object) -> str:
        """Return the group inverse."""
        return f"r{(-self._index(element)) % self.n}"

    def identity(self) -> str:
        return "r0"

    def regular_representation(self) -> Representation:
        """Return the ``n``-dimensional left regular representation."""
        matrices: Dict[Element, NDArray] = {}
        for element in self.elements:
            matrix = np.zeros((self.n, self.n), dtype=complex)
            for column, basis_element in enumerate(self.elements):
                row = self.elements.index(self.multiply(element, basis_element))
                matrix[row, column] = 1.0
            matrices[element] = matrix
        return Representation(f"Regular_Z{self.n}", self.n, matrices)

    def irreducible_representations(self) -> List[Representation]:
        """Return all ``n`` one-dimensional irreducible representations."""
        omega = np.exp(2j * np.pi / self.n)
        irreps: List[Representation] = []
        for mode in range(self.n):
            matrices = {
                f"r{power}": np.array([[omega ** (mode * power)]], dtype=complex)
                for power in range(self.n)
            }
            irreps.append(Representation(f"χ{mode}", 1, matrices))
        return irreps


class SymmetricGroup:
    """The symmetric group ``S_n`` using standard permutation composition."""

    def __init__(self, n: int):
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError("n must be an integer at least 2")
        if n > 8:
            raise ValueError("n must not exceed 8; this implementation materializes n! elements")
        self.n = n
        self.elements = list(permutations(range(n)))

    def _permutation(self, value: object) -> Tuple[int, ...]:
        if (
            not isinstance(value, tuple)
            or len(value) != self.n
            or any(type(item) is not int for item in value)
            or set(value) != set(range(self.n))
        ):
            raise ValueError(f"{value!r} is not a valid permutation in S_{self.n}")
        return value

    def multiply(self, first: Element, second: Element) -> Tuple[int, ...]:
        """Return ``first ∘ second``: apply ``second``, then ``first``."""
        left = self._permutation(first)
        right = self._permutation(second)
        return tuple(left[right[index]] for index in range(self.n))

    def inverse(self, value: object) -> Tuple[int, ...]:
        """Return a permutation inverse."""
        permutation_value = self._permutation(value)
        inverse_value = [0] * self.n
        for source, target in enumerate(permutation_value):
            inverse_value[target] = source
        return tuple(inverse_value)

    def identity(self) -> Tuple[int, ...]:
        return tuple(range(self.n))

    def sign(self, value: object) -> int:
        """Return ``+1`` for even and ``-1`` for odd permutations."""
        permutation_value = self._permutation(value)
        inversions = sum(
            permutation_value[i] > permutation_value[j]
            for i in range(self.n)
            for j in range(i + 1, self.n)
        )
        return 1 if inversions % 2 == 0 else -1

    def trivial_representation(self) -> Representation:
        matrices = {element: np.ones((1, 1)) for element in self.elements}
        return Representation("Trivial", 1, matrices)

    def sign_representation(self) -> Representation:
        matrices = {
            element: np.array([[self.sign(element)]], dtype=complex)
            for element in self.elements
        }
        return Representation("Sign", 1, matrices)

    def permutation_representation(self) -> Representation:
        """Return the natural ``n``-dimensional permutation representation."""
        matrices: Dict[Element, NDArray] = {}
        for element in self.elements:
            matrix = np.zeros((self.n, self.n), dtype=complex)
            for column in range(self.n):
                matrix[element[column], column] = 1.0
            matrices[element] = matrix
        return Representation("Permutation", self.n, matrices)

    def standard_representation(self) -> Representation:
        """Return the true ``(n-1)``-dimensional sum-zero representation."""
        basis = np.zeros((self.n, self.n - 1), dtype=float)
        for column in range(self.n - 1):
            count = column + 1
            scale = np.sqrt(count * (count + 1))
            basis[:count, column] = 1.0 / scale
            basis[count, column] = -count / scale

        permutation_rep = self.permutation_representation()
        matrices = {
            element: basis.T @ permutation_rep.matrices[element] @ basis
            for element in self.elements
        }
        return Representation("Standard", self.n - 1, matrices)

    def character_table(self) -> Dict[str, List[complex]]:
        """Return all irreducible character rows for ``S_3``."""
        if self.n != 3:
            raise NotImplementedError("Character table is implemented only for S_3")
        irreps = (
            self.trivial_representation(),
            self.sign_representation(),
            self.standard_representation(),
        )
        return {
            irrep.name: [irrep.character(element) for element in self.elements]
            for irrep in irreps
        }


class AgentSymmetryAnalyzer:
    """Analyze deterministic cyclic coordination-mode representations."""

    def __init__(self, group: object):
        self.group = group
        self._observations: Dict[Element, str] = {}

    @property
    def observations(self) -> List[Dict[str, object]]:
        """Return observations in group order when available."""
        elements = getattr(self.group, "elements", tuple(self._observations))
        return [
            {"element": element, "description": self._observations[element]}
            for element in elements
            if element in self._observations
        ]

    def observe_coordination(self, group_element: Element, description: str) -> None:
        """Record or replace a label for one known group element."""
        elements = getattr(self.group, "elements", None)
        if elements is None or group_element not in elements:
            raise ValueError(f"{group_element!r} is not a known group element")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        self._observations[group_element] = description

    def build_observed_representation(self, dim: int) -> Representation:
        """Build a deterministic synthetic cyclic representation from all labels.

        The complete ordered label set selects ``dim`` cyclic irreducible modes by
        SHA-256. This is a reproducible test adapter, not learned semantic inference.
        """
        if not isinstance(self.group, CyclicGroup):
            raise NotImplementedError("observed representations support cyclic groups only")
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if set(self._observations) != set(self.group.elements):
            raise ValueError("one observation is required for every group element")

        payload = json.dumps(
            [(element, self._observations[element]) for element in self.group.elements],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        modes = [
            int.from_bytes(
                hashlib.sha256(payload + coordinate.to_bytes(8, "big")).digest(),
                "big",
            )
            % self.group.n
            for coordinate in range(dim)
        ]
        omega = np.exp(2j * np.pi / self.group.n)
        matrices = {
            f"r{power}": np.diag([omega ** (mode * power) for mode in modes])
            for power in range(self.group.n)
        }
        return Representation("ObservedSynthetic", dim, matrices)

    def decompose_coordination(self, representation: Representation) -> Dict[str, int]:
        """Decompose a cyclic representation into irreducible modes."""
        if not isinstance(self.group, CyclicGroup):
            raise NotImplementedError("decomposition supports cyclic groups only")
        return representation.decompose(
            self.group.irreducible_representations(), self.group.elements
        )

    def detect_symmetry_breaking(
        self, representation: Representation, atol: float = 1e-9
    ) -> List[Element]:
        """Return elements whose represented action is non-identity.

        This diagnoses non-trivial action. Physical symmetry breaking additionally
        requires a state or observable, which this compact API does not model.
        """
        atol = _tolerance(atol)
        identity = np.eye(representation.dim, dtype=complex)
        return [
            element
            for element, matrix in representation.matrices.items()
            if not np.allclose(matrix, identity, atol=atol, rtol=0)
        ]

    def compute_selection_rules(
        self,
        initial_irrep: str,
        final_irrep: str,
        operator_irrep: str,
        irreps: Sequence[Representation],
        atol: float = 1e-8,
    ) -> bool:
        """Return whether ``initial ⊗ operator`` contains ``final``."""
        atol = _tolerance(atol)
        irrep_dict = {irrep.name: irrep for irrep in irreps}
        for name in (initial_irrep, final_irrep, operator_irrep):
            if name not in irrep_dict:
                raise ValueError(f"Unknown irrep: {name}")
        elements = _group_elements(getattr(self.group, "elements", ()))
        value = sum(
            irrep_dict[initial_irrep].character(element)
            * irrep_dict[operator_irrep].character(element)
            * np.conj(irrep_dict[final_irrep].character(element))
            for element in elements
        ) / len(elements)
        rounded = int(round(value.real))
        return bool(
            abs(value.imag) <= atol
            and rounded > 0
            and np.isclose(value.real, rounded, atol=atol, rtol=0)
        )


def demo() -> None:
    """Run a deterministic, locally verifiable example."""
    print("=" * 70)
    print("AGENT REPRESENTATION THEORY DEMO")
    print("=" * 70)

    cyclic = CyclicGroup(4)
    irreps = cyclic.irreducible_representations()
    regular = cyclic.regular_representation()
    valid = regular.is_representation(
        cyclic.elements,
        cyclic.multiply,
        cyclic.identity(),
    )
    print(f"Z_4 regular representation valid: {valid}")
    print(f"Z_4 regular decomposition: {regular.decompose(irreps, cyclic.elements)}")

    symmetric = SymmetricGroup(3)
    standard = symmetric.standard_representation()
    print(f"S_3 standard dimension: {standard.dim}")
    print(f"S_3 standard irreducible: {standard.is_irreducible(symmetric.elements)}")

    analyzer = AgentSymmetryAnalyzer(cyclic)
    for element, description in (
        ("r0", "synchronized"),
        ("r1", "quarter turn"),
        ("r2", "opposed"),
        ("r3", "three-quarter turn"),
    ):
        analyzer.observe_coordination(element, description)
    observed = analyzer.build_observed_representation(3)
    valid = observed.is_representation(cyclic.elements, cyclic.multiply, cyclic.identity())
    print(f"Observed representation valid: {valid}")
    print(f"Observed decomposition: {analyzer.decompose_coordination(observed)}")
    print(
        "Selection χ0 -> χ1 through χ1: "
        f"{analyzer.compute_selection_rules('χ0', 'χ1', 'χ1', irreps)}"
    )
    print("DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    demo()
