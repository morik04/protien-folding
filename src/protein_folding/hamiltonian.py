"""Protein folding Hamiltonian on a tetrahedral lattice.

The Hamiltonian is *diagonal* in the computational basis — its eigenvalue
for a basis state |z⟩ equals the classical energy of the conformation
encoded by z.  We enumerate all 2^n basis states, compute the energy for
each, then convert to a ``SparsePauliOp`` via a fast Walsh–Hadamard
transform (O(n · 2^n)).

Energy model
------------
H_fold  =  H_contact  +  λ · H_overlap

*  H_contact awards energy ε(i, j) whenever non-adjacent residues i and j
   are nearest neighbours on the lattice.
*  H_overlap penalises conformations where two residues occupy the same
   lattice site (self-avoidance constraint).
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .lattice import PINNED_AXES, compute_positions

_MAX_QUBITS = 20  # brute-force limit (2^20 ≈ 1 M states)


def build_hamiltonian(
    num_amino_acids: int,
    contact_energies: np.ndarray | None = None,
    penalty: float = 10.0,
    num_interaction_qubits: int = 0,
) -> SparsePauliOp:
    """Build H_fold as a ``SparsePauliOp``.

    Parameters
    ----------
    num_amino_acids : int
        Protein length *N*.
    contact_energies : np.ndarray, optional
        (N, N) symmetric matrix of pairwise contact energies.
        Defaults to −1 for every non-adjacent pair (HP-model style).
    penalty : float
        Coefficient λ multiplying the overlap penalty.
    num_interaction_qubits : int
        Auxiliary qubits (included in circuit width but not in energy).

    Returns
    -------
    SparsePauliOp
    """
    N = num_amino_acids
    n_config = 2 * (N - 3)
    n_total = n_config + num_interaction_qubits

    if n_total > _MAX_QUBITS:
        raise ValueError(
            f"Brute-force Hamiltonian requires ≤ {_MAX_QUBITS} qubits, "
            f"got {n_total}.  Use a symbolic builder for larger proteins."
        )

    if contact_energies is None:
        contact_energies = -np.ones((N, N))
        np.fill_diagonal(contact_energies, 0.0)

    # --- compute classical energy for every basis state ---
    num_states = 1 << n_total
    energies = np.zeros(num_states)

    for z in range(num_states):
        bitstring = format(z, f"0{n_total}b")
        turns = _decode_all_turns(bitstring, N)
        positions = compute_positions(turns)

        e_contact = 0.0
        e_overlap = 0.0
        for i in range(N):
            for j in range(i + 2, N):
                d2 = float(np.sum((positions[j] - positions[i]) ** 2))
                if abs(d2 - 1.0) < 0.1:
                    e_contact += contact_energies[i, j]
                if d2 < 0.01:
                    e_overlap += 1.0

        energies[z] = e_contact + penalty * e_overlap

    return _diagonal_to_sparse_pauli(energies, n_total)


# ───────────────────────────────────────────────────────────────────────────── #
#  Internal helpers                                                             #
# ───────────────────────────────────────────────────────────────────────────── #
def _decode_all_turns(bitstring: str, num_amino_acids: int) -> list[int]:
    """Same logic as ``lattice.decode_turns`` (inlined to avoid circular deps
    when only the config qubits matter for energy)."""
    bits = bitstring[::-1]
    turns: list[int] = list(PINNED_AXES)
    for t in range(num_amino_acids - 3):
        q1 = int(bits[2 * t])
        q2 = int(bits[2 * t + 1])
        turns.append(2 * q1 + q2)
    return turns


def _diagonal_to_sparse_pauli(energies: np.ndarray, n: int) -> SparsePauliOp:
    """Convert a diagonal energy vector to a sum of Pauli-Z strings.

    Uses an in-place fast Walsh–Hadamard transform so the cost is
    O(n · 2^n) rather than the naïve O(4^n).
    """
    coeffs = energies.copy()

    # Fast Walsh–Hadamard butterfly
    h = 1
    while h < len(coeffs):
        for i in range(0, len(coeffs), 2 * h):
            for k in range(h):
                u = coeffs[i + k]
                v = coeffs[i + k + h]
                coeffs[i + k] = u + v
                coeffs[i + k + h] = u - v
        h <<= 1

    coeffs /= 1 << n

    # Collect non-zero Pauli terms
    labels: list[str] = []
    values: list[float] = []
    for mask in range(1 << n):
        if abs(coeffs[mask]) < 1e-12:
            continue
        label = ["I"] * n
        for j in range(n):
            if mask & (1 << j):
                label[n - 1 - j] = "Z"
        labels.append("".join(label))
        values.append(float(coeffs[mask]))

    return SparsePauliOp(labels, values).simplify()
