"""Tetrahedral lattice geometry for the protein folding model.

Proteins are folded on a 3-D tetrahedral (diamond) lattice following
Robert et al. (2021).  Each bond between consecutive amino acids is
one of four unit-length tetrahedral vectors.  The lattice is bipartite,
so consecutive residues alternate between +v_i and -v_i steps.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
#  Tetrahedral bond vectors (unit length)                                      #
# --------------------------------------------------------------------------- #
BOND_VECTORS = np.array([
    [+1, +1, +1],   # axis 0
    [+1, -1, -1],   # axis 1
    [-1, +1, -1],   # axis 2
    [-1, -1, +1],   # axis 3
], dtype=float) / np.sqrt(3)

# First two turns are pinned to eliminate global rotation/translation symmetry
PINNED_AXES = [0, 1]


# --------------------------------------------------------------------------- #
#  Bitstring → turn sequence                                                   #
# --------------------------------------------------------------------------- #
def decode_turns(bitstring: str, num_amino_acids: int,
                 num_interaction_qubits: int = 0) -> list[int]:
    """Decode turn axes from a measurement bitstring (dense encoding).

    The configuration qubits occupy the lowest qubit indices.  Each pair
    ``(q_{2t}, q_{2t+1})`` encodes one active turn as ``axis = 2*q1 + q2``.

    Returns
    -------
    list[int]
        Full turn sequence of length *N-1* (including the two pinned turns).
    """
    num_active = num_amino_acids - 3
    # Qiskit bitstrings are MSB-first; reverse so index 0 → qubit 0
    bits = bitstring[::-1]
    turns: list[int] = list(PINNED_AXES)
    for t in range(num_active):
        q1 = int(bits[2 * t])
        q2 = int(bits[2 * t + 1])
        turns.append(2 * q1 + q2)
    return turns


# --------------------------------------------------------------------------- #
#  Turn sequence → 3-D coordinates                                             #
# --------------------------------------------------------------------------- #
def compute_positions(turns: list[int]) -> np.ndarray:
    """Compute residue positions from a turn sequence.

    Parameters
    ----------
    turns : list[int]
        *N-1* axis indices for a protein of *N* amino acids.

    Returns
    -------
    np.ndarray
        Shape ``(N, 3)`` array of Cartesian coordinates.
    """
    n_residues = len(turns) + 1
    positions = np.zeros((n_residues, 3))
    for step, axis in enumerate(turns):
        sign = 1.0 if step % 2 == 0 else -1.0
        positions[step + 1] = positions[step] + sign * BOND_VECTORS[axis]
    return positions


# --------------------------------------------------------------------------- #
#  Contact / overlap detection                                                 #
# --------------------------------------------------------------------------- #
def find_contacts(positions: np.ndarray) -> list[tuple[int, int]]:
    """Return non-adjacent residue pairs that are nearest neighbours."""
    n = len(positions)
    contacts: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 2, n):
            d2 = float(np.sum((positions[j] - positions[i]) ** 2))
            if abs(d2 - 1.0) < 0.1:          # bond-vector length = 1
                contacts.append((i, j))
    return contacts


def find_overlaps(positions: np.ndarray) -> list[tuple[int, int]]:
    """Return residue pairs that occupy the same lattice site."""
    n = len(positions)
    overlaps: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d2 = float(np.sum((positions[j] - positions[i]) ** 2))
            if d2 < 0.01:
                overlaps.append((i, j))
    return overlaps
