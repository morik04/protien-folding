"""Decode VQE measurement results into 3-D protein structures.

Given a bitstring from the quantum circuit, this module reconstructs
the full lattice conformation and can export it as a PDB file with
physically-scaled coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lattice import compute_positions, decode_turns, find_contacts, find_overlaps


# ───────────────────────────────────────────────────────────────────────────── #
#  Data class                                                                   #
# ───────────────────────────────────────────────────────────────────────────── #
@dataclass
class FoldedStructure:
    """A decoded protein conformation on the tetrahedral lattice."""

    sequence: str
    positions: np.ndarray            # (N, 3) lattice coordinates
    turns: list[int]
    contacts: list[tuple[int, int]]
    has_overlaps: bool
    energy: float | None = None


# ───────────────────────────────────────────────────────────────────────────── #
#  Decoding                                                                     #
# ───────────────────────────────────────────────────────────────────────────── #
def decode_structure(
    bitstring: str,
    sequence: str,
    num_interaction_qubits: int = 0,
    energy: float | None = None,
) -> FoldedStructure:
    """Convert a measurement bitstring to a ``FoldedStructure``.

    Parameters
    ----------
    bitstring : str
        Measurement result from the VQE circuit (MSB-first, Qiskit convention).
    sequence : str
        Amino-acid sequence in one-letter codes.
    num_interaction_qubits : int
        Number of auxiliary qubits appended after the config register.
    energy : float, optional
        Objective value returned by the optimiser.
    """
    N = len(sequence)
    turns = decode_turns(bitstring, N, num_interaction_qubits)
    positions = compute_positions(turns)
    contacts = find_contacts(positions)
    overlaps = find_overlaps(positions)

    return FoldedStructure(
        sequence=sequence,
        positions=positions,
        turns=turns,
        contacts=contacts,
        has_overlaps=len(overlaps) > 0,
        energy=energy,
    )


# ───────────────────────────────────────────────────────────────────────────── #
#  PDB export                                                                   #
# ───────────────────────────────────────────────────────────────────────────── #
_AA_MAP: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def to_pdb(structure: FoldedStructure, scale: float = 3.8) -> str:
    """Render a ``FoldedStructure`` as a PDB-format string.

    Lattice positions are multiplied by *scale* (default 3.8 Å, the
    typical CA–CA distance) so the resulting PDB has physically
    meaningful coordinates.
    """
    lines: list[str] = []
    for i, (aa, pos) in enumerate(zip(structure.sequence, structure.positions)):
        x, y, z = pos * scale
        resname = _AA_MAP.get(aa.upper(), "UNK")
        #        1-6    7-11 13-16  18-20 22 23-26    31-38   39-46   47-54  55-60  61-66       77-78
        lines.append(
            f"ATOM  {i+1:5d}  CA  {resname:>3s} A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
    lines.append("END")
    return "\n".join(lines)
