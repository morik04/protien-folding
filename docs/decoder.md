# Structure Decoder and PDB Export

**Explanation of `decoder.py`**

## Overview

After VQE converges, the most-probable measurement bitstring encodes a
protein conformation as a sequence of turn directions on the tetrahedral
lattice.  The decoder module converts this bitstring into 3-D Cartesian
coordinates and exports the result as a PDB file.

## Decoding Pipeline

```
Bitstring (e.g. "11011000")
       │
       ▼
 Extract qubit pairs → turn axes [0, 1, 0, 1, 2, 3]
       │
       ▼
 Prepend pinned turns → [0, 1, 0, 1, 2, 3]
       │
       ▼
 Walk tetrahedral lattice → (N, 3) positions
       │
       ▼
 Detect contacts and overlaps
       │
       ▼
 FoldedStructure  →  PDB string
```

### Step 1 — Turn Extraction (Dense Encoding)

Each active turn is encoded by 2 qubits (q₁, q₂). The axis index is
simply:

$$
\text{axis} = 2 q_1 + q_2
$$

The first two turns are pinned to axes 0 and 1 (removing rotational and
translational symmetry), so they are not read from the bitstring.

### Step 2 — Lattice Walk

Given a turn sequence [α₁, α₂, …, α_{N-1}], the position of residue k is:

$$
\vec{r}_k = \vec{r}_0 + \sum_{t=1}^{k-1} (-1)^{t+1}\, \vec{v}_{\alpha_t}
$$

The alternating sign enforces the bipartite sublattice structure of the
tetrahedral lattice (consecutive residues alternate between type-A and
type-B sites).

### Step 3 — Contact and Overlap Detection

- **Contacts:** Non-adjacent residue pairs (|i − j| ≥ 2) that are nearest
  neighbours on the lattice (distance ≈ 1 bond length).
- **Overlaps:** Residue pairs occupying the same lattice site (distance ≈ 0).
  A valid conformation has zero overlaps.

## FoldedStructure

The `decode_structure` function returns a `FoldedStructure` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `sequence` | `str` | Amino-acid sequence |
| `positions` | `np.ndarray` | (N, 3) lattice coordinates |
| `turns` | `list[int]` | Full turn sequence (including pinned) |
| `contacts` | `list[tuple]` | Non-adjacent nearest-neighbour pairs |
| `has_overlaps` | `bool` | Whether any two residues share a site |
| `energy` | `float \| None` | VQE optimal energy (if available) |

## PDB Export

The `to_pdb` function converts lattice coordinates to PDB format:

- Lattice positions are scaled by **3.8 Å** (the typical Cα–Cα distance
  in real proteins) to produce physically meaningful coordinates.
- Each residue is placed as a single **CA** (alpha-carbon) atom.
- One-letter amino-acid codes are mapped to standard three-letter PDB
  residue names (e.g. A → ALA, R → ARG).

```python
from protein_folding.decoder import decode_structure, to_pdb

structure = decode_structure("11011000", "APRLRFY")
pdb_text = to_pdb(structure, scale=3.8)

with open("folded.pdb", "w") as f:
    f.write(pdb_text)
```

## Example Output

```
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  PRO A   2       2.194   2.194   2.194  1.00  0.00           C
ATOM      3  CA  ARG A   3       0.000   4.388   4.388  1.00  0.00           C
ATOM      4  CA  LEU A   4       2.194   6.582   6.582  1.00  0.00           C
ATOM      5  CA  ARG A   5       0.000   8.776   8.776  1.00  0.00           C
ATOM      6  CA  PHE A   6      -2.194  10.970   6.582  1.00  0.00           C
ATOM      7  CA  TYR A   7       0.000  13.164   4.388  1.00  0.00           C
END
```
