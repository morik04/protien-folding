# Protein Folding Hamiltonian

**Explanation of `hamiltonian.py`**

## Overview

The Hamiltonian H_fold assigns an energy to each candidate protein
conformation on the tetrahedral lattice. It is a *diagonal* operator in the
computational basis — its eigenvalue for a basis state |z⟩ equals the
classical energy of the conformation encoded by z.

$$
H_{\mathrm{fold}} = H_{\mathrm{contact}} + \lambda \, H_{\mathrm{overlap}}
$$

## Contact Energy (H_contact)

Whenever two non-adjacent residues i and j land on nearest-neighbour lattice
sites, they contribute a contact energy ε(i, j):

$$
H_{\mathrm{contact}}
  = \sum_{\substack{i < j \\ |i - j| \geq 2}}
    \varepsilon(i,\,j) \;\cdot\; \delta_{\mathrm{nn}}(i,\,j)
$$

where δ_nn(i, j) = 1 if residues i and j are lattice nearest neighbours
(Euclidean distance ≈ 1 bond length) and 0 otherwise.

**Default energy model (HP-like):**
If no explicit contact-energy matrix is provided, the code uses
ε(i, j) = −1 for all non-adjacent pairs. This is equivalent to the
hydrophobic–polar (HP) model where every contact lowers the energy.

**Custom energies:**
Pass an (N × N) symmetric NumPy matrix as `contact_energies` to use
residue-specific interaction strengths (e.g. Miyazawa–Jernigan potentials).

## Overlap Penalty (H_overlap)

The self-avoidance constraint forbids two residues from occupying the same
lattice site. Rather than restricting the Hilbert space, this constraint is
enforced as a soft penalty:

$$
H_{\mathrm{overlap}}
  = \lambda \sum_{\substack{i < j}}
    \delta_{\mathrm{overlap}}(i,\,j)
$$

where δ_overlap(i, j) = 1 if residues i and j share the same position and
λ (default 10.0) is large enough to make any overlapping conformation
energetically unfavourable.

## Construction Method

Because H_fold is diagonal, it can be fully specified by a vector of 2^n
real energies (one per computational-basis state). The module:

1. **Enumerates** all 2^n basis states.
2. For each state, **decodes** the bitstring into turn directions, computes
   3-D lattice positions, and evaluates E_contact + λ · E_overlap.
3. **Converts** the diagonal energy vector into a sum of Pauli-Z strings
   using a fast Walsh–Hadamard transform.

### Walsh–Hadamard Transform

A diagonal operator D = diag(d₀, d₁, …, d_{2^n−1}) can be written as

$$
D = \sum_{k=0}^{2^n - 1} c_k \bigotimes_{j=0}^{n-1} Z_j^{b_j(k)}
$$

where b_j(k) is the j-th bit of k and the coefficients c_k are obtained by
applying the (unnormalised) Walsh–Hadamard butterfly to the diagonal vector d:

$$
\hat{d} = W_n \, d, \qquad c_k = \frac{\hat{d}_k}{2^n}
$$

This runs in O(n · 2^n) time, which is efficient for the small qubit counts
used in lattice protein folding (typically ≤ 20 qubits).

## Qubit Limit

The brute-force enumeration limits the module to proteins that require
≤ 20 total qubits.  For a dense-encoded chain of N amino acids with no
interaction qubits this corresponds to:

$$
2(N - 3) \leq 20 \implies N \leq 13
$$

Larger proteins would require a symbolic Hamiltonian builder that constructs
the Pauli terms directly from the indicator functions, without enumerating
all basis states.

## API

```python
from protein_folding.hamiltonian import build_hamiltonian

H = build_hamiltonian(
    num_amino_acids=7,
    contact_energies=None,       # default HP model (ε = -1)
    penalty=10.0,                # overlap weight λ
    num_interaction_qubits=0,
)

print(type(H))   # qiskit.quantum_info.SparsePauliOp
print(len(H))    # number of Pauli terms
```

## Example (5-mer)

For a 5-amino-acid peptide under dense encoding:

- Configuration qubits: 2 × (5 − 3) = 4
- Total basis states: 2⁴ = 16
- Each state is decoded into 4 turns → 5 residue positions
- Contact and overlap energies are computed and assembled into a
  16-element diagonal vector
- Walsh–Hadamard transform produces ≤ 16 Pauli-Z terms
