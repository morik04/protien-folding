# Quantum Protein Folding Simulation Circuit

**Explanation of `quantum_folding_circuit.ipynb`**

## Overview

This document explains the variational quantum circuit template implemented in
`quantum_folding_circuit.ipynb`. The notebook constructs a parameterised
ansatz for solving the protein folding problem on a quantum computer using the
Variational Quantum Eigensolver (VQE) framework, following the lattice model of
Robert et al. (2021).

## Tetrahedral Lattice Encoding

Proteins are folded on a three-dimensional tetrahedral (diamond) lattice. Each
bond between consecutive amino acids is represented by one of four tetrahedral
bond vectors:

$$
\vec{v}_0 = \frac{1}{\sqrt{3}}(+1,\,+1,\,+1), \qquad
\vec{v}_1 = \frac{1}{\sqrt{3}}(+1,\,-1,\,-1)
$$

$$
\vec{v}_2 = \frac{1}{\sqrt{3}}(-1,\,+1,\,-1), \qquad
\vec{v}_3 = \frac{1}{\sqrt{3}}(-1,\,-1,\,+1)
$$

The angle between any two distinct bond vectors is the tetrahedral angle:

$$
\theta_{\mathrm{tet}}
  = \arccos\!\left(-\frac{1}{3}\right)
  \approx 109.47°
$$

The lattice is bipartite (type-A / type-B sublattices), so consecutive residues
alternate between stepping by $+\vec{v}_i$ and $-\vec{v}_i$, preserving the
tetrahedral bond angle at every vertex.

## Dense Qubit Encoding

Under the *dense* encoding scheme, each turn direction is represented by
two qubits $(q_1, q_2)$. The chosen lattice axis is identified by four
**turn indicator functions**, exactly one of which evaluates to 1 for any
given qubit state:

$$
\begin{aligned}
f_0 &= (1-q_1)(1-q_2)  &\qquad |00\rangle &\;\mapsto\; \text{Axis } 0 \\
f_1 &= q_2\,(q_2-q_1)   &\qquad |01\rangle &\;\mapsto\; \text{Axis } 1 \\
f_2 &= q_1\,(q_1-q_2)   &\qquad |10\rangle &\;\mapsto\; \text{Axis } 2 \\
f_3 &= q_1 \cdot q_2     &\qquad |11\rangle &\;\mapsto\; \text{Axis } 3
\end{aligned}
$$

An alternative *sparse* encoding uses 4 qubits per turn in a one-hot
representation.

## Qubit Budget

For a protein chain of $N$ amino acids there are $N-1$ bonds and therefore
$N-2$ turns. The first two turns are **pinned** to eliminate global
rotational and translational symmetry, leaving $N-3$ active turns.

$$
n = N_{\mathrm{cf}} + N_{\mathrm{in}}
$$

where the number of **configuration qubits** under dense encoding is

$$
N_{\mathrm{cf}} = 2(N-3)
$$

and $N_{\mathrm{in}}$ is the number of auxiliary **interaction qubits**
used to track physical contacts between non-adjacent residues.

**Example (7-mer peptide):**

$$
N_{\mathrm{cf}} = 2(7-3) = 8, \qquad
N_{\mathrm{in}} = 1, \qquad
n = 9
$$

## Variational Ansatz Circuit $U(\vec\theta)$

The parameterised circuit is composed of three stages.

### Step A — Initialisation Layer

A Hadamard gate creates an equal superposition on every qubit, followed by a
layer of parameterised $R_Y$ rotations:

$$
U_{\mathrm{init}}
  = \prod_{i=0}^{n-1} R_Y(\theta_i)\; H_i
$$

### Step B — Entangling and Rotation Blocks

The following block is repeated $m$ times (where $m = \texttt{depth}$). For
each layer $d = 1, \dots, m$:

$$
U_d
  = \left[\,\prod_{i=0}^{n-1} R_Y(\theta_{d,i})\right]
    \cdot U_{\mathrm{ent}}
$$

**Closed-loop entangling layer.**
In the default `closed_loop` mode the entangling unitary is a linear
CNOT chain with a loopback from the last qubit to the first:

$$
U_{\mathrm{ent}}
  = \mathrm{CNOT}(n{-}1,\,0)
    \;\cdot\; \prod_{i=0}^{n-2} \mathrm{CNOT}(i,\,i{+}1)
$$

This topology is well-suited to NISQ hardware with nearest-neighbour
connectivity.

**All-to-all entangling layer.**
The alternative `all_to_all` mode applies a CNOT between every pair of
qubits:

$$
U_{\mathrm{ent}}
  = \prod_{i=0}^{n-1}\;\prod_{j=i+1}^{n-1}
    \mathrm{CNOT}(i,\,j)
$$

### Step C — Measurement

All $n$ qubits are measured in the computational ($Z$) basis.

## Parameter Count

The total number of variational parameters is

$$
|\vec\theta| = n\,(m+1)
$$

For the 7-mer example ($n=9$, $m=2$):

$$
|\vec\theta| = 9 \times 3 = 27
$$

## Gate Count Summary (7-mer Example)

| Gate | Count |
|------|-------|
| $H$ | 9 |
| $R_Y(\theta)$ | 27 |
| CNOT | 18 |
| Measure | 9 |
| **Circuit depth** | **23** |

*Gate counts for the 7-amino-acid ansatz with depth $m=2$
and closed-loop entanglement.*

## Classical Post-Processing

Given a measured bitstring $\mathbf{b} \in \{0,1\}^n$, the turn direction for
turn $t$ is decoded by extracting $(q_1, q_2)$ from the configuration register
and evaluating the indicator functions $f_0, \dots, f_3$. Exactly one indicator
equals 1, identifying the chosen lattice axis $\alpha(t) \in \{0,1,2,3\}$.

The three-dimensional position of residue $k$ is then reconstructed as

$$
\vec{r}_k
  = \vec{r}_0
    + \sum_{t=1}^{k-1} (-1)^{t+1}\, \vec{v}_{\,\alpha(t)}
$$

where the alternating sign $(-1)^{t+1}$ enforces the bipartite sublattice
structure of the tetrahedral lattice.

## VQE Context

The ansatz $U(\vec\theta)$ produces candidate folding conformations as quantum
states. A classical optimiser tunes $\vec\theta$ to minimise the expectation
value of the protein's energy Hamiltonian:

$$
\vec\theta^{*}
  = \arg\min_{\vec\theta}\;
    \langle 0| U^{\dagger}(\vec\theta)\, H_{\mathrm{fold}}\, U(\vec\theta) |0\rangle
$$

The Hamiltonian $H_{\mathrm{fold}}$ is constructed in `hamiltonian.py` and
includes nearest-neighbour contact energies and a penalty term that enforces
the self-avoidance constraint on the lattice.  See
[hamiltonian.md](hamiltonian.md) for the full specification.

## Related Documentation

- [pipeline.md](pipeline.md) — Hybrid AlphaFold 2 / quantum VQE pipeline
  overview
- [hamiltonian.md](hamiltonian.md) — Energy Hamiltonian construction
- [vqe_runner.md](vqe_runner.md) — VQE optimisation loop
- [decoder.md](decoder.md) — Bitstring decoding and PDB export
- [confidence_gate.md](confidence_gate.md) — AlphaFold pLDDT confidence
  evaluation
