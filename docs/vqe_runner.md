# VQE Runner

**Explanation of `vqe_runner.py`**

## Overview

The VQE (Variational Quantum Eigensolver) runner ties together the ansatz
circuit and the folding Hamiltonian to find the minimum-energy protein
conformation on the tetrahedral lattice.

$$
\vec\theta^{*}
  = \arg\min_{\vec\theta}\;
    \langle 0|\, U^{\dagger}(\vec\theta)\; H_{\mathrm{fold}}\; U(\vec\theta)\,|0\rangle
$$

## Method

1. **Build the ansatz** U(θ) — a parameterised quantum circuit (see
   [quantum_folding_circuit.md](quantum_folding_circuit.md)).
2. **Build the Hamiltonian** H_fold — the diagonal energy operator (see
   [hamiltonian.md](hamiltonian.md)).
3. **Initialise parameters** θ₀ uniformly at random in [−π, π].
4. **Optimise** using a classical optimiser (COBYLA by default) that
   repeatedly:
   - Binds θ into the circuit
   - Computes the exact statevector |ψ(θ)⟩
   - Evaluates ⟨ψ(θ)|H|ψ(θ)⟩
   - Updates θ to reduce the energy
5. **Extract the result** — the most-probable bitstring from the final
   statevector is the predicted conformation.

## Simulation Backend

The runner uses **exact statevector simulation** via Qiskit's `Statevector`
class.  This is ideal for the small qubit counts in lattice protein folding
(≤ 20 qubits) and avoids shot noise entirely.

For deployment on real quantum hardware, the `cost` function would be
replaced with a sampler- or estimator-based evaluation using
`qiskit-ibm-runtime`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_amino_acids` | — | Protein length N |
| `contact_energies` | `None` | (N, N) matrix; defaults to HP model (ε = −1) |
| `penalty` | 10.0 | Overlap-penalty weight λ |
| `num_interaction_qubits` | 0 | Auxiliary qubits for contact tracking |
| `depth` | 2 | Ansatz circuit depth |
| `max_iterations` | 200 | Maximum optimiser iterations |
| `optimizer` | `"COBYLA"` | Any `scipy.optimize.minimize` method |
| `seed` | `None` | Random seed for reproducibility |

## VQEResult

The function returns a `VQEResult` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `optimal_energy` | `float` | Final minimised energy value |
| `optimal_params` | `np.ndarray` | Optimised parameter vector θ* |
| `bitstring` | `str` | Most-probable measurement bitstring |
| `num_evaluations` | `int` | Total cost-function evaluations |

## Example

```python
from protein_folding.vqe_runner import run_vqe
from protein_folding.decoder import decode_structure, to_pdb

result = run_vqe(
    num_amino_acids=7,
    depth=2,
    max_iterations=200,
    seed=42,
)

print(f"Energy:    {result.optimal_energy:.4f}")
print(f"Bitstring: {result.bitstring}")

structure = decode_structure(result.bitstring, "APRLRFY")
print(to_pdb(structure))
```

## Convergence

For the 7-amino-acid example peptide (8 qubits, 24 parameters, depth 2),
VQE typically converges to near-zero energy within 200 COBYLA iterations.
The energy trajectory during a sample run:

| Iteration | Energy |
|-----------|--------|
| 50 | ~1.27 |
| 100 | ~0.28 |
| 150 | ~0.06 |
| 200 | ~0.02 |

Negative energies indicate conformations with favourable contacts.  An
energy near zero with no overlaps means the chain is self-avoiding but has
no non-adjacent contacts (extended conformation).
