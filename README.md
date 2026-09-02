# Hybrid Protein Folding Pipeline

A two-stage protein structure prediction pipeline that uses **AlphaFold 2** as the primary predictor and falls back to a **variational quantum eigensolver (VQE)** on a tetrahedral lattice when AlphaFold's confidence is too low.

```
Input (amino-acid sequence)
         │
         ▼
  ┌──────────────┐
  │  AlphaFold 2  │  predict structure + per-residue pLDDT
  └──────┬────────┘
         │
    pLDDT ≥ threshold? ──yes──▶ Return AlphaFold prediction
         │
        no
         ▼
  ┌──────────────────┐
  │  Quantum VQE      │  tetrahedral-lattice folding
  │  (lattice model)  │  (Robert et al. 2021)
  └──────┬────────────┘
         │
         ▼
   Return lowest-energy conformation (PDB)
```

## Quick Start

```bash
pip install .
```

```python
from protein_folding import FoldingPipeline

pipe = FoldingPipeline(confidence_threshold=70.0)

# With a precomputed AlphaFold PDB
result = pipe.run("APRLRFY", precomputed_pdb="af2/ranked_0.pdb")

# Or let the pipeline call AlphaFold via Docker
result = pipe.run("APRLRFY")

print(result.method)       # "alphafold" or "quantum_vqe"
print(result.pdb_string)   # PDB-formatted 3-D coordinates
```

### Run just the quantum path

```python
from protein_folding import run_vqe, decode_structure, to_pdb

result = run_vqe(num_amino_acids=7, depth=2, max_iterations=200, seed=42)
structure = decode_structure(result.bitstring, "APRLRFY")

with open("folded.pdb", "w") as f:
    f.write(to_pdb(structure))
```

## How It Works

### Stage 1 — AlphaFold 2

Runs AlphaFold 2 (via Docker, local install, or precomputed PDB) and extracts per-residue pLDDT confidence scores.

### Stage 2 — Confidence Gate

Accepts the prediction if mean pLDDT >= threshold (default 70) and no contiguous low-confidence regions are found. Otherwise rejects and falls back to VQE.

### Stage 3 — Quantum VQE

Folds the protein on a 3-D tetrahedral lattice using a variational quantum circuit:

1. **Ansatz** — parameterised circuit with Hadamard + RY rotations + CNOT entangling layers
2. **Hamiltonian** — diagonal energy operator encoding contact energies and self-avoidance penalties
3. **Optimisation** — COBYLA minimises the energy expectation value via exact statevector simulation
4. **Decoding** — the lowest-energy bitstring is decoded into 3-D coordinates and exported as PDB

The quantum approach follows the tetrahedral lattice model of Robert et al. (2021), using dense qubit encoding (2 qubits per turn direction).

## Project Structure

```
src/protein_folding/
├── pipeline.py              # top-level orchestrator
├── alphafold_predictor.py   # AlphaFold 2 wrapper
├── confidence.py            # pLDDT confidence gate
├── lattice.py               # tetrahedral geometry
├── ansatz.py                # parameterised VQE circuit
├── hamiltonian.py           # energy Hamiltonian (H_fold)
├── vqe_runner.py            # statevector VQE optimiser
└── decoder.py               # bitstring → 3-D structure → PDB
```

## Dependencies

- `qiskit >= 1.0`
- `qiskit-aer >= 0.13`
- `numpy`, `scipy`
- `biopython`
- `matplotlib`

For live AlphaFold predictions:

```bash
pip install ".[alphafold]"
```

## Documentation

Detailed documentation for each module is in [docs/](docs/):

- [Pipeline overview](docs/pipeline.md)
- [Quantum folding circuit](docs/quantum_folding_circuit.md)
- [Hamiltonian construction](docs/hamiltonian.md)
- [VQE runner](docs/vqe_runner.md)
- [Structure decoder](docs/decoder.md)
- [Confidence gate](docs/confidence_gate.md)

## References

- Robert, A. et al. (2021). *Resource-efficient quantum algorithm for protein folding.* npj Quantum Information.
- Perdomo-Ortiz, A. et al. (2012). *Finding low-energy conformations of lattice protein models by quantum annealing.* Scientific Reports.
- Doga, H. et al. (2024). *Quantum computing for protein folding.* (preprint)
