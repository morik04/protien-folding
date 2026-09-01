# Hybrid Protein Folding Pipeline

## Overview

This project implements a two-stage protein structure prediction pipeline.
The primary predictor is **AlphaFold 2**, a deep-learning model that
produces highly accurate structures for most protein sequences. When
AlphaFold's own confidence metric (pLDDT) indicates that a prediction is
unreliable, the pipeline falls back to a **variational quantum eigensolver
(VQE)** that searches for the minimum-energy conformation on a tetrahedral
lattice.

```
Input (amino-acid sequence)
         │
         ▼
  ┌──────────────┐
  │  AlphaFold 2  │  predict structure + per-residue pLDDT
  └──────┬────────┘
         │
    Confidence gate
    pLDDT ≥ threshold? ──yes──▶ Return AlphaFold prediction (PDB)
         │
        no
         ▼
  ┌──────────────────┐
  │  Quantum VQE      │  tetrahedral-lattice folding (Robert et al. 2021)
  │  (lattice model)  │
  └──────┬────────────┘
         │
         ▼
  Decode bitstring → 3-D coordinates → PDB
```

## Quick Start

```python
from protein_folding import FoldingPipeline

pipe = FoldingPipeline(confidence_threshold=70.0)

# With a precomputed AlphaFold PDB:
result = pipe.run("APRLRFY", precomputed_pdb="af2/ranked_0.pdb")

# Or let the pipeline call AlphaFold via Docker:
result = pipe.run("APRLRFY")

print(result.method)       # "alphafold" or "quantum_vqe"
print(result.pdb_string)   # PDB-formatted 3-D coordinates
```

## Pipeline Stages

### Stage 1 — AlphaFold 2 Prediction

The `AlphaFoldPredictor` class wraps AlphaFold 2 and supports three modes
of operation:

| Mode | When to use |
|------|-------------|
| **Docker** (default) | AF2 is installed as a Docker image |
| **Local** | AF2 is installed natively; pass `af2_dir` |
| **Precomputed** | You already have a PDB + pLDDT scores on disk |

The predictor returns an `AlphaFoldResult` containing:

- `pdb_path` — path to the best-ranked PDB file
- `plddt_scores` — NumPy array of per-residue pLDDT values (0–100)
- `positions` — Cα coordinates extracted from the PDB

### Stage 2 — Confidence Gate

The confidence gate examines the pLDDT scores and decides whether the
AlphaFold prediction is trustworthy.

**Decision logic:**

1. Compute the **mean pLDDT** across all residues.
2. Scan for **contiguous low-confidence regions** — stretches of ≥ 5
   residues where pLDDT < `region_threshold` (default 50).
3. **Accept** if mean pLDDT ≥ `accept_threshold` (default 70) **and** no
   low-confidence regions are found.
4. Otherwise **reject** and proceed to quantum VQE.

The thresholds are configurable:

```python
pipe = FoldingPipeline(
    confidence_threshold=80.0,   # stricter mean pLDDT cutoff
    region_threshold=60.0,       # flag regions below 60
)
```

The `ConfidenceVerdict` returned by the gate includes:

- `accept` — boolean
- `mean_plddt`, `min_plddt` — summary statistics
- `low_confidence_regions` — list of `(start, end)` residue spans
- `reason` — human-readable explanation

### Stage 3 — Quantum VQE Fallback

When AlphaFold confidence is insufficient, the pipeline folds the protein
on a tetrahedral lattice using a variational quantum circuit.

This stage has four sub-steps:

1. **Build the ansatz** — a parameterised quantum circuit
   (see [quantum_folding_circuit.md](quantum_folding_circuit.md))
2. **Build the Hamiltonian** — the energy operator H_fold
   (see [hamiltonian.md](hamiltonian.md))
3. **Run VQE** — a classical optimiser (COBYLA) tunes the circuit
   parameters to minimise ⟨ψ(θ)|H|ψ(θ)⟩
4. **Decode** — the most-probable measurement bitstring is converted to
   3-D lattice coordinates and exported as a PDB file

Key VQE parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vqe_depth` | 2 | Ansatz circuit depth (number of entangle→rotate blocks) |
| `vqe_max_iter` | 200 | Maximum optimiser iterations |
| `penalty` | 10.0 | Overlap-penalty weight λ in H_fold |
| `optimizer` | `"COBYLA"` | Scipy minimiser method |

## Pipeline Result

`FoldingPipeline.run()` returns a `PipelineResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `sequence` | `str` | Input amino-acid sequence |
| `method` | `str` | `"alphafold"` or `"quantum_vqe"` |
| `alphafold_result` | `AlphaFoldResult` | AF2 prediction (always present) |
| `confidence` | `ConfidenceVerdict` | Confidence evaluation |
| `vqe_result` | `VQEResult \| None` | VQE output (only if quantum path taken) |
| `structure` | `FoldedStructure \| None` | Decoded lattice structure (VQE only) |
| `pdb_string` | `str` | PDB-formatted coordinates |

## Project Structure

```
src/protein_folding/
├── __init__.py              # public API exports
├── pipeline.py              # top-level orchestrator (this document)
├── alphafold_predictor.py   # AlphaFold 2 wrapper
├── confidence.py            # pLDDT confidence gate
├── lattice.py               # tetrahedral geometry and coordinate reconstruction
├── ansatz.py                # parameterised VQE circuit
├── hamiltonian.py           # H_fold energy operator
├── vqe_runner.py            # statevector VQE with scipy optimiser
└── decoder.py               # bitstring → 3-D structure → PDB export
```

## Dependencies

Core dependencies (installed via `pip install .`):

- `qiskit >= 1.0` — quantum circuit construction and simulation
- `qiskit-aer >= 0.13` — high-performance statevector simulator
- `numpy`, `scipy` — numerics and optimisation
- `biopython` — sequence I/O
- `matplotlib` — visualisation

Optional (for live AlphaFold predictions):

```bash
pip install ".[alphafold]"   # adds alphafold, jax[cuda], openmm
```
