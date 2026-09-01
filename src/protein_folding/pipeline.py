"""Top-level orchestrator: AlphaFold 2 → confidence gate → quantum VQE.

Usage
-----
::

    from protein_folding import FoldingPipeline

    pipe = FoldingPipeline(confidence_threshold=70.0)
    result = pipe.run("APRLRFY", precomputed_pdb=Path("af2/ranked_0.pdb"))

    print(result.method)       # "alphafold" or "quantum_vqe"
    print(result.pdb_string)   # PDB-formatted coordinates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .alphafold_predictor import AlphaFoldPredictor, AlphaFoldResult
from .confidence import ConfidenceVerdict, evaluate_confidence
from .decoder import FoldedStructure, decode_structure, to_pdb
from .vqe_runner import VQEResult, run_vqe

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────── #
#  Pipeline result                                                              #
# ───────────────────────────────────────────────────────────────────────────── #
@dataclass
class PipelineResult:
    """Everything produced by a single pipeline invocation."""

    sequence: str
    method: str                            # "alphafold" | "quantum_vqe"
    alphafold_result: AlphaFoldResult | None
    confidence: ConfidenceVerdict | None
    vqe_result: VQEResult | None
    structure: FoldedStructure | None      # only set for VQE path
    pdb_string: str | None


# ───────────────────────────────────────────────────────────────────────────── #
#  Pipeline                                                                     #
# ───────────────────────────────────────────────────────────────────────────── #
class FoldingPipeline:
    """Hybrid protein-folding pipeline.

    Runs AlphaFold 2 first.  If its per-residue pLDDT confidence is too
    low the pipeline falls back to quantum VQE on a tetrahedral lattice.

    Parameters
    ----------
    af2_dir : Path, optional
        Root of a local AlphaFold 2 installation (ignored when using Docker).
    af2_output_dir : Path
        Directory for AlphaFold outputs.
    use_docker : bool
        Run AlphaFold inside a Docker container.
    confidence_threshold : float
        Accept AlphaFold if mean pLDDT ≥ this value (0–100).
    region_threshold : float
        Flag contiguous stretches where pLDDT < this value.
    vqe_depth : int
        Ansatz circuit depth for the quantum fallback.
    vqe_max_iter : int
        Max classical-optimiser iterations.
    penalty : float
        Overlap-penalty weight λ in the Hamiltonian.
    """

    def __init__(
        self,
        af2_dir: str | Path | None = None,
        af2_output_dir: str | Path = "af2_output",
        use_docker: bool = True,
        confidence_threshold: float = 70.0,
        region_threshold: float = 50.0,
        vqe_depth: int = 2,
        vqe_max_iter: int = 200,
        penalty: float = 10.0,
    ):
        self.predictor = AlphaFoldPredictor(
            af2_dir=af2_dir,
            output_dir=af2_output_dir,
            use_docker=use_docker,
        )
        self.confidence_threshold = confidence_threshold
        self.region_threshold = region_threshold
        self.vqe_depth = vqe_depth
        self.vqe_max_iter = vqe_max_iter
        self.penalty = penalty

    # ── public API ─────────────────────────────────────────────────────
    def run(
        self,
        sequence: str,
        contact_energies: np.ndarray | None = None,
        precomputed_pdb: str | Path | None = None,
        precomputed_scores: str | Path | None = None,
        vqe_seed: int | None = None,
    ) -> PipelineResult:
        """Execute the full folding pipeline.

        Parameters
        ----------
        sequence : str
            Amino-acid sequence (one-letter codes).
        contact_energies : np.ndarray, optional
            (N, N) pairwise energy matrix for VQE (defaults to HP model).
        precomputed_pdb : Path, optional
            Skip the live AF2 run and load this PDB instead.
        precomputed_scores : Path, optional
            JSON file with a ``"plddt"`` key (list of per-residue scores).
        vqe_seed : int, optional
            Random seed for VQE parameter initialisation.
        """
        # ── Stage 1: AlphaFold 2 ──────────────────────────────────────
        log.info("Stage 1 — AlphaFold 2 prediction")
        if precomputed_pdb is not None:
            af = self.predictor.predict_from_precomputed(
                sequence, precomputed_pdb, precomputed_scores,
            )
        else:
            af = self.predictor.predict(sequence)
        log.info("AlphaFold mean pLDDT: %.1f", af.mean_plddt)

        # ── Stage 2: confidence gate ──────────────────────────────────
        log.info("Stage 2 — confidence evaluation")
        verdict = evaluate_confidence(
            af.plddt_scores,
            accept_threshold=self.confidence_threshold,
            region_threshold=self.region_threshold,
        )
        log.info("Verdict: %s", verdict.reason)

        if verdict.accept:
            log.info("Accepting AlphaFold prediction.")
            pdb_text = (
                af.pdb_path.read_text() if af.pdb_path else None
            )
            return PipelineResult(
                sequence=sequence,
                method="alphafold",
                alphafold_result=af,
                confidence=verdict,
                vqe_result=None,
                structure=None,
                pdb_string=pdb_text,
            )

        # ── Stage 3: quantum VQE fallback ─────────────────────────────
        log.info("Stage 3 — quantum VQE fallback")
        vqe = run_vqe(
            num_amino_acids=len(sequence),
            contact_energies=contact_energies,
            penalty=self.penalty,
            depth=self.vqe_depth,
            max_iterations=self.vqe_max_iter,
            seed=vqe_seed,
        )

        structure = decode_structure(
            vqe.bitstring, sequence, energy=vqe.optimal_energy,
        )
        pdb_text = to_pdb(structure)
        log.info("VQE energy: %.4f", vqe.optimal_energy)

        return PipelineResult(
            sequence=sequence,
            method="quantum_vqe",
            alphafold_result=af,
            confidence=verdict,
            vqe_result=vqe,
            structure=structure,
            pdb_string=pdb_text,
        )
