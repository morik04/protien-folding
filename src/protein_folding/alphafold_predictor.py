"""AlphaFold 2 structure prediction wrapper.

Supports two modes:
  * **live** — runs AF2 via Docker or a local install (``predict``).
  * **precomputed** — loads an existing PDB produced by AF2 / ColabFold
    (``predict_from_precomputed``).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ───────────────────────────────────────────────────────────────────────────── #
#  Data class                                                                   #
# ───────────────────────────────────────────────────────────────────────────── #
@dataclass
class AlphaFoldResult:
    """Container for an AlphaFold 2 prediction."""

    sequence: str
    pdb_path: Path | None
    plddt_scores: np.ndarray          # per-residue, 0-100
    positions: np.ndarray | None = None  # (N, 3) CA coordinates in Å

    @property
    def mean_plddt(self) -> float:
        return float(np.mean(self.plddt_scores))

    @property
    def min_plddt(self) -> float:
        return float(np.min(self.plddt_scores))


# ───────────────────────────────────────────────────────────────────────────── #
#  Predictor                                                                    #
# ───────────────────────────────────────────────────────────────────────────── #
class AlphaFoldPredictor:
    """Thin wrapper around AlphaFold 2 / ColabFold."""

    def __init__(
        self,
        af2_dir: str | Path | None = None,
        output_dir: str | Path = "af2_output",
        use_docker: bool = True,
    ):
        self.af2_dir = Path(af2_dir) if af2_dir else None
        self.output_dir = Path(output_dir)
        self.use_docker = use_docker

    # ── live prediction ────────────────────────────────────────────────
    def predict(self, sequence: str,
                fasta_path: str | Path | None = None) -> AlphaFoldResult:
        """Run AlphaFold 2 on *sequence* and return the best-ranked result."""
        if fasta_path is None:
            fasta_path = self._write_fasta(sequence)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_alphafold(Path(fasta_path))
        return self._parse_results(sequence)

    # ── precomputed ────────────────────────────────────────────────────
    def predict_from_precomputed(
        self,
        sequence: str,
        pdb_path: str | Path,
        scores_path: str | Path | None = None,
    ) -> AlphaFoldResult:
        """Load a previously-computed AF2 result from disk."""
        pdb_path = Path(pdb_path)
        positions = _parse_ca_coords(pdb_path)

        if scores_path and Path(scores_path).exists():
            data = json.loads(Path(scores_path).read_text())
            plddt = np.asarray(data["plddt"], dtype=float)
        else:
            plddt = _parse_bfactor_plddt(pdb_path)

        return AlphaFoldResult(
            sequence=sequence,
            pdb_path=pdb_path,
            plddt_scores=plddt,
            positions=positions,
        )

    # ── internals ──────────────────────────────────────────────────────
    def _write_fasta(self, sequence: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "input.fasta"
        path.write_text(f">query\n{sequence}\n")
        return path

    def _run_alphafold(self, fasta_path: Path) -> None:
        if self.use_docker:
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{fasta_path.parent}:/input",
                "-v", f"{self.output_dir}:/output",
                "alphafold2",
                f"--fasta_paths=/input/{fasta_path.name}",
                "--output_dir=/output",
                "--model_preset=monomer",
                "--max_template_date=2022-01-01",
            ]
        else:
            if self.af2_dir is None:
                raise ValueError("af2_dir is required when use_docker=False")
            cmd = [
                "python",
                str(self.af2_dir / "run_alphafold.py"),
                f"--fasta_paths={fasta_path}",
                f"--output_dir={self.output_dir}",
                "--model_preset=monomer",
            ]
        subprocess.run(cmd, check=True)

    def _parse_results(self, sequence: str) -> AlphaFoldResult:
        pdb_files = sorted(self.output_dir.glob("**/ranked_*.pdb"))
        if not pdb_files:
            pdb_files = sorted(self.output_dir.glob("**/*.pdb"))
        if not pdb_files:
            raise FileNotFoundError(
                f"No PDB files found in {self.output_dir}")

        best_pdb = pdb_files[0]
        return AlphaFoldResult(
            sequence=sequence,
            pdb_path=best_pdb,
            plddt_scores=_parse_bfactor_plddt(best_pdb),
            positions=_parse_ca_coords(best_pdb),
        )


# ───────────────────────────────────────────────────────────────────────────── #
#  PDB helpers                                                                  #
# ───────────────────────────────────────────────────────────────────────────── #
def _parse_ca_coords(pdb_path: Path) -> np.ndarray:
    """Extract CA-atom coordinates from a PDB file."""
    coords: list[list[float]] = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coords.append([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
    return np.array(coords)


def _parse_bfactor_plddt(pdb_path: Path) -> np.ndarray:
    """Read per-residue pLDDT from the B-factor column (AF2 convention)."""
    scores: list[float] = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            scores.append(float(line[60:66]))
    return np.array(scores)
