"""Confidence gate: decide whether to accept AlphaFold or fall back to VQE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfidenceVerdict:
    """Result of the confidence evaluation."""

    accept: bool
    mean_plddt: float
    min_plddt: float
    low_confidence_regions: list[tuple[int, int]]   # (start, end) spans
    reason: str


def evaluate_confidence(
    plddt_scores: np.ndarray,
    accept_threshold: float = 70.0,
    region_threshold: float = 50.0,
    min_region_length: int = 5,
) -> ConfidenceVerdict:
    """Evaluate AlphaFold pLDDT scores and decide accept/reject.

    Parameters
    ----------
    plddt_scores : np.ndarray
        Per-residue pLDDT values (0--100).
    accept_threshold : float
        Accept if mean pLDDT >= this AND no problematic low-confidence regions.
    region_threshold : float
        Residues below this pLDDT are flagged as low-confidence.
    min_region_length : int
        Only flag contiguous stretches of at least this many residues.

    Returns
    -------
    ConfidenceVerdict
    """
    mean = float(np.mean(plddt_scores))
    minimum = float(np.min(plddt_scores))
    regions = _find_contiguous_regions(plddt_scores < region_threshold,
                                      min_region_length)

    if mean >= accept_threshold and not regions:
        return ConfidenceVerdict(
            accept=True,
            mean_plddt=mean,
            min_plddt=minimum,
            low_confidence_regions=regions,
            reason=(f"Mean pLDDT {mean:.1f} >= {accept_threshold} "
                    f"with no low-confidence regions"),
        )

    reasons: list[str] = []
    if mean < accept_threshold:
        reasons.append(f"mean pLDDT {mean:.1f} < {accept_threshold}")
    if regions:
        reasons.append(
            f"{len(regions)} region(s) below pLDDT {region_threshold}"
        )

    return ConfidenceVerdict(
        accept=False,
        mean_plddt=mean,
        min_plddt=minimum,
        low_confidence_regions=regions,
        reason="Rejected: " + "; ".join(reasons),
    )


def _find_contiguous_regions(
    mask: np.ndarray, min_length: int,
) -> list[tuple[int, int]]:
    """Find contiguous *True* stretches in *mask* of at least *min_length*."""
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_length:
                regions.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_length:
        regions.append((start, len(mask)))
    return regions
