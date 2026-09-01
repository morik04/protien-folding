# Confidence Gate

**Explanation of `confidence.py`**

## Overview

The confidence gate sits between AlphaFold 2 and the quantum VQE fallback.
It examines AlphaFold's per-residue pLDDT (predicted Local Distance
Difference Test) scores and decides whether the prediction is reliable
enough to accept, or whether the sequence should be forwarded to the
quantum solver.

## pLDDT Score Interpretation

AlphaFold 2 outputs a pLDDT score (0–100) for each residue.  These scores
estimate the local accuracy of the predicted structure:

| pLDDT Range | Interpretation |
|-------------|----------------|
| 90–100 | Very high confidence |
| 70–90 | High confidence |
| 50–70 | Low confidence — backbone likely correct, side chains uncertain |
| < 50 | Very low confidence — may be disordered or misfolded |

## Decision Logic

The gate applies two checks:

### 1. Mean pLDDT Threshold

$$
\text{accept if } \; \overline{\text{pLDDT}} \geq T_{\text{accept}}
$$

Default: T_accept = 70. If the average confidence across all residues falls
below this threshold, the prediction is rejected.

### 2. Low-Confidence Region Detection

Even when mean pLDDT is acceptable, a long contiguous stretch of
low-confidence residues can indicate a misfolded domain. The gate scans for
stretches of ≥ `min_region_length` (default 5) consecutive residues where
pLDDT < `region_threshold` (default 50).

**Accept** only if:
- Mean pLDDT ≥ T_accept, **and**
- No low-confidence regions are found.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `accept_threshold` | 70.0 | Minimum mean pLDDT to accept |
| `region_threshold` | 50.0 | pLDDT below which a residue is flagged |
| `min_region_length` | 5 | Minimum contiguous stretch to trigger rejection |

## ConfidenceVerdict

The `evaluate_confidence` function returns a `ConfidenceVerdict` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `accept` | `bool` | Whether to accept the AlphaFold prediction |
| `mean_plddt` | `float` | Mean pLDDT across all residues |
| `min_plddt` | `float` | Lowest single-residue pLDDT |
| `low_confidence_regions` | `list[tuple[int, int]]` | (start, end) spans of flagged regions |
| `reason` | `str` | Human-readable explanation |

## Examples

**Accepted prediction:**

```python
import numpy as np
from protein_folding.confidence import evaluate_confidence

scores = np.array([92, 88, 85, 91, 78, 82, 90])
verdict = evaluate_confidence(scores)
print(verdict.accept)  # True
print(verdict.reason)  # "Mean pLDDT 86.6 >= 70.0 with no low-confidence regions"
```

**Rejected — low mean pLDDT:**

```python
scores = np.array([85, 90, 60, 45, 30, 80, 95])
verdict = evaluate_confidence(scores)
print(verdict.accept)  # False
print(verdict.reason)  # "Rejected: mean pLDDT 69.3 < 70.0"
```

**Rejected — low-confidence region:**

```python
scores = np.array([90, 85, 40, 35, 30, 42, 38, 88, 92])
verdict = evaluate_confidence(scores)
print(verdict.accept)  # False
print(verdict.reason)  # "Rejected: 1 region(s) below pLDDT 50"
```

## Tuning the Thresholds

- **More conservative** (send more sequences to VQE): raise
  `accept_threshold` to 80 or 90.
- **More permissive** (trust AlphaFold more): lower `accept_threshold`
  to 60 and/or raise `region_threshold`.
- For **intrinsically disordered proteins**, AlphaFold will often produce
  low pLDDT, which will correctly route them to the quantum solver. Note
  that the lattice model may also struggle with disordered regions since
  they lack a single well-defined conformation.
