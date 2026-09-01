"""Variational ansatz circuit for lattice protein folding (VQE).

Constructs the parameterised circuit described in the project documentation,
following the tetrahedral-lattice model of Robert et al. (2021).
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def build_ansatz(
    num_amino_acids: int,
    encoding: str = "dense",
    num_interaction_qubits: int = 0,
    depth: int = 2,
    entangling: str = "closed_loop",
    measure: bool = True,
) -> QuantumCircuit:
    """Build a parameterised VQE ansatz for lattice protein folding.

    Parameters
    ----------
    num_amino_acids : int
        Length of the protein sequence (*N*).
    encoding : str
        ``"dense"`` (2 qubits/turn) or ``"sparse"`` (4 qubits/turn).
    num_interaction_qubits : int
        Auxiliary qubits for contact tracking.
    depth : int
        Number of entangle→rotate block repetitions.
    entangling : str
        ``"closed_loop"`` (linear CNOT chain + loopback) or ``"all_to_all"``.
    measure : bool
        If *True* append ``measure_all()`` (set *False* for statevector VQE).

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for VQE or sampling.
    """
    if encoding == "dense":
        n_config = 2 * (num_amino_acids - 3)
    elif encoding == "sparse":
        n_config = 4 * (num_amino_acids - 3)
    else:
        raise ValueError(f"Unknown encoding: {encoding!r}")

    n = n_config + num_interaction_qubits
    qc = QuantumCircuit(n)
    theta = ParameterVector("θ", n * (depth + 1))
    idx = 0

    # ── Step A: Hadamard + first RY layer ──────────────────────────────
    for q in range(n):
        qc.h(q)
    for q in range(n):
        qc.ry(theta[idx], q)
        idx += 1
    qc.barrier()

    # ── Step B: repeated entangle → rotate blocks ──────────────────────
    for _ in range(depth):
        if entangling == "closed_loop":
            for q in range(n - 1):
                qc.cx(q, q + 1)
            if n > 2:
                qc.cx(n - 1, 0)
        elif entangling == "all_to_all":
            for i in range(n):
                for j in range(i + 1, n):
                    qc.cx(i, j)
        else:
            raise ValueError(f"Unknown entangling type: {entangling!r}")
        qc.barrier()

        for q in range(n):
            qc.ry(theta[idx], q)
            idx += 1
        qc.barrier()

    # ── Step C: measurement ────────────────────────────────────────────
    if measure:
        qc.measure_all()

    return qc
