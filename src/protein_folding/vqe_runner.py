"""VQE optimiser for quantum protein folding.

Uses exact statevector simulation to compute ⟨ψ(θ)|H|ψ(θ)⟩ and a
classical optimiser (COBYLA by default) to minimise the energy.
After convergence the most-probable bitstring is returned so it can
be decoded into a 3-D conformation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from .ansatz import build_ansatz
from .hamiltonian import build_hamiltonian

log = logging.getLogger(__name__)


@dataclass
class VQEResult:
    """Outcome of a VQE run."""

    optimal_energy: float
    optimal_params: np.ndarray
    bitstring: str
    num_evaluations: int


def run_vqe(
    num_amino_acids: int,
    contact_energies: np.ndarray | None = None,
    penalty: float = 10.0,
    num_interaction_qubits: int = 0,
    depth: int = 2,
    max_iterations: int = 200,
    optimizer: str = "COBYLA",
    seed: int | None = None,
) -> VQEResult:
    """Run VQE to find the minimum-energy lattice conformation.

    Parameters
    ----------
    num_amino_acids : int
        Protein length *N*.
    contact_energies : np.ndarray, optional
        (N, N) pairwise contact-energy matrix (default: −1 HP model).
    penalty : float
        Overlap-penalty weight λ.
    num_interaction_qubits : int
        Auxiliary qubits (default 0 for a minimal qubit count).
    depth : int
        Ansatz circuit depth.
    max_iterations : int
        Maximum number of optimiser iterations.
    optimizer : str
        Scipy optimiser method name (e.g. ``"COBYLA"``, ``"Nelder-Mead"``).
    seed : int, optional
        Random seed for reproducible initial parameters.

    Returns
    -------
    VQEResult
    """
    # --- build components ---
    ansatz = build_ansatz(
        num_amino_acids,
        num_interaction_qubits=num_interaction_qubits,
        depth=depth,
        measure=False,
    )

    hamiltonian = build_hamiltonian(
        num_amino_acids,
        contact_energies=contact_energies,
        penalty=penalty,
        num_interaction_qubits=num_interaction_qubits,
    )

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)

    eval_count = 0

    def cost(params: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        bound = ansatz.assign_parameters(params)
        sv = Statevector.from_instruction(bound)
        energy = float(sv.expectation_value(hamiltonian).real)
        if eval_count % 50 == 0:
            log.info("VQE iter %d  energy = %.6f", eval_count, energy)
        return energy

    log.info(
        "Starting VQE: %d qubits, %d parameters, depth %d",
        ansatz.num_qubits,
        ansatz.num_parameters,
        depth,
    )

    result = minimize(cost, x0, method=optimizer,
                      options={"maxiter": max_iterations})

    # --- extract most-probable bitstring ---
    bound = ansatz.assign_parameters(result.x)
    sv = Statevector.from_instruction(bound)
    probs = sv.probabilities_dict()
    bitstring = max(probs, key=probs.get)

    log.info(
        "VQE converged: energy = %.6f, evals = %d, bitstring = %s",
        result.fun,
        eval_count,
        bitstring,
    )

    return VQEResult(
        optimal_energy=float(result.fun),
        optimal_params=result.x,
        bitstring=bitstring,
        num_evaluations=eval_count,
    )
