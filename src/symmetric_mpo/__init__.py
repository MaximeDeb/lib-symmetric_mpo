"""
Symmetric Tensor Network library for Matrix Product Operators.

A Python library for efficient manipulation of symmetric Matrix Product
Operators (MPOs) with U(1) symmetry conservation. Designed for many-body 
operator dynamics using time evolution (TEBD), natural orbital rotations, 
and correlation function computations.

Main Classes
------------
SymmetricMPO
    The main MPO class with Vidal's Gamma-Lambda representation.
SymmetricTensor
    Block-sparse tensor with symmetry conservation.
SymmetricLambda
    Diagonal singular value matrices.
SymmetricGate
    Two-site gates for TEBD evolution.

Key Functions
-------------
tensor_contract
    Contract two symmetric tensors.
apply_gate
    Apply a two-site gate using TEBD.
compute_R_matrix
    Compute correlation matrix from MPO.
build_trotter_sequence
    Construct Trotter decomposition.
givens_rotations
    Compute Givens rotation decomposition.

Example
-------
>>> from symmetric_mpo import SymmetricMPO, build_trotter_sequence, apply_gate
>>> 
>>> # Create identity MPO
>>> L, d = 10, 2
>>> params = {'L': L, 'd': d, 'J': 1.0, 'Jz': 1.0}
>>> mpo = SymmetricMPO(L, d, L, 1, chi_max=256, alpha=-1, initial="Id")
>>> 
>>> # Build Trotter sequence for Heisenberg model
>>> steps, U, U_dag, _ = build_trotter_sequence(
...     order=2, n_steps=100, dt=0.01, n_parts=2,
...     obs_interval=10, params=params, model="Heis_nn"
... )
>>> 
>>> # Time evolve
>>> gate_layers = {"H0": [(i, i+1) for i in range(0, L-1, 2)],
...                "H1": [(i, i+1) for i in range(1, L-1, 2)]}
>>> for step in steps:
...     for (l1, l2) in gate_layers[step.layer]:
...         mpo, _ = apply_gate(mpo, U[step.layer, step.dt],
...                             U_dag[step.layer, step.dt], l1, l2)
"""

__version__ = "0.2.0"
__author__ = "MaximeD"

# Core tensor classes
from .tensor import (
    SymmetricTensor,
    SymmetricLambda,
    swap_indices,
    mask_coordinates,
)

# MPO and gate classes
from .mpo import (
    SymmetricMPO,
    SymmetricGate,
    apply_fermionic_op,
    apply_spin_op,
)

# Linear algebra operations
from .linalg import (
    tensor_contract,
    trace_mpo_product,
    site_pacman,
    compute_obdmo,
    compute_otoc,
)

# Time evolution
from .tebd import (
    apply_gate,
    apply_lambda,
)

# Trotter decomposition
from .trotter import (
    TrotterStep,
    build_trotter_sequence,
    commuting_sequence,
    get_gate_sequence,
)

# Givens rotations
from .givens import (
    givens_rotations,
    rotation_from_givens,
    decompose_unitary_givens,
)

# Sparse operations (usually internal)
from . import sparse_ops


__all__ = [
    # Version
    "__version__",
    # Tensor classes
    "SymmetricTensor",
    "SymmetricLambda",
    "swap_indices",
    "mask_coordinates",
    # MPO classes
    "SymmetricMPO",
    "SymmetricGate",
    "apply_fermionic_op",
    "apply_spin_op",
    # Linear algebra
    "tensor_contract",
    "trace_mpo_product",
    "site_pacman",
    "compute_obdmo",
    "compute_otoc",
    # TEBD
    "apply_gate",
    "apply_lambda",
    # Trotter
    "TrotterStep",
    "build_trotter_sequence",
    "commuting_sequence",
    "get_gate_sequence",
    # Givens
    "givens_rotations",
    "rotation_from_givens",
    "decompose_unitary_givens",
    # Internal modules
    "sparse_ops",
]
