"""
Linear algebra operations for symmetric tensor networks.

This module provides tensor contraction, trace operations, and
correlation matrix computations for symmetric MPOs.

The tensor contraction is split into a *symbolic* phase (sector matching,
block placement, output coordinates) and a *numeric* phase (block
assembly + GEMM). The symbolic phase depends only on the block structure
of the operands, which in TEBD is identical from one time step to the
next except right after truncation events. It is therefore memoized in
an LRU cache keyed on the operands' structure, so that in steady state
only the numeric phase runs. This is the analogue of the cached fusion
data used by TeNPy / ITensor and is the main lever for reducing the
small-chi overhead.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING
from collections import OrderedDict

from .tensor import SymmetricTensor, swap_indices, mask_coordinates
from . import sparse_ops as so

if TYPE_CHECKING:
    from .mpo import SymmetricMPO


# ---------------------------------------------------------------------------
# Contraction-plan cache
# ---------------------------------------------------------------------------

_PLAN_CACHE: "OrderedDict[tuple, _ContractionPlan]" = OrderedDict()
_PLAN_CACHE_MAXSIZE = 1024


def clear_contraction_cache() -> None:
    """Empty the memoized contraction-plan cache."""
    _PLAN_CACHE.clear()


def contraction_cache_info() -> tuple[int, int]:
    """Return (current size, max size) of the plan cache."""
    return len(_PLAN_CACHE), _PLAN_CACHE_MAXSIZE


def _structure_key(T) -> tuple:
    """Fingerprint of everything the symbolic phase depends on."""
    return (
        T.n_legs,
        T.n_sectors,
        bool(T.data_as_tensors),
        int(T.alpha),
        T.coordinates.tobytes(),
        T.shapes.tobytes(),
        T.arrows.tobytes(),
        T.leg_type.tobytes(),
    )


class _ContractionPlan:
    """Precomputed symbolic data for one (A-structure, B-structure, legs)."""

    __slots__ = (
        "left_A", "right_A", "left_B", "right_B",
        "out_n_legs", "out_phys_dims", "out_arrows", "out_leg_type",
        "out_coordinates", "out_shapes", "out_n_sectors",
        "sector_specs",
    )


def _reshaped_block_shapes(tensor, left_legs, right_legs) -> NDArray:
    """
    Shapes of the blocks as they appear after `reshape_data_tensors`,
    computed without touching the data (mirrors that function's logic).
    """
    n_legs = np.arange(tensor.n_legs)
    if tensor.data_as_tensors:
        shapes = np.empty((2, tensor.n_sectors), dtype=np.intp)
        if len(left_legs) > 0:
            shapes[0] = np.prod(tensor.shapes[left_legs, :], axis=0)
        else:
            shapes[0] = 1
        if len(right_legs) > 0:
            shapes[1] = np.prod(tensor.shapes[right_legs, :], axis=0)
        else:
            shapes[1] = 1
        return shapes

    mask_virtual = tensor.leg_type == "v"
    n_virtual_left = np.sum(tensor.leg_type[left_legs] == 'v')

    if n_virtual_left == 1:
        mask_L = np.isin(n_legs, left_legs)
        mask_R = np.isin(n_legs, right_legs)
        left_virtual = n_legs[mask_L & mask_virtual]
        right_virtual = n_legs[mask_R & mask_virtual]
        if left_virtual[0] < right_virtual[0]:
            return tensor.shapes[mask_virtual]
        return tensor.shapes[mask_virtual][::-1, :]

    shapes = np.ones((2, tensor.n_sectors), dtype=np.intp)
    if n_virtual_left != 0:
        shapes[0] = np.prod(tensor.shapes[mask_virtual], axis=0)
    else:
        shapes[1] = np.prod(tensor.shapes[mask_virtual], axis=0)
    return shapes


def _build_assembly_specs(block_coords, mat_shapes, sector_mask):
    """
    Turn a block-coordinate matrix into flat placement instructions:
    a list of (global_block_index, r0, r1, c0, c1), the matrix shape,
    and the first-valid-block lists used for output coordinates.
    """
    row_sizes = mat_shapes[0, :, 0]
    col_sizes = mat_shapes[1, 0, :]
    row_off = np.concatenate(([0], np.cumsum(row_sizes)))
    col_off = np.concatenate(([0], np.cumsum(col_sizes)))

    glob = np.where(sector_mask)[0]

    specs = []
    list_left = []
    for i, row_coords in enumerate(block_coords):
        valid = row_coords[row_coords != -1]
        if len(valid) > 0:
            list_left.append(valid[0])
        for j, idx in enumerate(row_coords):
            if idx != -1:
                specs.append((
                    int(glob[idx]),
                    int(row_off[i]), int(row_off[i + 1]),
                    int(col_off[j]), int(col_off[j + 1]),
                ))

    list_right = []
    for col in block_coords.T:
        valid = col[col != -1]
        if len(valid) > 0:
            list_right.append(valid[0])

    shape = (int(row_off[-1]), int(col_off[-1]))
    return specs, shape, list_left, list_right, row_off, col_off


def _build_contraction_plan(A, B, indices) -> _ContractionPlan:
    """Symbolic phase of the contraction (mirrors the original algorithm)."""
    alpha = A.alpha
    x_ind = np.asarray(indices[0], dtype=np.intp)
    y_ind = np.asarray(indices[1], dtype=np.intp)

    n_A = np.arange(A.n_legs)
    n_B = np.arange(B.n_legs)

    left_A = np.setdiff1d(n_A, x_ind)
    right_A = x_ind
    left_B = y_ind
    right_B = np.setdiff1d(n_B, y_ind)

    A_shapes = _reshaped_block_shapes(A, left_A, right_A)
    B_shapes = _reshaped_block_shapes(B, left_B, right_B)

    # Determine symmetry sectors for contraction
    R_arrow_A = A.arrows[right_A] == 'i'
    R_sigma_A = A.leg_type[right_A] == 's'
    L_arrow_B = B.arrows[left_B] == 'i'
    L_sigma_B = B.leg_type[left_B] == 's'

    sectors_A = np.sum(
        A.coordinates[right_A] *
        (-1 * R_arrow_A[:, None] + (~R_arrow_A)[:, None]) *
        (alpha * R_sigma_A[:, None] + (~R_sigma_A[:, None])),
        axis=0
    )
    sectors_B = np.sum(
        B.coordinates[left_B] *
        (L_arrow_B[:, None] - 1 * (~L_arrow_B)[:, None]) *
        (alpha * L_sigma_B[:, None] + (~L_sigma_B[:, None])),
        axis=0
    )

    unique_A = np.unique(sectors_A)
    unique_B = np.unique(sectors_B)
    sectors = np.intersect1d(unique_A, unique_B)
    n_sectors = len(sectors)

    A_sector_mask = sectors_A[None, :] == sectors[:, None]
    B_sector_mask = sectors_B[None, :] == sectors[:, None]

    L_coords, A_shapes_sect, n_L_A, n_R_A, id_A = so.construct_subblock_sectors(
        A, left_A, right_A, A_shapes, A_sector_mask, n_sectors
    )
    R_coords, B_shapes_sect, n_L_B, n_R_B, id_B = so.construct_subblock_sectors(
        B, left_B, right_B, B_shapes, B_sector_mask, n_sectors
    )

    L_coords, R_coords, A_shapes_sect, B_shapes_sect, empty = so.check_subblock_sectors(
        L_coords, R_coords, A_shapes_sect, B_shapes_sect, id_A, id_B
    )

    plan = _ContractionPlan()
    plan.left_A = left_A
    plan.right_A = right_A
    plan.left_B = left_B
    plan.right_B = right_B

    phys_dims_L = int(np.sum(A.leg_type[left_A] != 'v') / 2)
    phys_dims_R = int(np.sum(B.leg_type[right_B] != 'v') / 2)
    plan.out_phys_dims = phys_dims_L + phys_dims_R
    plan.out_n_legs = len(left_A) + len(right_B)
    plan.out_arrows = np.concatenate([A.arrows[left_A], B.arrows[right_B]])
    plan.out_leg_type = np.concatenate([A.leg_type[left_A], B.leg_type[right_B]])

    if not A.data_as_tensors and plan.out_leg_type.size > 0:
        virtual_mask = plan.out_leg_type == "v"
    else:
        virtual_mask = None  # use full shapes

    coord_cols = []
    shape_cols = []
    sector_specs = []
    slot = 0

    for s in range(n_sectors):
        if empty[s]:
            continue

        A_specs, A_matshape, list_L_A, _, row_off_A, _ = _build_assembly_specs(
            L_coords[s], A_shapes_sect[s], A_sector_mask[s]
        )
        B_specs, B_matshape, _, list_R_B, _, col_off_B = _build_assembly_specs(
            R_coords[s], B_shapes_sect[s], B_sector_mask[s]
        )

        if len(list_L_A) != n_L_A[s] or len(list_R_B) != n_R_B[s]:
            raise RuntimeError(
                "tensor_contract: a sub-block row/column has no valid block "
                "after sector alignment; the block structure is inconsistent."
            )

        A_coords_out = np.repeat(
            A.coordinates[left_A][:, A_sector_mask[s]][:, list_L_A],
            n_R_B[s], axis=1
        )
        B_coords_out = np.tile(
            B.coordinates[right_B][:, B_sector_mask[s]][:, list_R_B],
            (1, n_L_A[s])
        )
        coord_cols.append(np.concatenate([A_coords_out, B_coords_out], axis=0))

        A_shapes_out = np.repeat(
            A.shapes[left_A][:, A_sector_mask[s]][:, list_L_A],
            n_R_B[s], axis=1
        )
        B_shapes_out = np.tile(
            B.shapes[right_B][:, B_sector_mask[s]][:, list_R_B],
            (1, n_L_A[s])
        )
        sect_shapes = np.concatenate([A_shapes_out, B_shapes_out], axis=0)
        shape_cols.append(sect_shapes)

        if virtual_mask is None:
            reshape_shapes = sect_shapes
        else:
            reshape_shapes = sect_shapes[virtual_mask, :]

        out_specs = []
        k = 0
        for i in range(n_L_A[s]):
            r0, r1 = int(row_off_A[i]), int(row_off_A[i + 1])
            for j in range(n_R_B[s]):
                c0, c1 = int(col_off_B[j]), int(col_off_B[j + 1])
                dims = tuple(int(x) for x in reshape_shapes[:, k])
                out_specs.append((slot, r0, r1, c0, c1, dims))
                slot += 1
                k += 1

        sector_specs.append((A_specs, A_matshape, B_specs, B_matshape, out_specs))

    plan.sector_specs = sector_specs
    plan.out_n_sectors = slot

    if coord_cols:
        plan.out_coordinates = np.concatenate(coord_cols, axis=1)
        plan.out_shapes = np.concatenate(shape_cols, axis=1)
    else:
        plan.out_coordinates = np.zeros((plan.out_n_legs, 0), dtype=np.intp)
        plan.out_shapes = np.ones((plan.out_n_legs, 0), dtype=np.intp)

    return plan


def _get_plan(A, B, indices) -> _ContractionPlan:
    key = (
        _structure_key(A),
        _structure_key(B),
        tuple(int(i) for i in indices[0]),
        tuple(int(i) for i in indices[1]),
    )
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_contraction_plan(A, B, indices)
        _PLAN_CACHE[key] = plan
        if len(_PLAN_CACHE) > _PLAN_CACHE_MAXSIZE:
            _PLAN_CACHE.popitem(last=False)
    else:
        _PLAN_CACHE.move_to_end(key)
    return plan


def tensor_contract(
    A: SymmetricTensor,
    B: SymmetricTensor,
    indices: tuple[list[int], list[int]]
) -> SymmetricTensor:
    """
    Contract two symmetric tensors along specified indices.

    Performs the tensor product of A and B, summing over the legs
    specified in indices while preserving the block-sparse structure.

    Parameters
    ----------
    A : SymmetricTensor
        First tensor.
    B : SymmetricTensor
        Second tensor.
    indices : tuple of (list, list)
        (indices_A, indices_B) specifying which legs to contract.

    Returns
    -------
    SymmetricTensor
        The contracted tensor.
    """
    plan = _get_plan(A, B, indices)

    # Numeric phase: reshape blocks as matrices (mostly views), assemble,
    # multiply, scatter into output blocks.
    A_blocks, _ = so.reshape_data_tensors(A, plan.left_A, plan.right_A,
                                          return_shapes=False)
    B_blocks, _ = so.reshape_data_tensors(B, plan.left_B, plan.right_B,
                                          return_shapes=False)

    C = SymmetricTensor(
        A.L, A.d, plan.out_phys_dims,
        alpha=A.alpha,
        data_as_tensors=A.data_as_tensors,
        n_legs=plan.out_n_legs,
        n_sectors=plan.out_n_sectors
    )
    C.arrows = plan.out_arrows.copy()
    C.leg_type = plan.out_leg_type.copy()
    C.coordinates = plan.out_coordinates.copy()
    C.shapes = plan.out_shapes.copy()

    data = C.data
    for A_specs, A_matshape, B_specs, B_matshape, out_specs in plan.sector_specs:
        mat_A = np.zeros(A_matshape, dtype=complex)
        for g, r0, r1, c0, c1 in A_specs:
            mat_A[r0:r1, c0:c1] = A_blocks[g]
        mat_B = np.zeros(B_matshape, dtype=complex)
        for g, r0, r1, c0, c1 in B_specs:
            mat_B[r0:r1, c0:c1] = B_blocks[g]

        mat_C = mat_A @ mat_B

        for slot, r0, r1, c0, c1, dims in out_specs:
            data[slot] = mat_C[r0:r1, c0:c1].copy().reshape(dims)

    return C


# ---------------------------------------------------------------------------
# Traces and environments
# ---------------------------------------------------------------------------

def _conj_copy(tensor: SymmetricTensor) -> SymmetricTensor:
    """Metadata copy with conjugated blocks (originals untouched)."""
    T = tensor.copy(copy_data=False)
    for b in range(T.n_sectors):
        T.data[b] = T.data[b].conj()
    return T


def _transposed_view(tensor: SymmetricTensor) -> SymmetricTensor:
    """Metadata copy with sigma/sigma' leg types swapped (legs 1 and 2)."""
    T = tensor.copy(copy_data=False)
    T.leg_type[[1, 2]] = T.leg_type[[2, 1]]
    return T


def trace_mpo_product(
    A: 'SymmetricMPO',
    B: 'SymmetricMPO',
    conj_A: bool = False,
    conj_B: bool = False
) -> complex:
    """
    Compute Tr(A^dag B) (and variants) using the PacMan method.

    Contracts the MPO product efficiently without forming the full matrix.

    Parameters
    ----------
    A : SymmetricMPO
        First MPO.
    B : SymmetricMPO
        Second MPO.
    conj_A : bool
        Take complex conjugate of A.
    conj_B : bool
        Take complex conjugate of B.

    Returns
    -------
    complex
        conj_A only          : Tr(A^dag B)
        conj_B only          : Tr(A B^dag)
        neither              : Tr(A B)
        both                 : Tr(A^dag B^dag)  (previously conj_B was
                               silently ignored in this case)
    """
    # Initialize PacMan tensor
    pac = SymmetricTensor(
        A.L, A.d, 0,
        alpha=A.alpha,
        data_as_tensors=A.data_as_tensors,
        n_legs=2, n_sectors=1
    )

    if A.alpha == -1:
        pac.coordinates[:, 0] = (A.L, A.L)
        pac.leg_sectors = np.array([np.array([A.L]), np.array([A.L])], dtype=object)
    elif A.alpha == A.L + 1:
        pac.coordinates[:, 0] = (0, 0)
        pac.leg_sectors = np.array([np.array([0]), np.array([0])], dtype=object)

    pac.shapes = np.ones((2, 1), dtype=np.intp)
    pac.data[0] = np.ones((1, 1), dtype=complex)
    pac.arrows = np.array(['o', 'o'])
    pac.leg_type = np.array(['v', 'v'])

    for i in range(A.L):
        B_A = A.TN[f"B{i}"]
        B_B = B.TN[f"B{i}"]

        if conj_A:
            B_A = _conj_copy(B_A)
        if conj_B:
            B_B = _conj_copy(B_B)

        if conj_A == conj_B:
            # Plain product (or product of two daggers): the operator
            # transpose is encoded by swapping sigma/sigma' on A and
            # crossing the physical contraction pattern on B.
            B_A = _transposed_view(B_A)
            dims_B = [0, 2, 1]
        else:
            dims_B = [0, 1, 2]

        pac_A = tensor_contract(pac, B_A, ([0], [0]))
        pac_A.arrows[[1, 2]] = 'o'
        pac = tensor_contract(pac_A, B_B, ([0, 1, 2], dims_B))

    if len(pac.data) == 0:
        return 0.0
    return np.sum(pac.data[0])


def site_pacman(
    O1: 'SymmetricMPO',
    O2: 'SymmetricMPO',
    conj_A: bool = False,
    conj_B: bool = False,
    left: bool = True,
    right: bool = True
) -> tuple[dict, dict]:
    """
    Compute PacMan environments at each site.

    Stores the left and right partial contractions, useful for
    computing local observables efficiently.

    Parameters
    ----------
    O1 : SymmetricMPO
        First MPO.
    O2 : SymmetricMPO
        Second MPO.
    conj_A : bool
        Conjugate first MPO.
    conj_B : bool
        Conjugate second MPO.
    left : bool
        Compute left environments.
    right : bool
        Compute right environments.

    Returns
    -------
    L_PM : dict
        Left PacMan at each site.
    R_PM : dict
        Right PacMan at each site.

    Notes
    -----
    The returned environments are read-only snapshots; callers that
    modify them (e.g. shifting sectors) must copy first.
    """
    A = O1
    B = O2
    L = A.L

    # Initialize left/right PacMan
    L_pac = SymmetricTensor(
        A.L, A.d, 0,
        alpha=A.alpha,
        n_legs=2, n_sectors=1
    )
    R_pac = SymmetricTensor(
        A.L, A.d, 0,
        alpha=A.alpha,
        n_legs=2, n_sectors=1
    )

    A_sect = A.TN[f"B{L-1}"].coordinates[-1, 0]
    B_sect = B.TN[f"B{L-1}"].coordinates[-1, 0]

    if A.alpha == -1:
        L_pac.coordinates[:, 0] = (L, L)
        L_pac.leg_sectors = np.array([np.array([L]), np.array([L])], dtype=object)
        R_pac.coordinates[:, 0] = (A_sect, B_sect)
        R_pac.leg_sectors = np.array([np.array([A_sect]), np.array([B_sect])], dtype=object)
    elif A.alpha == L + 1:
        L_pac.coordinates[:, 0] = (0, 0)
        L_pac.leg_sectors = np.array([np.array([0]), np.array([0])], dtype=object)
        R_pac.coordinates[:, 0] = (A_sect, B_sect)
        R_pac.leg_sectors = np.array([np.array([A_sect]), np.array([B_sect])], dtype=object)

    L_pac.shapes = np.ones((2, 1), dtype=np.intp)
    L_pac.data[0] = np.ones((1, 1), dtype=complex)
    L_pac.arrows = np.array(['o', 'o'])
    L_pac.leg_type = np.array(['v', 'v'])

    R_pac.shapes = np.ones((2, 1), dtype=np.intp)
    R_pac.data[0] = np.ones((1, 1), dtype=complex)
    R_pac.arrows = np.array(['i', 'i'])
    R_pac.leg_type = np.array(['v', 'v'])

    L_PM = {}
    R_PM = {}

    for i in range(L):
        L_A = A.TN[f"B{i}"]
        L_B = B.TN[f"B{i}"]
        R_A = A.TN[f"B{L-1-i}"]
        R_B = B.TN[f"B{L-1-i}"]

        if conj_A:
            L_A = _conj_copy(L_A)
            R_A = _conj_copy(R_A)
            dims_L_A, dims_L_B = [0, 1, 2], [0, 1, 2]
            dims_R_A, dims_R_B = [1, 2, 3], [1, 2, 3]
        elif conj_B:
            L_B = _conj_copy(L_B)
            R_B = _conj_copy(R_B)
            dims_L_A, dims_L_B = [0, 1, 2], [0, 1, 2]
            dims_R_A, dims_R_B = [1, 2, 3], [1, 2, 3]
        else:
            dims_L_A, dims_L_B = [0, 1, 2], [0, 2, 1]
            dims_R_A, dims_R_B = [1, 2, 3], [2, 1, 3]
            L_A = _transposed_view(L_A)
            R_A = _transposed_view(R_A)

        if left:
            L_PM[i] = L_pac
            pac_A = tensor_contract(L_pac, L_A, ([0], [0]))
            pac_A.arrows[[1, 2]] = 'o'
            L_pac = tensor_contract(pac_A, L_B, (dims_L_A, dims_L_B))

        if right:
            R_PM[L - 1 - i] = R_pac
            pac_A = tensor_contract(R_A, R_pac, ([3], [0]))
            pac_A.arrows[[1, 2]] = 'o'
            R_pac = tensor_contract(R_B, pac_A, (dims_R_B, dims_R_A))
            R_pac = swap_indices(R_pac, [1, 0], [0, 1])

    return L_PM, R_PM


def compute_obdmo(
    mpo: 'SymmetricMPO',
    unitary: bool = False,
    optimized: bool = True
) -> NDArray[np.complexfloating]:
    """
    Compute the R matrix (correlation matrix) for an MPO.

    The R matrix is defined as R_ij = <c_i^dag c_j> where the expectation
    is taken with respect to the operator interpreted as a density matrix.

    Parameters
    ----------
    mpo : SymmetricMPO
        The operator.
    unitary : bool
        Whether the operator is unitary (enables optimizations).
    optimized : bool
        Use optimized O(L^2) algorithm vs naive O(L^3).

    Returns
    -------
    R : ndarray, shape (2L, 2L), complex
        The correlation matrix.
    """
    from .mpo import apply_fermionic_op, apply_spin_op, mask_coordinates

    O = mpo
    norm_O = np.abs(O.norm())
    L = O.L

    R = np.zeros((2 * L, 2 * L), dtype=complex)

    if optimized:
        _, R_PM_O = site_pacman(O, O, left=False, conj_A=True)

        O_sgn_L = apply_spin_op("sz", O, np.arange(L), side="L")
        O_sgn_R = apply_spin_op("sz", O, np.arange(L), side="R")

        for i in range(2 * L):
            ind_i = i // 2
            side_i = "L" if i % 2 == 0 else "R"
            O_s = O_sgn_R if side_i == "L" else O_sgn_L

            if not unitary:
                O1_R = apply_fermionic_op("c", O, ind_i, side_i, sign_side="R")
                L_PM_between, _ = site_pacman(O1_R, O, right=False, conj_A=True)

            O1_L = apply_fermionic_op("c", O, ind_i, side_i, sign_side="L")
            L_PM_start, _ = site_pacman(O1_L, O_s, right=False, conj_A=True)

            for j in range(i, 2 * L):
                if unitary and (i % 2 == j % 2):
                    continue

                ind_j = j // 2
                side_j = "L" if j % 2 == 0 else "R"

                O1_loc = _conj_copy(O1_L.TN[f"B{ind_j}"])

                O2_loc = O.TN[f"B{ind_j}"].copy(copy_data=False)

                # Apply c_j locally
                leg_nb = 2 if side_j == "L" else 1
                inc_st = 1 if side_j == "L" else 0
                mask = O2_loc.coordinates[leg_nb, :] == inc_st
                O2_loc = mask_coordinates(O2_loc, mask)

                O2_loc.coordinates[leg_nb, :] = (1 + inc_st) % 2
                O2_loc.coordinates[-1, :] = (
                    O2_loc.coordinates[0, :] +
                    O.alpha * O2_loc.coordinates[1, :] +
                    O2_loc.coordinates[2, :]
                )
                O2_loc.invalidate_leg_sectors()

                # Adapt right PacMan sectors (copy: environments are shared)
                R_pac = R_PM_O[ind_j].copy(copy_data=False)
                R_pac.coordinates[0, :] += -1 if side_i == "L" else O.alpha
                R_pac.coordinates[1, :] += -1 if side_j == "L" else O.alpha
                R_pac.invalidate_leg_sectors()

                # Contract
                L_PM = L_PM_between[ind_j] if side_i == side_j else L_PM_start[ind_j]
                pac_A = tensor_contract(L_PM, O1_loc, ([0], [0]))
                pac_A.arrows[[1, 2]] = 'o'
                B_out = tensor_contract(pac_A, O2_loc, ([0, 1, 2], [0, 1, 2]))

                result = tensor_contract(B_out, R_pac, ([0, 1], [0, 1]))
                R[L * (i % 2) + ind_i, L * (j % 2) + ind_j] = np.sum(result.data) / norm_O

        if unitary:
            diag_val = 0.5 * 2 ** float(L) / norm_O
            np.fill_diagonal(R[:L, :L], diag_val)
            np.fill_diagonal(R[L:, L:], 1 - diag_val)
    else:
        # Naive O(L^3) implementation
        for i in range(2 * L):
            ind_i = i // 2
            side_i = "L" if i % 2 == 0 else "R"
            O1 = apply_fermionic_op("c", O, ind_i, side_i)

            for j in range(i, 2 * L):
                ind_j = j // 2
                side_j = "L" if j % 2 == 0 else "R"
                O2 = apply_fermionic_op("c", O, ind_j, side_j)

                R[L * (i % 2) + ind_i, L * (j % 2) + ind_j] = (
                    trace_mpo_product(O1, O2, conj_A=True) / norm_O
                )

    # Make Hermitian
    R = R + R.conj().T - np.diag(R.diagonal())
    return R


def compute_otoc(
    mpo: 'SymmetricMPO',
    sites: NDArray | None = None
) -> NDArray:
    """
    Compute the out-of-time-order correlator.

    For a local operator O(t), computes Tr(O(t) Sz_i O(t) Sz_i) / Tr(O^dag O).

    Parameters
    ----------
    mpo : SymmetricMPO
        The time-evolved operator.
    sites : ndarray, optional
        Sites at which to compute OTOC. Default: all sites.

    Returns
    -------
    otoc : ndarray
        OTOC values at each site.
    """
    from .mpo import apply_spin_op

    O = mpo
    L = O.L
    otoc = np.zeros(L)

    if sites is None:
        sites = np.arange(L)

    L_PM, R_PM = site_pacman(O, O)

    # Compute normalization
    norm_tensor = tensor_contract(L_PM[1], R_PM[0], ([0, 1], [0, 1]))
    norm = np.sum(norm_tensor.data).real

    for i in sites:
        O_i = apply_spin_op("sz", O, i).TN[f"B{i}"].copy(copy_data=False)

        O_i_t = _transposed_view(O_i)
        A_tensor = tensor_contract(L_PM[i], O_i_t, ([0], [0]))
        A_tensor.arrows[[1, 2]] = 'o'
        B_tensor = tensor_contract(A_tensor, O_i, ([0, 1, 2], [0, 2, 1]))

        result = tensor_contract(B_tensor, R_PM[i], ([0, 1], [0, 1]))
        otoc[i] = np.sum(result.data).real / norm

    return otoc
