"""
Time-Evolving Block Decimation (TEBD) for symmetric MPOs.

This module implements TEBD time evolution with SVD-based truncation,
maintaining the symmetry-preserving block structure throughout.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import scipy.linalg as sla
from typing import TYPE_CHECKING, Literal

from .tensor import SymmetricTensor, SymmetricLambda, swap_indices, mask_coordinates
from .linalg import tensor_contract
from . import sparse_ops as so

if TYPE_CHECKING:
    from .mpo import SymmetricMPO, SymmetricGate


def apply_gate(
    mpo: 'SymmetricMPO',
    gate: 'SymmetricGate',
    gate_dag: 'SymmetricGate | None',
    site1: int,
    site2: int,
    both_sides: bool = True,
    side: Literal["L", "R"] = "L"
) -> tuple['SymmetricMPO', float]:
    """
    Apply a two-site gate to an MPO using TEBD.
    
    Contracts the gate with two adjacent MPO tensors, then splits
    them back using SVD truncation.
    
    Parameters
    ----------
    mpo : SymmetricMPO
        The MPO to evolve.
    gate : SymmetricGate
        The gate to apply on sigma (left physical indices).
    gate_dag : SymmetricGate or None
        The conjugate gate (used when both_sides=True).
    site1, site2 : int
        The two adjacent sites (should be site2 = site1 + 1).
    both_sides : bool
        If True, apply gate from left and gate_dag from right.
    side : {"L", "R"}
        Which side to apply when both_sides=False and right_gate=None.
        WARNING: the side="R" branch is faithfully preserved from the
        original code but appears to have never worked: the plain gate's
        leg_type ('s') is contracted against the primed physical legs,
        which mislabels the conserved charge. From the identity it
        silently produces a norm-collapsed operator, and once the bond
        dimension exceeds 1 it raises a shape mismatch. Verify against a
        dense reference before relying on it (right multiplication can
        instead be obtained as (U_dag O_dag)_dag via side="L").
    Returns
    -------
    mpo : SymmetricMPO
        The evolved MPO.
    discarded : float
        Sum of squared discarded singular values.
    """
    # Contract the two-site tensor
    B1 = mpo.TN[f"B{site1}"]
    B2 = mpo.TN[f"B{site2}"]
    
    if not both_sides and side == "R":
        ind = [3, 4]
        tr = [2, 3, 0, 4, 1, 5]
    else:
        ind = [1, 2]
        tr = [2, 0, 1, 3, 4, 5]
    
    PhiB = tensor_contract(B1, B2, ([3], [0]))
    PhiB = swap_indices(PhiB, [0, 1, 3, 2, 4, 5], [0, 1, 2, 3, 4, 5])
        
    # Swap to standard order
    UPhiB = tensor_contract(gate, PhiB, ([2, 3], ind))
    UPhiB = swap_indices(UPhiB, tr, [0, 1, 2, 3, 4, 5])

    if both_sides:
        UPhiB = tensor_contract(gate_dag, UPhiB, ([2, 3], [3, 4]))
        UPhiB = swap_indices(UPhiB, [2, 3, 4, 0, 1, 5], [0, 1, 2, 3, 4, 5])

    
    # Split back into two tensors
    mpo, discarded = _symmetric_canonicalize(
        UPhiB, mpo, site1, site2,
        truncation_type=mpo.truncation_type
    )
    
    return mpo, discarded


def apply_lambda(
    lam: SymmetricLambda,
    tensor: SymmetricTensor
) -> SymmetricTensor:
    """
    Multiply Lambda onto the left virtual leg of a tensor.
    
    Parameters
    ----------
    lam : SymmetricLambda
        The diagonal singular value matrix.
    tensor : SymmetricTensor
        The tensor to modify.
        
    Returns
    -------
    SymmetricTensor
        The scaled tensor.
    """
    A = tensor
    n_legs = np.arange(A.n_legs)
    left_legs = np.zeros(1, dtype=np.intp)
    right_legs = np.setdiff1d(n_legs, 0)

    # Lambda is block-diagonal in the symmetry sector, so it acts on each
    # tensor block independently: no need to assemble/split a big matrix.
    sectors = np.intersect1d(lam.right_sectors, A.leg_sectors[0])
    covered = np.isin(A.coordinates[0, :], sectors)

    # Blocks whose left sector carries no singular values are multiplied
    # by zero; drop them instead of silently corrupting them (old code
    # left them uninitialized and scattered stale entries into slot 0).
    if not np.all(covered):
        A = mask_coordinates(A.copy(copy_data=False), covered)

    A_blocks, _ = so.reshape_data_tensors(A, left_legs, right_legs,
                                          return_shapes=False)

    B = SymmetricTensor(
        A.L, A.d, A.phys_dims,
        alpha=A.alpha,
        data_as_tensors=A.data_as_tensors,
        n_legs=A.n_legs,
        n_sectors=A.n_sectors
    )
    B.arrows = A.arrows.copy()
    B.leg_type = A.leg_type.copy()
    B.coordinates = A.coordinates.copy()
    B.shapes = A.shapes.copy()

    lam_data = lam.data
    if A.data_as_tensors:
        for i in range(A.n_sectors):
            vec = lam_data[A.coordinates[0, i]]
            res = vec[:, None] * A_blocks[i]
            B.data[i] = np.asarray(res, dtype=complex).reshape(B.shapes[:, i])
    else:
        for i in range(A.n_sectors):
            vec = lam_data[A.coordinates[0, i]]
            B.data[i] = np.asarray(vec[:, None] * A_blocks[i], dtype=complex)

    return B


def _symmetric_canonicalize(
    PhiB: SymmetricTensor,
    mpo: 'SymmetricMPO',
    site1: int,
    site2: int,
    truncation_type: str = "global"
) -> tuple['SymmetricMPO', float]:
    """
    Split a two-site tensor back into separate tensors using SVD.
    
    Implements Vidal's canonical form with optional truncation.
    """
    alpha = PhiB.alpha
    lam = mpo.TN[f"Lam{site1}"]
    
    # Apply Lambda from left
    Phi = apply_lambda(lam, PhiB)
    
    # Partition legs for SVD
    left_legs = [0, 1, 3]   # left virtual, sigma1, sigma1'
    right_legs = [2, 4, 5]  # sigma2, sigma2', right virtual
    
    Phi_blocks, Phi_shapes = so.reshape_data_tensors(Phi, left_legs, right_legs)
    PhiB_blocks, PhiB_shapes = so.reshape_data_tensors(PhiB, left_legs, right_legs)
    
    # Determine sectors
    L_arrow = Phi.arrows[left_legs] == 'i'
    L_sigma = Phi.leg_type[left_legs] == 's'
    
    L_sectors = np.sum(
        Phi.coordinates[left_legs] *
        (L_arrow[:, None] - 1 * (~L_arrow)[:, None]) *
        (alpha * L_sigma[:, None] + (~L_sigma[:, None])),
        axis=0
    )
    
    sectors = np.unique(L_sectors)
    n_sectors = len(sectors)
    coord_sectors = (L_sectors == sectors[:, None])
    
    block_coords, shape_sect, n_L, n_R, _ = so.construct_subblock_sectors(
        Phi, left_legs, right_legs, Phi_shapes, coord_sectors, n_sectors
    )
    
    # Create new tensors
    Lam1 = SymmetricLambda.empty(Phi.L, Phi.d, lam.chi_max, lam.chi_block)
    
    n_phys_L = np.sum(Phi.leg_type[left_legs] == 's')
    n_phys_R = np.sum(Phi.leg_type[right_legs] == 's')
    
    B1 = SymmetricTensor(
        Phi.L, Phi.d, n_phys_L,
        alpha=alpha,
        data_as_tensors=Phi.data_as_tensors,
        n_legs=len(left_legs) + 1,
        n_sectors=np.sum(n_L)
    )
    B2 = SymmetricTensor(
        Phi.L, Phi.d, n_phys_R,
        alpha=alpha,
        data_as_tensors=Phi.data_as_tensors,
        n_legs=len(right_legs) + 1,
        n_sectors=np.sum(n_R)
    )
    
    B1.arrows = np.array(list(Phi.arrows[left_legs]) + ['o'])
    B1.leg_type = np.array(list(Phi.leg_type[left_legs]) + ['v'])
    B2.arrows = np.array(['i'] + list(Phi.arrows[right_legs]))
    B2.leg_type = np.array(['v'] + list(Phi.leg_type[right_legs]))
    
    virtual_B1 = B1.leg_type == "v"
    virtual_B2 = B2.leg_type == "v"
    
    # Perform block SVDs
    mat_PhiB = np.empty(n_sectors, dtype=object)
    S_list = np.empty(n_sectors, dtype=object)
    Vh_list = np.empty(n_sectors, dtype=object)
    shape_L = np.empty(n_sectors, dtype=object)
    shape_R = np.empty(n_sectors, dtype=object)
    list_L = np.empty(n_sectors, dtype=object)
    list_R = np.empty(n_sectors, dtype=object)
    
    norm = 0.0
    for s in range(n_sectors):
        mat_Phi, list_L[s], list_R[s], shape_L[s], shape_R[s] = so.construct_matrix_from_subblocks(
            Phi_blocks, shape_sect[s], block_coords[s], coord_sectors[s]
        )
        mat_PhiB[s], _, _, _, _ = so.construct_matrix_from_subblocks(
            PhiB_blocks, shape_sect[s], block_coords[s], coord_sectors[s]
        )
        
        try:
            U, S_temp, Vh_temp = np.linalg.svd(mat_Phi, full_matrices=False)
        except np.linalg.LinAlgError:
            U, S_temp, Vh_temp = sla.svd(mat_Phi, full_matrices=False, lapack_driver='gesvd')
        
        # Threshold singular values
        mask = S_temp > mpo.th_sing_vals
        S_list[s] = S_temp[mask]
        Vh_list[s] = Vh_temp[mask, :]
        
        norm += np.sum(S_list[s] ** 2)
    
    norm = np.sqrt(norm)
    
    # Renormalize and threshold again
    total_states = 0
    largest_S = -1
    largest_sect = 0
    
    for s in range(n_sectors):
        S_temp = S_list[s] / norm
        mask = S_temp > mpo.th_sing_vals
        S_list[s] = S_temp[mask]
        Vh_list[s] = Vh_list[s][mask, :]
        
        if len(S_list[s]) > 0 and S_list[s][0] > largest_S:
            largest_S = S_list[s][0]
            largest_sect = s
        
        total_states += len(S_list[s])
    
    # Truncate
    S_list, Vh_list, truncated, discarded = _truncate(
        S_list, Vh_list, Lam1.chi_max, Lam1.chi_block,
        truncation_type, total_states, largest_sect
    )
    
    # Reconstruct B tensors
    valid_B1 = np.ones(np.sum(n_L), dtype=bool)
    valid_B2 = np.ones(np.sum(n_R), dtype=bool)
    idx_B1, idx_B2 = 0, 0
    
    for s in range(n_sectors):
        if S_list[s].size == 0:
            valid_B1[idx_B1:idx_B1 + n_L[s]] = False
            valid_B2[idx_B2:idx_B2 + n_R[s]] = False
        else:
            Lam1.data[sectors[s]] = S_list[s]
            
            # B2 coordinates and data
            B2.coordinates[:, idx_B2:idx_B2 + n_R[s]] = np.concatenate([
                np.tile([[sectors[s]]], (1, n_R[s])),
                Phi.coordinates[right_legs][:, coord_sectors[s]][:, list_R[s]]
            ], axis=0)
            
            # B1 coordinates and data
            B1.coordinates[:, idx_B1:idx_B1 + n_L[s]] = np.concatenate([
                Phi.coordinates[left_legs][:, coord_sectors[s]][:, list_L[s]],
                np.tile([[sectors[s]]], (1, n_L[s]))
            ], axis=0)
            
            # Compute B1 = PhiB @ Vh^dag (Hastings trick)
            mat_B1 = mat_PhiB[s] @ Vh_list[s].conj().T
            split_B1 = np.array_split(mat_B1, np.cumsum(shape_L[s][:-1]), axis=0)
            
            # Split Vh for B2
            split_Vh = np.array_split(Vh_list[s], np.cumsum(shape_R[s][:-1]), axis=1)
            
            # Set shapes
            B2.shapes[0, idx_B2:idx_B2 + n_R[s]] = S_list[s].size
            B2.shapes[-1, idx_B2:idx_B2 + n_R[s]] = shape_R[s]
            B1.shapes[0, idx_B1:idx_B1 + n_L[s]] = shape_L[s]
            B1.shapes[-1, idx_B1:idx_B1 + n_L[s]] = S_list[s].size
            
            if B1.data_as_tensors:
                rsh1 = B1.shapes[:, idx_B1:idx_B1 + n_L[s]]
                rsh2 = B2.shapes[:, idx_B2:idx_B2 + n_R[s]]
            else:
                rsh1 = B1.shapes[virtual_B1, idx_B1:idx_B1 + n_L[s]]
                rsh2 = B2.shapes[virtual_B2, idx_B2:idx_B2 + n_R[s]]
            
            B1.data[idx_B1:idx_B1 + n_L[s]] = [
                split_B1[i].astype(complex, copy=False).reshape(rsh1[:, i])
                for i in range(n_L[s])
            ]
            B2.data[idx_B2:idx_B2 + n_R[s]] = [
                split_Vh[i].astype(complex, copy=False).reshape(rsh2[:, i])
                for i in range(n_R[s])
            ]
        
        idx_B1 += n_L[s]
        idx_B2 += n_R[s]
    
    B1 = mask_coordinates(B1, valid_B1)
    B2 = mask_coordinates(B2, valid_B2)
    
    Lam1.set_sectors(np.array(list(Lam1.data.keys())))
    
    mpo.TN[f"B{site1}"] = B1
    mpo.TN[f"Lam{site2}"] = Lam1
    mpo.TN[f"B{site2}"] = B2
    
    if truncated:
        mpo = _check_truncation(mpo, site1, site2)
    
    return mpo, discarded


def _truncate(
    S: NDArray,
    Vh: NDArray,
    chi_max: int | None,
    chi_block: int,
    truncation_type: str,
    total_states: int,
    largest_sect: int
) -> tuple[NDArray, NDArray, bool, float]:
    """
    Truncate singular values according to the chosen strategy.
    """
    if chi_max is None:
        return S, Vh, False, 0.0
    
    truncated = False
    discarded = 0.0
    
    if truncation_type == "block_threshold":
        if S[largest_sect].size > chi_max:
            chi = min(len(S[largest_sect]) - 1, chi_max)
            threshold = S[largest_sect][chi]
            
            for s in range(len(S)):
                mask = S[s] > threshold
                discarded += np.sum(S[s][~mask] ** 2)
                S[s] = S[s][mask]
                Vh[s] = Vh[s][mask, :]
            truncated = True
            
    elif total_states > chi_max:
        if truncation_type == "global":
            # Keep chi_block per sector, then globally truncate the rest
            S_keep = [s[:chi_block] for s in S]
            S_trunc = [s[chi_block:] for s in S]
            
            n_trunc = sum(len(s) for s in S_trunc)
            if n_trunc > chi_max:
                # Find global threshold
                all_S = np.concatenate(S_trunc)
                if len(all_S) > chi_max:
                    threshold_idx = len(all_S) - chi_max
                    threshold = np.partition(all_S, threshold_idx)[threshold_idx]
                    
                    discarded = np.sum(all_S[all_S <= threshold] ** 2)
                    
                    for s in range(len(S)):
                        mask_keep = S_trunc[s] > threshold
                        S[s] = np.concatenate([S_keep[s], S_trunc[s][mask_keep]])
                        
                        # A sector may hold fewer than chi_block values:
                        # np.arange(chi_block) would then index Vh out of
                        # bounds. Offset by the actual number kept.
                        n_keep = len(S_keep[s])
                        idx_keep = np.concatenate([
                            np.arange(n_keep),
                            n_keep + np.where(mask_keep)[0]
                        ])
                        Vh[s] = Vh[s][idx_keep.astype(int), :]
                    
                    truncated = True
                    
        elif truncation_type == "block":
            for s in range(len(S)):
                if len(S[s]) > chi_max:
                    discarded += np.sum(S[s][chi_max:] ** 2)
                    S[s] = S[s][:chi_max]
                    Vh[s] = Vh[s][:chi_max, :]
                    # Flag only if something was actually cut, so that the
                    # sector-consistency sweep is not triggered needlessly.
                    truncated = True
    
    return S, Vh, truncated, discarded


def _check_truncation(
    mpo: 'SymmetricMPO',
    site1: int,
    site2: int
) -> 'SymmetricMPO':
    """
    Propagate sector changes after truncation.
    
    When truncation removes sectors, neighboring tensors and Lambda
    matrices must be updated consistently.
    """
    # Check left propagation
    l = site1
    lam = mpo.TN[f"Lam{l}"]
    B = mpo.TN[f"B{l}"]
    diff = np.setdiff1d(np.array(list(lam.data.keys())), B.leg_sectors[0])
    
    while len(diff) > 0 and l > 0:
        lam.set_sectors(B.leg_sectors[0])
        mask = np.ones(mpo.TN[f"B{l-1}"].n_sectors, dtype=bool)
        for key in diff:
            if key in lam.data:
                del lam.data[key]
            mask &= mpo.TN[f"B{l-1}"].coordinates[-1] != key
        mpo.TN[f"B{l-1}"] = mask_coordinates(mpo.TN[f"B{l-1}"], mask)
        l -= 1
        lam = mpo.TN[f"Lam{l}"]
        B = mpo.TN[f"B{l}"]
        diff = np.setdiff1d(np.array(list(lam.data.keys())), B.leg_sectors[0])
    
    # Check right propagation
    l = site2
    B = mpo.TN[f"B{l}"]
    lam_next = mpo.TN[f"Lam{l+1}"] if l + 1 < mpo.L else None
    
    if lam_next is not None:
        diff = np.setdiff1d(B.leg_sectors[-1], np.array(list(lam_next.data.keys())))
        
        while len(diff) > 0 and l < mpo.L - 1:
            lam_next.set_sectors(B.leg_sectors[-1])
            mask = np.ones(mpo.TN[f"B{l+1}"].n_sectors, dtype=bool)
            for key in diff:
                if key in lam_next.data:
                    del lam_next.data[key]
                mask &= mpo.TN[f"B{l+1}"].coordinates[0] != key
            mpo.TN[f"B{l+1}"] = mask_coordinates(mpo.TN[f"B{l+1}"], mask)
            l += 1
            B = mpo.TN[f"B{l}"]
            if l + 1 < mpo.L:
                lam_next = mpo.TN[f"Lam{l+1}"]
                diff = np.setdiff1d(B.leg_sectors[-1], np.array(list(lam_next.data.keys())))
            else:
                break
    
    return mpo