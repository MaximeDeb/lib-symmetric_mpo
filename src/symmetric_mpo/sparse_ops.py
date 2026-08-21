"""
Sparse block operations for symmetric tensor networks.

This module provides utilities for manipulating block-sparse matrices
that arise from symmetry-preserving tensor network operations.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any


def block_split(
    array: NDArray, 
    cuts_rows: NDArray[np.intp], 
    cuts_cols: NDArray[np.intp]
) -> list[NDArray]:
    """
    Split a matrix into sub-matrices according to row and column cuts.
    
    Parameters
    ----------
    array : ndarray
        The matrix to split.
    cuts_rows : ndarray
        Indices where to split along rows.
    cuts_cols : ndarray
        Indices where to split along columns.
        
    Returns
    -------
    list of ndarray
        Flattened list of sub-matrices in row-major order.
    """
    split_rows = np.split(array, cuts_rows, axis=0)
    sub_matrices = []
    for row_block in split_rows:
        sub_matrices.extend(np.split(row_block, cuts_cols, axis=1))
    return sub_matrices


def reshape_data_tensors(
    tensor: Any, 
    left_legs: NDArray[np.intp], 
    right_legs: NDArray[np.intp],
    return_shapes: bool = True
) -> tuple[NDArray, NDArray]:
    """
    Reshape tensor data blocks as matrices for contraction.
    
    Reorganizes tensor data so that legs in `left_legs` form the row indices
    and legs in `right_legs` form the column indices.
    
    Parameters
    ----------
    tensor : SymmetricTensor
        The tensor to reshape.
    left_legs : ndarray
        Indices of legs to place on the left (rows).
    right_legs : ndarray
        Indices of legs to place on the right (columns).
        
    Returns
    -------
    matrices : ndarray of object
        Array of reshaped matrix blocks.
    shapes : ndarray
        Shape information for the blocks (2, n_sectors).
    """
    n_legs = np.arange(tensor.n_legs)
    matrices = np.empty(tensor.n_sectors, dtype=object)
    
    if tensor.data_as_tensors:
        # Each tensor block is reshaped as a matrix. The permutation that
        # brings the left legs to the front is the same for every block:
        # compute it once (np.moveaxis re-normalizes axes on every call,
        # which dominated the numeric phase at small chi).
        n_left = len(left_legs)
        perm = tuple(int(p) for p in np.concatenate(
            [left_legs, np.setdiff1d(n_legs, left_legs)]
        ))
        identity_perm = perm == tuple(range(tensor.n_legs))
        right_dims = np.prod(tensor.shapes[right_legs, :], axis=0) \
            if len(right_legs) > 0 else np.ones(tensor.n_sectors, dtype=np.intp)
        for i, data in enumerate(tensor.data):
            if not identity_perm:
                data = data.transpose(perm)
            matrices[i] = data.reshape(-1, right_dims[i])
        if not return_shapes:
            return matrices, None
        shapes = np.array([matrices[i].shape for i in range(tensor.n_sectors)]).T
    else:
        # Data already stored as matrices
        mask_virtual = tensor.leg_type == "v"
        n_virtual_left = np.sum(tensor.leg_type[left_legs] == 'v')
        
        if n_virtual_left == 1:
            # It's a proper matrix, check orientation
            mask_L = np.isin(n_legs, left_legs)
            mask_R = np.isin(n_legs, right_legs)
            left_virtual = n_legs[mask_L & mask_virtual]
            right_virtual = n_legs[mask_R & mask_virtual]
            
            if left_virtual[0] < right_virtual[0]:
                # Already stored correctly
                matrices = tensor.data
                shapes = tensor.shapes[mask_virtual]
            else:
                # Need transpose
                matrices[:] = [mat.T for mat in tensor.data]
                shapes = tensor.shapes[mask_virtual][::-1, :]
        else:
            # Both virtual legs on the same side - vectorize
            shapes = np.ones((2, tensor.n_sectors), dtype=np.uint32)
            if n_virtual_left != 0:
                matrices[:] = [mat.reshape(-1, 1) for mat in tensor.data]
                shapes[0] = np.prod(tensor.shapes[mask_virtual], axis=0)
            else:
                matrices[:] = [mat.reshape(1, -1) for mat in tensor.data]
                shapes[1] = np.prod(tensor.shapes[mask_virtual], axis=0)
                
    return matrices, shapes


def construct_subblock_sectors(
    tensor: Any,
    left_legs: NDArray[np.intp],
    right_legs: NDArray[np.intp],
    block_shapes: NDArray,
    coord_sectors: NDArray[np.bool_],
    n_sectors: int
) -> tuple[NDArray, NDArray, NDArray[np.intp], NDArray[np.intp], NDArray]:
    """
    Construct block coordinate mapping for matrix operations.
    
    For each symmetry sector, determines which tensor blocks contribute
    and where they appear in the assembled matrix.
    
    Parameters
    ----------
    tensor : SymmetricTensor
        The tensor being processed.
    left_legs : ndarray
        Indices of left (row) legs.
    right_legs : ndarray
        Indices of right (column) legs.
    block_shapes : ndarray
        Shapes of the tensor blocks.
    coord_sectors : ndarray of bool
        Mask indicating which coordinates belong to each sector.
    n_sectors : int
        Number of symmetry sectors.
        
    Returns
    -------
    block_coords : ndarray of object
        Block coordinate matrices for each sector.
    mat_shapes_sect : ndarray of object
        Shape arrays for each sector.
    n_left_sect : ndarray
        Number of left blocks per sector.
    n_right_sect : ndarray
        Number of right blocks per sector.
    blocks_sect : ndarray
        Block identification arrays.
    """
    block_coords = np.empty(n_sectors, dtype=object)
    blocks_sect = np.empty((n_sectors, 2), dtype=object)
    mat_shapes_sect = np.empty(n_sectors, dtype=object)
    n_left_sect = np.zeros(n_sectors, dtype=np.intp)
    n_right_sect = np.zeros(n_sectors, dtype=np.intp)

    # Compute unique identifiers for sub-blocks
    left_indices = np.arange(len(left_legs), dtype=np.intp)
    right_indices = np.arange(len(right_legs), dtype=np.intp)
    base = (tensor.L + 1) ** 2
    pow_left = np.power(base, left_indices)
    pow_right = np.power(base, right_indices)
    
    for s in range(n_sectors):
        sector_mask = coord_sectors[s]
        
        # Left blocks
        left_coords = tensor.coordinates[left_legs][:, sector_mask]
        left_ids = np.sum(pow_left[:, None] * left_coords, axis=0)
        left_unique = np.unique(left_ids)
        left_block_mask = (left_ids[None, :] == left_unique[:, None])
        
        # Right blocks
        right_coords = tensor.coordinates[right_legs][:, sector_mask]
        right_ids = np.sum(pow_right[:, None] * right_coords, axis=0)
        right_unique = np.unique(right_ids)
        right_block_mask = (right_ids[None, :] == right_unique[:, None])
        
        blocks_sect[s, 0] = left_unique
        blocks_sect[s, 1] = right_unique
        
        n_left = len(left_unique)
        n_right = len(right_unique)
        n_left_sect[s] = n_left
        n_right_sect[s] = n_right
        
        # Block coordinate mapping (-1 for empty blocks)
        block_coords[s] = np.full((n_left, n_right), -1, dtype=np.intp)
        
        # Find which tensor blocks go where
        lr_where = np.where(left_block_mask[:, None, :] & right_block_mask[None, :, :])
        block_coords[s][lr_where[0], lr_where[1]] = lr_where[2]
        
        # Block shapes
        mat_shapes_sect[s] = np.zeros((2, n_left, n_right), dtype=np.intp)
        mat_shapes_sect[s][:, lr_where[0], lr_where[1]] = block_shapes[:, sector_mask][:, lr_where[2]]
        mat_shapes_sect[s][0] = mat_shapes_sect[s][0].max(axis=1)[:, None]
        mat_shapes_sect[s][1] = mat_shapes_sect[s][1].max(axis=0)[None, :]
        
    return block_coords, mat_shapes_sect, n_left_sect, n_right_sect, blocks_sect


def construct_matrix_from_subblocks(
    block_matrices: NDArray,
    mat_shapes: NDArray,
    block_coords: NDArray[np.intp],
    sector_mask: NDArray[np.bool_]
) -> tuple[NDArray, list, list, NDArray, NDArray]:
    """
    Assemble a full matrix from sparse blocks.
    
    Parameters
    ----------
    block_matrices : ndarray of object
        The individual matrix blocks.
    mat_shapes : ndarray
        Shape information for the blocks (2, n_left, n_right).
    block_coords : ndarray
        Coordinate mapping for blocks.
    sector_mask : ndarray of bool
        Mask for selecting blocks in this sector.
        
    Returns
    -------
    matrix : ndarray
        The assembled matrix.
    block_list_left : list
        Indices of left blocks used.
    block_list_right : list
        Indices of right blocks used.
    block_shape_left : ndarray
        Row sizes of blocks.
    block_shape_right : ndarray
        Column sizes of blocks.
    """
    block_shape_left = mat_shapes[0, :, 0].copy()
    block_shape_right = mat_shapes[1, 0, :].copy()

    # Preallocate once instead of building rows of temporaries with
    # np.zeros + np.concatenate (empty blocks are implicit zeros).
    row_off = np.concatenate(([0], np.cumsum(block_shape_left)))
    col_off = np.concatenate(([0], np.cumsum(block_shape_right)))
    matrix = np.zeros((row_off[-1], col_off[-1]), dtype=complex)

    blocks = block_matrices[sector_mask]

    block_list_left = []
    for i, row_coords in enumerate(block_coords):
        # Find first non-empty block in this row
        valid_blocks = row_coords[row_coords != -1]
        if len(valid_blocks) > 0:
            block_list_left.append(valid_blocks[0])
        for j, block_idx in enumerate(row_coords):
            if block_idx != -1:
                matrix[row_off[i]:row_off[i + 1],
                       col_off[j]:col_off[j + 1]] = blocks[block_idx]

    # Right block list
    block_list_right = []
    for col in block_coords.T:
        valid = col[col != -1]
        if len(valid) > 0:
            block_list_right.append(valid[0])

    return matrix, block_list_left, block_list_right, block_shape_left, block_shape_right


def check_subblock_sectors(
    left_block_coords: NDArray,
    right_block_coords: NDArray,
    left_shapes: NDArray,
    right_shapes: NDArray,
    id_blocks_left: NDArray,
    id_blocks_right: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray[np.bool_]]:
    """
    Verify and align block sectors between two tensors for contraction.
    
    Ensures that only matching sub-sectors are kept for valid matrix
    multiplication.
    
    Parameters
    ----------
    left_block_coords : ndarray
        Block coordinates from left tensor.
    right_block_coords : ndarray
        Block coordinates from right tensor.
    left_shapes : ndarray
        Shape arrays for left tensor blocks.
    right_shapes : ndarray
        Shape arrays for right tensor blocks.
    id_blocks_left : ndarray
        Block identifiers for left tensor.
    id_blocks_right : ndarray
        Block identifiers for right tensor.
        
    Returns
    -------
    left_block_coords : ndarray
        Filtered left coordinates.
    right_block_coords : ndarray
        Filtered right coordinates.
    left_shapes : ndarray
        Filtered left shapes.
    right_shapes : ndarray
        Filtered right shapes.
    empty_sectors : ndarray of bool
        Mask indicating which sectors have no valid blocks.
    """
    n_sectors = len(left_block_coords)
    empty_sectors = np.zeros(n_sectors, dtype=bool)
    
    for s in range(n_sectors):
        mid_left = id_blocks_left[s, 1]
        mid_right = id_blocks_right[s, 0]
        
        common_ids, left_idx, right_idx = np.intersect1d(
            mid_left, mid_right, return_indices=True
        )
        
        if len(common_ids) == 0:
            empty_sectors[s] = True
        
        left_block_coords[s] = left_block_coords[s][:, left_idx]
        right_block_coords[s] = right_block_coords[s][right_idx, :]
        left_shapes[s] = left_shapes[s][:, :, left_idx]
        right_shapes[s] = right_shapes[s][:, right_idx, :]
        
    return left_block_coords, right_block_coords, left_shapes, right_shapes, empty_sectors