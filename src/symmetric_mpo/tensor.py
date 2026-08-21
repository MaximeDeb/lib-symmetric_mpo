"""
Symmetric tensor and MPO classes for quantum many-body systems.

This module provides the core data structures for symmetric Matrix Product
Operators (MPOs), including tensors with block-sparse structure imposed by
symmetry conservation.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Any
from dataclasses import dataclass, field

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# Note: sparse_ops imports are done locally where needed to avoid circular imports


@dataclass
class SymmetricLambda:
    """
    Diagonal matrix of singular values in Vidal's Gamma-Lambda representation.
    
    Each symmetry sector has its own set of singular values stored as
    a 1D array in a dictionary keyed by sector label.
    
    Attributes
    ----------
    L : int
        System size.
    d : int
        Local Hilbert space dimension.
    chi_max : int or None
        Maximum total bond dimension.
    chi_block : int
        Minimum states kept per block before global truncation.
    data : dict
        Singular values for each sector: {sector_label: ndarray}.
    arrows : ndarray
        Arrow directions for legs ('i' for in, 'o' for out).
    n_sectors : int
        Number of active symmetry sectors.
    left_sectors : ndarray
        Sector labels for the left leg.
    right_sectors : ndarray
        Sector labels for the right leg.
    """
    L: int
    d: int
    chi_max: int | None = None
    chi_block: int = 0
    data: dict = field(default_factory=dict)
    arrows: NDArray = field(default_factory=lambda: np.array(['i', 'o']))
    n_sectors: int = 0
    left_sectors: NDArray = field(default_factory=lambda: np.array([]))
    right_sectors: NDArray = field(default_factory=lambda: np.array([]))
    
    @classmethod
    def identity(
        cls,
        L: int,
        d: int,
        sectors: NDArray,
        chi_max: int | None = None,
        chi_block: int = 0
    ) -> 'SymmetricLambda':
        """Create a Lambda with unit singular values in each sector."""
        lam = cls(
            L=L, d=d, chi_max=chi_max, chi_block=chi_block,
            n_sectors=len(sectors),
            left_sectors=sectors.copy(),
            right_sectors=sectors.copy()
        )
        for s in sectors:
            lam.data[s] = np.ones(1, dtype=float)
        return lam
    
    @classmethod
    def empty(
        cls,
        L: int,
        d: int,
        chi_max: int | None = None,
        chi_block: int = 0
    ) -> 'SymmetricLambda':
        """Create an empty Lambda to be filled from SVD."""
        return cls(L=L, d=d, chi_max=chi_max, chi_block=chi_block)
    
    def set_sectors(self, sectors: NDArray) -> None:
        """Update the sector labels."""
        self.n_sectors = len(sectors)
        self.left_sectors = sectors
        self.right_sectors = sectors
        
    def copy(self) -> 'SymmetricLambda':
        """Create a deep copy."""
        new = SymmetricLambda(
            L=self.L, d=self.d, chi_max=self.chi_max, chi_block=self.chi_block,
            arrows=self.arrows.copy(),
            n_sectors=self.n_sectors,
            left_sectors=self.left_sectors.copy(),
            right_sectors=self.right_sectors.copy()
        )
        new.data = {k: v.copy() for k, v in self.data.items()}
        return new


class SymmetricTensor:
    """
    Block-sparse tensor with U(1) symmetry conservation.
    
    The tensor is stored in a sparse format where only non-zero symmetry
    blocks are kept. Each block is identified by coordinates specifying
    the symmetry sector of each leg.
    
    Attributes
    ----------
    L : int
        System size.
    d : int
        Local dimension.
    phys_dims : int
        Number of physical (sigma, sigma') leg pairs.
    n_legs : int
        Total number of tensor legs.
    n_sectors : int
        Number of non-zero blocks.
    alpha : int
        Super-charge parameter (-1 or L+1).
    is_symmetric : bool
        Whether the tensor has additional symmetry.
    data_as_tensors : bool
        If True, data stored as tensors; if False, as matrices.
    coordinates : ndarray, shape (n_legs, n_sectors)
        Symmetry sector labels for each block.
    data : ndarray of object
        The actual tensor data for each block.
    shapes : ndarray, shape (n_legs, n_sectors)
        Dimensions of each leg for each block.
    leg_sectors : ndarray of object
        Available sector labels for each leg.
    arrows : ndarray
        Arrow directions ('i' or 'o') for each leg.
    leg_type : ndarray
        Type of each leg: 'v' (virtual), 's' (sigma), 'p' (sigma').
    """
    
    def __init__(
        self,
        L: int,
        d: int,
        phys_dims: int,
        *,
        left_sectors: NDArray | None = None,
        right_sectors: NDArray | None = None,
        initial: str | None = None,
        alpha: int = -1,
        is_symmetric: bool = False,
        data_as_tensors: bool = True,
        # For creating from existing data
        n_legs: int | None = None,
        n_sectors: int | None = None
    ):
        """
        Initialize a symmetric tensor.
        
        Parameters
        ----------
        L : int
            System size.
        d : int
            Local Hilbert space dimension.
        phys_dims : int
            Number of physical dimension pairs (sigma, sigma').
        left_sectors : ndarray, optional
            Non-zero sector dimensions for left virtual leg.
        right_sectors : ndarray, optional
            Non-zero sector dimensions for right virtual leg.
        initial : str, optional
            Initialization type: "Id" for identity, "Sx" for sigma-x, etc.
        alpha : int
            Super-charge: -1 for particle-hole, L+1 for particle number.
        is_symmetric : bool
            Whether tensor has additional symmetries.
        data_as_tensors : bool
            Storage format for data blocks.
        n_legs : int, optional
            Number of legs (for pre-allocation).
        n_sectors : int, optional
            Number of sectors (for pre-allocation).
        """
        self.L = L
        self.d = d
        self.phys_dims = phys_dims
        self.alpha = alpha
        self.is_symmetric = is_symmetric
        self.data_as_tensors = data_as_tensors
        
        # Pre-allocated tensor (from other operations)
        if n_legs is not None and n_sectors is not None:
            self._init_empty(n_legs, n_sectors)
            return
            
        # Standard initialization
        if initial is not None:
            self._init_from_type(left_sectors, right_sectors, initial)
    
    def _init_empty(self, n_legs: int, n_sectors: int) -> None:
        """Initialize empty arrays for a tensor with known dimensions."""
        self.n_legs = n_legs
        self.n_sectors = n_sectors
        self.leg_type = np.zeros(n_legs, dtype='U1')
        self.coordinates = np.zeros((n_legs, n_sectors), dtype=np.intp)
        self.data = np.empty(n_sectors, dtype=object)
        self.shapes = np.ones((n_legs, n_sectors), dtype=np.intp)
        self.arrows = np.empty(n_legs, dtype='U1')
        # leg_sectors is computed lazily from coordinates (see property below).
        self._leg_sectors = None

    @property
    def leg_sectors(self) -> NDArray:
        """
        Available sector labels for each leg (lazy).

        Recomputed from `coordinates` on demand whenever the cached value
        has been invalidated (e.g. by `mask_coordinates` or coordinate
        edits). Callers may still assign it directly when they know the
        sectors (e.g. boundary PacMan tensors).
        """
        if self._leg_sectors is None:
            ls = np.empty(self.n_legs, dtype=object)
            for i in range(self.n_legs):
                ls[i] = np.unique(self.coordinates[i, :])
            self._leg_sectors = ls
        return self._leg_sectors

    @leg_sectors.setter
    def leg_sectors(self, value) -> None:
        self._leg_sectors = value

    def invalidate_leg_sectors(self) -> None:
        """Mark cached leg_sectors as stale (recomputed on next access)."""
        self._leg_sectors = None
    
    def _init_from_type(
        self,
        a_L: NDArray,
        a_R: NDArray,
        initial: str
    ) -> None:
        """Initialize tensor from sector dimensions and type."""
        L, d, phys_dims, alpha = self.L, self.d, self.phys_dims, self.alpha
        
        self.n_legs = 2 + 2 * phys_dims
        self.leg_type = np.zeros(self.n_legs, dtype='U1')
        self.leg_type[[0, -1]] = 'v'
        self.leg_type[1:phys_dims + 1] = 's'
        self.leg_type[phys_dims + 1:2 * phys_dims + 1] = 'p'
        
        self.arrows = np.array(
            ['i'] + ['i'] * (2 * phys_dims) + ['o'],
            dtype='U1'
        )
        
        # Find non-zero sectors
        L_sect = np.where(a_L)
        R_sect = np.where(a_R)
        
        # Build connectivity based on alpha
        loc_sig = np.arange(d)
        
        if alpha == L + 1:
            allowed_combinations = self._build_connections_particle_number(
                L_sect, R_sect, loc_sig, initial
            )
        elif alpha == -1:
            allowed_combinations = self._build_connections_particle_hole(
                L_sect, R_sect, loc_sig, initial
            )
        else:
            raise ValueError(f"Unsupported alpha value: {alpha}")
        
        self._populate_from_connections(
            allowed_combinations, L_sect, R_sect, a_L, a_R, initial
        )
    
    def _build_connections_particle_number(
        self,
        L_sect: tuple,
        R_sect: tuple,
        loc_sig: NDArray,
        initial: str
    ) -> NDArray[np.bool_]:
        """Build allowed sector connections for particle number conservation."""
        phys_dims = self.phys_dims
        
        # l_R = l_L + sum(sigma), lp_R = lp_L + sum(sigma')
        conn_l = L_sect[0][(...,) + (None,) * phys_dims]
        conn_lp = L_sect[1][(...,) + (None,) * phys_dims]
        
        for i in range(phys_dims):
            sig_shape = (None,) + (None,) * i + (...,) + (None,) * (phys_dims - i - 1)
            sig_i = loc_sig[sig_shape]
            conn_l = conn_l + sig_i
            conn_lp = conn_lp + sig_i
        
        # Check if connected sectors are allowed
        mask = np.isin(conn_l, R_sect[0]).reshape(conn_l.shape)
        maskp = np.isin(conn_lp, R_sect[1]).reshape(conn_lp.shape)
        
        # Combine masks
        # (Index must be built by tuple concatenation: `mask[..., (None,)*k]`
        # nests a tuple inside the index and raises IndexError.)
        allowed = (
            mask[(...,) + (None,) * phys_dims] * 
            maskp[(slice(None),) + (None,) * phys_dims + (...,)]
        )
        
        # For identity-like operators, require l = l'
        if initial in ("Id",) or (initial and initial[0] == "S"):
            allowed = allowed * (
                conn_l[(...,) + (None,) * phys_dims] == 
                conn_lp[(slice(None),) + (None,) * phys_dims + (...,)]
            )
        
        return allowed
    
    def _build_connections_particle_hole(
        self,
        L_sect: tuple,
        R_sect: tuple,
        loc_sig: NDArray,
        initial: str
    ) -> NDArray[np.bool_]:
        """Build allowed sector connections for particle-hole symmetry."""
        phys_dims = self.phys_dims
        
        # l_R - l'_R = l_L - l'_L + sum(sigma) - sum(sigma')
        conn = L_sect[0][(...,) + (None,) * (2 * phys_dims)]
        
        for i in range(phys_dims):
            sig_shape = (None,) + (None,) * i + (...,) + (None,) * (2 * phys_dims - i - 1)
            sigp_shape = (None,) + (None,) * (phys_dims + i) + (...,) + (None,) * (phys_dims - i - 1)
            conn = conn + loc_sig[sig_shape] - loc_sig[sigp_shape]
        
        allowed = np.isin(conn, R_sect[0]).reshape(conn.shape)
        
        return allowed
    
    def _populate_from_connections(
        self,
        allowed: NDArray[np.bool_],
        L_sect: tuple,
        R_sect: tuple,
        a_L: NDArray,
        a_R: NDArray,
        initial: str
    ) -> None:
        """Populate tensor data from allowed connections."""
        L, d, phys_dims, alpha = self.L, self.d, self.phys_dims, self.alpha
        
        mask_conn = np.array(np.where(allowed))
        
        self.n_sectors = mask_conn.shape[1]
        self.coordinates = np.zeros((2 * phys_dims + 2, self.n_sectors), dtype=np.intp)
        self.data = np.empty(self.n_sectors, dtype=object)
        self.shapes = np.ones((self.n_legs, self.n_sectors), dtype=np.intp)
        
        for i in range(self.n_sectors):
            loc_sig = tuple(mask_conn[1 + j, i] for j in range(phys_dims))
            loc_sigp = tuple(mask_conn[1 + phys_dims + j, i] for j in range(phys_dims))
            L_idx = mask_conn[0, i]
            
            if alpha == L + 1:
                lL, lpL = L_sect[0][L_idx], L_sect[1][L_idx]
                lR = lL + sum(loc_sig)
                lpR = lpL + sum(loc_sigp)
                self.coordinates[0, i] = lL * (L + 1) + lpL
                self.coordinates[-1, i] = lR * (L + 1) + lpR
            elif alpha == -1:
                l_min_lp_L = L_sect[0][L_idx]
                l_min_lp_R = l_min_lp_L + sum(loc_sig) - sum(loc_sigp)
                self.coordinates[0, i] = l_min_lp_L
                self.coordinates[-1, i] = l_min_lp_R
            
            self.coordinates[1:phys_dims + 1, i] = loc_sig
            self.coordinates[phys_dims + 1:-1, i] = loc_sigp
            
            if initial in ("Id",) or (initial and initial[0] == "S"):
                shape = (1,) * self.n_legs if self.data_as_tensors else (1, 1)
                self.data[i] = np.ones(shape, dtype=complex)
        
        # Set leg sectors
        self.leg_sectors = np.array([
            np.unique(self.coordinates[0, :]),
            *([np.array([0, 1])] * 2 * phys_dims),
            np.unique(self.coordinates[-1, :])
        ], dtype=object)
    
    def copy(self, copy_data: bool = True) -> 'SymmetricTensor':
        """
        Create a copy of the tensor.

        Parameters
        ----------
        copy_data : bool
            If True (default), deep-copy the data blocks. If False, the
            data blocks themselves are shared with the original (the object
            array holding them is still copied). This is safe as long as
            blocks are only *replaced* (``t.data[i] = ...``), never mutated
            in place -- which is the convention throughout this library.
        """
        B = SymmetricTensor(
            self.L, self.d, self.phys_dims,
            alpha=self.alpha,
            data_as_tensors=self.data_as_tensors,
            n_legs=self.n_legs,
            n_sectors=self.n_sectors
        )
        B.coordinates = self.coordinates.copy()
        B.data = self.data.copy()
        if copy_data:
            for i in range(self.n_sectors):
                if self.data[i] is not None:
                    B.data[i] = self.data[i].copy()
        B.shapes = self.shapes.copy()
        if self._leg_sectors is None:
            B._leg_sectors = None
        else:
            # Deep-copy the inner arrays: the old shallow copy shared them,
            # which was fragile (any in-place edit would alias).
            ls = np.empty(self.n_legs, dtype=object)
            for i in range(self.n_legs):
                ls[i] = self._leg_sectors[i].copy()
            B._leg_sectors = ls
        B.arrows = self.arrows.copy()
        B.leg_type = self.leg_type.copy()
        return B


def swap_indices(
    tensor: SymmetricTensor,
    old_indices: list[int] | NDArray[np.intp],
    new_indices: list[int] | NDArray[np.intp]
) -> SymmetricTensor:
    """
    Permute tensor leg indices.
    
    Parameters
    ----------
    tensor : SymmetricTensor
        The tensor to modify (in-place).
    old_indices : array-like
        Current positions of legs.
    new_indices : array-like
        New positions for legs.
        
    Returns
    -------
    SymmetricTensor
        The modified tensor (same object).
    """
    old_indices = np.asarray(old_indices)
    new_indices = np.asarray(new_indices)
    
    tensor.coordinates[new_indices, :] = tensor.coordinates[old_indices, :]
    tensor.shapes[new_indices, :] = tensor.shapes[old_indices, :]
    tensor.arrows[new_indices] = tensor.arrows[old_indices]
    tensor.leg_type[new_indices] = tensor.leg_type[old_indices]
    if tensor._leg_sectors is not None:
        tensor._leg_sectors[new_indices] = tensor._leg_sectors[old_indices]
    
    if tensor.data_as_tensors:
        # Precompute the transpose permutation once instead of paying
        # np.moveaxis's axis-normalization overhead for every block.
        # moveaxis(x, old, new) == transpose(perm) with perm[new] = old
        # and the remaining axes filling the gaps in original order.
        perm = np.empty(tensor.n_legs, dtype=np.intp)
        perm[new_indices] = old_indices
        rest_dst = np.setdiff1d(np.arange(tensor.n_legs), new_indices)
        rest_src = np.setdiff1d(np.arange(tensor.n_legs), old_indices)
        perm[rest_dst] = rest_src
        perm_t = tuple(int(p) for p in perm)
        if perm_t != tuple(range(tensor.n_legs)):
            for b in range(tensor.n_sectors):
                if tensor.data[b] is not None:
                    tensor.data[b] = tensor.data[b].transpose(perm_t)
    
    return tensor


def mask_coordinates(
    tensor: SymmetricTensor,
    mask: NDArray[np.bool_]
) -> SymmetricTensor:
    """
    Filter tensor blocks by a boolean mask.
    
    Parameters
    ----------
    tensor : SymmetricTensor
        The tensor to filter (modified in-place).
    mask : ndarray of bool
        Which blocks to keep.
        
    Returns
    -------
    SymmetricTensor
        The filtered tensor (same object).
    """
    tensor.coordinates = tensor.coordinates[:, mask]
    tensor.data = tensor.data[mask]
    tensor.shapes = tensor.shapes[:, mask]
    tensor.n_sectors = np.sum(mask)
    # Lazy: recomputed from coordinates on next access.
    tensor.invalidate_leg_sectors()
    return tensor