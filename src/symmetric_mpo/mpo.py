"""
Symmetric Matrix Product Operators (MPOs) and quantum gates.

This module provides the main MPO class for representing operators in
the symmetric tensor network formalism, along with Trotter gates for
time evolution.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import scipy.linalg as sla
from typing import Literal, Any

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from .tensor import SymmetricTensor, SymmetricLambda, mask_coordinates


class SymmetricGate:
    """
    Two-site gate for TEBD time evolution.
    
    Represents a unitary or operator acting on two adjacent sites,
    stored in the same block-sparse format as MPO tensors.
    
    Attributes
    ----------
    phys_dims : int
        Number of physical dimension pairs.
    n_legs : int
        Number of tensor legs (2 * phys_dims).
    n_sectors : int
        Number of non-zero blocks.
    alpha : int
        Super-charge parameter.
    data_as_tensors : bool
        Storage format for blocks.
    L, d : int
        System parameters.
    coordinates : ndarray
        Symmetry sector labels for each block.
    data : ndarray of object
        Gate elements for each block.
    shapes : ndarray
        Block dimensions.
    arrows, leg_type, leg_sectors : ndarray
        Leg metadata.
    """
    
    def __init__(
        self,
        phys_dims: int,
        dt: float,
        params: dict[str, Any],
        *,
        gate_type: str = "Hamiltonian",
        model: str = "Heis_nn",
        dag: bool = False,
        alpha: int = -1,
        data_as_tensors: bool = True,
        step: str | None = None,
        add_terms: NDArray | None = None,
        rot: NDArray | None = None
    ):
        """
        Create a two-site quantum gate.
        
        Parameters
        ----------
        phys_dims : int
            Total physical dimensions (should be 2 for two-site gate).
        dt : float
            Time step for Hamiltonian evolution.
        params : dict
            Model parameters including 'L', 'd', and model-specific couplings.
        gate_type : str
            Type of gate: "Hamiltonian" or "Swap".
        model : str
            Physical model: "Heis_nn", "IRLM", or "givens".
        dag : bool
            If True, create the Hermitian conjugate gate.
        alpha : int
            Super-charge parameter.
        data_as_tensors : bool
            Storage format for blocks.
        step : str, optional
            Hamiltonian step identifier (e.g., "H0", "H1").
        add_terms : ndarray, optional
            Additional terms to add to the Hamiltonian.
        rot : ndarray, optional
            Rotation matrix for "givens" model.
        """
        self.phys_dims = phys_dims
        self.arrows = np.array(['i'] * phys_dims + ['o'] * phys_dims, dtype='U1')
        self.n_legs = 2 * phys_dims
        self.leg_type = np.array(['s'] * self.n_legs, dtype='U1')
        self.leg_sectors = np.array([np.array([0, 1])] * self.n_legs, dtype=object)
        self.alpha = alpha
        self.data_as_tensors = data_as_tensors
        self.L = params["L"]
        self.d = params["d"]
        
        if gate_type == "Hamiltonian":
            self._init_hamiltonian_gate(dt, params, model, dag, step, add_terms, rot)
        elif gate_type == "Swap":
            self._init_swap_gate()
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
    
    def _init_hamiltonian_gate(
        self,
        dt: float,
        params: dict,
        model: str,
        dag: bool,
        step: str | None,
        add_terms: NDArray | None,
        rot: NDArray | None
    ) -> None:
        """Initialize a Hamiltonian evolution gate."""
        d = self.d
        
        # Pauli operators
        Sp = np.array([[0, 0], [1, 0]], dtype=complex)
        Sz = np.array([[1, 0], [0, -1]], dtype=complex) * 0.5
        
        # Build two-site Hamiltonian
        if model == "Heis_nn":
            J = params.get('J', 1.0)
            Jz = params.get('Jz', 1.0)
            hi = J / 2 * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp)) + Jz * np.kron(Sz, Sz)
            
            # Floquet driving if present
            period_T = params.get('periodT', 0)
            if period_T != 0:
                t = params.get('t', 0)
                h_mean = params.get('hmean', 0)
                h_drive = params.get('hdrive', 0)
                h_field = h_mean + h_drive * np.cos(2 * np.pi * t / period_T)
                hi += h_field * (np.kron(np.eye(d), Sz) + np.kron(Sz, np.eye(d)))
                
        elif model == "IRLM":
            U_int = params.get('Uint', 0)
            V = params.get('V', 0)
            gamma = params.get('gamma', 0)
            
            if step == "H0":
                # Impurity-bath coupling
                hi = V * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp)) + U_int * np.kron(Sz, Sz)
            else:
                # Bath hopping
                hi = gamma * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp))
                
        elif model == "givens":
            hi = np.eye(4, dtype=complex)
            if rot is not None:
                hi[1:-1, 1:-1] = rot
        else:
            raise ValueError(f"Unknown model: {model}")
        
        if add_terms is not None:
            hi += add_terms
        
        # Exponentiate
        if model == "givens":
            U = hi  # Already a unitary for Givens
        else:
            U = sla.expm(-1j * hi * dt)
        
        # Store in block-sparse format
        self.n_sectors = 6
        self.coordinates = np.zeros((4, 6), dtype=np.intp)
        self.data = np.empty(6, dtype=object)
        self.shapes = np.ones((4, 6), dtype=np.intp)
        
        # Non-zero blocks: (sigma1, sigma2, sigma1', sigma2')
        #
        # NOTE (convention / latent transpose ambiguity):
        # With basis index b = 2*sigma1 + sigma2, the element assigned to
        # coordinates (s1, s2, s1', s2') is U[2*s1+s2, 2*s1'+s2'] for the
        # off-diagonal blocks below, i.e. rows are unprimed (output) and
        # columns are primed (input): the gate acts as U |s1 s2><s1' s2'|.
        # For dag=True the CONJUGATE is stored at the SAME coordinates
        # with leg_type 'p', so together with the crossed contraction in
        # apply_gate this represents U^dagger.
        # All Hamiltonians currently implemented (Heis_nn, IRLM, Givens
        # with real couplings) yield a SYMMETRIC U (U[1,2] == U[2,1]), so
        # any accidental row/column swap would be invisible here. If a
        # model with complex hopping (non-symmetric U) is added, verify
        # this convention against a dense reference before trusting
        # results.
        self.coordinates[:, 0] = [0, 0, 0, 0]
        self.coordinates[:, 1] = [0, 1, 0, 1]
        self.coordinates[:, 2] = [1, 0, 0, 1]
        self.coordinates[:, 3] = [0, 1, 1, 0]
        self.coordinates[:, 4] = [1, 0, 1, 0]
        self.coordinates[:, 5] = [1, 1, 1, 1]
        
        n_legs = 4 if self.data_as_tensors else 2
        shape = (1,) * n_legs
        
        if not dag:
            self.data[0] = np.ones(shape, dtype=complex) * U[0, 0]
            self.data[1] = np.ones(shape, dtype=complex) * U[1, 1]
            self.data[2] = np.ones(shape, dtype=complex) * U[1, 2]
            self.data[3] = np.ones(shape, dtype=complex) * U[2, 1]
            self.data[4] = np.ones(shape, dtype=complex) * U[2, 2]
            self.data[5] = np.ones(shape, dtype=complex) * U[3, 3]
        else:
            self.leg_type[:] = 'p'
            self.data[0] = np.ones(shape, dtype=complex) * U[0, 0].conj()
            self.data[1] = np.ones(shape, dtype=complex) * U[1, 1].conj()
            self.data[2] = np.ones(shape, dtype=complex) * U[1, 2].conj()
            self.data[3] = np.ones(shape, dtype=complex) * U[2, 1].conj()
            self.data[4] = np.ones(shape, dtype=complex) * U[2, 2].conj()
            self.data[5] = np.ones(shape, dtype=complex) * U[3, 3].conj()
    
    def _init_swap_gate(self) -> None:
        """Initialize a SWAP gate."""
        self.n_sectors = 4
        self.coordinates = np.zeros((4, 4), dtype=np.intp)
        self.data = np.empty(4, dtype=object)
        self.shapes = np.ones((4, 4), dtype=np.intp)
        
        self.coordinates[:, 0] = [0, 0, 0, 0]
        self.coordinates[:, 1] = [1, 0, 0, 1]
        self.coordinates[:, 2] = [0, 1, 1, 0]
        self.coordinates[:, 3] = [1, 1, 1, 1]
        
        n_legs = 4 if self.data_as_tensors else 2
        shape = (1,) * n_legs
        
        for i in range(4):
            self.data[i] = np.ones(shape, dtype=complex)


class SymmetricMPO:
    """
    Symmetric Matrix Product Operator in Gamma-Lambda (Vidal) representation.
    
    Represents an operator acting on L sites, decomposed into local tensors
    B_i connected by diagonal singular value matrices Lambda_i. The structure
    exploits U(1) symmetry conservation for efficient storage and manipulation.
    
    Attributes
    ----------
    L : int
        System size (number of sites).
    d : int
        Local Hilbert space dimension.
    q_alpha : int
        Global symmetry sector of the operator.
    chi_max : int
        Maximum bond dimension.
    chi_block : int
        Minimum states kept per symmetry block.
    alpha : int
        Super-charge parameter: -1 for particle-hole, L+1 for particle number.
    phys_dims : int
        Number of physical dimension pairs per site.
    is_symmetric : bool
        Whether tensors have additional structure.
    th_sing_vals : float
        Threshold for discarding small singular values.
    data_as_tensors : bool
        Storage format: True for n-leg tensors, False for matrices.
    truncation_type : str
        Truncation strategy: "global", "block", or "block_threshold".
    TN : dict
        Dictionary containing "B{i}" tensors and "Lam{i}" singular values.
    """
    
    def __init__(
        self,
        L: int,
        d: int,
        q_alpha: int,
        phys_dims: int,
        chi_max: int | None = None,
        chi_block: int = 0,
        th_sing_vals: float = 1e-8,
        data_as_tensors: bool = True,
        alpha: int = -1,
        initial: str | None = None,
        is_symmetric: bool = False,
        truncation_type: str = "global"
    ):
        """
        Initialize a symmetric MPO.
        
        Parameters
        ----------
        L : int
            System size.
        d : int
            Local dimension.
        q_alpha : int
            Symmetry sector of the operator.
        phys_dims : int
            Physical dimension pairs per site.
        chi_max : int, optional
            Maximum bond dimension.
        chi_block : int
            Minimum states per block.
        th_sing_vals : float
            Singular value threshold.
        data_as_tensors : bool
            Storage format.
        alpha : int
            Super-charge: -1 or L+1.
        initial : str, optional
            Initialize as "Id" for identity.
        is_symmetric : bool
            Additional symmetry structure.
        truncation_type : str
            Truncation method.
        """
        self.L = L
        self.d = d
        self.q_alpha = q_alpha
        self.chi_max = chi_max
        self.chi_block = chi_block
        self.alpha = alpha
        self.phys_dims = phys_dims
        self.is_symmetric = is_symmetric
        self.th_sing_vals = th_sing_vals
        self.data_as_tensors = data_as_tensors
        self.truncation_type = truncation_type
        self.TN: dict[str, SymmetricTensor | SymmetricLambda] = {}
        
        if initial == "Id":
            self._init_identity()
    
    def _init_identity(self) -> None:
        """Initialize as the identity operator."""
        bond_dims = _init_virtual_sectors(
            self.L, self.d, self.q_alpha, self.phys_dims,
            self.chi_max, "Id", self.alpha
        )
        
        for i in range(self.L):
            self.TN[f"B{i}"] = SymmetricTensor(
                self.L, self.d, self.phys_dims,
                left_sectors=bond_dims[i],
                right_sectors=bond_dims[i + 1],
                initial="Id",
                alpha=self.alpha,
                data_as_tensors=self.data_as_tensors,
                is_symmetric=self.is_symmetric
            )
            self.TN[f"Lam{i}"] = SymmetricLambda.identity(
                self.L, self.d,
                self.TN[f"B{i}"].leg_sectors[0],
                chi_max=self.chi_max,
                chi_block=self.chi_block
            )
    
    def copy(self, copy_data: bool = True) -> 'SymmetricMPO':
        """
        Create a copy of the MPO.

        Parameters
        ----------
        copy_data : bool
            If False, tensor data blocks are shared with the original
            (metadata is still copied). Safe as long as blocks are only
            replaced, never mutated in place -- the library convention.
        """
        B = SymmetricMPO(
            self.L, self.d, self.q_alpha, self.phys_dims,
            chi_max=self.chi_max,
            chi_block=self.chi_block,
            th_sing_vals=self.th_sing_vals,
            alpha=self.alpha,
            is_symmetric=self.is_symmetric,
            data_as_tensors=self.data_as_tensors,
            truncation_type=self.truncation_type
        )
        
        for i in range(self.L):
            B.TN[f"B{i}"] = self.TN[f"B{i}"].copy(copy_data=copy_data)
            B.TN[f"Lam{i}"] = self.TN[f"Lam{i}"].copy()
        
        return B
    
    def bond_dimensions(self) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """
        Get bond dimensions at each site.
        
        Returns
        -------
        left : ndarray
            Total left bond dimension at each site.
        right : ndarray
            Total right bond dimension at each site.
        """
        left = np.zeros(self.L, dtype=np.intp)
        right = np.zeros(self.L, dtype=np.intp)
        
        for i in range(self.L):
            B = self.TN[f"B{i}"]
            for sect in B.leg_sectors[0]:
                mask = B.coordinates[0, :] == sect
                if np.any(mask):
                    left[i] += B.shapes[0, mask][0]
            for sect in B.leg_sectors[-1]:
                mask = B.coordinates[-1, :] == sect
                if np.any(mask):
                    right[i] += B.shapes[-1, mask][0]
        
        return left, right
    
    def print_bond_dimensions(self) -> None:
        """Print formatted bond dimensions."""
        left, right = self.bond_dimensions()
        dims = [f"{left[i]}+{right[i]}" for i in range(self.L)]
        print('--'.join(dims))
    
    def entanglement_entropy(self, return_bond_dim: bool = False) -> NDArray | None:
        """
        Compute entanglement entropy at each bond.
        
        Parameters
        ----------
        return_bond_dim : bool
            If True, also return effective bond dimensions.
            
        Returns
        -------
        bond_dims : ndarray, optional
            Effective bond dimensions if requested.
        """
        bond_dims = []
        
        for l in range(self.L):
            # Collect all singular values
            s = np.concatenate(list(self.TN[f"Lam{l}"].data.values()))
            s = s / np.linalg.norm(s)
            
            # von Neumann entropy
            s_sq = s ** 2
            s_sq = s_sq[s_sq > 1e-12]  # Avoid log(0)
            S_entropy = -np.sum(s_sq * np.log(s_sq))
            print(f"Site {l}: S = {S_entropy:.6f}, BondDim={len(s_sq)}")
            
            if return_bond_dim:
                bond_dims.append(np.sum(s ** 2 > 1e-14))
        
        if return_bond_dim:
            return np.array(bond_dims)
        return None
    
    def get_lambda(self, site: int) -> NDArray:
        """Get sorted singular values at a site."""
        lam = self.TN[f"Lam{site}"]
        values = np.concatenate(list(lam.data.values()))
        return np.sort(values)
    
    def norm(self) -> complex:
        """Compute the Frobenius norm of the MPO."""
        from .linalg import trace_mpo_product
        return trace_mpo_product(self, self, conj_A=True)
    
    def trace(self) -> complex:
        """Compute Tr(O)."""
        # PacMan method - contract from left
        from .tensor import SymmetricTensor
        
        pac = SymmetricTensor(
            self.L, self.d, 0,
            alpha=self.alpha,
            n_legs=1, n_sectors=1
        )
        
        s0 = self.L if self.alpha == -1 else 0
        pac.coordinates[:, 0] = s0
        pac.leg_sectors = np.array([np.array([s0])], dtype=object)
        pac.shapes = np.ones((1, 1), dtype=np.intp)
        pac.data[0] = np.ones((1,))
        pac.arrows = np.array(['o'])
        pac.leg_type = np.array(['v'])
        
        from .linalg import tensor_contract
        
        for i in range(self.L):
            B = self.TN[f"B{i}"]
            pac = tensor_contract(pac, B, ([0], [0]))
            
            # Sum over sigma = sigma' (trace over physical)
            mask = pac.coordinates[0, :] == pac.coordinates[1, :]
            
            # Project to right virtual leg only. Each surviving block is a
            # (1, 1, chi_r) tensor: keep the FULL chi_r vector. (The old
            # code took .flat[0], silently discarding all but the first
            # element -- wrong for any bond dimension > 1.)
            right_coords = pac.coordinates[-1, mask]
            right_shapes = pac.shapes[-1, mask]
            blocks = pac.data[mask]
            
            # Combine equal sectors
            coords, idx, inv = np.unique(
                right_coords, return_index=True, return_inverse=True
            )
            
            new_data = np.empty(len(coords), dtype=object)
            for a in range(len(coords)):
                eq_sectors = np.where(inv == a)[0]
                acc = blocks[eq_sectors[0]].reshape(-1)
                if len(eq_sectors) > 1:
                    acc = acc.copy()
                    for s in eq_sectors[1:]:
                        acc += blocks[s].reshape(-1)
                new_data[a] = acc
            
            new_pac = SymmetricTensor(
                self.L, self.d, 0,
                alpha=self.alpha,
                n_legs=1, n_sectors=len(coords)
            )
            new_pac.coordinates = coords.reshape(1, -1)
            new_pac.leg_sectors = np.array([coords], dtype=object)
            new_pac.shapes = right_shapes[idx].reshape(1, -1)
            new_pac.arrows = np.array(['o'])
            new_pac.leg_type = np.array(['v'])
            new_pac.data = new_data
            pac = new_pac
        
        if len(pac.data) == 0:
            return 0.0
        return complex(sum(np.sum(d) for d in pac.data))
    
    def export(self, filename: str) -> None:
        """
        Export MPO to HDF5 file.
        
        Parameters
        ----------
        filename : str
            Path to output file.
        """
        with h5py.File(filename, 'w') as f:
            # Store scalar attributes. h5py cannot store None: encode
            # chi_max=None (no truncation) as the sentinel -1.
            for attr in ['L', 'd', 'q_alpha', 'chi_max', 'chi_block', 
                        'alpha', 'phys_dims', 'th_sing_vals', 
                        'data_as_tensors', 'truncation_type']:
                val = getattr(self, attr)
                if attr == 'chi_max' and val is None:
                    val = -1
                f.attrs[attr] = val
            
            for i in range(self.L):
                B = self.TN[f"B{i}"]
                f.create_dataset(f"B{i}_coordinates", data=B.coordinates)
                for j in range(B.n_sectors):
                    f.create_dataset(f"B{i}_data_{j}", data=B.data[j])
                f.create_dataset(f"B{i}_shapes", data=B.shapes)
                
                for leg in range(B.n_legs):
                    f.create_dataset(f"B{i}_leg_sectors_{leg}", data=B.leg_sectors[leg])
                    f.attrs[f"B{i}_arrows_{leg}"] = B.arrows[leg]
                    f.attrs[f"B{i}_leg_type_{leg}"] = B.leg_type[leg]
                
                lam = self.TN[f"Lam{i}"]
                for key, val in lam.data.items():
                    f.create_dataset(f"Lam{i}_data_{key}", data=val)
                f.create_dataset(f"Lam{i}_left_sectors", data=lam.left_sectors)
                f.create_dataset(f"Lam{i}_right_sectors", data=lam.right_sectors)
                f.attrs[f"Lam{i}_n_sectors"] = lam.n_sectors
        
        print(f"MPO exported to {filename}")
    
    @classmethod
    def load(cls, filename: str) -> 'SymmetricMPO':
        """
        Load MPO from HDF5 file.
        
        Parameters
        ----------
        filename : str
            Path to input file.
            
        Returns
        -------
        SymmetricMPO
            The loaded operator.
        """
        with h5py.File(filename, 'r') as f:
            attrs = dict(f.attrs)
            chi_max = attrs['chi_max']
            chi_max = None if chi_max == -1 else int(chi_max)
            mpo = cls(
                L=attrs['L'],
                d=attrs['d'],
                q_alpha=attrs['q_alpha'],
                phys_dims=attrs['phys_dims'],
                chi_max=chi_max,
                chi_block=attrs['chi_block'],
                th_sing_vals=attrs['th_sing_vals'],
                alpha=attrs['alpha'],
                data_as_tensors=attrs['data_as_tensors'],
                truncation_type=attrs['truncation_type']
            )
            
            for i in range(mpo.L):
                coords = np.array(f[f"B{i}_coordinates"])
                n_legs, n_sectors = coords.shape
                
                B = SymmetricTensor(
                    mpo.L, mpo.d, mpo.phys_dims,
                    alpha=mpo.alpha,
                    data_as_tensors=mpo.data_as_tensors,
                    n_legs=n_legs,
                    n_sectors=n_sectors
                )
                B.coordinates = coords
                B.data = np.empty(n_sectors, dtype=object)
                for j in range(n_sectors):
                    B.data[j] = np.array(f[f"B{i}_data_{j}"])
                B.shapes = np.array(f[f"B{i}_shapes"])
                B.leg_sectors = np.empty(n_legs, dtype=object)
                for leg in range(n_legs):
                    B.leg_sectors[leg] = np.array(f[f"B{i}_leg_sectors_{leg}"])
                    B.arrows[leg] = attrs[f"B{i}_arrows_{leg}"]
                    B.leg_type[leg] = attrs[f"B{i}_leg_type_{leg}"]
                
                mpo.TN[f"B{i}"] = B
                
                lam = SymmetricLambda.empty(
                    mpo.L, mpo.d, mpo.chi_max, mpo.chi_block
                )
                lam.left_sectors = np.array(f[f"Lam{i}_left_sectors"])
                lam.right_sectors = np.array(f[f"Lam{i}_right_sectors"])
                lam.n_sectors = attrs[f"Lam{i}_n_sectors"]
                for sect in lam.left_sectors:
                    lam.data[sect] = np.array(f[f"Lam{i}_data_{sect}"])
                
                mpo.TN[f"Lam{i}"] = lam
        
        print(f"MPO loaded from {filename}")
        return mpo


def _init_virtual_sectors(
    L: int,
    d: int,
    s: int,
    phys_dims: int,
    chi_max: int | None,
    initial: str,
    alpha: int
) -> list[NDArray]:
    """
    Compute allowed symmetry sectors for virtual bonds.
    
    Determines which symmetry sectors are populated at each bond
    given the global symmetry sector and initial operator type.
    
    Parameters
    ----------
    L : int
        System size.
    d : int
        Local dimension.
    s : int
        Global symmetry sector.
    phys_dims : int
        Physical dimension pairs.
    chi_max : int or None
        Maximum bond dimension.
    initial : str
        Initialization type.
    alpha : int
        Super-charge parameter.
        
    Returns
    -------
    list of ndarray
        Bond dimension arrays for each virtual bond (L+1 total).
    """
    loc_sig = np.arange(d)
    chi_max = chi_max or 2 ** L  # Default to full dimension
    
    if alpha == L + 1:
        # Particle number conservation
        list_LR = np.zeros((L + 1, L + 1, L + 1), dtype=np.intp)
        list_RL = np.zeros((L + 1, L + 1, L + 1), dtype=np.intp)
        list_LR[0, 0, 0] = 1
        list_RL[L, s, s] = 1
    elif alpha == -1:
        # Particle-hole symmetry
        list_LR = np.zeros((L + 1, 2 * L + 1), dtype=np.intp)
        list_RL = np.zeros((L + 1, 2 * L + 1), dtype=np.intp)
        list_LR[0, L] = 1
        list_RL[L, L] = 1
    else:
        raise ValueError(f"Unsupported alpha: {alpha}")
    
    # Forward pass (L -> R)
    for i in range(L):
        _propagate_sectors_forward(
            list_LR, i, loc_sig, phys_dims, L, alpha, chi_max, initial
        )
    
    # Backward pass (R -> L)
    for i in range(L, 0, -1):
        _propagate_sectors_backward(
            list_RL, i, loc_sig, phys_dims, L, s, alpha, chi_max, initial
        )
    
    # Intersection of forward and backward
    list_intersec = [np.minimum(list_LR[i], list_RL[i]) for i in range(L + 1)]
    
    if alpha == L + 1 and (initial == "Id" or (initial and initial[0] == "S")):
        list_intersec = [np.minimum(x, 1) for x in list_intersec]
    
    return list_intersec


def _propagate_sectors_forward(
    list_LR: NDArray,
    i: int,
    loc_sig: NDArray,
    phys_dims: int,
    L: int,
    alpha: int,
    chi_max: int,
    initial: str
) -> None:
    """Propagate sectors from left to right."""
    current = np.where(list_LR[i].ravel())[0]
    degen = list_LR[i].ravel()[current]
    
    if len(current) == 0:
        return
    
    # Expand with physical indices
    expanded = current.copy()
    deg_exp = degen.copy()
    
    # For alpha=L+1 the bond charge is flat-encoded as u = l*(L+1) + l':
    # the first phys_dims legs (sigma) increment l, so they add
    # sigma*(L+1); the last phys_dims legs (sigma') increment l' and add
    # sigma. (Previously all legs added sigma, so under the diagonal
    # l = l' constraint only the vacuum chain survived and the sector
    # projector was empty for any q_alpha > 0.)
    for leg in range(2 * phys_dims):
        if alpha == L + 1 and leg < phys_dims:
            inc = loc_sig * (L + 1)
        else:
            inc = loc_sig
        expanded = expanded[:, None] + inc[None, :]
        expanded = expanded.ravel()
        deg_exp = np.repeat(deg_exp, len(loc_sig))
    
    # For alpha=-1: l_R = l_L + sigma - sigma'
    # For alpha=L+1: l_R = l_L + sigma, l'_R = l'_L + sigma'
    if alpha == -1:
        # Apply constraints
        mask = expanded >= 0
        if initial in ("Id",) or (initial and initial[0] == "S"):
            mask = expanded == L
    elif alpha == L + 1:
        mask = (expanded % (L + 1) <= L) & (expanded // (L + 1) <= L)
        if initial in ("Id",) or (initial and initial[0] == "S"):
            mask &= (expanded // (L + 1) == expanded % (L + 1))
    
    expanded = expanded[mask]
    deg_exp = deg_exp[mask]
    
    # Accumulate unique sectors
    if len(expanded) == 0:
        return
        
    unique, inv = np.unique(expanded, return_inverse=True)
    for j, u in enumerate(unique):
        total = np.sum(deg_exp[inv == j])
        if alpha == L + 1:
            new_l, new_lp = int(u) // (L + 1), int(u) % (L + 1)
            list_LR[i + 1, new_l, new_lp] += min(max(0, total), chi_max)
        elif alpha == -1:
            list_LR[i + 1, int(u)] += min(max(0, total), chi_max)


def _propagate_sectors_backward(
    list_RL: NDArray,
    i: int,
    loc_sig: NDArray,
    phys_dims: int,
    L: int,
    s: int,
    alpha: int,
    chi_max: int,
    initial: str
) -> None:
    """Propagate sectors from right to left."""
    current = np.where(list_RL[i].ravel())[0]
    degen = list_RL[i].ravel()[current]
    
    if len(current) == 0:
        return
    
    # Expand with physical indices (going backward: subtract instead of add)
    expanded = current.copy()
    deg_exp = degen.copy()
    
    # Same flat-encoding weighting as in the forward pass (see there):
    # for alpha=L+1 the sigma legs act on l and carry a factor (L+1).
    for leg in range(2 * phys_dims):
        if alpha == L + 1 and leg < phys_dims:
            inc = loc_sig * (L + 1)
        else:
            inc = loc_sig
        expanded = expanded[:, None] - inc[None, :]
        expanded = expanded.ravel()
        deg_exp = np.repeat(deg_exp, len(loc_sig))
    
    if alpha == -1:
        if initial in ("Id",) or (initial and initial[0] == "S"):
            mask = expanded == L
        else:
            mask = expanded >= 0
    elif alpha == L + 1:
        mask = (expanded % (L + 1) <= s) & (expanded >= 0)
        if initial in ("Id",) or (initial and initial[0] == "S"):
            mask &= (expanded // (L + 1) == expanded % (L + 1))
    
    expanded = expanded[mask]
    deg_exp = deg_exp[mask]
    
    if len(expanded) == 0:
        return
    
    unique, inv = np.unique(expanded, return_inverse=True)
    for j, u in enumerate(unique):
        total = np.sum(deg_exp[inv == j])
        if alpha == L + 1:
            new_l, new_lp = int(u) // (L + 1), int(u) % (L + 1)
            list_RL[i - 1, new_l, new_lp] += min(max(0, total), chi_max)
        elif alpha == -1:
            list_RL[i - 1, int(u)] += min(max(0, total), chi_max)


# Operator application functions

def apply_fermionic_op(
    op_type: Literal["c", "c+"],
    mpo: SymmetricMPO,
    site: int,
    side: Literal["L", "R"],
    sign_side: Literal["L", "R"] = "L"
) -> SymmetricMPO:
    """
    Apply a fermionic creation/annihilation operator.
    
    Parameters
    ----------
    op_type : {"c", "c+"}
        Annihilation or creation operator.
    mpo : SymmetricMPO
        The MPO to modify.
    site : int
        Site to apply operator.
    side : {"L", "R"}
        Act from left or right.
    sign_side : {"L", "R"}
        Where to place the Jordan-Wigner string.
        
    Returns
    -------
    SymmetricMPO
        Modified MPO (copy).
    """
    # Shallow copy: coordinates/metadata are copied (they get mutated
    # below); data blocks are shared with the input (they are only ever
    # replaced, never edited in place). This removes the two full deep
    # copies per call that previously made compute_R_matrix O(L^2) in
    # whole-MPO copies.
    O = mpo.copy(copy_data=False)
    
    leg_nb = 2 if side == "L" else 1
    inc_st = 1 if ((op_type == "c" and side == "L") or 
                   (op_type == "c+" and side == "R")) else 0
    
    B = O.TN[f"B{site}"]
    mask = B.coordinates[leg_nb, :] == inc_st
    O.TN[f"B{site}"] = mask_coordinates(B, mask)
    
    B = O.TN[f"B{site}"]
    B.coordinates[leg_nb, :] = (1 + inc_st) % 2
    B.coordinates[-1, :] = (
        B.coordinates[0, :] + 
        O.alpha * B.coordinates[1, :] + 
        B.coordinates[2, :]
    )
    B.invalidate_leg_sectors()
    
    # Propagate charge change to sites on the right
    if op_type == "c":
        delta = -1 if side == "L" else O.alpha
    else:  # "c+"
        delta = 1 if side == "L" else -O.alpha
    
    for j in range(site + 1, O.L):
        Bj = O.TN[f"B{j}"]
        Bj.coordinates[0, :] += delta
        Bj.coordinates[-1, :] = (
            Bj.coordinates[0, :] + 
            O.alpha * Bj.coordinates[1, :] + 
            Bj.coordinates[2, :]
        )
        Bj.invalidate_leg_sectors()
        
        # Keep the Vidal form self-consistent: the Lambda on the bond to
        # the left of site j carries the same sector labels as B_j's left
        # leg, so its keys must shift by the same amount. (Previously the
        # Lambdas kept their STALE labels; any subsequent apply_gate then
        # found no overlap between Lambda sectors and tensor sectors,
        # which fed the apply_lambda corruption path.)
        lam = O.TN[f"Lam{j}"]
        lam.data = {k + delta: v for k, v in lam.data.items()}
        if len(lam.left_sectors):
            lam.left_sectors = lam.left_sectors + delta
        if len(lam.right_sectors):
            lam.right_sectors = lam.right_sectors + delta
    
    # Jordan-Wigner string (in place: O is already our private copy)
    sign_sites = np.arange(site) if sign_side == "L" else np.arange(site + 1, O.L)
    O = apply_spin_op("sz", O, sign_sites, side=side, inplace=True)
    
    return O


def apply_spin_op(
    op_type: Literal["sz"],
    mpo: SymmetricMPO,
    sites: int | list[int] | NDArray,
    side: Literal["L", "R"] = "L",
    inplace: bool = False
) -> SymmetricMPO:
    """
    Apply a spin operator (Pauli Z) to specified sites.
    
    Parameters
    ----------
    op_type : {"sz"}
        Operator type.
    mpo : SymmetricMPO
        The MPO to modify.
    sites : int or array-like
        Sites to apply operator.
    side : {"L", "R"}
        Act from left or right.
    inplace : bool
        Modify `mpo` directly instead of returning a copy. Used
        internally to avoid redundant full-MPO copies.
        
    Returns
    -------
    SymmetricMPO
        Modified MPO (copy unless inplace=True). The copy shares
        unmodified data blocks with the input; sign flips create new
        block arrays, so the input is never altered.
    """
    O = mpo if inplace else mpo.copy(copy_data=False)
    leg_nb = 2 if side == "L" else 1
    
    if isinstance(sites, (int, np.integer)):
        sites = [sites]
    
    if op_type == "sz":
        for j in sites:
            B = O.TN[f"B{j}"]
            for n in np.where(B.coordinates[leg_nb, :] == 1)[0]:
                B.data[n] = B.data[n] * (-1)
    
    return O
