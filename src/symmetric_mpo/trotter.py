"""
Trotter decomposition for time evolution.

This module provides functions to construct Trotter sequences of
different orders for TEBD time evolution.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any
from dataclasses import dataclass

from .mpo import SymmetricGate


@dataclass
class TrotterStep:
    """
    A single step in a Trotter sequence.
    
    Attributes
    ----------
    layer : str
        Identifier for the Hamiltonian layer (e.g., "H0", "H1").
    dt : float
        Time step for this layer.
    new_time : bool
        Whether this step completes a full time step.
    compute_obs : bool
        Whether to compute observables after this step.
    """
    layer: str
    dt: float
    new_time: bool = False
    compute_obs: bool = False


def commuting_sequence(n_parts: int, dt: float) -> list[tuple[str, float]]:
    """
    Create a sequence of commuting Hamiltonian parts.
    
    Parameters
    ----------
    n_parts : int
        Number of commuting parts (e.g., even/odd bond layers).
    dt : float
        Time step for each part.
        
    Returns
    -------
    list of tuple
        Sequence of (layer_name, time_step) pairs.
    """
    return [(f"H{i}", dt) for i in range(n_parts)]


def build_trotter_sequence(
    order: int,
    n_steps: int,
    dt: float,
    n_parts: int,
    obs_interval: int,
    params: dict[str, Any],
    *,
    alpha: int = -1,
    data_as_tensors: bool = True,
    model: str = "Heis_nn",
    layer_names: list[str] | None = None
) -> tuple[list[TrotterStep], dict, dict, int]:
    """
    Build a Trotter decomposition sequence with gates.
    
    Constructs the full time evolution sequence with specified Trotter
    order, including gate creation and step compaction.
    
    Parameters
    ----------
    order : int
        Trotter order: 1, 2, or 4.
    n_steps : int
        Number of time steps.
    dt : float
        Base time step.
    n_parts : int
        Number of commuting Hamiltonian parts.
    obs_interval : int
        Compute observables every this many steps.
    params : dict
        Physical parameters for the model.
    alpha : int
        Super-charge parameter.
    data_as_tensors : bool
        Storage format for gates.
    model : str
        Physical model name.
    layer_names : list of str, optional
        Names of Hamiltonian layers. Default: ["H0", "H1", ...].
        
    Returns
    -------
    steps : list of TrotterStep
        The compacted Trotter sequence.
    U : dict
        Forward gates keyed by (layer, dt).
    U_dag : dict
        Adjoint gates keyed by (layer, dt).
    n_layers : int
        Number of layers per time step.
    """
    phys_dims = 2 * params.get('phys_dims', 1)
    
    if layer_names is None:
        layer_names = [f"H{i}" for i in range(n_parts)]
    
    U = {}
    U_dag = {}
    
    if order == 1:
        # First order: O(dt^2) error
        dt_list = [dt]
        raw_steps = commuting_sequence(n_parts, dt)
        
    elif order == 2:
        # Second order (Strang): O(dt^3) error
        dt_list = [dt, dt / 2]
        raw_steps = (
            commuting_sequence(n_parts, dt / 2) +
            commuting_sequence(n_parts, dt / 2)[::-1]
        )
        
    elif order == 4:
        # Fourth order (Yoshida): O(dt^5) error
        dt1 = dt / (4.0 - 4.0 ** (1 / 3))
        dt2 = dt - 4 * dt1
        
        U2_1 = (
            commuting_sequence(n_parts, dt1 / 2) +
            commuting_sequence(n_parts, dt1 / 2)[::-1]
        )
        U2_2 = (
            commuting_sequence(n_parts, dt2 / 2) +
            commuting_sequence(n_parts, dt2 / 2)[::-1]
        )
        
        dt_list = [dt1, dt2, dt1 / 2, dt2 / 2, dt1 + dt2, (dt1 + dt2) / 2]
        raw_steps = U2_1 + U2_1 + U2_2 + U2_1 + U2_1
    else:
        raise ValueError(f"Trotter order {order} not supported. Use 1, 2, or 4.")
    
    # Create gates for all needed time steps
    for layer in layer_names:
        for DT in dt_list:
            U[layer, DT] = SymmetricGate(
                phys_dims, DT, params,
                gate_type="Hamiltonian",
                model=model,
                alpha=alpha,
                data_as_tensors=data_as_tensors,
                step=layer
            )
            U_dag[layer, DT] = SymmetricGate(
                phys_dims, DT, params,
                gate_type="Hamiltonian",
                model=model,
                dag=True,
                alpha=alpha,
                data_as_tensors=data_as_tensors,
                step=layer
            )
    
    # First compaction: merge consecutive identical layers
    compact_steps = _compact_consecutive(raw_steps)
    n_layers = len(compact_steps)
    
    # Replicate for all time steps
    full_sequence = compact_steps * n_steps
    
    # Second compaction: merge across time step boundaries and mark observables
    final_steps = _compact_with_observables(
        full_sequence, n_layers, obs_interval
    )
    
    return final_steps, U, U_dag, n_layers


def _compact_consecutive(
    steps: list[tuple[str, float]]
) -> list[tuple[str, float]]:
    """
    Merge consecutive steps with the same layer.
    """
    if not steps:
        return []
    
    compact = []
    i = 0
    
    while i < len(steps) - 1:
        if steps[i][0] == steps[i + 1][0]:
            compact.append((steps[i][0], steps[i][1] + steps[i + 1][1]))
            i += 2
        else:
            compact.append(steps[i])
            i += 1
    
    if i == len(steps) - 1:
        compact.append(steps[-1])
    
    return compact


def _compact_with_observables(
    steps: list[tuple[str, float]],
    n_layers: int,
    obs_interval: int
) -> list[TrotterStep]:
    """
    Compact steps and mark when to compute observables.
    """
    result = []
    i = 0
    time_step_count = 0
    
    while i < len(steps) - 1:
        # Check if we're crossing a time step boundary
        current_layer = i // n_layers
        can_merge = 2 if steps[i][0] == steps[i + 1][0] else 1
        next_layer = (i + can_merge) // n_layers
        
        new_time = (next_layer != current_layer)
        if new_time:
            time_step_count += 1
        
        compute_obs = new_time and (time_step_count % obs_interval == 0)
        
        if steps[i][0] == steps[i + 1][0] and not compute_obs:
            # Merge consecutive same-layer steps
            result.append(TrotterStep(
                layer=steps[i][0],
                dt=steps[i][1] + steps[i + 1][1],
                new_time=new_time,
                compute_obs=False
            ))
            i += 2
        else:
            result.append(TrotterStep(
                layer=steps[i][0],
                dt=steps[i][1],
                new_time=new_time,
                compute_obs=compute_obs
            ))
            i += 1
    
    # Handle last step
    if i == len(steps) - 1:
        result.append(TrotterStep(
            layer=steps[-1][0],
            dt=steps[-1][1],
            new_time=True,
            compute_obs=True
        ))
    
    return result


def get_gate_sequence(
    steps: list[TrotterStep],
    gates: dict[str, list],
    U: dict,
    U_dag: dict
) -> list[tuple[SymmetricGate, SymmetricGate, list, TrotterStep]]:
    """
    Convert Trotter steps to a sequence of gate applications.
    
    Parameters
    ----------
    steps : list of TrotterStep
        The Trotter sequence.
    gates : dict
        Mapping from layer names to lists of site pairs.
    U : dict
        Forward gates.
    U_dag : dict
        Adjoint gates.
        
    Returns
    -------
    list of tuple
        Each tuple contains (gate, gate_dag, site_pairs, step_info).
    """
    result = []
    
    for step in steps:
        layer = step.layer
        dt = step.dt
        
        # Fail loudly: silently skipping a missing (layer, dt) key used
        # to drop Trotter steps and evolve with the WRONG total time.
        # (E.g. n_parts=1 with obs_interval >= 2 merges boundary steps
        # into a 2*dt gate that was never precomputed.)
        if (layer, dt) not in U:
            raise KeyError(
                f"get_gate_sequence: no precomputed gate for layer "
                f"{layer!r} with dt={dt!r}. Available keys: "
                f"{sorted(U.keys())}. Precompute this gate or adjust "
                f"n_parts/obs_interval so merged steps match dt_list."
            )
        if (layer, dt) not in U_dag:
            raise KeyError(
                f"get_gate_sequence: no adjoint gate for layer "
                f"{layer!r} with dt={dt!r}."
            )
        gate = U[layer, dt]
        gate_dag = U_dag[layer, dt]
        site_pairs = gates.get(layer, [])
        result.append((gate, gate_dag, site_pairs, step))
    
    return result