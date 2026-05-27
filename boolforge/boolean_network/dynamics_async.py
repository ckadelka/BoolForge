#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:52:32 2026

@author: ckadelka
"""


from collections.abc import Sequence

import numpy as np

from .. import utils
    
# load optional but desirable package
try:
    import numba
    njit = numba.njit
    int64 = numba.int64
    __LOADED_NUMBA__ = True
except ModuleNotFoundError:
    __LOADED_NUMBA__ = False

class BooleanNetworkDynamicsAsyncMixin:
    def get_attractors_asynchronous_exact(self) -> dict:
        pass
    
    def get_steady_states_asynchronous(
        self,
        n_simulations: int = 500,
        initial_states: Sequence[int] | None = None,
        search_depth: int = 50,
        debug: bool = False,
        *,
        rng=None,
    ) -> dict:
        """
        Approximate steady states of a Boolean network under asynchronous updates.
    
        This method performs a Monte Carlo–style exploration of the asynchronous
        state space by simulating asynchronous updates from a collection of initial
        states. Each simulation proceeds until a steady state is reached or until
        a maximum search depth is exceeded.
    
        Unlike ``get_steady_states_asynchronous_exact``, this method does *not*
        exhaustively explore the full state space and does not guarantee that all
        steady states will be found. It is intended for large networks where exact
        enumeration is infeasible.
    
        Parameters
        ----------
        n_simulations : int, optional
            Number of asynchronous simulations to perform (default is 500).
        initial_states : sequence of int or None, optional
            Initial states to use for the simulations, given as decimal
            representations of network states. If None (default), ``n_simulations``
            random initial states are generated.
        search_depth : int, optional
            Maximum number of asynchronous update steps per simulation before
            giving up on convergence (default is 50).
        debug : bool, optional
            If True, print detailed debugging information during simulation.
        rng : optional
            Random number generator or seed, passed to ``utils._coerce_rng``.
    
        Returns
        -------
        dict
            Dictionary with the following entries:
    
            - SteadyStates : list of int  
              Decimal representations of steady states encountered.
            - NumberOfSteadyStatesLowerBound : int  
              Number of unique steady states found.
            - BasinSizesApproximation : list of int  
              Proportion of simulations that converged to each steady state.
            - STGAsynchronous : dict  
              Partial cache of asynchronous transitions encountered during
              simulation. Keys are ``(state, node_index)`` and values are
              successor states (all in decimal form).
            - InitialSamplePoints : list of int  
              Decimal initial states used in the simulations (either provided
              explicitly or generated randomly).
    
        Notes
        -----
        - This method detects only *steady states* (fixed points). If the
          asynchronous dynamics contain limit cycles, simulations may fail
          to converge within ``search_depth``.
        - The returned asynchronous transition graph is generally incomplete
          and should be interpreted as a cache of explored transitions rather
          than the full STG.
        - There is no guarantee that all steady states will be identified.
        """
        rng = utils._coerce_rng(rng)
    
        sampled_states: list[int] = []
        STG_asynchronous: dict[tuple[int, int], int] = {}
    
        steady_states: list[int] = []
        basin_sizes: list[int] = []
        steady_state_dict: dict[int, int] = {}
    
        for iteration in range(n_simulations):
            # Initialize state
            if initial_states is None:
                x = rng.integers(2, size=self.N)
                xdec = utils.bin2dec(x)
                sampled_states.append(xdec)
            else:
                xdec = initial_states[iteration]
                x = utils.dec2bin(xdec, self.N)
    
            if debug:
                print(iteration, -1, -1, False, xdec, x)
    
            for step in range(search_depth):
                found_new_state = False
    
                # Check if state is already known to be steady
                if xdec in steady_state_dict:
                    basin_sizes[steady_state_dict[xdec]] += 1
                    break
    
                update_order = rng.permutation(self.N)
                for i in map(int, update_order):
                    try:
                        fxdec = STG_asynchronous[(xdec, i)]
                    except KeyError:
                        fx_i = self.update_single_node(i, x[self.I[i]])
                        if fx_i > x[i]:
                            fxdec = xdec + 2 ** (self.N - 1 - i)
                            x[i] = 1
                            found_new_state = True
                        elif fx_i < x[i]:
                            fxdec = xdec - 2 ** (self.N - 1 - i)
                            x[i] = 0
                            found_new_state = True
                        else:
                            fxdec = xdec
                        STG_asynchronous[(xdec, i)] = fxdec
    
                    if fxdec != xdec:
                        xdec = fxdec
                        found_new_state = True
                        break
    
                if debug:
                    print(iteration, step, i, found_new_state, xdec, x)
    
                if not found_new_state:
                    # New steady state found
                    if xdec in steady_state_dict:
                        basin_sizes[steady_state_dict[xdec]] += 1
                    else:
                        steady_state_dict[xdec] = len(steady_states)
                        steady_states.append(xdec)
                        basin_sizes.append(1)
                    break
    
            if debug:
                print()
    
        if sum(basin_sizes) < n_simulations:
            print(
                f"Warning: only {sum(basin_sizes)} of the {n_simulations} simulations "
                "reached a steady state. Consider increasing search_depth. "
                "The network may also contain asynchronous limit cycles."
            )
        
        if sum(basin_sizes)>0:
            sum_basin_sizes  = sum(basin_sizes)
            basin_sizes = np.array([size/sum_basin_sizes for size in basin_sizes])
        
        return {
            "SteadyStates": steady_states,
            "NumberOfSteadyStatesLowerBound": len(steady_states),
            "BasinSizesApproximation": basin_sizes,
            "STGAsynchronous": STG_asynchronous,
            "InitialSamplePoints": (
                initial_states if initial_states is not None else sampled_states
            ),
        }
    
    def get_steady_states_asynchronous_given_one_initial_condition(
        self,
        initial_condition: int | Sequence[int] = 0,
        n_simulations: int = 500,
        stochastic_weights: Sequence[float] | None = None,
        search_depth: int = 50,
        debug: bool = False,
        *,
        rng=None,
    ) -> dict:
        """
        Approximate steady states reachable from a single initial condition under
        asynchronous updates.
    
        This method performs multiple asynchronous simulations starting from the
        same initial condition. In each simulation, nodes are updated one at a time
        according to either a uniform random order or node-specific stochastic
        update propensities. The simulation proceeds until a steady state is reached
        or a maximum number of update steps is exceeded.
    
        The method is sampling-based and does *not* guarantee that all reachable
        steady states are found. It is intended for exploratory analysis and for
        networks where exhaustive asynchronous analysis is infeasible.
    
        Parameters
        ----------
        initial_condition : int or sequence of int, optional
            Initial network state. If an integer is provided, it is interpreted as
            the decimal encoding of a Boolean state. If a sequence is provided, it
            must be a binary vector of length ``N``. Default is 0.
        n_simulations : int, optional
            Number of asynchronous simulation runs (default is 500).
        stochastic_weights : sequence of float or None, optional
            Relative update propensities for each node. If provided, must have
            length ``N`` and be strictly positive. The weights are normalized
            internally. If None (default), nodes are updated uniformly at random.
        search_depth : int, optional
            Maximum number of asynchronous update steps per simulation.
        debug : bool, optional
            If True, print detailed debugging information during simulation.
        rng : optional
            Random number generator or seed, passed to ``utils._coerce_rng``.
    
        Returns
        -------
        dict
            Dictionary with the following entries:
    
            - SteadyStates : list of int  
              Decimal representations of steady states reached.
            - NumberOfSteadyStatesLowerBound : int  
              Number of unique steady states found.
            - BasinSizesApproximation : list of int  
              Proportion of simulations converging to each steady state.
            - TransientTimes : list of list of int  
              For each steady state, a list of transient lengths (number of update
              steps before convergence).
            - STGAsynchronous : dict  
              Partial cache of asynchronous transitions encountered during
              simulation. Keys are ``(state, node_index)`` and values are successor
              states (all in decimal form).
            - UpdateQueues : list of list of int  
              For each simulation, the sequence of visited states (in decimal form).
    
        Notes
        -----
        - Only steady states (fixed points) are detected. If the asynchronous
          dynamics contain limit cycles, simulations may fail to converge within
          ``search_depth``.
        - The returned asynchronous transition graph is incomplete and represents
          only transitions encountered during sampling.
        - There is no guarantee that all steady states will be identified.
        """
        rng = utils._coerce_rng(rng)
    
        # --- Initialize initial condition ---
        if isinstance(initial_condition, int):
            x0 = utils.dec2bin(initial_condition, self.N)
            x0dec = initial_condition
        else:
            x0 = np.asarray(initial_condition, dtype=int)
            if x0.shape[0] != self.N:
                raise ValueError(
                    f"Initial condition must have length {self.N}, got {x0.shape[0]}."
                )
            x0dec = utils.bin2dec(x0)
    
        # --- Handle stochastic weights ---
        if stochastic_weights is not None:
            stochastic_weights = np.asarray(stochastic_weights, dtype=float)
            if stochastic_weights.shape[0] != self.N:
                raise ValueError("stochastic_weights must have length N.")
            if np.any(stochastic_weights <= 0):
                raise ValueError("stochastic_weights must be strictly positive.")
            stochastic_weights = stochastic_weights / stochastic_weights.sum()
    
        # --- Bookkeeping ---
        STG_async: dict[tuple[int, int], int] = {}
        steady_states: list[int] = []
        basin_sizes: list[int] = []
        transient_times: list[list[int]] = []
        steady_state_dict: dict[int, int] = {}
        queues: list[list[int]] = []
    
        # --- Simulations ---
        for iteration in range(n_simulations):
            x = x0.copy()
            xdec = x0dec
            queue = [xdec]
    
            for step in range(search_depth):
                found_new_state = False
    
                # If already known steady state, stop
                if xdec in steady_state_dict:
                    idx = steady_state_dict[xdec]
                    basin_sizes[idx] += 1
                    transient_times[idx].append(step)
                    queues.append(queue)
                    break
    
                # Choose update order
                if stochastic_weights is None:
                    update_order = rng.permutation(self.N)
                else:
                    update_order = rng.choice(
                        self.N, size=self.N, replace=False, p=stochastic_weights
                    )
    
                for i in map(int, update_order):
                    try:
                        fxdec = STG_async[(xdec, i)]
                    except KeyError:
                        fx_i = self.update_single_node(i, x[self.I[i]])
                        if fx_i > x[i]:
                            fxdec = xdec + 2 ** (self.N - 1 - i)
                            x[i] = 1
                        elif fx_i < x[i]:
                            fxdec = xdec - 2 ** (self.N - 1 - i)
                            x[i] = 0
                        else:
                            fxdec = xdec
                        STG_async[(xdec, i)] = fxdec
    
                    if fxdec != xdec:
                        xdec = fxdec
                        queue.append(xdec)
                        found_new_state = True
                        break
    
                if debug:
                    print(iteration, step, i, found_new_state, xdec, x)
    
                if not found_new_state:
                    # New steady state reached
                    if xdec in steady_state_dict:
                        idx = steady_state_dict[xdec]
                        basin_sizes[idx] += 1
                        transient_times[idx].append(step)
                    else:
                        steady_state_dict[xdec] = len(steady_states)
                        steady_states.append(xdec)
                        basin_sizes.append(1)
                        transient_times.append([step])
                    queues.append(queue)
                    break
    
            if debug:
                print()
    
        if sum(basin_sizes) < n_simulations:
            print(
                f"Warning: only {sum(basin_sizes)} of the {n_simulations} simulations "
                "reached a steady state. Consider increasing search_depth. "
                "The network may contain asynchronous limit cycles."
            )
            
        basin_sizes = np.array(basin_sizes)/n_simulations
    
        return {
            "SteadyStates": steady_states,
            "NumberOfSteadyStatesLowerBound": len(steady_states),
            "BasinSizesApproximation": basin_sizes,
            "TransientTimes": transient_times,
            "STGAsynchronous": STG_async,
            "UpdateQueues": queues,
        }