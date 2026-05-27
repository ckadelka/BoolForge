#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 01:26:57 2026

@author: ckadelka
"""

import math
import numpy as np

# load optional but desirable package
try:
    import numba
    from numba.typed import List
    njit = numba.njit
    int64 = numba.int64
    __LOADED_NUMBA__ = True
except ModuleNotFoundError:
    __LOADED_NUMBA__ = False


def _compress_with_known_cycle(traj, cycle_len):
    len_traj = len(traj)
    best_trajectory = []
    best_cycle_len = -1
    best_length = math.inf
    for s in range(len_traj):
        for p in range(1, min(cycle_len, len_traj - s) + 1):
            proposed_period = traj[s : s + p]
            good_proposal = True
            for i in range(s, len_traj):
                if traj[i] != proposed_period[(i - s) % p]:
                    good_proposal = False
                    break
            if not good_proposal:
                continue
            
            len_proposal = s + p
            if len_proposal < best_length:
                best_length = len_proposal
                best_trajectory = traj[:s] + proposed_period
                best_cycle_len = p
    return best_trajectory, best_cycle_len

if __LOADED_NUMBA__:
    @njit(fastmath=True)  # safe: operations are integer-only
    def _update_network_synchronously_numba(
        x,
        F_array_list,
        I_array_list,
    ):
        """
        Perform one synchronous update of a Boolean network.
    
        Given a binary state vector ``x``, this function computes the next network
        state under synchronous updating by evaluating each node’s Boolean update
        function based on its regulators.
    
        Parameters
        ----------
        x : np.ndarray
            Binary state vector of shape ``(N,)`` with dtype ``uint8``.
        F_array_list : list[np.ndarray]
            List of truth tables for each node, where the ``j``-th entry is an
            array of length ``2**k_j`` giving the update rule for node ``j`` with
            ``k_j`` regulators.
        I_array_list : list[np.ndarray]
            List of regulator index arrays, where the ``j``-th entry contains the
            indices of the regulators of node ``j``.
    
        Returns
        -------
        np.ndarray
            Updated binary state vector of shape ``(N,)`` with dtype ``uint8``.
        """
        N = x.shape[0]
        fx = np.empty(N, dtype=np.uint8)
        for j in range(N):
            regulators = I_array_list[j]
            if regulators.shape[0] == 0:
                fx[j] = F_array_list[j][0]
            else:
                idx = 0
                for k in range(regulators.shape[0]):
                    idx = (idx << 1) | x[regulators[k]]
                fx[j] = F_array_list[j][idx]
        return fx
    
    @njit
    def _compute_synchronous_stg_numba(
        F_array_list, 
        I_array_list, 
        N_variables
    ):
        """
        Compute the synchronous state transition graph (STG).
    
        This Numba-compiled function computes, for every possible binary state
        of a Boolean network, the index of its successor state under synchronous
        updating.
    
        Parameters
        ----------
        F_array_list : list[np.ndarray]
            List of Boolean update tables. The ``j``-th entry is a NumPy array of
            length ``2**k_j`` representing the update rule for node ``j`` with
            ``k_j`` regulators.
        I_array_list : list[np.ndarray]
            List of regulator index arrays. The ``j``-th entry contains the indices
            of the regulators of node ``j``.
        N_variables : int
            Number of variables (nodes) in the network.
    
        Returns
        -------
        np.ndarray
            One-dimensional array of length ``2**N_variables`` containing, for
            each state index, the index of the successor state under synchronous
            updating.
        """
        nstates = 2 ** N_variables
        states = np.zeros((nstates, N_variables), dtype=np.uint8)
        for i in range(nstates):
            # binary representation of i
            for j in range(N_variables):
                states[i, N_variables - 1 - j] = (i >> j) & 1
    
        next_states = np.zeros_like(states)
        powers_of_two = 2 ** np.arange(N_variables - 1, -1, -1)
    
        # Compute next state for each node
        for j in range(N_variables):
            regulators = I_array_list[j]
            if len(regulators) == 0:
                # constant node
                next_states[:, j] = F_array_list[j][0]
                continue
    
            n_reg = len(regulators)
            reg_powers = 2 ** np.arange(n_reg - 1, -1, -1)
            for s in range(nstates):
                idx = 0
                for k in range(n_reg):
                    idx += states[s, regulators[k]] * reg_powers[k]
                next_states[s, j] = F_array_list[j][idx]
    
        # Convert each next state to integer index
        next_indices = np.zeros(nstates, dtype=np.int64) # NOTE: this cannot be an unsigned int for safe indexing inside Numba kernels.
        for s in range(nstates):
            val = 0
            for j in range(N_variables):
                val += next_states[s, j] * powers_of_two[j]
            next_indices[s] = val
    
        return next_indices

    @njit
    def _compute_synchronous_stg_numba_low_memory(
        F_array_list,
        I_array_list,
        N_variables
    ):
        """
        Compute the synchronous state transition graph (STG) using minimal memory.
    
        For each integer state index ``i`` in ``[0, 2**N_variables)``, this function
        decodes ``i`` into its binary state vector, computes the synchronous update
        of the Boolean network, and encodes the resulting state back into an integer
        index.
    
        Parameters
        ----------
        F_array_list : list[np.ndarray]
            List of Boolean update tables. The ``j``-th entry is an array of length
            ``2**k_j`` representing the update rule for node ``j`` with ``k_j``
            regulators.
        I_array_list : list[np.ndarray]
            List of regulator index arrays. The ``j``-th entry contains the indices
            of the regulators of node ``j``.
        N_variables : int
            Number of variables (nodes) in the network.
    
        Returns
        -------
        np.ndarray
            One-dimensional array of length ``2**N_variables`` containing, for each
            state index, the index of the successor state under synchronous updating.
    
        Notes
        -----
        This implementation avoids storing the full state matrix and therefore
        reduces memory usage from ``O(N * 2**N)`` to ``O(N + 2**N)``. The time
        complexity remains exponential in ``N_variables``.
        """
        nstates = 2 ** N_variables
        next_indices = np.zeros(nstates, dtype=np.int64) # NOTE: this cannot be an unsigned int for safe indexing inside Numba kernels.
        powers_of_two = 2 ** np.arange(N_variables - 1, -1, -1)
    
        state = np.zeros(N_variables, dtype=np.uint8)
        next_state = np.zeros(N_variables, dtype=np.uint8)
    
        for i in range(nstates):
            # --- Decode i into binary vector (most-significant bit first)
            tmp = i
            for j in range(N_variables):
                state[N_variables - 1 - j] = tmp & 1
                tmp >>= 1
    
            # --- Compute next-state values
            for j in range(N_variables):
                regulators = I_array_list[j]
                if regulators.shape[0] == 0:
                    next_state[j] = F_array_list[j][0]
                else:
                    n_reg = regulators.shape[0]
                    idx = 0
                    for k in range(n_reg):
                        idx = (idx << 1) | state[regulators[k]]
                    next_state[j] = F_array_list[j][idx]
    
            # --- Encode next_state back to integer
            val = 0
            for j in range(N_variables):
                val += next_state[j] * powers_of_two[j]
            next_indices[i] = val
        return next_indices


    @njit(fastmath=True)  # safe: operations are integer-only
    def _hamming_distance(a, b):
        """
        Compute the Hamming distance between two binary vectors.
    
        Parameters
        ----------
        a : np.ndarray
            One-dimensional array of dtype ``uint8``.
        b : np.ndarray
            One-dimensional array of dtype ``uint8`` with the same shape as ``a``.
    
        Returns
        -------
        int
            Number of positions at which ``a`` and ``b`` differ.
        """
        dist = 0
        for i in range(a.size):
            dist += a[i] != b[i]
        return dist
    
    
    @njit(fastmath=True)  # safe: operations are integer-only
    def _derrida_simulation(
        F_array_list,
        I_array_list,
        N,
        n_simulations,
        seed
    ):
        """
        Perform a Monte Carlo simulation to estimate the Derrida value.
    
        This function estimates the Derrida value by repeatedly sampling a random
        initial state, flipping a single randomly chosen bit, synchronously
        updating both states, and computing the Hamming distance between the
        resulting successor states.
    
        Parameters
        ----------
        F_array_list : list[np.ndarray]
            List of Boolean update tables for each node.
        I_array_list : list[np.ndarray]
            List of regulator index arrays for each node.
        N : int
            Number of variables (nodes) in the network.
        n_simulations : int
            Number of Monte Carlo simulations to perform.
        seed : int
            Seed for the Numba-compatible random number generator.
    
        Returns
        -------
        float
            Estimated Derrida value, i.e., the expected Hamming distance after one
            synchronous update following a random single-bit perturbation.
        """

        # Numba RNG: seed once
        np.random.seed(seed)
        total_dist = 0.0
    
        X = np.empty(N, dtype=np.uint8)
        Y = np.empty(N, dtype=np.uint8)
    
        for _ in range(n_simulations):
            # Random initial state
            for i in range(N):
                X[i] = np.random.randint(0, 2)
            Y[:] = X
    
            # Flip one random bit
            idx = np.random.randint(0, N)
            Y[idx] = 1 - Y[idx]
    
            # Synchronous updates
            FX = _update_network_synchronously_numba(X, F_array_list, I_array_list)
            FY = _update_network_synchronously_numba(Y, F_array_list, I_array_list)
    
            total_dist += _hamming_distance(FX, FY)
    
        return total_dist / n_simulations

    @njit(cache=True)
    def _attractors_functional_graph(next_state):
        """
        Identify attractors and basins in a functional graph.
    
        Given a functional graph represented by a successor array, this function
        identifies all attractors (cycles), assigns each state to an attractor,
        and computes basin sizes and cycle properties.
    
        Parameters
        ----------
        next_state : np.ndarray
            One-dimensional integer array of length ``n`` such that
            ``next_state[x]`` gives the successor of state ``x`` and lies in
            ``[0, n-1]``.
    
        Returns
        -------
        attr_id : np.ndarray
            Integer array of length ``n`` mapping each state to its attractor
            index.
        basin_sizes : np.ndarray
            Integer array of length ``n_attr`` giving the basin size of each
            attractor.
        cycle_rep : np.ndarray
            Integer array of length ``n_attr`` containing one representative
            state from each attractor cycle.
        cycle_len : np.ndarray
            Integer array of length ``n_attr`` giving the length of each cycle.
        n_attr : np.int32
            Number of attractors in the functional graph.
        """

        n = next_state.shape[0]
        attr_id = np.full(n, -1, dtype=np.int32)
    
        # For detecting cycles within the current walk:
        # seen[u] == run_id  means u was visited in this run
        # pos[u] = index of u in the current path (when first visited this run)
        seen = np.zeros(n, dtype=np.int32)
        pos = np.zeros(n, dtype=np.int32)
    
        # Upper bounds: in the worst case every node could be its own 1-cycle
        basin_sizes_full = np.zeros(n, dtype=np.int32)
        cycle_rep_full = np.empty(n, dtype=np.int64)
        cycle_len_full = np.zeros(n, dtype=np.int32)
    
        n_attr = 0
    
        # Numba typed list for the current path
        path = List.empty_list(int64)
    
        for start in range(n):
            if attr_id[start] != -1:
                continue
    
            path.clear()
            u = start
            run_id = start + 1  # unique per start; safe while n << 2**31 (always true in practice)
    
            # Walk until we hit a known attractor or revisit a node in this run
            while attr_id[u] == -1 and seen[u] != run_id:
                seen[u] = run_id
                pos[u] = len(path)
                path.append(u)
                u = next_state[u]
    
            if attr_id[u] != -1:
                # This path flows into an already-known attractor
                aid = attr_id[u]
                for i in range(len(path)):
                    v = path[i]
                    attr_id[v] = aid
                    basin_sizes_full[aid] += 1
            else:
                # We found a cycle within the current run.
                # u is the first repeated node; cycle starts at pos[u] in path
                cyc_start = pos[u]
                aid = n_attr
                n_attr += 1
    
                # Representative and length of the cycle
                cycle_rep_full[aid] = u
                cycle_len_full[aid] = len(path) - cyc_start
    
                # Assign all nodes on the path to this new attractor
                for i in range(len(path)):
                    v = path[i]
                    attr_id[v] = aid
                    basin_sizes_full[aid] += 1
    
        return attr_id, basin_sizes_full[:n_attr], cycle_rep_full[:n_attr], cycle_len_full[:n_attr], np.int32(n_attr)
    
    @njit(cache=True)
    def _transient_lengths_functional_numba(
        succ,
        is_attr_mask
    ):
        """
        Compute exact transient length (distance to attractor) for a functional graph.
    
        Parameters
        ----------
        succ : int64 array, shape (n_states,)
            succ[x] = successor of state x
        is_attr_mask : uint8/bool array, shape (n_states,)
            1 if state lies on an attractor cycle, else 0
    
        Returns
        -------
        dist : int64 array, shape (n_states,)
            dist[x] = number of steps from x to its attractor
        """
        n = succ.shape[0]
        dist = np.full(n, -1, dtype=np.int64)
    
        # Attractor states have distance 0
        for i in range(n):
            if is_attr_mask[i]:
                dist[i] = 0
    
        for i in range(n):
            if dist[i] >= 0:
                continue
    
            v = i
    
            # Walk forward until we hit a known distance
            while dist[v] == -1:
                dist[v] = -2          # temporary marker: "in current path"
                v = succ[v]
    
            # Now dist[v] is either:
            #   0,1,2,...  (known)
            # or -2       (should not happen if cycles were pre-marked)
            d = dist[v]
    
            # Unwind path, assigning distances
            v = i
            while dist[v] == -2:
                d += 1
                nxt = succ[v]
                dist[v] = d
                v = nxt
    
        return dist
    
    @njit(cache=True)
    def _robustness_edge_traversal_numba(
        N,
        attractor_idx,
        is_attr_mask,
        dist_attr
    ):
        """
        Traverse hypercube edges to compute basin and attractor robustness measures.
    
        This function iterates over all undirected edges of the Boolean hypercube
        exactly once and accumulates coherence and fragility contributions for
        basins of attraction and for attractor states.
    
        Parameters
        ----------
        N : int
            Number of variables (dimension of the Boolean hypercube).
        attractor_idx : np.ndarray
            Integer array of shape ``(2**N,)`` mapping each state to its attractor
            index in ``[0, n_attr - 1]``.
        is_attr_mask : np.ndarray
            Boolean or uint8 array of shape ``(2**N,)`` indicating whether a state
            lies on an attractor.
        dist_attr : np.ndarray
            Two-dimensional array of shape ``(n_attr, n_attr)`` giving pairwise
            distances between attractors.
    
        Returns
        -------
        basin_coh : np.ndarray
            Array of length ``n_attr`` containing basin coherence values.
        basin_frag : np.ndarray
            Array of length ``n_attr`` containing basin fragility values.
        attr_coh : np.ndarray
            Array of length ``n_attr`` containing attractor coherence values.
        attr_frag : np.ndarray
            Array of length ``n_attr`` containing attractor fragility values.
        """

        n_states = attractor_idx.shape[0]
        n_attr = dist_attr.shape[0]
    
        basin_coh = np.zeros(n_attr, dtype=np.float64)
        basin_frag = np.zeros(n_attr, dtype=np.float64)
        attr_coh = np.zeros(n_attr, dtype=np.float64)
        attr_frag = np.zeros(n_attr, dtype=np.float64)
    
        # Iterate each undirected hypercube edge exactly once:
        # For x, flip only bits that are 0 -> y = x | (1<<bit), which guarantees y > x.
        for xdec in range(n_states):
            idx_x = attractor_idx[xdec]
            # (Should never be -1 if attractor_idx is filled for all states)
            for bit in range(N):
                if (xdec >> bit) & 1:
                    continue
                ydec = xdec | (1 << bit)
                idx_y = attractor_idx[ydec]
    
                if idx_x == idx_y:
                    # same basin: count both directions (like your +2)
                    basin_coh[idx_x] += 2.0
                    if is_attr_mask[xdec]:
                        attr_coh[idx_x] += 1.0
                    if is_attr_mask[ydec]:
                        attr_coh[idx_y] += 1.0
                else:
                    dxy = dist_attr[idx_x, idx_y]
                    basin_frag[idx_x] += dxy
                    basin_frag[idx_y] += dxy
                    if is_attr_mask[xdec]:
                        attr_frag[idx_x] += dxy
                    if is_attr_mask[ydec]:
                        attr_frag[idx_y] += dxy
    
        return basin_coh, basin_frag, attr_coh, attr_frag


    @njit(cache=True)
    def _robustness_edge_traversal_numba_stratified(
        N,
        attractor_idx,
        is_attr_mask,
        dist_attr,
        dist_state,
        max_dist
    ):
        n_states = attractor_idx.shape[0]
        n_attr = dist_attr.shape[0]
    
        basin_coh = np.zeros(n_attr, dtype=np.float64)
        basin_frag = np.zeros(n_attr, dtype=np.float64)
        attr_coh = np.zeros(n_attr, dtype=np.float64)
        attr_frag = np.zeros(n_attr, dtype=np.float64)
    
        strat_coh = np.zeros((n_attr, max_dist + 1), dtype=np.float64)
        strat_cnt = np.zeros((n_attr, max_dist + 1), dtype=np.int64)
    
        for xdec in range(n_states):
            dx = dist_state[xdec]
            idx_x = attractor_idx[xdec]
    
            for bit in range(N):
                if (xdec >> bit) & 1:
                    continue
    
                ydec = xdec | (1 << bit)
                dy = dist_state[ydec]
                idx_y = attractor_idx[ydec]
    
                strat_cnt[idx_x, dx] += 1
                strat_cnt[idx_y, dy] += 1
    
                if idx_x == idx_y:
                    basin_coh[idx_x] += 2.0
    
                    if is_attr_mask[xdec]:
                        attr_coh[idx_x] += 1.0
                    if is_attr_mask[ydec]:
                        attr_coh[idx_y] += 1.0
    
                    strat_coh[idx_x, dx] += 1.0
                    strat_coh[idx_y, dy] += 1.0
                else:
                    dxy = dist_attr[idx_x, idx_y]
                    basin_frag[idx_x] += dxy
                    basin_frag[idx_y] += dxy
    
                    if is_attr_mask[xdec]:
                        attr_frag[idx_x] += dxy
                    if is_attr_mask[ydec]:
                        attr_frag[idx_y] += dxy
    
        return (
            basin_coh,
            basin_frag,
            attr_coh,
            attr_frag,
            strat_coh,
            strat_cnt,
        )