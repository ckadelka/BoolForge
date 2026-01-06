#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the :class:`~boolforge.BooleanNetwork` class, which provides
a high-level framework for modeling, simulating, and analyzing Boolean networks.

A :class:`BooleanNetwork` represents a discrete dynamical system
:math:`F = (f_1, \\ldots, f_n)` composed of multiple
:class:`~boolforge.BooleanFunction` objects as update rules. The class includes
methods for constructing state transition graphs, identifying attractors,
computing robustness and sensitivity measures, and exporting truth tables.

Several computational routines—particularly those involving state space
exploration, attractor detection, and robustness estimation—offer optional
Numba-based just-in-time (JIT) acceleration. Installing Numba is **recommended**
for optimal performance but **not required**; all features remain functional
without it.

This module serves as the central interface for dynamic Boolean network
analysis within the BoolForge package.

Example
-------
>>> from boolforge import BooleanNetwork
>>> bn = BooleanNetwork(F = [[0,1],[0,0,0,1],[0,1]], I = [[1],[0,2],[1]])
>>> bn.get_attractors_synchronous_exact()
"""

import math
from collections import defaultdict
from copy import deepcopy

import numpy as np
import networkx as nx
import pandas as pd

from typing import Union, Optional

try:
    import boolforge.utils as utils
    from boolforge.boolean_function import BooleanFunction
except ModuleNotFoundError:
    import utils as utils
    from boolean_function import BooleanFunction
    
try:
    import cana.boolean_network
    __LOADED_CANA__=True
except ModuleNotFoundError:
    print('The module cana cannot be found. Ensure it is installed to use all functionality of this toolbox.')
    __LOADED_CANA__=False

try:
    from numba import njit, int64
    from numba.typed import List
    __LOADED_NUMBA__=True
except ModuleNotFoundError:
    print('The module numba cannot be found. Ensure it is installed to increase the run time of critical code in this toolbox.')
    __LOADED_NUMBA__=False


dict_weights = {'non-essential' : np.nan, 'conditional' : 0, 'positive' : 1, 'negative' : -1}

def get_entropy_of_basin_size_distribution(basin_sizes : Union[list, np.array]) -> float:
    """
    Compute the Shannon entropy of the basin size distribution.

    This function calculates the Shannon entropy of a probability distribution derived from the basin sizes.
    First, the basin sizes are normalized to form a probability distribution, and then the entropy is computed
    using the formula: H = - sum(p_i * log(p_i)), where p_i is the proportion of the basin size i.

    **Parameters:**
    
        - basin_sizes (list | np.array): A list where each element
          represents the size of a basin, i.e., the number of initial
          conditions that converge to a particular attractor.

    **Returns:**
    
        - float: The Shannon entropy of the basin size distribution.
    """
    total = sum(basin_sizes)
    probabilities = [size * 1.0 / total for size in basin_sizes]
    return sum([-np.log(p) * p for p in probabilities])


if __LOADED_NUMBA__:
    @njit(fastmath=True) #can safely use fastmath because computations are integers only
    def _update_network_synchronously_numba(x, 
                                            F_array_list, 
                                            I_array_list, 
                                            N):
        """
        Compute one synchronous network update for a given binary state vector x.
        Returns a new binary vector (uint8).
        """
        fx = np.empty(N, dtype=np.uint8)
        for j in range(N):
            regulators = I_array_list[j]
            if regulators.shape[0] == 0:
                fx[j] = F_array_list[j][0]
            else:
                n_reg = regulators.shape[0]
                idx = 0
                # convert substate bits → integer index
                for k in range(n_reg):
                    idx = (idx << 1) | x[regulators[k]]
                fx[j] = F_array_list[j][idx]
        return fx
    
    @njit
    def _compute_synchronous_stg_numba(F_list, 
                                       I_list, 
                                       N_variables):
        """
        Compute synchronous state transition graph (STG)
        in a fully numba-jitted function.
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
            regulators = I_list[j]
            if len(regulators) == 0:
                # constant node
                next_states[:, j] = F_list[j][0]
                continue
    
            n_reg = len(regulators)
            reg_powers = 2 ** np.arange(n_reg - 1, -1, -1)
            for s in range(nstates):
                idx = 0
                for k in range(n_reg):
                    idx += states[s, regulators[k]] * reg_powers[k]
                next_states[s, j] = F_list[j][idx]
    
        # Convert each next state to integer index
        next_indices = np.zeros(nstates, dtype=np.int64) # NOTE: this cannot be an unsigned int for safe indexing inside Numba kernels.
        for s in range(nstates):
            val = 0
            for j in range(N_variables):
                val += next_states[s, j] * powers_of_two[j]
            next_indices[s] = val
    
        return next_indices

    @njit    
    def _compute_synchronous_stg_numba_low_memory(F_array_list, 
                                                  I_array_list, 
                                                  N_variables):
        """
        Compute synchronous state transition graph (STG) without storing all states.
    
        For each integer state i in [0, 2^N):
          - decode i into its binary vector
          - compute its next state vector
          - encode back to integer
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

    @njit(fastmath=True) #can safely use fastmath because computations are integers only
    def _hamming_distance(a, b):
        """Fast Hamming distance for uint8 arrays."""
        dist = 0
        for i in range(a.size):
            dist += a[i] != b[i]
        return dist
    
    
    @njit(fastmath=True) #can safely use fastmath because computations are integers only
    def _derrida_simulation(F_array_list, 
                            I_array_list, 
                            N, 
                            nsim, 
                            seed):
        """
        Monte Carlo loop for Derrida value, using Numba-compatible RNG.
        """
        # Numba RNG: seed once
        np.random.seed(seed)
        total_dist = 0.0
    
        X = np.empty(N, dtype=np.uint8)
        Y = np.empty(N, dtype=np.uint8)
    
        for _ in range(nsim):
            # Random initial state
            for i in range(N):
                X[i] = np.random.randint(0, 2)
            Y[:] = X
    
            # Flip one random bit
            idx = np.random.randint(0, N)
            Y[idx] = 1 - Y[idx]
    
            # Synchronous updates
            FX = _update_network_synchronously_numba(X, F_array_list, I_array_list, N)
            FY = _update_network_synchronously_numba(Y, F_array_list, I_array_list, N)
    
            total_dist += _hamming_distance(FX, FY)
    
        return total_dist / nsim

    @njit(cache=True)
    def _attractors_functional_graph(next_state):
        """
        next_state: int array of length n with next_state[x] in [0, n-1]
        Returns:
            attr_id: int32 array length n, mapping each state -> attractor index
            basin_sizes: int32 array length n_attr
            cycle_rep: int64 array length n_attr, one representative node on each cycle
            cycle_len: int32 array length n_attr, cycle length
            n_attr: int32, number of attractors
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
            run_id = start + 1  # unique per start (int32 safe while n <= ~2e9)
    
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
    def _robustness_edge_traversal_numba(N, attractor_idx, is_attr_mask, dist_attr):
        """
        Core Numba kernel for Step 4 (hypercube edge traversal) using bit logic.
    
        Parameters
        ----------
        N : int
        attractor_idx : int32 array, shape (2**N,)
            attractor_idx[state] = attractor id in [0, A-1]
        is_attr_mask : uint8/bool array, shape (2**N,)
            1/True if state is in any attractor, else 0/False
        dist_attr : float64 array, shape (A, A)
            distance_between_attractors
    
        Returns
        -------
        basin_coh, basin_frag, attr_coh, attr_frag : float64 arrays, shape (A,)
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
    

class WiringDiagram(object):
    """
    A class representing a Wiring Diagram
    
    **Constructor Parameters:**

        - I (list[list[int]] | np.ndarray[list[int]]): A list of N lists
          representing the regulators (or inputs) for each Boolean function.

        - variables (list[str] | np.array[str], optional): A list of N strings
          representing the names of each variable, default = None.
          
        - weights (list[list[int | np.nan]]): #TODO

    **Members:**
        
        - I (list[np.array[int]]): As passed by the constructor.
        - variables (np.array[str]): As passed by the constructor.
        - N_variables (int): The number of variables in the Boolean network.
        - N_constants (int): The number of constants in the Boolean network.
        - N (int): The number of variables and constants in the Boolean network.
        - indegrees (list[int]): The indegrees for each node.
        - outdegrees (list[int]): The outdegrees of each node.
        - weights (list[list[int | np.nan]]): As passed by the constructor.
    """
    
    def __init__(self, 
                 I : Union[list, np.ndarray],
                 variables : Union[list, np.array, None] = None, 
                 weights = None):
        assert isinstance(I, (list, np.ndarray)), "I must be an array"
        #assert (len(I[i]) == ns[i] for i in range(len(ns))), "Malformed wiring diagram I"
        assert variables is None or len(I)==len(variables), "len(I)==len(variables) required if variable names are provided"
        assert weights is None or True, "weights assertion" # TODO: if weights are given, they must be valid
        
        self.I = [np.array(regulators,dtype=int) for regulators in I]
        self.N = len(I)
        self.indegrees = np.array(list(map(len, self.I)))
        
        if variables is None:
            variables = ['x'+str(i) for i in range(self.N)]
        
        self.N_constants = len(self.get_constants(False))
        self.N_variables = self.N - self.N_constants
        
        self.variables = np.array(variables)
        
        self.outdegrees = self.get_outdegrees()
        self.weights = weights

    @classmethod
    def from_DiGraph(cls, 
                     nx_DiGraph : "nx.DiGraph") -> "WiringDiagram":
        """
        **Compatibility Method:**
        
            Converts a `networkx.DiGraph` instance into a `WiringDiagram` object.
            Each node in the DiGraph represents a Boolean variable, and each
            directed edge (u → v) indicates that variable `u` regulates variable `v`.
        
        **Parameters:**
        
            - nx_DiGraph (nx.DiGraph): A directed graph where edges represent
              regulatory influences (u → v).
            
            - Node attributes (optional):
                
                - `'name'`: a string name of the variable (defaults to node label).
                - `'weight'`: numerical edge weights (stored in `weights` matrix, optional).
        
        **Returns:**
        
            - WiringDiagram: An instance constructed from the graph structure.
            
        **Example:**
        
            >>> import networkx as nx
            >>> G = nx.DiGraph()
            >>> G.add_edges_from([(0, 1), (1, 2), (2, 0)])
            >>> WD = WiringDiagram.from_DiGraph(G)
            >>> WD.I
            [array([2]), array([0]), array([1])]
            >>> WD.variables
            array(['x_0', 'x_1', 'x_2'], dtype='<U2')
        """
        # Ensure input is a DiGraph
        assert isinstance(nx_DiGraph, nx.DiGraph), "Input must be a networkx.DiGraph instance."
        
        # Sort nodes to ensure deterministic ordering
        nodes = list(nx_DiGraph.nodes)
        
        # Extract variable names, defaulting to "x0", "x1", ...
        variables = []
        for node in nodes:
            if 'name' in nx_DiGraph.nodes[node]:
                variables.append(str(nx_DiGraph.nodes[node]['name']))
            elif isinstance(node, str):
                variables.append(node)
            else:
                variables.append(f"x_{str(node)}")
    
        # Build regulator list I: for each node i, collect its predecessors (inputs)
        I = []
        for node in nodes:
            regulators = list(nx_DiGraph.predecessors(node))
            # Convert regulators to integer indices if nodes are not already 0..N-1
            if not all(isinstance(r, int) for r in regulators):
                regulators = [nodes.index(r) for r in regulators]
            I.append(regulators)
    
        # Optional: extract weights if available
        weights = None
        has_weights = all('weight' in nx_DiGraph[u][v] for u, v in nx_DiGraph.edges)
        if has_weights:
            weights = []
            for node in nodes:
                regs = list(nx_DiGraph.predecessors(node))
                w = [nx_DiGraph[u][node]['weight'] for u in regs]
                weights.append(w)
    
        # Instantiate WiringDiagram
        return cls(I=I, variables=variables, weights=weights)


    def to_DiGraph(self, 
                   USE_VARIABLE_NAMES : bool = True) -> nx.DiGraph:
        """
        Generate a NetworkX directed graph from a wiring diagram.

        Nodes are labeled with variable names (from variables) and constant
        names (from constants). Edges are added from each regulator to its
        target based on the wiring diagram I.

        **Parameters:**
            
            - USE_VARIABLE_NAMES (bool, optional): If True, nodes are labled using
            the variables names (default), otherwise indices are used.

        **Returns:**
            
            - networkx.DiGraph: The wiring diagram as directed graph.
        """
        G = nx.DiGraph()
        if USE_VARIABLE_NAMES:
            G.add_nodes_from(self.variables)
            G.add_edges_from([(self.variables[self.I[i][j]], self.variables[i]) for i in range(self.N) for j in range(self.indegrees[i])])
        else:
            G.add_nodes_from(range(self.N_variables))
            G.add_edges_from([(self.I[i][j], i) for i in range(self.N) for j in range(self.indegrees[i])])
        return G
    

    def __getitem__(self, index):
        return self.I[index]
    

    def get_outdegrees(self) -> np.array:
        """
        Returns the outdegree of each node.
        
        **Returns:**
            
            - np.array[int]: Outdegree of each node.
        """
        outdegrees = np.zeros(self.N, int)
        for regulators in self.I:
            for regulator in regulators:
                outdegrees[regulator] += 1
        return outdegrees


    def get_constants(self, 
                      AS_DICT : bool = True) -> Union[dict, np.array]:
        """
        Identify constants in a Boolean network.
        
        A node is considered a constant if it has no regulators.
        
        **Parameters:**
        
            - AS_DICT (bool, optional): Whether to return the indices of constants
              as a dictionary or array. If true, returns as a dictionary. Defaults
              to True.
        
        **Returns:**
        
            If AS_DICT is True:
                
                - dict[int:bool]: Dictionary determining if an index is a
                  constant or not.
                  
            else:
                - np.array[int]: Array of node indices that are constants.
        """
        rlI = range(len(self.I))
        is_constant = [self.indegrees[i] == 0 for i in rlI]
        if AS_DICT:
            return dict(zip(rlI, is_constant))
        return np.where(is_constant)[0]


    def get_strongly_connected_components(self) -> list:
        """
        Determine the strongly connected components of a wiring diagram.

        **Returns:**
            
            - list[set[int]]: A list of sets, each representing a strongly
              connected component.
        """
        edges_wiring_diagram = []
        for target, regulators in enumerate(self.I):
            for regulator in regulators:
                edges_wiring_diagram.append((int(regulator), target))
        subG = nx.from_edgelist(edges_wiring_diagram, create_using=nx.MultiDiGraph())
        return [scc for scc in nx.strongly_connected_components(subG)]


    def get_modular_structure(self):
        """
        Determine the modular structure of a Boolean network.

        The modular structure is defined by a directed acyclic graph (DAG) whose
        nodes are the strongly connected components (SCCs) of the underlying wiring
        diagram and whose directed edges indicate a regulation from one SCC to another SCC.

        **Returns:**
            
            - set[tuple[int]]: A set of edges, describing a directed acyclic graph
              indicating the regulations between modules (i.e., strongly connected
              components of the underlying wiring diagram).
        """
        sccs = self.get_strongly_connected_components()
        scc_dict = {}
        for j,s in enumerate(sccs):
            for el in s:
                scc_dict.update({el:j})
        dag = set()
        for target,regulators in enumerate(self.I):
            for regulator in regulators:
                edge = (scc_dict[regulator],scc_dict[target])
                if edge[0]!=edge[1] and (self.weights is None or not np.isnan(self.weights[target][list(self.weights[target]).index(regulator)])):
                    dag.add(edge)   
        return dag


    def get_ffls(self) -> Union[tuple, list]:
        """
        Identify feed-forward loops (FFLs) in a Boolean network based solely
        on the wiring diagram.

        The function uses the inverted wiring diagram to identify common
        targets and returns the FFLs found. If types_I (the type of each
        regulation) is provided, it also returns the corresponding regulation
        types.

        **Parameters:**
            
            - types_I (list[list[str]], optional): List of lists specifying
              the type (e.g., 'increasing' or 'decreasing') for each regulation.

        **Returns:**
            
            If self.weights is not None:
                
                - tuple[list[int], list[int]]: (ffls, types) where ffls is a
                  list of identified FFLs (each as a list [master regulator,
                  intermediate, target]), and types is a list of regulation type
                  triplets (master -> target, master -> intermediate,
                  intermediate -> target).
                
            Otherwise:
                
                - list[list[int]]: A list of identified FFLs.
        """
        I_inv = [[] for _ in range(self.N)]
        for target, regulators in enumerate(self.I):
            for regulator in regulators:
                I_inv[regulator].append(target)
        ffls = []
        types = []
        for i in range(self.N):  # master regulators
            for j in I_inv[i]:
                if i == j:
                    continue
                common_targets = list(set(I_inv[i]) & set(I_inv[j]))
                for k in common_targets:
                    if j == k or i == k:
                        continue
                    ffls.append([i, j, k])
                    if self.weights is not None:
                        direct = self.weights[k][list(self.I[k]).index(i)]
                        indirect1 = self.weights[j][list(self.I[j]).index(i)]
                        indirect2 = self.weights[k][list(self.I[k]).index(j)]
                        types.append([direct, indirect1, indirect2])
        if self.weights is not None:
            return (ffls, types)
        else:
            return ffls
        
        
    def get_fbls(self, max_length=4):
        """
        Compute all feedback loops (i.e., simple cycles) using a variant of Johnson's algorithm.
    
        This function finds simple cycles (elementary circuits) in the directed graph G
        with a maximum length of max_len. It first computes self-cycles (if any), 
        then iterates through the strongly connected components of G,
        recursively unblocking nodes to compute cycles.
    
        **Parameters:**
        
            - max_length (int, optional): Maximum length of cycles to consider (default is 4).
    
        **Returns:**
        
            - list[list[str]]: A list of lists, where each inner list represents a simple cycle.
        """
        G = self.to_DiGraph(USE_VARIABLE_NAMES=False)
    
        def _unblock(thisnode, blocked, B):
            stack = set([thisnode])
            while stack:
                node = stack.pop()
                if node in blocked:
                    blocked.remove(node)
                    stack.update(B[node])
                    B[node].clear()
    
        subG = nx.DiGraph(G.edges())
        sccs = [scc for scc in nx.strongly_connected_components(subG) if len(scc) > 1]
        
        fbls = []
        
        # Yield self-cycles and remove them.
        for v in subG:
            if subG.has_edge(v, v):
                fbls.append([v])
                subG.remove_edge(v, v)
        
        while sccs:
            scc = sccs.pop()
            sccG = subG.subgraph(scc)
            startnode = scc.pop()
            path = [startnode]
            len_path = 1
            blocked = set()
            closed = set()
            blocked.add(startnode)
            B = defaultdict(set)
            stack = [(startnode, list(sccG[startnode]))]
            while stack:
                thisnode, nbrs = stack[-1]
                if nbrs and len_path <= max_length:
                    nextnode = nbrs.pop()
                    if nextnode == startnode:
                        fbls.append( path[:] )
                        closed.update(path)
                    elif nextnode not in blocked:
                        path.append(nextnode)
                        len_path += 1
                        stack.append((nextnode, list(sccG[nextnode])))
                        closed.discard(nextnode)
                        blocked.add(nextnode)
                        continue
                if not nbrs or len_path > max_length:
                    if thisnode in closed:
                        _unblock(thisnode, blocked, B)
                    else:
                        for nbr in sccG[thisnode]:
                            if thisnode not in B[nbr]:
                                B[nbr].add(thisnode)
                    stack.pop()
                    path.pop()
                    len_path -= 1
            H = subG.subgraph(scc)
            sccs.extend(scc for scc in nx.strongly_connected_components(H) if len(scc) > 1)
        return fbls


    def get_types_of_fbls(self, fbls):
        if self.weights is None:
            return
        
        types = []
        n_negative_regulations_in_fbls = []
        for fbl in fbls:
            length = len(fbl)
            dummy = fbl[:]
            dummy.append(fbl[0])
            all_weights = []
            for i in range(length):
                all_weights.append( self.weights[dummy[i+1]][list(self.I[dummy[i+1]]).index(i)] )
            types.append(np.prod(all_weights))
            n_negative_regulations_in_fbls.append(sum([el==-1 for el in all_weights]))
                
        return types,n_negative_regulations_in_fbls



    def get_type_of_loop(self, loop : list) -> list:
        """
        Determine the regulation types along a feedback loop.

        For a given loop (a list of node indices), this function returns a
        list containing the type (e.g., 'increasing' or 'decreasing') of each
        regulation along the loop. The loop is assumed to be ordered such that
        the first node is repeated at the end.

        **Parameters:**
            
            - loop (list[int]): List of node indices representing the loop.

        **Returns:**
            
            - list[int]: A list of regulation types corresponding to each edge
              in the loop.
        """
        n = len(loop)
        dummy = loop[:]
        dummy.append(loop[0])
        res = []
        for i in range(n):
            # Assumes is_monotonic returns a tuple with the monotonicity information.
            #TODO: F does not exist here
            res.append(self.F[dummy[i+1]].is_monotonic(True)[1][list(self.I[dummy[i+1]]).index(dummy[i])])
        return res


class BooleanNetwork(WiringDiagram):
    """
    A class representing a Boolean network with N variables.
    
    **Constructor Parameters:**

        - F (list[BooleanFunction | list[int]] | np.ndarray[BooleanFunction |
          list[int]]): A list of N Boolean functions, or of N lists of length
          2^n representing the outputs of a Boolean function with n inputs.

        - I (list[list[int]] | np.ndarray[list[int]] | WiringDiagram):
          A list of N lists representing the regulators (or inputs) for each 
          Boolean function.

        - variables (list[str] | np.array[str], optional): A list of N strings
          representing the names of each variable, default = None.
          
        - SIMPLIFY_FUNCTIONS (bool, optional): Constructs this Boolean Network
          to only include its essential components. Defaults to False
          
    **Members:**
        
        - F (list[BooleanFunction]): As passed by the constructor.
        - I (list[np.array[int]]): As passed by the constructor.
        - variables (np.array[str]): As passed by the constructor.
        - N (int): The number of variables in the Boolean network.
        - N_constants (int): The number of constants in the Boolean network.
        - size (int): The number of variables and constants in the Boolean network.
        - indegrees (list[int]): The indegrees for each node.
        - outdegrees (list[int]): The outdegrees of each node.
        - STG (dict): The state transition graph.
        - weights (np.array[float] | None): Inherited from WiringDiagram. Default None.
    """

    def __init__(self, 
                 F : Union[list, np.ndarray], 
                 I : Union[list, np.ndarray, WiringDiagram],
                 variables : Union[list, np.array, None] = None,
                 SIMPLIFY_FUNCTIONS : Optional[bool] = False):
        assert isinstance(F, (list, np.ndarray)), "F must be an array or list."
        assert isinstance(I, (list, np.ndarray, WiringDiagram)), "I must be an array or list, or an instance of WiringDiagram."
        if isinstance(I, (list, np.ndarray)):
            super().__init__(I, variables)
        else:
            if variables is not None:
                print('Warning: Values of provided variables ignored. Variales of WiringDiagram I used instead.')
            super().__init__(I.I, I.variables)
        assert len(F)==self.N, "len(F)==len(I) required"
        
        self.F = []
        for ii,f in enumerate(F):
            if isinstance(f, (list, np.ndarray, str)):
                self.F.append(BooleanFunction(f,name = self.variables[ii]))
            elif isinstance(f, BooleanFunction):
                f.name = self.variables[ii]
                self.F.append(f)
            else:
                raise TypeError(f"F holds invalid data type {type(f)} : Expected either list, np.array, or BooleanFunction")
            assert self.F[ii].n == self.indegrees[ii], f"Index {ii}: Mismatch between the degree of the provided function {self.F[ii].n} and the degree of the wiring diagram {self.indegrees[ii]}."
        if not hasattr(self, 'constants'): #keeps track of all constants and nodes set to constants
            self.constants = {}
        if self.N_constants > 0:
            self.remove_constants()
        self.STG = None
        if SIMPLIFY_FUNCTIONS:
            self.simplify_functions() 

    def remove_constants(self, 
                         values_constants : Optional[list] = None) -> None:
        """
        Removes constants from this Boolean network.

        **Parameters:**
        
            - values_constants (list, optional): The values to fix for each constant
              node in the network. If None, takes the value provided by the constant
              function.
        """
        if values_constants is None:
            indices_constants = self.get_constants(AS_DICT=False)
            dict_constants = self.get_constants(AS_DICT=True)
            values_constants = [self.F[c][0] for c in indices_constants]
        else:
            indices_constants = self.get_source_nodes(AS_DICT=False) 
            dict_constants = self.get_source_nodes(AS_DICT=True)
            assert len(values_constants)==len(indices_constants),'The network contains {len(indices_constants)} source nodes but {len(values_constants)} values were provided.'
        #self.constants = dict(zip(self.variables[indices_constants],values_constants))
        for id_constant,value in zip(indices_constants,values_constants):
            regulated_nodes = []
            for i in range(self.N): # for all variables
                if dict_constants[i]:
                    continue
                try:
                    index = list(self.I[i]).index(id_constant) #check if the constant is part of regulators
                except ValueError:
                    continue
                truth_table = utils.get_left_side_of_truth_table(self.indegrees[i])
                indices_to_keep = np.where(truth_table[:,index]==value)[0]
                self.F[i].f = self.F[i].f[indices_to_keep]
                if self.weights is not None:
                    self.weights[i] = self.weights[i][self.I[i]!=id_constant]
                self.I[i] = self.I[i][self.I[i]!=id_constant]
                self.indegrees[i] -= 1
                self.F[i].n -= 1
                regulated_nodes.append(self.variables[i])
            self.constants[self.variables[id_constant]] = {'value' : value, 'regulatedNodes': regulated_nodes}
                
        for i in range(self.N): #check if any node has lost all its regulators, add an artificial non-essential regulation of the node by itself to avoid deletion of the node
            if dict_constants[i]:
                continue
            if self.indegrees[i] == 0:
                self.indegrees[i] = 1
                self.F[i].n = 1
                self.F[i].f = np.array([self.F[i][0],self.F[i][0]],dtype=int)
                self.I[i] = np.array([i],dtype=int)
                if self.weights is not None:
                    self.weights[i] = np.array([np.nan],dtype=int)
        self.F = [self.F[i] for i in range(self.N) if dict_constants[i]==False]
        adjustment_for_I = np.cumsum([dict_constants[i] for i in range(self.N)])
        self.I = [self.I[i]-adjustment_for_I[self.I[i]] for i in range(self.N) if dict_constants[i]==False]
        if self.weights is not None:
            self.weights = [self.weights[i] for i in range(self.N) if dict_constants[i]==False]
        self.variables = [self.variables[i] for i in range(self.N) if dict_constants[i]==False]
        self.outdegrees = [self.outdegrees[i] for i in range(self.N) if dict_constants[i]==False]
        self.indegrees = [self.indegrees[i] for i in range(self.N) if dict_constants[i]==False]
        self.N -= len(indices_constants)
        self.N_constants = len(self.constants)

    @classmethod
    def from_cana(cls, 
                  cana_BooleanNetwork : "cana.boolean_network.BooleanNetwork") -> "BooleanNetwork":
        """
        **Compatability Method:**
        
            Converts an instance of cana.boolean_network.BooleanNetwork from
            the cana module into a Boolforge BooleanNetwork object.
        
        **Returns**:
            
                - A BooleanNetwork object.
        """
        F = []
        I = []
        variables = []
        for entry in cana_BooleanNetwork.logic.values():
            try:
                variables.append(entry['name'])
            except KeyError:
                pass
            try:
                F.append(entry['out'])
                I.append(entry['in'])
            except KeyError:
                pass            
        return cls(F = F, I = I, variables=variables)

    @classmethod
    def from_string(cls, 
                    network_string : str, 
                    separator : Union[str, list, np.array] = ',', 
                    max_degree : int = 24, 
                    original_not : Union[str, list, np.array] = 'NOT', 
                    original_and : Union[str, list, np.array] = 'AND', 
                    original_or : Union[str, list, np.array] = 'OR') -> "BooleanNetwork":
        """
        **Compatability Method:**
        
            Converts a string into a Boolforge BooleanNetwork object.
        
        **Returns**:
            
                - A BooleanNetwork object.
        """
        sepstr, andop, orop, notop = "@", "∧", "∨", "¬"
        
        get_dummy_var = lambda i: "x%sy"%str(int(i))
        
        # reformat network string
        lines = network_string.replace('\t', ' ',).replace('(', ' ( ').replace(')', ' ) ')
        def __replace__(string, original, replacement):
            if isinstance(original, (list, np.ndarray)):
                for s in original:
                    string = string.replace(s, " %s "%replacement)
            elif isinstance(original, str):
                string = string.replace(original, " %s "%replacement)
            return string
        lines = __replace__(lines, separator, sepstr)
        lines = __replace__(lines, original_not, notop)
        lines = __replace__(lines, original_and, andop)
        lines = __replace__(lines, original_or, orop)
        
        lines = lines.splitlines()
        
        # remove empty lines
        while '' in lines:
            lines.remove('')
        
        # remove comments
        for i in range(len(lines)-1, -1, -1):
            if lines[i][0] == '#':
                lines.pop(i)
        
        n = len(lines)
        
        # find variables and constants
        var = ["" for i in range(n)]
        for i in range(n):
            var[i] = lines[i].split(sepstr)[0].replace(' ', '')
        consts_and_vars = []
        for line in lines:
            words = line.split(' ')
            for word in words:
                if word not in ['(', ')', sepstr, andop, orop, notop, ''] and not utils.is_float(word):
                    consts_and_vars.append(word)
        consts = list(set(consts_and_vars)-set(var))
        dict_var_const = dict(list(zip(var, [get_dummy_var(i) for i in range(len(var))])))
        dict_var_const.update(dict(list(zip(consts, [get_dummy_var(i+len(var)) for i in range(len(consts))]))))
        
        # replace all variables and constants with dummy names
        for i, line in enumerate(lines):
            words = line.split(' ')
            for j, word in enumerate(words):
                if word not in ['(', ')', sepstr, andop, orop, notop, ''] and not utils.is_float(word):
                    words[j] = dict_var_const[word]
            lines[i] = ' '.join(words)
        
        # update line to only be function
        for i in range(n):
            lines[i] = lines[i].split(sepstr)[1]
        
        # generate wiring diagram I
        I = []
        for i in range(n):
            try:
                idcs_open = utils.find_all_indices(lines[i], 'x')
                idcs_end = utils.find_all_indices(lines[i], 'y')
                regs = np.sort(np.array(list(map(int,list(set([lines[i][(begin+1):end] for begin,end in zip(idcs_open,idcs_end)]))))))
                I.append(regs)
            except ValueError:
                I.append(np.array([], int))
        
        deg = list(map(len, I))
        
        # generate functions F
        F = []
        for i in range(n):
            if deg[i] == 0:
                f = np.array([int(lines[i])], int)
            elif deg[i] <= max_degree:
                tt = utils.get_left_side_of_truth_table(deg[i])
                ldict = { get_dummy_var(I[i][j]) : tt[:, j].astype(bool) for j in range(deg[i]) }
                f = eval(lines[i].replace(andop, '&').replace(orop, '|').replace(notop, '~').replace(' ', ''), {"__builtins__" : None}, ldict)
            else:
                f = np.array([], int)
            F.append(f.astype(int))
        for i in range(len(consts)):
            F.append(np.array([0, 1], int))
            I.append(np.array([len(var) + i]))
        
        return cls(F, I, var+consts)


    @classmethod
    def from_DiGraph(cls, 
                     nx_DiGraph : "nx.DiGraph") -> "WiringDiagram":
        raise NotImplementedError("from_DiGraph is not supported in BooleanNetwork class.")
    
    
    def to_cana(self) -> "cana.boolean_network.BooleanNetwork":
        """
        **Compatability method:**
        
            Returns an instance of the class cana.BooleanNetwork from the
            cana module.

        **Returns:**
            
            - An instance of cana.boolean_network.BooleanNetwork
        """
        logic_dicts = []
        for bf,regulators,var in zip(self.F,self.I,self.variables):
            logic_dicts.append({'name':var, 'in': list(regulators), 'out': list(bf.f)})
        return cana.boolean_network.BooleanNetwork(Nnodes = self.N, logic = dict(zip(range(self.N),logic_dicts))) 

    def to_bnet(self, 
                separator=',\t', 
                AS_POLYNOMIAL : bool = True) -> str:
        """
        **Compatability method:**
            
            Returns a bnet string formatted as a polynomial.
        
        **Parameters:**

            - separator (str): A string used to separate the target variable
              from the function. Defaults to ',\t'.
              
            - AS_POLYNOMIAL (bool, optional): Determines whether to return
              the function as a polynomial or logical expression. If true,
              returns as a polynomial, and if false, returns as a logical
              expression. Defaults to true.
            
        **Returns:**
            
            - str: A string describing a bnet.
        """
        lines = []
        constants_indices = self.get_constants()
        for i in range(self.N):
            if constants_indices[i]:
                function = str(self.F[i].f[0])
            elif AS_POLYNOMIAL:
                function = utils.bool_to_poly(self.F[i], self.variables[self.I[i]])
            else:
                function = self.F[i].to_expression(" & ", " | ")
            lines.append(f'{self.variables[i]}{separator}{function}')
        return '\n'.join(lines)
    
    def to_truth_table(self, 
                       RETURN : bool = True, 
                       filename : str = None) -> pd.DataFrame:
        """
        Determines the full truth table of the Boolean network as pandas DataFrame.

        Each row shows the input combination (x1, x2, ..., xN)
        and the corresponding output(s) f(x).
        
        The output is returned as a pandas DataFrame and can optionally be
        exported to a file in CSV or Excel format.

        **Parameters:**
        
            - RETURN (bool, optional):
              Whether to return the truth table as a pandas DataFrame.
              Defaults to True.
            
            - filename (str, optional):
              If provided, the truth table is written to a file. The file
              extension determines the format and must be one of:
              `'csv'`, `'xls'`, or `'xlsx'`.
              Example: `"truth_table.csv"` or `"truth_table.xlsx"`.
              If `None` (default), no file is created.

        **Returns:**
            
            - pd.DataFrame: The full truth table with shape (2^N, 2N).
              Returned only if `RETURN=True`.
              
        **Notes:**
        
            - The function automatically computes the synchronous
              state transition graph (`STG`) if it has not been computed yet.
              
            - Each output row represents a deterministic transition from the
              current state to its next state under synchronous updating.
              
            - Exporting to Excel requires the `openpyxl` package to be installed.
        """
        
        columns = [name + '(t)' for name in self.variables]
        columns += [name + '(t+1)' for name in self.variables]
        if self.STG is None:
            self.compute_synchronous_state_transition_graph()
        data = np.zeros((2**self.N,2*self.N),dtype=int)
        data[:,:self.N] = utils.get_left_side_of_truth_table(self.N)
        for i in range(2**self.N):
            data[i,self.N:] = utils.dec2bin(self.STG[i],self.N)
        truth_table = pd.DataFrame(data,columns=columns)
        
        if filename is not None:
            ending = filename.split('.')[-1]
            assert ending in ['csv','xls','xlsx'],"filename must end in 'csv','xls', or 'xlsx'"
            if ending == 'csv':
                truth_table.to_csv(filename)
            else:
                truth_table.to_excel(filename)
        if RETURN:
            return truth_table
    
    def __len__(self):
        return self.N
    
    
    def __str__(self):
        return f"Boolean network of {self.N} nodes with indegrees {self.indegrees}"
    
    
    def __getitem__(self, index):
        return self.F[index]
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    
    def __call__(self, state):
        """
        Perform a synchronous update of a Boolean network.

        Each node's new state is determined by applying its Boolean function
        to the current states of its regulators.

        **Parameters:**
            
            - X (list[int] | np.array[int]): Current state vector of the network.

        **Returns:**
            
            - np.array[int]: New state vector after the update.
        """
        return self.update_network_synchronously(state)
    
    def get_types_of_regulation(self) -> np.array:
        """
        Computes the weights of this Boolean network and assigns them to the
        weights member variable.
        
        **Returns:**
        
            - weights (np.array): The weights of this network.
        """
        weights = []
        for bf in self.F:
            weights.append(np.array([dict_weights[el] for el in bf.get_type_of_inputs()]))
        self.weights = weights
        return weights
    



    ## Transform Boolean networks
    def simplify_functions(self) -> None:
        """
        Remove all non-essential inputs, i.e., inoperative edges from the Boolean network.

        For each node in a Boolean network, represented by its Boolean function
        and its regulators, this function extracts the “essential” part of the
        function by removing non-essential regulators. The resulting network
        contains, for each node, a reduced truth table (with only the essential
        inputs) and a corresponding list of essential regulators.

        **Returns:**
            
            - BooleanNetwork: A Boolean network object where:
                
                - F is a list of N Boolean functions containing functions of
                  length 2^(m_i), with m_i ≤ n_i, representing the functions
                  restricted to the essential regulators.
                  
                - I is a list of N lists containing the indices of the
                  essential regulators for each node.
        """
        self.get_types_of_regulation() #ensuring that self.weights is updated
        for i in range(self.N):
            regulator_is_non_essential = np.isnan(self.weights[i])
            if sum(regulator_is_non_essential)==0: #all variables are essential, nothing to change
                continue
            
            non_essential_variables = np.where(regulator_is_non_essential)[0]
            essential_variables = np.where(~regulator_is_non_essential)[0]
            self.outdegrees[non_essential_variables] -= 1
            if len(essential_variables)==0: #no variables are essential, introduce ``fake" auto-regulation to keep this variable and do not delete it as a constant
                self.indegrees[i] = 1
                self.F[i].f = np.array([self.F[i][0],self.F[i][0]],dtype=int)
                self.F[i].n = 1
                self.F[i].variables = self.variables[i]
                self.I[i] = np.array([i],dtype=int)
                self.weights[i] = np.array([np.nan],dtype=float)
                self.outdegrees[i] += 1 #add this, even though it's a fake regulation to keep sum(self.outdegrees)==sum(self.indegrees)
                continue
            
            left_side_of_truth_table = utils.get_left_side_of_truth_table(self.indegrees[i])
            self.F[i].f = self.F[i][np.sum(left_side_of_truth_table[:, non_essential_variables], 1) == 0]
            self.F[i].n = len(essential_variables)
            self.F[i].variables = self.F[i].variables[~regulator_is_non_essential]
            self.I[i] = self.I[i][essential_variables]
            self.weights[i] = self.weights[i][essential_variables]
            self.indegrees[i] = len(essential_variables)


    def get_source_nodes(self, 
                         AS_DICT : bool = False) -> Union[dict, np.array]:
        """
        Identify source nodes in a Boolean network.
        
        A node is considered a source node if it does not change over time. It has
        exactly one regulator and that regulator is the node itself.        
        
        **Parameters:**
        
            - AS_DICT (bool, optional): Whether to return the indices of source nodes
              as a dictionary or array. If true, returns as a dictionary. Defaults
              to False.
        
        **Returns:**
        
            If AS_DICT is True:
                
                - dict[int:bool]: Dictionary determining if an index is a
                  source nodes or not.
                  
            else:
                - np.array[int]: Array of all indices of source nodes.
        """

        rlI = range(self.N)
        is_source_node = [self.indegrees[i] == 1 and self.I[i][0] == i and self.F[i][0]==0 and self.F[i][1]==1 for i in rlI]
        if AS_DICT:
            return dict(zip(rlI, is_source_node))
        return np.where(is_source_node)[0]

    
    def get_network_with_fixed_source_nodes(self, 
                                            values_source_nodes : Union[list, np.array]) -> "BooleanNetwork":
        """
        Fix the values of source nodes within this Boolean Network.

        **Parameters:**
        
            - values_source_nodes (list | np.array): The values to fix for each
              source node within this network. Must be of length equivalent to
              the number of source nodes in the network, and each element must
              be either 0 or 1.

        **Returns:**
        
            - BooleanNetwork: A BooleanNetwork object with fixed source nodes.
        """
        indices_source_nodes = self.get_source_nodes(AS_DICT=False)
        assert len(values_source_nodes)==len(indices_source_nodes),f"The length of 'values_source_nodes', which is {len(values_source_nodes)}, must equal the number of source nodes, which is {len(indices_source_nodes)}."
        assert set(values_source_nodes) <= {0,1},"Controlled node values must be 0 or 1."
        F = deepcopy(self.F)
        I = deepcopy(self.I)
        for source_node,value in zip(indices_source_nodes,values_source_nodes):
            F[source_node].f = [value]
            I[source_node] = []
        bn = self.__class__(F, I, self.variables)
        bn.constants.update(self.constants)
        return bn

    def get_network_with_node_controls(self,
                                       indices_controlled_nodes : Union[list, np.array], 
                                       values_controlled_nodes : Union[list, np.array],
                                       KEEP_CONTROLLED_NODES : bool = False) -> "BooleanNetwork":
        """
        Fix the values of nodes within this BooleanNetwork.
        
        **Parameters:**
        
            - indices_controlled_nodes (list | np.array): The indices of the nodes
              to fix the value of.
              
            - values_controlled_nodes : (list | np.array): The values to fix for
              each specified node in the network.
            
            - KEEP_CONTROLLED_NODES : (bool, optional): Whether to turn controlled
              nodes into constants or not. If true, controlled nodes become constants
              and will be baked into the network. If false, they will not be considered
              as constants. Defaults to false.
        
        **Returns:**
        
            - BooleanNetwork: A BooleanNetwork object with specified nodes controlled.
        """
        assert len(values_controlled_nodes)==len(indices_controlled_nodes),f"The length of 'values_controlled_nodes', which is {len(values_controlled_nodes)}, must equal the length of 'indices_controlled_nodes', which is {len(indices_controlled_nodes)}."
        assert set(values_controlled_nodes) <= {0,1},"Controlled node values must be 0 or 1."
        F = deepcopy(self.F)
        I = deepcopy(self.I)
        for node,value in zip(indices_controlled_nodes,values_controlled_nodes):
            if KEEP_CONTROLLED_NODES:
                F[node].f = [value,value]
                I[node] = [node]        
            else:
                F[node].f = [value]
                I[node] = []
        bn = self.__class__(F, I, self.variables)
        if not KEEP_CONTROLLED_NODES:
            bn.constants.update(self.constants)
        return bn


    def get_network_with_edge_controls(self, 
                                       control_targets : Union[int,list,np.array], 
                                       control_sources : Union[int,list,np.array], 
                                       type_of_edge_controls : Union[int,list,np.array,None] = None) -> "BooleanNetwork":
        """
        Generate a perturbed Boolean network by removing the influence of
        specified regulators on specified targets.

        The function modifies the Boolean function for target nodes by
        restricting it to those entries in its truth table where the input
        from given regulators equals the specified type_of_control. The
        regulators are then removed from the wiring diagram for that node.

        **Parameters:**
            
            - control_targets (int | list[int] | np.array[int]): 
              Index of the target node(s) to be perturbed.
                
            - control_sources (int | list[int] | np.array[int]): 
              Index of the regulator(s) whose influence is to be fixed.
              
            - type_of_edge_controls (int | list[int] | np.array[int]) | None): 
              Source value in regulation of target after control. 
              Default is None (which is interpreted as 0).

        **Returns:**
            
            - BooleanNetwork object where:
                
                - F is the updated list of Boolean functions after perturbation.
                - I is the updated wiring diagram after removing the control
                  regulator from the target node.
        """

        # Normalize arguments to lists
        if np.isscalar(control_targets):
            control_targets = [control_targets]
            control_sources = [control_sources]
            type_of_edge_controls = [0 if type_of_edge_controls is None else type_of_edge_controls]
        elif type_of_edge_controls is None:
            type_of_edge_controls = [0] * len(control_targets)
    
        assert len(control_targets) == len(control_sources) == len(type_of_edge_controls), \
            "control_targets, control_sources, and type_of_edge_controls must have equal length."

        F_new = deepcopy(self.F)
        I_new = deepcopy(self.I)
        indegrees = np.copy(self.indegrees)

        for target, source, fixed_value in zip(control_targets, control_sources, type_of_edge_controls):
            assert fixed_value in [0, 1], f"type_of_edge_control must be 0 or 1 (got {fixed_value})."
            assert source in I_new[target], f"control_source={source} not in regulators of target={target}"
            idx_reg = list(I_new[target]).index(source)
            n_inputs = indegrees[target]
    
            # Compute bitmask indices efficiently
            indices = np.arange(2 ** n_inputs, dtype=np.uint32)
            mask = ((indices >> (n_inputs - 1 - idx_reg)) & 1) == fixed_value
            F_new[target] = F_new[target][mask]
    
            # Remove the regulator
            I_new[target] = np.delete(I_new[target], idx_reg)
            indegrees[target] -= 1
        
        return self.__class__(F_new, I_new, self.variables)

            
    
    def update_single_node(self, 
                           index : int, 
                           states_regulators : Union[list, np.array]) -> int:
        """
        Update the state of a single node.

        The new state is obtained by applying the Boolean function f to the
        states of its regulators. The regulator states are converted to a
        decimal index using utils.bin2dec.

        **Parameters:**
            
            - index (int): The index of the Boolean Function in F.
            - states_regulators (list[int] | np.array[int]): Binary vector
              representing the states of the node's regulators.

        **Returns:**
            
            - int: Updated state of the node (0 or 1).
        """
        return self.F[index].f[utils.bin2dec(states_regulators)].item()


    def update_network_synchronously(self, 
                                     X : Union[list, np.array]) -> np.array:
        """
        Perform a synchronous update of a Boolean network.

        Each node's new state is determined by applying its Boolean function
        to the current states of its regulators.

        **Parameters:**
            
            - X (list[int] | np.array[int]): Current state vector of the network.

        **Returns:**
            
            - np.array[int]: New state vector after the update.
        """
        if type(X)==list:
            X = np.array(X)
        Fx = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            Fx[i] = self.update_single_node(index = i, states_regulators = X[self.I[i]])
        return Fx


    def update_network_SDDS(self, 
                            X : Union[list, np.array], 
                            P : np.ndarray, 
                            *, 
                            rng=None) -> np.array:
        """
        Perform a stochastic update (SDDS) on a Boolean network.

        For each node, the next state is computed as nextstep = F[i] evaluated
        on the current states of its regulators. If nextstep > X[i], the node
        is activated with probability P[i,0]; if nextstep < X[i], the node is
        degraded with probability P[i,1]. Otherwise, the state remains unchanged.

        **Parameters:**
            
            - X (list[int] | np.array[int]): Current state vector.
            - P (np.array[float]): A len(F)×2 array of probabilities; for each
              node i, P[i,0] is the activation probability, and P[i,1] is the
              degradation probability.
            
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - np.array[int]: Updated state vector after applying the
              stochastic update.
        """
        rng = utils._coerce_rng(rng)
        if type(X)==list:
            X = np.array(X)
        Fx = X.copy()
        for i in range(self.N):
            nextstep = self.update_single_node(index = i, states_regulators = X[self.I[i]])
            if nextstep > X[i] and rng.random() < P[i, 0]:  # activation
                Fx[i] = nextstep
            elif nextstep < X[i] and rng.random() < P[i, 1]:  # degradation
                Fx[i] = nextstep
        return Fx


    def get_steady_states_asynchronous_exact(self,
                                             stochastic_weights : Union[list, np.array, None] = None,
                                             max_iterations: int = 1000,
                                             tol=1e-9):
        """
        Compute exhaustively the steady states of a Boolean network under general asynchronous update.

        This function simulates asynchronous updates of a Boolean network
        (with N nodes) for a given number of initial conditions (nsim). For
        each initial state, the network is updated asynchronously until a
        steady state (or attractor) is reached or until a maximum search depth
        is exceeded. The simulation starts from nsim random initial conditions.

        **Parameters:**
            
            - stochastic_weights (list, np.array, None): The propensity of update 
              for each node. If None (default), uniform update probabilities are assumed.
              
            - max_iterations (int): The maximal number of iterations of updates 
              before raising a convergence error. These errors occur if the network contains
              non-steady state attractors.
              
            - tol (float): Maximal allowable error before convergence is declared.
            

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - SteadyStates (list[int]): List of steady state
                  values (in decimal form) found.
                  
                - NumberOfSteadyStates (int): Total number of unique steady states.
                - BasinSizes (list[int]): List of counts showing how many
                  initial conditions converged to each steady state.
                  
                - STGAsynchronous (dict[tuple(int, int):int]):
                  The asynchronous state transition graph. 
                  STGAsynchronous[(a,i)] = c implies that state a transitions
                  to state c when the ith variable is updated. Here, a and c
                  are decimal representations of the state and i is in {0, 1,
                  ..., self.N-1}.
                  
                - FinalTransitionProbabilities (np.array[float]): The final transition
                  probability for each state in the system. Each row is a probability distribution.
        """

        left_side_of_truth_table = utils.get_left_side_of_truth_table(self.N)

        assert stochastic_weights is None or len(stochastic_weights) == self.N and min(stochastic_weights)>0, "one positive weight per node is required"    
        if stochastic_weights is None:
            stochastic_weights = np.ones(self.N,dtype=float)/self.N
        else:
            stochastic_weights = np.array(stochastic_weights) / sum(stochastic_weights)            
        
        steady_states = []
        steady_state_dict = {}
        STG = dict(zip(range(2**self.N),[{} for i in range(2**self.N)]))
        sped_up_STG = dict(zip(range(2**self.N),[[np.zeros(0,dtype=int),np.zeros(0,dtype=float)] for i in range(2**self.N)]))
        for xdec in range(2**self.N):
            x = left_side_of_truth_table[xdec].copy() #important: must create a copy here!
            to_be_distributed = 0
            for i in range(self.N):
                fx_i = self.update_single_node(i, x[self.I[i]])
                if fx_i > x[i]:
                    fxdec = xdec + 2**(self.N - 1 - i)
                elif fx_i < x[i]:
                    fxdec = xdec - 2**(self.N - 1 - i)
                else:
                    fxdec = xdec
                if fxdec in STG[xdec]:
                    STG[xdec][fxdec] += stochastic_weights[i]
                else:
                    STG[xdec][fxdec] = stochastic_weights[i]
                if fxdec!=xdec:
                    sped_up_STG[xdec][0] = np.append(sped_up_STG[xdec][0], fxdec)
                    sped_up_STG[xdec][1] = np.append(sped_up_STG[xdec][1], stochastic_weights[i])
                else:
                    to_be_distributed += stochastic_weights[i]
            sped_up_STG[xdec][1] /= (1-to_be_distributed)
            if len(STG[xdec])==1:
                steady_state_dict[xdec] = len(steady_states)
                steady_states.append(xdec)
                sped_up_STG[xdec][0] = np.append(sped_up_STG[xdec][0], xdec)
                sped_up_STG[xdec][1] = np.append(sped_up_STG[xdec][1], 1)
                
        # Probability vectors for all states
        final_probabilities = np.zeros((2**self.N, len(steady_states)), dtype=float)
    
        # Boundary conditions: absorbing states have probability 1 of themselves
        for xdec in steady_states:
            final_probabilities[xdec, steady_state_dict[xdec]] = 1.0
        transient_states = [xdec for xdec in range(2**self.N) if xdec not in steady_state_dict]
        
        for it in range(1, max_iterations + 1):
            max_delta = 0.0
    
            # In-place Gauss–Seidel  update:
            for xdec in transient_states:
                nxt, pr = sped_up_STG[xdec]

                old = final_probabilities[xdec].copy()
                final_probabilities[xdec] = np.dot(pr, final_probabilities[nxt, :])   # weighted average of successor probability vectors
    
                # track convergence (infinity norm per row)
                delta = np.max(np.abs(final_probabilities[xdec] - old))
                if delta > max_delta:
                    max_delta = delta
    
            if max_delta < tol:
                basin_sizes = final_probabilities.sum(0)/2**self.N
                
                        
                return dict(zip(["SteadyStates", "NumberOfSteadyStates", "BasinSizes", "STGAsynchronous", "FinalTransitionProbabilities"],
                                (steady_states, len(steady_states), basin_sizes, STG, final_probabilities)))
            
        raise RuntimeError(f"Did not converge in {max_iterations} iterations; last max_delta={max_delta:g}")
        

    def get_steady_states_asynchronous(self,
                                       nsim : int = 500,
                                       initial_sample_points : list = [], 
                                       search_depth : int = 50, 
                                       DEBUG : bool = False, 
                                       *, 
                                       rng=None) -> dict:
        """
        Compute the steady states of a Boolean network under asynchronous updates.

        This function simulates asynchronous updates of a Boolean network
        (with N nodes) for a given number of initial conditions (nsim). For
        each initial state, the network is updated asynchronously until a
        steady state (or attractor) is reached or until a maximum search depth
        is exceeded. The simulation starts from nsim random initial conditions.

        **Parameters:**
            
            - nsim (int, optional): Number of initial conditions to simulate
              (default is 500).
              
            - initial_sample_points (list[list[int]], optional): List of
              initial states (as binary vectors) to use. If provided and EXACT
              is False, these override random sampling.
              
            - search_depth (int, optional): Maximum number of asynchronous
              update iterations to attempt per simulation.
              
            - DEBUG (bool, optional): If True, print debugging information
              during simulation.
            
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - SteadyStates (list[int]): List of steady state
                  values (in decimal form) found.
                  
                - NumberOfSteadyStates (int): Total number of unique steady states.
                - BasinSizes (list[int]): List of counts showing how many
                  initial conditions converged to each steady state.
                  
                - STGAsynchronous (dict[tuple(int, int):int]):
                  The asynchronous state transition graph. 
                  STGAsynchronous[(a,i)] = c implies that state a transitions
                  to state c when the ith variable is updated. Here, a and c
                  are decimal representations of the state and i is in {0, 1,
                  ..., self.N-1}.
                  
                - InitialSamplePoints (list[int]): The list of initial sample
                  points used (if provided) or those generated during simulation.
        """
        rng = utils._coerce_rng(rng)

        sampled_points = []
        
                
        STG_asynchronous = dict()
        steady_states = []
        basin_sizes = []
        steady_state_dict = dict()   
        
        for iteration in range(nsim):
            if initial_sample_points == []:  # generate random initial states on the fly
                x = rng.integers(2, size=self.N)
                xdec = utils.bin2dec(x)
                sampled_points.append(xdec)
            else:                
                x = initial_sample_points[iteration]
                xdec = utils.bin2dec(x)
            
            if DEBUG:
                print(iteration, -1, -1, False, xdec, x)
            for jj in range(search_depth):  # update until a steady state is reached or search_depth is exceeded
                FOUND_NEW_STATE = False
                try:
                    # Check if this state is already recognized as a steady state.
                    index_ss = steady_state_dict[xdec]
                except KeyError:
                    # Asynchronously update the state until a new state is found.
                    update_order_to_try = rng.permutation(self.N)
                    for i in update_order_to_try:
                        try:
                            fxdec = STG_asynchronous[(xdec, i.item())]
                            if fxdec != xdec:
                                FOUND_NEW_STATE = True
                                x[i] = 1 - x[i]
                        except KeyError:
                            fx_i = self.update_single_node(i, x[self.I[i]])
                            if fx_i > x[i]:
                                fxdec = xdec + 2**(self.N - 1 - i.item())
                                x[i] = 1
                                FOUND_NEW_STATE = True
                            elif fx_i < x[i]:
                                fxdec = xdec - 2**(self.N - 1 - i.item())
                                x[i] = 0
                                FOUND_NEW_STATE = True
                            else:
                                fxdec = xdec
                            STG_asynchronous.update({(xdec, i.item()): fxdec})
                        if FOUND_NEW_STATE:
                            xdec = fxdec
                            break
                    if DEBUG:
                        print(iteration, jj, i, FOUND_NEW_STATE, xdec, x)
                if FOUND_NEW_STATE == False:  # steady state reached
                    try:
                        index_ss = steady_state_dict[xdec]
                        basin_sizes[index_ss] += 1
                        break
                    except KeyError:
                        steady_state_dict.update({xdec: len(steady_states)})
                        steady_states.append(xdec)
                        basin_sizes.append(1)
                        break
            if DEBUG:
                print()
        if sum(basin_sizes) < nsim:
            print('Warning: only %i of the %i tested initial conditions eventually reached a steady state. Try increasing the search depth. '
                  'It may however also be the case that your asynchronous state space contains a limit cycle.' %
                  (sum(basin_sizes), nsim))
        return dict(zip(["SteadyStates", "NumberOfSteadyStates", "BasinSizes", "STGAsynchronous", "InitialSamplePoints"],
                        (steady_states, len(steady_states), basin_sizes, STG_asynchronous,
                initial_sample_points if initial_sample_points != [] else sampled_points)))


    def get_steady_states_asynchronous_given_one_initial_condition(self,
        initial_condition : Union[int, list, np.array] = 0,
        nsim : int = 500, 
        stochastic_weights : Union[list, np.array, None] = None, 
        search_depth : int = 50,
        DEBUG : bool = False, 
        *, 
        rng = None) -> dict:
        """
        Determine the steady states reachable from one initial condition using
        weighted asynchronous updates.

        This function is similar to steady_states_asynchronous_given_one_IC but
        allows the update order to be influenced by provided stochastic weights
        (one per node). A weight vector (of length N) may be provided, and if
        given, it is used to select the next node to be updated.

        **Parameters:**

            - initial_condition (int | list[int] | np.array[int], optional):
              The initial state for all simulations. If an integer, it is
              converted to a binary vector. Default is 0.
              
            - nsim (int, optional): Number of simulation runs (default is 500).
            
            - stochastic_weights (list[float], optional): List of stochastic
              weights (one per node) used to bias update order. If None,
              nodes to be updated are chosen uniformly at random.
              
            - search_depth (int, optional): Maximum number of asynchronous
              update iterations per simulation (default is 50).
              
            - DEBUG (bool, optional): If True, print debugging information
              (default is False).
              
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - SteadyStates (list[int]): List of steady state values (in
                  decimal form) reached.
                
                - NumberOfSteadyStates (int): Total number of unique steady states.
                
                - BasinSizes (list[int]): List of counts of how many
                  simulations reached each steady state.
                  
                - TransientTimes (list[list[int]]): List of lists with
                  transient times (number of updates) for each steady state.
                  
                - STGAsynchronous (dict[tuple(int, int):int]):
                  A sample of the asynchronous state transition graph. 
                  STGAsynchronous[(a,i)] = c implies that state a transitions
                  to state c when the ith variable is updated. Here, a and c
                  are decimal representations of the state and i is in {0, 1,
                  ..., self.N-1}.
                  
                - UpdateQueues (list[list[int]]): List of state update queues
                  (the sequence of states encountered) for each simulation.
        """
        rng = utils._coerce_rng(rng)
        
        if type(initial_condition) == int:
            initial_condition = np.array(utils.dec2bin(initial_condition, self.N))
            initial_condition_bin = utils.bin2dec(initial_condition)
        else:
            initial_condition = np.array(initial_condition, dtype=int)
            initial_condition_bin = utils.bin2dec(initial_condition)
        
        assert stochastic_weights is None or len(stochastic_weights) == self.N, "one stochastic weight per node is required"    
        if stochastic_weights is not None:
            stochastic_weights = np.array(stochastic_weights) / sum(stochastic_weights)
        
        STG_async = dict()
        steady_states = []
        basin_sizes = []
        transient_times = []
        steady_state_dict = dict()   
        queues = []
        for iteration in range(nsim):
            x = initial_condition.copy()
            xdec = initial_condition_bin
            queue = [xdec]
            for jj in range(search_depth):  # update until a steady state is reached or search_depth is exceeded
                FOUND_NEW_STATE = False
                try:
                    index_ss = steady_state_dict[xdec]
                except KeyError:
                    if stochastic_weights != []:
                        update_order_to_try = rng.choice(self.N, size=self.N, replace=False, p=stochastic_weights)
                    else:
                        update_order_to_try = rng.permutation(self.N)
                    for i in update_order_to_try:
                        try:
                            fxdec = STG_async[(xdec, i.item())]
                            if fxdec != xdec:
                                FOUND_NEW_STATE = True
                                x[i] = 1 - x[i]
                        except KeyError:
                            fx_i = self.update_single_node(i, x[self.I[i]])
                            if fx_i > x[i]:
                                fxdec = xdec + 2**(self.N - 1 - i.item())
                                x[i] = 1
                                FOUND_NEW_STATE = True
                            elif fx_i < x[i]:
                                fxdec = xdec - 2**(self.N - 1 - i.item())
                                x[i] = 0
                                FOUND_NEW_STATE = True
                            else:
                                fxdec = xdec
                            STG_async.update({(xdec, i.item()): fxdec})
                        if FOUND_NEW_STATE:
                            xdec = fxdec
                            queue.append(xdec)
                            break
                    if DEBUG:
                        print(iteration, jj, i, FOUND_NEW_STATE, xdec, x)
                if not FOUND_NEW_STATE:  # steady state reached
                    queues.append(queue[:])
                    try:
                        index_ss = steady_state_dict[xdec]
                        basin_sizes[index_ss] += 1
                        transient_times[index_ss].append(jj)
                        break
                    except KeyError:
                        steady_state_dict.update({xdec: len(steady_states)})
                        steady_states.append(xdec)
                        basin_sizes.append(1)
                        transient_times.append([jj])
                        break
            if FOUND_NEW_STATE:
                print(jj)
                break
            if DEBUG:
                print()
        if sum(basin_sizes) < nsim:
            print('Warning: only %i of the %i tested initial conditions eventually reached a steady state. '
                  'Try increasing the search depth. It may also be that your asynchronous state space contains a limit cycle.' % (sum(basin_sizes), nsim))
        return dict(zip(["SteadyStates", "NumberOfSteadyStates", "BasinSizes", "TransientTimes", "STGAsynchronous", "UpdateQueues"],
                        (steady_states, len(steady_states), basin_sizes, transient_times, STG_async, queues)))


    def get_attractors_synchronous(
        self,
        nsim: int = 500,
        initial_sample_points: list = [],
        n_steps_timeout: int = 1000,
        INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS: bool = False,
        USE_NUMBA: bool = True,
        *,
        rng=None,
    ) -> dict:
        """
        Compute the number of attractors in a Boolean network using synchronous updates.
    
        This method is optimized for networks with long transient dynamics. Starting
        from a set of initial conditions, the network is updated synchronously until
        an attractor is reached or a timeout is exceeded. The returned attractors and
        basin sizes are lower-bound / unbiased estimates based on the sampled states.
    
        If Numba is available and USE_NUMBA=True, Boolean updates are accelerated using
        a compiled kernel; otherwise a pure-Python implementation is used.
    
        **Parameters:**
    
            - nsim (int, optional): Number of initial conditions to simulate
              (default 500). Ignored if initial_sample_points are provided.
    
            - initial_sample_points (list[int | list[int]], optional): Initial states
              to use. Interpretation is controlled by
              INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS.
    
            - n_steps_timeout (int, optional): Maximum number of update steps per
              simulation (default 1000).
    
            - INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS (bool, optional): If True,
              initial_sample_points are binary vectors; otherwise decimal states.
    
            - USE_NUMBA (bool, optional): If True (default), use Numba acceleration
              when available.
    
            - rng (None, optional): Random number generator, coerced via
              utils._coerce_rng.
    
        **Returns:**
    
            - dict[str:Variant] with keys:
                Attractors, NumberOfAttractors, BasinSizes, AttractorDict,
                InitialSamplePoints, STG, NumberOfTimeouts
        """
        rng = utils._coerce_rng(rng)
    
        dictF = {}          # memorized transitions
        attractors = []     # list of attractor cycles
        basin_sizes = []    # basin counts
        attr_dict = {}      # state -> attractor index
        STG = {}            # sampled state transition graph
        n_timeout = 0
        sampled_points = []
    
        INITIAL_SAMPLE_POINTS_EMPTY = utils.check_if_empty(initial_sample_points)
        if not INITIAL_SAMPLE_POINTS_EMPTY:
            nsim = len(initial_sample_points)
    
        # --- Decide update backend
        use_numba = __LOADED_NUMBA__ and USE_NUMBA
    
        if use_numba:
            F_array_list = List([np.array(bf.f, dtype=np.uint8) for bf in self.F])
            I_array_list = List([np.array(regs, dtype=np.int64) for regs in self.I])
            N = self.N
    
        # --- Main simulation loop
        for sim_idx in range(nsim):
            # --- Initialize state
            if INITIAL_SAMPLE_POINTS_EMPTY:
                x = rng.integers(2, size=self.N, dtype=np.uint8)
                xdec = utils.bin2dec(x)
                sampled_points.append(xdec)
            else:
                if INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS:
                    x = np.asarray(initial_sample_points[sim_idx], dtype=np.uint8)
                    xdec = utils.bin2dec(x)
                else:
                    xdec = int(initial_sample_points[sim_idx])
                    x = np.array(utils.dec2bin(xdec, self.N), dtype=np.uint8)
    
            visited = {xdec: 0}
            trajectory = [xdec]
            count = 0
    
            # --- Iterate until attractor or timeout
            while count < n_steps_timeout:
                if xdec in dictF:
                    fxdec = dictF[xdec]
                else:
                    if use_numba:
                        fx = _update_network_synchronously_numba(
                            x, F_array_list, I_array_list, N
                        )
                    else:
                        fx = self.update_network_synchronously(x)
    
                    fxdec = utils.bin2dec(fx)
                    dictF[xdec] = fxdec
                    x = fx
    
                if count == 0:
                    STG[xdec] = fxdec
    
                # already mapped to known attractor
                if fxdec in attr_dict:
                    idx_attr = attr_dict[fxdec]
                    basin_sizes[idx_attr] += 1
                    for s in trajectory:
                        attr_dict[s] = idx_attr
                    break
    
                # new attractor detected
                if fxdec in visited:
                    cycle_start = visited[fxdec]
                    attractor_states = trajectory[cycle_start:]
                    attractors.append(attractor_states)
                    basin_sizes.append(1)
                    idx_attr = len(attractors) - 1
                    for s in attractor_states:
                        attr_dict[s] = idx_attr
                    break
    
                # continue traversal
                visited[fxdec] = len(trajectory)
                trajectory.append(fxdec)
                xdec = fxdec
                count += 1
    
                if count == n_steps_timeout:
                    n_timeout += 1
                    break
    
        return {
            "Attractors": attractors,
            "NumberOfAttractors": len(attractors),
            "BasinSizes": basin_sizes,
            "AttractorDict": attr_dict,
            "InitialSamplePoints": (
                sampled_points if INITIAL_SAMPLE_POINTS_EMPTY else initial_sample_points
            ),
            "STG": STG,
            "NumberOfTimeouts": n_timeout,
        }

    
    def compute_synchronous_state_transition_graph(self, USE_NUMBA : bool = True):
        """
        Compute the synchronous state transition graph (STG) for all 2^N states.
        
        The STG is stored in `self.STG` as a one-dimensional NumPy array of length 2^N,
        where `self.STG[x]` is the decimal representation of the successor state 
        reached from state `x` under synchronous update.
        
        **Parameters:**
        
            - USE_NUMBA (bool, optional): If True (default), 
              Numba acceleration is used when available.
        """
    
        if __LOADED_NUMBA__ and USE_NUMBA:
            # Preprocess data into Numba-friendly types
            F_list = [np.array(bf.f, dtype=np.uint8) for bf in self.F]
            I_list = [np.array(regs, dtype=np.int64) for regs in self.I]
            
            if self.N <= 22:
                self.STG = _compute_synchronous_stg_numba(F_list, I_list, self.N)
            else:
                self.STG = _compute_synchronous_stg_numba_low_memory(F_list, I_list, self.N)
        else:
            # 1. Represent all possible network states as binary matrix
            states = utils.get_left_side_of_truth_table(self.N)
            
            # 2. Preallocate array for next states
            next_states = np.zeros_like(states)
            powers_of_two = 2 ** np.arange(self.N)[::-1]
        
            # 3. Compute next value for each node in vectorized form
            for j, bf in enumerate(self.F):
                regulators = self.I[j]
                if len(regulators) == 0:
                    # constant node
                    next_states[:, j] = bf.f[0]
                    continue
        
                # Extract substate of regulators for all states
                subspace = states[:, regulators]
        
                # Convert each substate to integer index (row of truth table)
                idx = np.dot(subspace, powers_of_two[-len(regulators):])
        
                # Lookup next-state value from Boolean function truth table
                next_states[:, j] = bf.f[idx]
        
            # 4. Convert each next-state binary vector to integer index
            self.STG = np.dot(next_states, powers_of_two).astype(np.int64)


    def get_attractors_synchronous_exact(self, USE_NUMBA : bool = True) -> dict:
        """
        Compute all attractors and their exact basin sizes under a synchronous
        updating scheme.
        
        **Parameters:**
        
            - USE_NUMBA (bool, optional): If True (default), 
              Numba acceleration is used when available.

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - Attractors (list[list[int]]): List of attractors (each
                  attractor is represented as a list of states forming the
                  cycle).
                
                - NumberOfAttractors (int): Number of unique attractors.
                
                - BasinSizes (np.ndarray[float]): Proportion of states that 
                  eventually transition to each attractor.
                
                - AttractorID (np.ndarray[int]): A one-dimensional integer array
                  representing the index of the attractor eventually reached from
                  each of the 2^N states.
                  
                - STG (np.ndarray[int]):
                  A one-dimensional integer array representing the synchronous 
                  state transition graph. For each of the 2^N states, the decimal
                  index of its successor state is stored.
        """        
        if self.STG is None:
            self.compute_synchronous_state_transition_graph(USE_NUMBA=USE_NUMBA)

        attractors = []
        
        if __LOADED_NUMBA__ and USE_NUMBA:
            attractor_id, basin_sizes, cycle_rep, cycle_len, n_attr = _attractors_functional_graph(self.STG)
        
            # Reconstruct explicit attractor cycles
            for k in range(int(n_attr)):
                rep = int(cycle_rep[k])
                L = int(cycle_len[k])
                cyc = [rep]
                x = rep
                for _ in range(L - 1):
                    x = int(self.STG[x])
                    cyc.append(x)
                attractors.append(cyc)

        else:                
            basin_sizes = []
            attractor_id = -np.ones(2**self.N,dtype=np.int32)
            n_attr = 0
            for xdec in range(2**self.N):
                queue = [xdec]
                while True:
                    fxdec = self.STG[xdec]
                    if attractor_id[fxdec]==-1:
                        try:
                            index = queue.index(fxdec)
                            attractor_id[np.array(queue)] = n_attr
                            n_attr += 1
                            attractors.append(queue[index:])
                            basin_sizes.append(1)
                            break
                        except ValueError:
                            pass
                    else: #already know the attractor of fxdec
                        index_attr = attractor_id[fxdec]
                        basin_sizes[index_attr] += 1
                        attractor_id[np.array(queue)] = index_attr
                        break
                    queue.append(fxdec)
                    xdec = fxdec

        basin_sizes = np.array(basin_sizes,dtype=np.float64)/2**self.N
        
        return dict(zip(
            ["Attractors", "NumberOfAttractors", "BasinSizes", "AttractorID", "STG"],
            (attractors, len(attractors), basin_sizes, attractor_id, self.STG)
        ))


    ## Robustness measures: synchronous Derrida value, entropy of basin size distribution, coherence, fragility
    def get_attractors_and_robustness_measures_synchronous_exact(self, USE_NUMBA : bool = True) -> dict:
        """
        Compute the attractors and several robustness measures of a synchronously 
        updated Boolean network.

        This function computes the exact attractors and robustness (coherence
        and fragility) of the entire network, as well as robustness measures
        for each basin of attraction and each attractor.
        
        **Parameters:**
        
            - USE_NUMBA (bool, optional): If True (default), 
              Numba acceleration is used when available.

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - Attractors (list[list[int]]): List of attractors (each
                  attractor is represented as a list of state decimal numbers).
                
                - NumberOfAttractors (int): Number of unique attractors.
                
                - BasinSizes (np.ndarray[float]): Proportion of states that 
                  eventually transition to each attractor.
                
                - AttractorID (np.ndarray[int]): A one-dimensional integer array
                  representing the index of the attractor eventually reached from
                  each of the 2^N states.
                  
                - Coherence (float): overall exact network coherence
                - Fragility (float): overall exact network fragility
                - BasinCoherence (np.ndarray[float]): exact coherence of each basin.
                - BasinFragility (np.ndarray[float]): exact fragility of each basin.
                - AttractorCoherence (np.ndarray[float]): exact coherence of each
                  attractor.
                  
                - AttractorFragility (np.ndarray[float]): exact fragility of each
                  attractor.
        
        **References:**
            
            1. Park, K. H., Costa, F. X., Rocha, L. M., Albert, R., & Rozum,
               J. C. (2023). Models of cell processes are far from the edge of
               chaos. PRX life, 1(2), 023009.
               
            2. Bavisetty, V. S. N., Wheeler, M., & Kadelka, C. (2025). Attractors
               are less stable than their basins: Canalization creates a coherence 
               gap in gene regulatory networks. bioRxiv 2025-11.
        """
        
        # 0) attractors + basins
        result = self.get_attractors_synchronous_exact(USE_NUMBA=USE_NUMBA)
        
        attractors = result["Attractors"]
        n_attractors = result["NumberOfAttractors"]
        basin_sizes = result["BasinSizes"]
        attractor_id = result["AttractorID"]
    
        # --- Single-attractor shortcut: values are trivial
        if n_attractors == 1:
            return dict(zip(
                ["Attractors", "NumberOfAttractors", "BasinSizes",
                 "AttractorID", "BasinCoherence", "BasinFragility",
                 "AttractorCoherence", "AttractorFragility", "Coherence", "Fragility"],
                (attractors, n_attractors, basin_sizes,
                 attractor_id, np.ones(1), np.zeros(1),
                 np.ones(1), np.zeros(1), 1.0, 0.0)
            ))
    
        # 1) arrays for O(1) lookup + attractor membership
        n_states = 2**self.N
        is_attr_mask = np.zeros(n_states, dtype=np.uint8)
        len_attractors = np.empty(len(attractors), dtype=np.int64)
        for i, states in enumerate(attractors):
            len_attractors[i] = len(states)
            # states is list[int] of decimals
            is_attr_mask[np.array(states, dtype=np.int64)] = 1
    
        # 2) mean binary vector per attractor (NumPy; usually small)
        mean_states_attractors = []
        for states in attractors:
            if len(states) == 1:
                mean_states_attractors.append(
                    np.array(utils.dec2bin(states[0], self.N), dtype=np.float64)
                )
            else:
                arr = np.array([utils.dec2bin(s, self.N) for s in states], dtype=np.float64)
                mean_states_attractors.append(arr.mean(axis=0))
        mean_states_attractors = np.stack(mean_states_attractors)
    
        # 3) distance matrix between attractors (NumPy vectorized)
        diff = mean_states_attractors[:, None, :] - mean_states_attractors[None, :, :]
        distance_between_attractors = np.sum(np.abs(diff), axis=2) / self.N
        distance_between_attractors = np.asarray(distance_between_attractors, dtype=np.float64)
    
        # 4) Numba kernel for edge traversal
        if __LOADED_NUMBA__ and USE_NUMBA:
            basin_coherences, basin_fragilities, attractor_coherences, attractor_fragilities = _robustness_edge_traversal_numba(
                int(self.N), attractor_id, is_attr_mask, distance_between_attractors
            )
        else:
            #left_side_of_truth_table = utils.get_left_side_of_truth_table(self.N)
            
            n_attractors = len(attractors)
            basin_coherences = np.zeros(n_attractors)
            basin_fragilities = np.zeros(n_attractors)
            attractor_coherences = np.zeros(n_attractors)
            attractor_fragilities = np.zeros(n_attractors)
        
            #powers_of_2 = (2 ** np.arange(self.N))[::-1]
        
            for xdec in range(n_states):
                for bit in range(self.N):
                    if (xdec >> bit) & 1:
                        continue # skip to avoid double-counting
                    ydec = xdec | (1 << bit)
        
            # for xdec, x in enumerate(left_side_of_truth_table):
            #     for i in range(self.N):
            #         if x[i] == 1:
            #             continue  # skip to avoid double-counting
            #         ydec = xdec + powers_of_2[i]
        
                    idx_x = attractor_id[xdec]
                    idx_y = attractor_id[ydec]
        
                    if idx_x == idx_y:
                        basin_coherences[idx_x] += 2  # count both directions
                        if is_attr_mask[xdec]:
                            attractor_coherences[idx_x] += 1
                        if is_attr_mask[ydec]:
                            attractor_coherences[idx_y] += 1
                    else:
                        dxy = distance_between_attractors[idx_x, idx_y]
                        basin_fragilities[idx_x] += dxy
                        basin_fragilities[idx_y] += dxy
                        if is_attr_mask[xdec]:
                            attractor_fragilities[idx_x] += dxy
                        if is_attr_mask[ydec]:
                            attractor_fragilities[idx_y] += dxy
    
        # 5) normalize (Python/NumPy; cheap)
        basin_counts = basin_sizes * n_states
        for i in range(n_attractors):
            basin_coherences[i] /= (basin_counts[i] * self.N)
            basin_fragilities[i] /= (basin_counts[i] * self.N)
            attractor_coherences[i] /= (len_attractors[i] * self.N)
            attractor_fragilities[i] /= (len_attractors[i] * self.N)
    
        #basin_sizes_norm = basin_sizes / (2 ** self.N)
        coherence = float(np.dot(basin_sizes, basin_coherences))
        fragility = float(np.dot(basin_sizes, basin_fragilities))

        return dict(zip(
            ["Attractors", "NumberOfAttractors",
             "BasinSizes", "AttractorID",
             "Coherence", "Fragility",
             "BasinCoherence", "BasinFragility",
             "AttractorCoherence", "AttractorFragility"],
            (attractors, n_attractors,
             basin_sizes, attractor_id,
             coherence, fragility,
             basin_coherences, basin_fragilities,
             attractor_coherences, attractor_fragilities)
        ))


    def get_attractors_and_robustness_measures_synchronous(self, 
                                                           number_different_IC : int = 500, 
                                                           RETURN_ATTRACTOR_COHERENCE : bool = True, 
                                                           *, 
                                                           rng=None) -> dict:
        """
        Approximate global robustness measures and attractors.

        This function samples the attractor landscape by simulating the network
        from a number of different initial conditions. It computes:
            
            - The coherence: the proportion of neighboring states (in the
              Boolean hypercube) that, after synchronous update, transition to
              the same attractor.
            
            - The fragility: a measure of how much the attractor state changes
              (assumed under synchronous update) in response to perturbations.
              
            - The final time-step Hamming distance between perturbed trajectories.

        In addition, it collects several details about each attractor (such as
        basin sizes, coherence of each basin, etc.).

        **Parameters:**
            
            - number_different_IC (int, optional): Number of different initial
              conditions to sample (default is 500).
              
            - RETURN_ATTRACTOR_COHERENCE (bool, optional): Determines whether
              the attractor coherence should also be computed (default True,
              i.e., Yes).
              
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - Attractors (list[list[int]]): List of attractors (each
                  attractor is represented as a list of state decimal numbers).
                
                - LowerBoundOfNumberOfAttractors (int): The lower bound on the
                  number of attractors found.
                  
                - BasinSizes (np.ndarray[float]): Proportion of states that 
                  eventually transition to each attractor.
                  
                - CoherenceApproximation (float): The approximate overall
                  network coherence.
                  
                - FragilityApproximation (float): The approximate overall
                  network fragility.
                  
                - FinalHammingDistanceApproximation (float): The approximate
                  final Hamming distance measure.
                  
                - BasinCoherenceApproximation (np.ndarray[float]): The approximate
                  coherence of each basin.
                  
                - BasinFragilityApproximation (np.ndarray[float]): The approximate
                  fragility of each basin.
                  
                - AttractorCoherence (np.ndarray[float]): The exact coherence of
                  each attractor (only computed and returned if
                  RETURN_ATTRACTOR_COHERENCE == True).
                  
                - AttractorFragility (np.ndarray[float]): The exact fragility of
                  each attractor (only computed and returned if
                  RETURN_ATTRACTOR_COHERENCE == True).

        **References:**
            
            1. Park, K. H., Costa, F. X., Rocha, L. M., Albert, R., & Rozum,
               J. C. (2023). Models of cell processes are far from the edge of
               chaos. PRX life, 1(2), 023009.
               
            2. Bavisetty, V. S. N., Wheeler, M., & Kadelka, C. (2025). Attractors
               are less stable than their basins: Canalization creates a coherence 
               gap in gene regulatory networks. bioRxiv 2025-11.
        """
        rng = utils._coerce_rng(rng)
        def lcm(a, b):
            return abs(a*b) // math.gcd(a, b)
        
        dictF = dict()
        attractors = []
        ICs_per_attractor_state = []
        basin_sizes = []
        attractor_dict = dict()
        attractor_state_dict = []
        distance_from_attractor_state_dict = []
        counter_phase_shifts = []
        
        height = []
        
        powers_of_2s = [np.array([2**i for i in range(NN)])[::-1] for NN in range(max(self.indegrees)+1)]
        if self.N<64:
            powers_of_2 = np.array([2**i for i in range(self.N)])[::-1]
        
        robustness_approximation = 0
        fragility_sum = 0
        basin_robustness = defaultdict(float)
        basin_fragility = defaultdict(float)
        final_hamming_distance_approximation = 0
        mean_states_attractors = []
        states_attractors = []
        
        for i in range(number_different_IC):
            index_attractors = []
            index_of_state_within_attractor_reached = []
            distance_from_attractor = []
            for j in range(2):
                if j == 0:
                    x = rng.integers(2, size=self.N)
                    if self.N<64:
                        xdec = np.dot(x, powers_of_2).item()
                    else: #out of range of np.int64
                        xdec = ''.join(str(bit) for bit in x)
                    x_old = x.copy()
                else:
                    x = x_old
                    random_flipped_bit = rng.integers(self.N)
                    x[random_flipped_bit] = 1 - x[random_flipped_bit]
                    if self.N<64:
                        xdec = np.dot(x, powers_of_2).item()
                    else: #out of range of np.int64
                        xdec = ''.join(str(bit) for bit in x)               
                queue = [xdec]
                try:
                    index_attr = attractor_dict[xdec]
                except KeyError:
                    while True:
                        try: #check if we already know F(xdec)
                            fxdec = dictF[xdec]
                        except KeyError: #if not, then compute the F(xdec)
                            fx = []
                            for jj in range(self.N):
                                if self.indegrees[jj]>0:
                                    fx.append(self.F[jj].f[np.dot(x[self.I[jj]], powers_of_2s[self.indegrees[jj]]).item()])
                                else:#constant functions whose regulators were all fixed to a specific value
                                    fx.append(self.F[jj].f[0])
                            if self.N<64:
                                fxdec = np.dot(fx, powers_of_2).item()
                            else:
                                fxdec = ''.join(str(bit) for bit in fx)               
                            dictF.update({xdec: fxdec})
                        try: #check if we already know the attractor of F(xdec) 
                            index_attr = attractor_dict[fxdec]
                            dummy_index_within_attractor_reached = attractor_state_dict[index_attr][fxdec]
                            dummy_distance_from_attractor = distance_from_attractor_state_dict[index_attr][fxdec]
                            attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                            attractor_state_dict[index_attr].update(list(zip(queue, [dummy_index_within_attractor_reached]*len(queue))))
                            distance_from_attractor_state_dict[index_attr].update(
                                list(zip(queue, list(range(len(queue) + dummy_distance_from_attractor, dummy_distance_from_attractor, -1))))
                            )
                            break
                        except KeyError: 
                            try: #if not, then check if F(xdec) is already in the queue, i.e., if F(xdec) is part of an attractor itself
                                index = queue.index(fxdec)
                                index_attr = len(attractors)
                                attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                                attractors.append(queue[index:])
                                basin_sizes.append(1)
                                attractor_state_dict.append(dict(zip(queue, [0]*index + list(range(len(attractors[-1])))))
                                )
                                distance_from_attractor_state_dict.append(
                                    dict(zip(queue, list(range(index, 0, -1)) + [0]*len(attractors[-1])))
                                )
                                ICs_per_attractor_state.append([0] * len(attractors[-1]))
                                counter_phase_shifts.append([0] * len(attractors[-1]))

                                if len(attractors[-1]) == 1:
                                    if self.N<64:
                                        fixed_point = np.array(utils.dec2bin(queue[index], self.N))
                                    else:
                                        fixed_point = np.array(list(queue[index]), dtype=int)
                                    states_attractors.append(fixed_point.reshape((1, self.N)))
                                    mean_states_attractors.append(fixed_point)
                                else:
                                    if self.N<64:
                                        limit_cycle = np.array([utils.dec2bin(state, self.N) for state in queue[index:]])
                                    else:
                                        limit_cycle = np.array([np.array(list(state), dtype=int) for state in queue[index:]])          
                                    states_attractors.append(limit_cycle)
                                    mean_states_attractors.append(limit_cycle.mean(0))
                                break
                            except ValueError: #if not, proceed by setting x = F(x)
                                x = np.array(fx)
                        queue.append(fxdec)
                        xdec = fxdec

                index_attractors.append(index_attr)
                index_of_state_within_attractor_reached.append(attractor_state_dict[index_attr][xdec])
                distance_from_attractor.append(distance_from_attractor_state_dict[index_attr][xdec])
                basin_sizes[index_attr] += 1
                ICs_per_attractor_state[index_attr][attractor_state_dict[index_attr][xdec]] += 1
            if index_attractors[0] == index_attractors[1]:
                robustness_approximation += 1
                basin_robustness[index_attractors[0]] += 1
                length_phaseshift = max(index_of_state_within_attractor_reached) - min(index_of_state_within_attractor_reached)
                counter_phase_shifts[index_attr][length_phaseshift] += 1
            else:
                fragility_sum += np.sum(np.abs(mean_states_attractors[index_attractors[0]] - mean_states_attractors[index_attractors[1]]))
                basin_fragility[index_attractors[0]] += np.sum(np.abs(mean_states_attractors[index_attractors[0]] - mean_states_attractors[index_attractors[1]]))
                required_n_states = lcm(len(attractors[index_attractors[0]]), len(attractors[index_attractors[1]]))
                index_j0 = index_of_state_within_attractor_reached[0]
                periodic_states_j0 = np.tile(states_attractors[index_attractors[0]], 
                                             (required_n_states // len(attractors[index_attractors[0]]) + 1, 1))[index_j0:(index_j0 + required_n_states), :]
                index_j1 = index_of_state_within_attractor_reached[1]
                periodic_states_j1 = np.tile(states_attractors[index_attractors[1]], 
                                             (required_n_states // len(attractors[index_attractors[1]]) + 1, 1))[index_j1:(index_j1 + required_n_states), :]
                final_hamming_distance_approximation += np.mean(periodic_states_j1 == periodic_states_j0)
                
            height.extend(distance_from_attractor)
        
        lower_bound_number_of_attractors = len(attractors)
        approximate_basin_sizes = np.array(basin_sizes)
        approximate_coherence = robustness_approximation * 1.0 / number_different_IC
        approximate_fragility = fragility_sum * 1.0 / number_different_IC / self.N
        
        approximate_basin_coherence = np.array([basin_robustness[index_att] * 2.0 / basin_sizes[index_att] for index_att in range(len(attractors))])
        approximate_basin_fragility = np.array([basin_fragility[index_att] * 2.0 / basin_sizes[index_att] / self.N for index_att in range(len(attractors))])
        
        for index_attr in range(len(attractors)):
            periodic_states_two_periods = np.tile(states_attractors[index_attr], (2, 1))
            for length_phaseshift, num_IC_with_that_phaseshift in enumerate(counter_phase_shifts[index_attr]):
                if num_IC_with_that_phaseshift > 0 and length_phaseshift > 0:
                    final_hamming_distance_approximation += num_IC_with_that_phaseshift * np.mean(
                        states_attractors[index_attr] ==
                        periodic_states_two_periods[length_phaseshift:(length_phaseshift + len(attractors[index_attr])), :]
                    )
                    
        final_hamming_distance_approximation = final_hamming_distance_approximation / number_different_IC
        
        #fixing the results here because the subsequent attractor coherence computation could in theory identify additional attractors, 
        #which would screw things up because the attractor regions of the state space have then been oversampled
        results = [attractors, lower_bound_number_of_attractors, approximate_basin_sizes/2./number_different_IC, 
                   approximate_coherence, approximate_fragility, final_hamming_distance_approximation,
                   approximate_basin_coherence, approximate_basin_fragility]
        if RETURN_ATTRACTOR_COHERENCE == False:
            return dict(zip(["Attractors", "LowerBoundOfNumberOfAttractors", "BasinSizesApproximation",
                             "CoherenceApproximation", "FragilityApproximation", "FinalHammingDistanceApproximation",
                             "BasinCoherenceApproximation", "BasinFragilityApproximation"],
                            tuple(results)))
        else:
            attractor_coherence = np.zeros(lower_bound_number_of_attractors)
            attractor_fragility = np.zeros(lower_bound_number_of_attractors)
            attractors_original = attractors[:] #needed because new attractors may be found
            for index_attr_original,attractor in enumerate(attractors_original):
                for attractor_state in attractor: #perturb each attractor state
                    for i in range(self.N):
                        if self.N<64:
                            x = np.array(utils.dec2bin(attractor_state, self.N))
                        else:
                            x = np.array(list(attractor_state), dtype=int)
                        x[i] = 1 - x[i]
                        if self.N<64:
                            xdec = np.dot(x, powers_of_2).item()
                        else:
                            xdec = ''.join(str(bit) for bit in x)
                        queue = [xdec]
                        try:
                            index_attr = attractor_dict[xdec]
                        except KeyError:
                            while True:
                                try: #check if we already know F(xdec)
                                    fxdec = dictF[xdec]
                                except KeyError: #if not, then compute the F(xdec)
                                    fx = []
                                    for jj in range(self.N):
                                        if self.indegrees[jj]>0:
                                            fx.append(self.F[jj].f[np.dot(x[self.I[jj]], powers_of_2s[self.indegrees[jj]]).item()])
                                        else:#constant functions whose regulators were all fixed to a specific value
                                            fx.append(self.F[jj].f[0])
                                    if self.N<64:
                                        fxdec = np.dot(fx, powers_of_2).item()
                                    else:
                                        fxdec = ''.join(str(bit) for bit in fx)               
                                    dictF.update({xdec: fxdec})
                                try: #check if we already know the attractor of F(xdec) 
                                    index_attr = attractor_dict[fxdec]
                                    dummy_index_within_attractor_reached = attractor_state_dict[index_attr][fxdec]
                                    dummy_distance_from_attractor = distance_from_attractor_state_dict[index_attr][fxdec]
                                    attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                                    attractor_state_dict[index_attr].update(list(zip(queue, [dummy_index_within_attractor_reached]*len(queue))))
                                    distance_from_attractor_state_dict[index_attr].update(
                                        list(zip(queue, list(range(len(queue) + dummy_distance_from_attractor, dummy_distance_from_attractor, -1))))
                                    )
                                    break
                                except KeyError: 
                                    try: #if not, then check if F(xdec) is already in the queue, i.e., if F(xdec) is part of an attractor itself
                                        index = queue.index(fxdec)
                                        index_attr = len(attractors)
                                        attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                                        attractors.append(queue[index:])
                                        #basin_sizes.append(1)
                                        attractor_state_dict.append(dict(zip(queue, [0]*index + list(range(len(attractors[-1])))))
                                        )
                                        distance_from_attractor_state_dict.append(
                                            dict(zip(queue, list(range(index, 0, -1)) + [0]*len(attractors[-1])))
                                        )
                                        ICs_per_attractor_state.append([0] * len(attractors[-1]))
                                        counter_phase_shifts.append([0] * len(attractors[-1]))
            
                                        if len(attractors[-1]) == 1:
                                            if self.N<64:
                                                fixed_point = np.array(utils.dec2bin(queue[index], self.N))
                                            else:
                                                fixed_point = np.array(list(queue[index]), dtype=int)
                                            states_attractors.append(fixed_point.reshape((1, self.N)))
                                            mean_states_attractors.append(fixed_point)
                                        else:
                                            if self.N<64:
                                                limit_cycle = np.array([utils.dec2bin(state, self.N) for state in queue[index:]])
                                            else:
                                                limit_cycle = np.array([np.array(list(state), dtype=int) for state in queue[index:]])          
                                            states_attractors.append(limit_cycle)
                                            mean_states_attractors.append(limit_cycle.mean(0))
                                        break
                                    except ValueError: #if not, proceed by setting x = F(x)
                                        x = np.array(fx)
                                queue.append(fxdec)
                                xdec = fxdec
                        if index_attr_original == index_attr:
                            attractor_coherence[index_attr_original] += 1
                        else:
                            attractor_fragility[index_attr_original] += np.sum(np.abs(mean_states_attractors[index_attr_original] - mean_states_attractors[index_attr]))
            attractor_coherence = np.array([s/self.N/size_attr for s,size_attr in zip(attractor_coherence,map(len,attractors_original))])
            attractor_fragility = np.array([s/self.N**2/size_attr for s,size_attr in zip(attractor_fragility,map(len,attractors_original))]) #something is wrong with attractor fragility, it returns values > 1 for small basins
            results[0] = attractors_original #important! It may be that new attractors were found, reset the count
            return dict(zip(["Attractors", "LowerBoundOfNumberOfAttractors", "BasinSizesApproximation",
                             "CoherenceApproximation", "FragilityApproximation", "FinalHammingDistanceApproximation",
                             "BasinCoherenceApproximation", "BasinFragilityApproximation",
                             "AttractorCoherence", "AttractorFragility"],
                            tuple(results + [attractor_coherence,attractor_fragility])))
        
    def get_derrida_value(self, 
                          nsim : int = 1000, 
                          EXACT : bool = False,
                          USE_NUMBA : bool = True,
                          *, 
                          rng = None) -> float:
        """
        Estimate the Derrida value for a Boolean network.

        The Derrida value is computed by perturbing a single node in a randomly
        chosen state and measuring the average Hamming distance between the
        resulting updated states of the original and perturbed networks.

        **Parameters:**
            
            - nsim (int, optional): Number of simulations to perform. Default
              is 1000.
              
            - EXACT (bool, optional): If True, the exact Derrida value is
              computed and 'nsim' is ignored. If False, the Derrida value is 
              approximated using Monte Carlo simulation (Numba-accelerated when available).
              
            - USE_NUMBA (bool, optional): If True (default), 
              Numba acceleration is used when available.
            
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - float: The average Hamming distance (Derrida value) over
              nsim simulations.

        **References:**
            
            #. Derrida, B., & Pomeau, Y. (1986). Random networks of automata:
               a simple annealed approximation. Europhysics letters, 1(2), 45.
        """
        if EXACT:
            return float(np.mean([
                bf.get_average_sensitivity(EXACT=True, NORMALIZED=False) for bf in self.F
            ]))
        else:
            rng = utils._coerce_rng(rng)
            if __LOADED_NUMBA__ and USE_NUMBA:
                # --- Numba-friendly preparation
                F_array_list = List([np.array(bf.f, dtype=np.uint8) for bf in self.F])
                I_array_list = List([np.array(regs, dtype=np.int64) for regs in self.I])
            
                # Derive reproducible seed from rng
                
                seed = int(rng.integers(19891989))
            
                return _derrida_simulation(F_array_list, I_array_list, self.N, nsim, seed)
            else:
                total_dist = 0.0
                
                for _ in range(nsim):
                    x = rng.integers(0, 2, size=self.N, dtype=np.uint8)
                    y = x.copy()
                
                    idx = rng.integers(0, self.N)
                    y[idx] ^= 1
                
                    fx = self.update_network_synchronously(x)
                    fy = self.update_network_synchronously(y)
                
                    total_dist += np.sum(fx != fy)
                
                return float(total_dist / nsim)
         

