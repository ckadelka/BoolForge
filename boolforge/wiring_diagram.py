#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wiring-diagram representation for Boolean networks.

This module defines the :class:`~boolforge.WiringDiagram` class, which encodes
the directed regulatory topology of a Boolean network independently of any
Boolean update functions.

A wiring diagram specifies, for each node, the set of regulating nodes
(predecessors). Nodes with no regulators are interpreted as constants within
the Boolean-network formalism.
"""

from collections import defaultdict
from collections.abc import Sequence
import numpy as np
import networkx as nx

__all__ = [
    "WiringDiagram",
]

class WiringDiagram(object):
    """
    Directed wiring diagram for a Boolean network.

    A wiring diagram specifies the regulatory topology of a Boolean network by
    listing, for each node, the indices of its regulators (incoming edges).
    It does not encode Boolean update functions or dynamical rules.

    Nodes with zero indegree are interpreted as constant nodes in the Boolean
    network formalism.

    Parameters
    ----------
    I : sequence of sequences of int
        Adjacency-list representation of the wiring diagram.
        Entry ``I[i]`` contains the indices of nodes regulating node ``i``.
        Regulator lists may have variable length.
    variables : list[str] or np.ndarray[str], optional
        Names of the variables corresponding to each node.
        Must have length ``N`` if provided. If None, default names
        ``['x0', 'x1', ..., 'x{N-1}']`` are assigned.
    weights : sequence of sequences of float, optional
        Interaction weights corresponding to ``I``.
        Entry ``weights[i][j]`` gives the weight of the interaction
        from regulator ``I[i][j]`` to node ``i``. Missing interactions
        may be encoded as ``np.nan``.

    Attributes
    ----------
    I : list[np.ndarray]
        Regulator index arrays, one per node.
    variables : np.ndarray[str]
        Names of variables corresponding to each node.
    N : int
        Total number of nodes, including constants.
    N_variables : int
        Number of non-constant variables.
    N_constants : int
        Number of constant nodes (nodes with zero indegree).
    indegrees : np.ndarray[int]
        Indegree of each node.
    outdegrees : np.ndarray[int]
        Outdegree of each node.
    weights : list[np.ndarray] or None
        Interaction weights associated with ``I`` if provided.

    Notes
    -----
    - Node indices are zero-based.
    - Constant nodes are identified solely by indegree.
    - The wiring diagram encodes topology only and does not define dynamics.

    Examples
    --------
    Nodes with zero indegree are interpreted as constants.
    
    >>> from boolforge import WiringDiagram
    >>> I = [
    ...     [],        # node 0 has no regulators -> constant
    ...     [0],       # node 1 is regulated by node 0
    ...     [0, 1],    # node 2 is regulated by nodes 0 and 1
    ... ]
    >>> wd = WiringDiagram(I)
    
    >>> wd.N
    3
    >>> wd.N_constants
    1
    >>> wd.N_variables
    2
    
    >>> wd.get_constants(AS_DICT=True)
    {0: True, 1: False, 2: False}
    
    >>> wd.get_constants(AS_DICT=False)
    array([0])
    
    See Also
    --------
    boolforge.BooleanNetwork
    """
    
    def __init__(
        self,
        I : Sequence[Sequence[int]],
        variables : list[str] | np.ndarray | None = None,
        weights : Sequence[Sequence[float]] | None = None,
    ):
        """
        Initialize a wiring diagram.
    
        Parameters
        ----------
        I : sequence of sequences of int
            Adjacency list representation of the wiring diagram.
            Entry ``I[i]`` contains the indices of nodes regulating node ``i``.
            Regulator lists may have variable length.
        variables : list[str] or np.ndarray[str], optional
            Names of the variables corresponding to each node.
            Must have length ``N`` if provided. If None, default names
            ``['x0', 'x1', ..., 'x{N-1}']`` are assigned.
        weights : sequence of sequences of float, optional
            Optional interaction weights corresponding to ``I``.
            Each entry ``weights[i][j]`` gives the weight of the interaction
            from regulator ``I[i][j]`` to node ``i``.
            Missing or undefined interactions may be encoded as ``np.nan``.
    
        Raises
        ------
        TypeError
            If ``I`` is not a sequence of sequences of integers.
            If ``weights`` is provided and is not a sequence of sequences of numbers.
            If any entry ``weights[i]`` is not a sequence of numbers.
        ValueError
            If ``variables`` is provided and does not have length ``N``.
            If ``weights`` is provided and does not have length ``N``.
            If any entry ``weights[i]`` does not have the same length as ``I[i]``.

    
        Notes
        -----
        Constant nodes are identified automatically as nodes with zero indegree.
        """
        if not isinstance(I, Sequence) or isinstance(I, (str, bytes)):
            raise TypeError("I must be a sequence of sequences of int")
            
        if variables is not None and len(I) != len(variables):
            raise ValueError("len(I) == len(variables) required if variable names are provided")

        
        self.I = [np.array(regulators,dtype=int) for regulators in I]
        self.N = len(I)
        self.indegrees = np.array([len(regulators) for regulators in self.I], dtype=int)
        
        if variables is None:
            variables = ['x'+str(i) for i in range(self.N)]
        
        self.N_constants = len(self.get_constants(False))
        self.N_variables = self.N - self.N_constants
        
        self.variables = np.array(variables, dtype=str)
        
        self.outdegrees = self.get_outdegrees()
        
        if weights is not None:
            if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)):
                raise TypeError("weights must be None or a sequence of sequences of numbers")

            if len(weights) != self.N:
                raise ValueError("weights must have the same length as I")
        
            self.weights = []
            for i, (regs, row) in enumerate(zip(self.I, weights)):
                if not isinstance(row, Sequence):
                    raise TypeError(f"weights[{i}] must be a sequence of floats")
                if len(row) != len(regs):
                    raise ValueError(f"weights[{i}] must have the same length as I[{i}]")
                self.weights.append(np.array(row, dtype=float))
        else:
            self.weights = None


    @classmethod
    def from_DiGraph(
        cls,
        nx_DiGraph: "nx.DiGraph",
    ) -> "WiringDiagram":
        """
        Construct a WiringDiagram from a NetworkX directed graph.
    
        Each node in the directed graph represents a Boolean variable, and each
        directed edge ``u -> v`` indicates that variable ``u`` regulates variable
        ``v``.
    
        Parameters
        ----------
        nx_DiGraph : nx.DiGraph
            Directed graph whose nodes represent variables and whose edges
            represent regulatory interactions.
    
            Node attributes (optional)
                name : str
                    Name of the variable. If not provided, the node label is used
                    when possible.
    
            Edge attributes (optional)
                weight : float
                    Weight of the regulatory interaction from ``u`` to ``v``.
                    If present on all edges, weights are stored in the
                    ``weights`` attribute of the resulting WiringDiagram.
    
        Returns
        -------
        WiringDiagram
            Wiring diagram representing the topology of the directed graph.
    
        Notes
        -----
        - Nodes are ordered according to the iteration order of
          ``nx_DiGraph.nodes``.
        - Regulator lists are constructed from incoming edges
          (graph predecessors).
        - Edge weights are only stored if *all* edges define a ``'weight'``
          attribute.
    
        Examples
        --------
        >>> import networkx as nx
        >>> from boolforge import WiringDiagram
        >>> G = nx.DiGraph()
        >>> G.add_edges_from([(0, 1), (1, 2)])
        >>> W = WiringDiagram.from_DiGraph(G)
        >>> W.I
        [array([], dtype=int64), array([0]), array([1])]
        >>> W.get_constants()
        {0: True, 1: False, 2: False}
        """
        if not isinstance(nx_DiGraph, nx.DiGraph):
            raise TypeError("nx_DiGraph must be a networkx.DiGraph")
    
        # Preserve NetworkX node iteration order
        nodes = list(nx_DiGraph.nodes)
    
        # Map nodes to indices
        node_to_idx = {node: i for i, node in enumerate(nodes)}
    
        # Extract variable names
        variables = []
        for node in nodes:
            if 'name' in nx_DiGraph.nodes[node]:
                variables.append(str(nx_DiGraph.nodes[node]['name']))
            elif isinstance(node, str):
                variables.append(node)
            else:
                variables.append(f"x_{node}")
    
        # Build adjacency list I (regulators / predecessors)
        I = []
        for node in nodes:
            regulators = [node_to_idx[r] for r in nx_DiGraph.predecessors(node)]
            I.append(regulators)
    
        # Extract weights only if all edges define a weight
        weights = None
        if nx_DiGraph.number_of_edges() > 0 and all(
            'weight' in nx_DiGraph[u][v] for u, v in nx_DiGraph.edges
        ):
            weights = []
            for node in nodes:
                row = [
                    nx_DiGraph[r][node]['weight']
                    for r in nx_DiGraph.predecessors(node)
                ]
                weights.append(row)
    
        return cls(I=I, variables=variables, weights=weights)


    def to_DiGraph(self, USE_VARIABLE_NAMES: bool = True) -> nx.DiGraph:
        """
        Convert the wiring diagram into a NetworkX directed graph.
        
        Each node in the resulting graph represents a variable or constant in the
        wiring diagram. A directed edge ``u -> v`` indicates that node ``u``
        regulates node ``v``.
        
        Parameters
        ----------
        USE_VARIABLE_NAMES : bool, optional
            If True (default), nodes are labeled using the variable names stored in
            ``self.variables``. If False, nodes are labeled by integer indices
            ``0, 1, ..., N-1``.
        
        Returns
        -------
        nx.DiGraph
            Directed graph representing the wiring diagram. If interaction weights
            are present, they are stored as edge attributes under the key
            ``'weight'``.
        
        Notes
        -----
        - The node ordering follows the internal ordering of the wiring diagram.
        - Incoming edges of node ``i`` correspond to the regulators listed in
          ``self.I[i]``.
        - Edge weights are included only if the wiring diagram defines weights.
        """
        G = nx.DiGraph()
    
        if USE_VARIABLE_NAMES:
            nodes = list(self.variables)
            idx_to_node = {i: self.variables[i] for i in range(self.N)}
        else:
            nodes = list(range(self.N))
            idx_to_node = {i: i for i in range(self.N)}
    
        G.add_nodes_from(nodes)
    
        for i in range(self.N):
            target = idx_to_node[i]
            for j, reg in enumerate(self.I[i]):
                source = idx_to_node[reg]
                if self.weights is not None:
                    G.add_edge(source, target, weight=self.weights[i][j])
                else:
                    G.add_edge(source, target)
    
        return G
        

    def __getitem__(self, index):
        return self.I[index]
    

    def get_outdegrees(self) -> np.ndarray:
        """
        Compute the outdegree of each node.
    
        The outdegree of a node is the number of nodes it regulates, i.e.,
        the number of outgoing edges from that node in the wiring diagram.
    
        Returns
        -------
        np.ndarray
            One-dimensional array of length ``N`` containing the outdegree of
            each node.
        """
        outdegrees = np.zeros(self.N, dtype=int)
        for regulators in self.I:
            for regulator in regulators:
                outdegrees[regulator] += 1
        return outdegrees


    def get_constants(self, AS_DICT: bool = True) -> dict[int, bool] | np.ndarray:
        """
        Identify constant nodes in the wiring diagram.
    
        A node is considered a constant if it has no regulators, i.e.,
        if its indegree is zero.
    
        Parameters
        ----------
        AS_DICT : bool, optional
            If True (default), return a dictionary mapping node indices to
            Boolean values indicating whether each node is a constant.
            If False, return an array of indices corresponding to constant nodes.
    
        Returns
        -------
        dict[int, bool] or np.ndarray
            If ``AS_DICT`` is True, a dictionary where keys are node indices and
            values indicate whether the node is a constant.
            If ``AS_DICT`` is False, a one-dimensional array containing the
            indices of constant nodes.
        """
        is_constant = self.indegrees == 0
        if AS_DICT:
            return dict(enumerate(is_constant.tolist()))
        return np.where(is_constant)[0]


    def get_strongly_connected_components(self) -> list:
        """
        Compute the strongly connected components of the wiring diagram.
    
        A strongly connected component (SCC) is a maximal set of nodes such that
        every node in the set is reachable from every other node via directed
        paths.
    
        Returns
        -------
        list of set of int
            List of strongly connected components, where each component is
            represented as a set of node indices.
        """
        edges = [(int(reg), target) for target, regs in enumerate(self.I) for reg in regs]
        subG = nx.DiGraph(edges)
        return list(nx.strongly_connected_components(subG))


    def get_modular_structure(self) -> set[tuple[int, int]]:
        """
        Compute the modular (condensation) structure of the wiring diagram.
    
        The modular structure is represented as a directed acyclic graph (DAG)
        whose nodes correspond to strongly connected components (SCCs) of the
        wiring diagram. A directed edge ``(i, j)`` indicates that there exists
        at least one regulatory interaction from SCC ``i`` to SCC ``j``.
    
        Returns
        -------
        set of tuple of int
            Set of directed edges representing the condensation DAG, where each
            edge ``(i, j)`` denotes a regulation from SCC ``i`` to SCC ``j``.
            
        Notes
        -----
        SCC indices correspond to the ordering returned by
        ``get_strongly_connected_components``.
        """
        sccs = self.get_strongly_connected_components()
    
        # Map node index -> SCC index
        scc_dict = {}
        for idx, scc in enumerate(sccs):
            for node in scc:
                scc_dict[node] = idx
    
        dag = set()
    
        for target, regulators in enumerate(self.I):
            for j, regulator in enumerate(regulators):
                src = scc_dict[int(regulator)]
                tgt = scc_dict[target]
    
                if src == tgt:
                    continue
    
                if self.weights is not None:
                    if np.isnan(self.weights[target][j]):
                        continue
    
                dag.add((src, tgt))
    
        return dag


    def get_ffls(
        self,
    ) -> tuple[list[list[int]], list[list[float]] | None]:
        """
        Identify feed-forward loops (FFLs) in the wiring diagram.
    
        A feed-forward loop is a triplet of nodes ``(i, j, k)`` such that
        ``i -> j``, ``j -> k``, and ``i -> k`` are all regulatory interactions.
    
        Returns
        -------
        tuple
            A tuple ``(ffls, types)`` where:
    
            - ``ffls`` is a list of feed-forward loops, each represented as
              ``[i, j, k]``.
            - ``types`` is a list of corresponding edge-weight triplets
              ``[i -> k, i -> j, j -> k]`` if interaction weights are defined,
              or ``None`` otherwise.
        """
        # Build inverted wiring diagram: regulator -> (target, index in I[target])
        I_inv = [[] for _ in range(self.N)]
        for target, regulators in enumerate(self.I):
            for idx, regulator in enumerate(regulators):
                I_inv[int(regulator)].append((target, idx))
    
        ffls: list[list[int]] = []
        types: list[list[float]] | None = [] if self.weights is not None else None
    
        for i in range(self.N):  # master regulator
            targets_i = {t for (t, _) in I_inv[i]}
    
            for j, idx_ij in I_inv[i]:  # i -> j
                if i == j:
                    continue
    
                targets_j = {t for (t, _) in I_inv[j]}
                common_targets = targets_i & targets_j
    
                for k in common_targets:
                    if k == i or k == j:
                        continue
    
                    ffls.append([i, j, k])
    
                    if types is not None:
                        # locate indices for k safely
                        idx_ik = next(idx for idx, r in enumerate(self.I[k]) if r == i)
                        idx_jk = next(idx for idx, r in enumerate(self.I[k]) if r == j)
    
                        direct = self.weights[k][idx_ik]
                        indirect1 = self.weights[j][idx_ij]
                        indirect2 = self.weights[k][idx_jk]
    
                        types.append([direct, indirect1, indirect2])
    
        return ffls, types
        
        
    def get_fbls(self, max_length: int = 4) -> list[list[int]]:
        """
        Identify feedback loops (simple directed cycles).
    
        A feedback loop is defined as a simple directed cycle in the underlying
        wiring diagram. This method enumerates elementary circuits using a
        variant of Johnson's algorithm, restricted to cycles of length at most
        ``max_length``.
    
        Parameters
        ----------
        max_length : int, optional
            Maximum length of feedback loops to consider. Cycles longer than
            ``max_length`` are not returned. Default is 4.
    
        Returns
        -------
        list of list of int
            List of feedback loops, where each loop is represented as a list
            of node indices forming a directed cycle. Self-loops are returned
            as single-element lists.
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


    def get_types_of_fbls(
        self,
        fbls: list[list[int]],
    ) -> tuple[list[float], list[int]] | None:
        """
        Determine the types of feedback loops based on interaction weights.
    
        For each feedback loop, this method computes the product of the
        interaction weights along the cycle and counts the number of
        negative regulations.
    
        Parameters
        ----------
        fbls : list of list of int
            List of feedback loops, where each loop is represented as a list
            of node indices forming a directed cycle.
    
        Returns
        -------
        tuple of list or None
            If interaction weights are defined, returns a tuple
            ``(types, n_negative)``, where:
    
            - ``types`` is a list containing the product of edge weights for
              each feedback loop. Interpretation:
              'non-essential' : np.nan, 'conditional' : 0, 
              'positive' : 1, 'negative' : -1
            - ``n_negative`` is a list containing the number of negative
              regulations in each feedback loop.
    
            If interaction weights are not defined, returns ``None``.
        """
        if self.weights is None:
            return None
    
        types: list[float] = []
        n_negative: list[int] = []
    
        for fbl in fbls:
            all_weights = []
    
            for u, v in zip(fbl, fbl[1:] + [fbl[0]]):
                # find index of regulator u in I[v]
                for idx, reg in enumerate(self.I[v]):
                    if reg == u:
                        w = self.weights[v][idx]
                        break
                else:
                    raise ValueError(f"No edge {u} -> {v} found in wiring diagram")
    
                all_weights.append(w)
    
            types.append(float(np.prod(all_weights)))
            n_negative.append(int(sum(w < 0 for w in all_weights)))
    
        return types, n_negative
    
