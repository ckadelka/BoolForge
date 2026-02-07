#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wiring-diagram representation for Boolean networks.

This module defines the :class:`~boolforge.WiringDiagram` class, which encodes
the directed regulatory topology of a Boolean network independently of any
Boolean update functions.

A wiring diagram specifies, for each node, the set of regulating nodes
(predecessors). Nodes with no regulators are source nodes in the wiring diagram.
Whether such nodes act as constants or identity nodes in Boolean networks
can only be determined after Boolean update functions are assigned.
"""

import warnings
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

    Nodes with zero indegree are source nodes. Whether a source node represents
    a constant, an identity node, or a dynamic variable is determined only after
    Boolean update functions are assigned.

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
        Total number of nodes, including source nodes.
    indegrees : np.ndarray[int]
        Indegree of each node.
    outdegrees : np.ndarray[int]
        Outdegree of each node.
    weights : list[np.ndarray] or None
        Interaction weights associated with ``I`` if provided.

    Notes
    -----
    - Node indices are zero-based.
    - Source nodes are identified solely by indegree == 0.
    - The wiring diagram encodes topology only and does not define dynamics.

    Examples
    --------
    Nodes with zero indegree are interpreted as source nodes.
    
    >>> from boolforge import WiringDiagram
    >>> I = [
            [],        # node 0 has no regulators -> source node
            [0],       # node 1 is regulated by node 0
            [0, 1],    # node 2 is regulated by nodes 0 and 1
        ]
    >>> W = WiringDiagram(I)
    
    >>> W.N
    3
    
    >>> W.get_source_nodes(AS_DICT=True)
    {0: True, 1: False, 2: False}
    
    >>> W.get_source_nodes(AS_DICT=False)
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
        Source nodes are identified automatically as nodes with zero indegree.
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
        - Edge weights are only stored if all edges define a ``'weight'``
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
        >>> W.get_source_nodes()
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
        
        A directed edge ``u -> v`` indicates that node ``u``
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
        
    def __str__(self):
        return (
            f"WiringDiagram(N={self.N}, "
            f"indegrees={self.indegrees.tolist()})"
        )

    def __getitem__(self, index):
        return self.I[index]

    def __repr__(self):
        return f"{type(self).__name__}(N={self.N})"

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


    def get_source_nodes(
        self, 
        AS_DICT: bool = True
    ) -> dict[int, bool] | np.ndarray:
        """
        Identify source nodes in the wiring diagram.
    
        A source node is a node with zero indegree. Source nodes represent
        inputs to the wiring diagram; whether they act as constants or
        identity nodes in Boolean networks depends on the associated
        Boolean update functions and is not determined at the wiring-diagram
        level.
    
        Parameters
        ----------
        AS_DICT : bool, optional
            If True (default), return a dictionary mapping node indices to
            Boolean values indicating whether each node is a source node.
            If False, return an array of indices corresponding to source nodes.
    
        Returns
        -------
        dict[int, bool] or np.ndarray
            If ``AS_DICT`` is True, a dictionary where keys are node indices and
            values indicate whether the node is a source node.
            If ``AS_DICT`` is False, a one-dimensional array containing the
            indices of source nodes.
        """
        is_source = self.indegrees == 0
        if AS_DICT:
            return dict(enumerate(is_source.tolist()))
        return np.where(is_source)[0]


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
    
                        direct = float(self.weights[k][idx_ik])
                        indirect1 = float(self.weights[j][idx_ij])
                        indirect2 = float(self.weights[k][idx_jk])
    
                        types.append([direct, indirect1, indirect2])
        
        return_dict = {'FFLs' : ffls}
        if types is not None:
            return_dict['Types'] = types
        
        return return_dict
    
        
        
    def get_fbls(self, max_length: int = 4, CLASSIFY : bool = False) -> dict:
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
        CLASSIFY : bool, optional
            If True and interaction weights are available, the type of each
            feedback loop is determined.
        
        Returns
        -------
        dict
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
        
        return_dict = {'FBLs' : fbls}
        if self.weights is not None and CLASSIFY == True:
            types,n_negative = self._get_types_of_fbls(fbls)
            return_dict['Types'] = types
            return_dict['NumberNegativeEdges'] = n_negative
        return return_dict


    def _get_types_of_fbls(
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
    
    def plot_modular_structure(
        self,
        ax=None,
        show=True,
        node_labels: bool = True,
        max_nodes: int = 50,
        curviness: float = 0.25,
    ):
        """
        Plot the wiring diagram as a directed acyclic graph of strongly connected components.
    
        The wiring diagram is first condensed into its strongly connected components (SCCs),
        yielding a directed acyclic graph (DAG). Each node in the plot represents one SCC.
    
        The layout is hierarchical (top-to-bottom) using topological generations, making
        feed-forward structure visually apparent, while condensing feedback loops.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, a new figure and axis are created.
        show : bool, default=True
            Whether to call ``plt.show()`` after plotting.
        node_labels : bool, default=True
            Whether to label SCC nodes by their size (only shown for SCCs of size > 1).
        max_nodes : int, default=50
            If the number of SCCs exceeds this value, edges are sparsified to reduce clutter.
        curviness : float, default=0.25
            Curvature of edges spanning multiple layers (0 = straight).
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis containing the plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, FancyArrowPatch
        
        
        def _ellipse_boundary_point(x0, y0, x1, y1, a, b):
            """
            Intersection of ray from (x0,y0) toward (x1,y1)
            with ellipse centered at (x0,y0) with semi-axes a,b.
            """
            dx = x1 - x0
            dy = y1 - y0
            if dx == 0 and dy == 0:
                return x0, y0
        
            t = 1.0 / np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
            return x0 + t * dx, y0 + t * dy
    
        N = self.N
        I = self.I
    
        # ------------------------------------------------------------------
        # Build directed graph of the wiring diagram
        # ------------------------------------------------------------------
        g_full = nx.DiGraph()
        g_full.add_nodes_from(range(N))
    
        outdegrees = np.zeros(N, dtype=int)
        for target, regulators in enumerate(I):
            for r in regulators:
                g_full.add_edge(r, target)
                outdegrees[r] += 1
    
        # ------------------------------------------------------------------
        # Compute SCCs
        # ------------------------------------------------------------------
        sccs = list(nx.strongly_connected_components(g_full))
        scc_sizes = np.array([len(scc) for scc in sccs])
    
        scc_index = {}
        for i, scc in enumerate(sccs):
            for node in scc:
                scc_index[node] = i
                
        if len(sccs)<2:
            warnings.warn('No plot created. The network consists of a single SCC', UserWarning)
            return None
    
        # ------------------------------------------------------------------
        # Build SCC DAG
        # ------------------------------------------------------------------
        dag_edges = set()
        for u, v in g_full.edges():
            su, sv = scc_index[u], scc_index[v]
            if su != sv:
                dag_edges.add((su, sv))
    
        dag = nx.DiGraph(dag_edges)
    
        if dag.number_of_nodes() > max_nodes:
            dag = nx.DiGraph(
                (u, v) for (u, v) in dag_edges
                if scc_sizes[u] > 1 or scc_sizes[v] > 1
            )
    
        # ------------------------------------------------------------------
        # Node types
        # ------------------------------------------------------------------
        types = np.zeros(len(sccs), dtype=int)
    
        for i, scc in enumerate(sccs):
            if scc_sizes[i] > 1:
                types[i] = 2
            else:
                node = next(iter(scc))
                if (g_full.in_degree(node) == 0 or
                    (g_full.in_degree(node) == 1 and
                     list(g_full.predecessors(node))[0] == node)
                ):
                    types[i] = -1
                elif outdegrees[node] == 0:
                    types[i] = 1
                else:
                    types[i] = 0
    
        # ------------------------------------------------------------------
        # Hierarchical layout: initial placement by generations
        # ------------------------------------------------------------------
        pos = {}
        layers = []
        
        generations = list(nx.topological_generations(dag))
        #max_n_per_generation = max([len(gen) for gen in generations])
        #n_generations = len(generations)
        for layer, generation in enumerate(generations):
            gen = list(generation)
            layers.append(gen)
    
            k = len(gen)
            if k == 1:
                pos[gen[0]] = (0.0, -layer)
            else:
                xs = np.linspace(-0.5, 0.5, k)
                for x, node in zip(xs, gen):
                    pos[node] = (x, -layer)
    
        # ------------------------------------------------------------------
        # NEW: barycentric horizontal refinement (FFL fix)
        # ------------------------------------------------------------------
        for gen in layers[1:]:  # skip first layer
            for v in gen:
                preds = list(dag.predecessors(v))
                if preds:
                    x_mean = np.mean([pos[p][0] for p in preds])
                    y = pos[v][1]
                    pos[v] = (x_mean, y)
    
        # tiny deterministic jitter to avoid exact overlaps
        eps = 1e-3
        for i, v in enumerate(dag.nodes()):
            x, y = pos[v]
            pos[v] = (x + eps * (i % 7), y)
            
        # ------------------------------------------------------------------
        # Post-process: spread nodes within each layer to use full width
        # ------------------------------------------------------------------
        max_width = 3
        for gen in layers:
            xs = np.array([pos[v][0] for v in gen])
            if len(xs) <= 1:
                continue
        
            # Sort nodes by x
            order = np.argsort(xs)
            gen_sorted = [gen[i] for i in order]
        
            # Reassign evenly spaced x positions
            width = max(1.0, min(len(gen_sorted) / 3, max_width))
            new_xs = np.linspace(-width / 2, width / 2, len(gen_sorted))
        
            for v, x in zip(gen_sorted, new_xs):
                pos[v] = (x, pos[v][1])
        
        # ------------------------------------------------------------------
        # Vertical micro-staggering within layers (reduce edge overlap)
        # ------------------------------------------------------------------
        epsilon = 0.25  # vertical spacing scale
        
        for gen in layers:
            if len(gen) <= 3:
                continue
        
            # sort nodes left-to-right
            gen_sorted = sorted(gen, key=lambda v: pos[v][0])
        
            for i, v in enumerate(gen_sorted):
                x, y = pos[v]
                # pattern: middle, down, up, middle, down, up, ...
                offset = (i % 3) * epsilon
                pos[v] = (x, y + offset)
    
        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------
        xs = np.array([x for x, y in pos.values()])
        ys = np.array([y for x, y in pos.values()])
        
        x_span = xs.max() - xs.min()
        y_span = ys.max() - ys.min()
        
        target_aspect = 1.5   # width / height
        current_aspect = x_span / max(y_span, 1e-6)

        if current_aspect < target_aspect:
            scale = target_aspect / current_aspect
            for v, (x, y) in pos.items():
                pos[v] = (x * scale, y)
        
        fig_width  = max(6, 0.6 * x_span * scale)
        fig_height = max(6, 0.6 * abs(y_span))
        
        
        if ax is None:
            _, ax = plt.subplots(figsize=(fig_width, fig_height))
            
        color_map = {
            -1: "#eeeeee",
             0: "#ffcccc",
             1: "#eeeeee",
             2: "#ff9999",
        }
    
        labels = None
        if node_labels:
            labels = {
                n: 'SCC of size '+str(scc_sizes[n]) if scc_sizes[n] > 1
                else self.variables[list(sccs[n])[0]]
                for n in dag.nodes()
            }
    
    
        xs = [x for x, y in pos.values()]
        ys = [y for x, y in pos.values()]
        
        pad_x = 1.0
        pad_y = 1.0
        
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    
        texts = {}
        for n, (x, y) in pos.items():
            label = labels[n] if labels else ""
            texts[n] = ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=8,
                zorder=3
            )
        
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


        # ------------------------------------------------------------------
        # Fix horizontal overlaps using measured text widths
        # ------------------------------------------------------------------
        half_width = {}
        half_height = {}
        
        for n, text in texts.items():
            bbox = text.get_window_extent(renderer=renderer)
            inv = ax.transData.inverted()
            (x0, y0), (x1, y1) = inv.transform(
                [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)]
            )
        
            half_width[n]  = 0.65 * (x1 - x0)
            half_height[n] = 0.90 * (y1 - y0)
        
        
        min_gap = 0.1  # extra spacing between nodes
        
        for gen in layers:
            if len(gen) <= 1:
                continue
        
            # sort left-to-right
            gen_sorted = sorted(gen, key=lambda v: pos[v][0])
        
            x_cursor = pos[gen_sorted[0]][0]
            new_pos = {gen_sorted[0]: x_cursor}
        
            for prev, curr in zip(gen_sorted[:-1], gen_sorted[1:]):
                required = (
                    half_width[prev] +
                    half_width[curr] +
                    min_gap
                )
                x_cursor = max(pos[curr][0], x_cursor + required)
                new_pos[curr] = x_cursor
        
            # re-center layer
            center = np.mean(list(new_pos.values()))
            for v in gen_sorted:
                pos[v] = (new_pos[v] - center, pos[v][1])
        
        # ------------------------------------------------------------------
        # FINAL barycentric refinement (FFL alignment fix)
        # ------------------------------------------------------------------

        for gen in layers[1:]:
            for v in gen:
                preds = list(dag.predecessors(v))
                if preds:
                    x_mean = np.mean([pos[p][0] for p in preds])
                    #pos[v] = (x_mean, pos[v][1])
                    alpha = 0.1   # 0 = no barycentric, 1 = full snap
                    x_new = alpha * x_mean + (1 - alpha) * pos[v][0]
                    pos[v] = (x_new, pos[v][1])
        
        node_layer = {}
        for i, gen in enumerate(layers):
            for v in gen:
                node_layer[v] = i

        for t in texts.values():
            t.remove()
        
        ellipse_axes = {}   # <-- ADD THIS
        
        for n, (x, y) in pos.items():
            a = half_width[n]
            b = half_height[n]#max(half_height[n], 0.3 * (2 * a))  # since height = max(2b, 0.6w)
        
            ellipse_axes[n] = (a, b)
        
        for n, (x, y) in pos.items():
            a, b = ellipse_axes[n]
        
            ellipse = Ellipse(
                (x, y),
                width=2*a,
                height=2*b,
                facecolor=color_map[types[n]],
                edgecolor="black",
                zorder=2,
            )
            ax.add_patch(ellipse)


        for u, v in dag.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
        
            a0, b0 = ellipse_axes[u]
            a1, b1 = ellipse_axes[v]
        
            sx, sy = _ellipse_boundary_point(x0, y0, x1, y1, a0, b0)
            tx, ty = _ellipse_boundary_point(x1, y1, x0, y0, a1, b1)
        
            if node_layer[v] - node_layer[u] > 1:
                if node_layer[v] - node_layer[u] > 1 and curviness != 0.0:
                    sign = -1 if pos[u][0] + pos[v][0] > 0 else 1
                    conn = f"arc3,rad={sign * curviness}"
                else:
                    conn = "arc3"
            else:
                conn = "arc3"
                
            arrow = FancyArrowPatch(
                (sx, sy),
                (tx, ty),
                arrowstyle='-|>',
                mutation_scale=10,
                linewidth=1.2,
                color='black',
                connectionstyle=conn,   # curved or straight
                zorder=1,
            )
            ax.add_patch(arrow)
            
        #Draw text again, now at the correct positions
        for n, (x, y) in pos.items():
            ax.text(
                x, y, labels[n],
                ha="center", va="center",
                fontsize=7,
                zorder=4,   # on top
            )
    
        ax.set_autoscale_on(False)

        # --------------------------------------------------
        # FINAL hard limits: guarantee everything is visible
        # --------------------------------------------------
        xs = []
        ys = []
        
        for n, (x, y) in pos.items():
            a, b = ellipse_axes[n]
            xs.extend([x - a, x + a])
            ys.extend([y - b, y + b])
        
        PAD_X = 0.5
        PAD_Y = 0.5
        
        ax.set_xlim(min(xs) - PAD_X, max(xs) + PAD_X)
        ax.set_ylim(min(ys) - PAD_Y, max(ys) + PAD_Y)
        
        ax.set_autoscale_on(False)

        ax.set_axis_off()
    
        if show:
            plt.show()
    
        return ax
    
    
    def plot(
        self,
        max_expanded_sccs: int = 4,
        min_scc_size: int = 2,
        show: bool = True,
        curviness: float = 0.25,
    ):
        """
        Plot an integrated overview of the wiring diagram.
    
        The plot consists of:
          1) A top panel showing the modular structure of the network as a DAG of
             strongly connected components (SCCs).
          2) Bottom panels showing the internal wiring of selected SCCs using a
             circular layout.
    
        By default, the largest SCCs of size >= ``min_scc_size`` are expanded,
        up to ``max_expanded_sccs``.
    
        Parameters
        ----------
        max_expanded_sccs : int, default=4
            Maximum number of SCCs to expand and show in detail.
        min_scc_size : int, default=2
            Minimum SCC size to be eligible for expansion.
        show : bool, default=True
            Whether to call ``plt.show()`` at the end.
        curviness : float, default=0.25
            Curvature of edges spanning multiple layers in the modular graph (0 = straight).
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        """
        import matplotlib.pyplot as plt
    
        N = self.N
        I = self.I
    
        # ------------------------------------------------------------
        # Build full directed graph
        # ------------------------------------------------------------
        g = nx.DiGraph()
        g.add_nodes_from(range(N))
        for target, regulators in enumerate(I):
            for r in regulators:
                g.add_edge(r, target)
    
        # ------------------------------------------------------------
        # Compute SCCs
        # ------------------------------------------------------------
        sccs = list(nx.strongly_connected_components(g))
        sccs = [sorted(scc) for scc in sccs]
    
        node_to_scc = {}
        for i, scc in enumerate(sccs):
            for v in scc:
                node_to_scc[v] = i
                
        G_scc = nx.DiGraph()
        G_scc.add_nodes_from(range(len(sccs)))
        
        for u, v in g.edges:
            su = node_to_scc[u]
            sv = node_to_scc[v]
            if su != sv:
                G_scc.add_edge(su, sv)
    
        # Select SCCs to expand
        expandable = [scc for scc in sccs if len(scc) >= min_scc_size]
        expandable.sort(key=len, reverse=True)
        expanded_sccs = expandable[:max_expanded_sccs]
    
        n_expanded = len(expanded_sccs)
    
        # ------------------------------------------------------------
        # Figure and GridSpec
        # ------------------------------------------------------------
        if n_expanded == 0: #if just showing the modular graph
            fig = plt.figure(figsize=(8, 4))
            gs = fig.add_gridspec(1, 1)
            ax_top = fig.add_subplot(gs[0, 0])
        elif len(sccs)==1: #if just showing the single SCC
            fig = plt.figure(figsize=(4 * n_expanded, 4))
            gs = fig.add_gridspec(1)
        else: #if showing both
            fig = plt.figure(figsize=(4 * n_expanded, 6))
            gs = fig.add_gridspec(
                2,
                n_expanded,
                height_ratios=[2.2, 1.5],
            )
            ax_top = fig.add_subplot(gs[0, :])
    
        # ------------------------------------------------------------
        # Top panel: modular structure
        # ------------------------------------------------------------
        if len(sccs)>1:
            self.plot_modular_structure(ax=ax_top, show=False,curviness=curviness)
            ax_top.set_title("Modular structure (DAG of SCCs)")
    
        # ------------------------------------------------------------
        # Bottom panels: internal SCC structure
        # ------------------------------------------------------------
        for j, scc in enumerate(expanded_sccs):
            if len(sccs)>1:
                ax = fig.add_subplot(gs[1, j])
            else:
                ax = fig.add_subplot(gs[j])
            
            C = set(scc)

            # direct external inputs only
            inputs = {
                u
                for v in C
                for u in g.predecessors(v)
                if u not in C
            }
            
            nodes_local = C | inputs
            subg = g.subgraph(nodes_local).copy()
    
            for u, v in list(subg.edges):
                if u not in nodes_local or v not in C:
                    subg.remove_edge(u, v)
    
            #subg = g.subgraph(scc).copy()
    
            pos = {}

            epsilon = 0.35  # vertical spacing scale
            
            # Inputs on top
            inputs = sorted(inputs)
            k_in = len(inputs)
            for j, v in enumerate(inputs):
                offset = 0 if k_in <= 3 else (j % 3 - 1) * epsilon
                pos[v] = ((j - (k_in - 1) / 2) /k_in*2, 2.0 + offset)
            
            # SCC nodes in a circle below
            pos_scc = nx.circular_layout(C, scale=1.0, center=(0.0, 0.0))
            pos.update(pos_scc)   
            
            # Color nodes: all feedback (same SCC)
            node_colors = [
                "#ff9999" if v in C else "#eeeeee"
                for v in subg.nodes
            ]
            
            nx.draw_networkx(
                subg,
                pos=pos,
                ax=ax,
                node_color=node_colors,
                node_size=200,
                labels={v: self.variables[v] for v in subg.nodes()},
                font_size=9,
                with_labels=True,
            )
    
            ax.set_title(f"SCC of size {len(scc)}")
            ax.set_axis_off()
    
        fig.tight_layout()
    
        if show:
            plt.show()
    
        return fig