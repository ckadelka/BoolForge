#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides utility functions and helper routines used throughout
BoolForge.

The :mod:`~boolforge.utils` module includes low-level operations for binary and
decimal conversions, truth table manipulations, and combinatorial helper
functions. These utilities are used internally by
:class:`~boolforge.BooleanFunction` and :class:`~boolforge.BooleanNetwork`
classes to enable efficient representation and analysis of Boolean functions
and networks.

Several functions in this module can take advantage of Numba-based JIT
compilation for significant speedups when processing large truth tables or
performing repeated bit-level operations. Installation of Numba is therefore
**encouraged** but **optional**; pure Python fallbacks are provided for all
functions.

Example
-------
>>> from boolforge import utils
>>> utils.bin2dec([1, 0, 1])
5
>>> utils.dec2bin(5, 3)
array([1, 0, 1])
"""


##Imports
from __future__ import annotations
import numpy as np
import random as _py_random
from numpy.random import Generator as _NPGen, RandomState as _NPRandomState, SeedSequence, default_rng

from typing import Union
from typing import Optional

def _coerce_rng(rng : Union[int, _NPGen, _NPRandomState, _py_random.Random, None] = None) -> _NPGen:
    """
    Return a NumPy Generator given a variety of rng-like inputs.

    **Accepts:**
        
      - None                -> default_rng()
      - int (seed)          -> default_rng(seed)
      - np.random.Generator -> returned as-is
      - np.random.RandomState -> converted via SeedSequence
      - random.Random       -> converted via SeedSequence

    **Raises:**
        
        - TypeError: for unsupported inputs.
    """
    if rng is None:
        return default_rng()
    if isinstance(rng, _NPGen):
        return rng
    if isinstance(rng, (int, np.integer)):
        return default_rng(int(rng))
    if isinstance(rng, _NPRandomState):
        # derive robust entropy from the legacy RNG
        entropy = rng.randint(0, 2**32, size=4, dtype=np.uint32)
        return default_rng(SeedSequence(entropy))
    if isinstance(rng, _py_random.Random):
        entropy = [_py_random.Random(rng.random()).getrandbits(32) for _ in range(4)]
        # simpler: entropy = [rng.getrandbits(32) for _ in range(4)]
        return default_rng(SeedSequence(entropy))
    raise TypeError(f"Unsupported rng type: {type(rng)!r}")

def is_float(element: any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def bin2dec(binary_vector : list) -> int:
    """
    Convert a binary vector to an integer.

    **Parameters:**
        
        - binary_vector (list[int]): List containing binary digits (0 or 1).

    **Returns:**
        
        - int: Integer value converted from the binary vector.
    """
    decimal = 0
    for bit in binary_vector:
        decimal = (decimal << 1) | bit
    return int(decimal)


def dec2bin(integer_value : int, num_bits : int) -> list:
    """
    Convert an integer to a binary vector.

    **Parameters:**
        
        - integer_value (int): Integer value to be converted.
        - num_bits (int): Number of bits in the binary representation.

    **Returns:**
        
        - list[int]: List containing binary digits (0 or 1).
    """
    binary_string = bin(integer_value)[2:].zfill(num_bits)
    return [int(bit) for bit in binary_string]

left_side_of_truth_tables = {}

def get_left_side_of_truth_table(N):
    if N in left_side_of_truth_tables:
        left_side_of_truth_table = left_side_of_truth_tables[N]
    else:
        #left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=self.N)))
        vals = np.arange(2**N, dtype=np.uint64)[:, None]              # shape (2^n, 1)
        masks = (1 << np.arange(N-1, -1, -1, dtype=np.uint64))[None]  # shape (1, n)
        left_side_of_truth_table = ((vals & masks) != 0).astype(np.uint8)
        left_side_of_truth_tables[N] = left_side_of_truth_table
    return left_side_of_truth_table


def find_all_indices(arr,el):
    '''
    Given a list arr, this function returns a list of all the indices i where arr[i]==el.
    If el not in arr, it raises a ValueError.
    '''
    res=[]
    for i,a in enumerate(arr):
        if a==el:
            res.append(i)
    if res==[]:
        raise ValueError('The element is not in the array at all')
    return res


def check_if_empty(my_list : Union[list, np.ndarray]) -> bool:
    """
    Check if the provided list or NumPy array is empty.

    **Parameters:**
        
        - my_list (list[Variant], np.ndarray[Variant]): The list or array
          to check.

    **Returns:**
        
        - bool: True if my_list is empty (or has size 0 for a NumPy array),
          False otherwise.
    """
    if isinstance(my_list, np.ndarray):
        if my_list.size == 0:
            return True
    elif my_list == []:
        return True
    return False
    
    
def is_list_or_array_of_ints(x : Union[list, np.ndarray],
    required_length : int = None) -> bool:
    """
    Determines if the array-like x contains elements of the 'integer' type.
    
    **Parameters**:
        - x (list | np.ndarray): The array-like to check.
        - required_length (int | None, optional): The exact length x must have
          to return true. If None, this check is ignored.
          
    **Returns**:
        - bool: True if x holds elements of type int or np.integer. If
          required_length is not None, then the length of x must equal required_length
          as well. Returns false otherwise.
    """
    # Case 1: Python list
    if isinstance(x, list):
        return (required_length is None or len(x) == required_length) and all(isinstance(el, (int, np.integer)) for el in x)
    
    # Case 2: NumPy array
    if isinstance(x, np.ndarray):
        return (required_length is None or x.shape == (required_length,)) and np.issubdtype(x.dtype, np.integer)
    
    return False

def is_list_or_array_of_floats(x : Union[list, np.ndarray],
    required_length : int = None) -> bool:
    """
    Determines if the array-like x contains elements of the 'floating point' type.
    
    **Parameters**:
        - x (list | np.ndarray): The array-like to check.
        - required_length (int | None, optional): The exact length x must have
          to return true. If None, this check is ignored.
          
    **Returns**:
        - bool: True if x holds elements of type float or np.floating. If
          required_length is not None, then the length of x must equal required_length
          as well. Returns false otherwise.
    """
    # Case 1: Python list
    if isinstance(x, list):
        return (required_length is None or len(x) == required_length) and all(isinstance(el, (float, np.floating)) for el in x)
    
    # Case 2: NumPy array
    if isinstance(x, np.ndarray):
        return (required_length is None or x.shape == (required_length,)) and np.issubdtype(x.dtype, np.floating)
    
    return False


def bool_to_poly(f : list, variables : Optional[list] = None,
    prefix : str = '') -> str:
    """
    Transform a Boolean function from truth table format to polynomial format
    in non-reduced DNF.

    **Parameters:**
        
        - f (list[int]): Boolean function as a vector (list of length 2^n,
          where n is the number of inputs).
          
        - variables (list[str] | None, optional): List of indices to use for
          variable naming. If empty or not matching the required number,
          defaults to list(range(n)).
          
        - prefix (str, optional): Prefix for variable names in the polynomial,
          default ''.

    **Returns:**
        
        - str: A string representing the Boolean function in disjunctive
          normal form (DNF).
    """
    len_f = len(f)
    n = int(np.log2(len_f))
    if variables is None or len(variables) != n:
        prefix = 'x'
        variables = [prefix+str(i) for i in range(n)]
    left_side_of_truth_table = get_left_side_of_truth_table(n)
    num_values = 2 ** n
    text = []
    for i in range(num_values):
        if f[i] == True:
            monomial = ' * '.join([('%s' % (v)) if entry == 1 else ('(1 - %s)' % (v)) 
                                  for v, entry in zip(variables, left_side_of_truth_table[i])])
            text.append(monomial)
    if text != []:
        return ' + '.join(text)
    else:
        return '0'


def f_from_expression(expr : str, max_degree : int = 16) -> tuple:
    """
    Extract a Boolean function from a string expression.

    The function converts an input expression into its truth table representation.
    The expression can include Boolean operators and comparisons, and the order
    of variables is determined by their first occurrence in the expression.

    **Parameters:**
        
        - expr (str): A text string containing an evaluable Boolean expression.

    **Returns:**
        
        - tuple[list[int], list[str]]:
            
            - f (list[int]): The right-hand side of the Boolean function
              (truth table) as a list of length 2**n, where n is the number
              of inputs.
              
            - var (list[str]): A list of variable names (of length n) in the
              order they were encountered.
    
    **Examples:**
        
        >>> f_from_expression('A AND NOT B') #nested canalizing function
        ([0, 0, 1, 0], ['A', 'B'])
        
        >>> f_from_expression('x1 + x2 + x3 > 1') #threshold function
        ([0, 0, 0, 1, 0, 1, 1, 1], ['x1', 'x2', 'x3'])
        
        >>> f_from_expression('(x1 + x2 + x3) % 2 == 0') % linear (XOR) function
        ([1, 0, 0, 1, 0, 1, 1, 0], ['x1', 'x2', 'x3'])
    """

    expr_mod = expr.replace('(', ' ( ').replace(')', ' ) ').replace('!','not ').replace('~','not ')
    expr_split = expr_mod.split(' ')
    variables = []
    dict_var = dict()
    n_var = 0
    for i, el in enumerate(expr_split):
        if el not in ['',' ','(',')','and','or','not','AND','OR','NOT','&','|','+','-','*','%','>','>=','==','<=','<'] and not is_float(el):#el.isdigit():
            try:
                new_var = dict_var[el]
            except KeyError:
                new_var = 'x[%i]' % n_var
                dict_var.update({el: new_var})
                variables.append(el)
                n_var += 1
            expr_split[i] = el
        elif el in ['AND', 'and']:
            expr_split[i] = '&'
        elif el in ['OR', 'or']:
            expr_split[i] = '|'
        elif el in ['NOT', 'not']:
            expr_split[i] = '~'
    expr_mod = ' '.join(expr_split)
    
    if n_var <= max_degree:
        truth_table = get_left_side_of_truth_table(n_var)
        local_dict = {var: truth_table[:, i].astype(bool) for i, var in enumerate(variables)}
        f = eval(expr_mod, {"__builtins__": None}, local_dict)
    else:
        f = []
    return np.array(f,dtype=np.uint8), np.array(variables)


def flatten(l : Union[list, np.array]) -> list:
    """
    Converts an array of arrays into an array containing the elements of each
    subarray, effectively reducing the dimension of the array by 1.
    
    **Paramters**:
        - l (list[list[Variant] | np.array[Variant]] | np.array[list[Variant]
          | np.array[Variant]]): Array of arrays to reduce the dimension of.
    
    **Returns**:
        - list[Variant]: Array with its dimensions reduced by 1.
    """
    return [item for sublist in l for item in sublist]


def get_layer_structure_of_an_NCF_given_its_Hamming_weight(n : int, w : int) -> tuple:
    """
    Compute the canalizing layer structure of a nested canalizing function
    (NCF) given its Hamming weight.

    There exists a bijection between the Hamming weight (with w equivalent to
    2^n - w) and the canalizing layer structure of an NCF. The layer structure
    is represented as [k_1, ..., k_r], where each k_i ≥ 1 and, if n > 1, for
    the last layer k_r ≥ 2.

    **Parameters:**
        
        - n (int): Number of inputs (variables) of the NCF.
        - w (int): Odd Hamming weight of the NCF, i.e., the number of 1s in
          the 2^n-vector representation of the function.

    **Returns:**
        
        - layer_structure_NCF (list[int]): A list [k_1, ..., k_r]
          describing the number of variables in each layer.

    **References:**
        
        #. Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence
           of canalization on the robustness of Boolean networks. Physica D:
           Nonlinear Phenomena, 353, 39-47.
    """
    if w == 1:
        layer_structure_NCF = [n]
    else:
        assert type(w) == int or type(w) == np.int64, 'Hamming weight must be an integer'
        assert 1 <= w <= 2**n - 1, 'Hamming weight w must satisfy 1 <= w <= 2^n - 1'
        assert w % 2 == 1, 'Hamming weight must be an odd integer since all NCFs have an odd Hamming weight.'
        w_bin = dec2bin(w, n)
        current_el = w_bin[0]
        layer_structure_NCF = [1]
        for el in w_bin[1:-1]:
            if el == current_el:
                layer_structure_NCF[-1] += 1
            else:
                layer_structure_NCF.append(1)
                current_el = el
        layer_structure_NCF[-1] += 1
    return layer_structure_NCF

# ===================== #
#   Modular BoolForge   #
# ===================== #

import math
import matplotlib.pyplot as plt
import networkx as nx

def merge_state_representation(x : Union[int, tuple], y : Union[int, tuple],
    b : Union[int, tuple]) -> Union[int, tuple]:
    """
    Combines two state representations *x* and *y* into a single decimal integer.
    
    **Parameters:**
        
        - x (int | tuple[int, int]): The first decimal state representation to
          merge. Can either be an integer or a pair of integers.
        
        - y (int | tuple[int, int]): The second decimal state representation to
          merge. Can either be an integer or a pair of integers.
          
        - b (int | tuple[int, int]): The size of the state *y*, in bits.
          Must be the same dimension as *y* (if *y* is an int, *b* must be an
          int, or if *y* is a pair of ints, *b* must be a pair of ints).

    **Returns:**
    
        - int | tuple[int, int]: The combined state representation of *x* and *y*.
        
            - If *x* and *y* are both ints, the merged state is returned as an int.
            - Otherwise, returns the merged state as a pair of ints.
    """
    is_pair_x = isinstance(x, (tuple, list))
    is_pair_y = isinstance(y, (tuple, list))
    if is_pair_x:
        if is_pair_y:
            return ((x[0] << b[0]) | y[0], (x[1] << b[1]) | y[1]) 
        return (x[0], (x[1] << b) | y)
    elif is_pair_y:
        return (y[0], (x << b[1]) | y[1])
    return (x << b) | y

def get_product_of_attractors(attrs_1 : list, attrs_2 : list,
    bits : Union[int | tuple]) -> list:
    """
    Computes the product of two sets of attractors.
    
    **Parameters:**
        
        - attrs_1 (list[list[int]] | list[list[tuple[int, int]]]): The first
          set of attractors to combine.
        
        - attrs_2 (list[list[int]] | list[list[tuple[int, int]]]): The second
          set of attractors to combine.
          
        - bits (int | tuple[int, int]): The size of the states in *attrs_2*, in bits.

    **Returns:**
    
        - list[list[int]] | list[list[tuple[int, int]]]: The set of attractors
          that is the product of *attrs_1* and *attrs_2*.
    """
    attractors = []
    for attr1 in attrs_1:
        attr = []
        for attr2 in attrs_2:
            m = len(attr1)
            n = len(attr2)
            for i in range(math.lcm(*[m, n])):
                attr.append(merge_state_representation(attr1[i % m], attr2[i % n], bits))
        attractors.append(attr)
    return attractors

def compress_trajectories(trajectories : list, num_bits : [int, None] = None) -> nx.DiGraph:
    # Helper method: determine the 'canon' ordering of a periodic pattern.
    # The canon ordering is the phase such that the lowest states come first
    # without changing the relative ordering of the states.
    def _canon_cycle_(pattern):
        return min([ tuple(pattern[i:] + pattern[:i]) for i in range(len(pattern)) ])
    
    # Helper method: determine which offset a given pattern is from the canon
    # ordering. That is, how much the pattern has been phased relative to the
    # canon ordering.
    def _cycle_offset_(pattern, canon):
        pattern = list(pattern)
        canon = list(canon)
        len_pattern = len(pattern)
        for offset in range(len_pattern):
            if canon[offset:] + canon[:offset] == pattern:
                return offset
        raise ValueError("Pattern does not match canonical rotations")
    
    G = nx.DiGraph()
    next_id = 0
    cycle_nodes = {}
    prefix_merge = {}
    for states, period in trajectories:
        len_traj = len(states)
        # First look through the non-periodic component of the trajectory,
        # also referred to in this code as the 'prefix' of the trajectory
        len_pref = len_traj - period
        pref_ids = []
        for i in range(len_pref):
            # Determine if this prefix can be merged elsewhere into the graph
            future = states[i:]
            prefix_tail = future[:-period]
            pattern = future[-period:]
            canon = _canon_cycle_(pattern)
            entry_offset = _cycle_offset_(pattern, canon)
            signature = (tuple(prefix_tail), canon, entry_offset)
            # If so, merge the it and mark the node as initial
            if signature in prefix_merge:
                node_id = prefix_merge[signature]
                if i == 0:
                    G.nodes[node_id]["StIn"] = True
            # Otherwise, make a new initial node
            else:
                node_id = next_id
                prefix_merge[signature] = node_id
                G.add_node(next_id, StIn=(i == 0),
                    NLbl=(str(dec2bin(states[i], num_bits)).replace(' ', '').replace(',', '').replace('[', '').replace(']', '')
                    if num_bits is not None else str(states[i])))
                pref_ids.append(next_id)
                next_id += 1
            pref_ids.append(node_id)
        # Once prefix nodes are added, create edges
        for i in range(len(pref_ids) - 1):
            if pref_ids[i] != pref_ids[i+1]:
                G.add_edge(pref_ids[i], pref_ids[i+1])
        # Second look through the periodic component of the trajectory,
        # also referred to in this code as the 'cycle' of the trajectory
        cycle = states[-period:]
        key = _canon_cycle_(cycle)
        # If we have found a new cycle, add it to the graph
        if key not in cycle_nodes:
            ids = []
            for s in key:
                # Create nodes based off of the canon ordering to ensure
                # predictable ordering in case we need to reference
                # this cycle again for another trajectory
                G.add_node(next_id, StIn=False,
                    NLbl=(str(dec2bin(s, num_bits)).replace(' ', '').replace(',', '').replace('[', '').replace(']', '')
                    if num_bits is not None else str(s)))
                ids.append(next_id)
                next_id += 1
            # Once nodes are added, add in edges
            for a, b in zip(ids, ids[1:]):
                G.add_edge(a, b)
            G.add_edge(ids[-1], ids[0])
            cycle_nodes[key] = ids
        # For a trajectory without a prefix, mark the first state of the trajectory
        # within the cycle as an initial node
        if len_pref == 0:
            G.nodes()[cycle_nodes[key][key.index(cycle[0])]]["StIn"] = True
        # Otherwise, we need to add an edge between the prefix and cycle
        else:
            G.add_edge(pref_ids[-1], cycle_nodes[key][_cycle_offset_(cycle, key)])
    return G

def product_of_trajectories(G1 : nx.DiGraph, G2 : nx.DiGraph) -> nx.DiGraph:
    _initial_1 = []
    _initial_2 = []
    for n in G1.nodes:
        if G1.nodes[n]["StIn"]:
            _initial_1.append(n)
    for n in G2.nodes:
        if G2.nodes[n]["StIn"]:
            _initial_2.append(n)
    G = nx.DiGraph()
    starting = []
    for n1 in _initial_1:
        for n2 in _initial_2:
            starting.append((n1, n2))
            G.add_node((n1, n2), StIn=G1.nodes[n1]["StIn"] and G2.nodes[n2]["StIn"],
                NLbl=f"{G1.nodes[n1]['NLbl']}{G2.nodes[n2]['NLbl']}")
    stack = starting[:]
    visited = set(starting)
    while stack:
        u1, u2 = stack.pop()
        for v1 in G1.successors(u1):
            for v2 in G2.successors(u2):
                new_pair = (v1, v2)
                if new_pair not in G:
                    G.add_node(new_pair, StIn=False,
                        NLbl=f"{G1.nodes[v1]['NLbl']}{G2.nodes[v2]['NLbl']}")
                G.add_edge((u1, u2), new_pair)
                if new_pair not in visited:
                    visited.add(new_pair)
                    stack.append(new_pair)
    return G

def plot_trajectory(compressed_trajectory_graph : nx.DiGraph) -> None:
    pos = nx.spring_layout(compressed_trajectory_graph, seed=62)
    nx.draw_networkx_nodes(compressed_trajectory_graph, pos, node_size=400, node_color="white")
    nx.draw_networkx_edges(compressed_trajectory_graph, pos, arrows=True, arrowstyle="->", arrowsize=20)
    
    normal = {}
    boxed = {}
    labels = nx.get_node_attributes(compressed_trajectory_graph, "NLbl")
    initial = nx.get_node_attributes(compressed_trajectory_graph, "StIn")
    for n in compressed_trajectory_graph.nodes():
        if initial[n]:
            boxed.update({n:labels[n]})
        else:
            normal.update({n:labels[n]})
    nx.draw_networkx_labels(compressed_trajectory_graph, pos, labels=normal, font_size=10)
    nx.draw_networkx_labels(compressed_trajectory_graph, pos, labels=boxed, font_size=10,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1))
    
    plt.axis("off")
    plt.show()