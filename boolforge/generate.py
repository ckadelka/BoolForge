#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 09:25:40 2025
Last Edited on Thu Aug 14 2025

@author: Claus Kadelka, Benjamin Coberly
"""

##Imports

import numpy as np
import itertools
import networkx as nx
import random

try:
    from boolforge.boolean_function import BooleanFunction as BF
    from boolforge.boolean_network import BooleanNetwork as BN
except ModuleNotFoundError:
    from boolean_function import BooleanFunction as BF
    from boolean_network import BooleanNetwork as BN


left_side_of_truth_tables = {}


def get_left_side_of_truth_table(n):
    if n in left_side_of_truth_tables:
        left_side_of_truth_table = left_side_of_truth_tables[n]
    else:
        left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=n)))
        left_side_of_truth_tables[n] = left_side_of_truth_table
    return left_side_of_truth_table

def random_function(n, bias=0.5):
    """
    Generate a random Boolean function in n variables.

    The Boolean function is represented as a truth table (an array of length 2^n) in which each entry is 0 or 1.
    Each entry is set to 1 with probability `bias`.

    Parameters:
        - n (int): Number of variables.
        - bias (float, optional): Probability that a given entry is 1 (default is 0.5).

    Returns:
        - BooleanFunction: Boolean function object.
    """
    return BF(np.array(np.random.random(2**n) < bias, dtype=int))


def random_linear_function(n):
    """
    Generate a random linear Boolean function in n variables.

    A random linear Boolean function is constructed by randomly choosing whether to include each variable or its negation in a linear sum.
    The resulting expression is then reduced modulo 2.

    Parameters:
        - n (int): Number of variables.

    Returns:
        - BooleanFunction: Boolean function object.
    """
    val = int(random.random()>0.5)
    f = [0] * 2**n
    for i in range(1 << n):
        if i.bit_count() % 2 == val:
            f[i] = 1
    return BF(f)


def random_non_degenerated_function(n, bias=0.5):
    """
    Generate a random non-degenerated Boolean function in n variables.

    A non-degenerated Boolean function is one in which every variable is essential (i.e. the output depends on every input).
    The function is repeatedly generated with the specified bias until a non-degenerated function is found.

    Parameters:
        - n (int): Number of variables.
        - bias (float, optional): Bias of the Boolean function (probability of a 1; default is 0.5).

    Returns:
        - BooleanFunction: Boolean function object.
    
    References:
        Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
        of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    while True:  # works well because most Boolean functions are non-degenerated
        f = random_function(n, bias)
        if not f.is_degenerated():
            return f


def random_degenerated_function(n, bias=0.5):
    """
    Generate a random degenerated Boolean function in n variables.

    A degenerated Boolean function is one in which at least one variable is non‐essential (its value never affects the output).
    The function is generated repeatedly until a degenerated function is found.

    Parameters:
        - n (int): Number of variables.
        - bias (float, optional): Bias of the Boolean function (default is 0.5, i.e., unbiased).

    Returns:
        - BooleanFunction: Boolean function object that is degenerated in the first input (and possibly others).
    
    References:
        Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
        of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    
    f_original = random_function(n-1, bias)
    index_non_essential_variable = int(random.random()*n)
    f = np.zeros(2**n, dtype=int)
    indices = (np.arange(2**n)//(2**index_non_essential_variable))%2==1
    f[indices] = f_original.f
    f[~indices] = f_original.f
    return BF(f)


def random_non_canalizing_function(n, bias=0.5):
    """
    Generate a random non-canalizing Boolean function in n (>1) variables.

    A Boolean function is canalizing if there exists at least one variable whose fixed value forces the output.
    This function returns one that is not canalizing.

    Parameters:
        - n (int): Number of variables (n > 1).
        - bias (float, optional): Bias of the Boolean function (default is 0.5, i.e., unbiased).

    Returns:
        - BooleanFunction: Boolean function object.
    
    References:
        Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
        of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    assert type(n)==int and n > 1, "n must be an integer > 1"
    while True:  # works because most functions are non-canalizing
        f = BF(np.array(np.random.random(2**n) < bias, dtype=int))
        if not f.is_canalizing():
            return f


def random_non_canalizing_non_degenerated_function(n, bias=0.5):
    """
    Generate a random Boolean function in n (>1) variables that is both non-canalizing and non-degenerated.

    Such a function has every variable essential and is not canalizing.

    Parameters:
        - n (int): Number of variables (n > 1).
        - bias (float, optional): Bias of the Boolean function (default is 0.5, i.e., unbiased).

    Returns:
        - BooleanFunction: Boolean function object.
    
    References:
        Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
        of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    assert type(n)==int and n > 1, "n must be an integer > 1"
    while True:  # works because most functions are non-canalizing and non-degenerated
        f = BF(np.array(np.random.random(2**n) < bias, dtype=int))
        if not f.is_canalizing() and not f.is_degenerated():
            return f


def random_k_canalizing_function(n, k, EXACT_DEPTH=False):
    """
    Generate a random k-canalizing Boolean function in n variables.

    A Boolean function is k-canalizing if it has at least k conditionally canalizing variables.
    If EXACT_DEPTH is True, the function will have exactly k canalizing variables; otherwise, its canalizing depth may exceed k.

    Parameters:
        - n (int): Total number of variables.
        - k (int): Number of canalizing variables. Set k==n to generate a random nested canalizing function.
        - EXACT_DEPTH (bool, optional): If True, enforce that the canalizing depth is exactly k (default is False).

    Returns:
        - BooleanFunction: Boolean function object.
    
    References:
        [1] He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth. 
            Physica D: Nonlinear Phenomena, 314, 1-8.
        [2] Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions: 
            Algorithms and applications. Automatica, 146, 110630.
    """
    assert n - k != 1 or EXACT_DEPTH == False,'There are no functions of exact canalizing depth n-1.\nEither set EXACT_DEPTH=False or ensure k != n-1'
    assert 0 <= k and k <= n,'Error:\nEnsure 0 <= k <= n.'

    left_side_of_truth_table = get_left_side_of_truth_table(n)
    num_values = 2**n
    aas = np.random.randint(2, size=k)  # canalizing inputs
    bbs = np.random.randint(2, size=k)  # canalized outputs

    # The activator_or_inhibitor parameter is currently not used.
    can_vars = np.random.choice(n, k, replace=False)
    f = np.zeros(num_values, dtype=int)
    if k < n:
        if EXACT_DEPTH:
            core_polynomial = random_non_canalizing_non_degenerated_function(n - k).f
        else:
            core_polynomial = random_non_degenerated_function(n - k).f
    else:
        core_polynomial = [1 - bbs[-1]]
    counter_non_canalized_positions = 0
    for i in range(num_values):
        for j in range(k):
            if left_side_of_truth_table[i][can_vars[j]] == aas[j]:
                f[i] = bbs[j]
                break
        else:
            f[i] = core_polynomial[counter_non_canalized_positions]
            counter_non_canalized_positions += 1
    return BF(f)


def random_k_canalizing_function_with_specific_layer_structure(n, layer_structure, EXACT_DEPTH=False):
    """
    Generate a random Boolean function in n variables with a specified canalizing layer structure.

    The layer structure is given as a list [k_1, ..., k_r], where each k_i indicates the number of canalizing variables 
    in that layer. If the function is fully canalizing (i.e. sum(layer_structure) == n and n > 1), the last layer must have at least 2 variables.

    Parameters:
        - n (int): Total number of variables.
        - layer_structure (list): List [k_1, ..., k_r] describing the canalizing layer structure. Each k_i ≥ 1, and if sum(layer_structure) == n and n > 1, then layer_structure[-1] ≥ 2. Set sum(layer_structure)==n to generate a random nested canalizing function.
        - EXACT_DEPTH (bool, optional): If True, the canalizing depth is exactly sum(layer_structure) (default is False).

    Returns:
        - BooleanFunction: Boolean function object.
    
    References:
        [1] He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
            Physica D: Nonlinear Phenomena, 314, 1-8.
        [2] Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
            of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    k = sum(layer_structure)  # canalizing depth
    if k == 0:
        layer_structure = [0]
        
    assert n - k != 1 or EXACT_DEPTH == False,'Error:\nThere are no functions of exact canalizing depth n-1.\nEither set EXACT_DEPTH=False or ensure k=sum(layer_structure)!=n-1.'
    assert 0 <= k and k <= n,'Error:\nEnsure 0 <= k = sum(layer_structure) <= n.'
    assert k < n or layer_structure[-1] > 1 or n == 1,'Error:\nThe last layer of an NCF (i.e., an n-canalizing function) has to have size >= 2 whenever n > 1.\nIf k=sum(layer_structure)=n, ensure that layer_structure[-1]>=2.'
    assert min(layer_structure) >= 1,'Error:\nEach layer must have at least one variable (each element of layer_structure must be >= 1).'
    
    left_side_of_truth_table = get_left_side_of_truth_table(n)

    num_values = 2**n
    aas = np.random.randint(2, size=k)  # canalizing inputs
    b0 = np.random.randint(2)
    bbs = [b0] * layer_structure[0]  # canalized outputs for first layer
    for i in range(1, len(layer_structure)):
        if i % 2 == 0:
            bbs.extend([b0] * layer_structure[i])
        else:
            bbs.extend([1 - b0] * layer_structure[i])
    can_vars = np.random.choice(n, k, replace=False)
    f = np.zeros(num_values, dtype=int)
    if k < n:
        if EXACT_DEPTH:
            core_polynomial = random_non_canalizing_non_degenerated_function(n - k).f
        else:
            core_polynomial = random_non_degenerated_function(n - k).f
    else:
        core_polynomial = [1 - bbs[-1]]
    counter_non_canalized_positions = 0
    for i in range(num_values):
        for j in range(k):
            if left_side_of_truth_table[i][can_vars[j]] == aas[j]:
                f[i] = bbs[j]
                break
        else:
            f[i] = core_polynomial[counter_non_canalized_positions]
            counter_non_canalized_positions += 1
    return BF(f)


def random_nested_canalizing_function(n,layer_structure=None):
    '''
    Generate a random nested canalizing Boolean function in n variables 
    with a specified canalizing layer structure (if provided).

    The layer structure is given as a list [k_1, ..., k_r], where each k_i indicates the number of canalizing variables 
    in that layer. If the function is fully canalizing (i.e. sum(layer_structure) == n and n > 1), the last layer must have at least 2 variables.

    Parameters:
        - n (int): Total number of variables.
        - layer_structure (list,optional): List [k_1, ..., k_r] describing the canalizing layer structure. Each k_i ≥ 1, and if sum(layer_structure) == n and n > 1, then layer_structure[-1] ≥ 2. Set sum(layer_structure)==n to generate a random nested canalizing function.

    Returns:
        - BooleanFunction: Boolean function object.
    
    References:
        [1] He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
            Physica D: Nonlinear Phenomena, 314, 1-8.
        [2] Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence of canalization on the robustness 
            of Boolean networks. Physica D: Nonlinear Phenomena, 353, 39-47.
    '''    
    if layer_structure is None:
        return random_k_canalizing_function(n,n,EXACT_DEPTH=False)
    else:
        assert sum(layer_structure) == n,'Error:\nEnsure sum(layer_structure) == n.'
        assert layer_structure[-1] > 1 or n == 1,'Error:\nThe last layer of an NCF has to have size >= 2 whenever n > 1.\nEnsure that layer_structure[-1]>=2.'
        return random_k_canalizing_function_with_specific_layer_structure(n,layer_structure,EXACT_DEPTH=False)

random_NCF = random_nested_canalizing_function

def random_adjacency_matrix(N, indegrees, NO_SELF_REGULATION=True, STRONGLY_CONNECTED=False):
    """
    Generate a random adjacency matrix for a network of N nodes.

    Each node i is assigned indegrees[i] outgoing edges (regulators) chosen at random.
    Optionally, self-regulation (an edge from a node to itself) can be disallowed,
    and the generated network can be forced to be strongly connected.

    Parameters:
        - N (int): Number of nodes.
        - indegrees (list or array-like): List of length N specifying the number of outgoing edges for each node.
        - NO_SELF_REGULATION (bool, optional): If True, self-regulation is disallowed (default is True).
        - STRONGLY_CONNECTED (bool, optional): If True, the generated network is forced to be strongly connected (default is False).

    Returns:
        - tuple: (matrix, indices) where:
            - matrix (np.array): An N x N adjacency matrix with entries 0 or 1.
            - indices (list): A list of length N, where each element is an array of selected target indices for the corresponding node.
    """
    matrix = np.zeros((N, N), dtype=int)
    indices = []
    for i in range(N):
        if NO_SELF_REGULATION:
            indexes = np.random.choice(np.append(np.arange(i), np.arange(i+1, N)), indegrees[i], replace=False)
        else:
            indexes = np.random.choice(np.arange(N), indegrees[i], replace=False)
        indexes = np.sort(indexes)
        indices.append(indexes)
        for index in indexes:
            matrix[i][index] = 1
    if STRONGLY_CONNECTED:
        G = nx.from_numpy_array(matrix, parallel_edges=False, create_using=nx.MultiDiGraph())
        if not nx.is_strongly_connected(G):
            return random_adjacency_matrix(N, indegrees, NO_SELF_REGULATION, STRONGLY_CONNECTED)
    return (matrix, indices)

    
def random_edge_list(N, indegrees, NO_SELF_REGULATION, AT_LEAST_ONE_REGULATOR_PER_NODE=False):
    """
    Generate a random edge list for a network of N nodes with optional constraints.

    Each node i receives indegrees[i] incoming edges chosen at random.
    Optionally, the function can ensure that every node regulates at least one other node.

    Parameters:
        - N (int): Number of nodes.
        - indegrees (list or array-like): List of length N specifying the number of regulators for each node.
        - NO_SELF_REGULATION (bool): If True, disallow self-regulation.
        - AT_LEAST_ONE_REGULATOR_PER_NODE (bool, optional): If True, ensure that each node has at least one outgoing edge (default is False).

    Returns:
        - list: A list of tuples (source, target) representing the edges.
    """
    if AT_LEAST_ONE_REGULATOR_PER_NODE == False:
        edge_list = []
        for i in range(N):
            if NO_SELF_REGULATION:
                indices = np.random.choice(np.append(np.arange(i), np.arange(i+1, N)), indegrees[i], replace=False)
            else:
                indices = np.random.choice(np.arange(N), indegrees[i], replace=False)
            edge_list.extend(list(zip(indices, i * np.ones(indegrees[i], dtype=int))))
    else:
        edge_list = []
        outdegree = np.zeros(N, dtype=int)
        sum_indegrees = sum(indegrees)  # total number of regulations
        for i in range(N):
            if NO_SELF_REGULATION:
                indices = np.random.choice(np.append(np.arange(i), np.arange(i+1, N)), indegrees[i], replace=False)
            else:
                indices = np.random.choice(np.arange(N), indegrees[i], replace=False)
            outdegree[indices] += 1
            edge_list.extend(list(zip(indices, i * np.ones(indegrees[i], dtype=int))))
        while min(outdegree) == 0:
            index_sink = np.where(outdegree == 0)[0][0]
            index_edge = int(random.random() * sum_indegrees)
            if NO_SELF_REGULATION:
                while edge_list[index_edge][1] == index_sink:
                    index_edge = int(random.random() * sum_indegrees)
            outdegree[index_sink] += 1
            outdegree[edge_list[index_edge][0]] -= 1
            edge_list[index_edge] = (index_sink, edge_list[index_edge][1])
    return edge_list


def random_rules(indegrees, k=0, EXACT_DEPTH=False, layer_structure=None, 
                 LINEAR=False, ALLOW_DEGENERATED_FUNCTIONS=True,
                 bias=0.5, absolute_bias = 0, USE_ABSOLUTE_BIAS=True):
    N = len(indegrees)
    F = []
    for i in range(N):
        if LINEAR:
            F.append(random_linear_function(indegrees[i]))
        elif type(k) in [int, np.int_] and k > 0 and layer_structure is None:
                F.append(random_k_canalizing_function(indegrees[i], min(k, indegrees[i]), EXACT_DEPTH=EXACT_DEPTH))
        elif type(k) in [list, np.ndarray] and layer_structure is None:
                F.append(random_k_canalizing_function(indegrees[i], min(k[i], indegrees[i]), EXACT_DEPTH=EXACT_DEPTH))
        elif layer_structure is not None:
            if np.all([type(el) in [int, np.int_] for el in layer_structure]):
                F.append(random_k_canalizing_function_with_specific_layer_structure(indegrees[i], layer_structure, EXACT_DEPTH=EXACT_DEPTH))
            else:
                F.append(random_k_canalizing_function_with_specific_layer_structure(indegrees[i], layer_structure[i], EXACT_DEPTH=EXACT_DEPTH))
        else:
            if USE_ABSOLUTE_BIAS:             
                bias_of_function = random.choice([0.5*(1-absolute_bias),0.5*(1+absolute_bias)])
            else:
                bias_of_function = bias
            if ALLOW_DEGENERATED_FUNCTIONS:
                if EXACT_DEPTH is True:
                    F.append(random_non_canalizing_non_degenerated_function(indegrees[i], bias_of_function))
                else:
                    F.append(random_non_degenerated_function(indegrees[i], bias_of_function))
            else:
                if EXACT_DEPTH is True:
                    F.append(random_non_canalizing_function(indegrees[i], bias_of_function))
                else:
                    F.append(random_function(indegrees[i], bias_of_function))
    return F


def random_degrees(N,n,indegree_distribution='constant',NO_SELF_REGULATION=True):
    if type(n) in [list, np.array]:
        assert (np.all([type(el) in [int, np.int_] for el in n]) and len(n) == N and min(n) >= 1 and max(n) <= N), 'A vector n was submitted.\nEnsure that n is an N-dimensional vector where each element is an integer between 1 and N representing the upper bound of a uniform degree distribution (lower bound == 1).'
        indegrees = np.array(n,dtype=int)
    elif indegree_distribution.lower() in ['constant', 'dirac', 'delta']:
        assert (type(n) in [int, np.int_] and n >= 1 and n <= N), 'n must be a single integer (or N-dimensional vector of integers) between 1 and N when using a constant degree distribution.'
        indegrees = np.ones(N, dtype=int) * n
    elif indegree_distribution.lower() == 'uniform':
        assert (type(n) in [int, np.int_] and n >= 1 and n <= N - int(NO_SELF_REGULATION)), 'n must be a single integer (or N-dimensional vector of integers) between 1 and ' + ('N-1' if NO_SELF_REGULATION else 'N')+' representing the upper bound of a uniform degree distribution (lower bound == 1).'
        indegrees = 1 + np.random.randint(n - 1, size=N)
    elif indegree_distribution.lower() == 'poisson':
        assert (type(n) in [int, np.int_, float, np.float64] and n>= 1 and n<=N), 'n must be a single number (or N-dimensional vector) > 0 representing the Poisson parameter.'
        indegrees = np.maximum(np.minimum(np.random.poisson(lam=n, size=N),N - int(NO_SELF_REGULATION)), 1)
    else:
        raise AssertionError('None of the predefined in-degree distributions were chosen.\nTo use a user-defined in-degree vector, submit an N-dimensional vector as argument for n; each element of n must an integer between 1 and N.')
    return indegrees

def random_network(N=None, n=None, k=0, EXACT_DEPTH=False, layer_structure=None, 
                   LINEAR=False, NO_SELF_REGULATION=True, ALLOW_DEGENERATED_FUNCTIONS=True,
                   STRONGLY_CONNECTED=False, indegree_distribution='constant', 
                   I=None, 
                   bias=0.5, absolute_bias = 0, USE_ABSOLUTE_BIAS=True, 
                   n_attempts_to_generate_strongly_connected_network = 1000):
    """
    Generate a random Boolean network (BN).

    This function creates a Boolean network of N nodes by first generating a wiring diagram
    (a set of regulatory interactions) according to a specified in-degree distribution and then assigning
    Boolean functions to each node. The functions can be canalizing with prescribed depth and/or specific layer structure,
    lineear, or random functions with a specified bias.

    Parameters:
        - N (int): Number of nodes in the network.
        - n (int, list, or np.array; float allowed if indegree_distribution=='poisson'):  Determines the in-degree of each node. If an integer, each node has the same number of regulators; if a vector, each element gives the number of regulators for the corresponding node.
        - k (int, list, or np.array, optional): Specifies the minimal canalizing depth for each node (exact canalizing depth if EXACT_DEPTH==True). If an integer, the same depth is used for all nodes; if a vector, each node gets its own depth. Default is 0.
        - EXACT_DEPTH (bool, optional): If True, Boolean functions are generated with exactly the specified canalizing depth; if False, the functions have at least that depth. Default is False.
        - layer_structure (optional): Specifies the canalizing layer structure for the Boolean functions. If provided, the parameter k is ignored.
        - LINEAR (bool, optional): If True, Boolean functions are generated to be linear. Default is False.
        - NO_SELF_REGULATION (bool, optional): If True, self-regulation (self-loops) is disallowed. Default is True.
        - ALLOW_DEGENERATED_FUNCTIONS(bool, optional): If True (default) and k==0 and layer_structure is None, degenerated functions may be created as in NK-Kauffman networks.
        - STRONGLY_CONNECTED (bool, optional): If True, ensures that the generated network is strongly connected. Default is False.
        - indegree_distribution (str, optional): In-degree distribution to use. Options include 'constant' (or 'dirac'/'delta'), 'uniform', or 'poisson'. Default is 'constant'.
        - I (list or numpy array, optional): A list of N lists representing the regulators (or inputs) for each node.
        - bias (float, optional): Bias of generated Boolean functions (probability of output 1). Default is 0.5. Ignored unless k==0 and LINEAR==False and layer_structure is None.
        - absolute_bias (float, optional): Absolute bias of generated Boolean functions. Default is 0. Ignored unless k==0 and LINEAR==False and layer_structure is None and USE_ABSOLUTE_BIAS==True.
        - USE_ABSOLUTE_BIAS (bool, optional): Determines if absolute bias or regular bias is used in the generation of functions. Default is True (i.e., absolute bias). Ignored unless k==0 and LINEAR==False and layer_structure is None.
        - n_attempts_to_generate_strongly_connected_network (integer, optional): Number of attempts to generate a strongly connected wiring diagram before raising an error and quitting.
    
    Returns:
        - BooleanNetwork: Boolean network object.
        
    Examples:
        >>> random_network(N,n) #creates a random NK-Kauffman network. The constant degree is specified by n.
        
        >>> random_network(N,n,k) #all functions have degree n and at least canalizing depth k.

        >>> random_network(N,n,k=n) #all functions are n-input NCFs of arbitrary layer structure.

        >>> random_network(N,n,k,EXACT_DEPTH=True) #all functions have degree n and exact canalizing depth k.

        >>> random_network(N,n,layer_structure=[n]) #all functions have degree n and are NCFs with exactly one layer.

        >>> random_network(N,n, LINEAR=True) #creates a random network with linear update rules

        >>> random_network(N,n,NO_SELF_REGULATION=False) #The underlying wiring diagram may contain self-loops.

        >>> random_network(N,n,ALLOW_DEGENERATED_FUNCTIONS=False) #creates a random NK-Kauffman network in which all inputs are essential.
        
        >>> random_network(N,n,ALLOW_DEGENERATED_FUNCTIONS=False,STRONGLY_CONNECTED=True) #The underlying wiring diagram is truly strongly connected.

        >>> random_network(N,n,indegree_distribution='Poisson') #Creates a random network where the degree of each node is drawn from a truncated Poisson distribution with parameter n, min = 1, max = N - int(NO_SELF_REGULATION).

        >>> random_network(I = I) #Creates a random network with defined wiring diagram (specified by I, see BooleanNetwork.I), only randomizes the update rules.

    """

    if I is None and N is not None and n is not None:
        # Generate the in-degree based on the specified in-degree distribution and the in-degree parameter n. If n is a vector of length N, this is used as in-degree.
        indegrees = random_degrees(N,n,indegree_distribution=indegree_distribution,NO_SELF_REGULATION=NO_SELF_REGULATION)

        # Generate the wiring diagram.
        counter = 0
        while True:  # Keep generating until we have a strongly connected graph
            edges_wiring_diagram = random_edge_list(N, indegrees, NO_SELF_REGULATION)
            if STRONGLY_CONNECTED:#may take a long time if n is small and N is large
                G = nx.from_edgelist(edges_wiring_diagram, create_using=nx.MultiDiGraph())
                if not nx.is_strongly_connected(G):
                    counter+=1
                    if counter>n_attempts_to_generate_strongly_connected_network:
                        raise RuntimeError('Made '+str(n_attempts_to_generate_strongly_connected_network)+' unsuccessful attempts to generate a strongly connected wiring diagram of '+str(N)+' nodes and degrees '+str(indegrees)+'.\nYou may increase the number of attempts by modulating the parameter n_attempts_to_generate_strongly_connected_network.')
                    continue
            break
        I = [[] for _ in range(N)]
        for edge in edges_wiring_diagram:
            I[edge[1]].append(edge[0])
        for i in range(N):
            I[i] = np.sort(I[i])
    elif I is not None:
        #TODO: add assertions for I
        assert isinstance(I, (list, np.ndarray)), "I must be an array"
        N = len(I)
        indegrees = list(map(len,I))
    else:
        raise AssertionError('At a minimum, the wiring diagram I must be provided or the network size N and degree parameter n.')
       
    # Process the canalizing depth / canalizing layer structure
    if layer_structure is None:
        if type(k) in [int, np.int_]:
            assert k >= 0 and k<=N,'The canalizing depth k must be an integer between 0 and N.'
            max_k = k
        elif type(k) in [list, np.array]:
            max_k = max(k)
            assert (len(k) == N and np.all([type(el) in [int, np.int_] for el in k]) and min(k) >= 0 and max_k <= N),'A vector k was submitted.\nTo use a user-defined vector k, ensure that k is an N-dimensional vector where each element is an integer between 0 and N.'
        else:
            raise AssertionError('Wrong input format for k.\nk must be a single integer (or N-dimensional vector of integers) between 0 and N, specifying the minimal canalizing depth or exact canalizing depth (if EXACT_DEPTH==True).')            
    else:  # layer_structure provided
        if np.all([type(el) in [int, np.int_] for el in layer_structure]):
            max_k = sum(layer_structure)
            assert np.all([type(el) in [int, np.int_] for el in layer_structure]) and np.all([el >= 1 for el in layer_structure]) and max_k <= N, 'The layer structure must be a vector of positive integers with 0 <= k = sum(layer_structure) <= N.'
        elif np.all([type(el) in [list, np.array] for el in layer_structure]):
            max_k = max([sum(el) for el in layer_structure])
            assert len(layer_structure) == N and type(layer_structure[0][0]) in [int, np.int_] and min([min(el) for el in layer_structure]) >= 0 and max_k <= N, 'Ensure that layer_structure is an N-dimensional vector where each element represents a layer structure and is a vector of positive integers with 1 <= k = sum(layer_structure[i]) <= N.'
        else:
            raise AssertionError('Wrong input format for layer_structure.\nlayer_structure must be a single vector (or N-dimensional vector of layer structures) where the sum of each element is between 0 and N.')

    F = random_rules(indegrees, k=k, EXACT_DEPTH=EXACT_DEPTH, layer_structure=layer_structure, 
                     LINEAR=LINEAR, ALLOW_DEGENERATED_FUNCTIONS=ALLOW_DEGENERATED_FUNCTIONS,
                     bias=bias, absolute_bias=absolute_bias, USE_ABSOLUTE_BIAS=USE_ABSOLUTE_BIAS)
                
    return BN(F, I)

random_BN = random_network

