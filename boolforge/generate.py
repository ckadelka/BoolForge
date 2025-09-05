#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 09:25:40 2025

@author: Claus Kadelka, Benjamin Coberly
"""

##Imports
import numpy as np
import networkx as nx

from typing import Union
from typing import Optional

try:
    from boolforge.boolean_function import BooleanFunction
    from boolforge.boolean_network import BooleanNetwork
    import boolforge.utils as utils
except ModuleNotFoundError:
    from boolean_function import BooleanFunction
    from boolean_network import BooleanNetwork
    import utils


## Random function generation

def random_function(n : int, depth : int = 0, EXACT_DEPTH : bool = False,
    layer_structure : Optional[list] = None, LINEAR : bool = False,
    ALLOW_DEGENERATED_FUNCTIONS : bool = False, bias : float = 0.5,
    absolute_bias : float = 0, USE_ABSOLUTE_BIAS : bool = False,
    hamming_weight : Optional[int] = None, *, rng=None) -> BooleanFunction:
    """
    Generate a random Boolean function in n variables under flexible constraints.

    Selection logic (first match applies):
        
        - If `LINEAR`: return a random **linear** Boolean function
          (`random_linear_function`).
          
        - Else if `layer_structure is not None`: return a function with the
          specified **canalizing layer structure** using
          `random_k_canalizing_function_with_specific_layer_structure`,
          with exact canalizing depth if `EXACT_DEPTH`.
          
        - Else if `depth > 0`: return a **k-canalizing** function with k =
          min(depth, n) using `random_k_canalizing_function`, with exact
          canalizing depth if `EXACT_DEPTH`.
          
        - Else if exact `hamming_weight` is provided: sample uniformly a truth
          table with the requested number of ones, and keep resampling until
          the additional constraints implied by `ALLOW_DEGENERATED_FUNCTIONS`
          and `EXACT_DEPTH` are satisfied:

            - If `ALLOW_DEGENERATED_FUNCTIONS` and `EXACT_DEPTH`: return a
              **non-canalizing** function with exact Hamming weight.
              
            - If `ALLOW_DEGENERATED_FUNCTIONS` and not `EXACT_DEPTH`: return
              a fully random function with exact Hamming weight.
              
            - If not `ALLOW_DEGENERATED_FUNCTIONS` and `EXACT_DEPTH`: return
              a **non-canalizing & non-degenerated** function with exact
              Hamming weight.
              
            - Else: return a **non-degenerated** function with exact Hamming
              weight.
            
        - Else: 
 
            - Choose a bias:
                
                - If `USE_ABSOLUTE_BIAS`, set `bias` randomly to
                  `0.5*(1−absolute_bias)` or `0.5*(1+absolute_bias)`.
                  
                - Else, use `bias` directly.
                
            - Then:
                
                - If `ALLOW_DEGENERATED_FUNCTIONS` and `EXACT_DEPTH`: return
                  a **non-canalizing** function with that bias
                  (`random_non_canalizing_function`).
                  
                - If `ALLOW_DEGENERATED_FUNCTIONS` and not `EXACT_DEPTH`:
                  return a fully random function with that bias, as used in
                  classical NK-Kauffman models (`random_function_with_bias`).
                  
                - If not `ALLOW_DEGENERATED_FUNCTIONS` and `EXACT_DEPTH`:
                  return a **non-canalizing, non-degenerated** function
                  (`random_non_canalizing_non_degenerated_function`).
                  
                - Else (default, if only 'n' is provided): return a
                  **non-degenerated** function with that bias
                  (`random_non_degenerated_function`).     
 

    **Parameters:**
        
        - n (int): Number of variables (n >= 1 for most nontrivial generators).
        - depth (int, optional): Requested canalizing depth (used when
          `layer_structure is None` and `depth > 0`). If `EXACT_DEPTH`,
          the function has exactly this depth (clipped at n); otherwise, at
          least this depth. Default 0.
          
        - EXACT_DEPTH (bool, optional): Enforce exact canalizing depth where
          applicable. For the case `depth == 0` this implies
          **non-canalizing**. Default False.
          
        - layer_structure (list[int] | None, optional): Canalizing layer
          structure [k1, ..., kr]. If provided, it takes precedence over
          `depth`. Exact depth behavior follows `EXACT_DEPTH`. Default None.
          
        - LINEAR (bool, optional): If True, ignore other generation options
          and return a random linear function. Default False.
          
        - ALLOW_DEGENERATED_FUNCTIONS (bool, optional): If True, generators
          in the “random” branches may return functions with non-essential
          inputs. If False, those branches insist on non-degenerated functions.
          Default False.
          
        - bias (float, optional): Probability of 1s when sampling with bias
          (ignored if `USE_ABSOLUTE_BIAS` or a different branch is taken).
          Must be in [0,1]. Default 0.5.
          
        - absolute_bias (float, optional): Absolute deviation parameter in
          [0,1] used when `USE_ABSOLUTE_BIAS`. The actual bias is chosen at
          random from {0.5*(1−absolute_bias), 0.5*(1+absolute_bias)}. Default 0.
          
        - USE_ABSOLUTE_BIAS (bool, optional): If True, use `absolute_bias` to
          set the distance from 0.5; otherwise use `bias` directly. Default False.
          
        - hamming_weight (int | None, optional): If provided, enforce an
          exact number of ones in the truth table (0..2^n). Additional
          constraints apply with `EXACT_DEPTH` and degeneracy settings (see
          selection logic above). Default None.
          
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanFunction: The generated Boolean function of arity n.

    **Raises:**
        
        - AssertionError: If parameter ranges are violated, e.g.:
            
            - `0 <= bias <= 1` (when used),
            - `0 <= absolute_bias <= 1` (when used),
            - `hamming_weight` in {0, ..., 2^n} (when used),
            - If `EXACT_DEPTH` and `depth==0`, then `hamming_weight`
              must be in {2,3,...,2^n−2} (since weights 0,1,2^n−1,2^n are
              canalizing).
            
        - AssertionError (from called generators): Some subroutines require
          `n > 1` for non-canalizing generation.

    **Notes:**
    
        - Extremely biased random functions (with bias very close to 0 or 1)
          are often degenerated and highly canalizing; some functions force
          bias in [0.001,0.999] to avoid RunTimeErrors.


    **Examples:**
        
        >>> # Unbiased, non-degenerated random function
        >>> f = random_function(n=3)
        
        >>> # Function with minimal canalizing depth 2
        >>> f = random_function(n=5, depth=2)

        >>> # Function with exact canalizing depth 2
        >>> f = random_function(n=5, depth=2, EXACT_DEPTH=True)

        >>> # With a specific layer structure (takes precedence over `depth`)
        >>> f = random_function(n=6, layer_structure=[2, 1], EXACT_DEPTH=False)

        >>> # Linear function
        >>> f = random_function(n=4, LINEAR=True)

        >>> # Fixed Hamming weight under non-canalizing + non-degenerated constraints
        >>> f = random_function(n=5, hamming_weight=10, EXACT_DEPTH=True,
        ...                     ALLOW_DEGENERATED_FUNCTIONS=False)
    """
    rng = utils._coerce_rng(rng)
    
    if LINEAR:
        return random_linear_function(n,rng=rng)
    elif layer_structure is not None:
        return random_k_canalizing_function_with_specific_layer_structure(n, layer_structure, EXACT_DEPTH=EXACT_DEPTH, ALLOW_DEGENERATED_FUNCTIONS=ALLOW_DEGENERATED_FUNCTIONS,rng=rng)
    elif depth>0:
        return random_k_canalizing_function(n, min(depth, n), EXACT_DEPTH=EXACT_DEPTH, ALLOW_DEGENERATED_FUNCTIONS=ALLOW_DEGENERATED_FUNCTIONS,rng=rng)
    elif hamming_weight is not None:
        assert isinstance(hamming_weight, (int, np.integer)) and 0<=hamming_weight<=2**n, "Hamming weight must be an integer in {0,1,...,2^n}"
        assert 1<hamming_weight<2**n-1 or not EXACT_DEPTH,"If EXACT_DEPTH=True and 'depth=0', Hamming_weight must be in 2,3,...,2^n-2. All functions with Hamming weight 0,1,2^n-1,2^n are canalizing"
        f=random_function_with_exact_hamming_weight(n, hamming_weight,rng=rng)
        while True:
            if ALLOW_DEGENERATED_FUNCTIONS and EXACT_DEPTH:
                if not f.is_canalizing():
                    return f
            elif ALLOW_DEGENERATED_FUNCTIONS:
                return f
            elif EXACT_DEPTH:
                if not f.is_canalizing() and not f.is_degenerated():
                    return f
            else:
                if not f.is_degenerated():
                    return f
            f=random_function_with_exact_hamming_weight(n, hamming_weight,rng=rng)
    else:
        if USE_ABSOLUTE_BIAS:
            assert 0<=absolute_bias<=1,"absolute_bias must be in [0,1]. Absolute bias determines the choice of `bias`, which is set randomly to `0.5*(1−absolute_bias)` or `0.5*(1+absolute_bias)`."
            bias_of_function = rng.choice([0.5*(1-absolute_bias),0.5*(1+absolute_bias)])
        else:
            assert 0<=bias<=1,"bias must be in [0,1]. It describes the probability of a 1 in the randomly generated function."            
            bias_of_function = bias
        if ALLOW_DEGENERATED_FUNCTIONS:
            if EXACT_DEPTH is True:
                return random_non_canalizing_function(n, bias_of_function,rng=rng)
            else: #completely random function
                return random_function_with_bias(n, bias_of_function,rng=rng)
        else:
            if EXACT_DEPTH is True:
                return random_non_canalizing_non_degenerated_function(n, bias_of_function,rng=rng)
            else: #generated by default
                return random_non_degenerated_function(n, bias_of_function,rng=rng)

def random_function_with_bias(n : int, bias : float = 0.5,
    *, rng=None) -> BooleanFunction:
    """
    Generate a random Boolean function in n variables with a specified bias.

    The Boolean function is represented as a truth table (an array of length
    2^n) in which each entry is 0 or 1. Each entry is set to 1 with
    probability `bias`.

    **Parameters:**
        
        - n (int): Number of variables.
        - bias (float, optional): Probability that a given entry is 1
          (default is 0.5).
        
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanFunction: Boolean function object.
    """
    rng = utils._coerce_rng(rng)
    return BooleanFunction(np.array(rng.random(2**n) < bias, dtype=int))


def random_function_with_exact_hamming_weight(n : int, hamming_weight : int,
    *, rng=None) -> BooleanFunction:
    """
    Generate a random Boolean function in n variables with exact Hamming
    weight (number of ones).

    The Boolean function is represented as a truth table (an array of length
    2^n) in which each entry is 0 or 1. Exactly 'hamming_weight' entries are
    set to 1.

    **Parameters:**
        
        - n (int): Number of variables.
        - hamming_weight (int): Probability that a given entry is 1
          (default is 0.5).
          
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanFunction: Boolean function object.
    """
    rng = utils._coerce_rng(rng)
    assert isinstance(hamming_weight, (int, np.integer)) and 0<=hamming_weight<=2**n,"Hamming weight must be an integer between 0 and 2^n."
    oneIndices = rng.choice(2**n,hamming_weight,replace=False)
    f = np.zeros(2**n,dtype=int)
    f[oneIndices] = 1    
    return BooleanFunction(f)


def random_linear_function(n : int, *, rng=None) -> BooleanFunction:
    """
    Generate a random linear Boolean function in n variables.

    A random linear Boolean function is constructed by randomly choosing
    whether to include each variable or its negation in a linear sum. The
    resulting expression is then reduced modulo 2.

    **Parameters:**
        
        - n (int): Number of variables.
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanFunction: Boolean function object.
    """
    rng = utils._coerce_rng(rng)
    assert isinstance(n, (int, np.integer)) and n>0,"n must be a positive integer"
    val = rng.integers(2)
    f = [0] * 2**n
    for i in range(1 << n):
        if i.bit_count() % 2 == val:
            f[i] = 1
    return BooleanFunction(f)


def random_non_degenerated_function(n : int, bias : float = 0.5,
    *, rng=None) -> BooleanFunction:
    """
    Generate a random non-degenerated Boolean function in n variables.

    A non-degenerated Boolean function is one in which every variable is
    essential (i.e. the output depends on every input). The function is
    repeatedly generated with the specified bias until a non-degenerated
    function is found.

    **Parameters:**
        
        - n (int): Number of variables.
        - bias (float, optional): Bias of the Boolean function (probability
          of a 1; default is 0.5).
        
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanFunction: Boolean function object.
    
    **References:**
        
        #. Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence
           of canalization on the robustness of Boolean networks. Physica D:
           Nonlinear Phenomena, 353, 39-47.
    """
    rng = utils._coerce_rng(rng)
    assert isinstance(n, (int, np.integer)) and n>0,"n must be a positive integer"
    assert isinstance(bias, (float, np.floating)) and 0.001<bias<0.999,"almost all extremely biased Boolean functions are degenerated. Choose a more balanced value for the 'bias'."
    while True:  # works well because most Boolean functions are non-degenerated
        f = random_function_with_bias(n, bias, rng=rng)
        if not f.is_degenerated():
            return f


def random_degenerated_function(n : int, bias : float = 0.5,
    *, rng=None) -> BooleanFunction:
    """
    Generate a random degenerated Boolean function in n variables.

    A degenerated Boolean function is one in which at least one variable is
    non‐essential (its value never affects the output). The function is
    generated repeatedly until a degenerated function is found.

    **Parameters:**
        
        - n (int): Number of variables.
        - bias (float, optional): Bias of the Boolean function (default is
          0.5, i.e., unbiased).
        
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanFunction: Boolean function object that is degenerated in
          the first input (and possibly others).
    
    **References:**
        
        #. Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence
           of canalization on the robustness of Boolean networks. Physica D:
           Nonlinear Phenomena, 353, 39-47.
    """
    rng = utils._coerce_rng(rng)
    assert isinstance(n, (int, np.integer)) and n>0,"n must be a positive integer"
    
    f_original = random_function_with_bias(n-1, bias,rng=rng)
    index_non_essential_variable = rng.integers(n)
    f = np.zeros(2**n, dtype=int)
    indices = (np.arange(2**n)//(2**index_non_essential_variable))%2==1
    f[indices] = f_original.f
    f[~indices] = f_original.f
    return BooleanFunction(f)


def random_non_canalizing_function(n : int, bias : float = 0.5,
    *, rng=None) -> BooleanFunction:
    """
    Generate a random non-canalizing Boolean function in n (>1) variables.

    A Boolean function is canalizing if there exists at least one variable
    whose fixed value forces the output. This function returns one that is
    not canalizing.

    **Parameters:**
        
        - n (int): Number of variables (n > 1).
        - bias (float, optional): Bias of the Boolean function (default is
          0.5, i.e., unbiased).
        
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanFunction: Boolean function object.
    
    **References:**
        
        #. Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence
           of canalization on the robustness of Boolean networks. Physica D:
           Nonlinear Phenomena, 353, 39-47.
    """
    rng = utils._coerce_rng(rng)
    assert isinstance(n, (int, np.integer)) and n > 1, "n must be an integer > 1"
    while True:  # works because most functions are non-canalizing
        f = random_function_with_bias(n, bias=bias, rng=rng)
        if not f.is_canalizing():
            return f


def random_non_canalizing_non_degenerated_function(n : int, bias : float = 0.5,
    *, rng=None) -> BooleanFunction:
    """
    Generate a random Boolean function in n (>1) variables that is both
    non-canalizing and non-degenerated.

    Such a function has every variable essential and is not canalizing.

    **Parameters:**
        
        - n (int): Number of variables (n > 1).
        - bias (float, optional): Bias of the Boolean function (default is
          0.5, i.e., unbiased).
        
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanFunction: Boolean function object.
    
    **References:**
        
        #. Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence
           of canalization on the robustness of Boolean networks. Physica D:
           Nonlinear Phenomena, 353, 39-47.
    """
    rng = utils._coerce_rng(rng)
    assert isinstance(n, (int, np.integer)) and n > 1, "n must be an integer > 1"
    while True:  # works because most functions are non-canalizing and non-degenerated
        f = random_function_with_bias(n, bias=bias, rng=rng)
        if not f.is_canalizing() and not f.is_degenerated():
            return f


def random_k_canalizing_function(n : int, k : int, EXACT_DEPTH : bool = False,
    ALLOW_DEGENERATED_FUNCTIONS : bool = False, *, rng=None) -> BooleanFunction:
    """
    Generate a random k-canalizing Boolean function in n variables.

    A Boolean function is k-canalizing if it has at least k conditionally
    canalizing variables. If EXACT_DEPTH is True, the function will have
    exactly k canalizing variables; otherwise, its canalizing depth may
    exceed k.

    **Parameters:**
        
        - n (int): Number of variables.
        - k (int): Number of canalizing variables. Set 'k=n' to generate a
          random nested canalizing function.
          
        - EXACT_DEPTH (bool, optional): If True, enforce that the canalizing
          depth is exactly k (default is False).
          
        - ALLOW_DEGENERATED_FUNCTIONS(bool, optional): If True (default False)
          and k==0 and layer_structure is None, degenerated functions may be
          created as in classical NK-Kauffman networks.
          
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanFunction: Boolean function object.
    
    **References:**
        
        #. He, Q., & Macauley, M. (2016). Stratification and enumeration of
           Boolean functions by canalizing depth. Physica D: Nonlinear
           Phenomena, 314, 1-8.
           
        #. Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022).
           Revealing the canalizing structure of Boolean functions: Algorithms
           and applications. Automatica, 146, 110630.
    """
    rng = utils._coerce_rng(rng)

    assert isinstance(n, (int, np.integer)) and n>0,"n must be a positive integer"
    assert n - k != 1 or not EXACT_DEPTH,'There are no functions of exact canalizing depth n-1.\nEither set EXACT_DEPTH=False or ensure k != n-1'
    assert isinstance(k, (int, np.integer)) and 0 <= k and k <= n,'k, the canalizing depth, must satisfy 0 <= k <= n.'

    num_values = 2**n
    aas = rng.integers(2, size=k)  # canalizing inputs
    bbs = rng.integers(2, size=k)  # canalized outputs

    can_vars = rng.choice(n, k, replace=False)
    f = np.zeros(num_values, dtype=int)
    if k < n:
        core_function = random_function(n=n-k,depth=0,EXACT_DEPTH=EXACT_DEPTH,
                                        ALLOW_DEGENERATED_FUNCTIONS=ALLOW_DEGENERATED_FUNCTIONS,rng=rng)
    else:
        core_function = [1 - bbs[-1]]
    
    left_side_of_truth_table = utils.get_left_side_of_truth_table(n)
    f = np.full(2**n, -1, dtype=np.int8)
    for j in range(k):
        mask = (left_side_of_truth_table[:, can_vars[j]] == aas[j]) & (f < 0)
        f[mask] = bbs[j]
    # fill remaining with core truth table
    f[f < 0] = np.asarray(core_function, dtype=np.int8)

    return BooleanFunction(f)


def random_k_canalizing_function_with_specific_layer_structure(n : int,
    layer_structure : list, EXACT_DEPTH : bool = False,
    ALLOW_DEGENERATED_FUNCTIONS : bool = False, *, rng=None) -> BooleanFunction:
    """
    Generate a random Boolean function in n variables with a specified
    canalizing layer structure.

    The layer structure is given as a list [k_1, ..., k_r], where each
    k_i indicates the number of canalizing variables in that layer. If the
    function is fully canalizing (i.e. sum(layer_structure) == n and n > 1),
    the last layer must have at least 2 variables.

    **Parameters:**
        
        - n (int): Total number of variables.
        - layer_structure (list[int]): List [k_1, ..., k_r] describing the
          canalizing layer structure. Each k_i ≥ 1, and if
          sum(layer_structure) == n and n > 1, then layer_structure[-1] ≥ 2.
          Set sum(layer_structure)==n to generate a random nested canalizing
          function.
          
        - EXACT_DEPTH (bool, optional): If True, the canalizing depth is
          exactly sum(layer_structure) (default is False).
          
        - ALLOW_DEGENERATED_FUNCTIONS(bool, optional): If True (default False),
          the core function may be degenerated, as in NK-Kauffman networks.
          
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanFunction: Boolean function object.
    
    **References:**
        
        #. He, Q., & Macauley, M. (2016). Stratification and enumeration of
           Boolean functions by canalizing depth. Physica D: Nonlinear
           Phenomena, 314, 1-8.
           
        #. Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence
           of canalization on the robustness of Boolean networks. Physica D:
           Nonlinear Phenomena, 353, 39-47.
    """
    rng = utils._coerce_rng(rng)
    depth = sum(layer_structure)  # canalizing depth
    if depth == 0:
        layer_structure = [0]
        
    assert isinstance(n, (int, np.integer)) and n > 0, "n must be an integer > 0"        
    assert n - depth != 1 or not EXACT_DEPTH,'There are no functions of exact canalizing depth n-1.\nEither set EXACT_DEPTH=False or ensure depth=sum(layer_structure)!=n-1.'
    assert 0 <= depth and depth <= n,'Ensure 0 <= depth = sum(layer_structure) <= n.'
    assert depth < n or layer_structure[-1] > 1 or n == 1,'The last layer of an NCF (i.e., an n-canalizing function) has to have size >= 2 whenever n > 1.\nIf depth=sum(layer_structure)=n, ensure that layer_structure[-1]>=2.'
    assert min(layer_structure) >= 1,'Each layer must have at least one variable (each element of layer_structure must be >= 1).'
    

    size_state_space = 2**n
    aas = rng.integers(2, size=depth)  # canalizing inputs
    b0 = rng.integers(2)
    bbs = [b0] * layer_structure[0]  # canalized outputs for first layer
    for i in range(1, len(layer_structure)):
        if i % 2 == 0:
            bbs.extend([b0] * layer_structure[i])
        else:
            bbs.extend([1 - b0] * layer_structure[i])
    can_vars = rng.choice(n, depth, replace=False)
    f = np.zeros(size_state_space, dtype=int)
    if depth < n:
        core_function = random_function(n=n-depth,depth=0,EXACT_DEPTH=EXACT_DEPTH,ALLOW_DEGENERATED_FUNCTIONS=ALLOW_DEGENERATED_FUNCTIONS,rng=rng)
    else:
        core_function = [1 - bbs[-1]]
    
    left_side_of_truth_table = utils.get_left_side_of_truth_table(n)
    f = np.full(2**n, -1, dtype=np.int8)
    for j in range(depth):
        mask = (left_side_of_truth_table[:, can_vars[j]] == aas[j]) & (f < 0)
        f[mask] = bbs[j]
    # fill remaining with core truth table
    f[f < 0] = np.asarray(core_function, dtype=np.int8)
            
    return BooleanFunction(f)



def random_nested_canalizing_function(n : int,
    layer_structure : Optional[list] = None, *, rng=None) -> BooleanFunction:
    """
    Generate a random nested canalizing Boolean function in n variables 
    with a specified canalizing layer structure (if provided).

    The layer structure is given as a list [k_1, ..., k_r], where each k_i
    indicates the number of canalizing variables in that layer. If the
    function is fully canalizing (i.e. sum(layer_structure) == n and n > 1),
    the last layer must have at least 2 variables.

    **Parameters:**
        
        - n (int): Total number of variables.
        - layer_structure (list[int] | optional): List [k_1, ..., k_r]
          describing the canalizing layer structure. Each k_i ≥ 1, and if
          sum(layer_structure) == n and n > 1, then layer_structure[-1] ≥ 2.
          Set sum(layer_structure)==n to generate a random nested canalizing
          function.
          
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanFunction: Boolean function object.
    
    **References:**
        
        #. He, Q., & Macauley, M. (2016). Stratification and enumeration of
           Boolean functions by canalizing depth. Physica D: Nonlinear
           Phenomena, 314, 1-8.
           
        #. Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence
           of canalization on the robustness of Boolean networks. Physica D:
           Nonlinear Phenomena, 353, 39-47.
    """ 
    rng = utils._coerce_rng(rng)
    if layer_structure is None:
        return random_k_canalizing_function(n,n,EXACT_DEPTH=False,rng=rng)
    else:
        assert sum(layer_structure) == n,'Ensure sum(layer_structure) == n.'
        assert layer_structure[-1] > 1 or n == 1,'The last layer of an NCF has to have size >= 2 whenever n > 1.\nEnsure that layer_structure[-1]>=2.'
        return random_k_canalizing_function_with_specific_layer_structure(n,layer_structure,EXACT_DEPTH=False,rng=rng)

def random_NCF(n : int, layer_structure : Optional[list] = None,
    *, rng=None) -> BooleanFunction:
    """
    Alias of random_nested_canalizing_function.
    """
    return random_nested_canalizing_function(n=n,layer_structure=layer_structure, rng=rng)


## Random network generation
def random_degrees(N : int, n : Union[int, float, list, np.ndarray],
    indegree_distribution : str ='constant', NO_SELF_REGULATION : bool = True,
    *, rng=None) -> np.ndarray:
    """
    Draw an in-degree vector for a network of N nodes.

    You can either (i) pass a full vector of in-degrees and use it as-is, or
    (ii) ask the function to *sample* in-degrees from a chosen distribution.

    **Parameters:**
        
        - N (int) :Number of nodes (>= 1).
        - n (int, float, list[int], np.ndarray[int]): Meaning depends on
          `indegree_distribution`:
            
            - If `n` is a length-N vector of integers, it is returned
              (after validation).
              
            - If `indegree_distribution` in {'constant','dirac','delta'}:
              the single integer `n` describes the in-degree of each node.
              
            - If `indegree_distribution` == 'uniform': `n` is an integer upper
              bound; each node gets an integer sampled *uniformly* from {1, 2,
              ..., n}.
              
            - If `indegree_distribution` == 'poisson': `n` is the Poisson
              rate λ (> 0); each node gets a Poisson(λ) draw, truncated to lie
              in [1, N - int(NO_SELF_REGULATION)].
            
        - indegree_distribution (str, optional): One of {'constant', 'dirac',
          'delta', 'uniform', 'poisson'}. Default 'constant'.
        
        - NO_SELF_REGULATION (bool, optional): If True, later wiring
          generation will disallow self-loops. This parameter is used here to
          cap sampled in-degrees at `N-1`. Default True.
        
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - indegrees (np.ndarray[int] (shape (N,))): The in-degree of each node,
          with values in `[1, N - int(NO_SELF_REGULATION)]` for sampled
          distributions.

    **Raises:**
        
        - AssertionError: If inputs are malformed or out of range.

    **Examples:**
        
        >>> random_degrees(5, n=2, indegree_distribution='constant')
        array([2, 2, 2, 2, 2])

        >>> random_degrees(4, n=2, indegree_distribution='uniform', NO_SELF_REGULATION=True)
        array([2, 1, 2, 2])  # each in {1,2}

        >>> random_degrees(6, n=1.7, indegree_distribution='poisson')
        array([1, 2, 1, 1, 2, 1])

        >>> random_degrees(3, n=[1, 2, 1])
        array([1, 2, 1])
    """
    rng = utils._coerce_rng(rng)

    if isinstance(n, (list, np.ndarray)):
        assert utils.is_list_or_array_of_ints(n,required_length=N) and min(n) >= 1 and max(n) <= N - int(NO_SELF_REGULATION), 'A vector n was submitted.\nEnsure that n is an N-dimensional vector where each element is an integer between 1 and '+ ('N-1' if NO_SELF_REGULATION else 'N')+' representing the indegree of each nodde.'
        indegrees = np.array(n,dtype=int)
    elif indegree_distribution.lower() in ['constant', 'dirac', 'delta']:
        assert (isinstance(n, (int, np.integer)) and n >= 1 and n <= N - int(NO_SELF_REGULATION)), 'n must be an integer between 1 and '+ ('N-1' if NO_SELF_REGULATION else 'N')+' describing the constant degree of each node.'
        indegrees = np.ones(N, dtype=int) * n
    elif indegree_distribution.lower() == 'uniform':
        assert (isinstance(n, (int, np.integer)) and n >= 1 and n <= N - int(NO_SELF_REGULATION)), 'n must be an integer between 1 and ' + ('N-1' if NO_SELF_REGULATION else 'N')+' representing the upper bound of a uniform degree distribution (lower bound == 1).'
        indegrees = rng.integers(1,n+1, size=N)
    elif indegree_distribution.lower() == 'poisson':
        assert (isinstance(n, (int, float, np.integer, np.floating)) and n>0), 'n must be a float > 0 representing the Poisson parameter.'
        indegrees = np.maximum(np.minimum(rng.poisson(lam=n, size=N),N - int(NO_SELF_REGULATION)), 1)
    else:
        raise AssertionError('None of the predefined in-degree distributions were chosen.\nTo use a user-defined in-degree vector, submit an N-dimensional vector as argument for n; each element of n must an integer between 1 and N.')
    return indegrees


def random_edge_list(N : int, indegrees : Union[list, np.array],
    NO_SELF_REGULATION : bool, AT_LEAST_ONE_REGULATOR_PER_NODE : bool = False,
    *, rng=None) -> list:
    """
    Generate a random edge list for a network of N nodes with optional
    constraints.

    Each node i receives indegrees[i] incoming edges chosen at random.
    Optionally, the function can ensure that every node regulates at least
    one other node.

    **Parameters:**
        
        - N (int): Number of nodes.
        - indegrees (list[int] | np.array[int]): List of length N specifying
          the number of regulators for each node.
          
        - NO_SELF_REGULATION (bool): If True, disallow self-regulation.
        - AT_LEAST_ONE_REGULATOR_PER_NODE (bool, optional): If True, ensure
          that each node has at least one outgoing edge (default is False).
        
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - list[tuple[int, int]]: A list of tuples (source, target)
          representing the edges.
    """
    rng = utils._coerce_rng(rng)
    if AT_LEAST_ONE_REGULATOR_PER_NODE == False:
        edge_list = []
        for i in range(N):
            if NO_SELF_REGULATION:
                indices = rng.choice(np.append(np.arange(i), np.arange(i+1, N)), indegrees[i], replace=False)
            else:
                indices = rng.choice(np.arange(N), indegrees[i], replace=False)
            edge_list.extend(list(zip(indices, i * np.ones(indegrees[i], dtype=int))))
    else:
        target_sources = [set() for _ in range(N)]
        for s, t in edge_list:
            target_sources[t].add(s)
        edge_list = []
        outdegrees = np.zeros(N, dtype=int)
        sum_indegrees = sum(indegrees)  # total number of regulations
        for i in range(N):
            if NO_SELF_REGULATION:
                indices = rng.choice(np.append(np.arange(i), np.arange(i+1, N)), indegrees[i], replace=False)
            else:
                indices = rng.choice(np.arange(N), indegrees[i], replace=False)
            outdegrees[indices] += 1
            edge_list.extend(list(zip(indices, i * np.ones(indegrees[i], dtype=int))))
        while min(outdegrees) == 0:
            index_sink = np.where(outdegrees == 0)[0][0]
            index_edge = rng.integers(sum_indegrees)
            t = edge_list[index_edge][1]
            if NO_SELF_REGULATION and t == index_sink: # avoid self-regulation
                continue
            if index_sink in target_sources[t]: # skip if it would duplicate (index_sink -> t)
                continue
            # perform replacement & update bookkeeping
            old_source = edge_list[index_edge][0]
            target_sources[t].discard(old_source)
            target_sources[t].add(index_sink)
            edge_list[index_edge] = (index_sink, t)
            outdegrees[index_sink] += 1
            outdegrees[old_source] -= 1
    return edge_list


def random_wiring_diagram(N : int, n : Union[int, list, np.array, float],
    NO_SELF_REGULATION : bool = True, STRONGLY_CONNECTED : bool = False,
    indegree_distribution : str ='constant',
    AT_LEAST_ONE_REGULATOR_PER_NODE : bool = False,
    n_attempts_to_generate_strongly_connected_network : int = 1000,
    *, rng=None) -> tuple:
    """
    Generate a random wiring diagram for a network of N nodes.

    Each node i is assigned indegrees[i] outgoing edges (regulators) chosen at random.
    Optionally, self-regulation (an edge from a node to itself) can be disallowed,
    and the generated network can be forced to be strongly connected.

    **Parameters:**
        
        - N (int): Number of nodes.
        - n (int | list[int] | np.array[int] | float (if
          indegree_distribution=='poisson')):  Determines the in-degree of
          each node. If an integer, each node has the same number of
          regulators; if a vector, each element gives the number of regulators
          for the corresponding node.
          
        - NO_SELF_REGULATION (bool, optional): If True, self-regulation is
          disallowed (default is True).
          
        - STRONGLY_CONNECTED (bool, optional): If True, the generated network
          is forced to be strongly connected (default is False).
          
        - indegree_distribution (str, optional): In-degree distribution to
          use. Options include 'constant' (or 'dirac'/'delta'), 'uniform', or
          'poisson'. Default is 'constant'.
          
        - AT_LEAST_ONE_REGULATOR_PER_NODE (bool, optional): If True, ensure
          that each node has at least one outgoing edge (default is False).
          
        - n_attempts_to_generate_strongly_connected_network (int, optional):
          Number of attempts to generate a strongly connected wiring diagram
          before raising an error and quitting.
          
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.
 
    **Returns:**
        
        - tuple[np.array[np.array[int]], list[int]]: (matrix, indices) where:
            
            - matrix (np.array[np.array[int]]): An N x N adjacency matrix
              with entries 0 or 1.
              
            - indices (list): A list of length N, where each element is an
              array of selected target indices for the corresponding node.
    """
    rng = utils._coerce_rng(rng)
    indegrees = random_degrees(N,n,indegree_distribution=indegree_distribution,NO_SELF_REGULATION=NO_SELF_REGULATION, rng=rng)

    counter = 0
    while True:  # Keep generating until we have a strongly connected graph
        edges_wiring_diagram = random_edge_list(N, indegrees, NO_SELF_REGULATION, AT_LEAST_ONE_REGULATOR_PER_NODE=AT_LEAST_ONE_REGULATOR_PER_NODE, rng=rng)
        if STRONGLY_CONNECTED:#may take a long time ("forever") if n is small and N is large
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
    return I, indegrees


def rewire_wiring_diagram(I : Union[list, np.array],
    average_swaps_per_edge : float = 10, DO_NOT_ADD_SELF_REGULATION : bool = True,
    FIX_SELF_REGULATION : bool = True, *, rng=None) -> list:
    """
    Degree-preserving rewiring of a wiring diagram (directed graph) via
    double-edge swaps.

    The wiring diagram is given in the “regulators” convention: `I[target]`
    lists all regulators (in-neighbors) of `target`. The routine performs
    random double-edge swaps `(u→v, x→y) → (u→y, x→v)` while **preserving both
    the in-degree and out-degree** of every node. Parallel edges are disallowed.

    **Parameters:**
        
        - I (list[list[int]] | list[np.ndarray[int]]): Representation of the
          adjacency matrix / wiring diagram as a list where `I[target]`
          contains the regulators of node `target`. Each inner list must
          contain distinct integers in `{0, ..., len(I)-1}`.
          
        - average_swaps_per_edge (float, optional): Target number of
          **successful** swaps per edge. Larger values typically yield better
          mixing (more randomized graphs) but take longer. Default 10.
          
        - DO_NOT_ADD_SELF_REGULATION (bool, optional): If True, proposed swaps
          that would create a self-loop `u→u` are rejected. Default True.
          
        - FIX_SELF_REGULATION (bool, optional): If True, *existing* self-loops
          are kept **fixed** and excluded from the pool of swappable edges
          (they remain as-is in the output). If False, self-loops, if present,
          may be swapped away; if `DO_NOT_ADD_SELF_REGULATION` is True, no new
          self-loops will be created. Default True.
          
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - J (list[np.ndarray[int]]): Rewired wiring diagram in the same format
          as `I`. Each `J[v]` is a sorted array of distinct regulators of `v`.

    **Guarantees:**
        
        - **In-degree** and **out-degree** of every node are preserved exactly.
        - No multi-edges (duplicate `u→v`) are introduced.
        - Self-loops are controlled by the two flags above.

    **Notes:**
        
        - If your input contains self-loops and you want to keep them exactly
          as in `I`, use the defaults (`FIX_SELF_REGULATION`,
          `DO_NOT_ADD_SELF_REGULATION`).

    **Example:**
        
        >>> I = random_network(8,3).I   
        >>> J = rewire_wiring_diagram(I)
        >>> sorted(map(len, I)) == sorted(map(len, J))          # in-degrees
        True
        >>> def outdeg(adj):
        ...     N = len(adj); od = [0]*N
        ...     for v, regs in enumerate(adj):
        ...         for u in regs: od[u] += 1
        ...     return od
        >>> outdeg(I) == outdeg(J)                               # out-degrees
        True
    """
    rng = utils._coerce_rng(rng)
    N = len(I)
    
    edges = [(int(regulator),target) for target in range(N) for regulator in I[target] if regulator!=target or not FIX_SELF_REGULATION]
    n_total_edges = len(edges)
    
    Jset = [set(regulators) for regulators in I]
    
    n_rewires_before_stop = int(average_swaps_per_edge * n_total_edges)
    successes = 0
    attempts = 0
    max_attempts = 50 * n_rewires_before_stop + 100

    # Helper to check if adding edge (regulator->target) is allowed
    def edge_ok(regulator, target):
        if DO_NOT_ADD_SELF_REGULATION and regulator == target:
            return False
        if regulator in Jset[target]:
            return False
        return True

    while successes < n_rewires_before_stop and attempts < max_attempts:
        attempts += 1
        
        # Pick two distinct edges uniformly at random
        i, j = rng.choice(n_total_edges,2,replace=False)

        (u, v) = edges[i]
        (x, y) = edges[j]

        # Swapping identical sources or identical targets is fine in principle,
        # but skip trivial cases that do nothing or re-create the same edges.
        if (u == x) or (v == y):
            continue

        # Proposed swapped edges
        a, b = u, y
        c, d = x, v

        # If the proposed edges are identical to originals, skip
        if (a, b) == (u, v) or (c, d) == (x, y):
            continue

        # Check constraints for both new edges
        if not edge_ok(a, b) or not edge_ok(c, d):
            continue

        # Perform the swap: update adjacency and edge list
        # Remove old edges
        Jset[v].discard(u)
        Jset[y].discard(x)
        # Add new edges
        Jset[b].add(a)
        Jset[d].add(c)
        # Commit edges
        edges[i] = (a, b)
        edges[j] = (c, d)

        successes += 1

    # Reconstruct J from adjacency sets
    J = [np.sort(list(Jset[target])) for target in range(N)]
    return J


#for testing:
# depths=0
# EXACT_DEPTH=False
# layer_structures=None
# ALLOW_DEGENERATED_FUNCTIONS=False
# LINEAR=False, 
# biases=0.5
# absolute_biases = 0.
# USE_ABSOLUTE_BIAS=True
# hamming_weights = None
# NO_SELF_REGULATION=True
# STRONGLY_CONNECTED=False
# indegree_distribution='constant'
# n_attempts_to_generate_strongly_connected_network = 1000

def random_network(N : Optional[int] = None, n : Union[int, float, list, np.ndarray, None] = None, 
    depths : Union[int, list, np.ndarray] = 0, EXACT_DEPTH : bool = False,
    layer_structures : Optional[list] = None, 
    ALLOW_DEGENERATED_FUNCTIONS : bool = False, LINEAR : bool = False, 
    biases : Union[float, list, np.ndarray] = 0.5,
    absolute_biases : Union[float, list, np.ndarray] = 0., USE_ABSOLUTE_BIAS : bool = True,
    hamming_weights : Union[int, list, np.ndarray, None] = None,
    NO_SELF_REGULATION : bool = True, STRONGLY_CONNECTED : bool = False, 
    indegree_distribution : str = 'constant', 
    AT_LEAST_ONE_REGULATOR_PER_NODE : bool =False,
    n_attempts_to_generate_strongly_connected_network : int = 1000, 
    I : Union[list, np.array, None] = None, *, rng=None) -> BooleanNetwork:
    """
    Construct a random Boolean network with configurable wiring and rule
    properties.

    The network is built in two stages:
        
        #. **Wiring diagram**:
            
            - If `I` is provided, use it as the wiring diagram  (each `I[v]`
              lists the regulators of node `v`).
            
            - Otherwise, sample a wiring diagram for `N` nodes using
              `random_wiring_diagram(N, n, ...)`, where the per-node
              in-degrees are determined by `n` and `indegree_distribution`.
              Self-loops can be disallowed and strong connectivity can be
              requested.
            
        #. **Update rules**:
            
            - For node `i`, draw a Boolean function with arity `indegrees[i]`
              using `random_function(...)` with the requested constraints on
              canalizing depth (or layer structure), linearity, bias /
              absolute bias, or exact Hamming weight.

    **Parameters:**
        
        - N (int | None, optional): Number of nodes. Required when `I` is not
          provided. Ignored if `I` is given.
          
        - n (int | float | list[int] | np.ndarray[int] | None, optional):
          Controls the **in-degree** distribution when generating a wiring
          diagram (ignored if `I` is given). Interpretation depends on
          `indegree_distribution`:
            
            - 'constant' / 'dirac' / 'delta': every node has constant
              in-degree `n`.
              
            - 'uniform': `n` is an integer upper bound; each node's in-degree
              is sampled uniformly from {1, ..., n}.
              
            - 'poisson': `n` is a positive rate lambda; in-degrees are Poisson
              (lambda) draws, truncated into [1, N - int(NO_SELF_REGULATION)].
              
            - If `n` is an N-length vector of integers, it is taken as the
              exact in-degrees.
            
        - depths (int | list[int] | np.ndarray[int], optional): Requested
          canalizing depth per node for rule generation. If an integer, it
          is broadcast to all nodes and clipped at each node's in-degree. If a
          vector, it must have length N. Interpreted as **minimum** depth
          unless `EXACT_DEPTH`. Default 0.
          
        - EXACT_DEPTH (bool, optional): If True, each function is generated
          with **exactly** the requested depth (or the sum of the
          corresponding `layer_structures[i]` if provided). If False, depth
          is **at least** as large as requested. Default False.
          
        - layer_structures (list | list[list[int]] | None, optional):
          Canalizing **layer structure** specifications.
        
            - If `None` (default), generation is controlled by `depths` /
              `EXACT_DEPTH`.
              
            - If a single list like `[k1, ..., kr]`, the same structure is
              used for all nodes.
              
            - If a list of lists of length N, `layer_structures[i]` is used
              for node i.
              
            - In all cases, `sum(layer_structure[i])` must be <= the node's
              in-degree. When provided, `layer_structures` takes precedence
              over `depths`.
            
        - ALLOW_DEGENERATED_FUNCTIONS (bool, optional): If True and
          `depths==0` and `layer_structures is None`, degenerated functions
          (with non-essential inputs) may be generated (classical NK-Kauffman
          models). If False, generated functions are essential in all
          variables. Default False.
          
        - LINEAR (bool, optional): If True, generate linear Boolean functions
          for all nodes; other rule parameters (bias, canalization, etc.) are
          ignored. Default False.
          
        - biases (float | list[float] | np.ndarray[float], optional):
          Probability of output 1 when generating random (nonlinear)
          functions, used only if `depths==0`, `layer_structures is None`,
          and `not LINEAR` and `not USE_ABSOLUTE_BIAS`. If a scalar, broadcast
          to length N. Must lie in [0, 1]. Default 0.5.
          
        - absolute_biases (float | list[float] | np.ndarray[float], optional):
          Absolute deviation from 0.5 (i.e., `|bias-0.5|*2`), used only if
          `depths==0`, `layer_structures is None`, `not LINEAR`, and
          `USE_ABSOLUTE_BIAS`. If a scalar, broadcast to length N. Must lie
          in [0, 1]. Default 0.0.
        
        - USE_ABSOLUTE_BIAS (bool, optional): If True, `absolute_biases`
          is used to set the bias per rule to either `0.5*(1 - abs_bias)` or
          `0.5*(1 + abs_bias)` at random. If False, `biases` is used. Only
          relevant when `depths==0`, `layer_structures is None`, and
          `not LINEAR`. Default True.
          
        - hamming_weights (int | list[int] | np.ndarray[int] | None,
          optional): Exact Hamming weights (number of ones in each truth
          table). If None, no exact constraint is enforced. If a scalar,
          broadcast to N. If a vector, must have length N. Values must be
          in {0, ..., 2^k} for a k-input rule. Additional constraints apply
          when requesting exact depth zero (see Notes).
          
        - NO_SELF_REGULATION (bool, optional): If True, forbids self-loops
          in **generated** wiring diagrams. Has no effect when `I` is
          provided. Default True.
          
        - STRONGLY_CONNECTED (bool, optional): If True, the wiring generation
          retries until a strongly connected directed graph is found (up to a
          maximum number of attempts) (ignored if `I` is provided). Default
          False.
          
        - indegree_distribution (str:{'constant', 'dirac', 'delta', 'uniform',
          'poisson'}, optional): Distribution used when sampling in-degrees
          (ignored if `I` is provided). Default 'constant'.
          
        - AT_LEAST_ONE_REGULATOR_PER_NODE (bool, optional): If True, ensure
          that each node has at least one outgoing edge (default is False).
          
        - n_attempts_to_generate_strongly_connected_network (int, optional):
          Max attempts for strong connectivity before raising. Default 1000.
          
        - I (list[list[int]] | list[np.ndarray[int]] | None, optional):
          Existing wiring diagram. If provided, `N` and `n` are ignored and
          `indegrees` are computed from `I`.
          
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.

    **Returns:**
        
        - BooleanNetwork: A new Boolean network with wiring `I` (given or
          generated) and a list of node functions `F` generated according to
          the specified constraints.

    **Raises:**
        
        - AssertionError: If input shapes/types are invalid or constraints
          are violated (e.g., requested depth > in-degree, malformed layer
          structures, invalid bias vectors, etc.).
          
        - RuntimeError: If `STRONGLY_CONNECTED` and a strongly connected
          wiring diagram cannot be generated within
          `n_attempts_to_generate_strongly_connected_network` tries.

    **Notes:**
        
        - **Precedence** for rule constraints: `LINEAR` → `layer_structures`
          (if provided) → `depths` (+ `EXACT_DEPTH`) and bias settings only
          apply when no canalization constraints are requested.
          
        - **Bias controls**: Use `USE_ABSOLUTE_BIAS` with `absolute_biases`
          to enforce a fixed distance from 0.5 while allowing either high or
          low bias with equal chance. Otherwise, set `USE_ABSOLUTE_BIAS=False`
          and provide `biases` directly.
          
        - **Hamming weights & canalization**: When `EXACT_DEPTH` and the
          target depth is 0, Hamming weights {0, 1, 2^k - 1, 2^k} correspond
          to canalizing functions and are therefore disallowed if forcing
          non-canalizing functions through `EXACT_DEPTH` (the implementation
          enforces this).

    **Examples:**
        
        >>> # Boolean network with only essential inputs
        >>> bn = random_network(N=10, n=2, ALLOW_DEGENERATED_FUNCTIONS=False)

        >>> # Classic NK-Kauffman network allowing degenerated rules
        >>> bn = random_network(N=10, n=3, ALLOW_DEGENERATED_FUNCTIONS=True)

        >>> # Fixed wiring: reuse an existing diagram but resample rules
        >>> bn0 = random_network(N=6, n=2)
        >>> bn  = random_network(I=bn0.I)

        >>> # Exact canalizing depth k for all nodes
        >>> bn = random_network(N=8, n=3, depths=1, EXACT_DEPTH=True)

        >>> # Nested canalizing update rules with specific layer structure (broadcast)
        >>> bn = random_network(N=5, n=3, layer_structures=[1,2])  # same for all nodes

        >>> # Linear rules
        >>> bn = random_network(N=7, n=2, LINEAR=True)

        >>> # Poisson in-degrees (truncated), no self-regulation, request strong connectivity
        >>> bn = random_network(N=12, n=1.6, indegree_distribution='poisson',
        ...                     NO_SELF_REGULATION=True, STRONGLY_CONNECTED=True)

        >>> # Exact Hamming weights (broadcast)
        >>> bn = random_network(N=6, n=3, hamming_weights=4)

        >>> # To ensure strong connectivity, set ALLOW_DEGENERATED_FUNCTIONS=False
        >>> # and STRONGLY_CONNECTED=True
        >>> bn = random_network(N,n,ALLOW_DEGENERATED_FUNCTIONS=False,STRONGLY_CONNECTED=True) 
    """
    rng = utils._coerce_rng(rng)
    if I is None and N is not None and n is not None: #generate wiring diagram
        I,indegrees = random_wiring_diagram(N,n,NO_SELF_REGULATION=NO_SELF_REGULATION, 
                                            STRONGLY_CONNECTED=STRONGLY_CONNECTED,
                                            indegree_distribution=indegree_distribution, 
                                            AT_LEAST_ONE_REGULATOR_PER_NODE=AT_LEAST_ONE_REGULATOR_PER_NODE,
                                            n_attempts_to_generate_strongly_connected_network = n_attempts_to_generate_strongly_connected_network,rng=rng)
    elif I is not None: #load wiring diagram
        assert isinstance(I, (list, np.ndarray)), "I must be a list or np.array of lists or np.arrays. Each inner list describes the regulators of node i (indexed by 0,1,...,len(I)-1)"
        N = len(I)
        for regulators in I:
            assert utils.is_list_or_array_of_ints(regulators) and min(regulators)>=0 and max(regulators)<=N-1, "Each element in I describes the regulators of a node (indexed by 0,1,...,len(I)-1)"
        indegrees = list(map(len,I))
    else:
        raise AssertionError('At a minimum, the wiring diagram I must be provided or the network size N and degree parameter n.')
       
        
       
    # Process the inputs, turn single inputs into vectors of length N
    if isinstance(depths, (int, np.integer)):
        assert depths >= 0 ,'The canalizing depth must be an integer between 0 and min(indegrees) or an N-dimensional vector of integers must be provided to use different depths per function.'
        depths = [min(indegrees[i],depths) for i in range(N)]
    elif utils.is_list_or_array_of_ints(depths, required_length=N):
        depths = [min(indegrees[i],depths[i]) for i in range(N)]
        assert min(depths) >= 0,"'depths' received a vector as input.\nTo use a user-defined vector, ensure that it is an N-dimensional vector where each element is a non-negative integer."
    else:
        raise AssertionError("Wrong input format for 'depths'.\nIt must be a single integer (or N-dimensional vector of integers) between 0 and N, specifying the minimal canalizing depth or exact canalizing depth (if EXACT_DEPTH==True).")            
    
    if layer_structures == None:
        layer_structures = [None] * N
    elif utils.is_list_or_array_of_ints(layer_structures):
        depth = sum(layer_structures)
        assert depth==0 or (min(layer_structures)>=1 and depth <= min(indegrees)), 'The layer structure must be [] or a vector of positive integers with 0 <= depth = sum(layer_structure) <= N.'
        layer_structures = [layer_structures[:]] * N
    elif np.all([utils.is_list_or_array_of_ints(el) for el in layer_structures]) and len(layer_structures) == N:
        for i,layer_structure in enumerate(layer_structures):
            depth = sum(layer_structure)
            assert depth==0 or (min(layer_structure)>=1 and depth <= indegrees[i]), 'Ensure that layer_structure is an N-dimensional vector where each element represents a layer structure and is either [] or a vector of positive integers with 0 <= depth = sum(layer_structure[i]) <= n = indegrees[i].'
    else:
        raise AssertionError("Wrong input format for 'layer_structure'.\nIt must be a single vector (or N-dimensional vector of layer structures) where the sum of each element is between 0 and N.")
    
    if isinstance(biases, (float, np.floating)):
        biases = [biases] * N
    elif not utils.is_list_or_array_of_floats(biases, required_length=N):
        raise AssertionError("Wrong input format for 'biases'.\nIt must be a single float (or N-dimensional vector of floats) in [0,1] , specifying the bias (probability of a 1) in the generation of the Boolean function.")            
    
    if isinstance(absolute_biases, (float, np.floating)):
        absolute_biases = [absolute_biases] * N
    elif not utils.is_list_or_array_of_floats(absolute_biases, required_length=N):
        raise AssertionError("Wrong input format for 'absolute_biases'.\nIt must be a single float (or N-dimensional vector of floats) in [0,1], specifying the absolute bias (divergence from the 'unbiased bias' of 0.5) in the generation of the Boolean function.")            

    if hamming_weights == None:
        hamming_weights = [None] * N    
    elif isinstance(hamming_weights, (int, np.integer)):
        hamming_weights = [hamming_weights] * N
    elif not utils.is_list_or_array_of_ints(hamming_weights, required_length=N):
        raise AssertionError("Wrong input format for 'hamming_weights'.\nIf provided, it must be a single integer (or N-dimensional vector of integers) in {0,1,...,2^n}, specifying the number of 1s in the truth table of each Boolean function.\nIf EXACT_DEPTH == True and depths==0, it must be in {2,3,...,2^n-2} because all functions with Hamming weight 0,1,2^n-1,2^n are canalizing.")            
            
    #generate functions
    F = [random_function(n=indegrees[i], depth=depths[i], EXACT_DEPTH=EXACT_DEPTH, layer_structure=layer_structures[i], 
                     LINEAR=LINEAR, ALLOW_DEGENERATED_FUNCTIONS=ALLOW_DEGENERATED_FUNCTIONS,
                     bias=biases[i], absolute_bias=absolute_biases[i], USE_ABSOLUTE_BIAS=USE_ABSOLUTE_BIAS,
                     hamming_weight=hamming_weights[i],rng=rng) for i in range(N)]

    return BooleanNetwork(F, I)


def random_null_model(bn : BooleanNetwork, wiring_diagram : str = 'fixed',
                      PRESERVE_BIAS : bool = True, PRESERVE_CANALIZING_DEPTH : bool = True,
                      *, rng=None, **kwargs) -> BooleanNetwork:
    """
    Generate a randomized Boolean network (null model) from an existing
    network, preserving selected properties of the wiring diagram and update
    rules.

    The output network has the same number of nodes as `bn`. You can choose to:
        
        - keep the wiring diagram fixed,
        - re-sample a wiring diagram that preserves each node’s **in-degree**
          only, or
          
        - rewire the original diagram via degree-preserving swaps to keep
          **both in-degrees and out-degrees** unchanged.

    Independently, the node update rules can be randomized while preserving:
        
        - the **bias** (Hamming weight) of each rule’s truth table,
        - the **canalizing depth** of each rule,
        - both simultaneously
        - neither (i.e., just the in-degree).

    **Parameters:**
        
        - bn (BooleanNetwork): The source network.
        - wiring_diagram (str:{'fixed', 'fixed_indegree',
          'fixed_in_and_outdegree'}, optional): How to handle the wiring
          diagram:
            
            - 'fixed' (default) : Use `bn.I` unchanged.
            - 'fixed_indegree' : Sample a fresh wiring diagram with the
              **same in-degree** per node as `bn` (calls
              `random_wiring_diagram` with `N=bn.N` and `n=bn.indegrees`).
              
            - 'fixed_in_and_outdegree' : Randomize the original wiring by
              **double-edge swaps** (calls `rewire_wiring_diagram`),
              preserving both in-degree and out-degree for every node.
            
        - PRESERVE_BIAS (bool, optional): If True, each node’s new function
          keeps the same Hamming weight (number of ones) as the original.
          Default True.
          
        - PRESERVE_CANALIZING_DEPTH (bool, optional): If True, each node’s new
          function has the same canalizing depth as the original. Default True.
        
        - rng (None, optional): Argument for the random number generator,
          implemented in 'utils._coerce_rng'.
        
        - `**kwargs`: Forwarded to the wiring-diagram routine selected above:
            
            - If `wiring_diagram == 'fixed_indegree'`: passed to
              `random_wiring_diagram` (e.g., `NO_SELF_REGULATION`,
              `STRONGLY_CONNECTED`, etc.).
              
            - If `wiring_diagram == 'fixed_in_and_outdegree'`: passed to
              `rewire_wiring_diagram` (e.g., `average_swaps_per_edge`,
              `DO_NOT_ADD_SELF_REGULATION`, `FIX_SELF_REGULATION`).

    **Returns:**
        
        - BooleanNetwork: A new network with randomized components according
          to the selected constraints.

    **Rule Randomization Details:**
        
        Let `f` be an original node rule with in-degree `n` and canalizing
        depth `k`:
            
            - If `PRESERVE_BIAS and PRESERVE_CANALIZING_DEPTH`: A new rule
              is assembled with:
                
                - the **same canalized outputs** sequence as `f`,
                - a **random canalizing order** and **random canalizing
                  inputs**,
                  
                - a **core** function with the **same Hamming weight** as
                  `f`’s core and that is **non-canalizing** and
                  **non-degenerated**.
                
            - If `PRESERVE_BIAS and not PRESERVE_CANALIZING_DEPTH`: A new
              rule with the same Hamming weight is drawn uniformly at random.
              
            - If `PRESERVE_CANALIZING_DEPTH and not PRESERVE_BIAS`: A random
              function with **exact** canalizing depth `d` is generated.
              
            - Else: A random **non-degenerated** function of the same
              in-degree is generated.

    **References:**
        
        #. Kadelka, C., & Murrugarra, D. (2024). *Canalization reduces the
           nonlinearity of regulation in biological networks.* npj Systems
           Biology & Applications, 10(1), 67.

    **Examples:**
        
        >>> # Keep wiring fixed; preserve both bias and canalizing depth (default)
        >>> bn2 = random_null_model(bn)

        >>> # Preserve in-degrees only (new wiring), and only bias of rules
        >>> bn3 = random_null_model(bn, wiring_diagram='fixed_indegree',
        ...                         PRESERVE_BIAS=True, PRESERVE_CANALIZING_DEPTH=False,
        ...                         NO_SELF_REGULATION=True)

        >>> # Preserve both in- and out-degrees via swaps
        >>> bn4 = random_null_model(bn, wiring_diagram='fixed_in_and_outdegree',
        ...                         average_swaps_per_edge=15)
    """
    rng = utils._coerce_rng(rng)
    if wiring_diagram == 'fixed':
        I = bn.I
    elif wiring_diagram == 'fixed_indegree':
        I,indegrees = random_wiring_diagram(N = bn.N, n = bn.indegrees,rng=rng,**kwargs)
    elif wiring_diagram == 'fixed_in_and_outdegree':
        I = rewire_wiring_diagram(I = bn.I, **kwargs)
    else:
        raise AssertionError("There are three choices for the wiring diagram: 1. 'fixed' (i.e., as in the provided BooleanNetwork), 2. 'fixed_indegree' (i.e., edges are shuffled but the indegree is preserved), 3. 'fixed_in_and_outdegree' (i.e., edges are shuffled but both the indegree and outdegree are preserved).")            
        
    F = []
    for i,f in enumerate(bn.F):
        # if i>=n_variables: #constants don't change #TODO: add constants
        #     newF.append(np.array([0,1]))
        #     continue
        if PRESERVE_CANALIZING_DEPTH:
            depth = f.get_canalizing_depth()           
        if PRESERVE_BIAS and PRESERVE_CANALIZING_DEPTH:
            core_function = f.properties['CoreFunction']
            can_outputs = f.properties['CanalizedOutputs']
            
            can_inputs = rng.choice(2,depth,replace=True)
            can_order = rng.choice(f.n,depth,replace=False)
            if f.n-depth==0:
                core_function = np.array([1 - can_outputs[-1]],dtype=int)
            elif f.n-depth==2:
                core_function = rng.choice([np.array([0,1,1,0],dtype=int),np.array([1,0,0,1],dtype=int)])
            else: #if f.n-depth>=3
                hamming_weight = sum(core_function)
                while True:
                    core_function = random_function_with_exact_hamming_weight(f.n-depth, hamming_weight,rng=rng)
                    if not core_function.is_canalizing():
                        if not core_function.is_degenerated():
                            break
            newf = -np.ones(2**bn.indegrees[i],dtype=int)
            for j in range(depth):
                newf[np.where(np.bitwise_and(newf==-1,utils.get_left_side_of_truth_table(bn.indegrees[i])[:,can_order[j]]==can_inputs[j]))[0]] = can_outputs[j]
            newf[np.where(newf==-1)[0]] = core_function
            newf = BooleanFunction(newf)
        elif PRESERVE_BIAS:  #and PRESERVE_CANALIZING_DEPTH==False
            hamming_weight = f.get_hamming_weight()
            newf = random_function_with_exact_hamming_weight(bn.indegrees[i],hamming_weight,rng=rng)
        elif PRESERVE_CANALIZING_DEPTH:
            newf = random_k_canalizing_function(n=bn.indegrees[i],k=depth,EXACT_DEPTH=True,rng=rng)
        else:
            newf = random_non_degenerated_function(n=bn.indegrees[i],rng=rng)
        F.append(newf)
    return BooleanNetwork(F, I)
