#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jul 29 09:25:40 2025

@author: Claus Kadelka, Benjamin Coberly
"""


##Imports
from __future__ import annotations
import numpy as np
import itertools
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
            monomial = ' * '.join([('%s' % (v)) if entry == 1 else ('(1-%s)' % (v)) 
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
        local_dict = {var: truth_table[:, i] for i, var in enumerate(variables)}
        f = eval(expr_mod, {"__builtins__": None}, local_dict)
    else:
        f = []
    return np.array(f,dtype=int), np.array(variables)


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
        
        - tuple[int, list[int]]: A tuple (r, layer_structure_NCF), where:
            
            - r (int): The number of canalizing layers.
            - layer_structure_NCF (list[int]): A list [k_1, ..., k_r]
              describing the number of variables in each layer.

    **References:**
        
        #. Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). The influence
           of canalization on the robustness of Boolean networks. Physica D:
           Nonlinear Phenomena, 353, 39-47.
    """
    if w == 1:
        r = 1
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
        r = len(layer_structure_NCF)
    return (r, layer_structure_NCF)

