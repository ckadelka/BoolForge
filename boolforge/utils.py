#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions used throughout BoolForge.

The :mod:`~boolforge.utils` module includes low-level operations for binary and
decimal conversions, truth table manipulations, and combinatorial helper
functions. These utilities are used internally by
:class:`~boolforge.BooleanFunction` and :class:`~boolforge.BooleanNetwork`
classes to enable efficient representation and analysis of Boolean functions
and networks.

Notes
-----
Most functions in this module are intended for internal use and are not part of
the stable public API.

Examples
--------
>>> from boolforge import utils
>>> utils.bin2dec([1, 0, 1])
5
>>> utils.dec2bin(5, 3)
array([1, 0, 1])
"""


##Imports
import numpy as np
import random as _py_random
from collections.abc import Sequence
from numpy.random import Generator as _NPGen, RandomState as _NPRandomState, SeedSequence, default_rng

__all__ = [
    "bin2dec",
    "dec2bin",
    "bool_to_poly",
    "f_from_expression",
    "hamming_weight_to_ncf_layer_structure",
    "get_left_side_of_truth_table"
]

def _coerce_rng(
    rng : int | _NPGen | _NPRandomState | _py_random.Random | None = None
) -> _NPGen:
    """
    Coerce a variety of RNG-like inputs to a NumPy ``Generator``.

    Parameters
    ----------
    rng : int | np.random.Generator | np.random.RandomState | random.Random | None, optional
        Random number generator or seed specification.

        - ``None``: return ``np.random.default_rng()``.
        - ``int``: interpreted as a seed for ``default_rng``.
        - ``np.random.Generator``: returned unchanged.
        - ``np.random.RandomState``: converted via ``SeedSequence``.
        - ``random.Random``: converted via ``SeedSequence``.

    Returns
    -------
    np.random.Generator
        A NumPy random number generator.

    Raises
    ------
    TypeError
        If ``rng`` is not one of the supported types.

    Notes
    -----
    This function provides a unified RNG interface across BoolForge by
    normalizing legacy and standard-library RNGs to the modern NumPy
    ``Generator`` API.

    Conversion from ``RandomState`` and ``random.Random`` is performed by
    extracting entropy and initializing a ``SeedSequence``. This preserves
    reproducibility while avoiding direct reliance on deprecated RNG APIs.
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

def is_float(element: object) -> bool:
    """
    Check whether an object can be coerced to a float.

    Parameters
    ----------
    element : object
        Object to test for float coercibility.

    Returns
    -------
    bool
        True if ``element`` can be converted to ``float`` without raising an
        exception, False otherwise.

    Notes
    -----
    This function tests coercibility, not type membership. For example,
    numeric strings and integers return True.

    Examples
    --------
    >>> is_float(3)
    True
    >>> is_float(3.14)
    True
    >>> is_float("2.7")
    True
    >>> is_float("abc")
    False
    >>> is_float(None)
    False
    """
    try:
        float(element)
        return True
    except (TypeError, ValueError):
        return False

def bin2dec(binary_vector: list[int]) -> int:
    """
    Convert a binary vector to an integer.

    Parameters
    ----------
    binary_vector : list of int
        Binary digits (0 or 1), ordered from most significant bit to least
        significant bit.

    Returns
    -------
    int
        Integer represented by the binary vector.

    Notes
    -----
    No validation is performed to ensure that entries of ``binary_vector`` are
    binary. Nonzero values are treated as 1 under bitwise conversion.

    Examples
    --------
    >>> bin2dec([1, 0, 1])
    5
    >>> bin2dec([0, 0, 1, 1])
    3
    """
    decimal = 0
    for bit in binary_vector:
        decimal = (decimal << 1) | bit
    return int(decimal)


def dec2bin(integer_value: int, num_bits: int) -> list[int]:
    """
    Convert a nonnegative integer to a binary vector.

    Parameters
    ----------
    integer_value : int
        Nonnegative integer to convert.
    num_bits : int
        Length of the binary representation.

    Returns
    -------
    list of int
        Binary digits (0 or 1), ordered from most significant bit to least
        significant bit.

    Notes
    -----
    - If ``integer_value`` requires more than ``num_bits`` bits, the most
      significant bits are truncated.
    - No validation is performed for negative inputs.

    Examples
    --------
    >>> dec2bin(5, 3)
    [1, 0, 1]
    >>> dec2bin(3, 5)
    [0, 0, 0, 1, 1]
    """
    binary_string = bin(integer_value)[2:].zfill(num_bits)
    return [int(bit) for bit in binary_string]


left_side_of_truth_tables = {}

def get_left_side_of_truth_table(N: int) -> np.ndarray:
    """
    Return the left-hand side of a Boolean truth table.

    The left-hand side is the binary representation of all ``2**N`` input
    combinations for ``N`` Boolean variables, ordered lexicographically from
    ``0`` to ``2**N - 1``.

    Parameters
    ----------
    N : int
        Number of Boolean variables.

    Returns
    -------
    np.ndarray
        Array of shape ``(2**N, N)`` with entries in ``{0, 1}``. Columns are
        ordered from most significant bit to least significant bit.

    Notes
    -----
    - The result is cached by ``N`` to avoid recomputation.
    - Row ``i`` corresponds to the binary expansion of integer ``i``.
    - The most significant bit appears in column 0.

    Examples
    --------
    >>> get_left_side_of_truth_table(2)
    array([[0, 0],
           [0, 1],
           [1, 0],
           [1, 1]], dtype=uint8)
    """
    if N in left_side_of_truth_tables:
        left_side_of_truth_table = left_side_of_truth_tables[N]
    else:
        vals = np.arange(2**N, dtype=np.uint64)[:, None]
        masks = (1 << np.arange(N-1, -1, -1, dtype=np.uint64))[None]
        left_side_of_truth_table = ((vals & masks) != 0).astype(np.uint8)
        left_side_of_truth_tables[N] = left_side_of_truth_table
    return left_side_of_truth_table


def find_all_indices(arr: list, el: object) -> list[int]:
    """
    Find all indices of a given element in a sequence.

    Parameters
    ----------
    arr : list
        Sequence to search.
    el : object
        Element to locate.

    Returns
    -------
    list of int
        Indices ``i`` such that ``arr[i] == el``.

    Raises
    ------
    ValueError
        If ``el`` does not occur in ``arr``.

    Examples
    --------
    >>> find_all_indices([1, 2, 1, 3], 1)
    [0, 2]
    >>> find_all_indices(['a', 'b', 'a'], 'a')
    [0, 2]
    """
    res: list[int] = []
    for i, a in enumerate(arr):
        if a == el:
            res.append(i)

    if not res:
        raise ValueError("Element not found in sequence")

    return res


def check_if_empty(my_list: list | np.ndarray) -> bool:
    """
    Check whether a list or NumPy array is empty.

    Parameters
    ----------
    my_list : list or np.ndarray
        Sequence to check.

    Returns
    -------
    bool
        True if ``my_list`` is empty, False otherwise.

    Notes
    -----
    For NumPy arrays, emptiness is determined by ``size == 0``.
    For Python lists, emptiness is determined by equality to ``[]``.
    """
    if isinstance(my_list, np.ndarray):
        return my_list.size == 0
    return my_list == []
    
    
def is_list_or_array_of_ints(
    x: list | np.ndarray,
    required_length: int | None = None,
) -> bool:
    """
    Check whether a list or NumPy array contains only integers.

    Parameters
    ----------
    x : list or np.ndarray
        Sequence to check.
    required_length : int or None, optional
        If provided, require that ``x`` has exactly this length.

    Returns
    -------
    bool
        True if ``x`` is a list of ``int`` / ``np.integer`` or a NumPy array
        with integer dtype, and (if specified) has length ``required_length``.
        False otherwise.

    Notes
    -----
    - For Python lists, each element is checked individually.
    - For NumPy arrays, the dtype is checked using ``np.issubdtype``.
    - One-dimensional arrays are required when ``required_length`` is given.
    """
    # Case 1: Python list
    if isinstance(x, list):
        return (
            (required_length is None or len(x) == required_length)
            and all(isinstance(el, (int, np.integer)) for el in x)
        )

    # Case 2: NumPy array
    if isinstance(x, np.ndarray):
        return (
            (required_length is None or x.shape == (required_length,))
            and np.issubdtype(x.dtype, np.integer)
        )

    return False


def is_list_or_array_of_floats(
    x: list | np.ndarray,
    required_length: int | None = None,
) -> bool:
    """
    Check whether a list or NumPy array contains only floating-point numbers.

    Parameters
    ----------
    x : list or np.ndarray
        Sequence to check.
    required_length : int or None, optional
        If provided, require that ``x`` has exactly this length.

    Returns
    -------
    bool
        True if ``x`` is a list of ``float`` / ``np.floating`` or a NumPy array
        with floating-point dtype, and (if specified) has length
        ``required_length``. False otherwise.

    Notes
    -----
    - For Python lists, each element is checked individually.
    - For NumPy arrays, the dtype is checked using ``np.issubdtype``.
    - One-dimensional arrays are required when ``required_length`` is given.
    """
    # Case 1: Python list
    if isinstance(x, list):
        return (
            (required_length is None or len(x) == required_length)
            and all(isinstance(el, (float, np.floating)) for el in x)
        )

    # Case 2: NumPy array
    if isinstance(x, np.ndarray):
        return (
            (required_length is None or x.shape == (required_length,))
            and np.issubdtype(x.dtype, np.floating)
        )

    return False


def bool_to_poly(
    f: list,
    variables: list[str] | None = None,
    prefix: str = '',
) -> str:
    """
    Convert a Boolean function from truth-table form to disjunctive normal form.

    The returned expression is a non-reduced disjunctive normal form (DNF),
    expressed as a sum of monomials corresponding to truth-table entries where
    the function evaluates to 1.

    Parameters
    ----------
    f : list
        Boolean function values ordered according to the standard truth-table
        convention. The length of ``f`` must be ``2**n`` for some integer ``n``.
    variables : list of str or None, optional
        Variable names to use in the expression. If None or if the length does
        not match the required number of variables, default names
        ``['x0', 'x1', ..., 'x{n-1}']`` are used.
    prefix : str, optional
        Prefix for automatically generated variable names. Ignored if
        ``variables`` is provided with the correct length.

    Returns
    -------
    str
        Boolean expression in non-reduced disjunctive normal form. Returns
        ``'0'`` if the function is identically zero.

    Notes
    -----
    - Variables are ordered from most significant bit to least significant bit.
    - Each monomial corresponds to a single truth-table row where ``f == 1``.
    - No simplification or reduction of the DNF is performed.

    Examples
    --------
    >>> bool_to_poly([0, 1, 1, 0])
    '(1 - x0) * x1 + x0 * (1 - x1)'
    """
    len_f = len(f)
    n = int(np.log2(len_f))
    if variables is None or len(variables) != n:
        prefix = 'x'
        variables = [prefix + str(i) for i in range(n)]

    left_side_of_truth_table = get_left_side_of_truth_table(n)
    num_values = 2 ** n
    text = []

    for i in range(num_values):
        if f[i] == True:
            monomial = ' * '.join(
                [
                    v if entry == 1 else f'(1 - {v})'
                    for v, entry in zip(variables, left_side_of_truth_table[i])
                ]
            )
            text.append(monomial)

    if text:
        return ' + '.join(text)
    return '0'



def f_from_expression(
    expr: str,
    max_degree: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a Boolean function from a string expression.
    
    The expression is evaluated symbolically over all Boolean input
    combinations to produce the truth table of the corresponding Boolean
    function. Variables are detected automatically based on their first
    occurrence in the expression.
    
    Parameters
    ----------
    expr : str
        Boolean expression to evaluate. The expression may contain logical
        operators (``AND``, ``OR``, ``NOT`` or their lowercase equivalents),
        arithmetic operators, and comparisons.
    max_degree : int, optional
        Maximum number of variables allowed. If the number of detected
        variables exceeds ``max_degree``, an empty truth table is returned.
    
    Returns
    -------
    f : np.ndarray
        Boolean function values as an array of shape ``(2**n,)`` with entries
        in ``{0, 1}``, where ``n`` is the number of detected variables.
    variables : np.ndarray
        Variable names in the order they were first encountered in the
        expression.
    
    Notes
    -----
    - Variables are ordered by first occurrence in ``expr``.
    - Truth-table rows follow the standard lexicographic ordering with the
      most significant bit first.
    - The expression is evaluated using ``eval`` with restricted builtins.
    - No syntactic or semantic validation of ``expr`` is performed beyond
      basic parsing.
    
    Examples
    --------
    >>> f_from_expression('A AND NOT B')
    (array([0, 0, 1, 0], dtype=uint8), array(['A', 'B'], dtype='<U1'))
    
    >>> f_from_expression('x1 + x2 + x3 > 1')
    (array([0, 0, 0, 1, 0, 1, 1, 1], dtype=uint8),
     array(['x1', 'x2', 'x3'], dtype='<U2'))
    
    >>> f_from_expression('(x1 + x2 + x3) % 2 == 0')
    (array([1, 0, 0, 1, 0, 1, 1, 0], dtype=uint8),
     array(['x1', 'x2', 'x3'], dtype='<U2'))
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


def flatten(l: Sequence[Sequence[object]]) -> list[object]:
    """
    Flatten a sequence of sequences by one level.

    Parameters
    ----------
    l : list or np.ndarray
        Sequence whose elements are themselves iterable.

    Returns
    -------
    list
        A flat list containing the elements of each sub-sequence in ``l``,
        in order.

    Notes
    -----
    This function performs a single-level flattening only. Nested sequences
    deeper than one level are not recursively flattened.

    Examples
    --------
    >>> flatten([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    >>> flatten(np.array([[1, 2], [3, 4]]))
    [1, 2, 3, 4]
    """
    return [item for sublist in l for item in sublist]


def hamming_weight_to_ncf_layer_structure(
    n: int,
    w: int,
) -> list[int]:
    """
    Compute the canalizing layer structure of a nested canalizing function (NCF)
    from its Hamming weight.

    For nested canalizing functions, there is a bijection between the (odd)
    Hamming weight ``w`` and the canalizing layer structure, with ``w`` and
    ``2**n - w`` corresponding to the same structure.

    Parameters
    ----------
    n : int
        Number of input variables of the NCF.
    w : int
        Odd Hamming weight of the NCF.

    Returns
    -------
    list of int
        Canalizing layer structure ``[k_1, ..., k_r]``.

    Raises
    ------
    TypeError
        If ``w`` is not an integer.
    ValueError
        If ``w`` is outside ``[1, 2**n - 1]`` or if ``w`` is even.

    Notes
    -----
    - All nested canalizing functions have odd Hamming weight.
    - The binary expansion of ``w`` (with ``n`` bits) determines the layer
      structure.
      
    References
    ----------
    Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). 
    The influence of canalization on the robustness of Boolean networks. 
    Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    if not isinstance(w, (int, np.integer)):
        raise TypeError("Hamming weight w must be an integer")

    if not (1 <= w <= 2**n - 1):
        raise ValueError("Hamming weight w must satisfy 1 <= w <= 2**n - 1")

    if w % 2 == 0:
        raise ValueError("Hamming weight w must be odd for nested canalizing functions")

    if w == 1:
        return [n]

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

