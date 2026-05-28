#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides functions for generating random Boolean functions and
Boolean networks with specified structural and dynamical properties.

The :mod:`~boolforge.generate` module enables the systematic creation of
Boolean functions and networks that satisfy particular constraints, such as
specified canalization depth, sensitivity range, bias, or connectivity.
Generated instances can be used for statistical analysis, benchmarking, or
simulation studies.

Several generation routines leverage Numba acceleration for efficient sampling
and evaluation of large function spaces. While Numba is **recommended** to
achieve near-native performance, it is **not required** for functionality; all
functions have pure Python fallbacks.

This module complements :mod:`~boolforge.boolean_function` and
:mod:`~boolforge.boolean_network` by facilitating reproducible generation of
synthetic test cases and large ensembles of random networks.

Example
-------
>>> import boolforge
>>> boolforge.random_function(n=3)
>>> boolforge.random_network(N=5, n=2)
"""

import numpy as np

from ..boolean_function import BooleanFunction
from .. import utils

from .canalization import random_k_canalizing_function_with_specific_layer_structure
from .canalization import random_k_canalizing_function
from .canalization import random_non_canalizing_function
from .canalization import random_non_canalizing_non_degenerate_function
from .validate import _validate_absolute_bias
from .validate import _validate_bias
from .validate import _validate_hamming_weight

def random_function(
    n: int,
    depth: int = 0,
    exact_depth: bool = False,
    uniform_over_functions: bool = True,
    layer_structure: list[int] | None = None,
    parity: bool = False,
    allow_degenerate_functions: bool = False,
    bias: float = 0.5,
    absolute_bias: float = 0,
    use_absolute_bias: bool = False,
    hamming_weight: int | None = None,
    *,
    rng = None,
) -> BooleanFunction:
    """
    Generate a random Boolean function under flexible structural constraints.

    This function acts as a high-level generator that unifies several common
    ensembles of Boolean functions, including parity functions, canalizing
    functions of specified depth or layer structure, functions with fixed
    Hamming weight, and biased random functions. The first applicable
    generation rule (in the order described below) is applied.

    Selection logic (first applicable rule is used)

    1. If ``parity`` is True, return a random parity function
       (see ``random_parity_function``).

    2. Else, if ``layer_structure`` is provided, return a Boolean function
       with the specified canalizing layer structure using
       ``random_k_canalizing_function_with_specific_layer_structure``.
       Exactness of the canalizing depth is controlled by ``exact_depth``.

    3. Else, if ``depth > 0``, return a k-canalizing function with
       ``k = min(depth, n)`` using ``random_k_canalizing_function``.
       If ``exact_depth`` is True, the function has exactly this depth;
       otherwise, its canalizing depth is at least ``k``.

       If ``uniform_over_functions`` is True, canalizing layer structures are
       sampled uniformly at random (up to the imposed constraints).
       If False, canalized outputs are sampled independently and uniformly
       as bitstrings, which biases the distribution toward more symmetric
       layer structures.

    4. Else, if ``hamming_weight`` is provided, repeatedly sample Boolean
       functions with the specified Hamming weight until additional
       constraints implied by ``exact_depth`` and
       ``allow_degenerate_functions`` are satisfied.

    5. Else, generate a random Boolean function using a Bernoulli model with
       either:

       - fixed bias ``bias``, or
       - an automatically chosen bias determined by ``absolute_bias`` if
         ``use_absolute_bias`` is True.

       Additional constraints on canalization and degeneracy are enforced
       depending on ``exact_depth`` and ``allow_degenerate_functions``.

    Parameters
    ----------
    n : int
        Number of input variables. Must be a positive integer.
    depth : int, optional
        Requested canalizing depth. Used only if ``layer_structure`` is None
        and ``depth > 0``. If ``exact_depth`` is True, the function has exactly
        this canalizing depth (clipped at ``n``); otherwise, its depth is at
        least ``depth``. Default is 0.
    exact_depth : bool, optional
        Enforce exact canalizing depth where applicable. If ``depth == 0``,
        setting ``exact_depth=True`` enforces that the function is
        non-canalizing. Default is False.
    uniform_over_functions : bool, optional
        If True (default), canalizing layer structures are sampled uniformly
        at random in canalizing-function branches. If False, canalized outputs
        are sampled independently as bitstrings, inducing a bias toward more
        symmetric structures.
    layer_structure : list[int] or None, optional
        Explicit canalizing layer structure ``[k1, ..., kr]``. If provided,
        this takes precedence over ``depth``. Default is None.
    parity : bool, optional
        If True, ignore all other options and return a random parity function.
        Default is False.
    allow_degenerate_functions : bool, optional
        If True, functions with non-essential variables may be returned in
        random-generation branches. If False, non-degenerate functions are
        enforced whenever possible. Default is False.
    bias : float, optional
        Probability of a 1 when sampling truth-table entries independently.
        Used only if ``use_absolute_bias`` is False and no other branch applies.
        Must lie in ``[0, 1]``. Default is 0.5.
    absolute_bias : float, optional
        Absolute deviation from 0.5 used to determine the bias when
        ``use_absolute_bias`` is True. The bias is chosen uniformly from
        ``{0.5*(1 - absolute_bias), 0.5*(1 + absolute_bias)}``. Must lie in
        ``[0, 1]``. Default is 0.
    use_absolute_bias : bool, optional
        If True, ignore ``bias`` and determine the bias using
        ``absolute_bias``. Default is False.
    hamming_weight : int or None, optional
        If provided, enforce that the Boolean function has exactly this many
        ones in its truth table. Additional constraints are enforced depending
        on ``exact_depth`` and ``allow_degenerate_functions``. Default is None.
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        A randomly generated Boolean function of arity ``n``.

    Raises
    ------
    TypeError
        If parameters have invalid types.
    ValueError
        If parameter values or combinations are invalid.

    Notes
    -----
    For any fixed combination of parameters, this function samples **uniformly
    at random** from the set of Boolean functions satisfying the corresponding
    constraints. Non-uniformity arises only when explicitly requested via
    ``uniform_over_functions=False``.

    Extremely biased functions are often degenerate or highly canalizing;
    under restrictive parameter choices, some branches may reject repeatedly
    before returning a valid function.

    Examples
    --------
    >>> # Unbiased, non-degenerate random function
    >>> f = random_function(n=3)

    >>> # Function with canalizing depth at least 2
    >>> f = random_function(n=5, depth=2)

    >>> # Function with exact canalizing depth 2
    >>> f = random_function(n=5, depth=2, exact_depth=True)

    >>> # Function with a specific canalizing layer structure
    >>> f = random_function(n=6, layer_structure=[2, 1])

    >>> # Parity function
    >>> f = random_function(n=4, parity=True)

    >>> # Fixed Hamming weight with non-canalizing and non-degenerate constraints
    >>> f = random_function(
    ...     n=5,
    ...     hamming_weight=10,
    ...     exact_depth=True,
    ...     allow_degenerate_functions=False
    ... )
    """

    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    rng = utils._coerce_rng(rng)

    # ------------------------------------------------------------
    # Parity branch (highest priority)
    # ------------------------------------------------------------
    if parity:
        return random_parity_function(n, rng=rng)

    # ------------------------------------------------------------
    # Layer structure branch
    # ------------------------------------------------------------
    if layer_structure is not None:
        return random_k_canalizing_function_with_specific_layer_structure(
            n,
            layer_structure,
            exact_depth=exact_depth,
            rng=rng,
        )

    # ------------------------------------------------------------
    # Canalizing depth branch
    # ------------------------------------------------------------
    if not isinstance(depth, (int, np.integer)) or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    
    if depth > 0:
        return random_k_canalizing_function(
            n,
            min(depth, n),
            exact_depth=exact_depth,
            uniform_over_functions=uniform_over_functions,
            rng=rng,
        )

    # ------------------------------------------------------------
    # Fixed Hamming weight branch
    # ------------------------------------------------------------
    if hamming_weight is not None:
        _validate_hamming_weight(n, hamming_weight, exact_depth=exact_depth)

        while True:
            f = random_function_with_exact_hamming_weight(
                n, hamming_weight, rng=rng
            )

            if allow_degenerate_functions and exact_depth:
                if not f.is_canalizing():
                    return f

            elif allow_degenerate_functions:
                return f

            elif exact_depth:
                if not f.is_canalizing() and not f.is_degenerate():
                    return f

            else:
                if not f.is_degenerate():
                    return f

    # ------------------------------------------------------------
    # Bias-based random generation
    # ------------------------------------------------------------
    if use_absolute_bias:
        _validate_absolute_bias(absolute_bias)
        bias_of_function = rng.choice(
            [0.5 * (1 - absolute_bias), 0.5 * (1 + absolute_bias)]
        )
    else:
        _validate_bias(bias)
        bias_of_function = bias

    if allow_degenerate_functions:
        if exact_depth:
            return random_non_canalizing_function(
                n, bias_of_function, rng=rng
            )
        else:
            return random_function_with_bias(
                n, bias_of_function, rng=rng
            )

    else:
        if exact_depth:
            return random_non_canalizing_non_degenerate_function(
                n, bias_of_function, rng=rng
            )
        else:
            return random_non_degenerate_function(
                n, bias_of_function, rng=rng
            )


def random_function_with_bias(
    n: int,
    bias: float = 0.5,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random Boolean function with a specified bias.

    The Boolean function is represented by its truth table of length
    ``2**n``, where each entry is independently set to 1 with probability
    ``bias`` and to 0 otherwise.

    Parameters
    ----------
    n : int
        Number of Boolean variables.
    bias : float, optional
        Probability that a given truth-table entry equals 1. Default is 0.5.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        Random Boolean function with the specified bias.
    """
    rng = utils._coerce_rng(rng)
    return BooleanFunction._from_f_unchecked(
        np.array(rng.random(2**n) < bias, dtype=int)
    )


def random_function_with_exact_hamming_weight(
    n: int,
    hamming_weight: int,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random Boolean function with a fixed Hamming weight.

    The Boolean function is represented by its truth table of length
    ``2**n``, containing exactly ``hamming_weight`` entries equal to 1.
    All such functions are sampled uniformly at random.

    Parameters
    ----------
    n : int
        Number of Boolean variables.
    hamming_weight : int
        Number of truth-table entries equal to 1.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        Random Boolean function with exactly ``hamming_weight`` ones in its
        truth table.

    Raises
    ------
    TypeError
        If ``hamming_weight`` is not an integer.
    ValueError
        If ``hamming_weight`` is not in the range ``[0, 2**n]``.
    """
    rng = utils._coerce_rng(rng)

    if not isinstance(hamming_weight, (int, np.integer)):
        raise TypeError("hamming_weight must be an integer")

    if not (0 <= hamming_weight <= 2**n):
        raise ValueError("hamming_weight must satisfy 0 <= hamming_weight <= 2**n")

    one_indices = rng.choice(2**n, hamming_weight, replace=False)
    f = np.zeros(2**n, dtype=int)
    f[one_indices] = 1

    return BooleanFunction._from_f_unchecked(f)


def random_parity_function(
    n: int,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random parity Boolean function.

    A parity Boolean function evaluates to the parity (sum modulo 2) of all
    input variables, optionally shifted by a constant. This function returns
    either the parity function or its complement, chosen uniformly at random.

    Parameters
    ----------
    n : int
        Number of Boolean variables.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        Random parity Boolean function on ``n`` variables.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.

    Notes
    -----
    - The returned function is either
      ``x1 XOR x2 XOR ... XOR xn`` or its complement.
    - All variables are included symmetrically.
    - Parity functions are never canalizing. All variables must always be known
      to determine the output; they have maximal average sensitivity.

    Examples
    --------
    >>> f = random_parity_function(3)
    >>> sum(f)
    4
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    rng = utils._coerce_rng(rng)

    # choose parity or its complement
    val = rng.integers(2)

    f = np.zeros(2**n, dtype=np.uint8)
    for i in range(2**n):
        if i.bit_count() % 2 == val:
            f[i] = 1

    return BooleanFunction._from_f_unchecked(f)



def random_non_degenerate_function(
    n: int,
    bias: float = 0.5,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random non-degenerate Boolean function.

    A Boolean function is non-degenerate if every variable is essential, i.e.,
    the function depends on all ``n`` input variables. Functions are sampled
    repeatedly from the Bernoulli(bias) ensemble until a non-degenerate
    function is obtained.

    Parameters
    ----------
    n : int
        Number of Boolean variables.
    bias : float, optional
        Probability that a truth-table entry equals 1. Default is 0.5.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        Random non-degenerate Boolean function.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.
    ValueError
        If ``bias`` is not strictly between 0 and 1.

    Notes
    -----
    - For moderate bias values, almost all Boolean functions are non-degenerate.
    - Extremely biased functions are very likely to be degenerate, which may
      lead to long rejection-sampling times.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(bias, (float, np.floating)) or not (0.0 < bias < 1.0):
        raise ValueError("bias must be a float strictly between 0 and 1")

    rng = utils._coerce_rng(rng)

    # Rejection sampling; almost all Boolean functions are non-degenerate
    while True:
        f = random_function_with_bias(n, bias=bias, rng=rng)
        if not f.is_degenerate():
            return f


def random_degenerate_function(
    n: int,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random degenerate Boolean function uniformly at random.
    
    A Boolean function is degenerate if at least one variable is
    non-essential, i.e., the function does not depend on that variable.
    By using appropriate acceptance weights, the resulting distribution
    is function-uniform over all degenerate Boolean functions on ``n``
    variables.
    
    Parameters
    ----------
    n : int
        Number of Boolean variables.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.
    
    Returns
    -------
    BooleanFunction
        Random degenerate Boolean function on ``n`` variables, sampled
        uniformly over all degenerate Boolean functions.
    
    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.
    
    Notes
    -----
    - A non-essential variable is forced by construction, but additional
      variables may also be non-essential by chance.
    - The forced non-essential variable is chosen uniformly at random
      from all ``n`` variables.
    - Function-uniformity is achieved by accepting a candidate function
      with ``k`` non-essential variables with probability ``1/k``,
      correcting for the fact that such functions are ``k`` times more
      likely to be proposed than functions with exactly one non-essential
      variable.
    - At bias != 0.5 this acceptance correction would no longer be valid,
      so bias is fixed at 0.5 internally.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    rng = utils._coerce_rng(rng)
    while True:
        # Choose forced non-essential variable uniformly at random
        index_non_essential_variable = rng.integers(n)
        
        # Generate an (n-1)-variable Boolean function
        f_original = random_function_with_bias(n - 1, bias=0.5, rng=rng)
        
        # Copy the (n-1)-variable function across both values of the non-essential variable
        block = 2 ** index_non_essential_variable
        indices = (np.arange(2**n) // block) % 2 == 1
        f = np.zeros(2**n, dtype=np.uint8)
        f[indices] = f_original.f
        f[~indices] = f_original.f
        candidate = BooleanFunction._from_f_unchecked(f)
        
        k = n - candidate.get_number_of_essential_variables()  # number of non-essential variables
        if rng.random() < 1.0 / k:
            return candidate
