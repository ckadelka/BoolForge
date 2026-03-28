#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:17:28 2026

@author: ckadelka
"""

import numpy as np

def _is_boolean(n_states, n_states_inputs):
    if n_states_inputs is None:
        return n_states == 2
    return n_states == 2 and all(k == 2 for k in n_states_inputs)

def _is_homogeneous(n_states, n_states_inputs):
    return n_states_inputs is None or all(k == n_states for k in n_states_inputs)

def normalize_and_validate_state_specs(
    n: int,
    n_states: int,
    n_states_inputs,
) -> tuple[int, np.ndarray]:
    
    # validate output states
    if not isinstance(n_states, (int, np.integer)):
        raise TypeError("n_states must be an integer")
    if n_states < 2:
        raise ValueError("n_states must be >= 2")

    # normalize input states
    if isinstance(n_states_inputs, (int, np.integer)):
        n_states_inputs = np.full(n, n_states_inputs, dtype=int)
    else:
        n_states_inputs = np.asarray(n_states_inputs, dtype=int)

    if n_states_inputs.shape != (n,):
        raise ValueError("n_states_inputs must have length n")

    if np.any(n_states_inputs < 2):
        raise ValueError("All n_states_inputs must be >= 2")

    is_boolean = _is_boolean(n_states, n_states_inputs)

    return n_states, n_states_inputs, is_boolean