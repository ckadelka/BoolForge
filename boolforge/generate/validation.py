#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def _validate_bias(bias: float) -> None:
    if not isinstance(bias, (float, int, np.floating)):
        raise TypeError("bias must be a float")
    if not (0.0 <= bias <= 1.0):
        raise ValueError("bias must be in [0, 1]")

def _validate_absolute_bias(absolute_bias: float) -> None:
    if not isinstance(absolute_bias, (float, int, np.floating)):
        raise TypeError("absolute_bias must be a float")
    if not (0.0 <= absolute_bias <= 1.0):
        raise ValueError("absolute_bias must be in [0, 1]")

def _validate_hamming_weight(
    n: int,
    hamming_weight: int,
    *,
    exact_depth: bool,
) -> None:
    if not isinstance(hamming_weight, (int, np.integer)):
        raise TypeError("hamming_weight must be an integer")
    if not (0 <= hamming_weight <= 2**n):
        raise ValueError("hamming_weight must satisfy 0 <= hamming_weight <= 2**n")

    if exact_depth and not (1 < hamming_weight < 2**n - 1):
        raise ValueError(
            "If exact_depth=True and depth=0, hamming_weight must be in "
            "{2, 3, ..., 2**n - 2}. "
            "Functions with weights 0, 1, 2**n-1, 2**n are canalizing."
        )

