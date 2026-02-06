#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import boolforge


def test_boolean_function_cana_bijection():
    """
    Generate a random Boolean function, convert it to cana format and back,
    and ensure the truth table is preserved.
    """
    rng = np.random.default_rng(0)

    n = rng.integers(1, 9)
    bf = boolforge.random_function(n, rng=rng)

    bf_converted_to_cana = bf.to_cana()
    bf_reconverted = boolforge.BooleanFunction.from_cana(bf_converted_to_cana)

    assert np.all(bf.f == bf_reconverted.f), (
        "BooleanFunction.to_cana / from_cana roundtrip failed"
    )


def test_boolean_network_cana_bijection():
    """
    Generate a random Boolean network, convert it to cana format and back,
    and ensure all functions, wiring, and variables are preserved.
    """
    rng = np.random.default_rng(1)

    N = rng.integers(3, 20)
    n = rng.integers(1, min(N, 8))

    bn = boolforge.random_network(N, n, rng=rng)

    cana_bn = bn.to_cana()
    bn_reconverted = boolforge.BooleanNetwork.from_cana(cana_bn)

    assert all(
        np.all(bn.F[i].f == bn_reconverted.F[i].f)
        for i in range(N)
    ), "Boolean functions differ after cana roundtrip"

    assert all(
        np.all(bn.I[i] == bn_reconverted.I[i])
        for i in range(N)
    ), "Wiring diagram differs after cana roundtrip"

    assert np.all(
        bn.variables == bn_reconverted.variables
    ), "Variables differ after cana roundtrip"


def test_boolean_network_bnet_bijection():
    """
    Generate a random Boolean network, convert it to a bnet string and back,
    and ensure all functions, wiring, and variables are preserved.
    """
    rng = np.random.default_rng(2)

    N = rng.integers(3, 20)
    n = rng.integers(1, min(N, 8))

    bn = boolforge.random_network(N, n, rng=rng)

    bnet = bn.to_bnet()
    bn_reconverted = boolforge.BooleanNetwork.from_string(
        bnet,
        original_not="1 - ",
        original_and=" * ",
        original_or=" + ",
    )

    assert all(
        np.all(bn.F[i].f == bn_reconverted.F[i].f)
        for i in range(N)
    ), "Boolean functions differ after bnet roundtrip"

    assert all(
        np.all(bn.I[i] == bn_reconverted.I[i])
        for i in range(N)
    ), "Wiring diagram differs after bnet roundtrip"

    assert np.all(
        bn.variables == bn_reconverted.variables
    ), "Variables differ after bnet roundtrip"
