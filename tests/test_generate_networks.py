#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import boolforge


def test_random_network_constant_indegree():
    """
    random_network(N, n) must produce a Boolean network
    with constant in-degree equal to n.
    """
    rng = np.random.default_rng(0)

    N = 20
    n = 3

    bn = boolforge.random_network(N, n, rng=rng)

    assert min(bn.indegrees) == max(bn.indegrees) == n, (
        "Failed to create BN with constant in-degree"
    )


def test_random_network_minimal_canalizing_depth():
    """
    All update rules must have canalizing depth at least k
    when depth=k is specified.
    """
    rng = np.random.default_rng(1)

    N = 20
    n = 3
    k = 1

    bn = boolforge.random_network(N, n, depth=k, rng=rng)

    depths = [bf.get_layer_structure()["CanalizingDepth"] for bf in bn]
    assert min(depths) >= k, (
        "Failed to enforce minimal canalizing depth"
    )


def test_random_network_exact_canalizing_depth():
    """
    All update rules must have canalizing depth exactly k
    when EXACT_DEPTH=True.
    """
    rng = np.random.default_rng(2)

    N = 20
    n = 3
    k = 1

    bn = boolforge.random_network(
        N, n, depth=k, EXACT_DEPTH=True, rng=rng
    )

    depths = [bf.get_layer_structure()["CanalizingDepth"] for bf in bn]
    assert min(depths) == k, (
        "Failed to enforce exact canalizing depth"
    )


def test_random_network_nested_canalizing():
    """
    depth = n must generate nested canalizing update rules.
    """
    rng = np.random.default_rng(3)

    N = 20
    n = 3

    bn = boolforge.random_network(N, n, depth=n, rng=rng)

    depths = [bf.get_layer_structure()["CanalizingDepth"] for bf in bn]
    assert min(depths) == n, (
        "Failed to create nested canalizing BN"
    )


def test_random_network_single_layer_ncf():
    """
    layer_structure=[n] must generate single-layer NCFs.
    """
    rng = np.random.default_rng(4)

    N = 20
    n = 3

    bn = boolforge.random_network(
        N, n, layer_structure=[n], rng=rng
    )

    checks = [
        bf.get_layer_structure()["CanalizingDepth"] == n
        and bf.get_layer_structure()["NumberOfLayers"] == 1
        for bf in bn
    ]

    assert np.all(checks), (
        "Failed to create BN with only single-layer NCFs"
    )


def test_random_network_linear_update_rules():
    """
    PARITY=True must generate linear (XOR-type) update rules,
    whose effective degree equals n.
    """
    rng = np.random.default_rng(5)

    N = 20
    n = 3

    bn = boolforge.random_network(N, n, PARITY=True, rng=rng)

    eff_degrees = [bf.get_effective_degree() for bf in bn]
    assert min(eff_degrees) == n, (
        "Failed to create linear BN with correct effective degree"
    )


def test_random_network_no_self_regulation():
    """
    NO_SELF_REGULATION=True must produce a wiring diagram
    without self-loops.
    """
    rng = np.random.default_rng(6)

    N = 20
    n = 3

    bn = boolforge.random_network(
        N, n, NO_SELF_REGULATION=True, rng=rng
    )

    assert np.all(
        i not in regulators
        for i, regulators in enumerate(bn.I)
    ), "Failed to create BN without self-loops"


def test_random_network_non_degenerate_update_rules():
    """
    ALLOW_DEGENERATE_FUNCTIONS=False must enforce that
    all update rules are non-degenerate.
    """
    rng = np.random.default_rng(7)

    N = 20
    n = 3

    bn = boolforge.random_network(
        N, n, ALLOW_DEGENERATE_FUNCTIONS=False, rng=rng
    )

    assert np.all(
        not bf.is_degenerate() for bf in bn
    ), "Failed to enforce non-degenerate update rules"


def test_random_network_with_fixed_wiring_diagram():
    """
    Passing a fixed wiring diagram I must preserve it
    while randomizing update rules.
    """
    rng = np.random.default_rng(8)

    N = 20
    n = 3

    bn1 = boolforge.random_network(N, n, rng=rng)
    bn2 = boolforge.random_network(I=bn1.I, rng=rng)

    assert np.all(
        np.all(I1 == I2)
        for I1, I2 in zip(bn1.I, bn2.I)
    ), "Failed to preserve wiring diagram when I is fixed"
