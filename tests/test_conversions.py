#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:16:55 2025

@author: ckadelka
"""

import numpy as np

import boolforge


#Generate a random Boolean function, turn it into a cana object and back and ensure it is the same function
n=np.random.randint(1,9)
bf = boolforge.random_function(n)
bf_converted_to_cana = bf.to_cana()
bf_reconverted = boolforge.BooleanFunction.from_cana(bf_converted_to_cana)
assert np.all(bf.f == bf_reconverted.f), 'failed BooleanFunction.to_cana or BooleanFunction.from_cana'


#Generate a random Boolean network, turn it into a cana BooleanNetwork and back and ensure it is the same network
N = np.random.randint(3,20)
n = np.random.randint(1,min(N,8))
bn = boolforge.random_network(N,n)
cana_bn = bn.to_cana()
bn_reconverted = boolforge.BooleanNetwork.from_cana(cana_bn)
assert (np.all([np.all(bn.F[i].f == bn_reconverted.F[i].f) for i in range(N)]) and
        np.all([np.all(bn.I[i] == bn_reconverted.I[i]) for i in range(N)]) and 
        np.all(bn.variables == bn_reconverted.variables)), 'failed BooleanNetwork.to_cana or BooleanNetwork.from_cana'


#Generate a random Boolean network, turn it into a bnet string and back and ensure it is the same network
N = np.random.randint(3,20)
n = np.random.randint(1,min(N,8))
bn = boolforge.random_network(N,n)
bnet = bn.to_bnet()
bn_reconverted = boolforge.BooleanNetwork.from_string(bnet, original_not = "1 - ", original_and = " * ", original_or = " + ")
assert (np.all([np.all(bn.F[i].f == bn_reconverted.F[i].f) for i in range(N)]) and
        np.all([np.all(bn.I[i] == bn_reconverted.I[i]) for i in range(N)]) and 
        np.all(bn.variables == bn_reconverted.variables)), 'failed BooleanNetwork.to_bnet or BooleanNetwork.from_bnet'
