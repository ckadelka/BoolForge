#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:16:55 2025

@author: ckadelka
"""

import sys #TODO: ideally remove this, keep for now
sys.path.append('../boolforge/')
sys.path.append('boolforge/')

import numpy as np

import generate
from boolean_function import BooleanFunction
from boolean_network import BooleanNetwork


#Generate a random Boolean function, turn it into a cana object and back and ensure it is the same function
n=np.random.randint(1,9)
bf = generate.random_function(n)
bf_converted_to_cana = bf.to_cana()
bf_reconverted = BooleanFunction.from_cana(bf_converted_to_cana)
assert np.all(bf.f == bf_reconverted.f), 'failed BooleanFunction.to_cana or BooleanFunction.from_cana'


#Generate a random Boolean network, turn it into a cana BooleanNetwork and back and ensure it is the same network
N = np.random.randint(3,20)
n = np.random.randint(1,min(N,8))
bn = generate.random_network(N,n)
cana_bn = bn.to_cana()
bn_reconverted = BooleanNetwork.from_cana(cana_bn)
assert (np.all([np.all(bn.F[i].f == bn_reconverted.F[i].f) for i in range(N)]) and
        np.all([np.all(bn.I[i] == bn_reconverted.I[i]) for i in range(N)]) and 
        np.all(bn.variables == bn_reconverted.variables)), 'failed pyboolnet_bnet_to_BooleanNetwork or to_pyboolnet_bnet'


#Generate a random Boolean network, turn it into a pyboolnet bnet and back and ensure it is the same network
N = np.random.randint(3,20)
n = np.random.randint(1,min(N,8))
bn = generate.random_network(N,n)
bnet = bn.to_bnet()
bn_reconverted = BooleanNetwork.from_bnet(bnet)
assert (np.all([np.all(bn.F[i].f == bn_reconverted.F[i].f) for i in range(N)]) and
        np.all([np.all(bn.I[i] == bn_reconverted.I[i]) for i in range(N)]) and 
        np.all(bn.variables == bn_reconverted.variables)), 'failed pyboolnet_bnet_to_BooleanNetwork or to_pyboolnet_bnet'
