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

import utils
import boolean_function
import generate

N=np.random.randint(10,21) #network size
n=np.random.randint(2,9) #degree parameter, require n>1 for the first few tests because all functions with 1 input are canalizing
bias = np.random.random() #bias: probability of ones in truth table format

## Testing random function generation

#Check that a non-canalizing function with exact canalizing depth k==0 is generated
bf = generate.random_non_canalizing_function(n,bias=bias)
assert bf.get_layer_structure()['CanalizingDepth']==0,"generate.random_non_canalizing_function failed"

#Check that a non-canalizing non-degenerated function with exact canalizing depth k==0 is generated
#All variables in non-degenerated functions have an edge effectiveness > 0
bf = generate.random_non_canalizing_non_degenerated_function(n,bias=bias)
assert bf.get_layer_structure()['CanalizingDepth']==0 and min(bf.get_edge_effectiveness())>0,"generate.random_non_canalizing_non_degenerated_function failed"

n=np.random.randint(1,9)
n=1
k=np.random.randint(0,n) #canalizing depth (exact or minimal depending on EXACT_DEPTH)
if k==n-1:#require k!=n-1 Boolean functions with exact canalizing depth k==n-1 do not exist
    k+=1

#Linear functions (XOR-type functions) must have normalized average sensitivity == 1
bf = generate.random_linear_function(n)
assert bf.get_average_sensitivity(EXACT=True)==1,"generate.random_linear_function or boolean_function.get_average_sensitivity(EXACT=True) failed"


#All variables in non-degenerated functions have an edge effectiveness > 0
bf = generate.random_non_degenerated_function(n,bias=bias)
assert min(bf.get_edge_effectiveness())>0,"generate.random_non_degenerated_function failed"


#At least one variable in a degenerated function must have an edge effectiveness == 0
bf = generate.random_degenerated_function(n,bias=bias)
assert min(bf.get_edge_effectiveness())==0,"generate.random_degenerated_function failed"


#Check that a function with exact canalizing depth k is generated
bf = generate.random_k_canalizing_function(n, k, EXACT_DEPTH=True)
assert bf.get_layer_structure()['CanalizingDepth']==k,"generate.random_k_canalizing failed"


#Check that a function with minimal canalizing depth k is generated
bf = generate.random_k_canalizing_function(n, k, EXACT_DEPTH=False)
assert bf.get_layer_structure()['CanalizingDepth']>=k,"generate.random_k_canalizing failed"


#All variables in an NCF are conditionally canalizing
bf = generate.random_nested_canalizing_function(n)
assert bf.is_k_canalizing(n),"generate.random_NCF failed"


#Generate all possible layer structures of n-input NCFs and test if the correct layer structure is recovered
for w in range(1,2**(n-1),2):
    layer_structure = utils.get_layer_structure_of_an_NCF_given_its_Hamming_weight(n,w)[-1]
    bf = generate.random_NCF(n,layer_structure=layer_structure)
    test = np.all(np.array(boolean_function.get_layer_structure_from_can_outputs(bf.get_layer_structure()['CanalizingOutputs'])) == np.array(layer_structure))
    assert test,"generate.random_NCF failed for n = {n} and layer_structure = {layer_structure}"


## Testing random network generation

#create a random NK-Kauffman network. The constant degree is specified by n.
bn = generate.random_network(N,n) 
assert min(bn.degrees)==max(bn.degrees)==n, "failed to create a BN with constant in-degree "+str(n)


#create a BN where all functions have degree n and at least canalizing depth k.
bn = generate.random_network(N,n,k)
depths = [bf.get_layer_structure()['CanalizingDepth'] for bf in bn]
assert min(depths)>=k, "failed to create a BN with minimal canalizing depth "+str(k)


#create a BN where all functions have degree n and exact canalizing depth k.
bn = generate.random_network(N,n,k, EXACT_DEPTH=True)
depths = [bf.get_layer_structure()['CanalizingDepth'] for bf in bn]
assert min(depths)==k, "failed to create a BN with exact canalizing depth "+str(k)


#create a BN where all functions are n-input NCFs of arbitrary layer structure.
bn = generate.random_network(N,n,k=n)
depths = [bf.get_layer_structure()['CanalizingDepth'] for bf in bn]
assert min(depths)==n, "failed to create a nested canalizing BN"


#create a BN where all functions are n-input NCFs with exactly one layer.
bn = generate.random_network(N,n,layer_structure=[n])
checks_per_function = [bf.get_layer_structure()['CanalizingDepth']==n and bf.get_layer_structure()['NumberOfLayers']==1 for bf in bn]
assert np.all(checks_per_function), "failed to create BN with only single-layer NCFs"


#creates a BN with linear update rules
bn = generate.random_network(N,n, LINEAR=True)
assert min([bf.get_effective_degree() for bf in bn]) == n, "failed to create a linear BN, in which the degree of update rules coincides with their effective degree"


#creates a BN where the underlying wiring diagram contains no self-loops
bn = generate.random_network(N,n,NO_SELF_REGULATION=True)
assert np.all([i not in regulators for i,regulators in enumerate(bn.I)]), "failed to create a BN without self-loops"


#creates a random NK-Kauffman network in which all inputs are essential.
bn = generate.random_network(N,n,ALLOW_DEGENERATED_FUNCTIONS=False) 
assert np.all([bf.is_degenerated()==False for bf in bn]), "failed to create a BN in which are inputs are essential"


#Creates a BN with defined wiring diagram (specified by I, see BooleanNetwork.I), only randomizes the update rules.
bn = generate.random_network(N,n)
bn2 = generate.random_network(I = bn.I) 
assert np.all([np.all(I1 == I2) for I1,I2 in zip(bn.I, bn2.I)]), "failed to create a BN with defined wiring diagram"








