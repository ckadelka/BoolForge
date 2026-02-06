#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 13:05:00 2026

@author: ckadelka
"""

import pandas as pd
import boolforge

n = 4
depth=2
EXACT_DEPTH = False

nsim = 100000

res = []
for _ in range(nsim):
    bf = boolforge.random_function(n, 
                                   k = depth, 
                                   UNIFORM_STRUCTURE = True, 
                                   EXACT_DEPTH=EXACT_DEPTH)
    res.append(boolforge.bin2dec(bf))
print(pd.Series(res).value_counts())

A = pd.Series(res).value_counts()
A = pd.DataFrame(A)
A['f'] = [boolforge.dec2bin(el,2**n) for el in A.index]
A['layer_structure'] = [str(boolforge.BooleanFunction(f).get_layer_structure()['LayerStructure']) for f in A['f']]
A['w'] = [min(sum(boolforge.dec2bin(el,2**n)), 2**n - sum(boolforge.dec2bin(el,2**n))) for el in A.index]
A.to_csv(f'out_n{n}_depth{depth}_nsim{nsim}.csv')
mean_by_layer = A.groupby("layer_structure", as_index=False)["count"].mean()
print(mean_by_layer)