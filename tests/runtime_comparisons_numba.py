#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 12:09:13 2026

@author: ckadelka
"""

import boolforge

N = 12
n = 3
nsim = 100

bns = []
for i in range(nsim):
   bn = boolforge.random_network(N,n)
   bns.append( BooleanNetwork(F = bn.F,I = bn.I) )
   #bns.append( bn )

# %%timeit
# for bn in bns: bn.get_attractors_synchronous_exact(True)

# %%timeit
# for bn in bns: bn.get_attractors_synchronous_exact(False)

# %%timeit
# for bn in bns: bn.get_attractors_and_robustness_measures_synchronous_exact(True)

# %%timeit
# for bn in bns: bn.get_attractors_and_robustness_measures_synchronous_exact(False)

# %%timeit
# for bn in bns: bn.get_derrida_value(USE_NUMBA=True,EXACT=False)

# %%timeit
# for bn in bns: bn.get_derrida_value(USE_NUMBA=False,EXACT=False)