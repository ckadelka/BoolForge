#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 14:45:48 2025

@author: ckadelka
"""


import itertools
import math
from collections import defaultdict
from copy import deepcopy

import numpy as np
import networkx as nx
import pandas as pd

from typing import Union, Optional


try:
    import boolforge.utils as utils
    from boolforge.boolean_function import BooleanFunction
    from boolforge.boolean_network import BooleanNetwork, WiringDiagram
except ModuleNotFoundError:
    import utils as utils
    from boolean_function import BooleanFunction
    from boolean_network import BooleanNetwork, WiringDiagram
    
class ModularBooleanNetwork(BooleanNetwork):
    def __init__(self, F : Union[list, np.ndarray], I : Union[list, np.ndarray, WiringDiagram],
                 variables : Union[list, np.array, None] = None,
                 SIMPLIFY_FUNCTIONS : Optional[bool] = False):
        super().__init__(F, I,variables,SIMPLIFY_FUNCTIONS)

    def get_attractors_synchronous_exact_with_external_inputs(self, input_patterns = [], starting_states = None):
        if starting_states is None:
            starting_states = list(range(2**self.N))
        
        n_const = len(input_patterns)
        
        length_input_patterns = list(map(len,input_patterns))
        lcm = math.lcm(*length_input_patterns)
        periodic_pattern_of_external_inputs = np.zeros((lcm,n_const),dtype=int)
        for i,pattern in enumerate(input_patterns):
            for j in range(int(lcm/len(pattern))):
                periodic_pattern_of_external_inputs[len(pattern)*j:len(pattern)*(j+1),i] = pattern
    
        n_initial_values = len(periodic_pattern_of_external_inputs)
        
        Fs_with_fixed_sources = []
        Is_with_fixed_sources = []
        degrees_with_fixed_sources = []
        for input_values in periodic_pattern_of_external_inputs:
            dummy = get_BN_with_fixed_source_nodes(self.F,self.I,self.N,n_const,input_values)
            Fs_with_fixed_sources.append(dummy[0])
            Is_with_fixed_sources.append(dummy[1])
            degrees_with_fixed_sources.append(list(map(len, dummy[1])))
        
        
        left_side_of_truth_table = boolforge.get_left_side_of_truth_table(self.N)
        
        powers_of_2 = np.array([2**i for i in range(self.N)])[::-1]
        
        dictF_with_fixed_source = []
        
        for iii in range(n_initial_values):
            state_space = np.zeros((2**N, N), dtype=int)
            for i in range(N):
                for j, x in enumerate(itertools.product([0, 1], repeat=degrees_with_fixed_sources[iii][i])):
                    if Fs_with_fixed_sources[iii][i][j]==1:
                        # For rows in left_side_of_truth_table where the columns I[i] equal x, set state_space accordingly.
                        state_space[np.all(left_side_of_truth_table[:, Is_with_fixed_sources[iii][i]] == np.array(x), axis=1), i] = 1
            dictF_with_fixed_source.append(dict(zip(list(range(2**N)), np.dot(state_space, powers_of_2))))
        
        attractors = []
        basin_sizes = []
        attractor_dict = dict()
        stg = dict()
        for iii_start in range(lcm):
            for xdec in starting_states:
                iii = iii_start
                queue = [xdec]
                while True:
                    fxdec = dictF_with_fixed_source[iii % n_initial_values][xdec]
                    stg.update({(int(boolforge.bin2dec(periodic_pattern_of_external_inputs[iii % n_initial_values])),int(xdec)):(int(boolforge.bin2dec(periodic_pattern_of_external_inputs[(iii + 1) % n_initial_values])),int(fxdec))})
                    iii += 1
                    try:
                        index_attr = attractor_dict[(iii % n_initial_values,fxdec)]
                        basin_sizes[index_attr] += 1
                        attractor_dict.update(list(zip(zip(np.arange(iii_start,len(queue)+iii_start)%n_initial_values,queue), [index_attr] * len(queue))))
                        break
                    except KeyError:
                        try: 
                            index = queue[-n_initial_values::-n_initial_values].index(fxdec)
                            dummy = np.arange(iii_start,len(queue)+iii_start)%n_initial_values
                            #print(iii_start,j,list(zip(dummy[-n_initial_values*(index+1):],queue[-n_initial_values*(index+1):])))
                            attractor_dict.update(list(zip(zip(dummy,queue), [len(attractors)] * len(queue))))
                            attractors.append(list(zip(dummy[-n_initial_values*(index+1):],queue[-n_initial_values*(index+1):])))
                            basin_sizes.append(1)
                            break
                        except ValueError:
                            pass
                    queue.append(fxdec)
                    xdec = fxdec
        attrs = []
        attr_dict = dict()
        for key in attractor_dict.keys():
            attr_dict[(int(boolforge.bin2dec(periodic_pattern_of_external_inputs[key[0]])), int(key[1]))] = int(attractor_dict[key])
        for attr in attractors:
            formatted_attr = []
            for state in attr:
                formatted_attr.append((int(boolforge.bin2dec(periodic_pattern_of_external_inputs[state[0]])), int(state[1])))
            attrs.append(formatted_attr)
            
            
        return (attrs, len(attractors), basin_sizes, attr_dict, state_space, stg)
    

