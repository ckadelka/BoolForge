#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: benco
"""

import math
import numpy as np
import itertools
import boolforge

# TODO: Make human readable
# TODO: Product of Trajectories
# TODO: Product of STG

# Example 2.8
F_8 = [[0,0,0,1],[0,1]]
I_8 = [[1,2],[0]]
np_8 = [[1]]
p_8 = [[1,0]]

# Example 2.9
F_9 = [[0,0,0,1],[1,0]]
I_9 = [[1,2],[0]]
np_9 = [[1]]
p_9 = [[1,0]]

# Example 2.10
F_10 = [[0,0,0,1],[1,0]]
I_10 = [[1,2],[0]]
np_10 = []
p_10 = [[0]]

# 2.8
# attr: [[(1, 0), (0, 0)], [(1, 1), (0, 2)]]
# stg: {(1, 0): (0, 0),
#       (0, 0): (1, 0),
#       (1, 1): (0, 2),
#       (0, 2): (1, 1),
#       (1, 2): (0, 1),
#       (0, 1): (1, 0),
#       (1, 3): (0, 3),
#       (0, 3): (1, 1)}

# 2.9
# attr: [[(1, 0), (0, 1), (1, 1), (0, 3)]]
# stg: {(1, 0): (0, 1),
#       (0, 1): (1, 1),
#       (1, 1): (0, 3),
#       (0, 3): (1, 0),
#       (1, 2): (0, 0),
#       (0, 0): (1, 1),
#       (1, 3): (0, 2),
#       (0, 2): (1, 0)}

# 2.8 x 2.9
# attr: [[(3, 0), (0, 1), (3, 1), (0, 3)],
#       [(3, 4), (0, 9), (3, 5), (0, 11)]]

def get_product_of_non_autonomous_attractors(attrs_1, attrs_2, merge_bits):
    attractors = []
    for attr1 in attrs_1:
        attractor = []
        for attr2 in attrs_2:
            m = len(attr1)
            n = len(attr2)
            for i in range(math.lcm(*[m, n])):
                attractor.append(_merge_state_representation(attr1[i % m], attr2[i % n], merge_bits))
        attractors.append(attractor)
    return attractors

def get_dynamics_product_of_two_non_autonomous_modules_slow(F1, I1, npseq1, pseq1, F2, I2, npseq2, pseq2):
    attrs1, _, _, _, _, stg1, istate1 = get_attractors_synchronous_exact_non_autonomous(F1, I1, npseq1, pseq1)
    attrs2, _, _, _, _, stg2, istate2 = get_attractors_synchronous_exact_non_autonomous(F2, I2, npseq2, pseq2)
    
    if len(npseq1) > 0:
        prefix1 = boolforge.bin2dec([ npseq1[j][0] for j in range(len(npseq1)) ])
    else:
        prefix1 = boolforge.bin2dec([ pseq1[j][0] for j in range(len(pseq1)) ])
    
    if len(npseq2) > 0:
        prefix2 = boolforge.bin2dec([ npseq2[j][0] for j in range(len(npseq2)) ])
    else:
        prefix2 = boolforge.bin2dec([ pseq2[j][0] for j in range(len(pseq2)) ])
    
    initial_states = []
    for state1 in istate1:
        for state2 in istate2:
            initial_states.append(((prefix1, state1), (prefix2, state2)))
    
    attractors = []
    basin_sizes = []
    attractor_dict = dict()
    stg = dict()
    for xdec in initial_states:
        queue = [xdec]
        while True:
            fxdec_1 = stg1[(xdec[0][0],xdec[0][1])]
            fxdec_2 = stg2[(xdec[1][0],xdec[1][1])]
            fxdec = (fxdec_1, fxdec_2)
            stg.update({xdec:fxdec})
            try:
                index_attr = attractor_dict[fxdec]
                basin_sizes[index_attr] += 1
                attractor_dict.update(list(zip(queue, [index_attr] * len(queue))))
                break
            except KeyError:
                try:
                    index = queue.index(fxdec)
                    attractor_dict.update(list(zip(queue, [len(attractors)] * len(queue))))
                    attractors.append(queue[index:])
                    basin_sizes.append(1)
                    break
                except ValueError:
                    pass
            queue.append(fxdec)
            xdec = fxdec
    return attractors, len(attractors), basin_sizes, attractor_dict, stg

def calculate_merge_bits(periodic_sequence, F):
    return len(periodic_sequence), len(F)

def _merge_state_representation(x, y, ybits):
    return ((x[0] << ybits[0] | y[0]), (x[1] << ybits[1]) | y[1])

def get_attractors_synchronous_exact_non_autonomous(F, I, non_periodic_seq, periodic_seq):
    if len(non_periodic_seq) > 0:
        initial_states = set()
        n_var = len(F)
        n_const = len(non_periodic_seq)
        
        max_len_pattern = max(list(zip(map(len,non_periodic_seq))))[0]
        for i,sequence in enumerate(non_periodic_seq):
            if len(sequence) < max_len_pattern:
                for j in range(max_len_pattern - len(sequence)):
                    val = periodic_seq[i][0]
                    sequence.append(val)
                    periodic_seq[i].pop(0)
                    periodic_seq[i].append(val)
                non_periodic_seq[i] = sequence
        
        for i in range(2 ** n_var):
            fxvec = boolforge.dec2bin(i, n_var)
            for iii in range(max_len_pattern):
                values = [ non_periodic_seq[j][iii] for j in range(n_const) ]
                F2, I2 = get_BN_with_fixed_source_nodes(F, I, n_var, n_const, values)
                fxvec = boolforge.BooleanNetwork(F2, I2).update_network_synchronously(fxvec)
            initial_states.add(boolforge.bin2dec(fxvec))
        initial_states = list(initial_states)
    else:
        initial_states = list(range(2**len(F)))
    
    attr_computation = get_attractors_synchronous_exact_with_external_inputs(F, I, periodic_seq, initial_states)
    return attr_computation[0], attr_computation[1], attr_computation[2], attr_computation[3], attr_computation[4], attr_computation[5], initial_states

def get_BN_with_fixed_source_nodes(F,I,n_variables,n_source_nodes,values_source_nodes):
    #NOTE: F, I must be arranged so that the source nodes appear last
    
    assert len(F) == len(I)
    F_new = [np.array(el) for el in F[:n_variables]]
    I_new = [np.array(el) for el in I[:n_variables]]
    
    for source_node,value in zip(list(range(n_variables,n_variables+n_source_nodes)),values_source_nodes):
        for i in range(n_variables):
            try:
                index = list(I[i]).index(source_node) #check if the constant is part of regulators
            except ValueError:
                continue
            truth_table = np.array(list(map(np.array, list(itertools.product([0, 1], repeat=len(I_new[i]))))))
            indices_to_keep = np.where(truth_table[:,index]==value)[0]
            F_new[i] = F_new[i][indices_to_keep]
            I_new[i] = I_new[i][~np.isin(I_new[i], source_node)]
    return F_new,I_new

def get_attractors_synchronous_exact_with_external_inputs(F, I, input_patterns = [], starting_states = None):
    if starting_states is None:
        starting_states = list(range(2**len(F)))
    
    n_var = len(F)
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
        dummy = get_BN_with_fixed_source_nodes(F,I,n_var,n_const,input_values)
        Fs_with_fixed_sources.append(dummy[0])
        Is_with_fixed_sources.append(dummy[1])
        degrees_with_fixed_sources.append(list(map(len, dummy[1])))
    
    N = len(F)
    
    left_side_of_truth_table = boolforge.get_left_side_of_truth_table(N)
    
    powers_of_2 = np.array([2**i for i in range(N)])[::-1]
    
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