#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 11:08:44 2025

@author: Benjamin Coberly, Claus Kadelka
"""

import itertools
import math
from collections import defaultdict
from copy import deepcopy

import numpy as np
import networkx as nx

from typing import Union, Optional

try:
    import boolforge.utils as utils
    from boolforge.boolean_function import BooleanFunction
except ModuleNotFoundError:
    import utils as utils
    from boolean_function import BooleanFunction
    
try:
    import cana.boolean_network
    __LOADED_CANA__=True
except ModuleNotFoundError:
    print('The module cana cannot be found. Ensure it is installed to use all functionality of this toolbox.')
    __LOADED_CANA__=False

class WiringDiagram(object):
    """
    A class representing a Wiring Diagram
    
    **Constructor Parameters:**

        - I (list[list[int]] | np.ndarray[list[int]]): A list of N lists
          representing the regulators (or inputs) for each Boolean function.

        - variables (list[str] | np.array[str], optional): A list of N strings
          representing the names of each variable, default = None.
          
        - weights (list[list[int | np.nan]]): #TODO

    **Members:**
        
        - I (list[np.array[int]]): As passed by the constructor.
        - variables (np.array[str]): As passed by the constructor.
        - N_variables (int): The number of variables in the Boolean network.
        - N_constants (int): The number of constants in the Boolean network.
        - N (int): The number of variables and constants in the Boolean network.
        - indegrees (list[int]): The indegrees for each node.
        - outdegrees (list[int]): The outdegrees of each node.
        - weights (list[list[int | np.nan]]): As passed by the constructor.
    """
    
    def __init__(self, I : Union[list, np.ndarray],
                 variables : Union[list, np.array, None] = None, weights = None):
        assert isinstance(I, (list, np.ndarray)), "I must be an array"
        #assert (len(I[i]) == ns[i] for i in range(len(ns))), "Malformed wiring diagram I"
        assert variables is None or len(I)==len(variables), "len(I)==len(variables) required if variable names are provided"
        assert weights is None or True, "weights assertion" # TODO: if weights are given, they must be valid
        
        self.I = [np.array(regulators,dtype=int) for regulators in I]
        self.N = len(I)
        self.indegrees = list(map(len, self.I))
        
        if variables is None:
            variables = ['x'+str(i) for i in range(self.N)]
        
        self.N_constants = len(self.get_constants(False))
        self.N_variables = self.N - self.N_constants
        
        # if self.N_constants > 0:
        #     constants_dict = self.get_constants()
        #     remap = ([], [])
        #     for node in constants_dict.keys():
        #         if constants_dict[node]:
        #             remap[1].append(node)
        #         else:
        #             remap[0].append(node)
        #     self.__CRD__ = dict(zip(range(self.N), remap[0] + remap[1]))
        #     self.I = [ self.I[self.__CRD__[i]] for i in range(len(self.I)) ]
        #     self.indegrees = list(map(len, self.I)) #could also instead remap, both fast
        #     variables = np.array([ variables[self.__CRD__[i]] for i in range(len(variables)) ])
        
        self.variables = np.array(variables)
        
        self.outdegrees = self.get_outdegrees()
        self.weights = weights

    def __getitem__(self, index):
        return self.I[index]

    def get_outdegrees(self) -> np.array:
        """
        Returns the outdegree of each node.
        
        **Returns:**
            
            - np.array[int]: Outdegree of each node.
        """
        outdegrees = np.zeros(self.N, int)
        for regulators in self.I:
            for regulator in regulators:
                outdegrees[regulator] += 1
        return outdegrees


    def get_constants(self, AS_DICT : bool = True) -> Union[dict, np.array]:
        """
        Identify constants in a Boolean network.
        
        A node is considered a constant if it has no regulators.
        
        **Parameters:**
        
            - AS_DICT (bool, optional): Whether to return the indices of constants
              as a dictionary or array. If true, returns as a dictionary. Defaults
              to True.
        
        **Returns:**
        
            If AS_DICT is True:
                
                - dict[int:bool]: Dictionary determining if an index is a
                  constant or not.
                  
            else:
                - np.array[int]: Array of node indices that are constants.
        """
        rlI = range(len(self.I))
        is_constant = [self.indegrees[i] == 0 for i in rlI]
        if AS_DICT:
            return dict(zip(rlI, is_constant))
        return np.where(is_constant)[0]


    def get_strongly_connected_components(self) -> list:
        """
        Determine the strongly connected components of a wiring diagram.

        **Returns:**
            
            - list[set[int]]: A list of sets, each representing a strongly
              connected component.
        """
        edges_wiring_diagram = []
        for target, regulators in enumerate(self.I):
            for regulator in regulators:
                edges_wiring_diagram.append((regulator, target))
        subG = nx.from_edgelist(edges_wiring_diagram, create_using=nx.MultiDiGraph())
        return [scc for scc in nx.strongly_connected_components(subG)]

    def get_modular_structure(self):
        """
        Determine the modular structure of a Boolean network.

        The modular structure is defined by a directed acyclic graph (DAG) whose
        nodes are the strongly connected components (SCCs) of the underlying wiring
        diagram and whose directed edges indicate a regulation from one SCC to another SCC.

        **Returns:**
            
            - set[tuple[int]]: A set of edges, describing a directed acyclic graph
              indicating the regulations between modules (i.e., strongly connected
              components of the underlying wiring diagram).
        """
        sccs = self.get_strongly_connected_components()
        scc_dict = {}
        for j,s in enumerate(sccs):
            for el in s:
                scc_dict.update({el:j})
        dag = set()
        for target,regulators in enumerate(self.I):
            for regulator in regulators:
                edge = (scc_dict[regulator],scc_dict[target])
                if edge[0]!=edge[1] and (self.weights is None or not np.isnan(self.weights[target][list(self.weights[target]).index(regulator)])):
                    dag.add(edge)   
        return dag


    # def get_signed_effective_graph(self, type_of_each_regulation : list,
    #     constants : list = [], IGNORE_SELFLOOPS : bool = False,
    #     IGNORE_CONSTANTS : bool = True) -> np.array:
    #     """
    #     Construct the signed effective graph of a Boolean network.

    #     This function computes an effective graph in which each edge is
    #     weighted by its effectiveness. Effectiveness is obtained via
    #     get_edge_effectiveness on the corresponding Boolean function. Edges
    #     are signed according to the type of regulation ('increasing' or
    #     'decreasing').

    #     **Parameters:**
            
    #         - type_of_each_regulation (list[str]): List of lists specifying
    #           the type of regulation for each edge.
              
    #         - constants (list[int], optional): List of constant nodes.
    #         - IGNORE_SELFLOOPS (bool, optional): If True, self-loops are ignored.
    #         - IGNORE_CONSTANTS (bool, optional): If True, constant nodes
    #           are excluded.

    #     **Returns:**
            
    #         - np.array[float]: The signed effective graph as a matrix of edge
    #           effectiveness values.
    #     """
    #     n = len(self.I)
    #     n_constants = len(constants)
    #     if IGNORE_CONSTANTS:
    #         m = np.zeros((n - n_constants, n - n_constants), dtype=float)
    #         for i, (regulators, type_of_regulation) in enumerate(zip(self.I, type_of_each_regulation)):
    #             effectivenesses = self.F[i].get_edge_effectiveness() #TODO: F does not exist here
    #             for j, t, e in zip(regulators, type_of_regulation, effectivenesses):
    #                 if j < n - n_constants and (not IGNORE_SELFLOOPS or i != j):
    #                     if t == 'increasing':
    #                         m[j, i] = e
    #                     elif t == 'decreasing':
    #                         m[j, i] = -e
    #                     else:
    #                         m[j, i] = np.nan
    #         return m
    #     else:
    #         return self.get_signed_effective_graph(type_of_each_regulation, [], IGNORE_CONSTANTS=True)


    def get_ffls(self) -> Union[tuple, list]:
        """
        Identify feed-forward loops (FFLs) in a Boolean network based solely
        on the wiring diagram.

        The function uses the inverted wiring diagram to identify common
        targets and returns the FFLs found. If types_I (the type of each
        regulation) is provided, it also returns the corresponding regulation
        types.

        **Parameters:**
            
            - types_I (list[list[str]], optional): List of lists specifying
              the type (e.g., 'increasing' or 'decreasing') for each regulation.

        **Returns:**
            
            If self.weights is not None:
                
                - tuple[list[int], list[int]]: (ffls, types) where ffls is a
                  list of identified FFLs (each as a list [master regulator,
                  intermediate, target]), and types is a list of regulation type
                  triplets (master -> target, master -> intermediate,
                  intermediate -> target).
                
            Otherwise:
                
                - list[list[int]]: A list of identified FFLs.
        """
        I_inv = [[] for _ in self.N]
        for target, regulators in enumerate(self.I):
            for regulator in regulators:
                I_inv[regulator].append(target)
        ffls = []
        types = []
        for i in range(self.N):  # master regulators
            for j in I_inv[i]:
                if i == j:
                    continue
                common_targets = list(set(I_inv[i]) & set(I_inv[j]))
                for k in common_targets:
                    if j == k or i == k:
                        continue
                    ffls.append([i, j, k])
                    if self.weights is not None:
                        direct = self.weights[k][self.I[k].index(i)]
                        indirect1 = self.weights[j][self.I[j].index(i)]
                        indirect2 = self.weights[k][self.I[k].index(j)]
                        types.append([direct, indirect1, indirect2])
        if self.weights is not None:
            return (ffls, types)
        else:
            return ffls


    def generate_networkx_graph(self) -> nx.DiGraph:
        """
        Generate a NetworkX directed graph from a wiring diagram.

        Nodes are labeled with variable names (from variables) and constant
        names (from constants). Edges are added from each regulator to its
        target based on the wiring diagram I.

        **Parameters:**
            
            - constants (list[str]): List of constant names.
            - variables (list[str]): List of variable names.

        **Returns:**
            
            - networkx.DiGraph: The wiring diagram as directed graph.
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.variables)
        G.add_edges_from([(self.variables[self.I[i][j]], self.variables[i]) for i in range(self.N) for j in range(self.indegrees[i])])
        return G


    def generate_networkx_graph_from_edges(self, n_variables : int) -> nx.DiGraph:
        """
        Generate a NetworkX directed graph from an edge list derived from the
        wiring diagram.

        Only edges among the first n_variables (excluding constant self-loops)
        are included.

        **Parameters:**
            
            - n_variables (int): Number of variable nodes (constants are
              excluded).

        **Returns:**
            
            - networkx.DiGraph: The generated directed graph.
        """
        edges = []
        for j, regulators in enumerate(self.I):
            if j >= n_variables:  # Exclude constant self-loops
                break
            for i in regulators:
                edges.append((i, j))
        return nx.DiGraph(edges)

    def get_type_of_loop(self, loop : list) -> list:
        """
        Determine the regulation types along a feedback loop.

        For a given loop (a list of node indices), this function returns a
        list containing the type (e.g., 'increasing' or 'decreasing') of each
        regulation along the loop. The loop is assumed to be ordered such that
        the first node is repeated at the end.

        **Parameters:**
            
            - loop (list[int]): List of node indices representing the loop.

        **Returns:**
            
            - list[int]: A list of regulation types corresponding to each edge
              in the loop.
        """
        n = len(loop)
        dummy = loop[:]
        dummy.append(loop[0])
        res = []
        for i in range(n):
            # Assumes is_monotonic returns a tuple with the monotonicity information.
            #TODO: F does not exist here
            res.append(self.F[dummy[i+1]].is_monotonic(True)[1][list(self.I[dummy[i+1]]).index(dummy[i])])
        return res


class BooleanNetwork(WiringDiagram):
    """
    A class representing a Boolean network with N variables.
    
    **Constructor Parameters:**

        - F (list[BooleanFunction | list[int]] | np.ndarray[BooleanFunction |
          list[int]]): A list of N Boolean functions, or of N lists of length
          2^n representing the outputs of a Boolean function with n inputs.

        - I (list[list[int]] | np.ndarray[list[int]]): A list of N lists
          representing the regulators (or inputs) for each Boolean function.

        - variables (list[str] | np.array[str], optional): A list of N strings
          representing the names of each variable, default = None.
          
        - SIMPLIFY_FUNCTIONS (bool, optional): Constructs this Boolean Network
          to only include its essential components. Defaults to False
          
    **Members:**
        
        - F (list[BooleanFunction]): As passed by the constructor.
        - I (list[np.array[int]]): As passed by the constructor.
        - variables (np.array[str]): As passed by the constructor.
        - N (int): The number of variables in the Boolean network.
        - N_constants (int): The number of constants in the Boolean network.
        - size (int): The number of variables and constants in the Boolean network.
        - indegrees (list[int]): The indegrees for each node.
        - outdegrees (list[int]): The outdegrees of each node.
        - STG (dict): The state transition graph.
        - weights (np.array[float] | None): Inherited from WiringDiagram. Default None.
    """

    def __init__(self, F : Union[list, np.ndarray], I : Union[list, np.ndarray, WiringDiagram],
                 variables : Union[list, np.array, None] = None,
                 SIMPLIFY_FUNCTIONS : Optional[bool] = False):
        assert isinstance(F, (list, np.ndarray)), "F must be an array or list."
        assert isinstance(I, (list, np.ndarray, WiringDiagram)), "I must be an array or list, or an instance of WiringDiagram."
        if isinstance(I, (list, np.ndarray)):
            super().__init__(I, variables)
        else:
            if variables is not None:
                print('Warning: Values of provided variables ignored. Variales of WiringDiagram I used instead.')
            super().__init__(I.I, I.variables)
        assert len(F)==self.N, "len(F)==len(I) required"
        
        self.F = []
        for ii,f in enumerate(F):
            if isinstance(f, (list, np.ndarray, str)):
                self.F.append(BooleanFunction(f,name = self.variables[ii]))
            elif isinstance(f, BooleanFunction):
                f.name = self.variables[ii]
                self.F.append(f)
            else:
                raise TypeError(f"F holds invalid data type {type(f)} : Expected either list, np.array, or BooleanFunction")
        if not hasattr(self, 'constants'): #keeps track of all constants and nodes set to constants
            self.constants = {}
        if self.N_constants > 0:
            self.remove_constants()
        self.STG = None
        if SIMPLIFY_FUNCTIONS:
            self.simplify_functions() 

    def remove_constants(self,values_constants=None):
        if values_constants is None:
            indices_constants = self.get_constants(AS_DICT=False)
            dict_constants = self.get_constants(AS_DICT=True)
            values_constants = [self.F[c][0] for c in indices_constants]
        else:
            indices_constants = self.get_source_nodes(AS_DICT=False) 
            dict_constants = self.get_source_nodes(AS_DICT=True)
            assert len(values_constants)==len(indices_constants),'The network contains {len(indices_constants)} source nodes but {len(values_constants)} values were provided.'
        #self.constants = dict(zip(self.variables[indices_constants],values_constants))
        for id_constant,value in zip(indices_constants,values_constants):
            regulated_nodes = []
            for i in range(self.N): # for all variables
                if dict_constants[i]:
                    continue
                try:
                    index = list(self.I[i]).index(id_constant) #check if the constant is part of regulators
                except ValueError:
                    continue
                truth_table = utils.get_left_side_of_truth_table(self.indegrees[i])
                indices_to_keep = np.where(truth_table[:,index]==value)[0]
                self.F[i].f = self.F[i].f[indices_to_keep]
                if self.weights is not None:
                    self.weights[i] = self.weights[i][self.I[i]!=id_constant]
                self.I[i] = self.I[i][self.I[i]!=id_constant]
                self.indegrees[i] -= 1
                self.F[i].n -= 1
                regulated_nodes.append(self.variables[i])
            self.constants[self.variables[id_constant]] = {'value' : value, 'regulatedNodes': regulated_nodes}
                
        for i in range(self.N):
            if dict_constants[i]:
                continue
            if self.indegrees[i] == 0:
                self.indegrees[i] = 1
                self.F[i].n = 1
                self.F[i].f = np.array([self.F[i][0],self.F[i][0]],dtype=int)
                self.I[i] = np.array([i],dtype=int)
                if self.weights is not None:
                    self.weights[i] = np.array([np.nan],dtype=int)
        self.F = [self.F[i] for i in range(self.N) if dict_constants[i]==False]
        adjustment_for_I = np.cumsum([dict_constants[i] for i in range(self.N)])
        self.I = [self.I[i]-adjustment_for_I[self.I[i]] for i in range(self.N) if dict_constants[i]==False]
        if self.weights is not None:
            self.weights = [self.weights[i] for i in range(self.N) if dict_constants[i]==False]
        self.variables = [self.variables[i] for i in range(self.N) if dict_constants[i]==False]
        self.outdegrees = [self.outdegrees[i] for i in range(self.N) if dict_constants[i]==False]
        self.indegrees = [self.indegrees[i] for i in range(self.N) if dict_constants[i]==False]
        self.N -= len(indices_constants)
        self.N_constants = 0

    @classmethod
    def from_cana(cls, cana_BooleanNetwork : "cana.boolean_network.BooleanNetwork") -> "BooleanNetwork":
        """
        **Compatability Method:**
        
            Converts an instance of cana.boolean_network.BooleanNetwork from
            the cana module into a Boolforge BooleanNetwork object.
        
        **Returns**:
            
                - A BooleanNetwork object.
        """
        F = []
        I = []
        variables = []
        for entry in cana_BooleanNetwork.logic.values():
            try:
                variables.append(entry['name'])
            except KeyError:
                pass
            try:
                F.append(entry['out'])
                I.append(entry['in'])
            except KeyError:
                pass            
        return cls(F = F, I = I, variables=variables)

    @classmethod
    def from_string(cls, network_string : str, separator : Union[str, list, np.array] = ',',
        max_degree : int = 24, original_not : Union[str, list, np.array] = 'NOT',
        original_and : Union[str, list, np.array] = 'AND',
        original_or : Union[str, list, np.array] = 'OR') -> "BooleanNetwork":
        """
        **Compatability Method:**
        
            Converts a string into a Boolforge BooleanNetwork object.
        
        **Returns**:
            
                - A BooleanNetwork object.
        """
        sepstr, andop, orop, notop = "@", "∧", "∨", "¬"
        
        get_dummy_var = lambda i: "x%sy"%str(int(i))
        
        # reformat network string
        lines = network_string.replace('\t', ' ',).replace('(', ' ( ').replace(')', ' ) ')
        def __replace__(string, original, replacement):
            if isinstance(original, (list, np.ndarray)):
                for s in original:
                    string = string.replace(s, " %s "%replacement)
            elif isinstance(original, str):
                string = string.replace(original, " %s "%replacement)
            return string
        lines = __replace__(lines, separator, sepstr)
        lines = __replace__(lines, original_not, notop)
        lines = __replace__(lines, original_and, andop)
        lines = __replace__(lines, original_or, orop)
        
        lines = lines.splitlines()
        
        # remove empty lines
        while '' in lines:
            lines.remove('')
        
        # remove comments
        for i in range(len(lines)-1, -1, -1):
            if lines[i][0] == '#':
                lines.pop(i)
        
        n = len(lines)
        
        # find variables and constants
        var = ["" for i in range(n)]
        for i in range(n):
            var[i] = lines[i].split(sepstr)[0].replace(' ', '')
        consts_and_vars = []
        for line in lines:
            words = line.split(' ')
            for word in words:
                if word not in ['(', ')', sepstr, andop, orop, notop, ''] and not utils.is_float(word):
                    consts_and_vars.append(word)
        consts = list(set(consts_and_vars)-set(var))
        dict_var_const = dict(list(zip(var, [get_dummy_var(i) for i in range(len(var))])))
        dict_var_const.update(dict(list(zip(consts, [get_dummy_var(i+len(var)) for i in range(len(consts))]))))
        
        # replace all variables and constants with dummy names
        for i, line in enumerate(lines):
            words = line.split(' ')
            for j, word in enumerate(words):
                if word not in ['(', ')', sepstr, andop, orop, notop, ''] and not utils.is_float(word):
                    words[j] = dict_var_const[word]
            lines[i] = ' '.join(words)
        
        # update line to only be function
        for i in range(n):
            lines[i] = lines[i].split(sepstr)[1]
        
        # generate wiring diagram I
        I = []
        for i in range(n):
            try:
                idcs_open = utils.find_all_indices(lines[i], 'x')
                idcs_end = utils.find_all_indices(lines[i], 'y')
                regs = np.sort(np.array(list(map(int,list(set([lines[i][(begin+1):end] for begin,end in zip(idcs_open,idcs_end)]))))))
                I.append(regs)
            except ValueError:
                I.append(np.array([], int))
        
        deg = list(map(len, I))
        
        # generate functions F
        F = []
        for i in range(n):
            if deg[i] == 0:
                f = np.array([int(lines[i])], int)
            elif deg[i] <= max_degree:
                tt = utils.get_left_side_of_truth_table(deg[i])
                ldict = { get_dummy_var(I[i][j]) : tt[:, j].astype(bool) for j in range(deg[i]) }
                f = eval(lines[i].replace(andop, '&').replace(orop, '|').replace(notop, '~').replace(' ', ''), {"__builtins__" : None}, ldict)
            else:
                f = np.array([], int)
            F.append(f.astype(int))
        for i in range(len(consts)):
            F.append(np.array([0, 1], int))
            I.append(np.array([len(var) + i]))
        
        return cls(F, I, var+consts)


    def to_cana(self) -> "cana.boolean_network.BooleanNetwork":
        """
        **Compatability method:**
        
            Returns an instance of the class cana.BooleanNetwork from the
            cana module.

        **Returns:**
            
            - An instance of cana.boolean_network.BooleanNetwork
        """
        logic_dicts = []
        for bf,regulators,var in zip(self.F,self.I,self.variables):
            logic_dicts.append({'name':var, 'in': list(regulators), 'out': list(bf.f)})
        return cana.boolean_network.BooleanNetwork(Nnodes = self.N, logic = dict(zip(range(self.N),logic_dicts))) 

    def to_bnet(self, separator=',\t', AS_POLYNOMIAL : bool = True) -> str:
        """
        **Compatability method:**
            
            Returns a bnet string formatted as a polynomial.
        
        **Parameters:**

            - separator (str): A string used to separate the target variable
              from the function. Defaults to ',\t'.
              
            - AS_POLYNOMIAL (bool, optional): Determines whether to return
              the function as a polynomial or logical expression. If true,
              returns as a polynomial, and if false, returns as a logical
              expression. Defaults to true.
            
        **Returns:**
            
            - str: A string describing a bnet.
        """
        lines = []
        constants_indices = self.get_constants()
        for i in range(self.N):
            if constants_indices[i]:
                function = str(self.F[i].f[0])
            elif AS_POLYNOMIAL:
                function = utils.bool_to_poly(self.F[i], self.variables[self.I[i]])
            else:
                function = self.F[i].to_expression(" & ", " | ")
            lines.append(f'{self.variables[i]}{separator}{function}')
        return '\n'.join(lines)

    
    def __len__(self):
        return self.N
    
    
    def __str__(self):
        return f"Boolean network of {self.N} nodes with indegrees {self.indegrees}"
    
    
    def __getitem__(self, index):
        #return (self.F[index],self.I[index],self.variables[index])
        return self.F[index]
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
        
    
    def get_types_of_regulation(self):
        weights = []
        dict_weights = {'non-essential' : np.nan, 'conditional' : 0, 'positive' : 1, 'negative' : -1}
        for bf in self.F:
            weights.append(np.array([dict_weights[el] for el in bf.get_type_of_inputs()]))
        self.weights = weights
        return weights

    ## Transform Boolean networks
    def simplify_functions(self) -> None:
        """
        Remove all non-essential inputs, i.e., inoperative edges from the Boolean network.

        For each node in a Boolean network, represented by its Boolean function
        and its regulators, this function extracts the “essential” part of the
        function by removing non-essential regulators. The resulting network
        contains, for each node, a reduced truth table (with only the essential
        inputs) and a corresponding list of essential regulators.

        **Returns:**
            
            - BooleanNetwork: A Boolean network object where:
                
                - F is a list of N Boolean functions containing functions of
                  length 2^(m_i), with m_i ≤ n_i, representing the functions
                  restricted to the essential regulators.
                  
                - I is a list of N lists containing the indices of the
                  essential regulators for each node.
        """
        self.get_types_of_regulation() #ensuring that self.weights is updated
        for i in range(self.N):
            regulator_is_non_essential = np.isnan(self.weights[i])
            if sum(regulator_is_non_essential)==0: #all variables are essential, nothing to change
                continue
            
            non_essential_variables = np.where(regulator_is_non_essential)[0]
            essential_variables = np.where(~regulator_is_non_essential)[0]
            self.outdegrees[non_essential_variables] -= 1
            if len(essential_variables)==0: #no variables are essential, introduce ``fake" auto-regulation to keep this variable and do not delete it as a constant
                self.indegrees[i] = 1
                self.F[i].f = np.array([self.F[i][0],self.F[i][0]],dtype=int)
                self.F[i].n = 1
                self.F[i].variables = self.variables[i]
                self.I[i] = np.array([i],dtype=int)
                self.weights[i] = np.array([np.nan],dtype=float)
                self.outdegrees[i] += 1 #add this, even though it's a fake regulation to keep sum(self.outdegrees)==sum(self.indegrees)
                continue
            
            left_side_of_truth_table = utils.get_left_side_of_truth_table(self.indegrees[i])
            self.F[i].f = self.F[i][np.sum(left_side_of_truth_table[:, non_essential_variables], 1) == 0]
            self.F[i].n = len(essential_variables)
            self.F[i].variables = self.F[i].variables[~regulator_is_non_essential]
            self.I[i] = self.I[i][essential_variables]
            self.weights[i] = self.weights[i][essential_variables]
            self.indegrees[i] = len(essential_variables)


    def get_source_nodes(self, AS_DICT : bool = True) -> Union[dict, np.array]:
        """
        Identify source nodes in a Boolean network.
        
        A node is considered a source node if it does not change over time. It has
        exactly one regulator and that regulator is the node itself.        
        
        **Parameters:**
        
            - AS_DICT (bool, optional): Whether to return the indices of source nodes
              as a dictionary or array. If true, returns as a dictionary. Defaults
              to True.
        
        **Returns:**
        
            If AS_DICT is True:
                
                - dict[int:bool]: Dictionary determining if an index is a
                  source nodes or not.
                  
            else:
                - np.array[int]: Array of all indices of source nodes.
        """

        rlI = range(self.N)
        is_source_node = [self.indegrees[i] == 1 and self.I[i][0] == i and self.F[i][0]==0 and self.F[i][1]==1 for i in rlI]
        if AS_DICT:
            return dict(zip(rlI, is_source_node))
        return np.where(is_source_node)[0]

    
    def get_network_with_fixed_source_nodes(self,values_source_nodes : Union[list, np.array]) -> "BooleanNetwork":
        indices_source_nodes = self.get_source_nodes(AS_DICT=False)
        assert len(values_source_nodes)==len(indices_source_nodes),f"The length of 'values_source_nodes', which is {len(values_source_nodes)}, must equal the number of source nodes, which is {len(indices_source_nodes)}."
        assert set(values_source_nodes) in set([0,1]),"Controlled node values must be 0 or 1."
        F = deepcopy(self.F)
        I = deepcopy(self.I)
        for source_node,value in zip(indices_source_nodes,values_source_nodes):
            F[source_node].f = [value]
            I[source_node] = []
        bn = self.__class__(F, I, self.variables)
        bn.constants.update(self.constants)
        return bn

    def get_network_with_node_controls(self,indices_controlled_nodes : Union[list, np.array], 
                                       values_controlled_nodes : Union[list, np.array],
                                       KEEP_CONTROLLED_NODES : bool = False) -> "BooleanNetwork":
        assert len(values_controlled_nodes)==len(indices_controlled_nodes),f"The length of 'values_controlled_nodes', which is {len(values_controlled_nodes)}, must equal the length of 'indices_controlled_nodes', which is {len(indices_controlled_nodes)}."
        assert set(values_controlled_nodes) in set([0,1]),"Controlled node values must be 0 or 1."
        F = deepcopy(self.F)
        I = deepcopy(self.I)
        for node,value in zip(indices_controlled_nodes,values_controlled_nodes):
            if KEEP_CONTROLLED_NODES:
                F[node].f = [value,value]
                I[node] = [node]        
            else:
                F[node].f = [value]
                I[node] = []
        bn = self.__class__(F, I, self.variables)
        if not KEEP_CONTROLLED_NODES:
            bn.constants.update(self.constants)
        return bn


    def get_network_with_edge_controls(self, 
                                      control_targets : Union[int,list,np.array],
                                      control_sources : Union[int,list,np.array], 
                                      type_of_edge_controls : Union[int,list,np.array,None] = None) -> "BooleanNetwork":
        """
        Generate a perturbed Boolean network by removing the influence of
        specified regulators on specified targets.

        The function modifies the Boolean function for target nodes by
        restricting it to those entries in its truth table where the input
        from given regulators equals the specified type_of_control. The
        regulators are then removed from the wiring diagram for that node.

        **Parameters:**
            
            - control_targets (int | list[int] | np.array[int]): 
                Index of the target node(s) to be perturbed.
                
            - control_sources (int | list[int] | np.array[int]): 
                Index of the regulator(s) whose influence is to be fixed.
              
            - type_of_edge_controls (int | list[int] | np.array[int]) | None): 
                Source value in regulation of target after control. 
                Default is None (which is interpreted as 0).

        **Returns:**
            
            - BooleanNetwork object where:
                
                - F is the updated list of Boolean functions after perturbation.
                - I is the updated wiring diagram after removing the control
                  regulator from the target node.
        """

        # Normalize arguments to lists
        if np.isscalar(control_targets):
            control_targets = [control_targets]
            control_sources = [control_sources]
            type_of_edge_controls = [0 if type_of_edge_controls is None else type_of_edge_controls]
        elif type_of_edge_controls is None:
            type_of_edge_controls = [0] * len(control_targets)
    
        assert len(control_targets) == len(control_sources) == len(type_of_edge_controls), \
            "control_targets, control_sources, and type_of_edge_controls must have equal length."

        F_new = deepcopy(self.F)
        I_new = deepcopy(self.I)
        indegrees = np.copy(self.indegrees)

        for target, source, fixed_value in zip(control_targets, control_sources, type_of_edge_controls):
            assert fixed_value in [0, 1], f"type_of_edge_control must be 0 or 1 (got {fixed_value})."
            assert source in I_new[target], f"control_source={source} not in regulators of target={target}"
            idx_reg = list(I_new[target]).index(source)
            n_inputs = indegrees[target]
    
            # Compute bitmask indices efficiently
            indices = np.arange(2 ** n_inputs, dtype=np.uint32)
            mask = ((indices >> (n_inputs - 1 - idx_reg)) & 1) == fixed_value
            F_new[target] = F_new[target][mask]
    
            # Remove the regulator
            I_new[target] = np.delete(I_new[target], idx_reg)
            indegrees[target] -= 1
        
        return self.__class__(F_new, I_new, self.variables)

            
    
    def update_single_node(self, index : int,
        states_regulators : Union[list, np.array]) -> int:
        """
        Update the state of a single node.

        The new state is obtained by applying the Boolean function f to the
        states of its regulators. The regulator states are converted to a
        decimal index using utils.bin2dec.

        **Parameters:**
            
            - index (int): The index of the Boolean Function in F.
            - states_regulators (list[int] | np.array[int]): Binary vector
              representing the states of the node's regulators.

        **Returns:**
            
            - int: Updated state of the node (0 or 1).
        """
        return self.F[index].f[utils.bin2dec(states_regulators)].item()


    def update_network_synchronously(self, X : Union[list, np.array]) -> np.array:
        """
        Perform a synchronous update of a Boolean network.

        Each node's new state is determined by applying its Boolean function
        to the current states of its regulators.

        **Parameters:**
            
            - X (list[int] | np.array[int]): Current state vector of the network.

        **Returns:**
            
            - np.array[int]: New state vector after the update.
        """
        if type(X)==list:
            X = np.array(X)
        Fx = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            Fx[i] = self.update_single_node(index = i, states_regulators = X[self.I[i]])
        return Fx


    def update_network_synchronously_many_times(self, X : Union[list, np.array],
        n_steps : int) -> np.array:
        """
        Update the state of a Boolean network sychronously multiple time steps.

        Starting from the initial state, the network is updated synchronously
        n_steps times using the update_network_synchronously function.

        **Parameters:**
            
            - X (list[int] | np.array[int]): Initial state vector of the network.
            - n_steps (int): Number of update iterations to perform.

        **Returns:**
            
            - np.array[int]: Final state vector after n_steps updates.
        """
        for i in range(n_steps):
            X = self.update_network_synchronously(X)
        return X


    def update_network_SDDS(self, X : Union[list, np.array], P : np.array,
        *, rng=None) -> np.array:
        """
        Perform a stochastic update (SDDS) on a Boolean network.

        For each node, the next state is computed as nextstep = F[i] evaluated
        on the current states of its regulators. If nextstep > X[i], the node
        is activated with probability P[i,0]; if nextstep < X[i], the node is
        degraded with probability P[i,1]. Otherwise, the state remains unchanged.

        **Parameters:**
            
            - X (list[int] | np.array[int]): Current state vector.
            - P (np.array[float]): A len(F)×2 array of probabilities; for each
              node i, P[i,0] is the activation probability, and P[i,1] is the
              degradation probability.
            
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - np.array[int]: Updated state vector after applying the
              stochastic update.
        """
        rng = utils._coerce_rng(rng)
        if type(X)==list:
            X = np.array(X)
        Fx = X.copy()
        for i in range(self.N):
            nextstep = self.update_single_node(index = i, states_regulators = X[self.I[i]])
            if nextstep > X[i] and rng.random() < P[i, 0]:  # activation
                Fx[i] = nextstep
            elif nextstep < X[i] and rng.random() < P[i, 1]:  # degradation
                Fx[i] = nextstep
        return Fx


    def get_steady_states_asynchronous(self, nsim : int = 500, EXACT : bool = False,
        initial_sample_points : list = [], search_depth : int = 50,
        DEBUG : bool = False, *, rng=None) -> dict:
        """
        Compute the steady states of a Boolean network under asynchronous updates.

        This function simulates asynchronous updates of a Boolean network
        (with N nodes) for a given number of initial conditions (nsim). For
        each initial state, the network is updated asynchronously until a
        steady state (or attractor) is reached or until a maximum search depth
        is exceeded. The simulation can be performed either approximately
        (by sampling nsim random initial conditions) or exactly (by iterating
        over the entire state space when EXACT == True).

        **Parameters:**
            
            - nsim (int, optional): Number of initial conditions to simulate
              (default is 500).
              
            - EXACT (bool, optional): If True, iterate over the entire state
              space and guarantee finding all steady states (2^N initial
              conditions); otherwise, use nsim random initial conditions.
              (Default is False.)
              
            - initial_sample_points (list[list[int]], optional): List of
              initial states (as binary vectors) to use. If provided and EXACT
              is False, these override random sampling.
              
            - search_depth (int, optional): Maximum number of asynchronous
              update iterations to attempt per simulation.
              
            - DEBUG (bool, optional): If True, print debugging information
              during simulation.
            
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - SteadyStates (list[int]): List of steady state
                  values (in decimal form) found.
                  
                - NumberOfSteadyStates (int): Total number of unique steady states.
                - BasinSizes (list[int]): List of counts showing how many
                  initial conditions converged to each steady state.
                  
                - STGAsynchronous (dict[tuple(int, int):int]):
                  The asynchronous state transition graph. 
                  STGAsynchronous[(a,i)] = c implies that state a transitions
                  to state c when the ith variable is updated. Here, a and c
                  are decimal representations of the state and i is in {0, 1,
                  ..., self.N-1}.
                  
                - InitialSamplePoints (list[int]): The list of initial sample
                  points used (if provided) or those generated during simulation.
        """
        rng = utils._coerce_rng(rng)
        if EXACT:
            left_side_of_truth_table = utils.get_left_side_of_truth_table(self.N)

        sampled_points = []
        
        assert initial_sample_points == [] or not EXACT, (
            "Warning: sample points were provided but, with option EXACT==True, the entire state space is computed "
            "(and initial sample points ignored)"
        )
                
        STG_asynchronous = dict()
        steady_states = []
        basin_sizes = []
        steady_state_dict = dict()   
        
        for iteration in range(nsim if not EXACT else 2**self.N):
            if EXACT:
                x = left_side_of_truth_table[iteration]
                xdec = iteration
            else:
                if initial_sample_points == []:  # generate random initial states on the fly
                    x = rng.integers(2, size=self.N)
                    xdec = utils.bin2dec(x)
                    sampled_points.append(xdec)
                else:                
                    x = initial_sample_points[iteration]
                    xdec = utils.bin2dec(x)
            
            if DEBUG:
                print(iteration, -1, -1, False, xdec, x)
            for jj in range(search_depth):  # update until a steady state is reached or search_depth is exceeded
                FOUND_NEW_STATE = False
                try:
                    # Check if this state is already recognized as a steady state.
                    index_ss = steady_state_dict[xdec]
                except KeyError:
                    # Asynchronously update the state until a new state is found.
                    update_order_to_try = rng.permutation(self.N)
                    for i in update_order_to_try:
                        try:
                            fxdec = STG_asynchronous[(xdec, i.item())]
                            if fxdec != xdec:
                                FOUND_NEW_STATE = True
                                x[i] = 1 - x[i]
                        except KeyError:
                            fx_i = self.update_single_node(i, x[self.I[i]])
                            if fx_i > x[i]:
                                fxdec = xdec + 2**(self.N - 1 - i.item())
                                x[i] = 1
                                FOUND_NEW_STATE = True
                            elif fx_i < x[i]:
                                fxdec = xdec - 2**(self.N - 1 - i.item())
                                x[i] = 0
                                FOUND_NEW_STATE = True
                            else:
                                fxdec = xdec
                            STG_asynchronous.update({(xdec, i.item()): fxdec})
                        if FOUND_NEW_STATE:
                            xdec = fxdec
                            break
                    if DEBUG:
                        print(iteration, jj, i, FOUND_NEW_STATE, xdec, x)
                if FOUND_NEW_STATE == False:  # steady state reached
                    try:
                        index_ss = steady_state_dict[xdec]
                        basin_sizes[index_ss] += 1
                        break
                    except KeyError:
                        steady_state_dict.update({xdec: len(steady_states)})
                        steady_states.append(xdec)
                        basin_sizes.append(1)
                        break
            if DEBUG:
                print()
        if sum(basin_sizes) < (nsim if not EXACT else 2**self.N):
            print('Warning: only %i of the %i tested initial conditions eventually reached a steady state. Try increasing the search depth. '
                  'It may however also be the case that your asynchronous state space contains a limit cycle.' %
                  (sum(basin_sizes), nsim if not EXACT else 2**self.N))
        return dict(zip(["SteadyStates", "NumberOfSteadyStates", "BasinSizes", "STGAsynchronous", "InitialSamplePoints"],
                        (steady_states, len(steady_states), basin_sizes, STG_asynchronous,
                initial_sample_points if initial_sample_points != [] else sampled_points)))


    def get_steady_states_asynchronous_given_one_initial_condition(self,
        initial_condition : Union[int, list, np.array] = 0,
        nsim : int = 500, stochastic_weights : list = [],search_depth : int = 50,
        DEBUG : bool = False,*, rng = None) -> dict:
        """
        Determine the steady states reachable from one initial condition using
        weighted asynchronous updates.

        This function is similar to steady_states_asynchronous_given_one_IC but
        allows the update order to be influenced by provided stochastic weights
        (one per node). A weight vector (of length N) may be provided, and if
        given, it is normalized and used to bias the random permutation of
        node update order.

        **Parameters:**

            - initial_condition (int | list[int] | np.array[int], optional):
              The initial state for all simulations. If an integer, it is
              converted to a binary vector. Default is 0.
              
            - nsim (int, optional): Number of simulation runs (default is 500).
            
            - stochastic_weights (list[float], optional): List of stochastic
              weights (one per node) used to bias update order. If empty,
              uniform random order is used.
              
            - search_depth (int, optional): Maximum number of asynchronous
              update iterations per simulation (default is 50).
              
            - DEBUG (bool, optional): If True, print debugging information
              (default is False).
              
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - SteadyStates (list[int]): List of steady state values (in
                  decimal form) reached.
                
                - NumberOfSteadyStates (int): Total number of unique steady states.
                
                - BasinSizes (list[int]): List of counts of how many
                  simulations reached each steady state.
                  
                - TransientTimes (list[list[int]]): List of lists with
                  transient times (number of updates) for each steady state.
                  
                - STGAsynchronous (dict[tuple(int, int):int]):
                  A sample of the asynchronous state transition graph. 
                  STGAsynchronous[(a,i)] = c implies that state a transitions
                  to state c when the ith variable is updated. Here, a and c
                  are decimal representations of the state and i is in {0, 1,
                  ..., self.N-1}.
                  
                - UpdateQueues (list[list[int]]): List of state update queues
                  (the sequence of states encountered) for each simulation.
        """
        rng = utils._coerce_rng(rng)
        
        if type(initial_condition) == int:
            initial_condition = np.array(utils.dec2bin(initial_condition, self.N))
            initial_condition_bin = utils.bin2dec(initial_condition)
        else:
            initial_condition = np.array(initial_condition, dtype=int)
            initial_condition_bin = utils.bin2dec(initial_condition)
        
        assert stochastic_weights == [] or len(stochastic_weights) == self.N, "one stochastic weight per node is required"    
        if stochastic_weights != []:
            stochastic_weights = np.array(stochastic_weights) / sum(stochastic_weights)
        
        STG_async = dict()
        steady_states = []
        basin_sizes = []
        transient_times = []
        steady_state_dict = dict()   
        queues = []
        for iteration in range(nsim):
            x = initial_condition.copy()
            xdec = initial_condition_bin
            queue = [xdec]
            for jj in range(search_depth):  # update until a steady state is reached or search_depth is exceeded
                FOUND_NEW_STATE = False
                try:
                    index_ss = steady_state_dict[xdec]
                except KeyError:
                    if stochastic_weights != []:
                        update_order_to_try = rng.choice(self.N, size=self.N, replace=False, p=stochastic_weights)
                    else:
                        update_order_to_try = rng.permutation(self.N)
                    for i in update_order_to_try:
                        try:
                            fxdec = STG_async[(xdec, i.item())]
                            if fxdec != xdec:
                                FOUND_NEW_STATE = True
                                x[i] = 1 - x[i]
                        except KeyError:
                            fx_i = self.update_single_node(i, x[self.I[i]])
                            if fx_i > x[i]:
                                fxdec = xdec + 2**(self.N - 1 - i.item())
                                x[i] = 1
                                FOUND_NEW_STATE = True
                            elif fx_i < x[i]:
                                fxdec = xdec - 2**(self.N - 1 - i.item())
                                x[i] = 0
                                FOUND_NEW_STATE = True
                            else:
                                fxdec = xdec
                            STG_async.update({(xdec, i.item()): fxdec})
                        if FOUND_NEW_STATE:
                            xdec = fxdec
                            queue.append(xdec)
                            break
                    if DEBUG:
                        print(iteration, jj, i, FOUND_NEW_STATE, xdec, x)
                if not FOUND_NEW_STATE:  # steady state reached
                    queues.append(queue[:])
                    try:
                        index_ss = steady_state_dict[xdec]
                        basin_sizes[index_ss] += 1
                        transient_times[index_ss].append(jj)
                        break
                    except KeyError:
                        steady_state_dict.update({xdec: len(steady_states)})
                        steady_states.append(xdec)
                        basin_sizes.append(1)
                        transient_times.append([jj])
                        break
            if FOUND_NEW_STATE:
                print(jj)
                break
            if DEBUG:
                print()
        if sum(basin_sizes) < nsim:
            print('Warning: only %i of the %i tested initial conditions eventually reached a steady state. '
                  'Try increasing the search depth. It may also be that your asynchronous state space contains a limit cycle.' % (sum(basin_sizes), nsim))
        return dict(zip(["SteadyStates", "NumberOfSteadyStates", "BasinSizes", "TransientTimes", "STGAsynchronous", "UpdateQueues"],
                        (steady_states, len(steady_states), basin_sizes, transient_times, STG_async, queues)))


    def get_attractors_synchronous(self, nsim : int = 500,
        initial_sample_points : list = [], n_steps_timeout : int = 1000,
        INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS : bool = True, *, rng=None) -> dict:
        """
        Compute the number of attractors in a Boolean network using an
        alternative (v2) approach.

        This version is optimized for networks with longer average path
        lengths. For each of nb initial conditions, the network is updated
        synchronously until an attractor is reached or until n_steps_timeout
        is exceeded. The function returns the attractors found, their basin
        sizes, a mapping of states to attractors, the set of initial sample
        points used, the explored state space, and the number of simulations
        that timed out.

        **Parameters:**
            
            - nsim (int, optional): Number of initial conditions to simulate
              (default is 500). Ignored if 'initial_sample_points' are provided.
              
            - initial_sample_points (list[int | list[int]], optional): List of
              initial states (in decimal) to use.
              
            - n_steps_timeout (int, optional): Maximum number of update steps
              allowed per simulation (default 1000).
              
            - INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS (bool, optional): If
              True, initial_sample_points are provided as binary vectors; if
              False, they are given as decimal numbers. Default is True.
              
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - Attractors (list[list[int]]): List of attractors (each as a
                  list of states in the attractor cycle).
                
                - NumberOfAttractors (int): Total number of unique attractors
                  found. This is a lower bound.
                  
                - BasinSizes (list[int]): List of counts for each attractor.
                  This is an unbiased estimator.
                  
                - AttractorDict (dict[int:int]): Dictionary mapping states
                  (in decimal) to the index of their attractor.
                  
                - InitialSamplePoints (list[int]): The initial sample points
                  used (if provided, they are returned; otherwise, the 'nsim'
                  generated points are returned).
                  
                - STG (dict[int:int]):
                  A sample of the state transition graph as dictionary, with 
                  each state represented by its decimal representation.
                  
                - NumberOfTimeouts (int): Number of simulations that timed out
                  before reaching an attractor. Increase 'n_steps_timeout' to 
                  reduce this number.
        """
        rng = utils._coerce_rng(rng)
        dictF = dict()
        attractors = []
        basin_sizes = []
        attr_dict = dict()
        STG = dict()
        
        sampled_points = []
        n_timeout = 0
        
        INITIAL_SAMPLE_POINTS_EMPTY = utils.check_if_empty(initial_sample_points)
        if not INITIAL_SAMPLE_POINTS_EMPTY:
            nsim = len(initial_sample_points)
        
        for i in range(nsim):
            if INITIAL_SAMPLE_POINTS_EMPTY:
                x = rng.integers(2, size=self.N)
                xdec = utils.bin2dec(x)
                sampled_points.append(xdec)
            else:
                if INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS:
                    x = initial_sample_points[i]
                    xdec = utils.bin2dec(x)
                else:
                    xdec = initial_sample_points[i]
                    x = np.array(utils.dec2bin(xdec, self.N))
            queue = [xdec]
            count = 0
            while count < n_steps_timeout:
                try:
                    fxdec = dictF[xdec]
                except KeyError:
                    fx = self.update_network_synchronously(x)
                    fxdec = utils.bin2dec(fx)
                    dictF.update({xdec: fxdec})
                    x = fx
                if count == 0:
                    STG.update({xdec:fxdec})
                try:
                    index_attr = attr_dict[fxdec]
                    basin_sizes[index_attr] += 1
                    attr_dict.update(list(zip(queue, [index_attr] * len(queue))))
                    break
                except KeyError:
                    try:
                        index = queue.index(fxdec)
                        attr_dict.update(list(zip(queue[index:], [len(attractors)] * (len(queue) - index))))
                        attractors.append(queue[index:])
                        basin_sizes.append(1)
                        break
                    except ValueError:
                        pass
                queue.append(fxdec)
                xdec = fxdec
                count += 1
                if count == n_steps_timeout:
                    n_timeout += 1
        return dict(zip(["Attractors", "NumberOfAttractors", "BasinSizes", "AttractorDict", "InitialSamplePoints", "STG", "NumberOfTimeouts"],
                        (attractors, len(attractors), basin_sizes, attr_dict,
                sampled_points if INITIAL_SAMPLE_POINTS_EMPTY else initial_sample_points,
                STG, n_timeout)))


    def compute_synchronous_state_transition_graph(self) -> dict:
        """
        Compute the entire synchronous state transition graph for all 2^N states,
        which is of type dict[int:int]. That is, each state is represented by 
        its decimal representation.
        """  
    
        # 1. Represent all possible network states as binary matrix
        #    shape = (2**n, n), each row = one state
        states = utils.get_left_side_of_truth_table(self.N_variables)
        
        # 2. Preallocate array for next states
        next_states = np.zeros_like(states)
        powers_of_two = 2 ** np.arange(self.N_variables)[::-1]
    
        # 3. Compute next value for each node in vectorized form
        for j, bf in enumerate(self.F):
            regulators = self.I[j]
            if len(regulators) == 0:
                # constant node
                next_states[:, j] = bf.f[0]
                continue
    
            # Extract substate of regulators for all states
            subspace = states[:, regulators]
    
            # Convert each substate to integer index (row of truth table)
            idx = np.dot(subspace, powers_of_two[-len(regulators):])
    
            # Lookup next-state value from Boolean function truth table
            next_states[:, j] = bf.f[idx]
    
        # 4. Convert each next-state binary vector to integer index
        next_indices = np.dot(next_states, powers_of_two)
    
        self.STG = dict(zip(list(range(2**self.N_variables)), next_indices.tolist()))
                

    def get_attractors_synchronous_exact(self) -> dict:
        """
        Compute the exact number of attractors in a Boolean network using a
        fast, vectorized approach.

        This function computes all attractors and their basin sizes from the 
        the full state transition graph.

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - Attractors (list[list[int]]): List of attractors (each
                  attractor is represented as a list of states forming the
                  cycle).
                
                - NumberOfAttractors (int): Total number of unique attractors.
                
                - BasinSizes (list[int]): List of counts for each attractor.
                
                - AttractorDict (dict[int:int]): Dictionary mapping each state
                  (in decimal) to its attractor index.
                  
                - STG (dict[int:int]):
                  The state transition graph as dictionary, with each state
                  represented by its decimal representation.
        """        

        if self.STG is None:
            self.compute_synchronous_state_transition_graph()
        
        attractors = []
        basin_sizes = []
        attractor_dict = dict()
        for xdec in range(2**self.N):
            queue = [xdec]
            while True:
                fxdec = self.STG[xdec]
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
        return dict(zip(["Attractors", "NumberOfAttractors", "BasinSizes", "AttractorDict", "STG"],
                        (attractors, len(attractors), basin_sizes, attractor_dict, self.STG)))  






    ## Robustness measures: synchronous Derrida value, entropy of basin size distribution, coherence, fragility
    def get_derrida_value(self, nsim : int = 1000, EXACT : bool = False,
        *, rng = None) -> float:
        """
        Estimate the Derrida value for a Boolean network.

        The Derrida value is computed by perturbing a single node in a randomly
        chosen state and measuring the average Hamming distance between the
        resulting updated states of the original and perturbed networks.

        **Parameters:**
            
            - nsim (int, optional): Number of simulations to perform. Default
              is 1000.
              
            - EXACT (bool, optional): If True, the exact Derrida value is
              computed and 'nsim' is ignored. Otherwise, 'nsim' simulations
              are used to approximate the Derrida value.
            
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - float: The average Hamming distance (Derrida value) over
              nsim simulations.

        **References:**
            
            #. Derrida, B., & Pomeau, Y. (1986). Random networks of automata:
               a simple annealed approximation. Europhysics letters, 1(2), 45.
        """
        if EXACT:
            return np.mean([bf.get_average_sensitivity(EXACT=True,NORMALIZED=False) for bf in self.F])
        else:
            rng = utils._coerce_rng(rng)
            hamming_distances = []
            for i in range(nsim):
                X = rng.integers(2, size = self.N)
                Y = X.copy()
                index = rng.integers(self.N)
                Y[index] = 1 - Y[index]
                FX = self.update_network_synchronously(X)
                FY = self.update_network_synchronously(Y)
                hamming_distances.append(sum(FX != FY))
            return np.mean(hamming_distances)


    def get_attractors_and_robustness_measures_synchronous_exact(self) -> dict:
        """
        Compute the attractors and several robustness measures of a Boolean network.

        This function computes the exact attractors and robustness (coherence
        and fragility) of the entire network, as well as robustness measures
        for each basin of attraction and each attractor.

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - Attractors (list[list[int]]): List of attractors (each
                  attractor is represented as a list of state decimal numbers).
                
                - ExactNumberOfAttractors (int): The exact number of network
                  attractors.
                  
                - BasinSizes (list[int]): List of exact basin sizes for each
                  attractor.
                  
                - AttractorDict (dict[int:int]): Dictionary mapping each state
                  (in decimal) to its attractor index.
                  
                - Coherence (float): overall exact network coherence
                - Fragility (float): overall exact network fragility
                - BasinCoherence (list[float]): exact coherence of each basin.
                - BasinFragility (list[float]): exact fragility of each basin.
                - AttractorCoherence (list[float]): exact coherence of each
                  attractor.
                  
                - AttractorFragility (list[float]): exact fragility of each
                  attractor.
        
        **References:**
            
            #. Park, K. H., Costa, F. X., Rocha, L. M., Albert, R., & Rozum,
               J. C. (2023). Models of cell processes are far from the edge of
               chaos. PRX life, 1(2), 023009.
               
            #. Bavisetty, V. S. N., Wheeler, M., & Kadelka, C. (2025). xxxx
               arXiv preprint arXiv:xxx.xxx.
        """
        left_side_of_truth_table = utils.get_left_side_of_truth_table(self.N)

        result = self.get_attractors_synchronous_exact()
        attractors, n_attractors, basin_sizes, attractor_dict = result["Attractors"], result["NumberOfAttractors"], result["BasinSizes"], result["AttractorDict"]
        len_attractors = list(map(len,attractors))
        
        if n_attractors == 1:
            return dict(zip(["Attractors", "ExactNumberOfAttractors", "BasinSizes",
                             "AttractorDict", "BasinCoherence", "BasinFragility",
                             "AttractorCoherence", "AttractorFragility", "Coherence", "Fragility"],
                (attractors, n_attractors, np.array(basin_sizes)/2**self.N,
                 attractor_dict, np.ones(1), np.zeros(1), np.ones(1), np.zeros(1), 1, 0)))
        
        mean_states_attractors = []
        is_attr_dict = dict()
        for i in range(n_attractors):
            if len_attractors[i] == 1:
                mean_states_attractors.append(np.array(utils.dec2bin(attractors[i][0], self.N)))
            else:
                states_attractors = np.array([utils.dec2bin(state, self.N) for state in attractors[i]])
                mean_states_attractors.append(states_attractors.mean(0))
            for state in attractors[i]:
                is_attr_dict.update({state:i})
            
        distance_between_attractors = np.zeros((n_attractors,n_attractors),dtype=int)
        for i in range(n_attractors):
            for j in range(i+1,n_attractors):
                distance_between_attractors[i,j] = np.sum(np.abs(mean_states_attractors[i] - mean_states_attractors[j]))
                distance_between_attractors[j,i] = distance_between_attractors[i,j]
        distance_between_attractors = distance_between_attractors/self.N
        
        basin_coherences = np.zeros(n_attractors)
        basin_fragilities = np.zeros(n_attractors)
        attractor_coherences = np.zeros(n_attractors)
        attractor_fragilities = np.zeros(n_attractors)
        
        powers_of_2 = np.array([2**i for i in range(self.N)])[::-1]
        for xdec, x in enumerate(left_side_of_truth_table): #iterate over each edge of the n-dim Hypercube once
            for i in range(self.N):
                if x[i] == 0:
                    ydec = xdec + powers_of_2[i]
                else: #to ensure we are not double-counting each edge
                    continue
                index_attr_x = attractor_dict[xdec]
                index_attr_y = attractor_dict[ydec]
                if index_attr_x == index_attr_y:
                    basin_coherences[index_attr_x] += 1
                    basin_coherences[index_attr_y] += 1
                    try:
                        is_attr_dict[xdec]
                        attractor_coherences[index_attr_x] += 1
                    except KeyError:
                        pass
                    try:
                        is_attr_dict[ydec]
                        attractor_coherences[index_attr_y] += 1
                    except KeyError:
                        pass
                else:
                    normalized_Hamming_distance = distance_between_attractors[index_attr_x,index_attr_y]
                    basin_fragilities[index_attr_x] += normalized_Hamming_distance
                    basin_fragilities[index_attr_y] += normalized_Hamming_distance
                    try:
                        is_attr_dict[xdec]
                        attractor_fragilities[index_attr_x] += normalized_Hamming_distance
                    except KeyError:
                        pass
                    try:
                        is_attr_dict[ydec]
                        attractor_fragilities[index_attr_y] += normalized_Hamming_distance
                    except KeyError:
                        pass
                    
        #normalizations
        for i,(basin_size,length_attractor) in enumerate(zip(basin_sizes,len_attractors)):
            basin_coherences[i] = basin_coherences[i] / basin_size / self.N
            basin_fragilities[i] = basin_fragilities[i] / basin_size / self.N
            attractor_coherences[i] = attractor_coherences[i] / length_attractor / self.N
            attractor_fragilities[i] = attractor_fragilities[i] / length_attractor / self.N
        basin_sizes = np.array(basin_sizes)/2**self.N
        
        coherence = np.dot(basin_sizes,basin_coherences)
        fragility = np.dot(basin_sizes,basin_fragilities)
        
        return dict(zip(["Attractors", "ExactNumberOfAttractors", 
                         "BasinSizes","AttractorDict",
                         "Coherence", "Fragility",
                         "BasinCoherence", "BasinFragility",
                         "AttractorCoherence", "AttractorFragility"],
                    (attractors, n_attractors, 
                     basin_sizes, attractor_dict,
                     coherence,fragility,
                     basin_coherences, basin_fragilities,
                     attractor_coherences, attractor_fragilities)))


    def get_attractors_and_robustness_measures_synchronous(self, number_different_IC : int = 500,
        RETURN_ATTRACTOR_COHERENCE : bool = True, *, rng=None) -> dict:
        """
        Approximate global robustness measures and attractors.

        This function samples the attractor landscape by simulating the network
        from a number of different initial conditions. It computes:
            
            - The coherence: the proportion of neighboring states (in the
              Boolean hypercube) that, after synchronous update, transition to
              the same attractor.
            
            - The fragility: a measure of how much the attractor state changes
              (assumed under synchronous update) in response to perturbations.
              
            - The final time-step Hamming distance between perturbed trajectories.

        In addition, it collects several details about each attractor (such as
        basin sizes, coherence of each basin, etc.).

        **Parameters:**
            
            - number_different_IC (int, optional): Number of different initial
              conditions to sample (default is 500).
              
            - RETURN_ATTRACTOR_COHERENCE (bool, optional): Determines whether
              the attractor coherence should also be computed (default True,
              i.e., Yes).
              
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - dict[str:Variant]: A dictionary containing:
                
                - Attractors (list[list[int]]): List of attractors (each
                  attractor is represented as a list of state decimal numbers).
                
                - LowerBoundOfNumberOfAttractors (int): The lower bound on the
                  number of attractors found.
                  
                - BasinSizes (list[int]): List of basin sizes for each attractor.
                - CoherenceApproximation (float): The approximate overall
                  network coherence.
                  
                - FragilityApproximation (float): The approximate overall
                  network fragility.
                  
                - FinalHammingDistanceApproximation (float): The approximate
                  final Hamming distance measure.
                  
                - BasinCoherenceApproximation (list[float]): The approximate
                  coherence of each basin.
                  
                - BasinFragilityApproximation (list[float]): The approximate
                  fragility of each basin.
                  
                - AttractorCoherence (list[float]): The exact coherence of
                  each attractor (only computed and returned if
                  RETURN_ATTRACTOR_COHERENCE == True).
                  
                - AttractorFragility (list[float]): The exact fragility of
                  each attractor (only computed and returned if
                  RETURN_ATTRACTOR_COHERENCE == True).

        **References:**
            
            #. Park, K. H., Costa, F. X., Rocha, L. M., Albert, R., & Rozum,
               J. C. (2023). Models of cell processes are far from the edge of
               chaos. PRX life, 1(2), 023009.
               
            #. Bavisetty, V. S. N., Wheeler, M., & Kadelka, C. (2025). xxxx
               arXiv preprint arXiv:xxx.xxx.
        """
        rng = utils._coerce_rng(rng)
        def lcm(a, b):
            return abs(a*b) // math.gcd(a, b)
        
        dictF = dict()
        attractors = []
        ICs_per_attractor_state = []
        basin_sizes = []
        attractor_dict = dict()
        attractor_state_dict = []
        distance_from_attractor_state_dict = []
        counter_phase_shifts = []
        
        height = []
        
        powers_of_2s = [np.array([2**i for i in range(NN)])[::-1] for NN in range(max(self.indegrees)+1)]
        if self.N<64:
            powers_of_2 = np.array([2**i for i in range(self.N)])[::-1]
        
        robustness_approximation = 0
        fragility_sum = 0
        basin_robustness = defaultdict(float)
        basin_fragility = defaultdict(float)
        final_hamming_distance_approximation = 0
        mean_states_attractors = []
        states_attractors = []
        
        for i in range(number_different_IC):
            index_attractors = []
            index_of_state_within_attractor_reached = []
            distance_from_attractor = []
            for j in range(2):
                if j == 0:
                    x = rng.integers(2, size=self.N)
                    if self.N<64:
                        xdec = np.dot(x, powers_of_2).item()
                    else: #out of range of np.int64
                        xdec = ''.join(str(bit) for bit in x)
                    x_old = x.copy()
                else:
                    x = x_old
                    random_flipped_bit = rng.integers(self.N)
                    x[random_flipped_bit] = 1 - x[random_flipped_bit]
                    if self.N<64:
                        xdec = np.dot(x, powers_of_2).item()
                    else: #out of range of np.int64
                        xdec = ''.join(str(bit) for bit in x)               
                queue = [xdec]
                try:
                    index_attr = attractor_dict[xdec]
                except KeyError:
                    while True:
                        try: #check if we already know F(xdec)
                            fxdec = dictF[xdec]
                        except KeyError: #if not, then compute the F(xdec)
                            fx = []
                            for jj in range(self.N):
                                if self.indegrees[jj]>0:
                                    fx.append(self.F[jj].f[np.dot(x[self.I[jj]], powers_of_2s[self.indegrees[jj]]).item()])
                                else:#constant functions whose regulators were all fixed to a specific value
                                    fx.append(self.F[jj].f[0])
                            if self.N<64:
                                fxdec = np.dot(fx, powers_of_2).item()
                            else:
                                fxdec = ''.join(str(bit) for bit in fx)               
                            dictF.update({xdec: fxdec})
                        try: #check if we already know the attractor of F(xdec) 
                            index_attr = attractor_dict[fxdec]
                            dummy_index_within_attractor_reached = attractor_state_dict[index_attr][fxdec]
                            dummy_distance_from_attractor = distance_from_attractor_state_dict[index_attr][fxdec]
                            attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                            attractor_state_dict[index_attr].update(list(zip(queue, [dummy_index_within_attractor_reached]*len(queue))))
                            distance_from_attractor_state_dict[index_attr].update(
                                list(zip(queue, list(range(len(queue) + dummy_distance_from_attractor, dummy_distance_from_attractor, -1))))
                            )
                            break
                        except KeyError: 
                            try: #if not, then check if F(xdec) is already in the queue, i.e., if F(xdec) is part of an attractor itself
                                index = queue.index(fxdec)
                                index_attr = len(attractors)
                                attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                                attractors.append(queue[index:])
                                basin_sizes.append(1)
                                attractor_state_dict.append(dict(zip(queue, [0]*index + list(range(len(attractors[-1])))))
                                )
                                distance_from_attractor_state_dict.append(
                                    dict(zip(queue, list(range(index, 0, -1)) + [0]*len(attractors[-1])))
                                )
                                ICs_per_attractor_state.append([0] * len(attractors[-1]))
                                counter_phase_shifts.append([0] * len(attractors[-1]))

                                if len(attractors[-1]) == 1:
                                    if self.N<64:
                                        fixed_point = np.array(utils.dec2bin(queue[index], self.N))
                                    else:
                                        fixed_point = np.array(list(queue[index]), dtype=int)
                                    states_attractors.append(fixed_point.reshape((1, self.N)))
                                    mean_states_attractors.append(fixed_point)
                                else:
                                    if self.N<64:
                                        limit_cycle = np.array([utils.dec2bin(state, self.N) for state in queue[index:]])
                                    else:
                                        limit_cycle = np.array([np.array(list(state), dtype=int) for state in queue[index:]])          
                                    states_attractors.append(limit_cycle)
                                    mean_states_attractors.append(limit_cycle.mean(0))
                                break
                            except ValueError: #if not, proceed by setting x = F(x)
                                x = np.array(fx)
                        queue.append(fxdec)
                        xdec = fxdec

                index_attractors.append(index_attr)
                index_of_state_within_attractor_reached.append(attractor_state_dict[index_attr][xdec])
                distance_from_attractor.append(distance_from_attractor_state_dict[index_attr][xdec])
                basin_sizes[index_attr] += 1
                ICs_per_attractor_state[index_attr][attractor_state_dict[index_attr][xdec]] += 1
            if index_attractors[0] == index_attractors[1]:
                robustness_approximation += 1
                basin_robustness[index_attractors[0]] += 1
                length_phaseshift = max(index_of_state_within_attractor_reached) - min(index_of_state_within_attractor_reached)
                counter_phase_shifts[index_attr][length_phaseshift] += 1
            else:
                fragility_sum += np.sum(np.abs(mean_states_attractors[index_attractors[0]] - mean_states_attractors[index_attractors[1]]))
                basin_fragility[index_attractors[0]] += np.sum(np.abs(mean_states_attractors[index_attractors[0]] - mean_states_attractors[index_attractors[1]]))
                required_n_states = lcm(len(attractors[index_attractors[0]]), len(attractors[index_attractors[1]]))
                index_j0 = index_of_state_within_attractor_reached[0]
                periodic_states_j0 = np.tile(states_attractors[index_attractors[0]], 
                                             (required_n_states // len(attractors[index_attractors[0]]) + 1, 1))[index_j0:(index_j0 + required_n_states), :]
                index_j1 = index_of_state_within_attractor_reached[1]
                periodic_states_j1 = np.tile(states_attractors[index_attractors[1]], 
                                             (required_n_states // len(attractors[index_attractors[1]]) + 1, 1))[index_j1:(index_j1 + required_n_states), :]
                final_hamming_distance_approximation += np.mean(periodic_states_j1 == periodic_states_j0)
                
            height.extend(distance_from_attractor)
        
        lower_bound_number_of_attractors = len(attractors)
        approximate_basin_sizes = np.array(basin_sizes)
        approximate_coherence = robustness_approximation * 1.0 / number_different_IC
        approximate_fragility = fragility_sum * 1.0 / number_different_IC / self.N
        
        approximate_basin_coherence = np.array([basin_robustness[index_att] * 2.0 / basin_sizes[index_att] for index_att in range(len(attractors))])
        approximate_basin_fragility = np.array([basin_fragility[index_att] * 2.0 / basin_sizes[index_att] / self.N for index_att in range(len(attractors))])
        
        for index_attr in range(len(attractors)):
            periodic_states_two_periods = np.tile(states_attractors[index_attr], (2, 1))
            for length_phaseshift, num_IC_with_that_phaseshift in enumerate(counter_phase_shifts[index_attr]):
                if num_IC_with_that_phaseshift > 0 and length_phaseshift > 0:
                    final_hamming_distance_approximation += num_IC_with_that_phaseshift * np.mean(
                        states_attractors[index_attr] ==
                        periodic_states_two_periods[length_phaseshift:(length_phaseshift + len(attractors[index_attr])), :]
                    )
                    
        final_hamming_distance_approximation = final_hamming_distance_approximation / number_different_IC
        
        #fixing the results here because the subsequent attractor coherence computation could in theory identify additional attractors, 
        #which would screw things up because the attractor regions of the state space have then been oversampled
        results = [attractors, lower_bound_number_of_attractors, approximate_basin_sizes/2./number_different_IC, 
                   approximate_coherence, approximate_fragility, final_hamming_distance_approximation,
                   approximate_basin_coherence, approximate_basin_fragility]
        if RETURN_ATTRACTOR_COHERENCE == False:
            return dict(zip(["Attractors", "LowerBoundOfNumberOfAttractors", "BasinSizesApproximation",
                             "CoherenceApproximation", "FragilityApproximation", "FinalHammingDistanceApproximation",
                             "BasinCoherenceApproximation", "BasinFragilityApproximation"],
                            tuple(results)))
        else:
            attractor_coherence = np.zeros(lower_bound_number_of_attractors)
            attractor_fragility = np.zeros(lower_bound_number_of_attractors)
            attractors_original = attractors[:] #needed because new attractors may be found
            for index_attr_original,attractor in enumerate(attractors_original):
                for attractor_state in attractor: #perturb each attractor state
                    for i in range(self.N):
                        if self.N<64:
                            x = np.array(utils.dec2bin(attractor_state, self.N))
                        else:
                            x = np.array(list(attractor_state), dtype=int)
                        x[i] = 1 - x[i]
                        if self.N<64:
                            xdec = np.dot(x, powers_of_2).item()
                        else:
                            xdec = ''.join(str(bit) for bit in x)
                        queue = [xdec]
                        try:
                            index_attr = attractor_dict[xdec]
                        except KeyError:
                            while True:
                                try: #check if we already know F(xdec)
                                    fxdec = dictF[xdec]
                                except KeyError: #if not, then compute the F(xdec)
                                    fx = []
                                    for jj in range(self.N):
                                        if self.indegrees[jj]>0:
                                            fx.append(self.F[jj].f[np.dot(x[self.I[jj]], powers_of_2s[self.indegrees[jj]]).item()])
                                        else:#constant functions whose regulators were all fixed to a specific value
                                            fx.append(self.F[jj].f[0])
                                    if self.N<64:
                                        fxdec = np.dot(fx, powers_of_2).item()
                                    else:
                                        fxdec = ''.join(str(bit) for bit in fx)               
                                    dictF.update({xdec: fxdec})
                                try: #check if we already know the attractor of F(xdec) 
                                    index_attr = attractor_dict[fxdec]
                                    dummy_index_within_attractor_reached = attractor_state_dict[index_attr][fxdec]
                                    dummy_distance_from_attractor = distance_from_attractor_state_dict[index_attr][fxdec]
                                    attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                                    attractor_state_dict[index_attr].update(list(zip(queue, [dummy_index_within_attractor_reached]*len(queue))))
                                    distance_from_attractor_state_dict[index_attr].update(
                                        list(zip(queue, list(range(len(queue) + dummy_distance_from_attractor, dummy_distance_from_attractor, -1))))
                                    )
                                    break
                                except KeyError: 
                                    try: #if not, then check if F(xdec) is already in the queue, i.e., if F(xdec) is part of an attractor itself
                                        index = queue.index(fxdec)
                                        index_attr = len(attractors)
                                        attractor_dict.update(list(zip(queue, [index_attr]*len(queue))))
                                        attractors.append(queue[index:])
                                        #basin_sizes.append(1)
                                        attractor_state_dict.append(dict(zip(queue, [0]*index + list(range(len(attractors[-1])))))
                                        )
                                        distance_from_attractor_state_dict.append(
                                            dict(zip(queue, list(range(index, 0, -1)) + [0]*len(attractors[-1])))
                                        )
                                        ICs_per_attractor_state.append([0] * len(attractors[-1]))
                                        counter_phase_shifts.append([0] * len(attractors[-1]))
            
                                        if len(attractors[-1]) == 1:
                                            if self.N<64:
                                                fixed_point = np.array(utils.dec2bin(queue[index], self.N))
                                            else:
                                                fixed_point = np.array(list(queue[index]), dtype=int)
                                            states_attractors.append(fixed_point.reshape((1, self.N)))
                                            mean_states_attractors.append(fixed_point)
                                        else:
                                            if self.N<64:
                                                limit_cycle = np.array([utils.dec2bin(state, self.N) for state in queue[index:]])
                                            else:
                                                limit_cycle = np.array([np.array(list(state), dtype=int) for state in queue[index:]])          
                                            states_attractors.append(limit_cycle)
                                            mean_states_attractors.append(limit_cycle.mean(0))
                                        break
                                    except ValueError: #if not, proceed by setting x = F(x)
                                        x = np.array(fx)
                                queue.append(fxdec)
                                xdec = fxdec
                        if index_attr_original == index_attr:
                            attractor_coherence[index_attr_original] += 1
                        else:
                            attractor_fragility[index_attr_original] += np.sum(np.abs(mean_states_attractors[index_attr_original] - mean_states_attractors[index_attr]))
            attractor_coherence = np.array([s/self.N/size_attr for s,size_attr in zip(attractor_coherence,map(len,attractors_original))])
            attractor_fragility = np.array([s/self.N**2/size_attr for s,size_attr in zip(attractor_fragility,map(len,attractors_original))]) #something is wrong with attractor fragility, it returns values > 1 for small basins
            results[0] = attractors_original #important! It may be that new attractors were found, reset the count
            return dict(zip(["Attractors", "LowerBoundOfNumberOfAttractors", "BasinSizesApproximation",
                             "CoherenceApproximation", "FragilityApproximation", "FinalHammingDistanceApproximation",
                             "BasinCoherenceApproximation", "BasinFragilityApproximation",
                             "AttractorCoherence", "AttractorFragility"],
                            tuple(results + [attractor_coherence,attractor_fragility])))
        
# n = 14
# k=4
# bn = boolforge.random_network(N=10,n=4)
# bn_new = BooleanNetwork(bn.F,bn.I)
# bn_new.compute_synchronous_state_transition_graph_old()
# STG_old = bn_new.STG
# bn_new.compute_synchronous_state_transition_graph()
# STG = bn_new.STG
# print(STG_old == STG)