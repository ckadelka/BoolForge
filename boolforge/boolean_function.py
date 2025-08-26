#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:03:49 2025
Last Edited on Thu Aug 14 2025

@author: Benjamin Coberly, Claus Kadelka
"""

import numpy as np
import itertools

try:
    import boolforge.utils as utils
except ModuleNotFoundError:
    import utils
    
try:
    import cana.boolean_node
    __LOADED_CANA__=True
except ModuleNotFoundError:
    print('The module cana cannot be found. Ensure it is installed to use all functionality of this toolbox.')
    __LOADED_CANA__=False
    

def get_layer_structure_from_canalized_outputs(can_outputs):
    canalizing_depth = len(can_outputs)
    if canalizing_depth == 0:
        return []
    size_of_layer = 1
    layer_structure = []
    for i in range(1, canalizing_depth):
        if can_outputs[i] == can_outputs[i - 1]:
            size_of_layer += 1
        else:
            layer_structure.append(size_of_layer)
            size_of_layer = 1
    layer_structure.append(size_of_layer)
    return layer_structure


class BooleanFunction(object):
    """
    A class representing a Boolean function.

    Parameters:
        - f (list or numpy array or str): A list of length 2^n representing the outputs of a Boolean function with n inputs, or a string that can be properly evaluated, see utils.f_from_expression.
        - name (str, optional): The name of the node regulated by the Boolean function (default '').
        
    Members:
        - f (numpy array): A numpy array of length 2^n representing the outputs of a Boolean function with n inputs.
        - n (int): The number of inputs for the Boolean function.
        - variables (numpy array): A numpy array of n strings with variable names, default x0, ..., x_{n-1}.
        - name (str, optional): The name of the node regulated by the Boolean function (default '')
        - properties: Dynamically created dictionary with additional information about the function (canalizing layer structure, type of inputs, etc.)
    """
    __slots__ = ['f','n','variables','name','properties']
    
    left_side_of_truth_tables = {}
    
    def __init__(self, f, name=''):
        self.name = name
        if isinstance(f, str):
            f, self.variables = utils.f_from_expression(f)
            self.n = len(self.variables)
        else:
            assert isinstance(f, (list, np.ndarray)), "f must be a list, numpy array or interpretable string"
            assert len(f) > 0, "f cannot be empty"
            
            _n = int(np.log2(len(f)))
            assert abs(np.log2(len(f)) - _n) < 1e-6, "f must be of size 2^n, n >= 0"
            self.n = _n
            self.variables = np.array(['x%i' % i for i in range(self.n)])
            
        self.f = np.array(f, dtype=int)
        
        self.properties = {}

    @classmethod
    def from_cana(cls, cana_BooleanNode):         
        return cls(np.array(cana_BooleanNode.outputs,dtype=int))

    def __str__(self):
        return f"{self.f}"
        #return f"{self.f.tolist()}"
        
    def str_expr(self):
        return utils.bool_to_poly(self.f,variables=self.variables)
    
    def __repr__(self):
        if self.n < 6:
            return f"{type(self).__name__}(f={self.f.tolist()})"
        else:
            return f"{type(self).__name__}(f={self.f})"
    
    def __len__(self):
        return 2**self.n

    def __getitem__(self, index):
        return int(self.f[index])

    def __setitem__(self, index, value):
        self.f[index] = value
    
    def to_cana(self):
        """
        Compatability method: Returns an instance of cana.boolean_node.BooleanNode from the cana module.

        Returns:
            - An instance of cana.boolean_node.BooleanNode
        """
        if __LOADED_CANA__:
            return cana.boolean_node.BooleanNode(k=self.n, outputs=self.f)
        print('The method \'to_cana_BooleanNode\' requires the module cana, which cannot be found. Ensure it is installed to use this functionality.')
        return None
    
    def _get_left_side_of_truth_table(self):
        """
        Internal method that enables computing the left hand side of the truth table only once per degree n.
        
        Returns:
            - np.ndarray: Array of size 2^n x n representing all input combinations of an n-input Boolean function.
        """
        if self.n in BooleanFunction.left_side_of_truth_tables:
            left_side_of_truth_table = BooleanFunction.left_side_of_truth_tables[self.n]
        else:
            left_side_of_truth_table = np.array(list(itertools.product([0, 1], repeat=self.n)))
            BooleanFunction.left_side_of_truth_tables[self.n] = left_side_of_truth_table
        return left_side_of_truth_table
    
    
    def get_hamming_weight(self):
        return int(self.f.sum())
    
    def is_constant(self):
        """
        Check whether a Boolean function is constant.

        Returns:
            - bool: True if f is constant (all outputs are 0 or all are 1), False otherwise.
        """
        return self.get_hamming_weight() in [0, 2**self.n]
    
    def is_degenerated(self):
        """
        Determine if a Boolean function contains non-essential variables.

        A variable is non-essential if the function's output does not depend on it.

        Returns:
            - bool: True if f contains at least one non-essential variable, False if all variables are essential.
        """
        for i in range(self.n):
            dummy_add = (2**(self.n-1-i))
            dummy = np.arange(2**self.n) % (2**(self.n-i)) // dummy_add
            depends_on_i = False
            for j in range(2**self.n):
                if dummy[j] == 1:
                    continue
                else:
                    if self.f[j] != self.f[j + dummy_add]:
                        depends_on_i = True
                        break
            if depends_on_i == False:
                return True
        return False

    def get_essential_variables(self):
        """
        Determine the indices of essential variables in a Boolean function.

        A variable is essential if changing its value (while holding the others constant) can change the output of f.

        Returns:
            - list: List of indices corresponding to the essential variables.
        """
        if len(self.f) == 0:
            return []
        essential_variables = list(range(self.n))
        for i in range(self.n):
            dummy_add = (2**(self.n-1-i))
            dummy = np.arange(2**self.n) % (2**(self.n-i)) // dummy_add
            depends_on_i = False
            for j in range(2**self.n):
                if dummy[j] == 1:
                    continue
                else:
                    if self.f[j] != self.f[j + dummy_add]:
                        depends_on_i = True
                        break
            if depends_on_i == False:
                essential_variables.remove(i)
        return essential_variables 

    def get_number_of_essential_variables(self):
        """
        Count the number of essential variables in a Boolean function.

        Returns:
            - int: The number of essential variables.
        """
        return len(self.get_essential_variables())
    
    
    def get_type_of_inputs(self):
        """
        Determine for each input of the Boolean function whether it is positive, negative, conditional or non-essential.

        Returns:
            - np.ndarray of str: The type of each input of the Boolean function.
        """
        
        if 'InputTypes' in self.properties:
            return self.properties['InputTypes']
        else:
            types = []
            for i in range(self.n):
                dummy_add=(2**(self.n-1-i))
                dummy=np.arange(2**self.n)%(2**(self.n-i))//dummy_add
                diff = self.f[dummy==1]-self.f[dummy==0]
                min_diff = min(diff)
                max_diff = max(diff)
                if min_diff==0 and max_diff==0:
                    types.append('non-essential')
                elif min_diff==-1 and max_diff==1:
                    types.append('conditional')
                elif min_diff>=0 and max_diff==1:
                    types.append('positive')            
                elif min_diff==-1 and max_diff<=0:
                    types.append('negative')
            types = np.array(types)
            self.properties.update({'InputTypes':types})
            return types


    def is_monotonic(self):
        """
        Determine if a Boolean function is monotonic.

        A Boolean function is monotonic if it is monotonic in each variable. 
        That is, if for all i=1,...,n: f(x_1,...,x_i=0,...,x_n) >= f(x_1,...,x_i=1,...,x_n) for all (x_1,...,x_n) or f(x_1,...,x_i=0,...,x_n) <= f(x_1,...,x_i=1,...,x_n) for all (x_1,...,x_n)

        Returns:
            - bool: True if f contains no conditional variables, False if at least one variable is conditional.
        """            
        return 'conditional' not in self.get_type_of_inputs()
    
    
    def get_symmetry_groups(self):
        """
        Determine all symmetry groups of input variables for a Boolean function.

        Two variables are in the same symmetry group if swapping their values does not change the output
        of the function for any input of the other variables.

        Returns:
            - list: A list of lists where each inner list contains indices of variables that form a symmetry group.
        """
        
        symmetry_groups = []
        left_to_check = np.ones(self.n)
        for i in range(self.n):
            if left_to_check[i] == 0:
                continue
            else:
                symmetry_groups.append([i])
                left_to_check[i] = 0
            for j in range(i + 1, self.n):
                diff = sum(2**np.arange(self.n - i - 2, self.n - j - 2, -1))
                for ii, x in enumerate(self._get_left_side_of_truth_table()):
                    if x[i] != x[j] and x[i] == 0 and self.f[ii] != self.f[ii + diff]:
                        break
                else:
                    left_to_check[j] = 0
                    symmetry_groups[-1].append(j)
        return symmetry_groups
    
    def get_absolute_bias(self):
        """
        Compute the absolute bias of a Boolean function.

        The absolute bias is defined as `|(self.get_hamming_weight() / 2^(n-1)) - 1|`, which quantifies how far the function's output distribution deviates from being balanced.

        Returns:
            - float: The absolute bias of the Boolean function.
        """
        return abs(self.get_hamming_weight() * 1.0 / 2**(self.n - 1) - 1)
    
    def get_average_sensitivity(self, nsim=10000, EXACT=False, NORMALIZED=True, *, rng=None):
        """
        Compute the average sensitivity of a Boolean function.

        The average sensitivity is equivalent to the Derrida value D(F,1) when the update rule is sampled
        from the same space. This function can compute the exact sensitivity by exhaustively iterating over all inputs (if EXACT is True)
        or estimate it via Monte Carlo sampling (if EXACT is False). The result can be normalized by the number of inputs.

        Parameters:
            - nsim (int, optional): Number of random samples (default is 10000, used when EXACT is False).
            - EXACT (bool, optional): If True, compute the exact sensitivity by iterating over all inputs; otherwise, use sampling (default).
            - NORMALIZED (bool, optional): If True, return the normalized sensitivity (divided by the number of function inputs); otherwise, return the total count.

        Returns:
            - float: The (normalized) average sensitivity of the Boolean function.
        """        
        size_state_space = 2**self.n
        s = 0
        if EXACT:
            for ii, X in enumerate(self._get_left_side_of_truth_table()):
                for i in range(self.n):
                    Y = X.copy()
                    Y[i] = 1 - X[i]
                    Ydec = utils.bin2dec(Y)
                    s += int(self.f[ii] != self.f[Ydec])
            if NORMALIZED:
                return s / (size_state_space * self.n)
            else:
                return s / size_state_space
        else:
            rng = utils._coerce_rng(rng)

            for i in range(nsim):
                xdec = rng.integers(size_state_space)
                Y = utils.dec2bin(xdec, self.n)
                index = rng.integers(self.n)
                Y[index] = 1 - Y[index]
                Ybin = utils.bin2dec(Y)
                s += int(self.f[xdec] != self.f[Ybin])
            if NORMALIZED:
                return s / nsim
            else:
                return self.n * s / nsim
    
    
    def is_canalizing(self):
        """
        Determine if a Boolean function is canalizing.

        A Boolean function f(x_1, ..., x_n) is canalizing if there exists at least one variable x_i and a value a ∈ {0, 1} 
        such that f(x_1, ..., x_i = a, ..., x_n) is constant.

        Returns:
            - bool: True if f is canalizing, False otherwise.
        """
        desired_value = 2**(self.n - 1)
        T = np.array(list(itertools.product([0, 1], repeat=self.n))).T
        A = np.r_[T, 1 - T]
        Atimesf = np.dot(A, self.f)
        if np.any(Atimesf == desired_value):
            return True
        elif np.any(Atimesf == 0):
            return True
        else:
            return False
    
    def is_k_canalizing(self, k):
        """
        Determine if a Boolean function is k-canalizing.

        A Boolean function is k-canalizing if it has at least k conditionally canalizing variables.
        This is checked recursively: after fixing a canalizing variable (with a fixed canalizing input that forces the output),
        the subfunction (core function) must itself be canalizing for the next variable, and so on.

        Parameters:
            - k (int): The desired canalizing depth (0 ≤ k ≤ n). Note: every function is 0-canalizing.

        Returns:
            - bool: True if f is k-canalizing, False otherwise.

        References:
            He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
                Physica D: Nonlinear Phenomena, 314, 1-8.
            Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions:
                Algorithms and applications. Automatica, 146, 110630.
        """
        if k > self.n:
            return False
        if k == 0:
            return True

        w = self.get_hamming_weight()  # Hamming weight of f
        if w == 0 or w == 2**self.n:  # constant function
            return False
        desired_value = 2**(self.n - 1)
        T = np.array(list(itertools.product([0, 1], repeat=self.n))).T
        A = np.r_[T, 1 - T]
        try:  # check for canalizing output 1
            index = list(np.dot(A, self.f)).index(desired_value)
            new_bf = BooleanFunction(self.f[np.where(A[index] == 0)[0]])
            return new_bf.is_k_canalizing(k - 1)
        except ValueError:
            try:  # check for canalizing output 0
                index = list(np.dot(A, 1 - self.f)).index(desired_value)
                new_bf = BooleanFunction(self.f[np.where(A[index] == 0)[0]])
                return new_bf.is_k_canalizing(k - 1)
            except ValueError:
                return False

    def _get_layer_structure(self, can_inputs, can_outputs, can_order, variables, depth, number_layers):
        """
        Only for internal use by recursively defined get_layer_structure.
        """
        n = self.n
        w = self.get_hamming_weight()
        if w == 0 or w == 2**n:  #eventually the recursion will end here (if self.f is a constant function)
            return (depth, number_layers, can_inputs, can_outputs, self, can_order)
        if type(variables) == np.ndarray:
            variables = list(variables)
        if variables == []:
            variables = list(range(n))
        desired_value = 2**(n - 1)
        T = np.array(list(itertools.product([0, 1], repeat=n))).T
        A = np.r_[T, 1 - T]

        indices1 = np.where(np.dot(A, self.f) == desired_value)[0]
        indices0 = np.where(np.dot(A, 1 - self.f) == desired_value)[0]
        if len(indices1) > 0:
            sorted_order = sorted(range(len(indices1)), key=lambda x: (indices1 % n)[x])
            inputs = (1 - indices1 // n)[np.array(sorted_order)]
            outputs = np.ones(len(indices1), dtype=int)
            new_canalizing_variables = []
            for index in np.sort(indices1 % n)[::-1]:
                new_canalizing_variables.append(variables.pop(index))
            new_canalizing_variables.reverse()
            new_f = self.f[np.sort(list(set.intersection(*[] + [set(np.where(A[index] == 0)[0]) for index, INPUT in zip(indices1, inputs)])))]
            new_bf = BooleanFunction(list(new_f))
            return new_bf._get_layer_structure(np.append(can_inputs, inputs), np.append(can_outputs, outputs),
                               np.append(can_order, new_canalizing_variables), variables, depth + len(new_canalizing_variables),
                               number_layers + 1)
        elif len(indices0):
            sorted_order = sorted(range(len(indices0)), key=lambda x: (indices0 % n)[x])
            inputs = (1 - indices0 // n)[np.array(sorted_order)]
            outputs = np.zeros(len(indices0), dtype=int)
            new_canalizing_variables = []
            for index in np.sort(indices0 % n)[::-1]:
                new_canalizing_variables.append(variables.pop(index))
            new_canalizing_variables.reverse()
            new_f = self.f[np.sort(list(set.intersection(*[] + [set(np.where(A[index] == 0)[0]) for index, INPUT in zip(indices0, inputs)])))]
            new_bf = BooleanFunction(list(new_f))
            return new_bf._get_layer_structure(np.append(can_inputs, inputs), np.append(can_outputs, outputs),
                               np.append(can_order, new_canalizing_variables), variables, depth + len(new_canalizing_variables),
                               number_layers + 1)
        else:  #or the recursion will end here (if self.f is non-canalizing)
        
            return (depth, number_layers, can_inputs, can_outputs, self, can_order)        

    def get_layer_structure(self):
        """
        Determine the canalizing layer structure of a Boolean function.

        This function decomposes a Boolean function into its canalizing layers (standard monomial form)
        by recursively identifying and removing conditionally canalizing variables.
        The output includes the canalizing depth, the number of layers, the canalizing inputs and outputs,
        the core function of the non-canalizing variables, and the order of the canalizing variables.

        Returns:
            - dict: A dictionary (self.properties) containing:
                - CanalizingDepth (int): Canalizing depth (number of conditionally canalizing variables).
                - NumberOfLayers (int): Number of distinct canalizing layers.
                - CanalizingInputs (np.array): Array of canalizing input values.
                - CanalizedOutputs (np.array): Array of canalized output values.
                - CoreFunction (np.array): The core function (truth table) after removing canalizing variables. Inputs: non-canalizing variables.
                - OrderOfCanalizingVariables (np.array): Array of indices representing the order of canalizing variables.
                - LayerStructure (np.array): Indicates the number of variables in each canalizing layer
                
        References:
            He, Q., & Macauley, M. (2016). Stratification and enumeration of Boolean functions by canalizing depth.
                Physica D: Nonlinear Phenomena, 314, 1-8.
            Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022). Revealing the canalizing structure of Boolean functions:
                Algorithms and applications. Automatica, 146, 110630.
        """
        if "CanalizingDepth" not in self.properties:
            dummy = dict(zip(["CanalizingDepth", "NumberOfLayers", "CanalizingInputs", "CanalizedOutputs", "CoreFunction", "OrderOfCanalizingVariables"],
                                            self._get_layer_structure(can_inputs=np.array([], dtype=int), can_outputs=np.array([], dtype=int),
                                                                      can_order=np.array([], dtype=int), variables=[], depth=0, number_layers=0)))
            dummy.update({'LayerStructure': get_layer_structure_from_canalized_outputs(dummy["CanalizedOutputs"])})
            self.properties.update(dummy)
            return dummy
        else:
            return {key: self.properties[key] for key in ["CanalizingDepth", "NumberOfLayers", "CanalizingInputs", "CanalizedOutputs", "CoreFunction", "OrderOfCanalizingVariables",'LayerStructure']}


    def get_canalizing_depth(self):
        """
        Returns the canalizing depth of the function.
        
        Returns:
            - int: The canalizing depth (number of conditionally canalizing variables).
        """
        if "CanalizingDepth" not in self.properties:
            self.get_layer_structure()
        return self.properties["CanalizingDepth"]

    
    def get_kset_canalizing_proportion(self, k):
        """
        Compute the proportion of k-set canalizing input sets for a Boolean function.

        For a given k, this function calculates the probability that a randomly chosen set of k inputs canalizes the function,
        i.e., forces the output regardless of the remaining variables.

        Parameters:
            - k (int): The size of the variable set (0 ≤ k ≤ n).

        Returns:
            - float: The proportion of k-set canalizing input sets.

        References:
            Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively canalizing Boolean functions.
            Advances in Applied Mathematics, 145, 102475.
        """
        assert type(k)==int and 0<=k<=self.n, "k must be an integer and satisfy 0 <= k <= degree n"
        
        if k == 0:
            return float(self.is_constant())
        desired_value = 2**(self.n - k)
        T = self._get_left_side_of_truth_table().T
        Tk = list(itertools.product([0, 1], repeat=k))
        A = np.r_[T, 1 - T]
        Ak = []
        for indices in itertools.combinations(range(self.n), k):
            for canalizing_inputs in Tk:
                indices_values = np.array(indices) + self.n * np.array(canalizing_inputs)
                dummy = np.sum(A[indices_values, :], 0) == k
                if sum(dummy) == desired_value:
                    Ak.append(dummy)
        Ak = np.array(Ak)
        is_there_canalization = np.isin(np.dot(Ak, self.f), [0, desired_value])
        return sum(is_there_canalization) / len(is_there_canalization)

    def is_kset_canalizing(self, k):
        """
        Determine if a Boolean function is k-set canalizing.

        A Boolean function is k-set canalizing if there exists a set of k variables such that setting these variables to specific values
        forces the output of the function, irrespective of the other n - k inputs.

        Parameters:
            - k (int): The size of the variable set (with 0 ≤ k ≤ n).

        Returns:
            - bool: True if f is k-set canalizing, False otherwise.

        References:
            Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively canalizing Boolean functions.
            Advances in Applied Mathematics, 145, 102475.
        """
        return self.get_kset_canalizing_proportion(k)>0


    def get_canalizing_strength(self):
        """
        Compute the canalizing strength of a Boolean function via exhaustive enumeration.

        The canalizing strength is defined as a weighted average of the proportions of k-set canalizing inputs for k = 1 to n-1.
        It is 0 for minimally canalizing functions (e.g., Boolean parity functions) and 1 for maximally canalizing functions
        (e.g., nested canalizing functions with one layer).

        Returns:
            - tuple:
                - float: The canalizing strength of f.
                - list: A list of the k-set canalizing proportions for k = 1, 2, ..., n-1.

        References:
            Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively canalizing Boolean functions.
            Advances in Applied Mathematics, 145, 102475.
        """
        if self.n==1:
            print("Warning:\nCanalizing strength is only properly defined for Boolean functions with n > 1 inputs. Returned 1 for n==1.")
            return 1.0
        res = []
        for k in range(1, self.n):
            res.append(self.get_kset_canalizing_proportion(k))
        return np.mean(np.multiply(res, 2**np.arange(1, self.n) / (2**np.arange(1, self.n) - 1))), res
    
    def get_input_redundancy(self):
        """
        .. attention::
            This method requires an installation of CANA. See :any:`Extended Functionality <installation>` for more information.

        Compute the input redundancy of a Boolean function.

        The input redundancy quantifies how many inputs are not required to determine the function’s output.
        Constant functions have an input redundancy of 1 (none of the inputs are needed), whereas parity functions have an input redundancy of 0 (all inputs are necessary).

        Returns:
            - float: Normalized input redundancy in the interval [0, 1].

        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        if __LOADED_CANA__:
            return self.to_cana().input_redundancy()
        print('The method \'get_input_redundancy\' requires the module cana, which cannot be found. Ensure it is installed to use this functionality.')
        return None
    
    def get_edge_effectiveness(self):
        """
        .. attention::
            This method requires an installation of CANA. See :any:`Extended Functionality <installation>` for more information.

        Compute the edge effectiveness for each regulator of a Boolean function.

        Edge effectiveness measures how much flipping a given input (regulator) influences the output.
        Non-essential inputs have an effectiveness of 0, whereas inputs that always flip the output when toggled have an effectiveness of 1.
        
        Returns:
            - list: A list of n floats in [0, 1] representing the edge effectiveness for each input.

        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        if __LOADED_CANA__:
            return self.to_cana().edge_effectiveness()
        print('The method \'get_edge_effectiveness\' requires the module cana, which cannot be found. Ensure it is installed to use this functionality.')
        return None

    def get_effective_degree(self):
        """
        .. attention::
            This method requires an installation of CANA. See :any:`Extended Functionality <installation>` for more information.

        Compute the effective degree, i.e., the sum of the edge effectivenesses of each regulator, of a Boolean function.

        Edge effectiveness measures how much flipping a given input (regulator) influences the output.
        Non-essential inputs have an effectiveness of 0, whereas inputs that always flip the output when toggled have an effectiveness of 1.

        Returns:
            - list: A value in [0, 1] representing the effective degree for each input.

        References:
            [1] Marques-Pita, M., & Rocha, L. M. (2013). Canalization and control in automata networks: body segmentation in Drosophila melanogaster. PloS One, 8(3), e55946.
            [2] Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018). CANA: a python package for quantifying control and canalization in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        if __LOADED_CANA__:
            return sum(self.get_edge_effectiveness())
        print('The method \'get_effective_degree\' requires the module cana, which cannot be found. Ensure it is installed to use this functionality.')
        return None
    