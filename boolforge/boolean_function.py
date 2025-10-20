#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:03:49 2025

@author: Benjamin Coberly, Claus Kadelka
"""

import numpy as np
import pandas as pd
import itertools
import math
from pyeda.inter import exprvar, Or, And, Not, espresso_exprs
from pyeda.boolalg.expr import OrOp, AndOp, NotOp, Complement

from typing import Union
from typing import Optional

try:
    import boolforge.utils as utils
except ModuleNotFoundError:
    import utils
    
try:
    import cana.boolean_node
    __LOADED_CANA__=True
except ModuleNotFoundError:
    print("The module cana cannot be found. Ensure it is installed to use all functionality of this toolbox.")
    __LOADED_CANA__=False
    
try:
    from numba import njit
    __LOADED_NUMBA__=True
except ModuleNotFoundError:
    print('The module numba cannot be found. Ensure it is installed to increase the run time of critical code in this toolbox.')
    __LOADED_NUMBA__=False

if __LOADED_NUMBA__:
    @njit
    def _is_degenerated_numba(f: np.ndarray, n: int) -> bool:
        """
        Numba-accelerated check for non-essential variables in a Boolean function.
        """
        N = 1 << n  # 2**n
        for i in range(n):
            stride = 1 << (n - 1 - i)
            step = stride << 1  # 2 * stride
            depends_on_i = False
            # Iterate in blocks that differ only in bit i
            for base in range(0, N, step):
                for offset in range(stride):
                    if f[base + offset] != f[base + offset + stride]:
                        depends_on_i = True
                        break
                if depends_on_i:
                    break
            if not depends_on_i:
                return True  # found non-essential variable
        return False

def display_truth_table(*functions: "BooleanFunction", labels = None):
    """
    Display the full truth table of a BooleanFunction in a formatted way.

    Each row shows the input combination (x1, x2, ..., xn)
    and the corresponding output(s) f(x).

    **Parameters:**
    
        - \\*functions (BooleanFunction): One or more BooleanFunction objects.
        - labels (list[str], optional): Column labels for each function
          (defaults to f1, f2, ...).

    **Example:**
    
        >>> f = BooleanFunction("(x1 & ~x2) | x3")
        >>> display_truth_table(f)
    """
    if not functions:
        raise ValueError("Please provide at least one BooleanFunction.")
    n = functions[0].n
    if any(f.n != n for f in functions):
        raise ValueError("All BooleanFunction objects must have the same number of variables.")
    f = functions[0]
    if isinstance(labels,str):
        labels = [labels]
    if labels is not None and len(labels)!=len(functions):
        raise ValueError("The number of labels (if provided) must equal the number of functions.")
        
    if np.all([np.all(f.variables == g.variables) for g in functions]):
        header = "\t".join([f.variables[i] for i in range(f.n)]) 
    else:
        header = "\t".join([f'x{i+1}' for i in range(f.n)]) 
    if labels is None:
        labels = [f"f{i}" for i in range(len(functions))]
    header += '\t|\t' + '\t'.join(labels)
    
    print(header)
    print("-" * len(header.expandtabs()))

    for inputs, outputs in zip(utils.get_left_side_of_truth_table(f.n), np.c_[*[f.f for f in functions]]):
        inputs_str = "\t".join(map(str, inputs))
        outputs_str = "\t".join(map(str, outputs))
        print(f"{inputs_str}\t|\t{outputs_str}")

def get_layer_structure_from_canalized_outputs(can_outputs : list) -> list:
    """
    Composes the layer structure when given canalized outputs.
    
    **Parameters**:
        
        - can_outputs (list[int] | np.array[int]): Array of canalized output
          values.
    
    **Returns**:
        
        - list[int]: The composed layer structure for the provided canalized
          outputs.
    """
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

    **Constructor Parameters:**
        
        - f (list[int] | np.array[int] | str): A list of length 2^n
          representing the outputs of a Boolean function with n inputs, or a
          string that can be properly evaluated, see utils.f_from_expression.
        
        - name (str, optional): The name of the node regulated by the Boolean
          function (default '').
        
    **Members:**
        
        - f (np.array[int]): A numpy array of length 2^n representing the
          outputs of a Boolean function with n inputs.
          
        - n (int): The number of inputs for the Boolean function.
        - variables (np.array[str]): A numpy array of n strings with variable
          names, default x0, ..., x_{n-1}.
          
        - name (str): The name of the node regulated by the Boolean function
          (default '').
          
        - properties (dict[str:Variant]): Dynamically created dictionary with
          additional information about the function (canalizing layer
          structure, type of inputs, etc.).
    """
    
    __slots__ = ['f','n','variables','name','properties']
    
    def __init__(self, f : Union[list, np.array, str], name : str = ""):
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
    def from_cana(cls, cana_BooleanNode : "cana.boolean_node.BooleanNode")-> "BooleanFunction":
        """
        **Compatability Method:**
        
            Converts an instance of cana.boolean_node.BooleanNode from the
            cana module into a Boolforge BooleanFunction object.
        
        **Returns**:
            
                - A BooleanFunction object.
        """
        return cls(np.array(cana_BooleanNode.outputs,dtype=int))

    def __str__(self):
        return f"{self.f}"
        #return f"{self.f.tolist()}"
        
    def __add__(self,value):
        if isinstance(value, int):
            return self.__class__((self.f + value) % 2)
        elif isinstance(value, BooleanFunction):
            return self.__class__((self.f + value.f) % 2)
        
    def to_polynomial(self) -> str:
        """
        Returns the Boolean function converted into polynomial format in
        non-reduced DNF.
        
        **Returns**:
            
            - str: Polynomial format in non-reduced DNF of the Boolean function.
        """
        return utils.bool_to_poly(self.f,variables=self.variables)

    def to_truth_table(self,RETURN=True,filename=None):
        """
        Returns or saves the full truth table of the Boolean function as a pandas DataFrame.
    
        Each row shows the input combination (x1, x2, ..., xn)
        and the corresponding output f(x).
    
        **Parameters**
            - RETURN (bool, optional): Whether to return the DataFrame (default: True).
              If False, the function only writes the table to file when `filename` is provided.
            - filename (str, optional): File name (including extension) to which the truth table
              should be saved. Supported formats are 'csv', 'xls', and 'xlsx'.
              If provided, the truth table is automatically saved in the specified format.
    
        **Returns**
            - pd.DataFrame: The full truth table, if `RETURN=True`.
              Otherwise, nothing is returned.
    
        **Example**
            >>> f = BooleanFunction("(x1 & ~x2) | x3")
            >>> f.to_truth_table()
                x1  x2  x3  f
            0    0   0   0  0
            1    0   0   1  1
            2    0   1   0  0
            3    0   1   1  1
            4    1   0   0  1
            5    1   0   1  1
            6    1   1   0  0
            7    1   1   1  1
    
        **Notes**
            - The column names correspond to the function's variables followed by its name.
            - When saving to a file, the file extension determines the format.
        """
        
        columns = np.append(self.variables,self.name if self.name != '' else 'f')
        truth_table = pd.DataFrame(np.c_[utils.get_left_side_of_truth_table(self.n),self.f],
                                   columns=columns)
        if filename is not None:
            ending = filename.split('.')[-1]
            assert ending in ['csv','xls','xlsx'],"filename must end in 'csv','xls', or 'xlsx'"
            if ending == 'csv':
                truth_table.to_csv(filename)
            else:
                truth_table.to_excel(filename)
        if RETURN:
            return truth_table
    
    def __repr__(self):
        if self.n < 6:
            return f"{type(self).__name__}(f={self.f.tolist()})"
        else:
            return f"{type(self).__name__}(f={self.f})"
    
    def __len__(self):
        return 2**self.n

    def __getitem__(self, index):
        try:
            return int(self.f[index])
        except TypeError:
            return self.f[index]

    def __setitem__(self, index, value):
        self.f[index] = value
    
    def to_cana(self) -> "cana.boolean_node.BooleanNode":
        """
        **Compatability method:**
            
            Returns an instance of cana.boolean_node.BooleanNode from the
            cana module.

        **Returns:**
            
            - An instance of cana.boolean_node.BooleanNode.
        """
        if __LOADED_CANA__:
            return cana.boolean_node.BooleanNode(k=self.n, outputs=self.f)
        print('The method \'to_cana_BooleanNode\' requires the module cana, which cannot be found. Ensure it is installed to use this functionality.')
        return None
    
    def to_logical(self, AND : str = '&', OR : str = '|', NOT : str = '!',
        MINIMIZE_EXPRESSION : bool = True) -> str:
        """
        Transform a Boolean function from truth table format to logical expression format.

        **Parameters:**
            
            - AND (str, optional): Character(s) to use for the And operator.
              Defaults to '&'.
            
            - OR (str, optional): Character(s) to use for the Or operator. Defaults
              to '|'.
            
            - NOT (str, optional): Character(s) to use for the Not operator.
              Defaults to '!'.
            
            - MINIMIZE_EXPRESSION (bool, optional): Whether or not to minimize
              the expression using Espresso. If true, minimizes the expression.
              If false, keeps the expression in DNF form. Defaults to true.

        **Returns:**
            
            - str: A string representing the Boolean function.
        """
        variables = [ exprvar(str(var)) for var in self.variables ]
        minterms = [ i for i in range(2**self.n) if self.f[i] ]
        terms = []
        for m in minterms:
            bits = [(variables[i] if (m >> (self.n - 1 - i)) & 1 else ~variables[i]) for i in range(self.n)]
            terms.append(And(*bits))
        func_expr = Or(*terms).to_dnf()
        if func_expr == ZERO: #TODO
            return '0'
        if MINIMIZE_EXPRESSION:
            func_expr, = espresso_exprs(func_expr)
        def __pyeda_to_string__(e):
            if isinstance(e, OrOp):
                return '('+(")%s("%OR).join(__pyeda_to_string__(arg) for arg in e.xs)+')'
            elif isinstance(e, AndOp):
                return AND.join(__pyeda_to_string__(arg) for arg in e.xs)
            elif isinstance(e, (NotOp)):
                return "%s(%s)"%(NOT, __pyeda_to_string__(e.x))
            elif isinstance(e, Complement):
                return "(%s%s)"%(NOT, str(e)[1::])
            return str(e)
        return __pyeda_to_string__(func_expr)
    
    def get_hamming_weight(self) -> int:
        """
        Calculate the number of non-zero bits in the bit vector representing
        a Boolean function.
        
        **Returns:**
        
            - int: The number of non-zero bits in the bit vector.
        """
        return int(self.f.sum())
    
    def is_constant(self) -> bool:
        """
        Check whether a Boolean function is constant.

        **Returns:**
            
            - bool: True if f is constant (all outputs are 0 or all are 1),
              False otherwise.
        """
        return np.all(self.f == self.f[0])
    
    if __LOADED_NUMBA__:
        def is_degenerated(self) -> bool:
            """
            Determine if a Boolean function contains non-essential variables.
            Numba-accelerated version.
            """
            f = np.asarray(self.f, dtype=np.uint8)
            return _is_degenerated_numba(f, self.n)
    
    else:
        def is_degenerated(self) -> bool:
            """
            Determine if a Boolean function contains non-essential variables.
    
            A variable is non-essential if the function's output does not depend
            on it.
    
            **Returns:**
                
                - bool: True if f contains at least one non-essential variable,
                  False if all variables are essential.
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
    
    def get_essential_variables(self) -> list:
        """
        Determine the indices of essential variables in a Boolean function.

        A variable is essential if changing its value (while holding the others
        constant) can change the output of f.

        **Returns:**
            
            - list[int]: List of indices corresponding to the essential variables.
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

    def get_number_of_essential_variables(self) -> int:
        """
        Count the number of essential variables in a Boolean function.

        **Returns:**
            
            - int: The number of essential variables.
        """
        return len(self.get_essential_variables())
    
    
    def get_type_of_inputs(self) -> np.ndarray:
        """
        Determine for each input of the Boolean function whether it is
        positive, negative, conditional or non-essential.

        **Returns:**
            
            - np.ndarray[str]: The type of each input of the Boolean function.
        """

        if 'InputTypes' in self.properties:
            return self.properties['InputTypes']
    
        f = np.asarray(self.f, dtype=np.int8)
        n = self.n
        types = np.empty(n, dtype=object)
    
        # Compute all pairwise differences for each bit position simultaneously
        # Each variable toggles every 2**i entries in the truth table.
        for i in range(n):
            period = 2 ** (i + 1)
            half = period // 2
            # Vectorized reshape pattern: consecutive blocks of 0...1 transitions
            f_reshaped = f.reshape(-1, period)
            diff = f_reshaped[:, half:] - f_reshaped[:, :half]
            min_diff = diff.min()
            max_diff = diff.max()
            if min_diff == 0 and max_diff == 0:
                types[i] = 'non-essential'
            elif min_diff == -1 and max_diff == 1:
                types[i] = 'conditional'
            elif min_diff >= 0 and max_diff == 1:
                types[i] = 'positive'
            elif min_diff == -1 and max_diff <= 0:
                types[i] = 'negative'
    
        types = np.array(types, dtype=str)
        self.properties['InputTypes'] = types
        return types

    def is_monotonic(self) -> bool:
        """
        Determine if a Boolean function is monotonic.

        A Boolean function is monotonic if it is monotonic in each variable. 
        That is, if for all i=1,...,n: f(x_1, ..., x_i=0, ..., x_n) >= f(x_1,
        ..., x_i=1, ..., x_n) for all (x_1, ..., x_n) or f(x_1, ..., x_i=0,
        ..., x_n) <= f(x_1, ..., x_i=1, ..., x_n) for all (x_1, ..., x_n).

        **Returns:**
            
            - bool: True if f contains no conditional variables, False if at
              least one variable is conditional.
        """            
        return 'conditional' not in self.get_type_of_inputs()
    
    
    def get_symmetry_groups(self) -> list:
        """
        Determine all symmetry groups of input variables for a Boolean function.

        Two variables are in the same symmetry group if swapping their values
        does not change the output of the function for any input of the other
        variables.

        **Returns:**
            
            - list[list[int]]: A list of lists where each inner list contains
              indices of variables that form a symmetry group.
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
                for ii, x in enumerate(utils.get_left_side_of_truth_table(self.n)):
                    if x[i] != x[j] and x[i] == 0 and self.f[ii] != self.f[ii + diff]:
                        break
                else:
                    left_to_check[j] = 0
                    symmetry_groups[-1].append(j)
        return symmetry_groups
    
    def get_absolute_bias(self) -> float:
        """
        Compute the absolute bias of a Boolean function.

        The absolute bias is defined as `|(self.get_hamming_weight() /
        2^(n-1)) - 1|`, which quantifies how far the function's output
        distribution deviates from being balanced.

        **Returns:**
            
            - float: The absolute bias of the Boolean function.
        """
        return abs(self.get_hamming_weight() * 1.0 / 2**(self.n - 1) - 1)


    def get_activities(self, 
                       nsim : int = 10000, 
                       EXACT : bool = False, 
                       *, 
                       rng = None) -> np.array:
        """
        Compute the activities of all variables of a Boolean function.

        This function can compute the activities  by exhaustively iterating
        over all inputs (if EXACT is True) or estimate it via Monte Carlo sampling 
        (if EXACT is False).

        **Parameters:**
            
            - nsim (int, optional): Number of random samples (default is 10000,
              used when EXACT is False).
            
            - EXACT (bool, optional): If True, compute the exact sensitivity by
              iterating over all inputs; otherwise, use sampling (default).
              
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - np.array(float): The activities of the variables of the Boolean function.
        """        
        size_state_space = 2**self.n
        activities = np.zeros(self.n,dtype=np.float64)
        if EXACT:
            # Compute all integer representations of inputs (0 .. 2^n - 1)
            X = np.arange(size_state_space, dtype=np.uint32)
        
            # For each bit position i, flipping that bit corresponds to XOR with (1 << self.n-1-i)
            for i in range(self.n):
                flipped = X ^ (1 << self.n-1-i) 
                activities[i] = np.count_nonzero(self.f != self.f[flipped])
            return activities / size_state_space
        else:
            rng = utils._coerce_rng(rng)

            random_states = rng.integers(0,size_state_space,nsim) #
            for i in range(self.n):
                flipped_random_states = random_states ^ (1 << self.n-1-i) 
                activities[i] = np.count_nonzero(self.f[random_states] != self.f[flipped_random_states])
            return activities / nsim
    
    
    def get_average_sensitivity(self, nsim : int = 10000, EXACT : bool = False,
        NORMALIZED : bool = True, *, rng = None) -> float:
        """
        Compute the average sensitivity of a Boolean function.

        The average sensitivity is equivalent to the Derrida value D(F,1) when
        the update rule is sampled from the same space. This function can
        compute the exact sensitivity by exhaustively iterating over all inputs
        (if EXACT is True) or estimate it via Monte Carlo sampling (if EXACT
        is False). The result can be normalized by the number of inputs.

        **Parameters:**
            
            - nsim (int, optional): Number of random samples (default is 10000,
              used when EXACT is False).
            
            - EXACT (bool, optional): If True, compute the exact sensitivity by
              iterating over all inputs; otherwise, use sampling (default).
              
            - NORMALIZED (bool, optional): If True, return the normalized
              sensitivity (divided by the number of function inputs);
              otherwise, return the total count.
              
            - rng (None, optional): Argument for the random number generator,
              implemented in 'utils._coerce_rng'.

        **Returns:**
            
            - float: The (normalized) average sensitivity of the Boolean function.
        """        
        activities = self.get_activities(nsim,EXACT,rng=rng)
        s = sum(activities)
        if NORMALIZED:
            return s / self.n
        else:
            return s
    

    def is_canalizing(self) -> bool:
        """
        Determine if a Boolean function is canalizing.

        A Boolean function f(x_1, ..., x_n) is canalizing if there exists at
        least one variable x_i and a value a ∈ {0, 1} such that f(x_1, ...,
        x_i = a, ..., x_n) is constant.

        **Returns:**
            
            - bool: True if f is canalizing, False otherwise.
        """
        indices = np.arange(2**self.n, dtype=np.uint32)
    
        # Iterate over each variable
        for i in range(self.n):
            mask = 1 << i #really this should be 1 << self.n-1-i but it's symmetric and faster as is
            bit_is_0 = (indices & mask) == 0
            bit_is_1 = ~bit_is_0
    
            # Restrict outputs where x_i = 0 or x_i = 1
            f0 = self.f[bit_is_0]
            f1 = self.f[bit_is_1]
    
            # If any restriction is constant, function is canalizing
            if np.all(f0 == f0[0]) or np.all(f1 == f1[0]):
                return True
    
        return False
    
    def is_k_canalizing(self, k : int) -> bool:
        """
        Determine if a Boolean function is k-canalizing.

        A Boolean function is k-canalizing if it has at least k conditionally
        canalizing variables. This is checked recursively: after fixing a
        canalizing variable (with a fixed canalizing input that forces the
        output), the subfunction must itself be canalizing for
        the next variable, and so on.

        **Parameters:**
            
            - k (int): The desired canalizing depth (0 ≤ k ≤ n).
              Note: every function is 0-canalizing.

        **Returns:**
            
            - bool: True if f is k-canalizing, False otherwise.

        **References:**
            
            #. He, Q., & Macauley, M. (2016). Stratification and enumeration of
               Boolean functions by canalizing depth. Physica D: Nonlinear
               Phenomena, 314, 1-8.
            
            #. Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D.
               (2022). Revealing the canalizing structure of Boolean functions:
               Algorithms and applications. Automatica, 146, 110630.
        """

        # Base cases
        if k > self.n:
            return False
        if k == 0:
            return True
        if np.all(self.f == self.f[0]):  # constant function is by definition not canalizing
            return False
    
        # Precompute input indices for masking
        indices = np.arange(2**self.n, dtype=np.uint32)
    
        # Try each variable to see if it is canalizing
        for i in range(self.n):
            mask = 1 << i #really this should be 1 << self.n-1-i but it's symmetric and faster as is
            bit_is_0 = (indices & mask) == 0
            bit_is_1 = ~bit_is_0
    
            f0, f1 = self.f[bit_is_0], self.f[bit_is_1]
    
            # Case 1: x_i = 0 is canalizing
            if np.all(f0 == f0[0]):
                if k == 1:
                    return True
                # recurse on subfunction with x_i fixed to 0 → drop that variable
                return BooleanFunction(f1).is_k_canalizing(k - 1)
    
            # Case 2: x_i = 1 is canalizing
            elif np.all(f1 == f1[0]):
                if k == 1:
                    return True
                return BooleanFunction(f0).is_k_canalizing(k - 1)
        return False


    def _get_layer_structure(self, can_inputs, can_outputs, can_order,
                             variables, depth, number_layers):
        """
        Faster internal version of _get_layer_structure using bitwise operations.
        """

        # base cases
        if np.all(self.f == self.f[0]):
            # recursion ends when function becomes constant
            return depth, number_layers, can_inputs, can_outputs, self, can_order
    
        if not variables:
            variables = list(range(self.n))
        elif isinstance(variables, np.ndarray):
            variables = variables.tolist()
    
        indices = np.arange(2**self.n, dtype=np.uint32)
    
        # candidate canalizing variables (x_i, a)
        new_canalizing_vars = []
        new_can_inputs = []
        new_can_outputs = []
        new_f = None
    
        for i in range(self.n):
            mask = 1 << (self.n-1-i)
            bit0 = (indices & mask) == 0
            bit1 = ~bit0
            f0, f1 = self.f[bit0], self.f[bit1]
    
            # check both possible canalizing directions
            if np.all(f0 == f0[0]):
                new_canalizing_vars.append(variables[i])
                new_can_inputs.append(0)
                new_can_outputs.append(int(f0[0]))
            elif np.all(f1 == f1[0]):
                new_canalizing_vars.append(variables[i])
                new_can_inputs.append(1)
                new_can_outputs.append(int(f1[0]))
    
        if not new_canalizing_vars:
            # non-canalizing core function
            return depth, number_layers, can_inputs, can_outputs, self, can_order
    
        # reduce variable list (remove canalizing ones)
        indices_new_canalizing_vars = [i for i,v in enumerate(variables) if v in new_canalizing_vars]
        remaining_vars = [v for v in variables if v not in new_canalizing_vars]
    
        # build the restricted subfunction (“core function”)
        # start with all indices, then keep those where none of the canalizing
        # variables take their canalizing inputs
        mask_keep = np.ones(2**self.n, dtype=bool)
        for var, val in zip(indices_new_canalizing_vars, new_can_inputs):
            bitmask = (indices >> (self.n-1-var)) & 1
            mask_keep &= (bitmask != val)
        new_f = self.f[mask_keep]
    

        # recurse on reduced function
        new_bf = self.__class__(list(new_f))
        return new_bf._get_layer_structure(
            np.append(can_inputs, new_can_inputs),
            np.append(can_outputs, new_can_outputs),
            np.append(can_order, new_canalizing_vars),
            remaining_vars,
            depth + len(new_canalizing_vars),
            number_layers + 1,
        )


    def get_layer_structure(self) -> dict:
        """
        Determine the canalizing layer structure of a Boolean function.

        This function decomposes a Boolean function into its canalizing layers
        (standard monomial form) by recursively identifying and removing
        conditionally canalizing variables. The output includes the canalizing
        depth, the number of layers, the canalizing inputs and outputs, the
        core function of the non-canalizing variables, and the order of the
        canalizing variables.

        **Returns:**
            
            - dict: A dictionary (self.properties) containing:
                
                - CanalizingDepth (int): Canalizing depth (number of
                  conditionally canalizing variables).
                
                - NumberOfLayers (int): Number of distinct canalizing layers.
                - CanalizingInputs (np.array[int]): Array of canalizing input
                  values.
                  
                - CanalizedOutputs (np.array[int]): Array of canalized output
                  values.
                  
                - CoreFunction (BooleanFunction): The core function (truth
                  table) after removing canalizing variables. Inputs:
                  non-canalizing variables.
                  
                - OrderOfCanalizingVariables (np.array[int]): Array of indices
                  representing the order of canalizing variables.
                  
                - LayerStructure (np.array[int]): Indicates the number of
                  variables in each canalizing layer.
                
        **References:**
            
            #. He, Q., & Macauley, M. (2016). Stratification and enumeration
               of Boolean functions by canalizing depth. Physica D: Nonlinear
               Phenomena, 314, 1-8.
               
            #. Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D.
               (2022). Revealing the canalizing structure of Boolean functions:
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


    def get_canalizing_depth(self) -> int:
        """
        Returns the canalizing depth of the function.
        
        **Returns:**
            
            - int: The canalizing depth (number of conditionally canalizing
              variables).
        """
        if "CanalizingDepth" not in self.properties:
            self.get_layer_structure()
        return self.properties["CanalizingDepth"]

    
    def get_kset_canalizing_proportion(self, k : int) -> float:
        """
        Compute the proportion of k-set canalizing input sets for a Boolean
        function.

        For a given k, this function calculates the probability that a randomly
        chosen set of k inputs canalizes the function, i.e., forces the output
        regardless of the remaining variables.

        **Parameters:**
            
            - k (int): The size of the variable set (0 ≤ k ≤ n).

        **Returns:**
            
            - float: The proportion of k-set canalizing input sets.

        **References:**
            
            #. Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively
               canalizing Boolean functions. Advances in Applied Mathematics,
               145, 102475.
        """
        assert type(k)==int and 0<=k<=self.n, "k must be an integer and satisfy 0 <= k <= degree n"
        
        # trivial case
        if k == 0:
            return float(self.is_constant())
        
        # precompute binary representation of all inputs
        #indices = np.arange(2**self.n, dtype=np.uint32)
        #bits = ((indices[:, None] >> np.arange(self.n)) & 1).astype(np.uint8)  # shape (2**n, n)
        left_side_of_truth_table = utils.get_left_side_of_truth_table(self.n)
        
        total_tests = 0
        canalizing_hits = 0
    
        # iterate over variable subsets of size k
        for subset in itertools.combinations(range(self.n), k):
            Xsub = left_side_of_truth_table[:, subset]  # shape (2**n, k)
            # For each possible assignment to this subset
            for assignment in itertools.product([0, 1], repeat=k):
                mask = np.all(Xsub == assignment, axis=1)
                if not np.any(mask):
                    continue
                # If all outputs equal when these vars are fixed → canalizing
                f_sub = self.f[mask]
                if np.all(f_sub == f_sub[0]):
                    canalizing_hits += 1
                total_tests += 1
    
        return canalizing_hits / total_tests if total_tests > 0 else 0.0


    def get_kset_canalizing_proportion_of_variables(self, k : int) -> float:
        """
        Compute the proportion of k-set canalizing input sets that contain a specific variable.

        For a given k, this function calculates the probability that a randomly
        chosen set of k inputs (including a specific variable) canalizes the function,
        i.e., forces the output regardless of the remaining variables.

        **Parameters:**
            
            - k (int): The size of the variable set (0 ≤ k ≤ n).

        **Returns:**
            
            - float: The proportion of k-set canalizing input sets.

        **References:**
            
            #. Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively
               canalizing Boolean functions. Advances in Applied Mathematics,
               145, 102475.
        """
        assert type(k)==int and 0<=k<=self.n, "k must be an integer and satisfy 0 <= k <= degree n"
        
        # trivial case
        if k == 0:
            return float(self.is_constant())
        
        # precompute binary representation of all inputs
        #indices = np.arange(2**self.n, dtype=np.uint32)
        #bits = ((indices[:, None] >> np.arange(self.n)) & 1).astype(np.uint8)  # shape (2**n, n)
        left_side_of_truth_table = utils.get_left_side_of_truth_table(self.n)
        
        canalizing_hits = np.zeros(self.n,dtype=np.float64)
        
        # iterate over variable subsets of size k
        for subset in itertools.combinations(range(self.n), k):
            Xsub = left_side_of_truth_table[:, subset]  # shape (2**n, k)
            subset = np.array(subset)
            # For each possible assignment to this subset
            for assignment in itertools.product([0, 1], repeat=k):
                mask = np.all(Xsub == assignment, axis=1)
                if not np.any(mask):
                    continue
                # If all outputs equal when these vars are fixed → canalizing
                f_sub = self.f[mask]
                if np.all(f_sub == f_sub[0]):
                    canalizing_hits[subset] += 1
    
        return canalizing_hits / (k/self.n * math.comb(self.n,k) * 2**k)


    def is_kset_canalizing(self, k : int) -> bool:
        """
        Determine if a Boolean function is k-set canalizing.

        A Boolean function is k-set canalizing if there exists a set of k
        variables such that setting these variables to specific values forces
        the output of the function, irrespective of the other n - k inputs.

        **Parameters:**
            
            - k (int): The size of the variable set (with 0 ≤ k ≤ n).

        **Returns:**
            
            - bool: True if f is k-set canalizing, False otherwise.

        **References:**
            
            #. Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively
               canalizing Boolean functions. Advances in Applied Mathematics,
               145, 102475.
        """
        return self.get_kset_canalizing_proportion(k)>0

    def get_canalizing_strength(self) -> tuple:
        """
        Compute the canalizing strength of a Boolean function via exhaustive
        enumeration.

        The canalizing strength is defined as a weighted average of the
        proportions of k-set canalizing inputs for k = 1 to n-1. It is 0 for
        minimally canalizing functions (e.g., Boolean parity functions) and 1
        for maximally canalizing functions (e.g., nested canalizing functions
        with one layer).

        **Returns:**
            
            - The canalizing strength of f.

        **References:**
            
            #. Kadelka, C., Keilty, B., & Laubenbacher, R. (2023). Collectively
               canalizing Boolean functions. Advances in Applied Mathematics,
               145, 102475.
        """
        if self.n==1:
            print("Warning:\nCanalizing strength is only properly defined for Boolean functions with n > 1 inputs. Returned 1 for n==1.")
            return 1.0
        res = []
        for k in range(1, self.n):
            res.append(self.get_kset_canalizing_proportion(k))
        return np.mean(np.multiply(res, 2**np.arange(1, self.n) / (2**np.arange(1, self.n) - 1)))


    def get_canalizing_strength_of_variables(self) -> tuple:
        """
        Compute the canalizing strength of each variable in a Boolean function 
        via exhaustive enumeration.

        The canalizing strength is defined as a weighted average of the
        proportions of k-set canalizing inputs for k = 1 to n-1. It is 0 for
        minimally canalizing functions (e.g., Boolean parity functions) and 1
        for maximally canalizing functions (e.g., nested canalizing functions
        with one layer).

        **Returns:**
            
            - np.array(float): The canalizing strength of each variable of f.

        """
        if self.n==1:
            print("Warning:\nCanalizing strength is only properly defined for Boolean functions with n > 1 inputs. Returned 1 for n==1.")
            return np.ones(1,dtype=np.float64)
        res = np.zeros((self.n-1,self.n))
        for k in range(1, self.n):
            res[k-1] = self.get_kset_canalizing_proportion_of_variables(k)
        multipliers = 2**np.arange(1, self.n) / (2**np.arange(1, self.n) - 1)
        return np.mean(res * multipliers[:, np.newaxis],0)
    
    
    
    def get_input_redundancy(self) -> Optional[float]:
        """
        .. important::
            This method requires an installation of CANA (See
            `Extended Functionality`_). If CANA is not found, this method will
            return None.
        .. _Extended Functionality: https://ckadelka.github.io/BoolForge/install.html#extended-functionality

        Compute the input redundancy of a Boolean function.

        The input redundancy quantifies how many inputs are not required to
        determine the function’s output. Constant functions have an input
        redundancy of 1 (none of the inputs are needed), whereas parity
        functions have an input redundancy of 0 (all inputs are necessary).

        **Returns:**
            
            - float: Normalized input redundancy in the interval [0, 1].

        **References:**
            
            #. Marques-Pita, M., & Rocha, L. M. (2013). Canalization and
               control in automata networks: body segmentation in Drosophila
               melanogaster. PloS One, 8(3), e55946.
               
            #. Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018).
               CANA: a python package for quantifying control and canalization
               in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        if __LOADED_CANA__:
            return self.to_cana().input_redundancy()
        print('The method \'get_input_redundancy\' requires the module cana, which cannot be found. Ensure it is installed to use this functionality.')
        return None
    
    def get_edge_effectiveness(self) -> Optional[list]:
        """
        .. important::
            This method requires an installation of CANA (See
            `Extended Functionality`_). If CANA is not found, this method will
            return None.
        .. _Extended Functionality: https://ckadelka.github.io/BoolForge/install.html#extended-functionality

        Compute the edge effectiveness for each regulator of a Boolean function.

        Edge effectiveness measures how much flipping a given input (regulator)
        influences the output. Non-essential inputs have an effectiveness of 0,
        whereas inputs that always flip the output when toggled have an
        effectiveness of 1.
        
        **Returns:**
            
            - list[float]: A list of n floats in [0, 1] representing the edge
              effectiveness for each input.

        **References:**
            
            #. Marques-Pita, M., & Rocha, L. M. (2013). Canalization and
               control in automata networks: body segmentation in Drosophila
               melanogaster. PloS One, 8(3), e55946.
               
            #. Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018).
               CANA: a python package for quantifying control and canalization
               in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        if __LOADED_CANA__:
            return self.to_cana().edge_effectiveness()
        print('The method \'get_edge_effectiveness\' requires the module cana, which cannot be found. Ensure it is installed to use this functionality.')
        return None

    def get_effective_degree(self) -> Optional[float]:
        """
        .. important::
            This method requires an installation of CANA (See
            `Extended Functionality`_). If CANA is not found, this method will
            return None.
        .. _Extended Functionality: https://ckadelka.github.io/BoolForge/install.html#extended-functionality
        
        Compute the effective degree, i.e., the sum of the edge effectivenesses
        of each regulator, of a Boolean function.

        Edge effectiveness measures how much flipping a given input (regulator)
        influences the output. Non-essential inputs have an effectiveness of 0,
        whereas inputs that always flip the output when toggled have an
        effectiveness of 1.

        **Returns:**
            
            - float: The sum of the edge effectiveness of each regulator.

        **References:**

            #. Marques-Pita, M., & Rocha, L. M. (2013). Canalization and
               control in automata networks: body segmentation in Drosophila
               melanogaster. PloS One, 8(3), e55946.
               
            #. Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018).
               CANA: a python package for quantifying control and canalization
               in Boolean networks. Frontiers in Physiology, 9, 1046.
        """
        if __LOADED_CANA__:
            return sum(self.get_edge_effectiveness())
        print('The method \'get_effective_degree\' requires the module cana, which cannot be found. Ensure it is installed to use this functionality.')
        return None
    



