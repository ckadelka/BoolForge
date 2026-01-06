# BoolForge Tutorial #2: Advanced concepts for Boolean functions

In this tutorial, we will further explore the `BooleanFunction` class — the foundation of BoolForge. You will learn how to compute advanced properties of Boolean functions such as:
- the symmetry among inputs, or 
- the activities and sensitivity of a Boolean function.


```python
import boolforge
import numpy as np
```

## Exploring symmetry groups of Boolean functions

Symmetries in Boolean functions are permutations of input variables that leave the function unchanged. They reveal redundancy, equivalence among variables, and often relate to biological modularity or robustness.


```python
import boolforge
# Example: a fully symmetric 3-variable Boolean function
f = boolforge.BooleanFunction('(x + y + z) % 2')

# Example: a partially symmetric 3-variable Boolean function
g = boolforge.BooleanFunction('x1 | (x2 & x3)')

# Example: a non-symmetric 3-variable Boolean function
h = boolforge.BooleanFunction('a | (b & ~c)')

labels = ['f','g','h']

boolforge.display_truth_table(f,g,h,labels=labels)
print()

for func,label in zip([f,g,h],labels):
    symmetry_groups = func.get_symmetry_groups()
    for i,symmetry_group in enumerate(symmetry_groups):
        print(f"Symmetry group {i+1} of {label}: {func.variables[np.array(symmetry_group)]}")
    print()
```

    x1	x2	x3	|	f	g	h
    -------------------------------------------------
    0	0	0	|	0	0	0
    0	0	1	|	1	0	0
    0	1	0	|	1	0	1
    0	1	1	|	0	1	0
    1	0	0	|	1	1	1
    1	0	1	|	0	1	1
    1	1	0	|	0	1	1
    1	1	1	|	1	1	1
    
    Symmetry group 1 of f: ['x' 'y' 'z']
    
    Symmetry group 1 of g: ['x1']
    Symmetry group 2 of g: ['x2' 'x3']
    
    Symmetry group 1 of h: ['a']
    Symmetry group 2 of h: ['b']
    Symmetry group 3 of h: ['c']
    


The first function `f` is fully symmetric. That is, all variables are part of the same symmetry group. 

The second function `g` is partially symmetric. While x2 and x3 can be interchanged without ever changing the function output, this is not the case for x1. 

The last function `h` has three symmetry groups. None of its variables can be interchanged without changing the function output.

## Activities and sensitivities

The *activity* of an input $x_i$ to a Boolean function $f$ describes how sensitive the function output is to changes in this input. That is, $$a(f,x_i) = \frac{1}{2^n}\sum_{\mathbf x\in \{0,1\}^n} f(\mathbf x) \neq f(\mathbf x \oplus e_i) \in [0,1],$$
where $e_i=(0,\ldots,0,1,0,\ldots,0)$ is the ith unit vector.

The *average sensitivity* of a Boolean function describes how sensitive its output is to changes in its inputs, specifically to a random single-bit flip. The average sensitivity is the sum of all its activities. That is, $$S(f) = \sum_{i=1}^n a(f,x_i) = \frac{1}{2^n}\sum_{\mathbf x\in \{0,1\}^n} \sum_{i=1}^n f(\mathbf x) \neq f(\mathbf x \oplus e_i) \in [0,n].$$
Division by $n$ yields the *normalized average sensitivity* $s(f)$, which can be readily compared between functions of different degree $n$:
$$s(f) = \frac {S(f)}n  \in [0,1].$$

To investigate how to compute the activtities and the average sensitivity in `BoolForge`, we work with the linear / parity function / XOR function `f` from above, as well as with the function `g`. By default, activities and sensitivities are computed by Monte Carlo Simulation (which is possible, even for functions with very large degree). To ensure an exact computation, we specify `EXACT=True`.


```python
EXACT = True
print('Activities of f:',f.get_activities(EXACT=EXACT))
print('Activities of g:',g.get_activities(EXACT=EXACT))
print('Normalized average sensitivity of f:',f.get_average_sensitivity(EXACT=EXACT))
print('Normalized average sensitivity of g:',g.get_average_sensitivity(EXACT=EXACT))
```

    Activities of f: [1. 1. 1.]
    Activities of g: [0.75 0.25 0.25]
    Normalized average sensitivity of f: 1.0
    Normalized average sensitivity of g: 0.4166666666666667


A single-bit change in `f` always changes its output, thus the normalized average sensitivity (normalized by default) of 1. On the other hand, only 75% of $x_1$ flips and 25% of $x_2$ or $x_3$ flips change the output of `g`. Thus, the normalized average sensitivity of `g` is $\frac 13*75\% + \frac 23 25\% = \frac{5}{12}$.

For functions of many inputs, we require `EXACT=False` (the default). Also, when generating such a function it not recommended to require that all inputs are essential, as (i) this is almost certainly the case anyways (the probability that an n-input function does not depend on input $x_i$ is given $1/2^{n-1}$), and (ii) checking for input degeneracy is NP-hard (i.e., very computationally expensive). We thus set `ALLOW_DEGENERATE_FUNCTIONS=True`. You find more on this in the next tutorial. 


```python
# Define a Boolean function with 25 inputs and compute activities and sensitivity by Monte Carlo simulation
EXACT = False
nsim = 500
n = 25
h = boolforge.random_function(n=n,ALLOW_DEGENERATE_FUNCTIONS=True) 
activities = h.get_activities(EXACT=EXACT)
print(f'Mean of all n={n} activities of h: {np.mean(activities)}')
print(f'Standard deviation of all n={n} activities of h: {np.std(activities)}')
print('Normalized average sensitivity of h:',h.get_average_sensitivity(EXACT=EXACT))
```

    Mean of all n=25 activities of h: 0.49751200000000007
    Standard deviation of all n=25 activities of h: 0.004486541652542627
    Sensitivity of h: 0.501072


This shows that for a random Boolean function, the expected value of its activities is 0.5, as is its normalized average sensitivity.
