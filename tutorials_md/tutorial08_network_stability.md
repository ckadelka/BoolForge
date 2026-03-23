# Perturbation and sensitivity analysis of Boolean networks

In this tutorial, we study how Boolean networks respond to perturbations.
Rather than implementing perturbations manually, we leverage BoolForge’s
built-in robustness and sensitivity measures.

## What you will learn
You will learn how to:

- quantify robustness and fragility of Boolean networks under synchronous update,
- interpret basin-level and attractor-level robustness measures,
- perform exact and approximate robustness computations, and
- compute Derrida values as a measure of dynamical sensitivity.

Together, these tools allow us to assess dynamical stability and resilience of 
Boolean network models in a principled and computationally efficient way.

## Setup

```python
import boolforge as bf
import pandas as pd
```

We reuse the small Boolean network from the previous tutorial as a running example.

```python
string = """
x = y
y = x OR z
z = y
"""

bn = bf.BooleanNetwork.from_string(string, separator="=")

print("Variables:", bn.variables)
print("Number of nodes:", bn.N)
```

    Variables: ['x' 'y' 'z']
    Number of nodes: 3


## Exact attractors and robustness measures

BoolForge provides a single method that computes:

- all attractors,
- basin sizes,
- overall network coherence and fragility,
- basin-level coherence and fragility, and
- attractor-level coherence and fragility.

These quantities are defined via systematic single-bit perturbations
in the Boolean hypercube and can be computed *exactly* for small networks.

```python
results_exact = bn.get_attractors_and_robustness_synchronous_exact()
for key in results_exact.keys():
    print(key)
```

    Attractors
    NumberOfAttractors
    BasinSizes
    AttractorID
    Coherence
    Fragility
    BasinCoherences
    BasinFragilities
    AttractorCoherences
    AttractorFragilities


For convenience, information about the dynamics (attractors, basin sizes, etc),
described in detail in the previous tutorial, is also returned by this method.

```python
print("Number of attractors:", results_exact["NumberOfAttractors"])
print("Attractors (decimal states):", results_exact["Attractors"])
print("Eventual attractor of each state:", results_exact["AttractorID"])

print("Basin sizes:", results_exact["BasinSizes"])
```

    Number of attractors: 3
    Attractors (decimal states): [[0], [2, 5], [7]]
    Eventual attractor of each state: [0 1 1 2 1 1 2 2]
    Basin sizes: [0.125 0.5   0.375]


## Network-, basin- and attractor-level robustness

Robustness can be resolved at different structural levels. Network-level metrics
report the average robustness of any network state when subjected to perturbation. 

```python
print("Overall coherence:", results_exact["Coherence"])
print("Overall fragility:", results_exact["Fragility"])
```

    Overall coherence: 0.3333333333333333
    Overall fragility: 0.3333333333333333


The same robustness metrics, coherence and fragility, can also be averaged
across a smaller set of states, e.g., all states in one basin of attraction, or
an even smaller set of states, e.g., all states that form an attractor.

```python
df_basins = pd.DataFrame({
    "BasinSizes": results_exact["BasinSizes"],
    "BasinCoherences": results_exact["BasinCoherences"],
    "BasinFragilities": results_exact["BasinFragilities"],
})

df_attractors = pd.DataFrame({
    "AttractorCoherences": results_exact["AttractorCoherences"],
    "AttractorFragilities": results_exact["AttractorFragilities"],
})

print("Basin-level robustness:")
print(df_basins)

print("Attractor-level robustness:")
print(df_attractors)
```

    Basin-level robustness:
       BasinSizes  BasinCoherences  BasinFragilities
    0       0.125         0.000000          0.500000
    1       0.500         0.333333          0.333333
    2       0.375         0.444444          0.277778
    Attractor-level robustness:
       AttractorCoherences  AttractorFragilities
    0             0.000000              0.500000
    1             0.333333              0.333333
    2             0.666667              0.166667


Interpretation:

- **Coherence** measures the fraction of single-bit perturbations that do *not*
  change the final attractor.
- **Fragility** measures how much the attractor state changes.

The robustness metrics considered thus far describe how a single perturbation affects
the network dynamics in the long-term, i.e., at the attractor. 
These metrics are very meaningful biologically because attractors typically 
correspond to cell types of phenotypes.

It turns out that attractors in biological networks are often less stable 
than their basins, a phenomenon explored in detail in Tutorial 10.


## Approximate robustness for larger networks

For larger networks, exact enumeration of all $2^N$ states is infeasible.
BoolForge therefore provides a Monte Carlo approximation that samples
random initial conditions and perturbations.

```python
results_approx = bn.get_attractors_and_robustness_synchronous(n_simulations=500)

print("Number of attractors (lower bound):", results_approx["NumberOfAttractorsLowerBound"])
print("Approximate coherence:", results_approx["CoherenceApproximation"])
print("Approximate fragility:", results_approx["FragilityApproximation"])
```

    Number of attractors (lower bound): 3
    Approximate coherence: 0.352
    Approximate fragility: 0.324


Even when only using 500 random initial states, the approximate values closely match the exact ones.
For larger networks, these approximations are often the only feasible option.

## Derrida value: dynamical sensitivity

An older and very popular robustness metric, the Derrida value, 
measures how perturbations *propagate* after one synchronous update.
It is defined as the expected Hamming distance between updated states that initially
differed in exactly one bit. 

BoolForge includes routines for the exact calculation and estimation of Derrida values.
For networks with low degree, the exact calculation is strongly preferable. It is faster and more accurate.

```python
derrida_exact = bn.get_derrida_value(exact=True)
derrida_approx = bn.get_derrida_value(n_simulations=2000)

print("Exact Derrida value:", derrida_exact)
print("Approximate Derrida value:", derrida_approx)
```

    Exact Derrida value: 1.0
    Approximate Derrida value: 1.0


Interpretation:

- Small Derrida values indicate ordered, stable dynamics.
- Large Derrida values indicate sensitive or chaotic dynamics.

Derrida values are closely related to average sensitivity of the update functions,
and provide a complementary notion of robustness.

## Summary

In this tutorial you learned how to:

- compute exact robustness measures for small Boolean networks,
- interpret coherence and fragility at network, basin, and attractor levels,
- approximate robustness measures for larger networks, and
- assess dynamical sensitivity using the Derrida value.

In Tutorial 9, we will finally analyze biological Boolean network models and
design ensemble experiments. 
