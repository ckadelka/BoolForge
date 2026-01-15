import nbformat as nbf

MD = """# BoolForge Tutorial #7: Dynamics of Boolean networks

In this tutorial, we study the *dynamics* of Boolean networks.
Building on the construction and structural analysis from previous tutorials, we now focus on how Boolean networks evolve over time and how their long-term behavior can be characterized.

You will learn how to:
- simulate Boolean network dynamics under different updating schemes,
- compute and classify attractors,
- analyze basins of attraction, and
- relate network structure to dynamical behavior.

## 0. Setup

```python
import boolforge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## 1. State space of a Boolean network

A Boolean network with $N$ nodes defines a dynamical system on the discrete state space $\\{0,1\\}^N.$

Each state is a binary vector
$$
\\mathbf x = (x_0,\\ldots,x_{N-1}) \\in \\{0,1\\}^N,
$$
where $x_i$ denotes the state of node $i$.

The full state space consists of $2^N$ states. 
For small $N$, the entire state space can be enumerated and analyzed explicitly.

We will use a small Boolean network as a running example throughout this tutorial.

```python
string = '''x = y
y = x OR z
z = y'''

bn = boolforge.BooleanNetwork.from_string(string, separator='=')
print("Variables:", bn.variables)
print("N:", bn.N)
print('bn.I',bn.I)
print('bn.F',bn.F)
```

All state vectors in this tutorial will follow the variable order given by `bn.variables`. 
For small $N$-node Boolean networks, we can explicitly enumerate all $2^N$ states in $\\{0,1\\}^N$. 

```python
all_states = boolforge.get_left_side_of_truth_table(bn.N)
pd.DataFrame(all_states,columns = bn.variables)
```

Each state is represented as a numpy.array of length N, and we see the decimal representation of each state on the left of the displayed pandas.DataFrame.

## 2. Dynamics of synchronous Boolean networks

The update rules, encoded in `bn.F`, describe the dynamic transitions of the Boolean network. 

Under a *synchronous update* scheme, all nodes are updated simultaneously.
This defines a deterministic update map
$$
F : \\{0,1\\}^N \\to \\{0,1\\}^N,
\\qquad
\\mathbf x(t+1) = F(\\mathbf x(t)).
$$

### 2.1 Exact computation

We can compute the next, synchronously updated state for any initial state:

```python
for state in all_states:
    print(state,'-->',bn.update_network_synchronously(state))
```

Note that this is exactly the same output that we get when typing

```python
print(bn.to_truth_table())
```

Each state has exactly one successor under synchronous updating.
As a result, the dynamics consist of transient trajectories leading into *steady states* and *cycles*, collectively known as *attractors*.

In our example, the Boolean network has two steady states ($x=0,y=0,z=0$ and $x=1,y=1,z=1$) as well as a cyclic attractor of periodicity 2 ($x=0,y=1,z=0 \\leftrightarrow x=1,y=0,z=1$). 
All other states are transient and eventually end up at an attractor.

For any sufficiently small Boolean network (typically of size $N\\lesssim 20$-$30$), the entire dynamics and long-term behavior can be computed exhaustively, allowing the full dynamics and long-term behavior to be computed exactly.
    
```python
dict_dynamics = bn.get_attractors_synchronous_exact()
dict_dynamics
````

This function returns a dictionary with five keys that collectively describe the network’s dynamics:

- `STG`: The *state transition graph* under synchronous updating. Each key is one of the $2^N$ states (in decimal representation), and the corresponding value is the next state after one synchronous update.
Use `boolforge.dec2bin` to convert states back to binary representation:
```python
for state in range(2**bn.N):
    next_state = dict_dynamics['STG'][state]
    print(state,'=',
          boolforge.dec2bin(state,bn.N),
          '-->',
          next_state,'=',
          boolforge.dec2bin(next_state,bn.N))
```
- `NumberOfAttractors`: The total number of distinct attractors in the network.
- `Attractors`: A list of the network attractors, each represented as a sequence of states in decimal form.
Again, `boolforge.dec2bin` can be used to transform to binary state vectors:
```python
for attractor in dict_dynamics['Attractors']:
    print(f'Attractor of length {len(attractor)}:')
    for state in attractor:
        print(state,boolforge.dec2bin(state,bn.N))
    print()
```
- `AttractorDict`: While `STG` specifies the one-time step update of each state, this dictionary assigns each state to the index of the attractor that it eventually reaches (indexed as in `dict_dynamics['Attractors']`).
- `BasinSizes`: The *basin size* of each attractor, defined as the number of states that ultimately flow into it. The basin sizes always sum to $2^N$, since every state eventually transitions to an attractor. Often, it is more useful to consider *relative basin sizes* that sum up to 1.

### 2.2 Simulations

For larger networks, exhaustive enumeration of the full state space quickly becomes computationally infeasible.
In such cases, *Monte Carlo simulation* can be used to identify all large attractors and to approximate the basin size distribution:

```python
dict_dynamics = bn.get_attractors_synchronous(nsim=100)
dict_dynamics
````

In addition to the keys described above, the resulting dictionary contains two further entries:
- `InitialSamplePoints`: The decimal representations of the nsim=100 randomly chosen initial states whose trajectories were followed until an attractor was reached.
- `NumberOfTimeouts`: The number of trajectories that failed to reach an attractor within the allotted time. By default, each trajectory is updated for at most `n_steps_timeout = 1000` steps (optional keyword), so this value is typically zero for well-behaved networks.
```

We can compute the probability that an attractor of relative basin size $q\in(0,1]$ is found by simulation. If we start the simulation from $m\geq 1$ random initial states, the probability that the attractor is found is $p = 1-(1-q)^m$. Importantly, this probability does not depend on the size of the network.
                                                                              
```python
qs = [0.0001,0.001,0.01,0.1]
ms = np.logspace(0,4,1000)
f,ax = plt.subplots()
for q in qs:
    ax.semilogx(ms,1-(1-q)**ms,label=str(q))
ax.legend(loc='best',title=r'$q$',frameon=False)
ax.set_xlabel(r'number of initial states ($m$)')
ax.set_ylabel('probability that an attractor of\nrelative basin size q is found')
````       

## 3. Dynamics of asynchronous Boolean networks

The synchronous updating scheme assumes that all nodes are updated simultaneously. While this is computationally nice, leading to deterministic dynamics, it is biologically unrealistic. In reality, updates (i.e., transition across the gene-specific threshold between 0 and 1) occur at random points on a continuous time scale. 

Asynchronous updating schemes capture this stochasticity by assuming that one (or a few) genes are updated at a time. `BoolForge` implements two different asynchronous updating schemes. 

## 3.1 Steady states under general asynchronous update

Under a general asynchronous updating scheme, one gene is updated at a time. The stochastic state transition graph describes all possible transitions from a given state (up to $N$ in a network of $N$ nodes). Unsurprisingly, computing the entire dynamics becomes a lot more complicated and there are a number of dedicated software solutions designed for this specific purpose, e.g. `pystablemotifs`.
`BoolForge` can compute the dynamics under this updating scheme both exhaustively and by simulation.

```python
dict_dynamics = bn.get_steady_states_asynchronous_exact()
dict_dynamics
````

This reveals the same $m=2$ steady states 0 (i.e., $x=0,y=0,z=0$) and 7 (i.e., $x=1,y=1,z=1$).
In addition, the entire asynchronous state transition graph is returned (`STGAsynchronous`), 
as well as the final transition probabilities from any of the $2^N$ states to the steady states.
If starting at a steady state, the system will remain at this steady state. However, when starting 
at a transient state there may be some uncertainty, which steady state the system finally settles in.
The basin sizes are computing by summing over the columns of this $2^N \times m$-matrix.

It is important to note that `BoolForge` does currently not contain methods to identify and deal with complex attractors.
In these cases, the use of `pystablemotifs` is advised.

Just as in the synchronous case, the steady states can also be approximated by simulation.
This is the only feasible option for larger network.

```python
dict_dynamics = bn.get_steady_states_asynchronous(nsim=500)
````

Rather than providing exact values, this provides 
- a lower bound of the number of steady states, 
- an approximation of the basin size distribution,
- a sample of the asynchronous state transition graph, as well as
- a list of `nsim` initial conditions used for the simulation.

Instead of starting from random initial conditions, `BoolForge` also allows to thoroughly
sample the possible trajectories from one defined initial condition. This is useful if one
knows e.g. the baseline condition of a biological network and wants to explore all possible
dynamics.

```python
dict_dynamics = bn.get_steady_states_asynchronous_given_one_initial_condition(initial_condition = [0,0,1], nsim=500)
````





"""

# Replace VSCode-unfriendly math delimiters if present
MD = MD.replace(r"\\(", "$").replace(r"\\)", "$").replace(r"\\[", "$$").replace(r"\\]", "$$")

def md_to_notebook(markdown_text: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []
    in_code = False
    buf = []

    for line in markdown_text.splitlines(True):
        if not in_code:
            if line.startswith("```"):
                lang = line.strip()[3:].strip().lower()
                if lang == "python":
                    if "".join(buf).strip():
                        cells.append(nbf.v4.new_markdown_cell("".join(buf).rstrip()))
                    buf = []
                    in_code = True
                else:
                    buf.append(line)
            else:
                buf.append(line)
        else:
            if line.startswith("```"):
                cells.append(nbf.v4.new_code_cell("".join(buf).rstrip()))
                buf = []
                in_code = False
            else:
                buf.append(line)

    if "".join(buf).strip():
        if in_code:
            cells.append(nbf.v4.new_code_cell("".join(buf).rstrip()))
        else:
            cells.append(nbf.v4.new_markdown_cell("".join(buf).rstrip()))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb

nb = md_to_notebook(MD)

out_path = "BoolForge_Tutorial_7_dynamics_of_Boolean_networks.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(out_path)
