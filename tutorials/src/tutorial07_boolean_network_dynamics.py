# %% [markdown]
# # Dynamics of Boolean Networks
#
# In this tutorial, we study the *dynamics* of Boolean networks.
# Building on the construction and structural analysis from previous tutorials,
# we now focus on characterizing the long-term behavior of Boolean networks.
#
# You will learn how to:
#
# - simulate Boolean network dynamics under different updating schemes,
# - compute and classify attractors,
# - analyze basins of attraction,
# - relate network structure to dynamical behavior.
#
# ## Setup

# %%
import boolforge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown]
# ## State space of a Boolean network
#
# A Boolean network with $N$ nodes defines a dynamical system on the discrete
# state space $\{0,1\}^N$.
#
# Each state is a binary vector
#
# $$
# \mathbf{x} = (x_0, \ldots, x_{N-1}) \in \{0,1\}^N,
# $$
#
# where $x_i$ denotes the state of node $i$.
#
# We use a small Boolean network as a running example.

# %%
string = """
x = y
y = x OR z
z = y
"""

bn = boolforge.BooleanNetwork.from_string(string, separator="=")

print("Variables:", bn.variables)
print("N:", bn.N)
print("bn.I:", bn.I)
print("bn.F:", bn.F)


# %% [markdown]
# All state vectors follow the variable order given by `bn.variables`.
# For small networks, we can enumerate all $2^N$ states explicitly.

# %%
all_states = boolforge.get_left_side_of_truth_table(bn.N)
print(pd.DataFrame(all_states, columns=bn.variables).to_string())


# %% [markdown]
# ## Dynamics of synchronous Boolean networks
#
# Under *synchronous updating*, all nodes are updated simultaneously, defining
# a deterministic update map
#
# $$
# \mathbf{x}(t+1) = F(\mathbf{x}(t)).
# $$


# %% [markdown]
# ### Exact computation
# For each state, we can call `update_network_synchronously` to identify the next
# state under synchronous update.

# %%
for state in all_states:
    print(state, "-->", bn.update_network_synchronously(state))


# %% [markdown]
# This output matches the synchronous truth table representation:

# %%
print(bn.to_truth_table().to_string())


# %% [markdown]
# Each state has exactly one successor, so the dynamics consist of transient
# trajectories leading into *attractors* (steady states or cycles).
#
# In this example, the network has:
#
# - two steady states: $(0,0,0)$ and $(1,1,1)$,
# - one cyclic attractor of length 2: $(0,1,0) \leftrightarrow (1,0,1)$.


# %% [markdown]
# ### Exhaustive attractor computation
# BoolForge contains a dedicated method to identify all attractors of a network
# under synchronous update.

# %%
dict_dynamics = bn.get_attractors_synchronous_exact()


# %% [markdown]
# The returned dictionary contains:
#
# - `STG`: the synchronous state transition graph,
# - `NumberOfAttractors`,
# - `Attractors`,
# - `AttractorID`,
# - `BasinSizes`.
#
# For computational reasons, binary states in $\{0,1\}^N$ are identified by their decimal representation.
# The state transition graph can be decoded as follows:

# %%
for state in range(2 ** bn.N):
    next_state = dict_dynamics["STG"][state]
    print(
        state,
        "=",
        boolforge.dec2bin(state, bn.N),
        "-->",
        next_state,
        "=",
        boolforge.dec2bin(next_state, bn.N),
    )

# %% [markdown]
# After repeated updates, the system settles into periodic behavior. That is,
# irrespective of the initial state, an attractor is reached. The list
# of all attractors (in decimal representation) can be displayed. 

# %%
print(dict_dynamics['Attractors'])

# %% [markdown]
# Attractors can be printed in binary representation:

# %%
for attractor in dict_dynamics["Attractors"]:
    print(f"Attractor of length {len(attractor)}:")
    for state in attractor:
        print(state, boolforge.dec2bin(state, bn.N))
    print()

# %% [markdown]
# The information which state transitions to which attractor is stored in a dictionary.
# Here, the indices correspond to the list of attractors in `dict_dynamics['Attractors']`.

# %%
print(dict_dynamics['AttractorID'])


# %% [markdown]
# Finally, the basin size of each attractor is determined by the number of states that eventually transition to an attractor.
# By definition, the sum of all basin sizes is always $2^N$. To simplify the comparison of
# the basin size distribution for networks of different size, `BoolForge` normalizes the basin sizes by default.

# %%
print(dict_dynamics['BasinSizes'])

# %% [markdown]
# From the previous two commands, that there is no state (other than 000) that eventually
# transitions to 000. Half the states transition to the 2-cycle, while 3 out of 8
# states transition to the attractor 111.
#
# ### Monte Carlo simulation
#
# For larger networks, exhaustive enumeration is infeasible.
# Monte Carlo simulation approximates the attractor landscape.

# %%
dict_dynamics = bn.get_attractors_synchronous(n_simulations=100)
print(dict_dynamics['Attractors'])
print(dict_dynamics['BasinSizes'])


# %% [markdown]
# The simulation returns additional information:
#
# - sampled initial states,
# - the number of timeouts (trajectories not reaching an attractor before timeout).
#
# In the absence of timeouts: If an attractor has relative basin size $q$, 
# the probability that it is found from $m$ random initializations is $1 - (1-q)^m$.

# %%
qs = [0.0001, 0.001, 0.01, 0.1]
ms = np.logspace(0, 4, 1000)

fig, ax = plt.subplots()
for q in qs:
    ax.semilogx(ms, 1 - (1 - q) ** ms, label=str(q))

ax.legend(title=r"$q$", frameon=False)
ax.set_xlabel("number of initial states ($m$)")
ax.set_ylabel("probability attractor of basin size q is found")
plt.show()


# %% [markdown]
# ## Dynamics of asynchronous Boolean networks
#
# Synchronous updating is computationally convenient but biologically unrealistic.
# Asynchronous updating assumes that only one node changes at a time.


# %% [markdown]
# ### Steady states under general asynchronous update
#
# BoolForge can compute steady states under **general asynchronous updating**,
# where at each step only a single node updates according to its Boolean rule.

# %%
dict_dynamics = bn.get_steady_states_asynchronous_exact()
print(dict_dynamics['SteadyStates'])
print(dict_dynamics['NumberOfSteadyStates'])

# %% [markdown]
# The result reveals the same two steady states as in the synchronous case.
# However, the limit cycle observed under synchronous updating disappears
# under asynchronous dynamics.
#
# In addition, BoolForge returns the **full asynchronous state transition graph**.


# %%
for state, successors in dict_dynamics["STGAsynchronous"].items():
    print(state, successors)

# %% [markdown]
# The state transition graph describes for each state the possible next states 
# that the system may transition to, in addition to the transition probabilities. 
# This graph can be interpreted as a **sparse transition matrix**
# of a Markov chain. Each directed edge corresponds to a possible single-node update.
#
# By repeatedly composing this transition matrix with itself (equivalently,
# raising it to higher powers), BoolForge computes the **absorption probabilities**,
# i.e., the probability that a trajectory starting from any state eventually
# reaches each steady state.

# %%
print(dict_dynamics['FinalTransitionProbabilities'])


# %% [markdown]
# The size of each basin of attraction is the (column-wise) average of these probabilities.

# %%
print(dict_dynamics['BasinSizes'])
print(dict_dynamics['BasinSizes'] == 
      np.mean(dict_dynamics['FinalTransitionProbabilities'],0))

# %% [markdown]
# Note that `BoolForge` currently does not detect complex cyclic attractors under
# asynchronous update; for this task, specialized tools such as
# `pystablemotifs` are recommended.


# %% [markdown]
# ### Monte Carlo approximation
#
# As in synchronous case, `BoolForge` also contains a Monte Carlo routine
# for sampling asynchronous dynamics.
#
# The simulation provides:
#
# - a lower bound on the number of steady states,
# - approximate basin size distributions,
# - samples of the asynchronous state transition graph.

# %%
dict_dynamics = bn.get_steady_states_asynchronous(n_simulations=500)
print(dict_dynamics['SteadyStates'])
print(dict_dynamics['NumberOfSteadyStates'])
print(dict_dynamics['BasinSizes'])

# %% [markdown]
# ### Sampling from a fixed initial condition

# %%
dict_dynamics = bn.get_steady_states_asynchronous_given_one_initial_condition(
    initial_condition=[0, 0, 1], n_simulations=500
)
dict_dynamics


# %% [markdown]
# ## Summary and outlook
#
# In this tutorial you learned how to:
#
# - simulate Boolean network dynamics,
# - compute synchronous attractors exactly and approximately,
# - analyze basin sizes,
# - compute steady states under asynchronous updating.
#
# This concludes the function- and network-level analysis.
# Subsequent tutorials focus on analyzing stability to perturbations, control analysis, 
# and ensemble experiments.
