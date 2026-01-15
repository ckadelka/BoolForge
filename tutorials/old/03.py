import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
r"""# BoolForge Tutorial #3: Canalization

In this tutorial, we will focus on canalization, a key property of Boolean functions, especially those that constitute biologically meaningful update rules in biological networks.

## What you will learn
*What you will learn:* In this tutorial you will:

- determine if a Boolean function is canalizing, $k$-canalizing, and nested canalizing,
- compute the canalizing layer structure of any Boolean function, and
- compute properties related to collective canalization, such as canalizing strength and quantities related to effective degree and input redundancy.

---
## 0. Setup"""
))

cells.append(nbf.v4.new_code_cell(
"""import boolforge
import numpy as np
import matplotlib.pyplot as plt"""
))

cells.append(nbf.v4.new_markdown_cell(
r"""---
## 1. Canalizing variables and layers

A Boolean function $f(x_1, \ldots, x_n)$ is *canalizing* if there exists at least one *canalizing variable* $x_i$ and a *canalizing input value* $a \in \{0,1\}$ such that
$f(x_1,\ldots,x_i=a,\ldots,x_n)=b$,
where $b \in \{0,1\}$ is a constant, the *canalized output*.

A Boolean function is *$k$-canalizing* if it has at least $k$ conditionally canalizing variables. This is checked recursively: after fixing a canalizing variable $x_i$ to its non-canalizing input value $\bar a$, the resulting subfunction must itself contain another canalizing variable, and so on. The maximal possible value of $k$ is called the *canalizing depth*. If all variables are conditionally canalizing (i.e., if the canalizing depth is $n$), the function is called a *nested canalizing function* (NCF).

Per He and Macaulay (Physica D, 2016), any Boolean function can be decomposed into a unique standard monomial form by recursively identifying and removing all conditionally canalizing variables. The set of variables removed at each step forms a *canalizing layer*. Each variable appears in exactly one layer or (if it is not conditionally canalizing) in the non-canalizing core function that is evaluated only if all conditionally canalizing variables receive their non-canalizing input value.

The *canalizing layer structure* $[k_1,\ldots,k_r]$ describes the number of variables in each canalizing layer. We thus have $r \ge 0$, $k_i \ge 1$, and $k_1+\cdots+k_r \le n$.

### 1.1 Examples

In the following code, we define four 3-input functions with different canalizing properties."""
))

cells.append(nbf.v4.new_code_cell(
"""# Example: a non-canalizing XOR function.
f = boolforge.BooleanFunction('(x0 + x1 + x2) % 2')

# Example: a 1-canalizing function
g = boolforge.BooleanFunction('(x0 | (x1 & x2 | ~x1 & ~x2)) % 2')

# Example: an NCF with 3 canalizing variables in the outer layer
h = boolforge.BooleanFunction('~x0 & x1 & x2')

# Example: an NCF with 1 canalizing variable in the outer layer and two in the inner layer
k = boolforge.BooleanFunction('x0 | (x1 & x2)')

labels = ['f','g','h','k']
boolforge.display_truth_table(f, g, h, k, labels=labels)"""
))

cells.append(nbf.v4.new_markdown_cell(
r"""### 1.2 Canalizing depth and nested canalization

For each function, we can determine whether it is canalizing and/or nested canalizing via its canalizing depth. An $n$-input function is canalizing if its canalizing depth is non-zero, and nested canalizing if its canalizing depth equals $n$."""
))

cells.append(nbf.v4.new_code_cell(
"""for func, label in zip([f, g, h, k], labels):
    canalizing_depth = func.get_canalizing_depth()
    print(f'Canalizing depth of {label}: {canalizing_depth}')

    CANALIZING = func.is_canalizing()
    print(f'{label} is canalizing: {CANALIZING}')

    NESTED_CANALIZING = func.is_k_canalizing(k=func.n)
    print(f'{label} is nested canalizing: {NESTED_CANALIZING}')
    print()"""
))

cells.append(nbf.v4.new_markdown_cell(
r"""### 1.3 Canalizing layer structure

We can also compute the full canalizing layer structure, which yields information on canalizing input values, canalized output values, the order of canalizing variables, the layer structure, and the core function."""
))

# NOTE: this code cell fixes the quoting bug from the markdown version by using double-quotes
cells.append(nbf.v4.new_code_cell(
"""for func, label in zip([f, g, h, k], labels):
    canalizing_info = func.get_layer_structure()
    print(f"Canalizing input values of {label}: {canalizing_info['CanalizingInputs']}")
    print(f"Canalized output values of {label}: {canalizing_info['CanalizedOutputs']}")
    print(f"Order of canalizing variables of {label}: {canalizing_info['OrderOfCanalizingVariables']}")
    print(f"Layer structure of {label}: {canalizing_info['LayerStructure']}")
    print(f"Number of canalizing layers of {label}: {canalizing_info['NumberOfLayers']}")
    print(f"Non-canalizing core function of {label}: {canalizing_info['CoreFunction']}")
    print()"""
))

cells.append(nbf.v4.new_markdown_cell(
r"""Consider, for example, the output for `k`. The canalizing input values corresponding to canalizing variables $x_0, x_1, x_2$ are $1,0,0$, respectively. Likewise, the corresponding canalized output values are also $1,0,0$. This tells us that `k` can be evaluated as follows:

$$
k(x_0,x_1,x_2) =
\begin{cases}
1 & \ \text{if}\ x_0 = 1,\\
0 & \ \text{if}\ x_0 \neq 1 \ \text{and} \ x_1 = 0,\\
0 & \ \text{if}\ x_0 \neq 1 \ \text{and} \ x_1 \neq 0 \ \text{and} \ x_2 = 0,\\
1 & \ \text{if}\ x_0 \neq 1 \ \text{and} \ x_1 \neq 0 \ \text{and} \ x_2 \neq 0.
\end{cases}
$$

Since $x_1$ and $x_2$ are both part of the second canalizing layer, `k` can equivalently be evaluated as:

$$
k(x_0,x_1,x_2) =
\begin{cases}
1 & \ \text{if}\ x_0 = 1,\\
0 & \ \text{if}\ x_0 \neq 1 \ \text{and} \ x_2 = 0,\\
0 & \ \text{if}\ x_0 \neq 1 \ \text{and} \ x_2 \neq 0 \ \text{and} \ x_1 = 0,\\
1 & \ \text{if}\ x_0 \neq 1 \ \text{and} \ x_2 \neq 0 \ \text{and} \ x_1 \neq 0.
\end{cases}
$$"""
))

cells.append(nbf.v4.new_markdown_cell(
r"""---
## 2. Collective canalization

More recently, the idea of collective canalization was introduced (Reichhardt & Bassler, Journal of Physics A, 2007). Rather than defining canalization purely as a property of each individual variable, collective canalization treats it as a property of the function itself.

Extending the basic definition, a Boolean $n$-input function is *$k$-set canalizing* if there exists a set of $k$ variables such that setting these variables to specific values forces the output of the function, irrespective of the other $n-k$ inputs (Kadelka et al., Advances in Applied Mathematics, 2023). Naturally:

- any Boolean function is $n$-set canalizing,
- the only two Boolean functions that are not $(n-1)$-set canalizing are the parity / XOR functions,
- the 1-set canalizing functions are exactly the canalizing functions.

For any function and a given $k$, we can quantify the proportion of $k$-sets that collectively canalize this function (i.e., suffice to determine its output). This is the *$k$-set canalizing proportion* $P_k(f)$.

It is fairly obvious that:

- nested canalizing functions of a single layer such as `h` have $P_k(f) = 1-1/2^k$ (among non-degenerate functions),
- $P_{k-1}(f) \le P_k(f)$,
- the $(n-1)$-set canalizing proportion $P_{n-1}(f)$ is $1$ minus the function’s normalized average sensitivity.

### 2.1 Computing $k$-set canalizing proportions"""
))

cells.append(nbf.v4.new_code_cell(
"""for func, label in zip([f, g, h, k], labels):
    print(f'1-set canalizing proportions of {label}: {func.get_kset_canalizing_proportion(k=1)}')
    print(f'2-set canalizing proportions of {label}: {func.get_kset_canalizing_proportion(k=2)}')
    print(f'Normalized average sensitivity of {label}: {func.get_average_sensitivity(EXACT=True)}')
    print(f'3-set canalizing proportions of {label}: {func.get_kset_canalizing_proportion(k=3)}')
    print()"""
))

cells.append(nbf.v4.new_markdown_cell(
r"""### 2.2 Canalizing strength

The *canalizing strength* is a measure of the degree of canalization of any Boolean function (Kadelka et al., Advances in Applied Mathematics, 2023). It is computed as a weighted average of the $k$-set canalizing proportions. It is:

- 1 for the most canalizing non-degenerate functions (nested canalizing functions of a single canalizing layer such as `h`),
- 0 for the least canalizing functions (parity / XOR functions such as `f`),
- strictly between 0 and 1 for all other non-degenerate Boolean functions.

It helps to view canalizing strength as a probability: given that you know a random number of function inputs (drawn uniformly at random from $1,\ldots,n-1$), how likely are you to already know the function output?"""
))

cells.append(nbf.v4.new_code_cell(
"""for func, label in zip([f, g, h, k], labels):
    canalizing_strength = func.get_canalizing_strength()
    print(f'Canalizing strength of {label}: {canalizing_strength}')
    print()"""
))

cells.append(nbf.v4.new_markdown_cell(
r"""---
## 3. Distribution of canalizing strength (small $n$)

An enumeration of all non-degenerate 3-input Boolean functions reveals the distribution of the canalizing strength. This brute-force code can also run (in less than a minute) for all $2^{2^4}=2^{16}=65{,}536$ 4-input functions but will take days for all $2^{2^5}=2^{32}=4{,}294{,}967{,}296$ 5-input functions."""
))

cells.append(nbf.v4.new_code_cell(
"""n = 3
all_functions = boolforge.get_left_side_of_truth_table(2**n)

canalizing_strengths = []
for binary_vector in all_functions:
    func = boolforge.BooleanFunction(f=binary_vector)
    if func.is_degenerate() == False:
        canalizing_strength = func.get_canalizing_strength()
        canalizing_strengths.append(canalizing_strength)

fig, ax = plt.subplots()
ax.hist(canalizing_strengths, bins=50)
ax.set_xlabel('canalizing strength')
ax.set_ylabel('Count')"""
))

cells.append(nbf.v4.new_markdown_cell(
r"""---
## 4. Canalization as a measure of input redundancy

Canalization, symmetry and redundancy are related concepts. A highly symmetric Boolean function with few (e.g., one) symmetry group exhibits high input redundancy and is on average more canalizing, irrespective of the measure of canalization. Recently, it was shown that almost all Boolean functions (except the parity / XOR functions) exhibit some level of *input redundancy* (Gates et al., PNAS, 2021).

The input redundancy of a variable is defined as 1 minus its *edge effectiveness*, which describes the proportion of times that this variable is needed to determine the output of the function. Edge effectiveness is very similar to the activity of a variable but is not the same (the difference is defined as *excess canalization*). The sum of all edge effectivenesses is the *effective degree*. The average input redundancy serves as another measure of canalization.

In BoolForge, these quantities are computed via the optional `CANA` package (install with `pip install cana`). To exemplify this, we reconsider the four 3-input functions from above."""
))

cells.append(nbf.v4.new_code_cell(
"""for func, label in zip([f, g, h, k], labels):
    edge_effectiveness = func.get_edge_effectiveness()
    activities = func.get_activities()
    effective_degree = func.get_effective_degree()
    input_redundancy = func.get_input_redundancy()

    print(f'Edge effectiveness of the variables of {label}: {edge_effectiveness}')
    print(f'Activities of the variables of {label}: {activities}')
    print(f'Excess canalization of the variables of {label}: {edge_effectiveness - activities}')
    print(f'Effective degree of {label}: {effective_degree}')
    print(f'Average edge effectiveness of {label}: {effective_degree/n}')
    print(f'Normalized input redundancy of {label}: {input_redundancy}')
    print()"""
))
    
cells.append(nbf.v4.new_markdown_cell(
r"""---
## 5. Summary and next steps

In this tutorial you learned how to:

- compute canalizing depth and identify nested canalizing functions,
- compute the canalizing layer structure and interpret layers and core functions,
- quantify collective canalization via $k$-set canalizing proportions,
- summarize canalization via canalizing strength, and
- relate canalization to redundancy-based measures such as edge effectiveness and effective degree (via CANA).

Canalization provides a compact structural explanation for why many biologically motivated Boolean rules are robust to perturbations: large subsets of inputs often become irrelevant once a few “decisive” variables take specific values.

*Next steps:* In the next tutorials, we build on these concepts to (i) generate random Boolean functions with prescribed canalization properties (e.g., fixed depth or layer structure), and (ii) study how canalization shapes the sensitivity of Boolean functions and the dynamics of Boolean networks (attractors, stability, and perturbation propagation)."""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

path = "BoolForge_Tutorial_3_canalization.ipynb"
with open(path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

path
