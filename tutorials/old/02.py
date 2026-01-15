import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title + intro markdown with corrected math delimiters
cells.append(nbf.v4.new_markdown_cell(
"""# BoolForge Tutorial #2  
## Advanced Concepts for Boolean Functions  

Understanding the structure of a Boolean function is essential for analyzing the behavior of the Boolean networks they define. In this tutorial, we move beyond the basics of `BooleanFunction` and explore three core concepts:

- **Symmetries** among inputs  
- **Activities** of inputs  
- **Average sensitivity** of a Boolean function  

These quantities are tied to redundancy, robustness, and dynamical behavior -- concepts that will play a central role in later tutorials on canalization and network dynamics.

## What you will learn
*What you will learn:* This tutorial will teach you how to:

- Identify symmetry groups of Boolean functions  
- Compute activities and sensitivities  
- Choose between exact and Monte Carlo computation  
- Interpret these quantities in terms of robustness and redundancy  

---
## 0. Setup
"""))

cells.append(nbf.v4.new_code_cell(
"""import boolforge
import numpy as np"""))

# Section 1 markdown
cells.append(nbf.v4.new_markdown_cell(
"""---
## 1. Symmetries in Boolean Functions

Symmetries reveal when inputs to a Boolean function are interchangeable. This matters for:

- Model reduction  
- Identifying redundant regulators  
- Understanding robustness in gene regulation  

### 1.1 What is a symmetry?

A symmetry of a Boolean function is a permutation of input variables that does **not** change its output.

- Inputs in the same symmetry group can be swapped freely  
- Inputs in different groups cannot  

These groups provide an algebraic fingerprint of the function’s structure.

### 1.2 Examples

Below we define three Boolean functions demonstrating full, partial, and no symmetry.
"""))

# Code: definitions and truth table
cells.append(nbf.v4.new_code_cell(
"""# Fully symmetric (parity / XOR)
f = boolforge.BooleanFunction('(x1 + x2 + x3) % 2')

# Partially symmetric
g = boolforge.BooleanFunction('x1 | (x2 & x3)')

# No symmetry
h = boolforge.BooleanFunction('x1 | (x2 & ~x3)')

labels = ['f','g','h']
boolforge.display_truth_table(f, g, h, labels=labels)"""))

# Code: symmetry groups
cells.append(nbf.v4.new_code_cell(
"""for func, label in zip([f, g, h], labels):
    print(f"Symmetry groups of {label}:")
    for group in func.get_symmetry_groups():
        print("  ", func.variables[np.array(group)])
    print()"""))

# Interpretation markdown
cells.append(nbf.v4.new_markdown_cell(
"""*Interpretation:*  
- `f` is fully symmetric: all variables are interchangeable.  
- `g` has partial symmetry: `x2` and `x3` are equivalent but `x1` is distinct.  
- `h` has no symmetries: all inputs play unique roles.

These patterns foreshadow the concepts of canalization, and specifically canalizing layers, explored in later tutorials.
"""))

# Section 2 markdown with $...$ and $$...$$
cells.append(nbf.v4.new_markdown_cell(
"""---
## 2. Activities and Sensitivities

Activities and sensitivity quantify how much each input affects the output of a Boolean function. 

### 2.1 Activity

The activity of input $x_i$ is the probability that flipping $x_i$ changes the function’s output:

$$
a(f,x_i) = \\Pr[f(\\mathbf{x}) \\neq f(\\mathbf{x} \\oplus e_i)].
$$

- If $a = 1$: the variable always matters  
- If $a = 0$: the variable is irrelevant (degenerate)  
- Random Boolean functions typically yield $a \\approx 0.5$

### 2.2 Average Sensitivity

The (unnormalized) average sensitivity is

$$
S(f) = \\sum_i a(f,x_i).
$$

The normalized average sensitivity is

$$
s(f) = \\frac{S(f)}{n}.
$$

*Interpretation:*
In Boolean network theory, the mean normalized average sensitivity $s(f)$ determines how perturbations tend to propagate through the system.

- If $s(f) < 1$, perturbations tend to die out and the dynamics lie in an *ordered regime*, characterized by stability and short attractors.
- If $s(f) > 1$, small perturbations typically amplify, producing a *chaotic regime* with sensitive, unpredictable dynamics.
- The boundary case $s(f) = 1$ defines the *critical regime*, where perturbations neither vanish nor explode, and where many biological models seem to operate.

This connection links the structure of update functions to global dynamical behavior.


### 2.3 Exact vs Monte Carlo computation

- **Exact (`EXACT=True`)** enumerates all $2^n$ states; feasible for small $n$.  
- **Monte Carlo (`EXACT=False`)** approximates using random samples; scalable to large $n$.

### 2.4 Computing activities and sensitivities
"""))

# Code: activities and sensitivities
cells.append(nbf.v4.new_code_cell(
"""EXACT = True

print("Activities of f:", f.get_activities(EXACT=EXACT))
print("Activities of g:", g.get_activities(EXACT=EXACT))

print("Normalized average sensitivity of f:", f.get_average_sensitivity(EXACT=EXACT))
print("Normalized average sensitivity of g:", g.get_average_sensitivity(EXACT=EXACT))"""))

# Interpretation markdown
cells.append(nbf.v4.new_markdown_cell(
"""*Interpretation:*  
- For `f` (XOR), flipping any input always flips the output, so $s(f) = 1$.  
- For `g`, `x1` matters more often than `x2` or `x3`.

This unequal influence is a precursor to canalization.
"""))

# Section 3 markdown
cells.append(nbf.v4.new_markdown_cell(
"""---
## 3. Large-Input Boolean Functions

Exact computation is infeasible for large $n$, so Monte Carlo simulation must be used.

### Example: random 25-input function
"""))

# Code: random 25-input example
cells.append(nbf.v4.new_code_cell(
"""EXACT = False
nsim = 500
n = 25

h = boolforge.random_function(n=n, ALLOW_DEGENERATE_FUNCTIONS=True)

activities = h.get_activities(EXACT=EXACT)
print(f"Mean activity: {np.mean(activities):.4f}")
print(f"Std of activities: {np.std(activities):.4f}")
print(f"Normalized average sensitivity: {h.get_average_sensitivity(EXACT=EXACT):.4f}")"""))

# Interpretation markdown
cells.append(nbf.v4.new_markdown_cell(
"""*Interpretation:*
Random Boolean functions satisfy, approximately:

- Mean activity $\\approx 0.5$  
- Normalized sensitivity $\\approx 0.5$  

This aligns with known theoretical results and defines the typical behavior against which biological functions can be compared.
"""))

# Section 4 + summary markdown
cells.append(nbf.v4.new_markdown_cell(
"""---
## 4. Practical Notes

### Degenerate functions

A function is **degenerate** if it ignores one or more inputs. Detecting degeneracy is NP-hard in general, but such functions are extremely rare unless intentionally created.

BoolForge therefore:

- allows degenerate functions by default  
- avoids expensive essential-variable checks unless requested  

### Forward reference

This tutorial introduced the concept of symmetry and the perturbation-based measures of activity and sensitivity.
In **Tutorial #3**, you will see how these relate to **canalization**, a key organizing principle of biological Boolean functions.

---
## 5. Summary

In this tutorial you learned:

- How to compute symmetry groups  
- How to compute activities and sensitivities  
- How to use exact vs Monte Carlo methods  
- How these quantities relate to robustness and structure  

These concepts provide essential foundations for understanding canalization and the dynamics of Boolean networks.
"""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}
}

path = "../tutorials/BoolForge_Tutorial_2_professional_mathfixed.ipynb"
with open(path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

path
