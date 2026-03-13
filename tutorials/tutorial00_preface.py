# %% [markdown]
# # Preface {-}
#
# \addcontentsline{toc}{section}{Preface}

# **BoolForge** is a Python toolbox for generating, analyzing, and simulating
# Boolean functions and Boolean networks. Boolean network models are widely used
# in systems biology, theoretical biology, and complex systems research to study
# regulatory systems whose components operate in two qualitative states
# (e.g., active/inactive or ON/OFF).
#
# In gene regulatory network models, for instance, each node represents a molecular component
# (such as a gene, protein, or signaling molecule), and each node is updated by a
# Boolean function that represents the regulatory logic controlling that component.
#
# The tutorials in this document provide a structured introduction to BoolForge
# and demonstrate how it can be used to perform computational experiments on
# Boolean functions and Boolean networks.
#
# ## Philosophy and scope of BoolForge
#
# BoolForge was designed to support both **methodological research on Boolean
# networks** and **applied analysis of biological regulatory models**.
#
# Three principles guide its design:
#
# **1. Fundamental representations**
#
# Boolean functions are stored internally as truth tables, the most fundamental
# representation of Boolean logic. Logical expressions and polynomial forms can
# be derived from this representation when needed.
#
# **2. Controlled random model generation**
#
# Many research questions require comparing biological networks with suitable
# **null models**. BoolForge therefore provides various tools for generating random
# Boolean functions and Boolean networks with prescribed structural properties.
#
# **3. Integration of structure and dynamics**
#
# Structural properties of regulatory rules (such as canalization, redundancy,
# and symmetry) influence dynamical behavior, including attractors, robustness,
# and sensitivity to perturbations. BoolForge enables analysis across these levels,
# connecting function-level structure to network-level dynamics.
#
# Together, these capabilities support a typical research workflow:
#
# ```
# Boolean functions
#        ↓
# Structural analysis (bias, activities, canalization)
#        ↓
# Random function generation (controlled null models)
#        ↓
# Boolean network construction
#        ↓
# Network dynamics (attractors, basins)
#        ↓
# Robustness and perturbation analysis
#        ↓
# Comparison with random ensembles and biological models
# ```
#
# ## Structure of the tutorials
#
# The tutorials gradually introduce the main concepts and tools provided by
# BoolForge:
#
# - **Boolean functions:** representation and structural analysis
# - **Canalization:** redundancy and robustness of regulatory rules
# - **Random function generation:** sampling functions with prescribed properties
# - **Boolean networks:** construction and wiring diagrams
# - **Network dynamics:** attractors and state transition graphs
# - **Perturbation and robustness:** sensitivity to state changes
# - **Random network ensembles:** statistical analysis of network dynamics
# - **Biological models:** analysis of curated regulatory networks
#
# Each tutorial contains executable code examples illustrating how these ideas
# can be explored using BoolForge.
#
# Readers are encouraged to run the code cells and modify the examples to
# explore their own Boolean functions and networks.