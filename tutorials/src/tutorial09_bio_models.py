# %% [markdown]
# # Curated biological Boolean network models
#
# In this tutorial, we study how to analyze curate biological Boolean networks.
#
# You will learn how to:
# - load repositories of curated biological Boolean network models,
# - analyze these models,
# - generate null models to test the statistical significance of features in biological models.
#
# These tools enable real research findings, namely the identification of 
# design principles of regulatory functions and networks.
#
# ## Setup

# %%
import boolforge
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Loading model repositories.
# BoolForge makes it very easy to load all models included in three different 
# repositories of curated biological Boolean networks.

# %%
models = boolforge.get_bio_models_from_repository()
n_models = len(models)
bns = models['BooleanNetworks']

# %% [markdown]
# The function `get_bio_models_from_repository` loads, by default, all 122 distinct 
# biological Boolean network models, analyzed in Kadelka et al., Sci Adv, 2024.
# The models are parsed directly from the associated Github repository, meaning
# a wireless connection is required to successfully execute this function.
#
# Models from the two other available repositories can be loaded by selecting the 
# respective Github repository name.

# %%
models_sm = boolforge.get_bio_models_from_repository('pystablemotifs (jcrozum)')
n_models_sm = len(models_sm)
bns_sm = models_sm['BooleanNetworks']

#models_bd = boolforge.get_bio_models_from_repository('biodivine (sybila)')
#n_models_bd = len(models_bd)
#bns_bd = models_bd['BooleanNetworks']

# %% [markdown]
# Note that the last repository is very large, which is why this code is commented out.

# %% [markdown]
# ## Analyzing model repositories.
# By applying BoolForge functions to all models in a repository, we can swiftly
# generate summary statistics, such as the size distribution of the models, or their average degree.

# %%
sizes = [bn.N for bn in bns]
average_degrees = [np.mean(bn.indegrees) for bn in bns]

# %% [markdown]
# Plotting the size of a model against its average degree, we observe that,
# for these 122 models, there exists no strong correlation between size and degree.

# %%
f,ax = plt.subplots()
ax.semilogx(sizes,average_degrees,'x')
ax.set_xlabel('network size')
ax.set_ylabel('average degree')







