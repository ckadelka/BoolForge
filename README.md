# BoolForge

This package is designed to generate and analyze random Boolean functions and networks, with a focus on the concept of canalization.

## Installation

** Latest Stable Release **

*BoolForge* currently has no stable release.

** Latest Development Release **

Install the code directly from the GitHub page:

`pip install git+https://github.com/ckadelka/BoolForge`

Note that *BoolForge* uses [*networkx*](https://networkx.org/). You may additionally need to install the following:

`pip install networkx`

### Extended Functionality

*BoolForge* has a handful of methods that make use of the [*CANA*](https://github.com/CASCI-lab/CANA) package. 
However, as these methods are not integral to the functionality of this package, *CANA* is not considered a dependency. 
To access these methods, install *CANA*:

`pip install cana`

Moreover, *BoolForge* contains capabilities to plot wiring diagrams of Boolean networks.
To enable this, ensure the [*matplotlib*](https://matplotlib.org/stable/install/index.html) package is installed:

`pip install matplotlib`

### Compatability

*BoolForge* has methods to convert data for use both to and from other commonly used Boolean Network python packages. These packages would need to be installed separately.

Supported packages include:

 1. [CANA](https://github.com/CASCI-lab/CANA)
 2. [pyboolnet](https://github.com/hklarner/pyboolnet)

## Documentation

The *BoolForge* documentation can be found at: [https://ckadelka.github.io/BoolForge/](https://ckadelka.github.io/BoolForge/)
