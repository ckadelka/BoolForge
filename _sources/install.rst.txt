Installation
============

.. toctree::
   :maxdepth: 4

Python
------
*BoolForge* is developed for Python 3.10+ and is not compatible with prior versions of Python.

Windows
-------
It is recommended to install the package using *pip*. If you do not already have *pip* installed, run::

   C:> py -m ensurepip --upgrade

To install *BoolForge* with *pip*, run::

   C:> py -m pip install boolforge

You should now be able to import BoolForge::

   C:> py
   >>> import boolforge

To remove *BoolForge* using *pip*, run::

   C:> py -m pip uninstall boolforge

Mac OS / Linux
--------------
It is recommended to install the package using *pip*. If you do not already have *pip* installed, run::

   $ python -m ensurepip --upgrade

To install *BoolForge* with *pip*, run::

   $ python -m pip install boolforge

You should now be able to import BoolForge::

   $ python
   >>> import boolforge

To remove *BoolForge* using *pip*, run::

   $ python -m pip uninstall boolforge

Extended Functionality
----------------------
BoolForge is fully usable with its core dependencies, but some features rely on optional packages that can be installed via *extras*.

Performance Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~

Some internal routines are automatically accelerated if `numba <https://numba.pydata.org/>`__ is available.

To enable numba acceleration::

   pip install boolforge[speed]

When numba is not installed, BoolForge transparently falls back to pure-Python implementations.

Plotting and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting of wiring diagrams and network structure requires `matplotlib <https://matplotlib.org/>`__.

To enable plotting::

   pip install boolforge[plot]

CANA Integration
~~~~~~~~~~~~~~~~

Some methods interface with the `CANA <https://github.com/CASCI-lab/CANA>`__ package for advanced canalization
measures.

To enable CANA-based functionality::

   pip install boolforge[cana]

Symbolic Logic and Expression Minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbolic representations and logical expression minimization rely on `PyEDA <https://pyeda.readthedocs.io/>`__.

To enable symbolic functionality::

   pip install boolforge[symbolic]

Biological Model Retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~

The retrival and loading of hundreds of published biological Boolean
network models relies on the `requests <https://requests.readthedocs.io/en/latest/>`__ package for web access.

To enable biological model retrieval::

   pip install boolforge[bio]

All Optional Features
~~~~~~~~~~~~~~~~~~~~~

To install BoolForge with **all optional dependencies**::

   pip install boolforge[all]

Compatability and Interoperability
----------------------------------
*BoolForge* supports import and export of Boolean network representations used by other software packages.

In particular, BoolForge supports the **BNet format** commonly used by `pyboolnet <https://github.com/hklarner/pyboolnet>`__, without requiring pyboolnet itself to be installed.

Boolforge also supports converstion to and from the format used by `CANA <https://github.com/CASCI-lab/CANA>`__.