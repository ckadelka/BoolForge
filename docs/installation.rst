Installation
============

.. toctree::
   :maxdepth: 2

Python
------
*BoolForge* is developed for Python 3 and is not compatible with Python 2.

Windows
-------
It is recommended to install the package using *pip*. If you do not already have *pip* installed, run::

   C:> py -m ensurepip --upgrade

To install *BoolForge* with *pip*, run::

   C:> py -m pip install git+https://github.com/ckadelka/BoolForge

You should now be able to import BoolForge::

   C:> py
   >>> import boolforge

To remove *BoolForge* using *pip*, run::

   C:\> py -m pip uninstall BoolForge

Mac OS / Linux
--------------
It is recommended to install the package using *pip*. If you do not already have *pip* installed, run::

   $ python -m ensurepip --upgrade

To install *BoolForge* with *pip*, run::

   $ python -m pip install git+https://github.com/ckadelka/BoolForge

You should now be able to import BoolForge::

   $ python
   >>> import boolforge

To remove *BoolForge* using *pip*, run::

   $ python -m pip uninstall BoolForge

Extended Functionality
----------------------
*BoolForge* has a handful of methods that make use of the `CANA <https://github.com/CASCI-lab/CANA>`__ package. However, as these methods are not integral to the functionality of this package, *CANA* is not considered a dependency of *BoolForge* and must be downloaded independently. `You can learn more about CANA here <https://casci-lab.github.io/CANA/index.html>`__.

Compatability
-------------
*BoolForge* has methods to convert data for use both to and from other commonly used Boolean Network python packages.

Supported packages include:
   #. `CANA <https://github.com/CASCI-lab/CANA>`__
   #. `pyboolnet <https://github.com/hklarner/pyboolnet>`__