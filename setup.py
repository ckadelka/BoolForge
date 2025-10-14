from setuptools import setup, find_packages

__package_name__ = "boolforge"
__description__ = "This package provides methods to generate and analyze random Boolean functions and networks, with a focus on the concept of canalization."

__version__ = open("boolforge/_version.py", "rt").read().split('\'')[1]

setup(
      name = __package_name__,
      version = __version__,
      description = __description__,
      long_description = __description__,
      
      author = "Claus Kadelka, Benjamin Coberly",
      author_email = "ckadelka@iastate.edu",
      url = "https://github.com/ckadelka/BoolForge",
      
      license = "MIT",
      
      packages = find_packages(),
      
      classifiers = [
          "Programming Language :: Python :: 3",
      ],
      
      install_requires = [
          "numpy",
          "networkx",
          "scipy",
          "pyeda"
      ]
)