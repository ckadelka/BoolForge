from boolforge.utils import *
from boolforge.generate import *
from boolforge.boolean_function import *
from boolforge.boolean_network import *
from boolforge.bio_models import *

try:
    from boolforge._version import __version__
except ImportError:
    __version__ = 'unknown'