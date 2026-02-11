# Core public classes
from .boolean_function import (
    BooleanFunction, 
    display_truth_table,
    get_layer_structure_from_canalized_outputs
)
from .boolean_network import (
    BooleanNetwork, 
    get_entropy_of_basin_size_distribution
)
from .wiring_diagram import WiringDiagram

# Canonical public generators
from .generate import (
    random_function,
    random_NCF,
    random_k_canalizing_function,
    random_wiring_diagram,
    random_network,
    random_null_model,
)

# Version
try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "BooleanFunction",
    "display_truth_table",
    "get_layer_structure_from_canalized_outputs",
    "BooleanNetwork",
    "get_entropy_of_basin_size_distribution",
    "WiringDiagram",
    "random_function",
    "random_NCF",
    "random_k_canalizing_function",
    "random_wiring_diagram",
    "random_network",
    "random_null_model",
    "__version__",
]