"""HYPE (SMHI) as a GPU forward model.

See :class:`~foresight_gpu.models.hype.model.HYPEModel` for the contract and the two
usage caveats (do not scale ``X``; keep ``shuffle=False``).
"""

from .autocal import AutoCalResult, autocalibrate
from .dates import as_X, as_ordinals, from_ordinals, window_ordinals
from .files import InfoFile, ParFile, read_bestsims, read_respar, read_series, write_optpar
from .forcing import ALIASES, FORCING_FILES, read_hype_table
from .frozen import TableModel, freeze
from .model import HYPEModel, load_observations
from .parameters import CATALOGUE, HypeParameter, Layout, build_layout, effective_options

__all__ = [
    "ALIASES",
    "AutoCalResult",
    "CATALOGUE",
    "FORCING_FILES",
    "HYPEModel",
    "HypeParameter",
    "InfoFile",
    "Layout",
    "ParFile",
    "TableModel",
    "as_X",
    "as_ordinals",
    "autocalibrate",
    "build_layout",
    "effective_options",
    "freeze",
    "from_ordinals",
    "load_observations",
    "read_hype_table",
    "read_bestsims",
    "read_respar",
    "read_series",
    "window_ordinals",
    "write_optpar",
]
