"""Deterministic forward models for the GPU engine.

Ship two defaults — a tiny :class:`MLPModel` and the hydrological :class:`GR4JModel` — and
a documented base class, :class:`BaseForwardModel`, for bringing your own.
"""

from .base import BaseForwardModel
from .gr4j import GR4JModel
from .mlp import MLPModel
from .mlp_opencl import MLPModelOpenCL

__all__ = ["BaseForwardModel", "MLPModel", "GR4JModel", "MLPModelOpenCL"]
