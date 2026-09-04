"""OpenCL overload point for the MLP forward pass (interface stub).

NumPy is the default and only tested backend. This module documents *where* a kernel-backed
``forward`` would slot in: subclass :class:`~foresight_gpu.models.mlp.MLPModel` and override
:meth:`forward` with a PyOpenCL implementation that keeps the same shapes
(``X: [n_samples, n_features]``, ``params: [n_particles, n_params]`` -> ``[n_samples,
n_particles]``). No kernels ship yet, so instantiating and calling this raises a clear
error. Kernels arrive with the ``foresight_gpu[opencl]`` extra in a later iteration.
"""

from .mlp import MLPModel


class MLPModelOpenCL(MLPModel):
    """Placeholder OpenCL-backed MLP. Use :class:`MLPModel` (NumPy) for now."""

    def forward(self, X, params):
        raise NotImplementedError(
            "The OpenCL backend is not shipped yet. Use MLPModel (NumPy), or install "
            "foresight_gpu[opencl] and override forward() with a kernel implementation "
            "(same shapes as MLPModel.forward)."
        )
