"""
ACDC_Optim — Optimal shim-coil current computation for B₀ field inhomogeneity
cancellation.

**Solvers**

- ``solve_shim_static_qp``            — QP for a single time frame.
- ``solve_shim_waveform_qp``          — Per‑time‑point QP (parallel).
- ``solve_shim_waveform_fista``       — FISTA for time‑resolved waveforms
                                        (NumPy, supports SIRF pre‑emphasis).
- ``solve_shim_waveform_fista_torch`` — GPU‑capable FISTA (PyTorch,
                                        FFT‑based SIRF convolution).

**Utilities**

- ``build_conv_matrix``        — Build a dense Toeplitz convolution matrix
                                 from a SIRF kernel.
- ``plot_convergence_curves``  — Plot loss & NRMSE convergence diagnostics.
"""

# === Public API — Solvers ==================================================

from .ACDC_optimization import (
    solve_shim_static_qp,
    solve_shim_waveform_fista,
    solve_shim_waveform_fista_torch,
    solve_shim_waveform_qp,
)

# === Public API — Utilities ================================================

from .ACDC_optimization import (
    build_conv_matrix,
    plot_convergence_curves,
)

__all__ = [
    # Solvers
    "solve_shim_static_qp",
    "solve_shim_waveform_qp",
    "solve_shim_waveform_fista",
    "solve_shim_waveform_fista_torch",
    # Utilities
    "build_conv_matrix",
    "plot_convergence_curves",
]
