# ACDC Optim: A Unified Optimization Framework for  Multi-channel Dynamic $\bf{B_0}$ Field Control

A unified optimization framework for **multi‑channel, dynamic $\bf{B_0}$ field control**
in MRI.  Supports both static and dynamic (time‑resolved) shimming with the
AC/DC array coil and Rev.D waveform generator & power amplifier system.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9, and the following dependencies:
+ `numpy>=1.20`
+ `scipy>=1.7`
+ `quadprog>=0.1.8` (for QP solvers)
+ `torch>=2.0` (for GPU‑accelerated FISTA solver)

## Quick start

```python
import numpy as np
from ACDC_Optim import solve_shim_static_qp

# Synthetic example — replace with real fieldmap data
b0map = np.random.randn(1000)          # M = 1000 spatial voxels
fieldmap = np.random.randn(32, 1000)   # 32 coils × 1000 voxels

currents, nrmse = solve_shim_static_qp(
    b0map, fieldmap,
    amp_limit=2.0,    # ±2 A per coil
    l1_limit=15.0,    # 15 A total across all coils
)
print(f"NRMSE = {nrmse:.4f}")
```

## Problem formulation

### Static shimming (QP)

For a single time frame, find coil currents $\mathbf{x}\in\mathbb{R}^{N_c}$ that
best cancel a measured fieldmap $\mathbf{d}\in\mathbb{R}^{M}$:

$$
\min_{\mathbf{x}}\ \frac{1}{2}\|\mathbf{W}^T\mathbf{x} - \mathbf{d}\|_2^2
+ \frac{\lambda}{2}\|\mathbf{x}\|_2^2
\quad\text{s.t.}\quad
|x_c|\le I_{\max},\ \ \sum_c|x_c|\le I_{\Sigma,\max}
$$

where $\mathbf{W}\in\mathbb{R}^{N_c\times M}$ is the ACDC coil sensitivity matrix.
Solved exactly via Quadratic Programming.

### Dynamic shimming (FISTA)

For time‑varying fields $\mathbf{B}\in\mathbb{R}^{T\times M}$ with an amplifier
system impulse response convolution operator $\mathbf{C}\in\mathbb{R}^{T\times T}$ (SIRF):

$$
\min_{\mathbf{X}}\ \frac{1}{2}\|\mathbf{C}\mathbf{X}\mathbf{W}
- \mathbf{B}\|_F^2 + \frac{\lambda}{2}\|\mathbf{X}\|_F^2
\quad\text{s.t.}\quad
|X_{t,c}|\le I_{\max},\ \ \sum_c|X_{t,c}|\le I_{\Sigma,\max}\ \forall t
$$

Solved via FISTA (Nesterov‑accelerated projected gradient) with FFT‑based
convolution for the SIRF terms.

> 📖 **Full derivation** — See [`Notes/ACDC optim_notes.md`](Notes/ACDC%20optim_notes.md)
> for the complete mathematical derivation, including the auxiliary‑variable
> reformulation of the L1 constraint, the `quadprog` standard form, numerical
> scaling, and the FISTA algorithm details.

## Solvers

| Function | Scope | Algorithm | SIRF |
|---|---|---|---|
| `solve_shim_static_qp` | Single time frame | QP — exact convex optimum (`quadprog`) | — |
| `solve_shim_waveform_qp` | Time‑resolved, no SIRF | QP per time point (parallel, `joblib`) | — |
| `solve_shim_waveform_fista` | Time‑resolved, with SIRF | FISTA — NumPy / scipy FFT convolution | ✓ |
| `solve_shim_waveform_fista_torch` | Time‑resolved, with SIRF, GPU | FISTA — PyTorch / CUDA `conv1d` | ✓ |

Both FISTA solvers accept `sirf_kernel` (FFT‑based, recommended) or
`conv_matrix` (legacy dense Toeplitz) for modelling the amplifier chain's
temporal response.  See the docstrings for full API details.

## Examples

Jupyter notebooks are provided in `examples/`:

| Notebook | Description |
|---|---|
| `ACDC_static_shim.ipynb` | Single‑frame shimming with a quadratic target field + L‑curve parameter sweep |
| `ACDC_dynamic_shim.ipynb` | Time‑resolved shimming — compares QP, NumPy FISTA, and Torch FISTA with a first‑order SIRF model |

> **Example data** — The notebooks reference `ACDC_fieldmap/ACDC_3T.npz`
> and `ACDC_fieldmap/ACDC_7T.npz`, which are not publicly distributed.  To request access,
> please email hhong6@mgh.harvard.edu

## License

This project is under the [MIT](LICENSE) License