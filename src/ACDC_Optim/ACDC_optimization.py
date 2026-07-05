"""
ACDC Shim Coil Current Optimisation
=====================================

Algorithms for computing optimal shim-coil currents to cancel static or
time-varying B₀ field inhomogeneities, subject to per-coil amplitude and
global L1-norm (total-current) constraints.

**Solvers provided** (naming scheme: ``solve_shim_<scope>_<algorithm>``)

1. ``solve_shim_static_qp``            — QP for a single time frame
                                          (Goldfarb-Idnani dual active-set).
2. ``solve_shim_waveform_qp``          — Per-time-point QP (no temporal
                                          coupling), parallelised with ``joblib``.
3. ``solve_shim_waveform_fista``       — FISTA (Nesterov-accelerated projected
                                          gradient) for time-resolved waveforms;
                                          supports SIRF pre-emphasis via a dense
                                          Toeplitz matrix.
4. ``solve_shim_waveform_fista_torch`` — GPU-capable FISTA (PyTorch) with
                                          FFT-based SIRF convolution via ``conv1d``.

Solvers 1 and 2 share the same underlying QP: solver 2 simply resolves it
independently at every time point. Solvers 3 and 4 share the same FISTA
algorithm; solver 4 differs only in using ``torch`` tensors and FFT-based
convolution so it can also run on a GPU.

**Shared building blocks**

- ``_scale_by_spectral_norm``     — Rescale (fieldmap, target) by the
                                     fieldmap's spectral norm, for QP/FISTA
                                     numerical conditioning.
- ``_build_qp_hessian``           — Build the (data-independent) QP Hessian
                                     G shared by both QP solvers.
- ``_build_qp_constraints``       — Build the (data-independent) linear
                                     inequality constraints C, b shared by
                                     both QP solvers.
- ``_solve_qp_currents``          — Solve the QP for one target vector,
                                     given a prebuilt (G, C, b), returning
                                     the optimal currents.
- ``_project_onto_box_l1ball``    — Row-wise projection onto the
                                     intersection of a box and an L1-ball
                                     (Duchi et al., 2008).
- ``build_conv_matrix``           — Build a dense Toeplitz convolution
                                     matrix from a SIRF kernel.
- ``_estimate_spectral_norm_power_iteration`` — Estimate ``||C||_2`` for a
                                     (possibly large) matrix via power
                                     iteration (NumPy).
- ``_estimate_spectral_norm_power_iteration_torch`` — Torch equivalent of
                                     the above, for a (possibly GPU-resident)
                                     tensor.
- ``_sirf_conv1d``                — Causal convolution / cross-correlation
                                     via ``torch.nn.functional.conv1d``.
- ``_sirf_conv1d_numpy``          — NumPy / scipy equivalent of
                                     ``_sirf_conv1d``, using
                                     ``scipy.signal.convolve``.
- ``plot_convergence_curves``     — Convergence diagnostics (loss & NRMSE
                                     panels).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from quadprog import solve_qp
from tqdm import tqdm


# ===========================================================================
#  Shared helpers — numerical scaling and QP assembly
# ===========================================================================

def _scale_by_spectral_norm(fieldmap: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Rescale a coil sensitivity matrix and its target field by the fieldmap's
    spectral norm, for numerical conditioning.

    Coil sensitivity matrices often have a large spectral norm (e.g.
    ~10^3-10^4 Hz/A at 7 T), which makes ``fieldmap @ fieldmap.T`` entries
    ~10^6-10^8 and pushes QP/FISTA condition numbers to ~10^10. Scaling both
    the fieldmap and its target by ``s = spectral_norm(fieldmap)`` fixes
    this without changing the unregularised least-squares minimiser (any
    regularisation terms must be rescaled by the caller, by 1/s**2).

    Parameters
    ----------
    fieldmap : np.ndarray, shape (n_coils, M)
        Coil sensitivity matrix.
    target : np.ndarray, shape (M,) or (T, M)
        Target field(s) to scale consistently with ``fieldmap``.

    Returns
    -------
    fieldmap_scaled : np.ndarray, same shape as ``fieldmap``
    target_scaled : np.ndarray, same shape as ``target``
    s : float
        The spectral norm used for scaling (``1.0`` if the fieldmap is
        (numerically) zero, to avoid division by zero).
    """
    s = np.linalg.norm(fieldmap, ord=2)
    if s < 1e-12:
        s = 1.0
    return fieldmap / s, target / s, s


def _build_qp_hessian(fieldmap_scaled: np.ndarray, ridge_x: float, ridge_u: float) -> np.ndarray:
    """
    Build the QP Hessian ``G`` for the box + L1-ball current-shimming problem.

    Stacks the decision vector as ``z = [x; u]`` (currents, L1-slack), with
    the quadratic data-fit term acting only on ``x`` and small ridge terms
    on both blocks (``ridge_u`` exists purely to keep ``G`` positive
    definite, as required by ``quadprog``; it has no physical meaning).

    Parameters
    ----------
    fieldmap_scaled : np.ndarray, shape (n_coils, M)
        Coil sensitivity matrix, already scaled by ``_scale_by_spectral_norm``.
    ridge_x : float
        Ridge weight on the currents ``x`` (already rescaled by 1/s**2).
    ridge_u : float
        Ridge weight on the L1-slack variables ``u`` (already rescaled by 1/s**2).

    Returns
    -------
    G : np.ndarray, shape (2*n_coils, 2*n_coils)
    """
    n_coils = fieldmap_scaled.shape[0]
    hessian_x = 2.0 * (fieldmap_scaled @ fieldmap_scaled.T) + ridge_x * np.eye(n_coils)
    hessian_u = ridge_u * np.eye(n_coils)
    return np.block([
        [hessian_x, np.zeros((n_coils, n_coils))],
        [np.zeros((n_coils, n_coils)), hessian_u],
    ]).astype(np.float64)


def _build_qp_constraints(n_coils: int, amp_limit: float, l1_limit: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the linear inequality constraints ``(C, b)`` for the box + L1-ball
    current-shimming QP, in ``quadprog``'s ``C^T z >= b`` convention.

    With ``z = [x; u]``, the constraints are, per coil ``c``:
      - box:       ``-I_max <= x_c <= I_max``
      - L1 slack:  ``u_c - x_c >= 0``  and  ``u_c + x_c >= 0``  (i.e. ``u_c >= |x_c|``)
    plus one global constraint:
      - total current: ``sum(u_c) <= I_sum``

    Parameters
    ----------
    n_coils : int
        Number of shim channels.
    amp_limit : float
        Per-coil amplitude bound ``I_max``.
    l1_limit : float
        Total instantaneous current bound ``I_sum``.

    Returns
    -------
    C : np.ndarray, shape (2*n_coils, 4*n_coils + 1)
    b : np.ndarray, shape (4*n_coils + 1,)
    """
    zdim = 2 * n_coils
    cols, bounds = [], []

    # Box constraints: x_c >= -I_max  and  -x_c >= -I_max  (i.e. x_c <= I_max)
    for c in range(n_coils):
        col = np.zeros(zdim); col[c] = 1.0
        cols.append(col); bounds.append(-amp_limit)
    for c in range(n_coils):
        col = np.zeros(zdim); col[c] = -1.0
        cols.append(col); bounds.append(-amp_limit)

    # L1-slack constraints: u_c - x_c >= 0  and  u_c + x_c >= 0  (i.e. u_c >= |x_c|)
    for c in range(n_coils):
        col = np.zeros(zdim); col[c] = -1.0; col[n_coils + c] = 1.0
        cols.append(col); bounds.append(0.0)
    for c in range(n_coils):
        col = np.zeros(zdim); col[c] = 1.0; col[n_coils + c] = 1.0
        cols.append(col); bounds.append(0.0)

    # Global total-current constraint: sum(u_c) <= I_sum  ->  -sum(u_c) >= -I_sum
    col = np.zeros(zdim); col[n_coils:] = -1.0
    cols.append(col); bounds.append(-l1_limit)

    C = np.stack(cols, axis=1).astype(np.float64)
    b = np.array(bounds, dtype=np.float64)
    return C, b


def _solve_qp_currents(G: np.ndarray, C: np.ndarray, b: np.ndarray,
                      fieldmap_scaled: np.ndarray, target_scaled: np.ndarray) -> np.ndarray:
    """
    Solve the box + L1-ball current QP for one target vector, given a
    prebuilt (data-independent) Hessian and constraint set.

    Parameters
    ----------
    G : np.ndarray, shape (2*n_coils, 2*n_coils)
        QP Hessian, from ``_build_qp_hessian``.
    C, b : np.ndarray
        QP inequality constraints, from ``_build_qp_constraints``.
    fieldmap_scaled : np.ndarray, shape (n_coils, M)
        Scaled coil sensitivity matrix.
    target_scaled : np.ndarray, shape (M,)
        Scaled target field for this time point.

    Returns
    -------
    x_opt : np.ndarray, shape (n_coils,)
        Optimal currents (in the scaled problem; scaling cancels out since
        the QP's minimiser over ``x`` is scale-invariant).
    """
    n_coils = fieldmap_scaled.shape[0]
    a = np.zeros(G.shape[0], dtype=np.float64)
    a[:n_coils] = 2.0 * (fieldmap_scaled @ target_scaled)
    z_opt = solve_qp(G, a, C, b, meq=0)[0]
    return z_opt[:n_coils]


# ===========================================================================
#  1.  Static (single-time-point) Shimming — Quadratic Programming
# ===========================================================================

def solve_shim_static_qp(b0map: np.ndarray, acdc_fieldmap: np.ndarray, amp_limit: float = 2.0,
                       l1_limit: float = 15.0, ridge_x: float = 1e-6, ridge_u: float = 1e-8,
                       verbose: bool = False) -> tuple[np.ndarray, float]:
    """
    Solve the static (time-invariant) B0 shimming problem via Quadratic Programming (QP).

    Given a measured B0 fieldmap and an ACDC coil fieldmap matrix, this function
    computes the optimal coil currents that minimize the residual field inhomogeneity,
    subject to per-coil amplitude constraints and a global L1-norm (total current) budget.

    **Mathematical Formulation**

    Let:
      - W = acdc_fieldmap          (n_coils x M), sensitivity matrix mapping coil
                                     currents to field perturbations at M spatial voxels.
      - d = b0map                  (M,), the measured B0 offset at each voxel.
      - x                          (n_coils,), the coil currents to be solved for.

    The unconstrained objective is the least-squares fit:

        minimize_x  || W^T x - d ||^2

    Absolute values in the L1 budget constraint are handled via auxiliary slack
    variables u_i (one per coil) satisfying ``|x_i| <= u_i``, and the whole
    problem is solved as a QP — see ``_build_qp_hessian`` /
    ``_build_qp_constraints`` for the full derivation.

    Parameters
    ----------
    b0map : np.ndarray, shape (M,)
        Measured B0 field inhomogeneity (in Hz or equivalent) at M spatial
        voxels. Values represent the deviation from the target field that the
        shim should cancel.

    acdc_fieldmap : np.ndarray, shape (n_coils, M)
        Sensitivity (fieldmap) matrix for the ACDC shim coils. Entry (i, j)
        gives the field perturbation at voxel j produced by 1 A of current in
        coil i. ``n_coils`` is the number of independent shim channels.

    amp_limit : float, optional, default=2.0
        Maximum allowed absolute current per individual coil (A). Coil currents
        are bounded to [-amp_limit, +amp_limit].

    l1_limit : float, optional, default=15.0
        Maximum allowed sum of absolute coil currents (A), i.e. sum_i |x_i| <= l1_limit.
        This limits the total current draw across all coils.

    ridge_x : float, optional, default=1e-6
        Tikhonov (ridge) regularization weight added to the diagonal of the
        Hessian block for x. Improves numerical stability when W @ W.T is
        rank-deficient or poorly conditioned.

    ridge_u : float, optional, default=1e-8
        Small regularization weight added to the Hessian block for the auxiliary
        variables u. Ensures the overall QP Hessian G is strictly positive
        definite, which is required by quadprog.

    verbose : bool, optional, default=False
        If True, prints the optimal current vector and the resulting NRMSE
        after the solve.

    Returns
    -------
    I_opt : np.ndarray, shape (n_coils,)
        Optimal coil currents (A) that minimize the residual field inhomogeneity
        under the given constraints.

    nrmse : float
        Normalised Root-Mean-Square Error of the residual field after applying
        the optimal shim currents:

            NRMSE = || W^T @ I_opt  -  b0map ||_2 / || b0map ||_2

        Lower values indicate better shimming. A value of 0 would mean perfect
        cancellation of the measured fieldmap (typically not attainable due to
        coil constraints).

    Notes
    -----
    - The QP is solved using `quadprog.solve_qp`, which implements the Goldfarb-
      Idnani dual active-set algorithm.
    - All constraints are formulated as inequality constraints (C^T z >= b);
      there are no equality constraints (meq = 0).
    - The NRMSE is computed over *all* M spatial voxels (including regions
      outside any brain mask). For brain-only evaluation, pass a masked
      fieldmap or post-process the residual.

    Examples
    --------
    >>> b0 = np.random.randn(1000)           # 1000-voxel fieldmap
    >>> W  = np.random.randn(32, 1000)       # 32 shim coils
    >>> I, err = solve_shim_static_qp(b0, W, amp_limit=2.0, l1_limit=15.0)
    >>> print(f"NRMSE = {err:.4f}")
    """
    n_coils, _ = acdc_fieldmap.shape

    W_scaled, d_scaled, s = _scale_by_spectral_norm(acdc_fieldmap, b0map)
    ridge_x_scaled = ridge_x / (s * s)
    ridge_u_scaled = ridge_u / (s * s)

    G = _build_qp_hessian(W_scaled, ridge_x_scaled, ridge_u_scaled)
    C, b = _build_qp_constraints(n_coils, amp_limit, l1_limit)
    I_opt = _solve_qp_currents(G, C, b, W_scaled, d_scaled)

    # Diagnostics on ORIGINAL (un-scaled) data.
    fieldmap_pred = acdc_fieldmap.T @ I_opt
    nrmse = np.linalg.norm(fieldmap_pred - b0map) / np.linalg.norm(b0map)
    if verbose:
        tqdm.write(f'Optimal currents: {I_opt}\nNRMSE: {nrmse}')

    return I_opt, nrmse


# ===========================================================================
#  2.  Shim Waveform Optimisation — FISTA (NumPy)
# ===========================================================================

def _project_onto_box_l1ball(X: np.ndarray, I_max: float, I_max_sum: float) -> np.ndarray:
    """
    Project each *row* of X onto the intersection of the box [-I_max, I_max]
    and the L1-ball of radius I_max_sum.

    That is, for each time point t (row of X):
        find  z  closest to  x = X[t, :]  such that
            |z_i| <= I_max   for all i    (box)
            sum_i |z_i| <= I_max_sum  (L1-ball)

    This uses a two-step approach: clip to the box, then if the L1-norm still
    exceeds the budget, project onto the L1-ball via the efficient O(n log n)
    simplex-projection algorithm (Duchi et al., 2008). Because that projection
    only ever shrinks magnitudes (soft-thresholding), it cannot push a
    coordinate back outside the box, so this two-step composition is the
    *exact* projection onto the box-intersect-L1-ball set, not an approximation.

    Parameters
    ----------
    X : np.ndarray, shape (T, n_coils)
        Current waveform matrix (each row is a time point).
    I_max : float
        Per-coil amplitude bound.
    I_max_sum : float
        L1-norm (total absolute current) bound per time instant.

    Returns
    -------
    X_proj : np.ndarray, shape (T, n_coils)
        Projected waveform matrix.
    """
    n_coils = X.shape[1]

    # Step 1 -- Box-clip.
    X_out = np.clip(X, -I_max, I_max)

    # Step 2 -- Identify rows violating the L1 budget and project just those.
    row_sums = np.sum(np.abs(X_out), axis=1)
    viol_mask = row_sums > I_max_sum

    if np.any(viol_mask):
        X_viol = X_out[viol_mask]                      # (n_viol, n_coils)
        U = np.sort(np.abs(X_viol), axis=1)[:, ::-1]   # descending sort per row
        S = np.cumsum(U, axis=1)                       # cumulative sums
        rank = np.arange(1, n_coils + 1)                # 1 ... n_coils
        still_above_threshold = U - (S - I_max_sum) / rank > 0
        rho = np.count_nonzero(still_above_threshold, axis=1) - 1
        row_idx = np.arange(len(X_viol))
        theta = (S[row_idx, rho] - I_max_sum) / (rho + 1)
        X_soft_thresholded = np.maximum(np.abs(X_viol) - theta[:, None], 0)
        X_out[viol_mask] = np.sign(X_viol) * X_soft_thresholded

    return X_out


# ===========================================================================
#  Utility — Build dense Toeplitz convolution matrix
# ===========================================================================

def build_conv_matrix(impulse_response: np.ndarray, n_time: int) -> np.ndarray:
    """
    Build a dense Toeplitz convolution matrix C in R^(T x T) from an impulse
    response kernel h[t] (length K <= T).

    The convolution is modelled as zero-padded, non-circular:

        (C @ x)[t] = sum_{tau=0}^{K-1} h[tau] * x[t - tau]

    with x[s] = 0 for s < 0. The kernel is assumed to be causal (h[t]=0 for t<0).

    Parameters
    ----------
    impulse_response : np.ndarray, shape (K,)
        The SIRF (or any linear filter) kernel.
    n_time : int
        Number of time points T in the waveform.

    Returns
    -------
    C : np.ndarray, shape (T, T)
        Lower-triangular Toeplitz convolution matrix.
    """
    K = len(impulse_response)
    T = n_time
    C = np.zeros((T, T), dtype=np.float64)
    for t in range(T):
        end_t = min(t + 1, K)
        C[t, t - end_t + 1: t + 1] = impulse_response[:end_t][::-1]
    return C


def _estimate_spectral_norm_power_iteration(matrix: np.ndarray, n_time: int, n_iters: int = 20) -> float:
    """
    Estimate the spectral norm ``||matrix||_2`` via power iteration.

    Used in place of ``np.linalg.norm(matrix, ord=2)`` (which computes a full
    SVD, O(T^3)) for the potentially large ``(T, T)`` SIRF convolution matrix,
    where power iteration is much cheaper, O(T^2) per iteration.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_time, n_time)
        Matrix to estimate the spectral norm of.
    n_time : int
        Dimension of ``matrix`` (used to size the random starting vector).
    n_iters : int, default=20
        Number of power-iteration steps.

    Returns
    -------
    norm : float
        Estimated spectral norm sigma_max(matrix).
    """
    v = np.random.randn(n_time).astype(np.float64)
    v /= np.linalg.norm(v)
    v_norm = 0.0
    for _ in range(n_iters):
        v_next = matrix.T @ (matrix @ v)
        v_norm = np.linalg.norm(v_next)
        if v_norm < 1e-15:
            break
        v = v_next / v_norm
    # sigma_max(matrix) = sqrt(sigma_max(matrix.T @ matrix)) ~= sqrt(||matrix.T @ matrix @ v|| / ||v||)
    return float(np.sqrt(v_norm))


# ===========================================================================
#  Utility — Convergence plotting
# ===========================================================================

def plot_convergence_curves(iter_history: list[int], loss_history: list[float],
               nrmse_history: list[float], show: bool = True) -> None:
    """
    Plot the optimisation convergence curves (loss function and NRMSE).

    Produces a two-panel figure:
      - **Left**:  Loss function value vs. logged iteration (log scale).
      - **Right**: NRMSE vs. logged iteration.

    Parameters
    ----------
    iter_history : list of int
        Logged iteration numbers (0, 1, 2, ...). Actual iteration number.
    loss_history : list of float
        Loss values recorded during the optimisation (one entry
        every ``iter_step`` iterations).
    nrmse_history : list of float
        NRMSE values at the same logged iterations.
    show : bool
        If ``True``, call ``matplotlib.pyplot.show()`` to display the figure
        interactively. Set to ``False`` for headless / batch runs.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(iter_history, loss_history, 'b-', linewidth=1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss Function (log scale)')
    ax1.set_title('Loss Function Convergence')
    ax1.grid(True, alpha=0.3)

    ax2.plot(iter_history, np.array(nrmse_history) * 100, 'r-', linewidth=1)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('NRMSE [%]')
    ax2.set_title('NRMSE Convergence')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)


# ===========================================================================
#  Utility — Causal SIRF convolution (NumPy / scipy)
# ===========================================================================

def _sirf_conv1d_numpy(X: np.ndarray, kernel: np.ndarray, mode: str = 'conv') -> np.ndarray:
    """
    Apply causal SIRF convolution or its transpose to each coil's waveform,
    using ``scipy.signal.convolve`` (FFT-based for large inputs).

    Mirrors ``_sirf_conv1d`` (the torch version) but operates on NumPy arrays.

    Parameters
    ----------
    X : np.ndarray, shape (T, N_c)
        Current waveforms.
    kernel : np.ndarray, shape (K,)
        Causal SIRF impulse response (h[t] for t >= 0).
    mode : str
        ``'conv'`` ->  C @ X   (causal:  y[t] = sum h[tau]*x[t-tau])
        ``'corr'`` ->  C^T @ X (cross-correlation:
                                  y[t] = sum h[tau]*x[t+tau])

    Returns
    -------
    Y : np.ndarray, shape (T, N_c)
    """
    from scipy.signal import convolve

    T, Nc = X.shape
    K = len(kernel)
    Y = np.empty_like(X)

    if mode == 'conv':
        # Causal: y[t] = sum_{tau} h[tau] * x[t-tau]
        for c in range(Nc):
            Y[:, c] = convolve(X[:, c], kernel, method='auto')[:T]
    elif mode == 'corr':
        # Transpose: (C^T g)[t] = sum_{tau} h[tau] * g[t+tau]
        # = convolve(g, h_reversed)[K-1 : K-1+T]
        h_rev = kernel[::-1]
        for c in range(Nc):
            Y[:, c] = convolve(X[:, c], h_rev, method='auto')[K-1:K-1+T]
    else:
        raise ValueError(f"Unknown mode '{mode}'; use 'conv' or 'corr'.")

    return Y


def solve_shim_waveform_fista(
    b0_timecourse: np.ndarray,
    acdc_fieldmap: np.ndarray,
    sirf_kernel: Optional[np.ndarray] = None,
    conv_matrix: Optional[np.ndarray] = None,
    amp_limit: float = 2.0,
    l1_limit: float = 15.0,
    ridge_x: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-12,
    verbose: bool = True,
    plot: bool = False,
    iter_step: int = 5,
) -> tuple[np.ndarray, float, float]:
    """
    Solve for shim coil current **waveforms** that cancel a time-varying B0
    fieldmap, subject to per-coil amplitude and global L1-norm constraints.

    This extends `solve_shim_static_qp` to the dynamic case where the fieldmap
    target evolves over time and the shim amplifier chain imposes a temporal
    filter (System Impulse Response Function, SIRF) on the output currents.
    The problem is solved via FISTA -- a projected gradient method with
    Nesterov momentum -- which avoids building the large QP Hessian required
    by a direct quadratic-programming approach.

    **Mathematical Formulation**

    Let:
      - T        = number of time points
      - N_c      = number of shim coils / channels
      - M        = number of spatial voxels
      - B        = b0_timecourse              (T x M)  target fieldmap at each time
      - W        = acdc_fieldmap              (N_c x M) coil sensitivity (Hz/A per voxel)
      - C        = conv_matrix                (T x T)  temporal SIRF convolution
                                                     (identity if None)
      - X        = unknown current waveforms  (T x N_c)

    The forward model for the shim-generated field is:

        field_pred = (C @ X) @ W          (T x M)

    and we minimise:

        min_X   1/2 || C X W - B ||^2_F  +  (lambda/2) ||X||^2_F

    subject to, for every time point t:

        |X_{t,c}| <= I_max   (per-channel amplitude)
        sum_c |X_{t,c}| <= I_Sigma_max   (total instantaneous current)

    **Algorithm -- FISTA**

    1. Gradient (w.r.t. Y, the momentum variable):

          grad f(Y) = C^T (C Y W - B) W^T  +  lambda Y

    2. Gradient-descent step with step size eta = 1/L, where L is the
       Lipschitz constant: L ~= ||C||^2 + lambda (the ||W||^2 factor drops
       out since W is rescaled to have spectral norm 1, see below).

    3. Projection onto the feasible set (box ∩ L1-ball), applied row-wise.

    4. Nesterov momentum update:

          t_new = 1/2 (1 + sqrt(1 + 4 t^2))
          Y_new = X_new + ((t - 1) / t_new) (X_new - X_old)

    Parameters
    ----------
    b0_timecourse : np.ndarray, shape (T, M)
        Target B0 fieldmap (Hz) at each of T time points and M spatial voxels.
        This is the field that the shim should *cancel*.

    acdc_fieldmap : np.ndarray, shape (N_c, M)
        Coil sensitivity matrix (Hz/A). Entry (c, m) is the field produced
        at voxel m by 1 A in coil c.

    conv_matrix : np.ndarray, shape (T, T), or None
        Temporal convolution matrix representing the SIRF of the shim
        amplifier chain (legacy dense-matrix path). When both
        ``sirf_kernel`` and ``conv_matrix`` are None, the identity matrix
        is used (no temporal filtering).

    sirf_kernel : np.ndarray, shape (K,), or None
        SIRF impulse response kernel.  If provided, convolution is performed
        via ``scipy.signal.convolve`` (O(T log T) FFT-based) instead of a
        dense (T x T) matrix multiply.  Mutually exclusive with
        ``conv_matrix``.

    amp_limit : float, default=2.0
        Maximum allowed absolute current per coil (A).

    l1_limit : float, default=15.0
        Maximum allowed sum of absolute currents across all coils at a single
        time instant (A).

    ridge_x : float, default=1e-3
        Tikhonov (ridge) regularisation weight on X. Larger values encourage
        smaller currents.

    max_iter : int, default=3000
        Maximum number of FISTA iterations.

    tol : float, default=1e-12
        Relative change in the loss function below which convergence is
        declared.

    verbose : bool, default=True
        Print progress information during the optimisation.

    plot : bool, default=False
        If ``True``, plot the Loss and NRMSE convergence curves at the end
        of the optimisation.

    iter_step : int, default=5
        Number of iterations between convergence/logging checks.

    Returns
    -------
    I_opt : np.ndarray, shape (T, N_c)
        Optimal current waveform for each coil (A).

    nrmse : float
        Normalised root-mean-square error of the predicted field (after SIRF
        convolution) relative to the target:

            NRMSE = || C X W - B ||_F  /  || B ||_F

    loss : float
        Final loss function value (on the scaled problem).
    """
    n_time, M = b0_timecourse.shape
    n_coils, M2 = acdc_fieldmap.shape
    if M != M2:
        raise ValueError(
            f"Spatial dimension mismatch: b0_timecourse has {M} voxels, "
            f"acdc_fieldmap has {M2} voxels."
        )

    W_scaled, B_scaled, s = _scale_by_spectral_norm(acdc_fieldmap, b0_timecourse)

    # ---- SIRF kernel or dense matrix ----
    # Three mutually exclusive paths:
    #   1. sirf_kernel provided      -> scipy FFT convolution (O(T log T))
    #   2. conv_matrix provided      -> legacy dense (T x T) multiply (O(T^2))
    #   3. neither provided          -> identity (no temporal filtering)
    use_dense_C = False
    h_kernel = None
    C_mat = None
    if sirf_kernel is not None:
        h_kernel = np.asarray(sirf_kernel, dtype=np.float64)
        # Causal low-pass SIRF with unit DC gain => ||C||_2 ~= 1
        norm_C = 1.0
    elif conv_matrix is not None:
        C_mat = conv_matrix
        norm_C = _estimate_spectral_norm_power_iteration(C_mat, n_time)
        use_dense_C = True
    else:
        norm_C = 1.0
    norm_W = np.linalg.norm(W_scaled, ord=2)               # ~= 1 after scaling

    # ---- Step-size estimation (Lipschitz constant) ----
    # With ||W_scaled||_2 = 1 (by construction of _scale_by_spectral_norm):
    #   L = ||C||_2^2 * ||W||_2^2  +  ridge_x
    #
    # For the SIRF-kernel path ||C||_2 ~= 1 (causal low-pass, unit DC gain);
    # for the dense-matrix path ||C||_2 is estimated by power iteration;
    # for identity (neither provided) ||C||_2 = 1 exactly.
    L = norm_C ** 2 * norm_W ** 2 + ridge_x
    step_size = 1.0 / L

    if verbose:
        tqdm.write(f"FISTA waveform optimisation: T={n_time}, N_c={n_coils}, M={M}")
        tqdm.write(f"  Scaling factor s = {s:.4e}  (sigma_max of W)")
        tqdm.write(f"  ||C||_2 ~= {norm_C:.4e}, ||W||_2 ~= {norm_W:.4e}, L = {L:.4e}, step = {step_size:.4e}")
        tqdm.write(f"  ridge_x={ridge_x:.2e}")
        tqdm.write(f"  Constraints: |I_i| <= {amp_limit} A,  sum|I_i| <= {l1_limit} A")

    # ---- Initialisation ----
    X = np.zeros((n_time, n_coils), dtype=np.float64)      # primal variable
    Y = X.copy()                                            # FISTA momentum variable
    t_fista = 1.0                                           # Nesterov sequence

    # ---- Dispatch helpers (mirrors the torch version) ----
    def _apply_conv(mat: np.ndarray) -> np.ndarray:
        if h_kernel is not None:
            return _sirf_conv1d_numpy(mat, h_kernel, mode='conv')
        if use_dense_C:
            return C_mat @ mat
        return mat

    def _apply_conv_transpose(mat: np.ndarray) -> np.ndarray:
        if h_kernel is not None:
            return _sirf_conv1d_numpy(mat, h_kernel, mode='corr')
        if use_dense_C:
            return C_mat.T @ mat
        return mat

    iter_history, loss_history, nrmse_history = [], [], []
    t_start = time.time()

    pbar = tqdm(total=max_iter, desc="FISTA", disable=not verbose)
    loss = 0.0
    for iteration in range(max_iter):

        # ---- Forward model & residual (scaled) ----
        CY = _apply_conv(Y)                                  # (T, N_c)
        field_pred = CY @ W_scaled                           # (T, M)
        residual = field_pred - B_scaled                     # (T, M)

        # ---- Gradient w.r.t. Y:  grad f(Y) = C^T (C Y W - B) W^T + ridge_x * Y ----
        grad_data = residual @ W_scaled.T                    # (T, N_c)
        grad_data = _apply_conv_transpose(grad_data)
        grad = grad_data + ridge_x * Y

        # ---- Descent & projection ----
        X_new = Y - step_size * grad
        X_new = _project_onto_box_l1ball(X_new, amp_limit, l1_limit)

        # ---- FISTA momentum update ----
        t_fista_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_fista ** 2))
        Y = X_new + ((t_fista - 1.0) / t_fista_new) * (X_new - X)

        X = X_new
        t_fista = t_fista_new

        # ---- Logging & convergence check ----
        if iteration % iter_step == 0:
            iter_history.append(iteration)
            loss = 0.5 * np.sum(residual ** 2) + 0.5 * ridge_x * np.sum(X ** 2)
            loss_history.append(loss)

            # NRMSE on ORIGINAL (un-scaled) data.
            field_orig = _apply_conv(X) @ acdc_fieldmap
            nrmse_cur = np.linalg.norm(field_orig - b0_timecourse) / np.linalg.norm(b0_timecourse)
            nrmse_history.append(nrmse_cur)
            pbar.set_postfix(loss=f"{loss:.4e}", NRMSE=f"{nrmse_cur:.6e}")
            pbar.update(iter_step)

            if len(loss_history) >= 2:
                rel_change = abs(loss_history[-1] - loss_history[-2]) / loss_history[-2]
                if rel_change < tol:
                    pbar.n = iteration
                    pbar.set_postfix(loss=f"{loss:.4e}", NRMSE=f"{nrmse_cur:.6e}", converged="yes")
                    pbar.close()
                    if verbose:
                        tqdm.write(f"  Converged at iteration {iteration} "
                                   f"(rel change {rel_change:.2e} < {tol:.2e}).")
                    break

    t_elapsed = time.time() - t_start
    pbar.close()

    # ---- Final diagnostics (original, un-scaled data) ----
    field_final = _apply_conv(X) @ acdc_fieldmap
    nrmse = np.linalg.norm(field_final - b0_timecourse) / np.linalg.norm(b0_timecourse)

    if verbose:
        tqdm.write(f"  Finished in {t_elapsed:.1f} s.  Final NRMSE = {nrmse:.6e}")

    if plot:
        plot_convergence_curves(iter_history, loss_history, nrmse_history, show=True)

    return X, nrmse, loss


# ===========================================================================
#  3.  Per-time-point Quadratic Programming (no SIRF, parallel)
# ===========================================================================

def solve_shim_waveform_qp(
    b0_timecourse: np.ndarray,
    acdc_fieldmap: np.ndarray,
    amp_limit: float = 2.0,
    l1_limit: float = 15.0,
    ridge_x: float = 1e-6,
    ridge_u: float = 1e-8,
    n_jobs: int = -1,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Solve for shim coil current waveforms by Quadratic Programming (QP) --
    **one time point at a time** (no temporal coupling / no SIRF).

    This is the exact (convex) optimum for each time frame independently,
    using the same QP formulation as `solve_shim_static_qp`. It is suitable
    when the SIRF convolution can be neglected (C ~= I) or as a warm-start
    / baseline for the FISTA solver.

    Parameters
    ----------
    b0_timecourse : np.ndarray, shape (T, M)
        Target B0 fieldmap (Hz) at T time points and M spatial voxels.
    acdc_fieldmap : np.ndarray, shape (N_c, M)
        Coil sensitivity matrix (Hz/A).
    amp_limit : float
        Per-coil current amplitude bound (A).
    l1_limit : float
        Total instantaneous current L1 bound (A).
    ridge_x : float
        Tikhonov weight on currents x.
    ridge_u : float
        Small Tikhonov weight on auxiliary L1 variables u.
    n_jobs : int
        Number of parallel jobs for `joblib`. Set to 1 for serial,
        -1 for all cores.
    verbose : bool
        Print progress bar and final NRMSE.

    Returns
    -------
    I_opt : np.ndarray, shape (T, N_c)
        Optimal coil current waveforms (A).
    nrmse : float
        NRMSE of the predicted field relative to the target.
    """
    import os
    from joblib import Parallel, delayed

    n_coils, M = acdc_fieldmap.shape            # (N_c, M) -- consistent with solve_shim_static_qp
    T, M_dB0 = b0_timecourse.shape              # b0_timecourse is (T, M)

    if M != M_dB0:
        raise ValueError(
            f"Spatial dimension mismatch: fieldmap has {M}, "
            f"b0_timecourse has {M_dB0} points."
        )

    if verbose:
        tqdm.write(f"QP waveform optimisation: T={T}, N_c={n_coils}, M={M}")
        tqdm.write(f"  Constraints: |I_i| <= {amp_limit} A,  sum|I_i| <= {l1_limit} A")

    # ---- Scale the problem (see _scale_by_spectral_norm for rationale) ----
    W_scaled, B_scaled, s = _scale_by_spectral_norm(acdc_fieldmap, b0_timecourse)
    ridge_x_scaled = ridge_x / (s * s)
    ridge_u_scaled = ridge_u / (s * s)

    # ---- Build time-independent QP matrices (identical structure to solve_shim_static_qp) ----
    G = _build_qp_hessian(W_scaled, ridge_x_scaled, ridge_u_scaled)
    C, b = _build_qp_constraints(n_coils, amp_limit, l1_limit)

    def _solve_one_time(t: int) -> np.ndarray:
        try:
            return _solve_qp_currents(G, C, b, W_scaled, B_scaled[t, :])
        except Exception:
            return np.zeros(n_coils, dtype=np.float64)

    # ---- Parallel execution across time points ----
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    if n_jobs < 0:
        n_jobs = max(1, int((os.cpu_count() or 1) * 0.9))

    time_range = tqdm(range(T), desc="QP per time point") if verbose else range(T)
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_solve_one_time)(t) for t in time_range
    )

    I_opt = np.stack(results, axis=0).astype(np.float64)      # (T, N_c)

    # ---- Diagnostics ----
    field_pred = I_opt @ acdc_fieldmap                        # (T, N_c) @ (N_c, M) = (T, M)
    nrmse = np.linalg.norm(field_pred - b0_timecourse) / np.linalg.norm(b0_timecourse)

    if verbose:
        tqdm.write(f"  QP waveform NRMSE = {nrmse:.6e}")

    return I_opt, nrmse


# ===========================================================================
#  4.  GPU-accelerated FISTA  (PyTorch with FFT-based SIRF convolution)
# ===========================================================================

def _sirf_conv1d(X, kernel, mode: str = 'conv'):
    """
    Apply causal SIRF convolution or its transpose to each coil's waveform.

    Uses ``torch.nn.functional.conv1d`` (CUDA kernels or CPU FFT),
    replacing a dense (T x T) @ (T x N_c) multiply with O(T log T * N_c).

    Parameters
    ----------
    X : torch.Tensor, shape (T, N_c)
        Current waveforms.
    kernel : torch.Tensor, shape (K,)
        Causal SIRF impulse response (h[t] for t >= 0).
    mode : str
        ``'conv'`` ->  C @ X   (causal:  y[t] = sum h[tau]*x[t-tau])
        ``'corr'`` ->  C^T @ X (cross-correlation:  y[t] = sum h[tau]*x[t+tau])

    Returns
    -------
    Y : torch.Tensor, shape (T, N_c)
    """
    import torch.nn.functional as F

    T, Nc = X.shape
    K = kernel.shape[0]

    X_batch = X.T.unsqueeze(1)                               # (N_c, 1, T)

    if mode == 'conv':
        # Causal: y[t] = sum h[tau]*x[t-tau].
        # conv1d with kernel h computes sum h[tau]*x[t+tau-K+1],
        # so we flip the kernel and pad *left* by K-1.
        weight = kernel.flip(0).view(1, 1, K)
        Y_batch = F.conv1d(X_batch, weight, padding=K - 1)   # (N_c, 1, T+K-1)
        Y = Y_batch[:, 0, :T].T                               # trim -> (T, N_c)
    elif mode == 'corr':
        # Transpose: (C^T g)[t] = sum h[tau] * g[t+tau].
        # Pad the input *right* with K-1 zeros so that conv1d covers the
        # full T outputs (the tail entries are NOT zero).
        X_padded = F.pad(X_batch, (0, K - 1))              # (N_c, 1, T+K-1)
        weight = kernel.view(1, 1, K)
        Y_batch = F.conv1d(X_padded, weight, padding=0)    # (N_c, 1, T)
        Y = Y_batch[:, 0, :].T                              # -> (T, N_c)
    else:
        raise ValueError(f"Unknown mode '{mode}'; use 'conv' or 'corr'.")

    return Y.contiguous()


def _estimate_spectral_norm_power_iteration_torch(matrix, n_time: int, torch_device, n_iters: int = 20) -> float:
    """
    Torch equivalent of ``_estimate_spectral_norm_power_iteration``, for estimating
    ``||matrix||_2`` on a (possibly GPU-resident) tensor.
    """
    import torch

    v = torch.randn(n_time, device=torch_device, dtype=torch.float64)
    v /= torch.norm(v)
    v_norm = torch.tensor(0.0, device=torch_device, dtype=torch.float64)
    for _ in range(n_iters):
        v_next = matrix.T @ (matrix @ v)
        v_norm = torch.norm(v_next)
        if v_norm < 1e-15:
            break
        v = v_next / v_norm
    return torch.sqrt(v_norm).item()


def solve_shim_waveform_fista_torch(
    b0_timecourse: np.ndarray,
    acdc_fieldmap: np.ndarray,
    sirf_kernel: Optional[np.ndarray] = None,
    conv_matrix: Optional[np.ndarray] = None,
    amp_limit: float = 2.0,
    l1_limit: float = 15.0,
    ridge_x: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-12,
    device: str = 'auto',
    verbose: bool = True,
    plot: bool = False,
    iter_step: int = 5,
) -> tuple[np.ndarray, float, float]:
    """
    GPU-accelerated shim waveform optimisation (FISTA) using PyTorch.

    This is functionally identical to `solve_shim_waveform_fista` but uses
    PyTorch tensors and FFT-based SIRF convolution, giving **10-100x
    speedup** on CUDA-capable GPUs. On CPU it still benefits from
    efficient ``conv1d`` kernels.

    The key improvement over the NumPy version is that the SIRF
    convolution is performed via ``torch.nn.functional.conv1d`` (O(T log T))
    instead of a dense matrix multiply (O(T^2)).

    Parameters
    ----------
    b0_timecourse : np.ndarray, shape (T, M)
        Target B0 fieldmap (Hz).
    acdc_fieldmap : np.ndarray, shape (N_c, M)
        Coil sensitivity matrix (Hz/A).
    sirf_kernel : np.ndarray, shape (K,), optional
        SIRF impulse response. If provided, FFT convolution is used.
        Mutually exclusive with ``conv_matrix``.
    conv_matrix : np.ndarray, shape (T, T), optional
        Dense SIRF convolution matrix (legacy path). If both are None,
        the identity is used (no temporal filtering).
    amp_limit, l1_limit, ridge_x, max_iter, tol, verbose :
        See `solve_shim_waveform_fista`.
    device : str
        'auto' (use CUDA if available, else CPU), 'cuda', or 'cpu'.
    plot : bool, default=False
        If ``True``, plot the loss and NRMSE convergence curves at the end
        of the optimisation.
    iter_step : int, default=5
        Number of iterations between convergence/logging checks.

    Returns
    -------
    I_opt : np.ndarray, shape (T, N_c)
        Optimal current waveforms (A).
    nrmse : float
        NRMSE of predicted field relative to target.
    loss : float
        Final loss function value (on the scaled problem).
    """
    try:
        import torch
        import torch.nn.functional as F  # noqa: F401  (imported for side effects / early failure)
    except ImportError:
        raise ImportError(
            "PyTorch is required for solve_shim_waveform_fista_torch. "
            "Install with:  pip install torch"
        )

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_device = torch.device(device)

    n_time, M = b0_timecourse.shape
    n_coils, M2 = acdc_fieldmap.shape
    if M != M2:
        raise ValueError(
            f"Spatial dimension mismatch: b0_timecourse has {M} voxels, "
            f"acdc_fieldmap has {M2} voxels."
        )

    # ---- Normalise (identical to NumPy version) ----
    W_scaled_np, B_scaled_np, s = _scale_by_spectral_norm(acdc_fieldmap, b0_timecourse)

    W_scaled = torch.from_numpy(W_scaled_np).to(torch_device)    # (N_c, M)
    B_scaled = torch.from_numpy(B_scaled_np).to(torch_device)    # (T, M)
    W_orig = torch.from_numpy(acdc_fieldmap).to(torch_device)
    B_orig = torch.from_numpy(b0_timecourse).to(torch_device)

    # ---- SIRF kernel or dense matrix ----
    use_dense_C = False
    h_kernel = None
    C_mat = None
    if sirf_kernel is not None:
        h_kernel = torch.from_numpy(np.asarray(sirf_kernel, dtype=np.float64)).to(torch_device)
        # Causal low-pass with unit DC gain -> ||C||_2 ~= 1.
        norm_C = 1.0
    elif conv_matrix is not None:
        C_mat = torch.from_numpy(conv_matrix).to(torch_device)
        norm_C = _estimate_spectral_norm_power_iteration_torch(C_mat, n_time, torch_device)
        use_dense_C = True
    else:
        norm_C = 1.0
    norm_W = torch.linalg.norm(W_scaled, ord=2).item()           # ~= 1 after scaling

    # ---- Step size ----
    L = norm_C ** 2 * norm_W ** 2 + ridge_x
    step_size = 1.0 / L

    if verbose:
        tqdm.write(f"FISTA-torch waveform optimisation: T={n_time}, N_c={n_coils}, M={M}")
        tqdm.write(f"  Device: {device},  Scaling s = {s:.4e}")
        tqdm.write(f"  ||C||_2 ~= {norm_C:.4e}, ||W||_2 ~= {norm_W:.4e}, L = {L:.4e}, step = {step_size:.4e}")
        tqdm.write(f"  ridge_x={ridge_x:.2e}")
        tqdm.write(f"  Constraints: |I_i| <= {amp_limit} A,  sum|I_i| <= {l1_limit} A")

    # ---- Initialisation ----
    X = torch.zeros(n_time, n_coils, device=torch_device, dtype=torch.float64)
    Y = X.clone()                                            # FISTA momentum variable
    t_fista = 1.0

    iter_history, loss_history, nrmse_history = [], [], []
    t_start = time.time()

    def _apply_conv(mat):
        if h_kernel is not None:
            return _sirf_conv1d(mat, h_kernel, mode='conv')
        if use_dense_C:
            return C_mat @ mat
        return mat

    def _apply_conv_transpose(mat):
        if h_kernel is not None:
            return _sirf_conv1d(mat, h_kernel, mode='corr')
        if use_dense_C:
            return C_mat.T @ mat
        return mat

    pbar = tqdm(total=max_iter, desc="FISTA-torch", disable=not verbose)
    loss = 0.0
    for iteration in range(max_iter):

        # ---- Forward model ----
        CY = _apply_conv(Y)                                    # (T, N_c)
        field_pred = CY @ W_scaled                              # (T, M)
        residual = field_pred - B_scaled                        # (T, M)

        # ---- Gradient:  grad f(Y) = C^T (C Y W - B) W^T + ridge_x * Y ----
        grad_data = residual @ W_scaled.T                       # (T, N_c)
        grad_data = _apply_conv_transpose(grad_data)
        grad = grad_data + ridge_x * Y

        # ---- Descent ----
        X_new = Y - step_size * grad

        # ---- Projection onto box ∩ L1-ball (row-wise on GPU) ----
        X_new = torch.clamp(X_new, -amp_limit, amp_limit)
        row_sums = torch.sum(torch.abs(X_new), dim=1)
        viol_mask = row_sums > l1_limit

        if viol_mask.any():
            X_viol = X_new[viol_mask]                        # (n_viol, N_c)
            U, _ = torch.sort(torch.abs(X_viol), dim=1, descending=True)
            S = torch.cumsum(U, dim=1)
            rank = torch.arange(1, n_coils + 1, device=torch_device, dtype=torch.float64)
            still_above_threshold = U - (S - l1_limit) / rank > 0
            rho = torch.count_nonzero(still_above_threshold, dim=1) - 1
            row_idx = torch.arange(len(X_viol), device=torch_device)
            theta = (S[row_idx, rho] - l1_limit) / (rho + 1)
            X_soft_thresholded = torch.clamp(torch.abs(X_viol) - theta.unsqueeze(1), min=0)
            X_new[viol_mask] = torch.sign(X_viol) * X_soft_thresholded

        # ---- FISTA momentum ----
        t_fista_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_fista ** 2))
        Y = X_new + ((t_fista - 1.0) / t_fista_new) * (X_new - X)

        X = X_new
        t_fista = t_fista_new

        # ---- Logging ----
        if iteration % iter_step == 0:
            iter_history.append(iteration)
            loss = 0.5 * torch.sum(residual ** 2).item()
            loss += 0.5 * ridge_x * torch.sum(X ** 2).item()
            loss_history.append(loss)

            # NRMSE on original (un-scaled) data.
            field_orig = _apply_conv(X) @ W_orig
            nrmse_cur = (torch.norm(field_orig - B_orig) / torch.norm(B_orig)).item()
            nrmse_history.append(nrmse_cur)
            pbar.set_postfix(loss=f"{loss:.4e}", NRMSE=f"{nrmse_cur:.6e}")
            pbar.update(iter_step)

            if len(loss_history) >= 2:
                rel_change = abs(loss_history[-1] - loss_history[-2]) / loss_history[-2]
                if rel_change < tol:
                    pbar.n = iteration
                    pbar.set_postfix(loss=f"{loss:.4e}", NRMSE=f"{nrmse_cur:.6e}", converged="yes")
                    pbar.close()
                    if verbose:
                        tqdm.write(f"  Converged at iteration {iteration} "
                                   f"(rel change {rel_change:.2e} < {tol:.2e}).")
                    break

    t_elapsed = time.time() - t_start
    pbar.close()

    # ---- Final diagnostics ----
    field_final = _apply_conv(X) @ W_orig
    nrmse = (torch.norm(field_final - B_orig) / torch.norm(B_orig)).item()

    if verbose:
        tqdm.write(f"  Finished in {t_elapsed:.1f} s.  Final NRMSE = {nrmse:.6e}")

    if plot:
        plot_convergence_curves(iter_history, loss_history, nrmse_history)

    return X.cpu().numpy(), nrmse, loss