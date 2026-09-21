"""
peakback_core.py
----------------
Pure-tensor math for PeakBack, kept separate from model/scheduler code so it
can be unit-tested without loading SD3 (see test_peakback_core.py).

Implements, with direct references to the derivations in the method section:
  - tweedie_potential        : Eq. 5  (U_k = sigma_k * ||v_c - v_uncond||)
  - joint_projector          : Eq. 10 (orthogonal to BOTH v and current x)
  - geodesic_step             : Eq. 11 (exact-norm move, no renormalization)
  - quantile_threshold        : Eq. 7  (u*)
  - leverage_surrogate        : Eq. 9  (zero-cost finite-difference leverage score)
"""

import math
import torch


def tweedie_potential(v_cond: torch.Tensor, v_uncond: torch.Tensor, sigma: float) -> torch.Tensor:
    """Eq. 5: U_k = sigma_k * ||v_cond - v_uncond||_2 (free — no extra forward pass)."""
    return sigma * (v_cond.float() - v_uncond.float()).norm()


def joint_projector(F: torch.Tensor, v: torch.Tensor, x: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """
    Eq. 10. Project flattened vector F onto the subspace orthogonal to BOTH
    v and x (the (D-2)-dim intersection), in one linear solve — NOT two
    sequential single-vector projections (which would not jointly satisfy
    both constraints; see Proposition 1 / Section 3.5).

    F, v, x: 1-D tensors of the same length D.
    ridge: small Tikhonov term added to the 2x2 Gram matrix for numerical
           safety when v and x are nearly parallel (breaks exact
           orthogonality by O(ridge), not exact zero — documented trade-off).
    """
    F = F.reshape(-1).float()
    v = v.reshape(-1).float()
    x = x.reshape(-1).float()

    A = torch.stack([v, x], dim=0)                      # [2, D]
    G = A @ A.t()                                        # [2, 2] Gram matrix
    G = G + ridge * torch.eye(2, dtype=G.dtype, device=G.device)
    rhs = (A @ F.unsqueeze(-1))                           # [2, 1]
    y = torch.linalg.solve(G, rhs)                         # G^{-1} A F
    w = F - (A.t() @ y).squeeze(-1)                        # Eq. 10
    return w


def geodesic_step(x: torch.Tensor, w: torch.Tensor, r: float, theta_max: float = None, eps: float = 1e-12):
    """
    Eq. 11. Exact great-circle move on the sphere of radius r, in the plane
    spanned by x and w (w must already be orthogonal to x, e.g. the output
    of joint_projector). Returns (x_new, theta_used).

    No renormalization step exists or is needed: ||x_new|| == r exactly,
    for any theta (Proposition 2).
    """
    x = x.float()
    w = w.float()
    w_norm = w.norm() + eps
    theta = w_norm / r
    if theta_max is not None:
        theta = torch.clamp(theta, max=theta_max)
    w_hat = w / w_norm
    x_new = x * torch.cos(theta) + r * w_hat * torch.sin(theta)
    return x_new, theta


def coupling_error_bound(theta: float, r: float, v_norm: float) -> float:
    """Eq. 12 upper bound: |delta_sigma'| <= (theta^2 / 2) * (r / ||v||)."""
    return 0.5 * (theta ** 2) * (r / v_norm)


def quantile_threshold(values, q: float) -> float:
    """Eq. 7: Quantile_{1-q}({U_k}) via simple sorted-index lookup (no numpy dependency)."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return float("inf")
    idx = round((1.0 - q) * (len(vals) - 1))
    idx = max(0, min(idx, len(vals) - 1))
    return vals[idx]


def leverage_surrogate(U_seq, sigma_seq, eps: float = 1e-8):
    """
    Eq. 9: free finite-difference leverage score, using only the cached
    {U_k, sigma_k} from the forward profiling pass — zero extra network calls.
    Returns a dict {index: score} over interior indices.
    """
    n = len(U_seq)
    scores = {}
    for k in range(1, n - 1):
        dU = abs(U_seq[k + 1] - U_seq[k - 1])
        dsig = abs(sigma_seq[k + 1] - sigma_seq[k - 1]) + eps
        scores[k] = U_seq[k] * (dU / dsig)
    return scores


def select_top_peaks(scores: dict, top_j: int, min_sep: int):
    """Greedy top-J selection by score, enforcing a minimum index separation."""
    ordered = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    selected = []
    for k in ordered:
        if all(abs(k - s) >= min_sep for s in selected):
            selected.append(k)
        if len(selected) >= top_j:
            break
    return sorted(selected)
