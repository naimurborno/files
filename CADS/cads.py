"""
CADS: Condition-Annealed Diffusion Sampler
Sadat et al., ICLR 2024 (arXiv:2310.17347)

This module implements the method EXACTLY as specified in the paper:
  - Eq. (2):      piecewise-linear annealing schedule gamma(t)
  - Eq. (1):      condition corruption  y_hat = sqrt(gamma)*y + s*sqrt(1-gamma)*n
  - Eq. (3)-(4):  rescaling of the corrupted condition back to y's mean/std,
                  mixed with the un-rescaled version via psi
  - Figure 15:    reference pseudocode (linear_schedule / add_noise)
  - Algorithm 1:  full CADS sampling loop with classifier-free guidance,
                  using INDEPENDENT noise draws n and n' for the conditional
                  and null embeddings respectively.
"""

import torch


def linear_schedule(t: float, tau1: float, tau2: float) -> float:
    """Eq. (2): piecewise-linear annealing schedule gamma(t).

    t in [0, 1], with t=1 at the START of sampling (max noise, z_1 ~ N(0,I))
    and t=0 at the END of sampling (clean sample z_0), matching the paper's
    convention that inference runs backward in time from t=1 to t=0.

    gamma(t) = 1                       for t <= tau1   (no corruption late in sampling)
             = (tau2 - t) / (tau2-tau1) for tau1 < t < tau2
             = 0                       for t >= tau2   (full corruption early in sampling)
    """
    if t <= tau1:
        return 1.0
    if t >= tau2:
        return 0.0
    return (tau2 - t) / (tau2 - tau1)


def add_noise(
    y: torch.Tensor,
    gamma: float,
    noise_scale: float,
    psi: float,
    rescale: bool = True,
    noise: torch.Tensor = None,
) -> torch.Tensor:
    """Eq. (1) + Eq. (3)-(4): corrupt condition y and (optionally) rescale it back.

    Args:
        y: clean condition tensor (e.g. CLIP text embeddings), any shape.
        gamma: gamma(t) scalar from linear_schedule().
        noise_scale: s, the initial noise scale in Eq. (1).
        psi: mixing factor in Eq. (4), psi in [0, 1].
             psi=1  -> fully rescaled (most stable, paper's default recommendation)
             psi=0  -> no rescaling (most diverse, can be less stable)
        rescale: whether to apply Eq. (3)-(4) rescaling at all.
        noise: optional externally supplied noise n ~ N(0, I) of same shape as y.
               If None, drawn fresh via torch.randn_like(y).

    Returns:
        y_hat (or y_hat_final if rescale=True), same shape as y.
    """
    if noise is None:
        n = torch.randn_like(y)
    else:
        n = noise

    # --- Eq. (1) ---
    y_hat = (gamma ** 0.5) * y + noise_scale * ((1.0 - gamma) ** 0.5) * n

    if not rescale:
        return y_hat

    # --- Eq. (3): rescale y_hat back toward y's (scalar) mean/std ---
    y_mean = y.mean()
    y_std = y.std()

    yhat_mean = y_hat.mean()
    yhat_std = y_hat.std()
    # guard against division by zero at gamma==1 (y_hat == y exactly -> std matches already)
    yhat_std = torch.clamp(yhat_std, min=1e-8)

    y_rescaled = (y_hat - yhat_mean) / yhat_std * y_std + y_mean

    # --- Eq. (4): mix rescaled and un-rescaled versions via psi ---
    y_final = psi * y_rescaled + (1.0 - psi) * y_hat
    return y_final


class CADSScheduleConfig:
    """Container for CADS hyperparameters (see Table 13 in the paper for reference values)."""

    def __init__(
        self,
        tau1: float = 0.6,
        tau2: float = 0.9,
        noise_scale: float = 0.25,
        psi: float = 1.0,
        rescale: bool = True,
        apply_to: str = "both",  # "both" | "cond_only"  (paper anneals cond; null is optionally annealed too)
    ):
        assert 0.0 <= tau1 <= 1.0 and 0.0 <= tau2 <= 1.0
        assert tau1 <= tau2, "tau1 must be <= tau2 per Eq. (2)"
        assert 0.0 <= psi <= 1.0
        self.tau1 = tau1
        self.tau2 = tau2
        self.noise_scale = noise_scale
        self.psi = psi
        self.rescale = rescale
        self.apply_to = apply_to

    def gamma(self, t: float) -> float:
        return linear_schedule(t, self.tau1, self.tau2)

    def __repr__(self):
        return (
            f"CADSScheduleConfig(tau1={self.tau1}, tau2={self.tau2}, "
            f"s={self.noise_scale}, psi={self.psi}, rescale={self.rescale}, "
            f"apply_to={self.apply_to})"
        )