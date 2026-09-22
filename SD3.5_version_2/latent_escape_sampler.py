# Hello my name is naimur asif borno
#
# ---------------------------------------------------------------------------
# This sampler now supports two modes, selected via `mode=`:
#
#   "two_pass"  — the ORIGINAL UGILE mechanism (unchanged, still here).
#                 Cost: 2N denoiser calls (N-step profiling pass + N-step
#                 final pass from the perturbed x0). This is left fully
#                 intact for backward-compatible experiments / re-runs.
#
#   "lite"      — NEW single-pass mechanism ("UGILE-Lite"). Cost: N denoiser
#                 calls, i.e. the SAME as an unmodified sampler / CADS / DAVE.
#                 It reuses the v_cond, v_uncond pair that CFG already
#                 computes at every step (no extra forward/backward calls),
#                 derives a per-step local semantic direction and curvature
#                 proxy from it, accumulates an EMA-persistent orthogonal
#                 escape direction across an early window of steps, and
#                 applies an annealed, norm-retracted nudge at each of those
#                 steps. See `_single_pass_with_escape` for the full logic
#                 and `run_lite` for the entry point.
#
# Default is "lite". Pass mode="two_pass" to reproduce old behavior exactly.
# ---------------------------------------------------------------------------
import math
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch.nn.functional as F


def load_prompts(path: str) -> List[str]:
    """Load ONLY the prompts list from a separate prompts yaml file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    prompts = data.get("prompts", [])
    if not prompts:
        raise ValueError(f"No 'prompts' key found in {path}")
    return prompts

from peakback_core import (
    tweedie_potential,
    joint_projector,
    geodesic_step,
)


class UGILESampler:
    """U-Profile Guided Initial Latent Escape sampler."""

    def __init__(
        self,
        unet,
        scheduler,
        cfg             : dict,
        device          : str   = "cuda",
        num_grad_steps  : int   = 5,
        sigma_lo        : float = 0.3,
        sigma_hi        : float = 0.9,
        escape_scale    : float = 3.0,
        theta_max       : float = 0.75,  # Increased to push Vendi Score up
        walk_steps      : int   = 10,
        J               : int   = 1,
        eps             : float = 1e-8,
        noise_scale     : float = 10.0,   # Lowered base scale for SD3 stability
        gamma           : float = 1.2,
        max_eps_frac    : float = 0.02,   # cap on Phase-2 perturbation, as a fraction of r
        # ---- UGILE-Lite (single-pass) parameters -------------------------
        mode                        : str   = "lite",  # "lite" or "two_pass"
        beta                        : float = 0.85,    # EMA persistence of escape direction
        window_frac                 : float = 0.18,    # fraction of steps used as intervention window (anchored on DAVE's 15-20% finding)
        disable_orthogonal_projection: bool  = False,   # ablation: if True, only remove radial component, NOT the semantic component (tests whether orthogonality-to-semantic-direction is load-bearing)
        lite_noise_scale            : Optional[float] = None,  # if None, reuses `noise_scale`
        lite_max_eps_frac           : Optional[float] = None,  # if None, reuses `max_eps_frac`
    ):
        self.unet           = unet
        self.scheduler      = scheduler
        self.cfg            = cfg
        self.device         = device
        self.num_grad_steps = num_grad_steps
        self.sigma_lo       = sigma_lo
        self.sigma_hi       = sigma_hi
        self.escape_scale   = escape_scale
        self.theta_max      = theta_max
        self.walk_steps     = walk_steps
        self.J              = J
        self.eps            = eps
        self.noise_scale    = noise_scale
        self.gamma          = gamma
        self.max_eps_frac   = max_eps_frac

        # UGILE-Lite state
        self.mode                         = mode
        self.beta                         = beta
        self.window_frac                  = window_frac
        self.disable_orthogonal_projection = disable_orthogonal_projection
        self.lite_noise_scale             = lite_noise_scale if lite_noise_scale is not None else noise_scale
        self.lite_max_eps_frac            = lite_max_eps_frac if lite_max_eps_frac is not None else max_eps_frac

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      50)
        self.guidance_scale = f_cfg.get("guidance_scale", 6.0)
        self.do_cfg         = self.guidance_scale > 1.0

    def run(
        self,
        x0                : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor] = None,
        seed              : int = 0,
        compute_baseline_for_eval: bool = False,
    ) -> Dict[str, Any]:
        """
        Dispatcher. mode="lite" (default) -> single denoiser pass, ~zero
        overhead. mode="two_pass" -> original mechanism, 2x denoiser passes.

        `compute_baseline_for_eval` only matters for mode="lite": it lets
        the caller additionally generate an unperturbed baseline for paired
        evaluation metrics (CLIP retention, paired Vendi, cos_xN, etc.).
        That extra pass is NOT part of UGILE-Lite's own inference cost —
        it exists purely so evaluation scripts can do a fair paired
        comparison the same way they already do for the two-pass method.
        """
        if self.mode == "two_pass":
            return self._run_two_pass(x0, text_embeddings, pooled_embeddings, seed=seed)
        elif self.mode == "lite":
            return self.run_lite(
                x0, text_embeddings, pooled_embeddings, seed=seed,
                compute_baseline_for_eval=compute_baseline_for_eval,
            )
        else:
            raise ValueError(f"Unknown mode '{self.mode}', expected 'lite' or 'two_pass'.")

    # ================================================================== #
    #  LEGACY — ORIGINAL TWO-PASS UGILE MECHANISM (unchanged)             #
    # ================================================================== #

    def _run_two_pass(
        self,
        x0                : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor] = None,
        seed              : int = 0,
    ) -> Dict[str, Any]:

        # Phase 1 — forward profiling pass
        cache = self._forward_pass_with_profiling(x0, text_embeddings, pooled_embeddings)

        # --- Phase 2 — semantic direction & trajectory analysis ---
        N = cache["N"]
        U_vals = torch.tensor(cache["U"], device=x0.device, dtype=torch.float32)
        U_total = U_vals.sum() + self.eps
        w = U_vals / U_total

        semantic_dir = torch.zeros_like(x0, dtype=torch.float32)
        diffs = []
        for k in range(N):
            diff = (cache["v_cond"][k] - cache["v_uncond"][k]).to(x0.device).float()
            diffs.append(diff)
            semantic_dir += w[k] * diff

        semantic_unit = semantic_dir / (semantic_dir.norm() + self.eps)
        s_flat = semantic_unit.flatten()

        # --- High-Level Advanced Noise Injection (Before Geodesic Step) ---

        # A3: U-Profile Curvature (2nd Derivative) as Noise Scale
        if N > 2:
            d2U = U_vals[:-2] - 2 * U_vals[1:-1] + U_vals[2:]
            kappa = d2U.abs().mean().item()
        else:
            kappa = 1.0

        # FIX 1: Clamp epsilon to a fraction of the latent radius to prevent
        # SD3 VAE breakdown (now tunable via max_eps_frac instead of hardcoded).
        r = x0.float().norm().item()
        max_eps = r * self.max_eps_frac
        epsilon_raw = self.noise_scale * (1.0 / (math.sqrt(kappa) + self.eps))
        epsilon = min(epsilon_raw, max_eps)
        # print(f"[UGILE debug] kappa={kappa:.4f}  epsilon_raw={epsilon_raw:.4f}  "
        #       f"max_eps={max_eps:.4f}  epsilon_used={epsilon:.4f}  "
        #       f"{'CAPPED' if epsilon_raw > max_eps else 'uncapped'}")

        # A1: Full Trajectory-Covariance-Anchored Noise (Background Diversity)
        rng_cov = torch.Generator(device=x0.device)
        rng_cov.manual_seed(seed * 10000 + 0)
        v_boundary = torch.randn(x0.shape, generator=rng_cov, dtype=torch.float32, device=x0.device).flatten()

        s_flat_for_diff = semantic_dir.flatten()
        diffs_centered = [(diff.flatten() - s_flat_for_diff) for diff in diffs]

        for _ in range(3):
            v_next = torch.zeros_like(v_boundary)
            for k in range(N):
                dot_product = torch.dot(diffs_centered[k], v_boundary)
                v_next += w[k] * diffs_centered[k] * dot_product

            v_boundary = v_next
            v_boundary = v_boundary - torch.dot(v_boundary, s_flat) * s_flat
            v_boundary = v_boundary / (v_boundary.norm() + self.eps)

        v_boundary = v_boundary.view_as(x0)

        rng = torch.Generator(device=x0.device)
        rng.manual_seed(seed * 10000 + 1)
        jitter = torch.randn(x0.shape, generator=rng, dtype=torch.float32, device=x0.device)

        eta = v_boundary + 0.1 * jitter

        # FIX 2: Remove spatial mean per channel to prevent VAE grid/band artifacts
        if eta.dim() == 4:
            eta = eta - eta.mean(dim=(2, 3), keepdim=True)

        # Project eta jointly onto span{s_hat, x0}^\perp (Eq. 10 — single linear
        # solve, NOT two sequential single-vector projections).
        x0_flat = x0.float().flatten()
        eta_flat = eta.flatten()
        eta_flat = joint_projector(eta_flat, s_flat, x0_flat)
        eta = eta_flat.view_as(x0)

        # Normalize and scale noise
        eta = epsilon * eta / (eta.norm() + self.eps)

        # Add noise to x0 and re-normalize to stay exactly on the sphere (r)
        x0_perturbed = x0.float() + eta
        x0_perturbed = x0_perturbed * (r / (x0_perturbed.norm() + self.eps))

        # --- Phase 3 — Geodesic Escape Step from perturbed x0 ---
        rng2 = torch.Generator(device=x0.device)
        rng2.manual_seed(seed * 10000 + 2)
        xi = torch.randn(x0_perturbed.shape, generator=rng2, dtype=torch.float32, device=x0.device)

        # FIX 3: True 11x11 Gaussian Blur (Replaces AvgPool)
        # AvgPool leaves subtle grid edges that SD3's 16-channel VAE decodes as artifacts.
        # An 11x11 Gaussian convolution with sigma=2.0 perfectly isolates low-frequency 
        # pose/layout structure with zero grid artifacts.
        if x0.dim() == 4:
            B, C, H, W = xi.shape
            sigma = 2.5
            coords = torch.arange(11, device=x0.device).float() - 5
            gauss_1d = torch.exp(-(coords**2) / (2 * sigma**2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            kernel = torch.outer(gauss_1d, gauss_1d).view(1, 1, 11, 11).repeat(C, 1, 1, 1)
            
            xi_low = F.conv2d(xi, kernel, padding=5, groups=C)
            
            # 60% low-freq (pose) + 40% high-freq (natural texture diversity)
            xi = 0.80 * xi_low + 0.20 * xi

        # Joint projection onto span{s_hat, x0_perturbed}^\perp (Eq. 14 — same
        # single linear solve used by the other architecture ports).
        xi_flat = xi.flatten()
        x0p_flat = x0_perturbed.flatten()
        xi_flat = joint_projector(xi_flat, s_flat, x0p_flat)

        e_hat = xi_flat.view_as(x0_perturbed)

        # Eq. 16-17: theta is the escape budget derived from the projected
        # vector's own norm, capped by theta_max — not always theta_max.
        # geodesic_step() also performs the exact-norm great-circle move
        # (Eq. 17-18), replacing the manual cos/sin computation below.
        # theta = min(escape_scale * ||w||/r, theta_max). escape_scale lets you
        # push the natural escape ratio up when it's small (as observed —
        # theta_max wasn't the binding constraint), while theta_max remains a
        # hard ceiling so theta can never exceed a safe bound.
        x0_new, theta = geodesic_step(
            x0_perturbed, self.escape_scale * e_hat, r, theta_max=self.theta_max
        )
        theta = theta.item()
        # print(f"[UGILE debug] ||w||/r = {e_hat.norm().item()/r:.4f}  "
        #       f"escape_scale = {self.escape_scale}  theta_max = {self.theta_max}  "
        #       f"theta_used = {theta:.4f}")
        x0_new = x0_new.to(x0.dtype)

        cos_x0 = torch.nn.functional.cosine_similarity(
            x0_new.reshape(1, -1).float(), x0.float().reshape(1, -1)
        ).item()

        # Phase 4 — full forward pass from x_0_new
        x_N_diverse = self._full_forward_pass(x0_new, text_embeddings, pooled_embeddings)

        cos_xN = torch.nn.functional.cosine_similarity(
            x_N_diverse.reshape(1, -1).float(),
            cache["x_N"].reshape(1, -1).float()
        ).item()

        return {
            "original_latents" : cache["x_N"],
            "branches"         : [{
                "branch_idx" : 0,
                "theta"      : theta,
                "cos_x0"     : cos_x0,
                "cos_xN"     : cos_xN,
                "latents"    : x_N_diverse,
            }],
        }

    # ================================================================== #
    #  UGILE-LITE — SINGLE-PASS, CFG-NATIVE TRAJECTORY ESCAPE              #
    # ================================================================== #
    #
    # Cost: exactly N denoiser calls (same as baseline / CADS / DAVE).
    # No separate profiling pass, no second full pass, no extra
    # forward/backward calls beyond what standard CFG already computes.
    #
    # Mechanism (see method docstring in run_lite for the full writeup):
    #   at each step in an early window W:
    #     s_t     = normalize(v_cond - v_uncond)          [free — CFG byproduct]
    #     U_t     = tweedie_potential(v_cond, v_uncond, sigma_t)  [free]
    #     g_t     = random unit tangent, projected orthogonal to
    #               {current latent (radial), s_t (semantic)}
    #     e_t     = normalize(beta * e_{t-1} + (1-beta) * g_t), reprojected
    #     eps_t   = min(noise_scale / (sqrt(U_t)+eps), max_eps_frac * r_t)
    #               * gamma(t)   [annealed, decaying across the window]
    #     z_t     = retract(z_t + eps_t * e_t)             [exact norm restore]
    #   outside W: identical to the unmodified scheduler step.
    # ================================================================== #

    def _project_radial_only(self, v_flat, z_flat):
        """
        Ablation helper: remove only the radial component (v - (v.zhat)zhat),
        NOT the semantic component. Used when
        disable_orthogonal_projection=True to test whether projecting away
        from the semantic direction is actually load-bearing, or whether a
        plain radially-orthogonal random walk (closer in spirit to CADS)
        gives the same result.
        """
        z_hat = z_flat / (z_flat.norm() + self.eps)
        v_flat = v_flat - torch.dot(v_flat, z_hat) * z_hat
        return v_flat

    def _lite_window_gamma(self, k: int, window_len: int) -> float:
        """
        Linear anneal across the intervention window: gamma=1 at the
        noisiest step of the window (k=0), decaying to 0 at the window's
        far edge (k=window_len). Mirrors CADS's own piecewise-linear
        annealing shape, just applied to our escape magnitude instead of
        the conditioning vector.
        """
        if window_len <= 0:
            return 0.0
        frac = k / float(window_len)
        return max(0.0, 1.0 - frac)

    def run_lite(
        self,
        x0                : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor] = None,
        seed              : int = 0,
        compute_baseline_for_eval: bool = False,
    ) -> Dict[str, Any]:
        """
        UGILE-Lite: one sampling pass, escape direction derived live from
        each step's own CFG computation (v_cond, v_uncond), accumulated
        with an EMA across an early window, applied with an annealed and
        norm-retracted nudge. No separate profiling pass, no second full
        forward pass.

        Returns a dict with the same keys the legacy two-pass `run()`
        returns ("original_latents", "branches"), so downstream code
        (run_sd3_ugile, evaluation scripts) does not need to change.
        `original_latents` is only populated if compute_baseline_for_eval
        is True (that pass is evaluation-only, not part of the method's
        own inference cost — see the docstring on `run`).
        """
        self.scheduler.set_timesteps(self.num_steps)
        timesteps = self.scheduler.timesteps
        N = len(timesteps)
        window_len = max(1, int(round(self.window_frac * N)))

        device = x0.device
        z = x0.clone()

        e_prev = None
        diag = {
            "window_len": window_len,
            "N": N,
            "kappa_sum": 0.0,
            "kappa_count": 0,
            "cap_hit_count": 0,
            "total_rel_displacement_sq": 0.0,  # accumulated ||delta||^2, summed across window
            "trace": [],                       # per-step diagnostics (see _lite_apply_escape_step)
        }

        for k, t in enumerate(timesteps):
            v_cfg, v_uncond, v_cond = self._velocity_forward(
                z, t, text_embeddings, pooled_embeddings, return_split=True
            )

            if k < window_len:
                z = self._lite_apply_escape_step(
                    z, k, window_len, t, v_cond, v_uncond, seed, diag
                )

            z = self.scheduler.step(v_cfg, t, z).prev_sample

        x_N_diverse = z

        result = {
            "original_latents": None,
            "branches": [{
                "branch_idx": 0,
                "theta": diag.get("total_rel_displacement", 0.0),
                "cos_x0": None,
                "cos_xN": None,
                "latents": x_N_diverse,
                # extra UGILE-Lite diagnostics (safe to ignore downstream)
                "window_len": window_len,
                "mean_kappa": (diag["kappa_sum"] / diag["kappa_count"]) if diag["kappa_count"] else 0.0,
                "cap_hit_rate": (diag["cap_hit_count"] / diag["kappa_count"]) if diag["kappa_count"] else 0.0,
                "trace": diag["trace"],
                "sigma_window": ([diag["trace"][0]["sigma"], diag["trace"][-1]["sigma"]] if diag["trace"] else None),
                "cos_e_first_last": (diag["trace"][-1]["cos_e_first"] if diag["trace"] else None),
            }],
        }

        if compute_baseline_for_eval:
            # Evaluation-only reference pass. NOT counted toward UGILE-Lite's
            # own inference cost (a real user generating one diverse image
            # never needs this). Kept separate and explicit so nobody
            # accidentally folds it into a cost comparison.
            baseline = self._full_forward_pass(x0, text_embeddings, pooled_embeddings)
            result["original_latents"] = baseline
            cos_xN = torch.nn.functional.cosine_similarity(
                x_N_diverse.reshape(1, -1).float(),
                baseline.reshape(1, -1).float(),
            ).item()
            result["branches"][0]["cos_xN"] = cos_xN

        return result

    def _lite_apply_escape_step(self, z, k, window_len, t, v_cond, v_uncond, seed, diag):
        """
        One in-window intervention step. Mutates nothing in place; returns
        the perturbed latent to feed into the scheduler's update.
        """
        z_f = z.float()
        r_t = z_f.norm() + self.eps

        # --- free, CFG-native local signals -------------------------------
        delta = (v_cond - v_uncond).float()
        s_t = delta / (delta.norm() + self.eps)

        sigma_t = self.scheduler.sigmas[k].item() if hasattr(self.scheduler, "sigmas") else 1.0 - k / diag["N"]
        sigma_t = max(sigma_t, 1e-4)
        U_t = tweedie_potential(v_cond, v_uncond, sigma_t).item()
        diag["kappa_sum"] += U_t
        diag["kappa_count"] += 1

        # --- candidate random tangent, orthogonalized ---------------------
        rng = torch.Generator(device=z.device)
        rng.manual_seed(seed * 1_000_003 + k)  # distinct stream from legacy seed*10000+{0,1,2}
        g = torch.randn(z.shape, generator=rng, dtype=torch.float32, device=z.device)

        g_flat = g.flatten()
        s_flat = s_t.flatten()
        z_flat = z_f.flatten()

        if self.disable_orthogonal_projection:
            # Ablation path: remove only the radial component. Tests
            # whether the semantic-orthogonality constraint is actually
            # doing anything (see the ablation check discussed in the
            # method writeup before trusting the full mechanism).
            g_flat = self._project_radial_only(g_flat, z_flat)
        else:
            g_flat = joint_projector(g_flat, s_flat, z_flat)
        g_flat = g_flat / (g_flat.norm() + self.eps)

        # --- EMA persistence across the window -----------------------------
        e_prev = diag.get("_e_prev", None)
        if e_prev is None:
            e_t = g_flat
        else:
            e_t = self.beta * e_prev + (1.0 - self.beta) * g_flat
            if self.disable_orthogonal_projection:
                e_t = self._project_radial_only(e_t, z_flat)
            else:
                e_t = joint_projector(e_t, s_flat, z_flat)
        e_t = e_t / (e_t.norm() + self.eps)
        cos_e_prev  = torch.dot(e_t, diag["_e_prev"]).item() if "_e_prev" in diag else None
        if "_e_first" not in diag:
            diag["_e_first"] = e_t.detach().clone()
        cos_e_first = torch.dot(e_t, diag["_e_first"]).item()
        diag["_e_prev"] = e_t.detach()

        # --- annealed, curvature-aware, norm-capped magnitude --------------
        gamma_t = self._lite_window_gamma(k, window_len)
        eps_uncapped = self.lite_noise_scale / (math.sqrt(U_t) + self.eps)
        eps_cap = r_t.item() * self.lite_max_eps_frac
        eps_t = min(eps_uncapped, eps_cap) * gamma_t
        if eps_uncapped >= eps_cap - 1e-12:
            diag["cap_hit_count"] += 1

        # --- apply + exact retraction ---------------------------------------
        z_pert_flat = z_flat + eps_t * e_t
        z_pert_flat = z_pert_flat * (r_t / (z_pert_flat.norm() + self.eps))

        diag["total_rel_displacement_sq"] = diag.get("total_rel_displacement_sq", 0.0) + (eps_t / r_t.item()) ** 2
        diag["total_rel_displacement"] = math.sqrt(diag["total_rel_displacement_sq"])

        # ---- per-step trace (for pilot diagnostics; negligible cost) -------
        diag["trace"].append({
            "k"            : k,
            "sigma"        : sigma_t,
            "U_t"          : U_t,
            "delta_norm"   : delta.norm().item(),
            "r_t"          : r_t.item(),
            "eps_uncapped" : eps_uncapped,
            "eps_cap"      : eps_cap,
            "eps_t"        : eps_t,
            "gamma_t"      : gamma_t,
            "rel_disp"     : ((z_pert_flat - z_flat).norm() / r_t).item(),  # actual ||dz||/r after retraction
            "cos_e_prev"   : cos_e_prev,
            "cos_e_first"  : cos_e_first,
            "cos_e_s"      : torch.dot(e_t, s_flat).item(),                 # ~0 when orthogonality is enforced
            "cap_hit"      : bool(eps_uncapped >= eps_cap - 1e-12),
        })

        return z_pert_flat.view_as(z).to(z.dtype)

    # ================================================================== #
    #  PHASE 1 — FORWARD PROFILING PASS  (legacy, two_pass mode only)     #
    # ================================================================== #

    def _forward_pass_with_profiling(self, x0, text_embeddings, pooled_embeddings):
        self.scheduler.set_timesteps(self.num_steps)
        timesteps = self.scheduler.timesteps
        N = len(timesteps)

        cached_x        = [None] * (N + 1)
        cached_v        = [None] * N
        cached_v_uncond = [None] * N
        cached_v_cond   = [None] * N
        cached_sigma    = [None] * N
        cached_t        = [None] * N
        cached_U        = [None] * N

        x = x0.clone()
        cached_x[0] = x.float().clone()

        for k, t in enumerate(timesteps):
            v, v_uncond, v_cond = self._velocity_forward(
                x, t, text_embeddings, pooled_embeddings, return_split=True
            )
            cached_v[k]        = v.detach().float().clone()
            cached_v_uncond[k] = v_uncond.detach().float().clone()
            cached_v_cond[k]   = v_cond.detach().float().clone()
            cached_t[k]        = t

            if hasattr(self.scheduler, "sigmas"):
                sigma_t    = self.scheduler.sigmas[k].item()
                sigma_next = self.scheduler.sigmas[k + 1].item()
            else:
                sigma_t, sigma_next = 1.0 - k / N, 1.0 - (k + 1) / N
            cached_sigma[k] = max(sigma_t, 1e-4)

            cached_U[k] = tweedie_potential(v_cond, v_uncond, cached_sigma[k]).item()

            x = self.scheduler.step(v, t, x).prev_sample
            cached_x[k + 1] = x.detach().float().clone()

        return {
            "x": cached_x, "v": cached_v,
            "v_uncond": cached_v_uncond, "v_cond": cached_v_cond,
            "sigma": cached_sigma, "t": cached_t,
            "U": cached_U, "x_N": x, "N": N,
        }

    # ================================================================== #
    #  PHASE 2 — U-WEIGHTED ESCAPE DIRECTION                              #
    # ================================================================== #

    def _compute_escape_direction(self, x0, cache, text_embeddings, pooled_embeddings):
        N = cache["N"]
        band = [k for k in range(N)
                if self.sigma_lo <= cache["sigma"][k] <= self.sigma_hi]
        if not band:
            band = list(range(N))
        selected = band

        U_selected = [cache["U"][k] for k in selected]
        U_sum = sum(U_selected) + self.eps
        weights = [u / U_sum for u in U_selected]

        escape_dir = torch.zeros_like(x0, dtype=torch.float32)

        for k, w_k in zip(selected, weights):
            x_k = cache["x"][k].to(dtype=torch.float32, device=self.device)
            t_k = cache["t"][k]
            grad = self._grad_U_at_xk(x_k, t_k, cache["sigma"][k],
                                       text_embeddings, pooled_embeddings)
            escape_dir = escape_dir + w_k * grad

        return escape_dir

    def _grad_U_at_xk(self, x_k, t, sigma, text_embeddings, pooled_embeddings,
                      eps_smooth=1e-6):
        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype

        x_req = x_k.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
        latent_input = torch.cat([x_req, x_req]) if self.do_cfg else x_req

        t_val   = t.item() if hasattr(t, "item") else float(t)
        t_batch = torch.tensor([t_val] * latent_input.shape[0],
                               device=device, dtype=dtype)

        kwargs = dict(
            hidden_states         = latent_input,
            timestep              = t_batch,
            encoder_hidden_states = text_embeddings.to(device=device, dtype=dtype),
        )
        if pooled_embeddings is not None:
            kwargs["pooled_projections"] = pooled_embeddings.to(device=device, dtype=dtype)

        with torch.enable_grad():
            output = self.unet(**kwargs).sample
            if self.do_cfg:
                v_uncond, v_cond = output.chunk(2)
            else:
                v_cond = v_uncond = output
            d = (v_cond - v_uncond).float()
            U = torch.sqrt((d * d).sum() + eps_smooth ** 2) * sigma
            grad = torch.autograd.grad(U, x_req)[0]

        return grad.detach().float()

    # ================================================================== #
    #  SHORT FORWARD PASS — on-manifold validity check                    #
    # ================================================================== #

    def _short_forward_pass(self, x0_new, text_embeddings, pooled_embeddings, n_steps=10):
        self.scheduler.set_timesteps(self.num_steps)
        timesteps = self.scheduler.timesteps[:n_steps]
        x = x0_new.clone()
        for t in timesteps:
            v = self._velocity_forward(x, t, text_embeddings, pooled_embeddings)
            x = self.scheduler.step(v, t, x).prev_sample
        return x

    # ================================================================== #
    #  PHASE 4 — FULL FORWARD PASS FROM MODIFIED x0                      #
    # ================================================================== #

    def _full_forward_pass(self, x0_new, text_embeddings, pooled_embeddings):
        self.scheduler.set_timesteps(self.num_steps)
        timesteps = self.scheduler.timesteps
        N = len(timesteps)
        x = x0_new.clone()

        for k, t in enumerate(timesteps):
            v = self._velocity_forward(x, t, text_embeddings, pooled_embeddings)
            x = self.scheduler.step(v, t, x).prev_sample

        return x

    # ================================================================== #
    #  VELOCITY FORWARD                                                    #
    # ================================================================== #

    def _velocity_forward(self, x, t, text_embeddings, pooled_embeddings,
                          return_split=False):
        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype

        latent_input = (torch.cat([x, x]) if self.do_cfg else x).to(device=device, dtype=dtype)
        text_embeddings = text_embeddings.to(device=device, dtype=dtype)

        t_val   = t.item() if hasattr(t, "item") else float(t)
        t_batch = torch.tensor([t_val] * latent_input.shape[0], device=device, dtype=dtype)

        kwargs = dict(
            hidden_states         = latent_input,
            timestep              = t_batch,
            encoder_hidden_states = text_embeddings,
        )
        if pooled_embeddings is not None:
            kwargs["pooled_projections"] = pooled_embeddings.to(device=device, dtype=dtype)

        with torch.no_grad():
            output = self.unet(**kwargs).sample

        if self.do_cfg:
            v_uncond, v_cond = output.chunk(2)
            v_cfg = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            if return_split:
                return v_cfg, v_uncond, v_cond
            return v_cfg

        if return_split:
            return output, output, output
        return output


# ══════════════════════════════════════════════════════════════════════ #
#  DROP-IN RUNNER                                                         #
# ══════════════════════════════════════════════════════════════════════ #

def run_sd3_ugile(opts: dict):
    from pipeline_wrapper import SD3PipelineWrapper

    cfg     = opts.get("_cfg", {})
    device  = opts["device"]
    seeds   = opts.get("seeds") or [opts["seed"]]

    # Prompts now come from a separate prompts-only yaml file
    # (config.yaml sets `prompts_file: path/to/prompts.yaml`)
    prompts_file = cfg.get("prompts_file", "prompts.yaml")
    prompts = load_prompts(prompts_file)

    ug_cfg = cfg.get("ugile", {})

    wrapper = SD3PipelineWrapper(cfg, device=device)
    wrapper.load()

    sampler = UGILESampler(
        unet            = wrapper.transformer,
        scheduler       = wrapper.scheduler,
        cfg             = cfg,
        device          = device,
        num_grad_steps  = ug_cfg.get("num_grad_steps",  5),
        sigma_lo        = ug_cfg.get("sigma_lo",        0.3),
        sigma_hi        = ug_cfg.get("sigma_hi",        0.9),
        escape_scale    = ug_cfg.get("escape_scale",    3.0),
        theta_max       = ug_cfg.get("theta_max",       0.75),
        walk_steps      = ug_cfg.get("walk_steps",      10),
        J               = ug_cfg.get("J",               1),
        noise_scale     = ug_cfg.get("noise_scale",     8.0),
        gamma           = ug_cfg.get("gamma",           1.2),
        max_eps_frac    = ug_cfg.get("max_eps_frac",    0.02),
        # UGILE-Lite config (all optional; defaults match __init__)
        mode                          = ug_cfg.get("mode",                          "lite"),
        beta                          = ug_cfg.get("beta",                          0.85),
        window_frac                   = ug_cfg.get("window_frac",                   0.18),
        disable_orthogonal_projection = ug_cfg.get("disable_orthogonal_projection", False),
        lite_noise_scale              = ug_cfg.get("lite_noise_scale",              None),
        lite_max_eps_frac             = ug_cfg.get("lite_max_eps_frac",             None),
    )

    # Whether to also generate an unperturbed baseline per prompt/seed.
    # For mode="lite" this is an EVALUATION-ONLY extra pass (see run_lite's
    # docstring) — it is what `save_original` below actually triggers.
    compute_baseline_for_eval = ug_cfg.get("save_original", True) and sampler.mode == "lite"

    base_out        = Path(opts["output"])
    multi_seed      = len(seeds) > 1
    diverse_folder  = Path(ug_cfg.get("diverse_output_dir",  "outputs/diverse"))
    original_folder = Path(ug_cfg.get("original_output_dir", "outputs/original"))
    diverse_folder.mkdir(parents=True, exist_ok=True)
    original_folder.mkdir(parents=True, exist_ok=True)
    save_original   = ug_cfg.get("save_original", True)

    def _base_path(prompt_idx, seed):
        # Filename = prompt position + seed, e.g. prompt 0, seed 41 -> "1_seed41.png"
        return original_folder / (f"{prompt_idx + 1}_seed{seed}" + base_out.suffix)

    def _branch_path(prompt_idx, seed, j):
        return diverse_folder / (f"{prompt_idx + 1}_seed{seed}_branch{j}" + base_out.suffix)

    records = []
    total = len(prompts) * len(seeds)
    done  = 0

    # Read the offset injected by the parallel runner
    prompt_offset = cfg.get("prompt_offset", 0)

    for p_idx, prompt in enumerate(prompts):
        # Calculate the absolute global index for filename safety
        global_idx = p_idx + prompt_offset

        prompt_embeds, pooled_embeds = wrapper.encode_prompt(
            prompt, opts["negative_prompt"]
        )

        for seed in seeds:
            done += 1
            print(f"[UGILE] image {done}/{total}  seed={seed}")

            latents = wrapper.get_initial_latents(seed=seed)
            result  = sampler.run(
                latents, prompt_embeds, pooled_embeds, seed=seed,
                compute_baseline_for_eval=compute_baseline_for_eval,
            )

            if save_original and result["original_latents"] is not None:

                base_path = _base_path(global_idx, seed)
                wrapper.decode_latents(result["original_latents"]).save(base_path)

            for br in result["branches"]:
                
                out_path = _branch_path(global_idx, seed, br["branch_idx"])
                wrapper.decode_latents(br["latents"]).save(out_path)

                records.append({
                    "prompt_idx": global_idx,
                    "prompt"    : prompt,
                    "seed"      : seed,
                    "branch"    : br["branch_idx"],
                    "theta"     : br["theta"],          # lite: total relative displacement (sqrt sum eps_t^2)/r ; two_pass: geodesic angle
                    "cos_x0"    : br["cos_x0"],
                    "cos_xN"    : br["cos_xN"],
                    "out_path"  : str(out_path),
                    # UGILE-Lite diagnostics (absent / None in two_pass mode)
                    "window_len"      : br.get("window_len"),
                    "sigma_window"    : br.get("sigma_window"),
                    "mean_kappa"      : br.get("mean_kappa"),
                    "cap_hit_rate"    : br.get("cap_hit_rate"),
                    "cos_e_first_last": br.get("cos_e_first_last"),
                    "trace"           : br.get("trace"),
                })

    # Persist run metadata + diagnostics next to the images so every sweep
    # arm is self-describing (inference.py itself discards the return value).
    meta = {
        "sampler_params": {
            "mode": sampler.mode, "beta": sampler.beta, "window_frac": sampler.window_frac,
            "disable_orthogonal_projection": sampler.disable_orthogonal_projection,
            "lite_noise_scale": sampler.lite_noise_scale, "lite_max_eps_frac": sampler.lite_max_eps_frac,
            "num_steps": sampler.num_steps, "guidance_scale": sampler.guidance_scale,
        },
        "seeds": list(seeds), "n_prompts": len(prompts), "prompt_offset": prompt_offset,
        "records": records,
    }
    with open(diverse_folder / f"records_offset{prompt_offset}.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)

    return records