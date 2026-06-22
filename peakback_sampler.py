"""
peakback_sampler.py
--------------------
PeakBack: Peak-Triggered Null-Space Backtracking for SD3 Flow Matching.

Drop-in sibling of onlb_sampler.py — same constructor/run() shape so it slots
into the same MODEL_REGISTRY pattern in inference.py. Differs from ONLB in
that it does NOT walk the whole trajectory backward; it profiles the forward
pass once (free), jumps to a small number of well-chosen mid-trajectory
points, and only locally searches at those points.

Algorithm (method-section numbering):
  Phase 1 — Forward profiling pass : cache (x_k, v_k, v_cond_k, v_uncond_k,
                                             sigma_k, t_k, U_k);  N model calls.
  Phase 2 — Leverage-score peak selection : zero-cost finite-difference
                                             surrogate (Eq. 9) over cached U_k.
  Phase 3 — Constrained null-space Langevin walk : <=K_max (forward+backward)
                                             pairs PER peak, using the joint
                                             projector + exact geodesic step
                                             from peakback_core.py.
  Phase 4 — Resume decode : manual Euler continuation from each accepted
                                             point to step N (reuses the last
                                             velocity evaluated in Phase 3).

Honest scope notes (see method section, Section 3.10):
  - The walk's reference velocity v_{k*}^cfg is held FIXED from the Phase-1
    cache throughout the local search (Remark 4) — re-evaluating it at every
    sub-step would cost an extra forward pass per step. theta_max bounds how
    far this approximation is allowed to drift (Corollary 1).
  - Only the "euler" solver is supported for Phase 1/Phase 4 stepping (same
    scope restriction as onlb_sampler.py).
  - Acceptance (Eq. 14) is a guardrail against the "bland sample" failure
    mode, not a proof that it cannot happen. Treat acceptance diagnostics
    (printed per branch) as something to correlate with CLIP/Vendi scores
    empirically, not as a guarantee.
"""

import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

from peakback_core import (
    tweedie_potential,
    joint_projector,
    geodesic_step,
    quantile_threshold,
    leverage_surrogate,
    select_top_peaks,
)


class PeakBackSampler:

    def __init__(
        self,
        unet,
        scheduler,
        cfg            : dict,
        device         : str   = "cuda",
        sigma_lo       : float = 0.3,
        sigma_hi       : float = 0.7,
        quantile_alpha : float = 0.2,   # Eq. 7 — escape quantile
        peak_eta       : float = 0.05,  # Langevin step size during the walk
        K_max          : int   = 5,
        delta          : float = 0.2,   # Eq. 14 fidelity tolerance
        theta_max      : float = 0.3,   # caps Corollary 1 drift per step
        J              : int   = 2,     # number of diverse branches per seed
        min_sep        : int   = 3,     # minimum index separation between peaks
        exact_leverage : bool  = False, # Eq. 8 (costly) vs Eq. 9 (free) selection
        eps            : float = 1e-8,
    ):
        self.unet           = unet
        self.scheduler      = scheduler
        self.cfg            = cfg
        self.device         = device
        self.sigma_lo       = sigma_lo
        self.sigma_hi       = sigma_hi
        self.quantile_alpha = quantile_alpha
        self.peak_eta       = peak_eta
        self.K_max          = K_max
        self.delta          = delta
        self.theta_max      = theta_max
        self.J              = J
        self.min_sep        = min_sep
        self.exact_leverage = exact_leverage
        self.eps            = eps

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      50)
        self.guidance_scale = f_cfg.get("guidance_scale", 7.5)
        self.do_cfg         = self.guidance_scale > 1.0

        if f_cfg.get("solver", "euler") != "euler":
            print("[PeakBack] WARNING: only the 'euler' solver is supported "
                  "for trajectory caching/resume; 'heun' config is ignored here.")

        print(f"[PeakBack] band=[{sigma_lo},{sigma_hi}]  alpha={quantile_alpha}  "
              f"eta={peak_eta}  K_max={K_max}  delta={delta}  theta_max={theta_max}  "
              f"J={J}  min_sep={min_sep}  exact_leverage={exact_leverage}")

    # ================================================================== #
    #  PUBLIC ENTRY                                                       #
    # ================================================================== #

    def run(
        self,
        latents           : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor] = None,
        seed              : int = 0,
    ) -> Dict[str, Any]:

        # Phase 1 — forward profiling pass (free base image included)
        print("\n[PeakBack] ── Phase 1: Forward profiling pass ──")
        cache = self._forward_pass_with_profiling(latents, text_embeddings, pooled_embeddings)
        x_N = cache["x_N"]
        print(f"[PeakBack]   base x_N  mean={x_N.mean():.4f}  std={x_N.std():.4f}  "
              f"norm={x_N.norm():.4f}")
        print(f"[PeakBack]   U_k range over trajectory: "
              f"min={min(cache['U']):.4f}  max={max(cache['U']):.4f}")

        # Phase 2 — leverage-score peak selection (free by default)
        print("\n[PeakBack] ── Phase 2: Leverage-score peak selection ──")
        peak_indices = self._select_peaks(cache, text_embeddings, pooled_embeddings)
        print(f"[PeakBack]   selected peak step-indices: {peak_indices}")
        if not peak_indices:
            print("[PeakBack]   WARNING: no peaks found in band — "
                  "no diverse branches will be produced.")

        u_star = quantile_threshold(
            [cache["U"][k] for k in range(len(cache["U"]))
             if self.sigma_lo <= cache["sigma"][k] <= self.sigma_hi] or cache["U"],
            self.quantile_alpha,
        )
        print(f"[PeakBack]   escape threshold u* = {u_star:.4f}")

        # Phases 3+4 — per-peak constrained walk, then resume decode
        branches = []
        for j, k_star in enumerate(peak_indices[: self.J]):
            print(f"\n[PeakBack] ── Branch {j} (peak step k*={k_star}, "
                  f"sigma={cache['sigma'][k_star]:.4f}) ──")

            x_tilde, accepted, U_history, v_cfg_final = self._constrained_walk(
                x0               = cache["x"][k_star],
                v_anchor         = cache["v"][k_star],
                sigma_star       = cache["sigma"][k_star],
                t_star           = cache["t"][k_star],
                orig_cond_norm   = cache["v_cond"][k_star].float().norm(),
                text_embeddings  = text_embeddings,
                pooled_embeddings= pooled_embeddings,
                u_star           = u_star,
                seed             = seed,
                branch_idx       = j,
            )
            print(f"[PeakBack]   walk finished | accepted={accepted} | "
                  f"steps_used={len(U_history)} | U trace={['%.3f'%u for u in U_history]}")

            x_N_diverse = self._resume_decode(
                x_tilde, k_star, v_cfg_final, cache, text_embeddings, pooled_embeddings
            )

            cos_sim = torch.nn.functional.cosine_similarity(
                x_N_diverse.reshape(1, -1).float(), x_N.reshape(1, -1).float()
            ).item()
            print(f"[PeakBack]   branch {j} | cos(x_N_diverse, x_N) = {cos_sim:.4f} | "
                  f"accepted={accepted}")

            branches.append({
                "branch_idx"   : j,
                "k_star"       : k_star,
                "accepted"     : accepted,
                "steps_used"   : len(U_history),
                "U_history"    : U_history,
                "latents"      : x_N_diverse,
                "cos_to_base"  : cos_sim,
            })

        return {
            "original_latents" : x_N,
            "branches"         : branches,
            "u_star"           : u_star,
            "peak_indices"     : peak_indices,
        }

    # ================================================================== #
    #  PHASE 1 — FORWARD PROFILING PASS                                  #
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

            # Eq. 5 — free potential identity
            cached_U[k] = tweedie_potential(v_cond, v_uncond, cached_sigma[k]).item()

            x = self.scheduler.step(v, t, x).prev_sample
            cached_x[k + 1] = x.detach().float().clone()

            if (k + 1) % 10 == 0 or k == 0:
                print(f"  [Phase1] step {k+1:>3}/{N} | sigma={sigma_t:.4f} | "
                      f"U_k={cached_U[k]:.4f} | mean={x.mean():.4f} | std={x.std():.4f}")

        return {
            "x": cached_x, "v": cached_v, "v_uncond": cached_v_uncond,
            "v_cond": cached_v_cond, "sigma": cached_sigma, "t": cached_t,
            "U": cached_U, "x_N": x, "N": N,
        }

    # ================================================================== #
    #  PHASE 2 — LEVERAGE-SCORE PEAK SELECTION                           #
    # ================================================================== #

    def _select_peaks(self, cache, text_embeddings, pooled_embeddings) -> List[int]:
        N = cache["N"]
        band = [k for k in range(N) if self.sigma_lo <= cache["sigma"][k] <= self.sigma_hi]
        if not band:
            print("[PeakBack]   WARNING: sigma band matched no cached steps; "
                  "falling back to the full trajectory.")
            band = list(range(N))

        if self.exact_leverage:
            # Eq. 8 — exact leverage score; costs one extra backward pass per candidate.
            scores = {}
            for k in band:
                _, grad, _, _ = self._potential_and_grad(
                    cache["x"][k], cache["t"][k], text_embeddings, pooled_embeddings
                )
                proj = joint_projector(
                    grad.reshape(-1), cache["v"][k].reshape(-1), cache["x"][k].reshape(-1)
                )
                scores[k] = cache["U"][k] * proj.norm().item()
        else:
            # Eq. 9 — free finite-difference surrogate, using only the Phase-1 cache.
            scores = leverage_surrogate(
                [cache["U"][k] for k in range(N)],
                [cache["sigma"][k] for k in range(N)],
            )
            scores = {k: v for k, v in scores.items() if k in band}

        return select_top_peaks(scores, top_j=self.J, min_sep=self.min_sep)

    # ================================================================== #
    #  PHASE 3 — CONSTRAINED NULL-SPACE LANGEVIN WALK                    #
    # ================================================================== #

    def _constrained_walk(
        self, x0, v_anchor, sigma_star, t_star, orig_cond_norm,
        text_embeddings, pooled_embeddings, u_star, seed, branch_idx,
    ):
        x = x0.clone()
        r = x0.float().norm().item()
        v = v_anchor.float()
        v_cfg_final = v_anchor
        accepted = False
        history = []

        for m in range(self.K_max):
            v_cfg, v_uncond, v_cond = self._velocity_forward(
                x, t_star, text_embeddings, pooled_embeddings, return_split=True
            )
            U_m = tweedie_potential(v_cond, v_uncond, sigma_star)
            history.append(U_m.item())
            v_cfg_final = v_cfg

            cond_norm = v_cond.float().norm()
            if U_m.item() <= u_star and cond_norm.item() >= (1 - self.delta) * orig_cond_norm.item():
                accepted = True
                break

            if m == self.K_max - 1:
                break   # don't spend a backward pass on a step we won't take

            # Eq. 6 — gradient of the potential, one backward pass
            _, grad, _, _ = self._potential_and_grad(x, t_star, text_embeddings, pooled_embeddings)

            rng = torch.Generator(device=x.device)
            rng.manual_seed(seed * 10000 + branch_idx * 100 + m)
            xi = torch.randn(x.shape, generator=rng, dtype=torch.float32, device=x.device)

            F = -self.peak_eta * grad + (2 * self.peak_eta) ** 0.5 * xi    # Eq. (drift+noise)
            w = joint_projector(F.reshape(-1), v.reshape(-1), x.float().reshape(-1)).reshape(x.shape)
            x_new, theta = geodesic_step(x.float(), w, r, theta_max=self.theta_max)
            x = x_new.to(x0.dtype)

            if not torch.isfinite(x).all():
                print(f"  [Phase3] WARNING: NaN/Inf at branch {branch_idx} step {m}, "
                      f"reverting to pre-step value")
                x = x0.clone()
                break

        return x, accepted, history, v_cfg_final

    # ================================================================== #
    #  PHASE 4 — RESUME DECODE                                            #
    # ================================================================== #

    def _resume_decode(self, x_tilde, k_star, v_cfg_at_kstar, cache, text_embeddings, pooled_embeddings):
        """
        Manual Euler continuation, mathematically identical to
        FlowMatchEulerDiscreteScheduler.step() for the 'euler' solver
        (prev_sample = sample + (sigma_next - sigma)*v == sample - dt*v),
        but driven directly from cached sigmas rather than the scheduler's
        internal (private) step-index state — avoids relying on
        scheduler._step_index to "resume" mid-trajectory.
        """
        N = cache["N"]
        sigma = cache["sigma"]
        x = x_tilde.clone()

        sigma_k    = sigma[k_star]
        sigma_next = sigma[k_star + 1] if k_star + 1 < N else 0.0
        dt = sigma_k - sigma_next
        x = (x.float() - dt * v_cfg_at_kstar.float()).to(x_tilde.dtype)

        for k in range(k_star + 1, N):
            t_k = cache["t"][k]
            v_cfg, _, _ = self._velocity_forward(
                x, t_k, text_embeddings, pooled_embeddings, return_split=True
            )
            sigma_k    = sigma[k]
            sigma_next = sigma[k + 1] if k + 1 < N else 0.0
            dt = sigma_k - sigma_next
            x = (x.float() - dt * v_cfg.float()).to(x_tilde.dtype)

        return x

    # ================================================================== #
    #  VELOCITY FORWARD (no-grad — for ordinary stepping/checks)          #
    # ================================================================== #

    def _velocity_forward(self, x, t, text_embeddings, pooled_embeddings, return_split=False):
        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype

        latent_input = (torch.cat([x, x]) if self.do_cfg else x).to(device=device, dtype=dtype)
        text_embeddings = text_embeddings.to(device=device, dtype=dtype)

        t_val   = t.item() if hasattr(t, "item") else float(t)
        t_batch = torch.tensor([t_val] * latent_input.shape[0], device=device, dtype=dtype)

        kwargs = dict(
            hidden_states         = latent_input,
            timestep               = t_batch,
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

    # ================================================================== #
    #  VELOCITY FORWARD + GRADIENT (Eq. 6 — one backward pass)            #
    # ================================================================== #

    def _potential_and_grad(self, x, t, text_embeddings, pooled_embeddings, eps_smooth=1e-6):
        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype

        x_req = x.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
        latent_input = torch.cat([x_req, x_req]) if self.do_cfg else x_req
        text_embeddings_d = text_embeddings.to(device=device, dtype=dtype)

        t_val   = t.item() if hasattr(t, "item") else float(t)
        t_batch = torch.tensor([t_val] * latent_input.shape[0], device=device, dtype=dtype)

        kwargs = dict(
            hidden_states         = latent_input,
            timestep               = t_batch,
            encoder_hidden_states = text_embeddings_d,
        )

        with torch.enable_grad():
            output = self.unet(**kwargs).sample
            v_uncond, v_cond = output.chunk(2)
            d = (v_cond - v_uncond).float()
            U_eps = torch.sqrt((d * d).sum() + eps_smooth ** 2)   # Remark 2 smoothing
            grad = torch.autograd.grad(U_eps, x_req)[0]

        return U_eps.detach(), grad.detach().float(), v_cond.detach(), v_uncond.detach()


# ══════════════════════════════════════════════════════════════════════ #
#  DROP-IN RUNNER  (multi-seed, single model load)                        #
# ══════════════════════════════════════════════════════════════════════ #

def run_sd3_peakback(opts: dict):
    """
    Drop-in runner for inference.py MODEL_REGISTRY.
    Set model_name: "sd3_peakback" in config.yaml to activate.

    For each seed: one base image (free) + up to `peakback.J` diverse
    branches from the SAME forward pass — no extra seeds needed for
    diversity, unlike sd3_onlb's per-seed re-walk.
    """
    from pipeline_wrapper import SD3PipelineWrapper

    cfg    = opts.get("_cfg", {})
    device = opts["device"]

    seeds = opts.get("seeds") or [opts["seed"]]

    print("\n" + "═" * 60)
    print("[PeakBack] Loading model (once for all seeds)…")
    print("═" * 60)
    wrapper = SD3PipelineWrapper(cfg, device=device)
    wrapper.load()

    print("\n[PeakBack] Encoding prompt (shared across all seeds)…")
    prompt_embeds, pooled_embeds = wrapper.encode_prompt(opts["prompt"], opts["negative_prompt"])

    pb_cfg = cfg.get("peakback", {})
    sampler = PeakBackSampler(
        unet           = wrapper.transformer,
        scheduler      = wrapper.scheduler,
        cfg            = cfg,
        device         = device,
        sigma_lo       = pb_cfg.get("sigma_lo",       0.3),
        sigma_hi       = pb_cfg.get("sigma_hi",       0.7),
        quantile_alpha = pb_cfg.get("quantile_alpha", 0.2),
        peak_eta       = pb_cfg.get("eta",            0.05),
        K_max          = pb_cfg.get("K_max",          5),
        delta          = pb_cfg.get("delta",          0.2),
        theta_max      = pb_cfg.get("theta_max",      0.3),
        J              = pb_cfg.get("J",              2),
        min_sep        = pb_cfg.get("min_sep",        3),
        exact_leverage = pb_cfg.get("exact_leverage", False),
    )

    base_out   = Path(opts["output"])
    multi_seed = len(seeds) > 1
    diverse_folder  = Path(pb_cfg.get("diverse_output_dir",  base_out.parent))
    original_folder = Path(pb_cfg.get("original_output_dir", base_out.parent))
    diverse_folder.mkdir(parents=True, exist_ok=True)
    original_folder.mkdir(parents=True, exist_ok=True)
    save_original = pb_cfg.get("save_original", True)

    def _base_path(seed):
        stem = base_out.stem + (f"_seed{seed}" if multi_seed else "")
        return original_folder / (stem + "_base" + base_out.suffix)

    def _branch_path(seed, j):
        stem = base_out.stem + (f"_seed{seed}" if multi_seed else "")
        return diverse_folder / (stem + f"_branch{j}" + base_out.suffix)

    records = []
    for idx, seed in enumerate(seeds):
        print("\n" + "═" * 60)
        print(f"[PeakBack] Seed {seed}  ({idx + 1}/{len(seeds)})")
        print("═" * 60)

        latents = wrapper.get_initial_latents(seed=seed)
        result  = sampler.run(latents, prompt_embeds, pooled_embeds, seed=seed)

        if save_original:
            base_path = _base_path(seed)
            wrapper.decode_latents(result["original_latents"]).save(base_path)
            print(f"[PeakBack] Base image → {base_path}")

        for br in result["branches"]:
            out_path = _branch_path(seed, br["branch_idx"])
            wrapper.decode_latents(br["latents"]).save(out_path)
            print(f"[PeakBack] Branch {br['branch_idx']} → {out_path}  "
                  f"(accepted={br['accepted']}, steps_used={br['steps_used']}, "
                  f"cos_to_base={br['cos_to_base']:.4f})")

            records.append({
                "seed": seed, "branch": br["branch_idx"], "k_star": br["k_star"],
                "accepted": br["accepted"], "steps_used": br["steps_used"],
                "cos_to_base": br["cos_to_base"], "out_path": str(out_path),
            })

    print("\n" + "═" * 60)
    print(f"[PeakBack] ── Summary ({len(seeds)} seed(s)) ──")
    print("═" * 60)
    header = f"{'Seed':>6} | {'Br':>3} | {'k*':>4} | {'Accept':>6} | {'Steps':>5} | {'cos':>7} | Output"
    print(header)
    print("-" * len(header))
    n_accepted = 0
    for r in records:
        print(f"{r['seed']:>6} | {r['branch']:>3} | {r['k_star']:>4} | "
              f"{str(r['accepted']):>6} | {r['steps_used']:>5} | {r['cos_to_base']:>7.4f} | {r['out_path']}")
        n_accepted += int(r["accepted"])
    print("-" * len(header))
    print(f"[PeakBack] Acceptance rate (guardrail Eq. 14 satisfied): "
          f"{n_accepted}/{len(records)}")
    print("[PeakBack] NOTE: acceptance is a geometric+fidelity guardrail, "
          "not proof of semantic diversity — validate against CLIP/Vendi scores.")
    print("═" * 60)
