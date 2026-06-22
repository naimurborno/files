"""
onlb_sampler.py
---------------
Orthogonal Null-Space Langevin Backtracking (ONLB) Sampler for SD3 Flow Matching.

Algorithm:
  Phase 1 — Forward pass  : scheduler.step() ODE, cache (x_k, v_k, dt_k, σ_k)
  Phase 2 — Backward pass : score-informed orthogonal Langevin walk
  Phase 3 — Shell project : preserve direction, match original norm
  Phase 4 — Forward decode: scheduler.step() ODE from diverse seed x̃_0

Backward update at step k:
  x̃_k = x̃_{k+1}
         - Δt·v_k                          (attraction / time reversal)
         + Δt·v_k⊥                         (soft repulsion — orthogonal escape)
         + (α·Δt/σ_k)·v_k                  (hard repulsion — score-based basin escape)
         + √(2η·Δt)·ξ⊥                     (orthogonal Langevin noise)

  v_k⊥ and ξ⊥ are both Gram-Schmidt projected onto null-space of v_k,
  so neither term can corrupt the flow direction.

Total model calls: 2N.

Multi-seed entry point (run_sd3_onlb):
  - Model is loaded ONCE before the seed loop.
  - Each seed runs the full ONLB pipeline independently.
  - Per-seed and average cosine similarities are reported at the end.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional


class ONLBSampler:

    def __init__(
        self,
        unet,
        scheduler,
        cfg          : dict,
        device       : str   = "cuda",
        lam          : float = 0.05,   # soft repulsion weight (orthogonal escape)
        eta          : float = 0.01,   # Langevin temperature
        alpha        : float = 0.1,    # hard repulsion weight (score-based basin escape)
        mu           : float = 0.3,    # CFG directional push weight
        max_drift    : float = 3.0,    # hard clamp: max ‖x̃ − x_anchor‖ / √D per step
        eps          : float = 1e-8,
    ):
        self.unet      = unet
        self.scheduler = scheduler
        self.cfg       = cfg
        self.device    = device
        self.lam       = lam
        self.eta       = eta
        self.alpha     = alpha
        self.mu        = mu
        self.max_drift = max_drift
        self.eps       = eps

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      20)
        self.guidance_scale = f_cfg.get("guidance_scale", 7.5)
        self.do_cfg         = self.guidance_scale > 1.0
        print(f"[ONLB] λ={self.lam}  η={self.eta}  α={self.alpha}  μ={self.mu}  "
              f"max_drift={self.max_drift}")

    # ================================================================== #
    #  PUBLIC ENTRY                                                       #
    # ================================================================== #

    def run(
        self,
        latents           : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor] = None,
        seed:int=0
    ) -> Dict[str, Any]:

        x0_original = latents.clone()

        # Phase 1
        print("\n[ONLB] ── Phase 1: Forward pass & trajectory caching ──")
        cached_x, cached_v, cached_v_uncond, cached_v_cond, cached_dt, cached_sigma, x_N = self._forward_pass(
            latents, text_embeddings, pooled_embeddings, tag="Cache"
        )

        # sanity: x_N should look like an image latent, not noise
        print(f"[ONLB]   x_N  mean={x_N.mean():.4f}  std={x_N.std():.4f}  "
              f"norm={x_N.norm():.4f}")

        # Phase 2
        print("\n[ONLB] ── Phase 2: Orthogonal Langevin backward pass ──")
        x0_tilde = self._backward_pass(
            cached_x, cached_v, cached_v_uncond, cached_v_cond, cached_dt, cached_sigma, seed
        )
        cos_pre = torch.nn.functional.cosine_similarity(
            x0_tilde.reshape(1, -1).float(),
            x0_original.reshape(1, -1).float()
        ).item()
        print(f"[ONLB] cosine sim BEFORE Phase 3 = {cos_pre:.4f}")

        # Phase 3
        print("\n[ONLB] ── Phase 3: Norm-preserving direction projection ──")
        x0_diverse = self._project_to_shell(x0_tilde, x0_original)

        escape = (x0_diverse - x0_original).norm().item()
        print(f"[ONLB]   escape distance ‖x̃₀ − x₀‖₂ = {escape:.4f}")
        cos_sim = torch.nn.functional.cosine_similarity(
            x0_diverse.reshape(1, -1).float(),
            x0_original.reshape(1, -1).float()
        ).item()
        print(f"[ONLB] cosine sim x0_diverse vs x0_original = {cos_sim:.4f}")
        print(f"[ONLB] if cos_sim > 0.95 → increase interp or lam")

        # Phase 4
        print("\n[ONLB] ── Phase 4: Forward decode from diverse seed ──")
        _, _, _, _, _, _, x_N_diverse = self._forward_pass(
            x0_diverse, text_embeddings, pooled_embeddings, tag="Decode"
        )
        print(f"[ONLB] x_N norm={x_N.norm():.4f}")
        print(f"[ONLB] x_N_diverse norm={x_N_diverse.norm():.4f}")
        cos_post4 = torch.nn.functional.cosine_similarity(
            x_N_diverse.reshape(1, -1).float(),
            x_N.reshape(1, -1).float()
        ).item()
        print(f"[ONLB] cosine sim AFTER Phase 4 = {cos_post4:.4f}")

        return {
            "original_latents" : x_N,
            "diverse_latents"  : x_N_diverse,
            "x0_original"      : x0_original,
            "x0_diverse"       : x0_diverse,
            "escape_distance"  : escape,
            # cosine similarities for external aggregation
            "cos_pre_phase3"   : cos_pre,
            "cos_x0"           : cos_sim,
            "cos_post_phase4"  : cos_post4,
        }

    # ================================================================== #
    #  PHASE 1 — FORWARD PASS                                            #
    # ================================================================== #

    def _forward_pass(
        self,
        x0                : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor],
        tag               : str = "Cache",
    ):
        """
        Use scheduler.step() — the same path as custom_flow_loop.py — so the
        trajectory is numerically identical to what the pipeline produces.

        Also record the actual dt and σ_k at each step so the backward pass
        can compute score-based repulsion without extra model calls.

        Returns:
          cached_x     [N+1]  — latent at each step boundary (fp32)
          cached_v     [N]    — velocity at each step (fp32)
          cached_dt    [N]    — actual |dt| used at each step (sigma-based)
          cached_sigma [N]    — σ_k at each step (for score repulsion)
          x_N                 — final image latent
        """
        # fresh scheduler state
        self.scheduler.set_timesteps(self.num_steps)
        timesteps = self.scheduler.timesteps   # e.g. [999, 966, ..., 0]
        N         = len(timesteps)

        cached_x        = [None] * (N + 1)
        cached_v        = [None] * N
        cached_v_uncond = [None] * N
        cached_v_cond   = [None] * N
        cached_dt       = [None] * N
        cached_sigma    = [None] * N

        x           = x0.clone()
        cached_x[0] = x.float().clone()

        for k, t in enumerate(timesteps):
            t_val = t.item() if hasattr(t, "item") else float(t)

            v, v_uncond, v_cond = self._velocity_forward(
                x, t, text_embeddings, pooled_embeddings, return_split=True
            )
            cached_v[k]        = v.detach().float().clone()
            cached_v_uncond[k] = v_uncond.detach().float().clone()
            cached_v_cond[k]   = v_cond.detach().float().clone()

            # ── actual dt and sigma from scheduler ─────────────────── #
            if hasattr(self.scheduler, "sigmas"):
                sigma_t    = self.scheduler.sigmas[k].item()
                sigma_next = self.scheduler.sigmas[k + 1].item()
                dt         = abs(sigma_next - sigma_t)
            else:
                sigma_t = 1.0 - k / N
                dt      = 1.0 / N
            cached_dt[k]    = dt
            cached_sigma[k] = max(sigma_t, 1e-4)   # guard against σ=0 at last step

            # ── step with scheduler (correct) ──────────────────────── #
            x = self.scheduler.step(v, t, x).prev_sample
            cached_x[k + 1] = x.detach().float().clone()

            if (k + 1) % 5 == 0 or k == 0:
                print(f"  [{tag}] step {k+1:>3}/{N} | t={t_val:>6.1f} | "
                      f"σ={sigma_t:.4f} | dt={dt:.4f} | "
                      f"mean={x.mean():.4f} | std={x.std():.4f}")

        return cached_x, cached_v, cached_v_uncond, cached_v_cond, cached_dt, cached_sigma, x

    # ================================================================== #
    #  PHASE 2 — ORTHOGONAL LANGEVIN BACKWARD PASS                       #
    # ================================================================== #

    def _backward_pass(
        self,
        cached_x       : list,
        cached_v       : list,
        cached_v_uncond: list,
        cached_v_cond  : list,
        cached_dt      : list,
        cached_sigma   : list,
        seed:int=0
    ) -> torch.Tensor:
        """
        Score-informed orthogonal Langevin backward walk: x_N → x̃_0.

        Full update at step k:

          x̃_k = x̃_{k+1}
                 - Δt·v_k                       (attraction: time reversal)
                 + Δt·v_k⊥                      (soft repulsion: orthogonal escape)
                 + (α·Δt/σ_k)·v_k               (hard repulsion: score-based basin escape)
                 + √(2η·Δt)·ξ⊥                  (orthogonal Langevin noise)
                 + μ·(dₖ/‖dₖ‖)·‖vₖ‖             (CFG directional push)

        where dₖ = v_uncond_k − v_cond_k  (points from cond basin → uncond basin)
        """
        N          = len(cached_v)
        orig_dtype = cached_x[N].dtype
        x_tilde    = cached_x[N].clone().float()
        D          = x_tilde.numel()
        step=0
        for k in range(N - 1, -1, -1):
            dt        = cached_dt[k]
            sigma_k   = cached_sigma[k]
            x_anchor  = cached_x[k].float()
            v_forward = cached_v[k].float()

            v_fwd_flat = v_forward.reshape(-1)
            v_fwd_sq   = v_fwd_flat.dot(v_fwd_flat) + self.eps
            v_fwd_norm = v_forward.norm() + self.eps

            # ── soft repulsion: orthogonal escape ────────────────────── #
            delta      = x_tilde - x_anchor
            delta_norm = delta.norm() + self.eps
            max_norm   = self.max_drift * (D ** 0.5)
            if delta_norm > max_norm:
                delta = delta * (max_norm / delta_norm)

            u       = self.lam * delta
            u_flat  = u.reshape(-1)
            v_ortho = u - (torch.dot(u_flat, v_fwd_flat) / v_fwd_sq) * v_forward
            # magnitude-match so soft escape is proportional to flow magnitude
            v_ortho = v_ortho * (v_fwd_norm / (v_ortho.norm() + self.eps))

            # ── hard repulsion: score-based basin escape ─────────────── #
            sigma_k     = max(sigma_k, 0.1)
            if sigma_k > 0.3:
                score_repulsion = (self.alpha * dt / sigma_k) * v_forward
            else:
                score_repulsion = torch.zeros_like(v_forward)

            # ── orthogonal Langevin noise ─────────────────────────────── #
            rng = torch.Generator(device=x_tilde.device)
            rng.manual_seed(seed * 10000 + step)   # unique per (seed, step)
            xi  = torch.randn(x_tilde.shape, generator=rng,
                              dtype=x_tilde.dtype, device=x_tilde.device)
            noise_scale = (2 * self.eta * dt) ** 0.5 * v_fwd_norm.item()

            # ── CFG directional push ─────────────────────────────────── #
            v_u      = cached_v_uncond[k]
            v_c      = cached_v_cond[k]
            d_k      = v_u - v_c
            d_k_norm = d_k.norm() + self.eps
            cfg_push = self.mu * (d_k / d_k_norm) * v_fwd_norm.item()
            F_raw=cfg_push+noise_scale*xi
            F_shear=F_raw-(torch.dot(F_raw.reshape(-1),v_fwd_flat)/v_fwd_sq)*v_forward
            sigma_thresh = self.cfg.get("onlb", {}).get("sigma_thresh", 0.3)
            if sigma_k > sigma_thresh:
                beta_k = max(0.0, 1.0 - (self.alpha * dt / sigma_k))
            else:
               beta_k  = 1.0
            # ── full update ──────────────────────────────────────────── #
            x_tilde = (
                x_tilde
                +0.001*beta_k*v_forward
                + score_repulsion
                +F_shear

                # + noise_scale * xi
                # + cfg_push
            )

            # NaN guard
            if not torch.isfinite(x_tilde).all():
                print(f"  [Backward] WARNING: NaN/Inf at k={k}, clamping")
                x_tilde = x_tilde.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)

            if (N - k) % 5 == 0 or k == N - 1:
                print(f"  [Backward] step {N-k:>3}/{N} | k={k} | σ={sigma_k:.4f} | "
                      f"|δ|={delta.norm():.2f} | |v⊥|={v_ortho.norm():.2f} | "
                      f"|score|={score_repulsion.norm():.2f} | "
                      f"|cfg_push|={cfg_push.norm():.2f} | "
                      f"noise={noise_scale:.4f} | "
                      f"mean={x_tilde.mean():.4f} | std={x_tilde.std():.4f}")
            step+=1

        # ── final sphere projection: enforce x̃₀ ∈ S ─────────────────── #
        # corrects norm drift accumulated over N backward steps
        # preserves direction completely — only rescales magnitude to √D
        # cached_x[0] is x0_original whose norm defines the shell radius
        target_norm = cached_x[0].float().norm()
        x_tilde = x_tilde * (target_norm / (x_tilde.norm() + self.eps))
        print(f"  [Backward] final sphere projection | "
              f"norm after = {x_tilde.norm():.4f} (target = {target_norm:.4f})")

        return x_tilde.to(orig_dtype)

    # ================================================================== #
    #  PHASE 3 — NORM-PRESERVING DIRECTION PROJECTION                    #
    # ================================================================== #

    # def _project_to_shell(self, x, x0_original):
    #   x0_orig_f  = x0_original.float()
    #   orig_norm  = x0_orig_f.norm() + self.eps
    #   x_hat      = x0_orig_f / orig_norm              # unit vector along x0

    #   # normalize Phase 2 output onto the shell
    #   x_on_shell = x.float() * (orig_norm / (x.float().norm() + self.eps))

    #   # measure cosine similarity — rho is the angular displacement
    #   rho = torch.nn.functional.cosine_similarity(
    #       x_on_shell.reshape(1, -1), x0_orig_f.reshape(1, -1)
    #   ).item()

    #   tau     = self.cfg.get("onlb", {}).get("target_cos", -0.5)
    #   abs_tau = abs(tau)

    #   if rho <= tau:
    #       # Case 1: natural escape — Phase 2 achieved separation naturally
    #       x_diverse = x_on_shell
    #       case      = 1

    #   elif rho > abs_tau:
    #       # Case 2: Householder reflection
    #       # cos(x_diverse, x0) = −rho < −|τ| = τ  ✓
    #       parallel  = x_on_shell.reshape(-1).dot(x_hat.reshape(-1)) * x_hat
    #       x_diverse = x_on_shell - 2 * parallel
    #       x_diverse = x_diverse * (orig_norm / (x_diverse.norm() + self.eps))
    #       case      = 2

    #   else:
    #       # Case 3: dead zone (τ < rho ≤ |τ|) — orthogonal projection → cos = 0
    #       # safe in 65536-d: (D−1)-dimensional complement is statistically
    #       # far from Ω_core for any reasonable prompt
    #       parallel  = x_on_shell.reshape(-1).dot(x_hat.reshape(-1)) * x_hat
    #       x_orth    = x_on_shell - parallel
    #       x_diverse = x_orth * (orig_norm / (x_orth.norm() + self.eps))
    #       case      = 3

    #   cos_final = torch.nn.functional.cosine_similarity(
    #       x_diverse.reshape(1, -1), x0_orig_f.reshape(1, -1)
    #   ).item()
    #   print(f"[ONLB]   Phase 3 Case {case} | "
    #         f"cos before={rho:.4f} → after={cos_final:.4f} | "
    #         f"τ={tau:.4f} | |τ|={abs_tau:.4f}")
    #   print(f"[ONLB]   ‖x̃₀ − x₀‖ = {(x_diverse - x0_orig_f).norm().item():.4f}")

    #   return x_diverse
    def _project_to_shell(self, x, x0_original):
            x0_orig_f = x0_original.float()
            orig_norm  = x0_orig_f.norm() + self.eps

            # normalize Phase 2 output
            x_normalized = x * (orig_norm / (x.norm() + self.eps))

            # compute cosine sim
            cos_sim = torch.nn.functional.cosine_similarity(
                x_normalized.reshape(1, -1), x0_orig_f.reshape(1, -1)
            ).item()

            # if not already in negative similarity territory, push further
            target_cos = self.cfg.get("onlb", {}).get("target_cos", -0.3)
            if cos_sim > target_cos:
                x0_unit     = x0_orig_f / orig_norm
                parallel    = (x_normalized.reshape(-1).dot(x0_unit.reshape(-1))) * x0_unit
                x_reflected = x_normalized - 2 * parallel
                x_normalized = x_reflected * (orig_norm / (x_reflected.norm() + self.eps))

            cos_final = torch.nn.functional.cosine_similarity(
                x_normalized.reshape(1, -1), x0_orig_f.reshape(1, -1)
            ).item()
            print(f"[ONLB]   cos before reflection={cos_sim:.4f} → after={cos_final:.4f}")
            print(f"[ONLB]   ‖x̃_0 − x_0‖ = {(x_normalized - x0_orig_f).norm().item():.4f}")

            return x_normalized

    # ================================================================== #
    #  VELOCITY FORWARD                                                  #
    # ================================================================== #

    def _velocity_forward(
        self,
        x                 : torch.Tensor,
        t                 : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor],
        return_split      : bool = False,
    ):
        """
        v_θ(x, t) with CFG.  When return_split=True, also returns v_uncond and v_cond.
        """
        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype

        latent_input    = (torch.cat([x, x]) if self.do_cfg else x).to(device=device, dtype=dtype)
        text_embeddings = text_embeddings.to(device=device, dtype=dtype)

        t_val   = t.item() if hasattr(t, "item") else float(t)
        t_batch = torch.tensor(
            [t_val] * latent_input.shape[0],
            device=device, dtype=dtype
        )

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
#  DROP-IN RUNNER  (multi-seed, single model load)                        #
# ══════════════════════════════════════════════════════════════════════ #

def run_sd3_onlb(opts: dict):
    """
    Drop-in runner for inference.py MODEL_REGISTRY.
    Set model_name: "sd3_onlb" in config.yaml to activate.
 
    Multi-seed behaviour
    --------------------
    - Reads `seeds` list from opts (populated by inference.py from config.yaml).
    - The SD3 pipeline (transformer, VAE, text encoders) is loaded ONCE.
    - Each seed generates one independent diverse image without reloading weights.
    - Per-seed cosine similarities (pre-phase3, x0, post-phase4) are printed.
    - Average cosine similarity across all seeds is reported in a final summary.
 
    Output naming
    -------------
    Single seed  → uses opts["output"] as-is (original behaviour preserved).
    Multi-seed   → stem gets "_seed{N}" appended, e.g. output11_seed41.png
    """
    from pipeline_wrapper import SD3PipelineWrapper
 
    cfg    = opts.get("_cfg", {})
    device = opts["device"]
 
    # ── resolve seeds ──────────────────────────────────────────────── #
    seeds = opts.get("seeds") or [opts["seed"]]
 
    # ── LOAD MODEL ONCE ────────────────────────────────────────────── #
    print("\n" + "═" * 60)
    print("[ONLB] Loading model (once for all seeds)…")
    print("═" * 60)
    wrapper = SD3PipelineWrapper(cfg, device=device)
    wrapper.load()
 
    # ── encode prompt once (shared across all seeds) ────────────────── #
    print("\n[ONLB] Encoding prompt (shared across all seeds)…")
    prompt_embeds, pooled_embeds = wrapper.encode_prompt(
        opts["prompt"], opts["negative_prompt"]
    )
 
    # ── ONLB hyper-parameters ────────────────────────────────────────── #
    onlb_cfg      = cfg.get("onlb", {})
    lam           = onlb_cfg.get("lam",           0.05)
    eta           = onlb_cfg.get("eta",           0.01)
    alpha         = onlb_cfg.get("alpha",         0.1)
    mu            = onlb_cfg.get("mu",            0.3)
    max_drift     = onlb_cfg.get("max_drift",     3.0)
    save_original = onlb_cfg.get("save_original", True)
 
    # sampler is stateless between runs — create once, reuse
    sampler = ONLBSampler(
        unet      = wrapper.transformer,
        scheduler = wrapper.scheduler,
        cfg       = cfg,
        device    = device,
        lam       = lam,
        eta       = eta,
        alpha     = alpha,
        mu        = mu,
        max_drift = max_drift,
    )
 
    # ── output path helper ────────────────────────────────────────────── #
    base_out = Path(opts["output"])
    multi_seed = len(seeds) > 1
 
    # Folder overrides from config (optional — fall back to base_out.parent)
    diverse_folder  = Path(onlb_cfg.get("diverse_output_dir",  base_out.parent))
    original_folder = Path(onlb_cfg.get("original_output_dir", base_out.parent))
    diverse_folder.mkdir(parents=True, exist_ok=True)
    original_folder.mkdir(parents=True, exist_ok=True)
 
    def _out_path(seed: int) -> Path:
        stem = base_out.stem + (f"_seed{seed}" if multi_seed else "")
        return diverse_folder / (stem + base_out.suffix)
 
    def _orig_path(out: Path) -> Path:
        return original_folder / (out.stem + "_original" + out.suffix)
 
    # ── tracking ──────────────────────────────────────────────────────── #
    # cos_x0        : cosine sim between x0_diverse and x0_original (Phase 3 output)
    # cos_post4     : cosine sim between x_N_diverse and x_N (Phase 4 output)
    records = []   # list of dicts, one per seed
 
    # ══════════════════════════════════════════════════════════════════ #
    #  SEED LOOP                                                         #
    # ══════════════════════════════════════════════════════════════════ #
    for idx, seed in enumerate(seeds):
        print("\n" + "═" * 60)
        print(f"[ONLB] Seed {seed}  ({idx + 1}/{len(seeds)})")
        print(f"[ONLB] λ={lam}  η={eta}  α={alpha}  μ={mu}  "
              f"max_drift={max_drift}  steps={opts['num_steps']}")
        print("═" * 60)
 
        # fresh initial noise for this seed
        latents = wrapper.get_initial_latents(seed=seed)
 
        result = sampler.run(latents, prompt_embeds, pooled_embeds,seed=seed)
 
        # ── save diverse image ──────────────────────────────────────── #
        out_path = _out_path(seed)
        wrapper.decode_latents(result["diverse_latents"]).save(out_path)
        print(f"\n[ONLB] Diverse image  → {out_path}")
 
        # ── optionally save original for side-by-side comparison ──── #
        if save_original:
            orig_path = _orig_path(out_path)
            wrapper.decode_latents(result["original_latents"]).save(orig_path)
            print(f"[ONLB] Original image → {orig_path}")
 
        print(f"[ONLB] Escape distance = {result['escape_distance']:.4f}")
 
        records.append({
            "seed"         : seed,
            "cos_pre_p3"   : result["cos_pre_phase3"],
            "cos_x0"       : result["cos_x0"],
            "cos_post_p4"  : result["cos_post_phase4"],
            "escape"       : result["escape_distance"],
            "out_path"     : str(out_path),
        })
 
    # ══════════════════════════════════════════════════════════════════ #
    #  COSINE SIMILARITY SUMMARY                                         #
    # ══════════════════════════════════════════════════════════════════ #
    print("\n" + "═" * 60)
    print(f"[ONLB] ── Cosine Similarity Summary ({len(seeds)} seed(s)) ──")
    print("═" * 60)
 
    header = f"{'Seed':>8} | {'cos(pre-P3)':>11} | {'cos(x0_div,x0)':>14} | {'cos(post-P4)':>12} | {'escape':>10} | Output"
    print(header)
    print("-" * len(header))
 
    cos_pre_list   = []
    cos_x0_list    = []
    cos_post4_list = []
 
    for r in records:
        print(
            f"{r['seed']:>8} | "
            f"{r['cos_pre_p3']:>11.4f} | "
            f"{r['cos_x0']:>14.4f} | "
            f"{r['cos_post_p4']:>12.4f} | "
            f"{r['escape']:>10.4f} | "
            f"{r['out_path']}"
        )
        cos_pre_list.append(r["cos_pre_p3"])
        cos_x0_list.append(r["cos_x0"])
        cos_post4_list.append(r["cos_post_p4"])
 
    if len(seeds) > 1:
        avg_pre   = sum(cos_pre_list)   / len(cos_pre_list)
        avg_x0    = sum(cos_x0_list)    / len(cos_x0_list)
        avg_post4 = sum(cos_post4_list) / len(cos_post4_list)
 
        print("-" * len(header))
        print(
            f"{'AVERAGE':>8} | "
            f"{avg_pre:>11.4f} | "
            f"{avg_x0:>14.4f} | "
            f"{avg_post4:>12.4f} |"
        )
 
    print("═" * 60)
    print(
        "[ONLB] Column guide:\n"
        "  cos(pre-P3)    — cosine sim of x̃₀ vs x₀  BEFORE Phase 3 projection\n"
        "  cos(x0_div,x0) — cosine sim of x0_diverse vs x0_original (Phase 3 output)\n"
        "  cos(post-P4)   — cosine sim of x_N_diverse vs x_N (final image latents)\n"
        "  escape         — ‖x̃₀ − x₀‖₂  Euclidean escape distance"
    )
    print("═" * 60)