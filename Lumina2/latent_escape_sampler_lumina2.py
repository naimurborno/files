"""
latent_escape_sampler_lumina2.py
----------------------------------
U-Profile Guided Initial Latent Escape (UGILE) — Lumina-Image-2.0 port.

Same algorithm as latent_escape_sampler.py (SD3) / latent_escape_sampler_sana.py
(SANA), UNCHANGED:

  Phase 1 — Forward profiling pass (N model calls), caching
             {x_k, v_k, v_cond_k, v_uncond_k, sigma_k, U_k}.
  Phase 2 — U-weighted semantic direction from cached v_cond - v_uncond.
  Phase 3 — Geodesic escape step in x_0 space, orthogonal to that
             direction, by exactly theta_max radians.
  Phase 4 — Full fresh forward pass from the escaped x_0.

None of that math is backbone-specific — Tweedie's potential and the
null-space projector operate on velocity tensors and a scalar sigma only,
so peakback_core's tweedie_potential / joint_projector / geodesic_step
are reused as-is, unmodified, exactly like the SANA setup. run() below is
byte-for-byte identical to SanaUGILESampler.run(); only the private
_velocity_forward / _forward_pass_with_profiling helpers change, because
those are where backbone-specific calling conventions live.

What's mechanically different from the SANA version (all confined to the
forward-call helpers, none of it touches UGILE's math):

  • SANA's `timestep_scale` quirk has no Lumina2 analog — there is
    nothing to multiply the timestep by here.
  • Lumina2 uses a REVERSED timestep convention relative to SD3/SANA:
    t=0 ↔ image, t=1 ↔ noise. Every forward call must first compute
        current_timestep = 1 - t / scheduler.config.num_train_timesteps
    (see Lumina2PipelineWrapper.reversed_timestep, reused here so the
    convention lives in exactly one place).
  • The transformer's raw output must be NEGATED before it's handed to
    `scheduler.step` (see Lumina2PipelineWrapper.negate_velocity) — this
    flips Lumina2's native output sign into the velocity sign convention
    FlowMatchEulerDiscreteScheduler.step() expects. SD3/SANA need no such
    flip.
  • Lumina2's official pipeline supports `cfg_trunc_ratio` (skip the
    uncond forward pass entirely past a timestep fraction) and
    `cfg_normalization` (rescale CFG output to the conditional branch's
    norm). UGILE's Phase 1 profiling pass needs BOTH v_cond and v_uncond
    at every step to compute Tweedie's potential (Eq. 5), so this
    sampler always runs the uncond branch during profiling — exactly
    the same call shape SD3/SANA already use — and applies
    cfg_trunc_ratio / cfg_normalization only as optional, config-gated
    extras on top of that shared v_cond/v_uncond pair (default: both
    off, i.e. plain CFG, so behaviour matches SD3/SANA unless you opt in).
  • x_0 has whatever channel/spatial shape Lumina2PipelineWrapper.
    get_initial_latents already produced (transformer.config.in_channels
    at H/8, W/8, rounded to a multiple of 2) — this sampler doesn't
    hardcode it, same policy as the SANA version.

Slots into inference.py MODEL_REGISTRY as "lumina2_ugile".
Add to your config (see config_lumina2.yaml):
  model_name: "lumina2_ugile"
  model_id:   "Alpha-VLLM/Lumina-Image-2.0"

  ugile:
    num_grad_steps:   5
    sigma_lo:         0.3
    sigma_hi:         0.9
    escape_scale:     3.0
    theta_max:        0.8
    walk_steps:       10
    J:                1
    diverse_output_dir:  "outputs/diverse"
    original_output_dir: "outputs/original"
    save_original:    true
"""

import math
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import torch.nn.functional as F

from peakback_core import (
    tweedie_potential,
    joint_projector,
    geodesic_step,
)


class Lumina2UGILESampler:
    """U-Profile Guided Initial Latent Escape sampler — Lumina2 backbone."""

    def __init__(
        self,
        transformer,
        scheduler,
        cfg             : dict,
        device          : str   = "cuda",
        num_grad_steps  : int   = 5,
        sigma_lo        : float = 0.3,
        sigma_hi        : float = 0.9,
        escape_scale    : float = 3.0,
        theta_max       : float = 0.8,
        walk_steps      : int   = 10,
        J               : int   = 1,
        eps             : float = 1e-8,
        noise_scale     : float = 0.5,
        gamma           : float = 1.2,
        cfg_trunc_ratio   : float = 1.0,
        cfg_normalization : bool  = False,
    ):
        self.transformer     = transformer
        self.scheduler        = scheduler
        self.cfg              = cfg
        self.device           = device
        self.num_grad_steps   = num_grad_steps
        self.sigma_lo         = sigma_lo
        self.sigma_hi         = sigma_hi
        self.escape_scale     = escape_scale
        self.theta_max        = theta_max
        self.walk_steps       = walk_steps
        self.J                = J
        self.eps              = eps
        self.noise_scale      = noise_scale
        self.gamma            = gamma

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      30)
        self.guidance_scale = f_cfg.get("guidance_scale", 4.0)
        self.do_cfg         = self.guidance_scale > 1.0

        # Lumina2-specific CFG extras — default off so behaviour matches
        # SD3/SANA (plain CFG) unless explicitly opted into via config.
        # See module docstring for why these never affect Phase 1's
        # v_cond/v_uncond profiling pair.
        self.cfg_trunc_ratio   = cfg_trunc_ratio
        self.cfg_normalization = cfg_normalization

        # No SANA-style timestep_scale here — Lumina2 has no such field.
        # Reversed-timestep + sign-flip are applied directly in
        # _velocity_forward below instead.
        self.num_train_timesteps = getattr(
            self.scheduler.config, "num_train_timesteps", 1000
        )

    # ================================================================== #
    #  PUBLIC ENTRY  (identical control flow to the SANA UGILESampler)    #
    # ================================================================== #

    def run(
        self,
        x0               : torch.Tensor,
        text_embeddings  : torch.Tensor,
        attention_mask   : Optional[torch.Tensor] = None,
        seed             : int = 0,
    ) -> Dict[str, Any]:

        # Phase 1 — forward profiling pass
        cache = self._forward_pass_with_profiling(x0, text_embeddings, attention_mask)

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

        epsilon = self.noise_scale * (1.0 / (math.sqrt(kappa) + self.eps))

        # A1: Full Trajectory-Covariance-Anchored Noise (Background Diversity)
        rng_cov = torch.Generator(device=x0.device)
        rng_cov.manual_seed(seed * 10000 + 0)
        v_boundary = torch.randn(x0.shape, generator=rng_cov, dtype=torch.float32, device=x0.device).flatten()

        s_flat_for_diff = semantic_dir.flatten()
        diffs_centered = [(diff.flatten() - s_flat_for_diff) for diff in diffs]

        # Use ALL steps for covariance to ensure robust background variance
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

        # Project eta onto span{x0, s_hat}^\perp
        x0_flat = x0.float().flatten()
        eta_flat = eta.flatten()
        eta_flat = eta_flat - torch.dot(eta_flat, x0_flat) / (torch.dot(x0_flat, x0_flat) + self.eps) * x0_flat
        eta_flat = eta_flat - torch.dot(eta_flat, s_flat) * s_flat
        eta = eta_flat.view_as(x0)

        # Normalize and scale noise
        eta = epsilon * eta / (eta.norm() + self.eps)

        # Add noise to x0 and re-normalize to stay exactly on the sphere (r)
        r = x0.float().norm().item()
        x0_perturbed = x0.float() + eta
        x0_perturbed = x0_perturbed * (r / (x0_perturbed.norm() + self.eps))

        # --- Phase 3 — Geodesic Escape Step from perturbed x0 ---
        rng2 = torch.Generator(device=x0.device)
        rng2.manual_seed(seed * 10000 + 2)
        xi = torch.randn(x0_perturbed.shape, generator=rng2, dtype=torch.float32, device=x0.device)

        # Smooth Structural Low-Frequency Bias (No Mask, Milder Blur)
        if x0.dim() == 4:
            B, C, H, W = xi.shape
            # Downsample to 60% and back up. This smoothly filters out high-frequency texture
            # noise without creating the pixel-block artifacts a coarser downsample would cause.
            xi_low = F.interpolate(xi, scale_factor=0.55, mode='bilinear', recompute_scale_factor=False, align_corners=False)
            xi_low = F.interpolate(xi_low, size=(H, W), mode='bilinear', align_corners=False)

            # Blend to preserve some high-frequency detail for natural variation
            xi = 0.7 * xi_low + 0.3 * xi

        # Gram-Schmidt: remove component along semantic_unit and x0_perturbed
        xi_flat = xi.flatten()
        xi_flat = xi_flat - torch.dot(xi_flat, s_flat) * s_flat
        x0p_flat = x0_perturbed.flatten()
        xi_flat = xi_flat - torch.dot(xi_flat, x0p_flat) / (torch.dot(x0p_flat, x0p_flat) + self.eps) * x0p_flat

        e_hat = xi_flat.view_as(x0_perturbed)
        e_hat = e_hat / (e_hat.norm() + self.eps)

        theta = self.theta_max
        x0_new = math.cos(theta) * x0_perturbed + math.sin(theta) * r * e_hat
        x0_new = x0_new.to(x0.dtype)

        cos_x0 = torch.nn.functional.cosine_similarity(
            x0_new.reshape(1, -1).float(), x0.float().reshape(1, -1)
        ).item()

        # Phase 4 — full forward pass from x_0_new
        x_N_diverse = self._full_forward_pass(x0_new, text_embeddings, attention_mask)

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
    #  PHASE 1 — FORWARD PROFILING PASS                                   #
    # ================================================================== #

    def _forward_pass_with_profiling(self, x0, text_embeddings, attention_mask):
        self._set_timesteps_with_mu(x0)
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
                x, t, text_embeddings, attention_mask, return_split=True
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

            if False:  # step-level logging suppressed
                pass

        return {
            "x": cached_x, "v": cached_v,
            "v_uncond": cached_v_uncond, "v_cond": cached_v_cond,
            "sigma": cached_sigma, "t": cached_t,
            "U": cached_U, "x_N": x, "N": N,
        }

    # ================================================================== #
    #  PHASE 2 — U-WEIGHTED ESCAPE DIRECTION                              #
    #  (kept for parity with the SD3/SANA files; unused by run() above —  #
    #  left in so all three backbones expose the same surface / knobs.)   #
    # ================================================================== #

    def _compute_escape_direction(self, x0, cache, text_embeddings, attention_mask):
        N = cache["N"]

        band = [k for k in range(N)
                if self.sigma_lo <= cache["sigma"][k] <= self.sigma_hi]
        if not band:
            print("[UGILE-Lumina2]   WARNING: no steps in sigma band, using full trajectory")
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
                                       text_embeddings, attention_mask)
            escape_dir = escape_dir + w_k * grad

        return escape_dir

    def _grad_U_at_xk(self, x_k, t, sigma, text_embeddings, attention_mask,
                      eps_smooth=1e-6):
        """grad_{x_k}[ U_k ] via one backward pass. U_k = sigma * ||v_c - v_u||."""
        device = next(self.transformer.parameters()).device
        dtype  = next(self.transformer.parameters()).dtype

        x_req = x_k.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
        latent_input = torch.cat([x_req, x_req]) if self.do_cfg else x_req

        t_batch = self._reversed_timestep_batch(t, latent_input.shape[0], device, dtype)

        kwargs = dict(
            hidden_states         = latent_input,
            timestep              = t_batch,
            encoder_hidden_states = text_embeddings.to(device=device, dtype=dtype),
            return_dict           = False,
        )
        if attention_mask is not None:
            kwargs["encoder_attention_mask"] = attention_mask.to(device=device)

        with torch.enable_grad():
            output = self.transformer(**kwargs)[0]
            output = -output   # Lumina2 sign-flip — see module docstring
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

    def _short_forward_pass(self, x0_new, text_embeddings, attention_mask, n_steps=10):
        """Run only the first n_steps to cheaply check if x0_new is on-manifold."""
        self._set_timesteps_with_mu(x0_new)
        timesteps = self.scheduler.timesteps[:n_steps]
        x = x0_new.clone()
        for t in timesteps:
            v = self._velocity_forward(x, t, text_embeddings, attention_mask)
            x = self.scheduler.step(v, t, x).prev_sample
        return x

    # ================================================================== #
    #  PHASE 4 — FULL FORWARD PASS FROM MODIFIED x0                      #
    # ================================================================== #

    def _full_forward_pass(self, x0_new, text_embeddings, attention_mask):
        """Run all N Euler steps from the new initial latent."""
        self._set_timesteps_with_mu(x0_new)
        timesteps = self.scheduler.timesteps
        x = x0_new.clone()

        for t in timesteps:
            v = self._velocity_forward(x, t, text_embeddings, attention_mask)
            x = self.scheduler.step(v, t, x).prev_sample

        return x

    # ================================================================== #
    #  TIMESTEP SETUP  (Lumina2-specific: needs a sequence-length-        #
    #  dependent `mu` shift — no SD3/SANA analog)                         #
    # ================================================================== #

    def _set_timesteps_with_mu(self, x0):
        """
        Reproduces Lumina2Pipeline.__call__'s timestep setup: sigmas are
        linear in [1, 1/num_steps], and `mu` (the timestep-shift used by
        the scheduler's resolution-dependent shifting) is derived from
        the latent's flattened sequence length, exactly as
        diffusers.pipelines.lumina2.pipeline_lumina2.calculate_shift does.
        This is pure plumbing for matching Lumina2's own schedule — UGILE
        itself doesn't care how timesteps were spaced, only that
        self.scheduler.timesteps / .sigmas are populated before Phase 1/4.
        """
        import numpy as np

        num_steps = self.num_steps
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)

        # image_seq_len: number of "tokens" the transformer sees, i.e.
        # flattened H*W of the latent (channels excluded) — matches
        # `latents.shape[1]` in the official pipeline after its internal
        # patchify-on-the-fly bookkeeping. For an unpatchified (B,C,H,W)
        # latent like ours, H*W is the correct token count.
        _, _, H, W = x0.shape
        image_seq_len = H * W

        base_seq_len = self.scheduler.config.get("base_image_seq_len", 256)
        max_seq_len  = self.scheduler.config.get("max_image_seq_len", 4096)
        base_shift   = self.scheduler.config.get("base_shift", 0.5)
        max_shift    = self.scheduler.config.get("max_shift", 1.15)

        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b

        self.scheduler.set_timesteps(sigmas=sigmas, device=self.device, mu=mu)

    def _reversed_timestep_batch(self, t, batch_size, device, dtype):
        """current_timestep = 1 - t / num_train_timesteps, broadcast to batch."""
        t_val = t.item() if hasattr(t, "item") else float(t)
        current_t = 1 - t_val / self.num_train_timesteps
        return torch.full((batch_size,), current_t, device=device, dtype=dtype)

    # ================================================================== #
    #  VELOCITY FORWARD                                                    #
    # ================================================================== #

    def _velocity_forward(self, x, t, text_embeddings, attention_mask,
                          return_split=False):
        device = next(self.transformer.parameters()).device
        dtype  = next(self.transformer.parameters()).dtype

        latent_input = (torch.cat([x, x]) if self.do_cfg else x).to(device=device, dtype=dtype)
        text_embeddings = text_embeddings.to(device=device, dtype=dtype)

        # Lumina2-specific: reversed timestep convention (t=0 ↔ image,
        # t=1 ↔ noise) — no SD3/SANA analog. No timestep_scale multiply
        # either (that quirk is SANA-only and doesn't exist here).
        t_batch = self._reversed_timestep_batch(t, latent_input.shape[0], device, dtype)

        kwargs = dict(
            hidden_states         = latent_input,
            timestep              = t_batch,
            encoder_hidden_states = text_embeddings,
            return_dict           = False,
        )
        if attention_mask is not None:
            kwargs["encoder_attention_mask"] = attention_mask.to(device=device)

        with torch.no_grad():
            output = self.transformer(**kwargs)[0]
            # Lumina2-specific sign flip before scheduler.step — no
            # SD3/SANA analog. See module docstring.
            output = -output

        if self.do_cfg:
            v_uncond, v_cond = output.chunk(2)
            v_cfg = v_uncond + self.guidance_scale * (v_cond - v_uncond)

            # Optional Lumina2-only extra (default off, see __init__):
            # rescale CFG output to the conditional branch's norm.
            if self.cfg_normalization:
                cond_norm  = torch.norm(v_cond, dim=-1, keepdim=True)
                cfg_norm   = torch.norm(v_cfg,  dim=-1, keepdim=True) + 1e-12
                v_cfg = v_cfg * (cond_norm / cfg_norm)

            if return_split:
                return v_cfg, v_uncond, v_cond
            return v_cfg

        if return_split:
            return output, output, output
        return output


# ══════════════════════════════════════════════════════════════════════ #
#  DROP-IN RUNNER                                                         #
# ══════════════════════════════════════════════════════════════════════ #

def run_lumina2_ugile(opts: dict):
    """
    Drop-in runner for inference.py's MODEL_REGISTRY.
    Set model_name: "lumina2_ugile" in config.yaml to activate.

    Iterates over all prompts and seeds from config. The model is loaded once
    and the sampler is reused across all prompts and seeds.
    """
    from pipeline_wrapper_lumina2 import Lumina2PipelineWrapper

    cfg     = opts.get("_cfg", {})
    device  = opts["device"]
    seeds   = opts.get("seeds") or [opts["seed"]]
    prompts = cfg.get("prompts") or [opts["prompt"]]

    ug_cfg = cfg.get("ugile", {})

    print(f"[UGILE-Lumina2] Loading model (once for all prompts/seeds)…")
    wrapper = Lumina2PipelineWrapper(cfg, device=device)
    wrapper.load()

    sampler = Lumina2UGILESampler(
        transformer     = wrapper.transformer,
        scheduler       = wrapper.scheduler,
        cfg             = cfg,
        device          = device,
        num_grad_steps  = ug_cfg.get("num_grad_steps",  5),
        sigma_lo        = ug_cfg.get("sigma_lo",        0.3),
        sigma_hi        = ug_cfg.get("sigma_hi",        0.9),
        escape_scale    = ug_cfg.get("escape_scale",    3.0),
        theta_max       = ug_cfg.get("theta_max",       0.8),
        walk_steps      = ug_cfg.get("walk_steps",      10),
        J               = ug_cfg.get("J",               1),
        noise_scale     = ug_cfg.get("noise_scale",     0.2),
        gamma           = ug_cfg.get("gamma",           1.2),
        cfg_trunc_ratio   = cfg.get("flow", {}).get("cfg_trunc_ratio",   1.0),
        cfg_normalization = cfg.get("flow", {}).get("cfg_normalization", False),
    )

    base_out        = Path(opts["output"])
    multi_seed      = len(seeds) > 1
    diverse_folder  = Path(ug_cfg.get("diverse_output_dir",  "outputs/diverse"))
    original_folder = Path(ug_cfg.get("original_output_dir", "outputs/original"))
    diverse_folder.mkdir(parents=True, exist_ok=True)
    original_folder.mkdir(parents=True, exist_ok=True)
    save_original   = ug_cfg.get("save_original", True)

    def _base_path(prompt_idx, seed):
        stem = base_out.stem + f"_p{prompt_idx}" + (f"_seed{seed}" if multi_seed else "")
        return original_folder / (stem + "_base" + base_out.suffix)

    def _branch_path(prompt_idx, seed, j):
        stem = base_out.stem + f"_p{prompt_idx}" + (f"_seed{seed}" if multi_seed else "")
        return diverse_folder / (stem + f"_branch{j}" + base_out.suffix)

    records = []
    total = len(prompts) * len(seeds)
    done  = 0

    # 1. ADD THIS LINE: Read the offset injected by the parallel runner config
    prompt_offset = cfg.get("prompt_offset", 0)

    for p_idx, prompt in enumerate(prompts):
        # 2. ADD THIS LINE: Calculate the true absolute index for filename safety
        global_idx = p_idx + prompt_offset

        print(f"\n[UGILE-Lumina2] Prompt {global_idx + 1}/{len(prompts) + prompt_offset}: \"{prompt}\"")
        prompt_embeds, attention_mask = wrapper.encode_prompt(
            prompt, opts["negative_prompt"]
        )

        for seed in seeds:
            done += 1
            print(f"[UGILE-Lumina2]   Generating image {done}/{total}  (seed={seed})")

            latents = wrapper.get_initial_latents(seed=seed)
            result  = sampler.run(latents, prompt_embeds, attention_mask, seed=seed)

            if save_original:
                # 3. CHANGE THIS: Use global_idx instead of p_idx
                base_path = _base_path(global_idx, seed)
                wrapper.decode_latents(result["original_latents"]).save(base_path)
                print(f"[UGILE-Lumina2]   Base  → {base_path}")

            for br in result["branches"]:
                # 4. CHANGE THIS: Use global_idx instead of p_idx
                out_path = _branch_path(global_idx, seed, br["branch_idx"])
                wrapper.decode_latents(br["latents"]).save(out_path)
                print(f"[UGILE-Lumina2]   Branch {br['branch_idx']} → {out_path}")

                records.append({
                    "prompt_idx": global_idx, # 5. CHANGE THIS: Use global_idx
                    "prompt"    : prompt,
                    "seed"      : seed,
                    "branch"    : br["branch_idx"],
                    "theta"     : br["theta"],
                    "cos_x0"    : br["cos_x0"],
                    "cos_xN"    : br["cos_xN"],
                    "out_path"  : str(out_path),
                })

    print("\n" + "═" * 70)
    print(f"[UGILE-Lumina2] Done — {len(prompts)} prompt(s) × {len(seeds)} seed(s) = {total} image(s)")
    print("═" * 70)
    header = f"{'P':>2} | {'Seed':>6} | {'Br':>3} | {'cos_x0':>7} | {'cos_xN':>7} | Output"
    print(header)
    print("-" * len(header))
    for r in records:
        print(f"{r['prompt_idx']:>2} | {r['seed']:>6} | {r['branch']:>3} | "
              f"{r['cos_x0']:>7.4f} | {r['cos_xN']:>7.4f} | {r['out_path']}")
    print("═" * 70)