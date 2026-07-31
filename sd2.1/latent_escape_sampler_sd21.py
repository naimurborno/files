"""
latent_escape_sampler_sd21.py
--------------------------------
UGILE ported to Stable Diffusion 2.1. Same algorithm as SD3/SANA/Lumina2/
PixArt-Sigma — only the four phases and joint-projector/geodesic-step math
(peakback_core.py) are shared; everything below is backbone plumbing:
  Phase 1 — forward profiling pass, caching {x_k, v_k, v_cond_k, v_uncond_k, sigma_k, U_k}
  Phase 2 — U-weighted semantic direction + noise injection
  Phase 3 — geodesic escape step in x0 space
  Phase 4 — full fresh forward pass from escaped x0

SD2.1 is the SIMPLEST backbone of the family — plain eps-prediction UNet,
even fewer quirks than PixArt-Sigma:
  - no reversed timestep, no output negation, no cfg_trunc_ratio
  - no attention_mask (CLIP has no real padding mask to pass through)
  - no added_cond_kwargs (no -MS style micro-conditioning)
  - UNet never returns learned-variance channels for the base checkpoint
    (out_channels == in_channels), so no chunk-splitting needed
  - DDIM/PNDM expose no .sigmas attribute the way DPMSolver does, so sigma_k
    is derived directly from scheduler.alphas_cumprod:
        sigma_t = sqrt((1 - a_bar_t) / a_bar_t)
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


class SD21UGILESampler:

    def __init__(
        self,
        unet,
        scheduler,
        cfg             : dict,
        wrapper         = None,          # unused for SD2.1, kept for interface parity
        device          : str   = "cuda",
        sigma_lo        : float = 0.3,
        sigma_hi        : float = 0.9,
        theta_max       : float = 0.5,
        noise_scale     : float = 1.5,
        eps             : float = 1e-8,
    ):
        self.unet        = unet
        self.scheduler   = scheduler
        self.cfg         = cfg
        self.wrapper     = wrapper
        self.device      = device
        self.sigma_lo    = sigma_lo
        self.sigma_hi    = sigma_hi
        self.theta_max   = theta_max
        self.noise_scale = noise_scale
        self.eps         = eps

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      50)
        self.guidance_scale = f_cfg.get("guidance_scale", 7.5)
        self.do_cfg         = self.guidance_scale > 1.0

        gen_cfg = cfg.get("generation", {})
        self.H  = gen_cfg.get("height", 512)
        self.W  = gen_cfg.get("width",  512)

    # ================================================================== #
    #  PUBLIC ENTRY                                                        #
    # ================================================================== #

    def run(self, x0, text_embeddings, attention_mask=None, seed: int = 0) -> Dict[str, Any]:
        cache = self._forward_pass_with_profiling(x0, text_embeddings, attention_mask)

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

        if N > 2:
            d2U = U_vals[:-2] - 2 * U_vals[1:-1] + U_vals[2:]
            kappa = d2U.abs().mean().item()
        else:
            kappa = 1.0

        epsilon = self.noise_scale * (1.0 / (math.sqrt(kappa) + self.eps))
        epsilon = min(epsilon, 10.0)

        rng = torch.Generator(device=x0.device)
        rng.manual_seed(seed * 10000 + 1)
        jitter = torch.randn(x0.shape, generator=rng, dtype=torch.float32, device=x0.device)

        eta = semantic_dir / (semantic_dir.norm() + self.eps) * 0 + jitter  # background noise
        x0_flat  = x0.float().flatten()
        eta_flat = eta.flatten()
        eta_flat = eta_flat - torch.dot(eta_flat, x0_flat) / (torch.dot(x0_flat, x0_flat) + self.eps) * x0_flat
        eta_flat = eta_flat - torch.dot(eta_flat, s_flat) * s_flat
        eta = eta_flat.view_as(x0)
        eta = epsilon * eta / (eta.norm() + self.eps)

        r = x0.float().norm().item()
        x0_perturbed = x0.float() + eta
        x0_perturbed = x0_perturbed * (r / (x0_perturbed.norm() + self.eps))

        rng2 = torch.Generator(device=x0.device)
        rng2.manual_seed(seed * 10000 + 2)
        xi = torch.randn(x0_perturbed.shape, generator=rng2, dtype=torch.float32, device=x0.device)

        if x0.dim() == 4:
            B, C, H, W = xi.shape
            xi_low = F.interpolate(xi, scale_factor=0.25, mode='bilinear',
                                    recompute_scale_factor=False, align_corners=False)
            xi_low = F.interpolate(xi_low, size=(H, W), mode='bilinear', align_corners=False)
            xi = 0.25 * xi + 0.75 * xi_low

        xi_flat  = xi.flatten()
        x0p_flat = x0_perturbed.flatten()
        w_proj = joint_projector(xi_flat, s_flat, x0p_flat, ridge=1e-3)

        x0_new_flat, theta_t = geodesic_step(x0p_flat, w_proj, r, theta_max=self.theta_max)
        theta = theta_t.item()
        x0_new = x0_new_flat.view_as(x0_perturbed).to(x0.dtype)

        cos_x0 = F.cosine_similarity(x0_new.reshape(1, -1).float(), x0.float().reshape(1, -1)).item()

        x_N_diverse = self._full_forward_pass(x0_new, text_embeddings, attention_mask)

        cos_xN = F.cosine_similarity(
            x_N_diverse.reshape(1, -1).float(), cache["x_N"].reshape(1, -1).float()
        ).item()

        return {
            "original_latents": cache["x_N"],
            "branches": [{
                "branch_idx": 0, "theta": theta,
                "cos_x0": cos_x0, "cos_xN": cos_xN,
                "latents": x_N_diverse,
            }],
        }

    # ================================================================== #
    #  PHASE 1 — FORWARD PROFILING PASS                                   #
    # ================================================================== #

    def _forward_pass_with_profiling(self, x0, text_embeddings, attention_mask):
        self.scheduler.set_timesteps(self.num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        N = len(timesteps)

        cached_x, cached_v = [None] * (N + 1), [None] * N
        cached_v_uncond, cached_v_cond = [None] * N, [None] * N
        cached_sigma, cached_t, cached_U = [None] * N, [None] * N, [None] * N

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

            # DDIM/PNDM don't expose .sigmas like DPMSolver/FlowMatch
            # schedulers do — derive sigma_k from alphas_cumprod instead.
            cached_sigma[k] = self._sigma_from_alphas_cumprod(t)

            cached_U[k] = tweedie_potential(v_cond, v_uncond, cached_sigma[k]).item()

            x = self.scheduler.step(v, t, x).prev_sample
            cached_x[k + 1] = x.detach().float().clone()

        return {
            "x": cached_x, "v": cached_v,
            "v_uncond": cached_v_uncond, "v_cond": cached_v_cond,
            "sigma": cached_sigma, "t": cached_t,
            "U": cached_U, "x_N": x, "N": N,
        }

    def _sigma_from_alphas_cumprod(self, t) -> float:
        """sigma_t = sqrt((1 - a_bar_t) / a_bar_t), the standard eps<->sigma
        relation for DDPM-derived schedulers (DDIM/PNDM included)."""
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        t_idx = int(t.item()) if torch.is_tensor(t) else int(t)
        t_idx = max(0, min(t_idx, len(alphas_cumprod) - 1))
        a_bar = alphas_cumprod[t_idx].item()
        sigma_t = math.sqrt(max((1.0 - a_bar) / max(a_bar, 1e-8), 0.0))
        return max(sigma_t, 1e-4)

    # ================================================================== #
    #  PHASE 4 — FULL FORWARD PASS FROM MODIFIED x0                       #
    # ================================================================== #

    def _full_forward_pass(self, x0_new, text_embeddings, attention_mask):
        self.scheduler.set_timesteps(self.num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        x = x0_new.clone()
        for t in timesteps:
            v = self._velocity_forward(x, t, text_embeddings, attention_mask)
            x = self.scheduler.step(v, t, x).prev_sample
        return x

    # ================================================================== #
    #  VELOCITY FORWARD — standard eps-prediction, no quirks               #
    # ================================================================== #

    def _velocity_forward(self, x, t, text_embeddings, attention_mask, return_split=False):
        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype

        latent_input = (torch.cat([x, x]) if self.do_cfg else x).to(device=device, dtype=dtype)
        # Official pipeline calls this before every forward pass —
        # required for schedulers that rescale the input by sigma.
        latent_input = self.scheduler.scale_model_input(latent_input, t)
        text_embeddings = text_embeddings.to(device=device, dtype=dtype)

        kwargs = dict(
            sample                 = latent_input,
            timestep               = t,
            encoder_hidden_states  = text_embeddings,
            return_dict            = False,
        )
        # attention_mask is always None for SD2.1's CLIP encoder (kept only
        # for interface parity with the T5-based ports).
        if attention_mask is not None:
            kwargs["encoder_attention_mask"] = attention_mask.to(device=device)

        with torch.no_grad():
            output = self.unet(**kwargs)[0]
            # SD2.1's base checkpoint does not predict learned variance
            # (out_channels == in_channels), so no chunk-splitting is needed
            # here (unlike PixArt-Sigma's optional 2x out_channels case).
            if output.shape[1] == 2 * x.shape[1]:
                output, _ = output.chunk(2, dim=1)

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

def run_sd21_ugile(opts: dict):
    from pipeline_wrapper_sd21 import SD21PipelineWrapper

    cfg     = opts.get("_cfg", {})
    device  = opts["device"]
    seeds   = opts.get("seeds") or [opts["seed"]]
    ug_cfg  = cfg.get("ugile", {})

    prompts_file = cfg.get("prompts_file")
    if prompts_file:
        import yaml
        with open(prompts_file) as f:
            prompts_cfg = yaml.safe_load(f)
        prompts = prompts_cfg.get("prompts") or prompts_cfg
        print(f"[UGILE-SD21] Loaded {len(prompts)} prompt(s) from {prompts_file}")
    else:
        prompts = cfg.get("prompts") or [opts["prompt"]]

    # Global position of the FIRST prompt in this process's slice — see the
    # PixArt-Sigma runner for the full rationale (multi-GPU filename clashes).
    prompt_offset = opts.get("prompt_offset", cfg.get("prompt_offset", 0))

    print(f"[UGILE-SD21] Loading model…")
    wrapper = SD21PipelineWrapper(cfg, device=device)
    wrapper.load()

    sampler = SD21UGILESampler(
        unet         = wrapper.unet,
        scheduler    = wrapper.scheduler,
        cfg          = cfg,
        wrapper      = wrapper,
        device       = device,
        sigma_lo     = ug_cfg.get("sigma_lo",  0.3),
        sigma_hi     = ug_cfg.get("sigma_hi",  0.9),
        theta_max    = ug_cfg.get("theta_max", 0.5),
        noise_scale  = ug_cfg.get("noise_scale", 1.5),
    )

    base_out = Path(opts["output"])
    diverse_folder  = Path(ug_cfg.get("diverse_output_dir",  "outputs/diverse"))
    original_folder = Path(ug_cfg.get("original_output_dir", "outputs/original"))
    diverse_folder.mkdir(parents=True, exist_ok=True)
    original_folder.mkdir(parents=True, exist_ok=True)
    save_original = ug_cfg.get("save_original", True)
    multi_seed = len(seeds) > 1

    def _base_path(p_idx, seed):
        stem = base_out.stem + f"_p{p_idx}" + (f"_seed{seed}" if multi_seed else "")
        return original_folder / (stem + "_base" + base_out.suffix)

    def _branch_path(p_idx, seed, j):
        stem = base_out.stem + f"_p{p_idx}" + (f"_seed{seed}" if multi_seed else "")
        return diverse_folder / (stem + f"_branch{j}" + base_out.suffix)

    records = []
    for local_idx, prompt in enumerate(prompts):
        p_idx = prompt_offset + local_idx  # absolute position in the full prompts file
        print(f"\n[UGILE-SD21] Prompt {p_idx + 1} (local {local_idx + 1}/{len(prompts)}): \"{prompt}\"")
        prompt_embeds, attention_mask = wrapper.encode_prompt(prompt, opts["negative_prompt"])

        for seed in seeds:
            print(f"[UGILE-SD21]   seed={seed}")
            latents = wrapper.get_initial_latents(seed=seed)
            result  = sampler.run(latents, prompt_embeds, attention_mask, seed=seed)

            if save_original:
                base_path = _base_path(p_idx, seed)
                wrapper.decode_latents(result["original_latents"]).save(base_path)
                print(f"[UGILE-SD21]   Base  → {base_path}")

            for br in result["branches"]:
                out_path = _branch_path(p_idx, seed, br["branch_idx"])
                wrapper.decode_latents(br["latents"]).save(out_path)
                print(f"[UGILE-SD21]   Branch {br['branch_idx']} → {out_path}")
                records.append({
                    "prompt_idx": p_idx, "prompt": prompt, "seed": seed,
                    "branch": br["branch_idx"], "theta": br["theta"],
                    "cos_x0": br["cos_x0"], "cos_xN": br["cos_xN"],
                    "out_path": str(out_path),
                })

    print(f"\n[UGILE-SD21] Done — {len(records)} image(s)")
    return records
