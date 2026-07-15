# Hello my name is naimur asif borno
import math
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
        noise_scale     : float = 8.0,   # Lowered base scale for SD3 stability
        gamma           : float = 1.2,
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

        # FIX 1: Hard Clamp Epsilon to 2% of latent radius to prevent SD3 VAE breakdown
        # (5% was too high and caused shredded grass artifacts)
        r = x0.float().norm().item()
        max_eps = r * 0.02 
        epsilon = self.noise_scale * (1.0 / (math.sqrt(kappa) + self.eps))
        epsilon = min(epsilon, max_eps)

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

        # Project eta onto span{x0, s_hat}^\perp
        x0_flat = x0.float().flatten()
        eta_flat = eta.flatten()
        eta_flat = eta_flat - torch.dot(eta_flat, x0_flat) / (torch.dot(x0_flat, x0_flat) + self.eps) * x0_flat
        eta_flat = eta_flat - torch.dot(eta_flat, s_flat) * s_flat
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
    #  PHASE 1 — FORWARD PROFILING PASS                                   #
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
    )

    base_out        = Path(opts["output"])
    multi_seed      = len(seeds) > 1
    diverse_folder  = Path(ug_cfg.get("diverse_output_dir",  "outputs/diverse"))
    original_folder = Path(ug_cfg.get("original_output_dir", "outputs/original"))
    diverse_folder.mkdir(parents=True, exist_ok=True)
    original_folder.mkdir(parents=True, exist_ok=True)
    save_original   = ug_cfg.get("save_original", True)

    def _base_path(prompt_idx, seed):
        # Filename = prompt position (1-indexed), e.g. prompt 0 -> "1.png"
        return original_folder / (f"{prompt_idx + 1}" + base_out.suffix)

    def _branch_path(prompt_idx, seed, j):
        return diverse_folder / (f"{prompt_idx + 1}" + base_out.suffix)

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
            result  = sampler.run(latents, prompt_embeds, pooled_embeds, seed=seed)

            if save_original:
                
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
                    "theta"     : br["theta"],
                    "cos_x0"    : br["cos_x0"],
                    "cos_xN"    : br["cos_xN"],
                    "out_path"  : str(out_path),
                })

    return records