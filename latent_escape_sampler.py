"""
latent_escape_sampler.py
------------------------
U-Profile Guided Initial Latent Escape (UGILE)

Core idea (distinct from PeakBack / ONLB):
  Instead of perturbing a mid-trajectory point and resuming,
  we use the FULL cached U_k profile from one forward pass to compute
  a basin-escape direction in x_0 space, then run a complete new
  forward pass from the modified x_0.

  Because the flow field runs from the NEW x_0, it works WITH the model
  rather than fighting the model's error-correction. The entire trajectory
  diverges from the original, not just a suffix of it.

Algorithm:
  Phase 1 — Forward profiling pass (N model calls, same as PeakBack Phase 1)
             Cache: {x_k, v_k, v_cond_k, v_uncond_k, sigma_k, U_k}

  Phase 2 — Compute U-weighted escape direction in x_0 space
             For each step k in a selected subset S:
               g_k = grad_{x_k}[ U_k ]          (one backward pass per k)
             Aggregate:
               d = sum_{k in S} [ w_k * g_k ]
               w_k = U_k / sum_{k in S}(U_k)    (high-U steps dominate)
             Project d onto null space of (v_0, x_0) via joint_projector
             → escape direction stays on the latent sphere, orthogonal
               to the base velocity (preserves flow-matching geometry)

  Phase 3 — Geodesic move on the latent sphere
             x_0_new = geodesic_step(x_0, projected_d, r, theta_max)
             Different branches get a fixed per-branch lateral offset
             added to d before projection → inter-branch diversity

  Phase 4 — Full new forward pass from x_0_new
             All 50 Euler steps run fresh → flow field evolves the
             perturbation naturally → no washout problem

Slots into inference.py MODEL_REGISTRY as "sd3_ugile".
Add to config.yaml:
  model_name: "sd3_ugile"
  model_id:   "stabilityai/stable-diffusion-3-medium-diffusers"

  ugile:
    num_grad_steps:   5        # how many cached steps to use for gradient
    sigma_lo:         0.3      # band for selecting gradient steps
    sigma_hi:         0.9
    escape_scale:     3.0      # multiplier on the projected escape direction
    theta_max:        0.8      # max geodesic angle per branch
    branch_noise:     0.3      # lateral offset scale for inter-branch diversity
    J:                3        # number of diverse branches per seed
    diverse_output_dir:  "outputs/diverse"
    original_output_dir: "outputs/original"
    save_original:    true
"""

import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

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
        theta_max       : float = 0.8,
        walk_steps      : int   = 10,
        J               : int   = 1,
        eps             : float = 1e-8,
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

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      50)
        self.guidance_scale = f_cfg.get("guidance_scale", 7.5)
        self.do_cfg         = self.guidance_scale > 1.0

        print(f"[UGILE] band=[{sigma_lo},{sigma_hi}]  num_grad_steps={num_grad_steps}  "
              f"escape_scale={escape_scale}  theta_max={theta_max}  "
              f"walk_steps={walk_steps}  J={J}")

    # ================================================================== #
    #  PUBLIC ENTRY                                                        #
    # ================================================================== #

    def run(
        self,
        x0                : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor] = None,
        seed              : int = 0,
    ) -> Dict[str, Any]:

        # Phase 1 — forward profiling pass
        print("\n[UGILE] ── Phase 1: Forward profiling pass ──")
        cache = self._forward_pass_with_profiling(x0, text_embeddings, pooled_embeddings)
        print(f"[UGILE]   U_k range: min={min(cache['U']):.4f}  max={max(cache['U']):.4f}")
        print(f"[UGILE]   base x_N  norm={cache['x_N'].norm():.4f}")

        # Phase 2 — compute semantic direction from cached velocity field
        # v_cond - v_uncond averaged across all steps = what the model attends to
        # Moving x_0 orthogonal to this = change appearance without destroying content
        print("\n[UGILE] ── Phase 2: Computing semantic direction ──")
        N = cache["N"]
        semantic_dir = torch.zeros_like(x0, dtype=torch.float32)
        U_total = sum(cache["U"]) + self.eps
        for k in range(N):
            w_k = cache["U"][k] / U_total
            diff = (cache["v_cond"][k] - cache["v_uncond"][k]).to(semantic_dir.device)
            semantic_dir += w_k * diff.float()
        semantic_unit = semantic_dir / (semantic_dir.norm() + self.eps)
        print(f"[UGILE]   semantic_dir norm={semantic_dir.norm():.6f}")

        # Phase 3 — build x_0_new orthogonal to semantic direction
        # Use a seeded random vector, remove semantic component, normalize to same r
        print("\n[UGILE] ── Phase 3: Building escaped x_0 ──")
        r = x0.float().norm().item()
        rng = torch.Generator(device=x0.device)
        rng.manual_seed(seed * 10000 + 1)
        noise = torch.randn(x0.shape, generator=rng,
                            dtype=torch.float32, device=x0.device)
        # remove semantic component from noise
        noise = noise - (noise.reshape(-1) @ semantic_unit.reshape(-1)) * semantic_unit
        noise = noise / (noise.norm() + self.eps)

        # blend: x0_new = cos(theta)*x0 + sin(theta)*r*noise
        # this is a geodesic step of exactly theta_max radians
        import math
        theta = self.theta_max
        x0_new = (math.cos(theta) * x0.float() + math.sin(theta) * r * noise)
        x0_new = x0_new.to(x0.dtype)

        cos_x0 = torch.nn.functional.cosine_similarity(
            x0_new.reshape(1, -1).float(), x0.float().reshape(1, -1)
        ).item()
        print(f"[UGILE]   cos(x0_new, x0) = {cos_x0:.4f}  (expected {math.cos(theta):.4f})")

        # Phase 4 — full forward pass from x_0_new
        print("\n[UGILE] ── Phase 4: Full forward pass from escaped x_0 ──")
        x_N_diverse = self._full_forward_pass(x0_new, text_embeddings, pooled_embeddings)

        cos_xN = torch.nn.functional.cosine_similarity(
            x_N_diverse.reshape(1, -1).float(),
            cache["x_N"].reshape(1, -1).float()
        ).item()
        print(f"[UGILE]   cos(x_N_diverse, x_N_base) = {cos_xN:.4f}")

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

            if (k + 1) % 10 == 0 or k == 0:
                print(f"  [Phase1] step {k+1:>3}/{N} | sigma={cached_sigma[k]:.4f} | "
                      f"U_k={cached_U[k]:.4f} | mean={x.mean():.4f} | std={x.std():.4f}")

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
        """
        For each selected step k, compute grad_{x_k}[U_k].
        Approximate grad_{x_0}[U_k] ≈ grad_{x_k}[U_k]
        (Euler Jacobian dx_0/dx_k ≈ identity for small per-step dt).
        Aggregate with U_k weights.
        """
        N = cache["N"]

        # Select steps in sigma band, evenly spaced up to num_grad_steps
        band = [k for k in range(N)
                if self.sigma_lo <= cache["sigma"][k] <= self.sigma_hi]
        if not band:
            print("[UGILE]   WARNING: no steps in sigma band, using full trajectory")
            band = list(range(N))

        # Use ALL steps in the band
        selected = band
        print(f"[UGILE]   gradient steps selected: {len(selected)} steps  "
              f"(sigma range {cache['sigma'][selected[0]]:.3f}–"
              f"{cache['sigma'][selected[-1]]:.3f})")

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
            print(f"  [Phase2] step k={k:>3} | U_k={cache['U'][k]:.4f} | "
                  f"w={w_k:.4f} | grad_norm={grad.norm():.4f}")

        return escape_dir

    def _grad_U_at_xk(self, x_k, t, sigma, text_embeddings, pooled_embeddings,
                      eps_smooth=1e-6):
        """grad_{x_k}[ U_k ] via one backward pass. U_k = sigma * ||v_c - v_u||."""
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
        """Run only the first n_steps to cheaply check if x0_new is on-manifold."""
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
        """Run all N Euler steps from the new initial latent."""
        self.scheduler.set_timesteps(self.num_steps)
        timesteps = self.scheduler.timesteps
        N = len(timesteps)
        x = x0_new.clone()

        for k, t in enumerate(timesteps):
            v = self._velocity_forward(x, t, text_embeddings, pooled_embeddings)
            x = self.scheduler.step(v, t, x).prev_sample

            if (k + 1) % 10 == 0 or k == 0:
                print(f"  [Phase4] step {k+1:>3}/{N} | "
                      f"mean={x.mean():.4f} | std={x.std():.4f}")

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
    """
    Drop-in runner for inference.py MODEL_REGISTRY.
    Set model_name: "sd3_ugile" in config.yaml to activate.

    For each seed: one profiling pass → escape direction → J full forward
    passes from J different modified x_0s. The model is loaded once.
    """
    from pipeline_wrapper import SD3PipelineWrapper

    cfg    = opts.get("_cfg", {})
    device = opts["device"]
    seeds  = opts.get("seeds") or [opts["seed"]]

    ug_cfg = cfg.get("ugile", {})

    print("\n" + "═" * 60)
    print("[UGILE] Loading model (once for all seeds)…")
    print("═" * 60)
    wrapper = SD3PipelineWrapper(cfg, device=device)
    wrapper.load()

    print("\n[UGILE] Encoding prompt (shared across all seeds)…")
    prompt_embeds, pooled_embeds = wrapper.encode_prompt(
        opts["prompt"], opts["negative_prompt"]
    )

    sampler = UGILESampler(
        unet            = wrapper.transformer,
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
    )

    base_out        = Path(opts["output"])
    multi_seed      = len(seeds) > 1
    diverse_folder  = Path(ug_cfg.get("diverse_output_dir",  "outputs/diverse"))
    original_folder = Path(ug_cfg.get("original_output_dir", "outputs/original"))
    diverse_folder.mkdir(parents=True, exist_ok=True)
    original_folder.mkdir(parents=True, exist_ok=True)
    save_original   = ug_cfg.get("save_original", True)

    def _base_path(seed):
        stem = base_out.stem + (f"_seed{seed}" if multi_seed else "")
        return original_folder / (stem + "_base" + base_out.suffix)

    def _branch_path(seed, j):
        stem = base_out.stem + (f"_seed{seed}" if multi_seed else "")
        return diverse_folder / (stem + f"_branch{j}" + base_out.suffix)

    records = []
    for idx, seed in enumerate(seeds):
        print("\n" + "═" * 60)
        print(f"[UGILE] Seed {seed}  ({idx + 1}/{len(seeds)})")
        print("═" * 60)

        latents = wrapper.get_initial_latents(seed=seed)
        result  = sampler.run(latents, prompt_embeds, pooled_embeds, seed=seed)

        if save_original:
            base_path = _base_path(seed)
            wrapper.decode_latents(result["original_latents"]).save(base_path)
            print(f"[UGILE] Base image → {base_path}")

        for br in result["branches"]:
            out_path = _branch_path(seed, br["branch_idx"])
            wrapper.decode_latents(br["latents"]).save(out_path)
            print(f"[UGILE] Branch {br['branch_idx']} → {out_path}  "
                  f"(theta={br['theta']:.3f}, cos_x0={br['cos_x0']:.4f}, "
                  f"cos_xN={br['cos_xN']:.4f})")

            records.append({
                "seed"      : seed,
                "branch"    : br["branch_idx"],
                "theta"     : br["theta"],
                "cos_x0"    : br["cos_x0"],
                "cos_xN"    : br["cos_xN"],
                "out_path"  : str(out_path),
            })

    print("\n" + "═" * 60)
    print(f"[UGILE] ── Summary ({len(seeds)} seed(s)) ──")
    print("═" * 60)
    header = (f"{'Seed':>6} | {'Br':>3} | {'theta':>6} | "
              f"{'cos_x0':>7} | {'cos_xN':>7} | Output")
    print(header)
    print("-" * len(header))
    for r in records:
        print(f"{r['seed']:>6} | {r['branch']:>3} | {r['theta']:>6.3f} | "
              f"{r['cos_x0']:>7.4f} | {r['cos_xN']:>7.4f} | {r['out_path']}")
    print("═" * 60)
    print("[UGILE] cos_x0: angular distance between original and escaped x_0")
    print("[UGILE] cos_xN: visual diversity between base and branch final latent")
    print("[UGILE] Target cos_xN range: 0.5–0.85 for meaningful visual diversity")