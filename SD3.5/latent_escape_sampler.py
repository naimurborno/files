# Md. Naimur Asif Borno
#
# ---------------------------------------------------------------------------
# Thermodynamic Dispersion Sampler (TDS)
#
# Replaces the UGILE-specific mechanism previously in this file. TDS treats
# N copies of one initial latent as an interacting gas: local crowding
# creates a pressure field that pushes particles apart, AND modulates each
# particle's own stochastic temperature (crowded particles get noisier,
# isolated particles calm down). No pairwise force is hand-designed — both
# the drift and the noise emerge from a single local KDE density estimate.
#
#   rho_i(t) = sum_j K_h(x_i - x_j)                        KDE, Gaussian K_h
#   P(x,t)   = rho(x,t)^gamma                              equation of state
#   D_i(t)   = D0 * [1 + alpha * (rho_i/rho_star)^beta]    density-modulated
#                                                           temperature
#   D0       = eta * T                                     Einstein relation
#
#   x_i <- x_i + dt*v_theta(x_i,t) - dt*eta*grad_P(x_i) + sqrt(2*D_i*dt)*xi_i
#
# active only for step index k in [window_start_idx, window_end_idx).
#
# Fixed by theory (NOT swept — see method section):
#   gamma = 1.0   ideal-gas / log-density equation of state
#   beta  = 1.0   linear crowding -> temperature coupling
#   eta   = 1.0   mobility; folded into T via D0 = eta * T
#   rho_star: NOT a hyperparameter — computed automatically as the mean
#             density at the first active step of each run.
#
# Actually-tunable knobs: num_particles, kde_bandwidth (None = auto
# SVGD-style median heuristic), temperature T, agitation_gain alpha,
# window_start_frac / window_end_frac, symmetry_break_std, max_step_frac.
# ---------------------------------------------------------------------------
import math
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_prompts(path: str) -> List[str]:
    """Load ONLY the prompts list from a separate prompts yaml file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    prompts = data.get("prompts", [])
    if not prompts:
        raise ValueError(f"No 'prompts' key found in {path}")
    return prompts


class TDSSampler:
    """Thermodynamic Dispersion Sampler — batched N-particle diversity sampler."""

    def __init__(
        self,
        unet,
        scheduler,
        cfg                : dict,
        device             : str   = "cuda",
        num_particles      : int   = 5,
        kde_bandwidth      : Optional[float] = None,   # h; None -> auto median heuristic
        pressure_exponent  : float = 1.0,   # gamma — fixed by theory, exposed only for ablation
        agitation_exponent : float = 1.0,   # beta  — fixed by theory, exposed only for ablation
        mobility           : float = 1.0,   # eta
        temperature        : float = 0.5,   # T ; D0 = eta * T (Einstein relation) — MAIN knob
        agitation_gain     : float = 2.0,   # alpha — crowding->noise coupling  — MAIN knob
        symmetry_break_std : float = 0.02,  # epsilon_init, per-element std (latents are raw N(0,1))
        window_start_frac  : float = 0.0,   # TDS active from this fraction of the trajectory...
        window_end_frac    : float = 0.4,   # ...through this fraction. Early steps = high noise
                                             # sigma = mode-level diversity; matches UGILE's own
                                             # early-window finding.
        max_step_frac      : float = 0.05,  # safety clamp: per-step nudge <= this * ||x_i||
        eps                : float = 1e-8,
    ):
        self.unet               = unet
        self.scheduler          = scheduler
        self.cfg                = cfg
        self.device             = device
        self.num_particles      = num_particles
        self.kde_bandwidth      = kde_bandwidth
        self.pressure_exponent  = pressure_exponent
        self.agitation_exponent = agitation_exponent
        self.mobility           = mobility
        self.temperature        = temperature
        self.agitation_gain     = agitation_gain
        self.symmetry_break_std = symmetry_break_std
        self.window_start_frac  = window_start_frac
        self.window_end_frac    = window_end_frac
        self.max_step_frac      = max_step_frac
        self.eps                = eps

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      28)
        self.guidance_scale = f_cfg.get("guidance_scale", 7.0)
        self.do_cfg         = self.guidance_scale > 1.0

    # ================================================================== #
    #  DENSITY / PRESSURE                                                  #
    # ================================================================== #

    def _median_bandwidth(self, x_flat: torch.Tensor) -> float:
        """
        SVGD-style median heuristic: h = median(||x_i - x_j||^2) / log(N).
        Computed fresh every active step from the CURRENT particle
        positions — this is what lets sigma auto-adapt as particles spread
        apart, instead of needing a hand-tuned bandwidth per resolution.
        """
        N = x_flat.shape[0]
        if N < 2:
            return 1.0
        sqdist = torch.cdist(x_flat, x_flat) ** 2
        iu = torch.triu_indices(N, N, offset=1, device=x_flat.device)
        med = sqdist[iu[0], iu[1]].median()
        h = med / max(math.log(N), self.eps)
        return float(h.clamp_min(self.eps).item())

    def _density_and_pressure_grad(self, x_flat: torch.Tensor):
        """
        x_flat: (N, D) flattened particle positions.
        Returns:
          S      : (N,)   unnormalized local density, S_i = sum_j K(x_i,x_j)
          grad_P : (N, D) gradient of P = S^gamma wrt each particle's OWN
                   position (i.e. d P_i / d x_i, not a joint gradient)
          h      : bandwidth actually used this step
        """
        N = x_flat.shape[0]
        h = self.kde_bandwidth if self.kde_bandwidth is not None else self._median_bandwidth(x_flat)

        diff   = x_flat.unsqueeze(1) - x_flat.unsqueeze(0)   # diff[i,j] = x_i - x_j,  (N,N,D)
        sqdist = (diff ** 2).sum(-1)                          # (N,N)
        K = torch.exp(-sqdist / h)                             # (N,N), K_ii = 1

        S = K.sum(dim=1)                                       # (N,)

        # d/dx_i sum_j K_ij = (2/h) * sum_j K_ij * (x_j - x_i)
        # — points TOWARD the crowd (ascent direction of local density).
        grad_S = (2.0 / h) * (K.unsqueeze(-1) * (-diff)).sum(dim=1)   # (N, D)

        # P = S^gamma  =>  grad_P = gamma * S^(gamma-1) * grad_S
        grad_P = (
            self.pressure_exponent
            * (S.clamp_min(self.eps) ** (self.pressure_exponent - 1.0)).unsqueeze(-1)
            * grad_S
        )
        return S, grad_P, h

    # ================================================================== #
    #  ONE IN-WINDOW TDS UPDATE                                            #
    # ================================================================== #

    def _tds_step(self, z: torch.Tensor, k: int, seed: int, diag: dict) -> torch.Tensor:
        """
        z: (N, C, H, W) — the full particle batch. Returns the updated batch.
        Applies: x_i <- x_i - dt*eta*grad_P(x_i) + sqrt(2*D_i*dt)*xi_i
        """
        N = z.shape[0]
        z_f = z.float()
        x_flat = z_f.reshape(N, -1)

        S, grad_P_flat, h_used = self._density_and_pressure_grad(x_flat)

        # rho_star is NOT a hyperparameter — set once, from the first
        # active step's own mean density, then held fixed for the rest
        # of this run's window.
        if diag["rho_star"] is None:
            diag["rho_star"] = S.mean().item()
        rho_star = max(diag["rho_star"], self.eps)

        D0 = self.mobility * self.temperature   # Einstein / fluctuation-dissipation relation
        D_i = D0 * (
            1.0 + self.agitation_gain * (S.clamp_min(self.eps) / rho_star) ** self.agitation_exponent
        )  # (N,)

        # local effective dt from the scheduler's own sigma spacing at this step
        if hasattr(self.scheduler, "sigmas") and k + 1 < len(self.scheduler.sigmas):
            dt_local = abs(self.scheduler.sigmas[k].item() - self.scheduler.sigmas[k + 1].item())
        else:
            dt_local = 1.0 / max(diag["N"], 1)

        drift_flat = -self.mobility * grad_P_flat * dt_local

        rng = torch.Generator(device=z.device)
        rng.manual_seed(seed * 7_000_003 + k)
        xi = torch.randn(x_flat.shape, generator=rng, dtype=torch.float32, device=z.device)
        noise_flat = torch.sqrt(2.0 * D_i.clamp_min(0.0).unsqueeze(-1) * dt_local) * xi

        step_flat = drift_flat + noise_flat

        # safety clamp: per-particle nudge <= max_step_frac * ||x_i|| (same
        # spirit as UGILE's max_eps_frac clamp — keeps the SDE from ever
        # blowing the latent off-manifold in one step).
        r_i = x_flat.norm(dim=-1, keepdim=True) + self.eps
        step_norm = step_flat.norm(dim=-1, keepdim=True)
        cap = self.max_step_frac * r_i
        scale = torch.clamp(cap / (step_norm + self.eps), max=1.0)
        step_flat = step_flat * scale

        diag["trace"].append({
            "k": k, "h": h_used, "rho_star": rho_star,
            "S_mean": S.mean().item(), "S_std": S.std().item(),
            "D_mean": D_i.mean().item(),
            "cap_hit_rate": (scale < 1.0).float().mean().item(),
        })

        z_new = x_flat + step_flat
        return z_new.view_as(z).to(z.dtype)

    # ================================================================== #
    #  BATCHED CFG VELOCITY (N particles at once)                          #
    # ================================================================== #

    def _velocity_forward_batch(self, x, t, text_embeddings, pooled_embeddings):
        """
        x: (N, C, H, W). text_embeddings: (2, seq, dim) == cat([uncond, cond])
        for ONE prompt (as produced by SD3PipelineWrapper.encode_prompt).
        Returns v_cfg: (N, C, H, W).

        NOTE: this runs a (2N, ...) batch through the transformer per call —
        N is limited by T4 memory, same constraint UGILE's branch count J
        already had, just applied to a jointly-coupled batch instead of
        independent branches.
        """
        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype
        N = x.shape[0]
        x = x.to(device=device, dtype=dtype)

        t_val = t.item() if hasattr(t, "item") else float(t)

        if self.do_cfg:
            uncond_emb, cond_emb = text_embeddings.chunk(2, dim=0)
            uncond_emb = uncond_emb.to(device=device, dtype=dtype).repeat(N, 1, 1)
            cond_emb   = cond_emb.to(device=device, dtype=dtype).repeat(N, 1, 1)
            text_batch   = torch.cat([uncond_emb, cond_emb], dim=0)   # (2N, seq, dim)
            latent_input = torch.cat([x, x], dim=0)                    # (2N, C, H, W)

            pooled_batch = None
            if pooled_embeddings is not None:
                p_uncond, p_cond = pooled_embeddings.chunk(2, dim=0)
                p_uncond = p_uncond.to(device=device, dtype=dtype).repeat(N, 1)
                p_cond   = p_cond.to(device=device, dtype=dtype).repeat(N, 1)
                pooled_batch = torch.cat([p_uncond, p_cond], dim=0)     # (2N, dim)
        else:
            text_batch   = text_embeddings.to(device=device, dtype=dtype).repeat(N, 1, 1)
            latent_input = x
            pooled_batch = (
                pooled_embeddings.to(device=device, dtype=dtype).repeat(N, 1)
                if pooled_embeddings is not None else None
            )

        t_batch = torch.tensor([t_val] * latent_input.shape[0], device=device, dtype=dtype)

        kwargs = dict(
            hidden_states         = latent_input,
            timestep              = t_batch,
            encoder_hidden_states = text_batch,
        )
        if pooled_batch is not None:
            kwargs["pooled_projections"] = pooled_batch

        with torch.no_grad():
            output = self.unet(**kwargs).sample

        if self.do_cfg:
            v_uncond, v_cond = output.chunk(2, dim=0)
            v_cfg = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            return v_cfg
        return output

    # ================================================================== #
    #  RUN                                                                 #
    # ================================================================== #

    def run(
        self,
        x0_single         : torch.Tensor,
        text_embeddings   : torch.Tensor,
        pooled_embeddings : Optional[torch.Tensor] = None,
        seed              : int = 0,
    ) -> Dict[str, Any]:
        """
        x0_single: (1, C, H, W) — ONE reference latent. Expanded into
        num_particles identical copies, then a small symmetry-breaking
        perturbation is added (otherwise rho is uniform everywhere and
        grad_P == 0 for every particle, forever).

        Returns {"particles": [ (1,C,H,W) x num_particles ], "diag": {...}}.
        """
        self.scheduler.set_timesteps(self.num_steps)
        timesteps = self.scheduler.timesteps
        N_steps = len(timesteps)

        window_start_idx = int(round(self.window_start_frac * N_steps))
        window_end_idx   = int(round(self.window_end_frac   * N_steps))
        window_end_idx   = max(window_end_idx, window_start_idx + 1)

        rng0 = torch.Generator(device=x0_single.device)
        rng0.manual_seed(seed * 9_000_001)
        z = x0_single.repeat(self.num_particles, 1, 1, 1).clone()
        z = z + self.symmetry_break_std * torch.randn(
            z.shape, generator=rng0, dtype=torch.float32, device=z.device
        ).to(z.dtype)

        diag = {"N": N_steps, "window_start_idx": window_start_idx,
                "window_end_idx": window_end_idx, "rho_star": None, "trace": []}

        for k, t in enumerate(timesteps):
            v_cfg = self._velocity_forward_batch(z, t, text_embeddings, pooled_embeddings)

            if window_start_idx <= k < window_end_idx:
                z = self._tds_step(z, k, seed, diag)

            z = self.scheduler.step(v_cfg, t, z).prev_sample

        return {
            "particles": [z[i:i + 1] for i in range(self.num_particles)],
            "diag": diag,
        }


# ══════════════════════════════════════════════════════════════════════ #
#  DROP-IN RUNNER                                                         #
# ══════════════════════════════════════════════════════════════════════ #

def _dispersion_stats(particles: List[torch.Tensor]) -> Dict[str, float]:
    """
    Quantify how far apart the final particle latents actually ended up.
    Returns mean/min/max pairwise L2 distance, plus that distance relative
    to the particles' own norm (scale-free — comparable across resolutions).
    """
    flat = torch.cat([p.reshape(1, -1).float() for p in particles], dim=0)  # (N, D)
    N = flat.shape[0]
    if N < 2:
        return {"mean_pairwise_dist": 0.0, "min_pairwise_dist": 0.0,
                "max_pairwise_dist": 0.0, "mean_relative_dist": 0.0}
    d = torch.cdist(flat, flat)                      # (N, N)
    iu = torch.triu_indices(N, N, offset=1)
    dists = d[iu[0], iu[1]]
    mean_norm = flat.norm(dim=-1).mean().clamp_min(1e-8)
    return {
        "mean_pairwise_dist":   dists.mean().item(),
        "min_pairwise_dist":    dists.min().item(),
        "max_pairwise_dist":    dists.max().item(),
        "mean_relative_dist":   (dists.mean() / mean_norm).item(),  # dist as a fraction of ||x||
    }


def run_sd3_tds(opts: dict):
    from pipeline_wrapper import SD3PipelineWrapper

    cfg    = opts.get("_cfg", {})
    device = opts["device"]
    seeds  = opts.get("seeds") or [opts["seed"]]

    prompts_file = cfg.get("prompts_file", "prompts.yaml")
    prompts = load_prompts(prompts_file)

    tds_cfg = cfg.get("tds", {})

    wrapper = SD3PipelineWrapper(cfg, device=device)
    wrapper.load()

    sampler = TDSSampler(
        unet      = wrapper.transformer,
        scheduler = wrapper.scheduler,
        cfg       = cfg,
        device    = device,
        num_particles      = tds_cfg.get("num_particles",      5),
        kde_bandwidth      = tds_cfg.get("kde_bandwidth",      None),
        pressure_exponent  = tds_cfg.get("pressure_exponent",  1.0),
        agitation_exponent = tds_cfg.get("agitation_exponent", 1.0),
        mobility           = tds_cfg.get("mobility",           1.0),
        temperature        = tds_cfg.get("temperature",        0.5),
        agitation_gain     = tds_cfg.get("agitation_gain",     2.0),
        symmetry_break_std = tds_cfg.get("symmetry_break_std", 0.02),
        window_start_frac  = tds_cfg.get("window_start_frac",  0.0),
        window_end_frac    = tds_cfg.get("window_end_frac",    0.4),
        max_step_frac      = tds_cfg.get("max_step_frac",      0.05),
    )

    base_out       = Path(opts["output"])
    diverse_folder = Path(tds_cfg.get("diverse_output_dir", "outputs/diverse_tds"))
    diverse_folder.mkdir(parents=True, exist_ok=True)

    records = []
    diag_summaries = []
    prompt_offset = cfg.get("prompt_offset", 0)

    for p_idx, prompt in enumerate(prompts):
        global_idx = p_idx + prompt_offset

        prompt_embeds, pooled_embeds = wrapper.encode_prompt(prompt, opts["negative_prompt"])

        for seed in seeds:
            print(f"[TDS] prompt {global_idx} seed={seed}  N={sampler.num_particles}")

            x0 = wrapper.get_initial_latents(seed=seed)
            result = sampler.run(x0, prompt_embeds, pooled_embeds, seed=seed)

            disp = _dispersion_stats(result["particles"])
            trace = result["diag"]["trace"]
            avg_cap_hit = (
                sum(t["cap_hit_rate"] for t in trace) / len(trace) if trace else None
            )
            print(f"  [TDS-diag] mean_pairwise_dist={disp['mean_pairwise_dist']:.4f} "
                  f"(relative={disp['mean_relative_dist']:.4%})  "
                  f"avg_cap_hit_rate={avg_cap_hit if avg_cap_hit is None else f'{avg_cap_hit:.2%}'}  "
                  f"rho_star={result['diag']['rho_star']}")

            diag_summaries.append({
                "prompt_idx": global_idx, "seed": seed,
                "dispersion": disp,
                "avg_cap_hit_rate": avg_cap_hit,
                "rho_star": result["diag"]["rho_star"],
                "trace": trace,   # full per-step S/D/cap history for this seed
            })

            for i, particle_latents in enumerate(result["particles"]):
                out_path = diverse_folder / f"{global_idx + 1}_seed{seed}_particle{i}{base_out.suffix}"
                wrapper.decode_latents(particle_latents).save(out_path)
                records.append({
                    "prompt_idx": global_idx,
                    "prompt"    : prompt,
                    "seed"      : seed,
                    "particle"  : i,
                    "out_path"  : str(out_path),
                })

    meta = {
        "sampler_params": {
            "num_particles": sampler.num_particles,
            "kde_bandwidth": sampler.kde_bandwidth,
            "pressure_exponent": sampler.pressure_exponent,
            "agitation_exponent": sampler.agitation_exponent,
            "mobility": sampler.mobility,
            "temperature": sampler.temperature,
            "agitation_gain": sampler.agitation_gain,
            "symmetry_break_std": sampler.symmetry_break_std,
            "window_start_frac": sampler.window_start_frac,
            "window_end_frac": sampler.window_end_frac,
            "max_step_frac": sampler.max_step_frac,
            "num_steps": sampler.num_steps,
            "guidance_scale": sampler.guidance_scale,
        },
        "seeds": list(seeds), "n_prompts": len(prompts), "prompt_offset": prompt_offset,
        "records": records,
        "diag_summaries": diag_summaries,
    }
    with open(diverse_folder / f"records_offset{prompt_offset}.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)

    return records