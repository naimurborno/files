# control_samplers.py
# ---------------------------------------------------------------------------
# Ablation controls for TDS (see latent_escape_sampler.py). Both reuse the
# EXACT same batched-CFG forward pass and KDE/pressure-gradient math as
# TDSSampler -- so the noise mechanism is the only thing that differs
# between these and TDS. That's what makes the comparison decisive rather
# than "different sampler, different everything."
#
#   PGSampler
#       Pure gradient repulsion, deterministic:
#           x_i <- x_i - dt*eta*grad_P(x_i)
#       No stochastic term at all. This is the classic Particle-Guidance-
#       style baseline. It CAN get stuck at force-balance configurations
#       (e.g. a particle sitting symmetrically between two others, where
#       grad_P == 0 even though all three are still redundant) -- the
#       exact failure mode TDS's noise term is supposed to escape.
#
#   MatchedNoisePGSampler
#       PG drift + isotropic stochastic term, but the noise magnitude D(t)
#       is a TIME-ONLY schedule -- the SAME scalar for every particle at a
#       given step k, regardless of that particle's own local crowding --
#       instead of TDS's per-particle, density-coupled D_i(rho_i).
#       Calibrated via build_matched_schedule_from_tds_records() so the
#       TOTAL injected noise mass (integral of D over the active window)
#       matches a real TDS run step-for-step. The only difference from TDS
#       is WHERE that noise mass goes: blind to density vs. coupled to it.
#
# Decisive reads:
#   TDS  vs  MatchedNoisePG  -> isolates whether density-coupling itself
#                                is doing real work (the paper's actual bet)
#   TDS / MatchedNoisePG  vs  PG -> isolates whether noise helps AT ALL
# ---------------------------------------------------------------------------

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from latent_escape_sampler import load_prompts, _dispersion_stats


# ============================================================== #
#  Shared KDE / pressure-gradient math (identical to TDS's)       #
# ============================================================== #

def _median_bandwidth(x_flat: torch.Tensor, eps: float = 1e-8) -> float:
    """SVGD-style median heuristic, identical to TDSSampler's."""
    N = x_flat.shape[0]
    if N < 2:
        return 1.0
    sqdist = torch.cdist(x_flat, x_flat) ** 2
    iu = torch.triu_indices(N, N, offset=1, device=x_flat.device)
    med = sqdist[iu[0], iu[1]].median()
    h = med / max(math.log(N), eps)
    return float(h.clamp_min(eps).item())


def _density_and_pressure_grad(
    x_flat: torch.Tensor, kde_bandwidth: Optional[float],
    pressure_exponent: float, eps: float = 1e-8,
):
    """Identical formula to TDSSampler._density_and_pressure_grad."""
    N = x_flat.shape[0]
    h = kde_bandwidth if kde_bandwidth is not None else _median_bandwidth(x_flat, eps)

    diff   = x_flat.unsqueeze(1) - x_flat.unsqueeze(0)
    sqdist = (diff ** 2).sum(-1)
    K = torch.exp(-sqdist / h)
    S = K.sum(dim=1)

    grad_S = (2.0 / h) * (K.unsqueeze(-1) * (-diff)).sum(dim=1)
    grad_P = (
        pressure_exponent
        * (S.clamp_min(eps) ** (pressure_exponent - 1.0)).unsqueeze(-1)
        * grad_S
    )
    return S, grad_P, h


# ============================================================== #
#  Batched CFG velocity forward (identical to TDS's)               #
# ============================================================== #

def _velocity_forward_batch(unet, x, t, text_embeddings, pooled_embeddings,
                             do_cfg: bool, guidance_scale: float):
    device = next(unet.parameters()).device
    dtype  = next(unet.parameters()).dtype
    N = x.shape[0]
    x = x.to(device=device, dtype=dtype)

    t_val = t.item() if hasattr(t, "item") else float(t)

    if do_cfg:
        uncond_emb, cond_emb = text_embeddings.chunk(2, dim=0)
        uncond_emb = uncond_emb.to(device=device, dtype=dtype).repeat(N, 1, 1)
        cond_emb   = cond_emb.to(device=device, dtype=dtype).repeat(N, 1, 1)
        text_batch   = torch.cat([uncond_emb, cond_emb], dim=0)
        latent_input = torch.cat([x, x], dim=0)

        pooled_batch = None
        if pooled_embeddings is not None:
            p_uncond, p_cond = pooled_embeddings.chunk(2, dim=0)
            p_uncond = p_uncond.to(device=device, dtype=dtype).repeat(N, 1)
            p_cond   = p_cond.to(device=device, dtype=dtype).repeat(N, 1)
            pooled_batch = torch.cat([p_uncond, p_cond], dim=0)
    else:
        text_batch   = text_embeddings.to(device=device, dtype=dtype).repeat(N, 1, 1)
        latent_input = x
        pooled_batch = (
            pooled_embeddings.to(device=device, dtype=dtype).repeat(N, 1)
            if pooled_embeddings is not None else None
        )

    t_batch = torch.tensor([t_val] * latent_input.shape[0], device=device, dtype=dtype)

    kwargs = dict(
        hidden_states          = latent_input,
        timestep                = t_batch,
        encoder_hidden_states  = text_batch,
    )
    if pooled_batch is not None:
        kwargs["pooled_projections"] = pooled_batch

    with torch.no_grad():
        output = unet(**kwargs).sample

    if do_cfg:
        v_uncond, v_cond = output.chunk(2, dim=0)
        return v_uncond + guidance_scale * (v_cond - v_uncond)
    return output


# ============================================================== #
#  PG SAMPLER  (pure gradient repulsion, no noise)                 #
# ============================================================== #

class PGSampler:
    """Deterministic gradient-repulsion baseline. Same KDE/pressure math,
    same window/clamp machinery as TDS. Zero stochastic term."""

    def __init__(
        self,
        unet, scheduler, cfg: dict, device: str = "cuda",
        num_particles      : int = 5,
        kde_bandwidth      : Optional[float] = None,
        pressure_exponent  : float = 1.0,
        mobility           : float = 1.0,
        symmetry_break_std : float = 0.02,
        window_start_frac  : float = 0.0,
        window_end_frac    : float = 0.4,
        max_step_frac      : float = 0.05,
        eps                : float = 1e-8,
    ):
        self.unet, self.scheduler, self.cfg, self.device = unet, scheduler, cfg, device
        self.num_particles      = num_particles
        self.kde_bandwidth      = kde_bandwidth
        self.pressure_exponent  = pressure_exponent
        self.mobility           = mobility
        self.symmetry_break_std = symmetry_break_std
        self.window_start_frac  = window_start_frac
        self.window_end_frac    = window_end_frac
        self.max_step_frac      = max_step_frac
        self.eps                = eps

        f_cfg = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps", 28)
        self.guidance_scale = f_cfg.get("guidance_scale", 7.0)
        self.do_cfg          = self.guidance_scale > 1.0

    def _pg_step(self, z: torch.Tensor, k: int, diag: dict) -> torch.Tensor:
        """x_i <- x_i - dt*eta*grad_P(x_i). No noise term at all."""
        N = z.shape[0]
        z_f = z.float()
        x_flat = z_f.reshape(N, -1)

        S, grad_P_flat, h_used = _density_and_pressure_grad(
            x_flat, self.kde_bandwidth, self.pressure_exponent, self.eps
        )

        if hasattr(self.scheduler, "sigmas") and k + 1 < len(self.scheduler.sigmas):
            dt_local = abs(self.scheduler.sigmas[k].item() - self.scheduler.sigmas[k + 1].item())
        else:
            dt_local = 1.0 / max(diag["N"], 1)

        step_flat = -self.mobility * grad_P_flat * dt_local   # deterministic only

        r_i = x_flat.norm(dim=-1, keepdim=True) + self.eps
        step_norm = step_flat.norm(dim=-1, keepdim=True)
        cap = self.max_step_frac * r_i
        scale = torch.clamp(cap / (step_norm + self.eps), max=1.0)
        step_flat = step_flat * scale

        diag["trace"].append({
            "k": k, "h": h_used,
            "S_mean": S.mean().item(), "S_std": S.std().item(),
            "cap_hit_rate": (scale < 1.0).float().mean().item(),
        })

        z_new = x_flat + step_flat
        return z_new.view_as(z).to(z.dtype)

    def run(self, x0_single, text_embeddings, pooled_embeddings=None, seed: int = 0) -> Dict[str, Any]:
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
                "window_end_idx": window_end_idx, "trace": []}

        for k, t in enumerate(timesteps):
            v_cfg = _velocity_forward_batch(self.unet, z, t, text_embeddings, pooled_embeddings,
                                             self.do_cfg, self.guidance_scale)
            if window_start_idx <= k < window_end_idx:
                z = self._pg_step(z, k, diag)
            z = self.scheduler.step(v_cfg, t, z).prev_sample

        return {"particles": [z[i:i + 1] for i in range(self.num_particles)], "diag": diag}


# ============================================================== #
#  MATCHED-NOISE-MASS PG  (PG drift + time-only noise schedule)    #
# ============================================================== #

class MatchedNoisePGSampler(PGSampler):
    """PG drift PLUS an isotropic noise term whose magnitude D(t) is a
    fixed, time-only schedule (identical for every particle at a given
    step) -- instead of TDS's per-particle density-coupled D_i(rho_i).
    Pass `noise_schedule` from build_matched_schedule_from_tds_records()
    so the total injected noise mass matches a real TDS run."""

    def __init__(self, *args, noise_schedule: Optional[Dict[int, float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_schedule = noise_schedule or {}

    def _matched_step(self, z: torch.Tensor, k: int, seed: int, diag: dict) -> torch.Tensor:
        N = z.shape[0]
        z_f = z.float()
        x_flat = z_f.reshape(N, -1)

        S, grad_P_flat, h_used = _density_and_pressure_grad(
            x_flat, self.kde_bandwidth, self.pressure_exponent, self.eps
        )

        if hasattr(self.scheduler, "sigmas") and k + 1 < len(self.scheduler.sigmas):
            dt_local = abs(self.scheduler.sigmas[k].item() - self.scheduler.sigmas[k + 1].item())
        else:
            dt_local = 1.0 / max(diag["N"], 1)

        drift_flat = -self.mobility * grad_P_flat * dt_local

        D_t = self.noise_schedule.get(k, 0.0)   # SAME scalar for every particle at step k
        rng = torch.Generator(device=z.device)
        rng.manual_seed(seed * 7_000_003 + k)
        xi = torch.randn(x_flat.shape, generator=rng, dtype=torch.float32, device=z.device)
        noise_flat = math.sqrt(max(2.0 * D_t * dt_local, 0.0)) * xi

        step_flat = drift_flat + noise_flat

        r_i = x_flat.norm(dim=-1, keepdim=True) + self.eps
        step_norm = step_flat.norm(dim=-1, keepdim=True)
        cap = self.max_step_frac * r_i
        scale = torch.clamp(cap / (step_norm + self.eps), max=1.0)
        step_flat = step_flat * scale

        diag["trace"].append({
            "k": k, "h": h_used, "D_t": D_t,
            "S_mean": S.mean().item(), "S_std": S.std().item(),
            "cap_hit_rate": (scale < 1.0).float().mean().item(),
        })

        z_new = x_flat + step_flat
        return z_new.view_as(z).to(z.dtype)

    def run(self, x0_single, text_embeddings, pooled_embeddings=None, seed: int = 0) -> Dict[str, Any]:
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
                "window_end_idx": window_end_idx, "trace": []}

        for k, t in enumerate(timesteps):
            v_cfg = _velocity_forward_batch(self.unet, z, t, text_embeddings, pooled_embeddings,
                                             self.do_cfg, self.guidance_scale)
            if window_start_idx <= k < window_end_idx:
                z = self._matched_step(z, k, seed, diag)
            z = self.scheduler.step(v_cfg, t, z).prev_sample

        return {"particles": [z[i:i + 1] for i in range(self.num_particles)], "diag": diag}


# ============================================================== #
#  Calibration: build a time-only noise schedule from a TDS run    #
# ============================================================== #

def build_matched_schedule_from_tds_records(
    records_path: str,
    prompt_idx: Optional[int] = None,
    seed: Optional[int] = None,
    average_over_all: bool = True,
) -> Dict[int, float]:
    """
    Read a records_offsetN.json produced by run_sd3_tds (must include
    diag_summaries[i].trace -- the diagnostic-enabled latent_escape_sampler.py)
    and build a step-indexed D(t) schedule carrying the SAME total noise
    mass TDS actually injected.

    average_over_all=True (default): average D_mean at each step k across
    EVERY (prompt, seed) entry in the file -- the most robust single
    schedule for a fair aggregate comparison. Set False and pass
    prompt_idx/seed to match one specific TDS run exactly instead.
    """
    with open(records_path) as f:
        meta = json.load(f)
    summaries = meta.get("diag_summaries", [])
    if not summaries:
        raise ValueError(
            f"{records_path} has no diag_summaries -- rerun sd3_tds with the "
            f"diagnostic-enabled latent_escape_sampler.py first."
        )

    if not average_over_all:
        for s in summaries:
            if (prompt_idx is None or s["prompt_idx"] == prompt_idx) and \
               (seed is None or s["seed"] == seed):
                return {t["k"]: t["D_mean"] for t in s["trace"]}
        raise ValueError(f"No matching diag_summary for prompt_idx={prompt_idx}, seed={seed}")

    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for s in summaries:
        for t in s["trace"]:
            k = t["k"]
            sums[k]   = sums.get(k, 0.0) + t["D_mean"]
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


# ============================================================== #
#  DROP-IN RUNNERS                                                 #
# ============================================================== #

def _run_control(opts: dict, mode: str):
    """Shared body for sd3_pg / sd3_pg_matched. mode in {'pg', 'matched'}."""
    from pipeline_wrapper import SD3PipelineWrapper

    cfg    = opts.get("_cfg", {})
    device = opts["device"]
    seeds  = opts.get("seeds") or [opts["seed"]]

    prompts_file = cfg.get("prompts_file", "prompts.yaml")
    prompts = load_prompts(prompts_file)

    pg_cfg = cfg.get("pg", {})

    wrapper = SD3PipelineWrapper(cfg, device=device)
    wrapper.load()

    common_kwargs = dict(
        unet      = wrapper.transformer,
        scheduler = wrapper.scheduler,
        cfg       = cfg,
        device    = device,
        num_particles      = pg_cfg.get("num_particles",      5),
        kde_bandwidth      = pg_cfg.get("kde_bandwidth",      None),
        pressure_exponent  = pg_cfg.get("pressure_exponent",  1.0),
        mobility           = pg_cfg.get("mobility",           1.0),
        symmetry_break_std = pg_cfg.get("symmetry_break_std", 0.02),
        window_start_frac  = pg_cfg.get("window_start_frac",  0.0),
        window_end_frac    = pg_cfg.get("window_end_frac",    0.4),
        max_step_frac      = pg_cfg.get("max_step_frac",      0.05),
    )

    if mode == "pg":
        sampler = PGSampler(**common_kwargs)
        tag = "PG"
        diverse_folder = Path(pg_cfg.get("diverse_output_dir", "outputs/diverse_pg"))
    else:
        matched_cfg  = cfg.get("pg_matched", {})
        records_path = matched_cfg.get("matched_from")
        if not records_path:
            raise ValueError(
                "config.yaml: pg_matched.matched_from must point at a "
                "records_offsetN.json produced by a prior sd3_tds run "
                "(with diag_summaries)."
            )
        schedule = build_matched_schedule_from_tds_records(
            records_path,
            prompt_idx       = matched_cfg.get("matched_prompt_idx"),
            seed              = matched_cfg.get("matched_seed"),
            average_over_all = matched_cfg.get("average_over_all", True),
        )
        sampler = MatchedNoisePGSampler(noise_schedule=schedule, **common_kwargs)
        tag = "PG-matched"
        # deliberately read from pg_matched (not pg) so this never collides
        # with the plain sd3_pg run's output folder
        diverse_folder = Path(matched_cfg.get("diverse_output_dir", "outputs/diverse_pg_matched"))

    base_out = Path(opts["output"])
    diverse_folder.mkdir(parents=True, exist_ok=True)

    records = []
    diag_summaries = []
    prompt_offset = cfg.get("prompt_offset", 0)

    for p_idx, prompt in enumerate(prompts):
        global_idx = p_idx + prompt_offset
        prompt_embeds, pooled_embeds = wrapper.encode_prompt(prompt, opts["negative_prompt"])

        for seed in seeds:
            print(f"[{tag}] prompt {global_idx} seed={seed}  N={sampler.num_particles}")
            x0 = wrapper.get_initial_latents(seed=seed)
            result = sampler.run(x0, prompt_embeds, pooled_embeds, seed=seed)

            disp = _dispersion_stats(result["particles"])
            trace = result["diag"]["trace"]
            avg_cap_hit = sum(t["cap_hit_rate"] for t in trace) / len(trace) if trace else None
            print(f"  [{tag}-diag] mean_pairwise_dist={disp['mean_pairwise_dist']:.4f} "
                  f"(relative={disp['mean_relative_dist']:.4%})  "
                  f"avg_cap_hit_rate={avg_cap_hit if avg_cap_hit is None else f'{avg_cap_hit:.2%}'}")

            diag_summaries.append({
                "prompt_idx": global_idx, "seed": seed,
                "dispersion": disp, "avg_cap_hit_rate": avg_cap_hit, "trace": trace,
            })

            for i, particle_latents in enumerate(result["particles"]):
                out_path = diverse_folder / f"{global_idx + 1}_seed{seed}_particle{i}{base_out.suffix}"
                wrapper.decode_latents(particle_latents).save(out_path)
                records.append({
                    "prompt_idx": global_idx, "prompt": prompt,
                    "seed": seed, "particle": i, "out_path": str(out_path),
                })

    meta = {
        "mode": mode,
        "sampler_params": {k: v for k, v in common_kwargs.items()
                            if k not in ("unet", "scheduler", "cfg", "device")},
        "seeds": list(seeds), "n_prompts": len(prompts), "prompt_offset": prompt_offset,
        "records": records, "diag_summaries": diag_summaries,
    }
    with open(diverse_folder / f"records_offset{prompt_offset}.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)

    return records


def run_sd3_pg(opts: dict):
    """Pure gradient repulsion (PG) baseline -- no stochastic term at all."""
    return _run_control(opts, mode="pg")


def run_sd3_pg_matched(opts: dict):
    """PG + matched-noise-mass, time-only D(t) schedule -- calibrated from
    a prior sd3_tds run via cfg['pg_matched']['matched_from']."""
    return _run_control(opts, mode="matched")
