"""
divin_sampler.py
-----------------
DivIn (Langevin-dynamics initial-latent diversification) ported into the
inference.py / pipeline_wrapper.py architecture, replacing UGILE/PeakBack/ONLB.

Core idea (unchanged from pipeline_sd3.py):
  Before the standard flow-matching denoising loop, run a short Langevin
  walk on `ipp` (num diverse images) initial latents so that their
  Tweedie-denoised predictions (x0_cond - x0_uncond) become spread apart,
  then hand the resulting latents to the normal CFG denoising loop.
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import yaml


def load_prompts(path: str) -> List[str]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, list):
        prompts = data
    else:
        prompts = data.get("prompts", [])
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


class DivInSampler:
    """Langevin-dynamics diverse-initialization sampler for SD3 flow matching."""

    def __init__(
        self,
        unet,
        scheduler,
        cfg          : dict,
        device       : str   = "cuda",
        lr           : float = 0.05,   # Langevin step size (eta)
        max_steps    : int   = 1,      # number of Langevin steps
        temperature  : float = 0.6,    # Langevin temperature (beta)
        gen_num      : int   = 4,      # number of diverse images per prompt (ipp)
        save_original: bool  = True,   # run an extra plain (non-Langevin) denoise
    ):
        self.unet        = unet
        self.scheduler   = scheduler
        self.cfg         = cfg
        self.device      = device
        self.lr          = lr
        self.max_steps   = max_steps
        self.temperature = temperature
        self.gen_num     = gen_num
        self.save_original = save_original

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      50)
        self.guidance_scale = f_cfg.get("guidance_scale", 7.0)
        self.do_cfg         = self.guidance_scale > 1.0

    def run(
        self,
        latents           : torch.Tensor,
        text_embeddings    : torch.Tensor,
        pooled_embeddings  : Optional[torch.Tensor] = None,
        seed               : int = 0,
    ) -> Dict[str, Any]:
        """
        `latents` is a single seeded init latent (batch=1). It is expanded
        to `gen_num` copies, Langevin-optimized jointly, then denoised.
        Returns dict with 'original_latents' (plain CFG bake of the un-
        optimized latent) and 'diverse_latents' (list of final decoded
        latents, one per gen_num branch).
        """
        ipp = self.gen_num
        self.scheduler.set_timesteps(self.num_steps)
        timesteps = self.scheduler.timesteps
        sigma0 = self.scheduler.sigmas[0]
        t0 = timesteps[0]

        # split embeddings: encode_prompt() already returns [uncond, cond] concatenated
        p_neg_single, p_pos_single = text_embeddings.chunk(2)
        pool_neg_single, pool_pos_single = (
            pooled_embeddings.chunk(2) if pooled_embeddings is not None else (None, None)
        )

        p_tot = torch.cat([p_neg_single] * ipp + [p_pos_single] * ipp)
        pool_tot = (
            torch.cat([pool_neg_single] * ipp + [pool_pos_single] * ipp)
            if pooled_embeddings is not None else None
        )

        # --- ORIGINAL (non-diverse) latent: plain seeded init, no Langevin ---
        # Skipped unless explicitly requested — this is an extra full denoise
        # pass the original DivIn script never performed.
        original_final = None
        if self.save_original:
            original_final = self._full_forward_pass(latents.clone(), text_embeddings, pooled_embeddings)
            torch.cuda.empty_cache()

        # --- Phase: Langevin walk on `ipp` initial latents ---
        model_dtype = next(self.unet.parameters()).dtype
        with torch.enable_grad():
            torch.manual_seed(seed)
            lat = torch.randn(
                (ipp, *latents.shape[1:]), device=latents.device, dtype=model_dtype
            )
            lat.requires_grad_(True)

            step_cnt = 0
            while step_cnt < self.max_steps + 1:
                lat_model_input = torch.cat([lat] * 2)
                noise_pred = self._run_transformer(lat_model_input, t0, p_tot, pool_tot)

                uc_pred, c_pred = noise_pred.chunk(2)
                x0_uc = lat - sigma0 * uc_pred
                x0_c = lat - sigma0 * c_pred

                v_vec = (x0_c - x0_uc).reshape(ipp, -1)
                loss_indiv = v_vec.norm(dim=1) ** 2
                loss = loss_indiv.sum()

                if step_cnt == self.max_steps:
                    break

                grad = torch.autograd.grad(loss, lat)[0]

                with torch.no_grad():
                    noise = torch.randn_like(lat)
                    sigma_langevin = math.sqrt(2 * self.lr)
                    lat = lat - self.lr * (grad * self.temperature + lat) + sigma_langevin * noise
                    lat.requires_grad_(True)

                step_cnt += 1

            diverse_init = lat.detach().clone()
            del lat, uc_pred, c_pred, v_vec, loss_indiv, loss, noise_pred, lat_model_input
        torch.cuda.empty_cache()

        # --- Denoise each diverse-init branch with standard CFG loop ---
        # (kept sequential, batch=1, so peak memory ≈ one single-image denoise
        # rather than a batched gen_num-wide denoise)
        branches = []
        for j in range(ipp):
            x_final = self._full_forward_pass(
                diverse_init[j:j + 1], text_embeddings, pooled_embeddings
            )
            branches.append({"branch_idx": j, "latents": x_final.detach()})
            torch.cuda.empty_cache()

        return {"original_latents": original_final, "branches": branches}

    # ------------------------------------------------------------------ #
    #  Transformer / velocity helpers                                     #
    # ------------------------------------------------------------------ #

    def _run_transformer(self, latents_in, t_val, p_embeds, p_pooled):
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        ts = t_val.reshape(1).expand(latents_in.shape[0]).to(device=device, dtype=dtype)
        kwargs = dict(
            hidden_states=latents_in.to(device=device, dtype=dtype),
            timestep=ts,
            encoder_hidden_states=p_embeds.to(device=device, dtype=dtype),
        )
        if p_pooled is not None:
            kwargs["pooled_projections"] = p_pooled.to(device=device, dtype=dtype)
        return self.unet(**kwargs).sample

    def _velocity_forward(self, x, t, text_embeddings, pooled_embeddings):
        latent_input = torch.cat([x, x]) if self.do_cfg else x
        with torch.no_grad():
            output = self._run_transformer(latent_input, t, text_embeddings, pooled_embeddings)
        if self.do_cfg:
            v_uncond, v_cond = output.chunk(2)
            return v_uncond + self.guidance_scale * (v_cond - v_uncond)
        return output

    def _full_forward_pass(self, x0, text_embeddings, pooled_embeddings):
        self.scheduler.set_timesteps(self.num_steps)
        timesteps = self.scheduler.timesteps
        x = x0.clone()
        for t in timesteps:
            v = self._velocity_forward(x, t, text_embeddings, pooled_embeddings)
            x = self.scheduler.step(v, t, x).prev_sample
        return x


# ══════════════════════════════════════════════════════════════════════ #
#  DROP-IN RUNNER (same shape as run_sd3_ugile / run_sd3_peakback)        #
# ══════════════════════════════════════════════════════════════════════ #

def run_sd3_divin(opts: dict):
    from pipeline_wrapper import SD3PipelineWrapper

    cfg    = opts.get("_cfg", {})
    device = opts["device"]
    seeds  = opts.get("seeds") or [opts["seed"]]

    prompts_file = cfg.get("prompts_file", "prompts.yaml")
    prompts = load_prompts(prompts_file)

    dv_cfg = cfg.get("divin", {})

    wrapper = SD3PipelineWrapper(cfg, device=device)
    wrapper.load()

    sampler = DivInSampler(
        unet        = wrapper.transformer,
        scheduler   = wrapper.scheduler,
        cfg         = cfg,
        device      = device,
        lr          = dv_cfg.get("lr",          0.05),
        max_steps   = dv_cfg.get("max_steps",   1),
        temperature = dv_cfg.get("temperature", 0.6),
        gen_num     = dv_cfg.get("gen_num",     4),
        save_original = dv_cfg.get("save_original", True),
    )

    base_out        = Path(opts["output"])
    diverse_folder  = Path(dv_cfg.get("diverse_output_dir",  "outputs/diverse"))
    original_folder = Path(dv_cfg.get("original_output_dir", "outputs/original"))
    diverse_folder.mkdir(parents=True, exist_ok=True)
    original_folder.mkdir(parents=True, exist_ok=True)
    save_original   = dv_cfg.get("save_original", True)

    prompt_offset = cfg.get("prompt_offset", 0)

    records = []
    total = len(prompts) * len(seeds)
    done = 0

    for p_idx, prompt in enumerate(prompts):
        global_idx = p_idx + prompt_offset

        prompt_embeds, pooled_embeds = wrapper.encode_prompt(prompt, opts["negative_prompt"])

        for seed in seeds:
            done += 1
            print(f"[DivIn] image {done}/{total}  seed={seed}")

            latents = wrapper.get_initial_latents(seed=seed)
            result = sampler.run(latents, prompt_embeds, pooled_embeds, seed=seed)

            if save_original and result["original_latents"] is not None:
                base_path = original_folder / (f"{global_idx + 1}" + base_out.suffix)
                wrapper.decode_latents(result["original_latents"]).save(base_path)

            for br in result["branches"]:
                out_path = diverse_folder / (f"{global_idx + 1}_{br['branch_idx']}" + base_out.suffix)
                wrapper.decode_latents(br["latents"]).save(out_path)

                records.append({
                    "prompt_idx": global_idx,
                    "prompt": prompt,
                    "seed": seed,
                    "branch": br["branch_idx"],
                    "out_path": str(out_path),
                })

    print(f"[DivIn] Done. {len(records)} diverse image(s) written to {diverse_folder}")
    return records