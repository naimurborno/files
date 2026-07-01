"""
custom_flow_loop.py  (+ Q1 step tracking)
-----------------------------------------
One change from original:
  __init__ accepts optional q1_analyzer
  _step_callback calls q1_analyzer.update_step(step_idx, t) each step
"""

import torch
from typing import Dict, Any


class FlowMatchingLoop:

    def __init__(
        self,
        unet,
        scheduler,
        cfg      : dict,
        device   : str  = "cuda",
        q1_analyzer     = None,      # ← Q1 addition
    ):
        self.unet        = unet
        self.scheduler   = scheduler
        self.cfg         = cfg
        self.device      = device
        self.q1_analyzer = q1_analyzer   # ← Q1 addition

        f_cfg               = cfg.get("flow", {})
        self.num_steps      = f_cfg.get("num_steps",      50)
        self.guidance_scale = f_cfg.get("guidance_scale", 7.5)
        self.solver         = f_cfg.get("solver",         "euler")
        self.do_cfg         = self.guidance_scale > 1.0

        self.scheduler.set_timesteps(self.num_steps)
        self.timesteps = self.scheduler.timesteps

    # ================================================================== #
    #  MAIN ENTRY                                                         #
    # ================================================================== #

    def run(
        self,
        latents          : torch.Tensor,
        text_embeddings  : torch.Tensor,
        pooled_embeddings: torch.Tensor = None,
    ) -> Dict[str, Any]:

        # Reset scheduler state for each new prompt
        self.scheduler.set_timesteps(self.num_steps)
        self.timesteps = self.scheduler.timesteps

        trajectory = []

        for i, t in enumerate(self.timesteps):

            latent_input = (
                torch.cat([latents] * 2) if self.do_cfg else latents
            )

            t_batch = t.reshape(1).expand(latent_input.shape[0]).to(self.device)
            with torch.no_grad():
                model_output = self._velocity_forward(
                    latent_input, t_batch, text_embeddings, pooled_embeddings
                )

            if self.do_cfg:
                model_output = self._apply_cfg(model_output)

            if self.solver == "heun" and i < len(self.timesteps) - 1:
                latents = self._heun_step(
                    latents, model_output, t, self.timesteps[i + 1],
                    text_embeddings, pooled_embeddings
                )
            else:
                latents = self.scheduler.step(model_output, t, latents).prev_sample

            latents = self._step_callback(latents, t, i, model_output)

            if self.cfg.get("save_trajectory", False):
                trajectory.append(latents.clone().cpu())

            if (i + 1) % 10 == 0 or i == 0:
                t_val = t.item() if hasattr(t, "item") else float(t)
                print(f"  [Flow ODE] step {i+1:>3}/{self.num_steps} | "
                      f"t={t_val:.3f} | "
                      f"latent_mean={latents.mean():.4f} | "
                      f"latent_std={latents.std():.4f}")

        return {"latents": latents, "trajectory": trajectory}

    # ================================================================== #
    #  VELOCITY FORWARD                                                   #
    # ================================================================== #

    def _velocity_forward(self, latent_input, t_batch, text_embeddings, pooled_embeddings=None):
        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype

        latent_input    = latent_input.to(device=device, dtype=dtype)
        text_embeddings = text_embeddings.to(device=device, dtype=dtype)
        t_batch         = t_batch.float().to(device=device, dtype=dtype)

        kwargs = dict(
            hidden_states         = latent_input,
            timestep              = t_batch,
            encoder_hidden_states = text_embeddings,
        )
        if pooled_embeddings is not None:
            kwargs["pooled_projections"] = pooled_embeddings.to(device=device, dtype=dtype)

        return self.unet(**kwargs).sample

    # ================================================================== #
    #  CFG                                                                #
    # ================================================================== #

    def _apply_cfg(self, velocity):
        v_uncond, v_cond = velocity.chunk(2)
        return v_uncond + self.guidance_scale * (v_cond - v_uncond)

    # ================================================================== #
    #  HEUN                                                               #
    # ================================================================== #

    def _heun_step(self, x_t, v_t, t, t_next, text_embeddings, pooled_embeddings=None):
        dt     = t_next - t
        x_pred = x_t + dt * v_t

        t_next_batch = t_next.expand(x_pred.shape[0])
        latent_input = torch.cat([x_pred] * 2) if self.do_cfg else x_pred

        with torch.no_grad():
            v_next = self._velocity_forward(latent_input, t_next_batch,
                                            text_embeddings, pooled_embeddings)
        if self.do_cfg:
            v_next = self._apply_cfg(v_next)

        return x_t + (dt / 2) * (v_t + v_next)

    # ================================================================== #
    #  STEP CALLBACK                                                      #
    # ================================================================== #

    def _step_callback(self, latents, t, step_idx, velocity):
        # ← Q1: tell the analyzer which step we're on
        if self.q1_analyzer is not None:
            self.q1_analyzer.update_step(step_idx, t)

        return latents