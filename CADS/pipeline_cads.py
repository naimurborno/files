"""
Algorithm 1 (CADS) applied to Stable Diffusion 1.5's text-conditioning embeddings.

We do NOT use the diffusers high-level `StableDiffusionPipeline.__call__`
because CADS requires per-step access to the conditional / unconditional
text embeddings BEFORE they are fed to the UNet (to corrupt them). Instead
we reimplement the minimal SD sampling loop ourselves, following Algorithm 1
line-by-line:

    Require: w_CFG, y (input condition), gamma(t), s
    z_1 ~ N(0, I)
    for t = T, ..., 1:
        y_hat      = sqrt(gamma(t))*y      + s*sqrt(1-gamma(t))*n
        y_hat_null = sqrt(gamma(t))*y_null + s*sqrt(1-gamma(t))*n'
        [rescale y_hat, y_hat_null if needed]
        D_CFG = D(z_t, t, y_hat_null) + w_CFG * (D(z_t, t, y_hat) - D(z_t, t, y_hat_null))
        z_{t-1} = diffusion_reverse(D_CFG, z_t, t)
    return z_0

n and n' are drawn independently for the conditional and null embeddings,
matching Algorithm 1's notation (n vs n').
"""

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

from cads import CADSScheduleConfig, add_noise


@dataclass
class CADSGenerationConfig:
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 50
    guidance_scale: float = 9.0          # w_CFG
    height: int = 512
    width: int = 512
    seed: Optional[int] = None
    use_cads: bool = True                # toggle CADS off to get plain DDIM/CFG for comparison


class CADSStableDiffusion:
    """Wraps SD 1.5 and implements Algorithm 1 (CADS sampling) manually."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        cads_config: CADSScheduleConfig = None,
        device: str = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device == "cuda" else torch.float32

        # Load full SD1.5 pipeline just to reuse its components (VAE, tokenizer,
        # text encoder, UNet); we bypass its __call__ entirely.
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype, safety_checker=None
        ).to(self.device)

        # DDIM is the deterministic-step sampler used as the base "diffusion_reverse"
        # step in Algorithm 1; CADS is sampler-agnostic (Table 3 in the paper shows
        # it composes with DDIM/DPM++/PNDM/UniPC identically). We use DDIM here.
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler

        self.cads_cfg = cads_config or CADSScheduleConfig()

    # ------------------------------------------------------------------ #
    # Text embedding helper
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tok = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        embeds = self.text_encoder(tok.input_ids.to(self.device))[0]
        return embeds.to(self.dtype)

    # ------------------------------------------------------------------ #
    # Main generation loop == Algorithm 1
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, cfg: CADSGenerationConfig) -> "PIL.Image.Image":
        generator = None
        if cfg.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(cfg.seed)

        # --- clean conditions y (prompt) and y_null (negative / empty prompt) ---
        y = self._encode_prompt(cfg.prompt)
        y_null = self._encode_prompt(cfg.negative_prompt)

        # --- initial latent z_1 ~ N(0, I) ---
        latents_shape = (
            1,
            self.unet.config.in_channels,
            cfg.height // 8,
            cfg.width // 8,
        )
        latents = torch.randn(
            latents_shape, generator=generator, device=self.device, dtype=self.dtype
        )

        self.scheduler.set_timesteps(cfg.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        latents = latents * self.scheduler.init_noise_sigma
        num_steps = len(timesteps)

        for i, t_diffusion in enumerate(timesteps):
            # --- Normalized annealing time: t=1 at the FIRST step (max noise),
            #     t=0 at the LAST step (clean sample), per the paper's convention. ---
            t_norm = 1.0 - i / max(num_steps - 1, 1)
            gamma_t = self.cads_cfg.gamma(t_norm)

            if cfg.use_cads:
                # independent noise draws n (for y) and n' (for y_null), Algorithm 1
                n = torch.randn(y.shape, generator=generator, device=self.device, dtype=y.dtype)
                n_prime = torch.randn(
                    y_null.shape, generator=generator, device=self.device, dtype=y_null.dtype
                )
                y_hat = add_noise(
                    y, gamma_t, self.cads_cfg.noise_scale, self.cads_cfg.psi,
                    rescale=self.cads_cfg.rescale, noise=n,
                )
                if self.cads_cfg.apply_to == "both":
                    y_hat_null = add_noise(
                        y_null, gamma_t, self.cads_cfg.noise_scale, self.cads_cfg.psi,
                        rescale=self.cads_cfg.rescale, noise=n_prime,
                    )
                else:
                    y_hat_null = y_null
            else:
                y_hat, y_hat_null = y, y_null

            latent_model_input = self.scheduler.scale_model_input(latents, t_diffusion)

            # D(z_t, t, y_hat_null) and D(z_t, t, y_hat) — batched for efficiency
            emb = torch.cat([y_hat_null, y_hat], dim=0)
            lat_in = torch.cat([latent_model_input, latent_model_input], dim=0)
            noise_pred = self.unet(lat_in, t_diffusion, encoder_hidden_states=emb).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)

            # --- classifier-free guidance, Eq. (10) ---
            noise_pred = noise_uncond + cfg.guidance_scale * (noise_cond - noise_uncond)

            # --- one reverse diffusion step z_{t-1} = diffusion_reverse(...) ---
            latents = self.scheduler.step(noise_pred, t_diffusion, latents).prev_sample

        image = self._decode_latents(latents)
        return image

    @torch.no_grad()
    def _decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")[0]
        from PIL import Image
        return Image.fromarray(image)
