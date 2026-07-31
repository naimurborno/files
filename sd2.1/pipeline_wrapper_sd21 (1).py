"""
pipeline_wrapper_sd21.py
-------------------------
Stable Diffusion 2.1 port of the PixArt-Sigma wrapper. Standard
eps-prediction UNet (DDIM/PNDM family) — even fewer quirks than PixArt-Sigma:
  - single CLIP text encoder (no T5, no attention_mask needed)
  - no added_cond_kwargs / micro-conditioning (not an -MS checkpoint concept)
  - UNet never predicts learned variance channels for the base v2.1 checkpoint
    (out_channels == in_channels), so no chunk-splitting is required
  - sigma_k has no native scheduler.sigmas on DDIM/PNDM, so it's derived from
    alphas_cumprod: sigma_t = sqrt((1 - a_bar_t) / a_bar_t)
"""

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image


class SD21PipelineWrapper:

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = device

        self.pipe             = None
        self.tokenizer        = None
        self.text_encoder     = None
        self.unet             = None
        self.vae              = None
        self.scheduler        = None
        self.vae_scale_factor = 8

    def load(self):
        model_id = self.cfg.get(
            "model_id", "stabilityai/stable-diffusion-2-1"
        )
        print(f"[Pipeline] Loading: {model_id}")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None,
        ).to(self.device)

        self.tokenizer    = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet         = self.pipe.unet
        self.vae          = self.pipe.vae
        self.vae_scale_factor = self.pipe.vae_scale_factor

        try:
            self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        except Exception as e:
            print(f"[Pipeline] WARNING: scheduler adapt failed ({e}); "
                  f"falling back to default DDIMScheduler().")
            self.scheduler = DDIMScheduler()

        self._freeze_all()
        print("[Pipeline] Loaded & frozen ✔")
        self._print_memory()

    def patch(self):
        pass  # no Q1 hooks wired in for SD2.1

    # ------------------------------------------------------------------ #
    #  PROMPT ENCODING                                                    #
    # ------------------------------------------------------------------ #

    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """
        Returns (prompt_embeds, attention_mask=None): uncond+cond concatenated
        on dim 0. CLIP is always run to its fixed context length (77 tokens,
        padded), so unlike T5 there is no real padding mask to pass through —
        attention_mask is kept as a return slot only for interface parity
        with the other UGILE ports.
        """
        do_cfg = self.cfg.get("flow", {}).get("guidance_scale", 7.5) > 1.0

        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompt,
            )

            if do_cfg:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds, None

    # ------------------------------------------------------------------ #
    #  LATENT HELPERS                                                     #
    # ------------------------------------------------------------------ #

    def get_initial_latents(self, seed: int = 42) -> torch.Tensor:
        gen_cfg   = self.cfg.get("generation", {})
        H         = gen_cfg.get("height", 512)
        W         = gen_cfg.get("width",  512)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        Hc = int(H) // self.vae_scale_factor
        Wc = int(W) // self.vae_scale_factor

        latents = torch.randn(
            (1, self.unet.config.in_channels, Hc, Wc),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        # Official pipeline scales raw noise by the scheduler's expected
        # noise level — required for DDIM/PNDM just like DPMSolver.
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        scaling_factor = self.vae.config.scaling_factor

        if not torch.isfinite(latents).all():
            print("[Pipeline] WARNING: latents contain NaN/Inf before decode — "
                  "returning blank image")
            import numpy as np
            gen_cfg = self.cfg.get("generation", {})
            return Image.fromarray(
                np.zeros(
                    (gen_cfg.get("height", 512), gen_cfg.get("width", 512), 3),
                    dtype="uint8"
                )
            )

        latents = latents.to(dtype=self.vae.dtype) / scaling_factor

        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype("uint8")
        return Image.fromarray(image)

    # ------------------------------------------------------------------ #
    #  MICRO-CONDITIONING — not applicable to SD2.1                       #
    # ------------------------------------------------------------------ #

    def added_cond_kwargs(self, batch_size: int, H: int, W: int, dtype, device):
        return None  # SD2.1 UNet has no resolution/aspect-ratio conditioning

    # ------------------------------------------------------------------ #
    #  GENERATE — standard path                                          #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, negative_prompt: str = "", seed: int = 42) -> Image.Image:
        f_cfg     = self.cfg.get("flow", {})
        gen_cfg   = self.cfg.get("generation", {})
        generator = torch.Generator(device="cpu").manual_seed(seed)

        print("[Pipeline] Running standard generation path.")
        result = self.pipe(
            prompt              = prompt,
            negative_prompt     = negative_prompt,
            height              = gen_cfg.get("height", 512),
            width               = gen_cfg.get("width",  512),
            num_inference_steps = f_cfg.get("num_steps", 50),
            guidance_scale      = f_cfg.get("guidance_scale", 7.5),
            generator           = generator,
        )
        return result.images[0]

    # ------------------------------------------------------------------ #
    #  INTERNALS                                                          #
    # ------------------------------------------------------------------ #

    def _freeze_all(self):
        for model in [self.text_encoder, self.unet, self.vae]:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = False

    def _print_memory(self):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            print(f"[Pipeline] GPU memory used: {alloc:.2f} GB")
