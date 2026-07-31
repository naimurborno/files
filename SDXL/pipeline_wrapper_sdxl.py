"""
pipeline_wrapper_sdxl.py
-------------------------
SDXL port of the PixArt-Sigma/SD3/SANA/Lumina2 wrappers. Standard
eps-prediction UNet diffusion (Euler/DPM-Solver family) — same "no
flow-matching quirks" category as PixArt-Sigma:
  - no reversed timestep
  - no output negation
  - no cfg_trunc_ratio / cfg_normalization
  - sigma_k comes straight from scheduler.sigmas (EulerDiscreteScheduler /
    DPMSolverMultistepScheduler both expose this)

SDXL specifics UGILE's forward call must handle (this is where SDXL differs
from PixArt-Sigma, everything else is identical in spirit):
  - TWO text encoders (CLIP ViT-L/14 + OpenCLIP ViT-bigG/14). encode_prompt
    concatenates their per-token hidden states on the feature dim for
    encoder_hidden_states, and takes the bigG encoder's pooled output as
    add_text_embeds.
  - No padding attention_mask is used by the stock SDXL pipeline (prompts are
    padded/truncated to a fixed 77 tokens with no mask passed to the UNet),
    so attention_mask is accepted for interface parity but ignored.
  - `added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids}` is
    REQUIRED (this replaces PixArt-Sigma's resolution/aspect_ratio pair) —
    add_time_ids encodes (original_size, crops_coords_top_left, target_size).
  - UNet2DConditionModel.forward returns eps directly — no learned-variance
    channel splitting needed (that was a PixArt-Sigma/-MS-only quirk).
  - VAE is numerically fragile in fp16 (produces black/NaN images on the
    stock SDXL VAE); we decode in float32, matching the official pipeline's
    `pipe.upcast_vae()` workaround.
"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image


class SDXLPipelineWrapper:

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = device

        self.pipe              = None
        self.tokenizer         = None
        self.tokenizer_2       = None
        self.text_encoder      = None
        self.text_encoder_2    = None
        self.unet              = None
        self.vae               = None
        self.scheduler         = None
        self.vae_scale_factor  = 8

    def load(self):
        model_id = self.cfg.get(
            "model_id", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        print(f"[Pipeline] Loading: {model_id}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16",
        ).to(self.device)

        self.tokenizer      = self.pipe.tokenizer
        self.tokenizer_2    = self.pipe.tokenizer_2
        self.text_encoder   = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        self.unet           = self.pipe.unet
        self.vae            = self.pipe.vae
        self.vae_scale_factor = self.pipe.vae_scale_factor

        try:
            self.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
        except Exception as e:
            print(f"[Pipeline] WARNING: scheduler adapt failed ({e}); "
                  f"falling back to default EulerDiscreteScheduler().")
            self.scheduler = EulerDiscreteScheduler()

        # SDXL's stock VAE overflows in fp16 — decode in fp32, same fix
        # the official pipeline applies via pipe.upcast_vae().
        self.vae.to(dtype=torch.float32)

        self._freeze_all()
        print("[Pipeline] Loaded & frozen ✔")
        self._print_memory()

    def patch(self):
        pass  # no Q1 hooks wired in for SDXL

    # ------------------------------------------------------------------ #
    #  PROMPT ENCODING                                                    #
    # ------------------------------------------------------------------ #

    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """
        Returns (prompt_embeds, pooled_embeds): both uncond+cond concatenated
        on dim 0. attention_mask is not used by SDXL (kept out of the return
        tuple; latent_escape_sampler_sdxl.py's _velocity_forward accepts a
        matching `None` in its place for interface parity with the other
        UGILE ports).

        prompt_embeds  : concatenated per-token hidden states from BOTH text
                         encoders on the feature dim -> [B, 77, 2048]
        pooled_embeds  : bigG (text_encoder_2) pooled output -> [B, 1280],
                         goes into added_cond_kwargs["text_embeds"]
        """
        do_cfg = self.cfg.get("flow", {}).get("guidance_scale", 5.0) > 1.0

        with torch.no_grad():
            (
                cond_emb, neg_emb,
                cond_pooled, neg_pooled,
            ) = self.pipe.encode_prompt(
                prompt              = prompt,
                device              = self.device,
                num_images_per_prompt = 1,
                do_classifier_free_guidance = do_cfg,
                negative_prompt     = negative_prompt,
            )

            if do_cfg:
                prompt_embeds  = torch.cat([neg_emb,     cond_emb])
                pooled_embeds  = torch.cat([neg_pooled,  cond_pooled])
            else:
                prompt_embeds  = cond_emb
                pooled_embeds  = cond_pooled

        return prompt_embeds, pooled_embeds

    # ------------------------------------------------------------------ #
    #  LATENT HELPERS                                                     #
    # ------------------------------------------------------------------ #

    def get_initial_latents(self, seed: int = 42) -> torch.Tensor:
        gen_cfg   = self.cfg.get("generation", {})
        H         = gen_cfg.get("height", 1024)
        W         = gen_cfg.get("width",  1024)
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
        # noise level — required for EulerDiscreteScheduler.
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
                    (gen_cfg.get("height", 1024), gen_cfg.get("width", 1024), 3),
                    dtype="uint8"
                )
            )

        # Decode in fp32 — the stock SDXL VAE is unstable in fp16.
        latents = latents.to(dtype=self.vae.dtype) / scaling_factor

        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype("uint8")
        return Image.fromarray(image)

    # ------------------------------------------------------------------ #
    #  MICRO-CONDITIONING (SDXL add_time_ids)                             #
    # ------------------------------------------------------------------ #

    def added_cond_kwargs(self, pooled_embeds, batch_size: int, H: int, W: int, dtype, device):
        """
        SDXL's equivalent of PixArt-Sigma's resolution/aspect_ratio pair:
        add_time_ids = (original_size, crops_coords_top_left, target_size),
        flattened, one row per batch element, broadcast across the CFG
        uncond/cond split. crop is (0, 0) since UGILE never crops.
        """
        original_size = (H, W)
        target_size   = (H, W)
        crops_coords_top_left = (0, 0)

        add_time_ids = torch.tensor(
            [list(original_size) + list(crops_coords_top_left) + list(target_size)],
            dtype=dtype, device=device,
        ).repeat(batch_size, 1)

        return {"text_embeds": pooled_embeds.to(device=device, dtype=dtype),
                "time_ids":    add_time_ids}

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
            height              = gen_cfg.get("height", 1024),
            width               = gen_cfg.get("width",  1024),
            num_inference_steps = f_cfg.get("num_steps", 30),
            guidance_scale      = f_cfg.get("guidance_scale", 5.0),
            generator           = generator,
        )
        return result.images[0]

    # ------------------------------------------------------------------ #
    #  INTERNALS                                                          #
    # ------------------------------------------------------------------ #

    def _freeze_all(self):
        for model in [self.text_encoder, self.text_encoder_2, self.unet, self.vae]:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = False

    def _print_memory(self):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            print(f"[Pipeline] GPU memory used: {alloc:.2f} GB")
