"""
pipeline_wrapper_pixart_sigma.py
---------------------------------
PixArt-Sigma port of the SD3/SANA/Lumina2 wrappers. Standard eps-prediction
diffusion (DDPM/DPM-Solver family) — NO flow-matching quirks:
  - no reversed timestep
  - no output negation
  - no cfg_trunc_ratio / cfg_normalization
  - t goes from num_train_timesteps -> 0, sigma-free (uses alphas_cumprod)

PixArt-Sigma specifics UGILE's forward call must handle:
  - T5 text encoder (not Gemma-2), attention_mask IS a real padding mask
  - Transformer needs `added_cond_kwargs={"resolution":..., "aspect_ratio":...}`
    micro-conditioning tensors for the -MS (multi-scale) checkpoints.
  - transformer.forward returns eps (and optionally learned variance channels
    if `transformer.config.out_channels == 2 * in_channels` — we split those
    off and only keep the eps half, matching diffusers' pipeline behavior).
"""

import torch
from diffusers import PixArtSigmaPipeline, DPMSolverMultistepScheduler
from PIL import Image
import diffusers
import transformers
diffusers.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()
import os, warnings, logging as pylogging
import contextlib, io

os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
pylogging.getLogger("diffusers").setLevel(pylogging.ERROR)
pylogging.getLogger("transformers").setLevel(pylogging.ERROR)

class PixArtSigmaPipelineWrapper:

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = device

        self.pipe             = None
        self.tokenizer        = None
        self.text_encoder     = None
        self.transformer      = None
        self.vae              = None
        self.scheduler        = None
        self.vae_scale_factor = 8

    def load(self):
        model_id = self.cfg.get(
            "model_id", "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
        )
        print(f"[Pipeline] Loading: {model_id}")
        buf = io.StringIO()
        
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            self.pipe = PixArtSigmaPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16,
            ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.tokenizer    = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.transformer   = self.pipe.transformer
        self.vae           = self.pipe.vae
        self.vae_scale_factor = self.pipe.vae_scale_factor

        try:
            self.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
        except Exception as e:
            print(f"[Pipeline] WARNING: scheduler adapt failed ({e}); "
                  f"falling back to default DPMSolverMultistepScheduler().")
            self.scheduler = DPMSolverMultistepScheduler()

        self._freeze_all()
        print("[Pipeline] Loaded & frozen ✔")
        self._print_memory()

    def patch(self):
        pass  # no Q1 hooks wired in for PixArt-Sigma

    # ------------------------------------------------------------------ #
    #  PROMPT ENCODING                                                    #
    # ------------------------------------------------------------------ #

    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """
        Returns (prompt_embeds, attention_mask): uncond+cond concatenated on
        dim 0. Unlike Gemma-2, T5's attention_mask is a real padding mask
        (still boolean/int, same downstream usage).
        """
        do_cfg      = self.cfg.get("flow", {}).get("guidance_scale", 4.5) > 1.0
        max_seq_len = self.cfg.get("max_sequence_length", 120)

        with torch.no_grad():
            (
                cond_emb, cond_mask,
                uncond_emb, uncond_mask,
            ) = self.pipe.encode_prompt(
                prompt,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompt,
                device=self.device,
                max_sequence_length=max_seq_len,
            )

            if do_cfg:
                prompt_embeds  = torch.cat([uncond_emb,  cond_emb])
                attention_mask = torch.cat([uncond_mask, cond_mask])
            else:
                prompt_embeds  = cond_emb
                attention_mask = cond_mask

        return prompt_embeds, attention_mask

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
            (1, self.transformer.config.in_channels, Hc, Wc),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        # Official pipeline scales raw noise by the scheduler's expected
        # noise level — required for DPMSolverMultistepScheduler.
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
    #  MICRO-CONDITIONING (PixArt-Sigma -MS checkpoints)                   #
    # ------------------------------------------------------------------ #

    def added_cond_kwargs(self, batch_size: int, H: int, W: int, dtype, device):
        if getattr(self.transformer.config, "sample_size", None) is None:
            return None  # non-MS checkpoint, no micro-conditioning needed
        resolution = torch.tensor([[H, W]], dtype=dtype, device=device).repeat(batch_size, 1)
        aspect_ratio = torch.tensor([[H / W]], dtype=dtype, device=device).repeat(batch_size, 1)
        return {"resolution": resolution, "aspect_ratio": aspect_ratio}

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
            num_inference_steps = f_cfg.get("num_steps", 30),
            guidance_scale      = f_cfg.get("guidance_scale", 4.5),
            generator           = generator,
        )
        return result.images[0]

    # ------------------------------------------------------------------ #
    #  INTERNALS                                                          #
    # ------------------------------------------------------------------ #

    def _freeze_all(self):
        for model in [self.text_encoder, self.transformer, self.vae]:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = False

    def _print_memory(self):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            print(f"[Pipeline] GPU memory used: {alloc:.2f} GB")