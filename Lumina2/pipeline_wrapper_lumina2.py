"""
pipeline_wrapper_lumina2.py  (Flow Matching edition — Lumina-Image-2.0 port)
-----------------------------------------------------------------------------
SD3PipelineWrapper / SanaPipelineWrapper, ported to Lumina2 so the UGILE
sampler (latent_escape_sampler_lumina2.py) can run against Lumina2 exactly
the way it runs against SD3 / SANA today. Same cfg.get(...) call shapes.

What's different vs. SANA, and why:

  ┌───────────────────────────┬──────────────────────────────────────────┐
  │     SanaPipelineWrapper    │      Lumina2PipelineWrapper              │
  ├───────────────────────────┼──────────────────────────────────────────┤
  │ Linear-attention DiT       │ Standard DiT (Lumina2Transformer2DModel) │
  │ single Gemma-2 encoder     │ single Gemma-2 encoder (same family)     │
  │ AutoencoderDC, 32x VAE     │ AutoencoderKL, 8x VAE                    │
  │ scaling_factor only        │ scaling_factor + shift_factor            │
  │ encoder_attention_mask     │ encoder_attention_mask (same convention) │
  │ in_channels = 32           │ in_channels = transformer.config.in_chan │
  │ t in [0,1], noise at t=0   │ t REVERSED: current_timestep = 1 - t/T,  │
  │                            │ noise_pred negated before scheduler.step │
  │ no CFG truncation/normaliz.│ cfg_trunc_ratio + cfg_normalization      │
  │ timestep_scale quirk       │ no analogous quirk                       │
  └───────────────────────────┴──────────────────────────────────────────┘

Lumina2's official __call__ has two backbone-specific quirks that the
sampler must reproduce bit-for-bit (these are NOT UGILE's math — they are
how Lumina2 itself defines its forward velocity, so getting them right is
what makes "don't change UGILE's method" possible at all):

  1. Timestep convention is flipped relative to SD3/SANA: Lumina2 uses
     t=0 ↔ image, t=1 ↔ noise (opposite of the flow-matching convention
     SD3/SANA use). The pipeline computes
         current_timestep = 1 - t / scheduler.config.num_train_timesteps
     before calling the transformer.
  2. The transformer's raw output is negated (`noise_pred = -noise_pred`)
     before being handed to `scheduler.step`, to convert it back into the
     velocity sign convention `FlowMatchEulerDiscreteScheduler.step`
     expects.

Both quirks live in this wrapper's `velocity_forward` helper (mirrors
SanaPipelineWrapper exposing `timestep_scale` for the sampler to read /
apply) so the UGILE sampler file itself only needs cosmetic renames, not
new math — same design rule as the SANA port.

Design rule unchanged: Load once → patch freely → weights stay frozen
unless explicitly unfrozen.
"""

import torch
from diffusers import Lumina2Pipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image


# Copied verbatim from diffusers' pipeline_lumina2.py — needed to build the
# same per-call mu (timestep-shift) value the official pipeline uses for
# `scheduler.set_timesteps(..., mu=mu)`. Not UGILE math; pure plumbing.
def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class Lumina2PipelineWrapper:

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = device

        self.pipe              = None
        self.tokenizer         = None
        self.text_encoder      = None
        self.transformer       = None
        self.vae               = None
        self.scheduler         = None
        self.vae_scale_factor  = 8   # AutoencoderKL default; refined in load()

        self.q1_analyzer        = None   # kept for parity with SD3/SANA wrappers

    # ================================================================== #
    #  LOAD                                                               #
    # ================================================================== #

    def load(self):
        model_id = self.cfg.get(
            "model_id", "Alpha-VLLM/Lumina-Image-2.0"
        )

        print(f"[Pipeline] Loading: {model_id}")

        self.pipe = Lumina2Pipeline.from_pretrained(
            model_id,
            torch_dtype = torch.float16,
        ).to(self.device)

        self.tokenizer    = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.transformer  = self.pipe.transformer
        self.vae          = self.pipe.vae
        self.vae.to(torch.float32)   # matches SD3/SANA fp32 decode

        # Lumina2Pipeline hardcodes this to 8 in __init__ (AutoencoderKL,
        # not DC-AE) — reuse the pipe's own attribute rather than
        # hardcoding here, same caution as the SANA wrapper.
        self.vae_scale_factor = self.pipe.vae_scale_factor

        # Already a FlowMatchEulerDiscreteScheduler out of the box — no
        # swap needed (unlike SANA, which ships DPMSolverMultistep and
        # needs an explicit re-instantiation). Kept as an from_config
        # call anyway for parity / safety if a checkpoint ships a
        # different scheduler.
        try:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
        except Exception as e:
            print(f"[Pipeline] WARNING: could not adapt source scheduler "
                  f"config ({e}); falling back to a default "
                  f"FlowMatchEulerDiscreteScheduler().")
            self.scheduler = FlowMatchEulerDiscreteScheduler()

        self._freeze_all()
        print("[Pipeline] Loaded & frozen ✔")
        self._print_memory()

    # ================================================================== #
    #  PATCH                                                              #
    # ================================================================== #

    def patch(self):
        """
        Q1 hooks (Q1EntropyAnalyzer) were written against SD3's MMDiT
        block list (`self.transformer.transformer_blocks`). Lumina2's
        Lumina2Transformer2DModel exposes its blocks under
        `self.transformer.layers`, so this would need a small attribute
        rename if/when that analyzer gets wired in here. Left as a no-op
        stub since it isn't part of the active UGILE path (same status
        as the SANA wrapper's patch()).
        """
        pass

    # ================================================================== #
    #  PROMPT ENCODING                                                    #
    # ================================================================== #

    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """
        Returns (prompt_embeds, attention_mask) — uncond+cond concatenated
        on dim 0, same convention as SanaPipelineWrapper.encode_prompt:
        the second tensor is an attention MASK (Gemma has no pooled
        output here either), not a pooled vector.

        Delegates to the pipeline's own `encode_prompt` for the same
        reason the SANA wrapper does: Lumina2's official encode_prompt
        prepends a fixed system prompt ("You are an assistant designed
        to generate superior images...") before tokenization — this is
        load-bearing for output quality and is reused verbatim rather
        than reimplemented by hand.
        """
        do_cfg = self.cfg.get("flow", {}).get("guidance_scale", 4.0) > 1.0
        max_seq_len = self.cfg.get("max_sequence_length", 256)

        with torch.no_grad():
            (
                cond_emb, cond_mask,
                uncond_emb, uncond_mask,
            ) = self.pipe.encode_prompt(
                prompt,
                do_cfg,
                negative_prompt        = negative_prompt,
                device                  = self.device,
                max_sequence_length     = max_seq_len,
            )

            if do_cfg:
                prompt_embeds  = torch.cat([uncond_emb,  cond_emb])
                attention_mask = torch.cat([uncond_mask, cond_mask])
            else:
                prompt_embeds  = cond_emb
                attention_mask = cond_mask

        return prompt_embeds, attention_mask

    # ================================================================== #
    #  LATENT HELPERS                                                     #
    # ================================================================== #

    def get_initial_latents(self, seed: int = 42) -> torch.Tensor:
        gen_cfg   = self.cfg.get("generation", {})
        H         = gen_cfg.get("height", 512)
        W         = gen_cfg.get("width",  512)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Mirror Lumina2Pipeline.prepare_latents: round H/W down to a
        # multiple of (vae_scale_factor * 2) before dividing, since the
        # official pipeline packs 2x2 patches on top of the 8x VAE.
        Hc = 2 * (int(H) // (self.vae_scale_factor * 2))
        Wc = 2 * (int(W) // (self.vae_scale_factor * 2))

        latents = torch.randn(
            (1, self.transformer.config.in_channels, Hc, Wc),
            generator = generator,
            device    = self.device,
            dtype     = torch.float32,   # matches SD3/SANA fp32 noise init
        )
        # Lumina2, like SD3/SANA, operates directly on unscaled N(0,I)
        # noise — no pre-scaling here; VAE decode handles unscaling.
        return latents

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        # AutoencoderKL decode convention (has BOTH scaling_factor and
        # shift_factor, unlike SANA's DC-AE which has scaling_factor only):
        #   pixels = VAE.decode(z / scaling_factor + shift_factor)
        scaling_factor = self.vae.config.scaling_factor
        shift_factor   = self.vae.config.shift_factor

        if not torch.isfinite(latents).all():
            print("[Pipeline] WARNING: latents contain NaN/Inf before decode — "
                  "returning blank image")
            from PIL import Image as PILImage
            import numpy as np
            gen_cfg = self.cfg.get("generation", {})
            return PILImage.fromarray(
                np.zeros(
                    (gen_cfg.get("height", 512), gen_cfg.get("width", 512), 3),
                    dtype="uint8"
                )
            )

        latents = latents.to(dtype=self.vae.dtype) / scaling_factor + shift_factor

        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype("uint8")
        from PIL import Image as PILImage
        return PILImage.fromarray(image)

    # ================================================================== #
    #  TIMESTEP / VELOCITY CONVENTION HELPERS                             #
    #  (Lumina2-specific — no SD3/SANA analog; see module docstring)      #
    # ================================================================== #

    def reversed_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """current_timestep = 1 - t / num_train_timesteps, broadcast-ready."""
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        return 1 - t / num_train_timesteps

    @staticmethod
    def negate_velocity(noise_pred: torch.Tensor) -> torch.Tensor:
        """Lumina2's transformer output must be negated before scheduler.step."""
        return -noise_pred

    # ================================================================== #
    #  GENERATE  (standard path — parity with SD3/SANA's .generate())     #
    # ================================================================== #

    def generate(self, prompt: str, negative_prompt: str = "", seed: int = 42) -> Image.Image:
        """Standard diffusers pipeline path — unchanged behaviour."""
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
            guidance_scale      = f_cfg.get("guidance_scale", 4.0),
            cfg_trunc_ratio     = f_cfg.get("cfg_trunc_ratio", 0.25),
            cfg_normalization   = f_cfg.get("cfg_normalization", True),
            generator           = generator,
        )
        return result.images[0]

    # ================================================================== #
    #  INTERNALS                                                          #
    # ================================================================== #

    def _freeze_all(self):
        for model in [self.text_encoder, self.transformer, self.vae]:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = False

    def _print_memory(self):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            print(f"[Pipeline] GPU memory used: {alloc:.2f} GB")