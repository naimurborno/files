"""
pipeline_wrapper_sana.py  (Flow Matching edition — SANA port)
---------------------------------------------------------------
This is pipeline_wrapper.py's SD3PipelineWrapper, ported to SANA so the
UGILE sampler (latent_escape_sampler_sana.py) can run against SANA exactly
the way it runs against SD3 today.

What changed vs. SD3, and why:

  ┌───────────────────────────┬──────────────────────────────────────┐
  │     SD3PipelineWrapper    │       SanaPipelineWrapper            │
  ├───────────────────────────┼──────────────────────────────────────┤
  │ MMDiT transformer         │ Linear-attention DiT                │
  │                           │ (SanaTransformer2DModel)             │
  │ CLIP-L + CLIP-G (+T5)     │ single Gemma-2 decoder-only LLM      │
  │ pooled_projections kwarg  │ NONE — Gemma has no pooled output.   │
  │                           │ Replaced by encoder_attention_mask.  │
  │ AutoencoderKL, 8x VAE     │ AutoencoderDC, 32x VAE (DC-AE)        │
  │ scaling_factor+shift      │ scaling_factor only (no shift)       │
  │ no text attention mask    │ REQUIRES encoder_attention_mask      │
  │ in_channels = 16          │ in_channels = 32                     │
  └───────────────────────────┴──────────────────────────────────────┘

Everywhere the SD3 wrapper threaded `pooled_embeddings` through the
pipeline (encode_prompt → sampler → transformer kwargs), this wrapper
threads `attention_mask` instead. That's the one structural change that
ripples through every method below.

Design rule unchanged from the SD3 version: Load once → patch freely →
weights stay frozen unless explicitly unfrozen.
"""

import torch
from diffusers import SanaPipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image


class SanaPipelineWrapper:

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = device

        self.pipe              = None
        self.tokenizer         = None
        self.text_encoder      = None
        self.transformer       = None
        self.vae               = None
        self.scheduler         = None
        self.vae_scale_factor  = 32   # DC-AE default; refined in load()

        self.q1_analyzer        = None   # kept for parity with SD3 wrapper

    # ================================================================== #
    #  LOAD                                                               #
    # ================================================================== #

    def load(self):
        model_id = self.cfg.get(
            "model_id", "Efficient-Large-Model/Sana_600M_512px_diffusers"
        )

        print(f"[Pipeline] Loading: {model_id}")

        self.pipe = SanaPipeline.from_pretrained(
            model_id,
            variant     = "fp16",          # drop this if your repo has no fp16 variant
            torch_dtype = torch.float16,
        ).to(self.device)

        # Per SANA's model card: the transformer can stay fp16, but the
        # text encoder (Gemma-2) and VAE (DC-AE) need bf16/fp32 to stay
        # numerically stable. This has no SD3 analog — SD3's CLIP/T5/VAE
        # are fine in fp16.
        self.pipe.vae.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)

        self.tokenizer    = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.transformer  = self.pipe.transformer
        self.vae          = self.pipe.vae

        # SanaPipeline.__init__ already computes this from the VAE's
        # encoder_block_out_channels (32 for DC-AE); reuse it rather than
        # hardcoding, in case a future SANA VAE changes the ratio.
        self.vae_scale_factor = self.pipe.vae_scale_factor

        # Swap to an explicit flow-matching scheduler — the same move
        # pipeline_wrapper.py makes for SD3. SANA ships with
        # DPMSolverMultistepScheduler by default, but UGILE's math
        # (Tweedie's potential, the geodesic escape step) is written for
        # a continuous flow-matching sigma schedule, so we re-instantiate
        # the same FlowMatchEulerDiscreteScheduler class SD3 uses here too.
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
        block list (`self.transformer.transformer_blocks`). SANA exposes
        the same attribute name on SanaTransformer2DModel, so this would
        work unmodified if/when that analyzer gets wired in here. Left
        as a no-op stub since it isn't part of the active UGILE path.
        """
        pass

    # ================================================================== #
    #  PROMPT ENCODING                                                    #
    # ================================================================== #

    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """
        Returns (prompt_embeds, attention_mask) — uncond+cond concatenated
        on dim 0, the same shape convention as SD3's
        (prompt_embeds, pooled_prompt_embeds) — except the second tensor
        here is an attention MASK, not a pooled vector (Gemma has no
        pooled output for SANA to use).

        This delegates to the pipeline's own `encode_prompt` rather than
        reimplementing Gemma tokenization by hand the way the SD3 wrapper
        reimplements CLIP encoding. Reason: SANA's official pipeline can
        optionally prepend a "complex human instruction" (CHI) template
        to the prompt before encoding, then slices the resulting
        embeddings with a specific index pattern. That logic is
        load-bearing for output quality and easy to get subtly wrong by
        hand, so it's safer to reuse the tested implementation.

        We deliberately do NOT pass `complex_human_instruction` here, so
        CHI stays off and the prompt is encoded as literally written —
        mirroring SD3's behavior, which never rewrites/enhances the
        prompt either. Pass `complex_human_instruction=[...]` into
        `self.pipe.encode_prompt(...)` below if you want SANA's official
        prompt-enhancement trick.
        """
        do_cfg = self.cfg.get("flow", {}).get("guidance_scale", 4.5) > 1.0

        with torch.no_grad():
            (
                cond_emb, cond_mask,
                uncond_emb, uncond_mask,
            ) = self.pipe.encode_prompt(
                prompt,
                do_classifier_free_guidance = do_cfg,
                negative_prompt              = negative_prompt,
                device                       = self.device,
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
        latents   = torch.randn(
            (1, self.transformer.config.in_channels,
             H // self.vae_scale_factor, W // self.vae_scale_factor),
            generator = generator,
            device    = self.device,
            dtype     = torch.float32,   # matches SANA's own pipeline (fp32 noise init)
        )
        # SANA, like SD3, operates directly on unscaled N(0,I) noise —
        # no pre-scaling here; the VAE decode handles unscaling.
        return latents

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        # DC-AE decode convention (simpler than SD3's VAE — no shift_factor):
        #   pixels = VAE.decode(z / scaling_factor)
        scaling_factor = self.vae.config.scaling_factor

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

        latents = latents.to(dtype=self.vae.dtype) / scaling_factor

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype("uint8")
        from PIL import Image as PILImage
        return PILImage.fromarray(image)

    # ================================================================== #
    #  GENERATE  (standard path — parity with SD3's _generate_standard)   #
    # ================================================================== #

    def generate(self, prompt: str, negative_prompt: str = "", seed: int = 42) -> Image.Image:
        """Standard diffusers pipeline path — unchanged behaviour."""
        f_cfg     = self.cfg.get("flow", {})
        gen_cfg   = self.cfg.get("generation", {})
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print("[Pipeline] Running standard generation path.")
        result = self.pipe(
            prompt              = prompt,
            negative_prompt     = negative_prompt,
            height              = gen_cfg.get("height", 512),
            width               = gen_cfg.get("width",  512),
            num_inference_steps = f_cfg.get("num_steps", 20),
            guidance_scale      = f_cfg.get("guidance_scale", 4.5),
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