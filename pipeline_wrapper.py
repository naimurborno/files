"""
Md. Naimur Asif Borno
pipeline_wrapper.py  (Flow Matching edition)
--------------------------------------------
Loads SD3 / FLUX via Diffusers and exposes each component individually.

Architecture shift from SD 1.5 → SD3:
  ┌─────────────────────────┬────────────────────────────────────┐
  │       SD 1.5 (DDPM)     │       SD3 / FLUX (Flow Matching)   │
  ├─────────────────────────┼────────────────────────────────────┤
  │ UNet (conv-heavy)       │ MMDiT / DiT (pure transformer)     │
  │ CLIPTokenizer (77 tok)  │ CLIP-L + CLIP-G + T5-XXL (3 enc.) │
  │ DDIMScheduler           │ FlowMatchEulerDiscreteScheduler    │
  │ ε-prediction            │ v-prediction (velocity field)      │
  │ t ∈ {0..1000}  discrete │ t ∈ [0.0, 1.0]  continuous        │
  └─────────────────────────┴────────────────────────────────────┘

Design rule:
    Load once → patch freely → weights stay frozen unless you explicitly unfreeze.
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image

# from q1_entropy_analysis import Q1EntropyAnalyzer   # ← Q1 addition
from stochastic_sampler  import StochasticVelocitySampler


class SD3PipelineWrapper:

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = device

        self.pipe              = None
        self.tokenizer         = None
        self.tokenizer_2       = None
        self.tokenizer_3       = None
        self.text_encoder      = None
        self.text_encoder_2    = None
        self.text_encoder_3    = None
        self.transformer       = None
        self.vae               = None
        self.scheduler         = None

        self.q1_analyzer       = None   # ← Q1 addition

    # ================================================================== #
    #  LOAD                                                               #
    # ================================================================== #

    def load(self):
        model_id = self.cfg.get("model_id", "stabilityai/stable-diffusion-3-medium-diffusers")

        print(f"[Pipeline] Loading: {model_id}")

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype      = torch.float16,
            text_encoder_3   = None,
            tokenizer_3      = None,
        ).to(self.device)
        # self.pipe.enable_model_cpu_offload()

        self.tokenizer      = self.pipe.tokenizer
        self.tokenizer_2    = self.pipe.tokenizer_2
        self.tokenizer_3    = None
        self.text_encoder   = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        self.text_encoder_3 = None
        self.transformer    = self.pipe.transformer
        self.vae            = self.pipe.vae

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

        self._freeze_all()
        print("[Pipeline] Loaded & frozen ✔")
        self._print_memory()

    # ================================================================== #
    #  PATCH                                                              #
    # ================================================================== #

    def patch(self):
        """
        Q1: Register entropy analyzer hooks on all MMDiT blocks.
        Hooks capture text Q and image K projections + block output hidden states.
        """
        self.q1_analyzer = Q1EntropyAnalyzer(self.transformer)  # ← Q1
        self.q1_analyzer.register_hooks()                        # ← Q1
        print(f"[Pipeline] Q1 hooks registered on "
              f"{len(self.transformer.transformer_blocks)} blocks.")

    # ================================================================== #
    #  PROMPT ENCODING                                                    #
    # ================================================================== #

    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        do_cfg = self.cfg.get("flow", {}).get("guidance_scale", 7.5) > 1.0

        with torch.no_grad():
            cond_emb, cond_pooled = self._encode_prompt_single(prompt)

            if do_cfg:
                uncond_emb, uncond_pooled = self._encode_prompt_single(negative_prompt)
                prompt_embeds        = torch.cat([uncond_emb,    cond_emb])
                pooled_prompt_embeds = torch.cat([uncond_pooled, cond_pooled])
            else:
                prompt_embeds        = cond_emb
                pooled_prompt_embeds = cond_pooled

        return prompt_embeds, pooled_prompt_embeds

    def _encode_prompt_single(self, text: str):
        max_len_clip = self.tokenizer.model_max_length

        def _tok(tokenizer, text, max_length):
            return tokenizer(
                text,
                padding        = "max_length",
                max_length     = max_length,
                truncation     = True,
                return_tensors = "pt",
            ).input_ids.to(self.device)

        ids_l    = _tok(self.tokenizer, text, max_len_clip)
        out_l    = self.text_encoder(ids_l, output_hidden_states=True)
        emb_l    = out_l.hidden_states[-2]
        pooled_l = out_l.hidden_states[-1][:, -1, :]

        ids_g    = _tok(self.tokenizer_2, text, max_len_clip)
        out_g    = self.text_encoder_2(ids_g, output_hidden_states=True)
        emb_g    = out_g.hidden_states[-2]
        pooled_g = out_g.text_embeds

        D_joint      = 4096
        emb_l_pad    = torch.nn.functional.pad(emb_l, (0, D_joint - emb_l.shape[-1]))
        emb_g_pad    = torch.nn.functional.pad(emb_g, (0, D_joint - emb_g.shape[-1]))
        emb_t5_zero  = torch.zeros(1, 256, D_joint, dtype=emb_l.dtype, device=self.device)

        emb_all = torch.cat([emb_l_pad, emb_g_pad, emb_t5_zero], dim=1)
        pooled  = torch.cat([pooled_l, pooled_g], dim=-1)

        return emb_all, pooled

    # ================================================================== #
    #  LATENT HELPERS                                                     #
    # ================================================================== #

    def get_initial_latents(self, seed: int = 42) -> torch.Tensor:
        gen_cfg   = self.cfg.get("generation", {})
        H         = gen_cfg.get("height", 512)
        W         = gen_cfg.get("width",  512)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents   = torch.randn(
            (1, self.transformer.config.in_channels, H // 8, W // 8),
            generator = generator,
            device    = self.device,
            dtype     = torch.float16,
        )
        # SD3: raw N(0,I) — no pre-scaling. The transformer operates
        # directly on unscaled noise; VAE decode handles unscaling.
        return latents

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        # SD3 VAE decode convention:
        #   VAE expects: latents / scaling_factor + shift_factor
        # (encoder does the inverse: (x - shift) * scale)
        scaling_factor = self.vae.config.scaling_factor          # ~1.5305
        shift_factor   = getattr(self.vae.config, 'shift_factor', 0.0609)

        # guard: abort early if latents are already NaN/Inf
        if not torch.isfinite(latents).all():
            print("[Pipeline] WARNING: latents contain NaN/Inf before decode — "
                  "returning blank image")
            from PIL import Image as PILImage
            gen_cfg = self.cfg.get("generation", {})
            return PILImage.fromarray(
                __import__("numpy").zeros(
                    (gen_cfg.get("height", 512), gen_cfg.get("width", 512), 3),
                    dtype="uint8"
                )
            )

        latents = latents.to(dtype=torch.float32)               # fp32 for stable decode
        latents = latents / scaling_factor + shift_factor        # correct unscaling
        latents = latents.to(dtype=torch.float16)               # back to fp16 for VAE

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype("uint8")
        from PIL import Image as PILImage
        return PILImage.fromarray(image)

    # ================================================================== #
    #  GENERATE                                                           #
    # ================================================================== #

    def generate(self, prompt: str, negative_prompt: str = "", seed: int = 42) -> Image.Image:
        """
        Top-level entry point. Reads `stochastic_sampler.enabled` from cfg
        and routes accordingly:

          stochastic_sampler:
            enabled: true          ← uses StochasticVelocitySampler
            K:         5
            sigma_max: 1.0
            lam:       0.5
            alpha:     1.0

          stochastic_sampler:
            enabled: false         ← uses pipe() directly (standard path)
        """
        ss_cfg  = self.cfg.get("stochastic_sampler", {})
        enabled = ss_cfg.get("enabled", False)

        if enabled:
            return self._generate_stochastic(prompt, negative_prompt, seed, ss_cfg)
        else:
            return self._generate_standard(prompt, negative_prompt, seed)

    def _generate_standard(self, prompt: str, negative_prompt: str, seed: int) -> Image.Image:
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
            num_inference_steps = f_cfg.get("num_steps", 50),
            guidance_scale      = f_cfg.get("guidance_scale", 7.5),
            generator           = generator,
        )
        return result.images[0]

    def _generate_stochastic(
        self, prompt: str, negative_prompt: str, seed: int, ss_cfg: dict
    ) -> Image.Image:
        """Stochastic velocity branching path."""
        print(f"[Pipeline] Running stochastic sampler "
              f"(K={ss_cfg.get('K', 5)}, "
              f"σ_max={ss_cfg.get('sigma_max', 1.0)}, "
              f"λ={ss_cfg.get('lam', 0.5)}, "
              f"α={ss_cfg.get('alpha', 1.0)}).")

        latents                    = self.get_initial_latents(seed)
        text_embeddings, pooled    = self.encode_prompt(prompt, negative_prompt)
        
        sampler = StochasticVelocitySampler(
            unet      = self.transformer,
            scheduler = self.scheduler,
            cfg       = self.cfg,
            device    = self.device,
            K         = ss_cfg.get("K",         5),
            sigma_max = ss_cfg.get("sigma_max", 1.0),
            lam       = ss_cfg.get("lam",       0.5),
            alpha     = ss_cfg.get("alpha",     1.0),
        )

        result  = sampler.run(latents, text_embeddings, pooled)

        # save trajectory log for plot_trajectory.py
        # import json
        # log_path = self.cfg.get("output", "output.png").replace(".png", "_traj.json")
        # with open(log_path, "w") as f:
        #     json.dump(result["chosen_log"], f, indent=2)
        # print(f"[Pipeline] Trajectory log saved → {log_path}")

        return self.decode_latents(result["latents"])

    # ================================================================== #
    #  INTERNALS                                                          #
    # ================================================================== #

    def _freeze_all(self):
        for model in [self.text_encoder, self.text_encoder_2,
                      self.text_encoder_3, self.transformer, self.vae]:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = False

    def _print_memory(self):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            print(f"[Pipeline] GPU memory used: {alloc:.2f} GB")