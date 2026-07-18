"""
SPARKE Pipeline with YAML Prompt Loading + Dual GPU Parallel Execution

Usage:
    python sparke_parallel.py --config config.yaml

Config YAML format:
    model_id: "stabilityai/stable-diffusion-2-1"
    prompts_file: "prompts.yaml"
    output_dir: "./outputs"
    num_inference_steps: 50
    guidance_scale: 7.5
    height: 768
    width: 768

    # SPARKE settings
    sparke_enabled: true
    criteria_guidance_scale: 0.03
    guidance_freq: 10
    sigma_img: 0.8
    sigma_text: 0.3

    # Generation settings
    num_images_per_prompt: 1
    seed: 42
    dtype: "float16"

Prompts YAML format:
    prompts:
      - "a photo of a dog in a park"
      - "a photo of a cat on a sofa"
      - "a portrait of a wizard"
      - ...
"""

import os
import sys
import yaml
import argparse
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg, retrieve_timesteps
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging, is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


# =============================================================================
# RKE GUIDANCE (SPARKE Core)
# =============================================================================

class RKEGuidedSampling:
    """RKE (Renyi Kernel Entropy) guided sampling for diversity."""

    def __init__(
        self,
        device: str = "cuda",
        sigma_img: float = 0.8,
        sigma_text: float = 0.3,
        text_kernel: str = "rbf",
    ):
        self.device = device
        self.sigma_img = sigma_img
        self.sigma_text = sigma_text
        self.text_kernel = text_kernel
        self.latent_history: List[torch.Tensor] = []
        self.text_embed_history: List[torch.Tensor] = []
        self.noise_history: List[torch.Tensor] = []

    def reset(self):
        self.latent_history = []
        self.text_embed_history = []
        self.noise_history = []

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)
        dist_sq = x_norm + y_norm.T - 2.0 * torch.mm(x, y.T)
        return torch.exp(-dist_sq / (2.0 * sigma ** 2))

    def cosine_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.normalize(x.view(x.shape[0], -1), p=2, dim=1)
        y = torch.nn.functional.normalize(y.view(y.shape[0], -1), p=2, dim=1)
        return torch.mm(x, y.T)

    def compute_text_embeddings(self, prompts: List[str], tokenizer, text_encoder) -> torch.Tensor:
        with torch.no_grad():
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            text_embeds = text_encoder(text_inputs.input_ids)[0]
            text_embeds = text_embeds.mean(dim=1)
        return text_embeds

    def compute_conditional_rke_gradient(
        self,
        current_latents: torch.Tensor,
        current_text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if len(self.latent_history) == 0:
            return torch.zeros_like(current_latents)

        batch_size = current_latents.shape[0]
        current_flat = current_latents.view(batch_size, -1)
        hist_latents = torch.cat([h.to(self.device).view(h.shape[0], -1) for h in self.latent_history], dim=0)
        hist_texts = torch.cat([t.to(self.device) for t in self.text_embed_history], dim=0)

        gradients = []
        for b in range(batch_size):
            z_n = current_flat[b:b+1]
            y_n = current_text_embeds[b:b+1]
            k_z = self.rbf_kernel(hist_latents, z_n, self.sigma_img).squeeze(-1)
            if self.text_kernel == "rbf":
                k_y = self.rbf_kernel(hist_texts, y_n, self.sigma_text).squeeze(-1)
            else:
                k_y = self.cosine_kernel(hist_texts, y_n).squeeze(-1)
            diff = hist_latents - z_n
            k_grad = k_z.unsqueeze(-1) * diff / (self.sigma_img ** 2)
            weights = (k_y ** 2).unsqueeze(-1)
            grad = (weights * k_grad * k_z.unsqueeze(-1)).sum(dim=0)
            grad = grad.view_as(current_latents[b])
            gradients.append(grad)
        return torch.stack(gradients, dim=0)

    def cond_fn(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        index: int,
        noise_pred: torch.Tensor,
        extra_step_kwargs: dict,
        criteria_guidance_scale: float,
        prompt,
        clip_for_guidance,
        regularize: bool,
        regularize_weight: float,
        F_M,
        F_T,
        F_M_real,
        F_T_real,
        beta,
    ):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        text_embeds = self.compute_text_embeddings(
            prompts, clip_for_guidance.tokenizer, clip_for_guidance.text_encoder
        )

        with torch.enable_grad():
            latents_input = latents.detach().requires_grad_(True)
            grad = self.compute_conditional_rke_gradient(latents_input, text_embeds)

        grads = criteria_guidance_scale * grad
        self.latent_history.append(latents.detach().cpu())
        self.text_embed_history.append(text_embeds.detach().cpu())
        self.noise_history.append(noise_pred.detach().cpu())
        return grads, F_M, F_T


# =============================================================================
# SPARKE PIPELINE
# =============================================================================

class SPARKEGuidedStableDiffusionPipeline(DiffusionPipeline, StableDiffusionMixin):
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()
        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=safety_checker,
            feature_extractor=feature_extractor, image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def encode_prompt(
        self, prompt, device, num_images_per_prompt,
        do_classifier_free_guidance, negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True)
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        prompt_embeds_dtype = self.text_encoder.dtype if self.text_encoder is not None else self.unet.dtype
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(f"`negative_prompt` batch size mismatch")
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens, padding="max_length",
                max_length=max_length, truncation=True, return_tensors="pt",
            )
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=attention_mask)
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            return image, None
        if torch.is_tensor(image):
            feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
        else:
            feature_extractor_input = self.image_processor.numpy_to_pil(image)
        safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
        image, has_nsfw = self.safety_checker(images=image, clip_input=safety_checker_input.pixel_values.to(dtype))
        return image, has_nsfw

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra = {}
        if accepts_eta:
            extra["eta"] = eta
        accepts_gen = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_gen:
            extra["generator"] = generator
        return extra

    def check_inputs(self, prompt, height, width, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"height and width must be divisible by 8, got {height}x{width}")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both prompt and prompt_embeds")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either prompt or prompt_embeds")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both negative_prompt and negative_prompt_embeds")

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt=None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator=None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_kernels: bool = False,
        rke_guided_sampler: Optional[RKEGuidedSampling] = None,
        criteria: str = "vscore_clip",
        criteria_guidance_scale: float = 0.0,
        guidance_freq: int = 1,
        clip_for_guidance=None,
        regularize: bool = False,
        regularize_weight: float = 0.0,
        F_M=None, F_T=None, F_M_real=None, F_T_real=None, beta=None,
        logger_=None,
        **kwargs,
    ):
        if not height or not width:
            height = self.unet.config.sample_size * self.vae_scale_factor
            width = self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, negative_prompt, prompt_embeds, negative_prompt_embeds)
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_cfg = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, device, num_images_per_prompt, do_cfg, negative_prompt,
            prompt_embeds, negative_prompt_embeds, clip_skip=clip_skip,
        )
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, latents)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if rke_guided_sampler is not None:
            rke_guided_sampler.reset()

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale)

                if rke_guided_sampler is not None and criteria_guidance_scale != 0:
                    if i % guidance_freq == 0 and criteria == "vscore_clip":
                        grads, F_M, F_T = rke_guided_sampler.cond_fn(
                            latents=latents, timestep=t, index=i, noise_pred=noise_pred,
                            extra_step_kwargs=extra_step_kwargs,
                            criteria_guidance_scale=criteria_guidance_scale,
                            prompt=prompt, clip_for_guidance=clip_for_guidance,
                            regularize=regularize, regularize_weight=regularize_weight,
                            F_M=F_M, F_T=F_T, F_M_real=F_M_real, F_T_real=F_T_real, beta=beta,
                        )
                        if torch.isnan(grads).any() or torch.isinf(grads).any():
                            if logger_ is not None:
                                logger_.info("Skipping gradient update due to NaN/Inf")
                        else:
                            latents = latents + grads

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    cb_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    cb_out = callback_on_step_end(self, i, t, cb_kwargs)
                    latents = cb_out.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image, has_nsfw = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw = None

        do_denormalize = [True] * image.shape[0] if has_nsfw is None else [not x for x in has_nsfw]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        self.maybe_free_model_hooks()

        if return_kernels:
            return (image, F_M, F_T)
        if not return_dict:
            return (image, has_nsfw)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw)


# =============================================================================
# GENERATION WORKER (runs on a single GPU)
# =============================================================================

def generate_on_gpu(
    gpu_id: int,
    prompts: List[str],
    global_indices: List[int],
    config: dict,
    output_dir: str,
):
    """Generate images for a subset of prompts on a specific GPU."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = f"cuda:0"  # After CUDA_VISIBLE_DEVICES, this maps to the selected GPU

    dtype_str = config.get("dtype", "float16")
    dtype = torch.float16 if dtype_str == "float16" else torch.float32

    print(f"[GPU {gpu_id}] Loading model: {config['model_id']}")

    # Load base pipeline
    base_pipe = StableDiffusionPipeline.from_pretrained(
        config["model_id"],
        torch_dtype=dtype,
        use_safetensors=True,
    )
    base_pipe = base_pipe.to(device)

    # Use DPM Solver as per SPARKE paper
    base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_pipe.scheduler.config)

    # Wrap with SPARKE
    pipe = SPARKEGuidedStableDiffusionPipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        unet=base_pipe.unet,
        scheduler=base_pipe.scheduler,
        safety_checker=base_pipe.safety_checker,
        feature_extractor=base_pipe.feature_extractor,
        image_encoder=getattr(base_pipe, "image_encoder", None),
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # Setup SPARKE sampler
    sparke_enabled = config.get("sparke_enabled", True)
    rke_sampler = None
    clip_wrapper = None

    if sparke_enabled:
        rke_sampler = RKEGuidedSampling(
            device=device,
            sigma_img=config.get("sigma_img", 0.8),
            sigma_text=config.get("sigma_text", 0.3),
            text_kernel=config.get("text_kernel", "rbf"),
        )
        clip_wrapper = type("CLIPWrapper", (), {
            "tokenizer": pipe.tokenizer,
            "text_encoder": pipe.text_encoder,
        })()

    # Prepare generator
    seed = config.get("seed")
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    os.makedirs(output_dir, exist_ok=True)

    height = config.get("height", 768)
    width = config.get("width", 768)
    num_steps = config.get("num_inference_steps", 50)
    guidance_scale = config.get("guidance_scale", 7.5)
    num_images_per_prompt = config.get("num_images_per_prompt", 1)
    negative_prompt = config.get("negative_prompt", "")

    criteria_guidance_scale = config.get("criteria_guidance_scale", 0.03) if sparke_enabled else 0.0
    guidance_freq = config.get("guidance_freq", 10)

    print(f"[GPU {gpu_id}] Generating {len(prompts)} images...")

    for prompt, global_idx in zip(prompts, global_indices):
        print(f"[GPU {gpu_id}] Prompt {global_idx}: {prompt[:60]}...")

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            rke_guided_sampler=rke_sampler,
            criteria="vscore_clip",
            criteria_guidance_scale=criteria_guidance_scale,
            guidance_freq=guidance_freq,
            clip_for_guidance=clip_wrapper,
        )

        for img_idx, img in enumerate(output.images):
            # Filename uses ORIGINAL global index from the undivided prompt list
            # Format: 1.png, 2.png, 3.png, ... (1-indexed to match your existing workflow)
            filename = f"{global_idx + 1}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            print(f"[GPU {gpu_id}] Saved {filepath}")

    print(f"[GPU {gpu_id}] Done! Generated {len(prompts)} images.")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_parallel_generation(config_path: str):
    """
    Read config, split prompts across 2 GPUs, run generation in parallel.

    Args:
        config_path: Path to YAML config file
    """

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    prompts_file = config.get("prompts_file", "prompts.yaml")
    output_dir = config.get("output_dir", "./outputs")

    # Load prompts
    with open(prompts_file, "r") as f:
        prompts_data = yaml.safe_load(f) or {}

    if isinstance(prompts_data, list):
        all_prompts = prompts_data
    elif isinstance(prompts_data, dict):
        all_prompts = prompts_data.get("prompts", [])
    else:
        all_prompts = []

    if not all_prompts or not isinstance(all_prompts, list):
        print("Error: Could not find a list of prompts in the prompts file.")
        sys.exit(1)

    total = len(all_prompts)
    midpoint = total // 2

    gpu0_prompts = all_prompts[:midpoint]
    gpu1_prompts = all_prompts[midpoint:]
    gpu0_indices = list(range(0, midpoint))
    gpu1_indices = list(range(midpoint, total))

    print(f"Total prompts: {total}")
    print(f"GPU 0: prompts {gpu0_indices[0]+1} to {gpu0_indices[-1]+1} ({len(gpu0_prompts)} items)")
    print(f"GPU 1: prompts {gpu1_indices[0]+1} to {gpu1_indices[-1]+1} ({len(gpu1_prompts)} items)")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Launch parallel processes
    def stream_reader(pipe, prefix):
        for line in iter(pipe.readline, ""):
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()
        pipe.close()

    env0 = os.environ.copy()
    env0["CUDA_VISIBLE_DEVICES"] = "0"
    env0["PYTHONUNBUFFERED"] = "1"

    env1 = os.environ.copy()
    env1["CUDA_VISIBLE_DEVICES"] = "1"
    env1["PYTHONUNBUFFERED"] = "1"

    # We need to run the worker function in a subprocess.
    # Since functions can't be pickled for multiprocessing with spawn,
    # we write a small script for each GPU and execute it.

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Write GPU 0 script
    gpu0_script = f"""
import sys
sys.path.insert(0, "{script_dir}")
from sparke_parallel import generate_on_gpu

prompts = {repr(gpu0_prompts)}
indices = {repr(gpu0_indices)}

import yaml
with open("{config_path}", "r") as f:
    config = yaml.safe_load(f)

generate_on_gpu(0, prompts, indices, config, "{output_dir}")
"""

    gpu0_script_path = os.path.join(output_dir, "_gpu0_worker.py")
    with open(gpu0_script_path, "w") as f:
        f.write(gpu0_script)

    # Write GPU 1 script
    gpu1_script = f"""
import sys
sys.path.insert(0, "{script_dir}")
from sparke_parallel import generate_on_gpu

prompts = {repr(gpu1_prompts)}
indices = {repr(gpu1_indices)}

import yaml
with open("{config_path}", "r") as f:
    config = yaml.safe_load(f)

generate_on_gpu(1, prompts, indices, config, "{output_dir}")
"""

    gpu1_script_path = os.path.join(output_dir, "_gpu1_worker.py")
    with open(gpu1_script_path, "w") as f:
        f.write(gpu1_script)

    # Execute
    print("Launching GPU workers...")

    p0 = subprocess.Popen(
        [sys.executable, gpu0_script_path],
        env=env0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    p1 = subprocess.Popen(
        [sys.executable, gpu1_script_path],
        env=env1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    t0 = threading.Thread(target=stream_reader, args=(p0.stdout, "GPU 0"))
    t1 = threading.Thread(target=stream_reader, args=(p1.stdout, "GPU 1"))
    t0.start()
    t1.start()

    start_time = time.time()
    while p0.poll() is None or p1.poll() is None:
        time.sleep(5)
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)
        print(f"[ORCHESTRATOR] Elapsed: {mins}m {secs}s | GPU0: {'done' if p0.poll() is not None else 'running'} | GPU1: {'done' if p1.poll() is not None else 'running'}")
        sys.stdout.flush()

    t0.join()
    t1.join()

    # Cleanup temp scripts
    os.remove(gpu0_script_path)
    os.remove(gpu1_script_path)

    print(f"\n{'='*60}")
    print(f"All {total} prompts finished!")
    print(f"Images saved to: {output_dir}")
    print(f"Naming: 1.png, 2.png, ..., {total}.png (based on original prompt order)")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPARKE Parallel Image Generation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    run_parallel_generation(args.config)
