"""
inference.py  —  Multi-Model Inference
---------------------------------------
Supports: FLUX, Stable Diffusion (1.x/2.x/3.x), SANA, Lumina, Janus,
          CogVideo (video), Mini-Gemini (VLM / image understanding)

Usage:
    python inference.py --config config.yaml
    python inference.py --config config.yaml --prompt "a red apple" --output out.png

Model is selected via config.yaml:
    model_name: "flux"          # flux | sd | sd3 | sd3_divin | sana | lumina | janus | cogvideo | mini_gemini

Multi-seed (sd3_divin only):
    Add a `seeds` list to config.yaml. The model is loaded once; per seed, DivIn
    runs a Langevin walk on `divin.gen_num` initial latents (see the `divin:`
    config block) and denoises each into a diverse image. Per-branch results
    are reported at the end.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# DivIn import — graceful fallback if file not yet on path
try:
    from divin_sampler import run_sd3_divin as _run_sd3_divin
    _DIVIN_AVAILABLE = True
except ImportError:
    _DIVIN_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════ #
#  Registry: model_name → loader function                                #
# ══════════════════════════════════════════════════════════════════════ #

MODEL_REGISTRY = {}

def register_model(name):
    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


# ══════════════════════════════════════════════════════════════════════ #
#  CLI                                                                    #
# ══════════════════════════════════════════════════════════════════════ #

def parse_args():
    p = argparse.ArgumentParser(description="Multi-Model Inference")
    p.add_argument("--config",  type=str, default="config.yaml")
    p.add_argument("--prompt",  type=str, default=None, help="Overrides config prompt")
    p.add_argument("--output",  type=str, default=None, help="Overrides config output path")
    p.add_argument("--seed",    type=int, default=None, help="Overrides config seed (single seed)")
    p.add_argument("--device",  type=str, default=None, help="Overrides config device")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════ #
#  Config                                                                 #
# ══════════════════════════════════════════════════════════════════════ #

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompts_from_config(cfg: dict) -> list:
    """If cfg['prompts_file'] is set, load prompts from that YAML file
    (expects a top-level `prompts:` list). Otherwise fall back to the
    inline cfg['prompts'] list."""
    prompts_file = cfg.get("prompts_file")
    if prompts_file:
        with open(prompts_file) as f:
            pdata = yaml.safe_load(f)
        prompts = pdata.get("prompts", []) if isinstance(pdata, dict) else pdata
        if not prompts:
            raise ValueError(f"No `prompts:` list found in {prompts_file}")
        print(f"[Prompts] Loaded {len(prompts)} prompt(s) from {prompts_file}")
        return prompts
    return cfg.get("prompts", ["a photo of a cat"])


def resolve_args(args, cfg: dict) -> dict:
    """Merge CLI args on top of config. CLI always wins."""
    device = args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Single seed: CLI --seed > config `seed`
    single_seed = args.seed or cfg.get("seed", 42)

    # Multi-seed list: CLI --seed overrides the whole list to [seed]
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = cfg.get("seeds") or [single_seed]

    return {
        "model_name":      cfg.get("model_name", "sd3"),
        "model_id":        cfg.get("model_id", "stabilityai/stable-diffusion-3-medium-diffusers"),
        "prompt":          args.prompt or load_prompts_from_config(cfg)[0],
        "negative_prompt": cfg.get("negative_prompt", "blurry, low quality, ugly, deformed"),
        "output":          args.output or cfg.get("output", "output.png"),
        "height":          cfg.get("generation", {}).get("height", 512),
        "width":           cfg.get("generation", {}).get("width",  512),
        "num_steps":       cfg.get("flow", {}).get("num_steps", 50),
        "guidance_scale":  cfg.get("flow", {}).get("guidance_scale", 7.5),
        "solver":          cfg.get("flow", {}).get("solver", "euler"),
        "seed":            single_seed,   # kept for non-ONLB runners
        "seeds":           seeds,         # used by sd3_onlb multi-seed loop
        "device":          device,
        # model-specific extras (pass through wholesale)
        "model_kwargs":    cfg.get("model_kwargs", {}),
    }


# ══════════════════════════════════════════════════════════════════════ #
#  Seed                                                                   #
# ══════════════════════════════════════════════════════════════════════ #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════ #
#  Shared decode + save                                                   #
# ══════════════════════════════════════════════════════════════════════ #

def save_image(image, output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"\n[Done] Saved → {out}")


# ══════════════════════════════════════════════════════════════════════ #
#  MODEL LOADERS                                                          #
# ══════════════════════════════════════════════════════════════════════ #

# ─── FLUX ─────────────────────────────────────────────────────────────
@register_model("flux")
def run_flux(opts: dict):
    """
    FLUX.1 (Black Forest Labs) — Flow Matching DiT
    Recommended model_id: "black-forest-labs/FLUX.1-dev"
                       or "black-forest-labs/FLUX.1-schnell"
    """
    from diffusers import FluxPipeline

    print(f"[FLUX] Loading {opts['model_id']}...")
    pipe = FluxPipeline.from_pretrained(
        opts["model_id"],
        torch_dtype=torch.bfloat16,
        **opts["model_kwargs"],
    ).to(opts["device"])

    generator = torch.Generator(device=opts["device"]).manual_seed(opts["seed"])

    print("[FLUX] Generating...")
    result = pipe(
        prompt          = opts["prompt"],
        height          = opts["height"],
        width           = opts["width"],
        num_inference_steps = opts["num_steps"],
        guidance_scale  = opts["guidance_scale"],
        generator       = generator,
    )
    save_image(result.images[0], opts["output"])


# ─── Stable Diffusion 1.x / 2.x ──────────────────────────────────────
@register_model("sd")
def run_sd(opts: dict):
    """
    Stable Diffusion 1.x or 2.x (DDPM / DDIM)
    Recommended model_id: "runwayml/stable-diffusion-v1-5"
                       or "stabilityai/stable-diffusion-2-1"
    """
    from diffusers import StableDiffusionPipeline

    print(f"[SD] Loading {opts['model_id']}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        opts["model_id"],
        torch_dtype=torch.float16,
        **opts["model_kwargs"],
    ).to(opts["device"])

    generator = torch.Generator(device=opts["device"]).manual_seed(opts["seed"])

    print("[SD] Generating...")
    result = pipe(
        prompt              = opts["prompt"],
        negative_prompt     = opts["negative_prompt"],
        height              = opts["height"],
        width               = opts["width"],
        num_inference_steps = opts["num_steps"],
        guidance_scale      = opts["guidance_scale"],
        generator           = generator,
    )
    save_image(result.images[0], opts["output"])


# ─── Stable Diffusion 3 ───────────────────────────────────────────────
@register_model("sd3")
def run_sd3(opts: dict):
    """
    Stable Diffusion 3 (Flow Matching + MMDiT)
    Recommended model_id: "stabilityai/stable-diffusion-3-medium-diffusers"
                       or "stabilityai/stable-diffusion-3.5-large"
    """
    from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

    print(f"[SD3] Loading {opts['model_id']}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        opts["model_id"],
        torch_dtype    = torch.float16,
        text_encoder_3 = None,   # skip T5 to save VRAM
        tokenizer_3    = None,
        **opts["model_kwargs"],
    ).to(opts["device"])

    # Swap to explicit flow scheduler
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    generator = torch.Generator(device=opts["device"]).manual_seed(opts["seed"])

    print("[SD3] Generating...")
    result = pipe(
        prompt              = opts["prompt"],
        negative_prompt     = opts["negative_prompt"],
        height              = opts["height"],
        width               = opts["width"],
        num_inference_steps = opts["num_steps"],
        guidance_scale      = opts["guidance_scale"],
        generator           = generator,
    )
    save_image(result.images[0], opts["output"])


# ─── SANA ─────────────────────────────────────────────────────────────
@register_model("sana")
def run_sana(opts: dict):
    """
    SANA (NVIDIA) — Efficient linear-attention diffusion transformer
    Recommended model_id: "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
                       or "Efficient-Large-Model/Sana_600M_512px_diffusers"
    Requires: pip install diffusers>=0.31 transformers accelerate
    """
    from diffusers import SanaPipeline

    print(f"[SANA] Loading {opts['model_id']}...")
    pipe = SanaPipeline.from_pretrained(
        opts["model_id"],
        torch_dtype=torch.float16,
        **opts["model_kwargs"],
    ).to(opts["device"])

    generator = torch.Generator(device=opts["device"]).manual_seed(opts["seed"])

    print("[SANA] Generating...")
    result = pipe(
        prompt              = opts["prompt"],
        negative_prompt     = opts["negative_prompt"],
        height              = opts["height"],
        width               = opts["width"],
        num_inference_steps = opts["num_steps"],
        guidance_scale      = opts["guidance_scale"],
        generator           = generator,
    )
    save_image(result.images[0], opts["output"])


# ─── Lumina ───────────────────────────────────────────────────────────
@register_model("lumina")
def run_lumina(opts: dict):
    """
    Lumina-T2X (Alpha-VLLM) — Flow Matching image generation
    Recommended model_id: "Alpha-VLLM/Lumina-Next-SFT-diffusers"
    Requires: pip install diffusers transformers accelerate
    """
    from diffusers import LuminaText2ImgPipeline

    print(f"[Lumina] Loading {opts['model_id']}...")
    pipe = LuminaText2ImgPipeline.from_pretrained(
        opts["model_id"],
        torch_dtype=torch.bfloat16,
        **opts["model_kwargs"],
    ).to(opts["device"])

    generator = torch.Generator(device=opts["device"]).manual_seed(opts["seed"])

    print("[Lumina] Generating...")
    result = pipe(
        prompt              = opts["prompt"],
        negative_prompt     = opts["negative_prompt"],
        height              = opts["height"],
        width               = opts["width"],
        num_inference_steps = opts["num_steps"],
        guidance_scale      = opts["guidance_scale"],
        generator           = generator,
    )
    save_image(result.images[0], opts["output"])


# ─── Janus ────────────────────────────────────────────────────────────
@register_model("janus")
def run_janus(opts: dict):
    """
    Janus (DeepSeek) — Unified multimodal model (text-to-image via AR)
    Recommended model_id: "deepseek-ai/Janus-1.3B"
                       or "deepseek-ai/Janus-Pro-7B"
    Requires: pip install git+https://github.com/deepseek-ai/Janus.git transformers

    NOTE: Janus uses autoregressive image generation, not diffusion.
          The num_steps / guidance_scale config keys are unused here.
    """
    try:
        from janus.models import MultiModalityCausalLM, VLChatProcessor
    except ImportError:
        raise ImportError(
            "Janus not installed.\n"
            "  pip install git+https://github.com/deepseek-ai/Janus.git"
        )
    from transformers import AutoTokenizer
    from PIL import Image as PILImage

    print(f"[Janus] Loading {opts['model_id']}...")
    vl_chat_processor = VLChatProcessor.from_pretrained(opts["model_id"])
    tokenizer = vl_chat_processor.tokenizer

    model = MultiModalityCausalLM.from_pretrained(
        opts["model_id"], trust_remote_code=True
    )
    model = model.to(torch.bfloat16).to(opts["device"]).eval()

    # Build conversation
    conversation = [
        {"role": "User",      "content": opts["prompt"]},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations         = conversation,
        sft_format            = vl_chat_processor.sft_format,
        system_prompt         = "",
    )
    prompt_text = sft_format + vl_chat_processor.image_start_tag

    # Tokenize
    input_ids = torch.LongTensor(tokenizer.encode(prompt_text)).unsqueeze(0).to(opts["device"])

    # Generation config (uses model_kwargs for temperature / top_p / cfg_weight)
    cfg_weight  = opts["model_kwargs"].get("cfg_weight",    5.0)
    temperature = opts["model_kwargs"].get("temperature",   1.0)
    top_p       = opts["model_kwargs"].get("top_p",         1.0)
    image_token_num = opts["model_kwargs"].get("image_token_num", 576)
    patch_size      = opts["model_kwargs"].get("patch_size", 16)
    image_size      = opts["height"]

    print("[Janus] Generating image tokens...")
    torch.manual_seed(opts["seed"])
    tokens = model.generate_image(
        input_ids       = input_ids,
        width           = image_size // patch_size,
        height          = image_size // patch_size,
        cfg_weight      = cfg_weight,
        temperature     = temperature,
        top_p           = top_p,
    )

    # Decode image tokens → PIL
    images = model.decode_image_tokens(tokens, height=image_size, width=image_size)
    images = vl_chat_processor.process_images(images, model.config).to(
        dtype=torch.bfloat16, device=opts["device"]
    )
    with torch.no_grad():
        decoded = model.vision_model.decode(images)
    img_np = decoded[0].permute(1, 2, 0).cpu().float().numpy()
    img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype("uint8")
    image  = PILImage.fromarray(img_np)
    save_image(image, opts["output"])


# ─── CogVideo ─────────────────────────────────────────────────────────
@register_model("cogvideo")
def run_cogvideo(opts: dict):
    """
    CogVideoX (Zhipu AI) — Text-to-VIDEO generation
    Recommended model_id: "THUDM/CogVideoX-5b"
                       or "THUDM/CogVideoX-2b"
    Requires: pip install diffusers transformers accelerate imageio

    NOTE: Output is a VIDEO (.mp4), not an image.
          Set output path to e.g. "output.mp4" in config.
    """
    from diffusers import CogVideoXPipeline
    from diffusers.utils import export_to_video

    print(f"[CogVideo] Loading {opts['model_id']}...")
    pipe = CogVideoXPipeline.from_pretrained(
        opts["model_id"],
        torch_dtype=torch.bfloat16,
        **opts["model_kwargs"],
    ).to(opts["device"])

    pipe.enable_model_cpu_offload()   # recommended for 5B — saves VRAM
    pipe.vae.enable_tiling()

    generator = torch.Generator(device=opts["device"]).manual_seed(opts["seed"])

    num_frames  = opts["model_kwargs"].get("num_frames",   49)
    fps         = opts["model_kwargs"].get("fps",          8)

    print("[CogVideo] Generating video...")
    result = pipe(
        prompt              = opts["prompt"],
        num_videos_per_prompt = 1,
        num_inference_steps = opts["num_steps"],
        num_frames          = num_frames,
        guidance_scale      = opts["guidance_scale"],
        generator           = generator,
    )

    # Force .mp4 extension
    out_path = str(opts["output"])
    if not out_path.endswith(".mp4"):
        out_path = Path(out_path).with_suffix(".mp4")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.frames[0], str(out_path), fps=fps)
    print(f"\n[Done] Video saved → {out_path}")


# ─── Mini-Gemini ──────────────────────────────────────────────────────
@register_model("mini_gemini")
def run_mini_gemini(opts: dict):
    """
    Mini-Gemini (CUHK) — Multimodal VLM (image understanding / captioning)
    Recommended model_id: "YanweiLi/MGM-7B"
                       or "YanweiLi/MGM-13B"
    Requires: pip install git+https://github.com/dvlab-research/MiniGemini.git

    NOTE: Mini-Gemini is a VLM, not a T2I model.
          It can generate text descriptions from images, or answer questions about images.
          For IMAGE INPUT: set model_kwargs.image_path in config.yaml.
          For TEXT ONLY:   it will respond with a text caption / answer.

    Output is a .txt file with the model's response.
    """
    try:
        from minigemini import MiniGeminiForCausalLM, MiniGeminiConfig
        from minigemini.mm_utils import load_image, process_images
    except ImportError:
        raise ImportError(
            "Mini-Gemini not installed.\n"
            "  pip install git+https://github.com/dvlab-research/MiniGemini.git\n"
            "  Or use the transformers AutoModel path if available."
        )
    from transformers import AutoTokenizer

    print(f"[Mini-Gemini] Loading {opts['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(opts["model_id"], use_fast=False)
    model = MiniGeminiForCausalLM.from_pretrained(
        opts["model_id"],
        torch_dtype       = torch.float16,
        low_cpu_mem_usage = True,
    ).to(opts["device"]).eval()

    image_path = opts["model_kwargs"].get("image_path", None)
    images     = None
    if image_path:
        print(f"[Mini-Gemini] Loading image from {image_path}...")
        raw_image = load_image(image_path)
        images    = process_images([raw_image], model.get_vision_tower().image_processor,
                                   model.config).to(opts["device"], dtype=torch.float16)

    inputs = tokenizer(opts["prompt"], return_tensors="pt").to(opts["device"])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            images         = images,
            max_new_tokens = opts["model_kwargs"].get("max_new_tokens", 512),
            do_sample      = False,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n[Mini-Gemini] Response:\n{response}")

    out_path = str(opts["output"])
    if not out_path.endswith(".txt"):
        out_path = Path(out_path).with_suffix(".txt")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(response)
    print(f"[Done] Text saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════ #
#  MAIN                                                                   #
# ══════════════════════════════════════════════════════════════════════ #

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    opts = resolve_args(args, cfg)

    # make the raw cfg available to runners that need it (e.g. ONLB)
    opts["_cfg"] = cfg

    set_seed(opts["seed"])

    # Register DivIn runner if available
    if _DIVIN_AVAILABLE:
        MODEL_REGISTRY["sd3_divin"] = _run_sd3_divin

    model_name = opts["model_name"].lower().strip()

    print(f"[INFO] Model    : {opts['model_name']} ({opts['model_id']})")
    print(f"[INFO] Prompt   : {opts['prompt']}")
    print(f"[INFO] Steps    : {opts['num_steps']} | cfg={opts['guidance_scale']} | solver={opts['solver']}")
    print(f"[INFO] Device   : {opts['device']}")
    print(f"[INFO] Output   : {opts['output']}")

    if model_name in ("sd3_divin",):
        seeds = opts["seeds"]
        print(f"[INFO] Seeds    : {seeds}  ({len(seeds)} image(s) to generate)")

    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model_name '{model_name}'.\n"
            f"Supported: {supported}\n"
            f"Set model_name in config.yaml."
        )

    runner = MODEL_REGISTRY[model_name]
    runner(opts)


if __name__ == "__main__":
    main()