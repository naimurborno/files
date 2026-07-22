"""
Usage:
    python run.py config.yaml

Loads all hyperparameters from a YAML file and generates `num_samples`
images for the given prompt using CADS (Algorithm 1). If generation.use_cads
is false, it instead reproduces the plain classifier-free-guided DDIM
baseline (no condition annealing) for side-by-side comparison.
"""

import os
import sys
import yaml
import torch

from cads import CADSScheduleConfig
from pipeline_cads import CADSStableDiffusion, CADSGenerationConfig


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    dtype = torch.float16 if cfg["model"]["dtype"] == "float16" else torch.float32

    cads_cfg = CADSScheduleConfig(
        tau1=cfg["cads"]["tau1"],
        tau2=cfg["cads"]["tau2"],
        noise_scale=cfg["cads"]["noise_scale"],
        psi=cfg["cads"]["psi"],
        rescale=cfg["cads"]["rescale"],
        apply_to=cfg["cads"]["apply_to"],
    )
    print("Loaded CADS config:", cads_cfg)

    model = CADSStableDiffusion(
        model_id=cfg["model"]["model_id"],
        cads_config=cads_cfg,
        dtype=dtype,
    )

    gen_cfg = cfg["generation"]
    out_dir = cfg["output"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # `seed` can be a single int OR a list, e.g. [41, 42, 43, 44, 45].
    # Each seed in the list generates the SAME prompt once (multi-seed variants).
    raw_seed = gen_cfg.get("seed", None)
    if isinstance(raw_seed, (list, tuple)):
        seeds = list(raw_seed)
    elif raw_seed is None:
        seeds = [None] * gen_cfg.get("num_samples", 1)
    else:
        # backward compat: single base seed + num_samples -> base_seed + i
        seeds = [raw_seed + i for i in range(gen_cfg.get("num_samples", 1))]

    for i, seed_i in enumerate(seeds):
        g = CADSGenerationConfig(
            prompt=gen_cfg["prompt"],
            negative_prompt=gen_cfg.get("negative_prompt", ""),
            num_inference_steps=gen_cfg.get("num_inference_steps", 50),
            guidance_scale=gen_cfg.get("guidance_scale", 9.0),
            height=gen_cfg.get("height", 512),
            width=gen_cfg.get("width", 512),
            seed=seed_i,
            use_cads=gen_cfg.get("use_cads", True),
        )
        print(f"Sampling {i+1}/{len(seeds)} (seed={seed_i}, use_cads={g.use_cads}) ...")
        image = model.generate(g)
        tag = "cads" if g.use_cads else "baseline"
        out_path = os.path.join(out_dir, f"{tag}_seed{seed_i}.png")
        image.save(out_path)
        print("Saved:", out_path)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)