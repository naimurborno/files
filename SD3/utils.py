"""
utils.py
--------
Shared utilities: config loading, seeding, saving results.
Unchanged from the DDPM version — fully compatible with the flow model.
"""

import torch
import numpy as np
import random
import yaml
import json
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"[utils] Config loaded from: {path}")
    return cfg


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[utils] Seed: {seed}")


def save_results(results: list, output_dir: str):
    """Save latents and metadata for all prompts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for i, r in enumerate(results):
        latent_path = output_dir / f"latents_{i:03d}.pt"
        torch.save(r["latents"].cpu(), latent_path)

        metadata.append({
            "index"       : i,
            "prompt"      : r.get("prompt", ""),
            "latent_path" : str(latent_path),
        })

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[utils] Saved {len(results)} result(s) + metadata.json → {output_dir}")