"""
pareto_experiment.py
---------------------
Pareto-frontier hyperparameter sweep for the Lumina2 UGILE sampler.

For every combo in config["pareto"]["grid"]:
  1. Generate one image per prompt (config["pareto"]["num_prompts"] prompts).
  2. Compute CLIP score (↑ better), Vendi score (↑ better), FID (↓ better).
  3. Log combo + metrics.

Then computes the Pareto-optimal combo set and plots vendi-vs-fid,
fid-vs-clip, vendi-vs-clip with the frontier highlighted.

Usage:
    python pareto_experiment.py --config config_lumina2.yaml

Requires: pip install torchmetrics vendi-score transformers matplotlib pandas
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from pipeline_wrapper_lumina2 import Lumina2PipelineWrapper
from latent_escape_sampler_lumina2 import Lumina2UGILESampler


# ══════════════════════════════════════════════════════════════════════ #
#  METRICS                                                                #
# ══════════════════════════════════════════════════════════════════════ #

def _clip_model(device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, proc


@torch.no_grad()
def clip_score(images, prompts, model, proc, device):
    """Mean CLIP image-text cosine similarity (%), higher = better alignment."""
    inputs = proc(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
    out = model(**inputs)
    img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    sims = (img_emb * txt_emb).sum(dim=-1)
    return sims.mean().item() * 100


@torch.no_grad()
def _clip_embeddings(images, model, proc, device):
    inputs = proc(images=images, return_tensors="pt").to(device)
    try:
        out = model.get_image_features(**inputs)
        emb = out if torch.is_tensor(out) else getattr(out, "image_embeds", None)
        if emb is None:
            raise AttributeError
    except AttributeError:
        vision_out = model.vision_model(**inputs)
        emb = model.visual_projection(vision_out.pooler_output)
    return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()


def vendi_score(images, model, proc, device):
    """Vendi Score (Friedman & Dieng, 2022) on a CLIP-embedding cosine kernel.
    Higher = more diverse set of generations."""
    from vendi_score import vendi
    emb = _clip_embeddings(images, model, proc, device)
    K = emb @ emb.T
    return float(vendi.score_K(K))


def fid_score(gen_images, real_dir, device):
    """FID between generated images and a reference real-image folder.
    Lower = more realistic / closer to the real distribution."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def _batch(imgs):
        arrs = [np.array(im.convert("RGB").resize((299, 299))) for im in imgs]
        t = torch.tensor(np.stack(arrs)).permute(0, 3, 1, 2).float() / 255.0
        return t.to(device)

    real_paths = sorted(Path(real_dir).glob("*"))[: max(len(gen_images) * 3, 50)]
    if not real_paths:
        raise FileNotFoundError(f"No real reference images found in {real_dir}")
    real_imgs = [Image.open(p) for p in real_paths]

    fid.update(_batch(real_imgs), real=True)
    fid.update(_batch(gen_images), real=False)
    return float(fid.compute().item())


# ══════════════════════════════════════════════════════════════════════ #
#  GRID                                                                   #
# ══════════════════════════════════════════════════════════════════════ #

def build_grid(grid_cfg: dict):
    keys = list(grid_cfg.keys())
    vals = [grid_cfg[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


# ══════════════════════════════════════════════════════════════════════ #
#  GENERATE ONE COMBO                                                     #
# ══════════════════════════════════════════════════════════════════════ #

def run_combo(wrapper, cfg, combo, prompts, seeds):
    sampler = Lumina2UGILESampler(
        transformer       = wrapper.transformer,
        scheduler         = wrapper.scheduler,
        cfg               = cfg,
        device            = wrapper.device,
        num_grad_steps    = combo.get("num_grad_steps",  cfg.get("ugile", {}).get("num_grad_steps",  5)),
        sigma_lo          = combo.get("sigma_lo",        cfg.get("ugile", {}).get("sigma_lo",        0.3)),
        sigma_hi          = combo.get("sigma_hi",        cfg.get("ugile", {}).get("sigma_hi",        0.9)),
        escape_scale      = combo.get("escape_scale",    cfg.get("ugile", {}).get("escape_scale",    3.0)),
        theta_max         = combo.get("theta_max",       cfg.get("ugile", {}).get("theta_max",       0.8)),
        walk_steps        = combo.get("walk_steps",      cfg.get("ugile", {}).get("walk_steps",      10)),
        J                 = combo.get("J",                cfg.get("ugile", {}).get("J",               1)),
        noise_scale       = combo.get("noise_scale",      cfg.get("ugile", {}).get("noise_scale",     0.2)),
        gamma             = combo.get("gamma",             cfg.get("ugile", {}).get("gamma",           1.2)),
        cfg_trunc_ratio   = combo.get("cfg_trunc_ratio",   cfg.get("flow", {}).get("cfg_trunc_ratio",   1.0)),
        cfg_normalization = combo.get("cfg_normalization", cfg.get("flow", {}).get("cfg_normalization", False)),
        interp_scale_factor = combo.get("interp_scale_factor", cfg.get("ugile", {}).get("interp_scale_factor", 0.96)),
    )

    images = []
    for prompt in prompts:
        for seed in seeds:
            prompt_embeds, attention_mask = wrapper.encode_prompt(prompt, cfg.get("negative_prompt", ""))
            latents = wrapper.get_initial_latents(seed=seed)
            result  = sampler.run(latents, prompt_embeds, attention_mask, seed=seed)
            images.append(wrapper.decode_latents(result["branches"][0]["latents"]))
    return images


# ══════════════════════════════════════════════════════════════════════ #
#  PARETO FRONTIER                                                        #
# ══════════════════════════════════════════════════════════════════════ #

def pareto_frontier(df: pd.DataFrame, directions: dict) -> np.ndarray:
    """directions: {col: 'max'|'min'}. Returns bool mask, True = non-dominated."""
    signed = np.stack(
        [df[c].values * (1 if d == "max" else -1) for c, d in directions.items()],
        axis=1,
    )
    n = len(df)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominated = np.all(signed >= signed[i], axis=1) & np.any(signed > signed[i], axis=1)
        dominated[i] = False
        is_pareto[dominated] = False
    return is_pareto


# ══════════════════════════════════════════════════════════════════════ #
#  PLOTS                                                                  #
# ══════════════════════════════════════════════════════════════════════ #

def make_plots(df: pd.DataFrame, out_dir: Path):
    import matplotlib.pyplot as plt

    for x, y in [("vendi", "fid"), ("fid", "clip"), ("vendi", "clip")]:
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = df["is_pareto"].map({True: "crimson", False: "steelblue"})
        ax.scatter(df[x], df[y], c=colors, s=70, edgecolor="black", zorder=3)

        pf = df[df["is_pareto"]].sort_values(x)
        ax.plot(pf[x], pf[y], "--", color="crimson", alpha=0.7, zorder=2, label="Pareto frontier")

        for _, row in df.iterrows():
            ax.annotate(row["combo_id"], (row[x], row[y]), fontsize=7, alpha=0.75,
                        xytext=(3, 3), textcoords="offset points")

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{x}_vs_{y}.png", dpi=150)
        plt.close(fig)

    # Bar chart: per-metric ranking of Pareto-optimal combos
    pf = df[df["is_pareto"]].copy()
    if len(pf):
        fig, ax = plt.subplots(figsize=(7, 4))
        idx = np.arange(len(pf))
        w = 0.25
        ax.bar(idx - w, pf["vendi"], width=w, label="vendi")
        ax.bar(idx,       pf["clip"], width=w, label="clip")
        ax.bar(idx + w, pf["fid"],   width=w, label="fid")
        ax.set_xticks(idx)
        ax.set_xticklabels(pf["combo_id"], rotation=45, ha="right")
        ax.set_title("Pareto-optimal combos — metric comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "pareto_combo_comparison.png", dpi=150)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════ #
#  MAIN                                                                   #
# ══════════════════════════════════════════════════════════════════════ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_lumina2.yaml")
    ap.add_argument("--shard_id", type=int, default=0, help="This process's shard index")
    ap.add_argument("--num_shards", type=int, default=1, help="Total shards (1 = no sharding)")
    ap.add_argument("--merge", action="store_true", help="Merge shard CSVs + plot, skip generation")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pareto_cfg = cfg.get("pareto", {})
    out_dir    = Path(pareto_cfg.get("output_dir", "pareto_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        shard_csvs = sorted(out_dir.glob("results_shard*.csv"))
        if not shard_csvs:
            raise FileNotFoundError(f"No results_shard*.csv found in {out_dir}")
        df = pd.concat([pd.read_csv(p) for p in shard_csvs], ignore_index=True)
        df["is_pareto"] = pareto_frontier(df, {"vendi": "max", "fid": "min", "clip": "max"})
        df.to_csv(out_dir / "results.csv", index=False)
        print(df[["combo_id", "vendi", "fid", "clip", "is_pareto"]].to_string(index=False))
        make_plots(df, out_dir)
        print(f"[Pareto] Merged {len(shard_csvs)} shard(s) → {out_dir/'results.csv'} + plots")
        return

    device      = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    grid_cfg    = pareto_cfg.get("grid", {})
    num_prompts = pareto_cfg.get("num_prompts", 20)
    real_dir    = pareto_cfg.get("real_images_dir")

    prompts = (cfg.get("prompts") or [])[:num_prompts]
    seeds   = pareto_cfg.get("seeds") or (cfg.get("seeds") or [42, 43])[:2]
    if not prompts:
        raise ValueError("config['prompts'] is empty — add your prompts there.")
    if not grid_cfg:
        raise ValueError("config['pareto']['grid'] is empty — add hyperparameters to sweep.")

    print(f"[Pareto] Loading Lumina2 pipeline once for all combos…")
    wrapper = Lumina2PipelineWrapper(cfg, device=device)
    wrapper.load()

    clip_model, clip_proc = _clip_model(device)

    combos = build_grid(grid_cfg)
    combos = list(enumerate(combos))[args.shard_id::args.num_shards]  # [(global_idx, combo), ...]
    n_img = len(prompts) * len(seeds)
    print(f"[Pareto] Shard {args.shard_id}/{args.num_shards}: {len(combos)} combo(s) × "
          f"{len(prompts)} prompt(s) × {len(seeds)} seed(s) = {n_img} image(s)/combo")

    # prompts repeated per seed, matching run_combo's image ordering
    prompts_for_clip = [p for p in prompts for _ in seeds]

    rows = []
    for i, combo in combos:
        print(f"\n[Pareto] Combo {i} (shard {args.shard_id}): {combo}")
        images = run_combo(wrapper, cfg, combo, prompts, seeds)

        c = clip_score(images, prompts_for_clip, clip_model, clip_proc, device)
        v = vendi_score(images, clip_model, clip_proc, device)
        f = fid_score(images, real_dir, device) if real_dir else float("nan")

        rows.append({**combo, "combo_id": f"c{i}", "clip": c, "vendi": v, "fid": f})
        print(f"[Pareto]   clip={c:.2f}  vendi={v:.3f}  fid={f:.2f}")

        combo_dir = out_dir / f"combo_{i}"
        combo_dir.mkdir(exist_ok=True)
        for j, im in enumerate(images):
            im.save(combo_dir / f"img_{j}.png")

    df = pd.DataFrame(rows)
    shard_csv = out_dir / f"results_shard{args.shard_id}.csv"
    df.to_csv(shard_csv, index=False)
    print(f"\n[Pareto] Shard {args.shard_id} results → {shard_csv}")
    print(df[["combo_id", "vendi", "fid", "clip"]].to_string(index=False))
    print("[Pareto] Run with --merge (after all shards finish) to combine + plot.")


if __name__ == "__main__":
    main()