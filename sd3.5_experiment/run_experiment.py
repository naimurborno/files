"""
run_experiment.py
------------------
Full UGILE hyperparameter-sweep experiment, driven entirely by config.yaml
(`experiment:` block). For every combination in `experiment.sweep`:

  1. Generates `vendi_seeds` (default 5) images per prompt -> used ONLY to
     compute the per-prompt Vendi Score (intra-prompt diversity needs
     multiple seeds of the SAME prompt).
  2. Generates 1 image per prompt at `eval_seed` -> used for FID / KID /
     CLIP Score / Precision / Recall, matched 1-to-1 against:
       - a baseline set (standard SD3, no UGILE — generated ONCE and
         reused across the whole sweep), and
       - optionally a real reference dataset (`experiment.real_reference_dir`).
  3. Appends ONE row of results to a single CSV (results are flushed after
     every combo, so a sweep can be safely resumed/interrupted).

Images are saved per-combination under:
    <experiment.root_dir>/<combo_id>/eval_single/{1..N}.png
    <experiment.root_dir>/<combo_id>/vendi_seeds/<prompt_idx>/seed<seed>.png
    <experiment.root_dir>/_baseline/eval_single/{1..N}.png     (shared, once)

Usage:
    python run_experiment.py --config config.yaml
"""

import argparse
import csv
import itertools
from pathlib import Path

import torch

import metrics as M
from latent_escape_sampler import UGILESampler, load_prompts
from pipeline_wrapper import SD3PipelineWrapper
from utils import load_config


# ══════════════════════════════════════════════════════════════════════ #
#  GRID BUILDING                                                          #
# ══════════════════════════════════════════════════════════════════════ #

def build_grid(sweep_cfg: dict):
    """Cartesian product of every list-valued key in `sweep_cfg`. Scalar
    values are held fixed across the whole sweep."""
    keys = list(sweep_cfg.keys())
    values = [sweep_cfg[k] if isinstance(sweep_cfg[k], list) else [sweep_cfg[k]]
              for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def combo_id(combo: dict) -> str:
    parts = []
    for k, v in combo.items():
        vs = str(v).replace(".", "p").replace("-", "m")
        parts.append(f"{k}{vs}")
    return "__".join(parts)


# ══════════════════════════════════════════════════════════════════════ #
#  IMAGE GENERATION                                                       #
# ══════════════════════════════════════════════════════════════════════ #

def generate_baseline(wrapper, prompts, eval_seed, out_dir: Path):
    """Standard (non-UGILE) SD3 pipeline, 1 image/prompt. Generated once
    and reused as the comparison target for every combo in the sweep."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.png"))
    if len(existing) >= len(prompts):
        print(f"[Baseline] Reusing {len(existing)} cached baseline image(s).")
        return
    for i, prompt in enumerate(prompts):
        img = wrapper._generate_standard(prompt, wrapper.cfg.get("negative_prompt", ""), eval_seed)
        img.save(out_dir / f"{i + 1}.png")
        print(f"[Baseline] {i + 1}/{len(prompts)}")


def run_combo(wrapper, sampler_kwargs, prompts, vendi_seeds, eval_seed,
              run_root: Path, negative_prompt: str):
    """Generate all images (eval + vendi-diversity) for ONE hyperparameter
    combination."""
    vendi_dir = run_root / "vendi_seeds"
    eval_dir  = run_root / "eval_single"
    vendi_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Skip cleanly if this combo was already fully generated (resume support)
    already_done = len(list(eval_dir.glob("*.png"))) >= len(prompts)

    sampler = UGILESampler(
        unet=wrapper.transformer, scheduler=wrapper.scheduler,
        cfg=wrapper.cfg, device=wrapper.device, **sampler_kwargs,
    )

    for p_idx, prompt in enumerate(prompts):
        eval_path = eval_dir / f"{p_idx + 1}.png"
        prompt_dir = vendi_dir / f"{p_idx + 1}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        need_eval  = not eval_path.exists()
        need_vendi = len(list(prompt_dir.glob("*.png"))) < len(vendi_seeds)
        if not need_eval and not need_vendi:
            continue

        prompt_embeds, pooled = wrapper.encode_prompt(prompt, negative_prompt)

        # eval_seed is one of the 5 vendi_seeds -> generate it once as part of
        # the vendi batch below and copy it into eval_single/, instead of
        # running the sampler twice for the same seed.
        reuse_from_vendi = eval_seed in vendi_seeds

        if need_eval and not reuse_from_vendi:
            latents = wrapper.get_initial_latents(seed=eval_seed)
            result = sampler.run(latents, prompt_embeds, pooled, seed=eval_seed)
            wrapper.decode_latents(result["branches"][0]["latents"]).save(eval_path)

        if need_vendi or (need_eval and reuse_from_vendi):
            for seed in vendi_seeds:
                seed_path = prompt_dir / f"seed{seed}.png"
                if not seed_path.exists():
                    latents = wrapper.get_initial_latents(seed=seed)
                    result = sampler.run(latents, prompt_embeds, pooled, seed=seed)
                    wrapper.decode_latents(result["branches"][0]["latents"]).save(seed_path)

                if need_eval and reuse_from_vendi and seed == eval_seed and not eval_path.exists():
                    import shutil
                    shutil.copy(seed_path, eval_path)

        print(f"  [{prompt[:40]:<40}] {p_idx + 1}/{len(prompts)} done")

    return vendi_dir, eval_dir


# ══════════════════════════════════════════════════════════════════════ #
#  MAIN                                                                    #
# ══════════════════════════════════════════════════════════════════════ #

def main():
    ap = argparse.ArgumentParser(description="UGILE full hyperparameter sweep")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp_cfg = cfg.get("experiment", {})

    sweep_cfg     = exp_cfg.get("sweep", {})
    vendi_seeds   = exp_cfg.get("vendi_seeds", [41, 42, 43, 44, 45])
    eval_seed     = exp_cfg.get("eval_seed", cfg.get("seed", 41))
    real_dir      = exp_cfg.get("real_reference_dir")
    root_dir      = Path(exp_cfg.get("root_dir", "outputs/experiments"))
    csv_path      = Path(exp_cfg.get("results_csv", "experiment_results.csv"))
    device        = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    negative_prompt = cfg.get("negative_prompt", "")

    prompts = load_prompts(cfg.get("prompts_file", "prompts.yaml"))
    n_prompts = exp_cfg.get("num_prompts")
    if n_prompts:
        prompts = prompts[:n_prompts]

    print(f"[Setup] {len(prompts)} prompt(s) | {len(vendi_seeds)} vendi seed(s)/prompt "
          f"| eval_seed={eval_seed}")

    # ── Load pipeline once, reused across the whole sweep ────────────────
    wrapper = SD3PipelineWrapper(cfg, device=device)
    wrapper.load()

    # ── Baseline (standard SD3) images, generated once, shared by all combos ──
    baseline_dir = root_dir / "_baseline" / "eval_single"
    generate_baseline(wrapper, prompts, eval_seed, baseline_dir)

    # ── Metric models, loaded once ────────────────────────────────────────
    clip_bundle = M.load_clip_bundle(device=device)

    grid = list(build_grid(sweep_cfg))
    print(f"[Sweep] {len(grid)} hyperparameter combination(s) to run.")

    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "a", newline="")
    writer = None

    for i, combo in enumerate(grid):
        cid = combo_id(combo)
        print(f"\n=== [{i + 1}/{len(grid)}] {cid} ===")
        run_root = root_dir / cid

        vendi_dir, eval_dir = run_combo(
            wrapper, combo, prompts, vendi_seeds, eval_seed, run_root, negative_prompt,
        )

        row = {"combo_id": cid, **combo}

        # Vendi Score: per-prompt intra-prompt diversity, averaged over prompts
        row["vendi_score"] = M.average_per_prompt_vendi(vendi_dir, clip_bundle)

        # FID / KID / Precision / Recall vs the shared baseline set
        fid, kid, prec, rec = M.compare_folders(eval_dir, baseline_dir, device=device)
        row["fid_vs_baseline"]       = fid
        row["kid_vs_baseline"]       = kid
        row["precision_vs_baseline"] = prec
        row["recall_vs_baseline"]    = rec

        # Optional: also compare vs a real reference dataset
        if real_dir:
            fid_r, kid_r, prec_r, rec_r = M.compare_folders(eval_dir, real_dir, device=device)
            row["fid_vs_real"]       = fid_r
            row["kid_vs_real"]       = kid_r
            row["precision_vs_real"] = prec_r
            row["recall_vs_real"]    = rec_r

        # CLIP Score (image-prompt alignment) on the eval set
        row["clip_score"] = M.compute_clip_score_folder(eval_dir, prompts, clip_bundle)

        if writer is None:
            writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
        writer.writerow(row)
        csv_file.flush()

        print(f"[Result] vendi={row['vendi_score']} | fid={row['fid_vs_baseline']} | "
              f"kid={row['kid_vs_baseline']} | prec={row['precision_vs_baseline']} | "
              f"rec={row['recall_vs_baseline']} | clip={row['clip_score']}")

    csv_file.close()
    print(f"\n[Done] Full sweep results -> {csv_path}")


if __name__ == "__main__":
    main()
