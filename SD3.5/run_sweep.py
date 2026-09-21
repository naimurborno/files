"""
run_sweep.py — generates per-arm configs from your config.yaml and runs inference.py.

Usage (from any folder on Kaggle):
    python files/SD3.5/run_sweep.py --stage A                       # pilot: 10 prompts x 2 seeds, eps_frac sweep
    python files/SD3.5/run_sweep.py --stage B --eps 0.10            # 20 prompts x 5 seeds, ablation + window/beta arms
    python files/SD3.5/run_sweep.py --stage C --eps 0.10 --window 0.18 --beta 0.85 [--ortho off]   # final: 100 prompts x 5 seeds
    add --dry-run to only write configs and print the commands.

Everything lands under runs/<stage>/<arm>/{diverse,original}. Nothing is overwritten across arms.
"""
import argparse, copy, subprocess, sys
from pathlib import Path
import yaml

BIG = 1e4   # lite_noise_scale so large that the cap (lite_max_eps_frac) is ALWAYS the binding constraint

def arm_specs(stage, eps, window, beta, ortho):
    base = dict(mode="lite", beta=beta, window_frac=window, lite_noise_scale=BIG,
                lite_max_eps_frac=eps, disable_orthogonal_projection=not ortho)
    if stage == "A":
        return {
            "cap000": dict(base, lite_max_eps_frac=0.0),   # SANITY: zero push -> must reproduce baseline (cos_xN ~ 1.000)
            "default_as_is": dict(base, lite_noise_scale=8.0, lite_max_eps_frac=0.02),  # what you have now
            "cap002": dict(base, lite_max_eps_frac=0.02),
            "cap005": dict(base, lite_max_eps_frac=0.05),
            "cap010": dict(base, lite_max_eps_frac=0.10),
            "cap020": dict(base, lite_max_eps_frac=0.20),
        }
    if stage == "B":
        return {
            "main":    dict(base),
            "noortho": dict(base, disable_orthogonal_projection=True),
            "win30":   dict(base, window_frac=0.30),
            "win40":   dict(base, window_frac=0.40),
            "beta050": dict(base, beta=0.50),
            "beta095": dict(base, beta=0.95),
        }
    if stage == "C":
        return {"main": dict(base)}
    raise ValueError(stage)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B", "C"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--prompts", default=None, help="prompts yaml (defaults to cfg['prompts_file'])")
    ap.add_argument("--eps", type=float, default=0.10, help="lite_max_eps_frac for B/C main arm")
    ap.add_argument("--window", type=float, default=0.18)
    ap.add_argument("--beta", type=float, default=0.85)
    ap.add_argument("--ortho", choices=["on", "off"], default="on")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # --- KAGGLE PATH FIXES ---
    # Define reference directories explicitly using absolute paths
    base_dir = Path("/kaggle/working")
    code_dir = base_dir / "files" / "SD3.5"
    
    # Locate configuration base file dynamically if relative path provided
    config_src = Path(a.config) if Path(a.config).is_absolute() else code_dir / a.config

    cfg0 = yaml.safe_load(open(config_src))
    prompts_src = a.prompts or cfg0.get("prompts_file", "prompts.yaml")
    
    # Ensure prompts path handles absolute matching
    prompts_path = Path(prompts_src) if Path(prompts_src).is_absolute() else code_dir / prompts_src
    all_prompts = yaml.safe_load(open(prompts_path))

    n_prompts, seeds = {"A": (10, [41, 42]), "B": (20, [41, 42, 43, 44, 45]), "C": (100, [41, 42, 43, 44, 45])}[a.stage]
    prompts = all_prompts[:n_prompts]
    
    # Save runs folder at the root kaggle working directory level
    root = base_dir / "runs" / a.stage
    root.mkdir(parents=True, exist_ok=True)
    pfile = root / "prompts.yaml"
    yaml.safe_dump({"prompts": prompts}, open(pfile, "w"))
    orig_dir = root / "original"           # shared baseline images for the whole stage

    arms = arm_specs(a.stage, a.eps, a.window, a.beta, a.ortho == "on")
    for i, (name, spec) in enumerate(arms.items()):
        cfg = copy.deepcopy(cfg0)
        cfg["seeds"] = seeds
        cfg["prompts_file"] = str(pfile.resolve())
        ug = cfg.setdefault("ugile", {})
        ug.update(spec)
        ug["diverse_output_dir"] = str((root / name / "diverse").resolve())
        ug["original_output_dir"] = str(orig_dir.resolve())
        # Stage A: every arm computes baseline so cos_xN is logged per arm.
        # Stage B/C: only the first arm generates baseline images (saves ~1 extra pass/image per other arm);
        # cos to baseline is then computed in feature space by analyze.py.
        ug["save_original"] = True if a.stage == "A" else (i == 0)
        cfg["output"] = str((root / name / "unused.png").resolve())   # only its suffix (.png) is used
        
        cpath = root / name / "config.yaml"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(cfg, open(cpath, "w"), sort_keys=False)
        
        # 1. Use absolute path for inference.py
        # 2. Use absolute path for --config argument (.resolve() forces absolute)
        cmd = [sys.executable, str((code_dir / "inference.py").resolve()), "--config", str(cpath.resolve())]
        
        print(f"[{a.stage}/{name}] {' '.join(cmd)}   params={spec}")
        if not a.dry_run:
            # Execute directly with resolved paths, matching code execution environment
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
