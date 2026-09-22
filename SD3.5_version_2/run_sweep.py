"""
run_sweep.py — generates per-arm configs from your config.yaml and runs inference.py.

Kaggle usage (paths default to /kaggle/working and /kaggle/working/files/SD3.5):
    python files/SD3.5/run_sweep.py --stage A --steps 28
    python files/SD3.5/run_sweep.py --stage B --eps 0.20 --steps 28 [--arms main,strong]
        arms: main(eps) strong(2eps) weak(eps/2) noortho win30 win40 beta050 beta095
    python files/SD3.5/run_sweep.py --stage C --eps 0.20 --steps 28 --window 0.18 --beta 0.85 [--ortho off]
    add --dry-run to only write configs and print the commands.
    override locations with --work_dir / --code_dir if your layout differs.

Everything lands under <work_dir>/runs/<stage>/<arm>/{diverse,original}. Nothing is overwritten across arms.
"""
import argparse, copy, subprocess, sys
from pathlib import Path
import yaml

def window_len(frac, N):            # mirrors sampler: max(1, int(round(window_frac * N)))
    return max(1, int(round(frac * N)))

def sum_gamma(W):                   # mirrors sampler's linear anneal gamma_k = 1 - k/W, k = 0..W-1
    return sum(max(0.0, 1.0 - k / W) for k in range(W))

BIG = 1e4   # lite_noise_scale so large that the cap (lite_max_eps_frac) is ALWAYS the binding constraint

def arm_specs(stage, eps, window, beta, ortho, N):
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
            "strong":  dict(base, lite_max_eps_frac=2 * eps),
            "weak":    dict(base, lite_max_eps_frac=eps / 2),
            "noortho": dict(base, disable_orthogonal_projection=True),
            # Window arms keep the TOTAL push (eps * sum(gamma)) equal to `main`, so they test WHERE the push
            # is applied, not just how much of it there is.
            "win30":   dict(base, window_frac=0.30, lite_max_eps_frac=eps * sum_gamma(window_len(window, N)) / sum_gamma(window_len(0.30, N))),
            "win40":   dict(base, window_frac=0.40, lite_max_eps_frac=eps * sum_gamma(window_len(window, N)) / sum_gamma(window_len(0.40, N))),
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
    ap.add_argument("--work_dir", default="/kaggle/working", help="where runs/ is written")
    ap.add_argument("--code_dir", default=None, help="folder with inference.py, config.yaml, prompts.yaml (default: <work_dir>/files/SD3.5)")
    ap.add_argument("--allow-fewer", action="store_true", help="B/C: run with fewer prompts than planned")
    ap.add_argument("--steps", type=int, default=None, help="sampler steps (overrides flow.num_steps in the config); use the SAME value as your CADS/DAVE runs")
    ap.add_argument("--arms", default=None, help="comma-separated subset of arms to run, e.g. main,strong")
    a = ap.parse_args()

    base_dir = Path(a.work_dir)
    code_dir = Path(a.code_dir) if a.code_dir else base_dir / "files" / "SD3.5"
    def rel(p):   # relative paths are resolved against code_dir; absolute paths are kept
        return Path(p) if Path(p).is_absolute() else code_dir / p

    cfg0 = yaml.safe_load(open(rel(a.config)))
    prompts_path = rel(a.prompts or cfg0.get("prompts_file", "prompts.yaml"))
    pdata = yaml.safe_load(open(prompts_path))
    all_prompts = pdata.get("prompts", []) if isinstance(pdata, dict) else pdata   # accept `prompts:` dict or bare list
    if not all_prompts:
        sys.exit(f"[ERROR] no prompts found in {prompts_path}")
    print(f"[cfg] code_dir={code_dir}  prompts={prompts_path} ({len(all_prompts)} prompts)")

    n_prompts, seeds = {"A": (10, [41, 42]), "B": (20, [41, 42, 43, 44, 45]), "C": (100, [41, 42, 43, 44, 45])}[a.stage]
    if len(all_prompts) < n_prompts:
        msg = (f"prompts file has {len(all_prompts)} prompts but stage {a.stage} needs {n_prompts}.")
        if a.stage == "A" or a.allow_fewer:
            print(f"[WARN] {msg} Proceeding with {len(all_prompts)}.")
        else:
            sys.exit(f"[ERROR] {msg} Add prompts, or pass --allow-fewer to run anyway "
                     f"(then Vendi CIs and P/R will be far noisier than planned).")
    prompts = all_prompts[:n_prompts]
    root = base_dir / "runs" / a.stage
    root.mkdir(parents=True, exist_ok=True)
    pfile = root / "prompts.yaml"
    yaml.safe_dump({"prompts": prompts}, open(pfile, "w"))
    orig_dir = root / "original"           # shared baseline images for the whole stage

    N = a.steps if a.steps is not None else cfg0.get("flow", {}).get("num_steps", 50)
    W = window_len(a.window, N)
    print(f"[cfg] num_steps={N} (config.yaml says {cfg0.get('flow', {}).get('num_steps')}) -> window W={W}, "
          f"sum(gamma)={sum_gamma(W):.2f}  => coherent push ~ {sum_gamma(W):.1f} x eps_frac")
    arms = arm_specs(a.stage, a.eps, a.window, a.beta, a.ortho == "on", N)
    if a.arms:
        want = [x.strip() for x in a.arms.split(",")]
        bad = [x for x in want if x not in arms]
        if bad: sys.exit(f"[ERROR] unknown arm(s) {bad}; available: {list(arms)}")
        arms = {k: v for k, v in arms.items() if k in want}
    has_baseline = orig_dir.exists() and any(orig_dir.glob("*_seed*"))   # reuse baseline from an earlier invocation
    for i, (name, spec) in enumerate(arms.items()):
        cfg = copy.deepcopy(cfg0)
        cfg.setdefault("flow", {})["num_steps"] = N
        cfg["seeds"] = seeds
        cfg["prompts_file"] = str(pfile.resolve())
        ug = cfg.setdefault("ugile", {})
        ug.update(spec)
        ug["diverse_output_dir"] = str((root / name / "diverse").resolve())
        ug["original_output_dir"] = str(orig_dir.resolve())
        # Stage A: every arm computes baseline so cos_xN is logged per arm.
        # Stage B/C: only the first arm generates baseline images (saves ~1 extra pass/image per other arm);
        # cos to baseline is then computed in feature space by analyze.py.
        ug["save_original"] = True if a.stage == "A" else (i == 0 and not has_baseline)
        cfg["output"] = str((root / name / "unused.png").resolve())   # only its suffix (.png) is used
        cpath = root / name / "config.yaml"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(cfg, open(cpath, "w"), sort_keys=False)
        cmd = [sys.executable, str((code_dir / "inference.py").resolve()), "--config", str(cpath.resolve())]
        print(f"[{a.stage}/{name}] {' '.join(cmd)}   params={spec}")
        if not a.dry_run:
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()