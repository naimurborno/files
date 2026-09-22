"""
pareto_sweep.py
----------------
UGILE hyperparameter Pareto sweep: one fixed prompt, 5 fixed seeds, grid over
(theta_max, noise_scale, escape_scale). For each grid point, generates the
baseline + 5 diverse branches and records TWO tiers of metrics:

  Tier 1 (free, latent-space proxies — no extra deps, computed every run):
    fidelity_proxy  = mean(cos_xN) over the 5 seeds
    diversity_proxy = 1 - mean_pairwise_cos(x_N_diverse_i, x_N_diverse_j)

  Tier 2 (real paper-grade metrics, requires a CLIP model — this is what you
  actually report, not the proxies above):
    clip_score  = mean CLIP text-image similarity between the prompt and each
                  of the 5 diverse images (and separately for the 5 baseline
                  images, so you can see if UGILE costs you prompt fidelity)
    vendi_score = Vendi Score (Friedman & Dieng) computed on the 5 diverse
                  images' CLIP image embeddings — standard entropy-of-
                  eigenvalues diversity metric, exp(H(K/N)) on the similarity
                  kernel K.

NOTE on escape_scale: once escape_scale * (||w||/r) exceeds theta_max, the
clamp wins and escape_scale stops affecting output at all (you already
observed this: 1.7 and 3.0 landed at the same theta_used). It only matters
in the sub-clamp regime. Include small escape_scale values (e.g. 0.5, 1.0)
if you actually want to see its effect rather than redundant clamped rows.

Usage:
    python pareto_sweep.py --config config.yaml --prompt "..." \
        --theta_max 0.15 0.2 0.25 0.3 0.35 0.4 \
        --noise_scale 4 8 12 16 \
        --escape_scale 0.5 1.0 2.0 3.0 \
        --seeds 41 42 43 44 45

Output:
    outputs/pareto_sweep/results.csv         (per-config, per-seed rows)
    outputs/pareto_sweep/summary.csv         (per-config aggregated means)
    outputs/pareto_sweep/pareto_front.csv    (non-dominated configs, on clip_score/vendi_score)
    outputs/pareto_sweep/pareto_plot.png     (vendi vs clip_score scatter)
    outputs/pareto_sweep/images/<config>/    (decoded PNGs for spot-checking)
"""

import argparse
import csv
import itertools
from pathlib import Path

import torch
import yaml

from pipeline_wrapper import SD3PipelineWrapper
from latent_escape_sampler import UGILESampler


def pairwise_diversity(latents: list) -> float:
    """1 - mean pairwise cosine similarity across a list of final latents."""
    flats = [l.reshape(1, -1).float() for l in latents]
    sims = []
    for a, b in itertools.combinations(flats, 2):
        sims.append(torch.nn.functional.cosine_similarity(a, b).item())
    return 1.0 - (sum(sims) / len(sims))


class ClipMetrics:
    """Real CLIP score + Vendi Score, computed from decoded PIL images.
    Loaded lazily so the script still runs (proxy-only) if unavailable."""

    def __init__(self, device: str, model_id: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def image_embeddings(self, images: list) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_out.pooler_output
        feats = self.model.visual_projection(pooled)
        return torch.nn.functional.normalize(feats, dim=-1)

    @torch.no_grad()
    def text_embedding(self, prompt: str) -> torch.Tensor:
        inputs = self.processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
        text_out = self.model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
        )
        pooled = text_out.pooler_output
        feats = self.model.text_projection(pooled)
        return torch.nn.functional.normalize(feats, dim=-1)

    def clip_score(self, images: list, prompt: str) -> float:
        img_emb = self.image_embeddings(images)
        txt_emb = self.text_embedding(prompt)
        sims = (img_emb @ txt_emb.T).squeeze(-1)
        return (sims.clamp(min=0).mean() * 100).item()

    def vendi_score(self, images: list) -> float:
        """Friedman & Dieng: VS = exp(H(eigvals(K/N))), K = cosine-sim kernel."""
        emb = self.image_embeddings(images)  # [N, D], already L2-normalized
        n = emb.shape[0]
        K = (emb @ emb.T) / n
        eigvals = torch.linalg.eigvalsh(K).clamp(min=1e-12)
        p = eigvals / eigvals.sum()
        entropy = -(p * torch.log(p)).sum()
        return torch.exp(entropy).item()


def run_sweep(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda")
    ug_cfg = cfg.get("ugile", {})

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] Loading pipeline once (device={device})...")
    wrapper = SD3PipelineWrapper(cfg, device=device)
    wrapper.load()

    if args.use_clip:
        print("[sweep] Loading CLIP model for real clip_score/vendi_score...")
        try:
            clip_metrics = ClipMetrics(device=device)
        except Exception as e:
            print(f"[sweep] WARNING: could not load CLIP model ({e}). "
                  f"Falling back to proxy metrics only.")
            clip_metrics = None
    else:
        clip_metrics = None

    negative_prompt = cfg.get("negative_prompt", "")
    print(f"[sweep] Encoding fixed prompt: {args.prompt!r}")
    text_embeddings, pooled_embeddings = wrapper.encode_prompt(args.prompt, negative_prompt)

    rows = []
    grid = list(itertools.product(args.theta_max, args.noise_scale, args.escape_scale))
    print(f"[sweep] {len(grid)} configs x {len(args.seeds)} seeds = "
          f"{len(grid) * len(args.seeds)} runs")

    for cfg_idx, (theta_max, noise_scale, escape_scale) in enumerate(grid):
        tag = f"theta{theta_max}_noise{noise_scale}_esc{escape_scale}"
        print(f"\n[sweep] ({cfg_idx+1}/{len(grid)}) theta_max={theta_max}  "
              f"noise_scale={noise_scale}  escape_scale={escape_scale}")

        sampler = UGILESampler(
            unet            = wrapper.transformer,
            scheduler       = wrapper.scheduler,
            cfg             = cfg,
            device          = device,
            num_grad_steps  = ug_cfg.get("num_grad_steps",  5),
            sigma_lo        = ug_cfg.get("sigma_lo",        0.3),
            sigma_hi        = ug_cfg.get("sigma_hi",        0.9),
            escape_scale    = escape_scale,
            theta_max       = theta_max,
            walk_steps      = ug_cfg.get("walk_steps",      10),
            J               = ug_cfg.get("J",               1),
            noise_scale     = noise_scale,
            gamma           = ug_cfg.get("gamma",           1.2),
            max_eps_frac    = ug_cfg.get("max_eps_frac",    0.02),
        )

        cfg_img_dir = img_dir / tag
        cfg_img_dir.mkdir(parents=True, exist_ok=True)

        diverse_latents = []
        diverse_images = []
        baseline_images = []
        for seed in args.seeds:
            x0 = wrapper.get_initial_latents(seed)
            result = sampler.run(x0, text_embeddings, pooled_embeddings)
            branch = result["branches"][0]

            diverse_latents.append(branch["latents"])

            rows.append({
                "theta_max"    : theta_max,
                "noise_scale"  : noise_scale,
                "escape_scale" : escape_scale,
                "seed"         : seed,
                "theta_used"   : branch["theta"],
                "cos_x0"       : branch["cos_x0"],
                "cos_xN"       : branch["cos_xN"],
            })

            img_diverse  = wrapper.decode_latents(branch["latents"])
            img_original = wrapper.decode_latents(result["original_latents"])
            diverse_images.append(img_diverse)
            baseline_images.append(img_original)

            if args.save_images:
                img_diverse.save(cfg_img_dir / f"seed{seed}_diverse.png")
                img_original.save(cfg_img_dir / f"seed{seed}_original.png")

            print(f"    seed={seed}  theta_used={branch['theta']:.4f}  "
                  f"cos_x0={branch['cos_x0']:.4f}  cos_xN={branch['cos_xN']:.4f}")

        diversity_proxy = pairwise_diversity(diverse_latents)

        clip_score_diverse = clip_score_baseline = vendi_score = None
        if clip_metrics is not None:
            clip_score_diverse  = clip_metrics.clip_score(diverse_images, args.prompt)
            clip_score_baseline = clip_metrics.clip_score(baseline_images, args.prompt)
            vendi_score          = clip_metrics.vendi_score(diverse_images)

        for r in rows[-len(args.seeds):]:
            r["diversity_proxy_this_config"] = diversity_proxy
            r["clip_score_diverse"]  = clip_score_diverse
            r["clip_score_baseline"] = clip_score_baseline
            r["vendi_score"]         = vendi_score

        mean_cos_xN = sum(r["cos_xN"] for r in rows[-len(args.seeds):]) / len(args.seeds)
        print(f"    -> mean_cos_xN={mean_cos_xN:.4f}  diversity_proxy={diversity_proxy:.4f}  "
              f"clip_score_diverse={clip_score_diverse}  clip_score_baseline={clip_score_baseline}  "
              f"vendi_score={vendi_score}")

    # ---- write raw per-seed rows ----
    results_csv = out_dir / "results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[sweep] Wrote {results_csv}")

    # ---- aggregate per config ----
    have_clip = clip_metrics is not None
    summary = {}
    for r in rows:
        key = (r["theta_max"], r["noise_scale"], r["escape_scale"])
        if key not in summary:
            summary[key] = {
                "cos_xN"            : [],
                "diversity_proxy"   : r["diversity_proxy_this_config"],
                "clip_score_diverse" : r["clip_score_diverse"],
                "clip_score_baseline": r["clip_score_baseline"],
                "vendi_score"        : r["vendi_score"],
            }
        summary[key]["cos_xN"].append(r["cos_xN"])

    fieldnames = ["theta_max", "noise_scale", "escape_scale", "fidelity_proxy",
                  "diversity_proxy", "clip_score_diverse", "clip_score_baseline", "vendi_score"]
    summary_rows = []
    for (theta_max, noise_scale, escape_scale), v in summary.items():
        summary_rows.append({
            "theta_max"           : theta_max,
            "noise_scale"         : noise_scale,
            "escape_scale"        : escape_scale,
            "fidelity_proxy"      : sum(v["cos_xN"]) / len(v["cos_xN"]),
            "diversity_proxy"     : v["diversity_proxy"],
            "clip_score_diverse"  : v["clip_score_diverse"],
            "clip_score_baseline" : v["clip_score_baseline"],
            "vendi_score"         : v["vendi_score"],
        })

    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[sweep] Wrote {summary_csv}")

    # ---- Pareto frontier ----
    # Prefer the real metrics (clip_score_diverse vs vendi_score) if available;
    # fall back to the free proxies (fidelity_proxy vs diversity_proxy) if not.
    if have_clip:
        x_key, y_key = "vendi_score", "clip_score_diverse"
    else:
        x_key, y_key = "diversity_proxy", "fidelity_proxy"
        print("[sweep] NOTE: no CLIP model — Pareto front uses PROXY metrics only, "
              "not real Vendi/CLIP scores. Re-run with --use_clip for paper-grade numbers.")

    pareto = []
    for cand in summary_rows:
        dominated = False
        for other in summary_rows:
            if other is cand:
                continue
            if (other[x_key] >= cand[x_key] and other[y_key] >= cand[y_key] and
                    (other[x_key] > cand[x_key] or other[y_key] > cand[y_key])):
                dominated = True
                break
        if not dominated:
            pareto.append(cand)
    pareto.sort(key=lambda r: r[x_key])

    pareto_csv = out_dir / "pareto_front.csv"
    with open(pareto_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pareto)
    print(f"[sweep] Wrote {pareto_csv}  ({len(pareto)} non-dominated configs, "
          f"axes: {x_key} vs {y_key})")

    print(f"\n[sweep] Pareto-optimal configs ({x_key} vs {y_key}):")
    for r in pareto:
        print(f"    theta_max={r['theta_max']}  noise_scale={r['noise_scale']}  "
              f"escape_scale={r['escape_scale']}  {x_key}={r[x_key]:.4f}  {y_key}={r[y_key]:.4f}")

    # ---- plot ----
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        all_x = [r[x_key] for r in summary_rows]
        all_y = [r[y_key] for r in summary_rows]
        ax.scatter(all_x, all_y, c="lightgray", label="all configs")
        px = [r[x_key] for r in pareto]
        py = [r[y_key] for r in pareto]
        ax.plot(px, py, "o-", c="crimson", label="Pareto front")
        for r in summary_rows:
            ax.annotate(f"θ={r['theta_max']},n={r['noise_scale']},e={r['escape_scale']}",
                        (r[x_key], r[y_key]), fontsize=5, alpha=0.7)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(f"UGILE Pareto frontier — prompt: {args.prompt[:50]!r}")
        ax.legend()
        fig.tight_layout()
        plot_path = out_dir / "pareto_plot.png"
        fig.savefig(plot_path, dpi=150)
        print(f"[sweep] Wrote {plot_path}")
    except ImportError:
        print("[sweep] matplotlib not installed — skipping plot (CSV results still written).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--theta_max", type=float, nargs="+",
                         default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    parser.add_argument("--noise_scale", type=float, nargs="+",
                         default=[4, 8, 12, 16])
    parser.add_argument("--escape_scale", type=float, nargs="+",
                         default=[3.0])
    parser.add_argument("--seeds", type=int, nargs="+",
                         default=[41, 42, 43, 44, 45])
    parser.add_argument("--output_dir", default="outputs/pareto_sweep")
    parser.add_argument("--save_images", action="store_true", default=True)
    parser.add_argument("--use_clip", action="store_true", default=True,
                         help="Compute real CLIP score + Vendi Score (requires "
                              "transformers + a CLIP checkpoint download). "
                              "Pass --no-use_clip to skip and use free proxies only.")
    parser.add_argument("--no-use_clip", dest="use_clip", action="store_false")
    args = parser.parse_args()
    run_sweep(args)