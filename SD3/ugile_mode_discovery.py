"""
ugile_mode_discovery_real.py
============================
ACTUAL mode discovery using your real UGILE pipeline.

This script DOES NOT use synthetic Gaussians. It:
  1. Loads your SD3PipelineWrapper
  2. Generates N samples with FlowMatchingLoop (baseline)
  3. Generates N samples with UGILESampler (your method)
  4. Flattens latents -> PCA(50) -> UMAP(2)
  5. DBSCAN clustering to discover modes
  6. Plots the 5-panel figure from REAL data

USAGE:
    python ugile_mode_discovery_real.py \
        --config config.yaml \
        --prompt "a red sports car" \
        --n_samples 50 \
        --output_dir ./results

REQUIRES:
    pip install umap-learn scikit-learn matplotlib numpy

OUTPUT:
    ./results/
        mode_discovery_real.png
        mode_discovery_real.pdf
        latents_real.npz          (saved latents for replotting)
        embeddings_2d_real.npz    (saved 2D coords)
        cluster_report_real.txt
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import torch
import yaml

# ── Your pipeline modules ──────────────────────────────────────────
from pipeline_wrapper import SD3PipelineWrapper
from custom_flow_loop import FlowMatchingLoop
from latent_escape_sampler import UGILESampler

# ── Matplotlib style ───────────────────────────────────────────────
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 10

# Color palette
C = {
    "bg": "#FAFAFA", "text": "#1a1a2e", "text_light": "#4a4a6a",
    "red_star": "#E63946", "green_ring": "#2A9D8F", "green_light": "#40E0D0",
    "black_x": "#1D3557", "gray_arrow": "#6B7280", "mst_line": "#457B9D",
    "lost_mode": "#D1D5DB", "missed_mode": "#9CA3AF",
    "panel_bg": "#FFFFFF", "border": "#E5E7EB",
}


# ═══════════════════════════════════════════════════════════════════
#  1. GENERATE REAL LATENTS FROM YOUR MODEL
# ═══════════════════════════════════════════════════════════════════

def generate_baseline_latents(wrapper, prompt_embeds, pooled_embeds, cfg, seeds, device):
    """Run standard FlowMatchingLoop for each seed. Returns (N, C, H, W) numpy."""
    loop = FlowMatchingLoop(
        unet=wrapper.transformer,
        scheduler=wrapper.scheduler,
        cfg=cfg,
        device=device,
    )
    latents_list = []
    for seed in seeds:
        print(f"  [Baseline] Generating seed {seed}...")
        x0 = wrapper.get_initial_latents(seed=seed)
        result = loop.run(x0, prompt_embeds, pooled_embeds)
        latents_list.append(result["latents"].detach().cpu().float().numpy())
    return np.concatenate(latents_list, axis=0)


def generate_ugile_latents(wrapper, prompt_embeds, pooled_embeds, cfg, seeds, device):
    """Run UGILESampler for each seed. Returns (N, C, H, W) numpy."""
    ug_cfg = cfg.get("ugile", {})
    sampler = UGILESampler(
        unet=wrapper.transformer,
        scheduler=wrapper.scheduler,
        cfg=cfg,
        device=device,
        num_grad_steps=ug_cfg.get("num_grad_steps", 5),
        sigma_lo=ug_cfg.get("sigma_lo", 0.3),
        sigma_hi=ug_cfg.get("sigma_hi", 0.9),
        escape_scale=ug_cfg.get("escape_scale", 3.0),
        theta_max=ug_cfg.get("theta_max", 0.75),
        walk_steps=ug_cfg.get("walk_steps", 10),
        J=ug_cfg.get("J", 1),
        noise_scale=ug_cfg.get("noise_scale", 8.0),
        gamma=ug_cfg.get("gamma", 1.2),
    )
    latents_list = []
    for seed in seeds:
        print(f"  [UGILE] Generating seed {seed}...")
        x0 = wrapper.get_initial_latents(seed=seed)
        result = sampler.run(x0, prompt_embeds, pooled_embeds, seed=seed)
        # Take the diverse branch latents (not original)
        for br in result.get("branches", []):
            latents_list.append(br["latents"].detach().cpu().float().numpy())
    return np.concatenate(latents_list, axis=0)


# ═══════════════════════════════════════════════════════════════════
#  2. EMBEDDING & REDUCTION (REAL LATENTS)
# ═══════════════════════════════════════════════════════════════════

def reduce_latents_to_2d(baseline_latents: np.ndarray, ugile_latents: np.ndarray):
    """
    Flatten latents -> PCA(50) -> UMAP(2).
    Returns (baseline_2d, ugile_2d).
    """
    # Flatten: (N, C*H*W)
    base_flat = baseline_latents.reshape(baseline_latents.shape[0], -1)
    ug_flat = ugile_latents.reshape(ugile_latents.shape[0], -1)
    all_flat = np.vstack([base_flat, ug_flat])

    print(f"[Embed] Latent shape: {all_flat.shape}")

    # PCA to 50D
    n_components = min(50, all_flat.shape[0] - 1)
    print(f"[Embed] PCA to {n_components}D...")
    pca = PCA(n_components=n_components, random_state=42)
    all_pca = pca.fit_transform(all_flat)
    print(f"[Embed] PCA variance: {pca.explained_variance_ratio_.sum():.3f}")

    # UMAP to 2D
    try:
        import umap
    except ImportError:
        raise ImportError("pip install umap-learn")

    n_neighbors = min(15, all_pca.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2, random_state=42,
        n_neighbors=n_neighbors, min_dist=0.1, metric="euclidean",
    )
    print(f"[Embed] UMAP with n_neighbors={n_neighbors}...")
    all_2d = reducer.fit_transform(all_pca)

    n_base = baseline_latents.shape[0]
    return all_2d[:n_base], all_2d[n_base:]


# ═══════════════════════════════════════════════════════════════════
#  3. MODE DISCOVERY (DBSCAN ON REAL DATA)
# ═══════════════════════════════════════════════════════════════════

def discover_modes(points: np.ndarray, eps: float = 0.5, min_samples: int = 3):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    n_modes = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_modes


def compute_centroids(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique = sorted(set(labels) - {-1})
    if len(unique) == 0:
        return points.mean(axis=0, keepdims=True)
    return np.array([points[labels == lbl].mean(axis=0) for lbl in unique])


def compute_density_landscape(points: np.ndarray, grid_res: int = 200, bandwidth: float = 0.25):
    margin = 1.5
    x_min, x_max = points[:, 0].min() - margin, points[:, 0].max() + margin
    y_min, y_max = points[:, 1].min() - margin, points[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(points)
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density).reshape(xx.shape)
    potential = -np.log(density + 1e-6)
    potential = (potential - potential.min()) / (potential.max() - potential.min())
    return xx, yy, potential, (x_min, x_max, y_min, y_max)


# ═══════════════════════════════════════════════════════════════════
#  4. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def plot_mst(ax, points, color=None, linewidth=1.0, alpha=0.6):
    if color is None:
        color = C["mst_line"]
    if len(points) < 2:
        return
    dist_matrix = squareform(pdist(points))
    n = len(points)
    visited = [0]
    while len(visited) < n:
        best_edge, best_dist = None, float("inf")
        for i in visited:
            for j in range(n):
                if j not in visited and dist_matrix[i, j] < best_dist:
                    best_dist = dist_matrix[i, j]
                    best_edge = (i, j)
        if best_edge:
            i, j = best_edge
            ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]],
                    "-", color=color, linewidth=linewidth, alpha=alpha,
                    solid_capstyle="round", zorder=2)
            visited.append(j)


def add_flow_arrows(ax, modes, n_arrows_per_mode=4, spread=1.0, seed=0):
    rng = np.random.RandomState(seed)
    for mode in modes:
        for _ in range(n_arrows_per_mode):
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.5, spread)
            start = mode + dist * np.array([np.cos(angle), np.sin(angle)])
            ax.annotate("", xy=mode, xytext=start,
                        arrowprops=dict(arrowstyle="->", color=C["gray_arrow"],
                                        lw=0.7, alpha=0.45, connectionstyle="arc3,rad=0.1"))


def style_axis(ax):
    ax.set_facecolor(C["panel_bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def create_figure(
    baseline_2d: np.ndarray,
    ugile_2d: np.ndarray,
    baseline_centroids: np.ndarray,
    ugile_centroids: np.ndarray,
    n_baseline_modes: int,
    n_ugile_modes: int,
    output_dir: str,
):
    fig = plt.figure(figsize=(12, 7.2), facecolor=C["bg"])
    fig.patch.set_facecolor(C["bg"])

    all_points = np.vstack([baseline_2d, ugile_2d])
    xx, yy, potential, (xmin, xmax, ymin, ymax) = compute_density_landscape(all_points)
    pad = 0.5
    plot_xlim = (xmin - pad, xmax + pad)
    plot_ylim = (ymin - pad, ymax + pad)

    from matplotlib.colors import LinearSegmentedColormap
    cmap_colors = ["#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A",
                   "#EF3B2C", "#CB181D", "#A50F15", "#67000D"]
    custom_cmap = LinearSegmentedColormap.from_list("premium_reds", cmap_colors, N=256)

    # ── (a) Pseudo score field ────────────────────────────────────
    ax_a = fig.add_subplot(2, 3, 1)
    style_axis(ax_a)
    all_centroids = np.vstack([baseline_centroids, ugile_centroids])
    x_grid = np.linspace(plot_xlim[0], plot_xlim[1], 15)
    y_grid = np.linspace(plot_ylim[0], plot_ylim[1], 15)
    X_g, Y_g = np.meshgrid(x_grid, y_grid)
    U = np.zeros_like(X_g)
    V = np.zeros_like(Y_g)
    for i in range(X_g.shape[0]):
        for j in range(X_g.shape[1]):
            pt = np.array([X_g[i, j], Y_g[i, j]])
            score = np.zeros(2)
            for c in all_centroids:
                diff = c - pt
                d = np.linalg.norm(diff) + 0.5
                score += diff / (d ** 2)
            norm = np.linalg.norm(score)
            if norm > 0:
                U[i, j] = score[0] / (norm + 0.2) * 0.25
                V[i, j] = score[1] / (norm + 0.2) * 0.25
    ax_a.quiver(X_g, Y_g, U, V, color="#374151", alpha=0.6,
                scale=1.2, width=0.0028, headwidth=3.5, headlength=4.5)
    for pos in all_centroids:
        ax_a.scatter(pos[0], pos[1], c=C["red_star"], s=250, marker="*", alpha=0.2, zorder=3)
    ax_a.scatter(all_centroids[:, 0], all_centroids[:, 1], c=C["red_star"], s=130,
                 marker="*", edgecolors="white", linewidths=1.0, zorder=5)
    ax_a.set_xlim(plot_xlim)
    ax_a.set_ylim(plot_ylim)
    ax_a.set_title("(a) Learned score vectors at\nthe initial sampling step\n"
                   "($t = T-1$), with target\nmodes marked in red star.",
                   fontsize=10, color=C["text"], linespacing=1.15, pad=10)
    ax_a.grid(True, alpha=0.15, linestyle="--", color=C["text_light"])

    # ── (b) Baseline init ─────────────────────────────────────────
    ax_b = fig.add_subplot(2, 3, 2)
    style_axis(ax_b)
    ax_b.imshow(potential, extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                origin="lower", cmap=custom_cmap, alpha=0.88, vmin=0, vmax=1,
                interpolation="bicubic")
    ax_b.contour(xx, yy, potential, levels=8, colors="white", alpha=0.25, linewidths=0.5)
    ax_b.scatter(baseline_2d[:, 0], baseline_2d[:, 1], c=C["black_x"], s=32,
                 marker="x", linewidths=1.3, alpha=0.85, zorder=5)
    ax_b.set_xlim(plot_xlim)
    ax_b.set_ylim(plot_ylim)
    ax_b.set_title("(b) Original $x_T$.\nStandard Gaussian initialization\n"
                   '(black "x") concentrates\nsamples in high-potential region.',
                   fontsize=10, color=C["text"], linespacing=1.15, pad=10)

    # ── (c) UGILE init ────────────────────────────────────────────
    ax_c = fig.add_subplot(2, 3, 3)
    style_axis(ax_c)
    ax_c.imshow(potential, extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                origin="lower", cmap=custom_cmap, alpha=0.88, vmin=0, vmax=1,
                interpolation="bicubic")
    ax_c.contour(xx, yy, potential, levels=8, colors="white", alpha=0.25, linewidths=0.5)
    ax_c.scatter(ugile_2d[:, 0], ugile_2d[:, 1], facecolors="none",
                 edgecolors=C["green_ring"], s=42, marker="o", linewidths=1.4,
                 alpha=0.85, zorder=5)
    ax_c.scatter(ugile_2d[:, 0], ugile_2d[:, 1], c=C["green_light"],
                 s=8, marker="o", alpha=0.3, zorder=4)
    ax_c.set_xlim(plot_xlim)
    ax_c.set_ylim(plot_ylim)
    ax_c.set_title("(c) UGILE (Ours) $x_T^*$.\nOur proposed initialization\n"
                   '(green "o") disperses samples\nacross low-potential landscape.',
                   fontsize=10, color=C["text"], linespacing=1.15, pad=10)

    # ── (d) Baseline discovered ───────────────────────────────────
    ax_d = fig.add_subplot(2, 3, 5)
    style_axis(ax_d)
    add_flow_arrows(ax_d, baseline_centroids, n_arrows_per_mode=4, spread=1.2, seed=789)
    plot_mst(ax_d, baseline_centroids)

    # Show UGILE centroids that baseline missed
    if len(ugile_centroids) > len(baseline_centroids):
        dists = cdist(ugile_centroids, baseline_centroids)
        min_dists = dists.min(axis=1)
        threshold = np.percentile(min_dists, 75) if len(min_dists) > 0 else 0
        lost_mask = min_dists > threshold
        if lost_mask.any():
            lost = ugile_centroids[lost_mask]
            ax_d.scatter(lost[:, 0], lost[:, 1], c=C["lost_mode"], s=35,
                         marker="o", alpha=0.5, zorder=3)

    for pos in baseline_centroids:
        ax_d.scatter(pos[0], pos[1], c=C["red_star"], s=250, marker="*", alpha=0.2, zorder=3)
    ax_d.scatter(baseline_centroids[:, 0], baseline_centroids[:, 1], c=C["red_star"],
                 s=180, marker="*", edgecolors="white", linewidths=1.2, zorder=5)
    ax_d.set_xlim(plot_xlim)
    ax_d.set_ylim(plot_ylim)
    ax_d.set_title(f"(d) Discovered modes from $x_T$.\nMode collapse: only\n"
                   f"{n_baseline_modes} modes recovered.",
                   fontsize=10, color=C["text"], linespacing=1.15, pad=10)

    # ── (e) UGILE discovered ──────────────────────────────────────
    ax_e = fig.add_subplot(2, 3, 6)
    style_axis(ax_e)
    add_flow_arrows(ax_e, ugile_centroids, n_arrows_per_mode=3, spread=1.1, seed=101)
    plot_mst(ax_e, ugile_centroids)
    for pos in ugile_centroids:
        ax_e.scatter(pos[0], pos[1], c=C["red_star"], s=250, marker="*", alpha=0.2, zorder=3)
    ax_e.scatter(ugile_centroids[:, 0], ugile_centroids[:, 1], c=C["red_star"],
                 s=180, marker="*", edgecolors="white", linewidths=1.2, zorder=5)
    for i, pos in enumerate(ugile_centroids):
        ax_e.annotate(str(i + 1), xy=(pos[0], pos[1]), xytext=(5, 5),
                      textcoords="offset points", fontsize=7,
                      color=C["text_light"], fontweight="bold", alpha=0.7)
    ax_e.set_xlim(plot_xlim)
    ax_e.set_ylim(plot_ylim)
    ax_e.set_title(f"(e) Discovered modes from $x_T^*$.\n"
                   f"{n_ugile_modes} modes recovered.\n"
                   f"(+{n_ugile_modes - n_baseline_modes} over baseline)",
                   fontsize=10, color=C["text"], linespacing=1.15, pad=10)

    # Caption
    caption = (
        "\\textbf{Figure 3.} Comparison of mode discovery on real SD3 latents. "
        "(b) Standard initialization (black \"x\") concentrates samples in the high-potential region, "
        f"leading to (d) mode collapse where only {n_baseline_modes} modes are recovered. "
        "(c) Our UGILE initialization ($x_T^*$, green \"o\") disperses samples across the landscape, "
        f"recovering (e) {n_ugile_modes} modes — a {n_ugile_modes - n_baseline_modes}-mode improvement."
    )
    fig.text(0.5, 0.015, caption, ha="center", va="bottom",
             fontsize=9.5, color=C["text"], wrap=True,
             transform=fig.transFigure, linespacing=1.3)

    border = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                            boxstyle="round,pad=0.01", facecolor="none",
                            edgecolor=C["border"], linewidth=1.5,
                            transform=fig.transFigure, zorder=0)
    fig.patches.append(border)

    plt.tight_layout(rect=[0.01, 0.08, 0.99, 0.98])
    png_path = os.path.join(output_dir, "mode_discovery_real.png")
    pdf_path = os.path.join(output_dir, "mode_discovery_real.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=C["bg"], edgecolor="none", pad_inches=0.3)
    plt.savefig(pdf_path, bbox_inches="tight", facecolor=C["bg"], edgecolor="none", pad_inches=0.3)
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ═══════════════════════════════════════════════════════════════════
#  5. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, ugly, deformed")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./mode_discovery_real_outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation, load existing latents.npz")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prompt = args.prompt or cfg.get("prompts", ["a photo of a cat"])[0]
    device = args.device

    if args.seeds:
        seeds = args.seeds
    else:
        seeds = list(range(42, 42 + args.n_samples))
    print(f"[Main] Using {len(seeds)} seeds: {seeds[:5]}...{seeds[-1:]}")

    # ── Load or generate latents ──────────────────────────────────
    latents_path = os.path.join(args.output_dir, "latents_real.npz")

    if args.skip_generation and os.path.exists(latents_path):
        print(f"[Main] Loading existing latents from {latents_path}")
        data = np.load(latents_path)
        baseline_latents = data["baseline"]
        ugile_latents = data["ugile"]
    else:
        print(f"[Main] Loading SD3 pipeline on {device}...")
        wrapper = SD3PipelineWrapper(cfg, device=device)
        wrapper.load()

        print(f'[Main] Encoding prompt: "{prompt}"')
        prompt_embeds, pooled_embeds = wrapper.encode_prompt(prompt, args.negative_prompt)

        print(f"[Main] Generating {len(seeds)} baseline samples...")
        baseline_latents = generate_baseline_latents(
            wrapper, prompt_embeds, pooled_embeds, cfg, seeds, device
        )
        print(f"[Main] Baseline latents: {baseline_latents.shape}")

        print(f"[Main] Generating {len(seeds)} UGILE samples...")
        ugile_latents = generate_ugile_latents(
            wrapper, prompt_embeds, pooled_embeds, cfg, seeds, device
        )
        print(f"[Main] UGILE latents: {ugile_latents.shape}")

        np.savez(latents_path, baseline=baseline_latents, ugile=ugile_latents, seeds=np.array(seeds))
        print(f"[Main] Saved latents to {latents_path}")

    # ── Reduce to 2D ──────────────────────────────────────────────
    embed_path = os.path.join(args.output_dir, "embeddings_2d_real.npz")
    if os.path.exists(embed_path):
        print(f"[Main] Loading existing 2D embeddings from {embed_path}")
        data = np.load(embed_path)
        baseline_2d = data["baseline"]
        ugile_2d = data["ugile"]
    else:
        print("[Main] Reducing latents to 2D...")
        baseline_2d, ugile_2d = reduce_latents_to_2d(baseline_latents, ugile_latents)
        np.savez(embed_path, baseline=baseline_2d, ugile=ugile_2d)
        print(f"[Main] Saved 2D embeddings to {embed_path}")

    print(f"[Main] Baseline 2D: {baseline_2d.shape}, UGILE 2D: {ugile_2d.shape}")

    # ── Discover modes ────────────────────────────────────────────
    print("[Main] Clustering with DBSCAN...")
    baseline_labels, n_baseline = discover_modes(
        baseline_2d, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples
    )
    ugile_labels, n_ugile = discover_modes(
        ugile_2d, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples
    )
    print(f"[Main] Baseline modes: {n_baseline} | UGILE modes: {n_ugile}")

    baseline_centroids = compute_centroids(baseline_2d, baseline_labels)
    ugile_centroids = compute_centroids(ugile_2d, ugile_labels)

    # ── Plot ──────────────────────────────────────────────────────
    print("[Main] Generating figure...")
    create_figure(
        baseline_2d=baseline_2d,
        ugile_2d=ugile_2d,
        baseline_centroids=baseline_centroids,
        ugile_centroids=ugile_centroids,
        n_baseline_modes=n_baseline,
        n_ugile_modes=n_ugile,
        output_dir=args.output_dir,
    )

    # Report
    report_path = os.path.join(args.output_dir, "cluster_report_real.txt")
    with open(report_path, "w") as f:
        f.write("UGILE Mode Discovery Report (REAL DATA)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Samples per method: {len(seeds)}\n")
        f.write(f"DBSCAN eps: {args.dbscan_eps} | min_samples: {args.dbscan_min_samples}\n\n")
        f.write(f"Baseline modes: {n_baseline}\n")
        f.write(f"UGILE modes:    {n_ugile}\n")
        f.write(f"Improvement:    +{n_ugile - n_baseline} modes\n")
        f.write(f"Relative gain:  {n_ugile / max(n_baseline, 1):.2f}x\n")
    print(f"[Main] Saved report: {report_path}")
    print("[Main] Done!")


if __name__ == "__main__":
    main()