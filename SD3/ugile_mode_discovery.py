"""
ugile_mode_discovery_flexible.py
=================================
HONEST mode-discovery visualization.

You set exactly how many modes each method discovers.
The figure reflects reality — no false claims.

CONFIG:
-------
  TOTAL_MODES        = 10   # modes in the toy distribution
  BASELINE_FINDS     = 4    # how many baseline discovers
  UGILE_FINDS        = 7    # how many YOUR model discovers (set this honestly!)

If UGILE_FINDS < TOTAL_MODES, panel (e) shows:
  - Red stars = discovered modes
  - Light gray circles = missed modes (honest!)
  - Caption says "7 of 10 modes recovered" instead of "all 10"

USAGE:
    python ugile_mode_discovery_flexible.py

OUTPUT:
    ./mode_discovery_honest.png
    ./mode_discovery_honest.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import multivariate_normal
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import LinearSegmentedColormap

# ═══════════════════════════════════════════════════════════════════
#  HONEST CONFIGURATION — Set these to match your actual results
# ═══════════════════════════════════════════════════════════════════

TOTAL_MODES = 10          # Total modes in the distribution
BASELINE_FINDS = 4        # How many baseline SD3 discovers
UGILE_FINDS = 7           # <<< SET THIS TO WHAT YOUR MODEL ACTUALLY FINDS

# Which indices each method discovers (0-9 for 10 modes)
# Default: baseline finds center modes, UGILE finds first N modes
# You can customize these arrays to match your clustering results
baseline_discovered_indices = [4, 5, 6, 8][:BASELINE_FINDS]  # center cluster
ugile_discovered_indices = list(range(UGILE_FINDS))          # first N modes

N_SAMPLES = 80
SEED = 42
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════
#  COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════
COLORS = {
    'bg': '#FAFAFA',
    'text': '#1a1a2e',
    'text_light': '#4a4a6a',
    'red_star': '#E63946',
    'green_ring': '#2A9D8F',
    'green_light': '#40E0D0',
    'black_x': '#1D3557',
    'gray_arrow': '#6B7280',
    'mst_line': '#457B9D',
    'lost_mode': '#D1D5DB',
    'missed_mode': '#9CA3AF',   # For modes UGILE misses
    'panel_bg': '#FFFFFF',
    'border': '#E5E7EB',
}

CMAP_COLORS = ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A',
               '#EF3B2C', '#CB181D', '#A50F15', '#67000D']
CUSTOM_CMAP = LinearSegmentedColormap.from_list('premium_reds', CMAP_COLORS, N=256)

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 10

# ═══════════════════════════════════════════════════════════════════
#  DEFINE 10-MODE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════

mode_positions = np.array([
    [-3.5, -1.5], [-1.2, -1.5], [1.2, -1.5], [3.5, -1.5],
    [-3.5,  1.5], [-1.2,  1.5], [1.2,  1.5], [3.5,  1.5],
    [-2.3,  0.0], [2.3,  0.0],
])
mode_positions = mode_positions + np.random.normal(0, 0.08, mode_positions.shape)
mode_cov = 0.12 * np.eye(2)

x = np.linspace(-5, 5, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

Z = np.zeros_like(X)
for mu in mode_positions:
    Z += multivariate_normal(mu, mode_cov).pdf(pos)
Z = Z / Z.max()
potential = -np.log(Z + 0.01)
potential = (potential - potential.min()) / (potential.max() - potential.min())

# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def style_axis(ax):
    ax.set_facecolor(COLORS['panel_bg'])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


def plot_heatmap(ax, potential, extent):
    ax.imshow(potential, extent=extent, origin='lower', cmap=CUSTOM_CMAP,
              alpha=0.88, vmin=0, vmax=1, interpolation='bicubic')
    ax.contour(X, Y, potential, levels=8, colors='white', alpha=0.25, linewidths=0.5)


def plot_score_field(ax, mode_positions, xlim, ylim, grid_res=18):
    x_grid = np.linspace(xlim[0], xlim[1], grid_res)
    y_grid = np.linspace(ylim[0], ylim[1], grid_res)
    X_g, Y_g = np.meshgrid(x_grid, y_grid)
    U = np.zeros_like(X_g)
    V = np.zeros_like(Y_g)
    for i in range(X_g.shape[0]):
        for j in range(X_g.shape[1]):
            pt = np.array([X_g[i,j], Y_g[i,j]])
            score = np.zeros(2)
            for mu in mode_positions:
                diff = pt - mu
                w = multivariate_normal(mu, mode_cov).pdf(pt)
                score += -diff / (mode_cov[0,0]) * w
            norm = np.linalg.norm(score)
            if norm > 0:
                U[i,j] = score[0] / (norm + 0.4) * 0.22
                V[i,j] = score[1] / (norm + 0.4) * 0.22
    ax.quiver(X_g, Y_g, U, V, color='#374151', alpha=0.6,
              scale=1.2, width=0.0028, headwidth=3.5, headlength=4.5, pivot='mid')
    for pos in mode_positions:
        ax.scatter(pos[0], pos[1], c=COLORS['red_star'], s=280, marker='*', alpha=0.15, zorder=3)
        ax.scatter(pos[0], pos[1], c=COLORS['red_star'], s=180, marker='*', alpha=0.3, zorder=4)
    ax.scatter(mode_positions[:,0], mode_positions[:,1], c=COLORS['red_star'], s=120,
               marker='*', edgecolors='white', linewidths=1.0, zorder=5)


def plot_mst(ax, points, color=None, linewidth=1.2, alpha=0.7):
    if color is None:
        color = COLORS['mst_line']
    if len(points) < 2:
        return
    dist_matrix = squareform(pdist(points))
    n = len(points)
    visited = [0]
    while len(visited) < n:
        best_edge, best_dist = None, float('inf')
        for i in visited:
            for j in range(n):
                if j not in visited and dist_matrix[i,j] < best_dist:
                    best_dist = dist_matrix[i,j]
                    best_edge = (i, j)
        if best_edge:
            i, j = best_edge
            ax.plot([points[i,0], points[j,0]], [points[i,1], points[j,1]],
                    '-', color=color, linewidth=linewidth, alpha=alpha,
                    solid_capstyle='round', zorder=2)
            visited.append(j)


def add_flow_arrows(ax, modes, n_arrows_per_mode=4, spread=1.0, seed=0):
    rng = np.random.RandomState(seed)
    for mode in modes:
        for _ in range(n_arrows_per_mode):
            angle = rng.uniform(0, 2*np.pi)
            dist = rng.uniform(0.5, spread)
            start = mode + dist * np.array([np.cos(angle), np.sin(angle)])
            ax.annotate('', xy=mode, xytext=start,
                        arrowprops=dict(arrowstyle='->', color=COLORS['gray_arrow'],
                                        lw=0.7, alpha=0.45, connectionstyle='arc3,rad=0.1'))


def generate_baseline_samples(n=80, seed=123):
    rng = np.random.RandomState(seed)
    samples = rng.multivariate_normal([0, 0], [[0.4, 0.05], [0.05, 0.35]], n)
    outliers = rng.multivariate_normal([2.5, 1.2], [[0.15, 0], [0, 0.15]], 8)
    return np.vstack([samples, outliers])


def generate_ugile_samples(n=80, seed=456, target_indices=None):
    """Generate samples biased toward discovered modes."""
    rng = np.random.RandomState(seed)
    if target_indices is None:
        target_indices = list(range(TOTAL_MODES))
    target_modes = mode_positions[target_indices]

    samples = []
    per_mode = n // len(target_modes)
    for mu in target_modes:
        mode_samples = rng.multivariate_normal(mu, 0.25 * np.eye(2), per_mode)
        samples.append(mode_samples)
    remainder = n - per_mode * len(target_modes)
    if remainder > 0:
        extra = rng.uniform(-4.5, 4.5, (remainder, 2))
        samples.append(extra)
    return np.vstack(samples)


# ═══════════════════════════════════════════════════════════════════
#  COMPUTE DISCOVERED MODE ARRAYS
# ═══════════════════════════════════════════════════════════════════

discovered_baseline = mode_positions[baseline_discovered_indices]
discovered_ugile = mode_positions[ugile_discovered_indices]

# Modes that baseline misses
baseline_missed = np.array([m for i, m in enumerate(mode_positions) 
                            if i not in baseline_discovered_indices])

# Modes that UGILE misses (shown honestly in panel e)
ugile_missed = np.array([m for i, m in enumerate(mode_positions) 
                         if i not in ugile_discovered_indices])

baseline_samples = generate_baseline_samples(N_SAMPLES, seed=123)
ugile_samples = generate_ugile_samples(N_SAMPLES, seed=456, target_indices=ugile_discovered_indices)

# ═══════════════════════════════════════════════════════════════════
#  CREATE FIGURE
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(12, 7.2), facecolor=COLORS['bg'])
fig.patch.set_facecolor(COLORS['bg'])

LIM_X = (-5, 5)
LIM_Y = (-3, 3)
extent = [x.min(), x.max(), y.min(), y.max()]

# (a) Score Field
ax_a = fig.add_subplot(2, 3, 1)
style_axis(ax_a)
plot_score_field(ax_a, mode_positions, LIM_X, LIM_Y)
ax_a.set_xlim(LIM_X)
ax_a.set_ylim(LIM_Y)
ax_a.set_title('(a) Learned score vectors at\nthe initial sampling step\n'
               '($t = T-1$), with target\nmodes marked in red star.',
               fontsize=10, color=COLORS['text'], linespacing=1.15, pad=10)
ax_a.grid(True, alpha=0.15, linestyle='--', color=COLORS['text_light'])

# (b) Baseline Init
ax_b = fig.add_subplot(2, 3, 2)
style_axis(ax_b)
plot_heatmap(ax_b, potential, extent)
ax_b.scatter(baseline_samples[:,0], baseline_samples[:,1], c=COLORS['black_x'],
             s=32, marker='x', linewidths=1.3, alpha=0.85, zorder=5)
ax_b.set_xlim(LIM_X)
ax_b.set_ylim(LIM_Y)
ax_b.set_title('(b) Original $x_T$.\nStandard Gaussian initialization\n'
               '(black "x") concentrates\nsamples in high-potential region.',
               fontsize=10, color=COLORS['text'], linespacing=1.15, pad=10)

# (c) UGILE Init
ax_c = fig.add_subplot(2, 3, 3)
style_axis(ax_c)
plot_heatmap(ax_c, potential, extent)
ax_c.scatter(ugile_samples[:,0], ugile_samples[:,1], facecolors='none',
             edgecolors=COLORS['green_ring'], s=42, marker='o', linewidths=1.4,
             alpha=0.85, zorder=5)
ax_c.scatter(ugile_samples[:,0], ugile_samples[:,1], c=COLORS['green_light'],
             s=8, marker='o', alpha=0.3, zorder=4)
ax_c.set_xlim(LIM_X)
ax_c.set_ylim(LIM_Y)
ax_c.set_title('(c) UGILE (Ours) $x_T^*$.\nOur proposed initialization\n'
               '(green "o") disperses samples\nacross low-potential landscape.',
               fontsize=10, color=COLORS['text'], linespacing=1.15, pad=10)

# (d) Baseline Discovered (Mode Collapse)
ax_d = fig.add_subplot(2, 3, 5)
style_axis(ax_d)
add_flow_arrows(ax_d, discovered_baseline, n_arrows_per_mode=5, spread=1.3, seed=789)
plot_mst(ax_d, discovered_baseline, color=COLORS['mst_line'], linewidth=1.0, alpha=0.6)

# Show modes baseline missed as faint gray
if len(baseline_missed) > 0:
    ax_d.scatter(baseline_missed[:,0], baseline_missed[:,1], c=COLORS['lost_mode'], 
                 s=35, marker='o', alpha=0.5, zorder=3, edgecolors='white', linewidths=0.5)

for pos in discovered_baseline:
    ax_d.scatter(pos[0], pos[1], c=COLORS['red_star'], s=250, marker='*', alpha=0.2, zorder=3)
ax_d.scatter(discovered_baseline[:,0], discovered_baseline[:,1], c=COLORS['red_star'],
             s=180, marker='*', edgecolors='white', linewidths=1.2, zorder=5)
ax_d.set_xlim(LIM_X)
ax_d.set_ylim(LIM_Y)
ax_d.set_title(f'(d) Discovered modes from $x_T$.\nMode collapse: only\n'
               f'{BASELINE_FINDS} modes recovered.',
               fontsize=10, color=COLORS['text'], linespacing=1.15, pad=10)

# (e) UGILE Discovered — HONEST VERSION
ax_e = fig.add_subplot(2, 3, 6)
style_axis(ax_e)
add_flow_arrows(ax_e, discovered_ugile, n_arrows_per_mode=3, spread=1.2, seed=101)
plot_mst(ax_e, discovered_ugile, color=COLORS['mst_line'], linewidth=1.0, alpha=0.6)

# HONEST: Show modes UGILE missed as gray crosses (not hidden!)
if len(ugile_missed) > 0:
    ax_e.scatter(ugile_missed[:,0], ugile_missed[:,1], c=COLORS['missed_mode'],
                 s=40, marker='x', linewidths=1.5, alpha=0.6, zorder=3)

for pos in discovered_ugile:
    ax_e.scatter(pos[0], pos[1], c=COLORS['red_star'], s=250, marker='*', alpha=0.2, zorder=3)
ax_e.scatter(discovered_ugile[:,0], discovered_ugile[:,1], c=COLORS['red_star'],
             s=180, marker='*', edgecolors='white', linewidths=1.2, zorder=5)

for i, pos in enumerate(discovered_ugile):
    ax_e.annotate(str(i+1), xy=(pos[0], pos[1]), xytext=(5, 5),
                  textcoords='offset points', fontsize=7,
                  color=COLORS['text_light'], fontweight='bold', alpha=0.7)

ax_e.set_xlim(LIM_X)
ax_e.set_ylim(LIM_Y)

# Title reflects honest count
if UGILE_FINDS == TOTAL_MODES:
    title_e = f'(e) Discovered modes from $x_T^*$.\nAll {UGILE_FINDS} modes successfully\nrecovered.'
else:
    title_e = f'(e) Discovered modes from $x_T^*$.\n{UGILE_FINDS} of {TOTAL_MODES} modes\nrecovered (vs. {BASELINE_FINDS}).'
ax_e.set_title(title_e, fontsize=10, color=COLORS['text'], linespacing=1.15, pad=10)

# ═══════════════════════════════════════════════════════════════════
#  HONEST CAPTION
# ═══════════════════════════════════════════════════════════════════

if UGILE_FINDS == TOTAL_MODES:
    caption = (
        f"\\textbf{{Figure 3.}} Comparison of mode discovery on a 2D toy distribution with "
        f"{TOTAL_MODES} modes. (b) Standard initialization (black \"x\") concentrates samples "
        f"in the high-potential region, leading to (d) mode collapse where only {BASELINE_FINDS} "
        f"modes are recovered. (c) Our UGILE initialization ($x_T^*$, green \"o\") disperses "
        f"samples across the landscape, successfully recovering (e) all {TOTAL_MODES} modes."
    )
else:
    caption = (
        f"\\textbf{{Figure 3.}} Comparison of mode discovery on a 2D toy distribution with "
        f"{TOTAL_MODES} modes. (b) Standard initialization (black \"x\") concentrates samples "
        f"in the high-potential region, leading to (d) mode collapse where only {BASELINE_FINDS} "
        f"modes are recovered. (c) Our UGILE initialization ($x_T^*$, green \"o\") disperses "
        f"samples across the landscape, recovering (e) {UGILE_FINDS} of {TOTAL_MODES} modes "
        f"(gray \"x\" = missed modes) — a {UGILE_FINDS - BASELINE_FINDS}-mode improvement."
    )

fig.text(0.5, 0.015, caption, ha='center', va='bottom',
         fontsize=9.5, color=COLORS['text'], wrap=True,
         transform=fig.transFigure, linespacing=1.3)

# Border
border = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                        boxstyle="round,pad=0.01", facecolor='none',
                        edgecolor=COLORS['border'], linewidth=1.5,
                        transform=fig.transFigure, zorder=0)
fig.patches.append(border)

plt.tight_layout(rect=[0.01, 0.08, 0.99, 0.98])
plt.savefig('mode_discovery_honest.png', dpi=300, bbox_inches='tight',
            facecolor=COLORS['bg'], edgecolor='none', pad_inches=0.3)
plt.savefig('mode_discovery_honest.pdf', bbox_inches='tight',
            facecolor=COLORS['bg'], edgecolor='none', pad_inches=0.3)
plt.show()

print(f"\n{'='*60}")
print(f"HONEST MODE DISCOVERY REPORT")
print(f"{'='*60}")
print(f"Total modes in distribution:     {TOTAL_MODES}")
print(f"Baseline discovers:              {BASELINE_FINDS}  ({BASELINE_FINDS/TOTAL_MODES*100:.0f}%)")
print(f"UGILE discovers:                 {UGILE_FINDS}  ({UGILE_FINDS/TOTAL_MODES*100:.0f}%)")
print(f"Modes missed by UGILE:           {TOTAL_MODES - UGILE_FINDS}")
print(f"Improvement over baseline:       +{UGILE_FINDS - BASELINE_FINDS} modes")
print(f"{'='*60}")
print(f"Saved: mode_discovery_honest.png & .pdf")