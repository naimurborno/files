import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats as st
from PIL import Image
from sklearn.manifold import TSNE
from transformers import CLIPProcessor, CLIPVisionModel, CLIPModel, CLIPTokenizer
from torchvision import transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from vendi_score import vendi


# ══════════════════════════════════════════════════════════════════════ #
#  FEATURE EXTRACTION                                                    #
# ══════════════════════════════════════════════════════════════════════ #

def load_images_as_tensors(folder_path, img_size=299):
    """Load all images from a folder as a normalised tensor batch (Inception input)."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    tensors = []
    for img_name in sorted(os.listdir(folder_path)):
        if img_name.lower().endswith(valid_extensions):
            try:
                img = Image.open(os.path.join(folder_path, img_name)).convert('RGB')
                tensors.append(transform(img))
            except:
                pass
    return torch.stack(tensors) if tensors else None


def extract_inception_features(image_tensors, device, batch_size=32):
    """Extract 2048-d pool3 features from Inception-v3. Model is loaded once and reused."""
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()   # strip classifier → 2048-d output
    inception.eval()

    feats = []
    with torch.inference_mode():
        for i in range(0, len(image_tensors), batch_size):
            batch = image_tensors[i:i + batch_size].to(device)
            feats.append(inception(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def extract_inception_features_multi(tensor_dict, device, batch_size=32):
    """
    Extract Inception features for multiple folders in a single model load.
    tensor_dict: { label: image_tensor_batch }
    Returns:     { label: np.ndarray of features }
    """
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()
    inception.eval()

    results = {}
    with torch.inference_mode():
        for label, tensors in tensor_dict.items():
            feats = []
            for i in range(0, len(tensors), batch_size):
                batch = tensors[i:i + batch_size].to(device)
                feats.append(inception(batch).cpu().numpy())
            results[label] = np.concatenate(feats, axis=0)
    return results


def extract_clip_image_features(folder_path, processor, model, device):
    """Extract CLIP visual embeddings (pooler_output) from all images in a folder."""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    embeddings = []
    if not os.path.exists(folder_path):
        return None
    for img_name in sorted(os.listdir(folder_path)):
        if img_name.lower().endswith(valid_extensions):
            try:
                img = Image.open(os.path.join(folder_path, img_name)).convert('RGB')
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.inference_mode():
                    feat = model(**inputs).pooler_output
                embeddings.append(feat.cpu().numpy().flatten())
            except:
                pass
    return np.array(embeddings) if embeddings else None


# ══════════════════════════════════════════════════════════════════════ #
#  FID                                                                   #
# ══════════════════════════════════════════════════════════════════════ #

def compute_fid(feat_real, feat_generated):
    """
    Fréchet Inception Distance.
      feat_real      : Inception features of the real reference dataset
      feat_generated : Inception features of the generated images
    Lower is better — generated distribution is closer to the real one.
    """
    mu_r, sigma_r = feat_real.mean(axis=0),      np.cov(feat_real,      rowvar=False)
    mu_g, sigma_g = feat_generated.mean(axis=0), np.cov(feat_generated, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)

    # sqrtm can return tiny imaginary parts due to floating-point noise
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))


# ══════════════════════════════════════════════════════════════════════ #
#  PRECISION & RECALL  (Kynkäänniemi et al., 2019)                      #
# ══════════════════════════════════════════════════════════════════════ #

def _manifold_radii(features, k=3):
    """Distance to k-th nearest neighbour — defines each sample's manifold radius."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(features)
    distances, _ = nn.kneighbors(features)
    return distances[:, -1]


def compute_precision_recall(feat_real, feat_generated, k=3):
    """
    Precision : fraction of generated samples that lie inside the real manifold
                (fidelity — are generated images realistic?).
    Recall    : fraction of real samples that lie inside the generated manifold
                (diversity — does the generator cover the real distribution?).
    Both ∈ [0, 1]; higher is better.
    Reference: real dataset. Generated: baseline or proposed outputs.
    """
    from sklearn.neighbors import NearestNeighbors

    radii_real = _manifold_radii(feat_real, k)
    radii_gen  = _manifold_radii(feat_generated, k)

    nn_real = NearestNeighbors(n_neighbors=1).fit(feat_real)
    dists_gen_to_real, idx_real = nn_real.kneighbors(feat_generated)
    precision = float(np.mean(dists_gen_to_real[:, 0] <= radii_real[idx_real[:, 0]]))

    nn_gen = NearestNeighbors(n_neighbors=1).fit(feat_generated)
    dists_real_to_gen, idx_gen = nn_gen.kneighbors(feat_real)
    recall = float(np.mean(dists_real_to_gen[:, 0] <= radii_gen[idx_gen[:, 0]]))

    return precision, recall


# ══════════════════════════════════════════════════════════════════════ #
#  CLIP SCORE  (Hessel et al., 2021)                                    #
# ══════════════════════════════════════════════════════════════════════ #

def compute_clip_score(folder_path, prompts, clip_model, clip_processor,
                       clip_tokenizer, device):
    """
    CLIP Score = mean cosine similarity (image embedding, text embedding) × 100.
    `prompts`: single string (applied to all images) or list matched to sorted filenames.
    Higher is better; typical range 20–35 for modern diffusion models.
    """
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    image_files = sorted([f for f in os.listdir(folder_path)
                          if f.lower().endswith(valid_extensions)])
    if not image_files:
        return None

    if isinstance(prompts, str):
        prompts = [prompts] * len(image_files)

    scores = []
    clip_model.eval()
    for img_name, prompt in zip(image_files, prompts):
        try:
            img = Image.open(os.path.join(folder_path, img_name)).convert('RGB')
            img_inputs  = clip_processor(images=img, return_tensors="pt").to(device)
            text_inputs = clip_tokenizer([prompt], return_tensors="pt",
                                          padding=True, truncation=True).to(device)
            with torch.inference_mode():
                img_feat  = clip_model.get_image_features(**img_inputs)
                text_feat = clip_model.get_text_features(**text_inputs)
            img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            scores.append((img_feat @ text_feat.T).item() * 100)
        except:
            pass

    return float(np.mean(scores)) if scores else None


# ══════════════════════════════════════════════════════════════════════ #
#  MAIN ANALYSIS                                                         #
# ══════════════════════════════════════════════════════════════════════ #

def plot_multimodal_distribution(
    folder_proposed,
    folder_baseline,
    folder_real,
    prompts=None,
    clip_model_name="openai/clip-vit-base-patch32",
):
    """
    Full evaluation suite with proper FID using a real reference dataset.

    FID, Precision, and Recall are all computed against `folder_real` as the
    ground-truth reference — once for the baseline outputs and once for the
    proposed outputs — so the numbers are directly comparable.

    Args:
        folder_proposed  : path to diverse/proposed generated images
        folder_baseline  : path to baseline/original generated images
        folder_real      : path to real reference dataset images
        prompts          : str or list[str] for CLIP Score; None skips CLIP
        clip_model_name  : HuggingFace model ID for CLIP
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. CLIP models ───────────────────────────────────────────────────
    print("Initialising CLIP model…")
    processor      = CLIPProcessor.from_pretrained(clip_model_name)
    clip_vision    = CLIPVisionModel.from_pretrained(clip_model_name).to(device)
    clip_vision.eval()
    clip_full      = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    # ── 2. CLIP features (for Vendi + t-SNE) ────────────────────────────
    print("Extracting CLIP features…")
    feat_proposed = extract_clip_image_features(folder_proposed, processor, clip_vision, device)
    feat_baseline = extract_clip_image_features(folder_baseline, processor, clip_vision, device)

    if feat_proposed is None or feat_baseline is None:
        print("❌ CLIP feature extraction failed. Check folder paths."); return

    # ── 3. Vendi Scores ──────────────────────────────────────────────────
    print("Computing Vendi Scores…")
    vs_proposed = vendi.score_dual(feat_proposed, normalize=True)
    vs_baseline = vendi.score_dual(feat_baseline, normalize=True)

    # ── 4. Inception features — load model ONCE for all three folders ────
    print("Extracting Inception features (real / baseline / proposed)…")
    tensor_map = {
        "real"     : load_images_as_tensors(folder_real),
        "baseline" : load_images_as_tensors(folder_baseline),
        "proposed" : load_images_as_tensors(folder_proposed),
    }
    for key, t in tensor_map.items():
        if t is None:
            print(f"❌ No images found in {key} folder."); return

    inc = extract_inception_features_multi(tensor_map, device)

    # ── 5. FID: real → baseline  and  real → proposed ───────────────────
    print("Computing FID (vs real dataset)…")
    fid_baseline = compute_fid(inc["real"], inc["baseline"])
    fid_proposed = compute_fid(inc["real"], inc["proposed"])

    # ── 6. Precision & Recall (real as reference manifold) ───────────────
    print("Computing Precision & Recall (vs real dataset)…")
    prec_baseline, rec_baseline = compute_precision_recall(inc["real"], inc["baseline"])
    prec_proposed, rec_proposed = compute_precision_recall(inc["real"], inc["proposed"])

    # ── 7. CLIP Score ────────────────────────────────────────────────────
    cs_baseline = cs_proposed = None
    if prompts is not None:
        print("Computing CLIP Scores…")
        cs_baseline = compute_clip_score(
            folder_baseline, prompts, clip_full, processor, clip_tokenizer, device)
        cs_proposed = compute_clip_score(
            folder_proposed, prompts, clip_full, processor, clip_tokenizer, device)

    # ── 8. t-SNE (baseline vs proposed; real excluded to keep plot clean) ─
    print("Computing t-SNE projections…")
    combined = np.vstack([feat_baseline, feat_proposed])
    opt_perplexity = min(12, max(5, min(len(feat_baseline), len(feat_proposed)) // 2))
    tsne = TSNE(n_components=2, perplexity=opt_perplexity, random_state=42)
    embeds_2d   = tsne.fit_transform(combined)
    base_2d     = embeds_2d[:len(feat_baseline)]
    prop_2d     = embeds_2d[len(feat_baseline):]

    x, y = embeds_2d[:, 0], embeds_2d[:, 1]
    xmin, xmax = x.min() - 5, x.max() + 5
    ymin, ymax = y.min() - 5, y.max() + 5
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    kernel = st.gaussian_kde(np.vstack([x, y]))
    f = np.reshape(kernel(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)

    # ── 9. Console summary ───────────────────────────────────────────────
    W = 62
    print("\n" + "═" * W)
    print(f"  {'Metric':<24} {'Baseline':>12} {'Proposed':>12} {'Δ':>8}")
    print("─" * W)
    print(f"  {'Vendi Score (↑)':<24} {vs_baseline:>12.4f} {vs_proposed:>12.4f} {vs_proposed - vs_baseline:>+8.4f}")
    print(f"  {'FID ↓  (vs real)':<24} {fid_baseline:>12.2f} {fid_proposed:>12.2f} {fid_proposed - fid_baseline:>+8.2f}")
    print(f"  {'Precision ↑':<24} {prec_baseline:>12.4f} {prec_proposed:>12.4f} {prec_proposed - prec_baseline:>+8.4f}")
    print(f"  {'Recall ↑':<24} {rec_baseline:>12.4f} {rec_proposed:>12.4f} {rec_proposed - rec_baseline:>+8.4f}")
    if cs_baseline is not None:
        print(f"  {'CLIP Score ↑':<24} {cs_baseline:>12.2f} {cs_proposed:>12.2f} {cs_proposed - cs_baseline:>+8.2f}")
    print("═" * W)

    # ── 10. Plots ─────────────────────────────────────────────────────────
    print("\nRendering plots…")
    fig = plt.figure(figsize=(20, 7), dpi=300)

    # ── 10a. t-SNE panels ────────────────────────────────────────────────
    panel_titles = [
        f"(a) Baseline\nVendi: {vs_baseline:.2f}",
        f"(b) Proposed Method\nVendi: {vs_proposed:.2f}",
    ]
    datasets = [base_2d, prop_2d]
    colors   = ['#E74C3C', '#2980B9']
    markers  = ['x', '^']

    for i in range(2):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.contourf(xx, yy, f, cmap='Oranges', alpha=0.5, levels=15)
        ax.scatter(datasets[i][:, 0], datasets[i][:, 1],
                   c=colors[i], marker=markers[i], s=60, label='Generated')
        ax.set_title(panel_titles[i], fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.axis('off'); ax.legend(loc='lower right', fontsize=9)

    # ── 10b. Grouped bar chart ────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)

    bar_metrics   = ['Vendi\nScore', 'FID\n(÷10, ↓)', 'Precision', 'Recall']
    bar_baseline  = [vs_baseline,   fid_baseline / 10, prec_baseline, rec_baseline]
    bar_proposed  = [vs_proposed,   fid_proposed / 10, prec_proposed, rec_proposed]

    if cs_baseline is not None:
        bar_metrics.append('CLIP\nScore\n(÷10)')
        bar_baseline.append(cs_baseline / 10)
        bar_proposed.append(cs_proposed / 10)

    x_pos = np.arange(len(bar_metrics))
    width = 0.35
    b1 = ax3.bar(x_pos - width / 2, bar_baseline, width,
                 label='Baseline', color='#E74C3C', alpha=0.85)
    b2 = ax3.bar(x_pos + width / 2, bar_proposed, width,
                 label='Proposed', color='#2980B9', alpha=0.85)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bar_metrics, fontsize=9)
    ax3.set_ylabel('Score', fontsize=10)
    ax3.set_ylim(0, max(max(bar_baseline), max(bar_proposed)) * 1.3)
    ax3.set_title("(c) Metric Comparison\n(FID & CLIP scaled ÷10 for display)",
                  fontsize=11, fontweight='bold', pad=10)
    ax3.legend(fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        if h > 0:
            ax3.text(rect.get_x() + rect.get_width() / 2, h + 0.01,
                     f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    output_filename = 'vendi_multimodal_density.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"✨ Plot saved as '{output_filename}'")
    plt.show()


# ─── RUN ANALYSIS ──────────────────────────────────────────────────────
dir_proposed = "/content/outputs/diverse"    # your proposed / diverse images
dir_baseline = "/content/outputs/original"   # your baseline generated images
dir_real     = "/content/data/real"          # ← your real reference dataset here

# Single string if all images share one prompt;
# list of strings (one per image, sorted filename order) if prompts vary.
prompt = "a photo of an astronaut riding a horse on mars"

plot_multimodal_distribution(dir_proposed, dir_baseline, dir_real, prompts=prompt)