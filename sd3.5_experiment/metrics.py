"""
metrics.py
----------
Unified evaluation metrics for the UGILE hyperparameter sweep.

  - FID, Precision, Recall   : Inception-v3 features (adapted from the
                                provided evaluation code)
  - KID                      : added — unbiased polynomial-kernel MMD^2
                                (Binkowski et al. 2018), subset-averaged
  - CLIP Score                : adapted from the provided evaluation code
  - Vendi Score (per-prompt)  : vendi_score library on CLIP features,
                                computed WITHIN each prompt's seed group
                                and averaged — this is the correct measure
                                of intra-prompt diversity gained by UGILE.

All heavy models (CLIP, Inception) are loaded once and reused across the
whole sweep via `load_clip_bundle()` / the internal Inception cache.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.models import inception_v3
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPVisionModel
from vendi_score import vendi

VALID_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ══════════════════════════════════════════════════════════════════════ #
#  MODEL LOADING (once, reused across the whole sweep)                   #
# ══════════════════════════════════════════════════════════════════════ #

def load_clip_bundle(clip_model_name="openai/clip-vit-base-patch32", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor      = CLIPProcessor.from_pretrained(clip_model_name)
    clip_vision    = CLIPVisionModel.from_pretrained(clip_model_name).to(device).eval()
    clip_full      = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    return {
        "processor": processor, "vision": clip_vision, "full": clip_full,
        "tokenizer": clip_tokenizer, "device": device,
    }


_INCEPTION = None

def _get_inception(device):
    global _INCEPTION
    if _INCEPTION is None:
        inc = inception_v3(pretrained=True, transform_input=False).to(device)
        inc.fc = torch.nn.Identity()
        inc.eval()
        _INCEPTION = inc
    return _INCEPTION


def _list_images(folder):
    folder = str(folder)
    if not os.path.exists(folder):
        return []
    return [
        os.path.join(folder, f) for f in sorted(os.listdir(folder))
        if f.lower().endswith(VALID_EXT)
    ]


# ══════════════════════════════════════════════════════════════════════ #
#  FEATURE EXTRACTION                                                    #
# ══════════════════════════════════════════════════════════════════════ #

def clip_image_features(image_paths, bundle):
    processor, model, device = bundle["processor"], bundle["vision"], bundle["device"]
    feats = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.inference_mode():
                f = model(**inputs).pooler_output
            feats.append(f.cpu().numpy().flatten())
        except Exception:
            pass
    return np.array(feats) if feats else None


def inception_features(image_paths, device, img_size=299, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tensors = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        except Exception:
            pass
    if not tensors:
        return None
    tensors = torch.stack(tensors)
    inc = _get_inception(device)
    feats = []
    with torch.inference_mode():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i + batch_size].to(device)
            feats.append(inc(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


# ══════════════════════════════════════════════════════════════════════ #
#  FID                                                                    #
# ══════════════════════════════════════════════════════════════════════ #

def compute_fid(feat_real, feat_gen):
    mu_r, sigma_r = feat_real.mean(axis=0), np.cov(feat_real, rowvar=False)
    mu_g, sigma_g = feat_gen.mean(axis=0),  np.cov(feat_gen,  rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))


# ══════════════════════════════════════════════════════════════════════ #
#  KID  (added)                                                          #
# ══════════════════════════════════════════════════════════════════════ #

def _poly_kernel(x, y, degree=3, gamma=None, coef0=1.0):
    gamma = gamma or (1.0 / x.shape[1])
    return (gamma * x @ y.T + coef0) ** degree


def compute_kid(feat_real, feat_gen, n_subsets=100, subset_size=None, seed=0):
    """Unbiased polynomial-kernel MMD^2 estimator (Binkowski et al., 2018),
    averaged over random subsets of `subset_size` — the standard approach
    for making KID stable when the sample count is small (per-combo runs
    here typically have O(#prompts) images, not O(50k))."""
    m = min(len(feat_real), len(feat_gen))
    if m < 2:
        return float("nan")
    subset_size = subset_size or max(2, min(50, m))
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_subsets):
        idx_r = rng.choice(len(feat_real), subset_size, replace=False)
        idx_g = rng.choice(len(feat_gen),  subset_size, replace=False)
        x, y = feat_real[idx_r], feat_gen[idx_g]
        n = subset_size
        Kxx = _poly_kernel(x, x)
        Kyy = _poly_kernel(y, y)
        Kxy = _poly_kernel(x, y)
        sum_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        sum_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
        sum_xy = Kxy.sum() / (n * n)
        scores.append(sum_xx + sum_yy - 2 * sum_xy)
    return float(np.mean(scores))


# ══════════════════════════════════════════════════════════════════════ #
#  PRECISION & RECALL  (Kynkäänniemi et al., 2019)                       #
# ══════════════════════════════════════════════════════════════════════ #

def _manifold_radii(features, k=3):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(features)
    d, _ = nn.kneighbors(features)
    return d[:, -1]


def compute_precision_recall(feat_real, feat_gen, k=3):
    from sklearn.neighbors import NearestNeighbors

    radii_real = _manifold_radii(feat_real, k)
    radii_gen  = _manifold_radii(feat_gen, k)

    nn_real = NearestNeighbors(n_neighbors=1).fit(feat_real)
    d1, idx1 = nn_real.kneighbors(feat_gen)
    precision = float(np.mean(d1[:, 0] <= radii_real[idx1[:, 0]]))

    nn_gen = NearestNeighbors(n_neighbors=1).fit(feat_gen)
    d2, idx2 = nn_gen.kneighbors(feat_real)
    recall = float(np.mean(d2[:, 0] <= radii_gen[idx2[:, 0]]))

    return precision, recall


# ══════════════════════════════════════════════════════════════════════ #
#  CLIP SCORE  (Hessel et al., 2021)                                     #
# ══════════════════════════════════════════════════════════════════════ #

def compute_clip_score_folder(folder, prompts, bundle):
    """Images must be named 1.png, 2.png, ... matched by position to `prompts`
    (this matches how run_experiment.py names the eval images)."""
    model, processor = bundle["full"], bundle["processor"]
    tokenizer, device = bundle["tokenizer"], bundle["device"]

    image_paths = _list_images(folder)
    if not image_paths:
        return None
    n = min(len(image_paths), len(prompts))
    scores = []
    for i in range(n):
        try:
            img = Image.open(image_paths[i]).convert("RGB")
            img_in = processor(images=img, return_tensors="pt").to(device)
            txt_in = tokenizer([prompts[i]], return_tensors="pt",
                                padding=True, truncation=True).to(device)
            with torch.inference_mode():
                img_f = model.get_image_features(**img_in)
                txt_f = model.get_text_features(**txt_in)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            scores.append((img_f @ txt_f.T).item() * 100)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else None


# ══════════════════════════════════════════════════════════════════════ #
#  VENDI SCORE — per-prompt intra-diversity, averaged                    #
# ══════════════════════════════════════════════════════════════════════ #

def average_per_prompt_vendi(vendi_root, bundle):
    """
    Expects: vendi_root/<prompt_idx>/seed*.png  (one subfolder per prompt,
    each containing the N seed images for that prompt — see run_experiment.py).

    Computes one Vendi Score per prompt (diversity across the N seeds for
    THAT prompt) and averages over all prompts. This is the metric that
    actually reflects UGILE's job: producing varied images for the SAME
    prompt, not diversity across unrelated prompts/classes.
    """
    root = Path(vendi_root)
    if not root.exists():
        return None
    scores = []
    for prompt_dir in sorted(root.iterdir()):
        if not prompt_dir.is_dir():
            continue
        paths = _list_images(prompt_dir)
        if len(paths) < 2:
            continue
        feats = clip_image_features(paths, bundle)
        if feats is None or len(feats) < 2:
            continue
        scores.append(vendi.score_dual(feats, normalize=True))
    return float(np.mean(scores)) if scores else None


# ══════════════════════════════════════════════════════════════════════ #
#  FOLDER-VS-FOLDER COMPARISON (FID + KID + Precision + Recall)          #
# ══════════════════════════════════════════════════════════════════════ #

def compare_folders(folder_gen, folder_ref, device=None, k=3):
    """FID, KID, Precision, Recall between two folders of images, using
    `folder_ref` as the reference manifold (real data, or the baseline
    model's outputs)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths_gen = _list_images(folder_gen)
    paths_ref = _list_images(folder_ref)
    if not paths_gen or not paths_ref:
        return (float("nan"),) * 4

    feat_gen = inception_features(paths_gen, device)
    feat_ref = inception_features(paths_ref, device)
    if feat_gen is None or feat_ref is None:
        return (float("nan"),) * 4

    fid = compute_fid(feat_ref, feat_gen)
    kid = compute_kid(feat_ref, feat_gen)
    prec, rec = compute_precision_recall(feat_ref, feat_gen, k=k)
    return fid, kid, prec, rec
