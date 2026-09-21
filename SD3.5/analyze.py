"""
analyze.py — two subcommands.

  python analyze.py diag runs/A/*/diverse                      # stdlib only; reads records_offset*.json
  python analyze.py metrics --prompts runs/B/prompts.yaml \
        --baseline runs/B/original --arm main=runs/B/main/diverse --arm noortho=runs/B/noortho/diverse \
        [--real_dir /path/to/real/images] --out metrics_B.json

`metrics` is a quick decision tool for the sweep (per-prompt Vendi over seeds, CLIP score, paired similarity
to baseline, optional PRDC). For the FINAL paper table use your existing 7-metric pipeline so numbers are
comparable with CADS/DAVE.
"""
import argparse, glob, json, math, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# ───────────────────────── metric math (numpy only) ─────────────────────────
def vendi_score(X):
    """Vendi score (q=1) with cosine kernel. X: [n, d]. Range [1, n]."""
    X = np.asarray(X, dtype=np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n = len(X)
    K = X @ X.T / n
    w = np.clip(np.linalg.eigvalsh(K), 0, None)
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))

def prdc(real, fake, k=5):
    """Naeem et al. 2020: precision, recall, density, coverage."""
    real = np.asarray(real, np.float64); fake = np.asarray(fake, np.float64)
    def pd(a, b):
        return np.sqrt(np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2 * a @ b.T, 0))
    dr = pd(real, real); np.fill_diagonal(dr, np.inf)
    rad = np.sort(dr, axis=1)[:, k - 1]                     # k-th NN radius per real sample
    d_rf = pd(real, fake)                                   # [n_real, n_fake]
    inside = d_rf < rad[:, None]
    precision = inside.any(0).mean()
    df = pd(fake, fake); np.fill_diagonal(df, np.inf)
    radf = np.sort(df, axis=1)[:, k - 1]                    # k-th NN radius per fake sample
    recall = (d_rf.T < radf[:, None]).any(0).mean()         # real sample inside any fake ball
    density = inside.sum(0).mean() / k
    coverage = (d_rf.min(1) < rad).mean()
    return dict(precision=float(precision), recall=float(recall), density=float(density), coverage=float(coverage))

def boot_ci(diffs, n=5000, seed=0):
    """95% bootstrap CI of the mean of per-prompt differences."""
    d = np.asarray(diffs, float); rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

# ───────────────────────────── diag subcommand ─────────────────────────────
def cmd_diag(dirs):
    print(f"{'arm':28s} {'n':>3s} {'σ-window':>13s} {'Σrel_disp':>9s} {'θ(√Σ²)':>7s} {'cap_hit':>7s} "
          f"{'cos_xN mean/min':>16s} {'cos(e0,eL)':>10s} {'max|cos(e,s)|':>13s}  verdict")
    for d in dirs:
        files = sorted(glob.glob(str(Path(d) / "records_offset*.json")))
        if not files:
            print(f"{d}: no records_offset*.json"); continue
        recs = [r for f in files for r in json.load(open(f))["records"]]
        tr = [r for r in recs if r.get("trace")]
        if not tr:
            print(f"{d}: records have no trace (two_pass mode?)"); continue
        sum_rel = np.mean([sum(t["rel_disp"] for t in r["trace"]) for r in tr])
        theta = np.mean([r["theta"] for r in tr])
        cap = np.mean([r["cap_hit_rate"] for r in tr])
        cx = [r["cos_xN"] for r in tr if r.get("cos_xN") is not None]
        cxs = f"{np.mean(cx):.3f}/{np.min(cx):.3f}" if cx else "n/a"
        c0L = np.mean([r["cos_e_first_last"] for r in tr])
        maxes = max(abs(t["cos_e_s"]) for r in tr for t in r["trace"])
        sw = tr[0]["sigma_window"]; sws = f"{sw[0]:.2f}→{sw[1]:.2f}"
        verdict = ""
        if cx:
            m = np.mean(cx)
            verdict = "WEAK (≈baseline)" if m > 0.98 else ("STRONG (inspect images)" if m < 0.5 else "moderate")
        name = str(Path(d).parent.name)
        print(f"{name:28s} {len(tr):3d} {sws:>13s} {sum_rel:9.4f} {theta:7.4f} {cap:7.2f} {cxs:>16s} {c0L:10.3f} {maxes:13.4f}  {verdict}")

# ─────────────────────── metrics subcommand (needs torch) ───────────────────
PAT = re.compile(r"^(\d+)_seed(\d+)(?:_branch(\d+))?\.(png|jpg|jpeg|webp)$")
def scan(d):
    out = {}
    for p in sorted(Path(d).iterdir()):
        m = PAT.match(p.name)
        if m: out[(int(m.group(1)), int(m.group(2)))] = p
    return out

def extract_features(paths, texts=None, device="cuda"):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor
    dino_p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
    F_d, F_ci, F_ct = [], [], []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i+32]]
            o = dino(**{k: v.to(device) for k, v in dino_p(images=ims, return_tensors="pt").items()})
            F_d.append(o.last_hidden_state[:, 0].float().cpu().numpy())
            ci = clip.get_image_features(**{k: v.to(device) for k, v in clip_p(images=ims, return_tensors="pt").items()})
            F_ci.append(ci.float().cpu().numpy())
            if texts is not None:
                tk = clip_p(text=texts[i:i+32], return_tensors="pt", padding=True, truncation=True, max_length=77)
                F_ct.append(clip.get_text_features(**{k: v.to(device) for k, v in tk.items()}).float().cpu().numpy())
    cat = np.concatenate
    return cat(F_d), cat(F_ci), (cat(F_ct) if texts is not None else None)

def cmd_metrics(a):
    import yaml
    prompts = yaml.safe_load(open(a.prompts))["prompts"]
    arms = {"baseline": scan(a.baseline)}
    for s in a.arm:
        name, path = s.split("=", 1); arms[name] = scan(path)
    keys = sorted(set.intersection(*[set(v) for v in arms.values()]))     # (prompt_idx, seed) present in ALL arms
    print(f"[info] {len(keys)} common (prompt,seed) pairs across {list(arms)}")
    feats = {}
    for name, m in arms.items():
        paths = [m[k] for k in keys]; texts = [prompts[k[0] - 1] for k in keys]
        feats[name] = extract_features(paths, texts, a.device)
    res = {}
    by_prompt = defaultdict(list)
    for i, k in enumerate(keys): by_prompt[k[0]].append(i)
    def per_prompt_vendi(F):
        return {p: vendi_score(F[idx]) for p, idx in by_prompt.items() if len(idx) >= 2}
    vb = per_prompt_vendi(feats["baseline"][0])
    def clip_scores(Fci, Fct):
        Fci = Fci / np.linalg.norm(Fci, axis=1, keepdims=True); Fct = Fct / np.linalg.norm(Fct, axis=1, keepdims=True)
        return 100 * (Fci * Fct).sum(1)
    cb = clip_scores(feats["baseline"][1], feats["baseline"][2]).mean()
    real_f = None
    if a.real_dir:
        rp = sorted(p for p in Path(a.real_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        real_f = extract_features(rp, None, a.device)[0]
    print(f"\n{'arm':14s} {'Vendi':>7s} {'ΔVendi [95% CI]':>26s} {'CLIP':>6s} {'ΔCLIP':>7s} {'cos→base':>9s}" + ("  P     R     D     C" if real_f is not None else ""))
    for name, (Fd, Fci, Fct) in feats.items():
        v = per_prompt_vendi(Fd); common = sorted(set(v) & set(vb))
        diffs = [v[p] - vb[p] for p in common]
        m, lo, hi = boot_ci(diffs) if diffs else (0, 0, 0)
        cs = clip_scores(Fci, Fct).mean()
        A = Fd / np.linalg.norm(Fd, axis=1, keepdims=True); B = feats["baseline"][0]; B = B / np.linalg.norm(B, axis=1, keepdims=True)
        cosb = (A * B).sum(1).mean()
        row = dict(vendi=float(np.mean([v[p] for p in common])), dvendi=m, ci=[lo, hi], clip=float(cs), dclip=float(cs - cb), cos_to_base=float(cosb))
        line = f"{name:14s} {row['vendi']:7.3f} {m:+8.3f} [{lo:+.3f},{hi:+.3f}] {cs:6.2f} {cs-cb:+7.2f} {cosb:9.3f}"
        if real_f is not None:
            q = prdc(real_f, Fd, k=a.k); row.update(q)
            line += f"  {q['precision']:.3f} {q['recall']:.3f} {q['density']:.3f} {q['coverage']:.3f}"
        print(line); res[name] = row
    if a.out: json.dump(res, open(a.out, "w"), indent=1)

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diag"); d.add_argument("dirs", nargs="+")
    m = sub.add_parser("metrics")
    m.add_argument("--prompts", required=True); m.add_argument("--baseline", required=True)
    m.add_argument("--arm", action="append", required=True, help="name=diverse_dir")
    m.add_argument("--real_dir", default=None); m.add_argument("--k", type=int, default=5)
    m.add_argument("--device", default="cuda"); m.add_argument("--out", default=None)
    a = ap.parse_args()
    cmd_diag(a.dirs) if a.cmd == "diag" else cmd_metrics(a)

if __name__ == "__main__":
    main()
