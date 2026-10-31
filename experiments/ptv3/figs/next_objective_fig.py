"""bento 3.0 design figure: proposed next training objective (no training code)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
INK, MUTED, LINE = "#1f2328", "#59636e", "#d0d7de"
GRAY, BLUE, TEAL, AMBER, RED = "#8c959f", "#2f6fdf", "#1a8a7a", "#c98a00", "#cf3a3a"

fig = plt.figure(figsize=(17, 8.0), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 170); ax.set_ylim(0, 80); ax.axis("off")

def box(x, y, w, h, text, ec=LINE, fc="white", fs=10.5, color=INK, bold=False, lw=1.3, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                ec=ec, fc=fc, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            weight="bold" if bold else "normal", linespacing=1.35)

def arrow(x0, y0, x1, y1, color=MUTED, ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))

def cloud(cx, cy, r, n, seed, drop=0.0, jitter=0.0, rot=0.0, mask_idx=None):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 2)) * r * 0.38
    pts = pts[np.hypot(*pts.T) < r]
    keep = rng.random(len(pts)) > drop
    pts = pts[keep]
    c, s = np.cos(rot), np.sin(rot)
    pts = pts @ np.array([[c, -s], [s, c]]) + rng.normal(size=pts.shape) * jitter
    cols = rng.choice([BLUE, TEAL, AMBER, "#8250df", "#bf3989"], size=len(pts))
    ax.add_patch(Circle((cx, cy), r * 0.42, fc="#f6f8fa", ec=LINE, lw=1, zorder=1))  # nucleus hint
    ax.scatter(pts[:, 0] + cx, pts[:, 1] + cy, s=9, c=cols, lw=0, zorder=2)
    if mask_idx is not None:
        ax.scatter([cx + 1.2], [cy + 0.8], s=70, fc="white", ec=INK, lw=1.2, zorder=3)
        ax.text(cx + 1.2, cy + 0.75, "?", ha="center", va="center", fontsize=8, weight="bold", zorder=4)

def panel(x, w, letter, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, 3), w, 66, boxstyle="round,pad=0,rounding_size=2",
                                ec=LINE, fc="#fbfcfd", lw=1))
    ax.text(x + 2.5, 64.5, letter, fontsize=17, weight="bold", color=color, va="center")
    ax.text(x + 7, 64.8, title, fontsize=13.5, weight="bold", color=INK, va="center")
    ax.text(x + 7, 60.9, sub, fontsize=10, color=MUTED, va="center")

fig.text(0.012, 0.965, "bento 3.0 — next training objective", fontsize=16, weight="bold", color=INK, va="top")
fig.text(0.012, 0.925, "Segmentation-free PTv3 over molecules; hierarchy inferred afterwards. Proposal only — no code changed.",
         fontsize=10.5, color=MUTED, va="top")

# ---------------- A: current ----------------
panel(2, 48, "A", "Current: masked gene classification", "what we trained (7 arms, seed 0, 15k steps)", GRAY)
cloud(13, 47, 8, 160, 1, mask_idx=0)
ax.text(13, 36.3, "molecules only\n(one masked)", ha="center", fontsize=9, color=MUTED, va="top")
box(25, 49.5, 22, 7, "gene id\n+ frozen NuSPIRe DAPI emb", fs=9.5)
box(25, 39, 22, 6.5, "PTv3 encoder", ec=GRAY, bold=True, fs=10.5)
box(25, 29, 22, 6.5, "predict masked\nmolecule's gene", fs=9.5)
arrow(21.5, 50, 24.7, 52.5); arrow(36, 49.2, 36, 45.8); arrow(36, 38.7, 36, 35.8)
# result
box(5, 7, 42, 17.5, "", ec=RED, fc="#fff5f5", lw=1.2)
ax.text(26, 20.6, "≈ 10% top-1 in every arm", ha="center", fontsize=12, weight="bold", color=RED)
ax.text(26, 13.8, "shuffled DAPI ≈ real DAPI\n→ the objective never needs the image;\n    single-label target is noisy & abundance-driven",
        ha="center", va="center", fontsize=9.6, color=INK, linespacing=1.4)

# ---------------- B: proposed ----------------
panel(53, 60, "B", "Proposed: two-view self-distillation", "PTv3-style pretraining (Sonata / Concerto)", BLUE)
cloud(62, 47, 7, 200, 2)
ax.text(62, 37.2, "tissue window", ha="center", fontsize=9, color=MUTED, va="top")
cloud(77, 53, 5.2, 120, 3, drop=0.3, jitter=0.25, rot=0.5)
cloud(77, 38.5, 5.2, 120, 4, drop=0.45, jitter=0.35, rot=-0.7)
ax.text(77, 46, "crop · drop/mask · jitter · rotate", ha="center", fontsize=8.3, color=MUTED, va="center")
arrow(69.5, 48.5, 71.5, 51.5); arrow(69.5, 45.5, 71.5, 41)
box(86, 50, 13, 6, "teacher\nPTv3 (EMA)", ec=BLUE, ls="--", fs=9.8)
box(86, 35.5, 13, 6, "student\nPTv3", ec=BLUE, bold=True, fs=9.8)
arrow(82.6, 53, 85.7, 53); arrow(82.6, 38.5, 85.7, 38.5)
ax.annotate("", xy=(92.5, 49.7), xytext=(92.5, 41.8),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1, ls=":", mutation_scale=10))
ax.text(93.4, 45.7, "EMA", fontsize=8, color=BLUE, va="center")
box(102, 41.5, 10, 9, "align\nembed-\ndings", ec=BLUE, fc="#f1f6ff", fs=9.2, bold=True)
arrow(99.3, 53, 104, 50.8, color=BLUE); arrow(99.3, 38.5, 104, 41.2, color=BLUE)
# cross-modal
box(57, 7, 54, 21.5, "", ec=TEAL, fc="#f2fbf9", ls="--", lw=1.1)
ax.text(59, 25.5, "+ optional cross-modal heads (make the image matter)", fontsize=10, weight="bold", color=TEAL, va="center")
box(59.5, 10, 23.5, 11.5, "masked neighborhood\ngene composition\n(distribution, not 1 label)", fs=9)
box(85.5, 10, 23.5, 11.5, "held-out DAPI patch\nembedding\npredicted from molecules", fs=9)
arrow(92.5, 34.9, 92.5, 28.9, color=TEAL)

# ---------------- C: evaluate + fine-tune ----------------
panel(116, 52, "C", "Evaluate, then fine-tune", "frozen embeddings first; adapt only if they work", TEAL)
rows = [
    ("linear probes", "nucleus vs cytoplasm · cell type", 51.5),
    ("abundance-aware", "macro recall · accuracy per abundance bin", 43.5),
    ("controls kept", "shuffled-DAPI · DAPI-intensity only", 35.5),
    ("hierarchy post hoc", "cluster embeddings → cell-like / niche-like", 27.5),
]
for head, body, y in rows:
    ax.add_patch(Circle((120.5, y + 0.2), 0.9, fc=TEAL, ec="none"))
    ax.text(123, y + 1.4, head, fontsize=10.3, weight="bold", color=INK, va="center")
    ax.text(123, y - 1.6, body, fontsize=9.3, color=MUTED, va="center")
box(119, 7, 46, 14, "", ec=AMBER, fc="#fffaf0", lw=1.1)
ax.text(121, 18.2, "fine-tune from pretrained (Utonia / Sonata)", fontsize=10, weight="bold", color=AMBER, va="center")
ax.text(121, 15, "LoRA on attention  vs  full fine-tune", fontsize=9.6, color=INK, va="center")
ax.text(121, 11.6, "LoRA: less memory/time, but won't fix a weak objective", fontsize=8.9, color=MUTED, va="center")
ax.text(121, 8.9, "open question: Utonia arms did worst → does transfer help?", fontsize=8.9, color=MUTED, va="center")

# panel-to-panel arrows
arrow(50.3, 36, 52.7, 36, color=GRAY, lw=1.8); arrow(113.3, 36, 115.7, 36, color=GRAY, lw=1.8)

out = Path(__file__).parent
for ext in ("png", "svg"):
    fig.savefig(out / f"next_objective.{ext}", facecolor="white")
print("ok")
