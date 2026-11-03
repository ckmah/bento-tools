"""bento 3.0 design figure: the training objective (what we tried, what bento3_ptv3 trains now, how it is scored).
Writes training_objective.{png,svg} next to this file."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from pathlib import Path

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
INK, MUTED, LINE = "#1f2328", "#59636e", "#d0d7de"
GRAY, BLUE, TEAL, AMBER, RED = "#8c959f", "#2f6fdf", "#1a8a7a", "#c98a00", "#cf3a3a"
GENES = [BLUE, TEAL, AMBER, "#8250df", "#bf3989"]

fig = plt.figure(figsize=(17, 8.0), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 170); ax.set_ylim(0, 80); ax.axis("off")


def box(x, y, w, h, text, ec=LINE, fc="white", fs=10.5, color=INK, bold=False, lw=1.3, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", ec=ec, fc=fc, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight="bold" if bold else "normal", linespacing=1.35)


def arrow(x0, y0, x1, y1, color=MUTED, ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def cell(cx, cy, r, n, seed, masks=(), big_mask=None):
    """Molecules of one cell (nucleus hint), with masked squares (side in plot units) drawn on top."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 2)) * r * 0.38; pts = pts[np.hypot(*pts.T) < r]
    cols = rng.choice(GENES, size=len(pts))
    ax.add_patch(Circle((cx, cy), r * 1.02, fc="none", ec=LINE, lw=1, ls="--", zorder=1))   # cell outline (eval label only)
    ax.add_patch(Circle((cx, cy), r * 0.42, fc="#f6f8fa", ec=LINE, lw=1, zorder=1))          # nucleus (eval label only)
    keep = np.ones(len(pts), bool)
    for (mx, my, s) in list(masks) + ([big_mask] if big_mask else []):
        inside = (np.abs(pts[:, 0] + cx - mx) < s / 2) & (np.abs(pts[:, 1] + cy - my) < s / 2); keep &= ~inside
    ax.scatter(pts[keep, 0] + cx, pts[keep, 1] + cy, s=9, c=cols[keep], lw=0, zorder=2)
    for (mx, my, s) in masks:
        ax.add_patch(Rectangle((mx - s / 2, my - s / 2), s, s, fc="white", ec=INK, lw=1.0, zorder=3))
        ax.text(mx, my - 0.05, "?", ha="center", va="center", fontsize=7.5, weight="bold", zorder=4)
    if big_mask:
        mx, my, s = big_mask
        ax.add_patch(Rectangle((mx - s / 2, my - s / 2), s, s, fc="#ffffffcc", ec=RED, lw=1.2, zorder=3))
        ax.text(mx, my, "?", ha="center", va="center", fontsize=12, weight="bold", color=RED, zorder=4)


def panel(x, w, letter, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, 3), w, 66, boxstyle="round,pad=0,rounding_size=2", ec=LINE, fc="#fbfcfd", lw=1))
    ax.text(x + 2.5, 64.5, letter, fontsize=17, weight="bold", color=color, va="center")
    ax.text(x + 7, 64.8, title, fontsize=13.5, weight="bold", color=INK, va="center")
    ax.text(x + 7, 60.9, sub, fontsize=10, color=MUTED, va="center")


fig.text(0.012, 0.965, "bento 3.0 — training objective", fontsize=16, weight="bold", color=INK, va="top")
fig.text(0.012, 0.925, "Segmentation-free PTv3 over molecules, aimed at subcellular embeddings. No labels in training; "
         "segmentation is used only to score.", fontsize=10.5, color=MUTED, va="top")

# ---------------- A: what we tried ----------------
panel(2, 48, "A", "Tried: cell-scale targets", "masked gene classification · then DINO with 10 µm masks", GRAY)
cell(14, 46, 8, 170, 1, big_mask=(15, 46, 11))
ax.text(14, 35.3, "10 µm mask ≈ a whole cell:\nonly neighbouring cells visible", ha="center", fontsize=8.8, color=MUTED, va="top")
box(27, 49.5, 20.5, 7, "target: one hidden\nmolecule's identity", fs=9.4)
box(27, 39.5, 20.5, 6.5, "PTv3 + LoRA\n(EMA teacher, Sinkhorn)", ec=GRAY, fs=9.2)
arrow(24, 47, 26.7, 51.5); arrow(37, 49.2, 37, 46.3)
box(5, 7, 42, 22, "", ec=RED, fc="#fff5f5", lw=1.2)
ax.text(26, 25.6, "what went wrong", ha="center", fontsize=11.5, weight="bold", color=RED)
ax.text(7, 16.5, "• one molecule's gene is barely predictable:\n   20-NN context → 8.8% top-1 vs 5.4% majority\n"
        "• embedding pushed to cell / niche scale\n• LoRA proj in level 3 grew until ΔW ≈ W\n   (encoder blew up; frozen LoRA was stable)",
        ha="left", va="center", fontsize=9.3, color=INK, linespacing=1.45)

# ---------------- B: now ----------------
panel(53, 60, "B", "Now: subcellular masked reconstruction", "objective='recon' (Point-MAE style; default)", BLUE)
masks = [(60.5, 49.5, 2.6), (65.5, 44.5, 2.6), (62.5, 41.5, 2.0), (67.5, 50.5, 3.0)]
cell(63.5, 46, 8, 220, 2, masks=masks)
ax.text(65, 35.3, "~40% of 1–3 µm squares removed,\none query point (mask token) each", ha="center", fontsize=8.8, color=MUTED, va="top")
box(77, 43, 14, 7.5, "PTv3 encoder\n(frozen + LoRA)", ec=BLUE, bold=True, fs=9.6)
arrow(73.2, 46.5, 76.7, 46.5, color=BLUE)
box(94, 43, 15.5, 7.5, "query\nembedding (64-d)", ec=BLUE, fc="#f1f6ff", fs=9.4)
arrow(91.3, 46.5, 93.7, 46.5, color=BLUE)
ax.text(83, 39.8, "sees only the visible molecules,\nmostly of the same cell", ha="center", fontsize=8.5, color=MUTED, va="top")
# targets
box(57, 9, 54, 22.5, "", ec=TEAL, fc="#f2fbf9", ls="--", lw=1.1)
ax.text(59, 28.0, "predict what each removed square held, as log1p counts\n(the log scale keeps abundant genes and dense squares in check)", fontsize=10, weight="bold", color=TEAL, va="center")
for i, (h, b) in enumerate([("count", "how many\nmolecules"), ("layout", "radial × z bins\n(rotation-invariant)"), ("genes", "count per gene\n(280 genes)")]):
    box(59.5 + i * 17, 11.5, 15.5, 13, f"{h}\n\n{b}", fs=8.9)
arrow(101.7, 42.7, 101.7, 32.0, color=TEAL)
ax.text(103, 37.5, "fixed targets:\nno teacher,\nno collapse", fontsize=8.2, color=TEAL, va="center")
ax.text(84, 4.9, "compared with objective='dino_pooled' (teacher targets averaged per square)\nand 'dino' at the same 1–3 µm mask scale", ha="center",
        fontsize=8.0, color=MUTED, va="center", linespacing=1.3)

# ---------------- C: evaluate ----------------
panel(116, 52, "C", "Score at subcellular scale", "linear probes on frozen embeddings, held-out 1 mm blocks", TEAL)
rows = [
    ("nucleus vs cytoplasm", "AUROC", 51.5),
    ("signed distance to nucleus boundary", "R², Spearman ρ  (µm; < 0 inside)", 43.5),
    ("distance to cell boundary", "R², Spearman ρ  (µm)", 35.5),
    ("10x cell cluster", "balanced accuracy (cell-level, secondary)", 27.5),
]
for head, body, y in rows:
    ax.add_patch(Circle((120.5, y + 0.2), 0.9, fc=TEAL, ec="none"))
    ax.text(123, y + 1.4, head, fontsize=10.3, weight="bold", color=INK, va="center")
    ax.text(123, y - 1.6, body, fontsize=9.3, color=MUTED, va="center")
box(119, 7, 46, 14, "", ec=AMBER, fc="#fffaf0", lw=1.1)
ax.text(121, 18.2, "the bar: beat raw local composition", fontsize=10, weight="bold", color=AMBER, va="center")
ax.text(121, 14.9, "comp20 = gene counts of the 20 nearest molecules", fontsize=9.4, color=INK, va="center")
ax.text(121, 11.8, "(≈ 2.6 µm radius; nucleus AUROC 0.82 on its own)", fontsize=8.9, color=MUTED, va="center")
ax.text(121, 9.0, "labels from 10x segmentation, used only here", fontsize=8.9, color=MUTED, va="center")

arrow(50.3, 36, 52.7, 36, color=GRAY, lw=1.8); arrow(113.3, 36, 115.7, 36, color=GRAY, lw=1.8)

out = Path(__file__).parent
for ext in ("png", "svg"):
    fig.savefig(out / f"training_objective.{ext}", facecolor="white")
print("ok")
