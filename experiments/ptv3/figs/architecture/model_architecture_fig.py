"""bento3_ptv3 model architecture: what each part sees and learns (objective='recon'). Writes model_architecture.{png,svg}.
Drawn schematically; sizes and counts are the real ones (4-5 um squares, 36-d stem, 5 encoder levels 36..432 ch, 64-d embedding)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from pathlib import Path

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
INK, MUTED, LINE = "#1f2328", "#59636e", "#d0d7de"
TRAIN, FROZEN, OBJ, EVAL, DATA = "#2f6fdf", "#8c959f", "#d9822b", "#1a8a7a", "#a07a55"
GENES = ["#2f6fdf", "#1a8a7a", "#c98a00", "#8250df", "#bf3989", "#cf3a3a"]
rng = np.random.default_rng(3)

fig = plt.figure(figsize=(17, 8.9), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 170); ax.set_ylim(0, 89); ax.axis("off")


def panel(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", ec=LINE, fc="#fbfcfd", lw=1))
    ax.text(x + 1.6, y + h - 2.6, title, fontsize=11.5, weight="bold", color=color, va="center")
    ax.text(x + 1.6, y + h - 5.4, sub, fontsize=8.6, color=MUTED, va="center", linespacing=1.25)


def arrow(x0, y0, x1, y1, color=MUTED, lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def tag(x, y, text, color):
    ax.text(x, y, text, fontsize=7.6, weight="bold", color="white", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.5", fc=color, ec="none"))


ax.text(2, 86.4, "bento3_ptv3 — what each part sees and learns", fontsize=16, weight="bold", color=INK, va="center")
ax.text(2, 83.0, "objective='recon'.  Colour = role:", fontsize=10.5, color=MUTED, va="center")
for x, (t, c) in zip((28.5, 41, 61, 76), [("trained", TRAIN), ("pretrained, frozen", FROZEN), ("objective", OBJ), ("eval only (labels)", EVAL)]):
    ax.text(x, 83.0, f"■ {t}", fontsize=10.5, color=c, va="center", weight="bold")

# =============================== TRAINING PATH (top row) ===============================
ax.text(2, 78.6, "TRAINING  ·  every step  ·  training blocks only  ·  no labels", fontsize=10, weight="bold", color=INK)
Y, H = 44, 33

# 1 window of molecules
panel(2, Y, 26, H, "1  Data", "100 µm window, molecules only\n(gene, x, y, z)", DATA)
pts = rng.uniform(0, 1, (900, 2)); keep = rng.random(900) < 0.55 + 0.4 * np.sin(pts[:, 0] * 6) * np.cos(pts[:, 1] * 5)
pts = pts[keep]; cols = rng.choice(GENES, len(pts))
X0, Y0, S = 5, Y + 3.5, 20
ax.add_patch(Rectangle((X0, Y0), S, S, fc="white", ec=DATA, lw=1.2))
ax.scatter(X0 + pts[:, 0] * S, Y0 + pts[:, 1] * S, s=2.2, c=cols, lw=0)
zx, zy, zs = X0 + 0.55 * S, Y0 + 0.5 * S, 0.22 * S
ax.add_patch(Rectangle((zx, zy), zs, zs, fc="none", ec=INK, lw=0.9, ls="--"))

# 2 masking: zoom with removed squares + query points
panel(30, Y, 26, H, "2  Mask", "~40% of 4–5 µm squares removed;\none query point (mask token) each", DATA)
X1, Y1, S1 = 33, Y + 3.5, 20
ax.add_patch(Rectangle((X1, Y1), S1, S1, fc="white", ec=DATA, lw=1.2))
zp = rng.uniform(0, 1, (230, 2)); zc = rng.choice(GENES, len(zp))
sq = [(0.12, 0.55), (0.55, 0.15), (0.6, 0.62)]; w = 0.3
inside = np.zeros(len(zp), bool)
for (a, b) in sq: inside |= (zp[:, 0] > a) & (zp[:, 0] < a + w) & (zp[:, 1] > b) & (zp[:, 1] < b + w)
ax.scatter(X1 + zp[~inside, 0] * S1, Y1 + zp[~inside, 1] * S1, s=7, c=zc[~inside], lw=0)
for (a, b) in sq:
    ax.add_patch(Rectangle((X1 + a * S1, Y1 + b * S1), w * S1, w * S1, fc="#f6f8fa", ec=MUTED, lw=0.9, ls="--"))
    ax.scatter([X1 + (a + w / 2) * S1], [Y1 + (b + w / 2) * S1], s=46, fc="white", ec=INK, lw=1.4, zorder=3)
ax.text(X1 + S1 / 2, Y1 - 1.6, "removed molecules → targets only", fontsize=7.6, color=OBJ, ha="center", va="center")
arrow(25.6, Y + 14, 32.6, Y + 14)

# 3 stem
panel(58, Y, 22, H, "3  Stem", "per point: gene embedding (32)\n+ coords (3) → linear → 36-d", TRAIN)
for r, (lab, n, c) in enumerate([("gene", 8, TRAIN), ("x y z", 3, DATA)]):
    for k in range(n): ax.add_patch(Rectangle((61 + k * 1.3, Y + 17 - r * 4.5), 1.1, 2.4, fc=c, alpha=0.35 + 0.08 * (k % 4), ec="white"))
    ax.text(60.6, Y + 18.2 - r * 4.5, lab, fontsize=7.6, color=MUTED, ha="right", va="center")
arrow(71.6, Y + 15.5, 73.6, Y + 15.5, color=TRAIN)
for k in range(9): ax.add_patch(Rectangle((74 + (k % 3) * 1.3, Y + 11.5 + (k // 3) * 2.6), 1.1, 2.3, fc=TRAIN, alpha=0.3 + 0.07 * k, ec="white"))
ax.text(76, Y + 9.6, "36-d", fontsize=7.6, color=TRAIN, ha="center")
ax.text(60.5, Y + 6.5, "query points use the\npretrained mask token", fontsize=7.6, color=MUTED, va="center")
tag(75.5, Y + 3, "trained · lr 1e-3", TRAIN)
arrow(53.6, Y + 14, 57.8, Y + 14)

# 4 encoder: 5 levels, frozen weights + LoRA qkv
panel(82, Y, 31, H, "4  PTv3 encoder (Utonia-small)", "5 levels, points pooled ×2 per level;\nattention weights frozen + LoRA on qkv", FROZEN)
ch = [36, 72, 144, 288, 432]; npts = [20, 13, 8, 5, 3]
for i, (c, n) in enumerate(zip(ch, npts)):
    x = 86 + i * 5.3; hgt = 3 + c / 432 * 10
    ax.add_patch(Rectangle((x, Y + 5), 3.6, hgt, fc="#eef1f4", ec=FROZEN, lw=1, hatch="////"))
    ax.add_patch(Rectangle((x + 3.6, Y + 5 + hgt * 0.55), 1.0, hgt * 0.4, fc=TRAIN, ec="none"))
    ax.text(x + 1.8, Y + 3.2, f"L{i}\n{c}", fontsize=7, color=MUTED, ha="center", va="top")
    yy = Y + 5 + hgt + 1.0; ax.scatter(x + 1.8 + np.linspace(-1.4, 1.4, n), np.full(n, yy), s=4, c=FROZEN, lw=0)
ax.text(85, Y + 27.4 - 3.2, "", fontsize=7)
ax.add_patch(Rectangle((85, Y + 24.2), 1.0, 1.6, fc=TRAIN, ec="none")); ax.text(86.5, Y + 25, "LoRA on qkv (r=16), trained · lr 5e-5", fontsize=7.4, color=TRAIN, va="center", weight="bold")
ax.add_patch(Rectangle((85, Y + 21.6), 1.6, 1.6, fc="#eef1f4", ec=FROZEN, hatch="////")); ax.text(87.2, Y + 22.4, "pretrained weights, frozen", fontsize=7.4, color=MUTED, va="center")
arrow(79.2, Y + 14, 81.8, Y + 14)

# 5 neck
panel(115, Y, 22, H, "5  Neck", "LayerNorm per level, concat 972\n→ linear → 64-d embedding", TRAIN)
y0 = Y + 6
for i, c in enumerate(ch):
    hgt = c / 432 * 4.2; ax.add_patch(Rectangle((118, y0), 4.5, hgt, fc=FROZEN, alpha=0.5, ec="white")); y0 += hgt + 0.4
ax.text(120.2, Y + 4.3, "levels", fontsize=7.2, color=MUTED, ha="center")
arrow(123.4, Y + 13, 126.4, Y + 13, color=TRAIN)
for k in range(8): ax.add_patch(Rectangle((127.5 + (k % 2) * 1.5, Y + 8 + (k // 2) * 2.4), 1.3, 2.2, fc=TRAIN, alpha=0.35 + 0.07 * k, ec="white"))
ax.text(129.2, Y + 6.2, "64-d", fontsize=7.6, color=TRAIN, ha="center")
tag(126, Y + 3 - 0.8, "trained", TRAIN)
arrow(112.6, Y + 14, 114.8, Y + 14)

# 6 recon head + objective
panel(139, Y, 29, H, "6  Recon head → loss", "predict what each removed square held\n(log1p counts); head dropped after", OBJ)
# count
ax.text(143, Y + 20.5, "count", fontsize=8, color=INK, ha="center")
ax.add_patch(Rectangle((141.8, Y + 7), 2.4, 11, fc=OBJ, alpha=0.25, ec=OBJ)); ax.add_patch(Rectangle((141.8, Y + 7), 2.4, 8.5, fc=OBJ, alpha=0.8, ec="none"))
# layout: 4 radial x 2 z
ax.text(151, Y + 20.5, "layout", fontsize=8, color=INK, ha="center")
lay = rng.uniform(0.15, 1, (2, 4))
for zi in range(2):
    for ri in range(4): ax.add_patch(Rectangle((147 + ri * 2.1, Y + 8 + zi * 4.6), 2.0, 4.4, fc=OBJ, alpha=float(lay[zi, ri]), ec="white"))
ax.text(151.2, Y + 6.6, "radial × z", fontsize=7, color=MUTED, ha="center")
# genes
ax.text(162, Y + 20.5, "genes (280)", fontsize=8, color=INK, ha="center")
g = np.sort(rng.gamma(0.7, 1, 22))[::-1]
for k, v in enumerate(g): ax.add_patch(Rectangle((156.5 + k * 0.5, Y + 7), 0.42, 0.9 + v * 3.2, fc=GENES[k % 6], ec="none"))
ax.text(150, Y + 3.4, "MSE vs the removed molecules", fontsize=8, color=OBJ, ha="center", weight="bold")
arrow(134.6, Y + 14, 138.8, Y + 14)

# update loop
ax.annotate("", xy=(68, Y - 0.6), xytext=(150, Y - 0.6), arrowprops=dict(arrowstyle="-|>", color=OBJ, lw=1.4, connectionstyle="arc3,rad=0.0"))
ax.text(110, Y - 2.6, "gradients → AdamW updates stem, LoRA, neck, head   ·   an EMA copy of the model becomes the teacher",
        fontsize=8.8, color=OBJ, ha="center", va="center")

# =============================== EVAL PATH (bottom row) ===============================
ax.text(2, 36.2, "EVAL  ·  every 1,000 steps  ·  held-out blocks  ·  10x labels used only here", fontsize=10, weight="bold", color=INK)
Y2, H2 = 2, 31

panel(2, Y2, 40, H2, "Held-out 1 mm blocks", "random blocks never trained on;\nalternate probe-fit / probe-test", EVAL)
B = rng.random((6, 11)); role = np.zeros_like(B, int); idx = rng.choice(66, 10, replace=False); role.flat[idx[::2]] = 1; role.flat[idx[1::2]] = 2
for r in range(6):
    for c in range(11):
        col = {0: "#eef1f4", 1: "#a8ddd3", 2: EVAL}[role[r, c]]
        ax.add_patch(Rectangle((5 + c * 2.9, Y2 + 3.5 + r * 2.9), 2.7, 2.7, fc=col, ec="white"))
for k, (t, c) in enumerate([("train", "#d0d7de"), ("probe-fit", "#7fcabb"), ("probe-test", EVAL)]):
    ax.text(5 + k * 10, Y2 + 22.2, f"■ {t}", fontsize=7.8, color=c, weight="bold")

panel(44, Y2, 36, H2, "EMA teacher, unmasked", "same model, frozen copy;\n972-d features per molecule, no gradients", FROZEN)
for i, c in enumerate(ch):
    hgt = 3 + c / 432 * 10; ax.add_patch(Rectangle((48 + i * 4.2, Y2 + 5), 3.0, hgt, fc="#eef1f4", ec=FROZEN, lw=1))
ax.text(58.5, Y2 + 3.2, "every molecule visible", fontsize=7.6, color=MUTED, ha="center")
for k in range(10): ax.add_patch(Rectangle((71 + (k % 2) * 1.6, Y2 + 5 + (k // 2) * 2.4), 1.4, 2.2, fc=FROZEN, alpha=0.3 + 0.06 * k, ec="white"))
arrow(39.6, Y2 + 14, 46.4, Y2 + 14, color=EVAL)

panel(82, Y2, 86, H2, "Linear probes (fit on probe-fit, score on probe-test)", "labels from 10x segmentation; the bar is comp20 = gene counts of the 20 nearest molecules", EVAL)
# cell with nucleus, molecules coloured by signed distance
cx, cy, R = 94, Y2 + 12.5, 8
th = np.linspace(0, 2 * np.pi, 80); ax.plot(cx + R * np.cos(th) * 1.1, cy + R * np.sin(th) * 0.9, color=MUTED, lw=1)
ax.plot(cx + 3.4 * np.cos(th), cy + 3.0 * np.sin(th), color=INK, lw=1.2)
mp = rng.normal(size=(260, 2)) * [3.6, 3.0]; mp = mp[(mp[:, 0] / 1.1) ** 2 / R ** 2 + (mp[:, 1] / 0.9) ** 2 / R ** 2 < 1]
dn = np.hypot(mp[:, 0] / 3.4, mp[:, 1] / 3.0) - 1
ax.scatter(cx + mp[:, 0], cy + mp[:, 1], s=5, c=dn, cmap="coolwarm", vmin=-1, vmax=1.4, lw=0)
ax.text(cx, Y2 + 2.6, "distance to nucleus /\ncell boundary (µm)", fontsize=7.6, color=MUTED, ha="center")
rows = [("nucleus vs cytoplasm", "AUROC"), ("signed distance to nucleus boundary", "R², Spearman ρ"),
        ("distance to cell boundary", "R², Spearman ρ"), ("10x cell cluster (cell level, secondary)", "balanced accuracy")]
for i, (a, b) in enumerate(rows):
    yy = Y2 + 20 - i * 4.6; ax.add_patch(Circle((108, yy), 0.8, fc=EVAL, ec="none"))
    ax.text(110, yy + 0.7, a, fontsize=9, color=INK, weight="bold", va="center"); ax.text(110, yy - 1.3, b, fontsize=8, color=MUTED, va="center")
# predicted vs true mini scatter
sx, sy = 146, Y2 + 5; t = rng.uniform(-3, 6, 120); p = 0.6 * t + rng.normal(0, 1.4, 120)
ax.add_patch(Rectangle((sx, sy), 18, 16, fc="white", ec=LINE))
ax.scatter(sx + 1 + (t + 3) / 9 * 16, sy + 1 + (p + 3.5) / 11 * 14, s=4, c=EVAL, lw=0)
ax.plot([sx + 1, sx + 17], [sy + 1 + 0.5 / 11 * 14, sy + 1 + 9.5 / 11 * 14], color=MUTED, lw=0.8, ls="--")
ax.text(sx + 9, sy - 1.6, "predicted vs true distance", fontsize=7.6, color=MUTED, ha="center")
arrow(77.6, Y2 + 14, 85.4, Y2 + 14, color=EVAL)

out = Path(__file__).parent
for ext in ("png", "svg"): fig.savefig(out / f"model_architecture.{ext}", facecolor="white")
print("ok")
