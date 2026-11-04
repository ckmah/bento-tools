"""Figure 1 (representation): nucleus-centred 20 um discs of molecules, each molecule encoded with Fourier position features,
a disc encoder over molecules, a window encoder over discs, and JEPA training with spatial masking, rotations and two views.
Images are real data (10x Xenium FFPE human breast IDC, one 100 um window); the model rows are schematics; numbers are measured.
Inputs: figdata.npz (../overview/extract_local.py) and repr_stats.npz (disc statistics over the whole section).
usage: python representation_fig.py DATA_DIR   -> representation.png / .pdf next to this file"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon
from matplotlib.collections import PolyCollection
from pathlib import Path

HERE = Path(__file__).parent; DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
d, st = np.load(DATA / "figdata.npz"), np.load(DATA / "repr_stats.npz")
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.labelsize": 5.5,
                     "ytick.labelsize": 5.5, "axes.labelsize": 6, "pdf.fonttype": 42})
INK, MUTED, LINE, FAINT = "#222222", "#6b6b6b", "#b9b9b9", "#eeeeee"
TRAIN, FROZEN, OBJ, EVAL, PANEL = "#2f6fdf", "#8a8a8a", "#d9822b", "#1a8a7a", "#8250df"
W, H = 7.2, 8.0
fig = plt.figure(figsize=(W, H), dpi=300)
axin = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
letter = lambda x, y, s: fig.text(x / W, y / H, s, fontsize=9, weight="bold", va="top")
title = lambda x, y, s: fig.text(x / W, y / H, s, fontsize=7, weight="bold", va="top", color=INK)
note = lambda ax, x, y, s, **k: ax.text(x, y, s, fontsize=k.pop("fs", 5.3), color=k.pop("c", MUTED), va=k.pop("va", "top"), **k)


def box(ax, x, y, w, h, text, ec, fc="white", fs=5.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", ec=ec, fc=fc, lw=0.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, c=MUTED, lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def vec(ax, x, y, n, c, w=1.6, h=1.6):
    for i in range(n): ax.add_patch(Rectangle((x + i * w, y), w * 0.86, h, fc=c, ec="none", alpha=0.35 + 0.65 * ((i * 37) % 7) / 6))


# ---- data: one 100 um window ----
xyz, gene, genes = d["win_xyz"], d["win_gene"].astype(int), d["genes"]
polys = lambda off, pts: [pts[off[k]:off[k + 1]] for k in range(len(off) - 1)]
nucs, cells = polys(d["nuc_off"], d["nuc_poly"]), polys(d["cell_off"], d["cell_poly"])
cen = np.array([p.mean(0) for p in nucs])
top = np.argsort(-np.bincount(gene, minlength=len(genes)))[:8]
PAL = ["#2f6fdf", "#e0702a", "#1a8a7a", "#8250df", "#c23b6e", "#c9a000", "#4aa3d8", "#7a5230"]
gcol = np.full((len(gene), 4), (0.72, 0.72, 0.72, 0.55))
for i, t in enumerate(top): gcol[gene == t] = matplotlib.colors.to_rgba(PAL[i])
R = 10.0                                                       # disc radius (20 um diameter)
inner = np.flatnonzero((cen[:, 0] > 25) & (cen[:, 0] < 75) & (cen[:, 1] > 25) & (cen[:, 1] < 75))
hi = inner[np.argmin(np.hypot(cen[inner, 0] - 50, cen[inner, 1] - 50))]   # highlighted nucleus near the centre

# =========================== a  discs on real data ===========================
letter(0.02, 7.95, "a"); title(0.22, 7.93, "One 20 µm disc of molecules per nucleus")
ax = axin(0.1, 5.55, 2.25, 2.25)
ax.scatter(xyz[:, 0], xyz[:, 1], s=0.5, c=gcol, lw=0, rasterized=True)
ax.add_collection(PolyCollection(nucs, facecolors="none", edgecolors="#444444", linewidths=0.35, linestyles=(0, (2, 1))))
for c in cen: ax.add_patch(Circle(c, R, fc="none", ec=TRAIN, lw=0.35, alpha=0.45))
ax.add_patch(Circle(cen[hi], R, fc="none", ec=TRAIN, lw=1.4))
ax.scatter(cen[:, 0], cen[:, 1], s=3, c=INK, lw=0, zorder=5)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.plot([5, 25], [4, 4], color=INK, lw=1.2); ax.text(15, 6.5, "20 µm", ha="center", fontsize=5.5)
note(ax, 0, -0.03, "100 µm of the Xenium section; dots: nucleus centroids (from DAPI\nnuclei); dashed: nuclei; blue: 20 µm discs; colour: gene (top 8)",
     transform=ax.transAxes)

# measured disc statistics (whole section)
ax = axin(2.75, 6.95, 1.3, 0.75)
m = st["mol_per_disc"]; ax.hist(m, bins=np.arange(0, 1000, 25), color=TRAIN, alpha=0.75)
ax.axvline(np.median(m), color=INK, lw=0.6); ax.text(np.median(m) + 15, ax.get_ylim()[1] * 0.85, f"median {np.median(m):.0f}", fontsize=5)
ax.set_xlabel("molecules per disc", labelpad=1); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
ax = axin(2.75, 5.85, 1.3, 0.75)
c = st["cnt"]; b = np.bincount(np.minimum(c, 9), minlength=10) / len(c); ax.bar(np.arange(10), b, color=OBJ, alpha=0.8, width=0.8)
ax.set_xticks([0, 3, 6, 9]); ax.set_xticklabels(["0", "3", "6", "9+"]); ax.set_xlabel("discs containing a molecule", labelpad=1)
ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
ax.text(9.4, b.max() * 0.95, f"in ≥1 disc: {(c > 0).mean():.1%}\nmean {c.mean():.1f} discs", fontsize=5, ha="right", va="top")
fig.text(2.75 / W, 5.42 / H, f"whole section: {len(st['nn']):,} nuclei; nearest-nucleus\nspacing median {np.median(st['nn']):.1f} µm, so discs overlap",
         fontsize=5.3, color=MUTED, va="top")

# =========================== b  molecule token ===========================
letter(4.3, 7.95, "b"); title(4.5, 7.93, "Each molecule: gene + Fourier-encoded position")
ax = axin(4.35, 5.95, 1.55, 1.55)
inn = np.hypot(xyz[:, 0] - cen[hi, 0], xyz[:, 1] - cen[hi, 1]) < R; dxy = xyz[inn, :2] - cen[hi]
ax.scatter(dxy[:, 0], dxy[:, 1], s=2.2, c=gcol[inn], lw=0)
ax.add_patch(Polygon(nucs[hi] - cen[hi], closed=True, fc="none", ec="#444444", lw=0.6, ls=(0, (2, 1))))
ax.add_patch(Circle((0, 0), R, fc="none", ec=TRAIN, lw=1.0)); ax.scatter([0], [0], s=8, c=INK, zorder=5)
k = np.argmin(np.abs(np.hypot(dxy[:, 0], dxy[:, 1]) - 6.0)); px, py = dxy[k]; gk = genes[gene[np.flatnonzero(inn)[k]]]
ax.annotate("", xy=(px, py), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=OBJ, lw=0.9, mutation_scale=6))
ax.scatter([px], [py], s=16, facecolors="none", edgecolors=OBJ, lw=0.9, zorder=6)
ax.text(-R, R + 0.8, f"{gk}: (dx, dy) = ({px:+.1f}, {py:+.1f}) µm", fontsize=5.2, color=OBJ, va="bottom")
ax.set_xlim(-R - 0.5, R + 0.5); ax.set_ylim(-R - 0.5, R + 0.5); ax.set_aspect("equal"); ax.axis("off")
note(ax, 0, -0.01, f"one disc: {inn.sum()} molecules; offsets from the nucleus centroid; dz from local median z", transform=ax.transAxes)
ax = axin(6.05, 6.55, 0.85, 0.95)
lam = np.geomspace(0.5, 40, 10); xs = np.linspace(-R, R, 600)
for j, l in zip([1, 4, 7, 9], ["0.8", "3.4", "15", "40"]):
    yy = np.sin(2 * np.pi * xs / lam[j]); off_ = [3.3, 1.1, -1.1, -3.3][[1, 4, 7, 9].index(j)]
    ax.plot(xs, yy * 0.85 + off_, color=TRAIN, lw=0.6); ax.text(R + 0.6, off_, f"{l} µm", fontsize=4.6, va="center", color=MUTED)
    ax.scatter([px], [np.sin(2 * np.pi * px / lam[j]) * 0.85 + off_], s=6, c=OBJ, zorder=5)
ax.axvline(px, color=OBJ, lw=0.6, ls=(0, (2, 1))); ax.set_xlim(-R, R); ax.set_yticks([])
ax.set_xticks([-10, 0, 10]); ax.set_xlabel("dx (µm)", labelpad=1); ax.spines[["top", "right", "left"]].set_visible(False)
note(ax, 0, 1.08, "sin(2π·dx/λ), 4 of the 10 λ", transform=ax.transAxes, va="bottom")
ax = axin(4.35, 5.3, 2.8, 0.45); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
ax.text(0, 9, "token = gene_emb(g) + W · [sin, cos](2π · (dx, dy, dz) / λ)", fontsize=6, color=INK, va="top")
note(ax, 0, 4.2, "λ = 0.5 … 40 µm, 10 log-spaced (finest ≈ localization precision; coarsest\n> disc diagonal, so no wrap-around). 3 axes × 10 λ × sin, cos = 60 → W → 96.")

# =========================== c  model ===========================
letter(0.02, 4.95, "c"); title(0.22, 4.93, "Disc encoder over molecules, window encoder over discs")
ax = axin(0.1, 3.0, 7.0, 1.8); ax.set_xlim(0, 141); ax.set_ylim(0, 36); ax.axis("off")
box(ax, 0, 20, 22, 12, "molecule tokens\n~340 per disc\n(gene + position)", INK)
box(ax, 28, 20, 26, 12, "disc encoder\n3 layers, width 96;\nattention within the disc,\nbias −slope · distance (µm)", TRAIN, fc="#eaf1fd")
box(ax, 60, 20, 18, 12, "attention pool\n+ log count", TRAIN, fc="#eaf1fd")
vec(ax, 82, 24, 8, TRAIN, w=1.6, h=3.6); ax.text(88.5, 31, "disc token", fontsize=5.5, color=TRAIN, ha="center")
box(ax, 98, 20, 26, 12, "window encoder\nover neighbouring discs;\nbias by centroid distance", TRAIN, fc="#eaf1fd")
box(ax, 128, 20, 12, 12, "pool\nover\nwindow", TRAIN, fc="#eaf1fd")
for x0, x1 in [(22.3, 27.7), (54.3, 59.7), (78.3, 81.7), (95.3, 97.7), (124.3, 127.7)]: arrow(ax, x0, 26, x1, 26, TRAIN)
arrow(ax, 41, 19.6, 41, 13.4, EVAL); box(ax, 30, 4, 22, 9, "per-molecule outputs\n→ subcellular readout", EVAL, fs=5.5)
arrow(ax, 88.5, 23.6, 88.5, 13.4, EVAL); box(ax, 77, 4, 23, 9, "disc token\n→ cell readout", EVAL, fs=5.5)
arrow(ax, 134, 19.6, 134, 13.4, EVAL); box(ax, 121, 4, 19, 9, "window pool\n→ niche readout", EVAL, fs=5.5)
note(ax, 0, 15, "attribution back to molecules: per-molecule\noutputs, pool weights, gradients w.r.t. each\nmolecule's gene and position", fs=5.0)

# =========================== d  training ===========================
letter(0.02, 2.85, "d"); title(0.22, 2.83, "JEPA training: spatial masking, random rotation, two views")
ax = axin(0.1, 0.4, 2.25, 2.25)
mx0, my0, ms = 52, 30, 30                                      # masked square (um), illustration
hid = (xyz[:, 0] >= mx0) & (xyz[:, 0] < mx0 + ms) & (xyz[:, 1] >= my0) & (xyz[:, 1] < my0 + ms)
ax.scatter(xyz[~hid, 0], xyz[~hid, 1], s=0.5, c=gcol[~hid], lw=0, rasterized=True)
ax.add_patch(Rectangle((mx0, my0), ms, ms, fc="#f1f1f1", ec=OBJ, lw=1.0, zorder=3))
tgt = (cen[:, 0] >= mx0) & (cen[:, 0] < mx0 + ms) & (cen[:, 1] >= my0) & (cen[:, 1] < my0 + ms)
cut = ~tgt & (np.maximum(np.abs(cen[:, 0] - np.clip(cen[:, 0], mx0, mx0 + ms)), 0) ** 2 + (cen[:, 1] - np.clip(cen[:, 1], my0, my0 + ms)) ** 2 < R ** 2)
for i in np.flatnonzero(tgt): ax.add_patch(Circle(cen[i], R, fc="none", ec=OBJ, lw=0.8, zorder=4))
for i in np.flatnonzero(cut)[::2]: ax.add_patch(Circle(cen[i], R, fc="none", ec=TRAIN, lw=0.5, ls=(0, (2, 1)), zorder=4))
ax.scatter(cen[tgt, 0], cen[tgt, 1], s=4, c=OBJ, zorder=5)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
note(ax, 0, -0.03, "student view: every molecule in the grey region is removed from\nevery disc (dashed blue: neighbours lose part of their disc);\n"
     "orange: discs centred inside are the targets", transform=ax.transAxes)
ax = axin(2.55, 0.25, 4.6, 2.45); ax.set_xlim(0, 101); ax.set_ylim(0, 56); ax.axis("off")
box(ax, 0, 40, 30, 11, "student: disc + window\nencoders on the masked view", TRAIN, fc="#eaf1fd")
box(ax, 36, 40, 26, 11, "predictor\nqueried at each target\ndisc's centroid", TRAIN, fc="#eaf1fd")
box(ax, 0, 22, 30, 11, "teacher (EMA of student,\n0.996 → 1): full view", FROZEN, ls=(0, (3, 1.5)))
arrow(ax, 30.3, 45.5, 35.7, 45.5, TRAIN); arrow(ax, 15, 39.6, 15, 33.4, FROZEN, ls=(0, (2, 1.5)))
ax.text(16, 36.5, "EMA", fontsize=5, color=FROZEN)
box(ax, 68, 40, 32, 11, "1 · latent prediction: smooth-L1 between\npredicted and teacher target-disc tokens", OBJ, fs=5.3)
arrow(ax, 62.3, 45.5, 67.7, 45.5, OBJ); arrow(ax, 30.3, 32, 67.7, 41.5, FROZEN, ls=(0, (2, 1.5))); ax.text(48, 38.6, "targets", fontsize=5, color=FROZEN, rotation=13)
box(ax, 36, 22, 64, 11, "2 · masked-gene anchor: hide 15–30% of molecules' genes (positions kept) and\npredict each from its per-molecule output; head = gene embedding table (any panel)", OBJ, fs=5.3)
box(ax, 36, 4, 64, 13, "3 · two views: split each disc's molecules into random halves; pull the two window\nembeddings together, with SIGReg against collapse (LeJEPA)\n"
    "+ every window gets a random rotation and flip (tissue has no 'up')", OBJ, fs=5.3)
note(ax, 0, 17, "sizes to set in the pilot:\nwindow ~100 µm (~80 discs);\nmasked regions 20–50 µm;\nmodel ~1M parameters", fs=5.2)

for ext in ("png", "pdf"): fig.savefig(HERE / f"representation.{ext}", dpi=300, facecolor="white")
print("ok")
