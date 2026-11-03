"""bento 3.0 overview figure: segmentation-free, panel-agnostic multiscale JEPA over molecules.
Real data (10x Xenium FFPE human breast IDC, held-out tile) for images; schematics for the model; measured numbers only.
Inputs: figdata.npz, figdata2.npz (made by extract_local.py / extract2.py from the bento3_ptv3 data arrays).
Writes overview.png (300 dpi) and overview.pdf next to this file."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch, Circle
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree
from pathlib import Path

HERE = Path(__file__).parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
d, d2 = np.load(DATA / "figdata.npz"), np.load(DATA / "figdata2.npz")

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                     "axes.labelsize": 6, "pdf.fonttype": 42, "svg.fonttype": "none"})
INK, MUTED, LINE, FAINT = "#222222", "#6b6b6b", "#b9b9b9", "#eeeeee"
TRAIN, FROZEN, OBJ, EVAL, PANEL = "#2f6fdf", "#8a8a8a", "#d9822b", "#1a8a7a", "#8250df"
W, H = 7.2, 7.75
fig = plt.figure(figsize=(W, H), dpi=300)


def axin(x, y, w, h, **kw):   # inches from the bottom-left corner
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def letter(x, y, s):
    fig.text(x / W, y / H, s, fontsize=9, weight="bold", va="top", ha="left")


def title(x, y, s):
    fig.text(x / W, y / H, s, fontsize=7, weight="bold", va="top", ha="left", color=INK)


def box(ax, x, y, w, h, text, ec, fc="white", fs=6, bold=False, lw=0.8, ls="-", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", ec=ec, fc=fc, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", color=color, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, c=MUTED, lw=0.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, color=c, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def vec(ax, x, y, n, c, w=1.6, h=1.6, alpha0=0.35):   # a small embedding vector glyph
    for i in range(n): ax.add_patch(Rectangle((x + i * w, y), w * 0.86, h, fc=c, ec="none", alpha=alpha0 + (1 - alpha0) * ((i * 37) % 7) / 6))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")


# ---- shared data -------------------------------------------------------------------------------------------------------
xyz, gene, genes = d["win_xyz"], d["win_gene"].astype(int), d["genes"]
top = np.argsort(-np.bincount(gene, minlength=len(genes)))[:8]
PAL = ["#2f6fdf", "#e0702a", "#1a8a7a", "#8250df", "#c23b6e", "#c9a000", "#4aa3d8", "#7a5230"]
gcol = np.full((len(gene), 4), (0.72, 0.72, 0.72, 0.55))
for i, t in enumerate(top): gcol[gene == t] = matplotlib.colors.to_rgba(PAL[i])
polys = lambda off, pts: [pts[off[k]:off[k + 1]] for k in range(len(off) - 1)]
cells, nucs = polys(d["cell_off"], d["cell_poly"]), polys(d["nuc_off"], d["nuc_poly"])


def outlines(ax, lw=0.35, nuc=True):
    ax.add_collection(PolyCollection(cells, facecolors="none", edgecolors="#bdbdbd", linewidths=lw))
    if nuc: ax.add_collection(PolyCollection(nucs, facecolors="none", edgecolors="#555555", linewidths=lw * 0.9, linestyles=(0, (2, 1))))


# =========================================== a  data ===========================================
letter(0.02, 7.72, "a"); title(0.22, 7.70, "Molecules from one Xenium section, no segmentation")
ax = axin(0.08, 5.35, 2.05, 2.1)
dens = d["dens"].T.astype(float); ax.imshow(np.log1p(dens), origin="lower", cmap="Greys", vmax=np.percentile(np.log1p(dens), 99.5) * 1.35, extent=[0, dens.shape[1] * 20, 0, dens.shape[0] * 20])
o = d["win_origin"]; ax.add_patch(Rectangle(o, 100, 100, fc="none", ec="#d62728", lw=0.9)); ax_sec, win_o = ax, o
bu = float(d["block_um"]); role = d["role"]; bnx = int(d["bnx"])
# planned split (Training plan tab): tissue blocks 80% train, 5% validation, 7.5% probe-fit, 7.5% probe-test; one seeded draw
bny = int(d["bny"]); px = int(bu // 20)
occ = np.array([dens[(b // bnx) * px:(b // bnx + 1) * px, (b % bnx) * px:(b % bnx + 1) * px].sum() for b in range(bnx * bny)])
tissue = np.flatnonzero(occ > 0.7 * np.median(occ[occ > 0])); nt = len(tissue)
perm = np.random.default_rng(0).permutation(tissue); nv, npf = round(0.05 * nt), round(0.075 * nt)
for b in perm[:nv]: ax.add_patch(Rectangle(((b % bnx) * bu, (b // bnx) * bu), bu, bu, fc="none", ec=OBJ, lw=0.6, ls=(0, (2, 1.5))))
for b in perm[nv:nv + 2 * npf]: ax.add_patch(Rectangle(((b % bnx) * bu, (b // bnx) * bu), bu, bu, fc="none", ec=EVAL, lw=0.6, ls=(0, (2, 1.5))))
ax.plot([400, 2400], [300, 300], color=INK, lw=1.2); ax.text(1400, 450, "2 mm", ha="center", fontsize=5.5)
ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
ax.text(0.0, -0.05, "68.1 M molecules · 280 genes · 1 mm blocks:\n80% train, 5% validation (orange), 15% probe (teal)", transform=ax.transAxes, fontsize=5.5, color=MUTED, va="top")
ax = axin(2.3, 5.35, 2.05, 2.1)
ax.scatter(xyz[:, 0], xyz[:, 1], s=0.9, c=gcol, lw=0, rasterized=True); outlines(ax, lw=0.25)
for yy in (0, 1): fig.add_artist(matplotlib.patches.ConnectionPatch(xyA=(win_o[0] + 100, win_o[1] + 100 * yy), coordsA=ax_sec.transData, xyB=(0, 100 * yy), coordsB=ax.transData, color="#d62728", lw=0.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color("#d62728"); s.set_linewidth(0.9)
ax.plot([6, 26], [5, 5], color=INK, lw=1.2); ax.text(16, 7.5, "20 µm", ha="center", fontsize=5.5)
ax.text(0.0, -0.05, "100 µm window; colour = gene (top 8)\noutlines: 10x cells / nuclei, eval only", transform=ax.transAxes, fontsize=5.5, color=MUTED, va="top")
ax.add_patch(Rectangle((52, 52), 25, 25, fc="none", ec=INK, lw=0.6, ls=(0, (2, 1))))

# =========================================== b  tokens ===========================================
letter(4.5, 7.72, "b"); title(4.7, 7.70, "5 µm patches become tokens on any panel")
zx, zy = 52, 52; zoom = (xyz[:, 0] >= zx) & (xyz[:, 0] < zx + 25) & (xyz[:, 1] >= zy) & (xyz[:, 1] < zy + 25)
ax = axin(4.55, 5.75, 1.3, 1.3)
ax.scatter(xyz[zoom, 0], xyz[zoom, 1], s=2.2, c=gcol[zoom], lw=0, rasterized=True); outlines(ax, lw=0.5)
for k in range(6): ax.axvline(zx + 5 * k, color=LINE, lw=0.4); ax.axhline(zy + 5 * k, color=LINE, lw=0.4)
px, py = zx + 10, zy + 10; ax.add_patch(Rectangle((px, py), 5, 5, fc="none", ec=TRAIN, lw=1.1))
ax.set_xlim(zx, zx + 25); ax.set_ylim(zy, zy + 25); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_linewidth(0.6); s.set_linestyle((0, (2, 1)))
ax.text(0.0, -0.05, "25 µm, 5 µm grid", transform=ax.transAxes, fontsize=5.5, color=MUTED, va="top")
ax = axin(5.9, 5.3, 1.28, 2.05); blank(ax, (0, 40), (0, 64))
inp = zoom & (xyz[:, 0] >= px) & (xyz[:, 0] < px + 5) & (xyz[:, 1] >= py) & (xyz[:, 1] < py + 5)
pg = gene[inp][:9]
for i, gg in enumerate(pg):
    ax.add_patch(Circle((3, 60 - i * 3.0), 0.9, fc=gcol[inp][i], ec="none"))
    vec(ax, 6.5, 59.2 - i * 3.0, 5, TRAIN, w=1.3, h=1.6, alpha0=0.25)
ax.text(3, 64.5, "molecule", fontsize=5, color=MUTED, ha="center"); ax.text(16, 64.5, "gene emb.", fontsize=5, color=TRAIN, ha="center")
ax.text(20.5, 58, "shared HGNC\nvocabulary;\nco-expression\nprior", fontsize=5, color=TRAIN, va="top", linespacing=1.1)
box(ax, 0.5, 21, 26, 7.5, "attention pool\n+ position in patch", TRAIN, fs=5.5)
arrow(ax, 13, 32.5, 13, 28.8, TRAIN)
vec(ax, 7, 13.5, 8, TRAIN, w=1.6, h=2.4); ax.text(13.5, 11.8, "patch token", fontsize=5.5, color=TRAIN, ha="center", va="top")
arrow(ax, 13, 20.6, 13, 16.3, TRAIN)
vec(ax, 29, 23, 3, PANEL, w=1.6, h=2.4); ax.text(33.5, 21.5, "panel", fontsize=5, color=PANEL, ha="center", va="top")
vec(ax, 29, 11, 3, FROZEN, w=1.6, h=2.4); ax.text(33.5, 9.5, "platform", fontsize=5, color=FROZEN, ha="center", va="top")
arrow(ax, 28.5, 22.5, 21.3, 15.6, PANEL, lw=0.6); arrow(ax, 28.6, 12.2, 21.3, 14.4, FROZEN, lw=0.6)
ax.text(0.5, 4.5, "unmeasured genes never encoded;\npanel token = mean of its gene emb.", fontsize=5, color=MUTED, va="top", linespacing=1.15)

# =========================================== c  JEPA ===========================================
letter(0.02, 5.0, "c"); title(0.22, 4.98, "Multiscale JEPA: predict hidden regions' embeddings from context")
ax = axin(0.1, 2.85, 1.85, 1.85)
blocks = [(10, 78, 5), (30, 88, 5), (8, 30, 5), (14, 52, 10), (28, 8, 10), (46, 44, 50)]   # (x, y, side) in um
hidden = np.zeros(len(gene), bool)
for (bx, by, s) in blocks: hidden |= (xyz[:, 0] >= bx) & (xyz[:, 0] < bx + s) & (xyz[:, 1] >= by) & (xyz[:, 1] < by + s)
ax.scatter(xyz[~hidden, 0], xyz[~hidden, 1], s=0.5, c=gcol[~hidden], lw=0, rasterized=True)
for (bx, by, s) in blocks: ax.add_patch(Rectangle((bx, by), s, s, fc="#f3f3f3", ec=OBJ, lw=1.0, zorder=3))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.text(0.0, -0.04, "hidden blocks of 5, 10 and 50 µm (training\nhides 60–75% of the window; a few shown)", transform=ax.transAxes, fontsize=5.5, color=MUTED, va="top")
ax = axin(2.0, 2.65, 3.05, 2.15); blank(ax, (0, 100), (0, 70))
box(ax, 2, 46, 26, 13, "context encoder\nvisible patches", TRAIN, fc="#eaf1fd", fs=6)
box(ax, 37, 46, 26, 13, "predictor\n+ block position, size", TRAIN, fc="#eaf1fd", fs=6)
vec(ax, 71, 51, 9, TRAIN, w=2.4, h=3.6); ax.text(82, 57.5, "predicted", fontsize=5.5, color=TRAIN, ha="center")
box(ax, 2, 8, 26, 13, "target encoder\nEMA copy, all patches", FROZEN, fs=6, ls=(0, (3, 1.5)))
box(ax, 37, 8, 26, 13, "pool over each\nhidden block", FROZEN, fs=6)
vec(ax, 71, 12.5, 9, FROZEN, w=2.4, h=3.6); ax.text(82, 6.3, "target", fontsize=5.5, color=FROZEN, ha="center")
arrow(ax, 28.2, 52.5, 36.8, 52.5, TRAIN); arrow(ax, 63.2, 52.5, 70.6, 52.5, TRAIN)
arrow(ax, 28.2, 14.5, 36.8, 14.5, FROZEN); arrow(ax, 63.2, 14.5, 70.6, 14.5, FROZEN)
ax.plot([82, 82], [20, 46], color=OBJ, lw=1.0, ls=(0, (2, 1.5))); ax.text(83.5, 33, "smooth-L1\nloss", fontsize=6, color=OBJ, va="center", weight="bold")
ax.annotate("", xy=(15, 21.3), xytext=(15, 45.7), arrowprops=dict(arrowstyle="-|>", color=FROZEN, lw=0.7, ls=(0, (2, 1.5)), mutation_scale=6))
ax.text(16.5, 33, "EMA", fontsize=5.5, color=FROZEN, va="center")
ax.annotate("", xy=(50, 45.7), xytext=(50, 33), arrowprops=dict(arrowstyle="-|>", color=OBJ, lw=0.6, mutation_scale=5))
ax.text(50, 31.5, "hint: 10% of the hidden\nblocks' molecules", fontsize=5, color=OBJ, va="top", ha="center", linespacing=1.1)
box(ax, 37, 62, 61, 7.5, "composition head (w 0.1): pooled emb. · gene emb. → log1p counts", OBJ, fs=5.5)
arrow(ax, 82, 59.6, 82, 61.8, OBJ)

# =========================================== d  panels + density ===========================================
letter(5.2, 5.0, "d"); title(5.4, 4.98, "Panel dropout and token size")
ax = axin(5.3, 3.95, 1.85, 0.75); blank(ax, (0, 100), (0, 30))
rng = np.random.default_rng(4); keep = rng.random(56) < 0.55
for i in range(56):
    ax.add_patch(Rectangle((1 + i * 1.75, 19), 1.5, 7, fc=PAL[i % 8], ec="none"))
    ax.add_patch(Rectangle((1 + i * 1.75, 6), 1.5, 7, fc=PAL[i % 8] if keep[i] else FAINT, ec="none"))
ax.text(0, 29.5, "target: full panel", fontsize=5.5, color=FROZEN); ax.text(0, 16.5, "context: random 30–100% of genes", fontsize=5.5, color=TRAIN)
ax.text(0, 2.5, "→ same embedding from a partial panel;\n   molecules also thinned (keep 50–100%)", fontsize=5.5, color=INK, va="top")
ax = axin(5.62, 2.85, 1.45, 0.82)
side = np.array([1, 2, 3, 4, 5, 8, 10]); med = np.array([2, 5, 10, 18, 27, 70, 109]); p10 = np.array([1, 1, 2, 4, 6, 22, 38]); p90 = np.array([3, 9, 18, 30, 46, 112, 170])
ax.fill_between(side, p10, p90, color=LINE, alpha=0.5, lw=0); ax.plot(side, med, color=INK, lw=0.9, marker="o", ms=1.8)
ax.plot([5], [27], marker="o", ms=4, color=TRAIN); ax.annotate("5 µm: 27", (5, 27), (6.2, 8), fontsize=5.5, color=TRAIN)
ax.set_yscale("log"); ax.set_xlabel("patch side (µm)", labelpad=1); ax.set_ylabel("molecules / patch", labelpad=1)
ax.set_xticks([1, 2, 5, 10]); ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1", "10", "100"])
ax.spines[["top", "right"]].set_visible(False)
ax.text(1.0, 0.04, "median, p10–p90\n~1 molecule / µm²", transform=ax.transAxes, fontsize=5, color=MUTED, va="bottom", ha="right")

# =========================================== e  field ===========================================
letter(0.02, 2.48, "e"); title(0.22, 2.46, "One field, averaged at any scale")
tree = cKDTree(xyz[:, :2]); nb = tree.query_ball_point(xyz[:, :2], r=4.0)
G = len(genes); comp = np.zeros((len(gene), G), np.float32)
for i, n in enumerate(nb): np.add.at(comp[i], gene[n], 1)
Xc = np.log1p(comp / comp.sum(1, keepdims=True) * 100); Xc -= Xc.mean(0)
_, _, Vt = np.linalg.svd(Xc[::4], full_matrices=False); pc = Xc @ Vt[:3].T
lo_, hi_ = np.percentile(pc, [2, 98], 0); rgb = np.clip((pc - lo_) / (hi_ - lo_), 0, 1)
ax = axin(0.1, 0.35, 1.95, 1.95)
ax.scatter(xyz[:, 0], xyz[:, 1], s=0.35, c=rgb, lw=0, rasterized=True); outlines(ax, lw=0.3, nuc=False)
ci = int(np.argmax([np.ptp(c[:, 0]) * (np.ptp(c[:, 0]) < 25) for c in cells]))
ax.add_patch(Polygon(cells[ci], closed=True, fc="none", ec=INK, lw=1.0))
cc = cells[ci].mean(0); ax.add_patch(Rectangle(cc - 2.5 + [-6, 0], 5, 5, fc="none", ec=INK, lw=1.0))
ax.add_patch(Rectangle((40, 40), 50, 50, fc="none", ec=INK, lw=1.0, ls=(0, (3, 1.5))))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.text(0.0, -0.03, "schematic of E(x): colours from local composition\n(stand-in; the trained field is not shown)", transform=ax.transAxes, fontsize=5, color=MUTED, va="top")
ax = axin(2.12, 0.35, 1.2, 1.95); blank(ax, (0, 40), (0, 70))
for k, (lab, sub) in enumerate([("5 µm spot", "subcellular"), ("cell (any outline)", "cell"), ("50 µm window", "niche")]):
    y = 56 - k * 22
    ax.text(0, y + 7, lab, fontsize=5.8, color=INK, weight="bold"); ax.text(0, y + 2.5, "mean of E(x) →", fontsize=5.5, color=MUTED)
    vec(ax, 0, y - 6.5, 9, TRAIN, w=2.6, h=3.6); ax.text(25, y - 4.6, sub, fontsize=5.5, color=TRAIN, va="center")

# =========================================== f  evaluation ===========================================
letter(3.42, 2.48, "f"); title(3.62, 2.46, "Linear probes per scale; the bar is composition at that scale")
ax = axin(3.5, 0.62, 1.15, 1.15)
dn = d2["win_dnuc"]; ok = np.isfinite(dn)
ax.scatter(xyz[ok, 0], xyz[ok, 1], s=0.3, c=dn[ok], cmap="coolwarm", norm=matplotlib.colors.TwoSlopeNorm(0, -3, 3), lw=0, rasterized=True); outlines(ax, lw=0.3)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("subcellular", fontsize=6, pad=2, weight="bold")
ax.text(0.5, -0.05, "distance to nucleus\nboundary (blue = inside)", transform=ax.transAxes, fontsize=5, color=MUTED, ha="center", va="top")
ax = axin(4.72, 0.62, 1.15, 1.15)
cl = d["win_clust"].astype(int); okc = cl >= 0; cpal = plt.get_cmap("tab20")(np.arange(20))
ax.scatter(xyz[okc, 0], xyz[okc, 1], s=0.3, c=cpal[cl[okc] % 20], lw=0, rasterized=True); outlines(ax, lw=0.3, nuc=False)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("cell", fontsize=6, pad=2, weight="bold")
ax.text(0.5, -0.05, "cell type; field averaged\ninside each 10x cell", transform=ax.transAxes, fontsize=5, color=MUTED, ha="center", va="top")
ax = axin(5.94, 0.62, 1.15, 1.15)
ni = d2["niche"].T.astype(float); ni[ni < 0] = np.nan
cnt = np.bincount(d2["niche"][d2["niche"] >= 0].ravel(), minlength=8); rank = np.empty(8, int); rank[np.argsort(-cnt)] = np.arange(8)
npal = np.array([matplotlib.colors.to_rgba(c) for c in ["#d9dde3", "#aac4e0", "#f2c48d", "#4e79a7", "#59a14f", "#e15759", "#b07aa1", "#edc948"]])
img = np.ones(ni.shape + (4,)); okn = np.isfinite(ni); img[okn] = npal[rank[ni[okn].astype(int)]]; img[~okn] = (1, 1, 1, 0)
ax.imshow(img, origin="lower", interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("niche", fontsize=6, pad=2, weight="bold")
ax.text(0.5, -0.05, "niche labels (shown: 50 µm\ncomposition clusters, k = 8)", transform=ax.transAxes, fontsize=5, color=MUTED, ha="center", va="top")
fig.text(3.5 / W, 0.22 / H, "Bars to beat: subcellular, comp20 (~2.6 µm composition), measured: nucleus AUROC 0.82 ·\nnucleus-distance ρ 0.62 · cell-distance ρ 0.33; cell, the cell's own counts; niche, CellCompass",
         fontsize=5.5, color=EVAL, va="center")

for ext in ("png", "pdf"): fig.savefig(HERE / f"overview.{ext}", dpi=300, facecolor="white")
print("ok")
