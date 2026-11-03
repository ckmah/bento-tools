"""Training plan figure: data split, held-out ladder, phases with gates, and where each recipe piece comes from.
Section image is real data (10x Xenium FFPE human breast IDC); everything else is a schematic of the plan.
Input: figdata.npz (made by ../overview/extract_local.py). Writes training_plan.png (300 dpi) and .pdf next to this file."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch
from pathlib import Path

HERE = Path(__file__).parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
d = np.load(DATA / "figdata.npz")
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 6.5, "pdf.fonttype": 42})
INK, MUTED, LINE, FAINT = "#222222", "#6b6b6b", "#b9b9b9", "#eeeeee"
TRAIN, FROZEN, OBJ, EVAL, PANEL = "#2f6fdf", "#8a8a8a", "#d9822b", "#1a8a7a", "#8250df"
SRC = {"CellWorld": "#2f6fdf", "TERRA": "#c23b6e", "Novae": "#c9a000"}
W, H = 7.2, 5.6
fig = plt.figure(figsize=(W, H), dpi=300)
axin = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
letter = lambda x, y, s: fig.text(x / W, y / H, s, fontsize=9, weight="bold", va="top")
title = lambda x, y, s: fig.text(x / W, y / H, s, fontsize=7, weight="bold", va="top", color=INK)


def box(ax, x, y, w, h, ec, fc="white", lw=0.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", ec=ec, fc=fc, lw=lw, ls=ls))


def chip(ax, x, y, src, fs=4.6):
    t = ax.text(x, y, src, fontsize=fs, color="white", weight="bold", va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.5", fc=SRC[src], ec="none"))
    return t


# ============================== a  split within a section ==============================
letter(0.02, 5.52, "a"); title(0.22, 5.50, "Split within a section: 1 mm blocks, fixed once")
ax = axin(0.1, 3.05, 1.9, 2.15)
dens = d["dens"].T.astype(float)
ax.imshow(np.log1p(dens), origin="lower", cmap="Greys", vmax=np.percentile(np.log1p(dens), 99.5) * 1.35,
          extent=[0, dens.shape[1] * 20, 0, dens.shape[0] * 20], alpha=0.55)
bu, bnx, bny = float(d["block_um"]), int(d["bnx"]), int(d["bny"]); px = int(bu // 20)
occ = np.array([dens[(b // bnx) * px:(b // bnx + 1) * px, (b % bnx) * px:(b % bnx + 1) * px].sum() for b in range(bnx * bny)])
tissue = np.flatnonzero(occ > 0.7 * np.median(occ[occ > 0])); nt = len(tissue)
perm = np.random.default_rng(0).permutation(tissue); nv, npf = round(0.05 * nt), round(0.075 * nt)
role = {b: "train" for b in tissue}
for b in perm[:nv]: role[b] = "val"
for b in perm[nv:nv + npf]: role[b] = "fit"
for b in perm[nv + npf:nv + 2 * npf]: role[b] = "test"
fcs = {"train": "#d3e0f7", "val": OBJ, "fit": EVAL, "test": EVAL}
for b, r in role.items():
    x0, y0 = (b % bnx) * bu, (b // bnx) * bu
    ax.add_patch(Rectangle((x0, y0), bu, bu, fc=fcs[r], ec="white", lw=0.4, alpha=1 if r == "train" else 0.8,
                           hatch="////" if r == "test" else None))
ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
ax.plot([400, 2400], [250, 250], color=INK, lw=1.0); ax.text(1400, 380, "2 mm", ha="center", fontsize=5.5)
zb = perm[nv]; zx, zy = (zb % bnx) * bu, (zb // bnx) * bu
ax.add_patch(Rectangle((zx - 300, zy - 300), bu + 600, bu + 600, fc="none", ec=INK, lw=0.6, ls=(0, (2, 1))))
lg = axin(0.1, 2.62, 1.9, 0.38); lg.axis("off"); lg.set_xlim(0, 100); lg.set_ylim(0, 20)
for i, (lab, fc, h) in enumerate([("train 80%", fcs["train"], None), ("validation 5%", OBJ, None), ("probe-fit 7.5%", EVAL, None), ("probe-test 7.5%", EVAL, "////")]):
    xx, yy = (i % 2) * 50, 12 - (i // 2) * 9
    lg.add_patch(Rectangle((xx, yy), 5, 5.5, fc=fc, ec=LINE if i == 0 else "white", lw=0.4, alpha=1 if i == 0 else 0.75, hatch=h))
    lg.text(xx + 7, yy + 2.7, lab, fontsize=5.5, va="center", color=INK)

# zoom: guard band and windows
ax = axin(2.15, 3.05, 1.6, 2.15); ax.set_xlim(-300, 1300); ax.set_ylim(-300, 1300); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Rectangle((-300, -300), 1600, 1600, fc="#d3e0f7", ec=INK, lw=0.6, ls=(0, (2, 1))))
ax.add_patch(Rectangle((-50, -50), 1100, 1100, fc="white", ec="none"))
ax.add_patch(Rectangle((-50, -50), 1100, 1100, fc="none", ec=FROZEN, lw=0, hatch="xxxx", alpha=0.35))
ax.add_patch(Rectangle((0, 0), 1000, 1000, fc=EVAL, ec="white", lw=0.4, alpha=0.75))
ax.text(500, 560, "probe-fit block\n1 mm", ha="center", va="center", fontsize=5.5, color="white", weight="bold")
ax.text(500, 380, "labels loaded\nonly here", ha="center", va="center", fontsize=5, color="white")
for (wx, wy) in [(-260, 1150), (1150, 700), (300, -260), (-260, 200)]:
    ax.add_patch(Rectangle((wx, wy), 100, 100, fc="none", ec=TRAIN, lw=0.9))
ax.add_patch(Rectangle((1000, 1040), 100, 100, fc="none", ec="#d62728", lw=0.9))
ax.plot([1000, 1100], [1040, 1140], color="#d62728", lw=0.7); ax.plot([1000, 1100], [1140, 1040], color="#d62728", lw=0.7)
ax.text(-300, -380, "hatched: 50 µm guard band, molecules dropped;\n100 µm training windows (blue) may not reach it\n(red: rejected)", fontsize=5, color=MUTED, va="top")
ax.text(-300, 1340, "zoom (dashed box in left map)", fontsize=5, color=MUTED, va="bottom")

# ============================== b  held-out ladder ==============================
letter(3.95, 5.52, "b"); title(4.15, 5.50, "Held-out ladder: every claim names its rung")
ax = axin(4.0, 2.62, 3.15, 2.6); ax.set_xlim(0, 100); ax.set_ylim(0, 80); ax.axis("off")
rungs = [("0", "1 mm blocks", "fit, not generalization", "phases 1–2"),
         ("1", "whole sections, seen donor", "section variation", "phase 3"),
         ("2", "all sections of a donor", "donor variation", "phase 3, 5"),
         ("3", "dataset with a new panel", "panel shift", "phase 4"),
         ("4", "a whole platform", "platform shift", "phase 4")]
for i, (r, held, tests, ph) in enumerate(rungs):
    x, y, w = i * 9, 4 + i * 14.5, 100 - i * 9
    shade = 0.12 + 0.17 * i
    ax.add_patch(Rectangle((x, y), w, 12, fc=(0.10, 0.54, 0.48, shade), ec="none"))
    ax.text(x + 2.5, y + 6, r, fontsize=9, weight="bold", va="center", color=INK if i < 3 else "white")
    ax.text(x + 8, y + 8.2, held, fontsize=5.8, weight="bold", va="center", color=INK if i < 3 else "white")
    ax.text(x + 8, y + 3.6, tests, fontsize=5.3, va="center", color=MUTED if i < 3 else "#f0f0f0")
    ax.text(98.5, y + 6, ph, fontsize=5.3, va="center", ha="right", color=INK if i < 3 else "white")
ax.annotate("", xy=(1, 76), xytext=(1, 3), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6, mutation_scale=6))
ax.text(2.5, 77, "harder generalization", fontsize=5.3, color=INK, va="bottom")

# ============================== c  phases ==============================
letter(0.02, 2.42, "c"); title(0.22, 2.40, "Phases: each ends at a gate; a failed gate stops the next phase")
ax = axin(0.1, 0.05, 7.05, 2.2); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
phases = [
    ("0", "Data and\nvocabulary", "HGNC vocabulary\ngene prior emb.\nsplit file, baselines", [], "every gene → one token;\nsplit frozen;\nbaselines scored"),
    ("1", "Single-panel\npilot", "1 Xenium section\nCellWorld-patch\nfalsification first", ["CellWorld"], "no collapse;\nbeats composition\nat 2 of 3 scales"),
    ("2", "Panel\nrobustness", "panel dropout\nthinning, count noise\npanel/platform tokens", ["Novae"], "≥80% retention on\n35%-overlap split;\n<5% IID cost"),
    ("3", "Multi-\nsection", "10–20 sections\n2–3 platforms\nrandom context size", ["TERRA", "CellWorld"], "beats composition\non rungs 1–2"),
    ("4", "Cross-platform\ntransfer", "zero-shot, then\nplatform token, then\nlast block or LoRA", ["TERRA"], "beats composition\non shared genes,\ncell and niche"),
    ("5", "Scale\nup", "more donors\nmodel size; data\nfractions by section", ["TERRA", "CellWorld"], "final numbers on\nheld-out tissue\n(rungs 2–4)"),
]
pw, gap = 14.2, 2.45
for i, (n, name, work, srcs, gate) in enumerate(phases):
    x = i * (pw + gap)
    box(ax, x, 21, pw, 17.5, TRAIN if i else FROZEN, fc="#eaf1fd" if i else "#f4f4f4")
    ax.text(x + 1.2, 36.6, n, fontsize=10, weight="bold", va="top", color=TRAIN if i else FROZEN)
    ax.text(x + 4.2, 36.6, name, fontsize=5.8, weight="bold", va="top", color=INK, linespacing=1.05)
    ax.text(x + 1.2, 29.6, work, fontsize=5, va="top", color=INK, linespacing=1.15)
    # gate
    gx = x + pw / 2
    ax.add_patch(Polygon([(gx, 19.6), (gx + 1.3, 17.6), (gx, 15.6), (gx - 1.3, 17.6)], closed=True, fc=EVAL, ec="none"))
    ax.plot([gx, gx], [21, 19.6], color=EVAL, lw=0.6)
    ax.text(gx, 14.6, gate, fontsize=4.9, va="top", ha="center", color=EVAL, linespacing=1.15)
    if i < len(phases) - 1:
        ax.add_patch(FancyArrowPatch((x + pw + 0.2, 29.5), (x + pw + gap - 0.2, 29.5), arrowstyle="-|>", mutation_scale=6, color=MUTED, lw=0.7))
ax.text(0, 2.2, "recipe throughout: AdamW, BF16, batch 256, warmup to 3e-3 then cosine, EMA 0.999 → 0.9999, 10% hint, 10k steps in the pilot; "
        "block masking at 5/10/50 µm, random masking as first ablation; adaptation only via platform token, last block or LoRA.",
        fontsize=5, color=MUTED, va="bottom", wrap=True)
for ext in ("png", "pdf"): fig.savefig(HERE / f"training_plan.{ext}", dpi=300, facecolor="white")
print("ok")
