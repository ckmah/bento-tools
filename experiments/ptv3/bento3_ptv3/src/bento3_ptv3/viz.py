"""Interpretation panels on one fixed held-out (probe-test) tile, rendered from the EMA teacher (what the probes evaluate).
run_dir/viz/step_XXXXXXX.png + one JSON line per render in run_dir/viz/viz.jsonl. Best effort: never stops training."""
import os, json, numpy as np, torch
from .evaluate import collate
from .data import make_view

LEVELS = (36, 72, 144, 288, 432)   # channels per encoder level in the concatenated 972-d feature (finest first)
_C = {}


def _tile(M):
    if "t" not in _C:
        ts = M.eval_tiles(2, 24); _C["t"] = max(ts, key=lambda t: len(t[0]))
    return _C["t"]


@torch.no_grad()
def render(net, M, cfg, dev, amp, run_dir, step):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sel, core, x0, y0 = _tile(M)
        v = make_view(M, sel, None, cfg, augment=False, mask=False); b = collate([v], dev)
        net.eval()
        with torch.autocast(**amp):
            up, e = net.features(b); lg = net.head(e)
        up, e, lg = up.float(), e.float(), lg.float()
        xy = M.xyz[sel, :2] - np.array([x0, y0]); n = len(sel)
        edges = np.cumsum((0,) + LEVELS); lvl = [float(up[:, edges[i]:edges[i + 1]].abs().max()) for i in range(len(LEVELS))]
        info = dict(step=int(step), n_mol=int(n), up_absmax=float(up.abs().max()), level_absmax=lvl, emb_std=float(e.std(0).mean()))
        fig, ax = plt.subplots(2, 3, figsize=(15, 10), dpi=80); ax = ax.ravel(); kw = dict(s=1.5, linewidths=0, rasterized=True)
        # 1 gene identity (top 12 genes in the tile)
        g = M.gene[sel]; top = np.argsort(-np.bincount(g, minlength=M.G))[:12]; col = np.full((n, 4), (0.8, 0.8, 0.8, 0.5))
        cm = plt.get_cmap("tab20")
        for i, t in enumerate(top): col[g == t] = cm(i)
        ax[0].scatter(*xy.T, c=col, **kw); ax[0].set_title("gene (top 12 coloured)")
        ax[0].legend(handles=[plt.Line2D([], [], ls="", marker="o", color=cm(i), label=M.genes[t]) for i, t in enumerate(top)], fontsize=6, loc="upper right", ncol=2, markerscale=0.8)
        # 2 10x cluster of the molecule's cell (eval-only label)
        cl = M.clust[sel].astype(int); pal = np.vstack([plt.get_cmap("tab20")(np.arange(20)), plt.get_cmap("tab20b")(np.arange(20))])
        col = np.where((cl >= 0)[:, None], pal[np.clip(cl, 0, None) % 40], (0.85, 0.85, 0.85, 0.4))
        ax[1].scatter(*xy.T, c=col, **kw); ax[1].set_title("10x cell cluster (eval label; grey = unassigned)")
        # 3 nucleus vs cytoplasm (eval label)
        nu = M.nuc[sel]; col = np.where((nu == 1)[:, None], (0.8, 0.1, 0.5, 0.9), np.where((nu == 0)[:, None], (0.1, 0.6, 0.8, 0.7), (0.85, 0.85, 0.85, 0.4)))
        ax[2].scatter(*xy.T, c=col, **kw); ax[2].set_title("nucleus (magenta) / cytoplasm (blue) - eval label")
        # 4 embedding PCA -> RGB
        if torch.isfinite(e).all():
            ec = e - e.mean(0); _, S, V = torch.pca_lowrank(ec, q=3, center=False); pc = (ec @ V[:, :3]).cpu().numpy()
            lo, hi = np.percentile(pc, [2, 98], 0); rgb = np.clip((pc - lo) / (hi - lo + 1e-9), 0, 1)
            evr = (S[:3] ** 2 / ec.pow(2).sum()).cpu().numpy(); info["pca_evr"] = [float(x) for x in evr]
            ax[3].scatter(*xy.T, c=rgb, **kw); ax[3].set_title(f"teacher embedding PCA->RGB (var {evr[0]:.2f}/{evr[1]:.2f}/{evr[2]:.2f})")
        else:
            ax[3].text(0.5, 0.5, "non-finite embedding", ha="center", transform=ax[3].transAxes); ax[3].set_title("teacher embedding PCA")
        # 5 prototype argmax
        a = lg.argmax(1).cpu().numpy(); u, cnt = np.unique(a, return_counts=True); p = cnt / cnt.sum()
        info |= dict(protos_used=int(len(u)), proto_perplexity=float(np.exp(-(p * np.log(p)).sum())), proto_top_frac=float(p.max()))
        perm = np.random.default_rng(0).permutation(cfg.prototypes); col = plt.get_cmap("hsv")(perm[a] / cfg.prototypes)
        ax[4].scatter(*xy.T, c=col, **kw); ax[4].set_title(f"teacher prototype argmax ({len(u)} used, perplexity {info['proto_perplexity']:.0f}, top {p.max():.2f})")
        # 6 feature magnitude per encoder level (the lora_ssl_s0 failure mode)
        ax[5].bar(range(len(LEVELS)), np.maximum(lvl, 1e-3), color="0.4"); ax[5].set_yscale("log"); ax[5].axhline(cfg.max_feat_abs, color="r", ls="--", lw=1)
        ax[5].set_xticks(range(len(LEVELS)), [f"L{i} ({c}ch)" for i, c in enumerate(LEVELS)]); ax[5].set_title(f"max |encoder feature| per level (abort > {cfg.max_feat_abs:g})")
        for k in range(5): ax[k].set_aspect("equal"); ax[k].set_xticks([]); ax[k].set_yticks([])
        fig.suptitle(f"step {step} - held-out tile ({n:,} molecules, 100 um)", fontsize=13); fig.tight_layout()
        d = os.path.join(run_dir, "viz"); os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, f".step_{step:07d}.png"); fig.savefig(tmp); plt.close(fig); os.replace(tmp, os.path.join(d, f"step_{step:07d}.png"))
        with open(os.path.join(d, "viz.jsonl"), "a") as f: f.write(json.dumps(info) + "\n")
    except Exception as ex:
        print(f"viz skipped at step {step} ({type(ex).__name__}: {ex})", flush=True)
