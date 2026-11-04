"""Phase 0: composition baselines on the frozen split, fit on probe-fit blocks and scored on probe-test blocks.

Subcellular (per molecule): comp20 = gene counts of the 20 nearest molecules (excluding self; about a 2.6 um radius)
  -> nucleus vs cytoplasm (logistic, AUROC); signed distance to its own nucleus boundary and distance to its own cell boundary
     (ridge; R2 and Spearman rho). Molecules without a cell / polygon are skipped.
Cell (per 10x cell, assigned to the block of its centroid): the cell's own counts, log1p of counts per 10k
  -> 10x graph-based cluster (logistic, balanced accuracy).
Niche (per cell): target = composition of 10x clusters among other cells within 50 um of the cell's centroid. Predictors:
  own counts (expression only), gene composition of the 50 um window (matched-scale composition), and squidpy's CellCharter
  niches (one-hot; unsupervised, computed on all cells). Ridge; mean per-cell Pearson r and R2 over the composition vector.

usage: python baselines.py DATA_DIR PHASE0_DIR [--n-mol 200000] [--niche-k 12]
writes PHASE0_DIR/baselines.json
"""
import sys, os, json, time, argparse, numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, r2_score

FIT, TEST = 2, 3


def std_fit(Xf, Xt):
    s = StandardScaler().fit(Xf); s.scale_[s.scale_ < 1e-6] = 1.0; return s.transform(Xf), s.transform(Xt)


def comp(tree, gene, q, k, G, exclude_self=True):
    _, nn = tree.query(q, k=k + exclude_self); nn = nn[:, 1:] if exclude_self else nn
    X = np.zeros((len(q), G), np.float32); np.add.at(X, (np.repeat(np.arange(len(q)), nn.shape[1]), gene[nn].ravel()), 1); return X


def boundary_dist(xy, cid, nucflag, B, signed, clip=15.0, chunk=50_000):
    ids, off, V = B["ids"], B["off"], B["xy"]; j = np.clip(np.searchsorted(ids, cid), 0, len(ids) - 1); ok = (cid >= 0) & (ids[j] == cid)
    out = np.full(len(xy), np.nan, np.float32); w = np.flatnonzero(ok); vmax = int((off[1:] - off[:-1]).max())
    for s in range(0, len(w), chunk):
        r = w[s:s + chunk]; a, L = off[j[r]], off[j[r] + 1] - off[j[r]]; k = np.arange(vmax)[None]
        P0 = V[a[:, None] + np.minimum(k, L[:, None] - 1)]; P1 = V[a[:, None] + (np.minimum(k, L[:, None] - 1) + 1) % L[:, None]]
        x = xy[r, None]; d = P1 - P0; t = np.clip(((x - P0) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-12), 0, 1)
        out[r] = np.sqrt((((P0 + t[..., None] * d) - x) ** 2).sum(-1).min(1))
    if signed: out = np.where(nucflag == 1, -out, out)
    return np.clip(out, -clip, clip)


def reg_scores(yt, pt):
    return dict(r2=float(r2_score(yt, pt)), spearman=float(spearmanr(yt, pt).statistic))


def comp_scores(Yt, Pt):
    Yc, Pc = Yt - Yt.mean(1, keepdims=True), Pt - Pt.mean(1, keepdims=True)
    r = (Yc * Pc).sum(1) / np.maximum(np.sqrt((Yc ** 2).sum(1) * (Pc ** 2).sum(1)), 1e-12)
    return dict(pearson_per_cell=float(np.nanmean(r)), r2=float(r2_score(Yt, Pt)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("p0"); ap.add_argument("--n-mol", type=int, default=200_000)
    ap.add_argument("--niche-k", type=int, default=12); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    t0 = time.time(); L = lambda n: np.load(os.path.join(a.data, n + ".npy"), mmap_mode="r"); rng = np.random.default_rng(a.seed)
    role = np.load(os.path.join(a.p0, "role.npy")); split = json.load(open(os.path.join(a.p0, "split.json")))
    genes = json.load(open(os.path.join(a.data, "genes.json"))); G = len(genes)
    xy = np.asarray(L("xyz")[:, :2]); gene = np.asarray(L("gene")).astype(np.int64); nuc = np.asarray(L("nuc")); cell = np.asarray(L("cell")).astype(np.int64)
    clust = np.asarray(L("clust")); res = dict(split_sha256=split["role_sha256"])

    # ---------------- subcellular: comp20 per molecule ----------------
    held = np.flatnonzero(role >= FIT); tree = cKDTree(xy[held]); gh = gene[held]   # probe blocks only (plus their own molecules)
    pick = {r: rng.choice(np.flatnonzero((role == r) & (cell >= 0)), a.n_mol, replace=False) for r in (FIT, TEST)}
    X = {r: np.log1p(comp(tree, gh, xy[pick[r]], 20, G)) for r in pick}; print(f"comp20 features {time.time()-t0:.0f}s", flush=True)
    B = {k: dict(np.load(os.path.join(a.data, f"{k}_bnd.npz"))) for k in ("nucleus", "cell")}
    sub = {}
    y = {r: (nuc[pick[r]] == 1).astype(int) for r in pick}; Xf, Xt = std_fit(X[FIT], X[TEST])
    lr = LogisticRegression(max_iter=2000).fit(Xf, y[FIT]); sub["nucleus_auroc"] = float(roc_auc_score(y[TEST], lr.decision_function(Xt)))
    for name, kind, signed in [("nucleus_signed_dist", "nucleus", True), ("cell_boundary_dist", "cell", False)]:
        d = {r: boundary_dist(xy[pick[r]], cell[pick[r]], nuc[pick[r]], B[kind], signed) for r in pick}
        okf, okt = np.isfinite(d[FIT]), np.isfinite(d[TEST]); Xf, Xt = std_fit(X[FIT][okf], X[TEST][okt])
        sub[name] = dict(**reg_scores(d[TEST][okt], Ridge(alpha=1.0).fit(Xf, d[FIT][okf]).predict(Xt)), n_test=int(okt.sum()))
    res["subcellular_comp20"] = sub; print("subcellular", json.dumps(sub), f"{time.time()-t0:.0f}s", flush=True)

    # ---------------- cells: own counts, cell type ----------------
    ok = cell >= 0; ids, inv = np.unique(cell[ok], return_inverse=True); nC = len(ids)
    Xc = np.zeros((nC, G), np.float32); np.add.at(Xc, (inv, gene[ok]), 1)
    lab = np.full(nC, -1, np.int64); lab[inv] = clust[ok]
    Bc = B["cell"]; j = np.searchsorted(Bc["ids"], ids); has = (j < len(Bc["ids"])) & (Bc["ids"][np.minimum(j, len(Bc["ids"]) - 1)] == ids)
    cen = np.full((nC, 2), np.nan, np.float32); off = Bc["off"]
    sums = np.add.reduceat(Bc["xy"], off[:-1], axis=0) / np.diff(off)[:, None]; cen[has] = sums[j[has]]
    sumxy = np.zeros((nC, 2)); np.add.at(sumxy, inv, xy[ok]); n_c = np.bincount(inv, minlength=nC); cen[~has] = (sumxy / n_c[:, None])[~has]
    o = np.asarray(split["origin"]); bs = split["block_um"]; bnx = split["bnx"]; brole = np.asarray(split["block_role"])
    bi = (((cen[:, 1] - o[1]) // bs).astype(int).clip(0, split["bny"] - 1) * bnx + ((cen[:, 0] - o[0]) // bs).astype(int).clip(0, bnx - 1))
    crole = brole[bi]; Xn = np.log1p(Xc / np.maximum(Xc.sum(1, keepdims=True), 1) * 1e4)
    f, t = (crole == FIT) & (lab >= 0), (crole == TEST) & (lab >= 0); ncls = int(lab.max()) + 1
    Xf, Xt = std_fit(Xn[f], Xn[t]); lr = LogisticRegression(max_iter=3000).fit(Xf, lab[f])
    res["cell_own_counts"] = dict(balanced_accuracy=float(balanced_accuracy_score(lab[t], lr.predict(Xt))), n_fit=int(f.sum()), n_test=int(t.sum()), n_classes=ncls)
    print("cell", json.dumps(res["cell_own_counts"]), f"{time.time()-t0:.0f}s", flush=True)

    # ---------------- niche: 50 um cell-type composition ----------------
    ct = cKDTree(cen); nb = ct.query_ball_point(cen[f | t], r=50.0); idx = np.flatnonzero(f | t)
    Y = np.zeros((len(idx), ncls), np.float32)
    for i, (c, n) in enumerate(zip(idx, nb)):
        n = [m for m in n if m != c and lab[m] >= 0]
        if n: Y[i] = np.bincount(lab[n], minlength=ncls) / len(n)
    win = np.zeros((len(idx), G), np.float32); mt = cKDTree(xy[held]); W = mt.query_ball_point(cen[idx], r=25.0)
    for i, w in enumerate(W):
        if w: win[i] = np.bincount(gh[w], minlength=G)
    win = np.log1p(win / np.maximum(win.sum(1, keepdims=True), 1) * 1e4)
    import anndata as ad, squidpy as sq, scanpy as sc
    A = ad.AnnData(X=Xn.copy()); A.obsm["spatial"] = cen.astype(np.float64); sc.pp.pca(A, n_comps=30, random_state=a.seed)
    sq.gr.spatial_neighbors(A, coord_type="generic", delaunay=True)
    sq.gr.calculate_niche(A, flavor="cellcharter", distance=3, n_components=a.niche_k, use_rep="X_pca", aggregation="mean", rng=a.seed)
    nk = [c for c in A.obs.columns if "niche" in c.lower()][0]; niche = A.obs[nk].astype("category").cat.codes.to_numpy()
    oh = np.eye(niche.max() + 1, dtype=np.float32)[niche[idx]]
    sel_f, sel_t = f[idx], t[idx]; nres = {}
    for name, F in [("own_counts", Xn[idx]), ("window_composition_50um", win), ("cellcharter_niche_onehot", oh)]:
        Xf, Xt = std_fit(F[sel_f], F[sel_t]); nres[name] = comp_scores(Y[sel_t], Ridge(alpha=1.0).fit(Xf, Y[sel_f]).predict(Xt))
    res["niche_50um_composition"] = dict(**nres, n_fit=int(sel_f.sum()), n_test=int(sel_t.sum()), cellcharter_k=int(niche.max() + 1))
    print("niche", json.dumps(res["niche_50um_composition"]), f"{time.time()-t0:.0f}s", flush=True)
    res["seconds"] = round(time.time() - t0); json.dump(res, open(os.path.join(a.p0, "baselines.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
