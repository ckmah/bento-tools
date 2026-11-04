"""Shared evaluation for phase 0 baselines and phase 1 models: the same probe molecules, cells, targets and probes everywhere.

Roles come from the frozen split (split.py): probes fit on probe-fit blocks (2) and score on probe-test blocks (3).
- Subcellular, per molecule: nucleus vs cytoplasm (logistic, AUROC); signed distance to its own nucleus boundary and distance to
  its own cell boundary (ridge, R2 and Spearman). 200k cell-assigned molecules per role, seed 0.
- Cell, per 10x cell (block of its polygon centroid): 10x graph-based cluster (logistic, balanced accuracy).
- Niche, per cell: composition of the 26 clusters among the other cells within 50 um of its centroid (ridge, mean per-cell
  Pearson and R2)."""
import os, json, numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, r2_score

FIT, TEST = 2, 3


def load(data, p0):
    """Arrays from bento3_ptv3.data.build_from_10x plus the frozen split. xy float32, gene int32, cell int64."""
    L = lambda n: np.load(os.path.join(data, n + ".npy"), mmap_mode="r")
    D = dict(xy=np.asarray(L("xyz")[:, :2]), gene=np.asarray(L("gene")).astype(np.int32), nuc=np.asarray(L("nuc")),
             cell=np.asarray(L("cell")).astype(np.int64), clust=np.asarray(L("clust")), role=np.load(os.path.join(p0, "role.npy")),
             split=json.load(open(os.path.join(p0, "split.json"))), genes=json.load(open(os.path.join(data, "genes.json"))))
    D["bnd"] = {k: dict(np.load(os.path.join(data, f"{k}_bnd.npz"))) for k in ("nucleus", "cell")}; D["G"] = len(D["genes"])
    assert len(D["role"]) == len(D["xy"]), "role.npy does not match these arrays"
    return D


def std_fit(Xf, Xt):
    s = StandardScaler().fit(Xf); s.scale_[s.scale_ < 1e-6] = 1.0; return s.transform(Xf), s.transform(Xt)


# ---------------- subcellular ----------------
def mol_picks(D, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    return {r: rng.choice(np.flatnonzero((D["role"] == r) & (D["cell"] >= 0)), n, replace=False) for r in (FIT, TEST)}


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


def mol_targets(D, picks):
    T = dict(nucleus={r: (D["nuc"][p] == 1).astype(int) for r, p in picks.items()})
    for name, kind, signed in [("nucleus_signed_dist", "nucleus", True), ("cell_boundary_dist", "cell", False)]:
        T[name] = {r: boundary_dist(D["xy"][p], D["cell"][p], D["nuc"][p], D["bnd"][kind], signed) for r, p in picks.items()}
    return T


def reg_scores(yt, pt):
    return dict(r2=float(r2_score(yt, pt)), spearman=float(spearmanr(yt, pt).statistic))


def score_mol(F, T):
    """F: {FIT: features, TEST: features} for the picked molecules, in pick order."""
    out = {}; Xf, Xt = std_fit(F[FIT], F[TEST])
    lr = LogisticRegression(max_iter=2000).fit(Xf, T["nucleus"][FIT]); out["nucleus_auroc"] = float(roc_auc_score(T["nucleus"][TEST], lr.decision_function(Xt)))
    for name in ("nucleus_signed_dist", "cell_boundary_dist"):
        d = T[name]; okf, okt = np.isfinite(d[FIT]), np.isfinite(d[TEST]); Xf, Xt = std_fit(F[FIT][okf], F[TEST][okt])
        out[name] = dict(**reg_scores(d[TEST][okt], Ridge(alpha=1.0).fit(Xf, d[FIT][okf]).predict(Xt)), n_test=int(okt.sum()))
    return out


# ---------------- cells ----------------
def cell_table(D):
    """Per 10x cell: ids, raw counts Xc (cells x genes), cluster label, centroid (polygon vertex mean, else molecule mean), role of
    its centroid's block; plus inv mapping each cell-assigned molecule (mask ok) to its cell row."""
    gene, cell, clust, xy, sp = D["gene"], D["cell"], D["clust"], D["xy"], D["split"]
    ok = cell >= 0; ids, inv = np.unique(cell[ok], return_inverse=True); nC = len(ids)
    Xc = np.zeros((nC, D["G"]), np.float32); np.add.at(Xc, (inv, gene[ok]), 1)
    lab = np.full(nC, -1, np.int64); lab[inv] = clust[ok]
    Bc = D["bnd"]["cell"]; j = np.searchsorted(Bc["ids"], ids); has = (j < len(Bc["ids"])) & (Bc["ids"][np.minimum(j, len(Bc["ids"]) - 1)] == ids)
    cen = np.full((nC, 2), np.nan, np.float32); off = Bc["off"]
    sums = np.add.reduceat(Bc["xy"], off[:-1], axis=0) / np.diff(off)[:, None]; cen[has] = sums[j[has]]
    sumxy = np.zeros((nC, 2)); np.add.at(sumxy, inv, xy[ok]); n_c = np.bincount(inv, minlength=nC); cen[~has] = (sumxy / n_c[:, None])[~has]
    o = np.asarray(sp["origin"]); bs = sp["block_um"]; bnx = sp["bnx"]; brole = np.asarray(sp["block_role"])
    bi = (((cen[:, 1] - o[1]) // bs).astype(int).clip(0, sp["bny"] - 1) * bnx + ((cen[:, 0] - o[0]) // bs).astype(int).clip(0, bnx - 1))
    return dict(ids=ids, inv=inv, ok=ok, Xc=Xc, lab=lab, cen=cen, role=brole[bi], ncls=int(lab.max()) + 1)


def lognorm(X):
    return np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)


def cell_masks(C):
    return (C["role"] == FIT) & (C["lab"] >= 0), (C["role"] == TEST) & (C["lab"] >= 0)


def score_cell(F, C, rows=None):
    """F: features for every cell row, or only for `rows` (e.g. the probe cells, niche_targets()['idx']) when given."""
    f, t = cell_masks(C); lab = C["lab"]
    if rows is not None: f, t, lab = f[rows], t[rows], lab[rows]
    Xf, Xt = std_fit(F[f], F[t]); lr = LogisticRegression(max_iter=3000).fit(Xf, lab[f])
    return dict(balanced_accuracy=float(balanced_accuracy_score(lab[t], lr.predict(Xt))), n_fit=int(f.sum()), n_test=int(t.sum()), n_classes=C["ncls"])


# ---------------- niche ----------------
def niche_targets(C, r=50.0):
    f, t = cell_masks(C); idx = np.flatnonzero(f | t); lab, cen, ncls = C["lab"], C["cen"], C["ncls"]
    nb = cKDTree(cen).query_ball_point(cen[idx], r=r); Y = np.zeros((len(idx), ncls), np.float32)
    for i, (c, n) in enumerate(zip(idx, nb)):
        n = [m for m in n if m != c and lab[m] >= 0]
        if n: Y[i] = np.bincount(lab[n], minlength=ncls) / len(n)
    return dict(idx=idx, Y=Y, f=f[idx], t=t[idx])


def comp_scores(Yt, Pt):
    Yc, Pc = Yt - Yt.mean(1, keepdims=True), Pt - Pt.mean(1, keepdims=True)
    r = (Yc * Pc).sum(1) / np.maximum(np.sqrt((Yc ** 2).sum(1) * (Pc ** 2).sum(1)), 1e-12)
    return dict(pearson_per_cell=float(np.nanmean(r)), r2=float(r2_score(Yt, Pt)))


def score_niche(F, N):
    """F: features for the niche cells (rows aligned with N['idx'])."""
    Xf, Xt = std_fit(F[N["f"]], F[N["t"]]); return comp_scores(N["Y"][N["t"]], Ridge(alpha=1.0).fit(Xf, N["Y"][N["f"]]).predict(Xt))


def disc_mean(points, values, centers, r, chunk=4000):
    """Mean of `values` rows over points within r of each center (zeros where none). Chunked to bound memory."""
    tr = cKDTree(points); out = np.zeros((len(centers), values.shape[1]), np.float32)
    for s in range(0, len(centers), chunk):
        for i, w in enumerate(tr.query_ball_point(centers[s:s + chunk], r=r), s):
            if w: out[i] = values[w].mean(0)
    return out


def disc_counts(points, labels, centers, r, n, chunk=4000):
    """Counts of each label among points within r of each center. Chunked: a 50 um disc holds ~2,000 molecules."""
    tr = cKDTree(points); out = np.zeros((len(centers), n), np.float32)
    for s in range(0, len(centers), chunk):
        for i, w in enumerate(tr.query_ball_point(centers[s:s + chunk], r=r), s):
            if w: out[i] = np.bincount(labels[w], minlength=n)
    return out
