"""Phase 0: composition baselines on the frozen split, fit on probe-fit blocks and scored on probe-test blocks (see evalkit).

Subcellular: comp20 = gene counts of the 20 nearest molecules (2D, excluding self, searched among probe-block molecules), log1p.
Cell: the cell's own counts, scaled to 10,000 per cell, log1p.
Niche (50 um cell-type composition): own counts; gene composition of the 50 um disc (probe-block molecules only, so discs at a
block edge are truncated); squidpy CellCharter niches (unsupervised, all cells; one-hot).

usage: python -m bento_jepa.baselines DATA_DIR PHASE0_DIR [--n-mol 200000] [--niche-k 12]
writes PHASE0_DIR/baselines.json
"""
import os, json, time, argparse, numpy as np
from scipy.spatial import cKDTree
from . import evalkit as ek


def comp(tree, gene, q, k, G):
    _, nn = tree.query(q, k=k + 1); nn = nn[:, 1:]
    X = np.zeros((len(q), G), np.float32); np.add.at(X, (np.repeat(np.arange(len(q)), k), gene[nn].ravel()), 1); return X


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("p0"); ap.add_argument("--n-mol", type=int, default=200_000)
    ap.add_argument("--niche-k", type=int, default=12); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    t0 = time.time(); D = ek.load(a.data, a.p0); G = D["G"]; res = dict(split_sha256=D["split"]["role_sha256"])
    save = lambda: json.dump(res, open(os.path.join(a.p0, "baselines.json"), "w"), indent=1)

    picks = ek.mol_picks(D, a.n_mol, a.seed); T = ek.mol_targets(D, picks)
    held = np.flatnonzero(D["role"] >= ek.FIT); tree = cKDTree(D["xy"][held]); gh = D["gene"][held]
    F = {r: np.log1p(comp(tree, gh, D["xy"][p], 20, G)) for r, p in picks.items()}; del tree
    res["subcellular_comp20"] = ek.score_mol(F, T); save(); print("subcellular", json.dumps(res["subcellular_comp20"]), f"{time.time()-t0:.0f}s", flush=True)

    C = ek.cell_table(D); Xn = ek.lognorm(C["Xc"])
    res["cell_own_counts"] = ek.score_cell(Xn, C); save(); print("cell", json.dumps(res["cell_own_counts"]), f"{time.time()-t0:.0f}s", flush=True)

    N = ek.niche_targets(C); idx = N["idx"]
    win = ek.lognorm(ek.disc_counts(D["xy"][held], gh, C["cen"][idx], 25.0, G))
    import anndata as ad, squidpy as sq, scanpy as sc
    A = ad.AnnData(X=Xn.copy()); A.obsm["spatial"] = C["cen"].astype(np.float64); sc.pp.pca(A, n_comps=30, random_state=a.seed)
    sq.gr.spatial_neighbors(A, coord_type="generic", delaunay=True)
    sq.gr.calculate_niche(A, flavor="cellcharter", distance=3, n_components=a.niche_k, use_rep="X_pca", aggregation="mean", rng=a.seed)
    nk = [c for c in A.obs.columns if "niche" in c.lower()][0]; niche = A.obs[nk].astype("category").cat.codes.to_numpy()
    oh = np.eye(niche.max() + 1, dtype=np.float32)[niche[idx]]
    res["niche_50um_composition"] = dict({k: ek.score_niche(F_, N) for k, F_ in [("own_counts", Xn[idx]), ("window_composition_50um", win),
                                          ("cellcharter_niche_onehot", oh)]}, n_fit=int(N["f"].sum()), n_test=int(N["t"].sum()), cellcharter_k=int(niche.max() + 1))
    print("niche", json.dumps(res["niche_50um_composition"]), f"{time.time()-t0:.0f}s", flush=True)
    res["seconds"] = round(time.time() - t0); save()


if __name__ == "__main__":
    main()
