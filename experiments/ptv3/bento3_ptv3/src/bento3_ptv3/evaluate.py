"""Frozen-embedding linear probes on held-out 1 mm blocks: fit on probe-fit blocks, score on probe-test blocks.
Probes (labels are eval-only, from 10x segmentation/clustering; never used in training):
  nuc   nucleus vs cytoplasm (overlaps_nucleus, molecules assigned to a cell)      -> AUROC, balanced accuracy
  clust 10x graph-based cluster of the molecule's cell (26 clusters)               -> balanced accuracy, accuracy
  gene  masked-gene probe: the molecule's own gene hidden (mask token), 280 genes  -> top-1, macro recall, top-1 per abundance bin
Baselines on the same molecules and the same probe: own gene one-hot ('gene1h'), and the gene counts of the 20 nearest molecules
('comp20'; + own gene one-hot for nuc/clust, without it for the gene probe). Read the model numbers against these, not alone."""
import numpy as np, torch, torch.nn.functional as F
from scipy.spatial import cKDTree
from .data import make_view

ABIN = ["lt1k", "1k_10k", "10k_100k", "100k_1M", "ge1M"]   # training-molecule count bins for the 68M-molecule section


def collate(views, dev):
    out = dict(coord=torch.tensor(np.concatenate([v["coord"] for v in views])), grid=torch.tensor(np.concatenate([v["grid"] for v in views])),
               gene=torch.tensor(np.concatenate([v["gene"] for v in views])), mask=torch.tensor(np.concatenate([v["mask"] for v in views])),
               coord_n=torch.tensor(np.concatenate([v["coord_n"] for v in views])), offset=torch.tensor(np.cumsum([len(v["gene"]) for v in views])))
    return {k: v.to(dev, non_blocking=True) for k, v in out.items()}


@torch.no_grad()
def embed(net, M, tiles, cfg, dev, amp, gene_masked=False):
    """Per tile: core molecules (<= eval_per_tile, fixed subsample). Returns idx, X (972-d upcast feats, fp16 on dev), E (emb_dim, fp16 cpu)
    and, with gene_masked, Xm = feats of each molecule while its own gene is hidden (4 disjoint 25% partitions)."""
    net.eval(); I, X, E, XM = [], [], [], []
    for sel, core, _, _ in tiles:
        c = np.flatnonzero(core)
        if len(c) > cfg.eval_per_tile: c = np.sort(np.random.default_rng(len(sel)).choice(c, cfg.eval_per_tile, replace=False))
        v = make_view(M, sel, None, cfg, augment=False, mask=False); b = collate([v], dev)
        with torch.autocast(**amp): up, e = net.features(b)
        I.append(sel[c]); X.append(up[c].float()); E.append(e[c].float().cpu())   # float32: fp16 overflowed to inf in lora_ssl_s0
        if gene_masked:
            part = np.random.default_rng(99).integers(0, 4, len(sel)); xm = torch.zeros(len(c), up.shape[1], dtype=torch.float32, device=dev)
            for r in range(4):
                b["mask"] = torch.tensor(part == r, device=dev)
                with torch.autocast(**amp): u2, _ = net.features(b)
                w = part[c] == r; xm[torch.tensor(w, device=dev)] = u2[c[w]].float()
            XM.append(xm)
    out = dict(idx=np.concatenate(I), X=torch.cat(X), E=torch.cat(E))
    if gene_masked: out["Xm"] = torch.cat(XM)
    return out


def comp_feats(M, tiles, idx_per_tile, k=20):
    """Gene counts among the k nearest molecules (2D, same tile, excluding self) for each selected molecule."""
    rows = []
    for (sel, core, _, _), take in zip(tiles, idx_per_tile):
        tr = cKDTree(M.xyz[sel, :2]); q = M.xyz[take, :2]; _, nn_ = tr.query(q, k=k + 1); g = M.gene[sel][nn_[:, 1:]]
        X = np.zeros((len(take), M.G), np.float32); np.add.at(X, (np.repeat(np.arange(len(take)), k), g.ravel()), 1); rows.append(X)
    return np.concatenate(rows)


def linear_probe(Xf, yf, Xt, ncls, dev, iters=1500, bs=4096, lr=3e-3, wd=1e-4, seed=0):
    """Standardised multinomial logistic regression with Adam (minibatches). Returns test logits (cpu float)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xf = torch.as_tensor(Xf, device=dev).float(); Xt = torch.as_tensor(Xt, device=dev).float(); yf = torch.as_tensor(yf, device=dev).long()
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-5; W = torch.nn.Linear(Xf.shape[1], ncls).to(dev); opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=wd)
    for i in range(iters):
        j = torch.randint(0, len(Xf), (min(bs, len(Xf)),), generator=g).to(dev)
        loss = F.cross_entropy(W((Xf[j] - mu) / sd), yf[j]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return torch.cat([W((Xt[i:i + 65536] - mu) / sd) for i in range(0, len(Xt), 65536)]).float().cpu()


def auroc(y, s):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def bacc(y, p, ncls):
    rec = [(p[y == c] == c).mean() for c in range(ncls) if (y == c).any()]; return float(np.mean(rec))


def probes(M, F_fit, F_test, y_fit, y_test, kind, dev, ncls):
    lg = linear_probe(F_fit, y_fit, F_test, ncls, dev); p = lg.argmax(1).numpy()
    if kind == "nuc": return dict(auroc=auroc(y_test, (lg[:, 1] - lg[:, 0]).numpy()), bacc=bacc(y_test, p, 2))
    return dict(bacc=bacc(y_test, p, ncls), acc=float((p == y_test).mean()))


def evaluate(net, M, cfg, dev, amp, n_tiles, gene=False, baselines=False):
    """Returns a flat dict of metrics (probe/*, base_*/*, gene/*) and the probe-test embeddings (idx, E) for saving."""
    tf, tt = M.eval_tiles(1, n_tiles // 2), M.eval_tiles(2, n_tiles // 2)
    A, B = embed(net, M, tf, cfg, dev, amp, gene), embed(net, M, tt, cfg, dev, amp, gene); res = dict(eval_tiles=len(tf) + len(tt), eval_mol=len(A["idx"]) + len(B["idx"]))
    for nm, D in (("fit", A), ("test", B)):
        res[f"feat_absmax_{nm}"] = float(D["X"].abs().max()); res[f"feat_nonfinite_{nm}"] = int((~torch.isfinite(D["X"])).sum())
    model_ok = res["feat_nonfinite_fit"] == 0 and res["feat_nonfinite_test"] == 0
    if not model_ok: print(f"eval: non-finite model features ({res['feat_nonfinite_fit']} fit / {res['feat_nonfinite_test']} test); model probes = NaN", flush=True)
    feats = {"model": (A["X"], B["X"])} if model_ok else {}
    if not model_ok:
        res |= {"probe/nuc_auroc": float("nan"), "probe/nuc_bacc": float("nan"), "probe/clust_bacc": float("nan"), "probe/clust_acc": float("nan")}
    if baselines:
        split = lambda D, T: np.split(D["idx"], np.cumsum([min(core.sum(), cfg.eval_per_tile) for _, core, _, _ in T])[:-1])
        cf, ct = comp_feats(M, tf, split(A, tf)), comp_feats(M, tt, split(B, tt)); oh = lambda i: np.eye(M.G, dtype=np.float32)[M.gene[i]]
        feats["gene1h"] = (oh(A["idx"]), oh(B["idx"])); feats["comp20"] = (np.c_[cf, oh(A["idx"])], np.c_[ct, oh(B["idx"])])
    for lab, kind, ncls in [(M.nuc, "nuc", 2), (M.clust, "clust", int(M.clust.max()) + 1)]:
        kf, kt = lab[A["idx"]] >= 0, lab[B["idx"]] >= 0
        for name, (Ff, Ft) in feats.items():
            r = probes(M, Ff[torch.tensor(kf)] if torch.is_tensor(Ff) else Ff[kf], Ft[torch.tensor(kt)] if torch.is_tensor(Ft) else Ft[kt],
                       lab[A["idx"]][kf].astype(np.int64), lab[B["idx"]][kt].astype(np.int64), kind, dev, ncls)
            res |= {f"{'probe' if name == 'model' else 'base_' + name}/{kind}_{k}": v for k, v in r.items()}
    if gene:
        yf, yt = M.gene[A["idx"]], M.gene[B["idx"]]; G = M.G
        bins = np.digitize(M.ntr, [1e3, 1e4, 1e5, 1e6]); ptr = M.ntr / M.ntr.sum()
        def gscore(pre, lg):
            p = lg.argmax(1).numpy(); n_g = np.bincount(yt, minlength=G); h_g = np.bincount(yt[p == yt], minlength=G); seen = n_g > 0
            out = {f"{pre}/gene_top1": float((p == yt).mean()), f"{pre}/gene_macro_recall": float((h_g[seen] / n_g[seen]).mean())}
            out |= {f"{pre}/gene_top1_{ABIN[b]}": float(h_g[bins == b].sum() / max(n_g[bins == b].sum(), 1)) for b in range(5)}
            return out
        if model_ok: res |= gscore("probe", linear_probe(A["Xm"], yf, B["Xm"], G, dev))
        if baselines:
            res |= gscore("base_comp20", linear_probe(cf, yf, ct, G, dev))
            res |= {"base_majority/gene_top1": float((yt == M.ntr.argmax()).mean()), "base_prior/gene_macro_recall": float(ptr[np.bincount(yt, minlength=G) > 0].mean())}
    net.train(); return res, B["idx"], B["E"]
