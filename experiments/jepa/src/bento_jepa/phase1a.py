"""Phase 1a: frozen CellWorld on patches.

Each 5 um (or 10 um) square patch of molecules is fed to the frozen CellWorld encoder as if it were a cell: the patch's gene counts
go through CellWorld's tokenizer (value-weighted sum of its gene embeddings, plus the breast and Xenium embeddings) and the context
encoder attends over the patches of a window. Windows are 64 x 64 patches (320 um at 5 um, about CellWorld's 4,096-token context)
with a halo; only each window's core patches are kept, so every patch sees context on all sides. Only probe blocks are embedded.

Two value scalings: 'cell' treats a patch like a cell (counts scaled to 10,000 per patch, log1p, as in CellWorld's corpus);
'raw' is log1p of the counts.

Scored with exactly the phase 0 probes (evalkit), so numbers compare one to one with baselines.json:
- subcellular: the embedding of the molecule's own patch;
- cell: the mean patch embedding over the cell's molecules;
- niche: the mean embedding of patches whose centres lie within 25 um of the cell's centroid (a 50 um disc).
References: 'patch_counts' (log1p of the patch's own gene counts: the token's input, no model) for subcellular, and 'native'
(CellWorld on segmented 10x cells, its intended use; it has seen this section) for cell and niche.

usage: python -m bento_jepa.phase1a (DATA_DIR PHASE0_DIR CKPT | --paths paths.json) --out results.json [--patch 5 10] [--scaling cell raw] [--native]
       [--max-blocks N] [--device cuda]
"""
import os, json, time, argparse, numpy as np, scipy.sparse as sps
from importlib import resources
from . import evalkit as ek

ORGAN, PLATFORM = 2, 2          # CellWorld data/metadata_vocab.json: organ 'breast' = 2, platform 'xenium' = 2
BIG = 1 << 20                   # patch key = ix * BIG + iy (global patch indices from the split's origin)


class Grid:
    """Molecules sorted into 100 um tiles for fast rectangle queries; returns indices into the original arrays."""
    def __init__(s, xy, tile=100.0):
        s.lo = xy.min(0); s.t = tile; n = np.floor((xy.max(0) - s.lo) / tile).astype(int) + 1; s.nx, s.ny = int(n[0]), int(n[1])
        key = s._key(xy[:, 0], xy[:, 1]); s.order = np.argsort(key, kind="stable"); s.start = np.searchsorted(key[s.order], np.arange(s.nx * s.ny + 1))
        s.xy = xy

    def _key(s, x, y):
        tx = np.clip(((x - s.lo[0]) // s.t).astype(np.int64), 0, s.nx - 1); ty = np.clip(((y - s.lo[1]) // s.t).astype(np.int64), 0, s.ny - 1)
        return ty * s.nx + tx

    def rect(s, x0, y0, x1, y1):
        tx0, tx1 = int(max((x0 - s.lo[0]) // s.t, 0)), int(min((x1 - s.lo[0]) // s.t, s.nx - 1))
        ty0, ty1 = int(max((y0 - s.lo[1]) // s.t, 0)), int(min((y1 - s.lo[1]) // s.t, s.ny - 1))
        if tx1 < tx0 or ty1 < ty0: return np.zeros(0, np.int64)
        idx = s.order[np.concatenate([np.arange(s.start[ty * s.nx + tx0], s.start[ty * s.nx + tx1 + 1]) for ty in range(ty0, ty1 + 1)])]
        p = s.xy[idx]; return idx[(p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)]


def panel_tokens(genes):
    rows = json.loads(resources.files("bento_jepa").joinpath("assets/panel_xenium_breast_idc.json").read_text())
    assert [r["panel_symbol"] for r in rows] == list(genes), "panel map does not match genes.json"
    return np.array([r["token"] for r in rows], np.int64)


def load_cellworld(path, device):
    import torch
    from .cellworld import CellWorld
    blob = torch.load(path, map_location="cpu", weights_only=True)   # state dict + plain config only; no code is unpickled
    m = CellWorld(blob["config"]); m.load_state_dict(blob["model"]); return m.eval().to(device)


def encode(model, gene_ids, values, offsets, coords, device):
    """One window (batch of 1, no padding: CellWorld rescales positions over all tokens, so padding would shift them)."""
    import torch
    T = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    batch = dict(gene_ids=T(gene_ids, torch.long), values=T(values, torch.float32), gene_offsets=T(offsets, torch.long),
                 coordinates=T(coords[None], torch.float32), organ_id=T([ORGAN], torch.long), platform_id=T([PLATFORM], torch.long))
    with torch.no_grad(), torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu", dtype=torch.bfloat16,
                                         enabled=str(device).startswith("cuda")):
        return model.encode(batch)[0].float().cpu().numpy()


def bag(rowid, gene, counts, tok, scaling):
    """Rows sorted by rowid -> CellWorld bag inputs: gene token ids, values, offsets (one bag per row)."""
    start = np.flatnonzero(np.r_[True, rowid[1:] != rowid[:-1]]); tot = np.add.reduceat(counts, start)
    per = np.repeat(tot, np.diff(np.r_[start, len(rowid)]))
    v = np.log1p(counts / per * 1e4) if scaling == "cell" else np.log1p(counts)
    return tok[gene], v.astype(np.float32), start


def probe_blocks(sp, roles, max_blocks):
    br = np.asarray(sp["block_role"]); out = []
    for r in roles:
        b = np.flatnonzero(br == r); out += list(b[:max_blocks] if max_blocks else b)
    return out


def embed_patches(D, grid, model, tok, P, scaling, device, side=64, halo_frac=0.125, max_blocks=None, log=print):
    sp = D["split"]; o = np.asarray(sp["origin"], np.float64); B = sp["block_um"]; G = D["G"]
    S = side * P; h = round(halo_frac * side) * P; core = S - 2 * h; keys, embs = [], []; t0 = time.time()
    blocks = probe_blocks(sp, (ek.FIT, ek.TEST), max_blocks); nwin = len(blocks) * int(np.ceil(B / core)) ** 2; k = 0
    for b in blocks:
        X0, Y0 = o[0] + (b % sp["bnx"]) * B, o[1] + (b // sp["bnx"]) * B
        for cx in np.arange(X0, X0 + B, core):
            for cy in np.arange(Y0, Y0 + B, core):
                k += 1; idx = grid.rect(cx - h, cy - h, cx - h + S, cy - h + S)
                if len(idx) == 0: continue
                xy = D["xy"][idx]; gx = np.floor((xy[:, 0] - o[0]) / P).astype(np.int64); gy = np.floor((xy[:, 1] - o[1]) / P).astype(np.int64)
                code = (gx * BIG + gy) * G + D["gene"][idx]; u, c = np.unique(code, return_counts=True); pk, gg = u // G, u % G
                gid, val, off = bag(pk, gg, c.astype(np.float32), tok, scaling); pkeys = pk[off]
                px, py = pkeys // BIG, pkeys % BIG; coords = np.stack([(px + 0.5) * P, (py + 0.5) * P], 1).astype(np.float32)
                E = encode(model, gid, val, off, coords - coords.mean(0), device)
                cx0, cy0 = (cx - o[0]) / P, (cy - o[1]) / P; ex, ey = min(cx + core, X0 + B), min(cy + core, Y0 + B)
                keep = (px >= cx0) & (px < (ex - o[0]) / P) & (py >= cy0) & (py < (ey - o[1]) / P)
                keys.append(pkeys[keep]); embs.append(E[keep].astype(np.float16))
                if k % 50 == 0: log(f"  P={P} {scaling}: window {k}/{nwin}, {time.time()-t0:.0f}s")
    keys = np.concatenate(keys); embs = np.concatenate(embs); o_ = np.argsort(keys); return keys[o_], embs[o_]


def lookup(keys, q):
    j = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1); return j, keys[j] == q


def mol_keys(D, idx, P):
    o = np.asarray(D["split"]["origin"]); xy = D["xy"][idx]
    return np.floor((xy[:, 0] - o[0]) / P).astype(np.int64) * BIG + np.floor((xy[:, 1] - o[1]) / P).astype(np.int64)


def features_from_patches(D, C, N, picks, keys, E, P):
    """Molecule, cell and niche features from patch embeddings."""
    out = {}; d = E.shape[1]
    F = {}
    for r, p in picks.items():
        j, ok = lookup(keys, mol_keys(D, p, P)); F[r] = np.where(ok[:, None], E[j].astype(np.float32), 0); out[f"found_mol_{r}"] = float(ok.mean())
    rows = N["idx"]; isrow = np.full(len(C["ids"]), -1, np.int64); isrow[rows] = np.arange(len(rows))
    mi = np.flatnonzero(C["ok"]); rr = isrow[C["inv"]]; sel = rr >= 0; mi, rr = mi[sel], rr[sel]
    j, ok = lookup(keys, mol_keys(D, mi, P)); W = sps.csr_matrix((np.ones(ok.sum(), np.float32), (rr[ok], j[ok])), shape=(len(rows), len(keys)))
    n = np.asarray(W.sum(1)).ravel(); Fc = np.asarray(W @ E.astype(np.float32)) / np.maximum(n, 1)[:, None]; out["found_cell"] = float((n > 0).mean())
    o = np.asarray(D["split"]["origin"]); cen = np.stack([(keys // BIG + 0.5) * P + o[0], (keys % BIG + 0.5) * P + o[1]], 1)
    Fn = ek.disc_mean(cen, E.astype(np.float32), C["cen"][rows], 25.0)
    return F, Fc, Fn, out


def patch_count_features(D, picks, P):
    """log1p gene counts of each picked molecule's own patch (molecules of the probe blocks)."""
    G = D["G"]; allq = np.unique(np.concatenate([mol_keys(D, p, P) for p in picks.values()]))
    held = np.flatnonzero(D["role"] >= ek.FIT); mk = mol_keys(D, held, P); m = np.isin(mk, allq)
    u, c = np.unique(mk[m] * G + D["gene"][held[m]], return_counts=True)
    M = sps.csr_matrix((c.astype(np.float32), (np.searchsorted(allq, u // G), u % G)), shape=(len(allq), G))
    return {r: np.log1p(M[np.searchsorted(allq, mol_keys(D, p, P))].toarray()) for r, p in picks.items()}


def embed_native(D, C, model, tok, device, core=600.0, halo=100.0, max_blocks=None, log=print):
    """CellWorld on segmented cells (its intended input): windows of ~5k cells, centroid coordinates, log1p counts per 10k."""
    from scipy.spatial import cKDTree
    sp = D["split"]; o = np.asarray(sp["origin"]); B = sp["block_um"]; Xn = ek.lognorm(C["Xc"]); cen = C["cen"]
    out = np.zeros((len(C["ids"]), model.embed_dim), np.float32); done = np.zeros(len(C["ids"]), bool); t0 = time.time()
    for b in probe_blocks(sp, (ek.FIT, ek.TEST), max_blocks):
        X0, Y0 = o[0] + (b % sp["bnx"]) * B, o[1] + (b // sp["bnx"]) * B
        for cx in np.arange(X0, X0 + B, core):
            for cy in np.arange(Y0, Y0 + B, core):
                w = np.flatnonzero((cen[:, 0] >= cx - halo) & (cen[:, 0] < cx + core + halo) & (cen[:, 1] >= cy - halo) & (cen[:, 1] < cy + core + halo))
                if len(w) == 0: continue
                r, g = np.nonzero(Xn[w]); gid, val, off = tok[g], Xn[w][r, g].astype(np.float32), np.searchsorted(r, np.arange(len(w)))
                has = np.diff(np.r_[off, len(r)]) > 0; w, off = w[has], off[has]
                E = encode(model, gid, val, off, cen[w] - cen[w].mean(0), device)
                ex, ey = min(cx + core, X0 + B), min(cy + core, Y0 + B)
                keep = (cen[w, 0] >= cx) & (cen[w, 0] < ex) & (cen[w, 1] >= cy) & (cen[w, 1] < ey); out[w[keep]] = E[keep]; done[w[keep]] = True
        log(f"  native: block {b} done, {time.time()-t0:.0f}s")
    return out, done


def restrict_blocks(D, max_blocks):
    """Smoke mode: only the first max_blocks probe blocks per role stay probe blocks; the rest count as train, so every probe
    molecule and cell is embedded and scores are meaningful (on far fewer cells than the full run)."""
    sp = D["split"]; br = np.asarray(sp["block_role"]).copy(); keep = set(probe_blocks(sp, (ek.FIT, ek.TEST), max_blocks))
    drop = [b for b in np.flatnonzero(br >= ek.FIT) if b not in keep]; br[drop] = 0
    o = np.asarray(sp["origin"]); B = sp["block_um"]; xy = D["xy"]
    bi = ((xy[:, 1] - o[1]) // B).astype(np.int64) * sp["bnx"] + ((xy[:, 0] - o[0]) // B).astype(np.int64)
    m = np.isin(bi, drop) & (D["role"] >= ek.FIT); D["role"] = D["role"].copy(); D["role"][m] = 0
    D["split"] = dict(sp, block_role=br.tolist())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data", nargs="?"); ap.add_argument("p0", nargs="?"); ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--paths", help="paths.json from bento_jepa.hub prepare (data, phase0, ckpt)"); ap.add_argument("--out", required=True)
    ap.add_argument("--patch", type=float, nargs="+", default=[5.0, 10.0]); ap.add_argument("--scaling", nargs="+", default=["cell", "raw"])
    ap.add_argument("--native", action="store_true"); ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--device", default=None); ap.add_argument("--n-mol", type=int, default=200_000); a = ap.parse_args()
    if a.paths: p = json.load(open(a.paths)); a.data, a.p0, a.ckpt = p["data"], p["phase0"], p["ckpt"]
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
    D = ek.load(a.data, a.p0); tok = panel_tokens(D["genes"]); model = load_cellworld(a.ckpt, dev)
    if a.max_blocks: restrict_blocks(D, a.max_blocks); a.n_mol = min(a.n_mol, int(((D["role"] == ek.TEST) & (D["cell"] >= 0)).sum()) // 2)
    res = dict(ckpt=os.path.basename(os.path.dirname(a.ckpt)) or a.ckpt, device=dev, split_sha256=D["split"]["role_sha256"],
               max_blocks=a.max_blocks, baselines=json.loads(resources.files("bento_jepa").joinpath("assets/baselines.json").read_text()), runs={})
    save = lambda: json.dump(res, open(a.out, "w"), indent=1)
    picks = ek.mol_picks(D, a.n_mol); T = ek.mol_targets(D, picks); C = ek.cell_table(D); N = ek.niche_targets(C); grid = Grid(D["xy"])
    log(f"loaded: {len(D['xy']):,} molecules, {len(N['idx']):,} probe cells, model {res['ckpt']} on {dev}")
    from scipy.spatial import cKDTree
    from .baselines import comp
    held = np.flatnonzero(D["role"] >= ek.FIT); tr = cKDTree(D["xy"][held])   # comp20 on the same molecules: the bar (also in smoke mode)
    res["runs"]["comp20"] = dict(subcellular=ek.score_mol({r: np.log1p(comp(tr, D["gene"][held], D["xy"][p], 20, D["G"])) for r, p in picks.items()}, T))
    win = ek.lognorm(ek.disc_counts(D["xy"][held], D["gene"][held], C["cen"][N["idx"]], 25.0, D["G"]))   # niche bar: 50 um composition
    res["runs"]["own_counts"] = dict(cell=ek.score_cell(ek.lognorm(C["Xc"]), C)); res["runs"]["composition_50um"] = dict(niche=ek.score_niche(win, N))
    del tr, win; save()
    log(f"bars: {json.dumps({k: res['runs'][k] for k in ('comp20', 'own_counts', 'composition_50um')})}")
    for P in a.patch:
        name = f"patch_counts_{P:g}um"; res["runs"][name] = dict(subcellular=ek.score_mol(patch_count_features(D, picks, P), T)); save(); log(f"{name}: {json.dumps(res['runs'][name])}")
        for sc in a.scaling:
            keys, E = embed_patches(D, grid, model, tok, P, sc, dev, max_blocks=a.max_blocks, log=log)
            F, Fc, Fn, cov = features_from_patches(D, C, N, picks, keys, E, P); name = f"cellworld_{P:g}um_{sc}"
            res["runs"][name] = dict(n_patches=int(len(keys)), coverage=cov, subcellular=ek.score_mol(F, T),
                                     cell=ek.score_cell(Fc, C, rows=N["idx"]), niche=ek.score_niche(Fn, N)); save()
            log(f"{name}: {json.dumps(res['runs'][name])}")
    if a.native:
        E, done = embed_native(D, C, model, tok, dev, max_blocks=a.max_blocks, log=log); rows = N["idx"]
        res["runs"]["cellworld_native_cells"] = dict(coverage=dict(found_cell=float(done[rows].mean())), cell=ek.score_cell(E[rows], C, rows=rows),
                                                     niche=ek.score_niche(E[rows], N)); save(); log(f"native: {json.dumps(res['runs']['cellworld_native_cells'])}")
    res["seconds"] = round(time.time() - t0); save(); log("done")


if __name__ == "__main__":
    main()
