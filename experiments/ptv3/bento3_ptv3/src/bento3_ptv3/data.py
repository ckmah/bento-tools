"""Full-section Xenium molecules: download/build, spatial index, held-out blocks, and the two augmented views per region.

Dataset: 10x Genomics 'FFPE Human Breast with Pre-designed Panel', tissue sample 1 (IDC/DCIS), Xenium Onboard Analysis 1.0.2,
hBreast_v1 280-gene panel (the same 280 genes as the 2-FOV set used before). 68,113,826 molecules with qv >= 20 on real genes,
~0.74 cm2 of tissue, ~9.2k molecules per 100 um tile (median over tiles with tissue).
Labels used ONLY for evaluation probes: nuc (overlaps_nucleus of assigned molecules) and clust (10x graph-based cluster of the cell)."""
import os, json, time, hashlib, numpy as np

URL_10X = "https://cf.10xgenomics.com/samples/xenium/1.0.2/Xenium_V1_FFPE_Human_Breast_IDC/Xenium_V1_FFPE_Human_Breast_IDC_outs.zip"
ARRAYS = ["xyz", "gene", "nuc", "cell", "clust"]
CTRL = ("NegControl", "BLANK", "Unassigned", "Deprecated", "antisense", "Intergenic")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def build_from_10x(out, raw=None, url=URL_10X):
    """Stream transcripts.parquet (range-read out of the 24 GB 10x zip, ~1.4 GB) row group by row group into .npy memmaps.
    Peak RAM ~0.3 GB. Same filter as the 2-FOV build: qv >= 20 and real genes only."""
    import pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
    from .httpzip import extract
    raw = raw or os.path.join(out, "raw"); os.makedirs(raw, exist_ok=True); os.makedirs(out, exist_ok=True)
    for m in ["transcripts.parquet", "analysis/clustering/gene_expression_graphclust/clusters.csv"]:
        dst = os.path.join(raw, os.path.basename(m))
        if not os.path.exists(dst): t = time.time(); extract(url, m, dst); print(f"10x: {m} {os.path.getsize(dst) / 1e6:.0f} MB in {time.time() - t:.0f}s", flush=True)
    f = pq.ParquetFile(os.path.join(raw, "transcripts.parquet")); t0 = time.time()
    def keep(rg):
        fn = pc.cast(rg.column("feature_name"), "string"); m = pc.greater_equal(rg.column("qv"), 20.0)
        for c in CTRL: m = pc.and_(m, pc.invert(pc.starts_with(fn, c)))
        return m, fn
    n, names, nraw = 0, set(), 0
    for i in range(f.num_row_groups):
        rg = f.read_row_group(i, columns=["feature_name", "qv"]); m, fn = keep(rg); nraw += rg.num_rows
        n += pc.sum(m).as_py() or 0; names |= set(pc.unique(pc.filter(fn, m)).to_pylist())
    genes = np.array(sorted(names))
    cl = pd.read_csv(os.path.join(raw, "clusters.csv")); cmax = int(cl.Barcode.max())
    clmap = np.full(cmax + 2, -1, np.int16); clmap[cl.Barcode.values] = cl.Cluster.values.astype(np.int16)
    mm = lambda name, dt, shape: np.lib.format.open_memmap(os.path.join(out, name + ".tmp.npy"), "w+", dt, shape)
    A = dict(xyz=mm("xyz", np.float32, (n, 3)), gene=mm("gene", np.int16, (n,)), nuc=mm("nuc", np.int8, (n,)), cell=mm("cell", np.int32, (n,)), clust=mm("clust", np.int16, (n,)))
    k = 0
    for i in range(f.num_row_groups):
        rg = f.read_row_group(i); m, fn = keep(rg); rg = rg.filter(m); fn = pc.filter(fn, m); j = k + rg.num_rows
        A["xyz"][k:j] = np.stack([rg.column(c).to_numpy() for c in ["x_location", "y_location", "z_location"]], 1)
        A["gene"][k:j] = np.searchsorted(genes, np.asarray(fn.to_pylist(), dtype=object).astype(str))
        cid = rg.column("cell_id").to_numpy().astype(np.int64)
        A["nuc"][k:j] = np.where(cid >= 0, rg.column("overlaps_nucleus").to_numpy().astype(np.int8), -1)
        A["cell"][k:j] = cid; A["clust"][k:j] = np.where((cid >= 0) & (cid <= cmax), clmap[np.clip(cid, 0, cmax)], -1); k = j
    assert k == n
    for a in A.values(): a.flush()
    del A
    for name in ARRAYS: os.replace(os.path.join(out, name + ".tmp.npy"), os.path.join(out, name + ".npy"))
    json.dump([str(g) for g in genes], open(os.path.join(out, "genes.json"), "w"))
    man = dict(source=URL_10X, n_raw=int(nraw), n=int(n), n_genes=len(genes), sha256={a + ".npy": _sha(os.path.join(out, a + ".npy")) for a in ARRAYS},
               build_sec=round(time.time() - t0))
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1); print(f"built {n:,} molecules, {len(genes)} genes", flush=True)
    return out


def ensure_data(cfg):
    """data_dir/xenium_breast_idc_v1/{xyz,gene,nuc,cell,clust}.npy + genes.json + manifest.json. HF first (public, no token), else 10x."""
    out = os.path.join(cfg.data_dir, os.path.basename(cfg.hf_data_path)); ok = os.path.join(out, "ok")
    if os.path.exists(ok): return out
    if cfg.data_source == "hf":
        try:
            from huggingface_hub import hf_hub_download
            t = time.time()
            for f in ["manifest.json", "genes.json"] + [a + ".npy" for a in ARRAYS]:
                hf_hub_download(cfg.hf_data_repo, f"{cfg.hf_data_path}/{f}", local_dir=cfg.data_dir + "/_hf")
            src = os.path.join(cfg.data_dir, "_hf", cfg.hf_data_path); man = json.load(open(os.path.join(src, "manifest.json")))
            for f, h in man["sha256"].items(): assert _sha(os.path.join(src, f)) == h, f"sha256 mismatch: {f}"
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            if os.path.exists(out): os.rename(out, out + f".old{int(time.time())}")
            os.rename(src, out); print(f"data: hf.co/{cfg.hf_data_repo}/{cfg.hf_data_path} in {time.time() - t:.0f}s (sha256 OK)", flush=True)
        except Exception as e:
            print(f"data: HF download failed ({type(e).__name__}: {e}); rebuilding from the 10x bundle", flush=True); build_from_10x(out)
    else: build_from_10x(out)
    open(ok, "w").write(time.strftime("%F %T")); return out


class Molecules:
    """All molecules in RAM, sorted by 50 um tile for fast window queries; held-out = random 1 mm blocks (fixed split_seed)."""
    TILE = 50.0

    def __init__(s, path, cfg):
        t = time.time(); L = lambda a: np.load(os.path.join(path, a + ".npy"))
        xyz, gene, nuc, clust = L("xyz"), L("gene").astype(np.int64), L("nuc"), L("clust")
        s.genes = json.load(open(os.path.join(path, "genes.json"))); s.G = len(s.genes)
        if cfg.crop:
            x0, y0, w = cfg.crop; k = (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + w) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + w)
            xyz, gene, nuc, clust = xyz[k], gene[k], nuc[k], clust[k]
        s.lo = xyz[:, :2].min(0); s.nx, s.ny = (np.floor((xyz[:, :2].max(0) - s.lo) / s.TILE) + 1).astype(int)
        key = s._tile(xyz[:, 0], xyz[:, 1]); o = np.argsort(key, kind="stable")
        s.xyz, s.gene, s.nuc, s.clust = xyz[o], gene[o], nuc[o], clust[o]; key = key[o]
        s.start = np.searchsorted(key, np.arange(s.nx * s.ny + 1)); s.zmin = float(s.xyz[:, 2].min())
        # held-out 1 mm blocks, chosen among blocks with >= 25% of the median block's molecules; alternate probe-fit / probe-test
        B = s.B = cfg.block_um; s.bnx = int(np.ceil((s.xyz[:, 0].max() - s.lo[0] + 1) / B)); s.bny = int(np.ceil((s.xyz[:, 1].max() - s.lo[1] + 1) / B))
        cnt = np.bincount(s._block(s.xyz[:, 0], s.xyz[:, 1]), minlength=s.bnx * s.bny); full = np.flatnonzero(cnt >= 0.25 * np.median(cnt[cnt > 0]))
        rng = np.random.default_rng(1000 + cfg.split_seed); perm = rng.permutation(full); held, tot = [], 0
        for b in perm:
            if tot >= cfg.heldout_frac * cnt.sum() and held: break
            held.append(b); tot += cnt[b]
        s.role = np.zeros(s.bnx * s.bny, np.int8); s.role[held[0::2]] = 1; s.role[held[1::2]] = 2   # 0 train, 1 probe-fit, 2 probe-test
        mb = s.role[s._block(s.xyz[:, 0], s.xyz[:, 1])]
        s.split = dict(n=len(s.xyz), train=int((mb == 0).sum()), probe_fit=int((mb == 1).sum()), probe_test=int((mb == 2).sum()),
                       blocks_train=int(((s.role == 0) & (cnt > 0)).sum()), blocks_fit=len(held[0::2]), blocks_test=len(held[1::2]), load_sec=round(time.time() - t, 1))
        s.ntr = np.bincount(s.gene[mb == 0], minlength=s.G)

    def _tile(s, x, y):
        tx = np.clip(((x - s.lo[0]) // s.TILE).astype(np.int64), 0, s.nx - 1); ty = np.clip(((y - s.lo[1]) // s.TILE).astype(np.int64), 0, s.ny - 1)
        return ty * s.nx + tx

    def _block(s, x, y):
        B = s.B
        bx = np.clip(((np.asarray(x) - s.lo[0]) // B).astype(np.int64), 0, s.bnx - 1); by = np.clip(((np.asarray(y) - s.lo[1]) // B).astype(np.int64), 0, s.bny - 1)
        return by * s.bnx + bx

    def window(s, x0, y0, w):
        tx0, tx1 = int((x0 - s.lo[0]) // s.TILE), int((x0 + w - s.lo[0]) // s.TILE); ty0, ty1 = int((y0 - s.lo[1]) // s.TILE), int((y0 + w - s.lo[1]) // s.TILE)
        tx0, ty0 = max(tx0, 0), max(ty0, 0); tx1, ty1 = min(tx1, s.nx - 1), min(ty1, s.ny - 1)
        if tx1 < tx0 or ty1 < ty0: return np.zeros(0, np.int64)
        idx = np.concatenate([np.arange(s.start[ty * s.nx + tx0], s.start[ty * s.nx + tx1 + 1]) for ty in range(ty0, ty1 + 1)])
        p = s.xyz[idx]; return idx[(p[:, 0] >= x0) & (p[:, 0] < x0 + w) & (p[:, 1] >= y0) & (p[:, 1] < y0 + w)]

    def inside(s, x0, y0, w, role):
        """True if the square lies entirely in blocks of this role (checked on a 5x5 grid of points; blocks are 1 mm >> 140 um)."""
        g = np.linspace(0, w, 5); X, Y = np.meshgrid(x0 + g, y0 + g)
        return bool((s.role[s._block(X.ravel().clip(None, x0 + w - 1e-3), Y.ravel().clip(None, y0 + w - 1e-3))] == role).all())

    def sample_region(s, rng, cfg):
        """Two overlapping view squares inside training blocks with >= min_mol molecules each."""
        W, sh = cfg.win_um, cfg.shift_frac * cfg.win_um; ext = s.lo + np.array([s.nx, s.ny]) * s.TILE
        for _ in range(200):
            x0, y0 = rng.uniform(s.lo[0], ext[0] - W - sh), rng.uniform(s.lo[1], ext[1] - W - sh)
            if not s.inside(x0, y0, W + sh, 0): continue
            ia = s.window(x0, y0, W)
            if len(ia) < cfg.min_mol: continue
            dx, dy = rng.uniform(0, sh, 2); ib = s.window(x0 + dx, y0 + dy, W)
            if len(ib) < cfg.min_mol: continue
            return ia, ib
        raise RuntimeError("no training window found (crop too small or all held out?)")

    def eval_tiles(s, role, n, W=100.0, halo=15.0):
        """Up to n held-out tiles of this role: (sel, core) with core = molecules >= halo from the edge. Deterministic."""
        out = []
        for b in np.flatnonzero(s.role == role):
            bx, by = b % s.bnx, b // s.bnx; X0, Y0 = s.lo[0] + bx * s.B, s.lo[1] + by * s.B
            for y0 in np.arange(Y0, Y0 + s.B - W + 1, W - 2 * halo):
                for x0 in np.arange(X0, X0 + s.B - W + 1, W - 2 * halo):
                    sel = s.window(x0, y0, W)
                    if len(sel) < 500: continue
                    p = s.xyz[sel]; core = (p[:, 0] >= x0 + halo) & (p[:, 0] < x0 + W - halo) & (p[:, 1] >= y0 + halo) & (p[:, 1] < y0 + W - halo)
                    out.append((sel, core, x0, y0))
        if len(out) > n: out = [out[i] for i in np.linspace(0, len(out) - 1, n).round().astype(int)]
        return out


def make_view(M, idx, rng, cfg, augment=True, mask=True, mask_override=None):
    """One view: coords centred, rotated about z, flipped, scaled, jittered (augment); unique 0.5 um voxels; patch mask (mask)."""
    p = M.xyz[idx].astype(np.float64); c = p.copy(); c[:, :2] -= p[:, :2].mean(0); c[:, 2] -= M.zmin
    if augment:
        th = rng.uniform(0, 2 * np.pi); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        c[:, :2] = c[:, :2] @ R.T
        if rng.random() < 0.5: c[:, 0] = -c[:, 0]
        c *= rng.uniform(*cfg.scale_range); c += rng.normal(0, cfg.jitter_um, c.shape)
    c = c.astype(np.float32); gc = np.floor((c - c.min(0)) / cfg.ugrid_um).astype(np.int64)
    key = (gc[:, 0] << 42) + (gc[:, 1] << 21) + gc[:, 2]; o = np.argsort(key, kind="stable"); k = key[o]   # bump collisions along z
    pos = np.arange(len(k)); start = np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], pos, 0)); gc[o, 2] += pos - start
    for _ in range(100):
        _, first, cnt = np.unique(gc, axis=0, return_index=True, return_counts=True)
        if (cnt == 1).all(): break
        dup = np.ones(len(gc), bool); dup[first] = False; gc[dup, 2] += 1
    if mask_override is not None: m = mask_override
    elif mask:
        P = cfg.mask_patch_um; pk = np.floor(p[:, :2] / P).astype(np.int64); pk = pk[:, 0] * 1_000_003 + pk[:, 1]   # patches in tissue coords
        u, inv = np.unique(pk, return_inverse=True); m = (rng.random(len(u)) < cfg.mask_ratio)[inv]
    else: m = np.zeros(len(idx), bool)
    return dict(idx=idx, coord=c, grid=gc, gene=M.gene[idx], mask=m, coord_n=np.c_[c[:, :2] / cfg.win_um, c[:, 2:3] / 10.0].astype(np.float32))
