import importlib.util, sys, types, numpy as np, os, time
SRC, D, P = sys.argv[1], sys.argv[2], sys.argv[3]
pkg = types.ModuleType("b3"); pkg.__path__ = [SRC]; sys.modules["b3"] = pkg
def load(n):
    s = importlib.util.spec_from_file_location(f"b3.{n}", f"{SRC}/{n}.py"); m = importlib.util.module_from_spec(s); sys.modules[f"b3.{n}"] = m; s.loader.exec_module(m); return m
cfgm, data = load("config"), load("data")
t = time.time(); data.ensure_boundaries(D); print("bnd", round(time.time() - t, 1), flush=True)
cfg = cfgm.TrainConfig(data_dir=D); M = data.Molecules(D, cfg); print("loaded", M.split, flush=True)
xy = M.xyz[:, :2]; lo = xy.min(0)
H, _, _ = np.histogram2d(xy[:, 0] - lo[0], xy[:, 1] - lo[1], bins=[int(np.ptp(xy[:, 0]) // 20) + 1, int(np.ptp(xy[:, 1]) // 20) + 1])
ts = M.eval_tiles(2, 24); sel, core, x0, y0 = max(ts, key=lambda t: len(t[0]))
ce = M.cell[sel]; polys = {}
for kind in ("nucleus", "cell"):
    B = M.bnd[kind]; ids = np.unique(ce[ce >= 0]); j = np.clip(np.searchsorted(B["ids"], ids), 0, len(B["ids"]) - 1); j = j[B["ids"][j] == ids]
    polys[kind] = [B["xy"][B["off"][k]:B["off"][k + 1]] - np.array([x0, y0], np.float32) for k in j]
np.savez_compressed(f"{P}/figdata.npz", dens=H.astype(np.uint32), win_xyz=(M.xyz[sel] - np.array([x0, y0, 0], np.float32)).astype(np.float32),
    win_gene=M.gene[sel].astype(np.int16), win_nuc=M.nuc[sel], win_clust=M.clust[sel], win_cell=ce, genes=np.array(M.genes),
    win_origin=np.array([x0 - lo[0], y0 - lo[1]]), gene_total=np.bincount(M.gene, minlength=M.G), role=M.role, bnx=M.bnx, bny=M.bny, block_um=M.B,
    nuc_off=np.cumsum([0] + [len(a) for a in polys["nucleus"]]), nuc_poly=np.concatenate(polys["nucleus"]).astype(np.float32),
    cell_off=np.cumsum([0] + [len(a) for a in polys["cell"]]), cell_poly=np.concatenate(polys["cell"]).astype(np.float32), section_um=np.ptp(xy, 0))
print("ok", H.shape, len(sel), len(polys["nucleus"]), len(polys["cell"]), os.path.getsize(f"{P}/figdata.npz"))
