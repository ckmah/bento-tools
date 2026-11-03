import importlib.util, sys, types, numpy as np, time
SRC, D, P = sys.argv[1], sys.argv[2], sys.argv[3]
pkg = types.ModuleType("b3"); pkg.__path__ = [SRC]; sys.modules["b3"] = pkg
def load(n):
    s = importlib.util.spec_from_file_location(f"b3.{n}", f"{SRC}/{n}.py"); m = importlib.util.module_from_spec(s); sys.modules[f"b3.{n}"] = m; s.loader.exec_module(m); return m
cfgm, data = load("config"), load("data")
# evaluate.py imports torch; reimplement boundary_dist import-free by exec'ing just that function
src = open(f"{SRC}/evaluate.py").read(); i = src.index("def boundary_dist"); j = src.index("\ndef ", i + 10)
ns = {"np": np}; exec(src[i:j], ns); boundary_dist = ns["boundary_dist"]
cfg = cfgm.TrainConfig(data_dir=D); M = data.Molecules(D, cfg)
ts = M.eval_tiles(2, 24); sel, core, x0, y0 = max(ts, key=lambda t: len(t[0]))
dn = boundary_dist(M, sel, "nucleus"); dc = boundary_dist(M, sel, "cell")
# niche domains: 50 um bins over the section, log1p-normalised composition, k-means k=8
xy = M.xyz[:, :2]; lo = xy.min(0); b = ((xy - lo) // 50).astype(np.int64); nx, ny = b.max(0) + 1
key = b[:, 0] * ny + b[:, 1]; C = np.zeros((nx * ny, M.G), np.float32); np.add.at(C, (key, M.gene), 1)
tot = C.sum(1); keep = tot >= 60; X = np.log1p(C[keep] / tot[keep, None] * 1e3); X -= X.mean(0)
U, S, Vt = np.linalg.svd(X[np.random.default_rng(0).choice(len(X), 20000, replace=False)], full_matrices=False); Z = X @ Vt[:20].T
rng = np.random.default_rng(0); Cn = Z[rng.choice(len(Z), 8, replace=False)]
for _ in range(40):
    lab = ((Z[:, None] - Cn[None]) ** 2).sum(-1).argmin(1); Cn = np.stack([Z[lab == k].mean(0) for k in range(8)])
dom = np.full(nx * ny, -1, np.int16); dom[np.flatnonzero(keep)] = lab
np.savez_compressed(f"{P}/figdata2.npz", win_dnuc=dn, win_dcell=dc, niche=dom.reshape(nx, ny), niche_bin_um=50)
print("ok", np.isfinite(dn).mean(), nx, ny, np.bincount(lab))
