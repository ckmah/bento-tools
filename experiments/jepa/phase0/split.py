"""Phase 0: the frozen within-section split.

1 mm blocks over the section. Held-out roles are drawn once (seeded) from well-filled blocks (>= FULL x the median non-empty
block's molecule count), so every probe block is mostly tissue: 5% validation, 7.5% probe-fit, 7.5% probe-test of all non-empty
blocks; every other non-empty block is train. Train molecules within GUARD um of a held-out block are marked guard (-1) and are
never used for training or evaluation.

usage: python split.py DATA_DIR OUT_DIR   (DATA_DIR holds xyz.npy from bento3_ptv3.data.build_from_10x)
writes OUT_DIR/split.json (grid, block roles, counts) and OUT_DIR/role.npy (int8 per molecule, same order as xyz.npy:
-1 guard, 0 train, 1 validation, 2 probe-fit, 3 probe-test)
"""
import sys, os, json, hashlib, numpy as np

BLOCK, GUARD, FULL, SEED = 1000.0, 50.0, 0.7, 0
FRAC = dict(validation=0.05, probe_fit=0.075, probe_test=0.075)
ROLE = dict(guard=-1, train=0, validation=1, probe_fit=2, probe_test=3)


def main(data, out):
    xy = np.load(os.path.join(data, "xyz.npy"), mmap_mode="r")[:, :2]; xy = np.asarray(xy, np.float32)
    lo = np.floor(xy.min(0)); bnx, bny = (np.floor((xy.max(0) - lo) / BLOCK) + 1).astype(int)
    bx = ((xy[:, 0] - lo[0]) // BLOCK).astype(np.int32); by = ((xy[:, 1] - lo[1]) // BLOCK).astype(np.int32); b = by * bnx + bx
    cnt = np.bincount(b, minlength=bnx * bny); nonempty = np.flatnonzero(cnt > 0); full = np.flatnonzero(cnt >= FULL * np.median(cnt[nonempty]))
    n = len(nonempty); k = {r: int(round(f * n)) for r, f in FRAC.items()}
    perm = np.random.default_rng(SEED).permutation(full); brole = np.full(bnx * bny, -2, np.int8); brole[nonempty] = 0; i = 0
    for r in ("validation", "probe_fit", "probe_test"): brole[perm[i:i + k[r]]] = ROLE[r]; i += k[r]
    role = brole[b].copy()
    # guard band: train molecules within GUARD of any held-out block (edge neighbours by axis distance, corner neighbours by Euclidean)
    held = np.zeros((bny + 2, bnx + 2), bool); held[1:-1, 1:-1] = (brole > 0).reshape(bny, bnx)
    fx = xy[:, 0] - lo[0] - bx * BLOCK; fy = xy[:, 1] - lo[1] - by * BLOCK; gx, gy = bx + 1, by + 1; g = np.zeros(len(xy), bool)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
        ex = fx if dx < 0 else BLOCK - fx; ey = fy if dy < 0 else BLOCK - fy
        d = ex if dy == 0 else ey if dx == 0 else np.hypot(ex, ey)
        g |= held[gy + dy, gx + dx] & (d < GUARD)
    role[(role == 0) & g] = ROLE["guard"]
    os.makedirs(out, exist_ok=True); np.save(os.path.join(out, "role.npy"), role)
    mc = {r: int((role == v).sum()) for r, v in ROLE.items()}
    info = dict(block_um=BLOCK, guard_um=GUARD, full_frac=FULL, seed=SEED, origin=lo.tolist(), bnx=int(bnx), bny=int(bny),
                block_role=brole.tolist(), roles=ROLE, blocks={r: int((brole == v).sum()) for r, v in ROLE.items() if v >= 0},
                blocks_nonempty=n, molecules=mc, molecule_frac={r: round(v / len(xy), 4) for r, v in mc.items()},
                role_sha256=hashlib.sha256(open(os.path.join(out, "role.npy"), "rb").read()).hexdigest())
    json.dump(info, open(os.path.join(out, "split.json"), "w"))
    print(json.dumps({k: v for k, v in info.items() if k not in ("block_role",)}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
