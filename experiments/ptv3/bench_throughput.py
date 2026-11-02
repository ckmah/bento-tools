
import time, json, copy, numpy as np, torch, torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from bento3_ptv3.config import TrainConfig
from bento3_ptv3.data import ensure_data, Molecules
from bento3_ptv3.model import UtoniaLoRA, fetch_utonia
from bento3_ptv3.train import build_batch, sinkhorn, koleo, _amp
from bento3_ptv3.evaluate import collate
"""GPU throughput vs regions per step, and batch building serial vs threads. Run from the molab notebook dir: python bench_throughput.py"""
cfg = TrainConfig(out_dir="bento3_lora/runs", data_dir="bento3_lora/data"); dev = torch.device("cuda"); amp = _amp(cfg, dev)
M = Molecules(ensure_data(cfg), cfg)
net = UtoniaLoRA(M.G, cfg); net.load_pretrained(fetch_utonia(cfg)); net.setup_trainable(); net.to(dev)
tea = copy.deepcopy(net).eval()
for p in tea.parameters(): p.requires_grad_(False)
params = [p for p in net.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=1e-6)
log = lambda d: print(json.dumps(d), flush=True)
# batch building: serial vs threads (GIL check), regions=16
cfg.regions = 16
t = time.time(); [build_batch(M, np.random.default_rng(i), cfg) for i in range(8)]; ser = (time.time() - t) / 8
for nw in (4, 8):
    with ThreadPoolExecutor(nw) as ex:
        t = time.time(); list(ex.map(lambda i: build_batch(M, np.random.default_rng(i), cfg), range(16))); thr = (time.time() - t) / 16
    log(dict(test="build_threads", regions=16, workers=nw, serial_s_per_batch=round(ser, 3), threaded_s_per_batch=round(thr, 3), speedup=round(ser / thr, 2)))
net.train()
for R in (4, 8, 16, 32, 64):
    cfg.regions = R; rng = np.random.default_rng(0); bs_ = [build_batch(M, rng, cfg) for _ in range(6)]
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); times = []
    try:
        for S, Tv, pairs in bs_ * 2:
            torch.cuda.synchronize(); t = time.time()
            bs, bt = collate(S, dev), collate(Tv, dev); pr = torch.tensor(pairs, device=dev)
            with torch.no_grad(), torch.autocast(**amp): tl = tea.head(tea.features(bt)[1][pr[:, 1]]).float()
            tp = sinkhorn(tl, 0.05)
            with torch.autocast(**amp):
                _, e = net.features(bs); zs = net.head.mlp(e[pr[:, 0]]).float(); sl = net.head.proto(F.normalize(zs, dim=-1)).float()
            loss = -(tp * F.log_softmax(sl / 0.1, -1)).sum(-1).mean() + 0.1 * koleo(zs[:4096])
            opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize(); times.append(time.time() - t)
    except torch.cuda.OutOfMemoryError:
        log(dict(test="gpu_step", regions=R, oom=True)); break
    st = float(np.median(times[2:])); nm = float(np.mean([sum(len(v["gene"]) for v in S) for S, _, _ in bs_])); npair = float(np.mean([len(p) for *_, p in bs_]))
    log(dict(test="gpu_step", regions=R, step_s=round(st, 3), student_mol=int(nm), loss_pairs=int(npair), mol_per_s=int(nm / st), pairs_per_s=int(npair / st),
             peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1)))
