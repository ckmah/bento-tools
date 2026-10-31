"""train(cfg): plain PyTorch loop. Sonata-style self-distillation over molecules with a frozen pretrained Utonia-small encoder + LoRA.

Per step, for each of cfg.regions training regions: two overlapping view squares A and B (each rotated, flipped, scaled, jittered).
The student sees A and B with ~40% of 10 um patches masked (pretrained Utonia mask token); the EMA teacher sees them unmasked.
For molecules present in both views, the student's prototype distribution at a masked molecule in A must match the teacher's
(centred, sharpened) distribution for the same molecule in B, and vice versa. No labels are used.

Resume: run dir has ckpt.pt (trainable weights, EMA teacher, centre, optimiser, RNG states, step, total). train() first pulls the
run's newest checkpoint from the Hub when it is further along than the local one, then continues. A session that is cut off loses
at most cfg.ckpt_min minutes. train() stops by itself after cfg.session_hours (checkpoint + push) and returns status 'paused'."""
import os, sys, json, time, copy, math, queue, threading, numpy as np, torch, torch.nn.functional as F
from .config import TrainConfig
from .data import ensure_data, Molecules, make_view
from .model import UtoniaLoRA, fetch_utonia, param_report
from .evaluate import collate, evaluate
from .track import Hub, Tracker


def _amp(cfg, dev):
    return dict(device_type="cuda" if dev.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=cfg.amp == "bf16" and dev.type == "cuda")


def build_batch(M, rng, cfg):
    """Student views (masked), teacher views (same coords, unmasked) and the matched pairs (student row -> teacher row)."""
    S, Tv, pairs, off = [], [], [], 0
    for _ in range(cfg.regions):
        ia, ib = M.sample_region(rng, cfg)
        va, vb = make_view(M, ia, rng, cfg), make_view(M, ib, rng, cfg)
        _, pa, pb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
        na = len(ia)
        for (ps, pt, vs) in [(pa, na + pb, va), (na + pb, pa, vb)]:   # student A -> teacher B, student B -> teacher A
            ms = vs["mask"][ps if vs is va else ps - na]
            k = np.flatnonzero(ms) if cfg.loss_on == "masked" else np.arange(len(ps))
            if len(k) > cfg.match_max: k = rng.choice(k, cfg.match_max, replace=False)
            pairs.append(np.c_[off + ps[k], off + pt[k], ms[k]])
        S += [va, vb]; Tv += [dict(va, mask=np.zeros_like(va["mask"])), dict(vb, mask=np.zeros_like(vb["mask"]))]; off += na + len(ib)
    return S, Tv, np.concatenate(pairs)


@torch.no_grad()
def sinkhorn(logits, temp, iters=3):
    """Sinkhorn-Knopp teacher targets (as in Sonata / DINOv2): equal mass per prototype across the batch, so the teacher cannot
    collapse to one prototype or to a uniform distribution. Returns (B, K) rows summing to 1."""
    Q = torch.exp((logits - logits.max()) / temp).t()   # K x B
    Q /= Q.sum(); K, B = Q.shape
    for _ in range(iters):
        Q /= Q.sum(1, keepdim=True); Q /= K
        Q /= Q.sum(0, keepdim=True); Q /= B
    return (Q * B).t()


def koleo(z, eps=1e-8):
    """KoLeo spreading term (DINOv2): -mean log distance to the nearest other vector, on L2-normalised student bottlenecks.
    Keeps the per-molecule outputs from collapsing onto one direction, which otherwise makes the Sinkhorn targets uniform."""
    z = F.normalize(z, dim=-1); d = torch.cdist(z, z).masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), float("inf"))
    return -torch.log(d.min(1).values + eps).mean()


def _sched(step, total, warm, lo, hi):   # cosine from lo to hi over [warm, total]
    if total is None or step < warm: return lo
    return hi + (lo - hi) * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(total - warm, 1))))


def train(cfg: TrainConfig):
    t_start = time.time()
    dev = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))
    torch.set_num_threads(cfg.threads); amp = _amp(cfg, dev)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name); os.makedirs(run_dir, exist_ok=True)
    hub = Hub(cfg, run_dir); hub.restore(); Hub.restore_trackio(run_dir)
    if os.path.exists(os.path.join(run_dir, "result.json")):
        print(f"{run_dir}: already finished (result.json)", flush=True); return dict(status="done", **json.load(open(os.path.join(run_dir, "result.json"))))
    cfg.save(os.path.join(run_dir, "config.json"))

    # ---- data + model
    M = Molecules(ensure_data(cfg), cfg); print("data:", json.dumps(M.split), flush=True)
    json.dump(M.split | dict(heldout_blocks_fit=np.flatnonzero(M.role == 1).tolist(), heldout_blocks_test=np.flatnonzero(M.role == 2).tolist()),
              open(os.path.join(run_dir, "split.json"), "w"), indent=1)
    torch.manual_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    net = UtoniaLoRA(M.G, cfg); init = net.load_pretrained(fetch_utonia(cfg)); init["lora_modules"] = net.setup_trainable(); net.to(dev)
    rep = param_report(net) | dict(init=init); json.dump(rep, open(os.path.join(run_dir, "params.json"), "w"), indent=1)
    print(f"params: {rep['trainable']:,} trainable / {rep['frozen']:,} frozen ({100 * rep['trainable_frac']:.2f}%); {init['lora_modules']} LoRA layers; "
          f"Utonia tensors loaded {init['loaded_tensors']}/{init['ckpt_tensors']}", flush=True)
    teacher = copy.deepcopy(net).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    tr_names = [n for n, p in net.named_parameters() if p.requires_grad]; tp = dict(teacher.named_parameters()); sp = dict(net.named_parameters())
    params = [p for p in net.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]; no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg.weight_decay), dict(params=no_decay, weight_decay=0.0)], lr=cfg.lr)
    center = torch.zeros(cfg.prototypes, device=dev)
    step, sec0, total = 0, 0.0, (cfg.steps or None)

    CK = os.path.join(run_dir, "ckpt.pt"); log_path = os.path.join(run_dir, "train_log.tsv")
    COLS = ["step", "sec", "loss", "loss_dino", "koleo", "t_entropy", "t_batch_entropy", "agree", "lr", "ema", "t_temp", "s_per_step"]
    if os.path.exists(CK):
        ck = torch.load(CK, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["net"], strict=False); teacher.load_state_dict(ck["teacher"], strict=False); opt.load_state_dict(ck["opt"])
        center.copy_(ck["center"].to(dev)); rng.bit_generator.state = ck["rng"]; torch.set_rng_state(ck["torch_rng"].cpu())
        if dev.type == "cuda" and ck.get("cuda_rng") is not None: torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        step, sec0, total = ck["step"], ck["sec"], ck["total"]
        lines = open(log_path).read().splitlines() if os.path.exists(log_path) else ["\t".join(COLS)]
        open(log_path, "w").write("\n".join([lines[0]] + [l for l in lines[1:] if int(l.split("\t")[0]) < step]) + "\n")
        print(f"resumed {run_dir} at step {step}" + (f" of {total}" if total else ""), flush=True)
    else:
        open(log_path, "w").write("\t".join(COLS) + "\n")
    log = open(log_path, "a")
    trk = Tracker(cfg, run_dir, cfg.to_dict() | dict(trainable_params=rep["trainable"], frozen_params=rep["frozen"], device=str(dev),
                                                    gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", **{f"split_{k}": v for k, v in M.split.items()}))

    def save_ckpt(rng_state, why):
        tmp = CK + ".tmp"
        torch.save(dict(net={n: p for n, p in net.state_dict().items() if n in set(tr_names)}, teacher={n: tp[n].detach() for n in tr_names},
                        opt=opt.state_dict(), center=center.cpu(), rng=rng_state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if dev.type == "cuda" else None, step=step, sec=sec0 + time.time() - t0, total=total,
                        config=cfg.to_dict()), tmp)
        os.replace(tmp, CK); log.flush(); hub.push(f"{why} step {step}", wait=why in ("pause", "final"))

    def run_eval(tag, full):
        te = time.time(); n = cfg.eval_tiles if full else max(cfg.eval_tiles // 4, 4)
        res, idx, E = evaluate(teacher, M, cfg, dev, amp, n, gene=full and cfg.probe_gene, baselines=full)
        teacher.eval(); res["eval_sec"] = time.time() - te
        with open(os.path.join(run_dir, "evals.jsonl"), "a") as f: f.write(json.dumps(dict(tag=tag, step=step, **res)) + "\n")
        trk.log({f"eval/{k}": v for k, v in res.items()}, step)
        print(f"eval {tag} @ {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in res.items() if isinstance(v, float) and ("probe" in k or "base" in k)), flush=True)
        return res, idx, E

    if step == 0 and not os.path.exists(os.path.join(run_dir, "evals.jsonl")):
        run_eval("init", full=True)   # reference: pretrained encoder, random stem, LoRA = 0

    # ---- batches come with the RNG state right after they were drawn, so a checkpoint resumes the sample stream exactly
    npre = cfg.prefetch if dev.type == "cuda" else 0
    def produce(qq, stop):
        while not stop.is_set():
            b = build_batch(M, rng, cfg); qq.put((b, rng.bit_generator.state))
    if npre:
        qq, stop = queue.Queue(npre), threading.Event(); th = threading.Thread(target=produce, args=(qq, stop), daemon=True); th.start(); next_batch = qq.get
    else: next_batch = lambda: (build_batch(M, rng, cfg), rng.bit_generator.state)

    t0 = time.time(); last_ck = time.time(); rng_state = rng.bit_generator.state; net.train(); status = "running"; t_meas = None; s_begin = step
    while total is None or step < total:
        (S, Tv, pairs), rng_state = next_batch()
        if total is None:   # auto budget: s/step over steps 50..250, then fixed (and stored in the checkpoint)
            if step == 50: t_meas = time.time()
            if step == 250:
                sps = (time.time() - t_meas) / 200; total = int(min(cfg.max_steps, max(step + 1, cfg.train_hours * 3600 / sps)))
                print(f"auto step budget: {sps:.3f} s/step -> {total} steps in {cfg.train_hours} h", flush=True)
        lr = cfg.lr * min(1.0, (step + 1) / cfg.warmup) * (_sched(step, total, cfg.warmup, 1.0, 0.0) if total else 1.0)
        for g in opt.param_groups: g["lr"] = lr
        ttemp = cfg.t_temp[0] + (cfg.t_temp[1] - cfg.t_temp[0]) * min(1.0, step / max(cfg.t_temp_warmup * (total or 1e9), 1))
        mom = _sched(step, total, 0, cfg.ema[0], cfg.ema[1])
        bs, bt = collate(S, dev), collate(Tv, dev); pr = torch.tensor(pairs, device=dev)
        with torch.no_grad(), torch.autocast(**amp):
            tl = teacher.head(teacher.features(bt)[1][pr[:, 1]]).float()
        tprob = sinkhorn(tl, ttemp) if cfg.teacher_norm == "sinkhorn" else F.softmax((tl - center) / ttemp, -1)
        with torch.autocast(**amp):
            _, e = net.features(bs); zs = net.head.mlp(e[pr[:, 0]]).float(); sl = net.head.proto(F.normalize(zs, dim=-1)).float()
        lp = F.log_softmax(sl / cfg.s_temp, -1); per = -(tprob * lp).sum(-1); loss_dino = per.mean()
        loss_kl = koleo(zs[:cfg.koleo_max]) if cfg.koleo > 0 else torch.zeros((), device=dev)
        loss = loss_dino + cfg.koleo * loss_kl
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, cfg.clip); opt.step()
        with torch.no_grad():
            center.mul_(cfg.center_momentum).add_(tl.mean(0), alpha=1 - cfg.center_momentum)
            for n in tr_names: tp[n].mul_(mom).add_(sp[n].detach(), alpha=1 - mom)
        if step % cfg.log_every == 0:
            with torch.no_grad():
                mp = tprob.mean(0); ent = float(-(mp * (mp + 1e-12).log()).sum() / math.log(cfg.prototypes))
                pent = float((-(tprob * (tprob + 1e-12).log()).sum(-1)).mean() / math.log(cfg.prototypes)); agree = float((sl.argmax(1) == tl.argmax(1)).float().mean())
            el = sec0 + time.time() - t0; sps = (time.time() - t0) / max(step - s_begin, 1)
            row = dict(step=step, sec=round(el), loss=float(loss.detach()), loss_dino=float(loss_dino.detach()), koleo=float(loss_kl.detach()), t_entropy=ent, t_batch_entropy=pent, agree=agree, lr=lr, ema=mom, t_temp=ttemp, s_per_step=sps)
            log.write("\t".join(f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]) for c in COLS) + "\n"); log.flush()
            trk.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step)
            if not math.isfinite(float(loss)): raise FloatingPointError(f"loss is {float(loss)} at step {step}")
        step += 1
        if cfg.eval_every and step % cfg.eval_every == 0 and (total is None or step < total): run_eval("quick", full=False); net.train()
        if (time.time() - last_ck > cfg.ckpt_min * 60 or step % cfg.ckpt_every == 0) and (total is None or step < total):
            save_ckpt(rng_state, "checkpoint"); last_ck = time.time()
        if time.time() - t_start > cfg.session_hours * 3600 and (total is None or step < total):
            save_ckpt(rng_state, "pause"); status = "paused"; print(f"session budget {cfg.session_hours} h reached at step {step}: paused (Run all resumes)", flush=True); break
    if npre:
        stop.set()
        while th.is_alive():
            try: qq.get_nowait()
            except queue.Empty: th.join(0.05)
    if status == "paused": trk.finish(); return dict(status="paused", step=step, total=total)
    rng.bit_generator.state = rng_state; train_sec = sec0 + time.time() - t0
    save_ckpt(rng_state, "trained")
    res, idx, E = run_eval("final", full=True)
    np.save(os.path.join(run_dir, "emb_probe_test.npy"), E.numpy()); np.save(os.path.join(run_dir, "emb_probe_test_idx.npy"), idx)
    out = dict(status="done", run=cfg.run_name, steps=step, train_hours=train_sec / 3600, sec_per_step=train_sec / max(step, 1),
               trainable_params=rep["trainable"], frozen_params=rep["frozen"], **res)
    json.dump(out, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    hub.push("final", wait=True); trk.finish(); print(json.dumps(out), flush=True); return out
