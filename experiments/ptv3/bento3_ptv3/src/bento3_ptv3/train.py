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
from . import viz


class _Trace:
    """Span dicts (wigglystuff TraceView format) appended to run_dir/trace.jsonl; the latest line per span_id wins."""
    def __init__(s, path): s.f = open(path, "a"); s.tid = os.urandom(16).hex()
    def _w(s, d): s.f.write(json.dumps(d) + "\n"); s.f.flush()
    def start(s, name, parent=None, **attrs):
        d = dict(span_id=os.urandom(8).hex(), parent_id=parent and parent["span_id"], trace_id=s.tid, name=name, start_ms=time.time() * 1e3,
                 end_ms=None, status="unset", status_message="", attributes=attrs); s._w(d); return d
    def update(s, d, **attrs): d["attributes"].update(attrs); s._w(d)
    def end(s, d, status="ok", msg="", **attrs): d.update(end_ms=time.time() * 1e3, status=status, status_message=msg); d["attributes"].update(attrs); s._w(d)


_TR, _ROOT = None, None


class _span:
    def __init__(s, name, **attrs): s.name, s.attrs = name, attrs
    def __enter__(s): s.d = _TR.start(s.name, _ROOT, **s.attrs) if _TR else None; return s.d
    def __exit__(s, et, ev, tb):
        if s.d: _TR.end(s.d, "error" if et else "ok", f"{et.__name__}: {ev}" if et else "")


def _traced(namef, fn):
    def w(*a, **k):
        with _span(namef(*a, **k)): return fn(*a, **k)
    return w


def _proc_producer(M, cfg, seed, q, stop):
    """Forked batch builder (inherits M copy-on-write; numpy only, never touches CUDA)."""
    rng = np.random.default_rng(seed)
    while not stop.is_set(): q.put(build_batch(M, rng, cfg))


def _amp(cfg, dev):
    return dict(device_type="cuda" if dev.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=cfg.amp == "bf16" and (dev.type == "cuda" or cfg.amp_cpu))


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
    """Runs _train inside a root trace span (progress.done/total, status); SIGTERM (stop button) ends it as an error, not silently."""
    global _TR, _ROOT
    run_dir = os.path.join(cfg.out_dir, cfg.run_name); os.makedirs(run_dir, exist_ok=True)
    _TR = _Trace(os.path.join(run_dir, "trace.jsonl")); _ROOT = _TR.start(f"train {cfg.run_name}", None, **{"progress.done": 0, "progress.total": cfg.steps or 0})
    try:
        import signal
        def _term(*_): raise KeyboardInterrupt("stopped (SIGTERM)")
        signal.signal(signal.SIGTERM, _term)
    except ValueError: pass   # not the main thread (e.g. called from a notebook cell)
    try: out = _train(cfg)
    except BaseException as e: _TR.end(_ROOT, "error", f"{type(e).__name__}: {e}"); raise
    _TR.end(_ROOT, "ok", result=str(out.get("status"))); return out


def _train(cfg: TrainConfig):
    t_start = time.time()
    dev = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))
    torch.set_num_threads(cfg.threads); amp = _amp(cfg, dev)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name); os.makedirs(run_dir, exist_ok=True)
    hub = Hub(cfg, run_dir); hub.restore(); Hub.restore_trackio(run_dir)
    if os.path.exists(os.path.join(run_dir, "result.json")):
        print(f"{run_dir}: already finished (result.json)", flush=True); return dict(status="done", **json.load(open(os.path.join(run_dir, "result.json"))))
    cfg.save(os.path.join(run_dir, "config.json"))

    # ---- data + model
    with _span("load data"): M = Molecules(ensure_data(cfg), cfg)
    print("data:", json.dumps(M.split), flush=True)
    json.dump(M.split | dict(heldout_blocks_fit=np.flatnonzero(M.role == 1).tolist(), heldout_blocks_test=np.flatnonzero(M.role == 2).tolist()),
              open(os.path.join(run_dir, "split.json"), "w"), indent=1)
    torch.manual_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    procs, pq_ = [], None
    if cfg.loader_procs > 0 and dev.type == "cuda":
        import multiprocessing as mp
        _ck = os.path.join(run_dir, "ckpt.pt"); step0 = torch.load(_ck, map_location="cpu", weights_only=False)["step"] if os.path.exists(_ck) else 0
        ctx = mp.get_context("fork"); pq_, pstop = ctx.Queue(cfg.prefetch), ctx.Event()
        procs = [ctx.Process(target=_proc_producer, args=(M, cfg, s, pq_, pstop), daemon=True)
                 for s in np.random.SeedSequence([cfg.seed, step0]).spawn(cfg.loader_procs)]   # resume reseeds from the step (stream differs from an uninterrupted run)
        for p in procs: p.start()
        print(f"loader: {len(procs)} batch processes (regions={cfg.regions})", flush=True)
    with _span("build model"): net = UtoniaLoRA(M.G, cfg); init = net.load_pretrained(fetch_utonia(cfg)); init["lora_modules"] = net.setup_trainable(); net.to(dev)
    rep = param_report(net) | dict(init=init); json.dump(rep, open(os.path.join(run_dir, "params.json"), "w"), indent=1)
    print(f"params: {rep['trainable']:,} trainable / {rep['frozen']:,} frozen ({100 * rep['trainable_frac']:.2f}%); {init['lora_modules']} LoRA layers; "
          f"Utonia tensors loaded {init['loaded_tensors']}/{init['ckpt_tensors']}", flush=True)
    teacher = copy.deepcopy(net).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    tr_names = [n for n, p in net.named_parameters() if p.requires_grad]; tp = dict(teacher.named_parameters()); sp = dict(net.named_parameters())
    params = [p for p in net.parameters() if p.requires_grad]
    lora_ids = {id(p) for n, p in net.named_parameters() if p.requires_grad and (n.endswith(".A") or n.endswith(".B"))}
    lora = [p for p in params if id(p) in lora_ids]; rest = [p for p in params if id(p) not in lora_ids]
    decay = [p for p in rest if p.ndim >= 2]; no_decay = [p for p in rest if p.ndim < 2]
    lora_lr = cfg.lora_lr if cfg.lora_lr > 0 else cfg.lr
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg.weight_decay, base_lr=cfg.lr), dict(params=no_decay, weight_decay=0.0, base_lr=cfg.lr),
                             dict(params=lora, weight_decay=cfg.weight_decay, base_lr=lora_lr)], lr=cfg.lr)
    lora_mods = [m for m in net.modules() if hasattr(m, "B") and hasattr(m, "base")]
    lora_names = [n for n, m in net.named_modules() if hasattr(m, "B") and hasattr(m, "base")]
    if cfg.freeze_lora:
        for p in lora: p.requires_grad_(False)
        params = [p for p in params if p.requires_grad]; opt.param_groups[2]["params"] = []
    feat_max = torch.zeros((), device=dev)   # running max|encoder feature| since the last guard check (no host sync per step)
    center = torch.zeros(cfg.prototypes, device=dev)
    step, sec0, total = 0, 0.0, (cfg.steps or None)

    CK = os.path.join(run_dir, "ckpt.pt"); log_path = os.path.join(run_dir, "train_log.tsv")
    COLS = ["step", "sec", "loss", "loss_dino", "koleo", "t_entropy", "t_batch_entropy", "agree", "lr", "ema", "t_temp", "s_per_step", "up_absmax", "lora_ratio", "emb_std", "grad_norm", "lora_lr"]
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
        fm = float(feat_max)   # updates since the last log-step guard check
        if not fm <= cfg.max_feat_abs: raise FloatingPointError(f"encoder features blew up (max|up|={fm:.3g} > {cfg.max_feat_abs:g}) before the {why} checkpoint at step {step}; not saved")
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

    run_eval = _traced(lambda tag, full: f"eval {tag} @ {step}", run_eval)
    save_ckpt = _traced(lambda r, why: f"checkpoint ({why}) @ {step}", save_ckpt)
    _viz = _traced(lambda *a: f"viz @ {a[-1]}", viz.render)
    if step == 0 and not os.path.exists(os.path.join(run_dir, "evals.jsonl")):
        run_eval("init", full=True)   # reference: pretrained encoder, random stem, LoRA = 0

    # ---- batches come with the RNG state right after they were drawn, so a checkpoint resumes the sample stream exactly
    npre = cfg.prefetch if dev.type == "cuda" and not procs else 0
    def produce(qq, stop):
        while not stop.is_set():
            b = build_batch(M, rng, cfg); qq.put((b, rng.bit_generator.state))
    if npre:
        qq, stop = queue.Queue(npre), threading.Event(); th = threading.Thread(target=produce, args=(qq, stop), daemon=True); th.start(); next_batch = qq.get
    elif procs: next_batch = lambda: (pq_.get(), rng.bit_generator.state)
    else: next_batch = lambda: (build_batch(M, rng, cfg), rng.bit_generator.state)

    t0 = time.time(); last_ck = time.time(); rng_state = rng.bit_generator.state; net.train(); status = "running"; t_meas = None; s_begin = step
    while total is None or step < total:
        (S, Tv, pairs), rng_state = next_batch()
        if total is None:   # auto budget: s/step over steps 50..250, then fixed (and stored in the checkpoint)
            if step == 50: t_meas = time.time()
            if step == 250:
                sps = (time.time() - t_meas) / 200; total = int(min(cfg.max_steps, max(step + 1, cfg.train_hours * 3600 / sps)))
                print(f"auto step budget: {sps:.3f} s/step -> {total} steps in {cfg.train_hours} h", flush=True)
        fac = min(1.0, (step + 1) / cfg.warmup) * (_sched(step, total, cfg.warmup, 1.0, 0.0) if total else 1.0); lr = cfg.lr * fac
        for g in opt.param_groups: g["lr"] = g.get("base_lr", cfg.lr) * fac
        ttemp = cfg.t_temp[0] + (cfg.t_temp[1] - cfg.t_temp[0]) * min(1.0, step / max(cfg.t_temp_warmup * (total or 1e9), 1))
        mom = _sched(step, total, 0, cfg.ema[0], cfg.ema[1])
        bs, bt = collate(S, dev), collate(Tv, dev); pr = torch.tensor(pairs, device=dev)
        with torch.no_grad(), torch.autocast(**amp):
            tl = teacher.head(teacher.features(bt)[1][pr[:, 1]]).float()
        tprob = sinkhorn(tl, ttemp) if cfg.teacher_norm == "sinkhorn" else F.softmax((tl - center) / ttemp, -1)
        with torch.autocast(**amp):
            up_s, e = net.features(bs); feat_max = torch.maximum(feat_max, up_s.detach().abs().max().float()); zs = net.head.mlp(e[pr[:, 0]]).float(); sl = net.head.proto(F.normalize(zs, dim=-1)).float()
        lp = F.log_softmax(sl / cfg.s_temp, -1); per = -(tprob * lp).sum(-1); loss_dino = per.mean()
        loss_kl = koleo(zs[:cfg.koleo_max]) if cfg.koleo > 0 else torch.zeros((), device=dev)
        loss = loss_dino + cfg.koleo * loss_kl
        opt.zero_grad(set_to_none=True); loss.backward(); gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip)); opt.step()
        with torch.no_grad():
            center.mul_(cfg.center_momentum).add_(tl.mean(0), alpha=1 - cfg.center_momentum)
            for n in tr_names: tp[n].mul_(mom).add_(sp[n].detach(), alpha=1 - mom)
        if step % cfg.log_every == 0:
            with torch.no_grad():
                mp = tprob.mean(0); ent = float(-(mp * (mp + 1e-12).log()).sum() / math.log(cfg.prototypes))
                pent = float((-(tprob * (tprob + 1e-12).log()).sum(-1)).mean() / math.log(cfg.prototypes)); agree = float((sl.argmax(1) == tl.argmax(1)).float().mean())
                up_abs = float(feat_max); feat_max.zero_(); estd = float(e.detach().float().std(0).mean())
                lratio = max((float((m.B @ m.A).norm() * m.scale / m.base.weight.norm()) for m in lora_mods), default=0.0)
            el = sec0 + time.time() - t0; sps = (time.time() - t0) / max(step - s_begin, 1)
            row = dict(step=step, sec=round(el), loss=float(loss.detach()), loss_dino=float(loss_dino.detach()), koleo=float(loss_kl.detach()), t_entropy=ent, t_batch_entropy=pent, agree=agree, lr=lr, ema=mom, t_temp=ttemp, s_per_step=sps, up_absmax=up_abs, lora_ratio=lratio, emb_std=estd, grad_norm=gnorm, lora_lr=lora_lr * fac)
            log.write("\t".join(f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]) for c in COLS) + "\n"); log.flush()
            with torch.no_grad(), open(os.path.join(run_dir, "lora_blocks.jsonl"), "a") as fb:
                fb.write(json.dumps(dict(step=step, **{n.replace("bb.enc.", ""): float((m.B @ m.A).norm() * m.scale / m.base.weight.norm()) for n, m in zip(lora_names, lora_mods)})) + "\n")
            trk.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step)
            if _TR: _TR.update(_ROOT, **{"progress.done": step, "progress.total": total or 0, "loss_dino": round(row["loss_dino"], 4), "max_feat": round(up_abs, 3), "lora_ratio": round(lratio, 4)})
            if not math.isfinite(row["loss"]): raise FloatingPointError(f"loss is {row['loss']} at step {step}")
            if not up_abs <= cfg.max_feat_abs: raise FloatingPointError(f"encoder features blew up: max|up|={up_abs:.3g} > {cfg.max_feat_abs:g} at step {step} "
                                                                        f"(lora_ratio={lratio:.3g}); lower lora_lr. Last good checkpoint kept.")
        step += 1
        if cfg.eval_every and step % cfg.eval_every == 0 and (total is None or step < total): run_eval("quick", full=False); net.train()
        if cfg.viz_every and (step == 1 or step % cfg.viz_every == 0): _viz(teacher, M, cfg, dev, amp, run_dir, step); net.train()
        if (time.time() - last_ck > cfg.ckpt_min * 60 or step % cfg.ckpt_every == 0) and (total is None or step < total):
            save_ckpt(rng_state, "checkpoint"); last_ck = time.time()
        if time.time() - t_start > cfg.session_hours * 3600 and (total is None or step < total):
            save_ckpt(rng_state, "pause"); status = "paused"; print(f"session budget {cfg.session_hours} h reached at step {step}: paused (Run all resumes)", flush=True); break
    if procs:
        pstop.set(); pq_.cancel_join_thread()
        for p in procs: p.terminate(); p.join(5)
    if npre:
        stop.set()
        while th.is_alive():
            try: qq.get_nowait()
            except queue.Empty: th.join(0.05)
    if status == "paused": trk.finish(); return dict(status="paused", step=step, total=total)
    rng.bit_generator.state = rng_state; train_sec = sec0 + time.time() - t0
    save_ckpt(rng_state, "trained")
    res, idx, E = run_eval("final", full=True)
    if cfg.viz_every: _viz(teacher, M, cfg, dev, amp, run_dir, step)
    np.save(os.path.join(run_dir, "emb_probe_test.npy"), E.numpy()); np.save(os.path.join(run_dir, "emb_probe_test_idx.npy"), idx)
    out = dict(status="done", run=cfg.run_name, steps=step, train_hours=train_sec / 3600, sec_per_step=train_sec / max(step, 1),
               trainable_params=rep["trainable"], frozen_params=rep["frozen"], **res)
    json.dump(out, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    hub.push("final", wait=True); trk.finish(); print(json.dumps(out), flush=True); return out
