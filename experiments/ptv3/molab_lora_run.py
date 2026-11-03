# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "torch==2.11.0",  # PyPI Linux wheel = CUDA 13.0 build (Blackwell sm_120 OK), same pin as molab_gpu_run.py
#     "numpy==2.5.4",
#     "scipy==1.18.1",
#     "pandas==3.1.0",
#     "pyarrow==26.0.0",
#     "timm==1.0.30",
#     "addict==2.4.0",
#     "packaging",
#     "huggingface-hub==2.4.0",
#     "trackio==0.44.0",
#     "matplotlib",
#     "wigglystuff==0.5.36",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([mo.md(r"""
    # bento 3.0: pretrained PTv3 + LoRA over molecules (full Xenium section)

    Utonia-small PTv3 encoder (pretrained, frozen) + LoRA adapters on attention, trained on molecules only (gene + position; no DAPI),
    aimed at **subcellular** embeddings. Data: the full 10x Xenium human breast IDC section (68.1M molecules, 280 genes). Default
    objective: masked reconstruction of 1–3 µm squares (count, layout and genes as log1p counts). Scored by linear probes on held-out
    1 mm blocks: nucleus vs cytoplasm, signed distance to the nucleus boundary, distance to the cell boundary. All training code is the
    `bento3_ptv3` package in this repo.

    **Use:** GPU on, `HF_TOKEN` in molab Secrets (checkpoints sync to Hugging Face), **Run all**. Runs are defined in `RUNS` below and
    train as background subprocesses, so the kernel stays free: start/stop them from the controls and watch them in the dashboard.
    A stopped or paused run resumes from its newest checkpoint (local, or on Hugging Face).
    """),
        mo.Html("<img style='width:100%;max-width:1500px' src='https://raw.githubusercontent.com/ckmah/bento-tools/bento3-ptv3-molab/experiments/ptv3/figs/training_objective.png'>"),
    ])
    return


@app.cell
def _():
    import marimo as mo
    import os, shutil, subprocess, sys
    # the package (pure Python; deps are in the header above). --no-deps keeps molab's torch untouched.
    PKG = "git+https://github.com/ckmah/bento-tools@bento3-ptv3-molab#subdirectory=experiments/ptv3/bento3_ptv3"
    _cmd = ["uv", "pip", "install", "--python", sys.executable] if shutil.which("uv") else [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run(_cmd + ["--no-deps", "--reinstall" if shutil.which("uv") else "--force-reinstall", PKG], check=True)
    return mo, os, subprocess, sys


@app.cell(hide_code=True)
def run_defs(os, subprocess, sys):
    # Background training runs: subprocesses (python -m bento3_ptv3.cli train), so this kernel stays free.
    # Each named run below can be started/stopped from the controls.
    import io, json, signal, time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    OUT = "bento3_lora/runs"
    BASE = dict(seed=0, lr=1e-3, lora_lr=5e-5, lora_alpha=16.0, regions=16, loader_procs=2, out_dir=OUT, data_dir="bento3_lora/data")
    # the one model: masked reconstruction of 4-5 um squares (count, layout, genes as log1p counts), LoRA frozen (encoder exactly
    # pretrained: LoRA proj blew the encoder up in earlier runs), sized to train_hours on this GPU
    RUNS = {"recon": dict(BASE, name="recon", objective="recon", mask_patch_um=(4.0, 5.0), freeze_lora=1, train_hours=9.5, eval_every=1000, viz_every=250)}
    run_dir = lambda name: os.path.join(OUT, f"{name}_s0")
    _arg = lambda v: ",".join(map(str, v)) if isinstance(v, (tuple, list)) else str(v)   # the CLI takes tuples as comma lists


    def run_pid(name):
        try: pid = int(open(os.path.join(run_dir(name), "pid")).read())
        except (OSError, ValueError): return None
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid: return None   # exited; reaped
        except ChildProcessError: pass                              # not this kernel's child: fall back to /proc
        try: return pid if open(f"/proc/{pid}/stat").read().split()[2] != "Z" else None
        except OSError: return None


    def launch_run(name):
        if run_pid(name): return run_pid(name)
        d = run_dir(name); os.makedirs(d, exist_ok=True)
        args = [sys.executable, "-u", "-m", "bento3_ptv3.cli", "train"] + [a for k, v in RUNS[name].items() for a in (f"--{k.replace('_', '-')}", _arg(v))]
        p = subprocess.Popen(args, cwd=os.getcwd(), stdout=open(os.path.join(d, "stdout.log"), "a"), stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONFAULTHANDLER="1"), start_new_session=True)
        open(os.path.join(d, "pid"), "w").write(str(p.pid)); return p.pid


    def stop_run(name):
        pid = run_pid(name)
        if pid: os.killpg(pid, signal.SIGTERM)
        return pid

    return (
        OUT,
        RUNS,
        io,
        json,
        launch_run,
        np,
        pd,
        plt,
        run_dir,
        run_pid,
        stop_run,
        time,
    )


@app.cell(hide_code=True)
def run_controls(RUNS, mo):
    import base64, re
    import wigglystuff as ws
    run_pick = mo.ui.dropdown(options=list(RUNS), value=list(RUNS)[0], label="run")
    start_btn = mo.ui.run_button(label="▶ start / resume", kind="success")
    stop_btn = mo.ui.run_button(label="■ stop", kind="danger")
    refresh = mo.ui.refresh(options=["10s", "30s", "1m", "5m"], default_interval="30s")
    follow = mo.ui.checkbox(value=True, label="follow latest step (untick to scrub / play)")
    mo.hstack([run_pick, start_btn, stop_btn, refresh, follow], justify="start", align="center", wrap=True)
    return base64, follow, re, refresh, run_pick, start_btn, stop_btn, ws


@app.cell(hide_code=True)
def run_actions(launch_run, run_pick, start_btn, stop_btn, stop_run, time):
    if start_btn.value: launch_run(run_pick.value)
    if stop_btn.value: stop_run(run_pick.value)
    action_t = time.time()
    return (action_t,)


@app.cell(hide_code=True)
def widgets(RUNS, mo, ws):
    metrics_w = ws.ImageRefreshWidget(src="")
    blocks_w = ws.ImageRefreshWidget(src="")
    viz_w = ws.ImageRefreshWidget(src="")
    status_w = ws.HTMLRefreshWidget(html="")
    vizinfo_w = ws.HTMLRefreshWidget(html="")
    stdout_w = ws.HTMLRefreshWidget(html="")
    player_w = ws.PlaySlider(value=0, min_value=0, max_value=1, step=1, interval_ms=600, loop=False, width=900)
    trace_w = ws.TraceView(title="training runs", height=520)
    bars_w = {n: ws.ProgressBar(value=0, max_value=1, color="#1d4ed8", height=14) for n in RUNS}
    metrics_ui, blocks_ui, viz_ui, status_ui, vizinfo_ui, stdout_ui = (mo.ui.anywidget(w) for w in (metrics_w, blocks_w, viz_w, status_w, vizinfo_w, stdout_w))
    player, trace_ui = mo.ui.anywidget(player_w), mo.ui.anywidget(trace_w)
    bars_ui = {n: mo.ui.anywidget(w) for n, w in bars_w.items()}
    return (
        bars_ui,
        bars_w,
        blocks_ui,
        blocks_w,
        metrics_ui,
        metrics_w,
        player,
        player_w,
        status_ui,
        status_w,
        stdout_ui,
        stdout_w,
        trace_ui,
        trace_w,
        viz_ui,
        viz_w,
        vizinfo_ui,
        vizinfo_w,
    )


@app.cell(hide_code=True)
def run_data(
    OUT,
    RUNS,
    action_t,
    bars_w,
    follow,
    json,
    os,
    pd,
    player_w,
    re,
    refresh,
    run_dir,
    run_pick,
    run_pid,
    status_w,
    stdout_w,
    trace_w,
):
    refresh.value, action_t   # re-read on every tick / button press
    dir_of = lambda n: os.path.join(OUT, n) if n.endswith("_s0") else run_dir(n)


    def _tsv(n):
        try: return pd.read_csv(os.path.join(dir_of(n), "train_log.tsv"), sep="\t")
        except Exception: return pd.DataFrame()


    def _jsonl(p):
        try: return [json.loads(l) for l in open(p) if l.strip()]
        except OSError: return []


    def _total(n):
        try: m = re.findall(r"-> (\d+) steps", open(os.path.join(dir_of(n), "stdout.log"), errors="replace").read())
        except OSError: m = []
        return int(m[-1]) if m else int(RUNS.get(n, {}).get("steps", 0)) or None


    sel_name = run_pick.value; sel_dir = dir_of(sel_name)
    logs = {sel_name: _tsv(sel_name)}
    evals_all = {n: _jsonl(os.path.join(dir_of(n), "evals.jsonl")) for n in logs}
    evals = evals_all[sel_name]
    vizinfo = {d["step"]: d for d in _jsonl(os.path.join(sel_dir, "viz", "viz.jsonl"))}
    blocks = pd.DataFrame(_jsonl(os.path.join(sel_dir, "lora_blocks.jsonl")))
    try: viz_steps = sorted(int(f[5:12]) for f in os.listdir(os.path.join(sel_dir, "viz")) if f.startswith("step_") and f.endswith(".png"))
    except OSError: viz_steps = []

    # progress bars + status table for every launchable run
    _rows = []
    for _n in RUNS:
        _d, _pid, _tot = _tsv(_n), run_pid(_n), _total(_n)
        _st = int(_d["step"].max()) if len(_d) else 0
        bars_w[_n].max_value, bars_w[_n].value = max(_tot or 1, 1), _st
        bars_w[_n].color = "#1d4ed8" if _pid else "#94a3b8"
        _last = _d.iloc[-1] if len(_d) else None
        _rows.append(f"<tr><td><b>{_n}</b></td><td>{'🟢 running' if _pid else '⚪ stopped'}</td><td>{_st:,} / {f'{_tot:,}' if _tot else '? (sized at step 250)'}</td>"
                     + (f"<td>{_last['s_per_step']:.3f} s/step</td><td>max|feat| {_last['up_absmax']:.3g}</td><td>LoRA ΔW/W {_last.get('lora_ratio', float('nan')):.3g}</td>" if _last is not None else "<td colspan=3></td>") + "</tr>")
    status_w.html = "<style>.rs td{padding:2px 14px 2px 0}</style><table class='rs' style='font-size:13px'>" + "".join(_rows) + "</table>"

    # trace spans (every run; the TraceView menu picks the trace)
    _sp = {}
    for _n in RUNS:
        for _d in _jsonl(os.path.join(dir_of(_n), "trace.jsonl")): _sp[_d["span_id"]] = _d
    if _sp: trace_w.add_spans(list(_sp.values()))

    try: _out = "".join(open(os.path.join(sel_dir, "stdout.log"), errors="replace").readlines()[-60:])
    except OSError: _out = "(no stdout yet)"
    stdout_w.html = "<pre style='font-size:11px;max-height:420px;overflow:auto;white-space:pre-wrap'>" + _out.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"

    player_w.max_value = max(len(viz_steps) - 1, 1)
    if follow.value: player_w.value = player_w.max_value
    return blocks, evals_all, logs, sel_dir, sel_name, viz_steps, vizinfo


@app.cell(hide_code=True)
def render(
    base64,
    blocks,
    blocks_w,
    evals_all,
    follow,
    io,
    logs,
    metrics_w,
    np,
    os,
    pd,
    player,
    plt,
    re,
    sel_dir,
    sel_name,
    viz_steps,
    viz_w,
    vizinfo,
    vizinfo_w,
):
    # re-draws the monitor images for the step the player points at (no layout re-render)
    _k = int(player.value.get("value", 0) or 0)
    sel_step = (viz_steps[-1] if follow.value else viz_steps[min(max(_k, 0), len(viz_steps) - 1)]) if viz_steps else None
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False})
    _cols = {n: ("#1d4ed8" if n == sel_name else c) for n, c in zip(logs, ["#1d4ed8", "#ea580c", "#16a34a", "#9333ea", "#64748b", "#db2777", "#0891b2"])}
    _xmax = max([float(d["step"].max()) for d in logs.values() if len(d)] + [100]) * 1.03


    def _sm(y, a=0.15):   # EMA smoothing for the noisy per-25-step training curves
        return pd.Series(y).ewm(alpha=a).mean().to_numpy()


    def _lines(ax, col, logy=False, smooth=True):
        for n, d in logs.items():
            if col in d and d[col].notna().any():
                y = d[col].to_numpy(float)
                if smooth: ax.plot(d["step"], y, color=_cols[n], lw=0.6, alpha=0.18)
                ax.plot(d["step"], _sm(y) if smooth else y, color=_cols[n], lw=2.2 if n == sel_name else 1.4, alpha=1 if n == sel_name else 0.8, label=n)
        if logy: ax.set_yscale("log")


    _fig, _ax = plt.subplots(2, 3, figsize=(13, 7.4), dpi=72); _ax = _ax.ravel()   # ~940 px: fits medium width
    # 1 the selected run's objective (other objectives' losses are not on the same scale)
    _d = logs[sel_name]
    if len(_d) and _d.get("loss_genes", pd.Series(dtype=float)).notna().any():
        for _c, _cc, _l in (("loss_count", "#0891b2", "count"), ("loss_layout", "#9333ea", "layout"), ("loss_genes", "#ea580c", "genes")):
            _ax[0].plot(_d["step"], _sm(_d[_c].to_numpy(float)), color=_cc, lw=2, label=_l)
        _ax[0].set_title(f"{sel_name}: reconstruction loss (log1p MSE)")
    elif len(_d):
        _ax[0].plot(_d["step"], _sm(_d["loss_dino"].to_numpy(float)), color="#1d4ed8", lw=2, label="DINO")
        _ax[0].set_title(f"{sel_name}: DINO loss")
    _ax[0].legend(frameon=False)
    # 2 subcellular probes (held-out), solid = selected, faded = compared; dashed = comp20 baseline
    for _n, _ev in evals_all.items():
        _ev = [e for e in _ev if "probe/nuc_auroc" in e]
        for _key, _mk in (("probe/nuc_auroc", "o"), ("probe/dnuc_r2", "s"), ("probe/dcell_r2", "^")):
            _pts = [(e["step"], e[_key]) for e in _ev if _key in e]
            if _pts: _ax[1].plot(*zip(*_pts), marker=_mk, ms=5, color=_cols[_n], lw=1.6 if _n == sel_name else 1, alpha=1 if _n == sel_name else 0.6)
    for _key, _ls in (("base_comp20/nuc_auroc", "-"), ("base_comp20/dnuc_r2", "--"), ("base_comp20/dcell_r2", ":")):
        _b = next((e[_key] for e in evals_all.get(sel_name, []) if _key in e), None)
        if _b is not None: _ax[1].axhline(_b, color="0.45", ls=_ls, lw=1)
    _ax[1].set_title("probes: ● nuc AUROC ■ nuc-dist R² ▲ cell-dist R²\n(grey = comp20 baseline)", fontsize=10)
    _ax[1].legend(handles=[plt.Line2D([], [], color=_cols[n], lw=2, label=n) for n in logs], frameon=False, fontsize=9)
    # 3-6 health, shared across runs
    _lines(_ax[2], "up_absmax", logy=True); _ax[2].axhline(1e4, color="#dc2626", ls="--", lw=1); _ax[2].set_title("max |encoder feature| (abort at 1e4)")
    _lines(_ax[3], "emb_std"); _ax[3].set_title("embedding spread (std across molecules)")
    _lines(_ax[4], "lora_ratio", logy=True); _ax[4].set_title("largest LoRA ‖ΔW‖/‖W‖ (0 when frozen)")
    for n, d in logs.items():
        if "s_per_step" in d and len(d):
            _ax[5].plot(d["step"], d["s_per_step"], color=_cols[n], lw=1.6)
            if "data_wait" in d: _ax[5].plot(d["step"], _sm(d["data_wait"].to_numpy(float)), color=_cols[n], lw=1, ls=":")
    _ax[5].set_title("s / step (solid) · waiting on data (dotted)")
    for _a in _ax:
        _a.set_xlim(0, _xmax); _a.grid(alpha=0.25)
        if sel_step is not None: _a.axvline(sel_step, color="crimson", lw=0.9, alpha=0.5)
    for _a in _ax[3:]: _a.set_xlabel("step")
    _fig.tight_layout(); _buf = io.BytesIO(); _fig.savefig(_buf, format="png"); plt.close(_fig)
    metrics_w.src = "data:image/png;base64," + base64.b64encode(_buf.getvalue()).decode()

    # per-block LoRA growth (own tab; only meaningful when LoRA trains)
    if len(blocks) and (blocks.drop(columns="step").to_numpy() > 0).any():
        _B = blocks.set_index("step").sort_index(); _B = _B.reindex(sorted(_B.columns, key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]), axis=1)
        _f2, _h = plt.subplots(figsize=(13, 6.5), dpi=72)
        _im = _h.imshow(np.log10(_B.T.values + 1e-9), aspect="auto", cmap="magma", vmin=-4, vmax=0.5, interpolation="nearest",
                        extent=[_B.index.min(), _B.index.max(), len(_B.columns) - 0.5, -0.5])
        _h.set_yticks(range(len(_B.columns)), [c.replace(".attn.", " ") for c in _B.columns], fontsize=8); _f2.colorbar(_im, ax=_h, label="log10 ‖ΔW‖/‖W‖")
        _h.set_title(f"{sel_name}: LoRA ‖ΔW‖/‖W‖ per layer"); _h.set_xlabel("step"); _f2.tight_layout()
        _b2 = io.BytesIO(); _f2.savefig(_b2, format="png"); plt.close(_f2); blocks_w.src = "data:image/png;base64," + base64.b64encode(_b2.getvalue()).decode()
    else:
        blocks_w.src = ""

    if sel_step is not None:
        from PIL import Image as _Im
        _im2 = _Im.open(os.path.join(sel_dir, "viz", f"step_{sel_step:07d}.png")); _im2 = _im2.resize((940, round(940 * _im2.height / _im2.width)), _Im.LANCZOS)
        _b3 = io.BytesIO(); _im2.save(_b3, format="png"); viz_w.src = "data:image/png;base64," + base64.b64encode(_b3.getvalue()).decode()
        _vi = vizinfo.get(sel_step, {})
        vizinfo_w.html = (f"<b>{sel_name}</b> · step <b>{sel_step:,}</b> · max|feat| <b>{_vi.get('up_absmax', float('nan')):.3g}</b>"
                          f" · emb std <b>{_vi.get('emb_std', float('nan')):.3f}</b>" + (f" · protos used <b>{_vi['protos_used']}</b>" if "protos_used" in _vi else ""))
    else:
        viz_w.src = ""; vizinfo_w.html = "<i>no viz panels yet (first at step 1, then every viz_every steps)</i>"
    return


@app.cell(hide_code=True)
def model_tree(RUNS, mo, run_pick, ws):
    # patched model structure (no weights loaded): trainable / frozen per module, LoRA placement, per-level neck
    import importlib.util as _iu, sys as _sys
    _PKG = __import__("os").path.dirname(_iu.find_spec("bento3_ptv3").origin)   # the installed package, loaded fresh (the kernel may hold an older import)
    for _k in [k for k in _sys.modules if k.startswith("bento3_patched")]: del _sys.modules[_k]   # always load the installed version afresh
    if True:
        _spec = _iu.spec_from_file_location("bento3_patched", f"{_PKG}/__init__.py", submodule_search_locations=[_PKG])
        _m = _iu.module_from_spec(_spec); _sys.modules["bento3_patched"] = _m; _spec.loader.exec_module(_m)
    _pm = _sys.modules["bento3_patched"]
    from bento3_patched.model import UtoniaLoRA as _UL
    from bento3_patched.config import TrainConfig as _TC
    model_struct = _UL(280, _TC(**{k: v for k, v in RUNS[run_pick.value].items() if k in _TC.__dataclass_fields__})); model_struct.setup_trainable()
    model_tree = mo.ui.anywidget(ws.ModuleTreeWidget(model_struct, initial_expand_depth=2))
    return (model_tree,)


@app.cell(hide_code=True)
def statechart(mo):
    import urllib.request as _ur
    _RAW = "https://raw.githubusercontent.com/ckmah/bento-tools/bento3-ptv3-molab/experiments/ptv3/figs/statechart"
    import base64 as _b64
    def _svg(name):   # as an <img> data URI: the chart's own <style> stays inside the image and cannot restyle the page (icons etc.)
        _d = _b64.b64encode(_ur.urlopen(f"{_RAW}/{name}", timeout=20).read()).decode()
        return mo.Html(f"<div style='overflow-x:auto'><img style='max-width:100%' src='data:image/svg+xml;base64,{_d}'></div>")
    statechart_view = mo.vstack([
        mo.md("**Training loop with the guard fix.** TLC checks all three rules: no checkpoint holds blown-up weights, a pause is always resumable, no deadlock. "
              "The running max of |feature| is checked at each log step *and* before every checkpoint write."),
        _svg("loop_guard_fix.svg"),
        mo.accordion({"Before the fix: the 7-step path to a blown-up checkpoint (dashed = breaks the rule, dotted = can break it, bold = bug path)":
                      _svg("loop_before_fix.svg")}),
    ])
    return (statechart_view,)


@app.cell(hide_code=True)
def dashboard(
    RUNS,
    bars_ui,
    blocks_ui,
    metrics_ui,
    mo,
    model_tree,
    player,
    statechart_view,
    status_ui,
    stdout_ui,
    trace_ui,
    viz_ui,
    vizinfo_ui,
):
    mo.ui.tabs({
        "Monitor": mo.vstack([
            status_ui,
            mo.vstack([mo.hstack([mo.md(f"`{n}`"), bars_ui[n]], widths=[1, 5], align="center") for n in RUNS]),
            metrics_ui,
            mo.md("**Viz panels** (held-out tile; the red line in the plots above marks this step)"),
            player, vizinfo_ui, viz_ui,
        ], gap=0.75),
        "LoRA layers": blocks_ui,
        "Trace": trace_ui,
        "Model": model_tree,
        "Loop statechart": statechart_view,
        "stdout": stdout_ui,
    })
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
