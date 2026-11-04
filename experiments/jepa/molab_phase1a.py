# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "torch==2.11.0",
#     "numpy==2.5.4",
#     "scipy==1.18.1",
#     "pandas==3.1.0",
#     "pyarrow==26.0.0",
#     "scikit-learn==1.9.1",
#     "omegaconf==2.3.0",
#     "huggingface-hub==2.4.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phase 1a: frozen CellWorld on patches

    Feeds 5 µm and 10 µm patches of molecules to the **frozen** CellWorld encoder as if each patch were a cell, and scores the
    embeddings with exactly the phase 0 probes on the frozen split (10x Xenium breast IDC, 280 genes). Nothing is trained.

    - **Patch token:** the patch's gene counts through CellWorld's tokenizer (value-weighted sum of gene embeddings, plus breast
      and Xenium embeddings). Two value scalings: `cell` (counts scaled to 10,000 per patch, then log1p, like CellWorld's cells)
      and `raw` (log1p counts).
    - **Context:** windows of 64 × 64 patches (320 µm at 5 µm, about CellWorld's 4,096 tokens), keeping each window's core.
    - **Scores:** subcellular from the molecule's own patch embedding; cell from the mean over the cell's molecules; niche from
      the mean over patches within 25 µm of the cell centroid. References: the patch's own gene counts (no model), and
      CellWorld on segmented cells (its intended input; it has seen this section).

    **Use:** GPU on, Hugging Face token below (or `HF_TOKEN` in the environment) with read access to the private
    `ckmah/bento3-ptv3-runs`, then **Run**. The run is a background process; this page refreshes from its log. Results are pushed
    to `jepa/phase1a/<scale>/` in that repo when it finishes.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import os, shutil, subprocess, sys, json, signal
    # pure-Python packages from this branch; --no-deps keeps molab's torch untouched (deps are in the header)
    _BR = "git+https://github.com/ckmah/bento-tools@bento3-ptv3-molab#subdirectory=experiments"
    _cmd = ["uv", "pip", "install", "--python", sys.executable] if shutil.which("uv") else [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run(_cmd + ["--no-deps", "--reinstall" if shutil.which("uv") else "--force-reinstall",
                           f"{_BR}/ptv3/bento3_ptv3", f"{_BR}/jepa"], check=True)
    WORK = os.path.abspath("bento_phase1a")
    os.makedirs(WORK, exist_ok=True)
    return WORK, json, mo, os, signal, subprocess, sys


@app.cell(hide_code=True)
def _(mo, os):
    token = mo.ui.text(kind="password", value=os.environ.get("HF_TOKEN", ""), label="HF token", full_width=True)
    scale = mo.ui.dropdown(["small", "base", "large", "huge"], value="base", label="CellWorld scale")
    patches = mo.ui.multiselect(["5", "10"], value=["5", "10"], label="patch (µm)")
    scalings = mo.ui.multiselect(["cell", "raw"], value=["cell", "raw"], label="value scaling")
    native = mo.ui.checkbox(value=True, label="also CellWorld on segmented cells")
    smoke = mo.ui.checkbox(value=False, label="smoke test (1 block per role)")
    run_btn = mo.ui.run_button(label="▶ run", kind="success")
    stop_btn = mo.ui.run_button(label="■ stop", kind="danger")
    refresh = mo.ui.refresh(options=["10s", "30s", "1m"], default_interval="30s")
    mo.vstack([token, mo.hstack([scale, patches, scalings], justify="start", wrap=True),
               mo.hstack([native, smoke, run_btn, stop_btn, refresh], justify="start", align="center", wrap=True)])
    return native, patches, refresh, run_btn, scale, scalings, smoke, stop_btn, token


@app.cell(hide_code=True)
def _(WORK, os, signal, subprocess, sys):
    PID = os.path.join(WORK, "pid")
    LOG = os.path.join(WORK, "run.log")


    def running():
        try: pid = int(open(PID).read())
        except (OSError, ValueError): return None
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid: return None
        except ChildProcessError: pass
        try: return pid if open(f"/proc/{pid}/stat").read().split()[2] != "Z" else None
        except OSError: return None


    def launch(scale, patches, scalings, native, smoke, token):
        if running(): return running()
        out = os.path.join(WORK, "phase1a", scale); os.makedirs(out, exist_ok=True)
        py = sys.executable; tag = "smoke" if smoke else scale
        p1a = [py, "-u", "-m", "bento_jepa.phase1a", "--paths", f"{WORK}/paths.json", "--out", f"{out}/results.json",
               "--patch", *patches, "--scaling", *scalings] + (["--native"] if native else []) + (["--max-blocks", "1"] if smoke else [])
        script = "\n".join([
            "set -e",
            f"{py} -u -m bento_jepa.hub prepare {WORK} --scale {scale}",
            " ".join(p1a),
            f"cp {LOG} {out}/run.log",
            f"{py} -u -m bento_jepa.hub push {out} --prefix jepa/phase1a/{tag}",
        ])
        env = dict(os.environ, PYTHONFAULTHANDLER="1", **({"HF_TOKEN": token} if token else {}))
        p = subprocess.Popen(["bash", "-c", script], cwd=WORK, stdout=open(LOG, "a"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
        open(PID, "w").write(str(p.pid)); return p.pid


    def stop():
        pid = running()
        if pid: os.killpg(pid, signal.SIGTERM)
        return pid

    return LOG, launch, running, stop


@app.cell(hide_code=True)
def _(launch, native, patches, run_btn, scale, scalings, smoke, stop, stop_btn, token):
    if run_btn.value: launch(scale.value, sorted(patches.value, key=float), scalings.value, native.value, smoke.value, token.value)
    if stop_btn.value: stop()
    return


@app.cell(hide_code=True)
def _(LOG, WORK, json, mo, os, refresh, running, scale, smoke):
    refresh
    import pandas as pd
    _tail = open(LOG).read().splitlines()[-25:] if os.path.exists(LOG) else ["(no run yet)"]
    _res_p = os.path.join(WORK, "phase1a", scale.value, "results.json")
    _rows = []
    if os.path.exists(_res_p):
        _r = json.load(open(_res_p)); _b = _r["baselines"]
        _row = lambda name, s=None, c=None, n=None: dict(
            run=name,
            nucleus_auroc=s and s["nucleus_auroc"], nucleus_dist_rho=s and s["nucleus_signed_dist"]["spearman"],
            cell_dist_rho=s and s["cell_boundary_dist"]["spearman"], cell_type_bacc=c and c["balanced_accuracy"],
            niche_pearson=n and n["pearson_per_cell"])
        _rows.append(_row("baseline: comp20 / own counts / 50 µm composition", _b["subcellular_comp20"], _b["cell_own_counts"],
                          _b["niche_50um_composition"]["window_composition_50um"]))
        _rows.append(_row("baseline: CellCharter niches", n=_b["niche_50um_composition"]["cellcharter_niche_onehot"]))
        for _k, _v in _r["runs"].items(): _rows.append(_row(_k, _v.get("subcellular"), _v.get("cell"), _v.get("niche")))
    _table = mo.ui.table(pd.DataFrame(_rows).round(3), selection=None) if _rows else mo.md("_no results yet_")
    mo.vstack([
        mo.md(f"**status:** {'running (pid ' + str(running()) + ')' if running() else 'idle'}"
              + (" · smoke test: scores use one block per role, so they are not comparable" if smoke.value else "")),
        mo.md("**results** (probe-test; higher is better; baselines from phase 0)"), _table,
        mo.md("**log**"), mo.plain_text("\n".join(_tail)),
    ])
    return


if __name__ == "__main__":
    app.run()
