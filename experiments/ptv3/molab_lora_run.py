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
# ]
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # bento 3.0: pretrained PTv3 + LoRA over molecules (full Xenium section)

    One run: Utonia-small PTv3 encoder (pretrained, frozen) + LoRA adapters on attention, trained with Sonata-style
    teacher-student self-distillation on molecules only (gene + position; no DAPI). Data: the full 10x Xenium human breast
    IDC section (68.1M molecules, 280 genes). Scored by linear probes on held-out 1 mm blocks. All code is the
    `bento3_ptv3` package in this repo; this notebook only installs it, sets the config and calls `train()`.

    **Run:** GPU on (RTX Pro 6000), set `HF_TOKEN` (molab Secrets, or paste below), then **Run all**.
    Training is sized to ~9.5 h (step count measured on this GPU), checkpoints go to Hugging Face every 15 min, and the run
    pauses itself at 11.25 h. If the session ends for any reason, open a new session and **Run all**: it resumes from the
    newest checkpoint on Hugging Face.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import os, shutil, subprocess, sys
    # the package (pure Python; deps are in the header above). --no-deps keeps molab's torch untouched.
    PKG = "git+https://github.com/ckmah/bento-tools@bento3-ptv3-molab#subdirectory=experiments/ptv3/bento3_ptv3"
    _cmd = ["uv", "pip", "install", "--python", sys.executable] if shutil.which("uv") else [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run(_cmd + ["--no-deps", "--reinstall" if shutil.which("uv") else "--force-reinstall", PKG], check=True)
    return mo, os


@app.cell
def _(mo):
    hf_token_box = mo.ui.text(kind="password", label="HF write token (only if HF_TOKEN is not set in molab Secrets)", full_width=True)
    hf_token_box
    return (hf_token_box,)


@app.cell
def _(hf_token_box, mo, os):
    from bento3_ptv3 import TrainConfig, train
    if (hf_token_box.value or "").strip(): os.environ["HF_TOKEN"] = hf_token_box.value.strip()   # session memory only, never printed
    cfg = TrainConfig(
        name="lora_ssl", seed=0,           # run dir + HF path: lora/lora_ssl_s0
        lora_rank=16, lora_alpha=32.0,     # LoRA on attention qkv + proj in all 14 encoder blocks
        steps=0, train_hours=9.5,          # 0 = fit the step count to 9.5 h on this GPU (fixed after step 250, kept on resume)
        out_dir="bento3_lora/runs", data_dir="bento3_lora/data",
    )
    _url = f"https://huggingface.co/spaces/{cfg.trackio_space}"
    mo.md(f"""**HF token:** {'set' if os.environ.get('HF_TOKEN') else '**missing**: no checkpoints on HF, no dashboard'} ·
    **Dashboard:** [{_url}]({_url}) (project `{cfg.trackio_project}`, run `{cfg.run_name}`) ·
    **Checkpoints:** [hf.co/{cfg.hf_repo}/tree/main/{cfg.hf_prefix}/{cfg.run_name}](https://huggingface.co/{cfg.hf_repo}/tree/main/{cfg.hf_prefix}/{cfg.run_name})""")
    return cfg, train


@app.cell
def _(cfg, mo, train):
    result = train(cfg)
    mo.md("**Paused at the session budget.** Open a new session and Run all to continue." if result["status"] == "paused" else
          "**Done.** " + ", ".join(f"`{k}` {v:.3f}" for k, v in result.items() if k.startswith(("probe/", "base_comp20/")) and isinstance(v, float)))
    return


if __name__ == "__main__":
    app.run()
