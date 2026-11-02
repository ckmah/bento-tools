# bento3_ptv3

Pretrained PTv3 encoder + LoRA, trained with self-distillation over single Xenium molecules. This is the first arm of
bento 3.0 Phase 2, and it's the simplest version: molecules only, no DAPI.

```bash
pip install "git+https://github.com/ckmah/bento-tools@bento3-ptv3-molab#subdirectory=experiments/ptv3/bento3_ptv3"
# or, from a clone:  pip install -e experiments/ptv3/bento3_ptv3[track]
```

```python
from bento3_ptv3 import TrainConfig, train
result = train(TrainConfig(seed=0, lora_rank=16))      # every knob is a dataclass field (config.py)
```

The same run from the command line: `bento3-ptv3 train --seed 0 --lora-rank 16`.

## What it does

| | |
|---|---|
| Data | 10x *FFPE Human Breast with Pre-designed Panel*, tissue sample 1 (IDC/DCIS), XOA 1.0.2. 68,113,826 molecules (qv ≥ 20, real genes), 280 genes (same panel as the 2-FOV set), about 0.74 cm² of tissue, median 9.2k molecules per 100 µm tile. Prebuilt arrays live at `hf.co/ckmah/bento3-ptv3-runs/data/xenium_breast_idc_v1` and are sha256-checked. If they're missing, `data.build_from_10x` rebuilds them by range-reading `transcripts.parquet` out of the public 24 GB zip. |
| Split | Random 1 mm blocks (`split_seed=0`, the same for every training seed). About 15% of molecules are held out, and those blocks alternate between probe-fit and probe-test. Training windows must lie entirely inside training blocks. |
| Backbone | Utonia-small PTv3 encoder (`hf.co/Pointcept/Utonia`, rev `9457f025`; code Apache-2.0, vendored in `_vendor/utonia`; weights CC-BY-NC-4.0). Its sparse conv runs through a pure-PyTorch shim, so it works with any CUDA. The encoder is frozen, with 285 of 287 tensors loaded. The input linear is replaced, because the pretrained one takes coords+colour+normal. |
| Trainable | LoRA (r = 16, α = 32) on attention `qkv` + `proj` in all 14 blocks: 297k, in its own optimiser group at `lora_lr` (5e-5; `lr` = 1e-3 for the rest). New stem (gene embedding 32-d + 3 coords → 36) plus the pretrained stem LayerNorm and mask token: 10k. Neck: a LayerNorm per encoder level, then 972 → 64: 64k. DINO head (batch-standardised MLP) + 1,024 prototypes: 691k (the head is dropped after training). Total 1.06M trainable vs 37.2M frozen (2.8%). |
| Objective | Sonata-style self-distillation, which matches how the backbone was pretrained. Each region gives two overlapping 100 µm views (rotation, flip, ±10% scale, 0.1 µm jitter). The student sees each view with 40% of 10 µm patches masked by the pretrained mask token. The EMA teacher (momentum 0.994 → 1) sees the same views unmasked. Loss: cross-entropy from the teacher's Sinkhorn-normalised prototype distribution (temperature 0.04 → 0.07) to the student's (0.1), for molecules masked in the student view, matched across the two views by molecule id. A KoLeo term (weight 0.1) spreads the student bottleneck. No labels. Watch `train/t_batch_entropy` on the dashboard: if it sits near 1.0, the targets are uniform and the run isn't learning. |
| Eval | Linear probes on frozen teacher features from held-out blocks: nucleus vs cytoplasm (AUROC), 10x graph cluster of the cell (balanced accuracy), and the masked gene (top-1, macro recall, top-1 per abundance bin). Baselines use the same molecules: own-gene one-hot and 20-nearest-neighbour gene counts. Probes run at step 0 (pretrained, untrained adapters), quickly every 10k steps, and in full at the end. The labels come from 10x segmentation and clustering and are only used for evaluation. |
| Budget | `steps=0` measures s/step over steps 50–250 and fixes the step count so training takes `train_hours` (9.5 h). The count is stored in the checkpoint. `session_hours=11.25` checkpoints, pushes and returns `status="paused"` before molab's 12 h stop. |
| Batches | `regions` regions per step (4 by default; 16 fills an RTX Pro 6000: about 418k molecules/s at 22 GB, and more regions are slower). `loader_procs` forked processes build the batches; threads don't help, because batch building holds the GIL. |
| Health guard | Each step keeps a running max of \|encoder feature\| on the GPU. It is checked at every log step and again before every checkpoint write. Above `max_feat_abs` (1e4; the pretrained encoder gives about 5), training aborts and nothing is saved. `lora_ssl_s0` blew up this way: LoRA ΔW grew as large as W, and the encoder features reached 5e10. |
| Resume | `ckpt.pt` is written every 15 min (`ckpt_min`) and pushed to `hf.co/<hf_repo>/lora/<run>/`. `train()` first pulls that checkpoint whenever it's further along than the local one, then resumes. With `loader_procs > 0`, the loaders are reseeded from the resume step, so the sample stream differs from an uninterrupted run. |
| Tracking | Per run dir: `train_log.tsv` (including `up_absmax`, `lora_ratio`, `emb_std`, `grad_norm`), `lora_blocks.jsonl` (‖ΔW‖/‖W‖ per LoRA layer), `viz/step_*.png` (genes, 10x cluster, nucleus, embedding PCA, prototype map and per-level feature size on one held-out tile, every `viz_every` steps) and `trace.jsonl` (spans in wigglystuff `TraceView` format). Trackio logs in-process. Publishing to a Space (`trackio_space`) is off by default, because its static export segfaulted the training process. If HF or Trackio fails, training carries on. |

## Files

`config.py` (TrainConfig), `data.py` (build/download, spatial index, split, views), `model.py` (Utonia + LoRA + head),
`train.py` (the loop), `evaluate.py` (probes), `viz.py` (per-step panels), `track.py` (HF push/restore, Trackio), `cli.py`, `httpzip.py`.
`../figs/statechart/` has a TLA+ model of the loop's checkpoint guard and its blueprint statechart.

## Tests

`pytest experiments/ptv3/bento3_ptv3/tests` (CPU, no weights or data): the sparse-conv shim and a tiny forward/backward under
bf16 autocast, the same dtype path the GPU run takes. `TrainConfig(amp_cpu=True)` turns on CPU bf16 autocast for a full CPU smoke run.
