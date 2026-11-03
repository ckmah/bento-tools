# bento3_ptv3

Pretrained PTv3 encoder + LoRA, trained with self-distillation over single Xenium molecules. This is the first arm of
bento 3.0 Phase 2, and it's the simplest version: molecules only, no DAPI.

```bash
pip install "git+https://github.com/ckmah/bento-tools@bento3-ptv3-molab#subdirectory=experiments/ptv3/bento3_ptv3"
# or, from a clone:  pip install -e experiments/ptv3/bento3_ptv3[track]
```

```python
from bento3_ptv3 import TrainConfig, train
result = train(TrainConfig(seed=0, objective="recon"))   # every knob is a dataclass field (config.py)
```

The same run from the command line: `bento3-ptv3 train --seed 0 --objective recon --mask-patch-um 4,5`.

## What it does

| | |
|---|---|
| Data | 10x *FFPE Human Breast with Pre-designed Panel*, tissue sample 1 (IDC/DCIS), XOA 1.0.2. 68,113,826 molecules (qv ≥ 20, real genes), 280 genes (same panel as the 2-FOV set), about 0.74 cm² of tissue, median 9.2k molecules per 100 µm tile. Prebuilt arrays live at `hf.co/ckmah/bento3-ptv3-runs/data/xenium_breast_idc_v1` and are sha256-checked. If they're missing, `data.build_from_10x` rebuilds them by range-reading `transcripts.parquet` out of the public 24 GB zip. |
| Split | Random 1 mm blocks (`split_seed=0`, the same for every training seed). About 15% of molecules are held out, and those blocks alternate between probe-fit and probe-test. Training windows must lie entirely inside training blocks. |
| Backbone | Utonia-small PTv3 encoder (`hf.co/Pointcept/Utonia`, rev `9457f025`; code Apache-2.0, vendored in `_vendor/utonia`; weights CC-BY-NC-4.0). Its sparse conv runs through a pure-PyTorch shim, so it works with any CUDA. The encoder is frozen, with 285 of 287 tensors loaded. The input linear is replaced, because the pretrained one takes coords+colour+normal. |
| Trainable | LoRA (r = 16) on attention `qkv` in all 14 blocks (not `proj`: LoRA on `proj` in level 3 grew until ΔW ≈ W), in its own optimiser group at `lora_lr` (5e-5; `lr` = 1e-3 for the rest; `freeze_lora=1` keeps the encoder exactly pretrained). New stem (gene embedding 32-d + 3 coords → 36) plus the pretrained stem LayerNorm and mask token: 10k. Neck: a LayerNorm per encoder level, then 972 → 64: 64k. Objective head: the reconstruction MLP (`recon`) or the DINO head + 1,024 prototypes (`dino*`); it is dropped after training. |
| Objective | Aimed at subcellular embeddings. Each region gives two overlapping 100 µm views (rotation, flip, ±10% scale, 0.1 µm jitter), with 40% of square patches masked; the patch side is drawn per view from `mask_patch_um` (4–5 µm: at about 1 molecule/µm² that is 18–27 molecules, enough for a composition target, and about a third of a cell, so the rest of the cell stays visible). **`recon` (default, Point-MAE style):** masked patches are removed and each is replaced by one query point (the pretrained mask token at the patch centre). From the visible molecules only, the query embedding predicts what the patch held, as log1p counts with an MSE loss: molecule count, count per layout bin (4 radial × 2 z, rotation-invariant) and count per gene. The log scale keeps abundant genes and dense patches from dominating. Fixed targets, so there is no teacher to collapse. **`dino`:** Sonata-style self-distillation; per masked molecule, the student matches the EMA teacher's Sinkhorn prototype distribution for the same molecule in the other view (KoLeo 0.1). **`dino_pooled`:** as `dino`, but each target is averaged over the molecule's masked patch, because the student cannot tell the molecules of one masked patch apart. No labels. |
| Eval | Linear probes on frozen teacher (EMA) features from held-out blocks. Subcellular: nucleus vs cytoplasm (AUROC), signed distance to the own cell's nucleus boundary and distance to its cell boundary (ridge regression; R², Spearman ρ; µm, clipped to ±15). Cell level: 10x graph cluster (balanced accuracy). Masked gene (top-1, macro recall, per abundance bin) at the end. Baselines on the same molecules: own-gene one-hot and `comp20` (gene counts of the 20 nearest molecules, about a 2.6 µm radius); the bar is to beat `comp20`. Constant feature columns are left unscaled; the earlier `std + 1e-5` scaling understated the one-hot and `comp20` baselines (e.g. `comp20` cluster balanced accuracy 0.15 → 0.22). Probes run at step 0, every `eval_every` steps, and in full at the end. The labels (10x segmentation polygons, read once from the public zip by `data.ensure_boundaries`, and clusters) are only used for evaluation. |
| Budget | `steps=0` measures s/step over steps 50–250 and fixes the step count so training takes `train_hours` (9.5 h). The count is stored in the checkpoint. `session_hours=11.25` checkpoints, pushes and returns `status="paused"` before molab's 12 h stop. |
| Batches | `regions` regions per step (4 by default; 16 fills an RTX Pro 6000: about 418k molecules/s at 22 GB, and more regions are slower). `loader_procs` forked processes build the batches; threads don't help, because batch building holds the GIL. Voxel de-duplication uses a scalar key, so a 16-region batch takes about 0.1 s to build; `data_wait` in the log shows time blocked on the loader. |
| Health guard | Each step keeps a running max of \|encoder feature\| on the GPU. It is checked at every log step and again before every checkpoint write. Above `max_feat_abs` (1e4; the pretrained encoder gives about 5), training aborts and nothing is saved. `lora_ssl_s0` blew up this way: LoRA ΔW grew as large as W, and the encoder features reached 5e10. |
| Resume | `ckpt.pt` is written every 15 min (`ckpt_min`) and pushed to `hf.co/<hf_repo>/lora/<run>/`. `train()` first pulls that checkpoint whenever it's further along than the local one, then resumes. With `loader_procs > 0`, the loaders are reseeded from the resume step, so the sample stream differs from an uninterrupted run. |
| Tracking | Per run dir: `train_log.tsv` (losses, `up_absmax`, `lora_ratio`, `emb_std`, `grad_norm`, `data_wait`), `lora_blocks.jsonl` (‖ΔW‖/‖W‖ per LoRA layer), `viz/step_*.png` (genes, 10x cluster, nucleus, and the embedding as PCA → RGB over the tile plus a 25 µm zoom of the PCA and of a 12-cluster k-means, on one held-out tile, every `viz_every` steps) and `trace.jsonl` (spans in wigglystuff `TraceView` format). Trackio logs in-process. Publishing to a Space (`trackio_space`) is off by default, because its static export segfaulted the training process. If HF or Trackio fails, training carries on. |

## Files

`config.py` (TrainConfig), `data.py` (build/download, spatial index, split, views), `model.py` (Utonia + LoRA + head),
`train.py` (the loop), `evaluate.py` (probes), `viz.py` (per-step panels), `track.py` (HF push/restore, Trackio), `cli.py`, `httpzip.py`.
`../figs/training_objective.png` summarises the objective (regenerate with `training_objective_fig.py`). `../figs/architecture/` charts what each part of the model sees and learns (a TLA+ model; TLC checks that labels, removed molecules and held-out blocks never reach training).

## Tests

`pytest experiments/ptv3/bento3_ptv3/tests` (CPU, no weights or data): the sparse-conv shim and a tiny forward/backward under
bf16 autocast, the same dtype path the GPU run takes. `TrainConfig(amp_cpu=True)` turns on CPU bf16 autocast for a full CPU smoke run.
