# Model architecture

`model_architecture.png` (and `.svg`) is a drawn schematic of `bento3_ptv3` with `objective="recon"`: what each part sees and
learns. The training row runs from a 100 µm window of molecules, through masking (4–5 µm squares removed, one query point
each), the stem, the PTv3 encoder (pretrained weights frozen, LoRA on `qkv` trained) and the per-level neck, to the recon head
and its log1p-count loss. The eval row runs from the held-out blocks, through the EMA teacher, to the linear probes, which are
the only place the 10x labels enter. Regenerate it with `python model_architecture_fig.py`.

`checked_flow.svg` is the same path as a state chart from `Arch.tla`. TLC checks five rules on that model: labels appear only
in the eval probes, the removed molecules' genes never reach the student's input, the pretrained encoder weights are never
updated, held-out blocks are never trained on, and the eval path never updates the model. These hold on the model; the code
enforces them in `data.make_recon_view` (removed molecules), `model.UtoniaLoRA.setup_trainable` (frozen weights) and
`data.Molecules` (train / probe-fit / probe-test blocks). Regenerate it with the pstack `statechart` skill (TLC), then Pele
(`pele@1.9.0`):

```bash
statechart Arch.tla --config Arch.cfg --view view.ts --out out
node render_chart.mjs out/chart.mmd checked_flow.svg "bento3_ptv3 training and eval flow (TLC-checked)"
```
