# Model architecture chart

`model_architecture.svg` shows the path one batch takes through `bento3_ptv3` (`objective="recon"`), and the eval path:
which data each stage sees, which parts are trained (stem, LoRA on attention `qkv`, neck, recon head) and which are frozen
(the pretrained Utonia encoder weights), and where the 10x labels enter (only the eval probes).

`Arch.tla` is a TLA+ model of that path. TLC checks five rules on it: labels appear only in the eval probes, the removed
molecules' genes never reach the student's input, the pretrained encoder weights are never updated, held-out blocks are
never trained on, and the eval path never updates the model. These hold on the model; the code enforces them in
`data.make_recon_view` (removed molecules), `model.UtoniaLoRA.setup_trainable` (frozen weights) and `data.Molecules`
(train / probe-fit / probe-test blocks).

Regenerate with the pstack `statechart` skill (TLC), then render with Pele (`pele@1.9.0`) in the notebook's light/dark look:

```bash
statechart Arch.tla --config Arch.cfg --view view.ts --out out
node render_chart.mjs out/chart.mmd model_architecture.svg
```
