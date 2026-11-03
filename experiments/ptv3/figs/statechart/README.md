# Training-loop statechart

`TrainLoop.tla` models the `train.py` loop before the checkpoint-guard fix. There, the feature guard ran only on log
steps, so a checkpoint (on time, every `ckpt_every` steps, at pause, or at the end) could hold blown-up weights. TLC
finds a 7-step path to that state (`loop_before_fix.svg`). `TrainLoopFixed.tla` adds the check before every checkpoint
write, and TLC finds that all three rules hold: no blown-up checkpoint, a pause is always resumable, no deadlock
(`loop_guard_fix.svg`).

Regenerate with the pstack `statechart` skill (TLC), then render it with Pele (`pele@1.9.0`) in the notebook's light/dark look:

```bash
statechart TrainLoopFixed.tla --config TrainLoopFixed.cfg --view view.ts --out outfix
node render_chart.mjs outfix/chart.mmd loop_guard_fix.svg "bento3 training loop — with the guard fix"
```
