---- MODULE Arch ----
\* One batch's path through bento3_ptv3 (objective='recon'), and the eval path. Abstracts the code in train.py / evaluate.py:
\* which data each stage sees, and which weights each update touches.
VARIABLES stage, block, seesLabels, inputHasRemovedGenes, updated
vars == <<stage, block, seesLabels, inputHasRemovedGenes, updated>>
STUDENT == {"masked", "stem", "encoder", "neck", "head", "loss"}

Init == stage = "window" /\ block = "train" /\ seesLabels = FALSE /\ inputHasRemovedGenes = FALSE /\ updated = "none"

\* ---- training step (train blocks only)
Mask    == stage = "window" /\ stage' = "masked" /\ inputHasRemovedGenes' = FALSE   \* removed molecules leave the input; their genes become targets
           /\ UNCHANGED <<block, seesLabels, updated>>
Stem    == stage = "masked"  /\ stage' = "stem"    /\ UNCHANGED <<block, seesLabels, inputHasRemovedGenes, updated>>
Encode  == stage = "stem"    /\ stage' = "encoder" /\ UNCHANGED <<block, seesLabels, inputHasRemovedGenes, updated>>
Neck    == stage = "encoder" /\ stage' = "neck"    /\ UNCHANGED <<block, seesLabels, inputHasRemovedGenes, updated>>
Head    == stage = "neck"    /\ stage' = "head"    /\ UNCHANGED <<block, seesLabels, inputHasRemovedGenes, updated>>
Loss    == stage = "head"    /\ stage' = "loss"    /\ UNCHANGED <<block, seesLabels, inputHasRemovedGenes, updated>>
Update  == stage = "loss"    /\ stage' = "update"  /\ updated' = "stem, LoRA qkv, neck, head, EMA teacher"
           /\ UNCHANGED <<block, seesLabels, inputHasRemovedGenes>>
NextStep == stage = "update" /\ stage' = "window" /\ updated' = "none" /\ UNCHANGED <<block, seesLabels, inputHasRemovedGenes>>
\* ---- eval (every eval_every steps): held-out blocks, EMA teacher, linear probes with 10x labels
EvalDue == stage = "update" /\ stage' = "heldout" /\ block' = "heldout" /\ updated' = "none" /\ UNCHANGED <<seesLabels, inputHasRemovedGenes>>
Embed   == stage = "heldout" /\ stage' = "teacher" /\ UNCHANGED <<block, seesLabels, inputHasRemovedGenes, updated>>
Probe   == stage = "teacher" /\ stage' = "probes"  /\ seesLabels' = TRUE /\ updated' = "probe weights" /\ UNCHANGED <<block, inputHasRemovedGenes>>
Resume  == stage = "probes"  /\ stage' = "window"  /\ block' = "train" /\ seesLabels' = FALSE /\ updated' = "none" /\ UNCHANGED inputHasRemovedGenes

Next == Mask \/ Stem \/ Encode \/ Neck \/ Head \/ Loss \/ Update \/ NextStep \/ EvalDue \/ Embed \/ Probe \/ Resume
Spec == Init /\ [][Next]_vars

\* Rules the reader agrees to
LabelsOnlyInEval == seesLabels => stage = "probes"
TargetsHiddenFromStudent == stage \in STUDENT => ~inputHasRemovedGenes
PretrainedWeightsFrozen == updated \in {"none", "stem, LoRA qkv, neck, head, EMA teacher", "probe weights"}   \* never the pretrained encoder weights
HeldOutNeverTrained == stage \in STUDENT \cup {"update"} => block = "train"
ProbesNeverTouchModel == stage \in {"heldout", "teacher", "probes"} => updated \in {"none", "probe weights"}
====
