---- MODULE TrainLoopFixed ----
EXTENDS Naturals
CONSTANTS N, LOG          \* total steps, log/guard every LOG steps
VARIABLES phase, step, sick, hasCkpt, ckptSick
vars == <<phase, step, sick, hasCkpt, ckptSick>>

Init == phase = "start" /\ step = 0 /\ sick = FALSE /\ hasCkpt = FALSE /\ ckptSick = FALSE

Start == phase = "start" /\ phase' = "ready" /\ UNCHANGED <<step, sick, hasCkpt, ckptSick>>
\* forward + backward + AdamW + EMA; an update may push a feature level past the guard limit
Update == phase = "ready" /\ phase' = "updated" /\ sick' \in {sick, TRUE} /\ UNCHANGED <<step, hasCkpt, ckptSick>>
\* every LOG steps: write log row, then the guard checks max|feature|
LogAbort == phase = "updated" /\ step % LOG = 0 /\ sick /\ phase' = "aborted" /\ UNCHANGED <<step, sick, hasCkpt, ckptSick>>
LogPass  == phase = "updated" /\ step % LOG = 0 /\ ~sick /\ phase' = "periodic" /\ step' = step + 1 /\ UNCHANGED <<sick, hasCkpt, ckptSick>>
NoLog    == phase = "updated" /\ step % LOG # 0 /\ phase' = "periodic" /\ step' = step + 1 /\ UNCHANGED <<sick, hasCkpt, ckptSick>>
\* after the step counter moves: eval / viz when due, then a checkpoint on time (15 min) or every 5000 steps
\* guard fix: running max|feature| since the last check is checked again before any checkpoint write
GuardAtSave == phase = "periodic" /\ sick /\ phase' = "aborted" /\ UNCHANGED <<step, sick, hasCkpt, ckptSick>>
Checkpoint == phase = "periodic" /\ ~sick /\ step < N /\ phase' = "ready" /\ hasCkpt' = TRUE /\ ckptSick' = sick /\ UNCHANGED <<step, sick>>
NoCheckpoint == phase = "periodic" /\ step < N /\ phase' = "ready" /\ UNCHANGED <<step, sick, hasCkpt, ckptSick>>
Pause == phase = "periodic" /\ ~sick /\ step < N /\ phase' = "paused" /\ hasCkpt' = TRUE /\ ckptSick' = sick /\ UNCHANGED <<step, sick>>
Finish == phase = "periodic" /\ ~sick /\ step = N /\ phase' = "done" /\ hasCkpt' = TRUE /\ ckptSick' = sick /\ UNCHANGED <<step, sick>>
Stopped == phase \in {"done", "paused", "aborted"} /\ UNCHANGED vars

Next == Start \/ GuardAtSave \/ Update \/ LogAbort \/ LogPass \/ NoLog \/ Checkpoint \/ NoCheckpoint \/ Pause \/ Finish \/ Stopped
Spec == Init /\ [][Next]_vars

NoBlownUpCheckpoint == ~ckptSick
PausedIsResumable == phase = "paused" => hasCkpt
====
