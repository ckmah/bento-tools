const L: Record<string, string> = {
  start: "Starting", ready: "Ready for next step", updated: "Weights updated",
  periodic: "Step counted", done: "Done", paused: "Paused (resumable)", aborted: "Aborted by guard",
};
export const title = "bento3 training loop (guard fix)";
export const state = (v: any) => L[v.phase] + (v.sick && !["start", "aborted"].includes(v.phase) ? ", features blown up" : "");
export const group = (v: any) => (v.phase === "start" ? "setup" : ["done", "paused"].includes(v.phase) ? "end" : v.phase === "aborted" ? "stop" : "train");
export const colors = { setup: "tan", train: "blue", end: "teal", stop: "orange" };
export const step = (a: string, f: any, t: any) => {
  switch (a) {
    case "Start": return "load data, fork loaders, build model, restore checkpoint, init eval";
    case "Update": return !f.sick && t.sick ? "an update blows up a feature level" : "forward, backward, optimizer step, EMA teacher";
    case "LogAbort": return "guard trips: abort, no checkpoint [log step]";
    case "LogPass": return "log metrics, guard passes [log step]";
    case "NoLog": return "no log, no guard check [between log steps]";
    case "Checkpoint": return "save checkpoint [15 min / 5000 steps]";
    case "NoCheckpoint": return "eval / viz if due";
    case "Pause": return "session budget hit: checkpoint, pause";
    case "GuardAtSave": return "guard trips before checkpoint: abort, nothing saved";
    case "Finish": return "last step: checkpoint, final eval, viz";
    default: return null;
  }
};
