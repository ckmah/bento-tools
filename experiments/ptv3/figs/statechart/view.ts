const L: Record<string, string> = {
  start: "🚀 Starting", ready: "🔁 Ready for next step", updated: "🧮 Weights updated",
  periodic: "✅ Step counted", done: "🏁 Done", paused: "⏸️ Paused (resumable)", aborted: "🛑 Aborted by guard",
};
export const title = "bento3 training loop";
export const state = (v: any) => L[v.phase] + (v.sick && !["start", "aborted"].includes(v.phase) ? " · 💥 features blown up" : "");
export const group = (v: any) => (v.phase === "start" ? "setting up" : ["done", "paused"].includes(v.phase) ? "finished / resumable" : v.phase === "aborted" ? "aborted" : v.sick ? "features blown up" : "training");
export const colors = { "setting up": "tan", training: "blue", "features blown up": "orange", "finished / resumable": "teal", aborted: "pink" };
export const step = (a: string, f: any, t: any) => {
  switch (a) {
    case "Start": return "load data, fork loaders, build model, init eval";
    case "Update": return !f.sick && t.sick ? "an update blows up a feature level" : "masked step, optimizer, EMA";
    case "LogAbort": return "guard trips: abort [log step]";
    case "LogPass": return "log, guard passes [log step]";
    case "NoLog": return "no guard check [between log steps]";
    case "Checkpoint": return "save checkpoint";
    case "NoCheckpoint": return "eval / viz if due";
    case "Pause": return "session budget: checkpoint, pause";
    case "GuardAtSave": return "guard trips before save: abort, nothing saved";
    case "Finish": return "last step: checkpoint, final eval";
    default: return null;
  }
};
