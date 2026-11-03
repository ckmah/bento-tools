const L: Record<string, string> = {
  window:  "🗺️ 100 µm windows · train blocks · molecules only (gene, x, y, z)",
  masked:  "🎭 ~40% of 4–5 µm squares removed → 1 query point each",
  stem:    "🧬 Stem: gene embedding + coords → 36-d · TRAINED (lr 1e-3)",
  encoder: "🧠 PTv3 encoder (Utonia-small) · weights FROZEN + LoRA qkv TRAINED (lr 5e-5)",
  neck:    "📐 Neck: LayerNorm per level → 64-d embedding · TRAINED",
  head:    "🎯 Recon head at query points · TRAINED, dropped after",
  loss:    "📉 log1p-count MSE: count · layout · genes of the removed molecules",
  update:  "🔧 AdamW updates stem, LoRA, neck, head · EMA → teacher",
  heldout: "🧪 Held-out 1 mm blocks (probe-fit / probe-test) · unmasked",
  teacher: "👩‍🏫 EMA teacher → 972-d features · no gradients",
  probes:  "📊 Linear probes with 10x labels: nucleus, distance to nucleus / cell boundary",
};
export const title = "bento3_ptv3: what each part sees and learns";
export const state = (v: any) => L[v.stage];
export const group = (v: any) =>
  ["window", "masked", "heldout"].includes(v.stage) ? "data" :
  ["stem", "neck", "head"].includes(v.stage) ? "trained" :
  v.stage === "encoder" ? "frozen + LoRA" :
  ["loss", "update"].includes(v.stage) ? "objective / update" : "eval only";
export const colors = { data: "tan", trained: "blue", "frozen + LoRA": "purple", "objective / update": "orange", "eval only": "teal" };
export const step = (a: string, f: any, t: any) => ({
  Mask: "mask", Stem: "visible molecules + queries", Encode: "36-d per point", Neck: "972-d (5 levels)", Head: "64-d at queries",
  Loss: "predicted log1p counts", Update: "gradients", NextStep: "next batch", EvalDue: "every eval_every steps",
  Embed: "frozen copy", Probe: "fit on probe-fit, score on probe-test", Resume: "back to training",
} as Record<string, string>)[a] ?? null;
