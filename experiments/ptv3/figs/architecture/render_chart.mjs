// Render statechart .mmd files with Pele in the marimo notebook's look (light, with a dark variant). Run from a folder where
// `npm i pele@1.9.0` is installed:  node render_chart.mjs in.mmd out.svg [title] [in2.mmd out2.svg title2 ...]
import { render, PeleError } from "pele";
import { readFileSync, writeFileSync } from "node:fs";
const CSS = `<style>
svg.pele { background:#ffffff; border-radius:10px; }
.pele { --pele-bg:#ffffff; --pele-fg:#1f2328; --pele-muted:#59636e; --pele-line:#8c959f; --pele-surface:#f6f8fa; --pele-surface-alt:#eef1f4;
  --pele-border:#d0d7de; --pele-accent:#2f6fdf; --pele-radius:10px;
  --pele-font:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Apple Color Emoji","Segoe UI Emoji",sans-serif; }
.pele text { font-family:var(--pele-font) !important; }
.pele .pele-edge-label rect, .pele .pele-edge-labels rect { fill:#ffffff !important; stroke:none !important; opacity:.92; }
.pele .tan rect    { fill:#f3ece4 !important; stroke:#c9b29a !important; }
.pele .blue rect   { fill:#e8f0fe !important; stroke:#7aa5f5 !important; }
.pele .orange rect { fill:#fff1e5 !important; stroke:#f0a35e !important; }
.pele .teal rect   { fill:#e3f6f2 !important; stroke:#5dbdac !important; }
.pele .pink rect   { fill:#fde8ef !important; stroke:#e88aa9 !important; }
.pele .purple rect { fill:#f1ebfd !important; stroke:#a585ef !important; }
.pele .pele-node rect { stroke-width:1.6px; filter:drop-shadow(0 1px 1.5px rgba(31,35,40,.12)); }
.pele .breaks rect   { fill:#ffecee !important; stroke:#e5484d !important; stroke-width:2.6px !important; }
.pele .mayBreak rect { stroke:#e8730c !important; stroke-width:2.4px !important; stroke-dasharray:6 3; }
.pele .bugPath rect  { stroke:#2f6fdf !important; stroke-width:2.8px !important; }
@media (prefers-color-scheme: dark) {
  svg.pele { background:#181c1f; }
  .pele { --pele-bg:#181c1f; --pele-fg:#e6edf3; --pele-muted:#9ba7b4; --pele-line:#6e7681; --pele-surface:#22272e; --pele-surface-alt:#2d333b; --pele-border:#444c56; }
  .pele .pele-edge-label rect, .pele .pele-edge-labels rect { fill:#181c1f !important; }
  .pele .tan rect { fill:#3a3127 !important; } .pele .blue rect { fill:#1d2b45 !important; } .pele .orange rect { fill:#3d2a18 !important; }
  .pele .teal rect { fill:#173a35 !important; } .pele .pink rect { fill:#3d1f2a !important; } .pele .purple rect { fill:#2b2245 !important; } .pele .breaks rect { fill:#4a1d22 !important; }
}
</style>`;
const a = process.argv.slice(2);
for (let i = 0; i < a.length; i += 3) {
  const [src, out, title] = [a[i], a[i + 1], a[i + 2]];
  try {
    let mmd = readFileSync(src, "utf8"); if (title) mmd = mmd.replace(/^title: .*$/m, `title: "${title}"`);
    const r = render(mmd, { limit: 1_000_000, outputLimit: 8_000_000, maxEdges: 10_000, fontSize: 13, padding: 24 });
    const M = 60; const svg = r.svg.replace(/viewBox="0 0 ([\d.]+) ([\d.]+)"/, (_, w, h) => `viewBox="${-M} 0 ${+w + 2 * M} ${h}"`).replace(/width="([\d.]+)"/, (_, w) => `width="${+w + 2 * M}"`);
    writeFileSync(out, svg.replace(/(<svg[^>]*>)/, `$1${CSS}`)); console.log("wrote", out, r.width, r.height);
  } catch (e) { console.log(e instanceof PeleError ? "PeleError" : "Error", e.message); }
}
