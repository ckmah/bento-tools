import { render, PeleError } from "pele";
import { readFileSync, writeFileSync } from "node:fs";
const CSS = `<style>
svg.pele { background-color:#0f3460; background-image:
  linear-gradient(rgba(220,235,255,.10) 1px, transparent 1px), linear-gradient(90deg, rgba(220,235,255,.10) 1px, transparent 1px),
  linear-gradient(rgba(220,235,255,.05) 1px, transparent 1px), linear-gradient(90deg, rgba(220,235,255,.05) 1px, transparent 1px);
  background-size: 80px 80px, 80px 80px, 16px 16px, 16px 16px; }
.pele { --pele-bg:transparent; --pele-fg:#eaf2ff; --pele-muted:#b7cbe8; --pele-line:#dce9ff; --pele-surface:rgba(255,255,255,.03);
  --pele-surface-alt:rgba(255,255,255,.06); --pele-border:#dce9ff; --pele-accent:#ffffff; --pele-radius:2px;
  --pele-font:monospace; --pele-font-mono:monospace; }
.pele text { fill:#eaf2ff !important; font-size:11.5px !important; }
.pele rect { stroke:#dce9ff !important; stroke-width:1.2px; }
.pele .tan rect, .pele .blue rect, .pele .teal rect, .pele .orange rect { fill:rgba(255,255,255,.04) !important; }
.pele .orange rect { fill:rgba(255,255,255,.10) !important; }
.pele .teal rect { stroke-width:2.2px !important; }
.pele .breaks rect { fill:rgba(255,255,255,.16) !important; stroke:#ffffff !important; stroke-width:3px !important; stroke-dasharray:10 4; }
.pele .breaks text, .pele .mayBreak text { fill:#ffffff !important; font-weight:700; }
.pele .mayBreak rect { fill:rgba(255,255,255,.08) !important; stroke:#ffffff !important; stroke-width:2.2px !important; stroke-dasharray:2 4; }
.pele .bugPath rect { stroke:#ffffff !important; stroke-width:3.2px !important; }
.pele .pele-edge-label rect, .pele .pele-edge-labels rect { fill:#0f3460 !important; stroke:none !important; }
.pele path, .pele line, .pele polyline { stroke:#dce9ff !important; }
.pele marker path, .pele [class*=arrow] { fill:#dce9ff !important; }
</style>`;
for (const [src, out] of process.argv.slice(2).reduce((a, x, i, l) => (i % 2 ? a : [...a, [x, l[i + 1]]]), [])) {
  try {
    const r = render(readFileSync(src, "utf8"), { limit: 1_000_000, outputLimit: 8_000_000, maxEdges: 10_000, fontSize: 13, fontFamily: "monospace", fontFamilyMono: "monospace", padding: 24 });
    const M = 90; let svg = r.svg.replace(/viewBox="0 0 ([\d.]+) ([\d.]+)"/, (_, w, h) => `viewBox="${-M} 0 ${+w + 2 * M} ${h}"`).replace(/width="([\d.]+)"/, (_, w) => `width="${+w + 2 * M}"`);
    writeFileSync(out, svg.replace(/(<svg[^>]*>)/, `$1${CSS}`)); console.log("wrote", out, r.width, r.height);
  } catch (e) { console.log(e instanceof PeleError ? "PeleError" : "Error", e.message); }
}
