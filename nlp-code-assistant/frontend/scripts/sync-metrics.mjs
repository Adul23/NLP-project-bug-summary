// Copies evaluation results from ../results into src/data so Vite can import them statically:
//   model_comparison.json, summarization_metrics.json (if present) and figures/*.png.
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const results = resolve(here, "../../results");
const data = resolve(here, "../src/data");
mkdirSync(data, { recursive: true });

function syncJson(name, placeholder) {
  const source = join(results, name);
  const target = join(data, name);
  if (existsSync(source)) {
    copyFileSync(source, target);
    console.log(`sync-metrics: copied ${name}`);
  } else if (!existsSync(target)) {
    writeFileSync(target, JSON.stringify(placeholder, null, 2) + "\n");
    console.warn(`sync-metrics: ${name} not found; wrote an empty placeholder`);
  } else {
    console.warn(`sync-metrics: ${name} not found; keeping existing copy`);
  }
}

syncJson("model_comparison.json", { models: [] });
// Placeholder marks "not evaluated yet" so the UI can say so instead of failing the import.
syncJson("summarization_metrics.json", { systems: {}, not_evaluated: true });

const figuresSource = join(results, "figures");
const figuresTarget = join(data, "figures");
if (existsSync(figuresSource)) {
  rmSync(figuresTarget, { recursive: true, force: true });
  mkdirSync(figuresTarget, { recursive: true });
  const pngs = readdirSync(figuresSource).filter((f) => f.endsWith(".png"));
  for (const file of pngs) copyFileSync(join(figuresSource, file), join(figuresTarget, file));
  console.log(`sync-metrics: copied ${pngs.length} figures`);
} else {
  mkdirSync(figuresTarget, { recursive: true });
  console.warn("sync-metrics: results/figures not found; no figures to show");
}
