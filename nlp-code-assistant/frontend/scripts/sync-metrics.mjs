// Copies ../results/model_comparison.json into src/data so Vite can import it statically.
import { copyFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = resolve(here, "../../results/model_comparison.json");
const target = resolve(here, "../src/data/model_comparison.json");

mkdirSync(dirname(target), { recursive: true });
if (existsSync(source)) {
  copyFileSync(source, target);
  console.log(`sync-metrics: copied ${source}`);
} else if (!existsSync(target)) {
  writeFileSync(target, JSON.stringify({ models: [] }, null, 2));
  console.warn(`sync-metrics: ${source} not found; wrote an empty placeholder`);
} else {
  console.warn(`sync-metrics: ${source} not found; keeping existing copy`);
}
