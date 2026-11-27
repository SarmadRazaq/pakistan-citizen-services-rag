// Regenerate the API types without a running backend:
//   1. build the OpenAPI document straight from the FastAPI app factory (repo-root .venv),
//   2. write it to frontend/openapi.json,
//   3. run openapi-typescript on it -> src/lib/api-types.ts.
// Usage: `npm run gen:api:offline` (needs the Python venv from docs/WORK_PACKAGES.md).
import { spawnSync } from "node:child_process";
import { existsSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const frontendDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(frontendDir, "..");

const candidates = [
  process.env.PYTHON,
  join(repoRoot, ".venv", "Scripts", "python.exe"),
  join(repoRoot, ".venv", "bin", "python"),
].filter(Boolean);
const python = candidates.find((p) => existsSync(p));
if (!python) {
  console.error("No Python venv found at ../.venv. Create it first (see docs/WORK_PACKAGES.md) or set PYTHON.");
  process.exit(1);
}

const code = "import json; from app.main import create_app; print(json.dumps(create_app().openapi()))";
const py = spawnSync(python, ["-c", code], { cwd: repoRoot, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
if (py.status !== 0) {
  console.error(py.stderr);
  process.exit(py.status ?? 1);
}
// Keep only the JSON document, in case logging configuration printed anything before it.
const spec = JSON.parse(py.stdout.slice(py.stdout.indexOf("{")));
writeFileSync(join(frontendDir, "openapi.json"), JSON.stringify(spec, null, 2) + "\n");
console.log(`openapi.json written (${Object.keys(spec.paths).length} paths)`);

const ts = spawnSync("npx", ["openapi-typescript", "openapi.json", "-o", "src/lib/api-types.ts"], {
  cwd: frontendDir,
  stdio: "inherit",
  shell: process.platform === "win32",
});
process.exit(ts.status ?? 1);
