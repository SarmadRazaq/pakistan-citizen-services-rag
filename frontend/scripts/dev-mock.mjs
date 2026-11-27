// `npm run dev:mock`: next dev with NEXT_PUBLIC_USE_MOCKS=1, on any OS (no cross-env needed).
// Pass extra args through, e.g. `npm run dev:mock -- --port 3001`.
import { spawn } from "node:child_process";

const child = spawn("npx", ["next", "dev", ...process.argv.slice(2)], {
  stdio: "inherit",
  shell: process.platform === "win32",
  env: { ...process.env, NEXT_PUBLIC_USE_MOCKS: "1" },
});
child.on("exit", (code) => process.exit(code ?? 0));
