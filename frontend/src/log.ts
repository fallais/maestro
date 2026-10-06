import { LogFrontend } from "../wailsjs/go/main/App";

/** Logs to the Go side (stdout), since release builds have no devtools. */
export function log(level: "info" | "warn" | "error", msg: string) {
  (level === "error" ? console.error : console.log)(msg);
  LogFrontend(level, msg).catch(() => {});
}

export function installGlobalErrorLogging() {
  window.addEventListener("error", (e) => log("error", `${e.message} at ${e.filename}:${e.lineno}`));
  window.addEventListener("unhandledrejection", (e) => log("error", `unhandled rejection: ${String(e.reason)}`));
}
