import { useCallback, useEffect, useState } from "react";
import { CancelSeparation, InitialTrack, OpenAudio, SelfTest, SeparateStems } from "../wailsjs/go/main/App";
import type { main } from "../wailsjs/go/models";
import { EventsOn, Quit } from "../wailsjs/runtime/runtime";
import "./App.css";
import type { MultitrackEngine } from "./audio/engine";
import { log } from "./log";
import Player from "./player/Player";

interface Progress {
  stage: "loading-model" | "separating" | "writing";
  segment: number;
  segments: number;
  elapsedMs: number;
}

const STAGE_LABEL: Record<Progress["stage"], string> = {
  "loading-model": "Loading model…",
  separating: "Separating",
  writing: "Writing stems…",
};

export default function App() {
  const [track, setTrack] = useState<main.Track | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [busy, setBusy] = useState<"opening" | "separating" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [engine, setEngine] = useState<MultitrackEngine | null>(null);

  useEffect(() => EventsOn("separation:progress", (p: Progress) => setProgress(p)), []);

  useEffect(() => {
    InitialTrack()
      .then((t) => t && setTrack(t))
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (engine && track) void runSelfTestIfRequested(engine);
  }, [engine, track]);

  async function open() {
    setError(null);
    setNotice(null);
    setBusy("opening");
    try {
      const t = await OpenAudio();
      if (t) setTrack(t);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  async function separate() {
    setError(null);
    setProgress(null);
    setBusy("separating");
    try {
      const stems = await SeparateStems();
      setTrack((t) => (t ? ({ ...t, stems } as main.Track) : t));
    } catch (e) {
      setError(String(e));
      log("error", `separation: ${e}`);
    } finally {
      setBusy(null);
      setProgress(null);
    }
  }

  const onError = useCallback((msg: string) => setError(msg), []);
  const onNotice = useCallback((msg: string) => setNotice(msg), []);
  const pct = progress && progress.segments ? (100 * progress.segment) / progress.segments : 0;
  const stems = track?.stems ?? [];

  return (
    <main className="app">
      <header>
        <h1>Maestro</h1>
        <button onClick={open} disabled={busy !== null}>
          {busy === "opening" ? "Decoding…" : "Open song"}
        </button>
      </header>

      {error && (
        <div className="error" role="alert" onClick={() => setError(null)}>
          {error}
        </div>
      )}
      {notice && (
        <div className="notice" onClick={() => setNotice(null)}>
          {notice}
        </div>
      )}

      {track && (
        <section>
          <h2>{track.name}</h2>

          {!stems.length &&
            (busy === "separating" ? (
              <div className="progress">
                <div className="progress-label">
                  {progress ? STAGE_LABEL[progress.stage] : "Starting…"}
                  {progress?.stage === "separating" && progress.segments > 0 && ` ${progress.segment}/${progress.segments}`}
                  {progress && ` · ${(progress.elapsedMs / 1000).toFixed(0)} s`}
                </div>
                <progress max={100} value={pct} />
                <button onClick={() => CancelSeparation()}>Cancel</button>
              </div>
            ) : (
              <button className="primary" onClick={separate}>
                Separate stems
              </button>
            ))}

          <Player key={track.id} track={track} stems={stems}
            onEngine={setEngine} onError={onError} onNotice={onNotice} />
        </section>
      )}
    </main>
  );
}

/** `maestro --selftest song.mp3`: decode, play 1 s, log what happened, quit. */
async function runSelfTestIfRequested(engine: MultitrackEngine) {
  const t = performance.now();
  if (!(await SelfTest())) return;
  log("info", `selftest: SelfTest() binding took ${(performance.now() - t).toFixed(0)} ms`);
  const deadline = performance.now() + 20_000;
  while (!engine.has("mix") && performance.now() < deadline) await sleep(100);
  if (!engine.has("mix")) {
    log("error", "selftest: mix never decoded");
    return Quit();
  }
  const tp = performance.now();
  await engine.play();
  log("info", `selftest: play() resolved after ${(performance.now() - tp).toFixed(0)} ms`);
  const t0 = engine.ctx.currentTime;
  await sleep(1000);
  const advanced = engine.ctx.currentTime - t0;
  log("info", `selftest: ctx.state=${engine.ctx.state} sampleRate=${engine.ctx.sampleRate} clock advanced ${advanced.toFixed(2)} s, position ${engine.position.toFixed(2)} s`);
  engine.setState("mix", { soloed: true });
  engine.seek(5);
  await sleep(300);
  log("info", `selftest: after seek(5) position ${engine.position.toFixed(2)} s, playing=${engine.playing}`);
  engine.pause();
  log(advanced > 0.5 ? "info" : "error", `selftest: ${advanced > 0.5 ? "PASS" : "FAIL (audio clock not running)"}`);
  Quit();
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
