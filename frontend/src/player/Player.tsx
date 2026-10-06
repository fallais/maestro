import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import WaveSurfer from "wavesurfer.js";
import { ExportStem } from "../../wailsjs/go/main/App";
import type { main } from "../../wailsjs/go/models";
import { computePeaks, MultitrackEngine } from "../audio/engine";
import { log } from "../log";
import "./Player.css";

const STEM_ORDER = ["vocals", "guitar", "piano", "drums", "bass", "other"];
const COLORS: Record<string, string> = {
  mix: "#8a8f98",
  vocals: "#e879a6",
  drums: "#f5a524",
  bass: "#4f9cf7",
  other: "#5fcf8a",
  guitar: "#c084fc",
  piano: "#22d3ee",
};

interface Props {
  track: main.Track;
  stems: main.Stem[];
  /** Lets siblings (chord timeline, piano roll) follow the same clock. */
  onEngine?: (engine: MultitrackEngine | null) => void;
  onError: (msg: string) => void;
  onNotice: (msg: string) => void;
}

export default function Player({ track, stems: unsorted, onEngine, onError, onNotice }: Props) {
  const engine = useMemo(() => new MultitrackEngine(), []);
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [buffers, setBuffers] = useState<Record<string, AudioBuffer>>({});
  const version = useSyncExternalStore(
    (fn) => engine.subscribe(fn),
    () => engine.version,
  );

  useEffect(() => {
    onEngine?.(engine);
    return () => {
      onEngine?.(null);
      engine.dispose();
    };
  }, [engine, onEngine]);

  // Load the mix, then stems as they become available. The mix is muted once stems exist.
  const stems = useMemo(
    () => [...unsorted].sort((a, b) => STEM_ORDER.indexOf(a.name) - STEM_ORDER.indexOf(b.name)),
    [unsorted],
  );
  useEffect(() => {
    const wanted: [string, string][] = [["mix", track.mixUrl], ...stems.map((s) => [s.name, s.url] as [string, string])];
    for (const [name, url] of wanted) {
      if (engine.has(name)) continue;
      setLoading((l) => ({ ...l, [name]: true }));
      const t0 = performance.now();
      engine
        .load(name, url, { muted: name === "mix" && stems.length > 0 })
        .then((buf) => {
          log("info", `decoded ${name}: ${buf.duration.toFixed(2)} s, ${buf.numberOfChannels} ch @ ${buf.sampleRate} Hz in ${(performance.now() - t0).toFixed(0)} ms`);
          setBuffers((b) => ({ ...b, [name]: buf }));
        })
        .catch((e) => {
          log("error", `decode ${name} failed: ${e}`);
          onError(`Could not load ${name}: ${e}`);
        })
        .finally(() => setLoading((l) => ({ ...l, [name]: false })));
    }
    if (stems.length > 0) engine.setState("mix", { muted: true });
  }, [engine, track.mixUrl, stems, onError]);

  // Space toggles playback (unless typing in an input).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space" && !(e.target instanceof HTMLInputElement)) {
        e.preventDefault();
        void engine.toggle();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [engine]);

  async function exportTrack(name: string) {
    try {
      const path = await ExportStem(name);
      if (path) onNotice(`Saved ${path}`);
    } catch (e) {
      onError(String(e));
    }
  }

  const rows = ["mix", ...stems.map((s) => s.name)];
  const duration = engine.duration || track.duration;
  void version; // re-render on engine state changes

  return (
    <div className="player">
      <div className="transport">
        <button className="play" onClick={() => void engine.toggle()} disabled={!buffers.mix} aria-label={engine.playing ? "Pause" : "Play"}>
          {engine.playing ? "❚❚" : "▶"}
        </button>
        <button onClick={() => engine.seek(0)} disabled={!buffers.mix} aria-label="Back to start">⏮</button>
        <Clock engine={engine} duration={duration} />
      </div>

      <div className="tracks">
        {rows.map((name) => {
          const st = engine.state(name);
          return (
            <div className="track-row" key={name}>
              <div className="track-controls">
                <span className="track-name" style={{ color: COLORS[name] }}>{name === "mix" ? "Original mix" : name}</span>
                <div className="track-buttons">
                  <button className={st?.muted ? "on" : ""} disabled={!st} onClick={() => engine.setState(name, { muted: !st?.muted })} title="Mute">M</button>
                  <button className={st?.soloed ? "on solo" : ""} disabled={!st} onClick={() => engine.setState(name, { soloed: !st?.soloed })} title="Solo">S</button>
                  <input type="range" min={0} max={1} step={0.01} value={st?.volume ?? 1} disabled={!st}
                    onChange={(e) => engine.setState(name, { volume: Number(e.target.value) })} aria-label={`${name} volume`} />
                  <button onClick={() => exportTrack(name)} title="Export WAV">⤓</button>
                </div>
              </div>
              <div className="track-wave">
                {buffers[name] ? (
                  <Waveform buffer={buffers[name]} color={COLORS[name]} dim={!isAudible(engine, name)} onSeek={(t) => engine.seek(t)} />
                ) : (
                  <div className="wave-placeholder">{loading[name] ? "Loading…" : ""}</div>
                )}
              </div>
            </div>
          );
        })}
        <Playhead engine={engine} duration={duration} />
      </div>
    </div>
  );
}

function isAudible(engine: MultitrackEngine, name: string) {
  const st = engine.state(name);
  if (!st) return true;
  const anySolo = ["mix", ...STEM_ORDER].some((n) => engine.state(n)?.soloed);
  return anySolo ? st.soloed : !st.muted;
}

function Waveform({ buffer, color, dim, onSeek }: { buffer: AudioBuffer; color: string; dim: boolean; onSeek: (t: number) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const seekRef = useRef(onSeek);
  seekRef.current = onSeek;

  useEffect(() => {
    if (!ref.current) return;
    const ws = WaveSurfer.create({
      container: ref.current,
      height: 64,
      waveColor: color,
      progressColor: color,
      cursorWidth: 0,
      barWidth: 2,
      barGap: 1,
      normalize: false,
      interact: true,
      peaks: computePeaks(buffer, 3000),
      duration: buffer.duration,
    });
    ws.on("interaction", (t) => seekRef.current(t));
    return () => ws.destroy();
  }, [buffer, color]);

  return <div ref={ref} className={dim ? "wave dim" : "wave"} />;
}

/** One playhead line over all waveforms, moved every frame without re-rendering React. */
function Playhead({ engine, duration }: { engine: MultitrackEngine; duration: number }) {
  const line = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      if (line.current && duration > 0) line.current.style.left = `${(100 * engine.position) / duration}%`;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [engine, duration]);
  return (
    <div className="playhead-lane">
      <div ref={line} className="playhead" />
    </div>
  );
}

function Clock({ engine, duration }: { engine: MultitrackEngine; duration: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      if (ref.current) ref.current.textContent = `${fmt(engine.position)} / ${fmt(duration)}`;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [engine, duration]);
  return <span ref={ref} className="clock" />;
}

export function fmt(s: number) {
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}
