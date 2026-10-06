/**
 * Sample-synced multitrack playback on Web Audio.
 *
 * Every track is a decoded AudioBuffer; on play, one AudioBufferSourceNode per
 * track is started at the same AudioContext instant, so stems never drift.
 * (WebKitGTK cannot stream <audio> from Wails' custom scheme anyway, so media
 * elements are not an option on Linux.)
 */
export interface TrackState {
  volume: number; // 0..1
  muted: boolean;
  soloed: boolean;
}

interface Track extends TrackState {
  buffer: AudioBuffer;
  gain: GainNode;
  source: AudioBufferSourceNode | null;
}

const START_DELAY = 0.03; // s, lets all sources be scheduled before the first plays
const RAMP = 0.01; // s, click-free gain changes

export class MultitrackEngine {
  readonly ctx = new AudioContext({ sampleRate: 44100, latencyHint: "interactive" });
  private readonly tracks = new Map<string, Track>();
  private startedAt = 0; // ctx time at which `offset` was playing
  private offset = 0; // track position (s) at startedAt, or while paused
  private _playing = false;
  private listeners = new Set<() => void>();
  /** Increments on every emitted change (for useSyncExternalStore). */
  version = 0;

  get playing() {
    return this._playing;
  }

  get duration() {
    let d = 0;
    for (const t of this.tracks.values()) d = Math.max(d, t.buffer.duration);
    return d;
  }

  /** Current playback position in seconds. Cheap; call every animation frame. */
  get position() {
    if (!this._playing) return this.offset;
    return Math.min(this.duration, this.offset + Math.max(0, this.ctx.currentTime - this.startedAt));
  }

  has(name: string) {
    return this.tracks.has(name);
  }

  buffer(name: string): AudioBuffer | undefined {
    return this.tracks.get(name)?.buffer;
  }

  /** Fetches and decodes a WAV (decoding runs off the main thread). */
  async load(name: string, url: string, state: Partial<TrackState> = {}): Promise<AudioBuffer> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} loading ${name}`);
    const buffer = await this.ctx.decodeAudioData(await res.arrayBuffer());
    this.remove(name);
    const gain = this.ctx.createGain();
    gain.connect(this.ctx.destination);
    this.tracks.set(name, { buffer, gain, source: null, volume: 1, muted: false, soloed: false, ...state });
    this.applyGains();
    if (this._playing) this.restart();
    this.emit();
    return buffer;
  }

  remove(name: string) {
    const t = this.tracks.get(name);
    if (!t) return;
    this.stopSource(t);
    t.gain.disconnect();
    this.tracks.delete(name);
  }

  async play() {
    if (this._playing) return;
    await this.ctx.resume();
    if (this.offset >= this.duration) this.offset = 0;
    this._playing = true;
    this.restart();
    this.emit();
  }

  pause() {
    if (!this._playing) return;
    this.offset = this.position;
    this._playing = false;
    for (const t of this.tracks.values()) this.stopSource(t);
    this.emit();
  }

  toggle() {
    return this._playing ? this.pause() : this.play();
  }

  seek(seconds: number) {
    this.offset = Math.max(0, Math.min(this.duration, seconds));
    if (this._playing) this.restart();
    this.emit();
  }

  state(name: string): TrackState | undefined {
    const t = this.tracks.get(name);
    return t && { volume: t.volume, muted: t.muted, soloed: t.soloed };
  }

  setState(name: string, patch: Partial<TrackState>) {
    const t = this.tracks.get(name);
    if (!t) return;
    Object.assign(t, patch);
    this.applyGains();
    this.emit();
  }

  /** Subscribe to play/pause/seek/state changes (not to position ticks). */
  subscribe(fn: () => void) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  dispose() {
    for (const name of [...this.tracks.keys()]) this.remove(name);
    void this.ctx.close();
  }

  private restart() {
    for (const t of this.tracks.values()) this.stopSource(t);
    const when = this.ctx.currentTime + START_DELAY;
    const offset = this.offset;
    for (const t of this.tracks.values()) {
      if (offset >= t.buffer.duration) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = t.buffer;
      src.connect(t.gain);
      src.start(when, offset);
      t.source = src;
    }
    this.startedAt = when;
    this.offset = offset;
    this.scheduleEnd();
  }

  private endTimer = 0;
  private scheduleEnd() {
    clearTimeout(this.endTimer);
    const remaining = this.duration - this.offset + START_DELAY;
    this.endTimer = window.setTimeout(() => {
      if (this._playing && this.position >= this.duration - 0.01) {
        this.pause();
        this.offset = this.duration;
        this.emit();
      } else if (this._playing) {
        this.scheduleEnd();
      }
    }, remaining * 1000 + 20);
  }

  private stopSource(t: Track) {
    if (!t.source) return;
    try {
      t.source.stop();
    } catch {
      // never started
    }
    t.source.disconnect();
    t.source = null;
  }

  private applyGains() {
    const anySolo = [...this.tracks.values()].some((t) => t.soloed);
    const now = this.ctx.currentTime;
    for (const t of this.tracks.values()) {
      const audible = anySolo ? t.soloed : !t.muted;
      t.gain.gain.setTargetAtTime(audible ? t.volume : 0, now, RAMP);
    }
  }

  private emit() {
    this.version++;
    for (const fn of this.listeners) fn();
  }
}

/** Max-abs peaks per bucket, per channel, for wavesurfer. */
export function computePeaks(buffer: AudioBuffer, buckets: number): Float32Array[] {
  const out: Float32Array[] = [];
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    const peaks = new Float32Array(buckets);
    const size = data.length / buckets;
    for (let b = 0; b < buckets; b++) {
      let m = 0;
      const end = Math.min(data.length, Math.floor((b + 1) * size));
      for (let i = Math.floor(b * size); i < end; i++) {
        const v = Math.abs(data[i]);
        if (v > m) m = v;
      }
      peaks[b] = m;
    }
    out.push(peaks);
  }
  return out;
}
