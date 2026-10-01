// The engine: the HumanDiT models running on the visitor's own device (js/ai/worker.js), on its graphics chip through
// WebGPU, or on its processor where a browser has no WebGPU. Nothing is sent to a server: the models download once
// (Cloudflare serves them with the page), stay in the browser's storage, and every face is made where it is shown.
// Until the visitor switches live mode on, the page answers from results the AI made earlier, and says so.

const listeners = new Set();
export const status = { state: "saved", text: "Saved results", hardware: "", info: null };
const set = (state, text, extra = {}) => {
  Object.assign(status, { state, text }, extra);
  listeners.forEach((f) => f(status));
};
export const onStatus = (f) => (listeners.add(f), () => listeners.delete(f));

const MODELS = new URL("../models/", import.meta.url).href;
let manifestP = null;
/** The model manifest (sizes, packs, measured quality), or null when this copy of the site has no models. */
export const manifest = () =>
  (manifestP ||= fetch(`${MODELS}manifest.json`, { cache: "no-cache" })
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null));

const mb = (bytes) => Math.round(bytes / 1e6);
/** Megabytes a pack downloads the first time. */
export async function packSize(...packs) {
  const m = await manifest();
  if (!m) return 0;
  const keys = [...new Set(packs.flatMap((p) => m.packs[p]))];
  return mb(keys.reduce((s, k) => s + m.files[k].bytes, 0));
}

/** True when every file of the pack is already in this browser's storage (a return visit). */
export async function cached(pack) {
  const m = await manifest();
  if (!m) return false;
  try {
    const c = await caches.open("humandit-models");
    const urls = m.packs[pack].flatMap((k) =>
      Array.from({ length: m.files[k].parts }, (_, i) => `${MODELS}${m.version}/${k}.${String(i).padStart(2, "0")}`),
    );
    return (await Promise.all(urls.map((u) => c.match(u)))).every(Boolean);
  } catch {
    return false;
  }
}

// ------------------------------------------------------------------ what this device can do
/** Threads for the processor path: logical cores, capped (more threads than that only fight each other). */
const cpuThreads = (cores) => Math.max(1, Math.min(16, cores));

let deviceP = null;
const NAMES = { nvidia: "NVIDIA", amd: "AMD", intel: "Intel", apple: "Apple", qualcomm: "Qualcomm", arm: "Arm" };
/**
 * {gpu, label, ep, threads, mobile, memory}: WebGPU when the browser has it (any brand of graphics chip, built-in
 * or separate), else the processor through WebAssembly on every core it lets a page use.
 */
export const device = () =>
  (deviceP ||= (async () => {
    let adapter = null;
    // ?ai=cpu runs on the processor even where WebGPU exists (how the processor-only speed was measured)
    const cpuOnly = new URLSearchParams(location.search).get("ai") === "cpu";
    try {
      if (!cpuOnly) adapter = (await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" })) || null;
    } catch {}
    const v = (adapter?.info?.vendor || "").toLowerCase();
    const arch = (adapter?.info?.architecture || "").replace(/-/g, " ");
    const cores = navigator.hardwareConcurrency || 4;
    const asked = Number(new URLSearchParams(location.search).get("threads")) || 0; // for measuring
    const threads = self.crossOriginIsolated ? asked || cpuThreads(cores) : 1;
    const mobile = navigator.userAgentData?.mobile ?? /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
    return {
      gpu: !!adapter,
      label: adapter ? [NAMES[v] || v, arch].filter(Boolean).join(" ") || "a WebGPU graphics chip" : "",
      ep: adapter ? "webgpu" : "wasm",
      threads: adapter ? Math.min(4, threads) : threads,
      mobile,
      memory: navigator.deviceMemory || 0, // GB, rounded down and capped at 8 (Chrome only)
      cores,
    };
  })());

// ------------------------------------------------------------------ the worker
let worker = null;
let seq = 0;
const pending = new Map();
function rpc(op, args, onProgress) {
  worker ||= (() => {
    const w = new Worker(new URL("./ai/worker.js", import.meta.url), { type: "module" });
    w.onmessage = (e) => {
      const { id, result, error, progress } = e.data;
      const p = pending.get(id);
      if (!p) return;
      if (progress) return p.onProgress?.(progress);
      pending.delete(id);
      error ? p.reject(new Error(error)) : p.resolve(result);
    };
    w.onerror = (e) => {
      pending.forEach((p) => p.reject(new Error(e.message || "The AI stopped unexpectedly.")));
      pending.clear();
    };
    return w;
  })();
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ id, op, args });
  });
}

let started = null; // init: resolves to {hardware, ep}
const loaded = new Set();
async function start() {
  return (started ||= (async () => {
    const [m, d] = await Promise.all([manifest(), device()]);
    if (!m) throw new Error("NO_MODELS");
    try {
      return await rpc("init", { base: MODELS, manifest: m, ep: d.ep, threads: d.threads });
    } catch (err) {
      if (d.ep !== "webgpu") throw err;
      // a graphics chip that offers WebGPU but can't open it: the processor still works
      d.ep = "wasm";
      d.threads = self.crossOriginIsolated ? cpuThreads(d.cores) : 1;
      worker?.terminate();
      worker = null;
      return rpc("init", { base: MODELS, manifest: m, ep: "wasm", threads: d.threads });
    }
  })().catch((err) => {
    started = null;
    throw err;
  }));
}

async function loadPacks(packs, onProgress) {
  const m = await manifest();
  const need = packs.filter((p) => !m.packs[p].every((k) => loaded.has(k)));
  if (!need.length) return;
  await start();
  await rpc("load", { packs: need }, onProgress);
  need.forEach((p) => m.packs[p].forEach((k) => loaded.add(k)));
}

export const live = () => status.state === "ready" || status.state === "busy";

/**
 * Switch live mode on: download (once) and prepare the models for Instant mode, then the page's buttons run on this
 * device. onProgress({phase, got, total, toFetch}).
 */
export async function goLive(onProgress) {
  if (live()) return status;
  set("loading", "Loading the AI…");
  try {
    const info = await start();
    await loadPacks(["core"], (p) => {
      const pct = p.total ? Math.floor((100 * p.got) / p.total) : 0;
      set("loading", p.phase === "preparing" ? `Preparing the AI · ${pct}%` : `Loading the AI · ${pct}%`);
      onProgress?.(p);
    });
    const m = await manifest();
    set("ready", `Live on ${info.hardware}`, {
      hardware: info.hardware,
      info: { features: m.features, ep: info.ep },
    });
    try {
      localStorage.setItem("hd-live", "1");
    } catch {}
    return status;
  } catch (err) {
    set("saved", "Saved results", { error: plain(err) });
    throw new Error(plain(err));
  }
}

/** On a return visit, after an earlier opt-in, load from this browser's storage without asking again. */
export function resume() {
  let opted = false;
  try {
    opted = localStorage.getItem("hd-live") === "1";
  } catch {}
  if (!opted) return;
  const go = () => goLive().catch(() => {});
  if ("requestIdleCallback" in window) setTimeout(() => requestIdleCallback(go, { timeout: 3000 }), 1500);
  else setTimeout(go, 1500);
}

// ------------------------------------------------------------------ requests
let extras = { weak: [], fids: [] };
/** Page data the answers need: features Instant mode is weak at, and the FID of each race lane. */
export const configure = (o) => (extras = { ...extras, ...o });

const urls = (v) => {
  if (v instanceof Blob) return URL.createObjectURL(v);
  if (Array.isArray(v)) return v.map(urls);
  if (v && typeof v === "object") return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, urls(x)]));
  return v;
};

function dataUrlToBlob(s) {
  const [head, body] = s.split(",");
  const type = /data:([^;]+)/.exec(head)?.[1] || "application/octet-stream";
  const bin = /;base64/.test(head) ? atob(body) : decodeURIComponent(body);
  return new Blob([Uint8Array.from(bin, (c) => c.charCodeAt(0))], { type });
}

let running = 0;
/**
 * Run an endpoint on this device: make, edit, blend, compare, check. {onProgress({k, of, phase})}.
 * Throws Error("NOT_LIVE") (in plain words) when live mode is off; "check" works either way (a 47 MB reader).
 */
export async function call(name, payload = {}, { onProgress } = {}) {
  if (name === "check") {
    if (typeof payload.image === "string") payload = { ...payload, image: dataUrlToBlob(payload.image) };
    try {
      await loadPacks(["check"], onProgress);
    } catch (err) {
      throw new Error(plain(err));
    }
  } else {
    if (!live()) throw new Error(plain(new Error("NOT_LIVE")));
    if (payload.quality === "best" || name === "compare") {
      try {
        await loadPacks(["best"], (p) => {
          set("busy", `Loading Best quality · ${p.total ? Math.floor((100 * p.got) / p.total) : 0}%`);
          onProgress?.({ ...p, pack: "best" });
        });
      } catch (err) {
        if (live()) set("ready", `Live on ${status.hardware}`);
        throw new Error(plain(err));
      }
    }
  }
  running++;
  if (live()) set("busy", "Working on your device…");
  try {
    const r = urls(await rpc("run", { name, payload }, onProgress));
    if (name === "make" || name === "edit") {
      const asked = name === "make" ? payload.must || [] : [payload.feature_name];
      r.weak = (r.quality || payload.quality) === "instant" ? asked.filter((f) => extras.weak.includes(f)) : [];
    }
    if (name === "compare")
      r.runs.forEach((x) => {
        const f = extras.fids.find((y) => y.steps === x.steps);
        x.fid = f?.fid ?? null;
        x.fid_n = f?.fid_n ?? null;
      });
    return r;
  } catch (err) {
    throw new Error(plain(err));
  } finally {
    running--;
    if (!running && live()) set("ready", `Live on ${status.hardware}`);
  }
}

export function plain(err) {
  const m = String(err?.message || err || "");
  if (m === "NOT_LIVE") return "Live mode is off, so this shows results the AI made earlier.";
  if (m === "NO_MODELS") return "This copy of the site has no models to run, so it shows results the AI made earlier.";
  if (/quota|QuotaExceeded/i.test(m)) return "This browser has no room left to store the AI. Free some disk space and try again.";
  if (/out of memory|OOM|allocation|device.*lost|GPUDevice/i.test(m))
    return "Your device ran out of memory for the AI. Close some tabs or other apps and try again.";
  if (/Download failed|Failed to fetch|NetworkError|network/i.test(m))
    return "The download was interrupted. Try again: the parts that arrived are kept.";
  if (/damaged|not a picture|Unknown feature/i.test(m)) return m;
  return `The AI hit a problem on this device: ${m.slice(0, 160)}`;
}
