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
    if (!(await caches.has("humandit-models"))) return false; // asking must not create it
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
const SAYS = { nvidia: /nvidia|geforce|quadro|rtx/i, amd: /amd|radeon/i, intel: /intel/i, apple: /apple/i, qualcomm: /qualcomm|adreno/i, arm: /mali|arm/i };

/**
 * The graphics chip's model name ("Intel UHD Graphics", "NVIDIA GeForce RTX 4070", "Apple M2"). WebGPU only gives a
 * vendor and an architecture ("intel", "gen-9"), so the name comes from WebGL's renderer string, which Chrome and Edge
 * fill in; other browsers give a generic name or none.
 */
function chipName() {
  try {
    const gl = document.createElement("canvas").getContext("webgl");
    if (!gl) return "";
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const raw = String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER) || "");
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    // "ANGLE (Intel, Intel(R) UHD Graphics (0x00009BC4) Direct3D11 vs_5_0 ps_5_0, D3D11)" -> "Intel UHD Graphics"
    const inner = /^ANGLE \((.*)\)$/.exec(raw)?.[1].split(", ")[1] ?? raw;
    return inner
      .replace(/^ANGLE Metal Renderer:\s*/i, "")
      .replace(/\s*\(0x[0-9a-f]+\).*$/i, "")
      .replace(/\s+(Direct3D|OpenGL|Vulkan).*$/i, "")
      .replace(/\/PCIe.*$|, or similar$/i, "")
      .replace(/\((R|TM)\)|®|™/gi, "")
      .replace(/\s+/g, " ")
      .trim();
  } catch {
    return "";
  }
}

/** A graphics chip built into the processor (shares its memory) rather than a separate graphics card. */
const builtIn = (vendor, arch, name) =>
  vendor === "intel"
    ? !/arc|xe-hpg|alchemist|battlemage/i.test(`${arch} ${name}`)
    : vendor === "amd"
      ? /radeon graphics|vega \d+ graphics|radeon \d{3}m|^amd radeon$/i.test(name)
      : vendor === "qualcomm" || vendor === "arm";

/** Windows, macOS, Linux, Android, iOS: for the advice the page gives. */
export const os = () => {
  const ua = navigator.userAgent;
  return /Windows/.test(ua) ? "windows" : /iPhone|iPad/.test(ua) ? "ios" : /Mac OS X|Macintosh/.test(ua) ? "mac" : /Android/.test(ua) ? "android" : /Linux|CrOS/.test(ua) ? "linux" : "";
};

/** Which browser this is, by name, for the advice the page gives. */
export const browser = () => {
  const ua = navigator.userAgent;
  const brands = (navigator.userAgentData?.brands || []).map((b) => b.brand);
  if (brands.includes("Microsoft Edge") || /Edg\//.test(ua)) return "Edge";
  if (brands.includes("Brave")) return "Brave";
  if (brands.includes("Opera") || /OPR\//.test(ua)) return "Opera";
  if (/Firefox\//.test(ua)) return "Firefox";
  if (brands.includes("Google Chrome") || /Chrome\//.test(ua)) return "Chrome";
  if (/Safari\//.test(ua)) return "Safari";
  return "";
};

/**
 * {gpu, name, vendor, builtIn, label, ep, threads, mobile, memory, cores}: WebGPU when the browser has it (any brand
 * of graphics chip, built-in or separate), else the processor through WebAssembly on every core it lets a page use.
 * On Windows the browser, not the page, decides which chip WebGPU gets (it ignores "high-performance"): usually the
 * built-in one on a laptop with two, unless Windows is told otherwise for that browser.
 */
export const device = () =>
  (deviceP ||= (async () => {
    let adapter = null;
    // ?ai=cpu runs on the processor even where WebGPU exists (how the processor-only speed was measured)
    const cpuOnly = new URLSearchParams(location.search).get("ai") === "cpu";
    try {
      if (!cpuOnly) adapter = (await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" })) || null;
    } catch {}
    const vendor = (adapter?.info?.vendor || "").toLowerCase();
    const arch = adapter?.info?.architecture || "";
    const named = adapter ? chipName() : "";
    // WebGL and WebGPU normally sit on the same chip; if they disagree, trust WebGPU's vendor
    const name = named && (!SAYS[vendor] || SAYS[vendor].test(named))
      ? named
      : [NAMES[vendor] || vendor, arch.replace(/-/g, " ")].filter(Boolean).join(" ") || "a WebGPU graphics chip";
    const cores = navigator.hardwareConcurrency || 4;
    const asked = Number(new URLSearchParams(location.search).get("threads")) || 0; // for measuring
    const threads = self.crossOriginIsolated ? asked || cpuThreads(cores) : 1;
    const mobile = navigator.userAgentData?.mobile ?? /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
    return {
      gpu: !!adapter,
      name: adapter ? name : "",
      vendor,
      builtIn: !!adapter && builtIn(vendor, arch, name),
      label: adapter ? name : `your processor (${threads} thread${threads > 1 ? "s" : ""})`,
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
      return await rpc("init", { base: MODELS, manifest: m, ep: d.ep, threads: d.threads, name: d.name });
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
let liveP = null; // the start-up under way, so buttons pressed meanwhile wait for it instead of falling back
const progressFns = new Set();
export function goLive(onProgress) {
  if (onProgress) progressFns.add(onProgress);
  if (live()) return Promise.resolve(status);
  return (liveP ||= startLive().finally(() => {
    liveP = null;
    progressFns.clear();
  }));
}

/** True while the AI is downloading or preparing: the page asks people to wait until it is running. */
export const starting = () => status.state === "loading";

async function startLive() {
  set("loading", "Please wait · starting the AI");
  try {
    const info = await start();
    await loadPacks(["core"], (p) => {
      const pct = p.total ? Math.floor((100 * p.got) / p.total) : 0;
      set(
        "loading",
        p.phase === "preparing" ? `Please wait · preparing the AI` : p.phase === "downloading" ? `Please wait · downloading ${pct}%` : `Please wait · loading ${pct}%`,
        { progress: p },
      );
      progressFns.forEach((f) => f(p));
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

// ------------------------------------------------------------------ storage: where the AI lives, and removing it
/** Bytes this site keeps on this device (the models, mostly), as the browser counts them; null if it won't say. */
export async function stored() {
  try {
    return (await navigator.storage.estimate()).usage ?? null;
  } catch {
    return null;
  }
}

/** Delete the downloaded AI from this browser and go back to saved results. */
export async function remove() {
  worker?.terminate();
  worker = null;
  started = null;
  loaded.clear();
  pending.forEach((p) => p.reject(new Error("REMOVED")));
  pending.clear();
  try {
    localStorage.removeItem("hd-live");
  } catch {}
  try {
    await caches.delete("humandit-models");
  } finally {
    set("saved", "Saved results", { hardware: "", info: null });
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
 * Run an endpoint on this device: make, edit, compare, check. {onProgress({k, of, phase, retry})}.
 * Throws Error("NOT_LIVE") (in plain words) when live mode is off; "check" works either way (a 47 MB reader).
 */
export async function call(name, payload = {}, { onProgress } = {}) {
  let upgraded = [];
  if (name === "check") {
    if (typeof payload.image === "string") payload = { ...payload, image: dataUrlToBlob(payload.image) };
    try {
      await loadPacks(["check"], onProgress);
    } catch (err) {
      throw new Error(plain(err));
    }
  } else {
    if (starting() && liveP) {
      // pressed while the AI is still starting: wait for it rather than answer from saved results
      try {
        await liveP;
      } catch {}
    }
    if (!live()) throw new Error(plain(new Error("NOT_LIVE")));
    // Instant mode can't draw some features (measured: a goatee in 0 of 16 faces), so those requests use Best quality
    const asked = name === "make" ? payload.must || [] : name === "edit" ? [payload.feature_name] : [];
    const needBest = (payload.quality || "instant") === "instant" ? asked.filter((f) => extras.weak.includes(f)) : [];
    if (needBest.length) payload = { ...payload, quality: "best" };
    upgraded = needBest;
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
    r.upgraded = upgraded; // features that moved this request from Instant to Best quality
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
  if (m === "REMOVED") return "The AI was removed from this device.";
  if (m === "NO_MODELS") return "This copy of the site has no models to run, so it shows results the AI made earlier.";
  if (/quota|QuotaExceeded/i.test(m)) return "This browser has no room left to store the AI. Free some disk space and try again.";
  if (/out of memory|OOM|allocation|device.*lost|GPUDevice/i.test(m))
    return "Your device ran out of memory for the AI. Close some tabs or other apps and try again.";
  if (/Download failed|Failed to fetch|NetworkError|network/i.test(m))
    return "The download was interrupted. Try again: the parts that arrived are kept.";
  if (/damaged|not a picture|Unknown feature/i.test(m)) return m;
  return `The AI hit a problem on this device: ${m.slice(0, 160)}`;
}
