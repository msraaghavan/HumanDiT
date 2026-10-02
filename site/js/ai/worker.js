// The AI, running on the visitor's own device: ONNX Runtime Web on the graphics chip through WebGPU (NVIDIA, AMD, Intel
// or Apple, built-in or separate) or, where a browser has no WebGPU, on the processor through WebAssembly on every
// core. It runs in this worker so the page's animation never waits for it. Each request mirrors an endpoint of the
// Python engine (app/api.py) and answers in the same shape; the page cannot tell them apart except by the speed.

import * as ort from "../../vendor/ort/ort.webgpu.min.mjs";
import { withText } from "./png.js";
import { resizeRGB } from "./resize.js";
import { randn } from "./rng.js";
import { load, prune, stored } from "./store.js";

const L = 1024 * 16 * 16; // one latent: 1024 channels on a 16 x 16 grid
const SIZE = 256;
const PIX = SIZE * SIZE;
let M = null; // the manifest: files, packs, sampler settings, the checker, the watermark codeword
let BASE = "";
let EP = "wasm";
let HW = "";
let K = 12;
let SECRET = null;
const S = {}; // ONNX Runtime sessions, by file key

const post = (msg) => self.postMessage(msg);

// cancelling: a "cancel" message (handled at once, not queued) moves the epoch on; the job under way stops at its next
// step and any job queued before the cancel never starts. ONNX Runtime can't stop inside one model call, so the
// longest wait is one step: a fraction of a second on a graphics chip.
let EPOCH = 0;
let RUNNING = 0; // the epoch of the job under way
let downloads = null; // aborts the download under way
const alive = () => {
  if (EPOCH !== RUNNING) throw new Error("CANCELLED");
};
const now = () => performance.now();
const T = (type, data, dims) => new ort.Tensor(type, data, dims);

// ------------------------------------------------------------------ set-up
const NAMES = { nvidia: "NVIDIA", amd: "AMD", intel: "Intel", apple: "Apple", qualcomm: "Qualcomm", arm: "Arm" };
const pretty = (s) => s.replace(/-/g, " ").replace(/\b([a-z])/g, (c) => c.toUpperCase()).replace(/\bRdna\b/i, "RDNA");

async function wasmBinary() {
  const res = await fetch(new URL("../../vendor/ort/ort-wasm-simd-threaded.asyncify.wasm.gz", import.meta.url));
  if (!res.ok) throw new Error("The AI runtime could not be downloaded.");
  const bytes = new Uint8Array(await res.arrayBuffer());
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes.buffer; // a server already undid the gzip
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

async function init({ base, manifest, ep, threads, name }) {
  M = manifest;
  BASE = base;
  K = M.features.length;
  SECRET = Float32Array.from(M.watermark.bits);
  EP = ep;
  ort.env.logLevel = "error";
  ort.env.wasm.wasmPaths = new URL("../../vendor/ort/", import.meta.url).href;
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.wasmBinary = await wasmBinary();
  if (EP === "webgpu") {
    const a = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!a) throw new Error("NO_WEBGPU");
    const v = (a.info?.vendor || "").toLowerCase();
    const arch = a.info?.architecture || "";
    // the page names the chip from WebGL (a worker has no DOM); vendor and architecture are the fallback
    HW = `your graphics chip (${name || [NAMES[v] || pretty(v), arch && pretty(arch)].filter(Boolean).join(" ") || "WebGPU"})`;
  } else {
    HW = `your processor (${threads} thread${threads > 1 ? "s" : ""})`;
  }
  prune(M.version);
  return { hardware: HW, ep: EP };
}

/** Download (or read from storage), verify and prepare every file of these packs. */
async function loadPacks({ packs }, progress) {
  const keys = [...new Set(packs.flatMap((p) => M.packs[p]))].filter((k) => !S[k]);
  const have = await stored(BASE, M, keys);
  const total = keys.reduce((s, k) => s + M.files[k].bytes, 0);
  const toFetch = keys.filter((k) => !have[k]).reduce((s, k) => s + M.files[k].bytes, 0);
  let got = 0;
  let last = 0;
  const tick = (phase, key, force = false) => {
    const t = now();
    if (!force && t - last < 120) return;
    last = t;
    progress({ phase, key, got, total, toFetch });
  };
  for (const key of keys) {
    downloads = new AbortController();
    const bytes = await load(BASE, M, key, (n) => {
      got += n;
      tick(have[key] ? "reading" : "downloading", key);
    }, downloads.signal);
    alive();
    tick("preparing", key, true);
    S[key] = await ort.InferenceSession.create(bytes, {
      executionProviders: [EP === "webgpu" ? "webgpu" : "wasm"],
      graphOptimizationLevel: "all",
      logSeverityLevel: 3, // errors only: the native "shape ops placed on the CPU" notice is expected
    });
  }
  progress({ phase: "ready", got: total, total, toFetch });
  return { loaded: Object.keys(S) };
}

// ------------------------------------------------------------------ the models
const tokens = (choices) => BigInt64Array.from(choices.map((c) => BigInt(M.tokens[c])));
const repeat = (row, n) => {
  const out = new BigInt64Array(row.length * n);
  for (let i = 0; i < n; i++) out.set(row, i * row.length);
  return out;
};

/** The one-step student, face by face: noise -> clean latent. */
async function oneStep(noise, attrs, n, onFace) {
  const z = new Float32Array(n * L);
  for (let i = 0; i < n; i++) {
    const r = await S.generator.run({
      noise: T("float32", noise.slice(i * L, (i + 1) * L), [1, 1024, 16, 16]),
      attrs: T("int64", attrs.slice(i * K, (i + 1) * K), [1, K]),
    });
    z.set(r.out.data, i * L);
    alive();
    onFace?.(i + 1);
  }
  return z;
}

/** shift_t(linspace(1, 0, steps + 1)): the noise level at each step, as in Transport.time_grid. */
function grid(steps) {
  const s = M.shift;
  return Float32Array.from({ length: steps + 1 }, (_, i) => {
    const t = Math.fround(1 - i / steps);
    return (s * t) / (1 + (s - 1) * t);
  });
}

/**
 * The teacher's flow-matching ODE (Euler), with Internal Guidance and, when any feature is asked for, attribute
 * classifier-free guidance: guided_model_fn + Transport.sample, all n faces in one batch per step.
 */
async function teacher(noise, attrs, n, steps, cfg, onStep) {
  const ts = grid(steps);
  const useCfg = cfg !== 1 && attrs.some((v) => v !== 0n);
  const B = useCfg ? 2 * n : n;
  const att = new BigInt64Array(B * K);
  att.set(attrs); // the second half stays 0: the unconditional (null) rows
  const x = Float32Array.from(noise);
  const xin = new Float32Array(B * L);
  const nL = n * L;
  const { scale: ig, t_min: igLo, t_max: igHi } = M.ig;
  for (let i = 0; i < steps; i++) {
    const t = ts[i];
    xin.set(x);
    if (useCfg) xin.set(x, nL);
    const r = await S.teacher.run({
      x: T("float32", xin, [B, 1024, 16, 16]),
      t: T("float32", new Float32Array(B).fill(t), [B]),
      attrs: T("int64", att, [B, K]),
    });
    const xf = r.x_full.data;
    const xb = r.x_base.data;
    const igOn = ig !== 1 && t >= igLo && t <= igHi;
    const cfgOn = useCfg && t >= M.cfg.t_min && t <= M.cfg.t_max;
    const inv = 1 / Math.max(t, M.t_eps);
    const dt = ts[i + 1] - t;
    for (let k = 0; k < nL; k++) {
      let xc = igOn ? xb[k] + ig * (xf[k] - xb[k]) : xf[k];
      if (cfgOn) {
        const u = k + nL;
        const xu = igOn ? xb[u] + ig * (xf[u] - xb[u]) : xf[u];
        xc = xu + cfg * (xc - xu);
      }
      x[k] += dt * (x[k] - xc) * inv; // v = (x - x̂) / t, then one Euler step towards t = 0
    }
    alive();
    onStep?.(i + 1);
  }
  return x;
}

async function sample(quality, noise, attrs, n, cfg, onStep) {
  if (quality === "best") {
    if (!S.teacher) throw new Error("NEED_BEST");
    return teacher(noise, attrs, n, M.steps, cfg, onStep);
  }
  return oneStep(noise, attrs, n, onStep);
}

/** Latent -> RGBA bytes of a 256 x 256 face. */
async function decode(z, i) {
  const r = await S.decoder.run({ latent: T("float32", z.slice(i * L, (i + 1) * L), [1, 1024, 16, 16]) });
  const x = r.out.data;
  const rgba = new Uint8ClampedArray(PIX * 4);
  for (let p = 0; p < PIX; p++) {
    rgba[p * 4] = Math.round(x[p] * 255);
    rgba[p * 4 + 1] = Math.round(x[PIX + p] * 255);
    rgba[p * 4 + 2] = Math.round(x[2 * PIX + p] * 255);
    rgba[p * 4 + 3] = 255;
  }
  return rgba;
}

/**
 * TrustMark's invisible watermark ("HumanDiT", BCH-protected), as trustmark.TrustMark.encode does it for a
 * 256 x 256 picture: the encoder's residual minus its colour cast, added to the picture, edges feathered.
 */
async function mark(rgba) {
  const cover = new Float32Array(3 * PIX);
  for (let p = 0; p < PIX; p++) for (let c = 0; c < 3; c++) cover[c * PIX + p] = (rgba[p * 4 + c] / 255) * 2 - 1;
  const r = await S.tm_encoder.run({ cover: T("float32", cover, [1, 3, SIZE, SIZE]), secret: T("float32", SECRET, [1, 100]) });
  const st = r.stego.data;
  const res = new Float32Array(3 * PIX);
  const mean = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let p = 0; p < PIX; p++) {
      const v = Math.min(1, Math.max(-1, st[c * PIX + p])) - cover[c * PIX + p];
      res[c * PIX + p] = v;
      s += v;
    }
    mean[c] = s / PIX;
  }
  for (let y = 0; y < SIZE; y++) {
    const rowEdge = y === 0 || y === SIZE - 1;
    for (let x = 0; x < SIZE; x++) {
      const edge = rowEdge || x === 0 || x === SIZE - 1;
      const p = y * SIZE + x;
      for (let c = 0; c < 3; c++) {
        const orig = rgba[p * 4 + c];
        const w = Math.min(1, Math.max(-1, res[c * PIX + p] - mean[c] + orig / 127.5 - 1)) * 127.5 + 127.5;
        rgba[p * 4 + c] = Math.floor(edge ? 0.5 * w + 0.5 * orig : w);
      }
    }
  }
  return rgba;
}

/** The checker: a linear probe on the latent's average, P(feature present) for each of the 12 features. */
function probs(z, i) {
  const { mu, sigma, weight, bias, invert } = M.probe;
  const f = new Float64Array(1024);
  for (let c = 0; c < 1024; c++) {
    let s = 0;
    const o = i * L + c * 256;
    for (let k = 0; k < 256; k++) s += z[o + k];
    f[c] = (s / 256 - mu[c]) / sigma[c];
  }
  return weight.map((w, j) => {
    let l = bias[j];
    for (let c = 0; c < 1024; c++) l += w[c] * f[c];
    const p = 1 / (1 + Math.exp(-l));
    return invert[j] ? 1 - p : p;
  });
}

function canvas(w, h) {
  const cv = new OffscreenCanvas(w, h);
  return [cv, cv.getContext("2d")];
}

async function png(rgba, details) {
  const [cv, ctx] = canvas(SIZE, SIZE);
  ctx.putImageData(new ImageData(rgba, SIZE, SIZE), 0, 0);
  return withText(await cv.convertToBlob({ type: "image/png" }), {
    "AI-generated": "true",
    Generator: "HumanDiT (RAEv2 + DiT^DH), research demo; synthetic person, not a real individual",
    Details: details,
  });
}

async function jpeg(frames, cols, quality) {
  const rows = Math.ceil(frames.length / cols);
  const [cv, ctx] = canvas(cols * SIZE, rows * SIZE);
  frames.forEach((f, i) => ctx.putImageData(new ImageData(f, SIZE, SIZE), (i % cols) * SIZE, Math.floor(i / cols) * SIZE));
  return cv.convertToBlob({ type: "image/jpeg", quality });
}

const newSeed = () => Math.floor(Math.random() * 2 ** 31);
const seedOr = (s) => (Number(s) >= 0 ? Number(s) : newSeed());
const MODE = { instant: "Instant (1-step GAN student)", best: `Best quality (teacher, 25-step ODE)` };
const where = () => `in a web browser on ${HW}`;
const r3 = (v) => Math.round(v * 1000) / 1000;

// ------------------------------------------------------------------ getting what was asked for
// The models follow a request most of the time, not every time (measured on the saved faces: Instant drew glasses
// in 16 of 16, a man in 11 of 16). So each face is read by the checker while it is still a latent, before the
// expensive decode, and a face it rejects is drawn again from new noise. A face that never passes keeps its closest
// try, and its marks say so.
const TRIES = { instant: 6, best: 2 };
const noiseFor = (seed, round, n) => randn(round ? (seed + round * 1_000_003) % 2 ** 31 : seed, n * L);

/** n latents that pass the checker on every asked choice ({j: "yes"|"no"}); returns {z, tries, misses}. */
async function matching(quality, seed, n, choices, cfg, progress, what) {
  const asked = choices.map((c, j) => [c, j]).filter(([c]) => c !== "any");
  const passes = (p) => asked.every(([c, j]) => p[j] > 0.5 === (c === "yes"));
  const closeness = (p) => asked.reduce((s, [c, j]) => s + Math.log(1e-6 + (c === "yes" ? p[j] : 1 - p[j])), 0);
  const row = tokens(choices);
  const z = new Float32Array(n * L);
  const kept = new Array(n).fill(-Infinity);
  const from = new Array(n).fill(0); // the round each kept face was drawn in (0: the seed's own noise)
  let todo = [...Array(n).keys()];
  let tries = 0;
  const rounds = asked.length ? TRIES[quality] : 1;
  for (let r = 0; r < rounds && todo.length; r++) {
    const m = todo.length;
    const all = noiseFor(seed, r, n);
    const noise = new Float32Array(m * L);
    todo.forEach((i, k) => noise.set(all.subarray(i * L, (i + 1) * L), k * L));
    const of = quality === "best" ? M.steps : m;
    const zz = await sample(quality, noise, repeat(row, m), m, cfg, (k) => progress({ k, of, what, retry: r, redraw: r ? m : 0 }));
    tries += m;
    const left = [];
    todo.forEach((i, k) => {
      const p = probs(zz, k);
      const ok = passes(p);
      const c = ok ? Infinity : closeness(p);
      if (c > kept[i]) {
        kept[i] = c;
        from[i] = r;
        z.set(zz.subarray(k * L, (k + 1) * L), i * L);
      }
      if (!ok) left.push(i);
    });
    todo = left;
  }
  return { z, tries, misses: todo.length, from };
}

// ------------------------------------------------------------------ endpoints (app/api.py)
async function make({ count = 4, quality = "instant", must = [], never = [], seed = -1 }, progress) {
  const n = Math.max(1, Math.min(8, Number(count)));
  seed = seedOr(seed);
  const choices = new Array(K).fill("any");
  for (const f of never || []) choices[M.features.indexOf(f)] = "no";
  for (const f of must || []) choices[M.features.indexOf(f)] = "yes";
  let t = now();
  const { z, tries, misses, from } = await matching(quality, seed, n, choices, M.cfg.make, progress, "make");
  const tGen = now() - t;
  const imgs = [];
  t = now();
  for (let i = 0; i < n; i++) {
    imgs.push(await decode(z, i));
    alive();
    progress({ k: i + 1, of: n, what: "finish" });
  }
  const tDec = now() - t;
  t = now();
  for (const im of imgs) {
    await mark(im);
    alive();
  }
  const tWm = now() - t;
  const asked = choices.map((c, j) => [M.model_features[j], c]).filter(([, c]) => c !== "any");
  const askedText = asked.length ? `{${asked.map(([k, c]) => `'${k}': '${c}'`).join(", ")}}` : "unconditional";
  const faces = [];
  for (let i = 0; i < n; i++) {
    const p = probs(z, i);
    faces.push({
      png: await png(imgs[i], `${MODE[quality]}; seed ${seed} face ${i}${from[i] ? ` redraw ${from[i]}` : ""}; steps ${M.steps}; cfg ${M.cfg.make}; ig ${M.ig.scale}; attributes ${askedText}; made ${where()}`),
      marks: choices
        .map((c, j) => [c, j])
        .filter(([c]) => c !== "any")
        .map(([c, j]) => ({ feature: M.features[j], want: c === "yes", ok: p[j] > 0.5 === (c === "yes"), p: r3(p[j]) })),
      sees: M.features.filter((_, j) => p[j] > 0.5),
    });
  }
  const sec = (tGen + tDec + tWm) / 1000;
  return {
    seed,
    quality,
    tries, // faces drawn in all, counting the ones the checker sent back
    misses, // faces still not matching after every try (shown with their closest try)
    faces,
    seconds: r3(sec),
    ms_per_face: { make: +(tGen / n).toFixed(1), decode: +(tDec / n).toFixed(1), mark: +(tWm / n).toFixed(1) },
    hardware: HW,
  };
}

async function compare({ seed = -1, feature_name = "" }, progress) {
  seed = seedOr(seed);
  const noise = randn(seed, L);
  const attrs = new BigInt64Array(K);
  if (feature_name) attrs[M.features.indexOf(feature_name)] = BigInt(M.tokens.yes);
  const runs = [["instant", 1]];
  if (S.teacher) runs.push(["best", 10], ["best", 25], ["best", 50]);
  const total = runs.reduce((s, [, k]) => s + k, 0);
  let done = 0;
  const out = [];
  for (const [q, steps] of runs) {
    const t = now();
    const z =
      q === "instant"
        ? await oneStep(noise, attrs, 1)
        : await teacher(noise, attrs, 1, steps, M.cfg.make, (k) => progress({ k: done + k, of: total, what: "compare" }));
    const sec = (now() - t) / 1000;
    done += steps;
    progress({ k: done, of: total, what: "compare" });
    out.push({ label: q === "instant" ? "Instant" : `${steps} steps`, steps, seconds: +sec.toFixed(4), img: await jpeg([await mark(await decode(z, 0))], 1, 0.9) });
    alive();
  }
  return { seed, runs: out, hardware: HW };
}

/** Is the HumanDiT watermark in this picture? trustmark.TrustMark.decode, then its BCH check as a distance test. */
async function check({ image }) {
  let bmp;
  try {
    bmp = await createImageBitmap(image, { imageOrientation: "none", colorSpaceConversion: "none", premultiplyAlpha: "none" });
  } catch {
    bmp = await createImageBitmap(image).catch(() => null);
  }
  if (!bmp) throw new Error("That file is not a picture this can read.");
  const [w, h] = [bmp.width, bmp.height];
  // TrustMark reads a centred square from very long pictures (aspect over 2), the whole picture otherwise
  const sq = Math.max(w, h) / Math.min(w, h) > 2 ? Math.min(w, h) : 0;
  const [cw, ch] = sq ? [sq, sq] : [w, h];
  const [cv, ctx] = canvas(cw, ch);
  ctx.drawImage(bmp, sq ? Math.floor((w - sq) / 2) : 0, sq ? Math.floor((h - sq) / 2) : 0, cw, ch, 0, 0, cw, ch);
  const px = ctx.getImageData(0, 0, cw, ch).data;
  const rgb = new Uint8Array(cw * ch * 3);
  for (let p = 0; p < cw * ch; p++) rgb.set(px.subarray(p * 4, p * 4 + 3), p * 3);
  const R = M.watermark.dec_res;
  const small = resizeRGB(rgb, cw, ch, R, R);
  const x = new Float32Array(3 * R * R);
  for (let p = 0; p < R * R; p++) for (let c = 0; c < 3; c++) x[c * R * R + p] = (small[p * 3 + c] / 255) * 2 - 1;
  const bits = (await S.tm_decoder.run({ image: T("float32", x, [1, 3, R, R]) })).bits.data;
  const code = M.watermark.bits;
  // the last two bits name the schema (BCH-5 here); BCH-5 corrects up to 5 flipped bits among the first 96
  const read = Array.from(bits, (v) => (v > 0 ? 1 : 0));
  const schemaOk = read[98] === code[98] && read[99] === code[99];
  let flips = 0;
  for (let i = 0; i < 96; i++) flips += read[i] !== code[i];
  return { found: schemaOk && flips <= M.watermark.max_flips, size: [w, h], flips };
}

const ENDPOINTS = { make, compare, check };

// ------------------------------------------------------------------ messages: one job at a time, in order
let queue = Promise.resolve();
self.onmessage = (e) => {
  const { id, op, args } = e.data;
  if (op === "cancel") {
    EPOCH++;
    downloads?.abort();
    return;
  }
  const mine = EPOCH;
  queue = queue.then(async () => {
    const progress = (p) => post({ id, progress: p });
    RUNNING = mine;
    try {
      alive(); // cancelled while it waited its turn
      let result;
      if (op === "init") result = await init(args);
      else if (op === "load") result = await loadPacks(args, progress);
      else if (op === "run") {
        const fn = ENDPOINTS[args.name];
        if (!fn) throw new Error(`Unknown request: ${args.name}`);
        result = await fn(args.payload || {}, progress);
      } else throw new Error(`Unknown op: ${op}`);
      post({ id, result });
    } catch (err) {
      post({ id, error: String(err?.message || err) });
    }
  });
};
