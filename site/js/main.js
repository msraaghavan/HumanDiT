// HumanDiT website: every section starts from results the AI made earlier (assets/site.json, labelled as such). When
// the visitor switches live mode on, the AI downloads once and every button runs it on their own device (engine.js).

import * as engine from "./engine.js";
import { nightScene } from "./pixel.js";
import {
  animate,
  clamp,
  countTo,
  drawFrame,
  easeInOut,
  easeOut,
  fit,
  grain,
  isOnScreen,
  lerp,
  loadImage,
  magnetic,
  noiseTiles,
  reduced,
  reveal,
  split,
  staticStrip,
} from "./motion.js";

document.documentElement.classList.add("js");
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pct = (v) => `${Math.round(v * 100)}%`;
const ICON = {
  yes: '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5l2.5 2.5L10 3.5"/></svg>',
  no: '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3l6 6M9 3l-6 6"/></svg>',
  dl: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>',
};
const hwLabel = (hw) => (/^your /.test(hw) ? hw : /cpu/i.test(hw) ? "a CPU" : `a laptop GPU (${hw})`);
const madeEarlier = (hw) => `Made earlier on ${hwLabel(hw)}`;

// CSS resolves url() in a custom property against the stylesheet, so pass absolute URLs
const FACES_URL = `url(${new URL("assets/faces.jpg", location.href).href})`;
let SITE = null;
let TILES = null; // canvases of static
let STATIC = ""; // the 6-frame strip, as a CSS url()

function toast(msg, ms = 7000) {
  const t = $(".toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), ms);
}

// ================================================================== saved results: the bank
// While the live engine is away, every button answers from a bank of results the AI made earlier (uncurated, made
// by the same endpoints; scripts/site/build_bank.py) and says so. `paused` means no engine is configured at all.
let paused = true; // until live mode is on, every button answers from the bank
engine.onStatus(() => (paused = !engine.live()));
let BANK = null;
const bank = () =>
  (BANK ||= fetch("assets/bank.json")
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null));
const wait = (ms) => new Promise((r) => setTimeout(r, reduced ? 0 : ms));

/** Say why a saved result is showing: nothing if live mode is simply paused, else the engine's problem. */
function fellBack(err) {
  if (!paused) toast(`${err.message} Showing a saved result instead.`);
}

function staticLayer() {
  const s = el("span", "static");
  s.setAttribute("aria-hidden", "true");
  s.append(el("i"));
  return s;
}

/** A small live timer for a status line: returns stop(), with stop.label(text) to change the words. */
function ticking(node, label) {
  const t0 = performance.now();
  let text = label;
  const stop = animate(null, (now) => {
    node.innerHTML = `${esc(text)} <b>${((now - t0) / 1000).toFixed(1)} s</b>`;
  }, 12);
  const done = () => stop();
  done.label = (t) => (text = t);
  return done;
}

/** Progress from the engine, in words: a first-time download of the Best quality model, or steps of the work. */
const progressWords = (p, doing) =>
  p.pack === "best" || p.phase
    ? p.phase === "preparing"
      ? "Preparing the Best quality model for your device…"
      : `Downloading the Best quality model (once): ${Math.round(p.got / 1e6)} of ${Math.round(p.total / 1e6)} MB…`
    : `${doing} ${p.k} of ${p.of}…`;

// ================================================================== intro: the page denoises itself (once a visit)
function intro() {
  const box = $(".intro");
  let seen = false;
  try {
    seen = sessionStorage.getItem("hd-intro") === "1";
    sessionStorage.setItem("hd-intro", "1");
  } catch {}
  if (reduced || seen) {
    box.classList.add("gone");
    return Promise.resolve();
  }
  const cv = $("canvas", box);
  const W = 192;
  const aspect = innerWidth > 0 && innerHeight > 0 ? innerHeight / innerWidth : 0.6;
  const H = Math.round(clamp(W * aspect, 64, 420));
  cv.width = W;
  cv.height = H;
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(W, H);
  const order = new Float32Array(W * H).map(() => Math.random());
  const t0 = performance.now();
  box.classList.add("show");
  return new Promise((resolve) => {
    const stop = animate(null, (now) => {
      const t = (now - t0) / 1000;
      // static clears from the middle outwards after 0.45 s: each pixel goes when its time comes
      const clear = clamp((t - 0.45) / 0.75);
      const d = img.data;
      for (let y = 0, k = 0; y < H; y++) {
        for (let x = 0; x < W; x++, k++) {
          const r = Math.hypot((x - W / 2) / W, (y - H / 2) / H) * 1.5;
          const gone = clear > 0 && order[k] * 0.35 + r * 0.65 < easeOut(clear) * 1.1;
          const v = Math.random() * 255;
          const i = k * 4;
          d[i] = v * 0.86;
          d[i + 1] = v * 0.93;
          d[i + 2] = Math.min(255, v * 1.05 + 20);
          d[i + 3] = gone ? 0 : 150;
        }
      }
      ctx.putImageData(img, 0, 0);
      if (t > 0.9) box.classList.add("done");
      if (t > 1.25) {
        stop();
        box.classList.add("gone");
        resolve();
      }
    }, 30);
  });
}

// ================================================================== chrome: nav, progress, active section
function chrome() {
  const pill = $(".live-pill");
  const hero = $(".hero");
  new IntersectionObserver(([e]) => pill.classList.toggle("show", !e.isIntersecting), { threshold: 0.15 }).observe(hero);
}

// ================================================================== pointer-driven polish
function polish() {
  // section kickers draw their line when the heading arrives
  const heads = new IntersectionObserver(
    (es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), heads.unobserve(e.target))),
    { rootMargin: "0px 0px -15% 0px" },
  );
  $$("[data-reveal-head], .how-copy").forEach((h) => heads.observe(h));
  if (reduced || !matchMedia("(pointer: fine)").matches) return;
  $$(".panel").forEach((p) => {
    p.addEventListener("pointermove", (e) => {
      const r = p.getBoundingClientRect();
      p.style.setProperty("--gx", `${e.clientX - r.left}px`);
      p.style.setProperty("--gy", `${e.clientY - r.top}px`);
    });
    p.addEventListener("pointerenter", () => p.classList.add("lit"));
    p.addEventListener("pointerleave", () => p.classList.remove("lit"));
  });
  const hero = $(".hero");
  const frame = $(".portrait-frame");
  hero.addEventListener("pointermove", (e) => {
    const r = frame.getBoundingClientRect();
    const x = clamp((e.clientX - (r.left + r.width / 2)) / innerWidth, -0.5, 0.5);
    const y = clamp((e.clientY - (r.top + r.height / 2)) / innerHeight, -0.5, 0.5);
    frame.style.setProperty("--ry", `${x * 10}deg`);
    frame.style.setProperty("--rx", `${-y * 8}deg`);
  });
  hero.addEventListener("pointerleave", () => {
    frame.style.setProperty("--ry", "0deg");
    frame.style.setProperty("--rx", "0deg");
  });
}

// ================================================================== hero
function scenes() {
  // each scene is a different street of the same rainy town: where the fire, the lamp and the big windows stand
  const LOOK = {
    hero: { seed: 11, fire: 0.66, lamp: 0.535, faces: 0.9, moon: 0.62 },
    interlude: { seed: 23, fire: 0.34, lamp: 0.74, faces: 0.56, moon: 0.2, street: 0.84, tall: 0.78 },
    footer: { seed: 5, fire: 0.7, lamp: 0.18, faces: 0.42, moon: 0.62, street: 0.82 },
  };
  const made = $$("[data-scene]").map((n) => nightScene(n, LOOK[n.dataset.scene] || {}));
  loadImage(new URL(`assets/${SITE.faces.file}`, location.href).href)
    .then((im) => made.forEach((m) => m.faces(im)))
    .catch(() => {});
}

function stats() {
  const m = SITE.metrics;
  const gpu = SPEEDS?.devices?.find((d) => d.kind === "gpu");
  const cpu = SPEEDS?.devices?.find((d) => d.kind === "cpu");
  const rows = [
    gpu
      ? { to: gpu.instant_s, from: cpu?.instant_s || 10, dec: 2, unit: " s", label: `a face, made in this page on a 2020 laptop's GPU${cpu ? ` (${cpu.instant_s.toFixed(0)} s on its processor alone)` : ""}` }
      : { to: SITE.speed.student_ms / 1000, from: SITE.speed.teacher_s, dec: 2, unit: " s", label: "per face in Instant mode, on a laptop GPU" },
    { to: m.fid_after, from: m.fid_before, dec: 2, unit: "", label: `realism gap, lower is better (the first version: ${m.fid_before.toFixed(1)})` },
    { to: m.n_attributes, from: 0, dec: 0, unit: "", label: "features you can ask for, or rule out" },
    { to: Math.round(m.memorised * m.memo_n), from: 0, dec: 0, unit: ` / ${m.memo_n.toLocaleString("en-US")}`, label: "invented faces that copy a real person's photo" },
  ];
  const dl = $(".stats");
  rows.forEach((r) => {
    const d = el("div", "stat");
    d.innerHTML = `<dt><span class="n">${r.to.toFixed(r.dec)}</span><small>${esc(r.unit)}</small></dt><dd>${esc(r.label)}</dd>`;
    dl.append(d);
    dl.addEventListener("reveal", () => countTo($(".n", d), r.to, { from: r.from, dec: r.dec, dur: 2000 }), { once: true });
  });
}

async function portrait() {
  const cv = $(".portrait-canvas");
  const ctx = fit(cv, 1.5, 720);
  const heroes = SITE.preset.hero;
  const sheets = [];
  const step = $(".hud-step b");
  const left = $(".hud-noise b");
  const meter = $(".hud-meter i");
  sheets[0] = await loadImage(heroes[0].src);
  heroes.slice(1).forEach((h, i) => loadImage(h.src).then((im) => (sheets[i + 1] = im)).catch(() => {}));
  const FRAME = 85; // ms per drawing step
  const HOLD = 2600;
  const OUT = 700;
  const cycle = heroes[0].n * FRAME + HOLD + OUT;
  const t0 = performance.now();
  let lastKey = "";
  const show = (now) => {
    const t = reduced ? heroes[0].n * FRAME + 10 : now - t0;
    let k = Math.floor(t / cycle) % heroes.length;
    if (!sheets[k]) k = 0;
    const h = heroes[k];
    const u = t % cycle;
    const i = Math.min(h.n - 1, Math.floor(u / FRAME));
    const after = i + 1 < h.n ? h.noise_left[i + 1] : 0; // static left once this step is done
    let noise = u < h.n * FRAME ? after : 0;
    if (u > h.n * FRAME + HOLD) noise = easeInOut((u - h.n * FRAME - HOLD) / OUT);
    const key = `${k}:${i}:${noise.toFixed(3)}`;
    if (key === lastKey) return;
    lastKey = key;
    drawFrame(ctx, sheets[k], h, i);
    if (noise > 0.001) {
      ctx.globalAlpha = noise;
      ctx.drawImage(TILES[Math.floor(now / 70) % TILES.length], 0, 0, cv.width, cv.height);
      ctx.globalAlpha = 1;
    }
    step.textContent = String(i + 1).padStart(2, "0");
    left.textContent = pct(noise);
    meter.style.transform = `scaleX(${1 - noise})`;
  };
  animate(cv, show, 30);
}

function reel() {
  const dlg = $(".reel");
  const video = $(".reel-video");
  $$("[data-reel]").forEach((b) =>
    b.addEventListener("click", () => {
      dlg.showModal();
      video.play().catch(() => {});
    }),
  );
  const close = () => {
    video.pause();
    dlg.close();
  };
  $(".reel-close").addEventListener("click", close);
  dlg.addEventListener("click", (e) => e.target === dlg && close());
  dlg.addEventListener("close", () => video.pause());
}

function ticker() {
  const track = $(".ticker-track");
  const f = SITE.faces;
  const rows = Math.ceil(f.n / f.cols);
  track.style.setProperty("--sheet", FACES_URL);
  track.style.setProperty("--bs", `${f.cols * 100}% ${rows * 100}%`);
  const words = ["Not a real person", "Invented by AI", "Nobody here exists", "Watermarked", "Measured, not claimed"];
  const make = () =>
    words.map((w, j) => {
      const k = (j * 13 + 7) % f.n;
      const item = el("span", `ticker-item${j % 2 ? " fill" : ""}`);
      const face = el("span", "ticker-face");
      face.style.setProperty("--bp", `${((k % f.cols) / (f.cols - 1)) * 100}% ${(Math.floor(k / f.cols) / (rows - 1)) * 100}%`);
      item.append(document.createTextNode(w), face);
      return item;
    });
  track.append(...make(), ...make());
}

// ================================================================== how it works: scroll scrubs the drawing
async function how() {
  const sec = $(".how");
  const cv = $(".how-canvas");
  const ctx = fit(cv, 1.5, 900);
  const h = SITE.preset.hero[0];
  const story = SITE.edit_story;
  const [sheet, edits] = await Promise.all([loadImage(h.src), loadImage(story.sheet)]);
  const steps = $$(".how-steps li");
  const count = $(".how-count b");
  const label = $(".how-label");
  const rail = $(".how-rail i");
  const tags = $(".how-tags");
  const NAMES = { Eyeglasses: "Glasses", Smiling: "Smile", Gray_Hair: "Grey hair", Bangs: "Bangs" };
  const order = story.probe.slice(1).map((p) => p.on[p.on.length - 1]);
  const tagEls = order.map((name, j) => {
    const t = el("span", "how-tag", `${esc(NAMES[name] || name)} <b>${pct(story.probe[j + 1].probe[name])}</b>`);
    tags.append(t);
    return t;
  });
  const guess = $(".how-guess");
  const gctx = fit($("canvas", guess), 1.5, 320);
  const leftOut = $(".how-left b");
  // static left after each step (the teacher's time grid); the last step leaves none
  const after = h.noise_left.map((_, s) => (s + 1 < h.n ? h.noise_left[s + 1] : 0));
  // Scroll spends its length where the picture changes: most of the static goes in the last few steps, so a step's
  // share of the scroll follows how much picture it reveals (the square root evens out the early, nearly-static steps).
  const shown = after.map((a) => Math.sqrt(1 - a));
  const stepAt = (u) => {
    let s = 0;
    while (s < h.n - 1 && shown[s] < u) s++;
    return s;
  };
  let last = -1;
  const draw = (now) => {
    const r = sec.getBoundingClientRect();
    const p = clamp(-r.top / Math.max(1, r.height - innerHeight));
    const q = Math.round(p * 1000);
    if (q === last && !reduced) return;
    last = q;
    let phase;
    let noise = 0;
    let lbl;
    let n;
    let draft = -1;
    rail.style.transform = `scaleX(${p})`;
    if (p < 0.08) {
      phase = 0;
      n = 0;
      noise = 1;
      drawFrame(ctx, sheet, h, 0);
      lbl = "pure noise";
    } else if (p < 0.6) {
      phase = 1;
      const s = stepAt((p - 0.08) / 0.5);
      n = s + 1;
      noise = after[s];
      draft = s;
      drawFrame(ctx, sheet, h, s);
      lbl = noise > 0.6 ? "mostly static" : noise > 0 ? "a face appears" : "finished";
    } else if (p < 0.7) {
      phase = 2;
      n = h.n;
      drawFrame(ctx, sheet, h, h.n - 1);
      lbl = "not a real person";
    } else {
      phase = 3;
      n = h.n;
      const e = ((p - 0.7) / 0.3) * (story.n - 1);
      const a = Math.floor(e);
      const b = Math.min(story.n - 1, a + 1);
      const mix = easeInOut(clamp((e - a) * 2.2 - 0.6));
      if (a === 0 && e < 0.3) drawFrame(ctx, sheet, h, h.n - 1);
      drawFrame(ctx, edits, story, a, a === 0 && e < 0.3 ? easeInOut(e / 0.3) : 1);
      if (mix > 0) drawFrame(ctx, edits, story, b, mix);
      const on = Math.round(e);
      tagEls.forEach((t, j) => t.classList.toggle("on", j < on));
      lbl = on ? `+ ${(NAMES[order[on - 1]] || order[on - 1]).toLowerCase()}` : "no extras";
    }
    if (phase < 3) tagEls.forEach((t) => t.classList.remove("on"));
    if (noise > 0.001) {
      ctx.globalAlpha = noise;
      ctx.drawImage(TILES[(q >> 2) % TILES.length], 0, 0, cv.width, cv.height);
      ctx.globalAlpha = 1;
    }
    // the inset: what the AI currently thinks the finished face is (its draft at this step, without the static)
    guess.classList.toggle("on", draft >= 0 && noise > 0.02);
    if (draft >= 0) drawFrame(gctx, sheet, h, draft);
    count.textContent = String(n).padStart(2, "0");
    label.textContent = lbl;
    leftOut.textContent = pct(noise);
    steps.forEach((li, j) => li.classList.toggle("on", j === phase));
  };
  animate(sec, draw, 60);
}

// ================================================================== segmented controls
function segmented(root, onChange) {
  const buttons = $$("button", root);
  const pill = $(".seg-pill", root);
  const place = () => {
    const on = buttons.find((b) => b.getAttribute("aria-checked") === "true") || buttons[0];
    pill.style.setProperty("--x", `${on.offsetLeft}px`);
    pill.style.setProperty("--w", `${on.offsetWidth}px`);
  };
  buttons.forEach((b) =>
    b.addEventListener("click", () => {
      buttons.forEach((x) => x.setAttribute("aria-checked", String(x === b)));
      place();
      onChange?.(b.dataset.value);
    }),
  );
  root.addEventListener("keydown", (e) => {
    const i = buttons.findIndex((b) => b.getAttribute("aria-checked") === "true");
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const b = buttons[(i + d + buttons.length) % buttons.length];
    b.click();
    b.focus();
  });
  new ResizeObserver(place).observe(root);
  place();
  return () => buttons.find((b) => b.getAttribute("aria-checked") === "true")?.dataset.value;
}

// ================================================================== 01 make faces
function studio() {
  const quality = segmented($('.seg[data-name="quality"]'));
  const count = segmented($('.seg[data-name="count"]'));
  const chips = $(".chips");
  const grid = $(".faces");
  const statusLine = $(".results-status");
  const tag = $(".results-tag");
  const weakNote = $(".weak-note");
  const button = $(".btn-make");
  const NEXT = { any: "yes", yes: "no", no: "any" };
  const WORD = { any: "", yes: "must have", no: "must not have" };
  SITE.features.forEach((f) => {
    const c = el("button", "chip");
    c.type = "button";
    c.dataset.state = "any";
    c.dataset.feature = f;
    c.textContent = f;
    c.setAttribute("aria-label", f);
    c.addEventListener("click", () => {
      c.dataset.state = NEXT[c.dataset.state];
      c.setAttribute("aria-label", `${f}${WORD[c.dataset.state] ? `: ${WORD[c.dataset.state]}` : ""}`);
    });
    chips.append(c);
  });

  const NOT = { Man: "not a man", Smile: "no smile", "Open mouth": "mouth closed" };
  const card = (face, i, live) => {
    const fig = el("figure", "face");
    fig.style.setProperty("--i", i);
    const box = el("div", "face-img");
    const img = el("img");
    img.alt = "An invented face (not a real person)";
    box.append(img, staticLayer());
    const file = live ? face.png : face.src;
    if (file) {
      const a = el("a", "face-dl", ICON.dl);
      a.href = file;
      a.download = `humandit-face-${i + 1}.${live ? "png" : "jpg"}`;
      a.setAttribute("aria-label", "Download this face (it carries the invisible watermark)");
      box.append(a);
    }
    const cap = el("figcaption", "face-cap");
    const marks = face.marks || [];
    if (marks.length)
      marks.forEach((m) => {
        const text = m.want ? m.feature.toLowerCase() : NOT[m.feature] || `no ${m.feature.toLowerCase()}`;
        cap.append(el("span", `mark${m.ok ? "" : " bad"}`, `${m.ok ? ICON.yes : ICON.no}${esc(text)}`));
      });
    else {
      cap.append(el("span", "mark label", "sees"));
      const seen = face.sees?.length ? face.sees.slice(0, 3) : ["none of the 12"];
      seen.forEach((f) => cap.append(el("span", "mark sees", esc(f.toLowerCase()))));
    }
    fig.append(box, cap);
    return { fig, img, src: live ? face.png : face.src };
  };

  const placeholders = (n) => {
    grid.dataset.count = n;
    grid.replaceChildren();
    const out = [];
    for (let i = 0; i < n; i++) {
      const fig = el("figure", "face");
      fig.style.setProperty("--i", i);
      const box = el("div", "face-img");
      box.append(staticLayer());
      fig.append(box, el("figcaption", "face-cap"));
      grid.append(fig);
      out.push(fig);
    }
    requestAnimationFrame(() => out.forEach((f) => f.classList.add("in")));
  };

  const show = async (faces, live) => {
    grid.dataset.count = faces.length;
    const cards = faces.map((f, i) => card(f, i, live));
    // decode every picture before anything moves, so the faces develop together rather than pop in one by one
    await Promise.all(cards.map((c) => loadImage(c.src).then((im) => (c.img.src = im.src)).catch(() => {})));
    grid.replaceChildren(...cards.map((c) => c.fig));
    requestAnimationFrame(() => {
      cards.forEach((c) => c.fig.classList.add("in"));
      cards.forEach((c, i) => setTimeout(() => c.fig.classList.add("dev"), reduced ? 0 : 240 + i * 90));
    });
  };

  const pre = SITE.preset.studio;
  show(pre.faces, false);
  statusLine.innerHTML = `Made <b>${pre.faces.length} faces</b> in <b>${pre.seconds.toFixed(2)} s</b>, Instant mode.`;
  tag.textContent = madeEarlier(pre.hardware);
  tag.classList.remove("live");

  // saved mode: faces from the bank whose checker verdicts match the ticks (exact matches first, not the same faces
  // again soon); the marks under each face are the checker's real verdicts, misses included
  const recent = new Set();
  const pick = (b, n, q, must, never) => {
    const pool = b.faces[q] || b.faces.instant;
    const hits = (f) => must.filter((m) => f.sees.includes(m)).length + never.filter((x) => !f.sees.includes(x)).length;
    const want = must.length + never.length;
    const shuffled = pool.map((f) => [Math.random(), f]).sort((x, y) => x[0] - y[0]).map((x) => x[1]);
    const full = shuffled.filter((f) => hits(f) === want);
    let chosen = [...full.filter((f) => !recent.has(f.src)), ...full.filter((f) => recent.has(f.src))].slice(0, n);
    const exact = chosen.length;
    if (chosen.length < n)
      chosen = chosen.concat(shuffled.filter((f) => hits(f) < want).sort((x, y) => hits(y) - hits(x)).slice(0, n - exact));
    if (recent.size > pool.length / 2) recent.clear();
    chosen.forEach((f) => recent.add(f.src));
    const faces = chosen.map((f) => ({
      src: f.src,
      sees: f.sees,
      marks: [
        ...must.map((m) => ({ feature: m, want: true, ok: f.sees.includes(m) })),
        ...never.map((x) => ({ feature: x, want: false, ok: !f.sees.includes(x) })),
      ],
    }));
    return { faces, exact, pool: pool.length, matching: full.length, want };
  };

  let busy = false;
  button.addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    button.classList.add("busy");
    const n = Number(count());
    const q = quality();
    const must = $$(".chip", chips).filter((c) => c.dataset.state === "yes").map((c) => c.dataset.feature);
    const never = $$(".chip", chips).filter((c) => c.dataset.state === "no").map((c) => c.dataset.feature);
    const previous = [...grid.children];
    const prevCount = grid.dataset.count;
    placeholders(n);
    weakNote.hidden = true;
    const making = paused ? `Picking ${n} saved face${n > 1 ? "s" : ""}…` : `Making ${n} face${n > 1 ? "s" : ""}…`;
    const stop = ticking(statusLine, making);
    const off = () => {};
    try {
      const r = await engine.call("make", { count: n, quality: q, must, never, seed: -1 }, {
        onProgress: (p) => stop.label(progressWords(p, q === "best" ? "Cleaning up: step" : "Drawing face")),
      });
      off();
      stop();
      await show(r.faces, true);
      statusLine.innerHTML = `Made <b>${r.faces.length} face${r.faces.length > 1 ? "s" : ""}</b> in <b class="secs">0.00 s</b> on ${esc(r.hardware)}, ${q === "best" ? "best quality" : "Instant mode"}.`;
      countTo($(".secs", statusLine), r.seconds, { dec: 2, dur: 900, suffix: " s" });
      tag.textContent = "Live · made on this device";
      tag.classList.add("live");
      if (r.weak?.length) {
        weakNote.textContent = `Instant mode is weak at ${r.weak.join(", ").toLowerCase()}: measured, it draws them in under half the faces. Try Best quality.`;
        weakNote.hidden = false;
      }
    } catch (err) {
      off();
      const b = await bank();
      if (!b) {
        stop();
        grid.dataset.count = prevCount;
        grid.replaceChildren(...previous);
        statusLine.textContent = err.message;
        toast(err.message);
        return;
      }
      await wait(650);
      stop();
      fellBack(err);
      const r = pick(b, n, q, must, never);
      await show(r.faces, false);
      const kind = q === "best" ? "best-quality" : "Instant";
      statusLine.innerHTML =
        r.want === 0
          ? `Picked <b>${r.faces.length}</b> of the <b>${r.pool}</b> ${kind} faces the AI made earlier.`
          : r.exact === r.faces.length
            ? `Picked <b>${r.faces.length}</b> of the <b>${r.matching}</b> saved ${kind} faces the checker says match.`
            : `Only <b>${r.matching}</b> of the ${r.pool} saved ${kind} faces match everything you asked; the rest match most of it.`;
      tag.textContent = `Saved · ${madeEarlier(b.made_on).toLowerCase()}`;
      tag.classList.remove("live");
      const weak = q === "instant" ? must.filter((m) => SITE.weak.includes(m)) : [];
      if (weak.length) {
        weakNote.textContent = `Instant mode is weak at ${weak.join(", ").toLowerCase()}: measured, it draws them in under half the faces. Try Best quality.`;
        weakNote.hidden = false;
      }
    } finally {
      busy = false;
      button.classList.remove("busy");
    }
  });
}

// ================================================================== 02 change one thing
function editor() {
  const box = $(".compare");
  const a = $(".compare-a");
  const b = $(".compare-b");
  const pills = $(".edit-pills");
  const status = $(".edit-status");
  const mw = $(".m-without");
  const mwith = $(".m-with");
  const vw = $(".v-without");
  const vwith = $(".v-with");
  const lblR = $(".compare-lbl.r");
  const btn = $(".btn-edit");
  const presets = new Map(SITE.preset.edits.map((e) => [e.feature, e]));
  let current = SITE.preset.edits[0];
  let touched = false;
  box.append(staticLayer());
  box.classList.add("gone-static");

  const setP = (p) => {
    const w = box.clientWidth;
    box.style.setProperty("--p", `${p}%`);
    box.style.setProperty("--px", `${(p / 100) * w}px`);
    box.setAttribute("aria-valuenow", Math.round(p));
  };
  new ResizeObserver(() => setP(parseFloat(box.style.getPropertyValue("--p")) || 50)).observe(box);
  setP(50);

  const render = async (e, live) => {
    const [ia, ib] = await Promise.all([loadImage(e.without), loadImage(e.with)]);
    a.src = ia.src;
    b.src = ib.src;
    a.alt = `The invented person without ${e.feature.toLowerCase()}`;
    b.alt = `The same invented person with ${e.feature.toLowerCase()}`;
    lblR.textContent = `With ${e.feature.toLowerCase()}`;
    box.classList.remove("swap");
    void box.offsetWidth;
    box.classList.add("swap");
    const [p0, , p2] = e.checker || [0, 0, 0];
    mw.style.transform = `scaleX(${p0})`;
    mwith.style.transform = `scaleX(${p2})`;
    vw.textContent = pct(p0);
    vwith.textContent = pct(p2);
    status.textContent = live
      ? `Live · ${hwLabel(e.hardware)} · Instant · ${e.seconds.toFixed(2)} s`
      : `${madeEarlier(e.made_on || SITE.preset.made_on)}${e.of ? ` · person ${e.k + 1} of ${e.of}` : ""}`;
    current = e;
  };

  const pill = (name, live) => {
    const p = el("button", `pill${live ? " live-only" : ""}`, esc(name));
    p.type = "button";
    p.setAttribute("role", "radio");
    p.setAttribute("aria-checked", String(name === current.feature));
    p.addEventListener("click", () => {
      $$(".pill", pills).forEach((x) => x.setAttribute("aria-checked", String(x === p)));
      if (presets.has(name) && !live) render(presets.get(name), false);
      else ask(name, -1, true);
    });
    pills.append(p);
  };
  SITE.preset.edits.forEach((e) => pill(e.feature, false));
  // saved mode: the bank has every feature, so every feature gets a pill
  if (paused)
    bank().then((b) =>
      Object.keys(b?.edits || {})
        .filter((f) => !presets.has(f) && b.edits[f].length)
        .forEach((f) => pill(f, false)),
    );
  engine.onStatus((s) => {
    if (s.state !== "ready" || pills.dataset.full) return;
    pills.dataset.full = "1";
    (s.info?.features || SITE.features)
      .filter((f) => !presets.has(f) && f !== "Man")
      .forEach((f) => pill(f, true));
  });

  let busy = false;
  const turn = {}; // saved mode: which saved person each feature is showing
  const ask = async (feature, seed, first = false) => {
    if (busy) return;
    busy = true;
    btn.classList.add("busy");
    box.classList.remove("gone-static");
    const stop = ticking(status, paused ? "Finding another saved person…" : "Drawing the same person twice…");
    try {
      const r = await engine.call("edit", { feature_name: feature, quality: "instant", seed });
      stop();
      await render(r, true);
    } catch (err) {
      const list = (await bank())?.edits?.[feature];
      if (!list?.length) {
        stop();
        status.textContent = err.message;
        toast(err.message);
        return;
      }
      await wait(450);
      stop();
      fellBack(err);
      const k = first && turn[feature] == null ? 0 : ((turn[feature] ?? -1) + 1) % list.length;
      turn[feature] = k;
      await render({ ...list[k], feature, made_on: BANK.made_on, k, of: list.length }, false);
    } finally {
      busy = false;
      btn.classList.remove("busy");
      box.classList.add("gone-static");
    }
  };
  btn.addEventListener("click", () => ask(current.feature, -1));

  // drag, click, keys
  const fromEvent = (e) => {
    const r = box.getBoundingClientRect();
    return clamp((e.clientX - r.left) / r.width, 0, 1) * 100;
  };
  let dragging = false;
  box.addEventListener("pointerdown", (e) => {
    dragging = touched = true;
    box.setPointerCapture(e.pointerId);
    setP(fromEvent(e));
  });
  box.addEventListener("pointermove", (e) => dragging && setP(fromEvent(e)));
  box.addEventListener("pointerup", () => (dragging = false));
  box.addEventListener("keydown", (e) => {
    const p = parseFloat(box.style.getPropertyValue("--p")) || 50;
    const d = { ArrowLeft: -5, ArrowRight: 5, Home: -100, End: 100 }[e.key];
    if (d == null) return;
    e.preventDefault();
    touched = true;
    setP(clamp(p + d, 0, 100));
  });
  // the first time it comes into view, the line sweeps once each way to show there are two pictures
  box.addEventListener("reveal", () => {
    if (reduced) return;
    const keys = [50, 82, 18, 50];
    const t0 = performance.now() + 500;
    const stop = animate(null, (now) => {
      if (touched) return stop();
      const t = (now - t0) / 900;
      if (t < 0) return;
      const k = Math.min(keys.length - 2, Math.floor(t));
      setP(lerp(keys[k], keys[k + 1], easeInOut(clamp(t - k))));
      if (t >= keys.length - 1) stop();
    });
  }, { once: true });
  render(current, false);
}

// ================================================================== 03 blend
function blender() {
  const stage = $(".blend-stage");
  const cv = $(".blend-canvas");
  const ctx = fit(cv, 1.5, 800);
  const ca = $(".blend-a").getContext("2d");
  const cb = $(".blend-b").getContext("2d");
  const range = $(".blend-range");
  const status = $(".blend-status");
  const btn = $(".btn-blend");
  let meta = SITE.preset.blend;
  let sheet = null;
  let touchedAt = -1e9;
  let last = -1;
  let noise = 0;

  const paint = (v) => {
    if (!sheet) return;
    const f = (v / 1000) * (meta.n - 1);
    const i = Math.floor(f);
    drawFrame(ctx, sheet, meta, i);
    if (f - i > 0.001) drawFrame(ctx, sheet, meta, i + 1, f - i);
    if (noise > 0.001) {
      ctx.globalAlpha = noise;
      ctx.drawImage(TILES[Math.floor(performance.now() / 70) % TILES.length], 0, 0, cv.width, cv.height);
      ctx.globalAlpha = 1;
    }
    range.style.setProperty("--fill", `${v / 10}%`);
  };
  const load = async (m, live) => {
    const im = await loadImage(m.src);
    meta = m;
    sheet = im;
    drawFrame(ca, sheet, meta, 0);
    drawFrame(cb, sheet, meta, meta.n - 1);
    status.textContent = live
      ? `Live · ${hwLabel(m.hardware)} · Instant · ${m.n} faces in ${m.seconds.toFixed(2)} s`
      : `${madeEarlier(m.hardware)} · best quality${m.of ? ` · blend ${m.k + 1} of ${m.of}` : ""}`;
    last = -1;
  };
  range.addEventListener("input", () => {
    touchedAt = performance.now();
    paint(Number(range.value));
  });
  load(meta, false);
  const t0 = performance.now();
  animate(stage, (now) => {
    const idle = now - touchedAt > 3500;
    if (idle && !reduced) {
      // ping-pong A -> B -> A, easing at each end, while nobody is holding the slider
      const u = ((now - t0) / 7000) % 1;
      range.value = Math.round(1000 * easeInOut(u < 0.5 ? u * 2 : 2 - u * 2));
    }
    const v = Number(range.value);
    if (v !== last || noise > 0) {
      last = v;
      paint(v);
    }
  }, 30);

  let busy = false;
  let turn = -1;
  btn.addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    btn.classList.add("busy");
    const stop = ticking(status, paused ? "Loading another saved blend…" : "Inventing two people and 22 in between…");
    const fade = animate(null, () => (noise = Math.min(0.85, noise + 0.06)), 30);
    try {
      const r = await engine.call("blend", { quality: "instant", seed_a: -1, seed_b: -1, frames: 24 }, {
        onProgress: (p) => stop.label(progressWords(p, "Drawing face")),
      });
      stop();
      fade();
      await load(r, true);
      const out = animate(null, () => {
        noise = Math.max(0, noise - 0.05);
        if (!noise) out();
      }, 30);
    } catch (err) {
      const list = (await bank())?.blends;
      if (!list?.length) {
        stop();
        fade();
        noise = 0;
        status.textContent = err.message;
        toast(err.message);
        return;
      }
      await wait(500);
      stop();
      fade();
      fellBack(err);
      turn = (turn + 1) % list.length;
      await load({ ...list[turn], k: turn, of: list.length }, false);
      const out = animate(null, () => {
        noise = Math.max(0, noise - 0.05);
        if (!noise) out();
      }, 30);
    } finally {
      busy = false;
      btn.classList.remove("busy");
    }
  });
}

// ================================================================== 04 instant or best: the race
function race() {
  const box = $(".race");
  const lanes = $(".lanes");
  const hw = $(".race-hw");
  const btn = $(".btn-race");
  let stopRace = null;

  const build = (data, live, now = false) => {
    stopRace?.();
    const runs = data.runs;
    const tmax = Math.max(...runs.map((r) => r.seconds));
    const fids = runs.filter((r) => r.fid != null).map((r) => r.fid);
    const fmax = Math.max(...fids);
    const fbest = Math.min(...fids);
    hw.textContent = `${live ? "Live" : "Timed earlier"} on ${hwLabel(data.hardware)}`;
    lanes.replaceChildren();
    const rows = runs.map((r) => {
      const name = r.steps === 1 ? "Instant" : `Best quality`;
      const sub = r.steps === 1 ? "one step, a second AI" : `${r.steps} clean-up steps`;
      const lane = el("div", "lane");
      const star = r.fid_n && r.fid_n !== 50000 ? "*" : "";
      lane.innerHTML = `
        <div class="lane-img"><img alt="${esc(name)}, ${esc(sub)}"></div>
        <div class="lane-name"><b>${esc(name)}</b><span>${esc(sub)}</span></div>
        <div class="lane-bars"><div class="track"><i></i></div>${r.fid != null ? `<div class="track gap${r.fid === fbest ? " best" : ""}"><i></i></div>` : ""}</div>
        <div class="lane-nums"><span class="t">0.00 s</span><small>${r.fid != null ? `gap ${r.fid.toFixed(1)}${star}` : ""}</small></div>`;
      $("img", lane).src = r.img;
      lanes.append(lane);
      return { r, lane, bar: $(".track i", lane), gap: $(".gap i", lane), t: $(".t", lane) };
    });
    if (fids.length && runs.some((r) => r.fid_n && r.fid_n !== 50000)) {
      lanes.append(el("p", "small", "*Measured on 10,000 faces, which reads a little higher than 50,000."));
    }
    const go = () => {
      const t0 = performance.now() + 250;
      rows.forEach((x) => {
        x.lane.classList.remove("done");
        x.bar.style.transform = "scaleX(0)";
        if (x.gap) x.gap.style.transform = "scaleX(0)";
      });
      const end = tmax * 1000;
      stopRace = animate(null, (now) => {
        const ms = Math.max(0, now - t0);
        rows.forEach((x) => {
          const dur = x.r.seconds * 1000;
          const done = ms >= dur;
          const k = Math.min(1, ms / dur);
          x.bar.style.transform = `scaleX(${(k * x.r.seconds) / tmax})`;
          x.t.textContent = `${(Math.min(ms, dur) / 1000).toFixed(2)} s`;
          if (done && !x.lane.classList.contains("done")) {
            x.lane.classList.add("done");
            if (x.gap) {
              x.gap.style.transition = "transform 1s cubic-bezier(.16,1,.3,1)";
              x.gap.style.transform = `scaleX(${x.r.fid / fmax})`;
            }
          }
        });
        if (ms > end + 50) stopRace();
      });
    };
    if (reduced) {
      rows.forEach((x) => {
        x.lane.classList.add("done");
        x.bar.style.transform = `scaleX(${x.r.seconds / tmax})`;
        x.t.textContent = `${x.r.seconds.toFixed(2)} s`;
        if (x.gap) x.gap.style.transform = `scaleX(${x.r.fid / fmax})`;
      });
    } else if (isOnScreen(box) || live || now) go();
    else {
      const io = new IntersectionObserver((es) => {
        if (es.some((e) => e.isIntersecting)) {
          io.disconnect();
          go();
        }
      }, { threshold: 0.35 });
      io.observe(box);
    }
  };
  animate(box, () => {}, 1); // registers the box with the on-screen watcher
  build(SITE.preset.compare, false);

  let busy = false;
  let turn = -1;
  btn.addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    btn.classList.add("busy");
    const was = hw.textContent;
    try {
      const r = await engine.call("compare", { seed: -1, feature_name: "" }, {
        onProgress: (p) => (hw.textContent = progressWords(p, "Racing: step")),
      });
      build(r, true);
    } catch (err) {
      hw.textContent = was;
      const list = (await bank())?.compares;
      if (!list?.length) return toast(err.message);
      fellBack(err);
      turn = (turn + 1) % list.length;
      build(list[turn], false, true); // a saved race, replayed at the speeds it was timed at
    } finally {
      busy = false;
      btn.classList.remove("busy");
    }
  });
}

// ================================================================== 05 is it AI-made?
function checker() {
  const drop = $(".drop");
  const input = $(".drop-input");
  const img = $(".drop-img");
  const empty = $(".drop-empty");
  const stamp = $(".stamp");
  const verdict = $(".verdict");
  const title = $(".verdict-title");
  const body = $(".verdict-body");
  let busy = false;

  const toDataUrl = (blob) =>
    new Promise((res, rej) => {
      const r = new FileReader();
      r.onload = () => res(r.result);
      r.onerror = rej;
      r.readAsDataURL(blob);
    });
  // big photos are shrunk to 1024 px on this device before sending: the watermark is read at 256 px anyway
  const prepare = async (blob) => {
    const url = await toDataUrl(blob);
    if (blob.size < 3 * 2 ** 20) return url;
    const im = await loadImage(url);
    const s = Math.min(1, 1024 / Math.max(im.naturalWidth, im.naturalHeight));
    const c = document.createElement("canvas");
    c.width = Math.round(im.naturalWidth * s);
    c.height = Math.round(im.naturalHeight * s);
    c.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.95);
  };

  // saved: the engine's earlier verdict on each example, shown (and labelled) when the engine can't answer now
  const saved = SITE.preset.check || {};
  const show = (found, note) => {
    stamp.textContent = found ? "AI-made · HumanDiT" : "No watermark";
    stamp.classList.add(found ? "yes" : "no", "show");
    verdict.classList.add(found ? "yes" : "no");
    title.textContent = found ? "Yes: this picture carries the HumanDiT watermark." : "No HumanDiT watermark found.";
    body.textContent =
      (found
        ? "It was made by this AI. The mark is invisible and survives resizing and compression."
        : "This picture was not made here, or it was heavily altered.") + (note ? ` ${note}` : "");
  };

  const check = async (blob, example) => {
    if (busy) return;
    if (!blob.type.startsWith("image/")) return toast("That file is not a picture.");
    busy = true;
    const url = await prepare(blob);
    img.src = url;
    img.hidden = false;
    empty.hidden = true;
    stamp.className = "stamp";
    verdict.className = "verdict";
    title.textContent = "Looking for the watermark…";
    body.textContent = "Reading the hidden signal in the picture's pixels.";
    drop.style.setProperty("--h", `${drop.clientHeight - 3}px`);
    drop.classList.add("scanning");
    const t0 = performance.now();
    try {
      const r = await engine.call("check", { image: blob }, {
        onProgress: (p) =>
          p.phase &&
          (body.textContent =
            p.phase === "preparing"
              ? "Preparing the watermark reader…"
              : `Downloading the watermark reader (once): ${Math.round(p.got / 1e6)} of ${Math.round(p.total / 1e6)} MB…`),
      });
      await new Promise((res) => setTimeout(res, Math.max(0, 1100 - (performance.now() - t0))));
      show(r.found);
    } catch (err) {
      if (example in saved) {
        await new Promise((res) => setTimeout(res, Math.max(0, 1100 - (performance.now() - t0))));
        show(saved[example], `(Checked earlier: ${err.message})`);
      } else {
        title.textContent = "Couldn't check this picture right now.";
        body.textContent = err.message;
      }
    } finally {
      drop.classList.remove("scanning");
      busy = false;
    }
  };

  input.addEventListener("change", () => input.files[0] && check(input.files[0]));
  ["dragenter", "dragover"].forEach((t) =>
    drop.addEventListener(t, (e) => {
      e.preventDefault();
      drop.classList.add("over");
    }),
  );
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (f) check(f);
  });
  $$(".example").forEach((b) =>
    b.addEventListener("click", async () => {
      const blob = await (await fetch(b.dataset.src)).blob();
      check(blob, b.dataset.src);
    }),
  );
}

// ================================================================== 06 the numbers
function numbers() {
  $$(".arch-lines path").forEach((p, i) => p.style.setProperty("--i", i));
  $$(".arch-nodes g").forEach((g, i) => g.style.setProperty("--i", i));
  const rows = $(".score-rows");
  SITE.scoreboard.forEach((r, i) => {
    const best = r.fid === Math.min(...SITE.scoreboard.map((x) => x.fid));
    const row = el("div", `srow${best ? " best" : r.use ? " live" : ""}`);
    row.style.setProperty("--i", i);
    row.innerHTML = `<span class="srow-name">${esc(r.name)}${r.use ? `<em>${esc(r.use)}</em>` : ""}<small>${esc(r.tech)}</small></span>
      <div class="track"><i></i></div><b>${r.fid.toFixed(2)}</b>`;
    rows.append(row);
  });
  $(".score").addEventListener("reveal", () =>
    $$(".srow", rows).forEach((row, i) => ($(".track i", row).style.transform = `scaleX(${SITE.scoreboard[i].w})`)),
  );
  const m = SITE.metrics;
  const facts = [
    ["Realism gap, Best quality (FID-50k)", m.fid_after.toFixed(2)],
    ["Realism gap, Instant (FID-50k)", m.fid_gan.toFixed(1)],
    ["Precision / recall", `${m.precision.toFixed(2)} / ${m.recall.toFixed(2)}`],
    ["Features drawn as asked", pct(m.control_mean)],
    ["Teacher parameters", `${m.params_m}M`],
    ["Copies of a training photo", `${Math.round(m.memorised * m.memo_n)} / ${m.memo_n.toLocaleString("en-US")}`],
  ];
  const dl = $(".fact-list");
  facts.forEach(([k, v]) => dl.append(el("div", "fact", `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`)));
  const links = (window.HD_CONFIG || {}).links || {};
  $(".links").hidden = !Object.values(links).some(Boolean);
  $$("[data-link]").forEach((a) => {
    const href = links[a.dataset.link];
    if (href) {
      a.href = href;
      a.hidden = false;
      a.target = "_blank";
      a.rel = "noopener";
    }
  });
}

// ================================================================== live mode: the AI on this device
const SPEEDS_URL = "assets/speeds.json";
let SPEEDS = null;
let YOU = null; // {name, s}: this device's measured seconds per Instant face, once known

/** Faces a minute, per device measured on the maker's laptop, plus this device once it has run. */
function ladder(root) {
  const rows = [...(SPEEDS?.devices || []).map((d) => ({ name: d.name, sub: d.kind === "cpu" ? "processor only" : d.kind === "igpu" ? "built-in graphics" : "graphics card", s: d.instant_s }))];
  if (YOU) rows.unshift({ name: `This device: ${YOU.name}`, sub: "measured just now", s: YOU.s, you: true });
  const best = Math.max(...rows.map((r) => 60 / r.s));
  root.replaceChildren(
    ...rows.map((r) => {
      const n = el("div", `lrow${r.you ? " you" : ""}`);
      const perMin = 60 / r.s;
      n.innerHTML = `<div class="lrow-name"><b>${esc(r.name)}</b><small>${esc(r.sub)}</small></div>
        <div class="lrow-track"><i></i></div>
        <div class="lrow-num">${perMin >= 10 ? Math.round(perMin) : perMin.toFixed(1)}<small>${r.s.toFixed(r.s < 10 ? 2 : 1)} s a face</small></div>`;
      requestAnimationFrame(() => requestAnimationFrame(() => ($(".lrow-track i", n).style.transform = `scaleX(${Math.max(0.02, perMin / best)})`)));
      return n;
    }),
  );
}

async function live() {
  const root = document.documentElement;
  const pill = $(".live-pill");
  const pillText = $(".engine-text", pill);
  const dlg = $("dialog.live");
  const go = $(".live-go", dlg);
  const goText = $("span", go);
  const devName = $(".live-device-name", dlg);
  const devNote = $(".live-device-note", dlg);
  const size = $(".live-size", dlg);
  const bar = $(".live-bar i", dlg);
  const prog = $(".live-progress", dlg);
  const progText = $(".live-progress-text", dlg);
  const strip = $(".gpu-strip-text");
  try {
    YOU = JSON.parse(sessionStorage.getItem("hd-you") || "null");
  } catch {}
  $$(".ladder-rows").forEach((r) => ladder(r));

  const onLive = () => {
    root.classList.toggle("is-live", engine.live());
    pill.dataset.state = engine.status.state;
    pillText.textContent = engine.status.text;
    if (engine.live()) {
      $$("[data-saved]").forEach((n) => n.dataset.live && (n.textContent = n.dataset.live));
      strip.innerHTML = `<b>Live on ${esc(engine.status.hardware)}.</b> Every face below is made on this device; nothing is uploaded.`;
    }
  };
  engine.onStatus(onLive);
  onLive();

  const sizes = { core: await engine.packSize("core"), best: await engine.packSize("best") };
  const describe = async () => {
    const d = await engine.device();
    const cached = await engine.cached("core");
    if (d.gpu) {
      devName.textContent = `Your graphics chip: ${d.label}`;
      devNote.textContent = "It runs through WebGPU, which works on NVIDIA, AMD, Intel and Apple graphics alike.";
    } else {
      devName.textContent = `No WebGPU in this browser: your processor (${d.threads} thread${d.threads > 1 ? "s" : ""})`;
      devNote.textContent =
        "It still works, on every core, but expect seconds a face, not a fraction of one. Chrome, Edge and Safari 26 have WebGPU.";
    }
    if (d.mobile) devNote.textContent += " On a phone it may run out of memory: a laptop or desktop is the better place for it.";
    size.textContent = !sizes.core
      ? "This copy of the site has no models to run."
      : cached
        ? "Already on this device: it loads in seconds."
        : `Downloads ${sizes.core} MB once and keeps it in this browser. Best quality adds ${sizes.best} MB the first time you use it.`;
    go.disabled = !sizes.core;
  };

  const ready = () => {
    goText.textContent = "Start making faces";
    go.dataset.done = "1";
    progText.textContent = YOU ? `Live on ${engine.status.hardware}: ${YOU.s.toFixed(2)} s a face.` : `Live on ${engine.status.hardware}.`;
    bar.style.transform = "scaleX(1)";
  };

  const open = () => {
    if (!dlg.open) dlg.showModal();
    describe();
    $$(".live-ladder", dlg).forEach((r) => ladder(r));
    if (engine.live()) ready();
  };
  $$("[data-golive]").forEach((b) => b.addEventListener("click", open));
  $(".live-close", dlg).addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => e.target === dlg && dlg.close());

  go.addEventListener("click", async () => {
    if (go.dataset.done) {
      dlg.close();
      $("#studio").scrollIntoView({ behavior: reduced ? "auto" : "smooth" });
      return;
    }
    go.disabled = true;
    prog.hidden = false;
    progText.textContent = "Starting…";
    try {
      await engine.goLive((p) => {
        bar.style.transform = `scaleX(${p.total ? p.got / p.total : 0})`;
        progText.textContent =
          p.phase === "preparing"
            ? "Preparing the AI for your device (compiling it for your graphics chip)…"
            : p.phase === "reading"
              ? `Reading from this browser's storage: ${Math.round(p.got / 1e6)} of ${Math.round(p.total / 1e6)} MB`
              : p.phase === "ready"
                ? "Ready."
                : `Downloading: ${Math.round(p.got / 1e6)} of ${Math.round(p.total / 1e6)} MB`;
      });
      // a speed test on this device: one face to warm up (the GPU compiles its programs), two timed
      progText.textContent = "Timing this device: making three faces…";
      await engine.call("make", { count: 1, quality: "instant", seed: 7 });
      const r = await engine.call("make", { count: 2, quality: "instant", seed: 8 });
      const d = await engine.device();
      YOU = { name: d.gpu ? d.label : `processor, ${d.threads} threads`, s: r.seconds / 2 };
      try {
        sessionStorage.setItem("hd-you", JSON.stringify(YOU));
      } catch {}
      $$(".ladder-rows").forEach((x) => ladder(x));
      ready();
    } catch (err) {
      progText.textContent = err.message;
      goText.textContent = "Try again";
    } finally {
      go.disabled = false;
    }
  });
}

// ================================================================== start
async function main() {
  // saved mode: sentences that promise live drawing swap to their saved-mode wording before anything shows
  $$("[data-saved]").forEach((n) => ((n.dataset.live = n.textContent), (n.textContent = n.dataset.saved)));
  $$("[data-split]").forEach(split);
  $$("[data-magnetic]").forEach((b) => magnetic(b));
  document.documentElement.style.setProperty("--grain", `url(${grain()})`);
  STATIC = staticStrip(128);
  document.documentElement.style.setProperty("--static", `url(${STATIC})`);
  TILES = noiseTiles(6, 256);
  chrome();
  reel();
  polish();
  let introDone;
  try {
    introDone = intro();
  } catch (err) {
    console.error("[humandit] intro:", err);
    introDone = Promise.resolve();
  }
  // whatever happens, the intro never stays over the page
  setTimeout(() => $(".intro").classList.add("gone"), 2500);
  try {
    [SITE, SPEEDS] = await Promise.all([
      fetch("assets/site.json").then((r) => r.json()),
      fetch(SPEEDS_URL).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    ]);
    engine.configure({ weak: SITE.weak || [], fids: SITE.preset.compare.runs });
  } catch (err) {
    toast("This page could not load its pictures. Try reloading.");
    throw err;
  }
  scenes();
  stats();
  ticker();
  // rAF sleeps in a background tab, so never wait on the intro for long
  await Promise.race([introDone, new Promise((r) => setTimeout(r, 1600))]);
  reveal();
  const neon = $(".display .neon");
  if (neon && !reduced) setTimeout(() => neon.classList.add("flicker"), 700);
  const sections = [portrait, how, studio, editor, blender, race, checker, numbers, live];
  for (const s of sections) {
    try {
      await s();
    } catch (err) {
      console.error(`[humandit] ${s.name}:`, err);
    }
  }
  engine.resume();
}

main();
