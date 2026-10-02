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
  isOnScreen,
  lerp,
  loadImage,
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
    : p.what === "finish"
      ? `Finishing: face ${p.k} of ${p.of} (decoding and watermarking)…`
      : p.redraw
      ? `The checker sent ${p.redraw} face${p.redraw > 1 ? "s" : ""} back · redrawing: ${p.k} of ${p.of}…`
      : `${doing} ${p.k} of ${p.of}…`;

/**
 * While the AI is still starting, a status line asks people to wait (and says how far along it is) rather than
 * pretending to work; once it runs, the line goes back to `then`. Returns off().
 */
function pleaseWait(stop, then) {
  if (!engine.starting()) return () => {};
  const say = (s) => stop.label(`Please wait until the AI is running · ${s.text.replace(/^Please wait · /, "")}…`);
  say(engine.status);
  const off = engine.onStatus((s) => {
    if (s.state === "loading") return say(s);
    stop.label(then);
    off();
  });
  return off;
}

/** While the AI works, a button reads "Cancel" (and cancels); returns restore(). */
function asCancel(button) {
  const was = button.innerHTML;
  button.classList.add("cancelling");
  button.innerHTML = `<span>Cancel</span>${ICON.no}`;
  button.setAttribute("aria-label", "Cancel");
  return () => {
    button.classList.remove("cancelling");
    button.innerHTML = was;
    button.removeAttribute("aria-label");
  };
}
const cancelled = (err) => err?.message === "Cancelled.";

/** How often Instant mode drew a feature when asked, from the saved faces (e.g. "0 of 16"), or "". */
async function instantRate(f) {
  const b = await bank();
  const asked = (b?.faces?.instant || []).filter((x) => x.asked === f);
  return asked.length ? `${asked.filter((x) => x.sees.includes(f)).length} of ${asked.length}` : "";
}
const listWords = (xs) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`).toLowerCase();

// ================================================================== chrome: nav, progress, active section
function chrome() {
  const pill = $(".live-pill");
  const hero = $(".hero");
  let away = false;
  const place = () => pill.classList.toggle("show", away || engine.starting());
  new IntersectionObserver(([e]) => ((away = !e.isIntersecting), place()), { threshold: 0.15 }).observe(hero);
  engine.onStatus(place);
}

// ================================================================== pointer-driven polish
function polish() {
  // section kickers draw their line when the heading arrives
  const heads = new IntersectionObserver(
    (es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), heads.unobserve(e.target))),
    { rootMargin: "0px 0px -15% 0px" },
  );
  $$("[data-reveal-head], .how-copy").forEach((h) => heads.observe(h));
}

// ================================================================== hero
function scenes() {
  // two streets of the same rainy town: where the fire, the lamp and the tall house stand (its windows just lit;
  // the photo-real faces stay out of the pixel art)
  const LOOK = {
    interlude: { seed: 23, fire: 0.34, lamp: 0.74, faces: 0.56, moon: 0.2, street: 0.8, tall: 0.86 },
    footer: { seed: 5, fire: 0.7, lamp: 0.18, faces: 0.42, moon: 0.62, street: 0.82 },
  };
  $$("[data-scene]").forEach((n) => nightScene(n, LOOK[n.dataset.scene] || {}));
}

function stats() {
  const m = SITE.metrics;
  const gpu = SPEEDS?.devices?.find((d) => d.kind === "gpu");
  const cpu = SPEEDS?.devices?.find((d) => d.kind === "cpu");
  const rows = [
    gpu
      ? { to: gpu.instant_s, from: cpu?.instant_s || 10, dec: 2, unit: " s", label: "a face, in this page, on a 2020 laptop GPU" }
      : { to: SITE.speed.student_ms / 1000, from: SITE.speed.teacher_s, dec: 2, unit: " s", label: "a face on a laptop GPU" },
    { to: m.fid_after, from: m.fid_before, dec: 2, unit: "", label: `realism gap (FID), down from ${m.fid_before.toFixed(0)} in the first version` },
    { to: m.n_attributes, from: 0, dec: 0, unit: "", label: "features to ask for or rule out" },
    { to: Math.round(m.memorised * m.memo_n), from: 0, dec: 0, unit: ` / ${m.memo_n.toLocaleString("en-US")}`, label: "faces that copy a real photo" },
  ];
  const dl = $(".stats");
  rows.forEach((r) => {
    const d = el("div", "stat");
    d.innerHTML = `<dt><span class="n">${r.to.toFixed(r.dec)}</span><small>${esc(r.unit)}</small></dt><dd>${esc(r.label)}</dd>`;
    dl.append(d);
    dl.addEventListener("reveal", () => countTo($(".n", d), r.to, { from: r.from, dec: r.dec, dur: 2000 }), { once: true });
  });
}

// ================================================================== hero: one large face, forming from static
// Inspired by thispersondoesnotexist.com: one person, large, and another on request. Each saved face plays the AI's
// real drafts (assets/hero.json: its clean-picture guess at each of 25 steps) from static to the finished face; in live
// mode, "Another person" makes a new face on the visitor's own device instead.
async function stage() {
  const fig = $(".stage");
  const cv = $(".stage-canvas");
  const who = $(".stage-who");
  const next = $(".stage-next");
  const amb = $$(".stage-amb");
  const data = await fetch("assets/hero.json")
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);
  const faces = data?.faces?.length ? data.faces : SITE.preset.hero;
  const madeOn = data?.made_on || SITE.preset.made_on;
  const sheets = new Map();
  const sheet = (k) => {
    if (!sheets.has(k)) sheets.set(k, loadImage(faces[k].src).catch(() => null));
    return sheets.get(k);
  };
  let dirty = true;
  let ctx = fit(cv, 2, 1100);
  new ResizeObserver(() => {
    ctx = fit(cv, 2, 1100);
    dirty = true;
  }).observe(cv);

  // what is on screen: a sheet of drafts (saved) or one picture (live), and where the animation is
  let shown = { img: await sheet(0), meta: faces[0], live: false };
  let k = 0;
  let phase = reduced ? "hold" : "draw"; // out (to static), draw (drafts), reveal (live: static clears), hold
  let t0 = performance.now();
  let touched = false; // after a press, no more automatic changes
  let incoming = null; // the next face, while static covers the old one
  const DRAW = 2200;
  const OUT = 420;
  const REVEAL = 700;
  const HOLD = 6500;
  sheet(1);

  const caption = (f, live) => {
    who.textContent = live
      ? `Made just now on ${f.hardware}, in ${f.seconds.toFixed(2)} s.`
      : `Drawn by this AI from seed ${f.seed}, step by step, on a ${madeOn}.`;
  };
  caption(faces[0], false);

  // the soft colour behind the face: the face itself, tiny and smoothed, cross-faded when the face changes
  let ambOn = 0;
  const ambient = (src, meta) => {
    const c = amb[1 - ambOn];
    const a = c.getContext("2d");
    // a soft blur every browser draws alike (Safari's canvas has no filter): down to 16 pixels, smoothed back up
    const t = (ambient.tiny ||= Object.assign(document.createElement("canvas"), { width: 16, height: 16 })).getContext("2d");
    t.imageSmoothingQuality = a.imageSmoothingQuality = "high";
    if (meta) t.drawImage(src, ((meta.n - 1) % meta.cols) * meta.size, Math.floor((meta.n - 1) / meta.cols) * meta.size, meta.size, meta.size, 0, 0, 16, 16);
    else t.drawImage(src, 0, 0, 16, 16);
    a.clearRect(0, 0, c.width, c.height);
    a.drawImage(ambient.tiny, 0, 0, c.width, c.height);
    amb[ambOn].classList.remove("on");
    c.classList.add("on");
    ambOn = 1 - ambOn;
  };

  const noise = (alpha, now) => {
    if (alpha <= 0.002) return;
    ctx.globalAlpha = alpha;
    ctx.drawImage(TILES[Math.floor(now / 70) % TILES.length], 0, 0, cv.width, cv.height);
    ctx.globalAlpha = 1;
  };
  const final = (s) => (s.meta ? drawFrame(ctx, s.img, s.meta, s.meta.n - 1) : ctx.drawImage(s.img, 0, 0, cv.width, cv.height));

  const frame = $(".stage-frame");
  const go = async (live) => {
    // static covers the face at once; the next one is fetched (or made) meanwhile
    phase = "out";
    frame.classList.remove("drift");
    t0 = performance.now();
    next.classList.remove("timing");
    if (live) {
      who.textContent = engine.starting() ? "Please wait until the AI is running…" : "Making a new face on this device…";
      const r = await engine.call("make", { count: 1, quality: "instant", seed: -1 });
      const img = await loadImage(r.faces[0].png);
      incoming = { img, meta: null, live: true, info: { hardware: r.hardware, seconds: r.seconds } };
    } else {
      let img = null;
      for (let tries = 0; !img && tries < faces.length; tries++) {
        k = (k + 1) % faces.length;
        img = await sheet(k);
      }
      sheet((k + 1) % faces.length);
      incoming = img ? { img, meta: faces[k], live: false } : shown; // nothing loads: the same face comes back
    }
  };

  const press = async () => {
    touched = true;
    if (phase === "out") return;
    engine.prepare();
    const live = engine.live() || engine.starting();
    next.classList.add("busy");
    try {
      await go(live);
    } catch (err) {
      // live failed or was cancelled: the next saved face instead
      if (err?.message !== "Cancelled.") toast(`${err.message} Showing a face the AI made earlier.`);
      await go(false);
    } finally {
      next.classList.remove("busy");
    }
  };
  next.addEventListener("click", press);
  frame.addEventListener("click", press);
  frame.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), press()));

  animate(fig, (now) => {
    const t = now - t0;
    if (phase === "out") {
      final(shown);
      noise(easeOut(clamp(t / OUT)), now);
      if (t >= OUT && incoming) {
        shown = incoming;
        incoming = null;
        phase = shown.live ? "reveal" : "draw";
        t0 = now;
        if (shown.live) caption(shown.info, true);
        else caption(shown.meta, false);
      }
      return;
    }
    if (phase === "draw") {
      const m = shown.meta;
      const u = clamp(t / DRAW);
      const f = easeInOut(u) * (m.n - 1);
      const i = Math.floor(f);
      drawFrame(ctx, shown.img, m, i);
      if (f - i > 0.001) drawFrame(ctx, shown.img, m, i + 1, f - i);
      const left = lerp(m.noise_left[i], m.noise_left[Math.min(m.n - 1, i + 1)], f - i);
      noise(left * (1 - clamp((u - 0.82) / 0.18)), now);
      if (u >= 1) {
        phase = "hold";
        t0 = now;
        if (!reduced) frame.classList.add("drift");
        ambient(shown.img, m);
        if (!touched) next.classList.add("timing");
      }
      return;
    }
    if (phase === "reveal") {
      final(shown);
      noise(1 - easeOut(clamp(t / REVEAL)), now);
      if (t >= REVEAL) {
        phase = "hold";
        t0 = now;
        if (!reduced) frame.classList.add("drift");
        ambient(shown.img, null);
      }
      return;
    }
    // hold: draw once, then wait (automatic changes stop once someone presses)
    if (dirty) {
      final(shown);
      dirty = false;
    }
    if (!touched && !reduced && t > HOLD) go(false);
  }, 60);
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
  const quality = segmented($('.seg[data-name="quality"]'), () => hint());
  const count = segmented($('.seg[data-name="count"]'));
  const chips = $(".chips");
  const grid = $(".faces");
  const statusLine = $(".results-status");
  const tag = $(".results-tag");
  const weakNote = $(".weak-note");
  const button = $(".btn-make");
  const NEXT = { any: "yes", yes: "no", no: "any" };
  const mustNow = () => $$(".chip", chips).filter((c) => c.dataset.state === "yes").map((c) => c.dataset.feature);
  // asking Instant mode for a feature it can't draw: say before the press that it will use Best quality
  const hint = async () => {
    const weak = quality() === "instant" ? mustNow().filter((f) => SITE.weak.includes(f)) : [];
    if (!weak.length) return (weakNote.hidden = true);
    const rates = (await Promise.all(weak.map(instantRate))).filter(Boolean);
    const them = weak.length > 1 ? "them" : "it";
    weakNote.textContent = `${listWords(weak).replace(/^./, (c) => c.toUpperCase())} ${weak.length > 1 ? "use" : "uses"} Best quality: Instant mode can't draw ${them}${rates.length ? ` (in a test it drew ${them} in ${rates.join(", ")} faces)` : ""}. Best quality is slower.`;
    weakNote.hidden = false;
  };
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
      hint();
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
    if (busy) return engine.cancel();
    busy = true;
    engine.prepare();
    const restore = engine.live() || engine.starting() ? asCancel(button) : (button.classList.add("busy"), () => {});
    const n = Number(count());
    const must = mustNow();
    // features Instant mode can't draw are made in Best quality (live, and from the saved Best faces)
    const q = quality() === "instant" && must.some((f) => SITE.weak.includes(f)) ? "best" : quality();
    const never = $$(".chip", chips).filter((c) => c.dataset.state === "no").map((c) => c.dataset.feature);
    const previous = [...grid.children];
    const prevCount = grid.dataset.count;
    placeholders(n);
    const making = paused && !engine.starting() ? `Picking ${n} saved face${n > 1 ? "s" : ""}…` : `Making ${n} face${n > 1 ? "s" : ""}…`;
    const stop = ticking(statusLine, making);
    const off = pleaseWait(stop, making);
    try {
      const r = await engine.call("make", { count: n, quality: q, must, never, seed: -1 }, {
        onProgress: (p) => stop.label(progressWords(p, q === "best" ? "Cleaning up: step" : "Drawing face")),
      });
      off();
      stop();
      await show(r.faces, true);
      const back = r.tries - r.faces.length;
      statusLine.innerHTML =
        `Made <b>${r.faces.length} face${r.faces.length > 1 ? "s" : ""}</b> in <b class="secs">0.00 s</b> on ${esc(r.hardware)}, ${r.quality === "best" ? "Best quality" : "Instant mode"}.` +
        (back > 0 ? ` The checker sent ${back} back to be redrawn.` : "") +
        (r.misses ? ` ${r.misses} still miss${r.misses > 1 ? "" : "es"} something after every try: the marks say what.` : "");
      countTo($(".secs", statusLine), r.seconds, { dec: 2, dur: 900, suffix: " s" });
      tag.textContent = "Live · made on this device";
      tag.classList.add("live");
    } catch (err) {
      off();
      if (cancelled(err)) {
        stop();
        grid.dataset.count = prevCount;
        grid.replaceChildren(...previous);
        statusLine.textContent = "Cancelled. The faces before are back.";
        return;
      }
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
    } finally {
      busy = false;
      button.classList.remove("busy");
      restore();
    }
  });
}

// ================================================================== 02 instant or best: the race
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
    if (busy) return engine.cancel();
    busy = true;
    engine.prepare();
    const restore = engine.live() || engine.starting() ? asCancel(btn) : (btn.classList.add("busy"), () => {});
    const was = hw.textContent;
    const stop = ticking(hw, "Racing…");
    const off = pleaseWait(stop, "Racing…");
    if (!engine.starting()) stop();
    try {
      const r = await engine.call("compare", { seed: -1, feature_name: "" }, {
        onProgress: (p) => {
          stop();
          hw.textContent = progressWords(p, "Racing: step");
        },
      });
      off();
      stop();
      build(r, true);
    } catch (err) {
      off();
      stop();
      hw.textContent = cancelled(err) ? "Cancelled." : was;
      if (cancelled(err)) return;
      const list = (await bank())?.compares;
      if (!list?.length) return toast(err.message);
      fellBack(err);
      turn = (turn + 1) % list.length;
      build(list[turn], false, true); // a saved race, replayed at the speeds it was timed at
    } finally {
      busy = false;
      btn.classList.remove("busy");
      restore();
    }
  });
}

// ================================================================== 03 is it AI-made?
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

/**
 * Where this browser keeps the downloaded AI on disk. Browsers store a site's data inside their own profile folder,
 * in their own format (not the Downloads folder), so this names the folder; removing goes through the browser.
 */
function storagePlace() {
  const b = engine.browser();
  const os = engine.os();
  const ff = location.origin.replace("://", "+++").replace(/:(\d+)$/, "+$1");
  const W = "C:\\Users\\<you>\\AppData";
  const PATHS = {
    "windows Chrome": `${W}\\Local\\Google\\Chrome\\User Data\\Default\\Service Worker\\CacheStorage`,
    "windows Edge": `${W}\\Local\\Microsoft\\Edge\\User Data\\Default\\Service Worker\\CacheStorage`,
    "windows Brave": `${W}\\Local\\BraveSoftware\\Brave-Browser\\User Data\\Default\\Service Worker\\CacheStorage`,
    "windows Firefox": `${W}\\Roaming\\Mozilla\\Firefox\\Profiles\\<profile>\\storage\\default\\${ff}\\cache`,
    "mac Chrome": "~/Library/Application Support/Google/Chrome/Default/Service Worker/CacheStorage",
    "mac Edge": "~/Library/Application Support/Microsoft Edge/Default/Service Worker/CacheStorage",
    "mac Brave": "~/Library/Application Support/BraveSoftware/Brave-Browser/Default/Service Worker/CacheStorage",
    "mac Firefox": `~/Library/Application Support/Firefox/Profiles/<profile>/storage/default/${ff}/cache`,
    "linux Chrome": "~/.config/google-chrome/Default/Service Worker/CacheStorage",
    "linux Edge": "~/.config/microsoft-edge/Default/Service Worker/CacheStorage",
    "linux Brave": "~/.config/BraveSoftware/Brave-Browser/Default/Service Worker/CacheStorage",
    "linux Firefox": `~/.mozilla/firefox/<profile>/storage/default/${ff}/cache`,
  };
  const path = PATHS[`${os} ${b}`] || "";
  const name = b || "your browser";
  const profile = /Default/.test(path) ? ' "Default" is the first browser profile; others are "Profile 1", "Profile 2" and so on.' : "";
  const how =
    b === "Firefox"
      ? "the padlock left of the address bar › Clear cookies and site data"
      : b === "Safari"
        ? "Safari › Settings › Privacy › Manage Website Data › search for this site › Remove"
        : "the icon left of the address bar › Site settings › Delete data";
  return { name, path, profile, how };
}

/** Windows hands a browser the built-in graphics unless told otherwise: the steps to move it to the faster card. */
/**
 * On Windows, Chrome and Edge give every page the graphics chip the browser itself runs on, usually the built-in one
 * on a laptop, and ignore a page's request for the faster one. One browser flag changes it (Chrome's WebGPU guide).
 */
const FLAG = { Chrome: "chrome", Edge: "edge", Brave: "brave", Opera: "opera" };
const switchSteps = (name) => {
  const card = SPEEDS?.devices?.find((d) => d.kind === "gpu");
  const chip = SPEEDS?.devices?.find((d) => d.kind === "igpu");
  const x = card && chip ? Math.round(chip.instant_s / card.instant_s) : 0;
  const scheme = FLAG[engine.browser()];
  const url = scheme ? `${scheme}://flags/#force-high-performance-gpu` : "";
  return `<p class="live-tip-title">This is the built-in graphics. Got an NVIDIA or AMD card too?</p>
  <p class="small">${esc(name)} gives every website the chip it runs on itself, and a website can't pick another.${x > 1 ? ` On the laptop measured below, the graphics card is ${x} times faster.` : ""} To use yours:</p>
  ${url ? `<ol class="small"><li>Copy this into the address bar: <span class="flag"><code>${url}</code><button class="flag-copy" type="button" data-copy="${url}">Copy</button></span></li><li>Set <b>Force High Performance GPU</b> to <b>Enabled</b>, then press <b>Restart</b>.</li><li>Come back: this box will name your graphics card.</li></ol>
  <p class="small">Or in Windows: Settings › System › Display › Graphics › ${esc(name)} › High performance.</p>`
  : `<ol class="small"><li>Open <b>Settings › System › Display › Graphics</b>.</li><li>Choose <b>${esc(name)}</b> › <b>Options</b> › <b>High performance</b> › <b>Save</b>.</li><li>Restart ${esc(name)} and come back here.</li></ol>`}`;
};

async function live() {
  const root = document.documentElement;
  const pill = $(".live-pill");
  const pillText = $(".engine-text", pill);
  const dlg = $("dialog.live");
  const go = $(".live-go", dlg);
  const goText = $("span", go);
  const devName = $(".live-device-name", dlg);
  const devNote = $(".live-device-note", dlg);
  const tip = $(".live-tip", dlg);
  const size = $(".live-size", dlg);
  const waitBox = $(".live-wait", dlg);
  const waitClock = $(".live-wait-clock", dlg);
  const bar = $(".live-bar i", dlg);
  const progText = $(".live-progress-text", dlg);
  const steps = Object.fromEntries($$(".live-steps li", dlg).map((li) => [li.dataset.step, li]));
  const storeWhat = $(".live-store-what", dlg);
  const storePath = $(".live-path", dlg);
  const storeHow = $(".live-store-how", dlg);
  const removeBtn = $(".live-remove", dlg);
  const strip = $(".gpu-strip-text");
  let stripWasText = strip.innerHTML;
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
    } else if (engine.starting()) {
      strip.innerHTML = `<b>Please wait until it's running.</b> ${esc(engine.status.text.replace(/^Please wait · /, ""))}…`;
    } else {
      $$("[data-saved]").forEach((n) => (n.textContent = n.dataset.saved));
      strip.innerHTML = stripWasText;
    }
  };
  engine.onStatus(onLive);
  onLive();
  const early = async () => {
    const d = await engine.device();
    if (!d.switchable) return;
    stripWasText = `<b>Runs on your GPU.</b> This browser is on the built-in ${esc(d.name)}. If this computer also has an NVIDIA or AMD graphics card, switch the browser to it for faster faces.`;
    if (!engine.live() && !engine.starting()) strip.innerHTML = stripWasText;
  };
  setTimeout(() => ("requestIdleCallback" in window ? requestIdleCallback(early, { timeout: 2000 }) : early()), 1200);

  const sizes = { core: await engine.packSize("core"), best: await engine.packSize("best") };
  const where = storagePlace();
  const store = async () => {
    const [used, has] = await Promise.all([engine.stored(), engine.cached("core")]);
    const mb = used ? Math.round(used / 1e6) : 0;
    storeWhat.textContent =
      has || engine.live()
        ? `On this computer now: ${mb ? `${mb} MB` : "the AI"}, inside ${where.name}'s own storage for this website (not your Downloads folder).`
        : `When you switch it on, ${sizes.core} MB goes into ${where.name}'s own storage for this website, not your Downloads folder. Nothing else on your computer changes.`;
    storePath.textContent = where.path;
    storePath.hidden = !where.path;
    const inside = /CacheStorage$/.test(where.path)
      ? `This site's part is a sub-folder there with a long code for a name, in ${where.name}'s own format: don't delete it by hand.`
      : `The files there are in ${where.name}'s own format: don't delete them by hand.`;
    storeHow.textContent = `${where.path ? `${inside}${where.profile} ` : ""}To remove it, press the button below, or in ${where.name}: ${where.how}.`;
    removeBtn.hidden = !(has || engine.live() || mb > 50);
    removeBtn.dataset.mb = mb;
  };

  const describe = async () => {
    const d = await engine.device();
    const cached = await engine.cached("core");
    if (d.gpu) {
      devName.textContent = `Your graphics chip: ${d.name}${d.builtIn ? " (built into the processor)" : ""}`;
      devNote.textContent = "It runs through WebGPU, which works on NVIDIA, AMD, Intel and Apple graphics alike.";
    } else {
      devName.textContent = `No WebGPU in this browser: your processor (${d.threads} thread${d.threads > 1 ? "s" : ""})`;
      devNote.textContent =
        "It still works, on every core, but expect seconds a face, not a fraction of one. Chrome, Edge and Safari 26 have WebGPU.";
    }
    tip.hidden = !d.switchable;
    if (!tip.hidden) {
      tip.innerHTML = switchSteps(where.name);
      $$("[data-copy]", tip).forEach((c) =>
        c.addEventListener("click", () =>
          navigator.clipboard.writeText(c.dataset.copy).then(() => (c.textContent = "Copied"), () => (c.textContent = "Select and copy")),
        ),
      );
    }
    size.textContent = !sizes.core
      ? "This copy of the site has no models to run."
      : cached
        ? "Already on this computer: it starts in seconds."
        : `Downloads ${sizes.core} MB once. Best quality adds ${sizes.best} MB the first time you use it.`;
    go.disabled = !sizes.core || engine.starting();
    await store();
  };

  // the start-up, step by step: download (or read back), prepare for the chip, a short speed test
  const stepState = (name, state, info = "") => {
    const li = steps[name];
    li.dataset.state = state;
    $(".ls-info", li).textContent = info;
  };
  let clock = null;
  const waiting = (on) => {
    waitBox.hidden = !on;
    clock?.();
    clock = null;
    if (!on) return;
    const t0 = performance.now();
    clock = animate(null, (now) => {
      const s = Math.floor((now - t0) / 1000);
      waitClock.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }, 4);
  };

  const ready = () => {
    waiting(false);
    goText.textContent = "Start making faces";
    go.dataset.done = "1";
    go.disabled = false;
    size.textContent = YOU ? `Running on ${engine.status.hardware}: ${YOU.s.toFixed(2)} s a face.` : `Running on ${engine.status.hardware}.`;
    store();
  };

  const open = (e) => {
    if (!dlg.open) dlg.showModal();
    describe();
    $$(".live-ladder", dlg).forEach((r) => ladder(r));
    if (engine.live()) ready();
    if (e?.currentTarget?.dataset?.store != null) $(".live-store", dlg).scrollIntoView({ block: "nearest" });
  };
  $$("[data-golive]").forEach((b) => b.addEventListener("click", open));
  $(".live-close", dlg).addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => e.target === dlg && dlg.close());

  // a return visit loads by itself (engine.resume): the steps follow it if the dialog is opened meanwhile
  engine.onStatus((s) => {
    if (go.dataset.running) return;
    if (s.state === "loading" && s.progress) {
      waiting(true);
      show(s.progress);
    } else if (engine.live() && dlg.open && !go.dataset.done) ready();
  });
  const show = (p) => {
    const got = Math.round(p.got / 1e6);
    const total = Math.round(p.total / 1e6);
    if (p.phase === "preparing" && p.got < p.total) {
      // one part has arrived and is being prepared while the rest still comes: the download step stays current
      stepState("download", "doing", `${got} of ${total} MB`);
      bar.style.transform = `scaleX(${p.got / p.total})`;
      progText.textContent = p.toFetch
        ? "Downloading the AI, and preparing each part as it arrives."
        : "Reading the AI from this browser's storage, and preparing each part.";
    } else if (p.phase === "downloading" || p.phase === "reading") {
      stepState("download", "doing", p.phase === "reading" ? `already here, reading ${got} of ${total} MB` : `${got} of ${total} MB`);
      bar.style.transform = `scaleX(${p.total ? p.got / p.total : 0})`;
      progText.textContent = p.phase === "reading" ? "Reading the AI from this browser's storage." : "Downloading the AI. It's kept, so this happens once.";
    } else if (p.phase === "preparing") {
      stepState("download", "done", `${total} MB`);
      stepState("prepare", "doing");
      bar.style.transform = "scaleX(1)";
      progText.textContent = "Preparing the AI for your graphics chip. This can take up to a minute, and the page may pause for a moment.";
    } else if (p.phase === "ready") {
      stepState("download", "done", `${total} MB`);
      stepState("prepare", "done");
    }
  };

  go.addEventListener("click", async () => {
    if (go.dataset.running) {
      engine.cancel();
      return;
    }
    if (go.dataset.done) {
      dlg.close();
      $("#studio").scrollIntoView({ behavior: reduced ? "auto" : "smooth" });
      return;
    }
    go.dataset.running = "1";
    goText.textContent = "Cancel";
    Object.keys(steps).forEach((k) => stepState(k, "todo"));
    stepState("download", "doing");
    bar.style.transform = "scaleX(0)";
    progText.textContent = "Starting…";
    waiting(true);
    waitBox.scrollIntoView({ block: "nearest", behavior: reduced ? "auto" : "smooth" });
    try {
      await engine.goLive(show);
      // a speed test on this device: one face to warm up (the chip compiles its programs), two timed
      stepState("prepare", "done");
      stepState("test", "doing", "face 1 of 3");
      progText.textContent = "Timing this device: the first face is slower while the chip warms up.";
      await engine.call("make", { count: 1, quality: "instant", seed: 7 });
      stepState("test", "doing", "faces 2 and 3 of 3");
      const r = await engine.call("make", { count: 2, quality: "instant", seed: 8 });
      const d = await engine.device();
      YOU = { name: d.label, s: r.seconds / 2 };
      try {
        sessionStorage.setItem("hd-you", JSON.stringify(YOU));
      } catch {}
      stepState("test", "done", `${YOU.s.toFixed(2)} s a face`);
      $$(".ladder-rows").forEach((x) => ladder(x));
      ready();
    } catch (err) {
      if (engine.live()) return ready(); // cancelled during the speed test: it runs, just untimed
      waiting(false);
      waitBox.hidden = false;
      progText.textContent = cancelled(err) ? "Cancelled. What already downloaded is kept, so starting again picks up from there." : err.message;
      goText.textContent = cancelled(err) ? "Download and run" : "Try again";
      go.disabled = false;
    } finally {
      delete go.dataset.running;
    }
  });

  // remove: press twice (the second press says how much it frees)
  removeBtn.addEventListener("click", async () => {
    if (!removeBtn.dataset.sure) {
      removeBtn.dataset.sure = "1";
      removeBtn.textContent = `Press again to remove it${Number(removeBtn.dataset.mb) ? ` (${removeBtn.dataset.mb} MB)` : ""}`;
      setTimeout(() => {
        delete removeBtn.dataset.sure;
        removeBtn.textContent = "Remove the AI from this computer";
      }, 5000);
      return;
    }
    delete removeBtn.dataset.sure;
    removeBtn.disabled = true;
    try {
      await engine.remove();
      YOU = null;
      try {
        sessionStorage.removeItem("hd-you");
      } catch {}
      delete go.dataset.done;
      goText.textContent = "Download and run";
      $$(".ladder-rows").forEach((x) => ladder(x));
      await describe();
      storeWhat.textContent = `Removed. ${storeWhat.textContent.replace(/^When you switch it on/, "If you switch it on again")}`;
      toast("The AI was removed from this computer. The page shows saved results again.");
    } catch (err) {
      toast(`Couldn't remove it: ${err.message}`);
    } finally {
      removeBtn.disabled = false;
      removeBtn.textContent = "Remove the AI from this computer";
    }
  });
}

// ================================================================== start
async function main() {
  // a phone, a tablet, or a computer without a graphics chip the AI can use: js/gate.js shows its own screen instead
  const gate = window.HDGate;
  if (gate?.gated) return;
  addEventListener("hd-gated", () => engine.cancel());
  // saved mode: sentences that promise live drawing swap to their saved-mode wording before anything shows
  $$("[data-saved]").forEach((n) => ((n.dataset.live = n.textContent), (n.textContent = n.dataset.saved)));
  $$("[data-split]").forEach(split);
  STATIC = staticStrip(128);
  document.documentElement.style.setProperty("--static", `url(${STATIC})`);
  TILES = noiseTiles(6, 256);
  chrome();
  polish();
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
  if (gate && !(await gate.ready)) return;
  scenes();
  stats();
  reveal();
  const sections = [stage, how, studio, race, checker, numbers, live];
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
