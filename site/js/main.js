// HumanDiT website: every section starts from results the AI made earlier (assets/site.json, labelled as such) and
// switches to live results from the engine when a visitor asks for something.

import * as engine from "./engine.js";
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
const hwLabel = (hw) =>
  /zerogpu/i.test(hw) ? "a shared cloud GPU (Hugging Face ZeroGPU)" : /cpu/i.test(hw) ? "a CPU" : hw;
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
  const off = engine.onStatus((s) => s.state === "waking" && (text = WAKING));
  const done = () => (stop(), off());
  done.label = (t) => (text = t);
  return done;
}
const WAKING = "Waking the GPU server (the first request after a quiet spell can take a minute)…";
const ordinal = (n) => `${n}${["th", "st", "nd", "rd"][n % 100 > 10 && n % 100 < 14 ? 0 : n % 10 < 4 ? n % 10 : 0]}`;
const inLine = (pos) => `In line for a GPU: ${ordinal(pos)}…`;

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
  const nav = $(".nav");
  const bar = $(".nav-progress i");
  let lastY = scrollY;
  let queued = false;
  const update = () => {
    queued = false;
    const y = scrollY;
    const max = document.documentElement.scrollHeight - innerHeight;
    bar.style.transform = `scaleX(${max > 0 ? y / max : 0})`;
    if (Math.abs(y - lastY) > 6) {
      nav.classList.toggle("hide", y > lastY && y > 400);
      lastY = y;
    }
  };
  addEventListener("scroll", () => {
    if (!queued) {
      queued = true;
      requestAnimationFrame(update);
    }
  }, { passive: true });
  const links = new Map($$(".nav-links a").map((a) => [a.getAttribute("href").slice(1), a]));
  const spy = new IntersectionObserver(
    (entries) =>
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => a.classList.remove("on"));
        links.get(e.target.id)?.classList.add("on");
      }),
    { rootMargin: "-45% 0px -50% 0px" },
  );
  $$("main section[id]").forEach((s) => spy.observe(s));
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
  const wallBox = $(".wall");
  animate(hero, () => wallBox.style.setProperty("--py", `${-scrollY * 0.18}px`), 60);
}

// ================================================================== hero
function wall() {
  const box = $(".wall");
  const f = SITE.faces;
  const rows = Math.ceil(f.n / f.cols);
  const cols = innerWidth < 640 ? 5 : 8;
  const per = 9;
  box.style.setProperty("--sheet", FACES_URL);
  box.style.setProperty("--bs", `${f.cols * 100}% ${rows * 100}%`);
  for (let c = 0; c < cols; c++) {
    const col = el("div", "wall-col");
    col.style.setProperty("--t", `${70 + ((c * 17) % 40)}s`);
    const faces = Array.from({ length: per }, (_, j) => (c * per + j * 5) % f.n);
    [...faces, ...faces].forEach((k) => {
      const s = el("span");
      s.style.setProperty(
        "--bp",
        `${((k % f.cols) / (f.cols - 1)) * 100}% ${(Math.floor(k / f.cols) / (rows - 1)) * 100}%`,
      );
      col.append(s);
    });
    box.append(col);
  }
}

function stats() {
  const m = SITE.metrics;
  const sp = SITE.speed;
  const rows = [
    { to: sp.student_ms / 1000, from: sp.teacher_s, dec: 2, unit: " s", label: `per face in Instant mode, on a laptop GPU (careful mode: ${sp.teacher_s.toFixed(1)} s)` },
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
    if (live && face.png) {
      const a = el("a", "face-dl", ICON.dl);
      a.href = face.png;
      a.download = `humandit-face-${i + 1}.png`;
      a.setAttribute("aria-label", "Download this face (PNG with its watermark)");
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
    const making = `Making ${n} face${n > 1 ? "s" : ""}…`;
    const stop = ticking(statusLine, engine.status.state === "ready" ? making : "Connecting to the engine…");
    const off = engine.onStatus((s) => s.state === "busy" && stop.label(making));
    try {
      const r = await engine.call("make", { count: n, quality: q, must, never, seed: -1 }, {
        onQueue: (pos) => stop.label(inLine(pos)),
      });
      off();
      stop();
      await show(r.faces, true);
      statusLine.innerHTML = `Made <b>${r.faces.length} face${r.faces.length > 1 ? "s" : ""}</b> in <b class="secs">0.00 s</b>, ${q === "best" ? "best quality" : "Instant mode"}.`;
      countTo($(".secs", statusLine), r.seconds, { dec: 2, dur: 900, suffix: " s" });
      tag.textContent = `Live · ${hwLabel(r.hardware)}`;
      tag.classList.add("live");
      if (r.weak?.length) {
        weakNote.textContent = `Instant mode is weak at ${r.weak.join(", ").toLowerCase()}: measured, it draws them in under half the faces. Try Best quality.`;
        weakNote.hidden = false;
      }
    } catch (err) {
      stop();
      off();
      grid.dataset.count = prevCount;
      grid.replaceChildren(...previous);
      statusLine.textContent = err.message;
      toast(err.message);
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
      ? `Live · ${hwLabel(e.hardware)} · ${e.seconds.toFixed(2)} s`
      : madeEarlier(SITE.preset.made_on);
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
      else ask(name, -1);
    });
    pills.append(p);
  };
  SITE.preset.edits.forEach((e) => pill(e.feature, false));
  engine.onStatus((s) => {
    if (s.state !== "ready" || pills.dataset.full) return;
    pills.dataset.full = "1";
    (s.info?.features || SITE.features)
      .filter((f) => !presets.has(f) && f !== "Man")
      .forEach((f) => pill(f, true));
  });

  let busy = false;
  const ask = async (feature, seed) => {
    if (busy) return;
    busy = true;
    btn.classList.add("busy");
    box.classList.remove("gone-static");
    const stop = ticking(status, `Drawing the same person twice…`);
    try {
      const r = await engine.call("edit", { feature_name: feature, quality: "best", seed });
      stop();
      await render(r, true);
    } catch (err) {
      stop();
      status.textContent = err.message;
      toast(err.message);
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
      ? `Live · ${hwLabel(m.hardware)} · ${m.n} faces in ${m.seconds.toFixed(2)} s`
      : `${madeEarlier(m.hardware)} · best quality`;
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
  btn.addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    btn.classList.add("busy");
    const stop = ticking(status, "Inventing two people and 22 in between…");
    const fade = animate(null, () => (noise = Math.min(0.85, noise + 0.06)), 30);
    try {
      const r = await engine.call("blend", { quality: "best", seed_a: -1, seed_b: -1, frames: 24 });
      stop();
      fade();
      await load(r, true);
      const out = animate(null, () => {
        noise = Math.max(0, noise - 0.05);
        if (!noise) out();
      }, 30);
    } catch (err) {
      stop();
      fade();
      noise = 0;
      status.textContent = err.message;
      toast(err.message);
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

  const build = (data, live) => {
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
    } else if (isOnScreen(box) || live) go();
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
  btn.addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    btn.classList.add("busy");
    try {
      const r = await engine.call("compare", { seed: -1, feature_name: "" });
      build(r, true);
    } catch (err) {
      toast(err.message);
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

  const check = async (blob) => {
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
      const r = await engine.call("check", { image: url });
      await new Promise((res) => setTimeout(res, Math.max(0, 1100 - (performance.now() - t0))));
      stamp.textContent = r.found ? "AI-made · HumanDiT" : "No watermark";
      stamp.classList.add(r.found ? "yes" : "no", "show");
      verdict.classList.add(r.found ? "yes" : "no");
      title.textContent = r.found ? "Yes: this picture carries the HumanDiT watermark." : "No HumanDiT watermark found.";
      body.textContent = r.found
        ? "It was made by this AI. The mark is invisible and survives resizing and compression."
        : "This picture was not made here, or it was heavily altered.";
    } catch (err) {
      title.textContent = "Couldn't check this picture right now.";
      body.textContent = err.message;
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
      check(blob);
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

// ================================================================== start
async function main() {
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
    SITE = await (await fetch("assets/site.json")).json();
  } catch (err) {
    toast("This page could not load its pictures. Try reloading.");
    throw err;
  }
  wall();
  stats();
  ticker();
  // rAF sleeps in a background tab, so never wait on the intro for long
  await Promise.race([introDone, new Promise((r) => setTimeout(r, 1600))]);
  reveal();
  const neon = $(".display .neon");
  if (neon && !reduced) setTimeout(() => neon.classList.add("flicker"), 700);
  const sections = [portrait, how, studio, editor, blender, race, checker, numbers];
  for (const s of sections) {
    try {
      await s();
    } catch (err) {
      console.error(`[humandit] ${s.name}:`, err);
    }
  }
  engine.wakeSoon();
}

main();
