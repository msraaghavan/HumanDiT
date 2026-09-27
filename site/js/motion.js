// The motion runtime: one requestAnimationFrame loop whose jobs run only while their element is on screen (and at
// their own frame rate), reveal-on-view, split headings, number count-ups, magnetic buttons, and the pre-made static
// textures every "developing" picture uses. Everything that moves is a transform or an opacity, or a small canvas.

export const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
export const finePointer = matchMedia("(pointer: fine)").matches;
export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const easeOut = (t) => 1 - Math.pow(1 - t, 3);
export const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

// ------------------------------------------------------------------ the loop
const jobs = new Set();
const onScreen = new WeakMap();
const watcher = new IntersectionObserver(
  (entries) => entries.forEach((e) => onScreen.set(e.target, e.isIntersecting)),
  { rootMargin: "120px 0px" },
);
let raf = 0;

function tick(now) {
  raf = requestAnimationFrame(tick);
  for (const j of jobs) {
    if (j.el && !onScreen.get(j.el)) continue;
    if (now - j.last < j.gap) continue;
    j.last = now;
    j.fn(now);
  }
}

/** Run fn(now) every frame (at most `fps` a second) while `el` is on screen; returns a stop function. */
export function animate(el, fn, fps = 60) {
  const job = { el, fn, gap: 1000 / fps - 1.5, last: 0 };
  if (el) {
    onScreen.set(el, false);
    watcher.observe(el);
  }
  jobs.add(job);
  if (!raf) raf = requestAnimationFrame(tick);
  return () => jobs.delete(job);
}

export const isOnScreen = (el) => !!onScreen.get(el);

// ------------------------------------------------------------------ reveal on view
const revealer = new IntersectionObserver(
  (entries) =>
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("in");
      e.target.dispatchEvent(new CustomEvent("reveal"));
      revealer.unobserve(e.target);
    }),
  { rootMargin: "0px 0px -12% 0px", threshold: 0.05 },
);

export function reveal(root = document) {
  root.querySelectorAll("[data-reveal], [data-split], .arch, .score").forEach((el) => {
    if (reduced) {
      el.classList.add("in");
      queueMicrotask(() => el.dispatchEvent(new CustomEvent("reveal")));
    } else revealer.observe(el);
  });
}

/** Wrap each word of an element in a mask so it can rise into place; nested elements (.neon) keep their wrapper. */
export function split(el) {
  let i = 0;
  const walk = (node) => {
    [...node.childNodes].forEach((child) => {
      if (child.nodeType === Node.TEXT_NODE) {
        const frag = document.createDocumentFragment();
        child.textContent.split(/(\s+)/).forEach((part) => {
          if (!part) return;
          if (/^\s+$/.test(part)) return frag.append(" ");
          const w = document.createElement("span");
          w.className = "w";
          const inner = document.createElement("span");
          inner.textContent = part;
          inner.style.setProperty("--i", i++);
          w.append(inner);
          frag.append(w);
        });
        child.replaceWith(frag);
      } else if (child.nodeType === Node.ELEMENT_NODE) walk(child);
    });
  };
  el.setAttribute("aria-label", el.textContent.replace(/\s+/g, " ").trim());
  walk(el);
  el.querySelectorAll(".w").forEach((w) => w.setAttribute("aria-hidden", "true"));
}

// ------------------------------------------------------------------ numbers
export function countTo(el, to, { from = 0, dec = 0, dur = 1600, suffix = "" } = {}) {
  const fmt = (v) =>
    v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suffix;
  if (reduced) return void (el.textContent = fmt(to));
  const t0 = performance.now();
  const stop = animate(null, (now) => {
    const t = clamp((now - t0) / dur);
    el.textContent = fmt(lerp(from, to, easeOut(t)));
    if (t >= 1) stop();
  });
}

// ------------------------------------------------------------------ magnetic buttons
export function magnetic(el, pull = 0.22) {
  if (!finePointer || reduced) return;
  el.addEventListener("pointermove", (e) => {
    const r = el.getBoundingClientRect();
    el.style.setProperty("--mx", `${(e.clientX - r.left - r.width / 2) * pull}px`);
    el.style.setProperty("--my", `${(e.clientY - r.top - r.height / 2) * pull}px`);
  });
  el.addEventListener("pointerleave", () => {
    el.style.setProperty("--mx", "0px");
    el.style.setProperty("--my", "0px");
  });
}

// ------------------------------------------------------------------ static (noise) textures, made once
function noiseCanvas(w, h, tint = true) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(w, h);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    // Gaussian-ish static, a little cooler than grey, like the model's own starting noise when drawn
    const g = (Math.random() + Math.random() + Math.random()) / 3;
    const v = g * 255;
    d[i] = tint ? v * 0.86 + Math.random() * 30 : v;
    d[i + 1] = tint ? v * 0.92 + Math.random() * 20 : v;
    d[i + 2] = tint ? Math.min(255, v * 1.02 + Math.random() * 40) : v;
    d[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

/** Six frames of static side by side: the .static overlays step through them with a transform (no repaint). */
export function staticStrip(size = 128) {
  const strip = document.createElement("canvas");
  strip.width = size * 6;
  strip.height = size;
  const ctx = strip.getContext("2d");
  for (let k = 0; k < 6; k++) ctx.drawImage(noiseCanvas(size, size), k * size, 0);
  return strip.toDataURL("image/jpeg", 0.82);
}

/** Canvases of static for drawing under a picture at a given amount (the portrait, how-it-works, blend). */
export function noiseTiles(n = 6, size = 256) {
  return Array.from({ length: n }, () => noiseCanvas(size, size));
}

export function grain() {
  return noiseCanvas(180, 180, false).toDataURL("image/png");
}

// ------------------------------------------------------------------ sprite sheets
export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.decoding = "async";
    im.onload = () => (im.decode ? im.decode().catch(() => {}) : Promise.resolve()).then(() => resolve(im));
    im.onerror = () => reject(new Error(`could not load ${src.slice(0, 60)}`));
    im.src = src;
  });
}

/** Draw frame i of a sprite sheet ({cols, size}) into the whole of a 2D context's canvas. */
export function drawFrame(ctx, sheet, meta, i, alpha = 1) {
  const k = Math.max(0, Math.min(meta.n - 1, i));
  const sx = (k % meta.cols) * meta.size;
  const sy = Math.floor(k / meta.cols) * meta.size;
  ctx.globalAlpha = alpha;
  ctx.drawImage(sheet, sx, sy, meta.size, meta.size, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.globalAlpha = 1;
}

/** Size a canvas to its box at a capped device-pixel ratio. */
export function fit(canvas, maxDpr = 1.5, max = 1024) {
  const r = canvas.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, maxDpr);
  const w = Math.min(max, Math.round(r.width * dpr)) || canvas.width;
  const h = Math.min(max, Math.round(r.height * dpr)) || canvas.height;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return canvas.getContext("2d");
}
