// Animated pixel art for the page's backdrops: a cold, rainy town at night against warm pockets of firelight and lit
// windows. Rain falls in three depths and splashes, a brazier throws sparks, smoke drifts from it and the chimneys,
// the firelight flickers on wet stone, puddles mirror it all, and now and then lightning whitens the sky. The faces in
// the big lit windows are people this site's AI invented.
//
// It is drawn on a grid of real pixels about 216 tall (384 x 216 on a 1080p screen: exactly 5 x 5 screen pixels
// each) and scaled up by a whole number with image-rendering: pixelated, so it stays crisp full screen. The houses,
// the street and their light are painted once per size, in a worker, at six brightnesses of the fire; each frame
// copies one and draws only what moves. Particle pools are fixed and clocks wrap, so it can run for hours without drifting or
// growing. It animates only while on screen, at 30 frames a second.

import { animate, reduced } from "./motion.js";
import { BAYER, C, DARKER, E, M, TV0, TV1, U32, hash2, noise, rng, tex } from "./pixel-art.js";

// one worker paints every scene on the page, in turn
let painter = null;
let paintSeq = 0;
const paintJobs = new Map();
function paint(W, H, o) {
  painter ||= Object.assign(new Worker(new URL("./pixel-build.js", import.meta.url), { type: "module" }), {
    onmessage: (e) => {
      paintJobs.get(e.data.id)?.(e.data.S);
      paintJobs.delete(e.data.id);
    },
  });
  const id = ++paintSeq;
  return new Promise((res) => {
    paintJobs.set(id, res);
    painter.postMessage({ id, W, H, o });
  });
}

// ------------------------------------------------------------------ faces for the big windows, in firelight
// All 72 faces are shrunk onto one small canvas and read back once, then mapped onto the warm ramp; shared by every
// scene on the page.
const spriteCache = new WeakMap();
function faceSprites(img, cols, n, w, h) {
  if (!img) return [];
  if (spriteCache.has(img)) return spriteCache.get(img);
  const per = 12;
  const cv = document.createElement("canvas");
  cv.width = per * w;
  cv.height = Math.ceil(n / per) * h;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  const size = img.naturalWidth / cols;
  for (let k = 0; k < n; k++) {
    const sx = (k % cols) * size;
    const sy = Math.floor(k / cols) * size;
    ctx.drawImage(img, sx + size * 0.12, sy + size * 0.06, size * 0.76, size * 0.9, (k % per) * w, Math.floor(k / per) * h, w, h);
  }
  const d = ctx.getImageData(0, 0, cv.width, cv.height).data;
  const out = [];
  for (let k = 0; k < n; k++) {
    const ox = (k % per) * w;
    const oy = Math.floor(k / per) * h;
    const sprite = new Uint8Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const q = ((oy + y) * cv.width + ox + x) * 4;
        const l = (d[q] * 0.35 + d[q + 1] * 0.5 + d[q + 2] * 0.15) / 255;
        const v = Math.min(1, Math.max(0, (l - 0.12) * 1.35)) * 7.2 + (BAYER[(y & 3) * 4 + (x & 3)] - 0.5);
        sprite[y * w + x] = E(Math.min(8, Math.max(1, Math.round(v) + 1)));
      }
    out.push(sprite);
  }
  spriteCache.set(img, out);
  return out;
}

// ------------------------------------------------------------------ the living parts
export function nightScene(host, opts = {}) {
  const o = { seed: 7, street: 0.8, fire: 0.62, lamp: 0.2, faces: 0.45, moon: 0.78, rain: 1, tall: 1, ...opts };
  const cv = document.createElement("canvas");
  cv.className = "pixel-canvas";
  cv.setAttribute("aria-hidden", "true");
  host.append(cv);
  const ctx = cv.getContext("2d", { alpha: false });
  let S = null;
  let pending = "";
  let img = null;
  let px = null;
  let sprites = [];
  let facesImg = null;
  const R = rng(o.seed * 31 + 5);

  // particles: fixed pools
  const drops = [];
  const splash = Array.from({ length: 80 }, () => ({ on: false, x: 0, y: 0, a: 0, w: false }));
  const rings = Array.from({ length: 48 }, () => ({ on: false, x: 0, y: 0, a: 0 }));
  const sparks = Array.from({ length: 90 }, () => ({ on: false, x: 0, y: 0, vx: 0, vy: 0, a: 0, life: 1 }));
  const smoke = Array.from({ length: 56 }, () => ({ on: false, x: 0, y: 0, vx: 0, vy: 0, a: 0, life: 1, r0: 1, warm: false, s: 0 }));
  /** A free slot in a pool, or null: scanned from where the last one was found. */
  const cursors = new WeakMap();
  const free = (pool) => {
    let at = cursors.get(pool) || 0;
    for (let n = 0; n < pool.length; n++, at = (at + 1) % pool.length)
      if (!pool[at].on) {
        cursors.set(pool, (at + 1) % pool.length);
        return pool[at];
      }
    return null;
  };
  const FW = 26;
  const FH = 42;
  const heat = new Float32Array(FW * FH);
  let fireAmt = 0.8;
  let clock = 0;
  let nextFlash = 14 + R() * 20;
  let flashAt = -10;
  let sparkDebt = 0;
  let smokeDebt = 0;
  const faceState = [];
  const WIND = 0.24; // rain slants left by this much per pixel fallen

  function size() {
    const box = host.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    // about 216 art pixels tall (exactly x5 at 1080p), but never fewer than 240 across (tall phone screens)
    const s = Math.max(2, Math.min(Math.round((Math.max(box.height, 200) * dpr) / 216), Math.floor((box.width * dpr) / 240)));
    const W = Math.ceil((box.width * dpr) / s);
    const H = Math.ceil((box.height * dpr) / s);
    if ((S && S.W === W && S.H === H) || pending === `${W}x${H}`) return;
    const want = `${W}x${H}`;
    pending = want;
    paint(W, H, o).then((built) => {
      if (pending !== want) return; // resized again meanwhile
      ready(built, s, dpr);
    });
  }

  function ready(built, s, dpr) {
    const { W, H } = built;
    S = built;
    cv.width = W;
    cv.height = H;
    cv.style.width = `${(W * s) / dpr}px`;
    cv.style.height = `${(H * s) / dpr}px`;
    img = ctx.createImageData(W, H);
    px = new Uint32Array(img.data.buffer);
    // rain in three depths, as many as the area needs
    drops.length = 0;
    const area = (W * H) / (384 * 216);
    const layer = (n, v, len, cold, warm, bright, hit) => {
      for (let i = 0; i < Math.round(n * area * o.rain); i++)
        drops.push({ x: R() * (W + H * WIND), y: R() * H, v: v * (0.85 + R() * 0.3), len, cold, warm, bright, hit, end: 0 });
    };
    layer(110, 150, 2, C(6), E(4), C(7), 0);
    layer(80, 230, 4, C(8), E(6), C(9), 1);
    layer(34, 330, 7, C(9), E(7), C(10), 2);
    drops.forEach((d) => (d.end = endOf(d)));
    sprites = faceSprites(facesImg, 12, 72, 10, 12);
    faceState.length = 0;
    S.faceWins.forEach((w, k) => faceState.push({ face: (k * 7 + o.seed) % 72, dark: 0, next: 8 + R() * 30 }));
    if (reduced) frame(0.033);
  }
  const endOf = (d) =>
    d.hit === 0 ? S.street - R() * S.H * 0.18 : S.street + 3 + R() * (S.H - S.street - 3) * (d.hit === 2 ? 1 : 0.7);

  const setPx = (x, y, c) => {
    x |= 0;
    y |= 0;
    if (x >= 0 && x < S.W && y >= 0 && y < S.H) px[y * S.W + x] = U32[c];
  };
  let lit = null; // this frame's warm light, x100
  const warmAt = (x, y) => {
    x |= 0;
    y |= 0;
    if (x < 0 || x >= S.W || y < 0 || y >= S.H) return 0;
    return lit[y * S.W + x] / 100;
  };

  function fireStep() {
    const c = (FW - 1) / 2;
    for (let x = 0; x < FW; x++) {
      const e = Math.abs(x - c) / (c * 0.78);
      heat[(FH - 1) * FW + x] = e >= 1 ? 0 : (1 - e * e) * (0.82 + 0.18 * R());
    }
    const sway = Math.sin(clock * 1.1) * 0.8 + Math.sin(clock * 2.9) * 0.4;
    const rise = clock * 20;
    for (let y = 0; y < FH - 1; y++) {
      const up = 1 - y / FH;
      const shift = Math.round(sway * (1 - up) * 0.6);
      for (let x = 0; x < FW; x++) {
        const sx = Math.min(FW - 1, Math.max(0, x + shift));
        const l = heat[(y + 1) * FW + Math.max(0, sx - 1)];
        const m = heat[(y + 1) * FW + sx];
        const r = heat[(y + 1) * FW + Math.min(FW - 1, sx + 1)];
        const b = heat[Math.min(FH - 1, y + 2) * FW + sx];
        const coolN = noise(x * 0.42, (y + rise) * 0.3, 31);
        const edge = Math.abs(x - c) / c;
        heat[y * FW + x] = Math.max(0, (l + m + r + b) / 4 - (0.012 + coolN * coolN * 0.11 + edge * 0.02));
      }
    }
    let sum = 0;
    for (let i = FW * (FH >> 1); i < FW * FH; i++) sum += heat[i];
    const target = 0.5 + 0.55 * Math.min(1, sum / (FW * (FH >> 1) * 0.42));
    fireAmt += (target - fireAmt) * 0.3;
  }

  const FIRE = [0.07, 0.15, 0.24, 0.34, 0.45, 0.57, 0.7, 0.84];
  const FIRE_C = [E(2), E(3), E(4), E(5), E(6), E(7), E(8), E(9)];

  function frame(dt) {
    if (!S || !px) return;
    clock = (clock + dt) % 86400; // wraps daily: nothing grows
    const { W, H, street } = S;
    fireStep();
    const lvl = Math.max(0, Math.min(5, Math.round((fireAmt - 0.45) / 0.11)));
    px.set(S.levels[lvl]);
    lit = S.light[lvl];

    // lightning, now and then: a double flash that fades
    nextFlash -= dt;
    if (nextFlash <= 0) {
      flashAt = clock;
      nextFlash = 22 + R() * 38;
    }
    const ft = clock - flashAt;
    let flashI = 0;
    if (ft >= 0 && ft < 0.9) flashI = ft < 0.06 ? ft / 0.06 : ft < 0.14 ? 1 - (ft - 0.06) * 9 : ft < 0.2 ? 0.3 + (ft - 0.14) * 10 : Math.max(0, 0.9 - (ft - 0.2) * 1.4);
    if (flashI > 0.02) {
      const L = S.skyList;
      for (let k = 0; k < L.length; k++) {
        const i = L[k];
        const x = i % W;
        if (flashI > BAYER[(((i / W) | 0) & 3) * 4 + (x & 3)]) px[i] = S.flash[i];
      }
    }

    // windows: candles waver, screens flicker cold, the big windows show someone, who now and then leaves
    for (const w of S.windows) {
      if (w.kind === "steady") continue;
      if (w.kind === "candle") {
        const m = Math.round((noise(clock * 5 + w.phase, 0.5, 3) - 0.62) * 3.2);
        if (m === 0) continue;
        const P = w.pix;
        for (let q = 0; q < P.length; q += 2) px[P[q]] = U32[Math.max(E(4), Math.min(E(9), P[q + 1] + m))];
      } else if (w.kind === "tv") {
        const step = Math.floor(clock * 7 + w.phase);
        const r = hash2(step, 1, w.phase | 0);
        const col = r < 0.45 ? TV0 : r < 0.75 ? C(9) : r < 0.9 ? TV1 : C(8);
        const P = w.pix;
        for (let q = 0; q < P.length; q += 2) px[P[q]] = U32[col];
      }
    }
    for (let k = 0; k < S.faceWins.length; k++) {
      const w = S.windows[S.faceWins[k]];
      const st = faceState[k];
      st.next -= dt;
      if (st.next <= 0) {
        if (st.dark <= 0) {
          st.dark = 1.6 + R() * 1.4;
          st.face = (st.face + 1 + Math.floor(R() * 70)) % 72;
        }
        st.next = 14 + R() * 40;
      }
      if (st.dark > 0) {
        st.dark -= dt;
        const P = w.pix;
        for (let q = 0; q < P.length; q += 2) px[P[q]] = U32[st.dark > 0.5 ? E(1) : E(4)];
        continue;
      }
      const sp = sprites[st.face];
      if (!sp) continue;
      const breath = noise(clock * 2 + k * 9, 2.5, 4) > 0.78 ? -1 : 0;
      for (let j = 0; j < w.h; j++)
        for (let i = 0; i < w.w; i++) {
          const x = w.x + i;
          if (x < 0 || x >= W) continue;
          px[(w.y + j) * W + x] = U32[Math.max(E(1), sp[j * w.w + i] + breath)];
        }
    }

    // smoke: from the brazier (lit warm) and every chimney (cold), drifting with the wind
    smokeDebt += dt * (3.2 + S.chimneys.length * 0.7);
    while (smokeDebt >= 1) {
      smokeDebt -= 1;
      const p = free(smoke);
      if (!p) break;
      const fromFire = !S.chimneys.length || R() < 3.2 / (3.2 + S.chimneys.length * 0.7);
      const ch = fromFire ? null : S.chimneys[Math.floor(R() * S.chimneys.length)];
      p.on = true;
      p.x = (fromFire ? S.fire.x : ch.x) + (R() - 0.5) * 3;
      p.y = fromFire ? S.fire.base - Math.round(FH * 0.8) : ch.y;
      p.vx = 3 + R() * 5;
      p.vy = -(5 + R() * 7);
      p.a = 0;
      p.life = 5 + R() * 5;
      p.r0 = fromFire ? 1.5 : 1;
      p.warm = fromFire;
      p.s = R() * 100;
    }
    for (const p of smoke) {
      if (!p.on) continue;
      p.a += dt;
      if (p.a > p.life) {
        p.on = false;
        continue;
      }
      p.x += (p.vx + Math.sin(clock * 0.7 + p.s) * 2) * dt;
      p.y += p.vy * dt;
      p.vy *= 1 - 0.08 * dt;
      const t = p.a / p.life;
      const r = p.r0 + t * 9;
      const dens = (t < 0.15 ? t / 0.15 : 1 - (t - 0.15) / 0.85) * 0.55;
      const x0 = Math.floor(p.x - r);
      const y0 = Math.floor(p.y - r);
      for (let y = y0; y <= p.y + r; y++) {
        if (y < 0 || y >= H) continue;
        for (let x = x0; x <= p.x + r; x++) {
          if (x < 0 || x >= W) continue;
          const d = ((x - p.x) ** 2 + (y - p.y) ** 2) / (r * r);
          if (d >= 1) continue;
          const a = dens * (1 - d) * (0.5 + 0.5 * tex(x * 2 + (p.s | 0), y * 2 + ((clock * 10) | 0)));
          if (a > BAYER[(y & 3) * 4 + (x & 3)]) {
            const lit = warmAt(x, y);
            px[y * W + x] = U32[lit > 0.55 ? M(3) : lit > 0.3 ? M(2) : p.warm && t < 0.3 ? M(1) : C(5)];
          }
        }
      }
    }

    // the fire itself
    const fx0 = S.fire.x - (FW >> 1);
    const fy0 = S.fire.base - FH + 3;
    for (let y = 0; y < FH; y++)
      for (let x = 0; x < FW; x++) {
        const h = heat[y * FW + x];
        if (h < FIRE[0]) continue;
        let k = 0;
        while (k < FIRE.length - 1 && h >= FIRE[k + 1]) k++;
        setPx(fx0 + x, fy0 + y, FIRE_C[k]);
      }

    // puddles mirror what stands above them, broken by ripples
    const P = S.puddleList;
    for (let k = 0; k < P.length; k++) {
      const i = P[k];
      const x = i % W;
      const y = (i / W) | 0;
      const sy = Math.round(street - (y - street) * 1.8 - 2);
      const sx = Math.round(x + Math.sin(y * 1.9 + clock * 2.4 + x * 0.15) * 0.8);
      if (sy < 0 || sx < 0 || sx >= W) {
        px[i] = U32[C(2)];
        continue;
      }
      const src = px[sy * W + sx];
      const pi = U32.indexOf(src);
      px[i] = U32[pi >= 0 ? DARKER[pi] : C(3)];
    }
    for (const r of rings) {
      if (!r.on) continue;
      r.a += dt;
      if (r.a > 0.75) {
        r.on = false;
        continue;
      }
      const rad = 1 + r.a * 8;
      const fade = 1 - r.a / 0.75;
      for (let k = 0; k < 18; k++) {
        const ang = (k / 18) * Math.PI * 2;
        const x = Math.round(r.x + Math.cos(ang) * rad);
        const y = Math.round(r.y + Math.sin(ang) * rad * 0.35);
        if (x < 0 || x >= W || y < 0 || y >= H || !S.puddle[y * W + x]) continue;
        if (fade > BAYER[(y & 3) * 4 + (x & 3)]) px[y * W + x] = U32[warmAt(x, y) > 0.45 ? E(5) : C(7)];
      }
    }

    // ground mist, two layers drifting opposite ways, looping across the width
    for (let y = street - 14; y < H; y++) {
      const prof = Math.max(0, 1 - Math.abs(y - (street + 2)) / 22) * 0.42;
      if (prof <= 0) continue;
      for (let x = 0; x < W; x++) {
        const n = tex(x + ((clock * 4) | 0), y * 2) * 0.6 + tex(x * 2 - ((clock * 9) | 0) + 64, y * 3 + 40) * 0.4;
        const a = (n - 0.42) * 2.2 * prof;
        if (a > BAYER[(y & 3) * 4 + (x & 3)]) px[y * W + x] = U32[warmAt(x, y) > 0.5 ? M(2) : C(4)];
      }
    }

    // rain: warm where it falls through light, pale in a flash
    for (const d of drops) {
      d.y += d.v * dt;
      d.x -= d.v * WIND * dt;
      if (d.y >= d.end) {
        const hx = Math.round(d.x);
        const hy = Math.round(d.end);
        if (d.hit && hx >= 0 && hx < W && hy < H) {
          if (S.puddle[hy * W + hx]) {
            const r = free(rings);
            if (r) {
              r.on = true;
              r.x = hx;
              r.y = hy;
              r.a = 0;
            }
          } else if (d.hit === 2 || R() < 0.5) {
            const s = free(splash);
            if (s) {
              s.on = true;
              s.x = hx;
              s.y = hy;
              s.a = 0;
              s.w = warmAt(hx, hy) > 0.4;
            }
          }
        }
        d.y = -R() * 30;
        d.x = R() * (W + H * WIND);
        d.end = endOf(d);
      }
      if (d.x < -12) d.x += W + H * WIND;
      for (let j = 0; j < d.len; j++) {
        const x = Math.round(d.x + j * WIND);
        const y = Math.round(d.y - j);
        if (x < 0 || x >= W || y < 0 || y >= H) continue;
        const warm = warmAt(x, y) > 0.55;
        px[y * W + x] = U32[flashI > 0.3 ? C(11) : warm ? (j === 0 ? Math.min(E(9), d.warm + 1) : d.warm) : j === 0 ? d.bright : d.cold];
      }
    }
    for (const s of splash) {
      if (!s.on) continue;
      s.a += dt;
      if (s.a > 0.16) {
        s.on = false;
        continue;
      }
      const c = s.w ? E(7) : C(9);
      if (s.a < 0.05) {
        setPx(s.x, s.y - 1, c);
        setPx(s.x - 1, s.y, c);
        setPx(s.x + 1, s.y, c);
      } else if (s.a < 0.1) {
        setPx(s.x - 1, s.y - 2, c);
        setPx(s.x + 1, s.y - 2, c);
        setPx(s.x - 2, s.y - 1, c);
        setPx(s.x + 2, s.y - 1, c);
      } else {
        setPx(s.x - 2, s.y - 2, s.w ? E(5) : C(7));
        setPx(s.x + 2, s.y - 2, s.w ? E(5) : C(7));
      }
    }

    // sparks: a steady stream and the odd crackle, floating up and cooling
    sparkDebt += dt * 13 + (R() < dt * 0.35 ? 10 : 0);
    while (sparkDebt >= 1) {
      sparkDebt -= 1;
      const p = free(sparks);
      if (!p) break;
      p.on = true;
      p.x = S.fire.x + (R() - 0.5) * 8;
      p.y = S.fire.base - 16 - R() * 16;
      p.vx = (R() - 0.5) * 16 + 4;
      p.vy = -(22 + R() * 38);
      p.a = 0;
      p.life = 0.9 + R() * 2;
    }
    for (const p of sparks) {
      if (!p.on) continue;
      p.a += dt;
      if (p.a > p.life) {
        p.on = false;
        continue;
      }
      const ox = p.x;
      const oy = p.y;
      p.vx += Math.sin(clock * 9 + p.life * 40) * 30 * dt;
      p.vy += 10 * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      const t = p.a / p.life;
      const c = t < 0.2 ? E(9) : t < 0.45 ? E(8) : t < 0.7 ? E(7) : t < 0.88 ? E(5) : E(3);
      if (Math.abs(p.vy) > 18) setPx(ox, oy, t < 0.5 ? E(5) : E(3));
      setPx(p.x, p.y, c);
    }

    ctx.putImageData(img, 0, 0);
  }

  let last = 0;
  let dirty = true;
  const ro = new ResizeObserver(() => {
    dirty = true;
    if (reduced) size();
  });
  ro.observe(host);
  const stop = reduced
    ? () => {}
    : animate(host, (now) => {
        if (dirty) {
          dirty = false;
          size();
        }
        const dt = last && now - last < 200 ? (now - last) / 1000 : 1 / 30;
        last = now;
        frame(dt);
      }, 30);

  return {
    /** The site's invented faces, for the big windows (once their sprite sheet has loaded). */
    faces(image) {
      facesImg = image;
      if (!S) return;
      sprites = faceSprites(image, 12, 72, 10, 12);
    },
    stop() {
      stop();
      ro.disconnect();
    },
  };
}
