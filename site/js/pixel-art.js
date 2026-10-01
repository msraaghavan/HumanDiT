// The pixel-art town's paint: palette, noise and the one-off painting of a scene at a given size (the houses, the
// street, the light of every window, lamp and fire, at six brightnesses of the fire). Pure computation: it runs in
// a worker (js/pixel-build.js) so painting never holds up the page. js/pixel.js brings the scene to life.

// ------------------------------------------------------------------ palette: cold night, warm fire, and their meeting
const HEX = [
  // 0-11 cold: ink to rain highlight
  "#04060d", "#080d1b", "#0d152b", "#131e3b", "#1a294c", "#233760",
  "#2e4877", "#3d5d91", "#5679ab", "#7d9dc8", "#b3cbe8", "#e3eefb",
  // 12-21 warm: ember to white-hot
  "#1c0e0c", "#3a1611", "#5f2014", "#8c2f17", "#bf461c", "#e8672b",
  "#ff8f3d", "#ffb45a", "#ffd585", "#fff0c4",
  // 22-26 warm light on cold stone
  "#1f1726", "#2f1e2e", "#45262f", "#613232", "#84432f",
  // 27-28 the cold flicker of a screen behind a window
  "#6fd3ff", "#bdf0ff",
];
export const PAL = HEX.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
export const U32 = Uint32Array.from(PAL, ([r, g, b]) => ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0);
export const C = (k) => k; // cold ramp index
export const E = (k) => 12 + k; // warm ramp index
export const M = (k) => 22 + k; // warm-lit stone
export const TV0 = 27;
export const TV1 = 28;
const LIT_OK = [...Array(27).keys()].filter((i) => i !== C(11) && i !== E(9) && i !== E(8)); // what lit stone may become

// 4 x 4 ordered dither, thresholds in (0, 1)
export const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((v) => (v + 0.5) / 16);

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const hash2 = (x, y, s = 0) => {
  let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(s, 2246822519)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
};
const smooth = (t) => t * t * (3 - 2 * t);
/** Smooth 2D value noise; `px` makes it repeat every px units along x (for fog that loops). */
export function noise(x, y, s = 0, px = 0) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const fx = smooth(x - xi);
  const fy = smooth(y - yi);
  const w = (v) => (px ? ((v % px) + px) % px : v);
  const a = hash2(w(xi), w(yi), s);
  const b = hash2(w(xi + 1), w(yi), s);
  const c = hash2(w(xi), w(yi + 1), s);
  const d = hash2(w(xi + 1), w(yi + 1), s);
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}
const fbm = (x, y, s, px = 0) =>
  noise(x, y, s, px) * 0.55 + noise(x * 2, y * 2, s + 7, px * 2) * 0.3 + noise(x * 4, y * 4, s + 13, px * 4) * 0.15;

// 128 x 128 tileable noise, for the smoke's and the mist's texture (cheaper than noise() per pixel per frame)
const TILE = 128;
const NOISE = new Float32Array(TILE * TILE);
for (let y = 0; y < TILE; y++)
  for (let x = 0; x < TILE; x++) NOISE[y * TILE + x] = fbm(x / 16, y / 16, 41, 8);
export const tex = (x, y) => NOISE[((y & (TILE - 1)) << 7) | (x & (TILE - 1))];

// lit colour = nearest palette colour to (stone + warm light), dithered; memoised by (colour, light, dither cell)
const WARM = [255, 132, 52];
const shadeMemo = new Uint8Array(32 * 32 * 16).fill(255);
function shade(idx, light, cell) {
  if (light <= 0.02 || idx >= 27) return idx;
  const lq = Math.min(31, Math.round(light * 24));
  const key = (idx * 32 + lq) * 16 + cell;
  let out = shadeMemo[key];
  if (out !== 255) return out;
  const a = (lq / 24) * 0.62;
  const jitter = (BAYER[cell] - 0.5) * 16;
  const c = PAL[idx];
  const t0 = c[0] + WARM[0] * a + jitter;
  const t1 = c[1] + WARM[1] * a + jitter;
  const t2 = c[2] + WARM[2] * a * 0.6 + jitter;
  let best = idx;
  let bd = Infinity;
  for (let q = 0; q < LIT_OK.length; q++) {
    const k = LIT_OK[q];
    const p = PAL[k];
    const e0 = p[0] - t0;
    const e1 = p[1] - t1;
    const e2 = p[2] - t2;
    const d = e0 * e0 * 0.9 + e1 * e1 * 1.2 + e2 * e2 * 0.7;
    if (d < bd) {
      bd = d;
      best = k;
    }
  }
  shadeMemo[key] = best;
  return best;
}
export const DARKER = HEX.map((_, i) => (i === 0 ? 0 : i <= 11 ? i - 1 : i <= 21 ? Math.max(12, i - 2) : i <= 26 ? Math.max(22, i - 1) : C(6)));

// ------------------------------------------------------------------ the scene, painted once per size
export function build(W, H, o) {
  const R = rng(o.seed);
  const N = W * H;
  const idx = new Uint8Array(N); // palette index of every surface
  const emis = new Int16Array(N).fill(-1); // light sources: drawn as they are, never shaded
  const lightS = new Float32Array(N); // steady warm light (windows, lamp, lanterns)
  const lightF = new Float32Array(N); // the brazier's light at full strength
  const sky = new Uint8Array(N); // 1 sky, 2 an edge the lightning catches
  const puddle = new Uint8Array(N);
  const cell = (x, y) => (y & 3) * 4 + (x & 3);
  const put = (x, y, c) => {
    if (x >= 0 && x < W && y >= 0 && y < H) idx[y * W + x] = c;
  };
  const glow = (x, y, c) => {
    if (x >= 0 && x < W && y >= 0 && y < H) emis[y * W + x] = c;
  };
  const rect = (x, y, w, h, c) => {
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) put(x + i, y + j, c);
  };
  const addLight = (map, cx, cy, radius, amount, power = 2) => {
    const x0 = Math.max(0, Math.floor(cx - radius));
    const x1 = Math.min(W - 1, Math.ceil(cx + radius));
    const y0 = Math.max(0, Math.floor(cy - radius));
    const y1 = Math.min(H - 1, Math.ceil(cy + radius));
    for (let y = y0; y <= y1; y++)
      for (let x = x0; x <= x1; x++) {
        const ddx = x - cx;
        const ddy = (y - cy) * 1.1;
        const d = Math.sqrt(ddx * ddx + ddy * ddy) / radius;
        if (d < 1) map[y * W + x] += amount * (1 - d) ** power;
      }
  };
  // sizes follow U: the height on a wide screen, capped by the width on a tall one (a phone's town stays in proportion)
  const U = Math.min(H, Math.round(W * 0.9));
  const street = H - Math.round(U * (1 - o.street));

  // sky: clouds lit faintly from below by the town, a hidden moon
  const moonX = W * o.moon;
  const moonY = H * 0.13;
  for (let y = 0; y < street; y++)
    for (let x = 0; x < W; x++) {
      const t = y / street;
      const n = tex(Math.round(x * 0.7 + y * 0.25 + o.seed * 37), Math.round(y * 2.2 + o.seed * 11));
      const mx = x - moonX;
      const my = (y - moonY) * 1.6;
      const moon = Math.max(0, 1 - Math.sqrt(mx * mx + my * my) / (U * 0.42));
      const v = 0.6 + t * 2.6 + (n - 0.5) * 2.4 + moon * moon * 3;
      const lo = Math.floor(v);
      const c = Math.min(5, Math.max(1, lo + (v - lo > BAYER[cell(x, y)] ? 1 : 0)));
      idx[y * W + x] = c;
      sky[y * W + x] = 1;
    }

  // far town: dark roofs and spires against the glowing cloud, a few tiny windows
  const farBase = street - Math.round(U * 0.07);
  const far = [];
  for (let x = -4; x < W; ) {
    const w = 6 + Math.floor(R() * 16);
    const h = Math.round(U * (0.05 + R() * 0.16));
    const top = farBase - h;
    for (let i = 0; i < w; i++)
      for (let y = top; y < street; y++) {
        put(x + i, y, C(2));
        if (x + i >= 0 && x + i < W && y >= 0) sky[y * W + x + i] = 0;
      }
    if (R() < 0.3) for (let y = top - 3 - Math.floor(R() * 8); y < top; y++) put(x + (w >> 1), y, C(2)); // a spire
    for (let y = top + 2; y < street - 2; y += 3)
      for (let i = 1; i < w - 1; i += 2) if (R() < 0.1) far.push(y * W + x + i);
    x += w + (R() < 0.3 ? 2 + Math.floor(R() * 5) : 0);
  }
  for (const p of far) if (p >= 0 && p < N) emis[p] = R() < 0.7 ? E(5) : E(6);

  // the street's houses
  const windows = [];
  const chimneys = [];
  const faceWins = [];
  const lanterns = [];
  const house = (x0, w, h, feature) => {
    const top = street - h;
    const wall = R() < 0.5 ? C(3) : C(4);
    for (let y = top; y < street; y++)
      for (let x = x0; x < x0 + w; x++) {
        if (x < 0 || x >= W) continue;
        const r = hash2(x, y, o.seed + 3);
        let c = wall;
        if (r < 0.035) c = wall - 1;
        if ((y - top) % 6 === 5 && hash2(x >> 3, y, o.seed) < 0.5) c = wall - 1; // courses of stone
        idx[y * W + x] = c;
        sky[y * W + x] = 0;
      }
    for (let y = top; y < street; y++) {
      put(x0, y, C(5)); // wet edge catching the sky
      put(x0 + w - 1, y, C(2));
      if (x0 >= 0 && x0 < W) sky[y * W + x0] = 2;
    }
    rect(x0 - 1, top, w + 2, 2, C(5)); // cornice
    for (let x = x0 - 1; x < x0 + w + 1; x++) if (x >= 0 && x < W && top >= 0) sky[top * W + x] = 2;
    // roof: a gable, or a flat top with a parapet
    if (!feature && R() < 0.7) {
      const rh = Math.round(w * (0.32 + R() * 0.22));
      for (let j = 0; j < rh; j++) {
        const half = Math.round((w / 2 + 1) * (1 - j / rh));
        const cx = x0 + w / 2;
        for (let x = Math.floor(cx - half); x <= Math.ceil(cx + half); x++) {
          const y = top - 1 - j;
          if (x < 0 || x >= W || y < 0) continue;
          idx[y * W + x] = j % 3 === 2 ? C(1) : C(2);
          sky[y * W + x] = 0;
        }
        for (const ex of [Math.floor(cx - half), Math.ceil(cx + half)]) {
          put(ex, top - 1 - j, C(6)); // the wet ridge line
          if (ex >= 0 && ex < W && top - 1 - j >= 0) sky[(top - 1 - j) * W + ex] = 2;
        }
      }
      if (R() < 0.75) {
        const cx = x0 + Math.round(w * (R() < 0.5 ? 0.25 : 0.7));
        const ch = 6 + Math.floor(R() * 7);
        const cy = top - Math.round(rh * 0.45) - ch;
        rect(cx, cy, 4, ch + 6, C(2));
        for (let y = cy; y < cy + ch + 6; y++) put(cx, y, C(5));
        rect(cx - 1, cy - 1, 6, 2, C(6));
        chimneys.push({ x: cx + 2, y: cy - 2 });
      }
    } else {
      rect(x0 - 1, top - 4, w + 2, 4, C(3));
      rect(x0 - 1, top - 5, w + 2, 1, C(6));
      if (!feature && R() < 0.5) {
        const cx = x0 + 3 + Math.floor(R() * (w - 8));
        rect(cx, top - 13, 4, 9, C(2));
        rect(cx - 1, top - 14, 6, 2, C(6));
        chimneys.push({ x: cx + 2, y: top - 15 });
      }
    }
    // windows
    const ww = feature ? 10 : 5;
    const wh = feature ? 12 : 7;
    const gx = feature ? 15 : 10;
    const gy = feature ? 18 : 15;
    const cols = Math.max(1, Math.floor((w - 6) / gx));
    const pad = Math.floor((w - (cols - 1) * gx - ww) / 2);
    for (let wy = top + 7; wy + wh < street - (feature ? 6 : 14); wy += gy)
      for (let c = 0; c < cols; c++) {
        const wx = x0 + pad + c * gx;
        const lit = R() < (feature ? 0.85 : 0.52);
        rect(wx - 1, wy - 1, ww + 2, wh + 2, C(1)); // frame
        rect(wx - 1, wy + wh + 1, ww + 2, 1, C(6)); // sill
        const pix = [];
        for (let j = 0; j < wh; j++)
          for (let i = 0; i < ww; i++) {
            const x = wx + i;
            const y = wy + j;
            if (x < 0 || x >= W) continue;
            const mullion = !feature && (i === 2 || j === 3);
            if (!lit) {
              put(x, y, i === 0 && j === 0 ? C(4) : C(1));
              continue;
            }
            if (mullion) {
              put(x, y, C(0));
              continue;
            }
            const warm = j < wh * 0.35 ? E(8) : j < wh * 0.75 ? E(7) : E(6);
            glow(x, y, warm);
            pix.push(y * W + x, warm);
          }
        if (!lit) continue;
        const kind = feature ? "face" : R() < 0.07 ? "tv" : R() < 0.25 ? "candle" : "steady";
        windows.push({ pix: Int32Array.from(pix), kind, phase: R() * 100, x: wx, y: wy, w: ww, h: wh });
        if (feature) faceWins.push(windows[windows.length - 1]);
        addLight(lightS, wx + ww / 2, wy + wh / 2, feature ? 16 : 9, feature ? 0.42 : 0.38);
        addLight(lightS, wx + ww / 2, wy + wh + 2, 6, 0.25); // spill on the sill
      }
    // a door, with light under it and sometimes a lantern
    if (!feature && R() < 0.65) {
      const dx = x0 + Math.floor(w / 2) - 3;
      rect(dx - 1, street - 13, 8, 13, C(1));
      rect(dx, street - 12, 6, 12, E(0));
      rect(dx, street - 1, 6, 1, E(5));
      for (let i = 0; i < 6; i++) glow(dx + i, street - 1, E(5));
      if (R() < 0.6) {
        const lx = dx + 8;
        rect(lx, street - 15, 2, 3, E(7));
        for (let j = 0; j < 3; j++) for (let i = 0; i < 2; i++) glow(lx + i, street - 15 + j, j === 0 ? E(8) : E(7));
        addLight(lightS, lx + 1, street - 13, 15, 0.5);
        lanterns.push({ x: lx + 1, y: street - 13 });
      }
    }
  };
  // the facade of faces sits where it is asked to; the rest of the street fills in around it
  const fw = Math.max(64, Math.min(96, Math.round(W * 0.2)));
  const fx0 = Math.round(W * o.faces - fw / 2);
  const fh = Math.round(U * 0.5 * o.tall);
  const spans = [];
  for (let x = -8; x < W + 8; ) {
    if (x + 30 > fx0 && x < fx0 + fw) {
      x = fx0 + fw + 2;
      continue;
    }
    const w = 26 + Math.floor(R() * 30);
    const end = Math.min(x + w, x < fx0 ? fx0 - 2 : W + 8);
    if (end - x >= 18) spans.push([x, end - x, Math.round(U * (0.24 + R() * 0.2) * o.tall)]);
    x = end + (R() < 0.35 ? 3 + Math.floor(R() * 4) : 1);
  }
  spans.forEach(([x, w, h]) => house(x, w, h, false));
  house(fx0, fw, fh, true);

  // the street: wet cobbles, the warm light of every window smeared into it
  rect(0, street, W, 1, C(6));
  rect(0, street + 1, W, 2, C(3));
  let y = street + 3;
  let row = 0;
  while (y < H) {
    const rh = 2 + Math.floor((y - street) / 11);
    let x = -Math.floor(R() * 8);
    while (x < W) {
      const sw = 5 + Math.floor(R() * 6) + rh;
      const base = R() < 0.5 ? C(3) : C(2);
      for (let j = 0; j < rh; j++)
        for (let i = 0; i < sw; i++) {
          const c = j === rh - 1 || i === sw - 1 ? C(1) : j === 0 ? (R() < 0.6 ? C(5) : C(4)) : base;
          put(x + i, y + j, c);
        }
      x += sw;
    }
    y += rh;
    row++;
  }
  const smear = (cx, half, amount) => {
    for (let yy = street + 2; yy < H; yy++) {
      const f = 1 - (yy - street) / (H - street);
      const wob = Math.round(Math.sin(yy * 1.7 + cx) * 1.2);
      for (let xx = Math.floor(cx - half) + wob; xx <= Math.ceil(cx + half) + wob; xx++)
        if (xx >= 0 && xx < W) lightS[yy * W + xx] += amount * f * (0.8 + 0.2 * hash2(xx >> 1, yy >> 1, 5));
    }
  };
  windows.forEach((w) => smear(w.x + w.w / 2, w.w / 2, w.kind === "face" ? 0.3 : 0.2));
  lanterns.forEach((l) => smear(l.x, 1.5, 0.35));

  // puddles: they mirror the town (filled in every frame)
  const puddles = [];
  for (let k = 0; k < Math.round(W / 80) + 2; k++) {
    const cx = R() * W;
    const cy = street + 8 + R() * (H - street - 14);
    const rx = 9 + R() * (14 + (cy - street) * 0.5);
    const ry = 1.6 + (cy - street) * 0.06 + R() * 1.5;
    if (Math.abs(cx - W * o.fire) < rx + 12) continue;
    puddles.push({ cx, cy, rx, ry });
    for (let yy = Math.floor(cy - ry); yy <= Math.ceil(cy + ry); yy++)
      for (let xx = Math.floor(cx - rx); xx <= Math.ceil(cx + rx); xx++) {
        if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;
        const d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2;
        if (d < 1) puddle[yy * W + xx] = 1;
        else if (d < 1.35) idx[yy * W + xx] = C(5); // the rim
      }
  }

  // a street lamp
  let lamp = null;
  if (o.lamp != null) {
    const lx = Math.round(W * o.lamp);
    const top = street - Math.round(U * 0.32 * Math.max(0.8, o.tall));
    rect(lx, top, 2, street + 4 - top, C(1));
    for (let yy = top; yy < street + 4; yy++) put(lx, yy, C(4));
    rect(lx - 2, street + 2, 6, 2, C(1));
    rect(lx - 3, top - 8, 8, 1, C(1));
    rect(lx - 2, top - 7, 6, 7, C(0));
    for (let j = 0; j < 5; j++) for (let i = 0; i < 4; i++) glow(lx - 1 + i, top - 6 + j, j < 2 ? E(9) : E(8));
    rect(lx - 3, top - 9, 8, 1, C(5));
    lamp = { x: lx + 0.5, y: top - 4 };
    addLight(lightS, lamp.x, lamp.y, U * 0.24, 0.8, 1.6);
    for (let yy = Math.ceil(lamp.y); yy < H; yy++) {
      const dy = yy - lamp.y;
      const half = dy * 0.42;
      for (let xx = Math.floor(lamp.x - half); xx <= Math.ceil(lamp.x + half); xx++) {
        if (xx < 0 || xx >= W) continue;
        const edge = 1 - Math.abs(xx - lamp.x) / (half + 1);
        lightS[yy * W + xx] += 0.32 * edge * Math.max(0, 1 - dy / (U * 0.62));
      }
    }
    addLight(lightS, lamp.x, street + 6, U * 0.12, 0.45);
    smear(lamp.x, 2, 0.45);
  }

  // the brazier
  const bx = Math.round(W * o.fire);
  const by = Math.min(H - 4, street + Math.round(U * 0.085)); // its foot
  rect(bx - 8, by - 11, 16, 11, E(0));
  for (let yy = by - 11; yy < by; yy++) {
    put(bx - 7, yy, C(1));
    put(bx + 6, yy, C(1));
  }
  rect(bx - 7, by - 8, 14, 1, C(2));
  rect(bx - 7, by - 4, 14, 1, C(2));
  rect(bx - 8, by - 12, 16, 1, E(3));
  rect(bx - 6, by, 3, 2, C(1));
  rect(bx + 3, by, 3, 2, C(1));
  const fire = { x: bx, base: by - 12 };
  addLight(lightF, bx, by - 18, U * 0.45, 1, 1.7);
  smear(bx, 5, 0.5);

  // edges fall into the dark
  const vig = new Float32Array(N);
  for (let yy = 0; yy < H; yy++)
    for (let xx = 0; xx < W; xx++) {
      const dx = (xx / W - 0.5) * 1.6;
      const dy = (yy / H - 0.45) * 1.9;
      vig[yy * W + xx] = Math.max(0, Math.sqrt(dx * dx + dy * dy) - 0.7) * 1.4;
    }

  // paint: six brightnesses of the fire
  const surf = new Uint8Array(N);
  const cells = new Uint8Array(N);
  const fireLit = [];
  const base = new Uint32Array(N);
  for (let yy = 0, i = 0; yy < H; yy++)
    for (let xx = 0; xx < W; xx++, i++) {
      const cl = (yy & 3) * 4 + (xx & 3);
      let c = idx[i];
      if (vig[i] > BAYER[cl]) c = DARKER[c];
      if (vig[i] > 1 + BAYER[cl]) c = DARKER[c];
      surf[i] = c;
      cells[i] = cl;
      if (emis[i] >= 0) base[i] = U32[emis[i]];
      else {
        base[i] = U32[shade(c, lightS[i], cl)];
        if (lightF[i] > 0.004) fireLit.push(i);
      }
    }
  const levels = [];
  const light = [];
  for (let k = 0; k < 6; k++) {
    const f = 0.45 + k * 0.11;
    const lt = new Uint8Array(N);
    for (let i = 0; i < N; i++) lt[i] = Math.min(255, Math.round((lightS[i] + lightF[i] * f) * 100));
    light.push(lt);
    const out = new Uint32Array(base);
    for (let q = 0; q < fireLit.length; q++) {
      const i = fireLit[q];
      out[i] = U32[shade(surf[i], lightS[i] + lightF[i] * f, cells[i])];
    }
    levels.push(out);
  }
  // lightning: the sky and every edge facing it go pale
  const flash = new Uint32Array(levels[3]);
  const skyList = [];
  for (let i = 0; i < N; i++) {
    if (!sky[i] || emis[i] >= 0) continue;
    skyList.push(i);
    flash[i] = U32[Math.min(C(9), idx[i] + (sky[i] === 1 ? 4 : 3))];
  }
  const puddleList = [];
  for (let i = 0; i < N; i++) if (puddle[i]) puddleList.push(i);
  return {
    W, H, street, levels, light, flash, skyList: Int32Array.from(skyList), puddleList: Int32Array.from(puddleList),
    puddle, windows, faceWins: faceWins.map((w) => windows.indexOf(w)), chimneys, fire, lamp,
  };
}

/** Every buffer of a built scene, to hand over without copying. */
export function buffers(S) {
  return [
    ...S.levels.map((a) => a.buffer), ...S.light.map((a) => a.buffer), S.flash.buffer, S.skyList.buffer,
    S.puddleList.buffer, S.puddle.buffer, ...S.windows.map((w) => w.pix.buffer),
  ];
}
