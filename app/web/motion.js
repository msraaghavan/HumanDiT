/* HumanDiT demo: motion runtime.
 *
 * Gradio renders the page after this script runs, so every effect attaches itself when its element appears
 * (`when(selector, init)`, driven by the nodes Gradio adds). All loops share one requestAnimationFrame, pause while
 * their element is off screen or the tab is hidden, and are built to stay at 60 fps on a laptop: the wall of faces is
 * painted once into a cached layer and only slid each frame, TV static comes from a few pre-generated tiles, noisy
 * canvases redraw at 30 fps at capped resolution, and nothing animates layout or blurs what lies behind it.
 *
 * Nothing here fakes a result: the pictures are the model's own outputs (docs/assets/web, built by
 * scripts/showreel/build_web.py), the numbers are measured, and the race bars replay the real, live timings.
 * With prefers-reduced-motion, every effect shows its final state instead.
 */
(() => {
  "use strict";
  if (window.HD) return;
  const HD = (window.HD = {});
  const BASE = window.HD_BASE || "gradio_api/file=docs/assets/web/";
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const BG = "#05060b";

  // ------------------------------------------------------------------ small helpers
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const E = {
    out: t => 1 - Math.pow(1 - t, 3),
    inOut: t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    expo: t => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t)),
  };
  const tw = (t, a, b, e = E.out) => e(clamp((t - a) / (b - a)));
  const pad2 = n => String(n).padStart(2, "0");
  const hash = (a, b) => { const s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return s - Math.floor(s); };
  const mod = (a, n) => ((a % n) + n) % n;

  // ------------------------------------------------------------------ TV static: a few tiles, made once
  function staticCanvas(size, tint) {
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const x = c.getContext("2d"), d = x.createImageData(size, size);
    for (let i = 0; i < d.data.length; i += 4) {
      const v = Math.random() * 235;
      d.data[i] = v + (Math.random() - 0.5) * tint;           // a cool cast: a little more blue than red
      d.data[i + 1] = v + (Math.random() - 0.5) * tint * 0.6;
      d.data[i + 2] = v + 18 + (Math.random() - 0.5) * tint;
      d.data[i + 3] = 255;
    }
    x.putImageData(d, 0, 0);
    return c;
  }
  const STATIC = Array.from({ length: 6 }, () => staticCanvas(80, 70));
  let tick = 0;                                                // frames since the page opened
  function drawStatic(ctx, x, y, w, h, alpha = 1, k = 0) {
    if (alpha <= 0.004) return;
    const s = STATIC[(tick + k) % STATIC.length], o = (tick * 7 + k * 13) % 16;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(s, o, o, 64, 64, x, y, w, h);
    ctx.restore();
  }
  document.documentElement.style.setProperty("--hd-noise", `url(${staticCanvas(128, 40).toDataURL("image/png")})`);

  // ------------------------------------------------------------------ assets
  const cache = {};
  const img = name => (cache[name] ||= new Promise((res, rej) => {
    const i = new Image();
    i.decoding = "async";
    i.onload = () => res(i);
    i.onerror = rej;
    i.src = BASE + name;
  }));
  const data = () => (cache.json ||= fetch(BASE + "motion.json").then(r => r.json()));
  function frame(ctx, sheet, meta, k, x, y, w, h) {
    const s = meta.size, c = meta.cols;
    ctx.drawImage(sheet, (k % c) * s, Math.floor(k / c) * s, s, s, x, y, w, h);
  }

  // ------------------------------------------------------------------ one loop; each animator runs only while visible
  const live = new Set();
  const io = new IntersectionObserver(es => es.forEach(e => (e.target.__hdVis = e.isIntersecting)), { rootMargin: "80px" });
  function animate(el, fn, every = 1) {                        // every = 2: 30 fps, for static and slow scenes
    el.__hdVis = false;
    io.observe(el);
    live.add({ el, fn, every });
  }
  function loop(now) {
    tick++;
    const t = now / 1000;
    if (!document.hidden) for (const a of live) {
      if (!a.el.isConnected) { live.delete(a); io.unobserve(a.el); continue; }
      if (a.el.__hdVis && tick % a.every === 0) { try { a.fn(t); } catch (err) { live.delete(a); console.warn("[hd]", err); } }
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
  function fit(cv, maxDpr = 1.5) {
    const dpr = Math.min(maxDpr, devicePixelRatio || 1);
    const w = Math.max(1, Math.round(cv.clientWidth * dpr)), h = Math.max(1, Math.round(cv.clientHeight * dpr));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    return { w, h, dpr };
  }

  // ------------------------------------------------------------------ attach effects to the nodes Gradio adds
  const inits = [];
  const added = new Set();
  let queued = false;
  function attach(root) {
    for (const [sel, fn, key] of inits) {
      const hit = el => { if (el[key]) return; el[key] = true; try { fn(el); } catch (err) { console.warn("[hd]", sel, err); } };
      if (root.matches && root.matches(sel)) hit(root);
      if (root.querySelectorAll) root.querySelectorAll(sel).forEach(hit);
    }
  }
  function flush() {
    queued = false;
    const roots = [...added];
    added.clear();
    for (const r of roots) if (r.isConnected) attach(r);
  }
  const queue = n => { added.add(n); if (!queued) { queued = true; requestAnimationFrame(flush); } };
  function when(sel, fn) { inits.push([sel, fn, "__hd" + inits.length]); queue(document.documentElement); }
  new MutationObserver(ms => { for (const m of ms) for (const n of m.addedNodes) if (n.nodeType === 1) queue(n); })
    .observe(document.documentElement, { childList: true, subtree: true });

  // ------------------------------------------------------------------ text effects
  const GLYPHS = "01#%&*+=<>/\\|_-:;";
  function scrambleIn(el, delay, dur) {
    const text = el.textContent;
    if (RM) return;
    const t0 = performance.now();
    const step = () => {
      const u = ((performance.now() - t0) / 1000 - delay) / dur;
      if (u >= 1) { el.textContent = text; return; }
      let s = "";
      for (let i = 0; i < text.length; i++) {
        const r = (i / text.length) * 0.6;
        s += text[i] === " " ? " " : u > r + 0.4 ? text[i] : u > r ? GLYPHS[(Math.random() * GLYPHS.length) | 0] : " ";
      }
      el.textContent = s;
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  // counts the first text node of `el` from data-from to data-to, keeping any <small> suffix
  function countTo(el, delay, dur) {
    const node = [...el.childNodes].find(n => n.nodeType === 3);
    if (!node || RM) return;
    const to = +el.dataset.to, from = +(el.dataset.from ?? 0), dec = +(el.dataset.dec ?? 0);
    const fmt = x => (dec ? x.toFixed(dec) : Math.round(x).toLocaleString("en-US"));
    const final = node.textContent, t0 = performance.now(), scr = "scramble" in el.dataset;
    const step = () => {
      const e = (performance.now() - t0) / 1000, u = tw(e, delay, delay + dur, E.expo);
      if (u >= 1) { node.textContent = final; return; }
      node.textContent = scr ? (e < delay ? " " : String((Math.random() * 10) | 0).repeat(final.length)) : fmt(from + (to - from) * u);
      requestAnimationFrame(step);
    };
    node.textContent = scr ? " " : fmt(from);
    requestAnimationFrame(step);
  }

  // ------------------------------------------------------------------ hero: headline, numbers, the wall, the reel
  when(".hd-hero", el => {
    const go = () => el.classList.add("hd-go");
    const fonts = document.fonts ? Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 900))]) : Promise.resolve();
    fonts.then(() => requestAnimationFrame(go));
    el.querySelectorAll(".hd-over-t").forEach(o => scrambleIn(o, 0.15, 1.1));
    el.querySelectorAll("[data-to]").forEach((b, i) => countTo(b, 1.7 + i * 0.12, 1.5));
    const cv = el.querySelector(".hd-wall");
    if (cv) wall(cv, el);
    const v = el.querySelector("video");
    if (v) monitor(el, v);
  });

  // The wall of invented faces drifts behind the hero; the cursor is a torch that lights the faces near it.
  // The lattice is painted once into an off-screen layer one step larger than the view and slid each frame; it is
  // repainted only when the drift crosses a lattice step, and single tiles only while they "re-invent" themselves.
  function wall(cv, hero) {
    const ctx = cv.getContext("2d", { alpha: false });
    const ov = document.createElement("canvas");               // the shade above the wall, with a hole at the cursor
    ov.className = "hd-wall hd-shade";
    ov.setAttribute("aria-hidden", "true");
    cv.after(ov);
    const octx = ov.getContext("2d");
    const L = document.createElement("canvas"), lctx = L.getContext("2d", { alpha: false });
    const P = { x: 0, y: 0, tx: 0, ty: 0, r: 0, tr: 0, mx: 0, my: 0, gx: 0, gy: 0, w: 0, dirty: true };
    hero.addEventListener("pointermove", ev => {
      const b = hero.getBoundingClientRect();
      P.tx = ev.clientX - b.left; P.ty = ev.clientY - b.top; P.tr = 210;
      if (P.r < 1) { P.x = P.tx; P.y = P.ty; }
      P.gx = (ev.clientX - b.left) / b.width - 0.5; P.gy = (ev.clientY - b.top) / b.height - 0.5;
    });
    hero.addEventListener("pointerleave", () => { P.tr = 0; P.gx = 0; P.gy = 0; });
    const box = hero.querySelector(".hd-box"), copy = hero.querySelector(".hd-copy");
    Promise.all([img("faces.jpg"), data()]).then(([sheet, M]) => {
      const meta = M.faces, tiles = new Map(), dissolving = new Set();
      let base = null, geo = null;
      const tileOf = (C, R, t) => {
        const key = C + "," + R;
        let T = tiles.get(key);
        if (!T) { T = { C, R, k: Math.floor(hash(C, R) * meta.n), from: 0, at: -9 }; tiles.set(key, T); }
        return T;
      };
      const paint = (T, t) => {                                // one tile into the layer
        const { pitch, size, dpr } = geo;
        const x = (T.C - base.c + 2) * pitch + (mod(T.R, 2) * pitch) / 2, y = (T.R - base.r + 1) * pitch;
        if (x > L.width || y > L.height || x + size < 0 || y + size < 0) return;
        lctx.fillStyle = BG;
        lctx.fillRect(x - 1, y - 1, size + 2, size + 2);
        lctx.save();
        lctx.beginPath();
        if (lctx.roundRect) lctx.roundRect(x, y, size, size, 12 * dpr); else lctx.rect(x, y, size, size);
        lctx.clip();
        const u = clamp((t - T.at) / 1.1);
        frame(lctx, sheet, meta, u < 0.5 ? T.from : T.k, x, y, size, size);
        if (u < 1) drawStatic(lctx, x, y, size, size, Math.sin(u * Math.PI) * 0.95, T.C + T.R);
        lctx.restore();
      };
      const draw = t => {
        const { w, h, dpr } = fit(cv, 1.25);
        const size = Math.round(92 * dpr), pitch = size + Math.round(10 * dpr);
        const sx = t * 16 * dpr, sy = t * 10 * dpr;         // drift, px/s
        const c0 = Math.floor(sx / pitch), r0 = Math.floor(sy / pitch);
        const cols = Math.ceil(w / pitch) + 4, rows = Math.ceil(h / pitch) + 3;
        if (!base || base.c !== c0 || base.r !== r0 || L.width !== cols * pitch || L.height !== rows * pitch) {
          if (L.width !== cols * pitch || L.height !== rows * pitch) { L.width = cols * pitch; L.height = rows * pitch; }
          base = { c: c0, r: r0 };
          geo = { pitch, size, dpr };
          lctx.fillStyle = BG;
          lctx.fillRect(0, 0, L.width, L.height);
          for (let R = r0 - 1; R < r0 - 1 + rows; R++) for (let C = c0 - 2; C < c0 - 2 + cols; C++) paint(tileOf(C, R, t), t);
          if (tiles.size > 1500) for (const k of [...tiles.keys()].slice(0, 500)) tiles.delete(k);
        }
        if (!RM && tick % 2 === 0) {                           // re-invent a few faces: static, then someone new
          if (Math.random() < 0.06) {
            const T = tileOf(c0 + Math.floor(Math.random() * (cols - 3)), r0 + Math.floor(Math.random() * (rows - 2)), t);
            if (t - T.at > 1.2) { T.from = T.k; T.k = (T.k + 1 + Math.floor(Math.random() * (meta.n - 1))) % meta.n; T.at = t; dissolving.add(T); }
          }
          for (const T of dissolving) { paint(T, t); if (t - T.at > 1.1) dissolving.delete(T); }
        }
        ctx.drawImage(L, -(sx - c0 * pitch) - 2 * pitch, -(sy - r0 * pitch) - pitch);

        // the shade and the torch: repainted only when the torch moves
        P.x += (P.tx - P.x) * 0.14; P.y += (P.ty - P.y) * 0.14; P.r += (P.tr - P.r) * 0.1;
        const moving = Math.abs(P.tx - P.x) + Math.abs(P.ty - P.y) + Math.abs(P.tr - P.r) > 0.3;
        const W = fit(ov, 1).w;
        if (W !== P.w || moving || P.dirty) {
          const H = ov.height, d = W / Math.max(1, ov.clientWidth);
          P.w = W; P.dirty = false;
          octx.globalCompositeOperation = "source-over";
          octx.clearRect(0, 0, W, H);
          let g = octx.createLinearGradient(0, 0, W, 0);
          g.addColorStop(0, "rgba(5,6,11,.95)"); g.addColorStop(0.5, "rgba(5,6,11,.86)"); g.addColorStop(1, "rgba(5,6,11,.6)");
          octx.fillStyle = g; octx.fillRect(0, 0, W, H);
          g = octx.createLinearGradient(0, 0, 0, H);
          g.addColorStop(0, "rgba(5,6,11,.3)"); g.addColorStop(0.55, "rgba(5,6,11,0)"); g.addColorStop(1, "rgba(5,6,11,.85)");
          octx.fillStyle = g; octx.fillRect(0, 0, W, H);
          if (P.r > 1 && !RM) {
            const X = P.x * d, Y = P.y * d, Rr = P.r * d;
            octx.globalCompositeOperation = "destination-out";
            g = octx.createRadialGradient(X, Y, 0, X, Y, Rr);
            g.addColorStop(0, "rgba(0,0,0,.7)"); g.addColorStop(0.55, "rgba(0,0,0,.38)"); g.addColorStop(1, "rgba(0,0,0,0)");
            octx.fillStyle = g; octx.fillRect(X - Rr, Y - Rr, 2 * Rr, 2 * Rr);
            octx.globalCompositeOperation = "source-over";           // a faint neon rim on the torch
            g = octx.createRadialGradient(X, Y, Rr * 0.7, X, Y, Rr);
            g.addColorStop(0, "rgba(0,229,255,0)"); g.addColorStop(0.8, "rgba(0,229,255,.07)"); g.addColorStop(1, "rgba(255,61,240,0)");
            octx.fillStyle = g; octx.fillRect(X - Rr, Y - Rr, 2 * Rr, 2 * Rr);
          }
        }
        // depth: the headline drifts against the cursor, the reel's monitor tilts toward it
        P.mx += (P.gx - P.mx) * 0.08; P.my += (P.gy - P.my) * 0.08;
        if (!RM && Math.abs(P.gx - P.mx) + Math.abs(P.gy - P.my) > 0.0005) {
          if (box) box.style.transform = `perspective(1100px) rotateY(${(P.mx * 8).toFixed(2)}deg) rotateX(${(-P.my * 6).toFixed(2)}deg)`;
          if (copy) copy.style.transform = `translate(${(-P.mx * 12).toFixed(1)}px, ${(-P.my * 8).toFixed(1)}px)`;
        }
      };
      animate(cv, t => draw(RM ? 0 : t));
    });
  }

  function monitor(el, v) {
    v.muted = true;
    v.play().catch(() => {});
    const tc = el.querySelector(".hd-tc"), bar = el.querySelector(".hd-prog i"), snd = el.querySelector(".hd-sound");
    const fmt = s => `00:${pad2(Math.floor(s))}.${Math.floor((s % 1) * 10)}`;
    if (snd) snd.addEventListener("click", () => {
      v.muted = !v.muted;
      if (!v.muted) v.play().catch(() => {});
      snd.textContent = v.muted ? "Sound off" : "Sound on";
      snd.classList.toggle("on", !v.muted);
    });
    // decode the reel only while it can be seen: off screen it would keep decoding for nobody, on the same machine
    // that may be making the faces
    let seen = true;
    const sync = () => { if (seen && !document.hidden) v.play().catch(() => {}); else v.pause(); };
    new IntersectionObserver(es => { seen = es[0].isIntersecting; sync(); }).observe(v);
    document.addEventListener("visibilitychange", sync);
    let last = "";
    animate(el, () => {
      const d = v.duration || 15, t = v.currentTime || 0, s = `${fmt(t)} / ${fmt(d)}`;
      if (s !== last && tc) { tc.textContent = s; last = s; }
      if (bar) bar.style.transform = `scaleX(${(t / d).toFixed(4)})`;
    });
  }

  // ------------------------------------------------------------------ how it works: the story plays itself, left to right
  when(".hd-how", el => {
    Promise.all([img("traj.jpg"), img("edits.jpg"), data()]).then(([traj, edits, M]) => {
      const steps = [...el.querySelectorAll(".hd-step")], dots = [...el.querySelectorAll(".hd-rail b")];
      const rail = el.querySelector(".hd-rail i");
      const cvs = steps.map(s => s.querySelector("canvas")), ctxs = cvs.map(c => c.getContext("2d", { alpha: false }));
      const tags = steps.map(s => s.querySelector(".hd-tag"));
      const W = [0, 2.0, 5.4, 7.9, 13.4];                    // each card's turn (s); the loop is 13.4 s
      const FEAT = [["Glasses", "Eyeglasses"], ["Smile", "Smiling"], ["Grey hair", "Gray_Hair"], ["Bangs", "Bangs"]];
      const lastTag = [], last = [];                           // what each tag / canvas shows, to skip identical redraws
      let lastAct = -2;
      const setTag = (i, html) => { if (lastTag[i] !== html) { tags[i].innerHTML = html; lastTag[i] = html; } };
      animate(el, t => {
        const u = RM ? W[3] + 4.9 : t % W[4];
        const act = W.findIndex((_, i) => u >= W[i] && u < W[i + 1]);
        if (act !== lastAct) {
          steps.forEach((s, i) => s.classList.toggle("on", i === act));
          dots.forEach((d, i) => d.classList.toggle("on", i <= act));
          lastAct = act;
        }
        if (rail) rail.style.transform = `scaleX(${clamp(u / W[3]).toFixed(4)})`;

        // 1 · random static
        { const { w, h } = fit(cvs[0], 1); drawStatic(ctxs[0], 0, 0, w, h, 1, 0); setTag(0, "pure noise"); }

        // 2 · cleaning up: the teacher's draft at each step under the static that is left (x_t = t·noise + (1-t)·x̂₀)
        { const { w, h } = fit(cvs[1], 1.5), c = ctxs[1];
          const p = clamp((u - W[1]) / (W[2] - W[1] - 0.6));
          const k = act < 1 ? 0 : act > 1 ? 24 : Math.min(24, Math.floor(E.inOut(p) * 25));
          const noise = act < 1 ? 1 : 1 - (k + 1) / 25;
          if (noise > 0.01 || last[1] !== k) {
            frame(c, traj, M.traj, k, 0, 0, w, h);
            drawStatic(c, 0, 0, w, h, Math.pow(noise, 0.8), 1);
            last[1] = noise > 0.01 ? -1 : k;
          }
          setTag(1, `step <b>${pad2(k + 1)}</b> / 25`); }

        // 3 · a new face: develops left to right when its turn comes
        { const { w, h } = fit(cvs[2], 1.5), c = ctxs[2];
          const before = act >= 0 && act < 2, a = act === 2 ? tw(u, W[2], W[2] + 1.3, E.inOut) : before ? 0 : 1;
          if (a < 1 || last[2] !== 1) {
            frame(c, traj, M.traj, 24, 0, 0, w, h);
            if (a < 1) {
              const x = a * w;
              c.save(); c.beginPath(); c.rect(x, 0, w - x, h); c.clip(); drawStatic(c, 0, 0, w, h, 0.97, 2); c.restore();
              if (a > 0) {
                const g = c.createLinearGradient(x - 26, 0, x + 4, 0);
                g.addColorStop(0, "rgba(0,229,255,0)"); g.addColorStop(1, "rgba(0,229,255,.95)");
                c.fillStyle = g; c.fillRect(x - 26, 0, 30, h);
              }
            }
            last[2] = a >= 1 ? 1 : 0;
          }
          setTag(2, a >= 1 ? "not a real person" : "developing"); }

        // 4 · your choices: the same face, one feature added at a time; the checker's score for each
        { const { w, h } = fit(cvs[3], 1.5), c = ctxs[3];
          const q = act === 3 ? (u - W[3]) / 1.3 : 0;             // 1.3 s per feature
          const cur = Math.min(4, Math.floor(q)), f = clamp((q - cur) / 0.35), prev = Math.max(0, cur - 1);
          const key = cur * 100 + Math.round(f * 99);
          if (last[3] !== key) {
            if (cur > 0 && f < 1) { frame(c, edits, M.edits, prev, 0, 0, w, h); c.globalAlpha = E.inOut(f); }
            frame(c, edits, M.edits, cur, 0, 0, w, h);
            c.globalAlpha = 1;
            last[3] = key;
          }
          if (cur === 0) setTag(3, "no extras");
          else { const [label, name] = FEAT[cur - 1], v = M.edit_probe[cur].probe[name];
            setTag(3, `+ ${label} <b>${Math.round(v * 100)}%</b>`); } }
      }, 2);
    });
  });

  // ------------------------------------------------------------------ tabs: a neon bar that springs to the chosen tab
  when(".tab-container:not(.visually-hidden)", tl => {   // Gradio keeps a hidden copy for measuring: not that one
    tl.classList.add("hd-inked");
    const ink = document.createElement("i");
    ink.className = "hd-ink";
    tl.appendChild(ink);
    const place = () => {
      const s = tl.querySelector('button[role="tab"].selected, button[aria-selected="true"]');
      if (!s || !s.offsetWidth) return;
      ink.style.width = `${s.offsetWidth - 20}px`;
      ink.style.transform = `translateX(${s.offsetLeft + 10}px)`;
      if (!ink.isConnected) tl.appendChild(ink);
    };
    new MutationObserver(place).observe(tl, { attributes: true, subtree: true, attributeFilter: ["class", "aria-selected"] });
    addEventListener("resize", place);
    setTimeout(place, 60);
    setTimeout(place, 700);
  });

  // ------------------------------------------------------------------ loaders: static that calms down while a job runs
  const FIELD = document.createElement("canvas");
  FIELD.width = 6; FIELD.height = 4;
  const FX = FIELD.getContext("2d");
  when(".hd-load", blk => {
    const ov = document.createElement("div");
    ov.className = "hd-loader";
    ov.innerHTML = '<canvas></canvas><div class="hd-ll">Cleaning up the static <b>0.0 s</b></div>';
    blk.appendChild(ov);
    const cv = ov.querySelector("canvas"), ctx = cv.getContext("2d", { alpha: false }), lbl = ov.querySelector("b");
    let on = false, since = 0, seed = Math.random() * 100;
    const busy = () => {
      const w = blk.querySelector(":scope > .wrap.default.full");
      return !!w && !w.classList.contains("hide") && !w.classList.contains("error") && !!w.querySelector(".eta-bar, .progress-text, .loader");
    };
    const check = () => {
      const b = busy();
      if (b === on) return;
      on = b; since = performance.now(); seed = Math.random() * 100;
      ov.classList.toggle("on", on);
      blk.classList.toggle("hd-busy", on);
      if (!ov.isConnected) blk.appendChild(ov);
    };
    new MutationObserver(check).observe(blk, { attributes: true, subtree: true, attributeFilter: ["class"], childList: true });
    animate(blk, () => {
      if (!on) return;
      const { w, h } = fit(cv, 1), e = (performance.now() - since) / 1000;
      const calm = 1 - Math.exp(-e / 2.5);
      // a soft, drifting field in neon tones (what the noise slowly turns into) under the static that is left
      for (let r = 0; r < FIELD.height; r++) for (let c = 0; c < FIELD.width; c++) {
        const a = hash(r + seed, c) * 6.28 + e * 0.7, b = hash(c + seed, r);
        FX.fillStyle = `hsl(${b < 0.5 ? 188 + b * 30 : 290 + b * 30}, 80%, ${10 + 16 * (0.5 + 0.5 * Math.sin(a))}%)`;
        FX.fillRect(c, r, 1, 1);
      }
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(FIELD, 0, 0, w, h);
      drawStatic(ctx, 0, 0, w, h, 0.85 - 0.55 * calm, 3);
      lbl.textContent = `${e.toFixed(1)} s`;
    }, 2);
  });

  // ------------------------------------------------------------------ results develop out of static
  when(".hd-dev", box => {
    box.addEventListener("load", ev => {
      const im = ev.target;
      if (im.tagName !== "IMG" || RM) return;
      const host = im.closest(".thumbnail-item") || im.parentElement;
      const all = [...box.querySelectorAll(".thumbnail-item")];
      const i = Math.max(0, all.indexOf(host));
      host.classList.remove("hd-developing");
      void host.offsetWidth;
      host.style.setProperty("--d", `${i * 0.08}s`);
      host.classList.add("hd-developing");
      clearTimeout(host.__hdT);
      host.__hdT = setTimeout(() => host.classList.remove("hd-developing"), 2000 + i * 80);
    }, true);
  });

  // ------------------------------------------------------------------ "Made 8 faces in 2.2 s": the numbers count up
  when("#timing", box => {
    const run = () => box.querySelectorAll("strong").forEach(s => {
      if (s.__hd) return;
      s.__hd = true;
      const m = s.textContent.match(/^(\d+(?:\.\d+)?)(.*)$/);
      if (!m || RM) return;
      const to = parseFloat(m[1]), dec = (m[1].split(".")[1] || "").length, rest = m[2], t0 = performance.now();
      const step = () => {
        const u = tw((performance.now() - t0) / 1000, 0.15, 0.95, E.expo);
        s.textContent = (to * u).toFixed(dec) + rest;
        if (u < 1) requestAnimationFrame(step);
      };
      step();
    });
    new MutationObserver(run).observe(box, { childList: true, subtree: true });
    run();
  });

  // ------------------------------------------------------------------ instant or best?: a race at the real speed
  when(".hd-race", el => {
    const lanes = [...el.querySelectorAll(".hd-lane.time")], gaps = [...el.querySelectorAll(".hd-lane.gapl")];
    const max = Math.max(...lanes.map(l => +l.dataset.sec), 1e-3);
    const show = (l, u, v, txt) => { l.querySelector("i").style.transform = `scaleX(${u * v})`; l.querySelector("b").textContent = txt; };
    const suffix = g => g.querySelector("b").dataset.suffix || "";
    if (RM) {
      lanes.forEach(l => show(l, 1, +l.dataset.sec / max, `${(+l.dataset.sec).toFixed(2)} s`));
      gaps.forEach(g => show(g, 1, +g.dataset.val / +g.dataset.max, (+g.dataset.val).toFixed(1) + suffix(g)));
      return;
    }
    const t0 = performance.now();
    const step = () => {
      if (!el.isConnected) return;
      const e = (performance.now() - t0) / 1000 - 0.35;
      let busy = false;
      lanes.forEach(l => {
        const s = +l.dataset.sec, u = clamp(e / s);
        show(l, u, s / max, `${Math.max(0, Math.min(e, s)).toFixed(2)} s`);
        if (u >= 1) l.classList.add("done"); else busy = true;
      });
      const g0 = max + 0.5;
      gaps.forEach((g, i) => {
        const u = tw(e, g0 + i * 0.14, g0 + i * 0.14 + 1.0);
        show(g, u, +g.dataset.val / +g.dataset.max, (u * +g.dataset.val).toFixed(1) + (u >= 1 ? suffix(g) : ""));
        if (u < 1) busy = true;
      });
      if (busy) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });

  // ------------------------------------------------------------------ video HUDs: which step / how far into the blend
  function videoHud(blk, kind) {
    const hud = document.createElement("div");
    hud.className = "hd-vhud";
    hud.innerHTML = kind === "draw"
      ? `<span class="hd-pill">step <b>01</b> / 25</span><span class="hd-ticks">${"<i></i>".repeat(25)}</span>`
      : '<span class="hd-blendbar"><span class="a">Person A</span><span class="trk"><i></i></span><span class="b">Person B</span></span>';
    const pill = hud.querySelector(".hd-pill b"), ticks = [...hud.querySelectorAll(".hd-ticks i")];
    const trk = hud.querySelector(".trk"), knob = hud.querySelector(".trk i");
    let lastK = -1;
    animate(blk, () => {
      const v = blk.querySelector("video");
      if (!v || !v.duration) { if (hud.isConnected) hud.remove(); return; }
      if (hud.parentElement !== v.parentElement) { v.parentElement.style.position ||= "relative"; v.parentElement.appendChild(hud); }
      const t = v.currentTime;
      if (kind === "draw") {                                   // 25 drafts at 8 fps, then the finished face held
        const k = Math.min(24, Math.floor(t * 8));
        if (k !== lastK) { pill.textContent = pad2(k + 1); ticks.forEach((x, i) => x.classList.toggle("on", i <= k)); lastK = k; }
      } else {                                                 // N frames A→B at 12 fps, then the same frames back
        const n = Math.max(2, Math.round((v.duration * 12) / 2)), f = Math.floor(t * 12);
        const p = f < n ? f / (n - 1) : (2 * n - 1 - f) / (n - 1);
        knob.style.transform = `translateX(${(clamp(p) * trk.clientWidth).toFixed(1)}px)`;
      }
    });
  }
  when("#draw-video", blk => videoHud(blk, "draw"));
  when("#blend-video", blk => videoHud(blk, "blend"));

  // ------------------------------------------------------------------ is it AI-made?: scan the picture, then stamp the verdict
  when("#verdict", v => {
    let last = "", pending = 0;
    const run = () => {
      // the verdict's own text only: Gradio's status tracker ("processing | 0.1s") lives in the same block
      const txt = [...v.children].filter(c => !c.classList.contains("wrap")).map(c => c.textContent).join("").trim();
      if (txt === last) return;
      last = txt;
      clearTimeout(pending);
      v.classList.remove("hd-stamp", "hd-yes", "hd-no", "hd-wait");
      if (!txt) return;
      const pic = document.querySelector("#wm-in");
      v.classList.add("hd-wait");
      if (pic && !RM) { pic.classList.remove("hd-scan"); void pic.offsetWidth; pic.classList.add("hd-scan"); }
      pending = setTimeout(() => {
        v.classList.remove("hd-wait");
        v.classList.add("hd-stamp", txt.startsWith("Yes") ? "hd-yes" : "hd-no");
      }, RM ? 0 : 1050);
    };
    new MutationObserver(run).observe(v, { childList: true, subtree: true, characterData: true });
    run();
  });

  // ------------------------------------------------------------------ the page denoises itself when it opens (once a session)
  when(".hd-hero", hero => {
    let seen = false;
    try { seen = !!sessionStorage.getItem("hd-intro"); sessionStorage.setItem("hd-intro", "1"); } catch (_) { /* play it */ }
    if (RM || seen) return;
    const cv = document.createElement("canvas");
    cv.className = "hd-curtain";
    cv.setAttribute("aria-hidden", "true");
    document.body.appendChild(cv);
    const ctx = cv.getContext("2d"), t0 = performance.now(), b = hero.getBoundingClientRect();
    const step = () => {
      const e = (performance.now() - t0) / 1000, { w, h } = fit(cv, 0.75), d = w / innerWidth;
      const cx = (b.left + b.width * 0.3) * d, cy = (b.top + b.height * 0.45) * d;
      const R = tw(e, 0.12, 1.25, E.inOut) * Math.hypot(w, h);
      ctx.globalCompositeOperation = "source-over";
      drawStatic(ctx, 0, 0, w, h, 1, 4);
      ctx.fillStyle = "rgba(5,6,11,.4)";
      ctx.fillRect(0, 0, w, h);
      if (R > 0) {                                             // clear a growing hole; its rim stays noisy
        ctx.globalCompositeOperation = "destination-out";
        const g = ctx.createRadialGradient(cx, cy, R * 0.72, cx, cy, R);
        g.addColorStop(0, "rgba(0,0,0,1)"); g.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill();
      }
      if (e < 1.35) requestAnimationFrame(step); else cv.remove();
    };
    requestAnimationFrame(step);
  });

  // ------------------------------------------------------------------ scroll: a progress line, and sections that rise into view
  const bar = document.createElement("i");
  bar.className = "hd-scrollbar";
  document.documentElement.classList.add("hd-js");
  const mount = () => { if (!bar.isConnected && document.body) document.body.appendChild(bar); };
  mount();
  addEventListener("DOMContentLoaded", mount);
  let scrollQueued = false;
  const onScroll = () => {
    if (scrollQueued) return;
    scrollQueued = true;
    requestAnimationFrame(() => {
      scrollQueued = false;
      const h = document.documentElement.scrollHeight - innerHeight;
      bar.style.transform = `scaleX(${h > 0 ? clamp(scrollY / h).toFixed(4) : 0})`;
    });
  };
  addEventListener("scroll", onScroll, { passive: true });
  addEventListener("resize", onScroll);
  const rio = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add("hd-in"); rio.unobserve(e.target); }
  }), { threshold: 0.08 });
  // not the tabs block: rasterising that whole section as one layer for the reveal cost a 167 ms frame
  when(".hd-how, .hd-score", el => { el.classList.add("hd-reveal"); rio.observe(el); });

  // ------------------------------------------------------------------ buttons ripple from where they were pressed
  document.addEventListener("pointerdown", ev => {
    const btn = ev.target.closest && ev.target.closest("button.lg, button.md, button.sm, #make-btn");
    if (!btn || RM) return;
    const r = btn.getBoundingClientRect(), d = Math.max(r.width, r.height) * 2.2;
    const sp = document.createElement("span");
    sp.className = "hd-ripple";
    Object.assign(sp.style, { width: `${d}px`, height: `${d}px`, left: `${ev.clientX - r.left - d / 2}px`, top: `${ev.clientY - r.top - d / 2}px` });
    btn.classList.add("hd-rippling");
    btn.appendChild(sp);
    setTimeout(() => sp.remove(), 700);
  }, true);

  // ------------------------------------------------------------------ engineers: every model's measured realism gap, log scale
  when(".hd-score", el => {
    const rows = [...el.querySelectorAll(".hd-srow")];
    const run = () => {
      const t0 = performance.now();
      const step = () => {
        const e = (performance.now() - t0) / 1000;
        let busy = false;
        rows.forEach((r, i) => {
          const u = RM ? 1 : tw(e, 0.2 + i * 0.09, 1.2 + i * 0.09, E.expo), v = +r.dataset.v;
          r.querySelector("i").style.transform = `scaleX(${u * +r.dataset.w})`;
          r.querySelector(":scope > b").textContent = (v * u).toFixed(v < 10 ? 2 : 1);
          if (u < 1) busy = true;
        });
        if (busy) requestAnimationFrame(step);
      };
      step();
    };
    const sio = new IntersectionObserver(es => { if (es[0].isIntersecting) { sio.disconnect(); run(); } }, { threshold: 0.3 });
    sio.observe(el);
  });
})();
