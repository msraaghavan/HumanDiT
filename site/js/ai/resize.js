// Pillow's BILINEAR resize (a triangle filter that widens when shrinking, two passes through 8-bit), so a picture
// reaches the watermark reader the way it does in the Python engine: TrustMark reads at 245 x 245.

function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const fscale = Math.max(scale, 1);
  const support = fscale; // bilinear's support is 1, scaled when shrinking
  const out = [];
  for (let x = 0; x < outSize; x++) {
    const center = (x + 0.5) * scale;
    const lo = Math.max(Math.trunc(center - support + 0.5), 0);
    const hi = Math.min(Math.trunc(center + support + 0.5), inSize);
    const w = [];
    let sum = 0;
    for (let i = lo; i < hi; i++) {
      const v = Math.max(0, 1 - Math.abs((i - center + 0.5) / fscale));
      w.push(v);
      sum += v;
    }
    out.push({ lo, w: w.map((v) => (sum ? v / sum : 0)) });
  }
  return out;
}

const u8 = (v) => (v <= 0 ? 0 : v >= 255 ? 255 : Math.round(v));

/** Resize interleaved RGB bytes (w x h) to (W x H). */
export function resizeRGB(src, w, h, W, H) {
  const cx = coeffs(w, W);
  const cy = coeffs(h, H);
  const mid = new Uint8Array(W * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < W; x++) {
      const { lo, w: k } = cx[x];
      let r = 0, g = 0, b = 0;
      for (let i = 0; i < k.length; i++) {
        const p = (y * w + lo + i) * 3;
        r += src[p] * k[i];
        g += src[p + 1] * k[i];
        b += src[p + 2] * k[i];
      }
      const o = (y * W + x) * 3;
      mid[o] = u8(r);
      mid[o + 1] = u8(g);
      mid[o + 2] = u8(b);
    }
  }
  const out = new Uint8Array(W * H * 3);
  for (let y = 0; y < H; y++) {
    const { lo, w: k } = cy[y];
    for (let x = 0; x < W; x++) {
      let r = 0, g = 0, b = 0;
      for (let i = 0; i < k.length; i++) {
        const p = ((lo + i) * W + x) * 3;
        r += mid[p] * k[i];
        g += mid[p + 1] * k[i];
        b += mid[p + 2] * k[i];
      }
      const o = (y * W + x) * 3;
      out[o] = u8(r);
      out[o + 1] = u8(g);
      out[o + 2] = u8(b);
    }
  }
  return out;
}
