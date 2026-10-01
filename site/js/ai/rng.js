// torch.randn on the CPU, reproduced: the same seed gives the same starting noise (and so the same face) in the
// browser as in the Python engine. torch seeds an MT19937 with manual_seed(seed), draws one 24-bit uniform per value,
// then turns each block of 16 uniforms into 16 normals with Box-Muller (aten normal_fill). Torch does that last step in
// float32 SIMD maths, this in float64 then rounds, so a value can differ in its last bit: invisible in the picture.

function mt19937(seed) {
  const mt = new Uint32Array(624);
  mt[0] = seed >>> 0;
  for (let i = 1; i < 624; i++) {
    const p = mt[i - 1] ^ (mt[i - 1] >>> 30);
    // 1812433253 * p + i, modulo 2^32, without losing bits to float64
    mt[i] = (((((p & 0xffff0000) >>> 16) * 1812433253) << 16) + (p & 0x0000ffff) * 1812433253 + i) >>> 0;
  }
  let idx = 624;
  return () => {
    if (idx >= 624) {
      for (let k = 0; k < 624; k++) {
        const y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff);
        mt[k] = mt[(k + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      idx = 0;
    }
    let y = mt[idx++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  };
}

/** torch.randn(n, generator=torch.Generator().manual_seed(seed)) for n >= 16, as a Float32Array. */
export function randn(seed, n) {
  const next = mt19937(seed);
  const d = new Float32Array(n);
  const uniform = () => (next() & 0xffffff) / 16777216;
  for (let i = 0; i < n; i++) d[i] = uniform();
  const fill16 = (o) => {
    for (let j = 0; j < 8; j++) {
      const u1 = 1 - d[o + j];
      const u2 = d[o + j + 8];
      const r = Math.sqrt(-2 * Math.log(u1));
      const th = Math.fround(2 * Math.PI) * u2;
      d[o + j] = r * Math.cos(th);
      d[o + j + 8] = r * Math.sin(th);
    }
  };
  for (let i = 0; i + 16 <= n; i += 16) fill16(i);
  if (n % 16) {
    for (let j = 0; j < 16; j++) d[n - 16 + j] = uniform(); // torch redraws the last 16
    fill16(n - 16);
  }
  return d;
}
