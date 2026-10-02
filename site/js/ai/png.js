// PNG text chunks: every face saved from the page says, inside the file, that an AI made it (as the Python engine's
// provenance_pnginfo does). The browser's PNG encoder writes the pixels; this adds tEXt chunks after the header.

const CRC = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const v = new DataView(out.buffer);
  v.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  v.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** A PNG blob with Latin-1 text chunks ({keyword: text}) inserted after IHDR. */
export async function withText(png, texts) {
  const src = new Uint8Array(await png.arrayBuffer());
  const ihdrEnd = 8 + 12 + new DataView(src.buffer).getUint32(8); // signature, then IHDR's length, type, data, crc
  const latin1 = (s) => Uint8Array.from(String(s), (c) => (c.charCodeAt(0) < 256 ? c.charCodeAt(0) : 63));
  const chunks = Object.entries(texts).map(([k, t]) => {
    const key = latin1(k);
    const val = latin1(t);
    const data = new Uint8Array(key.length + 1 + val.length);
    data.set(key);
    data.set(val, key.length + 1);
    return chunk("tEXt", data);
  });
  return new Blob([src.subarray(0, ihdrEnd), ...chunks, src.subarray(ihdrEnd)], { type: "image/png" });
}

const SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

/**
 * RGBA pixels as a PNG, encoded here rather than by a canvas: for a browser whose workers have no OffscreenCanvas.
 * Rows are stored unfiltered and squeezed by the browser's own zlib (CompressionStream "deflate" is the zlib format
 * PNG wants): larger files than a canvas writes, the same pixels.
 */
export async function encodePng(rgba, w, h) {
  const row = w * 4;
  const raw = new Uint8Array((row + 1) * h); // each row: filter type 0, then its pixels
  for (let y = 0; y < h; y++) raw.set(rgba.subarray(y * row, (y + 1) * row), y * (row + 1) + 1);
  const z = new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream("deflate")));
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, w);
  v.setUint32(4, h);
  ihdr[8] = 8; // bits a channel
  ihdr[9] = 6; // RGBA
  const idat = new Uint8Array(await z.arrayBuffer());
  return new Blob([SIGNATURE, chunk("IHDR", ihdr), chunk("IDAT", idat), chunk("IEND", new Uint8Array(0))], { type: "image/png" });
}
