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
