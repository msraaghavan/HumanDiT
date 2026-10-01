// Model files: downloaded once in parts (Cloudflare serves files up to 25 MiB), kept in the browser's Cache Storage so
// the next visit starts in seconds, and checked against the manifest's SHA-256 before ONNX Runtime sees a byte.

const CACHE = "humandit-models";

/** Drop parts of older model versions. */
export async function prune(version) {
  try {
    const c = await caches.open(CACHE);
    for (const req of await c.keys()) if (!req.url.includes(`/${version}/`)) await c.delete(req);
  } catch {}
}

/** Which of these files are already stored, so the page can say "0 MB to download". */
export async function stored(base, man, keys) {
  try {
    const c = await caches.open(CACHE);
    const out = {};
    for (const k of keys) {
      const urls = partUrls(base, man, k);
      out[k] = (await Promise.all(urls.map((u) => c.match(u)))).every(Boolean);
    }
    return out;
  } catch {
    return Object.fromEntries(keys.map((k) => [k, false]));
  }
}

const partUrls = (base, man, key) =>
  Array.from({ length: man.files[key].parts }, (_, i) => `${base}${man.version}/${key}.${String(i).padStart(2, "0")}`);

async function sha256(bytes) {
  const h = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...h].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * The bytes of one model file. onBytes(n) reports bytes as they arrive (cached parts count at once).
 * Four parts download at a time.
 */
export async function load(base, man, key, onBytes) {
  const info = man.files[key];
  const urls = partUrls(base, man, key);
  let cache = null;
  try {
    cache = await caches.open(CACHE);
  } catch {} // private windows can refuse storage: then it simply downloads every visit
  const out = new Uint8Array(info.bytes);
  const PART = 24 * 2 ** 20;
  let next = 0;
  const one = async (i) => {
    const url = urls[i];
    let res = cache && (await cache.match(url));
    const fromCache = !!res;
    if (!res) {
      res = await fetch(url);
      if (!res.ok) throw new Error(`Download failed (${res.status}) for ${key}`);
    }
    const reader = res.body.getReader();
    let at = i * PART;
    const chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      out.set(value, at);
      at += value.length;
      onBytes(value.length);
      if (!fromCache && cache) chunks.push(value);
    }
    if (!fromCache && cache) {
      try {
        await cache.put(url, new Response(new Blob(chunks), { headers: { "content-type": "application/octet-stream" } }));
      } catch {} // a full disk only costs the next visit a download
    }
  };
  const worker = async () => {
    while (next < urls.length) await one(next++);
  };
  await Promise.all(Array.from({ length: Math.min(4, urls.length) }, worker));
  if ((await sha256(out)) !== info.sha256) {
    if (cache) await Promise.all(urls.map((u) => cache.delete(u)));
    throw new Error(`The ${key} file arrived damaged; it was deleted, so trying again downloads a fresh copy.`);
  }
  return out;
}
