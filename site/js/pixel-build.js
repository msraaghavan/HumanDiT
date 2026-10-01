// Paints pixel-art scenes off the main thread (js/pixel-art.js), so scrolling never waits for one.
import { buffers, build } from "./pixel-art.js";

self.onmessage = (e) => {
  const { id, W, H, o } = e.data;
  const S = build(W, H, o);
  self.postMessage({ id, S }, buffers(S));
};
