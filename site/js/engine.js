// The engine: the HumanDiT models on a Hugging Face ZeroGPU Space (or app/api.py on this machine), reached with the
// official Gradio client. The page wakes it as soon as it opens, shows its state in the header, and turns its errors
// (a used-up free GPU allowance, busy GPUs, a sleeping Space) into plain words.

import { Client } from "../vendor/gradio-client-2.7.0.min.js";

const cfg = window.HD_CONFIG || {};
const params = new URLSearchParams(location.search);
const onLocalhost = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
export const target = params.get("engine") || (onLocalhost ? cfg.local : cfg.space) || "";

const pill = document.querySelector(".engine");
const pillText = document.querySelector(".engine-text");
const listeners = new Set();
export const status = { state: "idle", text: "", hardware: "", info: null };

let client = null;
let connecting = null;
let running = 0;

function set(state, text) {
  status.state = state;
  status.text = text;
  if (pill) pill.dataset.state = state;
  if (pillText) pillText.textContent = text;
  listeners.forEach((f) => f(status));
}

export const onStatus = (f) => (listeners.add(f), () => listeners.delete(f));

/** Connect (and wake the Space if it sleeps). Resolves to the engine's /ping answer; rejects if unreachable. */
export function connect() {
  if (!target || target.includes("__SPACE__")) {
    // a copy of the site deployed before its engine: everything shown is the AI's saved work, said plainly
    set("saved", "Showing saved results");
    return Promise.reject(new Error("NOT_LIVE"));
  }
  if (connecting) return connecting;
  set("connecting", "Engine: connecting");
  connecting = (async () => {
    try {
      client = await Client.connect(target, {
        status_callback: (s) => {
          if (["sleeping", "building", "starting", "paused"].includes(s.status) || s.load_status === "pending")
            set("waking", "Engine: waking up");
          if (s.status === "error" || s.load_status === "error") set("offline", "Engine offline");
        },
      });
      const info = await run("ping", {});
      status.info = info;
      status.hardware = info.hardware;
      set("ready", `Engine ready · ${short(info.hardware)}`);
      return info;
    } catch (err) {
      connecting = null;
      set("offline", "Engine offline · saved results");
      throw err;
    }
  })();
  return connecting;
}

const short = (hw) => hw.replace(/^NVIDIA (GeForce )?/, "").replace("Hugging Face ", "");

async function run(name, payload, onQueue) {
  const job = client.submit(`/${name}`, payload);
  for await (const msg of job) {
    if (msg.type === "status") {
      if (msg.stage === "error") throw new Error(msg.message || "The engine could not finish that.");
      if (msg.stage === "pending" && msg.queue && msg.position > 0) onQueue?.(msg.position);
    } else if (msg.type === "data") return msg.data[0];
  }
  throw new Error("The engine closed the connection without an answer.");
}

/** Call an endpoint: {onQueue(position)} reports a place in line. Errors come back in plain words. */
export async function call(name, payload, { onQueue } = {}) {
  try {
    await connect();
  } catch (err) {
    throw new Error(plain(err));
  }
  running++;
  set("busy", "Engine: working");
  try {
    return await run(name, payload, onQueue);
  } catch (err) {
    throw new Error(plain(err));
  } finally {
    running--;
    if (!running && status.state === "busy") set("ready", `Engine ready · ${short(status.hardware)}`);
  }
}

export function plain(err) {
  const m = String(err?.message || err || "");
  if (m === "NOT_LIVE")
    return "Live mode is not switched on for this copy of the site yet, so it shows results the AI made earlier.";
  if (/quota/i.test(m)) {
    const wait = m.match(/(\d+:\d{2}:\d{2})/);
    return (
      "You have used today's free GPU time on this site (Hugging Face gives each visitor a few minutes a day)." +
      (wait ? ` It refills in ${wait[1]}.` : "") +
      " Logged-in Hugging Face users get more. Until then, the saved results stay on the page."
    );
  }
  if (/no gpu|gpus? .*busy|capacity/i.test(m)) return "Every shared GPU is busy right now. Try again in a minute.";
  if (/too large/i.test(m) || /not a picture/i.test(m) || /unknown feature/i.test(m)) return m;
  if (/fetch|network|connect|offline|configured/i.test(m))
    return "The engine can't be reached right now, so this shows results the AI made earlier.";
  return `The engine hit a problem: ${m.slice(0, 160)}`;
}

/** Wake the engine early, once the page has settled, so it is ready by the time a visitor scrolls to a tool. */
export function wakeSoon(delay = 1200) {
  const go = () => connect().catch(() => {});
  if ("requestIdleCallback" in window) setTimeout(() => requestIdleCallback(go, { timeout: 2000 }), delay);
  else setTimeout(go, delay);
}
