/* Image Upscaler worker: runs Swin2SR (x2) on one tile at a time, off the main thread.
   Everything is loaded from this site's own files: no model hub, no CDN. */
import { env, Swin2SRForImageSuperResolution, Tensor } from "./lib/transformers.min.js";

env.allowRemoteModels = false;      // never download anything
env.allowLocalModels = true;        // the model ships in models/swin2SR-x2/
env.localModelPath = "models/";
env.useBrowserCache = false;        // the files are already local; don't keep a second 21 MB copy
const onnx = env.backends.onnx;
// ort-wasm-simd-threaded.jsep.wasm lives in lib/. Absolute URL: ort resolves it with new URL(file, base),
// which throws on a bare relative base.
onnx.wasm.wasmPaths = new URL("lib/", self.location.href).href;
onnx.logLevel = "error";           // ORT prints warnings through console.error; keep only real errors
onnx.wasm.numThreads = 1;           // GitHub Pages has no cross-origin isolation, so no wasm threads
onnx.wasm.proxy = false;

let model = null, device = null;

async function hasWebGPU() {
  try { return !!(self.navigator.gpu && await self.navigator.gpu.requestAdapter()); } catch (e) { return false; }
}

async function open(dev) {
  return Swin2SRForImageSuperResolution.from_pretrained("swin2SR-x2", {
    device: dev, dtype: "q8", local_files_only: true,
    session_options: { logSeverityLevel: 3 }   // errors only: ORT's node-placement warnings are noise here
  });
}

async function load(want) {
  if (model && (want === "auto" || want === device)) return device;
  if (model) { try { await model.dispose(); } catch (e) { /* ignore */ } model = null; }
  let tried = "";
  if (want !== "wasm" && await hasWebGPU()) {
    try {
      model = await open("webgpu"); device = "webgpu";
      // prove the GPU path really works (and compiles its shaders) on a small tile before trusting it
      const t = await infer(new Float32Array(3 * 16 * 16).fill(0.5), 16, 16);
      if (!t.data.every(Number.isFinite)) throw new Error("WebGPU returned invalid numbers");
      return device;
    } catch (e) {
      tried = "WebGPU failed (" + (e && e.message || e) + "), using WASM. ";
      try { if (model) await model.dispose(); } catch (x) { /* ignore */ }
      model = null;
    }
  }
  model = await open("wasm"); device = "wasm";
  self.postMessage({ type: "note", text: tried });
  return device;
}

// data: planar RGB float32 in 0..1, length 3*w*h (w, h multiples of 8). Returns planar RGB 2w x 2h.
async function infer(data, w, h) {
  const out = await model({ pixel_values: new Tensor("float32", data, [1, 3, h, w]) });
  const r = out.reconstruction;
  const [, , H, W] = r.dims;
  const d = r.data instanceof Float32Array ? r.data : Float32Array.from(r.data);
  if (typeof r.dispose === "function" && d !== r.data) r.dispose();
  return { data: d, w: W, h: H };
}

self.onmessage = async function (e) {
  const m = e.data;
  try {
    if (m.type === "init") {
      const t0 = performance.now();
      const dev = await load(m.device || "auto");
      self.postMessage({ type: "ready", id: m.id, device: dev, ms: performance.now() - t0 });
    } else if (m.type === "run") {
      const t0 = performance.now();
      const r = await infer(m.data, m.w, m.h);
      const copy = r.data.slice(); // own buffer, safe to transfer
      self.postMessage({ type: "out", id: m.id, data: copy, w: r.w, h: r.h, ms: performance.now() - t0 }, [copy.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: "error", id: m.id, message: String(err && err.message || err) });
  }
};
