/* Image Upscaler — Swin2SR x2 runs in a Web Worker (worker.js) on overlapping tiles; the tiles are
   feather-blended here. The image never leaves the browser and the page makes no network requests. */
const $ = S3.$, $$ = S3.$$;

// ---------- limits ----------
const MAX_IN_MP = 1;                 // 2x: input up to 1 MP (e.g. 1000×1000 -> 2000×2000); 4x: 0.25 MP
const MAX_OUT_SIDE = 12000;          // browsers refuse bigger canvases
const OVERLAP = 16;                  // input px shared by neighbouring tiles (32 px of feathered output)

// ---------- pure image maths (UI + self-test) ----------
// Tile starts along one axis: full tiles of size T stepping T - O, the last one pushed back inside the image.
function spans(n, T, O) {
  if (n <= T) return [[0, n]];
  const out = [], step = T - O;
  for (let s = 0; ; s += step) {
    if (s + T >= n) { out.push([n - T, T]); break; }
    out.push([s, T]);
  }
  return out;
}
function planTiles(W, H, T, O) {
  const xs = spans(W, T, O), ys = spans(H, T, O), t = [];
  ys.forEach(y => xs.forEach(x => t.push({ x: x[0], w: x[1], y: y[0], h: y[1] })));
  return t;
}
const up8 = n => Math.ceil(n / 8) * 8;
// Planar RGB float tile in 0..1, padded right/bottom to a multiple of 8 by mirroring (as Swin2SR's own processor does).
function cutTile(rgba, W, t) {
  const pw = up8(t.w), ph = up8(t.h), plane = pw * ph, d = new Float32Array(3 * plane);
  const mir = (i, n) => i < n ? i : Math.max(0, 2 * n - 1 - i);
  for (let j = 0; j < ph; j++) {
    const row = (t.y + mir(j, t.h)) * W + t.x;
    for (let i = 0; i < pw; i++) {
      const p = (row + mir(i, t.w)) * 4, o = j * pw + i;
      d[o] = rgba[p] / 255; d[plane + o] = rgba[p + 1] / 255; d[2 * plane + o] = rgba[p + 2] / 255;
    }
  }
  return { data: d, w: pw, h: ph };
}
// Weight ramps from 0 to 1 across R output px on every side that touches another tile (not on the image border).
function ramp(u, n, R, lo, hi) {
  let w = 1;
  if (lo) w = Math.min(w, (u + 0.5) / R);
  if (hi) w = Math.min(w, (n - u - 0.5) / R);
  return w;
}
function addTile(acc, wsum, OW, out, t, W, H, R) {
  const ow = t.w * 2, oh = t.h * 2, plane = out.w * out.h, ox = t.x * 2, oy = t.y * 2;
  const L = t.x > 0, Rt = t.x + t.w < W, T = t.y > 0, B = t.y + t.h < H;
  const wx = new Float32Array(ow);
  for (let u = 0; u < ow; u++) wx[u] = ramp(u, ow, R, L, Rt);
  for (let v = 0; v < oh; v++) {
    const wy = ramp(v, oh, R, T, B), src = v * out.w, dst = (oy + v) * OW + ox;
    for (let u = 0; u < ow; u++) {
      const w = wx[u] * wy, s = src + u, q = dst + u;
      acc[q * 3] += w * out.data[s]; acc[q * 3 + 1] += w * out.data[plane + s]; acc[q * 3 + 2] += w * out.data[2 * plane + s];
      wsum[q] += w;
    }
  }
}
// Plain bilinear 2x (pixel centres aligned): the "ordinary resize" baseline for the self-test, and for alpha.
function bilinear2x(src, W, H) {
  const OW = W * 2, OH = H * 2, out = new Uint8ClampedArray(OW * OH * 4);
  for (let y = 0; y < OH; y++) {
    const fy = Math.min(H - 1, Math.max(0, (y + 0.5) / 2 - 0.5)), y0 = Math.floor(fy), y1 = Math.min(H - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < OW; x++) {
      const fx = Math.min(W - 1, Math.max(0, (x + 0.5) / 2 - 0.5)), x0 = Math.floor(fx), x1 = Math.min(W - 1, x0 + 1), tx = fx - x0;
      const a = (y0 * W + x0) * 4, b = (y0 * W + x1) * 4, c = (y1 * W + x0) * 4, d = (y1 * W + x1) * 4, o = (y * OW + x) * 4;
      for (let k = 0; k < 4; k++) out[o + k] = (src[a + k] * (1 - tx) + src[b + k] * tx) * (1 - ty) + (src[c + k] * (1 - tx) + src[d + k] * tx) * ty;
    }
  }
  return out;
}
function hasAlpha(rgba) { for (let i = 3; i < rgba.length; i += 4) if (rgba[i] < 255) return true; return false; }

class Cancelled extends Error { constructor() { super("Cancelled"); this.name = "Cancelled"; } }

// One 2x pass. run(data, w, h) -> {data, w, h} is the model; progress(done, total) after each tile.
async function upscale2x(rgba, W, H, o) {
  const tile = o.tile || 128, overlap = Math.min(o.overlap == null ? OVERLAP : o.overlap, tile / 2);
  const OW = W * 2, OH = H * 2, tiles = planTiles(W, H, tile, overlap);
  const acc = new Float32Array(OW * OH * 3), wsum = new Float32Array(OW * OH);
  let next = 0, done = 0, failed = false;
  async function lane(k) {
    if (o.wait && !(await o.wait(k))) return;     // this worker failed to load: the others carry on
    while (next < tiles.length && !failed) {
      if (o.isCancelled && o.isCancelled()) throw new Cancelled();
      const t = tiles[next++], inp = cutTile(rgba, W, t);
      const out = await o.run(inp.data, inp.w, inp.h, k);
      if (failed) return;
      if (out.w !== inp.w * 2 || out.h !== inp.h * 2) throw new Error("Model returned " + out.w + "×" + out.h + " for a " + inp.w + "×" + inp.h + " tile");
      addTile(acc, wsum, OW, out, t, W, H, overlap * 2);
      done++;
      if (o.progress) o.progress(done, tiles.length);
    }
  }
  const k = Math.max(1, Math.min(o.lanes || 1, tiles.length));
  await Promise.all(Array.from({ length: k }, (_, i) => lane(i).catch(e => { failed = true; throw e; })));
  const res = new Uint8ClampedArray(OW * OH * 4), alpha = hasAlpha(rgba) ? bilinear2x(rgba, W, H) : null;
  for (let q = 0; q < OW * OH; q++) {
    const w = wsum[q] || 1;
    res[q * 4] = Math.round(acc[q * 3] / w * 255); res[q * 4 + 1] = Math.round(acc[q * 3 + 1] / w * 255);
    res[q * 4 + 2] = Math.round(acc[q * 3 + 2] / w * 255); res[q * 4 + 3] = alpha ? alpha[q * 4 + 3] : 255;
  }
  return { data: res, w: OW, h: OH, tiles: tiles.length };
}

// RGB error against a reference: mean absolute error and PSNR.
function compare(a, b) {
  let se = 0, ae = 0, n = 0, mx = 0;
  for (let i = 0; i < a.length; i++) {
    if ((i & 3) === 3) continue;
    const d = a[i] - b[i]; se += d * d; ae += Math.abs(d); n++; if (Math.abs(d) > mx) mx = Math.abs(d);
  }
  const mse = se / n;
  return { mae: ae / n, max: mx, psnr: mse ? 10 * Math.log10(255 * 255 / mse) : Infinity };
}

// ---------- the model, in workers ----------
// The wasm build is single-threaded (no cross-origin isolation on static hosting), so speed comes from
// running several workers side by side, each on its own tile. With WebGPU, the first worker uses the GPU.
class Lane {
  constructor(want) { this.want = want; this.w = null; this.device = null; this.seq = 0; this.pending = new Map(); }
  start() {
    this.w = new Worker(new URL("worker.js", import.meta.url), { type: "module" });
    this.w.onmessage = e => {
      const m = e.data;
      if (m.type === "note") { if (m.text) console.warn(m.text); return; }
      const p = this.pending.get(m.id); if (!p) return;
      this.pending.delete(m.id);
      m.type === "error" ? p.rej(new Error(m.message)) : p.res(m);
    };
    this.w.onerror = e => { e.preventDefault(); this.fail(new Error("An AI worker crashed (" + (e.message || "out of memory?") + "). Try a smaller tile size or a smaller image.")); };
  }
  call(msg, transfer) {
    if (!this.w) this.start();
    const id = ++this.seq;
    return new Promise((res, rej) => { this.pending.set(id, { res, rej }); msg.id = id; this.w.postMessage(msg, transfer || []); });
  }
  async ready() { if (!this.device) this.device = (await this.call({ type: "init", device: this.want })).device; return this.device; }
  run(data, w, h) { return this.call({ type: "run", data, w, h }, [data.buffer]); }
  fail(err) { const p = [...this.pending.values()]; this.pending.clear(); if (this.w) this.w.terminate(); this.w = null; this.device = null; p.forEach(x => x.rej(err)); }
}
function cpuLanes() {
  const cores = navigator.hardwareConcurrency || 2, mem = navigator.deviceMemory || 8;
  return Math.max(1, Math.min(4, Math.floor(cores / 2), mem <= 2 ? 1 : mem <= 4 ? 2 : 4));
}
const engine = {
  lanes: [], want: null,
  ready(want) {
    if (this.loading && this.loading.want === want) return this.loading.p;
    if (this.lanes.length && this.want === want && this.lanes[0].w) return Promise.resolve(this.summary());
    const p = this.load(want), entry = { want, p };
    this.loading = entry;
    const done = () => { if (this.loading === entry) this.loading = null; };
    p.then(done, done);
    return p;
  },
  async load(want) {
    this.kill();
    const n = cpuLanes(), lanes = [];
    for (let i = 0; i < n; i++) lanes.push(new Lane(i === 0 && want === "auto" ? "auto" : "wasm"));
    this.lanes = lanes; this.want = want;
    // Building the model session takes ~10 s per worker (it is a big graph). The first worker loads alone (it decides
    // GPU vs CPU, fails fast if the model can't load, and fills the browser cache); the others load in the background
    // and join the work as they become ready.
    await lanes[0].ready();
    lanes.slice(1).forEach(l => { l.readyP = l.ready().then(() => true, () => false); });
    lanes[0].readyP = Promise.resolve(true);
    // This model is 8-bit quantized. On many GPUs WebGPU hands those integer ops back to the CPU and ends up
    // slower than a CPU worker, and some GPU drivers return wrong numbers. So run the same test tile on the GPU
    // worker and a CPU worker: keep the GPU only if it matches the CPU result AND is faster.
    if (lanes[0].device === "webgpu" && lanes.length > 1 && await lanes[1].readyP) {
      const probe = () => { const d = new Float32Array(3 * 32 * 32); for (let i = 0; i < d.length; i++) d[i] = ((i * 37) % 97 > 48 ? 0.85 : 0.15) * (0.6 + 0.4 * Math.sin(i / 50)); return d; };
      const time = async l => { const t = performance.now(), r = await l.run(probe(), 32, 32); r.ms = performance.now() - t; return r; };
      await time(lanes[0]); await time(lanes[1]);              // warm-up (shader compile, first allocation)
      const g = await time(lanes[0]), c = await time(lanes[1]);
      let mx = 0, sum = 0;
      for (let i = 0; i < c.data.length; i++) { const d = Math.abs(g.data[i] - c.data[i]) * 255; sum += d; if (!(d <= mx)) mx = d; }
      const mean = sum / c.data.length, same = mean < 2.5 && mx < 40;   // NaN fails too
      this.probe = { gpuMs: Math.round(g.ms), cpuMs: Math.round(c.ms), meanDiff: +mean.toFixed(2), maxDiff: Math.round(mx), gpuKept: same && g.ms <= c.ms };
      if (!this.probe.gpuKept) {
        lanes[0].fail(new Cancelled()); lanes[0] = new Lane("wasm");
        lanes[0].readyP = lanes[0].ready().then(() => true, () => false);
        [lanes[0], lanes[1]] = [lanes[1], lanes[0]];               // a loaded CPU worker goes first
      }
    }
    return this.summary();
  },
  wait(k) { const l = this.lanes[k]; return l ? l.readyP : Promise.resolve(false); },
  get device() { return this.lanes[0] && this.lanes[0].device; },
  summary() {
    const gpu = this.lanes[0] && this.lanes[0].want === "auto" && this.lanes[0].device === "webgpu" ? 1 : 0, cpu = this.lanes.length - gpu;
    return { gpu, cpu, lanes: this.lanes.length, short: (gpu ? "GPU + " : "") + cpu + " CPU", label: (gpu ? "GPU" + (cpu ? " + " : "") : "") + (cpu ? cpu + " CPU worker" + (cpu > 1 ? "s" : "") : "") };
  },
  run(data, w, h, lane) { return this.lanes[(lane || 0) % this.lanes.length].run(data, w, h); },
  kill() { this.lanes.forEach(l => l.fail(new Cancelled())); this.lanes = []; this.loading = null; }
};

// ---------- UI state ----------
const saved = S3.store("upscale") || {};
const st = {
  scale: saved.scale === 4 ? 4 : 2, eng: saved.eng === "wasm" ? "wasm" : "auto",
  tile: [64, 96, 128, 192].indexOf(saved.tile) >= 0 ? saved.tile : 64, before: saved.before === "pixels" ? "pixels" : "smooth",
  src: null,      // {name, canvas, w, h}
  out: null,      // {canvas, w, h, ms, device, scale}
  beforeC: null, busy: false, cancel: false,
  zoom: 1, px: 0, py: 0, split: 0.5, msPerMP: saved.msPerMP || 0
};
function save() { S3.store("upscale", { scale: st.scale, eng: st.eng, tile: st.tile, before: st.before, msPerMP: st.msPerMP }); }
function canvas(w, h) { const c = document.createElement("canvas"); c.width = Math.max(1, w); c.height = Math.max(1, h); return c; }
function mp(w, h) { return w * h / 1e6; }
function fmtMs(ms) {
  const s = Math.round(ms / 1000);
  if (s < 60) return Math.max(1, s) + " s";
  if (s < 3600) return Math.floor(s / 60) + " min " + String(s % 60).padStart(2, "0") + " s";
  return Math.floor(s / 3600) + " h " + String(Math.round(s % 3600 / 60)).padStart(2, "0") + " min";
}
function seg(el, value, onPick) {
  $$("button", el).forEach(b => {
    b.setAttribute("aria-pressed", String(b.dataset.v === String(value)));
    b.addEventListener("click", () => { if (st.busy) return; $$("button", el).forEach(x => x.setAttribute("aria-pressed", String(x === b))); onPick(b.dataset.v); });
  });
}

// ---------- loading ----------
async function loadFile(file) {
  if (st.busy) { S3.toast("Wait for the current upscale to finish, or cancel it", "error"); return; }
  const url = URL.createObjectURL(file), img = new Image();
  img.src = url;
  try { await img.decode(); } catch (e) { URL.revokeObjectURL(url); S3.toast("Can't read " + (file.name || "this image") + " (unsupported or damaged)", "error"); return; }
  const c = canvas(img.naturalWidth, img.naturalHeight);
  c.getContext("2d").drawImage(img, 0, 0);
  setSource(c, file.name || "image.png", url);
}
function setSource(c, name, thumbUrl) {
  if (st.src && st.src.url) URL.revokeObjectURL(st.src.url);
  st.src = { canvas: c, w: c.width, h: c.height, name, url: thumbUrl || null };
  st.out = null; st.beforeC = null;
  $("#drop").hidden = true; $("#file").hidden = false;
  $("#thumb").src = thumbUrl || c.toDataURL();
  $("#fname").textContent = name;
  $("#fdims").textContent = c.width + " × " + c.height + " px · " + mp(c.width, c.height).toFixed(2) + " MP";
  $("#stats").hidden = true; $("#save").disabled = true;
  checkSize(); fit(); draw();
  engine.ready(st.eng).catch(() => {});   // start loading the model now, while the user picks settings
}

// Too big? Explain, offer to shrink. Returns true if the current image can be processed.
function limitFor(scale) { return scale === 4 ? MAX_IN_MP / 4 : MAX_IN_MP; }
// Until the first run on this device, assume ~37 min per input megapixel per CPU worker (measured on a desktop CPU).
function estimateMs(w, h, scale) {
  const per = st.msPerMP || 2.2e6 / cpuLanes();
  return per * (mp(w, h) + (scale === 4 ? mp(w * 2, h * 2) : 0));
}
function checkSize() {
  const s = st.src, warn = $("#warn"), txt = $("#warnText"), shrink = $("#shrink");
  updateGo();
  if (!s) { warn.hidden = true; return false; }
  const lim = limitFor(st.scale), m = mp(s.w, s.h), side = Math.max(s.w, s.h) * st.scale;
  const msgs = [];
  let ok = true;
  if (m > lim + 1e-9 || side > MAX_OUT_SIDE) {
    ok = false;
    msgs.push("This image is " + m.toFixed(2) + " MP. For " + st.scale + "x the limit is " + lim.toFixed(2) + " MP" +
      (st.scale === 4 ? " (4x runs the model twice)" : "") + ", so the browser doesn't run out of memory or take hours. Crop it, or shrink it to the limit.");
  } else {
    const est = estimateMs(s.w, s.h, st.scale);
    const load = engine.lanes.length ? "" : " (plus 10–30 s to load the AI model the first time)";
    if (st.scale === 4) msgs.push("4x runs the model twice: the second pass works on an image four times larger, so it takes about 5x as long as 2x. Estimated: ~" + fmtMs(est) + load + ".");
    else if (est > 30000) msgs.push("This will take roughly " + fmtMs(est) + load + (st.msPerMP ? "." : ", a first guess that gets accurate after your first upscale.") + " You can cancel at any time.");
  }
  warn.hidden = !msgs.length; txt.textContent = msgs.join(" ");
  warn.classList.toggle("bad", !ok); shrink.hidden = ok;
  $("#go").disabled = !ok || st.busy;
  return ok;
}
function shrinkToFit() {
  const s = st.src; if (!s) return;
  const lim = limitFor(st.scale) * 1e6;
  let f = Math.min(1, Math.sqrt(lim / (s.w * s.h)), MAX_OUT_SIDE / st.scale / Math.max(s.w, s.h));
  const w = Math.max(1, Math.floor(s.w * f)), h = Math.max(1, Math.floor(s.h * f));
  const c = canvas(w, h), x = c.getContext("2d");
  x.imageSmoothingQuality = "high"; x.drawImage(s.canvas, 0, 0, w, h);
  setSource(c, s.name.replace(/(\.[^.]+)?$/, "-" + w + "x" + h + "$1"), null);
  S3.toast("Shrunk to " + w + " × " + h);
}
function updateGo() { $("#goLbl").textContent = "Upscale " + st.scale + "x"; }

// ---------- run ----------
async function upscale(quiet) {
  const s = st.src; if (!s || st.busy || !checkSize()) return;
  st.busy = true; st.cancel = false;
  $("#go").disabled = true; $("#cancel").hidden = false; $("#prog").hidden = false;
  const bar = $("#bar"), pText = $("#pText"), pEta = $("#pEta");
  bar.removeAttribute("value"); pText.textContent = "Loading the AI model…"; pEta.textContent = "";
  const t0 = performance.now();
  try {
    const device = await engine.ready(st.eng);
    if (st.cancel) throw new Cancelled();
    const pr = engine.probe;
    $("#engineNote").textContent = "Running on " + device.label + (device.gpu ? " (WebGPU + WebAssembly)." : " (WebAssembly).") +
      (pr && !pr.gpuKept ? " Your GPU was tested and skipped: " + (pr.meanDiff >= 2.5 || pr.maxDiff >= 40 ? "its results didn't match the CPU's." : "it was slower than the CPU for this model.") : "") +
      " Everything stays on this device.";
    let rgba = s.canvas.getContext("2d").getImageData(0, 0, s.w, s.h).data, w = s.w, h = s.h;
    const passes = st.scale === 4 ? 2 : 1;
    // progress weight = input pixels of each pass (pass 2 is 4x bigger)
    const weights = passes === 2 ? [1, 4] : [1], total = weights.reduce((a, b) => a + b, 0);
    const tStart = performance.now();
    let doneBefore = 0, pace = 0, lastT = tStart, lastF = 0;
    for (let p = 0; p < passes; p++) {
      const r = await upscale2x(rgba, w, h, {
        tile: st.tile, isCancelled: () => st.cancel,
        lanes: device.lanes, wait: k => engine.wait(k), run: (d, tw, th, k) => engine.run(d, tw, th, k),
        progress: (i, n) => {
          const frac = (doneBefore + weights[p] * i / n) / total;
          bar.value = frac;
          pText.textContent = (passes > 1 ? "Pass " + (p + 1) + " of 2 · " : "") + "tile " + i + " of " + n;
          // time left from the recent pace (workers that are still loading make the first tiles look slow)
          const now = performance.now(), dt = (now - lastT) / Math.max(1e-9, frac - lastF);
          pace = pace ? pace * 0.85 + dt * 0.15 : dt; lastT = now; lastF = frac;
          pEta.textContent = frac < 1 ? "~" + fmtMs(pace * (1 - frac)) + " left" : "";
        }
      });
      doneBefore += weights[p];
      rgba = r.data; w = r.w; h = r.h;
      await new Promise(r2 => setTimeout(r2, 0));
    }
    const workMs = performance.now() - tStart;
    st.msPerMP = workMs / (mp(s.w, s.h) + (passes === 2 ? mp(s.w * 2, s.h * 2) : 0)); save();
    const c = canvas(w, h);
    c.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
    st.out = { canvas: c, w, h, ms: performance.now() - t0, device, scale: st.scale };
    makeBefore(); showStats(); fitOrOne(); draw();
    $("#save").disabled = false;
    if (!quiet) S3.toast("Upscaled to " + w + " × " + h);
    S3.done();
    return st.out;
  } catch (e) {
    if (e instanceof Cancelled || e.name === "Cancelled") S3.toast("Cancelled");
    else { S3.toast(e.message || "Upscaling failed", "error"); throw e; }
  } finally {
    st.busy = false; $("#cancel").hidden = true; $("#prog").hidden = true; checkSize();
  }
}
function cancel() {
  if (!st.busy) return;
  st.cancel = true;
  engine.kill();   // stops the tile in flight right away; the model reloads next time
}

function showStats() {
  const o = st.out, s = st.src;
  const rows = [[s.w + "×" + s.h, "input"], [o.w + "×" + o.h, "output (" + o.scale + "x)"], [fmtMs(o.ms), "total time"],
    [o.device.short, (o.device.gpu ? "WebGPU + WebAssembly" : "WebAssembly") + " workers"]];
  $("#stats").innerHTML = rows.map(v => '<div class="stat"><b>' + S3.esc(v[0]) + "</b><span>" + S3.esc(v[1]) + "</span></div>").join("");
  $("#stats").hidden = false;
}

// ---------- compare view ----------
const view = $("#view"), cv = $("#cv");
function makeBefore() {
  if (!st.out || !st.src) { st.beforeC = null; return; }
  const c = canvas(st.out.w, st.out.h), x = c.getContext("2d");
  x.imageSmoothingEnabled = st.before === "smooth"; x.imageSmoothingQuality = "high";
  x.drawImage(st.src.canvas, 0, 0, st.out.w, st.out.h);
  st.beforeC = c;
}
function dims() { return st.out ? { w: st.out.w, h: st.out.h } : st.src ? { w: st.src.w, h: st.src.h } : null; }
function setZoom(z, cx, cy) {
  const vw = view.clientWidth, vh = view.clientHeight;
  if (cx == null) { cx = vw / 2; cy = vh / 2; }
  z = Math.max(0.05, Math.min(16, z));
  st.px = cx - (cx - st.px) * z / st.zoom; st.py = cy - (cy - st.py) * z / st.zoom; st.zoom = z;
  $("#zoomV").textContent = Math.round(z * 100) + "%";
  draw();
}
function center(z) {
  const d = dims(); if (!d) return;
  st.zoom = z; st.px = (view.clientWidth - d.w * z) / 2; st.py = (view.clientHeight - d.h * z) / 2;
  $("#zoomV").textContent = Math.round(z * 100) + "%";
}
function fit() { const d = dims(); if (d) center(Math.min(1, (view.clientWidth - 16) / d.w, (view.clientHeight - 16) / d.h)); }
// After a run: actual pixels (1:1), so the difference is visible straight away.
function fitOrOne() { center(1); }
function draw() {
  const dpr = window.devicePixelRatio || 1, W = view.clientWidth, H = view.clientHeight;
  cv.style.width = W + "px"; cv.style.height = H + "px";
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const x = cv.getContext("2d");
  x.setTransform(1, 0, 0, 1, 0, 0); x.clearRect(0, 0, cv.width, cv.height);
  $("#empty").hidden = !!st.src;
  $("#tagL").hidden = $("#tagR").hidden = !st.out;
  if (!st.src) return;
  x.setTransform(dpr * st.zoom, 0, 0, dpr * st.zoom, dpr * st.px, dpr * st.py);
  x.imageSmoothingEnabled = st.zoom < 1; x.imageSmoothingQuality = "high";
  if (!st.out) { x.drawImage(st.src.canvas, 0, 0); return; }
  x.drawImage(st.out.canvas, 0, 0);
  const sx = st.split * W * dpr;
  x.save(); x.setTransform(1, 0, 0, 1, 0, 0); x.beginPath(); x.rect(0, 0, sx, cv.height); x.clip();
  x.setTransform(dpr * st.zoom, 0, 0, dpr * st.zoom, dpr * st.px, dpr * st.py);
  x.drawImage(st.beforeC, 0, 0); x.restore();
  x.setTransform(1, 0, 0, 1, 0, 0);
  x.fillStyle = "#ffffff"; x.fillRect(sx - dpr, 0, 2 * dpr, cv.height);
  x.shadowColor = "rgba(0,0,0,.45)"; x.shadowBlur = 8 * dpr;
  x.beginPath(); x.arc(sx, cv.height / 2, 15 * dpr, 0, Math.PI * 2); x.fill(); x.shadowBlur = 0;
  x.fillStyle = "#14171b"; x.font = (13 * dpr) + "px Inter, system-ui, sans-serif"; x.textAlign = "center"; x.textBaseline = "middle";
  x.fillText("⇆", sx, cv.height / 2 + dpr);
  $("#tagL").textContent = st.before === "smooth" ? "Before · smooth resize" : "Before · pixels";
  $("#tagR").textContent = "After · AI " + st.out.scale + "x";
}
function setSplit(v) { st.split = Math.max(0, Math.min(1, v)); draw(); }

const ptrs = new Map();
let drag = null;
function local(e) { const r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
view.addEventListener("pointerdown", e => {
  if (!st.src || e.button > 0 || e.target.closest("button")) return;
  view.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (ptrs.size === 2) { const v = [...ptrs.values()]; drag = { kind: "pinch", d: Math.hypot(v[0].x - v[1].x, v[0].y - v[1].y), z: st.zoom }; return; }
  const p = local(e);
  if (st.out && Math.abs(p.x - st.split * view.clientWidth) < 24) { drag = { kind: "split" }; setSplit(p.x / view.clientWidth); }
  else drag = { kind: "pan", x: e.clientX, y: e.clientY };
});
view.addEventListener("pointermove", e => {
  if (!drag) {
    if (st.out) view.style.cursor = Math.abs(local(e).x - st.split * view.clientWidth) < 24 ? "ew-resize" : "grab";
    return;
  }
  if (!ptrs.has(e.pointerId)) return;
  ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (drag.kind === "pinch" && ptrs.size === 2) {
    const v = [...ptrs.values()], r = cv.getBoundingClientRect();
    setZoom(drag.z * Math.hypot(v[0].x - v[1].x, v[0].y - v[1].y) / drag.d, (v[0].x + v[1].x) / 2 - r.left, (v[0].y + v[1].y) / 2 - r.top);
  } else if (drag.kind === "split") setSplit(local(e).x / view.clientWidth);
  else if (drag.kind === "pan") { st.px += e.clientX - drag.x; st.py += e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY; draw(); }
});
function endPtr(e) { ptrs.delete(e.pointerId); if (!ptrs.size) drag = null; }
view.addEventListener("pointerup", endPtr); view.addEventListener("pointercancel", endPtr);
view.addEventListener("wheel", e => { if (!st.src) return; e.preventDefault(); const p = local(e); setZoom(st.zoom * Math.exp(-e.deltaY * 0.0015), p.x, p.y); }, { passive: false });
view.addEventListener("keydown", e => {
  if (!st.src) return;
  const k = e.key;
  if (k === "+" || k === "=") setZoom(st.zoom * 1.25);
  else if (k === "-" || k === "_") setZoom(st.zoom / 1.25);
  else if (k === "0") { fit(); draw(); }
  else if (k === "1") setZoom(1);
  else if (st.out && (k === "ArrowLeft" || k === "ArrowRight") && !e.shiftKey) setSplit(st.split + (k === "ArrowLeft" ? -0.03 : 0.03));
  else if (k.indexOf("Arrow") === 0) { const s = 40; if (k === "ArrowLeft") st.px += s; if (k === "ArrowRight") st.px -= s; if (k === "ArrowUp") st.py += s; if (k === "ArrowDown") st.py -= s; draw(); }
  else return;
  e.preventDefault();
});

// ---------- wiring ----------
seg($("#scale"), st.scale, v => { st.scale = +v; save(); checkSize(); });
seg($("#engine"), st.eng, v => { st.eng = v; save(); });
seg($("#before"), st.before, v => { st.before = v; save(); makeBefore(); draw(); });
$("#tile").value = String(st.tile);
$("#tile").addEventListener("change", e => { st.tile = +e.target.value; save(); });
$("#go").addEventListener("click", () => { upscale().catch(() => {}); });
$("#cancel").addEventListener("click", cancel);
$("#shrink").addEventListener("click", shrinkToFit);
$("#change").addEventListener("click", () => { if (!st.busy) $("#drop input").click(); });
$("#zoomIn").addEventListener("click", () => setZoom(st.zoom * 1.25));
$("#zoomOut").addEventListener("click", () => setZoom(st.zoom / 1.25));
$("#fitBtn").addEventListener("click", () => { fit(); draw(); });
$("#oneBtn").addEventListener("click", () => setZoom(1));
$("#save").addEventListener("click", () => {
  if (!st.out) return;
  st.out.canvas.toBlob(b => {
    if (!b) { S3.toast("The browser could not encode the PNG", "error"); return; }
    S3.download(b, st.src.name.replace(/\.[^.]+$/, "") + "-" + st.out.scale + "x.png"); S3.done();
  }, "image/png");
});
$("#sample").addEventListener("click", () => setSource(sampleImage(), "sample.png", null));
S3.drop($("#drop"), files => { if (files[0]) loadFile(files[0]); }, { accept: "image/*" });
document.addEventListener("paste", e => {
  const f = Array.prototype.slice.call((e.clipboardData && e.clipboardData.files) || []).find(x => /^image\//.test(x.type));
  if (!f) return;
  e.preventDefault(); loadFile(f.name && f.name !== "image.png" ? f : new File([f], "pasted.png", { type: f.type }));
});
if (window.ResizeObserver) new ResizeObserver(S3.debounce(() => { if (!st.out) fit(); draw(); }, 60)).observe(view);
updateGo(); draw();

// ---------- sample: a small, soft "low-res" picture with text and edges ----------
function drawScene(x, W, H) {
  const g = x.createLinearGradient(0, 0, W, H); g.addColorStop(0, "#14335f"); g.addColorStop(1, "#2a8c9c");
  x.fillStyle = g; x.fillRect(0, 0, W, H);
  x.fillStyle = "#ffd36b"; x.beginPath(); x.arc(W * 0.8, H * 0.32, H * 0.17, 0, Math.PI * 2); x.fill();
  x.strokeStyle = "#ffffff"; x.lineWidth = W / 160;
  for (let i = 0; i < 7; i++) { x.beginPath(); x.moveTo(W * 0.06 + i * W * 0.035, H * 0.92); x.lineTo(W * 0.16 + i * W * 0.035, H * 0.58); x.stroke(); }
  x.fillStyle = "#ffffff"; x.font = "700 " + Math.round(H * 0.17) + "px Inter, Arial, sans-serif"; x.textBaseline = "alphabetic";
  x.fillText("S3eidat", W * 0.06, H * 0.3);
  x.font = "500 " + Math.round(H * 0.085) + "px Inter, Arial, sans-serif"; x.fillStyle = "#cfe8ff";
  x.fillText("Sharp text · fine lines · 2026", W * 0.06, H * 0.45);
  x.fillStyle = "#e0708b"; x.fillRect(W * 0.52, H * 0.62, W * 0.16, H * 0.24);
  x.strokeStyle = "#0d0f12"; x.lineWidth = W / 220; x.strokeRect(W * 0.52, H * 0.62, W * 0.16, H * 0.24);
  for (let i = 0; i < 6; i++) { x.fillStyle = i % 2 ? "#ffffff" : "#0d0f12"; x.fillRect(W * 0.74 + i * W * 0.03, H * 0.66, W * 0.03, H * 0.2); }
}
function sampleImage() {
  // drawn big, then shrunk 3x: like a low-resolution photo or old screenshot
  const big = canvas(870, 510); drawScene(big.getContext("2d"), 870, 510);
  const c = canvas(290, 170), x = c.getContext("2d");
  x.imageSmoothingQuality = "high"; x.drawImage(big, 0, 0, 290, 170);
  return c;
}

window.S3_DEMO = async function () {
  setSource(sampleImage(), "sample.png", null);
  st.scale = 2; $$("#scale button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === "2")));
  checkSize();
  await upscale(true);
  center(1); st.split = 0.5; draw();
};

// ---------- self-test (scripts/smoke.py): the real model on synthetic images ----------
function boxDown2(src, W, H) {
  const w = W / 2, h = H / 2, out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let k = 0; k < 4; k++) {
    const a = ((2 * y) * W + 2 * x) * 4 + k, b = a + W * 4;
    out[(y * w + x) * 4 + k] = (src[a] + src[a + 4] + src[b] + src[b + 4] + 2) >> 2;
  }
  return out;
}
function testRef() {
  const c = canvas(96, 96), x = c.getContext("2d");
  x.fillStyle = "#fff"; x.fillRect(0, 0, 96, 96);
  x.fillStyle = "#111"; x.fillRect(8, 8, 30, 4); x.fillRect(8, 16, 4, 26); x.fillRect(20, 20, 18, 18);
  x.strokeStyle = "#1d4ed8"; x.lineWidth = 3; x.beginPath(); x.moveTo(50, 8); x.lineTo(88, 44); x.stroke();
  x.fillStyle = "#dc2626"; x.beginPath(); x.arc(70, 70, 16, 0, Math.PI * 2); x.fill();
  x.fillStyle = "#111"; x.font = "700 20px Arial, sans-serif"; x.fillText("Ab", 8, 84);
  for (let i = 0; i < 5; i++) x.fillRect(42 + i * 4, 52, 2, 34);
  return x.getImageData(0, 0, 96, 96).data;
}
window.S3_SELFTEST = async function () {
  const r = {};
  // pure: tiling covers every pixel, with full-size tiles and enough overlap
  const tl = planTiles(300, 100, 128, 16);
  const cov = new Uint8Array(300 * 100);
  tl.forEach(t => { for (let y = t.y; y < t.y + t.h; y++) for (let x = t.x; x < t.x + t.w; x++) cov[y * 300 + x] = 1; });
  r.plan = tl.length === 3 && tl.every(t => t.w === 128 && t.h === 100) && cov.every(v => v === 1) && tl[2].x === 172;
  const t1 = planTiles(40, 30, 128, 16);
  r.plan = r.plan && t1.length === 1 && t1[0].w === 40 && up8(30) === 32;

  const run = (d, w, h, k) => engine.run(d, w, h, k);
  const dev = await engine.ready(st.eng);
  const ref = testRef(), small = boxDown2(ref, 96, 96);
  const t0 = performance.now();
  const single = await upscale2x(small, 48, 48, { tile: 48, run });
  r.singleMs = Math.round(performance.now() - t0);
  const tiled = await upscale2x(small, 48, 48, { tile: 32, overlap: 16, run, lanes: dev.lanes, wait: k => engine.wait(k) }); // 4 tiles, in parallel
  r.size = single.w === 96 && single.h === 96 && single.data.length === 96 * 96 * 4 && single.tiles === 1 && tiled.tiles === 4;
  const odd = await upscale2x(small.slice(0, 48 * 33 * 4), 48, 33, { tile: 128, run }); // 48x33 (pads to 48x40, crops back)
  r.odd = odd.w === 96 && odd.h === 66;
  const seam = compare(tiled.data, single.data);
  const bil = bilinear2x(small, 48, 48);
  const eAI = compare(single.data, ref), eBil = compare(bil, ref);
  r.seamMAE = +seam.mae.toFixed(2); r.seamMax = seam.max;
  r.psnrAI = +eAI.psnr.toFixed(2); r.psnrBilinear = +eBil.psnr.toFixed(2); r.maeAI = +eAI.mae.toFixed(2); r.maeBilinear = +eBil.mae.toFixed(2);
  r.tilesMatch = seam.mae < 2.5;
  r.better = eAI.psnr > eBil.psnr + 1 && eAI.mae < eBil.mae;
  r.device = dev.label;
  window.S3_SELFTEST_RESULT = r;
  return (r.plan && r.size && r.odd && r.tilesMatch && r.better) || r;
};

export { planTiles, upscale2x, bilinear2x, compare, engine, boxDown2, testRef, Lane, cutTile };
