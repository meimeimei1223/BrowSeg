// Web Worker hosting the WASM module (keeps the UI responsive; the module itself spawns
// pthread workers for multithreaded work). Messages:
//   init                              -> ready
//   loadDicom {files}                 -> series   (list of the series found; nothing is built yet)
//   buildSeries {uid, keep}           -> volume   (keep: keep the slices to switch series later)
//   clearDicom                        -> cleared  (frees the kept slices)
//   loadVolume {data | raw+datatype+slope+inter, dims, affine} -> volume   (NIfTI, x fastest)
//   run {task, roi, custom}           -> result   (label map + label names)
//   mesh {labels, items:[{id,name}]}  -> meshes
'use strict';

const BASE = new URL('./', self.location.href).href;
const PARAMS = new URL(self.location.href).searchParams;
importScripts(new URL('config.js', BASE).href);  // TSC_CONFIG: where the weights are (relative to web/ or absolute)
// fp32 weights reproduce Python TotalSegmentator exactly; fp16 (?fp16=1) halves the download
const WDIR = PARAMS.has('fp16') ? TSC_CONFIG.weightsBaseFp16 : TSC_CONFIG.weightsBase;
const MODEL_FILES = {
  298: ['total6mm_298.tsw', '6mm crop model'],
  291: ['organs_291.tsw', 'organs model'],
  292: ['vertebrae_292.tsw', 'vertebrae model'],
  293: ['cardiac_293.tsw', 'cardiac model'],
  294: ['muscles_294.tsw', 'muscles model'],
  295: ['ribs_295.tsw', 'ribs model'],
  570: ['liver_segments_570.tsw', 'liver segments model'],
  8: ['liver_vessels_8.tsw', 'liver vessels model'],
  343: ['trunk_cavities_343.tsw', 'trunk cavities model'],   // V2 (private)
  299: ['body_299.tsw', 'body model'],                       // V2 (private)
  300: ['body6mm_300.tsw', 'body model (6 mm, fast)'],       // V2 (private)
};
const CACHE_NAME = 'tsc-weights-v1';
const FORCE_CPU = PARAMS.has('cpu');

let M = null;
let device = null;
const gpuNets = {};  // task id -> GpuUNet (WebGPU backend)
const configOnly = new Set();  // task ids whose weights C++ holds as config only (WebGPU); reloaded for the CPU
let versions = {};  // "weights/x.tsw" -> "size-mtime" (server manifest): cache key, so changed files are re-fetched
let backend = 'cpu';
let gpuLostOnce = false;
// Low-memory GPU mode (after an out-of-memory error, or ?gpulowmem=1): the weights stay in JS memory and only the model
// being run is on the GPU (uploaded per stage, freed after it); tiles are read back and accumulated by C++ instead of in
// a GPU volume buffer. The 117-structure task did not fit on a 4 GB GPU otherwise (T1200, 2026-10-03).
let gpuLowMem = PARAMS.get('gpulowmem') === '1';
const gpuBytes = {};  // task id -> .tsw bytes (low-memory mode)
const SPLIT_OPS = 4;  // steps per GPU submission when split (Intel GPUs, or after a failed single submission); 4 was the fastest split on Intel UHD
let gpuErrors = 0;  // WebGPU uncapturederror events so far (counted into every result for the benchmark records)

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const mem = () => M.wasmMemory.buffer;  // always the current (possibly grown) memory
// Typed-array view on the WASM heap. The pointer argument is evaluated before mem() is read (calls such as
// tsc_tile_output() may grow the heap, which replaces the SharedArrayBuffer) and read as unsigned (>2 GB heaps).
const view = (T, ptr, n) => new T(mem(), ptr >>> 0, n);
const lastError = () => M.UTF8ToString(M._tsc_result_ptr(5) >>> 0);

let fetchSource = '';
// .tsw integrity: magic, header length, and every tensor's byte range inside the file (a truncated download or a
// partial cache entry would otherwise be zero-filled silently on the GPU path).
function tswCheck(bytes) {
  if (bytes.length < 12) return 'file too short';
  if (new TextDecoder().decode(bytes.subarray(0, 8)) !== 'TSCPPW01') return 'not a .tsw file';
  const hlen = new DataView(bytes.buffer, bytes.byteOffset + 8, 4).getUint32(0, true);
  if (12 + hlen > bytes.length) return 'truncated header';
  const base = Math.ceil((12 + hlen) / 64) * 64;
  let end = 0, n = 0;
  for (const line of new TextDecoder().decode(bytes.subarray(12, 12 + hlen)).split('\n')) {
    const t = line.trim().split(/\s+/);
    if (t[0] !== 'tensor') continue;
    const nd = +t[3];
    end = Math.max(end, +t[4 + nd] + +t[5 + nd]); n++;
  }
  if (!n) return 'no tensors in header';
  if (base + end > bytes.length) return `truncated: ${bytes.length} of ${base + end} bytes`;
  return null;
}

async function fetchWithProgress(url, label, cacheable = true) {
  let cache = null;
  try { cache = cacheable ? await caches.open(CACHE_NAME) : null; } catch (e) { /* no Cache API */ }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) { fetchSource = 'browser cache'; return new Uint8Array(await hit.arrayBuffer()); }
  }
  fetchSource = 'download';
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed: ${url} (${res.status})`);
  // GitHub Pages serves the .tsw files gzip-compressed: Content-Length is then the compressed size, smaller than the bytes
  // received. The completeness check is then left to tswCheck() (ensureModels deletes a bad cache entry), and the
  // progress is approximate (2026-10-04: with the byte comparison nothing was ever cached on the public site).
  const encoded = !!res.headers.get('Content-Encoding');
  const total = +res.headers.get('Content-Length') || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    post({ type: 'progress', stage: `download ${label}`, frac: total ? Math.min(0.99, got / total) : 0 });
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  if (cache && (encoded || !total || got === total) && !tswCheck(buf)) {
    try {
      const path = new URL(url).pathname;
      for (const req of await cache.keys()) if (new URL(req.url).pathname === path && req.url !== url) await cache.delete(req);
      await cache.put(url, new Response(buf));
    } catch (e) { /* quota */ }
  }
  post({ type: 'log', text: `${label}: downloaded ${(got / 1e6).toFixed(1)} MB` });
  return buf;
}

function copyIn(bytes) {
  const p = M._malloc(bytes.byteLength);
  if (!p) throw new Error('out of memory (malloc)');
  view(Uint8Array, p, bytes.byteLength).set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  return p;
}
function withString(s, fn) {
  const len = M.lengthBytesUTF8(s) + 1, p = M._malloc(len);
  M.stringToUTF8(s, p, len);
  try { return fn(p); } finally { M._free(p); }
}

// dist/ (WASM threads) needs SharedArrayBuffer = a cross-origin isolated page; otherwise the single-thread build dist-st/
// is used (same results, slower pre/post-processing). ?st=1 forces the single-thread build.
const DIST = (self.crossOriginIsolated && !PARAMS.has('st')) ? 'dist' : 'dist-st';

async function init() {
  const script = new URL(DIST + '/tsc.js', BASE).href;
  importScripts(script);
  M = await createTSC({
    mainScriptUrlOrBlob: script,
    locateFile: (p) => new URL(DIST + '/' + p, BASE).href,
    onProgress: (stage, frac) => post({ type: 'progress', stage, frac }),
    print: (t) => post({ type: 'log', text: t }),
    printErr: (t) => post({ type: 'log', text: t }),
  });
  // ?order=3 reproduces TotalSegmentator <= 2.15 (input resampling spline order 3); default 1 = TS >= 2.16
  if (PARAMS.has('order') && M._tsc_set_resampling_order(+PARAMS.get('order'))) throw new Error('order must be 1 or 3');
  // ?threads=N limits the WASM thread pool (benchmark: estimate of a single-threaded build with threads=1)
  if (PARAMS.has('threads')) M._tsc_set_threads(Math.max(1, +PARAMS.get('threads')));
  const cmText = await (await fetch(new URL('../weights/classmaps.txt', BASE).href)).text();
  if (withString(cmText, (p) => M._tsc_load_classmaps(p))) throw new Error(lastError());
  if (!FORCE_CPU && self.navigator.gpu) {
    try {
      importScripts(new URL('gpu_unet.js', BASE).href);
      // ?gpuchunk=N: send at most N ops per command buffer and wait for each (for integrated GPUs whose driver
      // resets a job that runs longer than its time limit). Default 0 = whole network in one submission.
      if (PARAMS.has('gpuchunk')) GpuUNet.chunkOps = Math.max(0, +PARAMS.get('gpuchunk') | 0);
      if (PARAMS.get('gpuchunkwait') === '0') GpuUNet.chunkWait = false;  // split the submission without waiting for each part
      // ?gpuincomplete=once|all: test hook, the completion counter reports one command buffer missing (once / every time)
      if (['once', 'all'].includes(PARAMS.get('gpuincomplete'))) GpuUNet.simulateIncomplete = PARAMS.get('gpuincomplete');
      gpuErrors = 0;
      await openGpu();
    } catch (e) {
      post({ type: 'log', text: `WebGPU unavailable (${e.message}), using CPU/WASM` });
    }
  }
  // gpuWanted: the CPU was not forced with ?cpu=1, so a CPU backend means WebGPU was not available (the page shows a hint)
  post({ type: 'ready', threads: M._tsc_result_int(8), backend, gpuWanted: !FORCE_CPU, build: DIST, isolated: !!self.crossOriginIsolated,
         version: M.UTF8ToString(M._tsc_version() >>> 0) });
}

// Opens the WebGPU device (at start, and once more after a device loss, see run()).
async function openGpu() {
  device = await GpuUNet.createDevice();
  backend = 'webgpu';
  // Intel GPUs: submit the network in parts of SPLIT_OPS steps unless ?gpuchunk is given. On Intel UHD (Linux i915)
  // one patch submitted at once sometimes exceeded the driver's time limit and the GPU was reset; split, 33 runs
  // passed (2026-10-03). Other GPUs keep the single submission measured in the paper.
  if (!PARAMS.has('gpuchunk') && /intel/i.test(GpuUNet.adapterInfo.vendor) && GpuUNet.chunkOps === 0) {
    GpuUNet.chunkOps = SPLIT_OPS;
    post({ type: 'log', text: `WebGPU: Intel GPU (${GpuUNet.adapterInfo.architecture || 'unknown architecture'}), network submitted in parts of ${SPLIT_OPS} steps` });
  }
  const dev = device;
  // WebGPU errors outside an error scope (e.g. a buffer whose allocation failed being used in a submit, which the
  // specification turns into a dropped submission and all-zero output) only surface here; recorded in the log so
  // that an all-zero result has its cause next to it (Firefox 156, 2026-09-28; see paper_jiim/PENDING.md).
  dev.addEventListener('uncapturederror', (ev) => {
    if (++gpuErrors <= 20) post({ type: 'log', text: `WebGPU uncaptured error (${ev.error.constructor.name}): ${ev.error.message}` });
    else if (gpuErrors === 21) post({ type: 'log', text: 'WebGPU uncaptured error: further errors are not logged' });
  });
  dev.lost.then((info) => {  // GPU reset / driver update: continue on the CPU
    if (device !== dev) return;
    device = null; backend = 'cpu';
    for (const k of Object.keys(gpuNets)) delete gpuNets[k];
    post({ type: 'log', text: closingDevice ? 'WebGPU device closed to return its memory' : `WebGPU device lost (${info.message}); continuing with the CPU (WASM)` });
    post({ type: 'backend', backend: 'cpu' });
  });
}
let gpuReopened = 0;
let closingDevice = false;  // the device is being closed on purpose (out-of-memory retry), not lost

// After a GPU failure: waits up to 3 s for a pending device loss to be reported (Firefox reported it only after the
// retry had started, 2026-10-03), then opens a new device once per session if the device was lost.
// Returns 'alive' (same device), 'reopened' (new device) or 'none' (no device).
async function recoverDevice() {
  if (device) await Promise.race([device.lost, new Promise((r) => setTimeout(r, 3000))]);
  await new Promise((r) => setTimeout(r, 50));  // the device.lost handler runs first
  if (device) return 'alive';
  if (gpuReopened >= 1 || !self.navigator.gpu) return 'none';
  gpuReopened++;
  try {
    await openGpu();
    post({ type: 'log', text: 'WebGPU: new device opened for the retry' });
    post({ type: 'backend', backend: 'webgpu' });
    return 'reopened';
  } catch (e) {
    post({ type: 'log', text: `WebGPU: no new device (${e.message})` });
    return 'none';
  }
}

// Downloads (or takes from cache) every model the task needs. custom: {id: url} of fine-tuned models.
async function ensureModels(task, roi, custom, forceCpu = false) {
  const gpu = device && !forceCpu ? device : null;  // forceCpu: load full weights into C++ even though WebGPU is available
  const out = M._malloc(64);
  const n = withString(task, (tp) => withString(roi || '-', (rp) => M._tsc_task_models(tp, rp, out, 16)));
  const ids = Array.from(view(Int32Array, out, Math.max(n, 0)));
  M._free(out);
  if (n < 0) throw new Error(lastError());
  // Only the models of the current task stay on the GPU: the others are freed first. Firefox (156, Windows) returned
  // empty label maps without any WebGPU error when the weights of several tasks were kept (2026-09-29, see
  // paper_jiim/WORKLOG.md). Uploading a model again takes < 0.1 s. ?keepgpu=1 keeps them (old behaviour).
  if (gpu && !PARAMS.has('keepgpu'))
    for (const k of Object.keys(gpuNets)) if (!ids.includes(+k)) { gpuNets[k].destroy(); delete gpuNets[k]; }
  for (const k of Object.keys(gpuBytes)) if (!ids.includes(+k)) delete gpuBytes[k];
  for (const tid of ids) {
    if (M._tsc_has_model(tid) && (gpu ? (gpuNets[tid] || (gpuLowMem && gpuBytes[tid])) : !configOnly.has(tid))) continue;
    let url, label;
    let rel;
    if (MODEL_FILES[tid]) { rel = WDIR.replace('../', '') + MODEL_FILES[tid][0]; label = MODEL_FILES[tid][1]; url = new URL(WDIR + MODEL_FILES[tid][0], BASE).href; }
    else if (custom && custom[tid]) { rel = custom[tid]; label = `custom model ${tid}`; url = new URL('../' + rel, BASE).href; }
    else throw new Error(`no weights for model ${tid}`);
    if (versions[rel]) url += `?v=${versions[rel]}`;
    // timings of the model hand-over (logged): file -> JS -> WASM (config / CPU weights) -> GPU buffers
    const t0 = performance.now();
    const bytes = await fetchWithProgress(url, label, !!MODEL_FILES[tid]);
    const bad = tswCheck(bytes);
    if (bad) {  // truncated / corrupt file: never keep it in the cache
      try { const c = await caches.open(CACHE_NAME); await c.delete(url); } catch (e) { /* no Cache API */ }
      throw new Error(`${label}: invalid weight file (${bad})`);
    }
    const t1 = performance.now();
    post({ type: 'progress', stage: `prepare ${label}`, frac: 0 });
    // WebGPU: C++ only needs the config -> hand over just the header (magic + length + text), not 125 MB
    const head = gpu ? bytes.subarray(0, 12 + new DataView(bytes.buffer, bytes.byteOffset + 8, 4).getUint32(0, true)) : bytes;
    const p = copyIn(head);
    const r = M._tsc_load_weights(p, head.length, gpu ? 0 : 1);
    M._free(p);
    if (r < 0) throw new Error(lastError());
    if (gpu) configOnly.add(tid); else configOnly.delete(tid);
    const t2 = performance.now();
    let gpuNote = '';
    if (gpu && gpuLowMem) {
      gpuBytes[tid] = bytes;  // uploaded to the GPU only while this model runs (segmentGpu)
      gpuNote = ', kept in memory (low-memory mode: one model on the GPU at a time)';
    } else if (gpu) {
      gpuNets[tid] = new GpuUNet(gpu, bytes);  // parse + writeBuffer of every tensor + shader pipelines
      const t3 = performance.now();
      await gpu.queue.onSubmittedWorkDone();   // uploads actually finished
      gpuNote = `, GPU upload + pipelines ${((t3 - t2) / 1000).toFixed(2)} s (queue done ${((performance.now() - t2) / 1000).toFixed(2)} s)`;
    }
    post({ type: 'log', text: `model ${tid} (${(bytes.length / 1e6).toFixed(0)} MB): ${fetchSource} ${((t1 - t0) / 1000).toFixed(2)} s, ` +
                              `WASM ${gpu ? 'config' : 'weights'} ${((t2 - t1) / 1000).toFixed(2)} s${gpuNote}` });
  }
}

function volumeMessage() {
  const shape = [0, 1, 2].map((i) => M._tsc_ct_info(i));
  const zooms = [3, 4, 5].map((i) => M._tsc_ct_info(i));
  const n = shape[0] * shape[1] * shape[2];
  const f = view(Float32Array, M._tsc_ct_ptr(), n);
  const ct = new Int16Array(n);
  for (let i = 0; i < n; i++) { const v = Math.round(f[i]); ct[i] = v < -32768 ? -32768 : v > 32767 ? 32767 : v; }
  const affine = Array.from(view(Float64Array, M._tsc_ct_affine(), 16));
  post({ type: 'volume', ct, shape, zooms, affine, log: M.UTF8ToString(M._tsc_result_ptr(6) >>> 0) }, [ct.buffer]);
}

function loadDicom(files) {
  M._tsc_clear_dicom();
  let n = 0;
  for (let i = 0; i < files.length; i++) {
    const bytes = new Uint8Array(files[i]);
    const p = copyIn(bytes);
    n += M._tsc_add_dicom(p, bytes.length);
    M._free(p);
    if (i % 32 === 0) post({ type: 'progress', stage: 'read DICOM', frac: i / files.length });
  }
  if (n < 1) throw new Error(TSC_CONFIG.lang === 'en' ? 'No DICOM images found (uncompressed DICOM is required)' : 'DICOM の画像が見つかりません（非圧縮の DICOM が必要です）');
  post({ type: 'log', text: `DICOM: ${n} images of ${files.length} files` });
  post({ type: 'series', list: seriesList(), files: files.length, images: n });
}

// series in the added DICOM files (default choice first), see tsc_list_series
function seriesList() {
  const txt = M.UTF8ToString(M._tsc_list_series() >>> 0);
  return txt.split('\n').filter(Boolean).map((line) => {
    const f = line.split('\t');
    return { uid: f[0], description: f[1], modality: f[2], number: +f[3], slices: +f[4], other: +f[5], rows: +f[6], cols: +f[7],
             spacing: [+f[8], +f[9]], sliceSpacing: +f[10], extent: +f[11], time: f[12] || '', contrast: f[13] || '', problem: f[14] || '' };
  });
}

// builds the CT of one series; keep: keep the DICOM slices for choosing another series later
function buildSeries({ uid, keep }) {
  post({ type: 'progress', stage: 'dicom', frac: 0 });
  if (withString(uid || '', (p) => M._tsc_load_dicom_series(p, keep ? 1 : 0))) throw new Error((M._tsc_ct_info(0) ? '' : '[CT_LOST] ') + lastError());
  volumeMessage();
}

// data: Float32Array (x fastest), or raw: the voxel bytes in the NIfTI's own type (Nifti.readRaw; converted and reoriented
// in one pass by tsc_set_volume_raw, about half the peak memory of the float path for int16 CTs)
function loadVolume({ data, raw, datatype, slope, inter, dims, affine }) {
  const src = raw || new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const p = copyIn(src);
  const ap = M._malloc(16 * 8);
  view(Float64Array, ap, 16).set(affine);
  const r = raw ? M._tsc_set_volume_raw(p, datatype, slope, inter, dims[0], dims[1], dims[2], ap)
                : M._tsc_set_volume(p, dims[0], dims[1], dims[2], ap);
  M._free(p); M._free(ap);
  if (r) throw new Error((M._tsc_ct_info(0) ? '' : '[CT_LOST] ') + lastError());
  volumeMessage();
}

// WebGPU path: C++ does everything except the network; tiles are run and accumulated
// (fp16, bit-identical to C++) on the GPU, logits read back once per model.
async function segmentGpu(task, roi) {
  if (withString(task, (tp) => withString(roi || '-', (rp) => M._tsc_begin(tp, rp)))) throw new Error(lastError());
  let tid;
  while ((tid = M._tsc_stage())) {
    let net = gpuNets[tid], temporary = false;
    if (!net && gpuLowMem && gpuBytes[tid] && device) {  // low-memory mode: this model only, for this stage
      net = new GpuUNet(device, gpuBytes[tid]);
      await device.queue.onSubmittedWorkDone();
      temporary = true;
    }
    if (!net) throw new Error(`GPU model ${tid} unavailable (device lost)`);
    const name = M.UTF8ToString(M._tsc_stage_name() >>> 0);
    const nt = M._tsc_num_tiles();
    const P = net.cfg.patch[0] * net.cfg.patch[1] * net.cfg.patch[2];
    const ps = [3, 4, 5].map((a) => M._tsc_tile_geom(0, a));
    try {
    if (!net.plan) await net.buildPlan();
    const accBytes = net.cfg.numClasses * ps[0] * ps[1] * ps[2] * 4;
    post({ type: 'log', text: `model ${tid}: GPU memory weights ${(net.weightBytes / 2 ** 20).toFixed(0)} MiB, work buffers ${(net.planBytes / 2 ** 20).toFixed(0)} MiB` +
                              (gpuLowMem ? ' (low-memory mode)' : `, accumulation ${(accBytes / 2 ** 20).toFixed(0)} MiB`) });
    if (!gpuLowMem && net.canAccumulate(ps)) {
      const tp = performance.now();
      await net.beginVolume(view(Float32Array, M._tsc_gaussian_ptr(), P).slice(), ps);
      post({ type: 'log', text: `model ${tid}: GPU work buffers (plan) ${((performance.now() - tp) / 1000).toFixed(2)} s` });
      for (let t = 0; t < nt; t++) {
        post({ type: 'progress', stage: name, frac: t / nt });
        const x = view(Float32Array, M._tsc_tile_input(t), P).slice();
        M._tsc_tile_count(t);
        await net.forwardAccumulate(x, [0, 1, 2].map((a) => M._tsc_tile_geom(t, a)));  // resolves at once unless ?gpuchunk splits the submission
      }
      const logits = await net.readLogits();
      if (PARAMS.has('gpulost') && !gpuLostOnce) { gpuLostOnce = true; device.destroy(); }  // test hook: simulate losing the GPU
      if (allZero(logits) || PARAMS.get('gpufail') === String(tid) || PARAMS.get('gpufail') === 'all')  // ?gpufail=<model id>|all: test hook
        throw new Error(`model ${tid} returned an all-zero result`);  // Firefox 156 did this silently (2026-09-29)
      view(Uint16Array, M._tsc_logits_ptr(), logits.length).set(logits);
    } else {
      for (let t = 0; t < nt; t++) {
        post({ type: 'progress', stage: name, frac: t / nt });
        const x = view(Float32Array, M._tsc_tile_input(t), P).slice();
        const y = await net.forward(x);
        view(Float32Array, M._tsc_tile_output(), y.length).set(y);
        M._tsc_tile_accumulate(t);
      }
    }
    if (M._tsc_advance()) throw new Error(lastError());
    } finally {
      await net.popVolumeScopes();
      net.releasePlan();  // activation / accumulation buffers (weights stay)
      if (temporary) net.destroy();  // low-memory mode: the weights leave the GPU too
    }
  }
  if (M._tsc_finish()) throw new Error(lastError());
  // An empty label map from the GPU is checked on the CPU: a GPU reset once made the 6 mm locating model find nothing,
  // so that the task ended with an empty map and no error (Intel UHD, 2026-10-03). If the CPU also finds nothing, the
  // empty map is the answer.
  const n = [0, 1, 2].map((i) => M._tsc_ct_info(i)).reduce((a, b) => a * b, 1);
  const lab = view(Uint8Array, M._tsc_labels_ptr(), n);
  let any = false;
  for (let i = 0; i < n && !any; i++) any = lab[i] !== 0;
  if (!any) { const e = new Error('empty label map from WebGPU'); e.noGpuRetry = true; throw e; }
}

// fp16 words that are +0 or -0
function allZero(h) {
  for (let i = 0; i < h.length; i++) if ((h[i] & 0x7fff) !== 0) return false;
  return true;
}

// CPU networks for the models of a task (after a GPU failure the C++ side only holds their configuration).
// The device is not touched here: it may have been lost meanwhile (device.lost handler), and must stay lost.
async function ensureCpuModels(task, roi, custom) {
  await ensureModels(task, roi, custom, true);
}

async function run({ task, roi, custom, keep, echo }) {
  const t0 = performance.now();
  await ensureModels(task, roi, custom);
  const modelSeconds = (performance.now() - t0) / 1000;   // weights: cache/download, WASM config, GPU upload (0 when already loaded)
  let used = backend, fallback = null;
  if (backend === 'webgpu') {
    try {
      try {
        await segmentGpu(task, roi);
      } catch (e) {
        // One retry on the GPU, then the CPU. Out of GPU memory: low-memory mode (one model on the GPU at a time, tiles
        // accumulated by C++). Otherwise: the network submitted in parts (slow GPUs whose driver resets a long job).
        // Either mode stays on for the rest of the session.
        const msg = String(e && e.message || e);
        const oom = /out.of.(device.)?memory|OUT_OF_DEVICE_MEMORY|out-of-memory|allocat/i.test(msg);
        if (e.noGpuRetry || (oom ? gpuLowMem : GpuUNet.chunkOps > 0)) throw e;
        fallback = msg;
        // The failure may have cost the device (driver reset: Firefox on Windows; out of memory: Chrome/Vulkan on a 4 GB
        // GPU); a lost device is replaced once per session (recoverDevice).
        if ((await recoverDevice()) === 'none') throw e;
        if (oom) {
          gpuLowMem = true;
          for (const k of Object.keys(gpuNets)) { gpuNets[k].destroy(); delete gpuNets[k]; }
          // The memory of the failed attempt was not returned by destroying its buffers (T1200 4 GB, Chrome/Vulkan: still
          // 3.9 GB in use at the retry, 2026-10-04); it certainly is with the device: close it and open a new one.
          if (device && gpuReopened < 1) { closingDevice = true; device.destroy(); await recoverDevice(); closingDevice = false; }
          if (!device) throw e;
          post({ type: 'warning', text: `WebGPU ran out of GPU memory (${msg}); retrying with one model on the GPU at a time` });
          used = 'webgpu (low-memory mode after out of memory)';
        } else {
          GpuUNet.chunkOps = SPLIT_OPS;
          post({ type: 'warning', text: `WebGPU failed (${msg}); retrying on the GPU with the network submitted in parts of ${SPLIT_OPS} steps` });
          used = 'webgpu (split after failure)';
        }
        try {
          await ensureModels(task, roi, custom);
          await segmentGpu(task, roi);
        } catch (e2) {
          // the device was lost during the retry (its loss was reported late): once more on a new device
          if (e2.noGpuRetry || (await recoverDevice()) !== 'reopened') throw e2;
          await ensureModels(task, roi, custom);
          await segmentGpu(task, roi);
        }
      }
    } catch (e) {
      // A wrong GPU result must never be returned as if it were right: the label map is recomputed on the CPU.
      fallback = (fallback ? fallback + '; GPU retry: ' : '') + String(e && e.message || e);
      post({ type: 'warning', text: `WebGPU failed (${fallback}); recomputing on the CPU` });
      post({ type: 'progress', stage: 'cpu fallback', frac: 0 });
      await ensureCpuModels(task, roi, custom);
      used = 'cpu (after WebGPU failure)';
      if (withString(task, (tp) => withString(roi || '-', (rp) => M._tsc_run(tp, rp)))) throw new Error(lastError());
    }
  } else {
    // the device may have been lost while ensureModels() awaited (then the C++ side holds only the configurations)
    await ensureCpuModels(task, roi, custom);
    if (withString(task, (tp) => withString(roi || '-', (rp) => M._tsc_run(tp, rp)))) throw new Error(lastError());
  }
  const n = [0, 1, 2].map((i) => M._tsc_ct_info(i)).reduce((a, b) => a * b, 1);
  const names = [];
  for (let i = 0; i < M._tsc_num_label_names(); i++) names.push(M.UTF8ToString(M._tsc_label_name(i) >>> 0));
  const taskLog = M.UTF8ToString(M._tsc_result_ptr(6) >>> 0);
  // V3 liver bundle: the label map stays in the engine (tsc_bundle_keep) instead of coming back to the page
  if (keep) {
    // echo: a copy of the label map also comes back (verification bench: agreement of every task of the bundle)
    const copy = echo ? view(Uint8Array, M._tsc_labels_ptr(), n).slice() : null;
    if (withString(keep, (kp) => M._tsc_bundle_keep(kp))) throw new Error(lastError());
    post({ type: 'result', task, roi, labels: copy, names, seconds: (performance.now() - t0) / 1000, modelSeconds, backend: used, fallback,
           heap: mem().byteLength, gpuErrors, gpuChunk: typeof GpuUNet !== 'undefined' ? GpuUNet.chunkOps : null, log: taskLog },
         copy ? [copy.buffer] : []);
    return;
  }
  const labels = view(Uint8Array, M._tsc_labels_ptr(), n).slice();
  post({ type: 'result', task, roi, labels, names, seconds: (performance.now() - t0) / 1000, modelSeconds, backend: used, fallback, heap: mem().byteLength,
         gpuErrors, gpuChunk: typeof GpuUNet !== 'undefined' ? GpuUNet.chunkOps : null, log: taskLog }, [labels.buffer]);
}

// V3 liver bundle: the kept label maps -> the BLISS simulator file set (liver_bundle.cpp) + display label map
function bundle({ margin, tumorMin, lps, ivcVein, splenic, ivcMargin, version }) {
  const t0 = performance.now();
  if (withString(version || '', (vp) => M._tsc_bundle_build(margin, tumorMin, lps ? 1 : 0, ivcVein ? 1 : 0, splenic ? 1 : 0, ivcMargin ?? 10, vp))) throw new Error(lastError());
  const blob = view(Uint8Array, M._tsc_bundle_blob_ptr(), M._tsc_bundle_blob_len()).slice();
  const files = [], dec = new TextDecoder();
  let log = '';
  for (let p = 0; p < blob.length;) {
    const a = blob.indexOf(0, p), b = blob.indexOf(0, a + 1);
    if (a < 0 || b < 0) break;
    const name = dec.decode(blob.subarray(p, a)), body = blob.slice(a + 1, b);
    if (name === '__log__') log = dec.decode(body); else files.push({ name, data: body });
    p = b + 1;
  }
  M._tsc_bundle_clear();
  const n = [0, 1, 2].map((i) => M._tsc_ct_info(i)).reduce((x, y) => x * y, 1);
  const labels = view(Uint8Array, M._tsc_labels_ptr(), n).slice();
  const names = [];
  for (let i = 0; i < M._tsc_num_label_names(); i++) names.push(M.UTF8ToString(M._tsc_label_name(i) >>> 0));
  post({ type: 'bundle', files, log, labels, names, seconds: (performance.now() - t0) / 1000 }, [labels.buffer, ...files.map((f) => f.data.buffer)]);
}

function mesh({ labels, items, seq }) {
  const numel = [0, 1, 2].map((i) => M._tsc_ct_info(i)).reduce((x, y) => x * y, 1);
  if (labels.length !== numel) throw new Error(`label map size ${labels.length} does not match the volume (${numel})`);
  const p = copyIn(labels);
  const r = M._tsc_set_labels(p);
  M._free(p);
  if (r) throw new Error(lastError());
  const spec = items.map((it) => `${it.id}\t${it.name}`).join('\n');
  if (withString(spec, (sp) => M._tsc_build_meshes(sp))) throw new Error(lastError());
  const V = view(Float32Array, M._tsc_result_ptr(0), M._tsc_result_int(0) * 3);
  const N = view(Float32Array, M._tsc_result_ptr(2), M._tsc_result_int(0) * 3);
  const I = view(Uint32Array, M._tsc_result_ptr(1), M._tsc_result_int(1) * 3);
  const out = [], transfer = [];
  for (let i = 0; i < M._tsc_result_int(10); i++) {
    const info = (w) => M._tsc_label_info(i, w);
    const v0 = info(1), nv = info(2), i0 = info(3), ni = info(4);
    const vertices = V.slice(v0 * 3, (v0 + nv) * 3), normals = N.slice(v0 * 3, (v0 + nv) * 3);
    const indices = I.slice(i0, i0 + ni);
    for (let k = 0; k < indices.length; k++) indices[k] -= v0;
    out.push({ id: info(0), vertices, normals, indices, voxels: info(5), ml: info(6), meshMl: info(7) });
    transfer.push(vertices.buffer, normals.buffer, indices.buffer);
  }
  M._tsc_free_results();
  post({ type: 'meshes', items: out, seq, seconds: M._tsc_result_double(2) }, transfer);
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') await init();
    else if (m.type === 'loadDicom') loadDicom(m.files);
    else if (m.type === 'loadVolume') loadVolume(m);
    else if (m.type === 'buildSeries') buildSeries(m);
    else if (m.type === 'clearDicom') { M._tsc_clear_dicom(); post({ type: 'cleared' }); }
    else if (m.type === 'run') await run(m);
    else if (m.type === 'mesh') mesh(m);
    else if (m.type === 'bundleClear') { M._tsc_bundle_clear(); post({ type: 'bundleCleared' }); }
    else if (m.type === 'bundle') bundle(m);
  } catch (err) {
    post({ type: 'error', message: String(err && err.message || err), request: m.type });
  }
};
