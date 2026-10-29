'use strict';
// WebGPU implementation of the nnU-Net PlainConvUNet forward pass (fp32).
// Mirrors src/tsc/unet.cpp: Conv3d -> InstanceNorm3d(eps 1e-5, affine) -> LeakyReLU(0.01),
// strided-conv downsampling, ConvTranspose3d(k=s=2) upsampling, 1x1x1 segmentation head.
// Only the network runs on the GPU; pre/post-processing stays in the WASM C++ code.

// ---------------------------------------------------------------- .tsw parser
function parseTsw(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const magic = new TextDecoder().decode(u8.subarray(0, 8));
  if (magic !== 'TSCPPW01') throw new Error('not a .tsw file');
  const hlen = new DataView(u8.buffer, u8.byteOffset + 8, 4).getUint32(0, true);
  const header = new TextDecoder().decode(u8.subarray(12, 12 + hlen));
  const base = Math.ceil((12 + hlen) / 64) * 64;
  const cfg = { features: [], strides: [], nConvEnc: [], nConvDec: [], patch: [], labels: [] };
  const tensors = {};
  for (const line of header.split('\n')) {
    const t = line.trim().split(/\s+/);
    const k = t[0];
    if (k === 'patch') cfg.patch = t.slice(1).map(Number);
    else if (k === 'n_stages') cfg.nStages = +t[1];
    else if (k === 'features') cfg.features = t.slice(1).map(Number);
    else if (k === 'strides') { const v = t.slice(1).map(Number); for (let i = 0; i < v.length; i += 3) cfg.strides.push(v.slice(i, i + 3)); }
    else if (k === 'n_conv_enc') cfg.nConvEnc = t.slice(1).map(Number);
    else if (k === 'n_conv_dec') cfg.nConvDec = t.slice(1).map(Number);
    else if (k === 'num_classes') cfg.numClasses = +t[1];
    else if (k === 'name') cfg.name = t[1];
    else if (k === 'tensor') {
      const [, name, dt, ndS] = t; const nd = +ndS;
      const shape = t.slice(4, 4 + nd).map(Number);
      const off = +t[4 + nd], nbytes = +t[5 + nd];
      const start = u8.byteOffset + base + off;
      let data;
      if (dt === 'f32') {
        data = new Float32Array(nbytes / 4);
        new Uint8Array(data.buffer).set(u8.subarray(base + off, base + off + nbytes));
      } else {
        const h = new Uint16Array(nbytes / 2);
        new Uint8Array(h.buffer).set(u8.subarray(base + off, base + off + nbytes));
        data = new Float32Array(h.length);
        for (let i = 0; i < h.length; i++) data[i] = halfToFloat(h[i]);
      }
      void start;
      tensors[name] = { shape, data };
    }
  }
  return { cfg, tensors };
}

function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// ---------------------------------------------------------------- WGSL
// Implicit-GEMM convolution. out[M=Cout][N=positions] = W[M][K] * im2col[K][N] (+bias).
// Tile BM x BN per workgroup (256 threads), TM x TN outputs per thread, BK = 16.
// The im2col address pieces for every k (channel, dz, dy, dx) come from a
// precomputed table (kinfo) instead of integer divisions in the inner loop.
const BK = 16;
// Cout <= 32 (the full-resolution layers). Needs 18 KB workgroup memory; devices that only
// offer the 16 KB default get a 32x128 tile instead (see GpuUNet constructor).
let TILE_SMALL_M = { BM: 32, BN: 256, TM: 4, TN: 8 };
const TILE_LARGE_M = { BM: 64, BN: 128, TM: 4, TN: 8 };
const tileFor = (M) => (M <= 32 ? TILE_SMALL_M : TILE_LARGE_M);

function gemmShader(mode, { BM, BN, TM, TN }) {  // mode: 'conv' | 'tconv'
  const conv = mode === 'conv';
  const TX = BN / TN, TY = BM / TM;
  if (TX * TY !== 256) throw new Error('bad tile config');
  const AL = (BM * BK) / 256, BL = (BN * BK) / 256, KSTEP = 256 / BN;
  return /* wgsl */ `
struct Params { M: u32, N: u32, K: u32, C0: u32, D: u32, H: u32, W: u32, OH: u32,
                OW: u32, sz: u32, sy: u32, sx: u32, pz: u32, py: u32, px: u32, ON: u32,
                u0: u32, u1: u32, u2: u32, u3: u32 };
@group(0) @binding(0) var<storage, read> in0: array<f32>;
@group(0) @binding(1) var<storage, read> in1: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;
${conv ? '@group(0) @binding(6) var<storage, read> kinfo: array<vec4<i32>>;' : ''}
var<workgroup> As: array<array<f32, ${BM}>, ${BK}>;
var<workgroup> Bs: array<array<f32, ${BN}>, ${BK}>;

@compute @workgroup_size(${TX}, ${TY})
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u,
        @builtin(local_invocation_index) tid: u32) {
  let bm = wid.y * ${BM}u;
  let bn = wid.x * ${BN}u;
  let o = wid.z;
  let nn = tid % ${BN}u;
  let kb = tid / ${BN}u;
  let n = bn + nn;
  let nvalid = n < p.N;
  ${conv ? `
  let HW = i32(p.H * p.W);
  let DHW = p.D * p.H * p.W;
  let ow = n % p.OW;
  let tq = n / p.OW;
  let oh = tq % p.OH;
  let od = tq / p.OH;
  let iz0 = i32(od * p.sz) - i32(p.pz);
  let iy0 = i32(oh * p.sy) - i32(p.py);
  let ix0 = i32(ow * p.sx) - i32(p.px);
  let sp0 = (iz0 * i32(p.H) + iy0) * i32(p.W) + ix0;` : ''}
  var acc: array<f32, ${TM * TN}>;
  for (var i = 0u; i < ${TM * TN}u; i++) { acc[i] = 0.0; }
  for (var k0 = 0u; k0 < p.K; k0 += ${BK}u) {
    for (var r = 0u; r < ${AL}u; r++) {
      let idx = tid + 256u * r;
      ${conv ? `
      let mm = idx / ${BK}u; let kk = idx % ${BK}u;
      let m = bm + mm; let k = k0 + kk;
      var v = 0.0;
      if (m < p.M && k < p.K) { v = wt[m * p.K + k]; }` : `
      let mm = idx % ${BM}u; let kk = idx / ${BM}u;
      let m = bm + mm; let k = k0 + kk;
      var v = 0.0;
      if (m < p.M && k < p.K) { v = wt[(k * p.M + m) * (p.sz * p.sy * p.sx) + o]; }`}
      As[kk][mm] = v;
    }
    for (var r = 0u; r < ${BL}u; r++) {
      let kk = kb + ${KSTEP}u * r;
      let k = k0 + kk;
      var v = 0.0;
      if (nvalid && k < p.K) {
        ${conv ? `
        let ki = kinfo[k];
        let z = iz0 + ki.y;
        let y = iy0 + ki.z;
        let x = ix0 + ki.w;
        if (z >= 0 && y >= 0 && x >= 0 && u32(z) < p.D && u32(y) < p.H && u32(x) < p.W) {
          let sp = u32(sp0 + (ki.y * i32(p.H) + ki.z) * i32(p.W) + ki.w);
          let ci = u32(ki.x);
          if (ci < p.C0) { v = in0[ci * DHW + sp]; } else { v = in1[(ci - p.C0) * DHW + sp]; }
        }` : `
        v = in0[k * p.N + n];
        if (p.u0 == 1u) { v += in1[0]; }  // keeps binding 1 in the auto layout (never taken)`}
      }
      Bs[kk][nn] = v;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < ${BK}u; kk++) {
      var a: array<f32, ${TM}>;
      var b: array<f32, ${TN}>;
      for (var i = 0u; i < ${TM}u; i++) { a[i] = As[kk][lid.y * ${TM}u + i]; }
      for (var j = 0u; j < ${TN}u; j++) { b[j] = Bs[kk][lid.x + j * ${TX}u]; }
      for (var i = 0u; i < ${TM}u; i++) {
        for (var j = 0u; j < ${TN}u; j++) { acc[i * ${TN}u + j] = fma(a[i], b[j], acc[i * ${TN}u + j]); }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < ${TM}u; i++) {
    let m = bm + lid.y * ${TM}u + i;
    if (m >= p.M) { continue; }
    let bv = bias[m];
    for (var j = 0u; j < ${TN}u; j++) {
      let nq = bn + lid.x + j * ${TX}u;
      if (nq >= p.N) { continue; }
      ${conv ? `
      outp[m * p.N + nq] = acc[i * ${TN}u + j] + bv;` : `
      let w = nq % p.W; let tq = nq / p.W; let h = tq % p.H; let d = tq / p.H;
      let a = o / (p.sy * p.sx); let bb = (o / p.sx) % p.sy; let c = o % p.sx;
      let oidx = ((d * p.sz + a) * (p.H * p.sy) + h * p.sy + bb) * (p.W * p.sx) + w * p.sx + c;
      outp[m * p.ON + oidx] = acc[i * ${TN}u + j] + bv;`}
    }
  }
}`;
}

// per-channel reduction: mode 0 sum(x), mode 1 sum((x-mean)^2); partials[c][chunk]
const CHUNK = 4096;
const reduceShader = /* wgsl */ `
struct Params { N: u32, nchunks: u32, mode: u32, C: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3u) {
  let c = wid.y; let chunk = wid.x;
  let base = c * p.N;
  var s = 0.0;
  var mean = 0.0;
  if (p.mode == 1u) { mean = stats[c * 2u]; }
  for (var j = 0u; j < ${CHUNK / 256}u; j++) {
    let i = chunk * ${CHUNK}u + j * 256u + tid;
    if (i < p.N) {
      let v = x[base + i];
      if (p.mode == 0u) { s += v; } else { let d = v - mean; s += d * d; }
    }
  }
  sh[tid] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (tid < st) { sh[tid] += sh[tid + st]; }
    workgroupBarrier();
  }
  if (tid == 0u) { partials[c * p.nchunks + chunk] = sh[0]; }
}`;

// finish reduction: stats[c*2] = mean (mode 0) or stats[c*2+1] = var (mode 1)
const finalizeShader = /* wgsl */ `
struct Params { N: u32, nchunks: u32, mode: u32, C: u32 };
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3u) {
  let c = wid.x;
  var s = 0.0;
  for (var i = tid; i < p.nchunks; i += 256u) { s += partials[c * p.nchunks + i]; }
  sh[tid] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (tid < st) { sh[tid] += sh[tid + st]; }
    workgroupBarrier();
  }
  if (tid == 0u) { stats[c * 2u + p.mode] = sh[0] / f32(p.N); }
}`;

// y = x * alpha + beta ; leaky relu ; alpha = g / sqrt(var + eps), beta = b - mean * alpha
const applyShader = /* wgsl */ `
struct Params { N: u32, total: u32, C: u32, u0: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> stats: array<f32>;
@group(0) @binding(2) var<storage, read> gam: array<f32>;
@group(0) @binding(3) var<storage, read> bet: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let stride = nwg.x * 256u;
  for (var i = gid.x; i < p.total; i += stride) {
    let c = i / p.N;
    let mean = stats[c * 2u];
    let invstd = 1.0 / sqrt(stats[c * 2u + 1u] + 1e-5);
    let alpha = invstd * gam[c];
    let beta = bet[c] - mean * alpha;
    let y = x[i] * alpha + beta;
    x[i] = select(y, y * 0.01, y < 0.0);
  }
}`;

// ---- sliding-window accumulation on the GPU, bit-identical to SlidingWindow::accumulate (C++):
//   logits_h = f16(f32(logits_h) + f32(y * g))   (explicit RNE conversions, no builtins)
const F16_WGSL = /* wgsl */ `
fn f2h(x: f32) -> u32 {
  let b = bitcast<u32>(x);
  let s = (b >> 16u) & 0x8000u;
  let a = b & 0x7fffffffu;
  if (a > 0x7f800000u) { return s | 0x7e00u; }
  if (a >= 0x477ff000u) { return s | 0x7c00u; }
  if (a < 0x38800000u) {
    let e = a >> 23u;
    if (e < 102u) { return s; }
    let mant = (a & 0x7fffffu) | 0x800000u;
    let shift = 126u - e;
    var h = mant >> shift;
    let rem = mant & ((1u << shift) - 1u);
    let half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (h & 1u) == 1u)) { h += 1u; }
    return s | h;
  }
  var h = (a >> 13u) - (112u << 10u);
  let rem = a & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) == 1u)) { h += 1u; }
  return s | h;
}
fn h2f(h: u32) -> f32 {
  let s = (h & 0x8000u) << 16u;
  let e = (h >> 10u) & 0x1fu;
  let m = h & 0x3ffu;
  if (e == 0u) {
    let v = f32(m) * 5.9604644775390625e-8;
    return select(v, -v, s != 0u);
  }
  if (e == 31u) { return bitcast<f32>(s | 0x7f800000u | (m << 13u)); }
  return bitcast<f32>(s | ((e + 112u) << 23u) | (m << 13u));
}`;

// y *= g  (separate pass so the product can never be fused into an FMA with the add)
const weightShader = /* wgsl */ `
struct Params { total: u32, PP: u32, u0: u32, u1: u32 };
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let stride = nwg.x * 256u;
  for (var i = gid.x; i < p.total; i += stride) { y[i] = y[i] * g[i % p.PP]; }
}`;

const accumShader = /* wgsl */ `
struct Params { total: u32, P1: u32, P2: u32, PP: u32, o0: u32, o1: u32, o2: u32, ps1: u32, ps2: u32, NP: u32, u0: u32, u1: u32 };
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> acc: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
${F16_WGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let stride = nwg.x * 256u;
  for (var i = gid.x; i < p.total; i += stride) {
    let c = i / p.PP;
    let q = i % p.PP;
    let pk = q % p.P2;
    let pj = (q / p.P2) % p.P1;
    let pi = q / (p.P1 * p.P2);
    let v = c * p.NP + ((p.o0 + pi) * p.ps1 + p.o1 + pj) * p.ps2 + p.o2 + pk;
    let sum = acc[v] + y[i];
    acc[v] = h2f(f2h(sum));
  }
}`;

const packShader = /* wgsl */ `
struct Params { total: u32, words: u32, u0: u32, u1: u32 };
@group(0) @binding(0) var<storage, read> acc: array<f32>;
@group(0) @binding(1) var<storage, read_write> outw: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;
${F16_WGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let stride = nwg.x * 256u;
  for (var w = gid.x; w < p.words; w += stride) {
    let lo = f2h(acc[2u * w]);
    var hi = 0u;
    if (2u * w + 1u < p.total) { hi = f2h(acc[2u * w + 1u]); }
    outw[w] = lo | (hi << 16u);
  }
}`;

// Completion marker: one invocation adds 1 to a counter. It is dispatched last in every command buffer, so the counter
// equals the number of command buffers whose work really ran. A GPU reset that drops part of the work (seen on Intel UHD
// with Linux i915, 2026-10-03: no WebGPU error, no device loss, wrong or empty label map) leaves the counter short.
const tickShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> c: array<u32>;
@compute @workgroup_size(1)
fn main() { c[0] = c[0] + 1u; }`;

// ---------------------------------------------------------------- GpuUNet
class GpuUNet {
  static chunkOps = 0;   // see forward(): ops per command buffer (0 = all, the configuration measured in the paper)
  static chunkWait = true;
  static simulateIncomplete = null;  // test hook (worker ?gpuincomplete=once|all): the completion counter reports one missing
  static takeSimulatedMiss() {
    if (!GpuUNet.simulateIncomplete) return 0;
    if (GpuUNet.simulateIncomplete === 'once') GpuUNet.simulateIncomplete = null;
    return 1;
  }  // with chunkOps > 0: wait for each command buffer before submitting the next (false: only split)
  static async createDevice() {
    if (!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no WebGPU adapter');
    const info = adapter.info || {};
    GpuUNet.adapterInfo = { vendor: info.vendor || '', architecture: info.architecture || '', description: info.description || '' };
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      },
    });
    device.lost.then((i) => console.error('WebGPU device lost', i.message));
    return device;
  }

  constructor(device, tswBytes) {
    this.device = device;
    if (device.limits.maxComputeWorkgroupStorageSize < 18432) TILE_SMALL_M = { BM: 32, BN: 128, TM: 2, TN: 8 };
    const { cfg, tensors } = parseTsw(tswBytes);
    this.cfg = cfg;
    const dev = device;
    const upload = (arr) => {
      const b = dev.createBuffer({ size: Math.max(16, arr.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(b, 0, arr);
      return b;
    };
    this.w = {};
    for (const [k, t] of Object.entries(tensors)) this.w[k] = { buf: upload(t.data), shape: t.shape };
    this.weightBytes = Object.values(this.w).reduce((s, t) => s + t.buf.size, 0);
    this.pipes = {};
    const mk = (name, code) => {
      this.pipes[name] = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code }), entryPoint: 'main' } });
    };
    for (const [tn, t] of [['s', TILE_SMALL_M], ['l', TILE_LARGE_M]]) {
      mk('conv_' + tn, gemmShader('conv', t));
      mk('tconv_' + tn, gemmShader('tconv', t));
    }
    mk('reduce', reduceShader);
    mk('finalize', finalizeShader);
    mk('apply', applyShader);
    mk('weight', weightShader);
    mk('accum', accumShader);
    mk('pack', packShader);
    mk('tick', tickShader);
    this.dummy = dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    this.plan = null;
  }

  // Builds a static execution plan (buffers, bind groups) for the model's patch size.
  // A failed plan (validation / out of memory / too large for the adapter's limits) is discarded, never kept.
  // The GPU work queued before is waited for first (buffers destroyed just before are then really freed), and a
  // failed allocation is retried: Firefox 156 (Windows) intermittently failed to create the large activation buffers
  // right after another model's buffers were destroyed (2026-09-29).
  async buildPlan() {
    const dev = this.device;
    let err = null;
    for (let attempt = 0; attempt < 4; attempt++) {
      await dev.queue.onSubmittedWorkDone();
      if (attempt) await new Promise((r) => setTimeout(r, 500 * attempt));
      dev.pushErrorScope('out-of-memory');
      dev.pushErrorScope('validation');
      let jsError = null;
      try { this._buildPlan(); } catch (e) { jsError = e; }
      const ev = await dev.popErrorScope(), eo = await dev.popErrorScope();
      err = jsError || eo || ev;
      if (!err) { if (attempt) console.warn(`WebGPU plan built on attempt ${attempt + 1}`); return; }
      for (const b of this._owned || []) b.destroy();
      this._owned = null;
      this.plan = null;
      if (jsError) break;  // limits / configuration: retrying does not help
    }
    throw new Error('WebGPU plan: ' + err.message);
  }

  _buildPlan() {
    const dev0 = this.device, cfg = this.cfg, S = cfg.nStages;
    const owned = (this._owned = []);
    const lim = dev0.limits;
    const dev = { createBuffer: (d) => {
                    if (d.size > lim.maxBufferSize || ((d.usage & GPUBufferUsage.STORAGE) && d.size > lim.maxStorageBufferBindingSize))
                      throw new Error(`buffer of ${(d.size / 2 ** 20).toFixed(0)} MB exceeds this GPU's limits`);
                    const b = dev0.createBuffer(d); owned.push(b); return b; },
                  createBindGroup: (d) => dev0.createBindGroup(d), queue: dev0.queue };
    const [P0, P1, P2] = cfg.patch;
    const pool = new Map();  // size -> [buffers]
    const alloc = (floats) => {
      const size = floats * 4;
      const l = pool.get(size);
      if (l && l.length) return l.pop();
      return dev.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    };
    const free = (t) => { if (!pool.has(t.buf.size)) pool.set(t.buf.size, []); pool.get(t.buf.size).push(t.buf); };
    const uni = (arr) => {
      const b = dev.createBuffer({ size: arr.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(b, 0, arr);
      return b;
    };
    const ops = [];
    const tensor = (C, D, H, W) => ({ C, D, H, W, buf: alloc(C * D * H * W) });

    const conv = (ins, name, stride) => {  // stride: [sz, sy, sx]
      const Wt = this.w[name + '.w'];
      const [Cout, Cin, kd, kh, kw] = Wt.shape;
      const { D, H, W } = ins[0];
      const [pz, py, px] = [kd >> 1, kh >> 1, kw >> 1];
      const OD = Math.floor((D + 2 * pz - kd) / stride[0]) + 1, OH = Math.floor((H + 2 * py - kh) / stride[1]) + 1,
        OW = Math.floor((W + 2 * px - kw) / stride[2]) + 1;
      const out = tensor(Cout, OD, OH, OW);
      const kv = kd * kh * kw, N = OD * OH * OW, K = Cin * kv;
      const u = uni(new Uint32Array([Cout, N, K, ins[0].C, D, H, W, OH, OW, stride[0], stride[1], stride[2], pz, py, px, 0, 0, 0, 0, 0]));
      // im2col table: k -> (input channel, dz, dy, dx), k = ci * kv + (dz * kh + dy) * kw + dx
      const kinfo = new Int32Array(K * 4);
      for (let k = 0; k < K; k++) {
        const t = k % kv;
        kinfo.set([Math.floor(k / kv), Math.floor(t / (kh * kw)), Math.floor(t / kw) % kh, t % kw], k * 4);
      }
      const kbuf = dev.createBuffer({ size: kinfo.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(kbuf, 0, kinfo);
      const tile = tileFor(Cout), tn = tile === TILE_SMALL_M ? 's' : 'l';
      const bg = dev.createBindGroup({
        layout: this.pipes['conv_' + tn].getBindGroupLayout(0),
        entries: [ins[0].buf, ins[1] ? ins[1].buf : this.dummy, Wt.buf, this.w[name + '.b'].buf, out.buf, u, kbuf].map((b, i) => ({ binding: i, resource: { buffer: b } })),
      });
      ops.push({ pipe: 'conv_' + tn, bg, wg: [Math.ceil(N / tile.BN), Math.ceil(Cout / tile.BM), 1] });
      return out;
    };
    const norm = (x, name) => {
      const N = x.D * x.H * x.W, C = x.C, nchunks = Math.ceil(N / CHUNK);
      const partials = dev.createBuffer({ size: C * nchunks * 4, usage: GPUBufferUsage.STORAGE });
      const stats = dev.createBuffer({ size: C * 2 * 4, usage: GPUBufferUsage.STORAGE });
      for (const mode of [0, 1]) {
        const u = uni(new Uint32Array([N, nchunks, mode, C]));
        ops.push({ pipe: 'reduce', wg: [nchunks, C, 1], bg: dev.createBindGroup({ layout: this.pipes.reduce.getBindGroupLayout(0),
          entries: [x.buf, partials, stats, u].map((b, i) => ({ binding: i, resource: { buffer: b } })) }) });
        ops.push({ pipe: 'finalize', wg: [C, 1, 1], bg: dev.createBindGroup({ layout: this.pipes.finalize.getBindGroupLayout(0),
          entries: [partials, stats, u].map((b, i) => ({ binding: i, resource: { buffer: b } })) }) });
      }
      const u = uni(new Uint32Array([N, N * C, C, 0]));
      ops.push({ pipe: 'apply', wg: [Math.min(4096, Math.ceil(N * C / 256)), 1, 1], bg: dev.createBindGroup({ layout: this.pipes.apply.getBindGroupLayout(0),
        entries: [x.buf, stats, this.w[name + '.nw'].buf, this.w[name + '.nb'].buf, u].map((b, i) => ({ binding: i, resource: { buffer: b } })) }) });
    };
    const tconv = (x, name) => {
      const Wt = this.w[name + '.w'];
      const [Cin, Cout, sa, sb, sc] = Wt.shape;  // kernel == stride
      const out = tensor(Cout, x.D * sa, x.H * sb, x.W * sc);
      const N = x.D * x.H * x.W, KVt = sa * sb * sc;
      const u = uni(new Uint32Array([Cout, N, Cin, Cin, x.D, x.H, x.W, 0, 0, sa, sb, sc, 0, 0, 0, N * KVt, 0, 0, 0, 0]));
      const tile = tileFor(Cout), tn = tile === TILE_SMALL_M ? 's' : 'l';
      const bg = dev.createBindGroup({
        layout: this.pipes['tconv_' + tn].getBindGroupLayout(0),
        entries: [x.buf, this.dummy, Wt.buf, this.w[name + '.b'].buf, out.buf, u].map((b, i) => ({ binding: i, resource: { buffer: b } })),
      });
      ops.push({ pipe: 'tconv_' + tn, bg, wg: [Math.ceil(N / tile.BN), Math.ceil(Cout / tile.BM), KVt] });
      return out;
    };

    const input = tensor(1, P0, P1, P2);
    const skips = [];
    let cur = input;
    for (let s = 0; s < S; s++) {
      let x = conv([cur], `enc.${s}.0`, cfg.strides[s]);  // cur (input or previous skip) stays alive
      norm(x, `enc.${s}.0`);
      for (let c = 1; c < cfg.nConvEnc[s]; c++) {
        const y = conv([x], `enc.${s}.${c}`, [1, 1, 1]);
        norm(y, `enc.${s}.${c}`);
        free(x);
        x = y;
      }
      skips.push(x);
      cur = x;
    }
    let lres = skips.pop();
    for (let s = 0; s < S - 1; s++) {
      const up = tconv(lres, `up.${s}`);
      free(lres);
      const skip = skips.pop();
      let x = conv([up, skip], `dec.${s}.0`, [1, 1, 1]);
      norm(x, `dec.${s}.0`);
      free(up); free(skip);
      for (let c = 1; c < cfg.nConvDec[s]; c++) {
        const y = conv([x], `dec.${s}.${c}`, [1, 1, 1]);
        norm(y, `dec.${s}.${c}`);
        free(x);
        x = y;
      }
      lres = x;
    }
    const logits = conv([lres], 'seg', [1, 1, 1]);
    const staging = dev.createBuffer({ size: logits.buf.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    // completion counter of forward() (see tickShader) and its readback
    const tick = dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const tickStaging = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const tickBg = dev.createBindGroup({ layout: this.pipes.tick.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: tick } }] });
    this.plan = { ops, input, logits, staging, owned, tick, tickStaging, tickBg };
    this.planBytes = owned.reduce((s, b) => s + b.size, 0);  // reported in the worker log (GPU memory needed)
  }

  // x: Float32Array (P0*P1*P2) -> Float32Array (numClasses * P0*P1*P2), written into `out` if given
  async forward(x, out) {
    if (!this.plan) await this.buildPlan();
    const dev = this.device, { ops, input, logits, staging, tick, tickStaging, tickBg } = this.plan;
    dev.queue.writeBuffer(input.buf, 0, x);
    dev.queue.writeBuffer(tick, 0, new Uint32Array(4));
    let sent = 0;
    // GpuUNet.chunkOps = 0 (default, as measured in the paper): the whole network of one patch goes to the GPU as one
    // command buffer. chunkOps = N > 0: at most N ops per command buffer, waiting for each to finish before the next, so
    // that no single GPU job runs longer than the driver's preemption time limit on slow (integrated) GPUs. The same
    // split is applied in forwardAccumulate() (the normal path; this function is the fallback for large volumes). The
    // arithmetic and its order are unchanged; only the submission is split. chunkWait = false splits without waiting
    // (the queue executes command buffers in order, so splitting alone may already let the driver preempt between them).
    const chunk = GpuUNet.chunkOps > 0 ? GpuUNet.chunkOps : ops.length;
    for (let i = 0; i < ops.length; i += chunk) {
      const enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const op of ops.slice(i, i + chunk)) {
        pass.setPipeline(this.pipes[op.pipe]);
        pass.setBindGroup(0, op.bg);
        pass.dispatchWorkgroups(op.wg[0], op.wg[1], op.wg[2]);
      }
      pass.setPipeline(this.pipes.tick); pass.setBindGroup(0, tickBg); pass.dispatchWorkgroups(1);  // completion marker
      sent++;
      pass.end();
      if (i + chunk >= ops.length) {
        enc.copyBufferToBuffer(logits.buf, 0, staging, 0, logits.buf.size);
        enc.copyBufferToBuffer(tick, 0, tickStaging, 0, 16);
      }
      dev.queue.submit([enc.finish()]);
      if (GpuUNet.chunkWait && i + chunk < ops.length) await dev.queue.onSubmittedWorkDone();
    }
    await Promise.all([staging.mapAsync(GPUMapMode.READ), tickStaging.mapAsync(GPUMapMode.READ)]);
    const done = new Uint32Array(tickStaging.getMappedRange())[0] - GpuUNet.takeSimulatedMiss();
    tickStaging.unmap();
    if (done !== sent) { staging.unmap(); throw new Error(`GPU work incomplete (${done} of ${sent} command buffers ran)`); }
    const r = new Float32Array(staging.getMappedRange());
    if (out) out.set(r); else out = r.slice();
    staging.unmap();
    return out;
  }
}

// Frees everything this network holds on the GPU (plan and weights); the object cannot be used afterwards.
GpuUNet.prototype.destroy = function () {
  this.releasePlan();
  for (const t of Object.values(this.w)) t.buf.destroy();
  this.dummy.destroy();
  this.w = {};
};

// Frees the activation buffers of the execution plan (weights stay on the GPU).
GpuUNet.prototype.releasePlan = function () {
  this.endVolume();
  if (!this.plan) return;
  for (const b of this.plan.owned) b.destroy();
  this.plan = null;
  this._owned = null;
};

// Error scopes around one sliding-window run (beginVolume .. readLogits): GPU errors of the tile
// submissions are reported instead of silently giving empty logits.
GpuUNet.prototype.popVolumeScopes = async function () {
  if (!this._volScopes) return null;
  this._volScopes = false;
  const ev = await this.device.popErrorScope(), eo = await this.device.popErrorScope();
  // Out of memory first: a buffer whose allocation failed is invalid, so the validation scope then also holds an
  // "Invalid Buffer" error from its first use, which hid the cause. The worker's out-of-memory retry (low-memory mode)
  // recognises the out-of-memory message; with the validation message it retried in parts and then used the CPU
  // (T1200 4 GB, Chrome/Vulkan: total with roi liver, gallbladder, IVC, portal vein, accumulation >= 471 MiB, 2026-10-25).
  if (eo) return { message: 'out of memory: ' + eo.message };
  return ev;
};

// Sliding-window accumulation state kept on the GPU (one per model run).
GpuUNet.prototype.canAccumulate = function (ps) {
  const n = this.cfg.numClasses * ps[0] * ps[1] * ps[2] * 4;
  return n <= this.device.limits.maxBufferSize && n <= this.device.limits.maxStorageBufferBindingSize;
};

GpuUNet.prototype.beginVolume = async function (gauss, ps) {
  if (!this.plan) await this.buildPlan();
  await this.popVolumeScopes();  // (a previous run that did not finish)
  await this.device.queue.onSubmittedWorkDone();  // earlier destroys really freed before the volume buffers (see buildPlan)
  const dev = this.device, C = this.cfg.numClasses, [P0, P1, P2] = this.cfg.patch;
  dev.pushErrorScope('out-of-memory');
  dev.pushErrorScope('validation');
  this._volScopes = true;
  const NP = ps[0] * ps[1] * ps[2], PP = P0 * P1 * P2;
  this.endVolume();
  const acc = dev.createBuffer({ size: C * NP * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const g = dev.createBuffer({ size: PP * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(g, 0, gauss);
  const wu = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(wu, 0, new Uint32Array([C * PP, PP, 0, 0]));
  const logits = this.plan.logits.buf;
  const bg = (pipe, bufs) => dev.createBindGroup({ layout: this.pipes[pipe].getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  // completion counter (see tickShader): +1 per command buffer that ran; checked in readLogits()
  const tick = dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(tick, 0, new Uint32Array(4));
  this.vol = { acc, g, wu, C, NP, PP, ps, wbg: bg('weight', [logits, g, wu]), extra: [], bg, tick, tickBg: bg('tick', [tick]), sent: 0 };
};

GpuUNet.prototype.endVolume = function () {
  if (!this.vol) return;
  for (const b of [this.vol.acc, this.vol.g, this.vol.wu, this.vol.tick, ...this.vol.extra]) b.destroy();
  this.vol = null;
};

// Runs the network on one tile and accumulates it into the GPU volume (no readback). This is the normal path (the
// accumulation buffer fits the device limits); forward() is the fallback. Returns a promise only because of the
// optional chunked submission (GpuUNet.chunkOps, see forward()); with chunkOps = 0 everything is still queued in one
// command buffer without waiting, exactly as measured in the paper.
GpuUNet.prototype.forwardAccumulate = async function (x, origin) {
  const dev = this.device, v = this.vol, { ops, input, logits } = this.plan;
  const [, P1, P2] = this.cfg.patch;
  dev.queue.writeBuffer(input.buf, 0, x);
  // per-tile uniform (queued writes to one buffer would all land before the first submit executes)
  const au = dev.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(au, 0, new Uint32Array([v.C * v.PP, P1, P2, v.PP, origin[0], origin[1], origin[2], v.ps[1], v.ps[2], v.NP, 0, 0]));
  v.extra.push(au);
  const chunk = GpuUNet.chunkOps > 0 ? GpuUNet.chunkOps : ops.length;
  for (let i = 0; i < ops.length; i += chunk) {
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const op of ops.slice(i, i + chunk)) {
      pass.setPipeline(this.pipes[op.pipe]);
      pass.setBindGroup(0, op.bg);
      pass.dispatchWorkgroups(op.wg[0], op.wg[1], op.wg[2]);
    }
    if (i + chunk >= ops.length) {  // the Gaussian weighting and the accumulation go with the last chunk
      pass.setPipeline(this.pipes.weight); pass.setBindGroup(0, v.wbg); pass.dispatchWorkgroups(4096);
      pass.setPipeline(this.pipes.accum); pass.setBindGroup(0, v.bg('accum', [logits.buf, v.acc, au])); pass.dispatchWorkgroups(4096);
    }
    pass.setPipeline(this.pipes.tick); pass.setBindGroup(0, v.tickBg); pass.dispatchWorkgroups(1);  // completion marker
    v.sent++;
    pass.end();
    dev.queue.submit([enc.finish()]);
    if (GpuUNet.chunkWait && i + chunk < ops.length) await dev.queue.onSubmittedWorkDone();
  }
};

// Packs the accumulated logits to fp16 and returns them (Uint16Array, C x padded volume).
GpuUNet.prototype.readLogits = async function () {
  const dev = this.device, v = this.vol;
  const total = v.C * v.NP, words = Math.ceil(total / 2);
  const out = dev.createBuffer({ size: words * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const staging = dev.createBuffer({ size: words * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const u = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(u, 0, new Uint32Array([total, words, 0, 0]));
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(this.pipes.pack);
  pass.setBindGroup(0, v.bg('pack', [v.acc, out, u]));
  pass.dispatchWorkgroups(4096);
  pass.end();
  enc.copyBufferToBuffer(out, 0, staging, 0, words * 4);
  const tickStaging = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  enc.copyBufferToBuffer(v.tick, 0, tickStaging, 0, 16);
  dev.queue.submit([enc.finish()]);
  const sent = v.sent;
  let r, done;
  try {
    await Promise.all([staging.mapAsync(GPUMapMode.READ), tickStaging.mapAsync(GPUMapMode.READ)]);
    r = new Uint16Array(staging.getMappedRange(), 0, total).slice();
    done = new Uint32Array(tickStaging.getMappedRange())[0] - GpuUNet.takeSimulatedMiss();
    staging.unmap();
    tickStaging.unmap();
  } finally {
    for (const b of [out, staging, u, tickStaging]) b.destroy();
    this.endVolume();
  }
  const err = await this.popVolumeScopes();
  if (err) throw new Error('WebGPU: ' + err.message);
  if (done !== sent) throw new Error(`GPU work incomplete (${done} of ${sent} command buffers ran)`);
  return r;
};

if (typeof self !== 'undefined') self.GpuUNet = GpuUNet;
