#!/usr/bin/env node
/**
 * generate-bin.js
 *
 * Creates an MBJS-format .bin checkpoint compatible with mambacode.js / MambaKit.
 *
 * This writes a properly initialised model with the same weight shapes that
 * mambacode.js expects.  The weights are NOT pretrained — use pretrain.html
 * to run language-model training in the browser and produce meaningful output.
 *
 * Usage:
 *   node tools/generate-bin.js                        # nano model → model.bin
 *   node tools/generate-bin.js --size small           # small preset
 *   node tools/generate-bin.js --size nano --out my.bin
 *
 * Output can be loaded in any example via Advanced → Checkpoint URL (serve it)
 * or via session.load({ storage: 'indexedDB' }) after saving it there.
 */

import { writeFileSync } from 'fs';

// ── CLI args ──────────────────────────────────────────────────────────────────

const args  = process.argv.slice(2);
const flag  = (name) => { const i = args.indexOf(name); return i !== -1 ? args[i + 1] : null; };

const SIZE_PRESETS = {
  nano  : { dModel: 128,  numLayers: 4  },
  small : { dModel: 256,  numLayers: 6  },
  medium: { dModel: 512,  numLayers: 8  },
  large : { dModel: 768,  numLayers: 12 },
};

const sizeName = flag('--size') ?? 'nano';
if (!SIZE_PRESETS[sizeName]) {
  console.error(`Unknown size "${sizeName}". Choose: ${Object.keys(SIZE_PRESETS).join(', ')}`);
  process.exit(1);
}

const { dModel, numLayers } = SIZE_PRESETS[sizeName];
const outPath = flag('--out') ?? 'model.bin';

// Qwen2.5-Coder vocab size (matches what MambaKit loads by default)
const VOCAB_SIZE = 151936;

// ── Architecture constants ────────────────────────────────────────────────────

const D_STATE  = 16;                         // dState (N)
const D_CONV   = 4;                          // dConv  (K)
const EXPAND   = 2;                          // expand
const D_INNER  = EXPAND * dModel;            // dInner (D)
const DT_RANK  = Math.ceil(dModel / 16);     // dtRank (R)

// ── Random number generation (Box-Muller, seeded) ─────────────────────────────

let seed = 0x12345678;

function rand() {
  // LCG — good enough for weight init
  seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
  return (seed >>> 0) / 0xffffffff;
}

function randn(std = 0.02) {
  // Box-Muller
  const u1 = Math.max(rand(), 1e-10);
  const u2  = rand();
  return std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function gaussianFill(buf, offset, n, std) {
  for (let i = 0; i < n; i++) buf[offset + i] = randn(std);
}

function zeroFill(buf, offset, n) {
  buf.fill(0, offset, offset + n);
}

function oneFill(buf, offset, n) {
  buf.fill(1, offset, offset + n);
}

// A_log[d, n] = log(n + 1) — matches mambacode.js _initWeights()
function alogFill(buf, offset) {
  for (let d = 0; d < D_INNER; d++) {
    for (let n = 0; n < D_STATE; n++) {
      buf[offset + d * D_STATE + n] = Math.log(n + 1);
    }
  }
}

// ── Collect parameter shapes ──────────────────────────────────────────────────

// Returns [ { numel, fill }, ... ] in the exact order exportWeights() uses
function buildParamList() {
  const params = [];

  const add = (numel, fill) => params.push({ numel, fill });

  // Embedding first
  add(VOCAB_SIZE * dModel,  (b, o) => gaussianFill(b, o, VOCAB_SIZE * dModel, 0.02));  // embedding

  // Per-block (13 tensors each, exact order from mamba_block.js parameters())
  for (let layer = 0; layer < numLayers; layer++) {
    add(2 * D_INNER * dModel,             (b, o) => gaussianFill(b, o, 2 * D_INNER * dModel, 0.02));  // wInProj
    add(2 * D_INNER,                      (b, o) => zeroFill(b, o, 2 * D_INNER));                      // bInProj
    add(D_INNER * D_CONV,                 (b, o) => gaussianFill(b, o, D_INNER * D_CONV, 0.01));        // wConv
    add(D_INNER,                          (b, o) => zeroFill(b, o, D_INNER));                           // bConv
    add((DT_RANK + 2 * D_STATE) * D_INNER,(b, o) => gaussianFill(b, o, (DT_RANK + 2 * D_STATE) * D_INNER, 0.01)); // wXProj
    add(DT_RANK + 2 * D_STATE,            (b, o) => zeroFill(b, o, DT_RANK + 2 * D_STATE));             // bXProj
    add(D_INNER * DT_RANK,                (b, o) => gaussianFill(b, o, D_INNER * DT_RANK, 0.02));       // wDtProj
    add(D_INNER,                          (b, o) => zeroFill(b, o, D_INNER));                           // bDtProj
    add(D_INNER * D_STATE,                (b, o) => alogFill(b, o));                                    // A_log
    add(D_INNER,                          (b, o) => oneFill(b, o, D_INNER));                            // D_vec
    add(dModel * D_INNER,                 (b, o) => gaussianFill(b, o, dModel * D_INNER, 0.02));         // wOutProj
    add(dModel,                           (b, o) => zeroFill(b, o, dModel));                            // bOutProj
    add(dModel,                           (b, o) => oneFill(b, o, dModel));                             // normWeight
  }

  // final_norm comes after all blocks
  add(dModel, (b, o) => oneFill(b, o, dModel));  // final_norm

  return params;
}

// ── Write MBJS binary ─────────────────────────────────────────────────────────

function writeMbjs(params, path) {
  const totalFloats = params.reduce((s, p) => s + p.numel, 0);
  const nParams     = params.length;

  // Layout: 4 (magic) + 4 (version) + 4 (nParams) + 4*nParams (numels) + 4*totalFloats (data)
  const headerBytes = 12 + nParams * 4;
  const dataBytes   = totalFloats * 4;
  const buf         = Buffer.allocUnsafe(headerBytes + dataBytes);

  let pos = 0;

  // Header
  buf.writeUInt32LE(0x4D424A53, pos); pos += 4;  // magic 'MBJS'
  buf.writeUInt32LE(1,          pos); pos += 4;  // version
  buf.writeUInt32LE(nParams,    pos); pos += 4;  // param count

  for (const p of params) {
    buf.writeUInt32LE(p.numel, pos); pos += 4;
  }

  // Float data — write into a Float32Array view so fill functions can work with indices
  const f32 = new Float32Array(totalFloats);
  let f32Offset = 0;
  for (const p of params) {
    p.fill(f32, f32Offset);
    f32Offset += p.numel;
  }

  // Copy Float32Array bytes into buf
  const f32bytes = Buffer.from(f32.buffer);
  f32bytes.copy(buf, headerBytes);

  writeFileSync(path, buf);
  return { nParams, totalFloats };
}

// ── Main ──────────────────────────────────────────────────────────────────────

console.log(`\nMambaKit generate-bin`);
console.log(`  Size:       ${sizeName} (dModel=${dModel}, numLayers=${numLayers})`);
console.log(`  Vocab:      ${VOCAB_SIZE.toLocaleString()} tokens (Qwen2.5-Coder)`);
console.log(`  D_INNER:    ${D_INNER}  DT_RANK: ${DT_RANK}  D_STATE: ${D_STATE}`);
console.log(`  Output:     ${outPath}`);
console.log('');

const params = buildParamList();
const totalFloats = params.reduce((s, p) => s + p.numel, 0);
const sizeMb = (totalFloats * 4 / 1024 / 1024).toFixed(1);
console.log(`  Parameters: ${params.length} tensors, ${totalFloats.toLocaleString()} floats (${sizeMb} MB)`);
console.log('  Writing…');

const { nParams } = writeMbjs(params, outPath);

console.log(`  Done → ${outPath}\n`);
console.log(`Next steps:`);
console.log(`  1. Serve it:  npm run serve`);
console.log(`  2. Open pretrain.html and train it on a text corpus`);
console.log(`  3. Or load it directly via Advanced → Checkpoint URL\n`);
