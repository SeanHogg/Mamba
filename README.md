# MambaCode.js

> WebGPU-accelerated Mamba State Space Model library — written in **TypeScript**, compiled for use in any JavaScript application.

[![npm](https://img.shields.io/npm/v/mambacode.js)](https://www.npmjs.com/package/mambacode.js)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

MambaCode.js is a **TypeScript-first** library that brings the [Mamba SSM](https://arxiv.org/abs/2312.00752) architecture to the browser via WebGPU. It targets the Qwen3.5-Coder-0.8B model shape and supports full **on-device training** (backpropagation), allowing models to adapt to a user's private codebase locally — without any data leaving the browser.

> 📖 **New to MambaCode.js?** Start with the [Getting Started Guide](./docs/getting-started.md).

---

## Key Features

| Feature | Detail |
|---|---|
| **TypeScript-first** | Full type declarations shipped with the package |
| **Plain JS compatible** | Import the compiled `dist/` in any JavaScript project — no TypeScript toolchain required |
| **Architecture** | Selective State Space Model (S6) — linear O(N) context scaling |
| **Hardware target** | WebGPU (WGSL) — Chrome 113+, Edge 113+, Firefox Nightly |
| **Memory ceiling** | ≤ 3 GB VRAM (Chrome/Edge/Firefox stable) |
| **No heavy frameworks** | Zero TensorFlow.js / Transformers.js dependencies |
| **On-device training** | Tape-based autograd + AdamW GPU optimizer |
| **Quantization** | FP16 weights, Int8 activations |
| **Tokenizer** | Browser-side BPE (Qwen3.5-Coder compatible) |
| **WSLA mode** | Fine-tune only B & C matrices for rapid local adaptation |

---

## Installation

```bash
npm install mambacode.js
```

Build the library from source:

```bash
npm run build   # compiles TypeScript → dist/
```

---

## Documentation

| Guide | Description |
|---|---|
| **[Getting Started](docs/getting-started.md)** | Beginner-friendly introduction — what LLMs are, how Qwen fits in, step-by-step setup, and what to do next (including [builderforce.ai](https://builderforce.ai)) |

---

## Quick Start

### TypeScript

```ts
import {
  MambaModel,
  MambaTrainer,
  BPETokenizer,
  initWebGPU,
  type MambaModelConfig,
  type TrainOptions,
} from 'mambacode.js';

// 1. Initialise WebGPU
const { device } = await initWebGPU();

// 2. Load tokenizer
const tokenizer = new BPETokenizer();
await tokenizer.load('/vocab.json', '/merges.txt');

// 3. Create model
const config: MambaModelConfig = {
  vocabSize : tokenizer.vocabSize,   // e.g. 151936 for Qwen3.5-Coder
  dModel    : 512,
  numLayers : 8,
  dState    : 16,
  dConv     : 4,
  expand    : 2,
};
const model = new MambaModel(device, config);

// 4. Train on local code
const trainer = new MambaTrainer(model, tokenizer);
const opts: TrainOptions = {
  learningRate : 1e-4,
  epochs       : 5,
  onEpochEnd   : (epoch, loss) => console.log(`Epoch ${epoch}: loss=${loss.toFixed(4)}`),
};
const losses = await trainer.train(myCodeString, opts);

// 5. Generate code
const promptIds = tokenizer.encode('function fibonacci(');
const outputIds = await model.generate(promptIds, 200, { temperature: 0.8 });
console.log(tokenizer.decode(outputIds));
```

### JavaScript (ESM)

The compiled output in `dist/` is plain JavaScript with no TypeScript runtime dependency:

```js
import {
  MambaModel,
  MambaTrainer,
  BPETokenizer,
  initWebGPU,
} from 'mambacode.js';

// 1. Initialise WebGPU
const { device } = await initWebGPU();

// 2. Load tokenizer
const tokenizer = new BPETokenizer();
await tokenizer.load('/vocab.json', '/merges.txt');

// 3. Create model
const model = new MambaModel(device, {
  vocabSize : tokenizer.vocabSize,
  dModel    : 512,
  numLayers : 8,
});

// 4. Train on local code
const trainer = new MambaTrainer(model, tokenizer);
const losses = await trainer.train(myCodeString, {
  learningRate : 1e-4,
  epochs       : 5,
  onEpochEnd   : (epoch, loss) => console.log(`Epoch ${epoch}: loss=${loss.toFixed(4)}`),
});

// 5. Generate code
const promptIds = tokenizer.encode('function fibonacci(');
const outputIds = await model.generate(promptIds, 200, { temperature: 0.8 });
console.log(tokenizer.decode(outputIds));
```

### WSLA (Weight-Selective Local Adaptation)

Fine-tune only the B and C matrices for rapid private-codebase adaptation:

```ts
await trainer.train(apiUsageExamples, {
  learningRate : 1e-4,
  epochs       : 3,
  wsla         : true,   // only B and C matrices are updated
});
```

---

## Architecture

```
Token IDs
    │
    ▼
Embedding Lookup (GPU gather kernel)
    │
    ▼  ┌─────────────────────────────────────────┐
       │           Mamba Block × N               │
       │                                         │
       │  Input ──► RMSNorm                      │
       │               │                         │
       │      ┌────────┴────────┐                │
       │      ▼                 ▼                │
       │  in_proj(x)       in_proj(z)  [gate]    │
       │      │                                  │
       │  Conv1D (causal, K=4)                   │
       │      │                                  │
       │   SiLU activation                       │
       │      │                                  │
       │  x_proj → Δ, B, C  (selective)          │
       │      │                                  │
       │  Δ → dt_proj (full D_inner width)        │
       │      │                                  │
       │  ┌───▼──────────────────────────────┐   │
       │  │  Selective Scan S6               │   │
       │  │  (Kogge-Stone parallel prefix)   │   │
       │  │  h_t = Ā·h_{t-1} + B̄·x_t        │   │
       │  │  y_t = C·h_t + D·x_t            │   │
       │  └──────────────────────────────────┘   │
       │      │                                  │
       │  Gate: y * SiLU(z)                      │
       │      │                                  │
       │  out_proj → residual add ──► output     │
       └─────────────────────────────────────────┘
    │
    ▼
Final RMSNorm → LM Head (tied embedding) → Logits
```

---

## File Structure

```
src/                                ← TypeScript source (edit here)
├── index.ts                        ← public API entry point
├── kernels/
│   ├── selective_scan.ts           ← WGSL: S6 forward + backward (Kogge-Stone)
│   ├── conv1d.ts                   ← WGSL: 1D causal convolution
│   ├── linear_projection.ts        ← WGSL: tiled matrix multiplication
│   ├── weight_update.ts            ← WGSL: AdamW optimizer + gradient clipping
│   └── activations.ts              ← WGSL: SiLU, RMSNorm
├── model/
│   ├── mamba_block.ts              ← Mamba Mixer Block (forward pass)
│   └── mamba_model.ts              ← Full stacked model + generation
├── training/
│   ├── autograd.ts                 ← Tape-based AD engine + loss helpers
│   └── trainer.ts                  ← MambaTrainer class
├── tokenizer/
│   └── bpe.ts                      ← Browser-side BPE tokenizer
└── utils/
    ├── gpu_utils.ts                ← WebGPU device/buffer management
    └── quantization.ts             ← FP16 / Int8 quantization utilities

dist/                               ← Compiled output (JS + .d.ts, gitignored)
├── index.js                        ← ESM entry point for JS consumers
├── index.d.ts                      ← TypeScript declarations for TS consumers
└── ...                             ← mirrored sub-folders

tests/
├── kernels.test.ts                 ← WGSL kernel source smoke tests
├── autograd.test.ts                ← Autograd engine unit tests
├── bpe.test.ts                     ← BPE tokenizer unit tests
└── quantization.test.ts            ← Quantization round-trip tests

docs/
└── getting-started.md              ← Step-by-step guide (TS & JS)
```

---

## WGSL Kernels

### Parallel Selective Scan (`selective_scan.ts`)
Implements the S6 core using a **Kogge-Stone parallel prefix-sum** inside each workgroup tile. Each tile of 64 time steps is scanned in log₂(64) = 6 GPU barrier rounds, giving O(log N) wall-clock time on the GPU.

The associative operator for the recurrence `h_t = Ā·h_{t-1} + B̄·x_t` is:

```
(a₁, b₁) ∘ (a₂, b₂) = (a₁·a₂, a₁·b₂ + b₁)
```

Tiles are chained via a carry-in state, covering arbitrarily long sequences.

### 1D Causal Convolution (`conv1d.ts`)
Depthwise 1D causal conv (kernel size K=4) with zero left-padding. Enforces causality by only reading positions `t-k` for `k ≥ 0`, contributing 0 for `t < k`.

### Linear Projection (`linear_projection.ts`)
Tiled 16×16 GEMM in WGSL using workgroup shared memory. Handles arbitrary (M, K) × (N, K) → (M, N) shapes with boundary guards.

### AdamW Optimizer (`weight_update.ts`)
Fused single-kernel AdamW update with decoupled weight decay. Includes a two-pass gradient norm clipping kernel (reduce → scale).

---

## Testing

```bash
npm test        # run 58 unit tests (no GPU required)
npm run build   # compile TypeScript → dist/
npm run lint    # ESLint on src/ and tests/
```

Unit tests cover quantization, BPE tokenization, autograd, and WGSL kernel source validation. GPU execution tests require a real browser with WebGPU support.

---

## Browser Compatibility

| Browser | Version | Status |
|---|---|---|
| Chrome | 113+ | ✅ Supported |
| Edge | 113+ | ✅ Supported |
| Firefox | Nightly | ✅ Supported (flag: `dom.webgpu.enabled`) |
| Safari | 18+ | ⚠️ Partial (WebGPU in preview) |
| Node.js | — | ❌ Not supported (no `navigator.gpu`) |

---

## License

MIT
