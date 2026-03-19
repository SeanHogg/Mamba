# Mamba
MambaCode.js — WebGPU-accelerated Mamba SSM library for browser-based code model training and inference.

## Overview

MambaCode.js is a pure JavaScript/WGSL implementation of the **Mamba State Space Model (SSM)** architecture, optimised for on-device code model training and inference in the browser. It targets the Qwen3.5-Coder-0.8B logic and supports full **on-device training** (backpropagation) via WebGPU, allowing models to adapt to a user's private codebase locally — without any data leaving the browser.

### Key features

| Feature | Detail |
|---|---|
| **Architecture** | Selective State Space Model (S6) — linear O(N) context scaling |
| **Hardware target** | WebGPU (WGSL) — Chrome 113+, Edge 113+, Firefox Nightly |
| **Memory ceiling** | ≤ 3 GB VRAM (Chrome/Edge/Firefox stable) |
| **No heavy frameworks** | Zero TensorFlow.js / Transformers.js dependencies |
| **On-device training** | Tape-based autograd + AdamW GPU optimizer |
| **Quantization** | FP16 weights, Int8 activations |
| **Tokenizer** | Browser-side BPE (Qwen3.5-Coder compatible) |
| **WSLA mode** | Fine-tune only B & C matrices for rapid local adaptation |

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

## Documentation

| Guide | Description |
|---|---|
| **[Getting Started](docs/getting-started.md)** | Beginner-friendly introduction — what LLMs are, how Qwen fits in, step-by-step setup, and what to do next (including [builderforce.ai](https://builderforce.ai)) |

---

## Quick Start

```js
import { MambaModel, MambaTrainer, BPETokenizer, initWebGPU } from './src/index.js';

// 1. Initialise WebGPU
const { device } = await initWebGPU();

// 2. Load tokenizer
const tokenizer = new BPETokenizer();
await tokenizer.load('/vocab.json', '/merges.txt');

// 3. Create model
const model = new MambaModel(device, {
  vocabSize : tokenizer.vocabSize,   // e.g. 151936 for Qwen3.5-Coder
  dModel    : 512,
  numLayers : 8,
  dState    : 16,
  dConv     : 4,
  expand    : 2,
});

// 4. Train on local code
const trainer = new MambaTrainer(model, tokenizer);
const losses  = await trainer.train(myCodeString, {
  learningRate : 1e-4,
  epochs       : 5,
  device       : 'webgpu',
  onEpochEnd   : (epoch, loss) => console.log(`Epoch ${epoch}: loss=${loss.toFixed(4)}`),
});

// 5. Generate code
const promptIds = tokenizer.encode('function fibonacci(');
const outputIds = await model.generate(promptIds, 200, { temperature: 0.8 });
console.log(tokenizer.decode(outputIds));
```

### WSLA (Weight-Selective Local Adaptation)

Fine-tune only the B and C matrices for rapid private-codebase adaptation:

```js
await trainer.train(apiUsageExamples, {
  learningRate : 1e-4,
  epochs       : 3,
  wsla         : true,   // only B and C matrices are updated
});
```

---

## File Structure

```
src/
├── index.js                      ← public API entry point
├── kernels/
│   ├── selective_scan.js         ← WGSL: S6 forward + backward (Kogge-Stone)
│   ├── conv1d.js                 ← WGSL: 1D causal convolution
│   ├── linear_projection.js      ← WGSL: tiled matrix multiplication
│   ├── weight_update.js          ← WGSL: AdamW optimizer + gradient clipping
│   └── activations.js            ← WGSL: SiLU, RMSNorm
├── model/
│   ├── mamba_block.js            ← Mamba Mixer Block (forward pass)
│   └── mamba_model.js            ← Full stacked model + generation
├── training/
│   ├── autograd.js               ← Tape-based AD engine + loss helpers
│   └── trainer.js                ← MambaTrainer class
├── tokenizer/
│   └── bpe.js                    ← Browser-side BPE tokenizer
└── utils/
    ├── gpu_utils.js              ← WebGPU device/buffer management
    └── quantization.js           ← FP16 / Int8 quantization utilities
tests/
├── kernels.test.js               ← WGSL kernel source smoke tests
├── autograd.test.js              ← Autograd engine unit tests
├── bpe.test.js                   ← BPE tokenizer unit tests
└── quantization.test.js          ← Quantization round-trip tests
```

---

## WGSL Kernels

### Parallel Selective Scan (`selective_scan.js`)
Implements the S6 core using a **Kogge-Stone parallel prefix-sum** inside each workgroup tile. Each tile of 64 time steps is scanned in log₂(64) = 6 GPU barrier rounds, giving O(log N) wall-clock time on the GPU.

The associative operator for the recurrence `h_t = Ā·h_{t-1} + B̄·x_t` is:

```
(a₁, b₁) ∘ (a₂, b₂) = (a₁·a₂, a₁·b₂ + b₁)
```

Tiles are chained via a carry-in state, covering arbitrarily long sequences.

### 1D Causal Convolution (`conv1d.js`)
Depthwise 1D causal conv (kernel size K=4) with zero left-padding. Enforces causality by only reading positions `t-k` for `k ≥ 0`, contributing 0 for `t < k`.

### Linear Projection (`linear_projection.js`)
Tiled 16×16 GEMM in WGSL using workgroup shared memory. Handles arbitrary (M, K) × (N, K) → (M, N) shapes with boundary guards.

### AdamW Optimizer (`weight_update.js`)
Fused single-kernel AdamW update with decoupled weight decay. Includes a two-pass gradient norm clipping kernel (reduce → scale).

---

## Testing

```bash
npm test
```

Runs 58 unit tests covering quantization, BPE tokenization, autograd, and WGSL kernel source validation. GPU execution tests require a real browser with WebGPU support.

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
