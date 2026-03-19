# Getting Started with MambaCode.js

This guide walks you through installing MambaCode.js and running your first on-device code model — whether you're using **TypeScript** or plain **JavaScript**.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Node.js | 18 or later |
| Browser (runtime) | Chrome 113+, Edge 113+, or Firefox Nightly |
| WebGPU | Must be available in the target browser |

> **Node.js is only needed to build and bundle your project.** The compiled library runs entirely inside the browser using WebGPU. There is no Node.js server component.

---

## Installation

### From npm

```bash
npm install mambacode.js
```

The `dist/` folder shipped with the package contains:

| File | Purpose |
|---|---|
| `dist/index.js` | Compiled ESM entry point — for plain-JS consumers |
| `dist/index.d.ts` | TypeScript declaration file — auto-picked up by TS toolchains |
| `dist/**/*.js.map` | Source maps for debugging back to `.ts` source |

No additional build step is required when consuming from npm.

### From source

```bash
git clone https://github.com/SeanHogg/Mamba.git
cd Mamba
npm install
npm run build   # compiles TypeScript → dist/
```

---

## Project setup

### TypeScript project

1. Ensure your `tsconfig.json` targets ES2022 or later and enables ESM:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "strict": true
  }
}
```

2. Import directly — type declarations are resolved automatically:

```ts
import { MambaModel, BPETokenizer, initWebGPU } from 'mambacode.js';
```

### JavaScript project (ESM)

No configuration is needed. The `exports` field in the package points plain `import` statements at the compiled JS:

```js
import { MambaModel, BPETokenizer, initWebGPU } from 'mambacode.js';
```

If your bundler (Vite, Webpack, Rollup, esbuild) handles ESM you are ready to go without any extra setup.

---

## Step 1 — Initialise WebGPU

Before using any GPU functionality, acquire a `GPUDevice`:

```ts
import { initWebGPU } from 'mambacode.js';

const { device, adapter } = await initWebGPU({
  powerPreference: 'high-performance',   // optional
});
```

`initWebGPU` throws a descriptive error if:
- The browser does not expose `navigator.gpu` (no WebGPU support)
- No suitable GPU adapter is found

---

## Step 2 — Load a tokenizer

The `BPETokenizer` is compatible with Qwen3.5-Coder vocabulary files. You can serve them from your own CDN or bundle them in your app.

### Load from URL (recommended for large vocabularies)

```ts
import { BPETokenizer } from 'mambacode.js';

const tokenizer = new BPETokenizer();
await tokenizer.load('/assets/vocab.json', '/assets/merges.txt');

console.log('Vocabulary size:', tokenizer.vocabSize);
```

`vocab.json` format:

```json
{ "<token>": 0, "another": 1, ... }
```

`merges.txt` format (one merge rule per line, sorted by priority):

```
h e
e l
el l
...
```

### Load from in-memory objects (small/bundled vocabularies)

```ts
tokenizer.loadFromObjects(
  { 'hello': 0, 'world': 1, ... },   // vocab object
  ['h e', 'e l', ...]                 // merges array
);
```

---

## Step 3 — Create a model

### TypeScript (with type safety)

```ts
import { MambaModel, type MambaModelConfig } from 'mambacode.js';

const config: MambaModelConfig = {
  vocabSize  : tokenizer.vocabSize,
  dModel     : 512,    // embedding / hidden dimension
  numLayers  : 8,      // number of stacked Mamba blocks
  dState     : 16,     // SSM state dimension
  dConv      : 4,      // 1D conv kernel size
  expand     : 2,      // inner-dim expansion factor (dInner = expand × dModel)
};

const model = new MambaModel(device, config);
```

### JavaScript

```js
import { MambaModel } from 'mambacode.js';

const model = new MambaModel(device, {
  vocabSize  : tokenizer.vocabSize,
  dModel     : 512,
  numLayers  : 8,
});
```

Unspecified fields use sensible defaults (`dState: 16`, `dConv: 4`, `expand: 2`).

---

## Step 4 — Train on local code

The `MambaTrainer` class handles tokenisation, chunking, forward passes, loss computation, backpropagation, and the AdamW parameter update — all on the GPU.

### TypeScript

```ts
import { MambaTrainer, type TrainOptions } from 'mambacode.js';

const trainer = new MambaTrainer(model, tokenizer);

const opts: TrainOptions = {
  learningRate : 1e-4,
  epochs       : 5,
  seqLen       : 512,
  weightDecay  : 0.01,
  onEpochEnd   : (epoch: number, loss: number) => {
    console.log(`Epoch ${epoch}  loss=${loss.toFixed(4)}`);
  },
};

const losses: number[] = await trainer.train(myCodeString, opts);
```

### JavaScript

```js
import { MambaTrainer } from 'mambacode.js';

const trainer = new MambaTrainer(model, tokenizer);

const losses = await trainer.train(myCodeString, {
  learningRate : 1e-4,
  epochs       : 5,
  onEpochEnd   : (epoch, loss) => console.log(`Epoch ${epoch}  loss=${loss.toFixed(4)}`),
});
```

### `TrainOptions` reference

| Option | Type | Default | Description |
|---|---|---|---|
| `learningRate` | `number` | `1e-4` | AdamW learning rate |
| `epochs` | `number` | `5` | Number of full passes over the data |
| `batchSize` | `number` | `1` | Sequences per gradient step |
| `seqLen` | `number` | `512` | Token sequence length per chunk |
| `maxGradNorm` | `number` | `1.0` | Global gradient clipping threshold |
| `weightDecay` | `number` | `0.01` | AdamW decoupled weight decay |
| `beta1` | `number` | `0.9` | AdamW first-moment decay |
| `beta2` | `number` | `0.999` | AdamW second-moment decay |
| `eps` | `number` | `1e-8` | AdamW epsilon for numerical stability |
| `wsla` | `boolean` | `false` | WSLA mode — only fine-tunes B & C matrices |
| `onEpochEnd` | `(epoch, loss) => void` | — | Progress callback |

---

## Step 5 — Generate code

```ts
// TypeScript
const promptIds: number[] = tokenizer.encode('function fibonacci(n: number): number {');
const outputIds: number[] = await model.generate(promptIds, 200, {
  temperature : 0.8,   // diversity (higher = more random)
  topK        : 50,    // top-K filtering
  topP        : 0.9,   // nucleus (top-p) filtering
});
console.log(tokenizer.decode(outputIds));
```

```js
// JavaScript
const promptIds = tokenizer.encode('function fibonacci(n) {');
const outputIds = await model.generate(promptIds, 200, { temperature: 0.8 });
console.log(tokenizer.decode(outputIds));
```

---

## WSLA — Weight-Selective Local Adaptation

WSLA is a lightweight fine-tuning strategy that freezes all parameters **except** the B and C matrices of the selective scan. This dramatically reduces the number of trained parameters and allows fast domain adaptation on consumer hardware.

```ts
// TypeScript
await trainer.train(privateCodeSnippets, {
  learningRate : 1e-4,
  epochs       : 3,
  wsla         : true,
});
```

```js
// JavaScript
await trainer.train(privateCodeSnippets, {
  learningRate : 1e-4,
  epochs       : 3,
  wsla         : true,
});
```

---

## Evaluate perplexity

```ts
const ppl: number = await trainer.evaluate(heldOutCode);
console.log(`Perplexity: ${ppl.toFixed(2)}`);
```

---

## Working with quantization utilities

Reduce VRAM usage by storing weights as FP16 or Int8 before uploading to the GPU:

```ts
import {
  quantizeFp16, dequantizeFp16,
  quantizeInt8, dequantizeInt8,
  type QuantizeInt8Result,
} from 'mambacode.js';

// FP16 round-trip
const fp16: Uint16Array = quantizeFp16(float32Weights);
const restored: Float32Array = dequantizeFp16(fp16);

// Int8 round-trip
const { data, scale }: QuantizeInt8Result = quantizeInt8(float32Activations);
const dequantized: Float32Array = dequantizeInt8(data, scale);
```

```js
// JavaScript — same API, no type annotations
import { quantizeFp16, dequantizeFp16, quantizeInt8, dequantizeInt8 } from 'mambacode.js';

const fp16 = quantizeFp16(float32Weights);
const { data, scale } = quantizeInt8(float32Activations);
```

---

## Advanced — using raw WGSL kernels

All compiled WGSL shaders are exported for advanced users who want to build custom GPU pipelines:

```ts
import {
  SELECTIVE_SCAN_FORWARD_WGSL,
  LINEAR_FORWARD_WGSL,
  ACTIVATIONS_WGSL,
  createComputePipeline,
  createBindGroup,
  dispatchKernel,
} from 'mambacode.js';

const pipeline = createComputePipeline(device, SELECTIVE_SCAN_FORWARD_WGSL, 'forward_scan');
const bindGroup = createBindGroup(device, pipeline, [paramsBuffer, inputBuffer, outputBuffer]);
dispatchKernel(device, pipeline, bindGroup, [Math.ceil(seqLen / 64), dInner, batch]);
```

---

## Development — building and testing

```bash
npm run build   # tsc: TypeScript → dist/  (required before publishing)
npm test        # Jest: runs 58 unit tests (no GPU required)
npm run lint    # ESLint: checks src/ and tests/
```

Tests run entirely in Node.js without a GPU. GPU-dependent paths (model forward, training, generation) require a browser with WebGPU support and are exercised via manual browser testing.

---

## Troubleshooting

### "WebGPU is not available in this environment"

`initWebGPU()` requires `navigator.gpu` which is only available in browsers. It will throw this error in Node.js. Use a bundler (Vite, Webpack, esbuild) to target a browser environment.

### TypeScript: "Cannot find module 'mambacode.js'"

Ensure `dist/` exists by running `npm run build` first (if using a local clone), or that the npm package is installed (`npm install mambacode.js`).

### "Failed to acquire a GPUAdapter"

Your GPU driver or browser version may not support WebGPU. Verify at [webgpureport.org](https://webgpureport.org) or try Chrome Canary with `--enable-unsafe-webgpu`.

### Build errors in strict TypeScript

If you consume the library in a project with very strict settings (e.g. `noUncheckedIndexedAccess`), all exported types are non-nullable and well-typed. If you encounter an issue, please [open an issue](https://github.com/SeanHogg/Mamba/issues).
