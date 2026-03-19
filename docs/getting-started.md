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
> **New to AI and language models?** This guide walks you through everything from the basics of how large language models work, all the way to running your first on-device code model — entirely in the browser.

---

## Table of Contents

1. [What is a Language Model?](#1-what-is-a-language-model)
2. [Where Does Qwen Fit In?](#2-where-does-qwen-fit-in)
3. [What is Mamba and Why Does it Matter?](#3-what-is-mamba-and-why-does-it-matter)
4. [How MambaCode.js Brings It All Together](#4-how-mambacodejs-brings-it-all-together)
5. [Prerequisites](#5-prerequisites)
6. [Step-by-Step: Your First On-Device Model](#6-step-by-step-your-first-on-device-model)
7. [The Complete Flow at a Glance](#7-the-complete-flow-at-a-glance)
8. [What Happens Next?](#8-what-happens-next)
9. [Using builderforce.ai](#9-using-builderforceai)
10. [Common Questions](#10-common-questions)

---

## 1. What is a Language Model?

A **language model (LM)** is a type of AI that learns patterns in text (or code) and uses those patterns to predict what comes next. When you type a sentence and your phone suggests the next word, that is a tiny language model at work.

**Large Language Models (LLMs)** take the same idea much further: they are trained on billions of lines of text and code, giving them the ability to:

- Answer questions in natural language
- Write, explain, and debug code
- Summarise documents and translate text
- Power AI assistants and copilots

### How are they trained?

Training a language model means adjusting millions (or billions) of numerical *weights* so that the model gets better at predicting the next token. A **token** is roughly a word, part of a word, or a code symbol — the model never sees raw characters, only these numeric IDs.

```
Raw text:  "function add("
Tokens:    [1, 2543, 912, 40]   ← numeric IDs the model actually processes
```

Training is done by:

1. Feeding the model a sequence of tokens
2. Asking it to predict the next token
3. Measuring the error (the *loss*)
4. Adjusting the weights to reduce the loss (back-propagation + optimizer)

After millions of these steps the model "knows" a great deal about language and code.

---

## 2. Where Does Qwen Fit In?

**Qwen3.5-Coder** is a family of open-weight code-focused language models built by Alibaba Cloud. The 0.8 B (800 million parameter) variant is small enough to run locally yet capable enough to help with real code tasks.

MambaCode.js targets the **Qwen3.5-Coder-0.8B tokenizer vocabulary** (151 936 tokens). This means:

- The tokenizer included in this library produces the same token IDs that a Qwen3.5-Coder model expects.
- Weights exported from (or compatible with) Qwen3.5-Coder-0.8B can be loaded directly.
- You can fine-tune the model on your own codebase and the output remains Qwen-compatible.

> **In short:** Qwen provides the pre-trained knowledge. MambaCode.js provides the engine to run and adapt it — locally, in the browser, with no data leaving your machine.

---

## 3. What is Mamba and Why Does it Matter?

Traditional LLMs use a **Transformer** architecture whose attention mechanism has quadratic cost: doubling the sequence length roughly quadruples the compute. This is manageable in a data centre but challenging on a single GPU or browser.

**Mamba** is a *State Space Model (SSM)* — specifically the **S6 (Selective Scan)** variant. Instead of attending to every previous token, Mamba maintains a compact hidden state that is updated step-by-step:

```
h_t = Ā · h_{t-1}  +  B̄ · x_t     (update state with new input)
y_t = C  · h_t     +  D · x_t     (read from state to produce output)
```

Key advantages over a Transformer for on-device use:

| Property | Transformer | Mamba (S6) |
|---|---|---|
| Context scaling | O(N²) | **O(N)** |
| Memory per token | Grows with sequence | **Constant** |
| Inference speed | Slows with length | **Constant** |
| Good for local GPU | Difficult | **Yes** |

This linear scaling is what makes it possible to run a meaningful code model inside a browser tab.

---

## 4. How MambaCode.js Brings It All Together

```
┌──────────────────────────────────────────────────────────┐
│                       Browser Tab                        │
│                                                          │
│  Your code / text                                        │
│       │                                                  │
│       ▼                                                  │
│  BPETokenizer  ──►  token IDs (Qwen3.5-Coder vocab)     │
│       │                                                  │
│       ▼                                                  │
│  MambaModel  (WebGPU / WGSL kernels on your local GPU)  │
│       │                                                  │
│  ┌────┴────────────────────┐                             │
│  │  Optional: MambaTrainer │  ← fine-tune on your code  │
│  └────┬────────────────────┘                             │
│       │                                                  │
│       ▼                                                  │
│  Generated tokens  ──►  BPETokenizer.decode  ──► text   │
│                                                          │
│  ✅  No network calls. No cloud. Private by design.      │
└──────────────────────────────────────────────────────────┘
```

The three main pieces:

| Component | Role |
|---|---|
| **BPETokenizer** | Converts text ↔ token IDs using the Qwen3.5-Coder vocabulary |
| **MambaModel** | Runs the Mamba SSM forward pass on the GPU via WebGPU |
| **MambaTrainer** | Fine-tunes the model on your local code using AdamW + autograd |

---

## 5. Prerequisites

### Browser

| Browser | Minimum Version | Notes |
|---|---|---|
| Chrome | 113+ | Recommended — best WebGPU support |
| Edge | 113+ | Same Chromium engine as Chrome |
| Firefox | Nightly | Enable `dom.webgpu.enabled` in `about:config` |
| Safari | 18+ | WebGPU in preview; some features may be limited |

> Open `chrome://gpu` and look for **WebGPU** to confirm it is enabled.

### Hardware

A dedicated GPU is not required, but it dramatically speeds things up. The model fits within **≤ 3 GB VRAM**, which covers most modern integrated and discrete graphics.

### Node.js (for local development only)

If you want to run the test suite or bundle the library:

```bash
node --version   # must be >= 18
```

---

## 6. Step-by-Step: Your First On-Device Model

### Step 1 — Install (local development)

```bash
git clone https://github.com/SeanHogg/Mamba.git
cd Mamba
npm install
```

Or, if you are building a web app:

```html
<script type="module">
  import { MambaModel, MambaTrainer, BPETokenizer, initWebGPU }
    from 'https://cdn.jsdelivr.net/npm/mambacode.js/src/index.js';
</script>
```

### Step 2 — Check WebGPU is available

```js
import { initWebGPU } from './src/index.js';

const { device } = await initWebGPU();
console.log('WebGPU ready ✅', device.label);
```

If this throws, check that your browser supports WebGPU (see [Prerequisites](#5-prerequisites)).

### Step 3 — Load the tokenizer

Download the Qwen3.5-Coder vocabulary files (`vocab.json` and `merges.txt`) and serve them from your web server:

```js
import { BPETokenizer } from './src/index.js';

const tokenizer = new BPETokenizer();
await tokenizer.load('/vocab.json', '/merges.txt');

console.log('Vocabulary size:', tokenizer.vocabSize);  // 151936
```

### Step 4 — Create the model

```js
import { MambaModel } from './src/index.js';

const model = new MambaModel(device, {
  vocabSize : tokenizer.vocabSize,   // 151936
  dModel    : 512,   // hidden dimension
  numLayers : 8,     // number of Mamba blocks
  dState    : 16,    // SSM state size
  dConv     : 4,     // causal conv kernel width
  expand    : 2,     // inner expansion factor
});
```

> **Tip:** Start with the small configuration above (≈ 50 M parameters). Increase `dModel` or `numLayers` only if you have a discrete GPU with more VRAM.

### Step 5 — (Optional) Fine-tune on your code

This is the real power of MambaCode.js: you can teach the model about *your* private codebase without sending anything to a server.

```js
import { MambaTrainer } from './src/index.js';

// myCodeString can be any text, e.g. the contents of your project files
const myCodeString = `
function greet(name) {
  return `Hello, ${name}!`;
}
`;

const trainer = new MambaTrainer(model, tokenizer);
const losses = await trainer.train(myCodeString, {
  learningRate : 1e-4,
  epochs       : 5,
  device       : 'webgpu',
  onEpochEnd   : (epoch, loss) =>
    console.log(`Epoch ${epoch}: loss = ${loss.toFixed(4)}`),
});

console.log('Training complete. Final loss:', losses.at(-1).toFixed(4));
```

For rapid adaptation with minimal compute, use **WSLA** mode (only updates the B and C matrices):

```js
await trainer.train(myCodeString, {
  learningRate : 1e-4,
  epochs       : 3,
  wsla         : true,   // faster, lower memory
});
```

### Step 6 — Generate code

```js
const prompt      = 'function fibonacci(';
const promptIds   = tokenizer.encode(prompt);
const outputIds   = await model.generate(promptIds, 200, { temperature: 0.8 });
const outputText  = tokenizer.decode(outputIds);

console.log(prompt + outputText);
```

You should see the model continue the function — using everything it has learned from Qwen3.5-Coder *plus* whatever you trained it on in Step 5.

---

## 7. The Complete Flow at a Glance

```
1. Load tokenizer vocab  ──►  tokenizer.load()
         │
         ▼
2. Create model          ──►  new MambaModel(device, config)
         │
         ▼
3. (Optional) Train      ──►  trainer.train(codeString, options)
         │                      Runs entirely on your local GPU
         │                      No data leaves the browser
         ▼
4. Generate              ──►  model.generate(promptIds, maxTokens)
         │
         ▼
5. Decode output         ──►  tokenizer.decode(outputIds)
         │
         ▼
6. Use the generated code in your application
```

---

## 8. What Happens Next?

Once you have the basics working, here are some natural next steps:

### Load pre-trained weights

A randomly-initialised model generates nonsense. To get useful output immediately, load weights that have already been trained on large amounts of code:

```js
// Fetch serialised weights from your server (never from a third-party URL)
const response = await fetch('/models/mamba-coder-0.8b.bin');
const buffer   = await response.arrayBuffer();
await model.loadWeights(buffer);
```

### Build a code-completion UI

Wire the model to a `<textarea>` and call `model.generate()` on every keypress (or on a timer) to show real-time suggestions — entirely client-side.

### Export and share fine-tuned weights

After training on your codebase you can serialise the adapted weights and share them with your team, without exposing the underlying code:

```js
const weights = await model.exportWeights();
// save weights to a file or IndexedDB for later use
```

### Integrate with builderforce.ai

See [Section 9](#9-using-builderforceai) for how builderforce.ai makes all of the above easier.

### Explore the WGSL kernels

If you want to go deeper, the `src/kernels/` directory contains hand-written WGSL shaders for every operation. Reading these is a great introduction to GPU programming for ML.

---

## 9. Using builderforce.ai

[**builderforce.ai**](https://builderforce.ai) is the platform built around MambaCode.js, designed to make on-device AI development accessible without requiring a machine-learning background.

### What builderforce.ai provides

| Feature | How it helps |
|---|---|
| **Model library** | Browse and download pre-trained Mamba and Qwen-compatible weights |
| **Fine-tune UI** | Upload your code files and fine-tune a model through a web interface — no JavaScript required |
| **Prompt playground** | Experiment with code-generation prompts against any model in your library |
| **Team sharing** | Share fine-tuned model checkpoints with colleagues |
| **Integration guides** | Step-by-step recipes for VSCode extensions, CI pipelines, and web apps |

### Getting started with builderforce.ai

1. **Visit [builderforce.ai](https://builderforce.ai)** and create a free account.
2. Navigate to the **Model Library** and download a starter Mamba-Coder checkpoint.
3. Drop the downloaded `.bin` file into your project and load it with `model.loadWeights()`.
4. Use the **Fine-tune UI** to upload your private code — the fine-tuning runs in your browser; builderforce.ai never receives the code itself.
5. Share the resulting checkpoint with your team through the **Team Sharing** panel.

### Using the builderforce.ai API from MambaCode.js

```js
// The API is used only for account-gated model downloads.
// Your code and weights never leave your machine.
import { BuilderForceClient } from 'https://cdn.builderforce.ai/sdk/v1/client.js';

const client = new BuilderForceClient({ apiKey: 'YOUR_API_KEY' });
const modelMeta = await client.models.get('mamba-coder-0.8b-base');

const response = await fetch(modelMeta.downloadUrl);
const buffer   = await response.arrayBuffer();
await model.loadWeights(buffer);
```

---

## 10. Common Questions

**Q: Do I need to understand machine learning to use this?**
No. Follow the steps in [Section 6](#6-step-by-step-your-first-on-device-model) and you will have a working code model in minutes. The deeper concepts in Sections 1–3 are background reading, not prerequisites.

---

**Q: Does MambaCode.js send my code anywhere?**
No. Everything — tokenization, model weights, training, and inference — runs locally in your browser using WebGPU. The only network calls are the initial fetches for vocab files and model weights from URLs *you* choose.

---

**Q: What is the difference between Mamba and a Transformer (like GPT)?**
Both are language models, but Mamba processes sequences in linear time (O(N)) instead of quadratic (O(N²)). This makes Mamba significantly faster for long contexts and more practical on consumer hardware. See [Section 3](#3-what-is-mamba-and-why-does-it-matter) for details.

---

**Q: Why Qwen3.5-Coder specifically?**
Qwen3.5-Coder-0.8B strikes a good balance between model capability and size. Its tokenizer vocabulary (151 936 tokens) is designed for code, handling identifiers, operators, and indentation efficiently. The 0.8 B parameter count fits comfortably within the 3 GB VRAM budget of most consumer GPUs.

---

**Q: Can I use a different vocabulary / model architecture?**
Yes, with some code changes. The `BPETokenizer` class accepts any `vocab.json` / `merges.txt` pair. The `MambaModel` config can be adjusted to match different layer counts and hidden dimensions. If you need help adapting the library, check the [builderforce.ai integration guides](https://builderforce.ai).

---

**Q: The model generates gibberish. What is wrong?**
A freshly-initialised model with random weights always generates gibberish. You need either (a) pre-trained weights (see Section 8) or (b) to train the model on a large corpus first. For a quick sanity-check, train on a small repeated string and verify the loss decreases.

---

*Back to [README](../README.md)*
