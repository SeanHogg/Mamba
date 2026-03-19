/**
 * MambaCode.js – Entry Point
 *
 * High-performance JavaScript/WGSL Mamba SSM library for browser-based
 * code model training and inference.
 *
 * Quick-start example
 * -------------------
 * import { MambaModel, MambaTrainer, BPETokenizer, initWebGPU } from 'mambacode.js';
 *
 * const { device }   = await initWebGPU();
 * const tokenizer    = new BPETokenizer();
 * await tokenizer.load('/vocab.json', '/merges.txt');
 *
 * const model = new MambaModel(device, {
 *   vocabSize : tokenizer.vocabSize,
 *   dModel    : 512,
 *   numLayers : 8,
 * });
 *
 * const trainer = new MambaTrainer(model, tokenizer);
 * const losses  = await trainer.train(myCodeString, { learningRate: 1e-4, epochs: 5 });
 *
 * const generated = await model.generate(tokenizer.encode('function '), 100);
 * console.log(tokenizer.decode(generated));
 */

// ── Core model ────────────────────────────────────────────────────────────────
export { MambaModel }   from './model/mamba_model.js';
export { MambaBlock }   from './model/mamba_block.js';

// ── Training ──────────────────────────────────────────────────────────────────
export { MambaTrainer } from './training/trainer.js';
export {
    Tensor,
    backward,
    enableGrad,
    noGrad,
    clearTape,
    recordOperation,
    crossEntropyLoss,
    crossEntropyGrad,
} from './training/autograd.js';

// ── Tokenizer ─────────────────────────────────────────────────────────────────
export { BPETokenizer } from './tokenizer/bpe.js';

// ── WebGPU utilities ──────────────────────────────────────────────────────────
export {
    initWebGPU,
    createStorageBuffer,
    createEmptyStorageBuffer,
    createUniformBuffer,
    createComputePipeline,
    createBindGroup,
    dispatchKernel,
    readBuffer,
    uploadBuffer,
    cdiv,
} from './utils/gpu_utils.js';

// ── Quantization utilities ────────────────────────────────────────────────────
export {
    quantizeFp16,
    dequantizeFp16,
    floatToFp16,
    fp16ToFloat,
    quantizeInt8,
    dequantizeInt8,
    quantizeInt8PerChannel,
    dequantizeInt8PerChannel,
    estimateMemory,
} from './utils/quantization.js';

// ── Raw WGSL kernel sources (for advanced users / custom pipelines) ───────────
export { SELECTIVE_SCAN_FORWARD_WGSL, SELECTIVE_SCAN_BACKWARD_WGSL }
    from './kernels/selective_scan.js';
export { CONV1D_FORWARD_WGSL, CONV1D_BACKWARD_WGSL }
    from './kernels/conv1d.js';
export { LINEAR_FORWARD_WGSL, LINEAR_BACKWARD_WGSL }
    from './kernels/linear_projection.js';
export { WEIGHT_UPDATE_WGSL, GRAD_CLIP_WGSL }
    from './kernels/weight_update.js';
export { ACTIVATIONS_WGSL, ACTIVATIONS_BACKWARD_WGSL }
    from './kernels/activations.js';

// ── Library metadata ──────────────────────────────────────────────────────────
export const VERSION = '0.1.0';
export const DESCRIPTION = 'MambaCode.js: WebGPU-accelerated Mamba SSM for browser code models';
