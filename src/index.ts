/**
 * MambaCode.js – Entry Point
 */

export { MambaModel }   from './model/mamba_model';
export { MambaBlock }   from './model/mamba_block';

export { MambaTrainer } from './training/trainer';
export {
    Tensor,
    backward,
    enableGrad,
    noGrad,
    clearTape,
    recordOperation,
    crossEntropyLoss,
    crossEntropyGrad,
} from './training/autograd';

export { BPETokenizer } from './tokenizer/bpe';

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
} from './utils/gpu_utils';

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
} from './utils/quantization';

export { SELECTIVE_SCAN_FORWARD_WGSL, SELECTIVE_SCAN_BACKWARD_WGSL }
    from './kernels/selective_scan';
export { CONV1D_FORWARD_WGSL, CONV1D_BACKWARD_WGSL }
    from './kernels/conv1d';
export { LINEAR_FORWARD_WGSL, LINEAR_BACKWARD_WGSL }
    from './kernels/linear_projection';
export { WEIGHT_UPDATE_WGSL, GRAD_CLIP_WGSL }
    from './kernels/weight_update';
export { ACTIVATIONS_WGSL, ACTIVATIONS_BACKWARD_WGSL }
    from './kernels/activations';

export const VERSION = '1.0.1';
export const DESCRIPTION = 'MambaCode.js: WebGPU-accelerated Mamba SSM for browser code models';
