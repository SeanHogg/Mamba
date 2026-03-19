/**
 * autograd.js – Lightweight tape-based automatic differentiation engine.
 *
 * Design
 * ------
 * Every differentiable GPU operation appends an entry to a global "tape"
 * (a reverse-mode AD record).  During the backward pass we replay the tape
 * in reverse, dispatching backward GPU kernels that accumulate gradients
 * into per-parameter gradient buffers.
 *
 * A "Tensor" in this context is a thin wrapper that holds:
 *   - a GPUBuffer (the data)
 *   - shape metadata
 *   - an optional gradient GPUBuffer
 *   - a reference to the tape node that produced it
 *
 * The tape stores closures so that complex operations (selective scan,
 * conv, linear) can have their own custom backward logic.
 */

/** @type {TapeEntry[]} */
let _tape = [];
let _gradEnabled = true;

/**
 * @typedef {Object} TapeEntry
 * @property {() => void} backward  – closure that computes and accumulates gradients
 */

/**
 * Tensor – wraps a GPUBuffer with shape, gradient, and autograd metadata.
 */
export class Tensor {
    /**
     * @param {GPUBuffer}   data     – GPU buffer holding the tensor values (FP32)
     * @param {number[]}    shape    – dimensions, e.g. [batch, seqLen, dInner]
     * @param {boolean}     [requiresGrad=false]
     */
    constructor(data, shape, requiresGrad = false) {
        this.data         = data;
        this.shape        = shape;
        this.numel        = shape.reduce((a, b) => a * b, 1);
        this.requiresGrad = requiresGrad;
        this.grad         = null;   // GPUBuffer, populated during backward()
        this._gradFn      = null;   // tape node index
    }

    /** Number of bytes occupied by this tensor (FP32). */
    get byteSize() { return this.numel * 4; }

    /**
     * Manually zero-out the gradient buffer (keeps the GPUBuffer allocated).
     * @param {GPUDevice} device
     */
    zeroGrad(device) {
        if (this.grad) {
            device.queue.writeBuffer(this.grad, 0, new Float32Array(this.numel));
        }
    }

    /** Free GPU memory for both data and grad buffers. */
    destroy() {
        this.data?.destroy();
        this.grad?.destroy();
        this.data = null;
        this.grad = null;
    }
}

// ─── Tape control ─────────────────────────────────────────────────────────────

/** Start recording operations onto the tape. */
export function enableGrad()  { _gradEnabled = true;  }

/** Stop recording (inference-only mode). */
export function noGrad()      { _gradEnabled = false; }

/** Clear the tape without running backward. */
export function clearTape()   { _tape = []; }

/**
 * Register a backward closure onto the tape.
 * Called internally by differentiable operations.
 *
 * @param {() => void} backwardFn
 * @returns {number} tape index (for reference by the output Tensor)
 */
export function recordOperation(backwardFn) {
    if (!_gradEnabled) return -1;
    _tape.push({ backward: backwardFn });
    return _tape.length - 1;
}

// ─── Backward pass ────────────────────────────────────────────────────────────

/**
 * Run the backward pass by replaying the tape in reverse.
 * Gradients accumulate into the `.grad` GPUBuffers of leaf tensors.
 *
 * After backward() the tape is cleared automatically.
 */
export async function backward() {
    for (let i = _tape.length - 1; i >= 0; i--) {
        await _tape[i].backward();
    }
    clearTape();
}

// ─── Gradient buffer management ───────────────────────────────────────────────

/**
 * Ensure a Tensor has an allocated (zeroed) gradient buffer.
 *
 * @param {GPUDevice} device
 * @param {Tensor}    tensor
 */
export function ensureGradBuffer(device, tensor) {
    if (!tensor.grad) {
        tensor.grad = device.createBuffer({
            size  : tensor.byteSize,
            usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        // Zero-init
        device.queue.writeBuffer(tensor.grad, 0, new Float32Array(tensor.numel));
    }
}

/**
 * Allocate gradient buffers for a list of tensors.
 *
 * @param {GPUDevice}  device
 * @param {Tensor[]}   tensors
 */
export function allocateGradients(device, tensors) {
    for (const t of tensors) {
        if (t.requiresGrad) ensureGradBuffer(device, t);
    }
}

/**
 * Zero all gradient buffers in-place (GPU write).
 *
 * @param {GPUDevice}  device
 * @param {Tensor[]}   tensors
 */
export function zeroGradients(device, tensors) {
    for (const t of tensors) {
        if (t.grad) {
            device.queue.writeBuffer(t.grad, 0, new Float32Array(t.numel));
        }
    }
}

// ─── Loss helpers ─────────────────────────────────────────────────────────────

/**
 * Create a scalar "1.0" gradient tensor to seed the backward pass.
 * (Equivalent to calling loss.backward() with grad=1.)
 *
 * @param {GPUDevice} device
 * @returns {GPUBuffer}  – single-element FP32 buffer containing 1.0
 */
export function onesLikeScalar(device) {
    const buf = device.createBuffer({
        size  : 4,
        usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange()).set([1.0]);
    buf.unmap();
    return buf;
}

/**
 * Cross-entropy loss (computed on CPU after reading back logits).
 * Returns a scalar JS number.
 *
 * @param {Float32Array} logits    – (vocabSize,)
 * @param {number}       targetId  – correct token index
 * @returns {number}
 */
export function crossEntropyLoss(logits, targetId) {
    // Numerically stable softmax
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) {
        sumExp += Math.exp(logits[i] - maxLogit);
    }
    const logSumExp = Math.log(sumExp) + maxLogit;
    return logSumExp - logits[targetId];
}

/**
 * Gradient of the cross-entropy loss w.r.t. logits.
 * Returns a Float32Array of shape (vocabSize,).
 *
 * @param {Float32Array} logits
 * @param {number}       targetId
 * @returns {Float32Array}
 */
export function crossEntropyGrad(logits, targetId) {
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    let sumExp = 0;
    const exp_shifted = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
        exp_shifted[i] = Math.exp(logits[i] - maxLogit);
        sumExp += exp_shifted[i];
    }
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
        probs[i] = exp_shifted[i] / sumExp;
    }
    probs[targetId] -= 1.0;   // dL/d logit_i = prob_i - 1{i==target}
    return probs;
}
