/**
 * mamba_block.js – Mamba Mixer Block
 *
 * Implements one complete Mamba residual layer:
 *
 *   x  ──► Norm ──► Linear up (×2, for z-gate) ──► Conv1D ──► SiLU ──► Scan ──► × z ──► Linear down ──► + x
 *
 * Components (all dispatched as WebGPU compute passes):
 *   1. RMSNorm
 *   2. Linear up-projection: (D_model → 2 × D_inner)
 *   3. 1D Causal Convolution (depthwise, kernel_size=4)
 *   4. SiLU activation
 *   5. Selective Scan (S6 core)
 *   6. Gated multiplication: y * SiLU(z)
 *   7. Linear down-projection: (D_inner → D_model)
 *   8. Residual add
 */

import {
    createComputePipeline,
    createBindGroup,
    createStorageBuffer,
    createEmptyStorageBuffer,
    createUniformBuffer,
    dispatchKernel,
    cdiv,
} from '../utils/gpu_utils.js';

import { SELECTIVE_SCAN_FORWARD_WGSL }  from '../kernels/selective_scan.js';
import { CONV1D_FORWARD_WGSL }          from '../kernels/conv1d.js';
import { LINEAR_FORWARD_WGSL }          from '../kernels/linear_projection.js';
import { ACTIVATIONS_WGSL }             from '../kernels/activations.js';

/**
 * @typedef {Object} MambaBlockConfig
 * @property {number} dModel       – model dimension (embedding size)
 * @property {number} dState       – SSM state dimension (N, default 16)
 * @property {number} dConv        – 1D conv kernel size (default 4)
 * @property {number} expand       – expansion factor (default 2)  → dInner = expand * dModel
 * @property {number} dtRank       – rank of Δ projection (default ceil(dModel/16))
 * @property {boolean} [biasConv]  – use bias in conv (default true)
 */

export class MambaBlock {
    /**
     * @param {GPUDevice}       device
     * @param {MambaBlockConfig} config
     */
    constructor(device, config) {
        this.device  = device;
        this.config  = {
            dState  : 16,
            dConv   : 4,
            expand  : 2,
            biasConv: true,
            ...config,
        };

        const { dModel, dState, dConv, expand } = this.config;
        this.dInner  = expand * dModel;
        this.dtRank  = this.config.dtRank ?? Math.ceil(dModel / 16);

        // ---- Initialise learnable parameters (CPU → GPU) ----
        this._initWeights();

        // ---- Compile GPU pipelines (once) ----
        this._buildPipelines();
    }

    // ─── Weight initialisation ────────────────────────────────────────────────

    _initWeights() {
        const { dModel, dState, dConv } = this.config;
        const D = this.dInner;
        const N = dState;
        const K = dConv;
        const R = this.dtRank;

        const randn = (n, std = 0.02) => {
            const a = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                // Box-Muller
                const u1 = Math.random(), u2 = Math.random();
                a[i] = std * Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
            }
            return a;
        };

        const zeros  = (n)    => new Float32Array(n);
        const ones   = (n)    => new Float32Array(n).fill(1.0);
        const linspace = (n)  => {
            const a = new Float32Array(n);
            for (let i = 0; i < n; i++) a[i] = i;
            return a;
        };

        // in_proj: (2*D_inner, D_model) – up-projection (and z gate)
        this.wInProj  = randn(2 * D * dModel);
        this.bInProj  = zeros(2 * D);

        // conv1d: weight (D_inner, K), bias (D_inner,)
        this.wConv    = randn(D * K, 0.01);
        this.bConv    = zeros(D);

        // x_proj: (dt_rank + 2*N, D_inner) – projects x to Δ, B, C
        this.wXProj   = randn((R + 2 * N) * D, 0.01);
        this.bXProj   = zeros(R + 2 * N);

        // dt_proj: (D_inner, dt_rank) – projects Δ to full D_inner width
        this.wDtProj  = randn(D * R, 0.02);
        this.bDtProj  = zeros(D);

        // A: (D_inner, N) – log-space negative eigenvalues
        // Initialised to log(range(1, N+1)) per HiPPO theory
        this.A_log    = new Float32Array(D * N);
        for (let d = 0; d < D; d++) {
            for (let n = 0; n < N; n++) {
                this.A_log[d * N + n] = Math.log(n + 1);
            }
        }

        // D: (D_inner,) – skip connection scale (initialised to 1)
        this.D_vec    = ones(D);

        // out_proj: (D_model, D_inner) – down-projection
        this.wOutProj = randn(dModel * D, 0.02);
        this.bOutProj = zeros(dModel);

        // RMSNorm scale: (D_model,)
        this.normWeight = ones(dModel);

        // Upload all to GPU
        this._uploadWeightsToGPU();
    }

    _uploadWeightsToGPU() {
        const d  = this.device;
        const mk = (arr, readable = true) => createStorageBuffer(d, arr, readable);

        this.gpuWeights = {
            wInProj  : mk(this.wInProj),
            bInProj  : mk(this.bInProj),
            wConv    : mk(this.wConv),
            bConv    : mk(this.bConv),
            wXProj   : mk(this.wXProj),
            bXProj   : mk(this.bXProj),
            wDtProj  : mk(this.wDtProj),
            bDtProj  : mk(this.bDtProj),
            A_log    : mk(this.A_log),
            D_vec    : mk(this.D_vec),
            wOutProj : mk(this.wOutProj),
            bOutProj : mk(this.bOutProj),
            normWeight: mk(this.normWeight),
        };
    }

    // ─── Pipeline compilation ─────────────────────────────────────────────────

    _buildPipelines() {
        const d = this.device;

        this.pipelines = {
            linear    : createComputePipeline(d, LINEAR_FORWARD_WGSL,           'linear_forward'),
            conv1d    : createComputePipeline(d, CONV1D_FORWARD_WGSL,           'conv1d_forward'),
            silu      : createComputePipeline(d, ACTIVATIONS_WGSL,              'silu_forward'),
            rmsnorm   : createComputePipeline(d, ACTIVATIONS_WGSL,              'rmsnorm_forward'),
            scan_fwd  : createComputePipeline(d, SELECTIVE_SCAN_FORWARD_WGSL,   'forward_scan'),
            scan_reduce: createComputePipeline(d, SELECTIVE_SCAN_FORWARD_WGSL,  'forward_reduce'),
        };
    }

    // ─── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Run the Mamba block forward pass on GPU.
     *
     * @param {GPUBuffer} xBuf   – input (batch * seqLen, dModel)
     * @param {number}    batch
     * @param {number}    seqLen
     * @returns {{ output: GPUBuffer, cache: Object }}
     *   output  – (batch * seqLen, dModel)
     *   cache   – intermediate buffers needed for backward pass
     */
    forward(xBuf, batch, seqLen) {
        const d = this.device;
        const { dModel, dState, dConv } = this.config;
        const D = this.dInner;
        const N = dState;
        const B = batch;
        const L = seqLen;
        const M = B * L;
        const R = this.dtRank;

        // Intermediate buffers (will be freed after backward or cached)
        const cache = {};

        // 1. RMSNorm: (M, dModel)
        const normOut  = createEmptyStorageBuffer(d, M * dModel * 4, true);
        const normInv  = createEmptyStorageBuffer(d, M * 4,          true);
        cache.normInv  = normInv;
        cache.normIn   = xBuf;

        {
            // Pack params as Uint32 (num_rows, dim) + f32 (eps) ← 12 bytes padded to 16
            const params = new ArrayBuffer(16);
            new Uint32Array(params, 0, 2).set([M, dModel]);
            new Float32Array(params, 8, 1).set([1e-6]);
            const pBuf = createUniformBuffer(d, params);

            const bg = createBindGroup(d, this.pipelines.rmsnorm,
                [pBuf, xBuf, this.gpuWeights.normWeight, normOut, normInv]);
            dispatchKernel(d, this.pipelines.rmsnorm, bg, [cdiv(M, 64), 1, 1]);
        }

        // 2. in_proj: (M, 2*D) = normOut @ wInProj^T + bInProj
        const inProjOut = createEmptyStorageBuffer(d, M * 2 * D * 4, true);
        cache.normOut   = normOut;
        {
            const params = new Uint32Array([M, dModel, 2 * D]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines.linear,
                [pBuf, normOut, this.gpuWeights.wInProj, this.gpuWeights.bInProj, inProjOut]);
            dispatchKernel(d, this.pipelines.linear, bg, [cdiv(M, 16), cdiv(2 * D, 16), 1]);
        }

        // Split inProjOut into x (M, D) and z (M, D) – the z-gate
        // We reuse the same buffer with offsets since WGSL bindings can be offset.
        // For simplicity, allocate two separate buffers and copy.
        const xConvIn  = createEmptyStorageBuffer(d, M * D * 4, true);
        const zBuf     = createEmptyStorageBuffer(d, M * D * 4, true);
        {
            // Copy first D columns into xConvIn, last D columns into zBuf
            const enc = d.createCommandEncoder();
            enc.copyBufferToBuffer(inProjOut, 0,           xConvIn, 0, M * D * 4);
            enc.copyBufferToBuffer(inProjOut, M * D * 4,   zBuf,    0, M * D * 4);
            d.queue.submit([enc.finish()]);
        }
        cache.zBuf = zBuf;

        // 3. Conv1D on xConvIn: (B, L, D) – depthwise causal conv
        const convOut = createEmptyStorageBuffer(d, M * D * 4, true);
        cache.xConvIn = xConvIn;
        {
            const params = new Uint32Array([L, D, dConv, B]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines.conv1d,
                [pBuf, xConvIn, this.gpuWeights.wConv, this.gpuWeights.bConv, convOut]);
            dispatchKernel(d, this.pipelines.conv1d, bg, [cdiv(L, 16), cdiv(D, 16), B]);
        }

        // 4. SiLU(convOut) in-place
        const siluOut = createEmptyStorageBuffer(d, M * D * 4, true);
        cache.convOut = convOut;
        {
            const params = new Uint32Array([M * D]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines.silu,
                [pBuf, convOut, siluOut]);
            dispatchKernel(d, this.pipelines.silu, bg, [cdiv(M * D, 256), 1, 1]);
        }

        // 5. x_proj: (M, R+2N) = siluOut @ wXProj^T + bXProj
        const xProjOut = createEmptyStorageBuffer(d, M * (R + 2 * N) * 4, true);
        {
            const params = new Uint32Array([M, D, R + 2 * N]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines.linear,
                [pBuf, siluOut, this.gpuWeights.wXProj, this.gpuWeights.bXProj, xProjOut]);
            dispatchKernel(d, this.pipelines.linear, bg, [cdiv(M, 16), cdiv(R + 2 * N, 16), 1]);
        }

        // Split xProjOut → dtRaw (M, R), B_raw (M*N flattened) = (B, L, N), C_raw (B, L, N)
        const dtRaw = createEmptyStorageBuffer(d, M * R * 4,     true);
        const B_raw = createEmptyStorageBuffer(d, B * L * N * 4, true);
        const C_raw = createEmptyStorageBuffer(d, B * L * N * 4, true);
        {
            const enc = d.createCommandEncoder();
            enc.copyBufferToBuffer(xProjOut, 0,                  dtRaw, 0, M * R * 4);
            enc.copyBufferToBuffer(xProjOut, M * R * 4,          B_raw, 0, B * L * N * 4);
            enc.copyBufferToBuffer(xProjOut, M * (R + N) * 4,    C_raw, 0, B * L * N * 4);
            d.queue.submit([enc.finish()]);
        }

        // 6. dt_proj: (M, D) = dtRaw @ wDtProj^T + bDtProj
        const deltaFull = createEmptyStorageBuffer(d, M * D * 4, true);
        {
            const params = new Uint32Array([M, R, D]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines.linear,
                [pBuf, dtRaw, this.gpuWeights.wDtProj, this.gpuWeights.bDtProj, deltaFull]);
            dispatchKernel(d, this.pipelines.linear, bg, [cdiv(M, 16), cdiv(D, 16), 1]);
        }

        // 7. Selective Scan
        //    Allocate y (B, L, D) and h_cache (2 * B*L*D*N) – first half for h, second for y_partial
        const scanY      = createEmptyStorageBuffer(d, B * L * D * 4,         true);
        const hCache     = createEmptyStorageBuffer(d, 2 * B * L * D * N * 4, true);
        cache.siluOut    = siluOut;
        cache.deltaFull  = deltaFull;
        cache.B_raw      = B_raw;
        cache.C_raw      = C_raw;
        cache.hCache     = hCache;

        {
            const params = new Uint32Array([L, N, D, B]).buffer;
            const pBuf   = createUniformBuffer(d, params);

            // forward_scan pass
            const bg = createBindGroup(d, this.pipelines.scan_fwd,
                [pBuf, siluOut, deltaFull, this.gpuWeights.A_log, B_raw, C_raw,
                 this.gpuWeights.D_vec, scanY, hCache]);
            dispatchKernel(d, this.pipelines.scan_fwd, bg,
                [cdiv(D, 8), cdiv(N, 8), B]);

            // forward_reduce pass (collapses N dim → y)
            const bg2 = createBindGroup(d, this.pipelines.scan_reduce,
                [pBuf, siluOut, deltaFull, this.gpuWeights.A_log, B_raw, C_raw,
                 this.gpuWeights.D_vec, scanY, hCache]);
            dispatchKernel(d, this.pipelines.scan_reduce, bg2,
                [cdiv(L, 64), D, B]);
        }

        // 8. Gate: scanY *= SiLU(zBuf)  – element-wise product
        const siluZ   = createEmptyStorageBuffer(d, M * D * 4, true);
        const gatedOut = createEmptyStorageBuffer(d, M * D * 4, true);
        {
            // SiLU(z)
            const params = new Uint32Array([M * D]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines.silu,
                [pBuf, zBuf, siluZ]);
            dispatchKernel(d, this.pipelines.silu, bg, [cdiv(M * D, 256), 1, 1]);

            // Element-wise multiply scanY * siluZ → gatedOut
            // We encode this as a trivial compute pass using a small inline shader.
            const mulShader = /* wgsl */`
                @group(0) @binding(0) var<storage, read>       a : array<f32>;
                @group(0) @binding(1) var<storage, read>       b : array<f32>;
                @group(0) @binding(2) var<storage, read_write> c : array<f32>;
                @group(0) @binding(3) var<uniform>             n : u32;
                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    let i = gid.x;
                    if (i < n) { c[i] = a[i] * b[i]; }
                }
            `;
            const mulPipeline = createComputePipeline(d, mulShader, 'main');
            const nBuf = createUniformBuffer(d, new Uint32Array([M * D]).buffer);
            const bgMul = createBindGroup(d, mulPipeline,
                [scanY, siluZ, gatedOut, nBuf]);
            dispatchKernel(d, mulPipeline, bgMul, [cdiv(M * D, 256), 1, 1]);
        }

        // 9. out_proj: (M, dModel) = gatedOut @ wOutProj^T + bOutProj
        const outProjOut = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            const params = new Uint32Array([M, D, dModel]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines.linear,
                [pBuf, gatedOut, this.gpuWeights.wOutProj, this.gpuWeights.bOutProj, outProjOut]);
            dispatchKernel(d, this.pipelines.linear, bg, [cdiv(M, 16), cdiv(dModel, 16), 1]);
        }

        // 10. Residual add: output = outProjOut + x
        const output = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            const addShader = /* wgsl */`
                @group(0) @binding(0) var<storage, read>       a : array<f32>;
                @group(0) @binding(1) var<storage, read>       b : array<f32>;
                @group(0) @binding(2) var<storage, read_write> c : array<f32>;
                @group(0) @binding(3) var<uniform>             n : u32;
                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    let i = gid.x;
                    if (i < n) { c[i] = a[i] + b[i]; }
                }
            `;
            const addPipeline = createComputePipeline(d, addShader, 'main');
            const nBuf = createUniformBuffer(d, new Uint32Array([M * dModel]).buffer);
            const bgAdd = createBindGroup(d, addPipeline,
                [outProjOut, xBuf, output, nBuf]);
            dispatchKernel(d, addPipeline, bgAdd, [cdiv(M * dModel, 256), 1, 1]);
        }

        return { output, cache };
    }

    /**
     * Return a list of all parameter GPU buffers (for the optimizer).
     * @returns {Array<{buf: GPUBuffer, numel: number, name: string}>}
     */
    parameters() {
        const { dModel, dState, dConv } = this.config;
        const D = this.dInner;
        const N = dState;
        const K = dConv;
        const R = this.dtRank;

        return [
            { buf: this.gpuWeights.wInProj,   numel: 2 * D * dModel, name: 'wInProj'   },
            { buf: this.gpuWeights.bInProj,   numel: 2 * D,          name: 'bInProj'   },
            { buf: this.gpuWeights.wConv,     numel: D * K,           name: 'wConv'     },
            { buf: this.gpuWeights.bConv,     numel: D,               name: 'bConv'     },
            { buf: this.gpuWeights.wXProj,    numel: (R + 2*N) * D,   name: 'wXProj'   },
            { buf: this.gpuWeights.bXProj,    numel: R + 2 * N,       name: 'bXProj'   },
            { buf: this.gpuWeights.wDtProj,   numel: D * R,           name: 'wDtProj'  },
            { buf: this.gpuWeights.bDtProj,   numel: D,               name: 'bDtProj'  },
            { buf: this.gpuWeights.A_log,     numel: D * N,           name: 'A_log'    },
            { buf: this.gpuWeights.D_vec,     numel: D,               name: 'D_vec'    },
            { buf: this.gpuWeights.wOutProj,  numel: dModel * D,      name: 'wOutProj' },
            { buf: this.gpuWeights.bOutProj,  numel: dModel,          name: 'bOutProj' },
            { buf: this.gpuWeights.normWeight, numel: dModel,          name: 'normWeight'},
        ];
    }

    /**
     * WSLA (Weight-Selective Local Adaptation) mode.
     * Freezes all parameters except the B and C matrices (wXProj slice).
     * This allows rapid local adaptation with minimal compute.
     *
     * @param {boolean} enabled
     */
    setWSLAMode(enabled) {
        this._wslaMode = enabled;
        // Mark which parameters receive gradients
        // (The trainer checks this.getTrainableParams() during backward)
    }

    /**
     * Returns only the trainable parameters under WSLA mode.
     * @returns {Array<{buf: GPUBuffer, numel: number, name: string}>}
     */
    getTrainableParams() {
        if (this._wslaMode) {
            // Only B and C portions of wXProj
            return [
                { buf: this.gpuWeights.wXProj, numel: this.wXProj.length, name: 'wXProj' },
                { buf: this.gpuWeights.bXProj, numel: this.bXProj.length, name: 'bXProj' },
            ];
        }
        return this.parameters();
    }
}
