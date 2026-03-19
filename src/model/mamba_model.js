/**
 * mamba_model.js – Full Mamba language model.
 *
 * Architecture (matches Qwen3.5-Coder-0.8B-style Mamba):
 *
 *   Token IDs ──► Embedding ──► [MambaBlock × numLayers] ──► RMSNorm ──► LM Head
 *
 * The LM Head is a linear projection from dModel → vocabSize.
 * All computations run on WebGPU via the kernels in src/kernels/.
 */

import { MambaBlock } from './mamba_block.js';
import {
    createStorageBuffer,
    createEmptyStorageBuffer,
    createUniformBuffer,
    createComputePipeline,
    createBindGroup,
    dispatchKernel,
    readBuffer,
    cdiv,
} from '../utils/gpu_utils.js';
import { LINEAR_FORWARD_WGSL } from '../kernels/linear_projection.js';
import { ACTIVATIONS_WGSL }    from '../kernels/activations.js';

/**
 * @typedef {Object} MambaModelConfig
 * @property {number} vocabSize   – vocabulary size (Qwen3.5-Coder: 151936)
 * @property {number} dModel      – model (embedding) dimension
 * @property {number} numLayers   – number of Mamba blocks
 * @property {number} [dState]    – SSM state dimension (default 16)
 * @property {number} [dConv]     – conv kernel size (default 4)
 * @property {number} [expand]    – inner-dim expansion factor (default 2)
 */

export class MambaModel {
    /**
     * @param {GPUDevice}       device
     * @param {MambaModelConfig} config
     */
    constructor(device, config) {
        this.device = device;
        this.config = {
            dState    : 16,
            dConv     : 4,
            expand    : 2,
            ...config,
        };

        const { vocabSize, dModel, numLayers } = this.config;

        // Token embedding table: (vocabSize, dModel)
        const embedData = new Float32Array(vocabSize * dModel);
        // Xavier-style initialisation
        const std = 1.0 / Math.sqrt(dModel);
        for (let i = 0; i < embedData.length; i++) {
            const u1 = Math.random(), u2 = Math.random();
            embedData[i] = std * Math.sqrt(-2 * Math.log(u1 + 1e-12)) *
                           Math.cos(2 * Math.PI * u2);
        }
        this.gpuEmbedding = createStorageBuffer(device, embedData, true);

        // Stacked Mamba blocks
        this.blocks = Array.from({ length: numLayers }, () =>
            new MambaBlock(device, {
                dModel,
                dState  : this.config.dState,
                dConv   : this.config.dConv,
                expand  : this.config.expand,
            })
        );

        // Final RMSNorm
        const finalNormW = new Float32Array(dModel).fill(1.0);
        this.gpuFinalNorm = createStorageBuffer(device, finalNormW, true);

        // LM Head: (vocabSize, dModel) – tied to embedding by default
        // We share the embedding weight (weight tying saves memory).
        this.tiedEmbedding = true;

        // Compile pipelines
        this._lmHeadPipeline  = createComputePipeline(device, LINEAR_FORWARD_WGSL, 'linear_forward');
        this._rmsnormPipeline = createComputePipeline(device, ACTIVATIONS_WGSL,    'rmsnorm_forward');

        // LM Head bias (zeroed)
        this.gpuLMHeadBias = createStorageBuffer(device, new Float32Array(vocabSize), true);

        // Embedding lookup pipeline (gather rows)
        this._embedPipeline = createComputePipeline(device, EMBED_LOOKUP_WGSL, 'embed_lookup');
    }

    // ─── Embedding lookup ─────────────────────────────────────────────────────

    /**
     * Look up token embeddings.
     *
     * @param {Int32Array|Uint32Array} tokenIds  – (batch * seqLen,)
     * @param {number} batch
     * @param {number} seqLen
     * @returns {GPUBuffer}  – (batch * seqLen, dModel)
     */
    embedTokens(tokenIds, batch, seqLen) {
        const { dModel } = this.config;
        const M = batch * seqLen;

        const idsBuf  = createStorageBuffer(this.device,
            tokenIds instanceof Uint32Array ? tokenIds : new Uint32Array(tokenIds), false);
        const outBuf  = createEmptyStorageBuffer(this.device, M * dModel * 4, true);

        const params  = new Uint32Array([M, dModel]).buffer;
        const pBuf    = createUniformBuffer(this.device, params);

        const bg = createBindGroup(this.device, this._embedPipeline,
            [pBuf, idsBuf, this.gpuEmbedding, outBuf]);
        dispatchKernel(this.device, this._embedPipeline, bg, [cdiv(M, 64), 1, 1]);

        idsBuf.destroy();
        pBuf.destroy();
        return outBuf;
    }

    // ─── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Full model forward pass.
     *
     * @param {number[]|Uint32Array} tokenIds  – (batch * seqLen,) flat
     * @param {number}  batch
     * @param {number}  seqLen
     * @returns {Promise<{ logits: Float32Array, gpuLogits: GPUBuffer }>}
     *   logits   – CPU Float32Array of shape (batch * seqLen, vocabSize)
     *   gpuLogits – GPU buffer (same data, for chained backward)
     */
    async forward(tokenIds, batch, seqLen) {
        const { dModel, vocabSize } = this.config;
        const M = batch * seqLen;

        // 1. Token embedding lookup
        let hidden = this.embedTokens(tokenIds, batch, seqLen);

        // 2. Mamba blocks
        const caches = [];
        for (const block of this.blocks) {
            const { output, cache } = block.forward(hidden, batch, seqLen);
            caches.push(cache);
            hidden.destroy();
            hidden = output;
        }

        // 3. Final RMSNorm
        const normOut = createEmptyStorageBuffer(this.device, M * dModel * 4, true);
        const normInv = createEmptyStorageBuffer(this.device, M * 4,          false);
        {
            const params = new ArrayBuffer(16);
            new Uint32Array(params, 0, 2).set([M, dModel]);
            new Float32Array(params, 8, 1).set([1e-6]);
            const pBuf = createUniformBuffer(this.device, params);
            const bg = createBindGroup(this.device, this._rmsnormPipeline,
                [pBuf, hidden, this.gpuFinalNorm, normOut, normInv]);
            dispatchKernel(this.device, this._rmsnormPipeline, bg, [cdiv(M, 64), 1, 1]);
        }

        // 4. LM Head: (M, vocabSize) = normOut @ embedding^T + bias
        const gpuLogits = createEmptyStorageBuffer(this.device, M * vocabSize * 4, true);
        {
            const params = new Uint32Array([M, dModel, vocabSize]).buffer;
            const pBuf   = createUniformBuffer(this.device, params);
            const weightBuf = this.tiedEmbedding ? this.gpuEmbedding : this.gpuLMHeadWeight;
            const bg = createBindGroup(this.device, this._lmHeadPipeline,
                [pBuf, normOut, weightBuf, this.gpuLMHeadBias, gpuLogits]);
            dispatchKernel(this.device, this._lmHeadPipeline, bg,
                [cdiv(M, 16), cdiv(vocabSize, 16), 1]);
        }

        normOut.destroy();
        normInv.destroy();

        // 5. Read back logits to CPU
        const logits = await readBuffer(this.device, gpuLogits, M * vocabSize * 4);

        return { logits, gpuLogits, caches };
    }

    /**
     * Greedy / top-k / temperature-sampled autoregressive generation.
     *
     * @param {number[]} promptIds  – starting token IDs
     * @param {number}   maxNewTokens
     * @param {{ temperature?: number, topK?: number, topP?: number }} [samplingOpts]
     * @returns {Promise<number[]>}  – full sequence (prompt + generated)
     */
    async generate(promptIds, maxNewTokens = 200, samplingOpts = {}) {
        const { temperature = 1.0, topK = 50, topP = 0.9 } = samplingOpts;
        const { vocabSize } = this.config;

        let ids = [...promptIds];

        for (let step = 0; step < maxNewTokens; step++) {
            // Use the full context each step (linear cost with Mamba – no kv-cache needed)
            const { logits } = await this.forward(
                new Uint32Array(ids), 1, ids.length
            );
            // Get logits for the last position
            const lastLogits = logits.slice((ids.length - 1) * vocabSize, ids.length * vocabSize);

            const nextId = sampleToken(lastLogits, { temperature, topK, topP });
            ids.push(nextId);

            // Stop on EOS
            if (nextId === this.config.eosId) break;
        }

        return ids;
    }

    /**
     * Collect all trainable parameters across all blocks.
     * @returns {Array<{buf: GPUBuffer, numel: number, name: string}>}
     */
    parameters() {
        const params = [];

        // Embedding
        params.push({
            buf  : this.gpuEmbedding,
            numel: this.config.vocabSize * this.config.dModel,
            name : 'embedding',
        });

        // Blocks
        for (let i = 0; i < this.blocks.length; i++) {
            for (const p of this.blocks[i].parameters()) {
                params.push({ ...p, name: `block${i}.${p.name}` });
            }
        }

        // Final norm
        params.push({
            buf  : this.gpuFinalNorm,
            numel: this.config.dModel,
            name : 'final_norm',
        });

        return params;
    }

    /**
     * Enable WSLA (selective fine-tuning of B and C only) across all blocks.
     * @param {boolean} enabled
     */
    setWSLAMode(enabled) {
        for (const block of this.blocks) block.setWSLAMode(enabled);
        this._wslaMode = enabled;
    }
}

// ─── Embedding lookup WGSL kernel ────────────────────────────────────────────

const EMBED_LOOKUP_WGSL = /* wgsl */`
struct EmbedParams {
    num_tokens : u32,
    d_model    : u32,
};

@group(0) @binding(0) var<uniform>            params  : EmbedParams;
@group(0) @binding(1) var<storage, read>      ids     : array<u32>;
@group(0) @binding(2) var<storage, read>      table   : array<f32>;  // (V, D)
@group(0) @binding(3) var<storage, read_write> out    : array<f32>;  // (T, D)

@compute @workgroup_size(64, 1, 1)
fn embed_lookup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;
    if (token_idx >= params.num_tokens) { return; }

    let D   = params.d_model;
    let tok = ids[token_idx];
    let src = tok * D;
    let dst = token_idx * D;

    for (var i: u32 = 0u; i < D; i = i + 1u) {
        out[dst + i] = table[src + i];
    }
}
`;

// ─── Sampling helper ──────────────────────────────────────────────────────────

/**
 * Sample a token from logits using temperature + top-k + nucleus (top-p).
 *
 * @param {Float32Array} logits
 * @param {{ temperature?: number, topK?: number, topP?: number }} opts
 * @returns {number}
 */
function sampleToken(logits, { temperature = 1.0, topK = 50, topP = 0.9 } = {}) {
    const n = logits.length;

    // Apply temperature
    const scaled = new Float32Array(n);
    for (let i = 0; i < n; i++) scaled[i] = logits[i] / Math.max(temperature, 1e-7);

    // Softmax
    let maxL = -Infinity;
    for (let i = 0; i < n; i++) if (scaled[i] > maxL) maxL = scaled[i];
    let sumE = 0;
    const exps = new Float32Array(n);
    for (let i = 0; i < n; i++) { exps[i] = Math.exp(scaled[i] - maxL); sumE += exps[i]; }

    // Sort indices by probability (descending)
    const indices = Array.from({ length: n }, (_, i) => i)
        .sort((a, b) => exps[b] - exps[a]);

    // Top-K filter
    const topKIndices = indices.slice(0, topK);

    // Nucleus (top-p) filter
    let cumSum = 0;
    const nucleus = [];
    for (const idx of topKIndices) {
        cumSum += exps[idx] / sumE;
        nucleus.push(idx);
        if (cumSum >= topP) break;
    }

    // Sample from nucleus
    let nucleusSum = 0;
    for (const idx of nucleus) nucleusSum += exps[idx];
    const threshold = Math.random() * nucleusSum;
    let acc = 0;
    for (const idx of nucleus) {
        acc += exps[idx];
        if (acc >= threshold) return idx;
    }
    return nucleus[nucleus.length - 1];
}
