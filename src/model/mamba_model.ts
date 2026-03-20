/**
 * mamba_model.ts – Full Mamba language model.
 */

import { MambaBlock, BlockCache, BlockParam } from './mamba_block.js';
import {
    createStorageBuffer,
    createEmptyStorageBuffer,
    createUniformBuffer,
    createComputePipeline,
    createBindGroup,
    dispatchKernel,
    readBuffer,
    uploadBuffer,
    cdiv,
} from '../utils/gpu_utils.js';
import { LINEAR_FORWARD_WGSL } from '../kernels/linear_projection.js';
import { ACTIVATIONS_WGSL }    from '../kernels/activations.js';

export interface MambaModelConfig {
  vocabSize: number;
  dModel: number;
  numLayers: number;
  dState?: number;
  dConv?: number;
  expand?: number;
  eosId?: number;
}

export interface ModelForwardResult {
  logits: Float32Array;
  gpuLogits: GPUBuffer;
  caches: BlockCache[];
}

export interface SamplingOptions {
  temperature?: number;
  topK?: number;
  topP?: number;
}

export class MambaModel {
    device: GPUDevice;
    config: Required<MambaModelConfig>;
    gpuEmbedding: GPUBuffer;
    blocks: MambaBlock[];
    gpuFinalNorm: GPUBuffer;
    tiedEmbedding: boolean;
    gpuLMHeadBias: GPUBuffer;
    private _lmHeadPipeline: GPUComputePipeline;
    private _rmsnormPipeline: GPUComputePipeline;
    private _embedPipeline: GPUComputePipeline;
    private _wslaMode = false;

    constructor(device: GPUDevice, config: MambaModelConfig) {
        this.device = device;
        this.config = {
            dState    : 16,
            dConv     : 4,
            expand    : 2,
            eosId     : -1,
            ...config,
        } as Required<MambaModelConfig>;

        const { vocabSize, dModel, numLayers } = this.config;

        const embedData = new Float32Array(vocabSize * dModel);
        const std = 1.0 / Math.sqrt(dModel);
        for (let i = 0; i < embedData.length; i++) {
            const u1 = Math.random(), u2 = Math.random();
            embedData[i] = std * Math.sqrt(-2 * Math.log(u1 + 1e-12)) *
                           Math.cos(2 * Math.PI * u2);
        }
        this.gpuEmbedding = createStorageBuffer(device, embedData, true);

        this.blocks = Array.from({ length: numLayers }, () =>
            new MambaBlock(device, {
                dModel,
                dState  : this.config.dState,
                dConv   : this.config.dConv,
                expand  : this.config.expand,
            })
        );

        const finalNormW = new Float32Array(dModel).fill(1.0);
        this.gpuFinalNorm = createStorageBuffer(device, finalNormW, true);

        this.tiedEmbedding = true;

        this._lmHeadPipeline  = createComputePipeline(device, LINEAR_FORWARD_WGSL, 'linear_forward');
        this._rmsnormPipeline = createComputePipeline(device, ACTIVATIONS_WGSL,    'rmsnorm_forward');

        this.gpuLMHeadBias = createStorageBuffer(device, new Float32Array(vocabSize), true);

        this._embedPipeline = createComputePipeline(device, EMBED_LOOKUP_WGSL, 'embed_lookup');
    }

    embedTokens(tokenIds: number[] | Uint32Array, batch: number, seqLen: number): GPUBuffer {
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

    async forward(tokenIds: number[] | Uint32Array, batch: number, seqLen: number): Promise<ModelForwardResult> {
        const { dModel, vocabSize } = this.config;
        const M = batch * seqLen;

        let hidden = this.embedTokens(tokenIds, batch, seqLen);

        const caches: BlockCache[] = [];
        for (const block of this.blocks) {
            const { output, cache } = block.forward(hidden, batch, seqLen);
            caches.push(cache);
            hidden.destroy();
            hidden = output;
        }

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

        const gpuLogits = createEmptyStorageBuffer(this.device, M * vocabSize * 4, true);
        {
            const params = new Uint32Array([M, dModel, vocabSize]).buffer;
            const pBuf   = createUniformBuffer(this.device, params);
            const weightBuf = this.tiedEmbedding ? this.gpuEmbedding : this.gpuLMHeadBias;
            const bg = createBindGroup(this.device, this._lmHeadPipeline,
                [pBuf, normOut, weightBuf, this.gpuLMHeadBias, gpuLogits]);
            dispatchKernel(this.device, this._lmHeadPipeline, bg,
                [cdiv(M, 16), cdiv(vocabSize, 16), 1]);
        }

        normOut.destroy();
        normInv.destroy();

        const logits = await readBuffer(this.device, gpuLogits, M * vocabSize * 4);

        return { logits, gpuLogits, caches };
    }

    async generate(promptIds: number[], maxNewTokens = 200, samplingOpts: SamplingOptions = {}): Promise<number[]> {
        const { temperature = 1.0, topK = 50, topP = 0.9 } = samplingOpts;
        const { vocabSize } = this.config;

        let ids = [...promptIds];

        for (let step = 0; step < maxNewTokens; step++) {
            const { logits } = await this.forward(
                new Uint32Array(ids), 1, ids.length
            );
            const lastLogits = logits.slice((ids.length - 1) * vocabSize, ids.length * vocabSize);

            const nextId = sampleToken(lastLogits, { temperature, topK, topP });
            ids.push(nextId);

            if (nextId === this.config.eosId) break;
        }

        return ids;
    }

    parameters(): BlockParam[] {
        const params: BlockParam[] = [];

        params.push({
            buf  : this.gpuEmbedding,
            numel: this.config.vocabSize * this.config.dModel,
            name : 'embedding',
        });

        for (let i = 0; i < this.blocks.length; i++) {
            for (const p of this.blocks[i]!.parameters()) {
                params.push({ ...p, name: `block${i}.${p.name}` });
            }
        }

        params.push({
            buf  : this.gpuFinalNorm,
            numel: this.config.dModel,
            name : 'final_norm',
        });

        return params;
    }

    setWSLAMode(enabled: boolean): void {
        for (const block of this.blocks) block.setWSLAMode(enabled);
        this._wslaMode = enabled;
    }

    /**
     * Serialise all model parameters to an ArrayBuffer.
     *
     * Binary format:
     *   [0..3]   magic  : uint32  = 0x4D424A53 ('MBJS')
     *   [4..7]   version: uint32  = 1
     *   [8..11]  nParams: uint32
     *   [12 .. 12+4*nParams-1]  numel[i]: uint32 for each parameter i
     *   [12+4*nParams ..]       float32 data for each parameter, concatenated
     *
     * Save the returned buffer to a file or IndexedDB and reload it with
     * `loadWeights()` to resume from a checkpoint.
     */
    async exportWeights(): Promise<ArrayBuffer> {
        const params = this.parameters();
        const nParams = params.length;

        // Read all GPU buffers into CPU Float32Arrays
        const arrays: Float32Array[] = await Promise.all(
            params.map(p => readBuffer(this.device, p.buf, p.numel * 4))
        );

        // Calculate total byte size: header + numel table + all float data
        const headerBytes = 4 + 4 + 4 + nParams * 4;  // magic + version + nParams + numel[]
        const dataBytes   = arrays.reduce((acc, a) => acc + a.byteLength, 0);
        const out         = new ArrayBuffer(headerBytes + dataBytes);
        const view        = new DataView(out);

        let offset = 0;
        view.setUint32(offset, 0x4D424A53, true); offset += 4;  // magic 'MBJS'
        view.setUint32(offset, 1,           true); offset += 4;  // version
        view.setUint32(offset, nParams,     true); offset += 4;  // nParams

        for (const p of params) {
            view.setUint32(offset, p.numel, true);
            offset += 4;
        }

        for (const arr of arrays) {
            new Float32Array(out, offset, arr.length).set(arr);
            offset += arr.byteLength;
        }

        return out;
    }

    /**
     * Load model parameters from an ArrayBuffer previously produced by
     * `exportWeights()`.  The parameter count and element counts must match
     * the current model configuration exactly.
     *
     * @throws {Error} if the magic number, version, or parameter layout do
     *                 not match the current model.
     */
    async loadWeights(buffer: ArrayBuffer): Promise<void> {
        const view    = new DataView(buffer);
        let offset    = 0;

        const magic   = view.getUint32(offset, true); offset += 4;
        if (magic !== 0x4D424A53) {
            throw new Error(
                'Invalid weight file: bad magic number. ' +
                'Ensure the file was exported by MambaModel.exportWeights().'
            );
        }

        const version = view.getUint32(offset, true); offset += 4;
        if (version !== 1) {
            throw new Error(`Unsupported weight file version: ${version}. Expected version 1.`);
        }

        const nParams = view.getUint32(offset, true); offset += 4;
        const params  = this.parameters();

        if (nParams !== params.length) {
            throw new Error(
                `Weight file has ${nParams} parameters but this model has ${params.length}. ` +
                'Ensure the model configuration matches the one used when exporting.'
            );
        }

        const numels: number[] = [];
        for (let i = 0; i < nParams; i++) {
            numels.push(view.getUint32(offset, true));
            offset += 4;
        }

        for (let i = 0; i < nParams; i++) {
            // i is guaranteed in-bounds: nParams === params.length was verified above
            const p      = params[i]!;
            const numel  = numels[i]!;
            if (numel !== p.numel) {
                throw new Error(
                    `Parameter ${i} ("${p.name}") size mismatch: ` +
                    `file has ${numel} elements, model expects ${p.numel}.`
                );
            }

            const slice = new Float32Array(buffer, offset, p.numel);
            uploadBuffer(this.device, p.buf, slice);
            offset += p.numel * 4;
        }
    }
}

const EMBED_LOOKUP_WGSL: string = /* wgsl */`
struct EmbedParams {
    num_tokens : u32,
    d_model    : u32,
};

@group(0) @binding(0) var<uniform>            params  : EmbedParams;
@group(0) @binding(1) var<storage, read>      ids     : array<u32>;
@group(0) @binding(2) var<storage, read>      table   : array<f32>;
@group(0) @binding(3) var<storage, read_write> out    : array<f32>;

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

function sampleToken(logits: Float32Array, { temperature = 1.0, topK = 50, topP = 0.9 } = {}): number {
    const n = logits.length;

    const scaled = new Float32Array(n);
    for (let i = 0; i < n; i++) scaled[i] = logits[i]! / Math.max(temperature, 1e-7);

    let maxL = -Infinity;
    for (let i = 0; i < n; i++) if (scaled[i]! > maxL) maxL = scaled[i]!;
    let sumE = 0;
    const exps = new Float32Array(n);
    for (let i = 0; i < n; i++) { exps[i] = Math.exp(scaled[i]! - maxL); sumE += exps[i]!; }

    const indices = Array.from({ length: n }, (_, i) => i)
        .sort((a, b) => exps[b]! - exps[a]!);

    const topKIndices = indices.slice(0, topK);

    let cumSum = 0;
    const nucleus: number[] = [];
    for (const idx of topKIndices) {
        cumSum += exps[idx]! / sumE;
        nucleus.push(idx);
        if (cumSum >= topP) break;
    }

    let nucleusSum = 0;
    for (const idx of nucleus) nucleusSum += exps[idx]!;
    const threshold = Math.random() * nucleusSum;
    let acc = 0;
    for (const idx of nucleus) {
        acc += exps[idx]!;
        if (acc >= threshold) return idx;
    }
    return nucleus[nucleus.length - 1]!;
}
