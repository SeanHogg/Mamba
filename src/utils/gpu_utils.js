/**
 * gpu_utils.js – WebGPU device management and buffer helpers.
 *
 * Provides thin, consistent wrappers around the WebGPU API so that
 * the rest of MambaCode.js never calls navigator.gpu directly.
 */

/**
 * Initialise WebGPU and return the { device, adapter } pair.
 *
 * @param {{ powerPreference?: 'high-performance'|'low-power' }} [opts]
 * @returns {Promise<{ device: GPUDevice, adapter: GPUAdapter }>}
 */
export async function initWebGPU(opts = {}) {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
        throw new Error(
            'WebGPU is not available in this environment. ' +
            'Use Chrome 113+, Edge 113+, or Firefox Nightly with WebGPU enabled.'
        );
    }

    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: opts.powerPreference ?? 'high-performance',
    });

    if (!adapter) {
        throw new Error('Failed to acquire a GPUAdapter. Your GPU may not support WebGPU.');
    }

    // Request a device, capping requested limits to what the adapter supports.
    const adapterLimits = adapter.limits;
    const requested3GB  = 3 * 1024 * 1024 * 1024;
    const device = await adapter.requestDevice({
        requiredLimits: {
            maxBufferSize: Math.min(
                requested3GB,
                adapterLimits.maxBufferSize
            ),
            maxStorageBufferBindingSize: Math.min(
                requested3GB,
                adapterLimits.maxStorageBufferBindingSize
            ),
            maxComputeInvocationsPerWorkgroup: Math.min(
                256,
                adapterLimits.maxComputeInvocationsPerWorkgroup
            ),
        },
    });

    device.lost.then((info) => {
        console.error('WebGPU device lost:', info.message);
    });

    return { device, adapter };
}

// ─── Buffer factory helpers ───────────────────────────────────────────────────

const UNIFORM = GPUBufferUsage?.UNIFORM  ?? 0x40;
const STORAGE = GPUBufferUsage?.STORAGE  ?? 0x80;
const COPY_SRC = GPUBufferUsage?.COPY_SRC ?? 0x04;
const COPY_DST = GPUBufferUsage?.COPY_DST ?? 0x08;
const MAP_READ = GPUBufferUsage?.MAP_READ ?? 0x01;

/**
 * Create a GPU storage buffer pre-filled with Float32 data.
 *
 * @param {GPUDevice} device
 * @param {Float32Array|number[]} data
 * @param {boolean} [readable=false]  Also attach COPY_SRC so it can be read back.
 * @returns {GPUBuffer}
 */
export function createStorageBuffer(device, data, readable = false) {
    const arr    = data instanceof Float32Array ? data : new Float32Array(data);
    const usage  = STORAGE | COPY_DST | (readable ? COPY_SRC : 0);
    const buffer = device.createBuffer({ size: arr.byteLength, usage, mappedAtCreation: true });
    new Float32Array(buffer.getMappedRange()).set(arr);
    buffer.unmap();
    return buffer;
}

/**
 * Create a GPU storage buffer of `size` bytes, zeroed.
 *
 * @param {GPUDevice} device
 * @param {number} byteSize
 * @param {boolean} [readable=false]
 * @returns {GPUBuffer}
 */
export function createEmptyStorageBuffer(device, byteSize, readable = false) {
    const usage = STORAGE | COPY_DST | (readable ? COPY_SRC : 0);
    return device.createBuffer({ size: byteSize, usage });
}

/**
 * Create a uniform buffer for a plain-old-data struct.
 * The caller must supply a correctly-packed ArrayBuffer / TypedArray.
 *
 * @param {GPUDevice} device
 * @param {ArrayBuffer|TypedArray} data
 * @returns {GPUBuffer}
 */
export function createUniformBuffer(device, data) {
    const bytes  = ArrayBuffer.isView(data) ? data.buffer : data;
    const buffer = device.createBuffer({
        size  : bytes.byteLength,
        usage : UNIFORM | COPY_DST,
        mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(bytes));
    buffer.unmap();
    return buffer;
}

/**
 * Read back a GPU storage buffer to a Float32Array (async, for debugging/eval).
 *
 * @param {GPUDevice}  device
 * @param {GPUBuffer}  srcBuffer   Must have COPY_SRC usage.
 * @param {number}     byteSize
 * @returns {Promise<Float32Array>}
 */
export async function readBuffer(device, srcBuffer, byteSize) {
    const stagingBuffer = device.createBuffer({
        size  : byteSize,
        usage : MAP_READ | COPY_DST,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(srcBuffer, 0, stagingBuffer, 0, byteSize);
    device.queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode?.READ ?? 0x01);
    const result = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    return result;
}

/**
 * Upload a Float32Array to an existing GPU buffer.
 *
 * @param {GPUDevice}    device
 * @param {GPUBuffer}    buffer   Must have COPY_DST usage.
 * @param {Float32Array} data
 * @param {number}       [byteOffset=0]
 */
export function uploadBuffer(device, buffer, data, byteOffset = 0) {
    device.queue.writeBuffer(buffer, byteOffset, data);
}

// ─── Pipeline / Shader helpers ────────────────────────────────────────────────

/**
 * Compile a WGSL compute shader and return a GPUComputePipeline.
 *
 * @param {GPUDevice} device
 * @param {string}    wgslSource
 * @param {string}    entryPoint
 * @returns {GPUComputePipeline}
 */
export function createComputePipeline(device, wgslSource, entryPoint) {
    const shaderModule = device.createShaderModule({ code: wgslSource });
    return device.createComputePipeline({
        layout : 'auto',
        compute: { module: shaderModule, entryPoint },
    });
}

/**
 * Build a GPUBindGroup from an array of GPUBuffer bindings.
 *
 * @param {GPUDevice}           device
 * @param {GPUComputePipeline}  pipeline
 * @param {GPUBuffer[]}         buffers   Ordered list matching @binding(i).
 * @param {number}              [groupIndex=0]
 * @returns {GPUBindGroup}
 */
export function createBindGroup(device, pipeline, buffers, groupIndex = 0) {
    const entries = buffers.map((buf, i) => ({
        binding : i,
        resource: { buffer: buf },
    }));
    return device.createBindGroup({
        layout : pipeline.getBindGroupLayout(groupIndex),
        entries,
    });
}

/**
 * Dispatch a compute pipeline synchronously (encodes + submits in one call).
 *
 * @param {GPUDevice}           device
 * @param {GPUComputePipeline}  pipeline
 * @param {GPUBindGroup}        bindGroup
 * @param {[number, number, number]} workgroups  [x, y, z]
 */
export function dispatchKernel(device, pipeline, bindGroup, workgroups) {
    const encoder = device.createCommandEncoder();
    const pass    = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
}

/**
 * Ceil-divide helper: Math.ceil(a / b) in integer arithmetic.
 *
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
export function cdiv(a, b) {
    return Math.ceil(a / b);
}
