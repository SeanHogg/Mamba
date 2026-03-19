/**
 * quantization.js – FP16 and Int8 quantization utilities.
 *
 * MambaCode.js supports two quantization modes to reduce VRAM usage:
 *   • FP16  – weights stored as 16-bit floats (halves memory vs FP32)
 *   • Int8  – non-critical activations quantized to signed 8-bit integers
 *
 * All quantization/dequantization happens in JavaScript; the GPU kernels
 * always operate on FP32 tensors internally (dequantized on upload).
 */

// ─── FP16 Utilities ──────────────────────────────────────────────────────────

/**
 * Convert a 32-bit float to a 16-bit IEEE 754 float (represented as Uint16).
 * Uses bit manipulation to avoid the need for a Float16Array (not in spec yet).
 *
 * @param {number} val  – 32-bit float
 * @returns {number}    – 16-bit float packed as an integer (0–65535)
 */
export function floatToFp16(val) {
    const buf = new ArrayBuffer(4);
    const f32 = new Float32Array(buf);
    const u32 = new Uint32Array(buf);
    f32[0] = val;
    const bits = u32[0];

    const sign     = (bits >>> 31) & 0x1;
    const exponent = (bits >>> 23) & 0xFF;
    const mantissa =  bits         & 0x7FFFFF;

    if (exponent === 255) {
        // Inf / NaN
        return (sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0);
    }

    const expAdj = exponent - 127 + 15;  // re-bias from 127 to 15

    if (expAdj >= 31) {
        // Overflow → Inf
        return (sign << 15) | 0x7C00;
    }

    if (expAdj <= 0) {
        // Underflow or denormal
        if (expAdj < -10) { return sign << 15; }  // flush to zero
        const shift = 14 - expAdj;
        return (sign << 15) | ((mantissa | 0x800000) >> shift);
    }

    return (sign << 15) | (expAdj << 10) | (mantissa >> 13);
}

/**
 * Convert a 16-bit FP16 integer to a 32-bit float.
 *
 * @param {number} val – Uint16 representation of an FP16 value
 * @returns {number}   – JavaScript number (float64, but semantically float32)
 */
export function fp16ToFloat(val) {
    const sign     = (val >>> 15) & 0x1;
    const exponent = (val >>> 10) & 0x1F;
    const mantissa =  val         & 0x3FF;

    if (exponent === 0) {
        // Denormal or zero
        const f = mantissa / 1024.0;
        return sign ? -f : f;
    }

    if (exponent === 31) {
        // Inf / NaN
        return sign ? -Infinity : (mantissa ? NaN : Infinity);
    }

    const expUnbiased = exponent - 15;
    const f = (1 + mantissa / 1024.0) * Math.pow(2, expUnbiased);
    return sign ? -f : f;
}

/**
 * Quantize a Float32Array to FP16 (stored as Uint16Array).
 *
 * @param {Float32Array} f32
 * @returns {Uint16Array}
 */
export function quantizeFp16(f32) {
    const out = new Uint16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
        out[i] = floatToFp16(f32[i]);
    }
    return out;
}

/**
 * Dequantize a Uint16Array (FP16) back to Float32Array.
 *
 * @param {Uint16Array} fp16
 * @returns {Float32Array}
 */
export function dequantizeFp16(fp16) {
    const out = new Float32Array(fp16.length);
    for (let i = 0; i < fp16.length; i++) {
        out[i] = fp16ToFloat(fp16[i]);
    }
    return out;
}

// ─── Int8 Quantization ───────────────────────────────────────────────────────

/**
 * Symmetric per-tensor Int8 quantization.
 * Quantization: q = round(x / scale),  scale = max(|x|) / 127
 *
 * @param {Float32Array} f32
 * @returns {{ data: Int8Array, scale: number }}
 */
export function quantizeInt8(f32) {
    let maxAbs = 0;
    for (let i = 0; i < f32.length; i++) {
        const a = Math.abs(f32[i]);
        if (a > maxAbs) maxAbs = a;
    }

    const scale = maxAbs / 127.0 || 1.0;  // avoid division by zero
    const data  = new Int8Array(f32.length);

    for (let i = 0; i < f32.length; i++) {
        data[i] = Math.max(-128, Math.min(127, Math.round(f32[i] / scale)));
    }

    return { data, scale };
}

/**
 * Dequantize an Int8Array back to Float32Array.
 *
 * @param {Int8Array} int8
 * @param {number}    scale
 * @returns {Float32Array}
 */
export function dequantizeInt8(int8, scale) {
    const out = new Float32Array(int8.length);
    for (let i = 0; i < int8.length; i++) {
        out[i] = int8[i] * scale;
    }
    return out;
}

/**
 * Per-channel Int8 quantization (useful for weight matrices).
 * Each output channel gets its own scale factor for better accuracy.
 *
 * @param {Float32Array} f32          – Flat weight tensor, row-major
 * @param {number}       numChannels  – Number of output channels (rows)
 * @returns {{ data: Int8Array, scales: Float32Array }}
 */
export function quantizeInt8PerChannel(f32, numChannels) {
    const channelSize = f32.length / numChannels;
    const scales = new Float32Array(numChannels);
    const data   = new Int8Array(f32.length);

    for (let c = 0; c < numChannels; c++) {
        let maxAbs = 0;
        const base = c * channelSize;
        for (let j = 0; j < channelSize; j++) {
            const a = Math.abs(f32[base + j]);
            if (a > maxAbs) maxAbs = a;
        }
        scales[c] = maxAbs / 127.0 || 1.0;
        for (let j = 0; j < channelSize; j++) {
            data[base + j] = Math.max(-128, Math.min(127,
                Math.round(f32[base + j] / scales[c])
            ));
        }
    }

    return { data, scales };
}

/**
 * Dequantize per-channel Int8 data.
 *
 * @param {Int8Array}    int8
 * @param {Float32Array} scales
 * @param {number}       numChannels
 * @returns {Float32Array}
 */
export function dequantizeInt8PerChannel(int8, scales, numChannels) {
    const channelSize = int8.length / numChannels;
    const out = new Float32Array(int8.length);

    for (let c = 0; c < numChannels; c++) {
        const base = c * channelSize;
        for (let j = 0; j < channelSize; j++) {
            out[base + j] = int8[base + j] * scales[c];
        }
    }

    return out;
}

/**
 * Estimate memory usage for a weight tensor under different precisions.
 *
 * @param {number} numElements
 * @returns {{ fp32: number, fp16: number, int8: number }}  – bytes
 */
export function estimateMemory(numElements) {
    return {
        fp32: numElements * 4,
        fp16: numElements * 2,
        int8: numElements * 1,
    };
}
