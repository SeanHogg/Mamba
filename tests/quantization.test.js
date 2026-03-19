/**
 * tests/quantization.test.js
 * Unit tests for the quantization utilities (no GPU required).
 */

import {
    floatToFp16,
    fp16ToFloat,
    quantizeFp16,
    dequantizeFp16,
    quantizeInt8,
    dequantizeInt8,
    quantizeInt8PerChannel,
    dequantizeInt8PerChannel,
    estimateMemory,
} from '../src/utils/quantization.js';

// ── FP16 round-trip ───────────────────────────────────────────────────────────

test('floatToFp16 and fp16ToFloat are inverse for ordinary values', () => {
    const cases = [0.0, 1.0, -1.0, 0.5, -0.5, 3.14, -100.0, 0.001];
    for (const v of cases) {
        const roundTripped = fp16ToFloat(floatToFp16(v));
        // FP16 has ~3 decimal digits of precision
        expect(Math.abs(roundTripped - v)).toBeLessThan(Math.abs(v) * 0.01 + 1e-4);
    }
});

test('fp16ToFloat returns Infinity for overflow', () => {
    expect(fp16ToFloat(0x7C00)).toBe(Infinity);
    expect(fp16ToFloat(0xFC00)).toBe(-Infinity);
});

test('floatToFp16 clamps overflow to Inf', () => {
    const huge = 1e10;
    const bits = floatToFp16(huge);
    expect(fp16ToFloat(bits)).toBe(Infinity);
});

test('quantizeFp16 / dequantizeFp16 round-trip', () => {
    const data = new Float32Array([1.0, 2.5, -0.75, 0.0, 100.0, -100.0]);
    const fp16 = quantizeFp16(data);
    const back = dequantizeFp16(fp16);

    expect(back.length).toBe(data.length);
    for (let i = 0; i < data.length; i++) {
        const rel = Math.abs(data[i]) > 1 ? Math.abs(data[i]) * 0.02 : 0.02;
        expect(Math.abs(back[i] - data[i])).toBeLessThan(rel + 1e-3);
    }
});

// ── Int8 quantization ─────────────────────────────────────────────────────────

test('quantizeInt8 / dequantizeInt8 round-trip', () => {
    const data = new Float32Array([0.1, -0.2, 0.5, -0.5, 1.0, -1.0]);
    const { data: q, scale } = quantizeInt8(data);
    const back = dequantizeInt8(q, scale);

    expect(scale).toBeGreaterThan(0);
    expect(back.length).toBe(data.length);

    for (let i = 0; i < data.length; i++) {
        // Int8 has ~1/127 relative error
        expect(Math.abs(back[i] - data[i])).toBeLessThan(0.02);
    }
});

test('quantizeInt8 clamps to [-128, 127]', () => {
    const data = new Float32Array([1.0, -1.0, 0.5]);
    const { data: q } = quantizeInt8(data);
    for (let i = 0; i < q.length; i++) {
        expect(q[i]).toBeGreaterThanOrEqual(-128);
        expect(q[i]).toBeLessThanOrEqual(127);
    }
});

test('quantizeInt8PerChannel produces per-channel scales', () => {
    const data   = new Float32Array([10, 20, 30, 0.1, 0.2, 0.3]);  // 2 channels, 3 elements each
    const { data: q, scales } = quantizeInt8PerChannel(data, 2);
    expect(scales.length).toBe(2);
    expect(scales[0]).toBeGreaterThan(scales[1]);   // first channel has larger range
});

test('quantizeInt8PerChannel / dequantizeInt8PerChannel round-trip', () => {
    const data = new Float32Array([1, 2, 3, 4, -1, -2, -3, -4]);  // 2 channels
    const { data: q, scales } = quantizeInt8PerChannel(data, 2);
    const back = dequantizeInt8PerChannel(q, scales, 2);
    for (let i = 0; i < data.length; i++) {
        expect(Math.abs(back[i] - data[i])).toBeLessThan(0.1);
    }
});

test('quantizeInt8 handles all-zeros', () => {
    const data = new Float32Array([0, 0, 0]);
    const { data: q, scale } = quantizeInt8(data);
    expect(scale).toBe(1.0);
    expect([...q]).toEqual([0, 0, 0]);
});

// ── Memory estimation ─────────────────────────────────────────────────────────

test('estimateMemory returns correct byte counts', () => {
    const { fp32, fp16, int8 } = estimateMemory(100);
    expect(fp32).toBe(400);
    expect(fp16).toBe(200);
    expect(int8).toBe(100);
});
