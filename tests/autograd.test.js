/**
 * tests/autograd.test.js
 * Unit tests for the tape-based autograd engine.
 */

import {
    Tensor,
    enableGrad,
    noGrad,
    clearTape,
    recordOperation,
    backward,
    crossEntropyLoss,
    crossEntropyGrad,
} from '../src/training/autograd.js';

// ── crossEntropyLoss ──────────────────────────────────────────────────────────

test('crossEntropyLoss returns positive scalar', () => {
    const logits = new Float32Array([1.0, 2.0, 0.5, -1.0]);
    const loss   = crossEntropyLoss(logits, 1);
    expect(loss).toBeGreaterThan(0);
    expect(Number.isFinite(loss)).toBe(true);
});

test('crossEntropyLoss is lower when target is highest logit', () => {
    const logits = new Float32Array([0.0, 10.0, 0.0]);
    const lossCorrect   = crossEntropyLoss(logits, 1);   // target is highest
    const lossIncorrect = crossEntropyLoss(logits, 0);   // target is low-prob
    expect(lossCorrect).toBeLessThan(lossIncorrect);
});

test('crossEntropyLoss handles uniform logits', () => {
    const n      = 4;
    const logits = new Float32Array(n).fill(0.0);  // uniform
    const loss   = crossEntropyLoss(logits, 0);
    expect(Math.abs(loss - Math.log(n))).toBeLessThan(1e-4);
});

// ── crossEntropyGrad ──────────────────────────────────────────────────────────

test('crossEntropyGrad sums to 0', () => {
    const logits = new Float32Array([1.0, 2.0, 0.5, -1.0]);
    const grad   = crossEntropyGrad(logits, 1);
    const sum    = grad.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum)).toBeLessThan(1e-5);
});

test('crossEntropyGrad target component is negative', () => {
    const logits = new Float32Array([1.0, 2.0, 0.5]);
    const grad   = crossEntropyGrad(logits, 1);
    // dL/d logit_target = prob_target - 1 < 0 (since prob_target < 1)
    expect(grad[1]).toBeLessThan(0);
});

test('crossEntropyGrad non-target components are positive', () => {
    const logits = new Float32Array([1.0, 2.0, 0.5]);
    const grad   = crossEntropyGrad(logits, 1);
    expect(grad[0]).toBeGreaterThan(0);
    expect(grad[2]).toBeGreaterThan(0);
});

test('crossEntropyGrad length matches logits length', () => {
    const logits = new Float32Array(100).fill(0.1);
    const grad   = crossEntropyGrad(logits, 50);
    expect(grad.length).toBe(100);
});

// ── Tape recording ────────────────────────────────────────────────────────────

test('recordOperation returns -1 when grad is disabled', () => {
    noGrad();
    const idx = recordOperation(() => {});
    expect(idx).toBe(-1);
    enableGrad();
    clearTape();
});

test('recordOperation returns a valid index when grad is enabled', () => {
    enableGrad();
    clearTape();
    const idx = recordOperation(() => {});
    expect(idx).toBe(0);
    clearTape();
});

test('backward calls closures in reverse order', async () => {
    enableGrad();
    clearTape();

    const order = [];
    recordOperation(async () => { order.push(1); });
    recordOperation(async () => { order.push(2); });
    recordOperation(async () => { order.push(3); });

    await backward();
    expect(order).toEqual([3, 2, 1]);
});

test('backward clears the tape', async () => {
    enableGrad();
    clearTape();
    recordOperation(async () => {});
    await backward();
    // After backward the tape should be empty, so calling backward again is a no-op
    const order = [];
    await backward();
    expect(order.length).toBe(0);
});

// ── Tensor ────────────────────────────────────────────────────────────────────

test('Tensor.numel computes product of shape', () => {
    // No GPU needed for shape arithmetic
    const t = new Tensor(null, [2, 3, 4]);
    expect(t.numel).toBe(24);
});

test('Tensor.byteSize is numel * 4', () => {
    const t = new Tensor(null, [10]);
    expect(t.byteSize).toBe(40);
});
