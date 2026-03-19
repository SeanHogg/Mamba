/**
 * tests/bpe.test.js
 * Unit tests for the BPE tokenizer (no GPU required).
 */

import { BPETokenizer } from '../src/tokenizer/bpe.js';

// ── Minimal synthetic vocabulary for testing ──────────────────────────────────

function buildTinyTokenizer() {
    const tokenizer = new BPETokenizer();

    // Build a minimal vocab that covers ASCII printable characters + a few words
    // The byte encoder maps bytes 33–126 (! to ~) to themselves,
    // so single-char printable ASCII tokens are their own strings.

    const vocab = {};
    // Add byte-level tokens (printable ASCII 33–126 → same character)
    for (let b = 33; b <= 126; b++) {
        vocab[String.fromCodePoint(b)] = b - 33;
    }
    // Add a few merged tokens
    vocab['he'] = 100;
    vocab['ll'] = 101;
    vocab['hel'] = 102;
    vocab['hell'] = 103;
    vocab['hello'] = 104;

    // Space is in the non-printable range and gets mapped to Unicode 256+
    // byte 32 (space) → code point 256 + offset
    // Simplify: add it directly
    vocab['\u0100'] = 200;  // byte 32 (space) → code point 256

    const merges = [
        'h e',
        'l l',
        'h e l',  // will be skipped since 'he' + 'l' isn't a valid pair key
        'he l',
        'hel l',
        'hell o',
    ];

    tokenizer.loadFromObjects(vocab, merges);
    return tokenizer;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

test('BPETokenizer loads vocab from objects', () => {
    const t = buildTinyTokenizer();
    expect(t.vocabSize).toBeGreaterThan(0);
});

test('encode returns an array of numbers', () => {
    const t = buildTinyTokenizer();
    const ids = t.encode('hello');
    expect(Array.isArray(ids)).toBe(true);
    for (const id of ids) {
        expect(typeof id).toBe('number');
    }
});

test('decode inverts encode for ASCII tokens', () => {
    const t   = buildTinyTokenizer();
    // Use characters that are in the byte-encoder passthrough range (33–126)
    const text = 'abc';
    const ids  = t.encode(text);
    const back = t.decode(ids);
    // Should recover original string (may differ in whitespace handling)
    expect(back).toBe(text);
});

test('padOrTruncate truncates long sequences', () => {
    const t   = buildTinyTokenizer();
    const ids = [1, 2, 3, 4, 5];
    expect(t.padOrTruncate(ids, 3)).toEqual([1, 2, 3]);
});

test('padOrTruncate pads short sequences (right)', () => {
    const t   = buildTinyTokenizer();
    const ids = [1, 2];
    const out = t.padOrTruncate(ids, 5, 'right');
    expect(out.length).toBe(5);
    expect(out.slice(0, 2)).toEqual([1, 2]);
    // Padded with padId (defaults to 0 if not set)
    expect(out[2]).toBe(0);
});

test('padOrTruncate pads short sequences (left)', () => {
    const t   = buildTinyTokenizer();
    const ids = [1, 2];
    const out = t.padOrTruncate(ids, 4, 'left');
    expect(out.length).toBe(4);
    expect(out.slice(2)).toEqual([1, 2]);
});

test('encode handles empty string', () => {
    const t   = buildTinyTokenizer();
    const ids = t.encode('');
    expect(Array.isArray(ids)).toBe(true);
    expect(ids.length).toBe(0);
});

test('vocabSize reflects loaded vocabulary', () => {
    const t   = buildTinyTokenizer();
    expect(t.vocabSize).toBeGreaterThan(90);  // ASCII 33–126 = 94 entries + extras
});

test('_bpe returns single token for known word', () => {
    const t = new BPETokenizer();
    t.loadFromObjects({ 'hello': 0 }, []);
    expect(t._bpe('hello')).toEqual(['hello']);
});

test('_bpe splits unknown word into characters', () => {
    const t = new BPETokenizer();
    t.loadFromObjects({}, []);
    const result = t._bpe('abc');
    expect(result).toEqual(['a', 'b', 'c']);
});

test('_bpe applies merges correctly', () => {
    const t = new BPETokenizer();
    t.loadFromObjects({ 'ab': 0, 'c': 1 }, ['a b']);
    const result = t._bpe('abc');
    expect(result).toEqual(['ab', 'c']);
});
