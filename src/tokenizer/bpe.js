/**
 * bpe.js – Browser-side Byte Pair Encoding (BPE) tokenizer.
 *
 * Compatible with Qwen3.5-Coder's vocabulary and merge rules.
 * Implements the standard Hugging Face / tiktoken BPE algorithm:
 *
 *   1. Pre-tokenize text into Unicode code point sequences.
 *   2. Apply byte-level encoding for unknown characters.
 *   3. Iteratively merge the highest-priority adjacent byte-pair
 *      according to the learnt merge table.
 *
 * Usage
 * -----
 *   const tokenizer = new BPETokenizer();
 *   await tokenizer.load(vocabUrl, mergesUrl);
 *
 *   const ids  = tokenizer.encode("function foo() {}");
 *   const text = tokenizer.decode(ids);
 */

// Byte-level fallback map (matches GPT-2 / Qwen convention).
// Maps raw bytes 0-255 to printable Unicode characters so that every
// byte sequence has a valid string representation.
function buildByteEncoder() {
    /** @type {Map<number, string>} */
    const enc = new Map();
    const ranges = [
        [0x21, 0x7E],   // ! → ~
        [0xA1, 0xAC],   // ¡ → ¬
        [0xAE, 0xFF],   // ® → ÿ
    ];
    let n = 0;
    for (const [lo, hi] of ranges) {
        for (let b = lo; b <= hi; b++) {
            enc.set(b, String.fromCodePoint(b));
        }
    }
    for (let b = 0; b < 256; b++) {
        if (!enc.has(b)) {
            enc.set(b, String.fromCodePoint(256 + n));
            n++;
        }
    }
    return enc;
}

const BYTE_ENCODER = buildByteEncoder();
const BYTE_DECODER = new Map([...BYTE_ENCODER].map(([k, v]) => [v, k]));

// Pre-tokenisation regex: matches words, numbers, punctuation, whitespace
// (closely mirrors tiktoken's GPT-2/Qwen pre-tokenizer pattern).
const PRE_TOKENIZE_RE =
    /(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

export class BPETokenizer {
    constructor() {
        /** @type {Map<string, number>} token → id */
        this.vocab  = new Map();
        /** @type {Map<number, string>} id → token */
        this.idToToken = new Map();
        /** @type {Map<string, number>} "a b" → rank (lower = higher priority) */
        this.merges = new Map();

        // Special tokens (added after loading vocab)
        this.bosToken   = '<|im_start|>';
        this.eosToken   = '<|im_end|>';
        this.padToken   = '<|endoftext|>';
        this.unkToken   = '<unk>';

        this.bosId      = null;
        this.eosId      = null;
        this.padId      = null;
    }

    /**
     * Load vocabulary and merge rules from JSON / text URLs.
     *
     * vocab.json  – { "token": id, ... }
     * merges.txt  – one merge per line: "a b"  (sorted by rank)
     *
     * @param {string|object} vocab   – URL string or pre-parsed vocab object
     * @param {string|string[]} merges – URL string or array of merge strings
     * @returns {Promise<void>}
     */
    async load(vocab, merges) {
        // Load vocab
        let vocabObj;
        if (typeof vocab === 'string') {
            const res = await fetch(vocab);
            vocabObj = await res.json();
        } else {
            vocabObj = vocab;
        }
        this.vocab     = new Map(Object.entries(vocabObj).map(([k, v]) => [k, Number(v)]));
        this.idToToken = new Map([...this.vocab].map(([k, v]) => [v, k]));

        // Load merges
        let mergeLines;
        if (typeof merges === 'string') {
            const res = await fetch(merges);
            const txt = await res.text();
            mergeLines = txt.split('\n').filter(l => l && !l.startsWith('#'));
        } else {
            mergeLines = merges;
        }
        this.merges = new Map();
        mergeLines.forEach((line, rank) => {
            this.merges.set(line.trim(), rank);
        });

        // Resolve special token ids
        this.bosId = this.vocab.get(this.bosToken) ?? null;
        this.eosId = this.vocab.get(this.eosToken) ?? null;
        this.padId = this.vocab.get(this.padToken) ?? null;
    }

    /**
     * Load from plain JavaScript objects (no network fetch).
     * Useful for bundling a small vocabulary directly.
     *
     * @param {Object} vocabObj   – { token: id }
     * @param {string[]} mergeArr – ["a b", "c d", ...]
     */
    loadFromObjects(vocabObj, mergeArr) {
        this.vocab     = new Map(Object.entries(vocabObj).map(([k, v]) => [k, Number(v)]));
        this.idToToken = new Map([...this.vocab].map(([k, v]) => [v, k]));
        this.merges    = new Map(mergeArr.map((m, i) => [m, i]));
        this.bosId = this.vocab.get(this.bosToken) ?? null;
        this.eosId = this.vocab.get(this.eosToken) ?? null;
        this.padId = this.vocab.get(this.padToken) ?? null;
    }

    // ── Encoding ──────────────────────────────────────────────────────────────

    /**
     * Encode a string to an array of token IDs.
     *
     * @param {string} text
     * @param {{ addBos?: boolean, addEos?: boolean }} [opts]
     * @returns {number[]}
     */
    encode(text, opts = {}) {
        const words = text.match(PRE_TOKENIZE_RE) ?? [];
        const ids   = [];

        if (opts.addBos && this.bosId !== null) ids.push(this.bosId);

        for (const word of words) {
            // Convert to byte-level encoding
            const bytes    = new TextEncoder().encode(word);
            const byteStr  = Array.from(bytes).map(b => BYTE_ENCODER.get(b) ?? '?').join('');
            const bpeTokens = this._bpe(byteStr);

            for (const tok of bpeTokens) {
                const id = this.vocab.get(tok);
                if (id !== undefined) {
                    ids.push(id);
                } else {
                    // Fallback: encode each character individually
                    for (const ch of tok) {
                        const cid = this.vocab.get(ch);
                        if (cid !== undefined) ids.push(cid);
                    }
                }
            }
        }

        if (opts.addEos && this.eosId !== null) ids.push(this.eosId);
        return ids;
    }

    /**
     * Decode an array of token IDs back to a string.
     *
     * @param {number[]} ids
     * @returns {string}
     */
    decode(ids) {
        let byteStr = '';
        for (const id of ids) {
            const tok = this.idToToken.get(id);
            if (tok !== undefined) byteStr += tok;
        }
        // Convert byte-level string back to raw bytes then UTF-8 decode
        const bytes = new Uint8Array(
            [...byteStr].map(ch => BYTE_DECODER.get(ch) ?? ch.codePointAt(0))
        );
        try {
            return new TextDecoder('utf-8').decode(bytes);
        } catch {
            return byteStr;
        }
    }

    // ── BPE merge algorithm ───────────────────────────────────────────────────

    /**
     * Apply BPE merges to a byte-encoded word.
     *
     * @param {string} word  – Space-free string of byte-level characters
     * @returns {string[]}   – Merged token pieces
     */
    _bpe(word) {
        if (this.vocab.has(word)) return [word];

        // Start: each character is its own symbol
        let symbols = [...word];

        // Iteratively merge the highest-priority pair
        while (symbols.length > 1) {
            let bestRank = Infinity;
            let bestIdx  = -1;

            for (let i = 0; i < symbols.length - 1; i++) {
                const pair = symbols[i] + ' ' + symbols[i + 1];
                const rank = this.merges.get(pair);
                if (rank !== undefined && rank < bestRank) {
                    bestRank = rank;
                    bestIdx  = i;
                }
            }

            if (bestIdx === -1) break;  // no more merges available

            // Merge pair at bestIdx
            const merged = symbols[bestIdx] + symbols[bestIdx + 1];
            symbols = [
                ...symbols.slice(0, bestIdx),
                merged,
                ...symbols.slice(bestIdx + 2),
            ];
        }

        return symbols;
    }

    // ── Padding / truncation helpers ─────────────────────────────────────────

    /**
     * Pad or truncate a sequence to a fixed length.
     *
     * @param {number[]} ids
     * @param {number}   maxLen
     * @param {'right'|'left'} [side='right']
     * @returns {number[]}
     */
    padOrTruncate(ids, maxLen, side = 'right') {
        if (ids.length >= maxLen) return ids.slice(0, maxLen);
        const padId = this.padId ?? 0;
        const pad   = new Array(maxLen - ids.length).fill(padId);
        return side === 'right' ? [...ids, ...pad] : [...pad, ...ids];
    }

    /** @returns {number} */
    get vocabSize() { return this.vocab.size; }
}
