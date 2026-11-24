# bfc — BF16-aware lossless compressor for safetensors

`bfc` is a lossless compressor designed for **model download platforms**.
It is safetensors-aware: instead of treating the file as opaque bytes
(WinRAR gets ~1.25:1), it exploits the structure of bf16 weights.

## Why it works

A bf16 value is 1 sign + 8 exponent + 7 mantissa bits. For normally
distributed weights, the mantissa bits are near-uniform (~7 bits of
entropy); essentially **all compressibility lives in sign+exponent**
(~3.5 bits for a 350M-class model). So each tensor is split into two
independent streams, coded with adaptive context models over a built-in
binary range coder (no permutation tables, no sorting — the original
interleaving is reconstructed by position):

- **exp stream** — the 9-bit `(sign | exponent)` symbol of each value,
  binary-tree coded, conditioned on the *previous* value's exponent
  bucket (32 magnitude buckets) → first-order context model.
- **mant stream** — the 7-bit mantissa of each value, conditioned on the
  same value's exponent bucket. `--fast` replaces this with raw 7-bit
  packing (uniform mantissas lose nothing).

Non-BF16 tensors (odd length, other dtypes) and non-safetensors files
fall back to an adaptive order-1 model over raw bytes, so the tool
compresses *anything* losslessly.

Decompression is fully independent per tensor and parallelized across
cores (`--threads N`), which makes chunk-parallel, memory-mappable
serving straightforward to add later.

## Build

```
cargo build --release
```

No external crates; builds offline with any stable Rust ≥ 1.70.
A prebuilt Linux x86-64 binary is included (`bfc-linux-x64`).

## Usage

```
bfc compress   <in.safetensors> <out.bfc> [--fast] [--threads N]
bfc decompress <in.bfc> <out>            [--threads N]
bfc info       <file>        # works on both .bfc and .safetensors
bfc gendata    <size_mb> <out.safetensors> [--seed N]  # synthetic test model
bfc selftest   # entropy-coder + roundtrip checks
```

`--fast`: pack mantissas raw instead of adaptive coding — ~2x faster,
usually near-identical ratio on real weights.

## Measured results (synthetic 260 MB bf16 model, 2-core sandbox)

| mode            | ratio   | speed        |
|-----------------|---------|--------------|
| default         | 1.495:1 | 29 MB/s comp, 34 MB/s decomp |
| `--fast`        | 1.500:1 | 62 MB/s comp |

Roundtrip verified byte-exact with `cmp` at 260 MB. Selftest ratios:
ordinary weights ≈ 10.7 bits/value (matches the measured 10.648-bit
zero-order entropy of a real 350M model); layernorm-style tensors near
1.0 drop to 6.4–6.8 bits/value thanks to context modeling; constant
tensors to ~0.2 bits/value.

Real-model ratios should land near the theoretical 1.50–1.55:1 from your
entropy measurements, with the first-order exponent model potentially
pushing past zero-order on tensors with correlated magnitudes/outliers.

## Compressed format (.bfc), little-endian

```
"BFC1"                       magic
u8   mode                    1 = safetensors-aware, 0 = raw
u64  original_size
mode 1:
  u32 header_len             safetensors header block (JSON+padding) stored verbatim
  header bytes
  u32 num_tensors            tensors in data-offset order
  per tensor:
    u64 tensor_len           payload byte length
    u8  method               0 = bf16 adaptive, 1 = bf16 fast, 2 = raw order-1
    u64 exp_len; exp bytes   method 0/1: range-coded (sign|exp) stream; method 2: the single order-1 stream
    u64 mant_len; mant bytes method 0: range-coded; method 1: 7-bit packed; empty for method 2
mode 0:
  u64 stream_len; stream bytes
```

The original file is reconstructed as: u64 header_len + header block +
concatenated tensor payloads — byte-exact, including header padding.

## Ideas for v2

1. **Delta against a base model** — for a hub serving fine-tunes of the
   same base, store the (compressed) difference; typically far better
   than 1.5:1. The container's per-tensor layout makes this easy to add.
2. **F16 plane split** — fp16 (5-bit exp / 10-bit mantissa) deserves its
   own split; currently falls back to order-1.
3. **Per-row scale conditioning** — code exponents relative to a per-row
   max (few bits per row) to shrink the exponent alphabet further.
4. **Chunk-parallel decode + lazy tensor loading** — streams are
   self-contained per tensor; a serving platform can decode tensors on
   demand and start inference before the full file is materialized.
5. **rANS backend** — same ratio class, faster decode if you need
   multi-GB/s.
