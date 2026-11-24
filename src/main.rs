mod json;
mod rc;

use json::{parse_safetensors_header, TensorInfo};
use rc::{decode_sym, encode_sym, RangeDecoder, RangeEncoder, P_INIT};
use std::time::Instant;

// ======================================================================
// bfc — BF16-aware lossless compressor for safetensors files.
//
// Idea (see discussion): ~all of the compressibility of bf16 weights lives
// in the sign+exponent bits; the 7-bit mantissa is near-uniform. So each
// tensor is split into two streams, coded with adaptive context models:
//   stream 1 ("exp"):  9-bit (sign|exponent) symbol per value, binary-tree
//                      coded, conditioned on the *previous* value's
//                      exponent bucket (32 buckets) -> first-order model.
//   stream 2 ("mant"): 7-bit mantissa per value, conditioned on the same
//                      value's exponent bucket (--fast: raw 7-bit packing).
// No permutation tables: the original interleaving is reconstructed by
// position, regrouping is implicit.
// Non-BF16 tensors and non-safetensors files fall back to an order-1
// adaptive model over raw bytes.
// ======================================================================

const MAGIC: &[u8; 4] = b"BFC1";

// ---------------------------------------------------------------- byte helpers

struct W {
    v: Vec<u8>,
}
impl W {
    fn new() -> Self {
        W { v: Vec::new() }
    }
    fn u8(&mut self, x: u8) {
        self.v.push(x);
    }
    fn u32(&mut self, x: u32) {
        self.v.extend_from_slice(&x.to_le_bytes());
    }
    fn u64(&mut self, x: u64) {
        self.v.extend_from_slice(&x.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.v.extend_from_slice(b);
    }
}

struct R<'a> {
    b: &'a [u8],
    pos: usize,
}
impl<'a> R<'a> {
    fn new(b: &'a [u8]) -> Self {
        R { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.b.len() {
            return Err("unexpected end of file".into());
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

// ---------------------------------------------------------------- bit packing

struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    n: u32,
}
impl BitWriter {
    fn new() -> Self {
        BitWriter { buf: Vec::new(), acc: 0, n: 0 }
    }
    #[inline]
    fn write_bits(&mut self, v: u64, k: u32) {
        self.acc |= v << self.n;
        self.n += k;
        while self.n >= 8 {
            self.buf.push((self.acc & 0xFF) as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    fn finish(self) -> Vec<u8> {
        let mut b = self.buf;
        if self.n > 0 {
            b.push(self.acc as u8);
        }
        b
    }
}

struct BitReader<'a> {
    b: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
}
impl<'a> BitReader<'a> {
    fn new(b: &'a [u8]) -> Self {
        BitReader { b, pos: 0, acc: 0, n: 0 }
    }
    #[inline]
    fn read_bits(&mut self, k: u32) -> u64 {
        while self.n < k {
            let byte = if self.pos < self.b.len() { self.b[self.pos] } else { 0 };
            self.pos += 1;
            self.acc |= (byte as u64) << self.n;
            self.n += 8;
        }
        let v = self.acc & ((1u64 << k) - 1);
        self.acc >>= k;
        self.n -= k;
        v
    }
}

// ---------------------------------------------------------------- bf16 codec

#[inline]
fn exp_bucket(exp: u32) -> usize {
    // 32 magnitude buckets; bf16 weight exponents cluster around 96..135,
    // large outliers live above. exp < 96 (incl. exp=0 zeros) -> bucket 0.
    if exp < 96 {
        0
    } else {
        (1 + ((exp - 96) >> 2)).min(31) as usize
    }
}

/// Split-compress one bf16 tensor. Returns (exp_stream, mant_stream).
fn compress_bf16(data: &[u8], packed_mant: bool) -> (Vec<u8>, Vec<u8>) {
    let n = data.len() / 2;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut probs9 = vec![P_INIT; 32 * 512]; // (sign|exp) 9-bit tree, prev-exp bucket context
    let mut probs7 = vec![P_INIT; 32 * 128]; // mantissa 7-bit tree, own-exp bucket context
    let mut e9 = RangeEncoder::new();
    let mut e7 = RangeEncoder::new();
    let mut bw = BitWriter::new();
    let mut prev_exp: u32 = 127;
    for i in 0..n {
        let v = u16::from_le_bytes([data[2 * i], data[2 * i + 1]]);
        let sign = (v >> 15) as u32;
        let exp = ((v >> 7) & 0xFF) as u32;
        let man = (v & 0x7F) as u32;
        let b = exp_bucket(prev_exp) * 512;
        encode_sym(&mut e9, &mut probs9[b..b + 512], 9, (sign << 8) | exp);
        if packed_mant {
            bw.write_bits(man as u64, 7);
        } else {
            let b7 = exp_bucket(exp) * 128;
            encode_sym(&mut e7, &mut probs7[b7..b7 + 128], 7, man);
        }
        prev_exp = exp;
    }
    let exp = e9.finish();
    let mant = if packed_mant { bw.finish() } else { e7.finish() };
    (exp, mant)
}

fn decompress_bf16(exp: &[u8], mant: &[u8], packed_mant: bool, n_bytes: usize) -> Vec<u8> {
    let n = n_bytes / 2;
    let mut out = vec![0u8; n * 2];
    if n == 0 {
        return out;
    }
    let mut probs9 = vec![P_INIT; 32 * 512];
    let mut probs7 = vec![P_INIT; 32 * 128];
    let mut d9 = RangeDecoder::new(exp);
    let mut br = if packed_mant { Some(BitReader::new(mant)) } else { None };
    let mut d7 = if packed_mant { None } else { Some(RangeDecoder::new(mant)) };
    let mut prev_exp: u32 = 127;
    for i in 0..n {
        let b = exp_bucket(prev_exp) * 512;
        let s9 = decode_sym(&mut d9, &mut probs9[b..b + 512], 9);
        let sign = s9 >> 8;
        let expv = s9 & 0xFF;
        let man = if packed_mant {
            br.as_mut().unwrap().read_bits(7) as u32
        } else {
            let b7 = exp_bucket(expv) * 128;
            decode_sym(d7.as_mut().unwrap(), &mut probs7[b7..b7 + 128], 7)
        };
        let v = ((sign << 15) | (expv << 7) | man) as u16;
        out[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
        prev_exp = expv;
    }
    out
}

// ---------------------------------------------------------------- raw codec (order-1)

fn compress_raw(data: &[u8]) -> Vec<u8> {
    let mut probs = vec![P_INIT; 256 * 256];
    let mut e = RangeEncoder::new();
    let mut prev = 0usize;
    for &b in data {
        let ctx = prev * 256;
        encode_sym(&mut e, &mut probs[ctx..ctx + 256], 8, b as u32);
        prev = b as usize;
    }
    e.finish()
}

fn decompress_raw(stream: &[u8], n: usize) -> Vec<u8> {
    let mut probs = vec![P_INIT; 256 * 256];
    let mut d = RangeDecoder::new(stream);
    let mut prev = 0usize;
    let mut out = vec![0u8; n];
    for i in 0..n {
        let ctx = prev * 256;
        let s = decode_sym(&mut d, &mut probs[ctx..ctx + 256], 8);
        out[i] = s as u8;
        prev = s as usize;
    }
    out
}

// ---------------------------------------------------------------- parallelism

fn parallel_map<T, F>(items: usize, threads: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T + Send + Sync,
    T: Send,
{
    let threads = threads.max(1).min(items.max(1));
    if threads <= 1 || items <= 1 {
        return (0..items).map(|i| f(i)).collect();
    }
    let f = &f;
    std::thread::scope(|s| {
        let per = (items + threads - 1) / threads;
        let mut handles = Vec::new();
        let mut t = 0usize;
        while t * per < items {
            let lo = t * per;
            let hi = ((t + 1) * per).min(items);
            handles.push(s.spawn(move || (lo..hi).map(|i| f(i)).collect::<Vec<_>>()));
            t += 1;
        }
        let mut out = Vec::new();
        for h in handles {
            // any panic in a worker is a real bug; propagate it
            out.extend(h.join().expect("worker thread panicked"));
        }
        out
    })
}

// ---------------------------------------------------------------- container

struct TensorComp {
    tensor_len: usize,
    method: u8, // 0 = bf16 adaptive, 1 = bf16 fast (packed mantissa), 2 = raw order-1
    exp: Vec<u8>,
    mant: Vec<u8>,
}

fn detect_safetensors(data: &[u8]) -> Option<(usize, Vec<TensorInfo>)> {
    if data.len() < 8 {
        return None;
    }
    let hlen = u64::from_le_bytes(data[0..8].try_into().ok()?) as usize;
    if hlen == 0 || 8usize.saturating_add(hlen) > data.len() || hlen > (1 << 30) {
        return None;
    }
    if data[8] != b'{' {
        return None;
    }
    let header = &data[8..8 + hlen];
    let mut tensors = parse_safetensors_header(header)?;
    // tensors must exactly tile the data region (no gaps/padding between them)
    let data_len = data.len() - 8 - hlen;
    let mut expected = 0u64;
    for t in &tensors {
        if t.start != expected || t.end < t.start {
            return None;
        }
        expected = t.end;
    }
    if expected != data_len as u64 {
        return None;
    }
    Some((hlen, tensors))
}

pub fn compress_bytes(data: &[u8], fast: bool, threads: usize) -> Vec<u8> {
    let t0 = Instant::now();
    let (mode, header_block, mut tensors) = match detect_safetensors(data) {
        Some((hlen, tensors)) => (1u8, Some(&data[8..8 + hlen]), Some(tensors)),
        None => (0u8, None, None),
    };

    let comps: Vec<TensorComp> = if mode == 1 {
        let tensors = tensors.take().unwrap();
        let data_start = 8 + header_block.unwrap().len();
        let n = tensors.len();
        parallel_map(n, threads, |i| {
            let t = &tensors[i];
            let len = (t.end - t.start) as usize;
            let slice = &data[data_start + t.start as usize..data_start + t.end as usize];
            if t.dtype == "BF16" && len % 2 == 0 {
                let (exp, mant) = compress_bf16(slice, fast);
                let method = if fast { 1 } else { 0 };
                TensorComp { tensor_len: len, method, exp, mant }
            } else {
                let exp = compress_raw(slice);
                TensorComp { tensor_len: len, method: 2, exp, mant: Vec::new() }
            }
        })
    } else {
        let exp = compress_raw(data);
        vec![TensorComp { tensor_len: data.len(), method: 2, exp, mant: Vec::new() }]
    };

    let mut w = W::new();
    w.bytes(MAGIC);
    w.u8(mode);
    w.u64(data.len() as u64);
    if mode == 1 {
        let hb = header_block.unwrap();
        w.u32(hb.len() as u32);
        w.bytes(hb);
        w.u32(comps.len() as u32);
        for c in &comps {
            w.u64(c.tensor_len as u64);
            w.u8(c.method);
            w.u64(c.exp.len() as u64);
            w.bytes(&c.exp);
            w.u64(c.mant.len() as u64);
            w.bytes(&c.mant);
        }
    } else {
        w.u64(comps[0].exp.len() as u64);
        w.bytes(&comps[0].exp);
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let out = &w.v;
    let ratio = data.len() as f64 / out.len().max(1) as f64;
    eprintln!(
        "compressed {} bytes -> {} bytes (ratio {:.3}:1, {:.1} MB/s)",
        data.len(),
        out.len(),
        ratio,
        data.len() as f64 / 1e6 / elapsed.max(1e-9)
    );
    std::mem::take(&mut w.v)
}

pub fn decompress_bytes(data: &[u8], threads: usize) -> Result<Vec<u8>, String> {
    let mut r = R::new(data);
    if r.take(4)? != MAGIC {
        return Err("not a bfc file (bad magic)".into());
    }
    let mode = r.u8()?;
    let orig_len = r.u64()? as usize;
    let out: Vec<u8> = match mode {
        0 => {
            let slen = r.u64()? as usize;
            let stream = r.take(slen)?.to_vec();
            let decoded = decompress_raw(&stream, orig_len);
            decoded
        }
        1 => {
            let hlen = r.u32()? as usize;
            let header = r.take(hlen)?.to_vec();
            let n = r.u32()? as usize;
            let mut metas = Vec::with_capacity(n);
            for _ in 0..n {
                let tensor_len = r.u64()? as usize;
                let method = r.u8()?;
                let elen = r.u64()? as usize;
                let exp = r.take(elen)?.to_vec();
                let mlen = r.u64()? as usize;
                let mant = r.take(mlen)?.to_vec();
                metas.push((tensor_len, method, exp, mant));
            }
            let exps: Vec<Vec<u8>> = metas.iter().map(|(_, _, e, _)| e.clone()).collect();
            let mants: Vec<Vec<u8>> = metas.iter().map(|(_, _, _, m)| m.clone()).collect();
            let lens: Vec<usize> = metas.iter().map(|(l, _, _, _)| *l).collect();
            let methods: Vec<u8> = metas.iter().map(|(_, m, _, _)| *m).collect();
            // decode in parallel; closures only read the cloned stream vecs
            let decoded: Vec<Vec<u8>> = parallel_map(n, threads, |i| match methods[i] {
                0 | 1 => {
                    decompress_bf16(&exps[i], &mants[i], methods[i] == 1, lens[i])
                }
                2 => decompress_raw(&exps[i], lens[i]),
                _ => panic!("bad method"),
            });
            let mut body = Vec::with_capacity(decoded.iter().map(|d| d.len()).sum());
            for d in decoded {
                body.extend_from_slice(&d);
            }
            let mut out = Vec::with_capacity(8 + header.len() + body.len());
            out.extend_from_slice(&(header.len() as u64).to_le_bytes());
            out.extend_from_slice(&header);
            out.extend_from_slice(&body);
            if out.len() != orig_len {
                return Err(format!(
                    "size mismatch: reconstructed {} bytes, expected {}",
                    out.len(),
                    orig_len
                ));
            }
            out
        }
        _ => return Err(format!("unknown mode {mode}")),
    };
    Ok(out)
}

// ---------------------------------------------------------------- info

fn human(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1e3)
    } else {
        format!("{bytes} B")
    }
}

pub fn info_bytes(data: &[u8]) -> Result<(), String> {
    if data.len() >= 4 && &data[0..4] == MAGIC {
        let mut r = R::new(data);
        r.take(4)?;
        let mode = r.u8()?;
        let orig = r.u64()? as usize;
        println!("bfc compressed file");
        println!("  mode:            {}", if mode == 1 { "safetensors-aware" } else { "raw order-1" });
        println!("  original size:   {} ({})", orig, human(orig));
        println!("  compressed size: {} ({})", data.len(), human(data.len()));
        println!("  ratio:           {:.3}:1", orig as f64 / data.len().max(1) as f64);
        if mode == 1 {
            let hlen = r.u32()? as usize;
            let header = r.take(hlen)?;
            let n = r.u32()? as usize;
            let tensors = parse_safetensors_header(header).unwrap_or_default();
            println!("  tensors:         {n}");
            println!();
            println!(
                "  {:<44} {:>12} {:>6} {:>10} {:>10} {:>10}",
                "tensor", "bytes", "method", "exp KB", "mant KB", "bits/val"
            );
            let mut tot_exp = 0usize;
            let mut tot_mant = 0usize;
            let mut tot_len = 0usize;
            for (i, t) in tensors.iter().enumerate().take(n) {
                let tensor_len = r.u64()? as usize;
                let method = r.u8()?;
                let elen = r.u64()? as usize;
                r.take(elen)?; // skip payload
                let mlen = r.u64()? as usize;
                r.take(mlen)?; // skip payload
                tot_exp += elen;
                tot_mant += mlen;
                tot_len += tensor_len;
                if i < 20 {
                    let name: String = t.name.chars().take(42).collect();
                    let nvals = if method <= 1 { tensor_len / 2 } else { tensor_len };
                    let bits = if nvals > 0 {
                        ((elen + mlen) as f64) * 8.0 / nvals as f64
                    } else {
                        0.0
                    };
                    println!(
                        "  {:<44} {:>12} {:>6} {:>10.1} {:>10.1} {:>10.2}",
                        name,
                        tensor_len,
                        method,
                        elen as f64 / 1e3,
                        mlen as f64 / 1e3,
                        bits
                    );
                }
            }
            if n > 20 {
                println!("  ... ({} more tensors)", n - 20);
            }
            println!();
            let nvals = tot_len / 2;
            println!(
                "  totals: exp {} | mant {} | payload {} ({:.2} bits/value)",
                human(tot_exp),
                human(tot_mant),
                human(tot_len),
                if nvals > 0 { ((tot_exp + tot_mant) as f64) * 8.0 / nvals as f64 } else { 0.0 }
            );
        }
        return Ok(());
    }
    match detect_safetensors(data) {
        Some((hlen, tensors)) => {
            println!("safetensors file");
            println!("  size:    {} ({})", data.len(), human(data.len()));
            println!("  tensors: {}", tensors.len());
            println!();
            println!("  {:<52} {:>8} {:>14}", "tensor", "dtype", "bytes");
            for t in tensors.iter().take(40) {
                let name: String = t.name.chars().take(50).collect();
                println!("  {:<52} {:>8} {:>14}", name, t.dtype, t.end - t.start);
            }
            if tensors.len() > 40 {
                println!("  ... ({} more tensors)", tensors.len() - 40);
            }
            let _ = hlen;
        }
        None => {
            println!("unknown file: {} ({}) — bfc will fall back to raw order-1 compression", data.len(), human(data.len()));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- synthetic data

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self, mu: f64, sigma: f64) -> f64 {
        // Box-Muller
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        mu + sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    let rounded = b.wrapping_add(0x7FFF).wrapping_add((b >> 16) & 1);
    (rounded >> 16) as u16
}

/// Generate a synthetic safetensors file with realistic LLM-like bf16 tensors:
/// embeddings, large 2D weights (mostly small-magnitude normals), tensors with
/// rare large-magnitude outliers, layernorm gains near 1.0, and small biases.
pub fn gendata_bytes(target_mb: usize, seed: u64) -> Vec<u8> {
    let target = target_mb as u64 * 1_000_000;
    let mut rng = Rng(seed ^ 0x9E3779B97F4A7C15);
    let mut tensors: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut offset: u64 = 0;
    let mut total: u64 = 0;
    let mut layer: usize = 0;

    // embedding
    let d = 1024usize;
    let vocab = ((target / 5 / (2 * d as u64)) as usize).clamp(4096, 32000);
    {
        let n = vocab * d;
        let mut buf = Vec::with_capacity(n * 2);
        for _ in 0..n {
            let v = f32_to_bf16(rng.normal(0.0, 0.03) as f32);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let len = buf.len() as u64;
        tensors.push((
            "model.embed_tokens.weight".into(),
            vec![vocab as u64, d as u64],
            buf,
        ));
        offset += len;
        total += len;
    }

    let mut kind = 0usize;
    while total < target {
        let a = (512 + (rng.next_u64() as usize % 32) * 128) / 64 * 64;
        let b = (512 + (rng.next_u64() as usize % 16) * 128) / 64 * 64;
        let l = layer % 12;
        match kind {
            0 | 1 => {
                // ordinary weight matrices
                let sigma = if kind == 0 { 0.02 } else { 0.05 };
                let name = format!("model.layers.{l}.mlp.gate_proj.weight");
                let n = a * b;
                let mut buf = Vec::with_capacity(n * 2);
                for _ in 0..n {
                    let v = f32_to_bf16(rng.normal(0.0, sigma) as f32);
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                let len = buf.len() as u64;
                tensors.push((name, vec![a as u64, b as u64], buf));
                offset += len;
                total += len;
            }
            2 => {
                // weights with rare big outliers (LLM outlier channels)
                let name = format!("model.layers.{l}.self_attn.o_proj.weight");
                let n = a * b;
                let mut buf = Vec::with_capacity(n * 2);
                for _ in 0..n {
                    let x = if rng.next_f64() < 0.01 {
                        rng.normal(0.0, 1.0)
                    } else {
                        rng.normal(0.0, 0.02)
                    };
                    let v = f32_to_bf16(x as f32);
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                let len = buf.len() as u64;
                tensors.push((name, vec![a as u64, b as u64], buf));
                offset += len;
                total += len;
            }
            3 => {
                // layernorm gain: values tightly around 1.0
                let name = format!("model.layers.{l}.input_layernorm.weight");
                let mut buf = Vec::with_capacity(d * 2);
                for _ in 0..d {
                    let v = f32_to_bf16((1.0 + rng.normal(0.0, 0.05)) as f32);
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                let len = buf.len() as u64;
                tensors.push((name, vec![d as u64], buf));
                offset += len;
                total += len;
            }
            _ => {
                // small bias
                let name = format!("model.layers.{l}.mlp.up_proj.bias");
                let mut buf = Vec::with_capacity(b * 2);
                for _ in 0..b {
                    let v = f32_to_bf16(rng.normal(0.0, 0.001) as f32);
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                let len = buf.len() as u64;
                tensors.push((name, vec![b as u64], buf));
                offset += len;
                total += len;
            }
        }
        kind = (kind + 1) % 4;
        layer += 1;
    }

    // header JSON, padded with spaces so the data section stays 8-byte aligned
    let mut json = String::from("{");
    for (i, (name, shape, buf)) in tensors.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        let shape_s: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        json.push_str(&format!(
            "\"{}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name,
            shape_s.join(","),
            offset_zero(i, &tensors),
            offset_zero(i, &tensors) + buf.len() as u64
        ));
    }
    json.push('}');
    let hlen_needed = ((8 + json.len() + 7) / 8) * 8 - 8;
    while json.len() < hlen_needed {
        json.push(' ');
    }

    let mut out = Vec::with_capacity(8 + json.len() + total as usize);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    for (_, _, buf) in &tensors {
        out.extend_from_slice(buf);
    }
    out
}

fn offset_zero(i: usize, tensors: &[(String, Vec<u64>, Vec<u8>)]) -> u64 {
    tensors[..i].iter().map(|(_, _, b)| b.len() as u64).sum()
}

// ---------------------------------------------------------------- selftest

pub fn selftest() -> Result<(), String> {
    println!("bfc selftest");
    // 1. bf16 tensor roundtrip: several distributions, both mantissa modes
    let mut rng = Rng(12345);
    let mut cases: Vec<(String, Vec<u8>)> = Vec::new();
    {
        let n = 1 << 20;
        let mut a = Vec::with_capacity(n * 2);
        for _ in 0..n {
            a.extend_from_slice(&f32_to_bf16(rng.normal(0.0, 0.03) as f32).to_le_bytes());
        }
        cases.push(("normal(0,0.03)".into(), a));
        let mut b = Vec::with_capacity(n * 2);
        for i in 0..n {
            let x = if (i % 100) == 0 { rng.normal(0.0, 1.0) } else { rng.normal(0.0, 0.02) };
            b.extend_from_slice(&f32_to_bf16(x as f32).to_le_bytes());
        }
        cases.push(("1% outliers".into(), b));
        let mut c = Vec::with_capacity(65536 * 2);
        for _ in 0..65536 {
            c.extend_from_slice(&f32_to_bf16((1.0 + rng.normal(0.0, 0.05)) as f32).to_le_bytes());
        }
        cases.push(("near 1.0".into(), c));
        cases.push(("all zeros".into(), vec![0u8; 65536 * 2]));
        cases.push(("tiny tensor".into(), vec![0x3C, 0x00, 0x3D, 0x80, 0xBE, 0x4F, 0x00, 0x00]));
    }
    for (name, tensor) in &cases {
        for packed in [false, true] {
            let (exp, mant) = compress_bf16(tensor, packed);
            let back = decompress_bf16(&exp, &mant, packed, tensor.len());
            if back != *tensor {
                return Err(format!("FAIL bf16 roundtrip: {} (packed={})", name, packed));
            }
            let bits = ((exp.len() + mant.len()) as f64) * 8.0 / (tensor.len() / 2) as f64;
            println!("  PASS bf16 {:<16} packed={:<5} {:.3} bits/value", name, packed, bits);
        }
    }
    // 2. raw order-1 roundtrip on arbitrary binary (this test's own synthetic blob)
    {
        let mut blob = Vec::new();
        for i in 0..1 << 18 {
            blob.push((i * 7 + (i >> 5)) as u8);
        }
        let stream = compress_raw(&blob);
        let back = decompress_raw(&stream, blob.len());
        if back != blob {
            return Err("FAIL raw roundtrip".into());
        }
        println!("  PASS raw  order-1 roundtrip");
    }
    // 3. full container roundtrip on a synthetic safetensors file
    {
        let st = gendata_bytes(8, 777);
        let comp = compress_bytes(&st, false, 1);
        let back = decompress_bytes(&comp, 1)?;
        if back != st {
            return Err("FAIL container roundtrip".into());
        }
        println!(
            "  PASS container roundtrip ({:.3}:1 on synthetic 8 MB)",
            st.len() as f64 / comp.len() as f64
        );
    }
    println!("all tests passed");
    Ok(())
}

fn usage() {
    eprintln!(
        "bfc — lossless BF16-aware compressor for safetensors (download-platform tool)\n\
         \n\
         usage:\n\
         \x20 bfc compress <in.safetensors> <out.bfc> [--fast] [--threads N]\n\
         \x20 bfc decompress <in.bfc> <out> [--threads N]\n\
         \x20 bfc info <file>            (works on .bfc and .safetensors)\n\
         \x20 bfc gendata <size_mb> <out.safetensors> [--seed N]\n\
         \x20 bfc selftest\n\
         \n\
         notes:\n\
         \x20 --fast  raw 7-bit packing for mantissas instead of adaptive coding\n\
         \x20         (slightly worse ratio, faster); non-BF16 tensors and\n\
         \x20         non-safetensors files use a raw order-1 fallback.\n\
         \x20 decompression is byte-exact; see 'bfc selftest'."
    );
}

fn threads_from(args: &[String]) -> usize {
    for (i, a) in args.iter().enumerate() {
        if a == "--threads" {
            if let Some(v) = args.get(i + 1) {
                if let Ok(n) = v.parse::<usize>() {
                    return n.max(1);
                }
            }
        }
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2)
}

fn cmd_compress(args: &[String]) -> Result<(), String> {
    let fast = args.iter().any(|a| a == "--fast");
    let threads = threads_from(args);
    let data = std::fs::read(&args[1]).map_err(|e| format!("read {}: {e}", args[1]))?;
    let out = compress_bytes(&data, fast, threads);
    std::fs::write(&args[2], &out).map_err(|e| format!("write {}: {e}", args[2]))?;
    Ok(())
}

fn cmd_decompress(args: &[String]) -> Result<(), String> {
    let threads = threads_from(args);
    let data = std::fs::read(&args[1]).map_err(|e| format!("read {}: {e}", args[1]))?;
    let out = decompress_bytes(&data, threads)?;
    std::fs::write(&args[2], &out).map_err(|e| format!("write {}: {e}", args[2]))?;
    eprintln!("decompressed {} bytes (byte-exact)", out.len());
    Ok(())
}

fn cmd_info(args: &[String]) -> Result<(), String> {
    let data = std::fs::read(&args[1]).map_err(|e| format!("read {}: {e}", args[1]))?;
    info_bytes(&data)
}

fn cmd_gendata(args: &[String]) -> Result<(), String> {
    let mb: usize = args[1].parse().map_err(|_| "size_mb must be an integer")?;
    let seed = args
        .iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);
    let data = gendata_bytes(mb, seed);
    std::fs::write(&args[2], &data).map_err(|e| format!("write {}: {e}", args[2]))?;
    eprintln!("wrote {} ({})", args[2], human(data.len()));
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let res: Result<(), String> = match args.first().map(|s| s.as_str()) {
        Some("compress") if args.len() >= 3 => cmd_compress(&args),
        Some("decompress") if args.len() >= 3 => cmd_decompress(&args),
        Some("info") if args.len() >= 2 => cmd_info(&args),
        Some("gendata") if args.len() >= 3 => cmd_gendata(&args),
        Some("selftest") => selftest(),
        _ => {
            usage();
            std::process::exit(2);
        }
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
