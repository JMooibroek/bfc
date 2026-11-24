// Binary range coder (LZMA-style carry-handling encoder) with adaptive bit models.
//
// Convention: `prob` is a 12-bit fixed-point probability of bit == 1 (0..4096),
// initialized to 2048, adapted multiplicatively with shift RATE (1/32).
// Encoder and decoder are exact mirrors; losslessness is asserted by selftest.

pub const RATE: u32 = 5;
pub const P_INIT: u16 = 2048;

pub fn init_probs(probs: &mut [u16]) {
    for p in probs.iter_mut() {
        *p = P_INIT;
    }
}

// ---------------------------------------------------------------- encoder

pub struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    pub out: Vec<u8>,
}

impl RangeEncoder {
    pub fn new() -> Self {
        RangeEncoder {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            cache_size: 1,
            out: Vec::new(),
        }
    }

    #[inline]
    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add((self.low >> 32) as u8));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    #[inline]
    pub fn encode_bit(&mut self, prob: &mut u16, bit: u32) {
        // prob = P(bit == 1); bit 1 owns [0, bound), bit 0 owns [bound, range).
        let bound = ((self.range >> 12) as u64) * (*prob as u64);
        if bit != 0 {
            self.range = bound as u32;
            *prob += ((4096 - *prob) >> RATE) as u16;
        } else {
            self.low += bound;
            self.range -= bound as u32;
            *prob -= *prob >> RATE;
        }
        while self.range < (1 << 24) {
            self.range <<= 8;
            self.shift_low();
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

// ---------------------------------------------------------------- decoder

pub struct RangeDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = RangeDecoder {
            data,
            pos: 0,
            range: 0xFFFF_FFFF,
            code: 0,
        };
        // Encoder's first emitted byte is always 0 (cache warm-up); skip it.
        let _ = d.next_byte();
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next_byte();
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u32 {
        let b = if self.pos < self.data.len() {
            self.data[self.pos] as u32
        } else {
            0 // encoder flush guarantees we never need bytes past the stream in practice
        };
        self.pos += 1;
        b
    }

    #[inline]
    fn normalize(&mut self) {
        while self.range < (1 << 24) {
            self.range <<= 8;
            self.code = (self.code << 8) | self.next_byte();
        }
    }

    #[inline]
    pub fn decode_bit(&mut self, prob: &mut u16) -> u32 {
        let bound = ((self.range >> 12) as u64) * (*prob as u64);
        let bit;
        if (self.code as u64) < bound {
            // bit 1 owns the interval [0, bound)
            self.range = bound as u32;
            bit = 1;
            *prob += ((4096 - *prob) >> RATE) as u16;
        } else {
            // bit 0 owns the interval [bound, range)
            self.code -= bound as u32;
            self.range -= bound as u32;
            bit = 0;
            *prob -= *prob >> RATE;
        }
        self.normalize();
        bit
    }
}

// ------------------------------------------------- binary-tree symbol coding
//
// An n-bit symbol is coded MSB-first through a binary tree of adaptive bit
// models; `probs` holds 1 << n entries, index 0 unused.

#[inline]
pub fn encode_sym(enc: &mut RangeEncoder, probs: &mut [u16], nbits: u32, sym: u32) {
    let mut ctx = 1usize;
    for i in (0..nbits).rev() {
        let bit = (sym >> i) & 1;
        enc.encode_bit(&mut probs[ctx], bit);
        ctx = (ctx << 1) | bit as usize;
    }
}

#[inline]
pub fn decode_sym(dec: &mut RangeDecoder, probs: &mut [u16], nbits: u32) -> u32 {
    let mut ctx = 1usize;
    for _ in 0..nbits {
        let bit = dec.decode_bit(&mut probs[ctx]);
        ctx = (ctx << 1) | bit as usize;
    }
    (ctx - (1usize << nbits)) as u32
}
