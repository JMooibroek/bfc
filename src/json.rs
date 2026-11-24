// Minimal JSON scanner, sufficient for safetensors headers.
// Handles: objects, arrays, strings, numbers, booleans/null (skipped).
// The header block is stored verbatim in the compressed file; this parser is
// only used to extract dtypes and data_offsets at compress time.

pub struct Json<'a> {
    pub b: &'a [u8],
    pub pos: usize,
}

impl<'a> Json<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Json { b, pos: 0 }
    }

    fn ws(&mut self) {
        while self.pos < self.b.len() && matches!(self.b[self.pos], b' ' | b'\t' | b'\r' | b'\n') {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.b.get(self.pos).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), &'static str> {
        match self.peek() {
            Some(x) if x == c => {
                self.pos += 1;
                Ok(())
            }
            _ => Err("json: expected char"),
        }
    }

    pub fn parse_string(&mut self) -> Result<String, &'static str> {
        self.expect(b'"')?;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = self.b.get(self.pos).copied().ok_or("json: unterminated string")?;
            self.pos += 1;
            match c {
                b'"' => return Ok(String::from_utf8_lossy(&out).into_owned()),
                b'\\' => {
                    let e = self.b.get(self.pos).copied().ok_or("json: bad escape")?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'u' => {
                            if self.pos + 4 > self.b.len() {
                                return Err("json: bad \\u");
                            }
                            let hex = std::str::from_utf8(&self.b[self.pos..self.pos + 4])
                                .map_err(|_| "json: bad \\u")?;
                            let cp =
                                u32::from_str_radix(hex, 16).map_err(|_| "json: bad \\u")?;
                            self.pos += 4;
                            // Basic multilingual plane only; safetensors keys in practice are ASCII.
                            let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err("json: bad escape"),
                    }
                }
                _ => out.push(c),
            }
        }
    }

    pub fn parse_u64(&mut self) -> Result<u64, &'static str> {
        self.ws();
        let start = self.pos;
        while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start {
            return Err("json: expected number");
        }
        std::str::from_utf8(&self.b[start..self.pos])
            .ok()
            .and_then(|s| s.parse().ok())
            .ok_or("json: bad number")
    }

    pub fn skip_value(&mut self) -> Result<(), &'static str> {
        match self.peek().ok_or("json: expected value")? {
            b'"' => {
                self.parse_string()?;
                Ok(())
            }
            b'{' => {
                self.pos += 1;
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    return Ok(());
                }
                loop {
                    self.parse_string()?;
                    self.expect(b':')?;
                    self.skip_value()?;
                    match self.peek().ok_or("json: unterminated object")? {
                        b',' => self.pos += 1,
                        b'}' => {
                            self.pos += 1;
                            return Ok(());
                        }
                        _ => Err("json: expected , or }")?,
                    }
                }
            }
            b'[' => {
                self.pos += 1;
                if self.peek() == Some(b']') {
                    self.pos += 1;
                    return Ok(());
                }
                loop {
                    self.skip_value()?;
                    match self.peek().ok_or("json: unterminated array")? {
                        b',' => self.pos += 1,
                        b']' => {
                            self.pos += 1;
                            return Ok(());
                        }
                        _ => Err("json: expected , or ]")?,
                    }
                }
            }
            _ => {
                // number / true / false / null: consume until a delimiter
                while self.pos < self.b.len()
                    && !matches!(self.b[self.pos], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n')
                {
                    self.pos += 1;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub start: u64,
    pub end: u64,
}

/// Parse a safetensors header (JSON bytes, without the leading length field).
/// Returns tensors in file (offset) order, or None on any structural problem.
pub fn parse_safetensors_header(header: &[u8]) -> Option<Vec<TensorInfo>> {
    let mut p = Json::new(header);
    p.expect(b'{').ok()?;
    let mut tensors: Vec<TensorInfo> = Vec::new();
    if p.peek() == Some(b'}') {
        return Some(tensors);
    }
    loop {
        let name = p.parse_string().ok()?;
        p.expect(b':').ok()?;
        if name == "__metadata__" {
            p.skip_value().ok()?;
        } else {
            let mut dtype: Option<String> = None;
            let mut offs: Option<(u64, u64)> = None;
            p.expect(b'{').ok()?;
            if p.peek() == Some(b'}') {
                return None; // tensor entry without fields: invalid
            }
            loop {
                let key = p.parse_string().ok()?;
                p.expect(b':').ok()?;
                match key.as_str() {
                    "dtype" => dtype = Some(p.parse_string().ok()?),
                    "data_offsets" => {
                        p.expect(b'[').ok()?;
                        let s = p.parse_u64().ok()?;
                        p.expect(b',').ok()?;
                        let e = p.parse_u64().ok()?;
                        p.expect(b']').ok()?;
                        offs = Some((s, e));
                    }
                    _ => p.skip_value().ok()?,
                }
                match p.peek()? {
                    b',' => p.pos += 1,
                    b'}' => {
                        p.pos += 1;
                        break;
                    }
                    _ => return None,
                }
            }
            let dtype = dtype?;
            let (start, end) = offs?;
            if end < start {
                return None;
            }
            tensors.push(TensorInfo {
                name,
                dtype,
                start,
                end,
            });
        }
        match p.peek()? {
            b',' => p.pos += 1,
            b'}' => {
                p.pos += 1;
                break;
            }
            _ => return None,
        }
    }
    // trailing whitespace / garbage after '}' is tolerated by skipping
    tensors.sort_by_key(|t| t.start);
    Some(tensors)
}
