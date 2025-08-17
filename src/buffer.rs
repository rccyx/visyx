pub struct SharedBuf {
    data: Vec<f32>,
    write_idx: usize,
    filled: bool,
}

impl SharedBuf {
    pub fn new(cap: usize) -> Self {
        Self {
            data: vec![0.0; cap],
            write_idx: 0,
            filled: false,
        }
    }

    #[inline]
    pub fn push(&mut self, x: f32) {
        self.data[self.write_idx] = x;
        self.write_idx = (self.write_idx + 1) % self.data.len();
        if self.write_idx == 0 {
            self.filled = true;
        }
    }

    pub fn latest(&self) -> Vec<f32> {
        let len = if self.filled {
            self.data.len()
        } else {
            self.write_idx
        };
        
        if len == 0 {
            return Vec::new();
        }
        
        let mut result = Vec::with_capacity(len);
        
        if self.filled {
            result.extend_from_slice(&self.data[self.write_idx..]);
            result.extend_from_slice(&self.data[..self.write_idx]);
        } else {
            result.extend_from_slice(&self.data[..self.write_idx]);
        }
        
        result
    }
}
