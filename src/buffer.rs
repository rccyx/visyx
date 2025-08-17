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
    
    pub fn push(&mut self, x: f32) {
        self.data[self.write_idx] = x;
        self.write_idx = (self.write_idx + 1) % self.data.len();
        if self.write_idx == 0 {
            self.filled = true;
        }
    }
    
    pub fn latest(&self) -> Vec<f32> {
        if !self.filled {
            return self.data[..self.write_idx].to_vec();
        }
        let mut v = Vec::with_capacity(self.data.len());
        v.extend_from_slice(&self.data[self.write_idx..]);
        v.extend_from_slice(&self.data[..self.write_idx]);
        v
    }
} 