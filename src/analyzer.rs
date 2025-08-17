use crate::dsp::ema_tc;
use crate::filterbank::Tri;

pub struct SpectrumAnalyzer {
    pub spec_pow_smooth: Vec<f32>,
    pub filters: Vec<Tri>,
    pub bars_y: Vec<f32>,
    pub bars_v: Vec<f32>,
    pub eq_ref: Vec<f32>,
    pub db_low: f32,
    pub db_high: f32,
}

impl SpectrumAnalyzer {
    pub fn new(half_fft_size: usize) -> Self {
        Self {
            spec_pow_smooth: vec![0.0; half_fft_size],
            filters: Vec::new(),
            bars_y: Vec::new(),
            bars_v: Vec::new(),
            eq_ref: Vec::new(),
            db_low: -60.0,
            db_high: -20.0,
        }
    }

    pub fn resize(&mut self, num_bars: usize) {
        if self.bars_y.len() != num_bars {
            self.bars_y = vec![0.0; num_bars];
            self.bars_v = vec![0.0; num_bars];
            self.eq_ref = vec![1e-6; num_bars];
        }
    }

    pub fn update_spectrum(&mut self, spec_pow: &[f32], tau_spec: f32, dt_s: f32) {
        for i in 0..self.spec_pow_smooth.len().min(spec_pow.len()) {
            self.spec_pow_smooth[i] = ema_tc(
                self.spec_pow_smooth[i],
                spec_pow[i].max(1e-12),
                tau_spec,
                dt_s,
            );
        }
    }

    pub fn analyze_bands(&mut self, tilt_alpha: f32, dt_s: f32, gate_open: bool) -> Vec<f32> {
        let mut db_per_band = vec![0.0f32; self.filters.len()];
        
        for (i, tri) in self.filters.iter().enumerate() {
            let mut acc = 0.0f32;
            for &(idx, wgt) in &tri.taps {
                if idx < self.spec_pow_smooth.len() {
                    acc += self.spec_pow_smooth[idx] * wgt;
                }
            }
            let amp = acc.sqrt();

            let tilt = (tri.center_hz / 1000.0).max(0.001).powf(tilt_alpha);
            let amp_tilted = amp * tilt;

            self.eq_ref[i] = ema_tc(self.eq_ref[i], amp_tilted, 6.0, dt_s).max(1e-9);
            let rel = amp_tilted / self.eq_ref[i];

            db_per_band[i] = 20.0 * rel.max(1e-12).log10(); // relative dB
        }

        self.update_db_range(&db_per_band, dt_s);
        
        let low = self.db_low - 3.0;
        let high = self.db_high + 6.0;
        let range = (high - low).max(12.0);

        let mut bars_target = vec![0.0f32; db_per_band.len()];
        for i in 0..db_per_band.len() {
            let mut v = (db_per_band[i] - low) / range;
            if !gate_open {
                v = 0.0;
            }
            // soft comp: gamma < 1 brightens lows, then soft knee near 1
            v = v.clamp(0.0, 1.0).powf(0.85);
            v = 1.0 - (1.0 - v).powf(1.6);
            bars_target[i] = v;
        }
        
        bars_target
    }
    
    pub fn update_db_range(&mut self, db_per_band: &[f32], dt_s: f32) {
        let mut sorted = db_per_band.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p = |q: f32| -> f32 {
            if sorted.is_empty() {
                return -60.0;
            }
            let idx = ((sorted.len() - 1) as f32 * q).round() as usize;
            sorted[idx]
        };
        
        let q10 = p(0.10);
        let q90 = p(0.90);

        // Smooth the window. Low follows fairly quick, high a bit slower to avoid pumping.
        self.db_low = ema_tc(self.db_low, q10, 0.30, dt_s);
        self.db_high = ema_tc(self.db_high, q90, 0.50, dt_s);
    }
    
    pub fn apply_flow_and_spring(&mut self, bars_target: &[f32], flow_k: f32, spr_k: f32, spr_zeta: f32, dt_s: f32) {
        let n = bars_target.len();
        let mut flowed = vec![0.0f32; n];
        
        for i in 0..n {
            let left = if i > 0 { self.bars_y[i - 1] } else { self.bars_y[i] };
            let right = if i + 1 < n { self.bars_y[i + 1] } else { self.bars_y[i] };
            let flow = flow_k * (left + right - 2.0 * self.bars_y[i]);
            flowed[i] = (bars_target[i] + flow).clamp(0.0, 1.0);
        }

        // spring smoothing
        let c = 2.0 * (spr_k).sqrt() * spr_zeta;
        for i in 0..flowed.len() {
            let x = flowed[i];
            let a = spr_k * (x - self.bars_y[i]) - c * self.bars_v[i];
            self.bars_v[i] += a * dt_s;
            self.bars_y[i] = (self.bars_y[i] + self.bars_v[i] * dt_s).clamp(0.0, 1.0);
        }
    }
} 