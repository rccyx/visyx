use crate::dsp::{hz_to_mel, mel_to_hz};

#[derive(Clone)]
pub struct Tri {
    pub taps: Vec<(usize, f32)>,
    pub center_hz: f32,
}

pub fn build_filterbank(
    sr: f32,
    fft_size: usize,
    bands: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Tri> {
    let half = fft_size / 2;
    let hz_per_bin = sr / fft_size as f32;
    let mmin = hz_to_mel(fmin.max(hz_per_bin));
    let mmax = hz_to_mel(fmax.min(sr * 0.5 - hz_per_bin));
    let mel_step = (mmax - mmin) / (bands as f32 + 1.0);

    // Pre-calculate mel points
    let mut mel_points = Vec::with_capacity(bands + 2);
    for i in 0..(bands + 2) {
        mel_points.push(mmin + (i as f32) * mel_step);
    }
    
    // Convert to Hz
    let mut hz_points = Vec::with_capacity(bands + 2);
    for &mel in &mel_points {
        hz_points.push(mel_to_hz(mel));
    }

    // Convert to bin indices
    let mut bin_points = Vec::with_capacity(bands + 2);
    for &hz in &hz_points {
        let mut b = (hz / hz_per_bin).round() as isize;
        if b < 1 {
            b = 1;
        }
        if b as usize >= half {
            b = (half - 1) as isize;
        }
        bin_points.push(b as usize);
    }

    // Ensure monotonically increasing bin points
    for i in 1..bin_points.len() {
        if bin_points[i] <= bin_points[i - 1] {
            bin_points[i] = (bin_points[i - 1] + 1).min(half - 1);
        }
    }

    // Build triangular filters
    let mut filters = Vec::with_capacity(bands);
    for b in 0..bands {
        let l = bin_points[b];
        let c = bin_points[b + 1];
        let r = bin_points[b + 2];
        
        let mut taps = Vec::with_capacity(r - l + 1);
        
        // Left slope (ascending)
        let lc_diff = c - l;
        if lc_diff > 0 {
            let lc_diff_f = lc_diff as f32;
            for i in l..=c {
                let w = (i - l) as f32 / lc_diff_f;
                taps.push((i, w));
            }
        }
        
        // Right slope (descending)
        let cr_diff = r - c;
        if cr_diff > 0 {
            let cr_diff_f = cr_diff as f32;
            for i in (c + 1)..=r {
                let w = 1.0 - (i - c - 1) as f32 / cr_diff_f;
                taps.push((i, w));
            }
        }
        
        // Normalize weights
        let sumw = taps.iter().map(|(_, w)| *w).sum::<f32>().max(1e-6);
        let inv_sumw = 1.0 / sumw;
        for t in &mut taps {
            t.1 *= inv_sumw;
        }
        
        let center_hz = c as f32 * hz_per_bin;
        filters.push(Tri { taps, center_hz });
    }
    
    filters
}
