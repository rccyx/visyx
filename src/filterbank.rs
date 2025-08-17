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

    let mut mel_points = Vec::with_capacity(bands + 2);
    for i in 0..(bands + 2) {
        mel_points.push(
            mmin + (i as f32) * (mmax - mmin) / (bands as f32 + 1.0),
        );
    }
    let hz_points: Vec<f32> =
        mel_points.into_iter().map(mel_to_hz).collect();

    let mut bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| {
            let mut b = (hz / hz_per_bin).round() as isize;
            if b < 1 {
                b = 1;
            }
            if b as usize >= half {
                b = (half - 1) as isize;
            }
            b as usize
        })
        .collect();

    for i in 1..bin_points.len() {
        if bin_points[i] <= bin_points[i - 1] {
            bin_points[i] = (bin_points[i - 1] + 1).min(half - 1);
        }
    }

    let mut filters = Vec::with_capacity(bands);
    for b in 0..bands {
        let l = bin_points[b];
        let c = bin_points[b + 1];
        let r = bin_points[b + 2];
        let mut taps = Vec::new();
        for i in l..=c {
            let w = if c == l {
                0.0
            } else {
                (i - l) as f32 / (c - l) as f32
            };
            taps.push((i, w));
        }
        for i in c..=r {
            let w = if r == c {
                0.0
            } else {
                1.0 - (i - c) as f32 / (r - c) as f32
            };
            taps.push((i, w));
        }
        let sumw =
            taps.iter().map(|(_, w)| *w).sum::<f32>().max(1e-6);
        for t in &mut taps {
            t.1 /= sumw;
        }
        let center_hz = c as f32 * hz_per_bin;
        filters.push(Tri { taps, center_hz });
    }
    filters
} 