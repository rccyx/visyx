use rustfft::num_complex::Complex;

pub fn hann(n: usize) -> Vec<f32> {
    let den = (n.max(2) - 1) as f32;
    let pi2_div_den = 2.0 * std::f32::consts::PI / den;
    (0..n)
        .map(|i| 0.5 - 0.5 * f32::cos(pi2_div_den * i as f32))
        .collect()
}

#[inline]
pub fn ema_tc(prev: f32, x: f32, tau_s: f32, dt_s: f32) -> f32 {
    let a = (-dt_s / tau_s).exp();
    a * prev + (1.0 - a) * x
}

#[inline]
pub fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

#[inline]
pub fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

pub fn prepare_fft_input(
    samples: &[f32],
    window: &[f32],
) -> Vec<Complex<f32>> {
    let mut result = Vec::with_capacity(samples.len());
    for (i, &x) in samples.iter().enumerate() {
        result.push(Complex {
            re: x * window[i],
            im: 0.0,
        });
    }
    result
}

#[inline]
pub fn prepare_fft_input_inplace(
    samples: &[f32],
    window: &[f32],
    buf: &mut Vec<Complex<f32>>,
) {
    buf.clear();
    buf.reserve(samples.len());
    for (i, &x) in samples.iter().enumerate() {
        buf.push(Complex {
            re: x * window[i],
            im: 0.0,
        });
    }
}
