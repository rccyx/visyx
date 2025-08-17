use rustfft::num_complex::Complex;

pub fn hann(n: usize) -> Vec<f32> {
    let den = (n.max(2) - 1) as f32;
    (0..n)
        .map(|i| {
            0.5 - 0.5
                * f32::cos(
                    2.0 * std::f32::consts::PI * i as f32 / den,
                )
        })
        .collect()
}

#[inline]
pub fn ema_tc(prev: f32, x: f32, tau_s: f32, dt_s: f32) -> f32 {
    let a = (-dt_s / tau_s).exp();
    a * prev + (1.0 - a) * x
}

pub fn hz_to_mel(f: f32) -> f32 {
    2595.0 * ((1.0 + f / 700.0).log10())
}

pub fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

pub fn prepare_fft_input(
    samples: &[f32],
    window: &[f32],
) -> Vec<Complex<f32>> {
    samples
        .iter()
        .zip(window.iter())
        .map(|(x, w)| Complex { re: x * w, im: 0.0 })
        .collect()
}
