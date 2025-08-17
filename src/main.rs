use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, SampleFormat, SizedSample, StreamConfig};
use crossterm::{
    cursor, execute,
    style::{Color, SetForegroundColor},
    terminal::{self, ClearType},
};
use rustfft::num_traits::ToPrimitive;
use rustfft::{num_complex::Complex, FftPlanner};
use std::{
    cmp::min,
    env,
    io::{stdout, Stdout, Write},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

struct SharedBuf {
    data: Vec<f32>,
    write_idx: usize,
    filled: bool,
}
impl SharedBuf {
    fn new(cap: usize) -> Self {
        Self { data: vec![0.0; cap], write_idx: 0, filled: false }
    }
    fn push(&mut self, x: f32) {
        self.data[self.write_idx] = x;
        self.write_idx = (self.write_idx + 1) % self.data.len();
        if self.write_idx == 0 {
            self.filled = true;
        }
    }
    fn latest(&self) -> Vec<f32> {
        if !self.filled {
            return self.data[..self.write_idx].to_vec();
        }
        let mut v = Vec::with_capacity(self.data.len());
        v.extend_from_slice(&self.data[self.write_idx..]);
        v.extend_from_slice(&self.data[..self.write_idx]);
        v
    }
}

fn pick_input_device() -> Result<Device> {
    let host = cpal::default_host();
    if let Ok(want) = env::var("MYCAVA_DEVICE") {
        for dev in host.input_devices()? {
            if dev.name()?.to_lowercase().contains(&want.to_lowercase()) {
                return Ok(dev);
            }
        }
        anyhow::bail!("MYCAVA_DEVICE='{}' not found", want);
    }
    host.default_input_device().context("No default input device")
}

fn best_config_for(device: &Device) -> Result<StreamConfig> {
    let mut cfg = device.default_input_config()?.config();
    cfg.sample_rate.0 = cfg.sample_rate.0.clamp(44_100, 48_000);
    Ok(cfg)
}

fn build_stream<T>(device: Device, cfg: StreamConfig, shared: Arc<Mutex<SharedBuf>>) -> Result<cpal::Stream>
where
    T: Sample + SizedSample + ToPrimitive,
{
    let channels = cfg.channels as usize;
    let err_fn = |e| eprintln!("Stream error: {}", e);
    let stream = device.build_input_stream(
        &cfg,
        move |data: &[T], _| {
            let mut buf = shared.lock().unwrap();
            for frame in data.chunks_exact(channels) {
                let mut acc: f32 = 0.0;
                for &s in frame.iter() {
                    acc += s.to_f32().unwrap_or(0.0);
                }
                buf.push(acc / channels as f32);
            }
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * f32::cos(2.0 * std::f32::consts::PI * i as f32 / n as f32))
        .collect()
}

/* ---------------- Mel-ish triangular filterbank ---------------- */

#[derive(Clone)]
struct Tri {
    // (bin index, weight)
    taps: Vec<(usize, f32)>,
}

fn hz_to_mel(f: f32) -> f32 {
    2595.0 * ((1.0 + f / 700.0).log10())
}
fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

fn build_filterbank(sr: f32, fft_size: usize, bands: usize, fmin: f32, fmax: f32) -> Vec<Tri> {
    let half = fft_size / 2;
    let hz_per_bin = sr / fft_size as f32;

    let mmin = hz_to_mel(fmin.max(hz_per_bin));
    let mmax = hz_to_mel(fmax.min(sr * 0.5 - hz_per_bin));
    let mut mel_points = Vec::with_capacity(bands + 2);
    for i in 0..(bands + 2) {
        let t = i as f32 / (bands as f32 + 1.0);
        mel_points.push(mmin + t * (mmax - mmin));
    }
    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz).collect();
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
        let left = bin_points[b];
        let center = bin_points[b + 1];
        let right = bin_points[b + 2];
        let mut taps = Vec::new();
        for i in left..=center {
            let w = if center == left {
                0.0
            } else {
                (i - left) as f32 / (center - left) as f32
            };
            taps.push((i, w));
        }
        for i in center..=right {
            let w = if right == center {
                0.0
            } else {
                1.0 - (i - center) as f32 / (right - center) as f32
            };
            taps.push((i, w));
        }
        let sumw: f32 = taps.iter().map(|(_, w)| *w).sum::<f32>().max(1e-6);
        for t in &mut taps {
            t.1 /= sumw;
        }
        filters.push(Tri { taps });
    }
    filters
}

/* ---------------- Drawing ---------------- */

fn draw_frame(out: &mut Stdout, bars: &[f32], w: u16, h: u16) -> std::io::Result<()> {
    let bar_count = min(bars.len() as u16, w.saturating_sub(2));
    let max_h = h.saturating_sub(2);

    execute!(out, cursor::MoveTo(0, 1))?;
    let mut buf = String::with_capacity(((bar_count as usize) + 3) * ((max_h as usize) + 3));

    for row in (0..max_h).rev() {
        buf.push(' ');
        for i in 0..bar_count {
            let v = bars[i as usize].clamp(0.0, 1.0);
            let filled = (v * max_h as f32).round() as u16;
            if filled > row {
                buf.push('█');
            } else {
                buf.push(' ');
            }
        }
        buf.push('\n');
    }
    buf.push(' ');
    for _ in 0..bar_count {
        buf.push('▁');
    }
    buf.push('\n');

    out.write_all(buf.as_bytes())?;
    out.flush()?;
    Ok(())
}

/* ---------------- Main ---------------- */

fn main() -> Result<()> {
    // Tunables
    const FMIN: f32 = 30.0;
    const FMAX: f32 = 16_000.0;
    const TARGET_FPS_MS: u64 = 16;
    const FFT_SIZE: usize = 2048;
    const SPEC_EMA: f32 = 0.75;
    const BAR_ATTACK: f32 = 0.55;
    const BAR_DECAY: f32 = 0.88;
    const NORM_EMA: f32 = 0.92;
    const LOG_COMP: f32 = 250.0;
    const FLOOR_FRAC: f32 = 0.05;

    let mut out = stdout();
    terminal::enable_raw_mode()?;
    execute!(
        out,
        terminal::EnterAlternateScreen,
        cursor::Hide,
        terminal::Clear(ClearType::All),
        SetForegroundColor(Color::White),
    )?;
    let cleanup = scopeguard::guard((), |_| {
        let mut out = stdout();
        let _ = execute!(out, cursor::Show, terminal::LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    });

    // Audio
    let device = pick_input_device()?;
    let name = device.name().unwrap_or_else(|_| "<unknown>".into());
    let cfg = best_config_for(&device)?;
    let sample_rate = cfg.sample_rate.0 as f32;

    let ring_len = (sample_rate as usize / 10).max(FFT_SIZE * 2);
    let shared = Arc::new(Mutex::new(SharedBuf::new(ring_len)));
    let stream = match device.default_input_config()?.sample_format() {
        SampleFormat::F32 => build_stream::<f32>(device, cfg.clone(), shared.clone())?,
        SampleFormat::I16 => build_stream::<i16>(device, cfg.clone(), shared.clone())?,
        SampleFormat::U16 => build_stream::<u16>(device, cfg.clone(), shared.clone())?,
        _ => anyhow::bail!("Unsupported sample format"),
    };
    stream.play()?;

    // FFT
    let window = hann_window(FFT_SIZE);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let half = FFT_SIZE / 2;

    // State
    let mut last = Instant::now();
    let target_dt = Duration::from_millis(TARGET_FPS_MS);
    let mut spec_pow_smooth: Vec<f32> = vec![0.0; half]; // EMA on power spectrum
    let mut bars_state: Vec<f32> = Vec::new();
    let mut filters: Vec<Tri> = Vec::new();
    let mut norm_max = 0.1f32;

    loop {
        if crossterm::event::poll(Duration::from_millis(0))? {
            if let crossterm::event::Event::Key(k) = crossterm::event::read()? {
                if let crossterm::event::KeyCode::Char('q') = k.code {
                    break;
                }
            }
        }

        let now = Instant::now();
        if now.duration_since(last) < target_dt {
            thread::sleep(Duration::from_millis(1));
            continue;
        }
        last = now;

        let samples = { shared.lock().unwrap().latest() };
        if samples.len() < FFT_SIZE {
            continue;
        }

        // FFT on latest window
        let tail = &samples[samples.len() - FFT_SIZE..];
        let mut buf: Vec<Complex<f32>> = tail
            .iter()
            .zip(window.iter())
            .map(|(x, w)| Complex { re: x * w, im: 0.0 })
            .collect();
        fft.process(&mut buf);

        // Power spectrum + EMA smoothing
        for i in 0..half {
            let re = buf[i].re;
            let im = buf[i].im;
            let p = (re * re + im * im) / (FFT_SIZE as f32 * FFT_SIZE as f32);
            spec_pow_smooth[i] =
                SPEC_EMA * spec_pow_smooth[i] + (1.0 - SPEC_EMA) * p.max(1e-12);
        }

        // Bars from filterbank
        let (w, h) = terminal::size()?;
        let bands = (w.saturating_sub(2)).max(10) as usize;

        if filters.len() != bands {
            filters = build_filterbank(sample_rate, FFT_SIZE, bands, FMIN, FMAX);
            bars_state = vec![0.0; bands];
        }

        let mut bar_vals = vec![0.0f32; bands];
        for (i, tri) in filters.iter().enumerate() {
            let mut acc = 0.0f32;
            for &(idx, wgt) in &tri.taps {
                acc += spec_pow_smooth[idx] * wgt;
            }
            let amp = acc.sqrt();
            let v = (amp * LOG_COMP + 1.0).ln() / (LOG_COMP + 1.0).ln(); // 0..1
            bar_vals[i] = v;
        }

        // Stable normalization with EMA of the max and a small floor
        let frame_max = bar_vals.iter().cloned().fold(0.0f32, f32::max);
        norm_max = NORM_EMA * norm_max + (1.0 - NORM_EMA) * frame_max.max(1e-6);
        let floor = norm_max * FLOOR_FRAC;
        let scale = (norm_max - floor).max(1e-6);

        for i in 0..bands {
            let v = ((bar_vals[i] - floor) / scale).clamp(0.0, 1.0);
            if v > bars_state[i] {
                bars_state[i] =
                    BAR_ATTACK * bars_state[i] + (1.0 - BAR_ATTACK) * v;
            } else {
                bars_state[i] *= BAR_DECAY;
                if v > bars_state[i] {
                    bars_state[i] = v;
                }
            }
        }

        execute!(out, terminal::Clear(ClearType::All))?;
        {
            let header = format!("  mycava  |  input: {}  |  q to quit\n", name);
            out.write_all(header.as_bytes())?;
            out.flush()?;
        }
        draw_frame(&mut out, &bars_state, w, h)?;
    }; // make loop a statement so main returns Result

    drop(cleanup);
    Ok(())
}

/* ---------------- Scopeguard ---------------- */

mod scopeguard {
    pub fn guard<T, F: FnOnce(T)>(v: T, f: F) -> Guard<T, F> {
        Guard { v: Some(v), f: Some(f) }
    }
    pub struct Guard<T, F: FnOnce(T)> {
        v: Option<T>,
        f: Option<F>,
    }
    impl<T, F: FnOnce(T)> Drop for Guard<T, F> {
        fn drop(&mut self) {
            if let (Some(v), Some(f)) = (self.v.take(), self.f.take()) {
                f(v);
            }
        }
    }
}
