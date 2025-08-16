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
        Self {
            data: vec![0.0; cap],
            write_idx: 0,
            filled: false,
        }
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
            if dev
                .name()?
                .to_lowercase()
                .contains(&want.to_lowercase())
            {
                return Ok(dev);
            }
        }
        anyhow::bail!("MYCAVA_DEVICE='{}' not found", want);
    }
    host.default_input_device()
        .context("No default input device")
}

fn best_config_for(device: &Device) -> Result<StreamConfig> {
    let mut cfg = device.default_input_config()?.config();
    cfg.sample_rate.0 = cfg.sample_rate.0.clamp(44_100, 48_000);
    Ok(cfg)
}

fn build_stream<T>(
    device: Device,
    cfg: StreamConfig,
    shared: Arc<Mutex<SharedBuf>>,
) -> Result<cpal::Stream>
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
        .map(|i| {
            0.5 - 0.5
                * f32::cos(
                    2.0 * std::f32::consts::PI * i as f32
                        / (n as f32),
                )
        })
        .collect()
}

#[derive(Clone)]
struct BarBins {
    start: usize,
    end: usize, // exclusive
}

/// Build log-spaced mapping from bars to FFT bin ranges.
fn make_bar_bins(
    sr: f32,
    fft_size: usize,
    bars: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<BarBins> {
    let half = fft_size / 2;
    let hz_per_bin = sr / fft_size as f32;

    let log_min = fmin.ln();
    let log_max = fmax.ln();
    let mut bins = Vec::with_capacity(bars);

    for b in 0..bars {
        let t0 = b as f32 / bars as f32;
        let t1 = (b + 1) as f32 / bars as f32;
        let f0 = (log_min + t0 * (log_max - log_min)).exp();
        let f1 = (log_min + t1 * (log_max - log_min)).exp();
        let mut i0 =
            ((f0 / hz_per_bin).floor() as isize).max(1) as usize;
        let mut i1 =
            ((f1 / hz_per_bin).ceil() as isize).max(2) as usize;
        if i0 >= half {
            i0 = half - 1;
        }
        if i1 > half {
            i1 = half;
        }
        if i1 <= i0 {
            i1 = i0 + 1;
        }
        bins.push(BarBins { start: i0, end: i1 });
    }
    bins
}

/// RMS in a bin range with a tiny DC guard.
fn bucket_rms(spec: &[f32], start: usize, end: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0usize;
    for i in start..end {
        let v = spec[i];
        sum += v * v;
        n += 1;
    }
    let rms = if n > 0 { (sum / n as f32).sqrt() } else { 0.0 };
    // avoid log blowups downstream
    rms.max(1e-8)
}

fn draw_frame(
    out: &mut Stdout,
    bars: &[f32],
    peaks: &[f32],
    w: u16,
    h: u16,
) -> std::io::Result<()> {
    let bar_count = min(bars.len() as u16, w.saturating_sub(2));
    let max_h = h.saturating_sub(2);

    execute!(out, cursor::MoveTo(0, 1))?;
    let mut buf = String::with_capacity(
        ((bar_count as usize) + 3) * ((max_h as usize) + 3),
    );

    for row in (0..max_h).rev() {
        buf.push(' ');
        for i in 0..bar_count {
            let v = bars[i as usize].clamp(0.0, 1.0);
            let filled = (v * max_h as f32).round() as u16;
            // Peak dot one row above the fill
            let peak_row = (peaks[i as usize].clamp(0.0, 1.0)
                * max_h as f32)
                .round() as i32;

            if (row as i32) == peak_row {
                buf.push('▪');
            } else if filled > row {
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

fn main() -> Result<()> {
    // Look and feel
    const FMIN: f32 = 25.0;
    const FMAX: f32 = 16_000.0;
    const TARGET_FPS_MS: u64 = 16; // ~60 fps
    const FFT_SIZE: usize = 2048; // more resolution than 1024
    const SPEC_EMA: f32 = 0.7; // spectral smoothing (higher = smoother)
    const BAR_ATTACK: f32 = 0.6; // bar smoothing attack
    const BAR_DECAY: f32 = 0.85; // bar decay per frame
    const PEAK_DECAY: f32 = 0.96; // peak gravity
    const NORM_EMA: f32 = 0.92; // normalization max smoothing
    const FLOOR_FRAC: f32 = 0.06; // noise floor as fraction of max

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
        let _ = execute!(
            out,
            cursor::Show,
            terminal::LeaveAlternateScreen
        );
        let _ = terminal::disable_raw_mode();
    });

    // Audio
    let device = pick_input_device()?;
    let name = device.name().unwrap_or_else(|_| "<unknown>".into());
    let cfg = best_config_for(&device)?;
    let sample_rate = cfg.sample_rate.0 as f32;

    let ring_len = (sample_rate as usize / 10).max(FFT_SIZE * 2);
    let shared = Arc::new(Mutex::new(SharedBuf::new(ring_len)));

    let stream = match device.default_input_config()?.sample_format()
    {
        SampleFormat::F32 => {
            build_stream::<f32>(device, cfg.clone(), shared.clone())?
        }
        SampleFormat::I16 => {
            build_stream::<i16>(device, cfg.clone(), shared.clone())?
        }
        SampleFormat::U16 => {
            build_stream::<u16>(device, cfg.clone(), shared.clone())?
        }
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
    let mut spec_smooth: Vec<f32> = vec![0.0; half]; // EMA on spectrum
    let mut bars_state: Vec<f32> = Vec::new();
    let mut peaks: Vec<f32> = Vec::new();
    let mut bar_bins: Vec<BarBins> = Vec::new();
    let mut norm_max = 0.1f32;

    loop {
        if crossterm::event::poll(Duration::from_millis(0))? {
            if let crossterm::event::Event::Key(k) =
                crossterm::event::read()?
            {
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

        // Take FFT of the latest window
        let tail = &samples[samples.len() - FFT_SIZE..];
        let mut buf: Vec<Complex<f32>> = tail
            .iter()
            .zip(window.iter())
            .map(|(x, w)| Complex { re: x * w, im: 0.0 })
            .collect();
        fft.process(&mut buf);

        // Magnitude spectrum and EMA smoothing
        for i in 0..half {
            let mag = (buf[i].re * buf[i].re + buf[i].im * buf[i].im)
                .sqrt()
                / FFT_SIZE as f32;
            spec_smooth[i] = SPEC_EMA * spec_smooth[i]
                + (1.0 - SPEC_EMA) * mag.max(1e-9);
        }

        // Bars layout depends on terminal width
        let (w, h) = terminal::size()?;
        let bars = (w.saturating_sub(2)).max(10) as usize;

        if bar_bins.len() != bars {
            bar_bins = make_bar_bins(
                sample_rate,
                FFT_SIZE,
                bars,
                FMIN,
                FMAX,
            );
            bars_state = vec![0.0; bars];
            peaks = vec![0.0; bars];
        }

        // Bucket with RMS
        let mut bar_vals = vec![0.0f32; bars];
        for (i, bb) in bar_bins.iter().enumerate() {
            bar_vals[i] = bucket_rms(&spec_smooth, bb.start, bb.end);
        }

        // Gentle spatial smoothing across neighbors
        if bars >= 3 {
            let mut tmp = bar_vals.clone();
            for i in 1..bars - 1 {
                tmp[i] = 0.25 * bar_vals[i - 1]
                    + 0.5 * bar_vals[i]
                    + 0.25 * bar_vals[i + 1];
            }
            bar_vals = tmp;
        }

        // Stable normalization with EMA of max and a small floor
        let frame_max =
            bar_vals.iter().cloned().fold(0.0f32, f32::max);
        norm_max = NORM_EMA * norm_max
            + (1.0 - NORM_EMA) * frame_max.max(1e-6);
        let floor = norm_max * FLOOR_FRAC;

        // Bar envelope with attack/decay and peak hold
        for i in 0..bars {
            let v = ((bar_vals[i] - floor) / (norm_max - floor))
                .clamp(0.0, 1.0);
            if v > bars_state[i] {
                bars_state[i] = BAR_ATTACK * bars_state[i]
                    + (1.0 - BAR_ATTACK) * v;
            } else {
                bars_state[i] *= BAR_DECAY;
                if v > bars_state[i] {
                    bars_state[i] = v;
                }
            }
            peaks[i] *= PEAK_DECAY;
            if bars_state[i] > peaks[i] {
                peaks[i] = bars_state[i];
            }
        }

        // Draw
        execute!(out, terminal::Clear(ClearType::All))?;
        {
            let header = format!(
                "  mycava  |  input: {}  |  q to quit\n",
                name
            );
            out.write_all(header.as_bytes())?;
            out.flush()?;
        }
        draw_frame(&mut out, &bars_state, &peaks, w, h)?;
    }

    drop(cleanup);
    Ok(())
}

mod scopeguard {
    pub fn guard<T, F: FnOnce(T)>(v: T, f: F) -> Guard<T, F> {
        Guard {
            v: Some(v),
            f: Some(f),
        }
    }
    pub struct Guard<T, F: FnOnce(T)> {
        v: Option<T>,
        f: Option<F>,
    }
    impl<T, F: FnOnce(T)> Drop for Guard<T, F> {
        fn drop(&mut self) {
            if let (Some(v), Some(f)) = (self.v.take(), self.f.take())
            {
                f(v);
            }
        }
    }
}
