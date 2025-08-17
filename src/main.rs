use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, SampleFormat, SizedSample, StreamConfig};
use crossterm::{cursor, execute, queue, style::{Color, SetForegroundColor}, terminal::{self, ClearType}};
use rustfft::{num_complex::Complex, num_traits::ToPrimitive, FftPlanner};
use std::{env, io::{stdout, Stdout, Write}, sync::{Arc, Mutex}, thread, time::{Duration, Instant}};

/* =========================== Ring buffer =========================== */

struct SharedBuf { data: Vec<f32>, write_idx: usize, filled: bool }
impl SharedBuf {
    fn new(cap: usize) -> Self { Self { data: vec![0.0; cap], write_idx: 0, filled: false } }
    fn push(&mut self, x: f32) {
        self.data[self.write_idx] = x;
        self.write_idx = (self.write_idx + 1) % self.data.len();
        if self.write_idx == 0 { self.filled = true; }
    }
    fn latest(&self) -> Vec<f32> {
        if !self.filled { return self.data[..self.write_idx].to_vec(); }
        let mut v = Vec::with_capacity(self.data.len());
        v.extend_from_slice(&self.data[self.write_idx..]);
        v.extend_from_slice(&self.data[..self.write_idx]);
        v
    }
}

/* ======================= Audio device helpers ====================== */

fn pick_input_device() -> Result<Device> {
    let host = cpal::default_host();
    if let Ok(want) = env::var("MYCAVA_DEVICE") {
        for dev in host.input_devices()? {
            if dev.name()?.to_lowercase().contains(&want.to_lowercase()) { return Ok(dev); }
        }
        anyhow::bail!("MYCAVA_DEVICE='{}' not found", want);
    }
    for dev in host.input_devices()? {
        let name = dev.name().unwrap_or_else(|_| String::new()).to_lowercase();
        if name.contains("monitor") { return Ok(dev); }
    }
    host.default_input_device().context("No default input device")
}

fn best_config_for(device: &Device) -> Result<StreamConfig> {
    let mut cfg = device.default_input_config()?.config();
    cfg.sample_rate.0 = cfg.sample_rate.0.clamp(44_100, 48_000);
    Ok(cfg)
}

fn build_stream<T>(device: Device, cfg: StreamConfig, shared: Arc<Mutex<SharedBuf>>) -> Result<cpal::Stream>
where T: Sample + SizedSample + ToPrimitive {
    let ch = cfg.channels as usize;
    let err_fn = |e| eprintln!("Stream error: {}", e);
    let stream = device.build_input_stream(
        &cfg,
        move |data: &[T], _| {
            let mut buf = shared.lock().unwrap();
            for frame in data.chunks_exact(ch) {
                let mut acc = 0.0f32;
                for &s in frame { acc += s.to_f32().unwrap_or(0.0); }
                buf.push(acc / ch as f32);
            }
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

/* ====================== Window and filterbank ====================== */

fn hann(n: usize) -> Vec<f32> {
    let nf = n as f32;
    (0..n).map(|i| 0.5 - 0.5 * f32::cos(2.0 * std::f32::consts::PI * i as f32 / nf)).collect()
}

#[derive(Clone)]
struct Tri { taps: Vec<(usize, f32)> }

fn hz_to_mel(f: f32) -> f32 { 2595.0 * ((1.0 + f / 700.0).log10()) }
fn mel_to_hz(m: f32) -> f32 { 700.0 * (10f32.powf(m / 2595.0) - 1.0) }

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

    let mut bin_points: Vec<usize> = hz_points.iter().map(|&hz| {
        let mut b = (hz / hz_per_bin).round() as isize;
        if b < 1 { b = 1; }
        if b as usize >= half { b = (half - 1) as isize; }
        b as usize
    }).collect();

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
            let w = if c == l { 0.0 } else { (i - l) as f32 / (c - l) as f32 };
            taps.push((i, w));
        }
        for i in c..=r {
            let w = if r == c { 0.0 } else { 1.0 - (i - c) as f32 / (r - c) as f32 };
            taps.push((i, w));
        }
        let sumw = taps.iter().map(|(_, w)| *w).sum::<f32>().max(1e-6);
        for t in &mut taps { t.1 /= sumw; }
        filters.push(Tri { taps });
    }
    filters
}

/* ============================ Layout/Drawing ======================= */

struct Layout { bars: usize, bar_w: u16, gap: u16, left_pad: u16 }
fn compute_layout(w: u16) -> Layout {
    let mut bars = 64usize;
    let mut bar_w = 2u16;
    let gap = 0u16;
    loop {
        let needed = bars as u16 * bar_w + (bars.saturating_sub(1)) as u16 * gap;
        if needed + 2 <= w {
            let left_pad = (w - needed) / 2;
            return Layout { bars, bar_w, gap, left_pad };
        }
        if bar_w > 1 { bar_w -= 1; continue; }
        if bars > 32 { bars -= 8; continue; }
        return Layout { bars, bar_w, gap, left_pad: 1 };
    }
}

// map fractional cell to unicode partial block
fn frac_char(f: f32) -> char {
    let idx = (f * 8.0).clamp(0.0, 7.999).floor() as usize;
    const CH: [char; 9] = [' ', '▁','▂','▃','▄','▅','▆','▇','█'];
    CH[idx + 1]
}

fn draw_frame(out: &mut Stdout, bars: &[f32], peaks: &[u16], w: u16, h: u16, lay: &Layout) -> std::io::Result<()> {
    let max_h = h.saturating_sub(3);
    queue!(out, cursor::MoveTo(0, 1))?;

    for row_top in (0..max_h).rev() {
        let rb = max_h - 1 - row_top; // row index from bottom
        let mut line = String::with_capacity(w as usize);
        let mut cols_used: u16 = 0;

        // left padding
        for _ in 0..lay.left_pad { line.push(' '); cols_used += 1; }

        for i in 0..lay.bars.min(bars.len()) {
            let height = (bars[i].clamp(0.0, 1.0) * max_h as f32).max(0.0);
            let full = height.floor() as u16;
            let frac = height - full as f32;

            for _ in 0..lay.bar_w {
                if rb < full { line.push('█'); }
                else if rb == full && frac > 0.0 { line.push(frac_char(frac)); }
                else { line.push(' '); }
                cols_used += 1;
            }

            // peak marker on last column of the bar
            if peaks[i] > rb {
                for _ in 0..lay.gap { line.push(' '); cols_used += 1; }
                // replace last drawn char with '▀'
                if line.pop().is_some() { line.push('▀'); } else { line.push('▀'); }
            } else {
                for _ in 0..lay.gap { line.push(' '); cols_used += 1; }
            }
        }

        // pad to full width
        while cols_used < w { line.push(' '); cols_used += 1; }
        line.push('\n');
        out.write_all(line.as_bytes())?;
    }

    // baseline full width
    let mut base = String::with_capacity(w as usize + 1);
    for _ in 0..w { base.push('─'); }
    base.push('\n');
    out.write_all(base.as_bytes())?;
    out.flush()?;
    Ok(())
}

/* ============================= Main =============================== */

fn main() -> Result<()> {
    // Tunables
    const FMIN: f32 = 30.0;
    const FMAX: f32 = 16_000.0;
    const TARGET_FPS_MS: u64 = 16; // ~60 FPS
    const FFT_SIZE: usize = 2048;

    // smoothing/compression
    const SPEC_EMA: f32 = 0.80;
    const BAR_ATTACK: f32 = 0.45;
    const BAR_DECAY: f32 = 0.92;
    const NORM_EMA: f32 = 0.93;
    const LOG_COMP: f32 = 200.0;
    const FLOOR_FRAC: f32 = 0.05;

    // silence + peaks
    const SILENCE_DB: f32 = -60.0;
    const PEAK_HOLD_MS: u64 = 240;
    const PEAK_FALL: u16 = 1;

    // TUI setup
    let mut out = stdout();
    terminal::enable_raw_mode()?;
    execute!(out, terminal::EnterAlternateScreen, cursor::Hide, terminal::Clear(ClearType::All), SetForegroundColor(Color::White))?;
    let _cleanup = scopeguard::guard((), |_| {
        let mut out = stdout();
        let _ = execute!(out, cursor::Show, terminal::LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    });

    // Audio
    let device = pick_input_device()?;
    let name = device.name().unwrap_or_else(|_| "<unknown>".into());
    let cfg = best_config_for(&device)?;
    let sr = cfg.sample_rate.0 as f32;

    let ring_len = (sr as usize / 10).max(FFT_SIZE * 3);
    let shared = Arc::new(Mutex::new(SharedBuf::new(ring_len)));
    let stream = match device.default_input_config()?.sample_format() {
        SampleFormat::F32 => build_stream::<f32>(device, cfg.clone(), shared.clone())?,
        SampleFormat::I16 => build_stream::<i16>(device, cfg.clone(), shared.clone())?,
        SampleFormat::U16 => build_stream::<u16>(device, cfg.clone(), shared.clone())?,
        _ => anyhow::bail!("Unsupported sample format"),
    };
    stream.play()?;

    // FFT
    let window = hann(FFT_SIZE);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let half = FFT_SIZE / 2;

    // State
    let mut last = Instant::now();
    let target_dt = Duration::from_millis(TARGET_FPS_MS);
    let mut spec_pow_smooth: Vec<f32> = vec![0.0; half];
    let mut bars_state: Vec<f32> = Vec::new();
    let mut filters: Vec<Tri> = Vec::new();
    let mut norm_max = 0.12f32;
    let mut peaks: Vec<u16> = Vec::new();
    let mut peaks_timer: Vec<Instant> = Vec::new();

    loop {
        if crossterm::event::poll(Duration::from_millis(0))? {
            if let crossterm::event::Event::Key(k) = crossterm::event::read()? {
                if let crossterm::event::KeyCode::Char('q') = k.code { break; }
            }
        }

        let now = Instant::now();
        if now.duration_since(last) < target_dt { thread::sleep(Duration::from_millis(1)); continue; }
        last = now;

        let (w, h) = terminal::size()?;
        let lay = compute_layout(w);

        if filters.len() != lay.bars {
            filters = build_filterbank(sr, FFT_SIZE, lay.bars, FMIN, FMAX);
            bars_state = vec![0.0; lay.bars];
            peaks = vec![0u16; lay.bars];
            peaks_timer = vec![Instant::now(); lay.bars];
        }

        let samples = { shared.lock().unwrap().latest() };
        if samples.len() < FFT_SIZE { continue; }
        let tail = &samples[samples.len() - FFT_SIZE..];

        // dB power gate
        let rms = tail.iter().map(|x| x * x).sum::<f32>() / FFT_SIZE as f32;
        let db = 10.0 * (rms.max(1e-12)).log10();
        let gate_open = db > SILENCE_DB;

        // FFT
        let mut buf: Vec<Complex<f32>> = tail.iter().zip(window.iter()).map(|(x, w)| Complex { re: x * w, im: 0.0 }).collect();
        fft.process(&mut buf);

        for i in 0..half {
            let re = buf[i].re; let im = buf[i].im;
            let p = (re * re + im * im) / (FFT_SIZE as f32 * FFT_SIZE as f32);
            spec_pow_smooth[i] = SPEC_EMA * spec_pow_smooth[i] + (1.0 - SPEC_EMA) * p.max(1e-12);
        }

        // bands
        let mut bar_vals = vec![0.0f32; lay.bars];
        for (i, tri) in filters.iter().enumerate() {
            let mut acc = 0.0f32;
            for &(idx, wgt) in &tri.taps { acc += spec_pow_smooth[idx] * wgt; }
            let amp = acc.sqrt();
            let v = (amp * LOG_COMP + 1.0).ln() / (LOG_COMP + 1.0).ln();
            bar_vals[i] = if gate_open { v } else { 0.0 };
        }

        // Gaussian 7-tap spatial smooth
        if lay.bars >= 7 {
            let k = [1.0f32, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0];
            let mut sm = vec![0.0f32; lay.bars];
            for i in 0..lay.bars {
                let mut acc = 0.0; let mut wsum = 0.0;
                for (j, wgt) in k.iter().enumerate() {
                    let idx = (i as isize + j as isize - 3).clamp(0, (lay.bars - 1) as isize) as usize;
                    acc += bar_vals[idx] * *wgt;
                    wsum += *wgt;
                }
                sm[i] = acc / wsum;
            }
            bar_vals = sm;
        }

        // normalize and temporal easing
        let frame_max = bar_vals.iter().copied().fold(0.0f32, f32::max);
        norm_max = NORM_EMA * norm_max + (1.0 - NORM_EMA) * frame_max.max(1e-6);
        let floor = norm_max * FLOOR_FRAC;
        let scale = (norm_max - floor).max(1e-6);

        let max_h = h.saturating_sub(3);
        for i in 0..lay.bars {
            let v = (((bar_vals[i] - floor) / scale).clamp(0.0, 1.0)).powf(0.85);
            if v > bars_state[i] {
                bars_state[i] = BAR_ATTACK * bars_state[i] + (1.0 - BAR_ATTACK) * v;
            } else {
                bars_state[i] *= BAR_DECAY;
                if v > bars_state[i] { bars_state[i] = v; }
            }

            // peaks
            let cur_h = (bars_state[i] * max_h as f32).round() as u16;
            if cur_h > peaks[i] { peaks[i] = cur_h; peaks_timer[i] = now; }
            else if now.duration_since(peaks_timer[i]) > Duration::from_millis(PEAK_HOLD_MS) && peaks[i] > 0 {
                peaks[i] = peaks[i].saturating_sub(PEAK_FALL);
            }
        }

        // draw
        queue!(out, terminal::Clear(ClearType::All), cursor::MoveTo(0, 0), SetForegroundColor(Color::White))?;
        let header = format!("  mycava  |  input: {}  |  rate: {} Hz  |  level: {:>5.1} dB  |  q to quit\n",
                             name, sr as u32, db);
        out.write_all(header.as_bytes())?;
        draw_frame(&mut out, &bars_state, &peaks, w, h, &lay)?;
    }

    Ok(())
}

mod scopeguard {
    pub fn guard<T, F: FnOnce(T)>(v: T, f: F) -> Guard<T, F> { Guard { v: Some(v), f: Some(f) } }
    pub struct Guard<T, F: FnOnce(T)> { v: Option<T>, f: Option<F> }
    impl<T, F: FnOnce(T)> Drop for Guard<T, F> {
        fn drop(&mut self) {
            if let (Some(v), Some(f)) = (self.v.take(), self.f.take()) { f(v); }
        }
    }
}
