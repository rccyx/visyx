use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, SampleFormat, SizedSample, StreamConfig};
use crossterm::{
    cursor, execute, queue,
    style::{Color, SetForegroundColor},
    terminal::{self, ClearType},
};
use rustfft::{num_complex::Complex, num_traits::ToPrimitive, FftPlanner};
use std::{
    env,
    io::{stdout, Stdout, Write},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

/* =========================== Ring buffer =========================== */
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
        if self.write_idx == 0 { self.filled = true; }
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

/* ======================= Audio device helpers ====================== */
fn pick_input_device() -> Result<Device> {
    let host = cpal::default_host();
    if let Ok(want) = env::var("LOOKAS_DEVICE") {
        for dev in host.input_devices()? {
            if dev.name()?.to_lowercase().contains(&want.to_lowercase()) {
                return Ok(dev);
            }
        }
        anyhow::bail!("LOOKAS_DEVICE='{}' not found", want);
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
fn build_stream<T>(
    device: Device,
    cfg: StreamConfig,
    shared: Arc<Mutex<SharedBuf>>,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample + ToPrimitive,
{
    let ch = cfg.channels as usize;
    let err_fn = |e| eprintln!("Stream error: {}", e);
    let stream = device.build_input_stream(
        &cfg,
        move |data: &[T], _| {
            let mut buf = shared.lock().unwrap();
            for frame in data.chunks_exact(ch) {
                let mut acc = 0.0f32;
                for &s in frame {
                    acc += s.to_f32().unwrap_or(0.0);
                }
                buf.push(acc / ch as f32);
            }
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

/* ====================== DSP helpers ====================== */
fn hann(n: usize) -> Vec<f32> {
    let den = (n.max(2) - 1) as f32;
    (0..n).map(|i| 0.5 - 0.5 * f32::cos(2.0 * std::f32::consts::PI * i as f32 / den)).collect()
}
#[inline]
fn ema_tc(prev: f32, x: f32, tau_s: f32, dt_s: f32) -> f32 {
    let a = (-dt_s / tau_s).exp();
    a * prev + (1.0 - a) * x
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

    let mut bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| {
            let mut b = (hz / hz_per_bin).round() as isize;
            if b < 1 { b = 1; }
            if b as usize >= half { b = (half - 1) as isize; }
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

/* ============================ Braille draw ========================= */
/* vertical, symmetric fill to avoid diagonal lean */

// pair bits per full row from bottom: rows 0..3 => (7,8), (3,6), (2,5), (1,4)
#[inline]
fn braille_pair_mask(row_from_bottom: u8) -> u16 {
    match row_from_bottom {
        0 => 0x40 | 0x80, // 7,8
        1 => 0x04 | 0x20, // 3,6
        2 => 0x02 | 0x10, // 2,5
        _ => 0x01 | 0x08, // 1,4
    }
}
#[inline]
fn braille_half_mask(row_from_bottom: u8, left: bool) -> u16 {
    match (row_from_bottom, left) {
        (0, true) => 0x40, (0, false) => 0x80,
        (1, true) => 0x04, (1, false) => 0x20,
        (2, true) => 0x02, (2, false) => 0x10,
        (3, true) => 0x01, (3, false) => 0x08,
        _ => 0,
    }
}
#[inline]
fn braille_from_rem(rem: i32, rng: &mut u32) -> char {
    if rem <= 0 { return ' '; }
    if rem >= 8 { return '\u{28FF}'; }
    let full_rows = (rem / 2).clamp(0, 4);
    let half = rem & 1;

    let mut bits: u16 = 0;
    for r in 0..full_rows { bits |= braille_pair_mask(r as u8); }
    if half == 1 {
        *rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        let left = (*rng >> 31) == 0;
        bits |= braille_half_mask(full_rows as u8, left);
    }
    char::from_u32(0x2800 + bits as u32).unwrap()
}

struct Layout { bars: usize, gap: u16, left_pad: u16 }
fn compute_layout(w: u16) -> Layout {
    let margin = 2u16;
    let gap = 1u16;
    let avail = w.saturating_sub(margin);
    let stride = 1 + gap;
    let bars_fit = (avail / stride).max(10) as usize;
    let used = bars_fit as u16 * stride - gap;
    let left_pad = w.saturating_sub(used) / 2;
    Layout { bars: bars_fit, gap, left_pad }
}

fn draw_braille(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
    phase: u32,
) -> std::io::Result<()> {
    let rows = h.saturating_sub(3) as usize;
    if rows == 0 { return Ok(()); }

    queue!(out, cursor::MoveTo(0, 1))?;

    for row_top in 0..rows {
        let mut line = String::with_capacity(w as usize);
        for _ in 0..lay.left_pad { line.push(' '); }
        let row_from_bottom = rows - 1 - row_top;

        for i in 0..lay.bars.min(bars.len()) {
            let v = bars[i].clamp(0.0, 1.0);
            let dots_total = (v * (rows as f32 * 8.0)).round() as i32;
            let rem = dots_total - (row_from_bottom as i32) * 8;

            let mut rng = phase
                .wrapping_mul(0x9E3779B1)
                .wrapping_add((i as u32).wrapping_mul(0x85EBCA6B));

            let ch = braille_from_rem(rem, &mut rng);
            line.push(ch);
            for _ in 0..lay.gap { line.push(' '); }
        }

        while line.chars().count() < w as usize { line.push(' '); }
        line.push('\n');
        out.write_all(line.as_bytes())?;
    }

    // removed baseline line
    out.flush()?;
    Ok(())
}

/* ============================= Main =============================== */
fn main() -> Result<()> {
    // Tunables
    const FMIN: f32 = 30.0;
    const FMAX: f32 = 16_000.0;
    const TARGET_FPS_MS: u64 = 16;
    const FFT_SIZE: usize = 2048;

    // time constants (seconds)
    const TAU_SPEC: f32 = 0.10;
    const TAU_NORM: f32 = 0.45;
    const FLOOR_FRAC: f32 = 0.05;
    const LOG_COMP: f32 = 180.0;

    // spring smoother per bar
    const SPR_K: f32 = 60.0;
    const SPR_ZETA: f32 = 1.0;

    const SILENCE_DB: f32 = -60.0;

    // TUI setup
    let mut out = stdout();
    terminal::enable_raw_mode()?;
    execute!(
        out,
        terminal::EnterAlternateScreen,
        cursor::Hide,
        terminal::Clear(ClearType::All),
        SetForegroundColor(Color::White),
    )?;
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
    let mut filters: Vec<Tri> = Vec::new();
    let mut bars_target: Vec<f32> = Vec::new();

    // spring state
    let mut bars_y: Vec<f32> = Vec::new();
    let mut bars_v: Vec<f32> = Vec::new();

    let mut norm_max = 0.12f32;
    let mut frame_counter: u32 = 0;

    loop {
        if crossterm::event::poll(Duration::from_millis(0))? {
            if let crossterm::event::Event::Key(k) = crossterm::event::read()? {
                if let crossterm::event::KeyCode::Char('q') = k.code { break; }
            }
        }

        let now = Instant::now();
        let dt = now.duration_since(last);
        if dt < target_dt {
            thread::sleep(target_dt - dt);
            continue;
        }
        let dt_s = dt.as_secs_f32();
        last = now;
        frame_counter = frame_counter.wrapping_add(1);

        let (w, h) = terminal::size()?;
        let lay = compute_layout(w);
        if filters.len() != lay.bars {
            filters = build_filterbank(sr, FFT_SIZE, lay.bars, FMIN, FMAX);
            bars_target = vec![0.0; lay.bars];
            bars_y = vec![0.0; lay.bars];
            bars_v = vec![0.0; lay.bars];
        }

        let samples = { shared.lock().unwrap().latest() };
        if samples.len() < FFT_SIZE { continue; }
        let tail = &samples[samples.len() - FFT_SIZE..];

        let rms = tail.iter().map(|x| x * x).sum::<f32>() / FFT_SIZE as f32;
        let db = 10.0 * (rms.max(1e-12)).log10();
        let gate_open = db > SILENCE_DB;

        let mut buf: Vec<Complex<f32>> =
            tail.iter().zip(window.iter()).map(|(x, w)| Complex { re: x * w, im: 0.0 }).collect();
        fft.process(&mut buf);

        for i in 0..half {
            let re = buf[i].re;
            let im = buf[i].im;
            let p = (re * re + im * im) / (FFT_SIZE as f32 * FFT_SIZE as f32);
            spec_pow_smooth[i] = ema_tc(spec_pow_smooth[i], p.max(1e-12), TAU_SPEC, dt_s);
        }

        for (i, tri) in filters.iter().enumerate() {
            let mut acc = 0.0f32;
            for &(idx, wgt) in &tri.taps { acc += spec_pow_smooth[idx] * wgt; }
            let amp = acc.sqrt();
            let v = (amp * LOG_COMP + 1.0).ln() / (LOG_COMP + 1.0).ln();
            bars_target[i] = if gate_open { v } else { 0.0 };
        }

        if lay.bars >= 9 {
            let k = [1.0f32, 4.0, 11.0, 22.0, 27.0, 22.0, 11.0, 4.0, 1.0];
            let mut sm = vec![0.0f32; lay.bars];
            for i in 0..lay.bars {
                let mut acc = 0.0;
                let mut wsum = 0.0;
                for (j, wgt) in k.iter().enumerate() {
                    let idx = (i as isize + j as isize - 4).clamp(0, (lay.bars - 1) as isize) as usize;
                    acc += bars_target[idx] * *wgt;
                    wsum += *wgt;
                }
                sm[i] = acc / wsum;
            }
            bars_target.copy_from_slice(&sm);
        }

        let frame_max = bars_target.iter().copied().fold(0.0f32, f32::max);
        norm_max = ema_tc(norm_max, frame_max.max(1e-6), TAU_NORM, dt_s);
        let floor = norm_max * FLOOR_FRAC;
        let scale = (norm_max - floor).max(1e-6);

        let c = 2.0 * (SPR_K).sqrt() * SPR_ZETA;
        for i in 0..lay.bars {
            let x = ((bars_target[i] - floor) / scale).clamp(0.0, 1.0);
            let a = SPR_K * (x - bars_y[i]) - c * bars_v[i];
            bars_v[i] += a * dt_s;
            bars_y[i] = (bars_y[i] + bars_v[i] * dt_s).clamp(0.0, 1.0);
        }

        queue!(out, terminal::Clear(ClearType::All), cursor::MoveTo(0, 0), SetForegroundColor(Color::White))?;
        let header = format!("  lookas  |  input: {}  |  q quits\n", name);
        out.write_all(header.as_bytes())?;
        draw_braille(&mut out, &bars_y, w, h, &lay, frame_counter)?;
    }

    Ok(())
}

/* =========================== Scopeguard =========================== */
mod scopeguard {
    pub fn guard<T, F: FnOnce(T)>(v: T, f: F) -> Guard<T, F> { Guard { v: Some(v), f: Some(f) } }
    pub struct Guard<T, F: FnOnce(T)> { v: Option<T>, f: Option<F> }
    impl<T, F: FnOnce(T)> Drop for Guard<T, F> {
        fn drop(&mut self) {
            if let (Some(v), Some(f)) = (self.v.take(), self.f.take()) { f(v); }
        }
    }
}
