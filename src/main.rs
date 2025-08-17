use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, SampleFormat, SizedSample, StreamConfig};
use crossterm::{
    cursor, execute, queue,
    style::{Color, SetForegroundColor},
    terminal::{self, ClearType},
};
use rustfft::{
    num_complex::Complex, num_traits::ToPrimitive, FftPlanner,
};
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

/* ======================= Audio device helpers ====================== */
fn pick_input_device() -> Result<Device> {
    let host = cpal::default_host();
    if let Ok(want) = env::var("LOOKAS_DEVICE") {
        for dev in host.input_devices()? {
            if dev
                .name()?
                .to_lowercase()
                .contains(&want.to_lowercase())
            {
                return Ok(dev);
            }
        }
        anyhow::bail!("LOOKAS_DEVICE='{}' not found", want);
    }
    for dev in host.input_devices()? {
        let name = dev
            .name()
            .unwrap_or_else(|_| String::new())
            .to_lowercase();
        if name.contains("monitor") {
            return Ok(dev);
        }
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
fn ema_tc(prev: f32, x: f32, tau_s: f32, dt_s: f32) -> f32 {
    let a = (-dt_s / tau_s).exp();
    a * prev + (1.0 - a) * x
}

/* ================= Filterbank (Mel) with center freq ================= */
#[derive(Clone)]
struct Tri {
    taps: Vec<(usize, f32)>,
    center_hz: f32,
}
fn hz_to_mel(f: f32) -> f32 {
    2595.0 * ((1.0 + f / 700.0).log10())
}
fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}
fn build_filterbank(
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

/* ===================== High-PPI BLOCK renderer ===================== */
const BLOCKS: [char; 9] =
    [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const BAYER8: [[u8; 8]; 8] = [
    [0, 48, 12, 60, 3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [8, 56, 4, 52, 11, 59, 7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [2, 50, 14, 62, 1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58, 6, 54, 9, 57, 5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21],
];

#[derive(Clone, Copy, PartialEq)]
enum Orient {
    Vertical,
    Horizontal,
}
impl Orient {
    fn from_env() -> Self {
        match env::var("LOOKAS_ORIENT")
            .unwrap_or_else(|_| "vertical".into())
            .to_lowercase()
            .as_str()
        {
            "h" | "hor" | "horizontal" => Orient::Horizontal,
            _ => Orient::Vertical,
        }
    }
}

struct Layout {
    bars: usize,
    left_pad: u16,
    right_pad: u16,
}
fn layout_for(w: u16, _h: u16, orient: Orient) -> Layout {
    match orient {
        // Reserve a small blank right margin to avoid edge artifacts
        Orient::Vertical => {
            let left_pad = 1u16;
            let right_pad = 2u16;
            let usable = w.saturating_sub(left_pad + right_pad);
            Layout {
                bars: usable.max(10) as usize,
                left_pad,
                right_pad,
            }
        }
        // Horizontal uses one row per bar; bars count decided from height
        Orient::Horizontal => Layout {
            bars: 0,
            left_pad: 1,
            right_pad: 2,
        },
    }
}

/* vertical columns across width */
fn draw_blocks_vertical(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
) -> std::io::Result<()> {
    let rows = h.saturating_sub(3) as usize;
    if rows == 0 {
        return Ok(());
    }
    queue!(out, cursor::MoveTo(0, 1))?;
    for row_top in 0..rows {
        let row_from_bottom = rows - 1 - row_top;
        let mut line = String::with_capacity(w as usize);
        for _ in 0..lay.left_pad {
            line.push(' ');
        }
        for i in 0..bars.len() {
            let v = bars[i].clamp(0.0, 1.0);
            let cells = v * rows as f32;
            let full = cells.floor() as usize;
            let frac = (cells - full as f32).clamp(0.0, 0.999_9);
            let ch = if row_from_bottom < full {
                '█'
            } else if row_from_bottom == full {
                let threshold =
                    BAYER8[row_top & 7][i & 7] as f32 / 64.0;
                let mut level = (frac * 8.0).floor();
                if frac.fract() > threshold {
                    level += 1.0;
                }
                BLOCKS[level.clamp(0.0, 8.0) as usize]
            } else {
                ' '
            };
            line.push(ch);
        }
        for _ in 0..lay.right_pad {
            line.push(' ');
        }
        line.push('\n');
        out.write_all(line.as_bytes())?;
    }
    out.flush()?;
    Ok(())
}

/* horizontal rows up the screen */
fn draw_blocks_horizontal(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
) -> std::io::Result<()> {
    let rows = h.saturating_sub(3) as usize;
    let usable_w =
        w.saturating_sub(lay.left_pad + lay.right_pad) as usize;
    if rows == 0 || usable_w == 0 {
        return Ok(());
    }

    queue!(out, cursor::MoveTo(0, 1))?;
    for row in 0..rows.min(bars.len()) {
        let v = bars[row].clamp(0.0, 1.0);
        let cells = v * usable_w as f32;
        let full = cells.floor() as usize;
        let frac = (cells - full as f32).clamp(0.0, 0.999_9);

        let mut line = String::with_capacity(w as usize);
        for _ in 0..lay.left_pad {
            line.push(' ');
        }
        for _ in 0..full {
            line.push('█');
        }
        if full < usable_w {
            let threshold = BAYER8[row & 7][full & 7] as f32 / 64.0;
            let mut level = (frac * 8.0).floor();
            if frac.fract() > threshold {
                level += 1.0;
            }
            line.push(BLOCKS[level.clamp(0.0, 8.0) as usize]);
        }
        while line.chars().count()
            < (lay.left_pad as usize + usable_w)
        {
            line.push(' ');
        }
        for _ in 0..lay.right_pad {
            line.push(' ');
        }
        line.push('\n');
        out.write_all(line.as_bytes())?;
    }
    for _ in bars.len()..rows {
        let mut line = String::new();
        for _ in 0..w {
            line.push(' ');
        }
        line.push('\n');
        out.write_all(line.as_bytes())?;
    }
    out.flush()?;
    Ok(())
}

/* ============================= Main =============================== */
fn main() -> Result<()> {
    // Visualizer params (env-tunable, no recompile)
    let db_min: f32 = env::var("LOOKAS_DB_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(-70.0);
    let db_max: f32 = env::var("LOOKAS_DB_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(-15.0);
    let tilt_alpha: f32 = env::var("LOOKAS_TILT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.35);
    let gate_db: f32 = env::var("LOOKAS_GATE_DB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(-50.0);
    let gain_db: f32 = env::var("LOOKAS_GAIN_DB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let gain_lin: f32 = 10f32.powf(gain_db / 20.0);
    // Adaptive whitening (spreads energy across width)
    let eq_tau: f32 = env::var("LOOKAS_EQ_TAU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0); // seconds
    let eq_strength: f32 = env::var("LOOKAS_EQ_STRENGTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.80); // 0..1

    // Audio/DSP tunables
    const FMIN: f32 = 30.0;
    const FMAX: f32 = 16_000.0;
    const TARGET_FPS_MS: u64 = 16;
    const FFT_SIZE: usize = 2048;
    const TAU_SPEC: f32 = 0.06;
    const SPR_K: f32 = 60.0;
    const SPR_ZETA: f32 = 1.0;

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
    let sr = cfg.sample_rate.0 as f32;

    let ring_len = (sr as usize / 10).max(FFT_SIZE * 3);
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
    let mut bars_y: Vec<f32> = Vec::new();
    let mut bars_v: Vec<f32> = Vec::new();
    let mut eq_ref: Vec<f32> = Vec::new();

    let orient = Orient::from_env();

    loop {
        if crossterm::event::poll(Duration::from_millis(0))? {
            if let crossterm::event::Event::Key(k) =
                crossterm::event::read()?
            {
                if let crossterm::event::KeyCode::Char('q') = k.code {
                    return Ok(());
                }
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

        let (w, h) = terminal::size()?;
        let lay = layout_for(w, h, orient);

        let desired_bars = match orient {
            Orient::Vertical => lay.bars,
            Orient::Horizontal => h.saturating_sub(3) as usize, // one row per band
        };

        if filters.len() != desired_bars {
            filters = build_filterbank(
                sr,
                FFT_SIZE,
                desired_bars,
                FMIN,
                FMAX,
            );
            bars_target = vec![0.0; desired_bars];
            bars_y = vec![0.0; desired_bars];
            bars_v = vec![0.0; desired_bars];
            eq_ref = vec![1e-6; desired_bars];
        }

        let samples = { shared.lock().unwrap().latest() };
        if samples.len() < FFT_SIZE {
            continue;
        }
        let tail = &samples[samples.len() - FFT_SIZE..];

        // ambient gate
        let rms =
            tail.iter().map(|x| x * x).sum::<f32>() / FFT_SIZE as f32;
        let rms_db = 10.0 * (rms.max(1e-12)).log10();
        let gate_open = rms_db > gate_db;

        let mut buf: Vec<Complex<f32>> = tail
            .iter()
            .zip(window.iter())
            .map(|(x, w)| Complex { re: x * w, im: 0.0 })
            .collect();
        fft.process(&mut buf);

        for i in 0..half {
            let re = buf[i].re;
            let im = buf[i].im;
            let p = (re * re + im * im)
                / (FFT_SIZE as f32 * FFT_SIZE as f32);
            spec_pow_smooth[i] = ema_tc(
                spec_pow_smooth[i],
                p.max(1e-12),
                TAU_SPEC,
                dt_s,
            );
        }

        // Per-band feature -> 0..1 bars with whitening
        for (i, tri) in filters.iter().enumerate() {
            // band power -> amplitude
            let mut acc = 0.0f32;
            for &(idx, wgt) in &tri.taps {
                acc += spec_pow_smooth[idx] * wgt;
            }
            let amp = acc.sqrt() * gain_lin;

            // tilt to reduce low-end dominance
            let tilt =
                (tri.center_hz / 1000.0).max(0.001).powf(tilt_alpha);
            let amp_tilted = amp * tilt;

            // adaptive whitening: track slow baseline and express relative
            eq_ref[i] =
                ema_tc(eq_ref[i], amp_tilted, eq_tau, dt_s).max(1e-9);
            let rel = (amp_tilted / eq_ref[i]).powf(eq_strength);

            // map to dB window
            let db = 20.0 * rel.max(1e-12).log10();
            let mut v = (db - db_min) / (db_max - db_min);
            if !gate_open {
                v = 0.0;
            }
            bars_target[i] = v.clamp(0.0, 1.0);
        }

        // spring smoothing for motion
        let c = 2.0 * (SPR_K).sqrt() * SPR_ZETA;
        for i in 0..bars_target.len() {
            let x = bars_target[i];
            let a = SPR_K * (x - bars_y[i]) - c * bars_v[i];
            bars_v[i] += a * dt_s;
            bars_y[i] =
                (bars_y[i] + bars_v[i] * dt_s).clamp(0.0, 1.0);
        }

        queue!(
            out,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0),
            SetForegroundColor(Color::White)
        )?;
        let header = format!(
            "  lookas  |  input: {}  |  orient: {}  |  dB:[{:.0},{:.0}] gain:{:.0}dB tilt:{:.2} eq:{:.2}@{:.1}s  |  q quits\n",
            name,
            match orient { Orient::Vertical => "vertical", Orient::Horizontal => "horizontal" },
            db_min, db_max, gain_db, tilt_alpha, eq_strength, eq_tau
        );
        out.write_all(header.as_bytes())?;

        match orient {
            Orient::Vertical => {
                draw_blocks_vertical(&mut out, &bars_y, w, h, &lay)?
            }
            Orient::Horizontal => {
                draw_blocks_horizontal(&mut out, &bars_y, w, h, &lay)?
            }
        }
    }
}

/* =========================== Scopeguard =========================== */
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
