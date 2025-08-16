use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, SizedSample, SampleFormat, StreamConfig};
use crossterm::{
    cursor, execute,
    style::{Color, SetForegroundColor},
    terminal::{self, ClearType},
};
use rustfft::{num_complex::Complex, FftPlanner};
use rustfft::num_traits::ToPrimitive; 
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
                for c in 0..channels {
                    acc += frame[c].to_f32().unwrap_or(0.0);
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
    (0..n).map(|i| 0.5 - 0.5 * f32::cos(2.0 * std::f32::consts::PI * i as f32 / (n as f32))).collect()
}

fn bucket_log(freqs: &[f32], mags: &[f32], bars: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let mut out = vec![0.0; bars];
    let log_min = fmin.ln();
    let log_max = fmax.ln();
    for (i, &f) in freqs.iter().enumerate() {
        if f < fmin || f > fmax { continue; }
        let t = ((f.ln() - log_min) / (log_max - log_min)).clamp(0.0, 0.999999);
        let b = (t * bars as f32) as usize;
        if b < bars {
            out[b] = f32::max(out[b], mags[i]);
        }
    }
    out
}

fn draw_frame(out: &mut Stdout, bars: &[f32], w: u16, h: u16) -> std::io::Result<()> {
    let bar_count = min(bars.len() as u16, w.saturating_sub(2));
    let max_h = h.saturating_sub(2);
    execute!(out, cursor::MoveTo(0,0))?;
    println!();
    for row in (0..max_h).rev() {
        print!(" ");
        for i in 0..bar_count {
            let v = bars[i as usize].clamp(0.0, 1.0);
            let filled = (v * max_h as f32).round() as u16;
            if filled > row { print!("█"); } else { print!(" "); }
        }
        println!();
    }
    print!(" ");
    for _ in 0..bar_count { print!("▁"); }
    println!();
    out.flush()?;
    Ok(())
}

fn main() -> Result<()> {
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

    let device = pick_input_device()?;
    let name = device.name().unwrap_or_else(|_| "<unknown>".into());
    let cfg = best_config_for(&device)?;
    let sample_rate = cfg.sample_rate.0 as f32;

    let ring_len = (sample_rate as usize / 10).max(2048);
    let shared = Arc::new(Mutex::new(SharedBuf::new(ring_len)));

    let stream = match device.default_input_config()?.sample_format() {
        SampleFormat::F32 => build_stream::<f32>(device, cfg.clone(), shared.clone())?,
        SampleFormat::I16 => build_stream::<i16>(device, cfg.clone(), shared.clone())?,
        SampleFormat::U16 => build_stream::<u16>(device, cfg.clone(), shared.clone())?,
        _ => anyhow::bail!("Unsupported sample format"),
    };
    stream.play()?;

    let fft_size = 1024usize;
    let window = hann_window(fft_size);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut last = Instant::now();
    let target_dt = Duration::from_millis(16);
    let mut smoothed: Vec<f32> = vec![0.0; 120];
    let decay = 0.88f32;

    loop {
        if crossterm::event::poll(Duration::from_millis(0))? {
            if let crossterm::event::Event::Key(k) = crossterm::event::read()? {
                if let crossterm::event::KeyCode::Char('q') = k.code { break; }
            }
        }

        let now = Instant::now();
        if now.duration_since(last) < target_dt {
            thread::sleep(Duration::from_millis(2));
            continue;
        }
        last = now;

        let samples = { shared.lock().unwrap().latest() };
        if samples.len() < fft_size { continue; }

        let tail = &samples[samples.len()-fft_size..];
        let mut buf: Vec<Complex<f32>> = tail.iter().zip(window.iter())
            .map(|(x,w)| Complex{ re: x * w, im: 0.0 })
            .collect();
        fft.process(&mut buf);

        let half = fft_size/2;
        let mags: Vec<f32> = buf[..half]
            .iter()
            .map(|c| (c.norm() / fft_size as f32).powf(0.35))
            .collect();

        let freqs: Vec<f32> = (0..half).map(|i| i as f32 * sample_rate / fft_size as f32).collect();

        let (w, h) = terminal::size()?;
        let bars = (w.saturating_sub(2)).max(10) as usize;

        let mut bar_vals = bucket_log(&freqs, &mags, bars, 30.0, 16_000.0);

        let maxv = bar_vals.iter().cloned().fold(0.0001f32, f32::max);
        for v in &mut bar_vals { *v = (*v / maxv).clamp(0.0, 1.0); }

        if smoothed.len() != bars { smoothed = vec![0.0; bars]; }
        for i in 0..bars {
            smoothed[i] = smoothed[i] * decay;
            if bar_vals[i] > smoothed[i] {
                smoothed[i] = bar_vals[i];
            }
        }

        execute!(out, terminal::Clear(ClearType::All))?;
        println!("  mycava  |  input: {}  |  q to quit", name);
        draw_frame(&mut out, &smoothed, w, h)?;
    }

    drop(cleanup);
    Ok(())
}

mod scopeguard {
    pub fn guard<T, F: FnOnce(T)>(v: T, f: F) -> Guard<T, F> { Guard{ v: Some(v), f: Some(f) } }
    pub struct Guard<T, F: FnOnce(T)> { v: Option<T>, f: Option<F> }
    impl<T, F: FnOnce(T)> Drop for Guard<T, F> {
        fn drop(&mut self) {
            if let (Some(v), Some(f)) = (self.v.take(), self.f.take()) { f(v); }
        }
    }
}

