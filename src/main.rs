use anyhow::Result;
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::SampleFormat;
use crossterm::{
    cursor, execute, queue,
    style::{Color, SetForegroundColor},
    terminal::{self, ClearType},
};
use rustfft::FftPlanner;
use std::{
    env,
    io::{stdout, Write},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use visyx::{
    analyzer::SpectrumAnalyzer,
    audio::{best_config_for, build_stream, pick_input_device},
    buffer::SharedBuf,
    dsp::{hann, prepare_fft_input_inplace},
    filterbank::build_filterbank,
    render::{
        draw_blocks_vertical, layout_for,
    },
    utils::scopeguard,
};

#[inline]
fn get_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    match env::var(name) {
        Ok(val) => val.parse::<T>().unwrap_or(default),
        Err(_) => default,
    }
}


fn main() -> Result<()> {
    #[cfg(unix)]
    unsafe {
        use libc::dup2;
        use std::fs::OpenOptions;
        use std::os::unix::io::AsRawFd;

        if let Ok(devnull) =
            OpenOptions::new().write(true).open("/dev/null")
        {
            dup2(devnull.as_raw_fd(), libc::STDERR_FILENO);
        }
    }

    let fmin: f32 = get_env("VISYX_FMIN", 30.0);
    let fmax: f32 = get_env("VISYX_FMAX", 16_000.0);
    let target_fps_ms: u64 = get_env("VISYX_TARGET_FPS_MS", 16);
    let fft_size: usize = get_env("VISYX_FFT_SIZE", 2048);
    let tau_spec: f32 = get_env("VISYX_TAU_SPEC", 0.06);
    let gate_db: f32 = get_env("VISYX_GATE_DB", -55.0);
    let tilt_alpha: f32 = get_env("VISYX_TILT_ALPHA", 0.30);
    let flow_k: f32 = get_env("VISYX_FLOW_K", 0.18);
    let spr_k: f32 = get_env("VISYX_SPR_K", 60.0);
    let spr_zeta: f32 = get_env("VISYX_SPR_ZETA", 1.0);

    let show_hud: bool = get_env("VISYX_HUD", 0u8) != 0;
    let top_pad: u16 = if show_hud { 1 } else { 0 };

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

    let device = pick_input_device()?;
    let name = device.name().unwrap_or_else(|_| "<unknown>".into());
    let cfg = best_config_for(&device)?;
    let sr = cfg.sample_rate.0 as f32;

    let ring_len = (sr as usize / 10).max(fft_size * 3);
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

    execute!(
        out,
        terminal::Clear(ClearType::All),
        cursor::MoveTo(0, 0)
    )?;

    let window = hann(fft_size);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let half = fft_size / 2;

    let mut last = Instant::now();
    let target_dt = Duration::from_millis(target_fps_ms);
    let mut analyzer = SpectrumAnalyzer::new(half);

    let mut buf = Vec::with_capacity(fft_size);
    let mut spec_pow = vec![0.0; half];
    let mut header = String::with_capacity(256);

    loop {
        if crossterm::event::poll(Duration::ZERO)? {
            if let crossterm::event::Event::Key(k) =
                crossterm::event::read()?
            {
                use crossterm::event::KeyCode::*;
                match k.code {
                    Char('q') => return Ok(()),
                    _ => {}
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
        let lay = layout_for(w, top_pad);
        let desired_bars = lay.bars;

        if analyzer.filters.len() != desired_bars {
            analyzer.filters = build_filterbank(
                sr,
                fft_size,
                desired_bars,
                fmin,
                fmax,
            );
            analyzer.resize(desired_bars);
        }

        let samples = if let Ok(buf) = shared.try_lock() {
            buf.latest()
        } else {
            thread::sleep(Duration::from_millis(1));
            continue;
        };
        if samples.len() < fft_size {
            continue;
        }
        let tail = &samples[samples.len() - fft_size..];

        let mut rms = 0.0;
        for &x in tail {
            rms += x * x;
        }
        rms /= fft_size as f32;
        let rms_db = 10.0 * (rms.max(1e-12)).log10();
        let gate_open = rms_db > gate_db;

        prepare_fft_input_inplace(tail, &window, &mut buf);
        fft.process(&mut buf);

        for i in 0..half {
            let re = buf[i].re;
            let im = buf[i].im;
            spec_pow[i] = (re * re + im * im)
                / (fft_size as f32 * fft_size as f32);
        }

        analyzer.update_spectrum(&spec_pow, tau_spec, dt_s);
        let bars_target =
            analyzer.analyze_bands(tilt_alpha, dt_s, gate_open);
        analyzer.apply_flow_and_spring(
            &bars_target,
            flow_k,
            spr_k,
            spr_zeta,
            dt_s,
            gate_open,
        );

        queue!(
            out,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0),
            SetForegroundColor(Color::White)
        )?;

        if show_hud {
            header.clear();
            header.push_str("  visyx  |  input: ");
            header.push_str(&name);
            header.push_str("  |  auto gain [");
            use std::fmt::Write;
            let _ = write!(
                header,
                "{:.1}..{:.1} dB",
                analyzer.db_low - 3.0,
                analyzer.db_high + 6.0
            );
            header.push_str("]  |  q quits\n");
            out.write_all(header.as_bytes())?;
        }

        draw_blocks_vertical(
            &mut out,
            &analyzer.bars_y,
            w,
            h,
            &lay,
        )?;
    }
}
