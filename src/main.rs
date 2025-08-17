use anyhow::Result;
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, StreamTrait};
use crossterm::{
    cursor, execute, queue,
    style::{Color, SetForegroundColor},
    terminal::{self, ClearType},
};
use lookas::{
    analyzer::SpectrumAnalyzer,
    audio::{best_config_for, build_stream, pick_input_device},
    buffer::SharedBuf,
    dsp::{hann, prepare_fft_input},
    filterbank::build_filterbank,
    render::{draw_blocks_horizontal, draw_blocks_vertical, layout_for, Orient},
    utils::scopeguard,
};
use rustfft::FftPlanner;
use std::{
    io::{stdout, Write},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

fn main() -> Result<()> {
    // Constants tuned to behave like CAVA without manual gain
    const FMIN: f32 = 30.0;
    const FMAX: f32 = 16_000.0;
    const TARGET_FPS_MS: u64 = 16;
    const FFT_SIZE: usize = 2048;
    const TAU_SPEC: f32 = 0.06; // spectrum smoothing
    const GATE_DB: f32 = -55.0; // ignore room hiss
    const TILT_ALPHA: f32 = 0.30; // reduce low end dominance
    const FLOW_K: f32 = 0.18; // lateral energy flow
    const SPR_K: f32 = 60.0; // spring to keep motion smooth
    const SPR_ZETA: f32 = 1.0;

    // TUI
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
    let stream = match device.default_input_config()?.sample_format() {
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
    let mut analyzer = SpectrumAnalyzer::new(half);

    // Animation state
    let mut orient = Orient::Vertical;

    loop {
        if crossterm::event::poll(Duration::from_millis(0))? {
            if let crossterm::event::Event::Key(k) =
                crossterm::event::read()?
            {
                use crossterm::event::KeyCode::*;
                match k.code {
                    Char('q') => return Ok(()),
                    Char('v') => orient = Orient::Vertical,
                    Char('h') => orient = Orient::Horizontal,
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
        let lay = layout_for(w, h, orient);
        let desired_bars = match orient {
            Orient::Vertical => lay.bars,
            Orient::Horizontal => h.saturating_sub(3) as usize,
        };

        if analyzer.filters.len() != desired_bars {
            analyzer.filters = build_filterbank(
                sr,
                FFT_SIZE,
                desired_bars,
                FMIN,
                FMAX,
            );
            analyzer.resize(desired_bars);
        }

        let samples = { shared.lock().unwrap().latest() };
        if samples.len() < FFT_SIZE {
            continue;
        }
        let tail = &samples[samples.len() - FFT_SIZE..];

        // gate on room noise
        let rms = tail.iter().map(|x| x * x).sum::<f32>() / FFT_SIZE as f32;
        let rms_db = 10.0 * (rms.max(1e-12)).log10();
        let gate_open = rms_db > GATE_DB;

        let mut buf = prepare_fft_input(tail, &window);
        fft.process(&mut buf);

        // Calculate power spectrum
        let mut spec_pow = vec![0.0; half];
        for i in 0..half {
            let re = buf[i].re;
            let im = buf[i].im;
            spec_pow[i] = (re * re + im * im) / (FFT_SIZE as f32 * FFT_SIZE as f32);
        }

        analyzer.update_spectrum(&spec_pow, TAU_SPEC, dt_s);
        let bars_target = analyzer.analyze_bands(TILT_ALPHA, dt_s, gate_open);
        analyzer.apply_flow_and_spring(&bars_target, FLOW_K, SPR_K, SPR_ZETA, dt_s);

        queue!(
            out,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0),
            SetForegroundColor(Color::White)
        )?;
        let header = format!(
            "  lookas  |  input: {}  |  orient: {}  |  auto gain window [{:.1} dB .. {:.1} dB]  |  v/h to switch, q quits\n",
            name,
            match orient { Orient::Vertical => "vertical", Orient::Horizontal => "horizontal" },
            analyzer.db_low - 3.0, analyzer.db_high + 6.0
        );
        out.write_all(header.as_bytes())?;

        match orient {
            Orient::Vertical => {
                draw_blocks_vertical(&mut out, &analyzer.bars_y, w, h, &lay)?
            }
            Orient::Horizontal => {
                draw_blocks_horizontal(&mut out, &analyzer.bars_y, w, h, &lay)?
            }
        }
    }
}
