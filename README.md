<p align="center">
  <img src="https://raw.githubusercontent.com/rust-lang/rust-artwork/master/logo/rust-logo-256x256.png" height="80" alt="Rust Logo"/>
</p>

<h1 align="center">visyx</h1>

<p align="center">
  <a href="https://www.rust-lang.org">
    <img src="https://img.shields.io/badge/Rust-1.70+-black?logo=rust&logoColor=white&style=for-the-badge" alt="Rust"/>
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue?logo=open-source-initiative&logoColor=white&style=for-the-badge" alt="License: MIT"/>
  </a>
  <a href="https://crates.io/crates/visyx">
    <img src="https://img.shields.io/crates/v/visyx?color=orange&logo=rust&style=for-the-badge" alt="Crates.io"/>
  </a>
</p>

A high-performance, terminal-based audio spectrum visualizer written in Rust that transforms your audio into mesmerizing real-time visual patterns.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/db5daa81-a06b-45a9-b760-2e45e307eb61" />

## What It Does

visyx captures your system audio, breaks it into frequency bands with a mel-scale FFT, and renders it as smooth, physics-driven bars in the terminal. Adaptive gain and noise gating keep the visuals clean, while zero-copy rendering ensures 60+ FPS with minimal latency. Multiple layouts are supported, and it runs cross-platform on Linux, macOS, and Windows, given you have Rust installed.

### Installation

```bash
cargo install visyx
```

### Basic Usage

Simply run the visualizer with default settings:

```bash
visyx
```

**Controls:**

- `q` - Quit the application

## Configuration

You can fine-tune the visualization for your specific audio setup and preferences using these environment variables.

### Settings

| Variable         | Description                       | Default | Range                 |
| ---------------- | --------------------------------- | ------- | --------------------- |
| `VISYX_FMIN`     | Minimum frequency to display (Hz) | 30.0    | 10.0 - 1000.0         |
| `VISYX_FMAX`     | Maximum frequency to display (Hz) | 16000.0 | 1000.0 - 24000.0      |
| `VISYX_FFT_SIZE` | FFT window size (power of 2)      | 2048    | 512, 1024, 2048, 4096 |

### Performance & Display

| Variable              | Description                      | Default | Range             |
| --------------------- | -------------------------------- | ------- | ----------------- |
| `VISYX_TARGET_FPS_MS` | Target frame time (milliseconds) | 16      | 8 - 50            |
| `VISYX_MODE`          | Display mode                     | "rows"  | "rows", "columns" |
| `VISYX_HUD`           | Show header with device info     | 0       | 0, 1              |

### Audio Processing

| Variable           | Description                      | Default | Range         |
| ------------------ | -------------------------------- | ------- | ------------- |
| `VISYX_TAU_SPEC`   | Spectrum smoothing time constant | 0.06    | 0.01 - 0.20   |
| `VISYX_GATE_DB`    | Noise gate threshold (dB)        | -55.0   | -80.0 - -30.0 |
| `VISYX_TILT_ALPHA` | Frequency response tilt factor   | 0.30    | 0.0 - 1.0     |

### Animation Physics

| Variable         | Description                       | Default | Range        |
| ---------------- | --------------------------------- | ------- | ------------ |
| `VISYX_FLOW_K`   | Horizontal flow between bars      | 0.18    | 0.0 - 1.0    |
| `VISYX_SPR_K`    | Spring stiffness for bar movement | 60.0    | 10.0 - 200.0 |
| `VISYX_SPR_ZETA` | Spring damping factor             | 1.0     | 0.1 - 2.0    |

### Example Configurations

**High-Resolution Mode** (more frequency detail):

```bash
VISYX_FFT_SIZE=4096 VISYX_FMIN=20 VISYX_FMAX=20000 visyx
```

**Performance Mode** (smooth on lower-end systems):

```bash
VISYX_TARGET_FPS_MS=33 VISYX_FFT_SIZE=1024 visyx
```

**Bass-Heavy Music**:

```bash
VISYX_FMIN=20 VISYX_FMAX=8000 VISYX_TILT_ALPHA=0.1 visyx
```

**Classical Music**:

```bash
VISYX_FMIN=40 VISYX_FMAX=12000 VISYX_TAU_SPEC=0.12 visyx
```

## How It Works

visyx runs a fast, low-latency audio pipeline built to turn raw sound into smooth, responsive visuals. It starts by hooking directly into your system audio, through a loopback device, so what you see is exactly what you hear, no setup friction, no manual wiring.

The signal is then windowed with Hann smoothing to avoid leakage and pushed through an FFT to break the stream into frequency components. Those raw bins are remapped onto a mel-scale filterbank so the output reflects how we actually perceive sound rather than just a grid of math.

To keep the display clean and balanced, it constantly adjusts its dynamic range. It tracks energy percentiles to scale loudness in real time, cuts off background hiss with noise gating, and applies frequency tilt so no single band overwhelms the others. The result is a visual field that feels natural and reacts instantly to changes in the mix.

On top of that sits a lightweight physics model. Instead of raw bar jumps, visyx runs each band through a spring-damper system so motion carries weight and flow. Energy diffuses between neighbors, giving the spectrum a fluid, wave-like feel. Every calculation is written to avoid allocations on the hot path, which means it runs smooth even at high refresh.

Finally, the renderer pushes everything to the terminal with dense Unicode block characters, letting gradients show up clean without burning cycles. FFT calls are SIMD-accelerated where hardware allows, memory access is cache-friendly, and processing adapts automatically to the system it runs on.

On modern machines, this translates to a consistent 60+ frames per second with sub-5ms audio latency—snappy enough that the visuals feel like part of the sound itself.

## License

[MIT](LICENSE)
