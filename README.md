# Lookas

A terminal-based audio spectrum visualizer written in Rust.

![Lookas Visualization](https://github.com/user-attachments/assets/1190509f-400b-46cb-adbe-5ea93c186199)

## How It Works

Lookas captures audio from your system's input device, performs a Fast Fourier Transform (FFT) to analyze the frequency content, and displays the result as animated bars in your terminal.

## Installation

```bash
cargo install lookas
```

## Usage

Simply run:

```bash
lookas
```

### Configuration

Lookas can be configured using environment variables:

| Variable               | Description                          | Default |
| ---------------------- | ------------------------------------ | ------- |
| `LOOKAS_FMIN`          | Minimum frequency (Hz)               | 30.0    |
| `LOOKAS_FMAX`          | Maximum frequency (Hz)               | 16000.0 |
| `LOOKAS_TARGET_FPS_MS` | Target frame time in milliseconds    | 16      |
| `LOOKAS_FFT_SIZE`      | FFT size                             | 2048    |
| `LOOKAS_TAU_SPEC`      | Spectrum smoothing time constant     | 0.06    |
| `LOOKAS_GATE_DB`       | Noise gate threshold (dB)            | -55.0   |
| `LOOKAS_TILT_ALPHA`    | Frequency tilt factor                | 0.30    |
| `LOOKAS_FLOW_K`        | Flow coefficient for bar interaction | 0.18    |
| `LOOKAS_SPR_K`         | Spring stiffness                     | 60.0    |
| `LOOKAS_SPR_ZETA`      | Spring damping factor                | 1.0     |
| `LOOKAS_MODE`          | Default mode ("rows" or "columns")   | "rows"  |

Example:

```bash
LOOKAS_FMIN=50 LOOKAS_FMAX=10000 lookas
```

## License

MIT License
