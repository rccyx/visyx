# Lookas

A terminal-based audio spectrum visualizer written in Rust.

![Lookas Visualization](https://raw.githubusercontent.com/ashgw/lookas/main/.github/assets/lookas_preview.gif)

## Features

- Real-time audio spectrum visualization in your terminal
- Automatic input device selection
- Multiple visualization modes:
  - Horizontal bars (default)
  - Vertical bars
  - Row mode (one band per row)
  - Column mode (bands averaged into rows)
- Smooth animations with spring physics
- Adaptive gain control
- Low CPU usage
- Cross-platform support

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/ashgw/lookas
cd lookas

# Build and install
cargo install --path .
```

## Usage

Simply run:

```bash
lookas
```

### Controls

- `h` - Switch to horizontal orientation
- `v` - Switch to vertical orientation
- `m` - Toggle between row and column modes
- `q` - Quit

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

## How It Works

Lookas captures audio from your system's input device, performs a Fast Fourier Transform (FFT) to analyze the frequency content, and displays the result as animated bars in your terminal. The visualization includes:

1. Audio capture using the CPAL library
2. FFT processing with rustfft
3. Mel-scale frequency band analysis
4. Dynamic range compression with automatic gain control
5. Spring physics for smooth animations
6. Terminal rendering using crossterm

## Development

```bash
# Run the application
cargo run

# Format code
just format

# Run linter
just clippy

# Clean build artifacts
just clean
```

## License

MIT License
