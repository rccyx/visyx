# Lookas

A terminal-based audio spectrum visualizer written in Rust.

## Description

Lookas is a lightweight audio spectrum analyzer that displays real-time audio visualization in your terminal. It captures audio input, performs FFT analysis, and renders the frequency spectrum as either vertical bars (like a traditional spectrum analyzer) or horizontal bars.

## Features

- Real-time audio visualization in the terminal
- Automatic input device detection (prefers monitor devices)
- Mel-scale frequency bands for better representation of how humans perceive sound
- Vertical and horizontal visualization modes
- Automatic gain control to adapt to different audio levels
- Smooth animations with spring physics

## Installation

```
cargo install lookas
```

### From source

```
git clone git@github.com:ashgw/lookas.git
cd lookas
cargo build --release
```

The binary will be available at `target/release/lookas`.

## Usage

Run the program:

```
lookas
```

### Controls

- `v`: Switch to vertical visualization mode
- `h`: Switch to horizontal visualization mode
- `q`: Quit the application

## Requirements

- Rust 1.70 or higher
- ALSA development libraries (Linux)
- CoreAudio (macOS)
- WASAPI (Windows)

### Linux Dependencies

```
sudo apt install libasound2-dev # on your preferred package manager
```

### macOS Dependencies

No additional dependencies required.

### Windows Dependencies

No additional dependencies required.

## How It Works

1. Audio is captured from the default input device or a monitor device if available
2. The audio signal is processed through a Fast Fourier Transform (FFT)
3. The frequency spectrum is divided into Mel-scale bands for better perceptual representation
4. The spectrum is visualized in the terminal using Unicode block characters
5. Spring physics and lateral energy flow are applied to make the visualization more organic

## Configuration

Currently, configuration is hardcoded in the source. You can modify the following constants in `src/main.rs`:

- `FMIN`: Minimum frequency to analyze (default: 30 Hz)
- `FMAX`: Maximum frequency to analyze (default: 16,000 Hz)
- `TARGET_FPS_MS`: Target frame rate in milliseconds (default: 16ms, ~60 FPS)
- `FFT_SIZE`: Size of the FFT window (default: 2048)
- `GATE_DB`: Noise gate threshold (default: -55 dB)

## License

MIT
