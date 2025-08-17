# Lookas

A terminal-based audio spectrum visualizer written in Rust.

## Description

Lookas is a lightweight audio spectrum analyzer that displays real-time audio visualization in your terminal. It captures audio input, performs FFT analysis, and renders the frequency spectrum as either vertical bars (like a traditional spectrum analyzer) or horizontal bars.

<details>
  <summary>Click to see the preview</summary>

  ![Visualizer preview](https://github.com/user-attachments/assets/1533afdf-3826-4e14-839f-9b880864bac1)

</details>


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

## Usage

Simply run the binary:

```
lookas
```

### Controls

- `v`: Switch to vertical visualization mode
- `h`: Switch to horizontal visualization mode
- `q`: Quit the application

## Needed

- Rust 1.70 or higher
- ALSA development libraries (Linux)
- CoreAudio (macOS)
- WASAPI (Windows)

### Linux Dependencies

```
sudo apt install libasound2-dev # on your preferred package manager
```

## How It Works

1. Audio is captured from the default input device or a monitor device if available
2. The audio signal is processed through a Fast Fourier Transform (FFT)
3. The frequency spectrum is divided into Mel-scale bands for better perceptual representation
4. The spectrum is visualized in the terminal using Unicode block characters
5. Spring physics and lateral energy flow are applied to make the visualization more organic

## Configuration

You can configure it using environment variables. The default values are used if no environment variables are set.

| Environment Variable   | Description                                       | Default Value  |
| ---------------------- | ------------------------------------------------- | -------------- |
| `LOOKAS_FMIN`          | Minimum frequency to analyze                      | 30 Hz          |
| `LOOKAS_FMAX`          | Maximum frequency to analyze                      | 16,000 Hz      |
| `LOOKAS_TARGET_FPS_MS` | Target frame rate in milliseconds                 | 16ms (~60 FPS) |
| `LOOKAS_FFT_SIZE`      | Size of the FFT window                            | 2048           |
| `LOOKAS_GATE_DB`       | Noise gate threshold                              | -55 dB         |
| `LOOKAS_TAU_SPEC`      | Spectrum smoothing time constant                  | 0.06           |
| `LOOKAS_TILT_ALPHA`    | Frequency tilt factor (reduces low end dominance) | 0.30           |
| `LOOKAS_FLOW_K`        | Lateral energy flow coefficient                   | 0.18           |
| `LOOKAS_SPR_K`         | Spring stiffness for smooth motion                | 60.0           |
| `LOOKAS_SPR_ZETA`      | Spring damping factor                             | 1.0            |

### Example

```bash
# Run with custom settings
LOOKAS_FMIN=50 LOOKAS_FMAX=12000 LOOKAS_GATE_DB=-60 lookas
```

## License

[MIT](/LICENSE)
