# Vision Pointer — Gesture Daemon

A lightweight Rust daemon that captures video from a webcam and delivers
consistently-sized RGB frames, benchmarked for throughput and latency. It is
the foundation (Phase 1) for a gesture-driven control system.

## Overview

The program opens the default webcam, grabs a stream of frames, decodes them
into RGB, and downsamples each frame into a pre-allocated buffer. It then
prints rolling and average FPS so the capture pipeline can be measured.

## Features

- **Webcam capture** via [`nokhwa`](https://github.com/l15n42pw1r/nokhwa) using the auto/MediaFoundation backend.
- **Zero-copy / zero-allocation frames** — RGB data is written into a pre-allocated buffer reused across frames.
- **Nearest-neighbor downsampling** implemented without extra heap allocations.
- **Format fallback** — if the native format request fails, it retries with `AbsoluteHighestFrameRate`.
- **FPS benchmarking** — per-frame timing, rolling FPS, and average FPS output.

## Architecture

```
src/
├── main.rs    # CLI entry point, capture loop, benchmark output
└── camera.rs  # CameraCapture wrapper: open, decode, downsample
```

| Module | Responsibility |
| ------ | -------------- |
| `main.rs` | Initializes the camera and runs the capture/benchmark loop. |
| `camera.rs` | Encapsulates camera access, frame decoding, and downsampling into `CameraCapture`. |

## Requirements

- **Rust** toolchain (edition 2024). Install via [rustup](https://rustup.rs/).
- A working **webcam** accessible to the OS.
- **Windows**: MediaFoundation support is used automatically by `nokhwa`'s `input-native` feature.

## Dependencies

Defined in [`Cargo.toml`](Cargo.toml):

- `anyhow` — ergonomic error handling.
- `image` — image decoding / pixel format conversion.
- `nokhwa` (feature `input-native`) — cross-platform camera capture.

## Getting Started

```bash
# Clone / enter the project directory
cd vision-pointer

# Build
cargo build

# Run the Phase 1 capture benchmark
cargo run
```

On first launch, `cargo` downloads and compiles the dependencies, so the initial
run may take a moment.

## How It Works

1. `CameraCapture::new(0, 320, 240, 20)` opens camera index `0`, targeting a
   `320x240` processing buffer.
2. Available video devices are enumerated and printed.
3. The camera stream is opened with the driver's native format; a fallback
   format is attempted if the initial request fails.
4. For each of `100` frames:
   - The frame is grabbed and decoded into RGB.
   - If the source resolution differs from the target, it is downsampled
     into the pre-allocated buffer.
   - Every `20` frames, buffer size, frame capture time, rolling FPS, and
     average FPS are printed.
5. A summary of total frames, elapsed time, and average FPS is printed at the
   end.

## Configuration

Adjust behavior by editing the arguments in `src/main.rs`:

```rust
let mut cam = CameraCapture::new(0, 320, 240, 20)?;
//                                ^   ^   ^   ^
//                          index  w   h  fps (target)
```

- **Camera index** — select a different webcam (`0` is the default).
- **Target width / height** — processing buffer dimensions.
- **`max_frames`** — number of frames captured during the benchmark.

## What's Next

Phase 1 focuses on the capture pipeline. Future phases are expected to layer
gesture recognition and pointer control on top of these RGB frames.

## License

This project is currently unlicensed starter code. Add a `LICENSE` file before
distribution if needed.
