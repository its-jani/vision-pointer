# Vision Pointer — Gesture Daemon

A lightweight Rust daemon that captures video from a webcam and delivers
consistently-sized RGB frames, benchmarked for throughput and latency. It is
the foundation (Phase 1) for a gesture-driven control system.

## Overview

The program opens the default webcam, negotiates the cheapest stream the driver can
actually deliver at the requested frame rate, pumps frames on a dedicated capture
thread, and downsamples each one into a pre-allocated buffer. It then prints rolling
and average FPS so the capture pipeline can be measured.

## Features

- **Webcam capture** via [`nokhwa`](https://github.com/l15n42pw1r/nokhwa) using the auto/MediaFoundation backend.
- **Format negotiation** — the driver's full format list is scored and the stream with the
  best frame rate per USB byte is requested (640x480 MJPEG @ 30 FPS in preference to the
  1280x720 raw default).
- **Threaded frame pump** — a capture thread owns the device, decodes and downsamples, then
  publishes into a mailbox; the consumer takes the newest frame without stalling on driver I/O.
  If the pump cannot start, it falls back to blocking capture on the calling thread.
- **Zero-copy / zero-allocation frames** — RGB data is written into a pre-allocated buffer reused across frames.
- **Nearest-neighbor downsampling** implemented without extra heap allocations.
- **FPS benchmarking** — per-frame timing, rolling FPS, and average FPS output.

Measured on a *XiaoMi USB 2.0 Webcam*: negotiating 640x480 MJPEG @ 30 FPS over the threaded
pump lifts the loop from ~9.6 FPS (native 1280x720) to ~30 FPS steady / 27.6 FPS average.

## Architecture

```
src/
├── main.rs    # CLI entry point, capture loop, benchmark output
└── camera.rs  # CameraCapture wrapper: open, decode, downsample
```

| Module | Responsibility |
| ------ | -------------- |
| `main.rs` | Initializes the camera and runs the capture/benchmark loop. |
| `camera.rs` | Format negotiation, the capture thread + frame mailbox, decoding, and downsampling behind `CameraCapture`. |

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

1. `CameraCapture::new(0, 320, 240, 30)` opens camera index `0`, targeting a
   `320x240` processing buffer at 30 FPS.
2. Available video devices are enumerated and printed.
3. The driver's advertised format list is scored (`format_score`) — frame rate first, then
   the 640x480 pixel budget, then wire format (MJPEG > NV12/YUYV > raw RGB) — and the winner
   is requested with `RequestedFormatType::Exact`.
4. A capture thread opens the stream, and per frame it decodes to RGB, downsamples into its
   own fixed buffer, bumps a sequence counter and signals a condvar.
5. The loop reads the newest published frame and every `20` frames prints buffer size,
   frame capture time, rolling FPS, and average FPS.
6. A summary of total frames, elapsed time, average FPS, and the >= 20 FPS verdict is printed.

## Configuration

Adjust behavior by editing the arguments in `src/main.rs`:

```rust
let mut cam = CameraCapture::new(0, 320, 240, 30)?;
//                                ^   ^   ^   ^
//                          index  w   h  fps (target)
```

- **Camera index** — select a different webcam (`0` is the default).
- **Target width / height** — processing buffer dimensions.
- **`fps`** — the frame-rate target the negotiation scores formats against.
- **`max_frames`** — number of frames captured during the benchmark.
- Tuning constants live at the top of `src/camera.rs` (`NATIVE_MAX_WIDTH`/`HEIGHT`,
  `MIN_ACCEPTABLE_FPS`, `FRAME_TIMEOUT`).

## What's Next

Phase 1 (capture pipeline) is complete and running at ~30 FPS. Phase 2 layers ONNX Runtime
hand-landmark extraction on top of the pumped RGB frames, followed by gesture recognition
and pointer control.

## License

This project is currently unlicensed starter code. Add a `LICENSE` file before
distribution if needed.
