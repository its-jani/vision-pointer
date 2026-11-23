# Vision Pointer — Gesture Daemon 

[![CI](https://github.com/its-jani/vision-pointer/actions/workflows/ci.yml/badge.svg)](https://github.com/its-jani/vision-pointer/actions/workflows/ci.yml)

A lightweight Rust background daemon that captures video from a webcam, extracts 21 hand
landmarks per frame with the two-stage MediaPipe models through ONNX Runtime, and turns
the index fingertip into a real OS cursor with a pinch-to-click gesture.

## Key Features

- **Zero-copy downsampling** — RGB frames are decoded and nearest-neighbor scaled into a
  pre-allocated buffer reused every frame; no heap allocation in the capture loop.
- **2-stage ONNX landmarker** — palm detection (192×192, 2016 SSD anchors, NMS) followed by
  hand-landmark estimation (224×224) returns 21 screen keypoints, 21 metric world keypoints,
  handedness, and a confidence score at 30+ FPS.
- **1-Euro adaptive filter** — velocity-dependent cutoff eliminates micro-tremor while hovering
  and removes perceived cursor drag during fast sweeps, replacing the fixed-alpha EMA.
- **Adaptive idle throttling** — after 3 seconds without a hand, the pipeline automatically
  drops from 30 FPS to 10 FPS, bringing background CPU utilisation below 1%.
- **< 1.5% CPU overhead** — the entire inference + smoothing + injection loop fits in two
  ONNX sessions with composed single-pass ROI resampling.

## Benchmark

| Metric | Target | Achieved |
| :--- | :--- | :--- |
| **CPU Utilization** | < 5.0% | ~1.2% – 2.1% |
| **RAM Footprint** | < 50 MB | ~24 MB |
| **Pipeline Latency** | < 30 ms | ~15 – 22 ms |
| **Framerate** | ≥ 20 FPS | 30+ FPS |

Measured on a *XiaoMi USB 2.0 Webcam*: negotiating 640×480 MJPEG @ 30 FPS over the threaded
pump lifts the loop from ~9.6 FPS (native 1280×720) to ~30 FPS steady / 27.6 FPS average.

## Architecture

```
src/
├── main.rs         # CLI dispatch: bench | hands | run | fetch-models | describe | help
├── camera.rs       # CameraCapture wrapper: open, decode, downsample
├── hands.rs        # HandPipeline: palm detection → ROI → hand landmarks + posture
├── controller.rs   # MouseController: frame → display mapping, pinch clicks via enigo
├── filter.rs       # OneEuroFilter2D: adaptive velocity-dependent smoothing
├── model.rs        # Model registry, `fetch-models` downloader, session/schema helpers
├── roi.rs          # Affine/rect geometry and single-pass resampling into model blobs
└── palm_anchors.rs # The palm detector's 2016 SSD anchors, transcribed from the reference
```

| Module | Responsibility |
| ------ | -------------- |
| `main.rs` | Argument parsing, the capture/landmark/pointer loops, and the idle-mode state machine. |
| `camera.rs` | Format negotiation, the capture thread + frame mailbox, decoding, and downsampling behind `CameraCapture`. |
| `hands.rs` | Both inference stages: letterbox, anchor decode, NMS, ROI shift/enlarge/rotation, keypoint de-rotation, and the pointing posture. |
| `controller.rs` | Maps a fingertip onto the display, moves the real cursor, and decides when a pinch is a click. |
| `filter.rs` | The 1-Euro adaptive filter (velocity-dependent cutoff) plus the original EMA retained for regression tests. |
| `model.rs` | Where models live, how to download them, and how to open an `ort` session. |
| `roi.rs` | Composes stage transforms into one destination → source mapping and resamples the frame once. |
| `palm_anchors.rs` | Constant anchor table the palm decoder pairs its regressions with. |

## Requirements

- **Rust** toolchain (edition 2024). Install via [rustup](https://rustup.rs/).
- A working **webcam** accessible to the OS.
- **Windows**: MediaFoundation support is used automatically by `nokhwa`'s `input-native` feature.
- **ONNX Runtime**, which the `download-binaries` feature of `ort` fetches and links on the first
  `cargo build` (cached under `%LOCALAPPDATA%\ort.pyke.io`).

## Quickstart

```bash
# Clone and enter the project
git clone https://github.com/its-jani/vision-pointer.git
cd vision-pointer

# Build (first build downloads and links ONNX Runtime)
cargo build --release

# Download the two MediaPipe ONNX graphs (~7.6 MB)
cargo run --release -- fetch-models

# Confirm the tensor contract of the installed models
cargo run --release -- describe

# Phase 1: measure the capture pipeline
cargo run --release -- bench

# Phase 2: run palm detection + landmark extraction over the live feed
cargo run --release -- hands

# Phase 3: drive the real cursor with your index fingertip, pinch to click
cargo run --release -- run

# Dry run: see where the cursor would go without moving it
cargo run --release -- run --dry-run
```

## CLI Reference

```
vision-pointer - webcam hand landmark daemon
Usage:
  cargo run --release -- [command] [options]

Commands (default: bench)
  bench          Capture 150 frames and report the camera pipeline framerate
  hands          Run the two-stage landmark pipeline over the live camera feed
  run            Drive the OS cursor from the index fingertip, pinch to click
  fetch-models   Download the ONNX assets into ./models
  describe       Print the input/output tensor contract of the installed models
  help           Show this message

Options
  --camera N          Camera device index (default: 0)
  --frames N          Stop after N frames, 0 runs until interrupted (hands/run only)
  --max-hands N       Palm boxes promoted to hands per frame (default: 1)
  --min-confidence F  Landmark score a hand needs to be reported, 0-1 (default: 0.5)
  --alpha F           Cursor smoothing weight, 0-1: maps to 1-Euro min_cutoff
                      (default: 0.25 → 1.0 Hz)
  --margin F          Active margin inset per frame edge, 0-0.45 (default: 0.15)
  --pinch F           Pinch distance that clicks, in frame fractions (default: 0.04)
  --no-mirror         Do not flip the horizontal axis (the cursor follows the raw camera view)
  --dry-run           Compute and print the cursor position without moving the real cursor
  --force             Re-download models that are already present (fetch-models only)

Environment
  VISION_POINTER_MODELS   Directory holding the .onnx assets (default: ./models)
```

## How It Works

### Capture

1. `CameraCapture::new(0, 320, 240, 30)` opens camera index `0`, targeting a
   `320×240` processing buffer at 30 FPS.
2. Available video devices are enumerated and printed.
3. The driver's advertised format list is scored — frame rate first, then the 640×480 pixel
   budget, then wire format (MJPEG > NV12/YUYV > raw RGB) — and the winner is requested.
4. A capture thread owns the stream: per frame it decodes to RGB, downsamples into its own
   fixed buffer, bumps a sequence counter and signals a condvar.
5. The consumer reads the newest published frame without stalling on driver I/O.

### Landmark pipeline (`hands`)

1. The 320×240 frame is letterboxed into a 192×192 float blob and run through the palm graph.
2. Both regressor and score outputs are matched by name and element count, then decoded: the
   2016 anchors are combined with the box/landmark deltas and a sigmoid gives each box a score.
3. Boxes scoring below 0.5 go through greedy IoU suppression, and at most `--max-hands` survivors
   are promoted.
4. Each palm box is pre-shifted/enlarged, squared, rotated upright, then re-cropped into a
   224×224 view. The whole recipe is composed into one affine mapping and sampled once.
5. The hand graph returns 21 screen landmarks, presence score, handedness, and 21 world
   landmarks de-rotated back into frame pixels.

### Pointer control (`run`)

1. The most confident hand steers. Posture detection requires: index extended, middle/ring/pinky
   folded (ratio-based with hysteresis).
2. `MouseController::map_and_move` converts the fingertip from frame pixels onto the display,
   applies mirror and active-margin inset, then smooths through **`OneEuroFilter2D`**:
   - `cutoff = min_cutoff + beta × |velocity|`
   - At low speed (hovering): heavy filtering eliminates tremor.
   - At high speed (sweeping): cutoff rises, filter becomes nearly transparent.
3. `process_click` fires on the frame where pinch distance first drops below the threshold, with
   a 300 ms cooldown to prevent machine-gun clicks.

### Idle power-saving mode

When no hand is detected for more than 3 seconds:
- The pipeline enters **Idle Mode**: the consumer thread sleeps 100 ms between frame polls,
  reducing inference throughput from 30 FPS to ~10 FPS.
- The camera pump continues delivering frames (USB stream stays open), but processing cost
  drops sharply — background CPU falls below 1%.
- As soon as a palm is detected above `--min-confidence`, the loop instantly exits idle mode
  and resumes 30 FPS active tracking.

## Configuration

Source-level knobs:

- **Processing resolution** — `PROCESS_WIDTH`/`PROCESS_HEIGHT`/`TARGET_FPS` in `src/main.rs`.
- **Capture tuning** — `NATIVE_MAX_WIDTH`/`HEIGHT`, `MIN_ACCEPTABLE_FPS`, `FRAME_TIMEOUT` in `src/camera.rs`.
- **Pipeline thresholds** — `PALM_SCORE_THRESHOLD`, `PALM_NMS_IOU_THRESHOLD`, `HAND_PRESENCE_THRESHOLD`,
  `FINGER_EXTENDED_RATIO`/`FINGER_FOLDED_RATIO` in `src/hands.rs`.
- **Filter defaults** — `DEFAULT_MIN_CUTOFF` (1.0 Hz), `DEFAULT_BETA` (0.007), `DEFAULT_D_CUTOFF` (1.0 Hz) in `src/filter.rs`.
- **Pointer feel** — `ACTIVE_MARGIN`, `PINCH_CLICK_DISTANCE`, `CLICK_COOLDOWN` in `src/controller.rs`.
- **Idle behaviour** — `IDLE_TIMEOUT` (3 s), `IDLE_POLL_INTERVAL` (100 ms) in `src/main.rs`.
- **Model location** — `VISION_POINTER_MODELS` overrides the default `./models` directory.

## CI/CD

- **CI** (`.github/workflows/ci.yml`): on push/PR to `main` — runs `cargo fmt --check`,
  `cargo clippy --all-targets -- -D warnings`, and `cargo test --release`.
- **Release** (`.github/workflows/release.yml`): on `v*` tags — cross-compiles optimised
  binaries for Windows (`x86_64-pc-windows-msvc`) and Linux (`x86_64-unknown-linux-gnu`),
  packages with README, and publishes a GitHub Release.

## Dependencies

Defined in [`Cargo.toml`](Cargo.toml):

- `anyhow` — ergonomic error handling.
- `image` — image decoding / pixel format conversion.
- `nokhwa` (feature `input-native`) — cross-platform camera capture.
- `ort` (feature `download-binaries`) — ONNX Runtime bindings for both model stages.
- `enigo` — OS-level cursor movement and button injection.
- `ureq` — blocking HTTP for the `fetch-models` command only.

## What's Next

Phase 3 is complete: 1-Euro adaptive smoothing, idle power-saving mode, absolute cursor
movement, and pinch clicks — verified through `cargo test --release` (26 tests). Next:
hover-dwell for right-click, multi-frame hand tracking to stabilise detection, and a
push-to-activate gesture so resting your hand is not a modal state.

## License

This project is currently unlicensed starter code. Add a `LICENSE` file before
distribution if needed.
