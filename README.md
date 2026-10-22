# Vision Pointer — Gesture Daemon

A lightweight Rust daemon that captures video from a webcam and extracts 21 hand
landmarks per frame with the two-stage MediaPipe models through ONNX Runtime. It is
the foundation for a gesture-driven control system.

## Overview

The program opens the default webcam, negotiates the cheapest stream the driver can
actually deliver at the requested frame rate, pumps frames on a dedicated capture
thread, and downsamples each one into a pre-allocated buffer. That buffer is then fed
through palm detection and hand-landmark estimation, printing keypoints and per-stage
timings as they arrive.

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
- **Two-stage hand landmarks** — palm detection (192x192, 2016 SSD anchors, NMS) followed by
  hand-landmark estimation (224x224) via [`ort`](https://ort.pyke.io/), returning 21 screen
  keypoints in frame pixels, 21 metric world keypoints, handedness, and a confidence score.
- **Explicit model fetching** — nothing is downloaded by the build or by running `bench`; the
  `.onnx` assets are fetched only when `fetch-models` is invoked.

Measured on a *XiaoMi USB 2.0 Webcam*: negotiating 640x480 MJPEG @ 30 FPS over the threaded
pump lifts the loop from ~9.6 FPS (native 1280x720) to ~30 FPS steady / 27.6 FPS average.

## Architecture

```
src/
├── main.rs         # CLI dispatch: bench | hands | fetch-models | describe | help
├── camera.rs       # CameraCapture wrapper: open, decode, downsample
├── hands.rs        # HandPipeline: palm detection -> ROI -> hand landmarks
├── model.rs        # Model registry, `fetch-models` downloader, session/schema helpers
├── roi.rs          # Affine/rect geometry and single-pass resampling into model blobs
└── palm_anchors.rs # The palm detector's 2016 SSD anchors, transcribed from the reference
```

| Module | Responsibility |
| ------ | -------------- |
| `main.rs` | Argument parsing and the capture/benchmark and live landmark loops. |
| `camera.rs` | Format negotiation, the capture thread + frame mailbox, decoding, and downsampling behind `CameraCapture`. |
| `hands.rs` | Both inference stages: letterbox, anchor decode, NMS, ROI shift/enlarge/rotation, keypoint de-rotation. |
| `model.rs` | Where models live, how to download them, and how to open an `ort` session. |
| `roi.rs` | Composes stage transforms into one destination -> source mapping and resamples the frame once. |
| `palm_anchors.rs` | Constant anchor table the palm decoder pairs its regressions with. |

## Requirements

- **Rust** toolchain (edition 2024). Install via [rustup](https://rustup.rs/).
- A working **webcam** accessible to the OS.
- **Windows**: MediaFoundation support is used automatically by `nokhwa`'s `input-native` feature.
- **ONNX Runtime**, which the `download-binaries` feature of `ort` fetches and links on the first
  `cargo build` (cached under `%LOCALAPPDATA%\ort.pyke.io`).

## Dependencies

Defined in [`Cargo.toml`](Cargo.toml):

- `anyhow` — ergonomic error handling.
- `image` — image decoding / pixel format conversion.
- `nokhwa` (feature `input-native`) — cross-platform camera capture.
- `ort` (feature `download-binaries`) — ONNX Runtime bindings for both model stages.
- `ureq` — blocking HTTP for the `fetch-models` command only.

## Getting Started

```bash
# Clone / enter the project directory
cd vision-pointer

# Build (first build also downloads and links ONNX Runtime)
cargo build --release

# Phase 1 only: measure the capture pipeline
cargo run --release -- bench

# Phase 2: download the two MediaPipe ONNX graphs into ./models
cargo run --release -- fetch-models

# Confirm what the conversion actually exposes (tensor names, dtypes, shapes)
cargo run --release -- describe

# Run palm detection + landmark extraction over the live feed
cargo run --release -- hands
```

On first launch, `cargo` downloads and compiles the dependencies, so the initial
run may take a moment. The two float32 models total ~7.6 MB (3.7 MB palm + 3.9 MB handpose);
they are never fetched automatically, `./models` is git-ignored, and `VISION_POINTER_MODELS`
can point at a shared directory instead.

## How It Works

### Capture

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

### Landmark pipeline (`hands`)

1. The 320x240 frame is letterboxed into a 192x192 float blob (`/255`, the padding offset is
   recorded as `pad_bias`) and run through the palm graph.
2. Both regressor and score outputs are matched by name and element count, then decoded: the
   2016 anchors are combined with the box/landmark deltas and a sigmoid gives each box a score.
3. Boxes scoring below `0.5` are dropped, the rest go through greedy IoU suppression at `0.3`,
   and at most `--max-hands` survivors are promoted.
4. Each palm box is pre-shifted/enlarged, squared, rotated to upright using the wrist and
   middle-finger MCP points (`90° - atan2` of their vector), then re-shifted and enlarged into a
   224x224 crop. The whole recipe is composed into one affine mapping and sampled once.
5. The hand graph returns 21 screen landmarks, a presence score, handedness logits, and 21 world
   landmarks. Screen keypoints are de-rotated and un-letterboxed back into frame pixels; the
   hand is reported once the presence score clears `--min-confidence`.
6. Each frame with a hand prints fingertip position, depth, box size, confidence, handedness,
   and per-stage inference milliseconds.

The arithmetic follows the Apache-2.0 OpenCV Zoo reference (`mp_palmdet.py`, `mp_handpose.py`).
Unit tests in `hands.rs` cover the geometry that can be checked without weights — letterbox
mapping, rotation angle, ROI plan, output matching, box quantisation — so **the numeric output
against real weights has not been verified yet**; run `describe` first and compare the printed
tensor contract with the assumptions in `hands.rs` if the landmarks look wrong.

## Configuration

CLI options (`cargo run --release -- help`):

```
--camera N          Camera device index (default: 0)
--frames N          Stop after N frames, 0 runs until interrupted (hands only)
--max-hands N       Palm boxes promoted to hands per frame (default: 1)
--min-confidence F  Landmark score a hand needs to be reported, 0-1 (default: 0.5)
--force             Re-download models that are already present (fetch-models only)
```

Source-level knobs:

- **Processing resolution** — `PROCESS_WIDTH`/`PROCESS_HEIGHT`/`TARGET_FPS` in `src/main.rs`.
  These are the pixels the models see; lowering them trades accuracy for headroom, and the pump
  already delivers more frames than inference can consume, so the loop always uses the newest.
- **Capture tuning** — `NATIVE_MAX_WIDTH`/`HEIGHT`, `MIN_ACCEPTABLE_FPS`, `FRAME_TIMEOUT` at the
  top of `src/camera.rs`.
- **Pipeline thresholds** — `PALM_SCORE_THRESHOLD`, `PALM_NMS_IOU_THRESHOLD`,
  `HAND_PRESENCE_THRESHOLD` in `src/hands.rs`; ROI shift/enlarge factors are the
  `*_BOX_SHIFT`/`*_BOX_ENLARGE` constants taken from the reference.
- **Model location** — `VISION_POINTER_MODELS` overrides the default `./models` directory.

## What's Next

Phase 2 (landmark extraction) is wired and compiles clean; it needs a `fetch-models` run to
validate the decoded geometry against real weights. Next: track hands across frames, classify
gestures from the 21 keypoints, and map them to pointer input.

## License

This project is currently unlicensed starter code. Add a `LICENSE` file before
distribution if needed.
