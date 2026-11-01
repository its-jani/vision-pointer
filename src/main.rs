mod camera;
mod controller;
mod filter;
mod hands;
mod model;
mod palm_anchors;
mod roi;

use anyhow::{Result, bail};
use camera::CameraCapture;
use controller::MouseController;
use hands::{Hand, HandPipeline, Posture, Timings};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// Resolution the vision models run at; the camera downsamples its native stream into this.
const PROCESS_WIDTH: u32 = 320;
const PROCESS_HEIGHT: u32 = 240;
/// Frame rate asked of the camera driver.
const TARGET_FPS: u32 = 30;
/// Frames captured by the `bench` command.
const BENCH_FRAMES: u32 = 150;
/// Frames averaged into the live FPS readout.
const FPS_WINDOW: u32 = 30;
/// Seconds without a detected hand before the pipeline enters idle mode.
const IDLE_TIMEOUT: Duration = Duration::from_secs(3);
/// Sleep between processed frames in idle mode, capping inference throughput to ~10 FPS.
/// The camera pump still delivers frames; only the consumer slows down.
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug)]
struct Flags {
    command: String,
    camera: u32,
    frames: u32,
    max_hands: usize,
    min_confidence: f64,
    alpha: f32,
    margin: f32,
    pinch: f32,
    no_mirror: bool,
    dry_run: bool,
    force: bool,
}

impl Default for Flags {
    fn default() -> Self {
        Flags {
            command: String::from("bench"),
            camera: 0,
            frames: 0,
            max_hands: 1,
            min_confidence: hands::HAND_PRESENCE_THRESHOLD,
            alpha: filter::DEFAULT_ALPHA,
            margin: controller::ACTIVE_MARGIN,
            pinch: controller::PINCH_CLICK_DISTANCE,
            no_mirror: false,
            dry_run: false,
            force: false,
        }
    }
}

impl Flags {
    fn parse(args: &[String]) -> Result<Flags> {
        let mut rest = args.to_vec();
        let mut flags = if rest
            .first()
            .is_some_and(|argument| !argument.starts_with('-'))
        {
            Flags {
                command: rest.remove(0),
                ..Flags::default()
            }
        } else {
            Flags::default()
        };

        let mut index = 0;
        while index < rest.len() {
            let flag = rest[index].as_str();
            index += 1;
            match flag {
                "--camera" => flags.camera = number(flag, value(&mut index, &rest, flag)?)?,
                "--frames" => flags.frames = number(flag, value(&mut index, &rest, flag)?)?,
                "--max-hands" => flags.max_hands = number(flag, value(&mut index, &rest, flag)?)?,
                "--min-confidence" => {
                    flags.min_confidence = number(flag, value(&mut index, &rest, flag)?)?
                }
                "--alpha" => flags.alpha = number(flag, value(&mut index, &rest, flag)?)?,
                "--margin" => flags.margin = number(flag, value(&mut index, &rest, flag)?)?,
                "--pinch" => flags.pinch = number(flag, value(&mut index, &rest, flag)?)?,
                "--no-mirror" => flags.no_mirror = true,
                "--dry-run" => flags.dry_run = true,
                "--force" => flags.force = true,
                "-h" | "--help" => flags.command = String::from("help"),
                other => bail!("unknown option `{other}` - run `cargo run --release -- help`"),
            }
        }
        Ok(flags)
    }
}

/// Reads the value that follows a long option.
fn value(index: &mut usize, rest: &[String], flag: &str) -> Result<String> {
    if *index >= rest.len() {
        bail!("`{flag}` requires a value");
    }
    let text = rest[*index].clone();
    *index += 1;
    Ok(text)
}

fn number<T: std::str::FromStr>(flag: &str, text: String) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    text.parse::<T>()
        .map_err(|why| anyhow::anyhow!("`{flag}` expects a number, got `{text}` ({why})"))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flags = Flags::parse(&args)?;
    match flags.command.as_str() {
        "bench" => bench(&flags),
        "hands" => hands(&flags),
        "run" => run(&flags),
        "fetch-models" => fetch_models(&flags),
        "describe" => describe(),
        "help" => {
            usage();
            Ok(())
        }
        other => {
            usage();
            bail!("unknown command `{other}`")
        }
    }
}

fn usage() {
    println!(
        "vision-pointer - webcam hand landmark daemon
Usage:
  cargo run --release -- [command] [options]

Commands (default: bench)
  bench          Capture {BENCH_FRAMES} frames and report the camera pipeline framerate
  hands          Run the two-stage landmark pipeline over the live camera feed
  run            Drive the OS cursor from the index fingertip, pinch to click
  fetch-models   Download the ONNX assets into ./models (nothing is fetched automatically)
  describe       Print the input/output tensor contract of the installed models
  help           Show this message

Options
  --camera N          Camera device index (default: 0)
  --frames N          Stop after N frames, 0 runs until interrupted (hands/run only)
  --max-hands N       Palm boxes promoted to hands per frame (default: 1)
  --min-confidence F  Landmark score a hand needs to be reported, 0-1 (default: 0.5)
  --alpha F           Cursor smoothing weight, 0-1: higher follows the hand, lower is glassy
                      (default: {DEFAULT_ALPHA})
  --margin F          Active margin inset per frame edge, 0-0.45 (default: {ACTIVE_MARGIN})
  --pinch F           Pinch distance that clicks, in frame fractions (default: {PINCH_CLICK_DISTANCE})
  --no-mirror         Do not flip the horizontal axis (the cursor follows the raw camera view)
  --dry-run           Compute and print the cursor position without moving the real cursor
  --force             Re-download models that are already present (fetch-models only)

Environment
  VISION_POINTER_MODELS   Directory holding the .onnx assets (default: ./models)",
        DEFAULT_ALPHA = filter::DEFAULT_ALPHA,
        ACTIVE_MARGIN = controller::ACTIVE_MARGIN,
        PINCH_CLICK_DISTANCE = controller::PINCH_CLICK_DISTANCE,
    );
}

fn open_camera(flags: &Flags, header: &str) -> Result<CameraCapture> {
    println!("============================================================");
    println!("  {header}");
    println!("============================================================");
    let camera = CameraCapture::new(flags.camera, PROCESS_WIDTH, PROCESS_HEIGHT, TARGET_FPS)?;
    let stream = camera.stream_info();
    let (width, height) = camera.resolution();
    println!("Camera Stream Opened:");
    println!(
        " - Native Camera Resolution: {}x{} {} @ {} FPS",
        stream.width, stream.height, stream.fourcc, stream.fps
    );
    println!(
        " - Frame Delivery: {}",
        if stream.pumped {
            "threaded pump (non-blocking capture loop)"
        } else {
            "blocking grab on caller thread"
        }
    );
    println!(
        " - Target Processing Buffer: {}x{} RGB ({} bytes/frame)",
        width,
        height,
        width * height * 3
    );
    Ok(camera)
}

/// Phase 1: measure the capture path on its own.
fn bench(flags: &Flags) -> Result<()> {
    let mut camera = open_camera(flags, "Gesture Daemon - Phase 1: Camera Capture Benchmark")?;

    println!(
        "\nCapturing {BENCH_FRAMES} frames to measure zero-copy buffer & FPS performance...\n"
    );

    let started = Instant::now();
    let mut window = Instant::now();
    let mut frames_in_window = 0u32;
    let mut captured = 0u32;
    while captured < BENCH_FRAMES {
        let frame_started = Instant::now();
        let frame = camera.capture_frame()?;
        let duration = frame_started.elapsed();
        captured += 1;
        frames_in_window += 1;

        if captured.is_multiple_of(20) {
            let rolling = frames_in_window as f64 / window.elapsed().as_secs_f64();
            println!(
                "Frame {:>3}/{BENCH_FRAMES}: buffer = {} bytes | frame capture = {:>6.2?} | rolling FPS = {:>5.1} | avg FPS = {:>5.1}",
                captured,
                frame.len(),
                duration,
                rolling,
                captured as f64 / started.elapsed().as_secs_f64()
            );
            window = Instant::now();
            frames_in_window = 0;
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    let average = captured as f64 / elapsed;
    println!("\n------------------------------------------------------------");
    println!("Phase 1 Verification Complete!");
    println!(" - Total Frames: {captured}");
    println!(" - Total Time:   {elapsed:.2}s");
    println!(" - Average FPS:  {average:.1} FPS");
    println!("------------------------------------------------------------");
    if average >= 20.0 {
        println!("Framerate target met (>= 20 FPS) - inference can now share the loop.");
    } else {
        println!("Framerate target NOT met (< 20 FPS) - capture path still needs tuning.");
    }
    Ok(())
}

/// Phase 2: run both MediaPipe stages over the live feed.
fn hands(flags: &Flags) -> Result<()> {
    let environment = model::init_environment()?;
    let mut pipeline = HandPipeline::new(&environment)?
        .with_max_hands(flags.max_hands)?
        .with_hand_threshold(flags.min_confidence);
    let mut camera = open_camera(flags, "Gesture Daemon - Phase 2: Hand Landmark Extraction")?;

    let (width, height) = camera.resolution();
    println!(
        "\nLandmark pipeline ready (max {} hand(s), min confidence {:.2}, processing {}x{}). Ctrl-C stops.\n",
        flags.max_hands, flags.min_confidence, width, height
    );

    let started = Instant::now();
    let mut window = Instant::now();
    let mut frames_in_window = 0u32;
    let mut captured = 0u32;
    let mut detected = 0u32;
    let mut detect_ms = 0.0f64;

    // Idle power-saving state machine.
    let mut last_hand_seen = Instant::now();
    let mut idle = false;

    loop {
        if flags.frames > 0 && captured >= flags.frames {
            break;
        }
        let frame = camera.capture_frame()?;
        let infer_started = Instant::now();
        let result = pipeline.detect(frame, width, height);
        let elapsed = infer_started.elapsed().as_secs_f64() * 1000.0;
        captured += 1;
        frames_in_window += 1;
        detect_ms += elapsed;

        let hands = result
            .map_err(|why| anyhow::anyhow!("frame {captured}: landmark pipeline failed: {why}"))?;
        if hands.is_empty() {
            if last_hand_seen.elapsed() > IDLE_TIMEOUT && !idle {
                idle = true;
                println!(
                    "[idle] no hand for {:.1}s - throttling to ~10 FPS",
                    IDLE_TIMEOUT.as_secs_f64()
                );
            }
        } else {
            if idle {
                idle = false;
                println!("[active] hand detected - resuming full rate");
            }
            last_hand_seen = Instant::now();
        }
        detected = detected.max(hands.len() as u32);
        let fps = frames_in_window as f64 / window.elapsed().as_secs_f64();
        // Every frame with a hand, plus a heartbeat so a still camera does not look frozen.
        if !hands.is_empty() || captured.is_multiple_of(FPS_WINDOW) {
            readout(captured, &hands, pipeline.timings(), elapsed, fps);
        }
        if captured.is_multiple_of(FPS_WINDOW) {
            window = Instant::now();
            frames_in_window = 0;
        }
        if idle {
            std::thread::sleep(IDLE_POLL_INTERVAL);
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    println!("\n------------------------------------------------------------");
    println!("Frames processed:      {captured}");
    println!(
        "Framerate:             {:.1} FPS",
        captured as f64 / elapsed
    );
    println!(
        "Average inference:     {:.1} ms/frame",
        detect_ms / captured.max(1) as f64
    );
    println!("Hands seen in a frame: {detected}");
    println!("------------------------------------------------------------");
    Ok(())
}

/// Phase 3: point the OS cursor with the index fingertip and click on a pinch.
fn run(flags: &Flags) -> Result<()> {
    let environment = model::init_environment()?;
    let mut pipeline = HandPipeline::new(&environment)?
        .with_max_hands(flags.max_hands)?
        .with_hand_threshold(flags.min_confidence);
    let mut pointer = MouseController::new()?
        .with_alpha(flags.alpha)
        .with_margin(flags.margin)
        .with_pinch_distance(flags.pinch)
        .with_mirror(!flags.no_mirror)
        .with_dry_run(flags.dry_run);
    let mut camera = open_camera(flags, "Gesture Daemon - Phase 3: Pointer Control")?;

    let (width, height) = camera.resolution();
    let (display_w, display_h) = pointer.display_size();
    println!(
        "\nPointer ready: frame {width}x{height} on a {display_w}x{display_h} display | {}{}\nCtrl-C stops.\n",
        pointer.tuning(),
        if flags.dry_run {
            " | dry run: the real cursor stays put"
        } else {
            ""
        }
    );

    let started = Instant::now();
    let mut window = Instant::now();
    let mut frames_in_window = 0u32;
    let mut captured = 0u32;
    let mut pointing_frames = 0u32;
    let mut clicks = 0u32;

    // --- Adaptive power-saving state machine ---
    let mut last_hand_seen = Instant::now();
    let mut idle = false;

    loop {
        if flags.frames > 0 && captured >= flags.frames {
            break;
        }
        let frame = camera.capture_frame()?;
        let infer_started = Instant::now();
        let hands = pipeline.detect(frame, width, height)?;
        let elapsed = infer_started.elapsed().as_secs_f64() * 1000.0;
        captured += 1;
        frames_in_window += 1;
        let fps = frames_in_window as f64 / window.elapsed().as_secs_f64();

        // The most confident hand steers. Anything else in view is ignored rather than allowed to
        // fight the cursor for control.
        let pilot = hands.into_iter().max_by(|a, b| a.score.total_cmp(&b.score));
        let report = match pilot {
            None => {
                // Check idle transition: no hand present, see if we should throttle.
                if last_hand_seen.elapsed() > IDLE_TIMEOUT && !idle {
                    idle = true;
                    println!(
                        "[idle] no hand for {:.1}s - throttling to ~10 FPS",
                        IDLE_TIMEOUT.as_secs_f64()
                    );
                }
                pointer.reset();
                PointerReport {
                    frame: captured,
                    hand: None,
                    posture: Posture::default(),
                    pinch: 0.0,
                    pointing: false,
                    position: None,
                    clicked: false,
                    timings: pipeline.timings(),
                    elapsed,
                    fps,
                }
            }
            Some(hand) => {
                // Hand detected: exit idle immediately and reset the timer.
                if idle {
                    idle = false;
                    println!(
                        "[active] hand detected (conf {:.2}) - resuming 30 FPS",
                        hand.score
                    );
                }
                last_hand_seen = Instant::now();
                let pinch = hand.pinch_distance(width, height);
                // A pinch curls the index finger, so it is scored on every tracked frame rather
                // than only while pointing.
                let clicked = pointer.process_click(pinch);
                clicks += u32::from(clicked);
                let posture = hand.posture();
                let pointing = posture.pointing();
                if pointing {
                    let tip = hand.tip();
                    pointer.map_and_move(tip.x as f32, tip.y as f32, width, height)?;
                }
                // Otherwise the cursor stays where it is; the filter keeps its history so the
                // pointer glides back to the fingertip when pointing resumes.
                pointing_frames += u32::from(pointing);
                PointerReport {
                    frame: captured,
                    hand: Some(hand),
                    posture,
                    pinch,
                    pointing,
                    position: pointer.last_position(),
                    clicked,
                    timings: pipeline.timings(),
                    elapsed,
                    fps,
                }
            }
        };
        // Print every frame that moves or clicks, plus a heartbeat so a still camera does not look
        // frozen.
        if report.pointing || report.clicked || captured.is_multiple_of(FPS_WINDOW) {
            pointer_readout(&report);
        }
        if captured.is_multiple_of(FPS_WINDOW) {
            window = Instant::now();
            frames_in_window = 0;
        }
        // Idle mode: sleep between polls to reduce CPU from 30 to ~10 FPS.
        if idle {
            std::thread::sleep(IDLE_POLL_INTERVAL);
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    println!("\n------------------------------------------------------------");
    println!("Frames processed:   {captured}");
    println!("Framerate:          {:.1} FPS", captured as f64 / elapsed);
    println!("Pointing frames:    {pointing_frames}");
    println!(
        "Clicks injected:    {clicks}{}",
        if flags.dry_run { " (dry run)" } else { "" }
    );
    println!("------------------------------------------------------------");
    Ok(())
}

/// One frame's worth of pointer state, gathered for the status line.
struct PointerReport {
    frame: u32,
    hand: Option<Hand>,
    posture: Posture,
    pinch: f32,
    /// Whether the pointing posture engaged this frame, and so whether the cursor was sent anywhere.
    pointing: bool,
    /// Screen coordinate the cursor was last sent to; still reported while pointing is off, because
    /// that is where the cursor is holding.
    position: Option<(i32, i32)>,
    clicked: bool,
    timings: Timings,
    elapsed: f64,
    fps: f64,
}

/// Prints `[Pointing: Active | Smoothed X, Y | Pinch Dist | Click]`.
fn pointer_readout(report: &PointerReport) {
    let mut line = format!("Frame {:>5} | ", report.frame);
    match (report.hand, report.pointing) {
        (Some(hand), true) => {
            let (x, y) = report.position.unwrap_or((0, 0));
            let _ = write!(
                line,
                "Pointing: Active | Smoothed ({x}, {y}) | Pinch {:.3}",
                report.pinch
            );
            if report.clicked {
                line.push_str(" | Click");
            }
            let _ = write!(
                line,
                " | {:?} conf {:.2} | {:.1} ms ({:.1} palm + {:.1} hand) | {:.1} FPS",
                hand.handedness,
                hand.score,
                report.elapsed,
                report.timings.palm_ms,
                report.timings.hand_ms,
                report.fps
            );
        }
        // Tracked but not aimed: the posture explains why the cursor is holding still.
        (Some(hand), false) => {
            let _ = write!(
                line,
                "Pointing: {} | Pinch {:.3} | conf {:.2}",
                report.posture.label(),
                report.pinch,
                hand.score
            );
            if let Some((x, y)) = report.position {
                let _ = write!(line, " | cursor holding at ({x}, {y})");
            }
            let _ = write!(line, " | {:.1} FPS", report.fps);
        }
        (None, _) => {
            let _ = write!(
                line,
                "Pointing: Lost | no hand tracked | {:.1} FPS",
                report.fps
            );
        }
    }
    println!("{}", line.trim_end());
}

fn readout(frame: u32, hands: &[Hand], timings: Timings, elapsed: f64, fps: f64) {
    if hands.is_empty() {
        println!(
            "Frame {:>5} | no hands | palm {:>5.1} ms | {:.1} FPS",
            frame, timings.palm_ms, fps
        );
        return;
    }
    let mut line = format!(
        "Frame {:>5} | {} hand(s) | palm {:>5.1} ms + landmark {:>5.1} ms = {:>5.1} ms | {:.1} FPS | ",
        frame,
        hands.len(),
        timings.palm_ms,
        timings.hand_ms,
        elapsed,
        fps
    );
    for hand in hands {
        let tip = hand.tip();
        let _ = write!(
            line,
            "[tip ({:.0},{:.0}) depth {:.3} box {:.0}x{:.0} conf {:.2} {:?}] ",
            tip.x,
            tip.y,
            hand.tip_depth(),
            hand.bbox.w,
            hand.bbox.h,
            hand.score,
            hand.handedness
        );
    }
    println!("{}", line.trim_end());
}

fn fetch_models(flags: &Flags) -> Result<()> {
    let dir = model::models_dir();
    if flags.force {
        println!("Re-downloading every asset into {}", dir.display());
    }
    let fetched = model::fetch_models(flags.force)?;
    if fetched == 0 {
        println!(
            "Nothing to do - both models are already in {}.",
            dir.display()
        );
    } else {
        println!("\nDownloaded {fetched} model(s) into {}.", dir.display());
        println!("Check the tensor contract next:\n  cargo run --release -- describe");
        println!("Then watch the pipeline run:\n  cargo run --release -- hands");
    }
    Ok(())
}

fn describe() -> Result<()> {
    model::require_models()?;
    let environment = model::init_environment()?;
    model::describe_models(&environment)
}
