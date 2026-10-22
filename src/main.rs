mod camera;

use anyhow::Result;
use camera::CameraCapture;
use std::time::Instant;

fn main() -> Result<()> {
    println!("============================================================");
    println!("  Gesture Daemon - Phase 1: Camera Capture Benchmark");
    println!("============================================================");

    // Open default webcam (index 0) targeting 320x240 @ 30 FPS
    let mut cam = CameraCapture::new(0, 320, 240, 30)?;
    let (target_w, target_h) = cam.resolution();
    let (native_w, native_h) = cam.native_resolution();
    let stream = cam.stream_info();

    println!("Camera Stream Opened:");
    println!(" - Native Camera Resolution: {}x{} {} @ {} FPS", native_w, native_h, stream.fourcc, stream.fps);
    println!(" - Frame Delivery: {}", if stream.pumped { "threaded pump (non-blocking capture loop)" } else { "blocking grab on caller thread" });
    println!(" - Target Processing Buffer: {}x{} RGB ({} bytes/frame)", target_w, target_h, target_w * target_h * 3);

    println!("\nCapturing 150 frames to measure zero-copy buffer & FPS performance...\n");

    let start_time = Instant::now();
    let mut frame_count = 0u32;
    let max_frames = 150;
    let mut last_fps_time = Instant::now();

    while frame_count < max_frames {
        let frame_start = Instant::now();
        let frame_data = cam.capture_frame()?;
        let frame_duration = frame_start.elapsed();

        frame_count += 1;

        if frame_count % 20 == 0 {
            let total_elapsed = start_time.elapsed().as_secs_f64();
            let rolling_elapsed = last_fps_time.elapsed().as_secs_f64();
            let rolling_fps = 20.0 / rolling_elapsed;
            let avg_fps = frame_count as f64 / total_elapsed;

            println!(
                "Frame {:3}/{}: buffer = {} bytes | frame capture = {:>6.2?} | rolling FPS = {:>5.1} | avg FPS = {:>5.1}",
                frame_count,
                max_frames,
                frame_data.len(),
                frame_duration,
                rolling_fps,
                avg_fps
            );
            last_fps_time = Instant::now();
        }
    }

    let total_elapsed = start_time.elapsed().as_secs_f64();
    let avg_fps = frame_count as f64 / total_elapsed;

    println!("\n------------------------------------------------------------");
    println!("Phase 1 Verification Complete!");
    println!(" - Total Frames: {}", frame_count);
    println!(" - Total Time:   {:.2}s", total_elapsed);
    println!(" - Average FPS:  {:.1} FPS", avg_fps);
    println!("------------------------------------------------------------");
    if avg_fps >= 20.0 {
        println!("Framerate target met (>= 20 FPS) - ready for Phase 2.");
    } else {
        println!("Framerate target NOT met (< 20 FPS) - capture path still needs tuning.");
    }

    Ok(())
}
