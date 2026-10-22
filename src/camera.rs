use anyhow::{anyhow, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
};
use nokhwa::Camera;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Preferred ceiling for the native capture resolution. Larger frames are downsampled
/// anyway, so asking the driver for fewer pixels shrinks the USB payload and the decode cost.
const NATIVE_MAX_WIDTH: u32 = 640;
const NATIVE_MAX_HEIGHT: u32 = 480;

/// Minimum frame rate we are willing to accept from the driver before compromising elsewhere.
const MIN_ACCEPTABLE_FPS: u32 = 10;

/// How long the consumer waits for a freshly pumped frame before declaring the stream dead.
const FRAME_TIMEOUT: Duration = Duration::from_millis(2000);

/// MSMF reports variable-rate MJPEG streams back as 0 or 1 FPS. Anything implausible is
/// replaced by the rate the driver advertised in its own format list.
fn plausible_fps(reported: u32, advertised: u32) -> u32 {
    if reported < MIN_ACCEPTABLE_FPS && advertised >= MIN_ACCEPTABLE_FPS {
        advertised
    } else {
        reported
    }
}

/// The stream parameters the camera actually agreed to.
#[derive(Clone, Debug)]
pub struct StreamInfo {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub fourcc: String,
    pub pumped: bool,
}

/// Frame mailbox written by the capture thread.
struct FrameSlot {
    buffer: Vec<u8>,
    sequence: u64,
    stream: Option<StreamInfo>,
    error: Option<String>,
}

/// Shared plumbing for the threaded buffer pump.
struct Pump {
    slot: Mutex<FrameSlot>,
    frame_ready: Condvar,
}

impl Pump {
    fn lock_slot(&self) -> std::sync::MutexGuard<'_, FrameSlot> {
        // A panic while holding this lock would only corrupt frame data, so keep serving
        // frames instead of poisoning the whole capture pipeline.
        self.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

enum CaptureSource {
    /// Grab, decode and downsample on the calling thread.
    Direct(Camera),
    /// A dedicated thread owns the device and publishes frames into `pump`.
    Pumped {
        pump: Arc<Pump>,
        stop: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    },
}

pub struct CameraCapture {
    source: CaptureSource,
    target_width: usize,
    target_height: usize,
    stream: StreamInfo,
    raw_buffer: Vec<u8>,
    seen_sequence: u64,
}

impl CameraCapture {
    pub fn new(index: u32, target_width: u32, target_height: u32, fps: u32) -> Result<Self> {
        // Query available cameras to assist debugging
        if let Ok(devices) = nokhwa::query(ApiBackend::Auto) {
            println!("Found {} video input device(s):", devices.len());
            for dev in devices {
                println!(" - [{}] {}", dev.index(), dev.human_name());
            }
        }

        let target_width = target_width as usize;
        let target_height = target_height as usize;

        // Ask the driver what it can actually do and pick the cheapest stream that still
        // hits the frame-rate target (ideally 640x480 MJPEG @ 30, not the 1280x720 default).
        let chosen = negotiate_capture_format(index, fps)?;
        if let Some(fmt) = chosen {
            println!(
                "Negotiated native capture format: {}x{} {} @ {} FPS",
                fmt.resolution().width(),
                fmt.resolution().height(),
                fmt.format(),
                fmt.frame_rate()
            );
        } else {
            println!("Camera driver does not publish a format list, using driver default.");
        }

        // The threaded pump keeps device I/O and JPEG decode off the inference loop, which is
        // what Phase 2 needs. It is only adopted once its first frame has landed.
        match Self::new_pumped(index, chosen, target_width, target_height) {
            Ok(capture) => Ok(capture),
            Err(why) => {
                eprintln!("Warning: threaded frame pump unavailable ({why}), falling back to blocking capture...");
                Self::new_direct(index, chosen, target_width, target_height)
            }
        }
    }

    fn new_pumped(
        index: u32,
        chosen: Option<CameraFormat>,
        target_width: usize,
        target_height: usize,
    ) -> Result<Self> {
        let buffer_size = target_width * target_height * 3;
        let pump = Arc::new(Pump {
            slot: Mutex::new(FrameSlot {
                buffer: vec![0u8; buffer_size],
                sequence: 0,
                stream: None,
                error: None,
            }),
            frame_ready: Condvar::new(),
        });
        let stop = Arc::new(AtomicBool::new(false));

        // Every nokhwa / Media Foundation call for this mode happens on the same thread,
        // which keeps the COM apartment consistent for the lifetime of the stream.
        let thread_stop = stop.clone();
        let thread_pump = pump.clone();
        let handle = std::thread::spawn(move || {
            run_frame_pump(index, chosen, target_width, target_height, thread_pump, thread_stop);
        });

        // Block until the pump proves it can open the device; if it cannot, tear it down and
        // let the caller fall back to blocking capture on the main thread.
        let stream = match wait_for_stream(&pump) {
            Ok(stream) => stream,
            Err(why) => {
                stop.store(true, Ordering::SeqCst);
                let _ = handle.join();
                return Err(why);
            }
        };

        Ok(Self {
            source: CaptureSource::Pumped {
                pump,
                stop,
                handle: Some(handle),
            },
            target_width,
            target_height,
            stream,
            raw_buffer: vec![0u8; buffer_size],
            seen_sequence: 0,
        })
    }

    fn new_direct(
        index: u32,
        chosen: Option<CameraFormat>,
        target_width: usize,
        target_height: usize,
    ) -> Result<Self> {
        let mut camera = open_camera(index, chosen)?;
        camera
            .open_stream()
            .map_err(|why| anyhow!("Failed to open camera stream: {why}"))?;

        let fmt = camera.camera_format();
        let stream = StreamInfo {
            width: fmt.resolution().width(),
            height: fmt.resolution().height(),
            fps: plausible_fps(fmt.frame_rate(), chosen.map_or(0, |c| c.frame_rate())),
            fourcc: fmt.format().to_string(),
            pumped: false,
        };

        Ok(Self {
            source: CaptureSource::Direct(camera),
            target_width,
            target_height,
            stream,
            raw_buffer: vec![0u8; target_width * target_height * 3],
            seen_sequence: 0,
        })
    }

    pub fn resolution(&self) -> (u32, u32) {
        (self.target_width as u32, self.target_height as u32)
    }

    pub fn native_resolution(&self) -> (u32, u32) {
        (self.stream.width, self.stream.height)
    }

    pub fn stream_info(&self) -> &StreamInfo {
        &self.stream
    }

    /// Captures a frame and delivers it in exact 320x240 RGB format.
    /// Uses zero-heap-allocation nearest-neighbor downsampling directly into the pre-allocated buffer.
    pub fn capture_frame(&mut self) -> Result<&[u8]> {
        match &mut self.source {
            CaptureSource::Direct(camera) => {
                let frame = camera
                    .frame()
                    .map_err(|why| anyhow!("Failed to grab camera frame: {why}"))?;
                let decoded = frame
                    .decode_image::<RgbFormat>()
                    .map_err(|why| anyhow!("Failed to decode frame into RGB format: {why}"))?;

                let src_raw = decoded.as_raw();
                let src_w = decoded.width() as usize;
                let src_h = decoded.height() as usize;

                if src_w == self.target_width && src_h == self.target_height {
                    self.raw_buffer.copy_from_slice(src_raw);
                } else {
                    resize_nearest_rgb(
                        src_raw,
                        src_w,
                        src_h,
                        &mut self.raw_buffer,
                        self.target_width,
                        self.target_height,
                    );
                }
            }
            CaptureSource::Pumped { pump, .. } => {
                await_pumped_frame(pump, &mut self.seen_sequence, &mut self.raw_buffer)?;
            }
        }

        Ok(&self.raw_buffer)
    }
}

/// Waits for the capture thread to publish a frame newer than the one already consumed,
/// then copies it into the caller's buffer (a 320x240 RGB memcpy, ~0.02 ms).
fn await_pumped_frame(pump: &Pump, seen_sequence: &mut u64, out: &mut [u8]) -> Result<()> {
    let deadline = Instant::now() + FRAME_TIMEOUT;
    let mut slot = pump.lock_slot();

    while slot.sequence == *seen_sequence {
        if let Some(error) = &slot.error {
            return Err(anyhow!("Capture thread lost the stream: {error}"));
        }
        let Some(rest) = deadline.checked_duration_since(Instant::now()) else {
            return Err(anyhow!("Timed out waiting for a frame from the capture thread"));
        };
        let (next_slot, wait_result) = pump
            .frame_ready
            .wait_timeout(slot, rest)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slot = next_slot;
        if wait_result.timed_out() {
            return Err(anyhow!("Timed out waiting for a frame from the capture thread"));
        }
    }

    *seen_sequence = slot.sequence;
    out.copy_from_slice(&slot.buffer);
    Ok(())
}

impl Drop for CameraCapture {
    fn drop(&mut self) {
        if let CaptureSource::Pumped { stop, handle, .. } = &mut self.source {
            stop.store(true, Ordering::SeqCst);
            // The pump blocks inside `Camera::frame()` for at most one frame interval,
            // so joining here is quick and guarantees the device is released cleanly.
            if let Some(handle) = handle.take() {
                let _ = handle.join();
            }
        }
    }
}

fn run_frame_pump(
    index: u32,
    chosen: Option<CameraFormat>,
    target_width: usize,
    target_height: usize,
    pump: Arc<Pump>,
    stop: Arc<AtomicBool>,
) {
    let publish_error = |pump: &Pump, message: String| {
        let mut slot = pump.lock_slot();
        slot.error = Some(message);
        drop(slot);
        pump.frame_ready.notify_all();
    };

    let mut camera = match open_camera(index, chosen) {
        Ok(camera) => camera,
        Err(why) => {
            publish_error(&pump, why.to_string());
            return;
        }
    };

    if let Err(why) = camera.open_stream() {
        publish_error(&pump, format!("Failed to open camera stream: {why}"));
        return;
    }

    let fmt = camera.camera_format();
    {
        let mut slot = pump.lock_slot();
        slot.stream = Some(StreamInfo {
            width: fmt.resolution().width(),
            height: fmt.resolution().height(),
            fps: plausible_fps(fmt.frame_rate(), chosen.map_or(0, |c| c.frame_rate())),
            fourcc: fmt.format().to_string(),
            pumped: true,
        });
        pump.frame_ready.notify_all();
    }

    let buffer_size = target_width * target_height * 3;
    // Allocated once per thread: the decode + downsample path stays allocation free afterwards.
    let mut local_frame = vec![0u8; buffer_size];

    while !stop.load(Ordering::SeqCst) {
        let frame = match camera.frame() {
            Ok(frame) => frame,
            Err(why) => {
                // Transient drops are normal on MSMF; keep polling unless shutdown is requested.
                eprintln!("Warning: frame grab failed ({why}), retrying...");
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
        };

        let decoded = match frame.decode_image::<RgbFormat>() {
            Ok(decoded) => decoded,
            Err(why) => {
                eprintln!("Warning: frame decode failed ({why}), skipping...");
                continue;
            }
        };

        let src_w = decoded.width() as usize;
        let src_h = decoded.height() as usize;
        if src_w == target_width && src_h == target_height {
            local_frame.copy_from_slice(decoded.as_raw());
        } else {
            resize_nearest_rgb(
                decoded.as_raw(),
                src_w,
                src_h,
                &mut local_frame,
                target_width,
                target_height,
            );
        }

        {
            let mut slot = pump.lock_slot();
            slot.buffer.copy_from_slice(&local_frame);
            slot.sequence = slot.sequence.wrapping_add(1);
            slot.error = None;
            pump.frame_ready.notify_all();
        }
    }
}

/// Resolves the pump's handshake: the stream metadata it publishes right after the device
/// open is what tells the constructor the pump is genuinely usable.
fn wait_for_stream(pump: &Arc<Pump>) -> Result<StreamInfo> {
    let deadline = Instant::now() + FRAME_TIMEOUT;
    let mut slot = pump.lock_slot();
    loop {
        if let Some(stream) = &slot.stream {
            return Ok(stream.clone());
        }
        if let Some(error) = &slot.error {
            return Err(anyhow!("{error}"));
        }
        let Some(rest) = deadline.checked_duration_since(Instant::now()) else {
            return Err(anyhow!("Capture thread did not deliver a frame in time"));
        };
        let (next_slot, wait_result) = pump
            .frame_ready
            .wait_timeout(slot, rest)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slot = next_slot;
        if wait_result.timed_out() {
            return Err(anyhow!("Capture thread did not deliver a frame in time"));
        }
    }
}

fn open_camera(index: u32, chosen: Option<CameraFormat>) -> Result<Camera> {
    let requested = match chosen {
        // Drive the negotiation from the exact format we scored off the driver's own list,
        // so nokhwa only ever hands back something the camera really advertises.
        Some(fmt) => RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt)),
        None => RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
    };

    let camera = Camera::new(CameraIndex::Index(index), requested)
        .map_err(|why| anyhow!("Failed to initialize camera: {why}"))?;

    // MSMF reports the negotiated frame rate back inconsistently (variable-rate MJPEG streams
    // come back as 1 or 0 FPS), so only geometry and pixel format are worth enforcing here -
    // the benchmark loop is the real judge of delivered frame rate.
    if let Some(fmt) = chosen {
        let applied = camera.camera_format();
        if applied.resolution() != fmt.resolution() || applied.format() != fmt.format() {
            return Err(anyhow!(
                "Camera driver refused the negotiated stream (asked {} {} {}, got {} {} {})",
                fmt.resolution().width(),
                fmt.resolution().height(),
                fmt.format(),
                applied.resolution().width(),
                applied.resolution().height(),
                applied.format(),
            ));
        }
        if applied.frame_rate() != fmt.frame_rate() {
            println!(
                "Note: driver reports {} FPS for the {} {} stream (advertised {} FPS)",
                applied.frame_rate(),
                applied.resolution().width(),
                applied.resolution().height(),
                fmt.frame_rate()
            );
        }
    }

    Ok(camera)
}

/// Asks the driver for its full format list and scores every entry, preferring a stream that
/// reaches the frame-rate target with the smallest USB payload (MJPEG 640x480 @ 30 ideal).
/// Returns `None` when the backend cannot enumerate formats, which keeps the old behaviour.
fn negotiate_capture_format(index: u32, target_fps: u32) -> Result<Option<CameraFormat>> {
    let mut probe = Camera::new(
        CameraIndex::Index(index),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
    )
    .map_err(|why| anyhow!("Failed to initialize probe camera: {why}"))?;

    let formats = match probe.compatible_camera_formats() {
        Ok(formats) => formats,
        Err(why) => {
            eprintln!("Warning: camera cannot enumerate formats ({why}), using driver default...");
            return Ok(None);
        }
    };

    if !formats.is_empty() {
        println!("Camera advertises {} format(s), sampling the low-resolution tiers:", formats.len());
        for fmt in formats
            .iter()
            .filter(|fmt| fmt.resolution().width() <= NATIVE_MAX_WIDTH)
            .take(8)
        {
            println!(
                "   - {}x{} {} @ {} FPS",
                fmt.resolution().width(),
                fmt.resolution().height(),
                fmt.format(),
                fmt.frame_rate()
            );
        }
    }

    Ok(select_capture_format(&formats, target_fps))
}

fn select_capture_format(formats: &[CameraFormat], target_fps: u32) -> Option<CameraFormat> {
    formats
        .iter()
        // Only formats nokhwa can hand straight back as RGB888 (GRAY is dropped on purpose).
        .filter(|fmt| matches!(
            fmt.format(),
            FrameFormat::MJPEG
                | FrameFormat::YUYV
                | FrameFormat::NV12
                | FrameFormat::RAWRGB
                | FrameFormat::RAWBGR
        ))
        .filter(|fmt| fmt.frame_rate() >= MIN_ACCEPTABLE_FPS)
        .max_by_key(|fmt| format_score(fmt, target_fps))
        .copied()
}

/// Higher is better. Tiers are ordered by what actually limits the loop:
/// frame rate first, then pixel budget, then wire format, then sharpness.
fn format_score(fmt: &CameraFormat, target_fps: u32) -> (u32, u32, u32, i64, i64) {
    let resolution = fmt.resolution();
    let width = resolution.width();
    let height = resolution.height();
    let frame_rate = i64::from(fmt.frame_rate());
    let target = i64::from(target_fps);
    let floor = i64::from(MIN_ACCEPTABLE_FPS);
    let half_target = i64::from(target_fps / 2);

    let fps_band = if frame_rate >= target {
        3
    } else if frame_rate >= half_target.max(floor) {
        2
    } else {
        1
    };

    let within_budget = u32::from(width <= NATIVE_MAX_WIDTH && height <= NATIVE_MAX_HEIGHT);

    // Compressed transport buys the most frames per USB megabit, raw RGB buys the least.
    let wire_rank: u32 = match fmt.format() {
        FrameFormat::MJPEG => 3,
        FrameFormat::NV12 | FrameFormat::YUYV => 2,
        FrameFormat::GRAY => 1,
        FrameFormat::RAWRGB | FrameFormat::RAWBGR => 0,
    };

    let area = i64::from(width) * i64::from(height);
    // Inside the budget keep the sharpest image; outside it keep the cheapest one.
    let area_rank = if within_budget == 1 { area } else { -area };

    // Prefer not to burn bandwidth on 60 FPS when the target was already reached.
    let fps_overshoot = -(frame_rate - target).abs();

    (fps_band, within_budget, wire_rank, area_rank, fps_overshoot)
}

/// Zero-allocation nearest-neighbor downsampling for packed RGB buffers
#[inline]
fn resize_nearest_rgb(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) {
    for y in 0..dst_h {
        let src_y = (y * src_h) / dst_h;
        let src_row_start = src_y * src_w * 3;
        let dst_row_start = y * dst_w * 3;

        for x in 0..dst_w {
            let src_x = (x * src_w) / dst_w;
            let src_idx = src_row_start + src_x * 3;
            let dst_idx = dst_row_start + x * 3;

            dst[dst_idx..dst_idx + 3].copy_from_slice(&src[src_idx..src_idx + 3]);
        }
    }
}
