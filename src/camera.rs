use anyhow::{Context, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;

pub struct CameraCapture {
    camera: Camera,
    target_width: u32,
    target_height: u32,
    native_width: u32,
    native_height: u32,
    raw_buffer: Vec<u8>,
}

impl CameraCapture {
    pub fn new(index: u32, target_width: u32, target_height: u32, _fps: u32) -> Result<Self> {
        // Query available cameras to assist debugging
        if let Ok(devices) = nokhwa::query(ApiBackend::Auto) {
            println!("Found {} video input device(s):", devices.len());
            for dev in devices {
                println!(" - [{}] {}", dev.index(), dev.human_name());
            }
        }

        let cam_index = CameraIndex::Index(index);

        // MediaFoundation webcams reject strict format matching like FrameFormat::RAWRGB.
        // We request None so the driver selects its native supported format (e.g. MJPEG/YUY2).
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);

        let mut camera = match Camera::new(cam_index.clone(), requested) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: RequestedFormatType::None failed ({:?}), trying AbsoluteHighestFrameRate fallback...", e);
                let fallback = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
                Camera::new(cam_index, fallback)
                    .context("Failed to initialize camera with default or fallback format")?
            }
        };

        camera.open_stream().context("Failed to open camera stream")?;

        let res = camera.resolution();
        let buffer_size = (target_width * target_height * 3) as usize;

        Ok(Self {
            camera,
            target_width,
            target_height,
            native_width: res.width(),
            native_height: res.height(),
            raw_buffer: vec![0u8; buffer_size],
        })
    }

    pub fn resolution(&self) -> (u32, u32) {
        (self.target_width, self.target_height)
    }

    pub fn native_resolution(&self) -> (u32, u32) {
        (self.native_width, self.native_height)
    }

    /// Captures a frame and delivers it in exact 320x240 RGB format.
    /// Uses zero-heap-allocation nearest-neighbor downsampling directly into the pre-allocated buffer.
    pub fn capture_frame(&mut self) -> Result<&[u8]> {
        let frame = self.camera.frame().context("Failed to grab camera frame")?;
        let decoded = frame
            .decode_image::<RgbFormat>()
            .context("Failed to decode frame into RGB format")?;

        let src_raw = decoded.as_raw();
        let src_w = decoded.width();
        let src_h = decoded.height();

        if src_w == self.target_width && src_h == self.target_height {
            self.raw_buffer.copy_from_slice(src_raw);
        } else {
            // High-speed, zero-allocation downsample to 320x240 directly into pre-allocated memory
            resize_nearest_rgb(
                src_raw,
                src_w as usize,
                src_h as usize,
                &mut self.raw_buffer,
                self.target_width as usize,
                self.target_height as usize,
            );
        }

        Ok(&self.raw_buffer)
    }
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
