use anyhow::{Context, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};
use nokhwa::Camera;

pub struct CameraCapture {
    camera: Camera,
    target_width: u32,
    target_height: u32,
    captured_width: u32,
    captured_height: u32,
    raw_buffer: Vec<u8>,
}

impl CameraCapture {
    pub fn new(index: u32, target_width: u32, target_height: u32, fps: u32) -> Result<Self> {
        // Query available cameras to assist debugging if index fails
        if let Ok(devices) = nokhwa::query(ApiBackend::Auto) {
            println!("Found {} video input device(s):", devices.len());
            for dev in devices {
                println!(" - [{}] {}", dev.index(), dev.human_name());
            }
        }

        let cam_index = CameraIndex::Index(index);
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
            CameraFormat::new(
                Resolution::new(target_width, target_height),
                FrameFormat::RAWRGB,
                fps,
            ),
        ));

        let mut camera = Camera::new(cam_index, requested)
            .context("Failed to initialize camera with requested format")?;

        camera.open_stream().context("Failed to open camera stream")?;

        let res = camera.resolution();
        let buffer_size = (target_width * target_height * 3) as usize;

        Ok(Self {
            camera,
            target_width,
            target_height,
            captured_width: res.width(),
            captured_height: res.height(),
            raw_buffer: vec![0u8; buffer_size],
        })
    }

    pub fn resolution(&self) -> (u32, u32) {
        (self.target_width, self.target_height)
    }

    pub fn native_resolution(&self) -> (u32, u32) {
        (self.captured_width, self.captured_height)
    }

    /// Captures a frame and ensures it is delivered in 320x240 RGB format
    /// reusing the internal memory buffer to avoid allocations in the hot loop.
    pub fn capture_frame(&mut self) -> Result<&[u8]> {
        let frame = self.camera.frame().context("Failed to grab camera frame")?;
        let decoded = frame
            .decode_image::<RgbFormat>()
            .context("Failed to decode frame into RGB format")?;

        if decoded.width() == self.target_width && decoded.height() == self.target_height {
            let raw = decoded.as_raw();
            self.raw_buffer.copy_from_slice(raw);
        } else {
            // Downsample to target 320x240 if camera driver returned a higher native resolution
            let resized = image::imageops::resize(
                &decoded,
                self.target_width,
                self.target_height,
                image::imageops::FilterType::Nearest, // Fastest downsample for low CPU usage
            );
            self.raw_buffer.copy_from_slice(resized.as_raw());
        }

        Ok(&self.raw_buffer)
    }
}
