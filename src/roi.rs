//! Affine geometry and resampling helpers for the landmark pipeline.
//!
//! The reference MediaPipe implementation crops, zero-pads, rotates and resizes between stages.
//! Doing that literally allocates and resamples a large intermediate image per hand, so here the
//! stage transforms are composed into a single mapping and the source frame is sampled once.
//! Sampling outside the source bounds yields black, which is exactly the zero padding the
//! reference performs.

/// A 2x3 affine transform: `out = M * [x, y, 1]`.
#[derive(Copy, Clone, Debug, Default)]
pub struct Affine {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Affine {
    pub const IDENTITY: Affine = Affine {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        tx: 0.0,
        ty: 0.0,
    };

    pub fn translation(dx: f64, dy: f64) -> Self {
        Self { tx: dx, ty: dy, ..Self::IDENTITY }
    }

    /// Maps coordinates of a region whose top-left corner is `(x, y)` back to the parent space.
    pub fn offset_of(crop: &Rect) -> Self {
        Self::translation(crop.x, crop.y)
    }

    /// Scales about an arbitrary pivot: `pivot + (p - pivot) * factor`.
    pub fn scale_about(pivot: (f64, f64), factor: f64) -> Self {
        Self {
            a: factor,
            d: factor,
            tx: pivot.0 * (1.0 - factor),
            ty: pivot.1 * (1.0 - factor),
            ..Self::IDENTITY
        }
    }

    /// Mirrors `cv2.getRotationMatrix2D(center, angle_degrees, 1.0)`.
    ///
    /// OpenCV treats `M` as source -> destination, so sampling the rotated image at a destination
    /// point requires the inverse of this transform.
    pub fn rotation(center: (f64, f64), angle_degrees: f64) -> Self {
        let radians = angle_degrees.to_radians();
        let (alpha, beta) = (radians.cos(), radians.sin());
        Self {
            a: alpha,
            b: beta,
            c: -beta,
            d: alpha,
            tx: (1.0 - alpha) * center.0 - beta * center.1,
            ty: beta * center.0 + (1.0 - alpha) * center.1,
        }
    }

    /// Applies `self` first, then `next` (i.e. `next(self(p))`).
    pub fn then(&self, next: &Affine) -> Affine {
        Affine {
            a: next.a * self.a + next.b * self.c,
            b: next.a * self.b + next.b * self.d,
            c: next.c * self.a + next.d * self.c,
            d: next.c * self.b + next.d * self.d,
            tx: next.a * self.tx + next.b * self.ty + next.tx,
            ty: next.c * self.tx + next.d * self.ty + next.ty,
        }
    }

    pub fn apply(&self, point: (f64, f64)) -> (f64, f64) {
        (
            self.a * point.0 + self.b * point.1 + self.tx,
            self.c * point.0 + self.d * point.1 + self.ty,
        )
    }

    pub fn apply_slice(&self, points: &mut [(f64, f64)]) {
        for point in points.iter_mut() {
            *point = self.apply(*point);
        }
    }

    /// Inverse of a rigid (rotation + translation, unit scale) transform.
    ///
    /// Same construction MediaPipe's reference uses for de-rotating landmarks, and valid because
    /// `a^2 + c^2 == 1` for a pure rotation.
    pub fn invert_rigid(&self) -> Affine {
        let inverted_tx = -(self.a * self.tx + self.c * self.ty);
        let inverted_ty = -(self.b * self.tx + self.d * self.ty);
        Affine {
            a: self.a,
            b: self.c,
            c: self.b,
            d: self.d,
            tx: inverted_tx,
            ty: inverted_ty,
        }
    }

    /// Rotates a vector by only the linear part of `self`, matching `coords @ M[:, :2]` in the
    /// reference implementation.
    pub fn rotate_vector(&self, point: (f64, f64)) -> (f64, f64) {
        (
            self.a * point.0 + self.c * point.1,
            self.b * point.0 + self.d * point.1,
        )
    }
}

/// Axis-aligned rectangle in `corner` form: top-left plus width/height.
#[derive(Copy, Clone, Debug, Default)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, w, h }
    }

    pub fn from_corners(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self::new(x0, y0, x1 - x0, y1 - y0)
    }

    pub fn center(&self) -> (f64, f64) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    /// Moves the rectangle by a vector proportional to its own size, as MediaPipe's
    /// `*_BOX_SHIFT_VECTOR` constants do.
    pub fn shifted(&self, shift: (f64, f64)) -> Rect {
        Rect::new(
            self.x + shift.0 * self.w,
            self.y + shift.1 * self.h,
            self.w,
            self.h,
        )
    }

    /// Grows the rectangle about its centre by `factor` on each axis.
    pub fn enlarged(&self, factor_x: f64, factor_y: f64) -> Rect {
        let new_w = self.w * factor_x;
        let new_h = self.h * factor_y;
        Rect::from_corners(
            self.center().0 - new_w / 2.0,
            self.center().1 - new_h / 2.0,
            self.center().0 + new_w / 2.0,
            self.center().1 + new_h / 2.0,
        )
    }

    /// Grows the rectangle about its centre uniformly, matching MediaPipe's
    /// `*_BOX_ENLARGE_FACTOR` handling.
    pub fn enlarged_uniform(&self, factor: f64) -> Rect {
        self.enlarged(factor, factor)
    }

    pub fn clamped_to_image(&self, image_w: usize, image_h: usize) -> Rect {
        let x0 = self.x.clamp(0.0, image_w as f64);
        let y0 = self.y.clamp(0.0, image_h as f64);
        let x1 = (self.x + self.w).clamp(0.0, image_w as f64);
        let y1 = (self.y + self.h).clamp(0.0, image_h as f64);
        Rect::from_corners(x0, y0, x1, y1)
    }

    /// The square region that encloses this rectangle, centred on it - the crop-plus-pad step.
    /// `side` overrides the square size (MediaPipe uses the diagonal before rotation) and returns
    /// the transform that maps square coordinates into image coordinates.
    pub fn square_pad(&self, image_w: usize, image_h: usize, side_override: Option<f64>) -> (Rect, Affine) {
        let clamped = self.clamped_to_image(image_w, image_h);
        let side = side_override.unwrap_or_else(|| clamped.w.max(clamped.h)).ceil().max(1.0);
        let center = clamped.center();
        let square = Rect::new(center.0 - side / 2.0, center.1 - side / 2.0, side, side);
        (square, Affine::offset_of(&square))
    }

    /// Transform mapping a destination box of `out_size` into this rectangle's image space.
    pub fn map_from_unit_box(&self, out_w: usize, out_h: usize) -> Affine {
        Affine {
            a: self.w / out_w as f64,
            d: self.h / out_h as f64,
            tx: self.x,
            ty: self.y,
            ..Affine::IDENTITY
        }
    }

    pub fn contains(&self, point: (f64, f64)) -> bool {
        point.0 >= 0.0
            && point.1 >= 0.0
            && point.0 < self.w
            && point.1 < self.h
    }
}

/// Rectangle in corner form, as returned by the palm decoder.
#[derive(Copy, Clone, Debug, Default)]
pub struct Corners {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Corners {
    pub fn to_rect(self) -> Rect {
        Rect::from_corners(self.x0, self.y0, self.x1, self.y1)
    }

    pub fn area(self) -> f64 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }
}

pub fn iou(a: Corners, b: Corners) -> f64 {
    let x0 = a.x0.max(b.x0);
    let y0 = a.y0.max(b.y0);
    let x1 = a.x1.min(b.x1);
    let y1 = a.y1.min(b.y1);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let intersection = (x1 - x0) * (y1 - y0);
    let union = a.area() + b.area() - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Axis-aligned bounds of a point set.
pub fn bounding_box(points: &[(f64, f64)]) -> Option<Rect> {
    let first = *points.first()?;
    let mut box_ = Corners {
        x0: first.0,
        y0: first.1,
        x1: first.0,
        y1: first.1,
    };
    for &(x, y) in points {
        box_.x0 = box_.x0.min(x);
        box_.y0 = box_.y0.min(y);
        box_.x1 = box_.x1.max(x);
        box_.y1 = box_.y1.max(y);
    }
    Some(box_.to_rect())
}

/// Channel order of a model input blob.
///
/// MediaPipe graphs are exported from TFLite, which is `HWC`, but the ONNX conversion may keep
/// the `CHW` convention instead, so the layout is read from the session rather than assumed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Height x width x channels (`HWC`).
    Interleaved,
    /// Channels x height x width (`CHW`).
    Planar,
}

/// Packs RGB (0..=255) into the `float32` 0..=1 blob the MediaPipe graphs expect.
///
/// `mapping` takes a destination pixel `(x, y)` and returns where it came from in the source
/// frame; a `None` result becomes black.
///
/// Bilinear for mapping scale < 1 (rotations, and small upscales of a 640x480 source), box
/// averaging for downscales: nearest on its own would alias badly on the palm stage, where the
/// 320x240 letterbox into 192x192 is a downscale.
pub fn write_blob(
    source: &[u8],
    source_w: usize,
    source_h: usize,
    mapping: &Affine,
    out_w: usize,
    out_h: usize,
    layout: Layout,
    dest: &mut [f32],
) {
    debug_assert_eq!(dest.len(), out_w * out_h * 3);
    // `mapping` maps destination -> source, so `a > 1` means the destination is a downscale.
    let downscale = mapping.a > 1.05 || mapping.d > 1.05;
    let axis_aligned = mapping.b.abs() < 1e-6 && mapping.c.abs() < 1e-6;
    let use_box = downscale && axis_aligned;
    let plane = out_w * out_h;

    for out_y in 0..out_h {
        for out_x in 0..out_w {
            let mut rgb = [0.0f32; 3];
            if use_box {
                sample_box(
                    source,
                    source_w,
                    source_h,
                    mapping,
                    (out_x, out_y),
                    &mut rgb,
                );
            } else {
                let (sx, sy) = mapping.apply((out_x as f64 + 0.5, out_y as f64 + 0.5));
                sample_bilinear(source, source_w, source_h, sx - 0.5, sy - 0.5, &mut rgb);
            }
            match layout {
                Layout::Interleaved => {
                    let offset = (out_y * out_w + out_x) * 3;
                    dest[offset..offset + 3].copy_from_slice(&rgb);
                }
                Layout::Planar => {
                    let offset = out_y * out_w + out_x;
                    for (channel, value) in rgb.iter().enumerate() {
                        dest[channel * plane + offset] = *value;
                    }
                }
            }
        }
    }
}

fn sample_box(
    source: &[u8],
    source_w: usize,
    source_h: usize,
    mapping: &Affine,
    out: (usize, usize),
    dest: &mut [f32],
) {
    let dst_x = out.0 as f64;
    let dst_y = out.1 as f64;
    let src_x0 = mapping.a * dst_x + mapping.tx;
    let src_y0 = mapping.d * dst_y + mapping.ty;
    let src_x1 = mapping.a * (dst_x + 1.0) + mapping.tx;
    let src_y1 = mapping.d * (dst_y + 1.0) + mapping.ty;

    let x0 = src_x0.max(0.0).floor() as usize;
    let y0 = src_y0.max(0.0).floor() as usize;
    let x1 = src_x1.max(0.0).ceil() as usize;
    let y1 = src_y1.max(0.0).ceil() as usize;
    let x1 = x1.min(source_w);
    let y1 = y1.min(source_h);
    if x1 <= x0 || y1 <= y0 {
        dest.copy_from_slice(&[0.0, 0.0, 0.0]);
        return;
    }
    // Area weighting over the exact footprint, so partially covered edge pixels are not overcounted.
    let mut sums = [0.0f64; 3];
    let mut weights = 0.0f64;
    for sy in y0..y1 {
        let cover_y = src_y1.min(sy as f64 + 1.0) - src_y0.max(sy as f64);
        if cover_y <= 0.0 {
            continue;
        }
        for sx in x0..x1 {
            let cover_x = src_x1.min(sx as f64 + 1.0) - src_x0.max(sx as f64);
            if cover_x <= 0.0 {
                continue;
            }
            let weight = cover_x * cover_y;
            let index = (sy * source_w + sx) * 3;
            for channel in 0..3 {
                sums[channel] += f64::from(source[index + channel]) * weight;
            }
            weights += weight;
        }
    }
    if weights <= 0.0 {
        dest.copy_from_slice(&[0.0, 0.0, 0.0]);
        return;
    }
    let divisor = weights * 255.0;
    for channel in 0..3 {
        dest[channel] = (sums[channel] / divisor) as f32;
    }
}

fn sample_bilinear(
    source: &[u8],
    source_w: usize,
    source_h: usize,
    fx: f64,
    fy: f64,
    dest: &mut [f32],
) {
    if fx < -1.0 || fy < -1.0 || fx > source_w as f64 || fy > source_h as f64 {
        dest.copy_from_slice(&[0.0, 0.0, 0.0]);
        return;
    }
    let x0 = fx.floor() as i64;
    let y0 = fy.floor() as i64;
    let wx = fx - x0 as f64;
    let wy = fy - y0 as f64;

    let pixel = |x: i64, y: i64| -> Option<[u8; 3]> {
        if x < 0 || y < 0 || x >= source_w as i64 || y >= source_h as i64 {
            return None;
        }
        let index = (y as usize * source_w + x as usize) * 3;
        Some([source[index], source[index + 1], source[index + 2]])
    };

    let p00 = pixel(x0, y0);
    let p10 = pixel(x0 + 1, y0);
    let p01 = pixel(x0, y0 + 1);
    let p11 = pixel(x0 + 1, y0 + 1);

    // Missing neighbours (edge of frame) contribute black, matching the reference's zero pad.
    let mix = |channel: usize| -> f64 {
        let v = |p: Option<[u8; 3]>| p.map_or(0.0, |px| f64::from(px[channel]));
        let top = v(p00) * (1.0 - wx) + v(p10) * wx;
        let bottom = v(p01) * (1.0 - wx) + v(p11) * wx;
        top * (1.0 - wy) + bottom * wy
    };
    for channel in 0..3 {
        dest[channel] = (mix(channel) / 255.0) as f32;
    }
}
