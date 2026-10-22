//! Phase 2: hand landmark extraction with the two-stage MediaPipe pipeline.
//!
//! Stage 1 (palm detection) finds palm boxes plus the 7 points used to orient a hand; stage 2
//! (hand landmarks) consumes an upright, cropped 224x224 view of one hand and returns 21
//! keypoints. Both stages are plain ONNX graphs, so everything between them - letterboxing,
//! anchor decoding, NMS, ROI shift/enlarge, rotation and the de-rotation back into frame
//! pixels - is ours to implement.
//!
//! The arithmetic mirrors the Apache-2.0 reference implementations shipped with OpenCV Zoo
//! (`mp_palmdet.py` and `mp_handpose.py`) so results line up with MediaPipe's own output. Where
//! the reference resamples an intermediate image several times (crop, pad, warp, resize), the
//! stage transforms are composed into a single destination -> source mapping and the frame is
//! sampled once; see [`crate::roi`].
//!
//! Frame coordinates below are always pixels of the RGB buffer handed to [`HandPipeline::detect`].

use crate::model;
use crate::palm_anchors::PALM_ANCHORS;
use crate::roi::{self, Affine, Corners, Layout, Rect};
use anyhow::{Context, Result, anyhow, bail};
use ort::environment::Environment;
use ort::session::{Session, SessionOutputs};
use ort::value::{TensorRef, ValueType};

/// Square side of the palm detector input.
pub const PALM_INPUT_SIZE: usize = 192;
/// Square side of the hand landmark input.
pub const HAND_INPUT_SIZE: usize = 224;
/// Sigmoid score a palm box needs before it is considered.
pub const PALM_SCORE_THRESHOLD: f64 = 0.5;
/// IoU above which a lower-scoring palm box is suppressed.
pub const PALM_NMS_IOU_THRESHOLD: f64 = 0.3;
/// Upper bound on retained detections, matching the reference `topK`.
pub const PALM_TOP_K: usize = 5000;
/// Landmark presence confidence required to report a hand.
///
/// The reference demo uses 0.8; 0.5 keeps the daemon responsive at the cost of the occasional
/// phantom hand, so it is a field rather than a constant (see [`HandPipeline::with_hand_threshold`]).
pub const HAND_PRESENCE_THRESHOLD: f64 = 0.5;
/// Per-anchor predictor width of the palm regressor: 4 box deltas + 7 landmarks x 2.
const PALM_PREDICTOR_CHANNELS: usize = 18;

// ROI recipes copied from mp_handpose.py; the vectors are fractions of the box size.
const PALM_BOX_PRE_SHIFT: (f64, f64) = (0.0, 0.0);
const PALM_BOX_PRE_ENLARGE: f64 = 4.0;
const PALM_BOX_SHIFT: (f64, f64) = (0.0, -0.4);
const PALM_BOX_ENLARGE: f64 = 3.0;
const HAND_BOX_SHIFT: (f64, f64) = (0.0, -0.1);
const HAND_BOX_ENLARGE: f64 = 1.65;
/// Palm landmark indices used to decide the hand orientation: wrist and middle-finger MCP.
const INDEX_OF_PALM_BASE: usize = 0;
const INDEX_OF_MIDDLE_FINGER_BASE: usize = 2;
const PALM_LANDMARK_COUNT: usize = 7;
const HAND_LANDMARK_COUNT: usize = 21;

/// One keypoint. Screen landmarks are in frame pixels (`z` is relative to the wrist); world
/// landmarks are the model's metric coordinates, origin at the wrist, unit-agnostic.
#[derive(Copy, Clone, Debug, Default)]
pub struct Landmark {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Which hand the model thinks it is looking at.
///
/// MediaPipe assumes a mirrored (selfie) view, so on an unmirrored webcam feed this is flipped.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Handedness {
    Left,
    Right,
}

/// A hand resolved by both stages.
#[derive(Copy, Clone, Debug)]
pub struct Hand {
    /// 21 keypoints in frame pixels: wrist, then thumb/index/middle/ring/pinky rays.
    pub landmarks: [Landmark; HAND_LANDMARK_COUNT],
    /// 21 metric keypoints, useful for gestures that are depth-sensitive.
    pub world_landmarks: [Landmark; HAND_LANDMARK_COUNT],
    pub handedness: Handedness,
    /// Landmark presence confidence from stage 2.
    pub score: f64,
    /// Hand bounding box in frame pixels (the ROI a tracker would use next frame).
    pub bbox: Rect,
}

/// Per-stage inference cost of the most recent [`HandPipeline::detect`] call.
#[derive(Copy, Clone, Debug, Default)]
pub struct Timings {
    pub palm_ms: f64,
    pub hand_ms: f64,
}

/// One loaded model plus the input contract read from its metadata.
///
/// The output names live on the pipeline rather than here: `Session::run` keeps the session
/// mutably borrowed for as long as its outputs are alive, so a stage field cannot be read while
/// decoding.
struct Stage {
    session: Session,
    label: &'static str,
    /// Square input side the graph expects.
    size: usize,
    layout: Layout,
}

impl Stage {
    fn open(environment: &Environment, spec: &model::ModelSpec, expected_size: usize) -> Result<(Stage, Vec<String>)> {
        let session = model::open_session(environment, spec)?;
        let (size, layout) = inspect_input(&session, expected_size, spec.label)?;
        let output_names: Vec<String> = session.outputs().iter().map(|outlet| outlet.name().to_string()).collect();
        println!(
            "{}: input {size}x{size} {:?}, {} output(s): {}",
            spec.label,
            layout,
            output_names.len(),
            output_names.join(", ")
        );
        Ok((Stage { session, label: spec.label, size, layout }, output_names))
    }

    fn run<'a>(&'a mut self, blob: &[f32]) -> Result<SessionOutputs<'a>> {
        let label = self.label;
        let tensor = make_input(blob, self.size, self.layout)?;
        match self.session.run(ort::inputs![tensor]) {
            Ok(outputs) => Ok(outputs),
            Err(why) => Err(anyhow!("{label} inference failed: {why}")),
        }
    }
}

/// Reads the graph's input metadata instead of trusting the documented shape: the conversion may
/// expose `HWC` (as TFLite does) or `CHW`, and a quantised build can differ again.
fn inspect_input(session: &Session, expected: usize, label: &str) -> Result<(usize, Layout)> {
    let outlet = session.inputs().first().ok_or_else(|| anyhow!("the {label} model reports no inputs"))?;
    let dims: Vec<i64> = match outlet.dtype() {
        ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
        other => bail!("the {label} model input is not a tensor ({other:?}); run `cargo run --release -- describe`"),
    };
    if dims.len() != 4 {
        bail!("the {label} model input is not rank 4 ({dims:?}); run `cargo run --release -- describe`");
    }
    // Dynamic dimensions arrive as -1, so only the channel axis is dependable.
    let layout = if dims[1] == 3 {
        Layout::Planar
    } else if dims[3] == 3 {
        Layout::Interleaved
    } else {
        println!("Note: cannot tell the channel order of the {label} input, assuming HWC");
        Layout::Interleaved
    };
    let (height, width) = match layout {
        Layout::Planar => (dims[2], dims[3]),
        Layout::Interleaved => (dims[1], dims[2]),
    };
    let side = if height > 0 && width > 0 {
        if height != width {
            bail!("the {label} model input is {width}x{height}, but the pipeline assumes a square input");
        }
        height as usize
    } else {
        expected
    };
    if side != expected {
        println!("Note: {label} model wants a {side}x{side} input (reference default was {expected}x{expected})");
    }
    Ok((side, layout))
}

fn make_input(blob: &[f32], size: usize, layout: Layout) -> Result<TensorRef<'_, f32>> {
    let (shape, data) = match layout {
        Layout::Interleaved => ([1usize, size, size, 3], blob),
        Layout::Planar => ([1usize, 3, size, size], blob),
    };
    TensorRef::from_array_view((shape, data)).context("Failed to wrap the input blob as a tensor view")
}

/// A palm detection in frame coordinates.
#[derive(Copy, Clone, Debug)]
struct Candidate {
    bbox: Corners,
    score: f64,
    landmarks: [(f64, f64); PALM_LANDMARK_COUNT],
}

/// Everything stage 2 needs to know about one palm: how to build its upright crop, and how to
/// map the resulting keypoints back to frame pixels.
#[derive(Copy, Clone, Debug)]
struct HandPlan {
    /// Blob pixel -> frame pixel, used by the resampler.
    mapping: Affine,
    /// Rotation-only transform for de-rotating keypoints (no translation).
    vector_rotation: Affine,
    /// Inverse of the big rotation, mapping the crop box centre from R space back to P space.
    inverse_rotation: Affine,
    /// Centre of the stage-2 crop box in R space.
    crop_center: (f64, f64),
    /// Frame pixels per blob pixel; the palm stage scales its anchors by `max(width, height)`.
    pixel_scale: f64,
    /// Offset of the pre-rotation crop inside the frame.
    bias: (f64, f64),
    /// Blob side (224 unless the model asked otherwise).
    side: usize,
}

/// Runs both MediaPipe stages over the camera frames.
pub struct HandPipeline {
    palm: Stage,
    hand: Stage,
    palm_outputs: Vec<String>,
    hand_outputs: Vec<String>,
    palm_blob: Vec<f32>,
    hand_blob: Vec<f32>,
    candidates: Vec<Candidate>,
    order: Vec<usize>,
    kept: Vec<Candidate>,
    max_hands: usize,
    hand_threshold: f64,
    timings: Timings,
}

impl HandPipeline {
    /// Loads the palm and landmark graphs. Expects the assets in `models/`; call
    /// [`model::fetch_models`] (the `fetch-models` subcommand) first.
    pub fn new(environment: &Environment) -> Result<HandPipeline> {
        model::require_models()?;
        let (palm, palm_outputs) = Stage::open(environment, &model::PALM, PALM_INPUT_SIZE)?;
        let (hand, hand_outputs) = Stage::open(environment, &model::LANDMARK, HAND_INPUT_SIZE)?;
        let palm_blob = vec![0.0f32; palm.size * palm.size * 3];
        let hand_blob = vec![0.0f32; hand.size * hand.size * 3];
        Ok(HandPipeline {
            palm,
            hand,
            palm_outputs,
            hand_outputs,
            palm_blob,
            hand_blob,
            candidates: Vec::with_capacity(PALM_ANCHORS.len()),
            order: Vec::with_capacity(PALM_ANCHORS.len()),
            kept: Vec::with_capacity(4),
            max_hands: 1,
            hand_threshold: HAND_PRESENCE_THRESHOLD,
            timings: Timings::default(),
        })
    }

    /// How many palm boxes may be promoted to hands (MediaPipe's `max_num_hands`).
    pub fn with_max_hands(mut self, max_hands: usize) -> Result<Self> {
        if max_hands == 0 {
            bail!("max_hands must be at least 1");
        }
        self.max_hands = max_hands;
        Ok(self)
    }

    /// Landmark presence score a hand must reach to be reported (see [`HAND_PRESENCE_THRESHOLD`]).
    pub fn with_hand_threshold(mut self, threshold: f64) -> Self {
        self.hand_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn timings(&self) -> Timings {
        self.timings
    }

    /// Detects hands in an RGB8 frame of `width x height`.
    pub fn detect(&mut self, image_buffer: &[u8], width: u32, height: u32) -> Result<Vec<Hand>> {
        let frame_w = width as usize;
        let frame_h = height as usize;
        if frame_w == 0 || frame_h == 0 {
            bail!("the frame has no dimensions");
        }
        let Some(frame) = roi::Source::rgb(image_buffer, frame_w, frame_h) else {
            bail!(
                "expected {} RGB bytes for a {frame_w}x{frame_h} frame, got {}",
                frame_w * frame_h * 3,
                image_buffer.len()
            );
        };

        let palm_started = std::time::Instant::now();
        let plans = self.plan_hands(frame)?;
        self.timings.palm_ms = palm_started.elapsed().as_secs_f64() * 1000.0;

        let hand_started = std::time::Instant::now();
        let mut hands = Vec::with_capacity(plans.len());
        for plan in &plans {
            if let Some(hand) = self.resolve_hand(frame, plan)? {
                hands.push(hand);
            }
        }
        self.timings.hand_ms = hand_started.elapsed().as_secs_f64() * 1000.0;
        Ok(hands)
    }

    /// Stage 1 plus the ROI maths: one palm inference, then a crop plan per surviving detection.
    fn plan_hands(&mut self, frame: roi::Source<'_>) -> Result<Vec<HandPlan>> {
        let (frame_w, frame_h) = (frame.width, frame.height);
        let side = self.palm.size;
        let (mapping, pad_bias) = letterbox(frame_w, frame_h, side);
        roi::write_blob(frame, &mapping, (side, side), self.palm.layout, &mut self.palm_blob);

        self.candidates.clear();
        self.order.clear();
        self.kept.clear();

        let outputs = self.palm.run(&self.palm_blob)?;
        decode_palm_outputs(&outputs, &self.palm_outputs, side, (frame_w, frame_h), pad_bias, &mut self.candidates)?;
        drop(outputs);

        // Highest score first, greedy IoU suppression, then at most `max_hands` survivors.
        self.order.extend(0..self.candidates.len());
        self.order.sort_by(|&a, &b| self.candidates[b].score.total_cmp(&self.candidates[a].score));
        for &index in &self.order {
            if self.kept.len() >= PALM_TOP_K || self.kept.len() >= self.max_hands {
                break;
            }
            let candidate = self.candidates[index];
            let dominated = self.kept.iter().any(|kept| roi::iou(kept.bbox, candidate.bbox) >= PALM_NMS_IOU_THRESHOLD);
            if !dominated {
                self.kept.push(candidate);
            }
        }

        let hand_side = self.hand.size;
        let mut plans = Vec::with_capacity(self.kept.len());
        for index in 0..self.kept.len() {
            let candidate = self.kept[index];
            plans.push(plan_hand_roi(&candidate, frame_w, frame_h, hand_side));
        }
        Ok(plans)
    }

    /// Stage 2 for one palm: resample the upright crop, run the graph, de-rotate the keypoints.
    fn resolve_hand(&mut self, frame: roi::Source<'_>, plan: &HandPlan) -> Result<Option<Hand>> {
        let side = plan.side;
        roi::write_blob(frame, &plan.mapping, (side, side), self.hand.layout, &mut self.hand_blob);

        let outputs = self.hand.run(&self.hand_blob)?;
        decode_hand_outputs(&outputs, &self.hand_outputs, plan, self.hand_threshold)
    }
}

/// Maps a `side x side` letterboxed blob onto the frame, preserving aspect ratio.
///
/// Returns the destination -> source mapping and the padding offset in *frame* pixels, which the
/// palm decoder subtracts from its anchors (the reference does `(pad / ratio).astype(int32)`).
fn letterbox(source_w: usize, source_h: usize, side: usize) -> (Affine, (f64, f64)) {
    let ratio = (side as f64 / source_h as f64).min(side as f64 / source_w as f64);
    let scaled_w = ((source_w as f64 * ratio) as usize).max(1);
    let scaled_h = ((source_h as f64 * ratio) as usize).max(1);
    let left = (side - scaled_w) / 2;
    let top = (side - scaled_h) / 2;
    let inverse = 1.0 / ratio;
    // Dividing (never multiplying by a reciprocal) keeps `pad_bias` bit-identical to the
    // reference's `pad_bias / ratio`, whose `astype(int32)` truncation the decoder relies on.
    let mapping = Affine {
        a: inverse,
        d: inverse,
        tx: -(left as f64) / ratio,
        ty: -(top as f64) / ratio,
        ..Affine::IDENTITY
    };
    let pad_bias = ((left as f64 / ratio).trunc(), (top as f64 / ratio).trunc());
    (mapping, pad_bias)
}

/// Runs the sigmoid + anchor decode over the palm stage outputs and collects detections.
fn decode_palm_outputs(
    outputs: &SessionOutputs,
    names: &[String],
    input_size: usize,
    frame: (usize, usize),
    pad_bias: (f64, f64),
    out: &mut Vec<Candidate>,
) -> Result<()> {
    let mut regressor: Option<&[f32]> = None;
    let mut scores: Option<&[f32]> = None;
    for index in 0..names.len() {
        let (shape, data) = outputs[index]
            .try_extract_tensor::<f32>()
            .with_context(|| format!("Failed to extract palm output #{} ({})", index, names[index]))?;
        let per_anchor = if data.len() % PALM_ANCHORS.len() == 0 {
            data.len() / PALM_ANCHORS.len()
        } else {
            bail!(
                "palm output {} has {} values for {} anchors; run `cargo run --release -- describe` and check the graph",
                names[index],
                data.len(),
                PALM_ANCHORS.len()
            );
        };
        match per_anchor {
            1 => scores = Some(data),
            PALM_PREDICTOR_CHANNELS => regressor = Some(data),
            other => bail!(
                "unexpected palm output {} with shape {} and {other} values per anchor (expected 1 or {PALM_PREDICTOR_CHANNELS})",
                names[index],
                shape,
            ),
        }
    }
    let regressor = regressor.ok_or_else(|| anyhow!("the palm model exposes no {}-value regressor output", PALM_PREDICTOR_CHANNELS))?;
    let scores = scores.ok_or_else(|| anyhow!("the palm model exposes no 1-value score output"))?;

    // `scale = max(original_shape)`; anchors are unit fractions of that scale.
    let scale = frame.0.max(frame.1) as f64;
    let inv_size = 1.0 / input_size as f64;
    for (index, anchor) in PALM_ANCHORS.iter().enumerate() {
        let raw = f64::from(scores[index]);
        let score = 1.0 / (1.0 + (-raw).exp());
        if score < PALM_SCORE_THRESHOLD {
            continue;
        }
        let base = index * PALM_PREDICTOR_CHANNELS;
        let center_x = f64::from(regressor[base]) * inv_size;
        let center_y = f64::from(regressor[base + 1]) * inv_size;
        let width = f64::from(regressor[base + 2]) * inv_size;
        let height = f64::from(regressor[base + 3]) * inv_size;
        let (anchor_x, anchor_y) = (f64::from(anchor[0]), f64::from(anchor[1]));
        let to_frame = |offset_x: f64, offset_y: f64| {
            ((offset_x + anchor_x) * scale - pad_bias.0, (offset_y + anchor_y) * scale - pad_bias.1)
        };
        let (x0, y0) = to_frame(center_x - width / 2.0, center_y - height / 2.0);
        let (x1, y1) = to_frame(center_x + width / 2.0, center_y + height / 2.0);

        let mut landmarks = [(0.0, 0.0); PALM_LANDMARK_COUNT];
        for (point, landmark) in landmarks.iter_mut().enumerate() {
            let dx = f64::from(regressor[base + 4 + point * 2]) * inv_size;
            let dy = f64::from(regressor[base + 5 + point * 2]) * inv_size;
            *landmark = to_frame(dx, dy);
        }

        out.push(Candidate {
            bbox: Corners { x0, y0, x1, y1 },
            score,
            landmarks,
        });
    }
    Ok(())
}
/// Computes the stage-2 crop plan for one palm, following `_cropAndPadFromPalm` and the rotation
/// step of mp_handpose.py.
///
/// Two passes, exactly like the reference: a wide box (shift 0, enlarge x4) padded to the crop
/// diagonal, which is the space the rotation is computed in; then the box that actually feeds the
/// landmark model (shift `[0, -0.4]`, enlarge x3) padded to its longest side.
fn plan_hand_roi(candidate: &Candidate, frame_w: usize, frame_h: usize, side: usize) -> HandPlan {
    let grown = candidate
        .bbox
        .to_rect()
        .shifted(PALM_BOX_PRE_SHIFT)
        .enlarged_uniform(PALM_BOX_PRE_ENLARGE);
    let (x0, y0, x1, y1) = quantize(&grown, frame_w as i64, frame_h as i64);
    let crop_w = (x1 - x0).max(1);
    let crop_h = (y1 - y0).max(1);
    let square = integer_diagonal(crop_w, crop_h);
    let left = (square - crop_w) / 2;
    let top = (square - crop_h) / 2;
    // Origin of the pre-rotation crop (space P) inside the frame; `P + bias = frame`.
    let bias = (x0 - left, y0 - top);
    let box_p = Rect::new(left as f64, top as f64, crop_w as f64, crop_h as f64);

    let landmarks_p: [(f64, f64); PALM_LANDMARK_COUNT] =
        candidate.landmarks.map(|point| (point.0 - bias.0 as f64, point.1 - bias.1 as f64));
    let angle = rotation_degrees(&landmarks_p);
    let rotation = Affine::rotation(box_p.center(), angle); // P -> R
    let inverse_rotation = rotation.invert_rigid(); // R -> P

    let mut rotated = [(0.0, 0.0); PALM_LANDMARK_COUNT];
    for (source, target) in landmarks_p.iter().zip(rotated.iter_mut()) {
        *target = rotation.apply(*source);
    }
    let rotated_box = roi::bounding_box(&rotated).unwrap_or(box_p);

    let grown2 = rotated_box.shifted(PALM_BOX_SHIFT).enlarged_uniform(PALM_BOX_ENLARGE);
    let (bx0, by0, bx1, by1) = quantize(&grown2, square, square);
    let crop2_w = (bx1 - bx0).max(1);
    let crop2_h = (by1 - by0).max(1);
    let square2 = crop2_w.max(crop2_h);
    let left2 = (square2 - crop2_w) / 2;
    let top2 = (square2 - crop2_h) / 2;
    // Origin of the stage-2 crop (space R) inside R.
    let bias2 = (bx0 - left2, by0 - top2);

    let square_r = Rect::new(bias2.0 as f64, bias2.1 as f64, square2 as f64, square2 as f64);
    let box2_r = Rect::new(bx0 as f64, by0 as f64, crop2_w as f64, crop2_h as f64);
    let to_frame = Affine::translation(bias.0 as f64, bias.1 as f64);
    let mapping = square_r
        .map_from_unit_box(side, side)
        .then(&inverse_rotation)
        .then(&to_frame);

    HandPlan {
        mapping,
        vector_rotation: Affine::rotation((0.0, 0.0), angle),
        inverse_rotation,
        crop_center: box2_r.center(),
        pixel_scale: (crop2_w.max(crop2_h) as f64) / side as f64,
        bias: (bias.0 as f64, bias.1 as f64),
        side,
    }
}

/// Turns the wrist -> middle-finger-MCP vector into the upright hand angle, in degrees.
fn rotation_degrees(landmarks: &[(f64, f64); PALM_LANDMARK_COUNT]) -> f64 {
    let base = landmarks[INDEX_OF_PALM_BASE];
    let middle = landmarks[INDEX_OF_MIDDLE_FINGER_BASE];
    let radians = std::f64::consts::FRAC_PI_2 - (-(middle.1 - base.1)).atan2(middle.0 - base.0);
    let circle = 2.0 * std::f64::consts::PI;
    (radians - circle * ((radians + std::f64::consts::PI) / circle).floor()).to_degrees()
}

/// `astype(int32)` followed by `np.clip`: the reference crops on truncated integer pixels, and the
/// blob geometry has to agree or the keypoints drift by half a pixel.
fn quantize(rect: &Rect, image_w: i64, image_h: i64) -> (i64, i64, i64, i64) {
    let x0 = rect.x.trunc() as i64;
    let y0 = rect.y.trunc() as i64;
    let x1 = (rect.x + rect.w).trunc() as i64;
    let y1 = (rect.y + rect.h).trunc() as i64;
    (
        x0.clamp(0, image_w),
        y0.clamp(0, image_h),
        x1.clamp(0, image_w),
        y1.clamp(0, image_h),
    )
}

/// `int(np.linalg.norm([height, width]))` - the square that keeps every crop corner visible.
fn integer_diagonal(width: i64, height: i64) -> i64 {
    ((width as f64) * (width as f64) + (height as f64) * (height as f64)).sqrt() as i64
}

/// Which landmark output this tensor is, by name first and element count second.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Slot {
    Landmarks,
    World,
    Confidence,
    Handedness,
    Unknown,
}

fn classify(name: &str, values: usize, landmarks_taken: bool) -> Slot {
    let lowered = name.to_ascii_lowercase();
    // `handedness` contains `hand`, and `world_landmarks` contains `landmark`, so order matters.
    if lowered.contains("world") {
        return Slot::World;
    }
    if lowered.contains("handedness") || lowered.contains("multi_hand") {
        return Slot::Handedness;
    }
    if lowered.contains("landmark") {
        return if landmarks_taken { Slot::World } else { Slot::Landmarks };
    }
    if lowered.contains("confidence") || lowered.contains("presence") || lowered.contains("score") {
        return Slot::Confidence;
    }
    // tf2onnx exports keep generic names (`Identity`, `Identity_1`, ...), and then the element
    // count is unambiguous for this graph: 63 screen landmarks, 1 confidence, 2 handedness scores,
    // 63 world landmarks. Two 63-value tensors are told apart by declaration order, screen first.
    match values {
        1 => Slot::Confidence,
        2 => Slot::Handedness,
        _ if values == HAND_LANDMARK_COUNT * 3 => {
            if landmarks_taken {
                Slot::World
            } else {
                Slot::Landmarks
            }
        }
        _ => Slot::Unknown,
    }
}

/// Runs the landmark graph for one palm crop and maps its keypoints back to frame pixels.
fn decode_hand_outputs(outputs: &SessionOutputs, names: &[String], plan: &HandPlan, threshold: f64) -> Result<Option<Hand>> {
    let mut screen: Option<&[f32]> = None;
    let mut world: Option<&[f32]> = None;
    let mut confidence: Option<&[f32]> = None;
    let mut handedness: Option<&[f32]> = None;
    for index in 0..names.len() {
        let (shape, data) = outputs[index]
            .try_extract_tensor::<f32>()
            .with_context(|| format!("Failed to extract hand output #{} ({})", index, names[index]))?;
        match classify(&names[index], data.len(), screen.is_some()) {
            Slot::Landmarks => screen = Some(data),
            Slot::World => world = Some(data),
            Slot::Confidence => confidence = Some(data),
            Slot::Handedness => handedness = Some(data),
            Slot::Unknown => println!("Note: unrecognised hand output {} with shape {shape}", names[index]),
        }
    }

    let confidence = confidence.ok_or_else(|| anyhow!("the hand model exposes no confidence/presence output"))?;
    let screen = screen.ok_or_else(|| anyhow!("the hand model exposes no {}-value landmark output", HAND_LANDMARK_COUNT * 3))?;
    let score = f64::from(confidence[0]);
    if score < threshold {
        return Ok(None);
    }
    // Index 0 of the handedness pair, as the reference reports it: probability of a left hand.
    let left_score = handedness.map_or(0.5, |values| f64::from(values[0]));

    // Blob -> frame: centre the crop, undo the reference's `max(scale_factor)` stretch, undo the
    // rotation, then step back through the big rotation into the frame (reference `_postprocess`).
    let half = plan.side as f64 / 2.0;
    let original_center = plan.inverse_rotation.apply(plan.crop_center);
    let mut landmarks = [Landmark::default(); HAND_LANDMARK_COUNT];
    let mut points = [(0.0, 0.0); HAND_LANDMARK_COUNT];
    for (key, point) in points.iter_mut().enumerate() {
        let offset = key * 3;
        let stretched = (
            (f64::from(screen[offset]) - half) * plan.pixel_scale,
            (f64::from(screen[offset + 1]) - half) * plan.pixel_scale,
        );
        let rotated = plan.vector_rotation.rotate_vector(stretched);
        let landmark = Landmark {
            x: rotated.0 + original_center.0 + plan.bias.0,
            y: rotated.1 + original_center.1 + plan.bias.1,
            z: f64::from(screen[offset + 2]) * plan.pixel_scale,
        };
        landmarks[key] = landmark;
        *point = (landmark.x, landmark.y);
    }

    // World landmarks are metric and wrist-centred: the reference only de-rotates them.
    let mut world_landmarks = [Landmark::default(); HAND_LANDMARK_COUNT];
    if let Some(values) = world {
        for (key, landmark) in world_landmarks.iter_mut().enumerate() {
            let offset = key * 3;
            let rotated = plan
                .vector_rotation
                .rotate_vector((f64::from(values[offset]), f64::from(values[offset + 1])));
            *landmark = Landmark {
                x: rotated.0,
                y: rotated.1,
                z: f64::from(values[offset + 2]),
            };
        }
    }

    let bbox = roi::bounding_box(&points)
        .unwrap_or_default()
        .shifted(HAND_BOX_SHIFT)
        .enlarged_uniform(HAND_BOX_ENLARGE);

    Ok(Some(Hand {
        landmarks,
        world_landmarks,
        handedness: if left_score >= 0.5 { Handedness::Left } else { Handedness::Right },
        score,
        bbox,
    }))
}

impl Hand {
    /// The index-finger tip, which is what a pointer app turns into a cursor.
    pub fn tip(&self) -> Landmark {
        self.landmarks[8]
    }

    /// Index-finger tip depth from the landmark model's metric space, in its own unit.
    pub fn tip_depth(&self) -> f64 {
        self.world_landmarks[8].z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_frame_is_letterboxed_into_the_model_square() {
        let (mapping, pad_bias) = letterbox(320, 240, 192);
        // ratio = min(192/240, 192/320) = 0.6 -> 192x144 centred in 192 rows.
        assert!((mapping.a - 1.0 / 0.6).abs() < 1e-9, "scale = {}", mapping.a);
        assert!(pad_bias.0.abs() < 0.5, "horizontal pad = {}", pad_bias.0);
        assert!((pad_bias.1 - 40.0).abs() < 0.5, "vertical pad = {}", pad_bias.1);
    }

    #[test]
    fn an_upright_palm_needs_no_rotation() {
        let mut landmarks = [(0.0, 0.0); PALM_LANDMARK_COUNT];
        landmarks[INDEX_OF_PALM_BASE] = (50.0, 200.0);
        landmarks[INDEX_OF_MIDDLE_FINGER_BASE] = (50.0, 80.0);
        assert!(rotation_degrees(&landmarks).abs() < 1e-9, "angle = {}", rotation_degrees(&landmarks));
    }

    #[test]
    fn a_palm_pointing_right_rotates_a_quarter_turn() {
        let mut landmarks = [(0.0, 0.0); PALM_LANDMARK_COUNT];
        landmarks[INDEX_OF_PALM_BASE] = (0.0, 0.0);
        landmarks[INDEX_OF_MIDDLE_FINGER_BASE] = (100.0, 0.0);
        assert!((rotation_degrees(&landmarks) - 90.0).abs() < 1e-9, "angle = {}", rotation_degrees(&landmarks));
    }

    #[test]
    fn a_palm_plan_maps_its_crop_centre_back_into_the_frame() {
        let candidate = Candidate {
            bbox: Corners {
                x0: 100.0,
                y0: 60.0,
                x1: 160.0,
                y1: 140.0,
            },
            score: 0.9,
            landmarks: [
                (130.0, 130.0),
                (120.0, 100.0),
                (130.0, 70.0),
                (140.0, 75.0),
                (150.0, 85.0),
                (110.0, 90.0),
                (115.0, 95.0),
            ],
        };
        let plan = plan_hand_roi(&candidate, 320, 240, HAND_INPUT_SIZE);
        let centre = plan.mapping.apply((112.0, 112.0));
        assert!(
            centre.0 > 0.0 && centre.0 < 320.0 && centre.1 > 0.0 && centre.1 < 240.0,
            "centre = {centre:?}"
        );
        assert!(plan.pixel_scale > 0.0 && plan.pixel_scale < 10.0, "scale = {}", plan.pixel_scale);
    }

    #[test]
    fn hand_outputs_are_matched_by_name_then_by_size() {
        assert_eq!(classify("world_landmarks", 63, true), Slot::World);
        assert_eq!(classify("Identity_2", 2, true), Slot::Handedness);
        assert_eq!(classify("Identity", 63, false), Slot::Landmarks);
        assert_eq!(classify("Identity_1", 1, true), Slot::Confidence);
        assert_eq!(classify("Identity_9", 7, true), Slot::Unknown);
    }

    #[test]
    fn quantize_clamps_a_box_to_the_image() {
        // Truncate, then clamp: -40.6 -> 0, 259.8 -> 259, and both y edges collapse onto the
        // bottom of the frame because the box starts below it (so it is empty, as expected).
        let rect = Rect::new(-40.6, 250.9, 300.4, 200.7);
        assert_eq!(quantize(&rect, 320, 240), (0, 240, 259, 240));
        // A box that hangs off the top-left corner keeps just the part inside the image.
        let rect = Rect::new(-40.6, -20.4, 300.4, 200.7);
        assert_eq!(quantize(&rect, 320, 240), (0, 0, 259, 180));
    }
}
