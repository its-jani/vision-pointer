//! Phase 3: turn a tracked fingertip into OS cursor movement and pinch clicks.
//!
//! The camera sees a 320x240 slice of the world; the display it drives is something else entirely.
//! Between those two spaces sit three jobs: remap the frame position onto the screen, smooth the
//! jitter out ([`crate::filter`]), and decide when a thumb-and-finger pinch means a click.
//!
//! Movement uses absolute coordinates (`Coordinate::Abs`), so the cursor sits where the hand
//! points rather than accumulating drift, and on Windows it is delivered through
//! `MOUSEEVENTF_ABSOLUTE`, which bypasses the "enhance pointer precision" acceleration curve that
//! would otherwise make the mapping depend on how fast the hand moves.

use crate::filter::EmaFilter2D;
use anyhow::{Context, Result, anyhow, bail};
use enigo::{Button, Coordinate, Direction, Enigo, Mouse, Settings};
use std::time::{Duration, Instant};

/// Fraction of the frame, measured from each edge, that maps outside the screen.
///
/// A hand can physically travel much further left in view than the cursor can travel left on the
/// display. Compressing the useful range into the middle 70% of the frame means the screen corners
/// and edges are reached by pointing comfortably inside the frame, instead of by putting the
/// fingertip against the edge of the camera view where the crop goes unreliable.
pub const ACTIVE_MARGIN: f32 = 0.15;
/// Thumb-tip to index-tip distance (in frame fractions, see [`crate::hands::Hand::pinch_distance`])
/// below which the pinch counts as a click. Touching tips measure ~0; 0.04 is roughly 13 px apart
/// on a 320-wide frame.
pub const PINCH_CLICK_DISTANCE: f32 = 0.04;
/// Shortest spacing between injected clicks, so one sustained pinch is one click and a shaky hold
/// cannot rattle off double-clicks.
pub const CLICK_COOLDOWN: Duration = Duration::from_millis(300);

/// Drives the real cursor.
pub struct MouseController {
    enigo: Enigo,
    filter: EmaFilter2D,
    /// Main display size in pixels, asked of the OS at construction.
    display: (i32, i32),
    margin: f32,
    pinch_distance: f32,
    /// The webcam feed is not mirrored, so a hand moving right appears to move right in the frame
    /// while the person is moving it left. Mirroring the horizontal axis makes the cursor follow
    /// the hand the way a touch screen does.
    mirror_x: bool,
    /// Report where the cursor *would* go without touching the real one.
    dry_run: bool,
    last_click: Option<Instant>,
    /// Whether the previous frame was already inside the pinch threshold.
    pinching: bool,
    /// Last coordinate mapped, for the status line and for `dry_run`.
    last_position: Option<(i32, i32)>,
}

/// Frame pixels -> display pixels: mirror, apply the active-margin inset, then scale.
///
/// Returns fractional screen pixels so the caller can smooth before rounding.
fn map_point(
    raw_x: f32,
    raw_y: f32,
    frame: (u32, u32),
    display: (i32, i32),
    margin: f32,
    mirror_x: bool,
) -> (f32, f32) {
    // Frame pixels -> 0..1 across the frame.
    let mut x = raw_x / frame.0 as f32;
    let y = raw_y / frame.1 as f32;
    if mirror_x {
        x = 1.0 - x;
    }

    // The inset band in the middle of the frame owns the whole screen; anything outside it is
    // clamped to the edge, which is what makes the corners reachable.
    let span = 1.0 - 2.0 * margin;
    let x = ((x - margin) / span).clamp(0.0, 1.0);
    let y = ((y - margin) / span).clamp(0.0, 1.0);

    // Screen pixels, inset by one so the extreme position lands on the last addressable pixel.
    (x * (display.0 - 1) as f32, y * (display.1 - 1) as f32)
}

impl MouseController {
    /// Connects to the display and asks it for its size.
    pub fn new() -> Result<MouseController> {
        let enigo =
            Enigo::new(&Settings::default()).context("Failed to connect to the display for input injection")?;
        let display = enigo
            .main_display()
            .map_err(|why| anyhow!("Failed to read the display size ({why})"))?;
        if display.0 <= 0 || display.1 <= 0 {
            bail!("the display reported an unusable size {display:?}");
        }
        println!("Controlling the main display at {}x{}", display.0, display.1);
        Ok(MouseController {
            enigo,
            filter: EmaFilter2D::new(),
            display,
            margin: ACTIVE_MARGIN,
            pinch_distance: PINCH_CLICK_DISTANCE,
            mirror_x: true,
            dry_run: false,
            last_click: None,
            pinching: false,
            last_position: None,
        })
    }

    /// Weight of the newest cursor sample; see [`EmaFilter2D`].
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.filter = EmaFilter2D::with_alpha(alpha);
        self
    }

    pub fn with_margin(mut self, margin: f32) -> Self {
        // Below 0.5 keeps `1 - 2 * margin` positive; the clamp stops a degenerate mapping.
        self.margin = margin.clamp(0.0, 0.45);
        self
    }

    pub fn with_pinch_distance(mut self, threshold: f32) -> Self {
        self.pinch_distance = threshold.max(0.0);
        self
    }

    pub fn with_mirror(mut self, mirror: bool) -> Self {
        self.mirror_x = mirror;
        self
    }

    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    pub fn display_size(&self) -> (i32, i32) {
        self.display
    }

    /// The tuning actually in force, after clamping - so a banner can report what will happen
    /// rather than what was asked for.
    pub fn tuning(&self) -> String {
        format!(
            "alpha {:.2} | inset {:.0}% each edge | pinch < {:.3}",
            self.filter.alpha(),
            self.margin * 100.0,
            self.pinch_distance
        )
    }

    /// Screen coordinate the cursor was last sent to, if pointing has engaged at all.
    pub fn last_position(&self) -> Option<(i32, i32)> {
        self.last_position
    }

    /// Maps a fingertip in frame pixels onto the display, smooths it, and moves the cursor there.
    ///
    /// The new coordinate is available afterwards through [`last_position`][Self::last_position],
    /// which is what the status line prints.
    pub fn map_and_move(
        &mut self,
        raw_x: f32,
        raw_y: f32,
        frame_w: u32,
        frame_h: u32,
    ) -> Result<()> {
        if frame_w == 0 || frame_h == 0 {
            bail!("cannot map onto the display from a {frame_w}x{frame_h} frame");
        }
        let (target_x, target_y) = map_point(raw_x, raw_y, (frame_w, frame_h), self.display, self.margin, self.mirror_x);

        let (smoothed_x, smoothed_y) = self.filter.filter(target_x, target_y);
        let position = (smoothed_x.round() as i32, smoothed_y.round() as i32);
        self.last_position = Some(position);

        if !self.dry_run {
            let (x, y) = position;
            self.enigo
                .move_mouse(x, y, Coordinate::Abs)
                .map_err(|why| anyhow!("Failed to move the cursor to ({x}, {y}) ({why})"))?;
        }
        Ok(())
    }

    /// Clicks when a pinch starts.
    ///
    /// A click fires on the frame where the distance first drops under the threshold, and not again
    /// until the fingertips separate past it and [`CLICK_COOLDOWN`] has passed. Edge-triggering is
    /// what keeps a sustained pinch - which is how people naturally hold a click - from becoming a
    /// machine gun of clicks every 300 ms.
    pub fn process_click(&mut self, pinch_dist: f32) -> bool {
        let pinched = pinch_dist < self.pinch_distance;
        let started = pinched && !self.pinching;
        self.pinching = pinched;
        if !started {
            return false;
        }
        if let Some(previous) = self.last_click
            && previous.elapsed() < CLICK_COOLDOWN
        {
            return false;
        }
        self.last_click = Some(Instant::now());
        if self.dry_run {
            return true;
        }
        if let Err(why) = self.enigo.button(Button::Left, Direction::Click) {
            eprintln!("Warning: left click failed ({why})");
            return false;
        }
        true
    }

    /// Forgets everything, for the frame after tracking drops out: the next sample is a new hand,
    /// not the old one that moved there, and a pinch held across the dropout is a new gesture.
    pub fn reset(&mut self) {
        self.filter.reset();
        self.pinching = false;
        self.last_position = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: (u32, u32) = (320, 240);
    const DISPLAY: (i32, i32) = (1920, 1080);

    /// Mirroring off, so the expectations below read as pure mapping arithmetic.
    fn mapped(x: f32, y: f32) -> (f32, f32) {
        map_point(x, y, FRAME, DISPLAY, ACTIVE_MARGIN, false)
    }

    #[test]
    fn the_inset_edges_own_the_screen_corners() {
        // 15% of 320 is 48 px, 15% of 240 is 36 px: pointing there reaches the extreme pixel.
        let (x, y) = mapped(48.0, 36.0);
        assert_eq!((x.round() as i32, y.round() as i32), (0, 0), "top-left: {x}, {y}");
        let (x, y) = mapped(272.0, 204.0);
        assert_eq!((x.round() as i32, y.round() as i32), (1919, 1079), "bottom-right: {x}, {y}");
    }

    #[test]
    fn pointing_past_the_frame_edge_clamps_to_the_screen_edge() {
        let (x, y) = mapped(0.0, 0.0);
        assert_eq!((x, y), (0.0, 0.0), "the frame corner must not overshoot the display");
        let (x, y) = mapped(320.0, 240.0);
        assert_eq!((x.round() as i32, y.round() as i32), (1919, 1079), "far corner: {x}, {y}");
    }

    #[test]
    fn the_frame_centre_is_the_screen_centre() {
        let (x, y) = mapped(160.0, 120.0);
        assert!((x - 959.5).abs() < 0.001 && (y - 539.5).abs() < 0.001, "centre = {x}, {y}");
    }

    #[test]
    fn mirroring_flips_only_the_horizontal_axis() {
        let left = map_point(48.0, 36.0, FRAME, DISPLAY, ACTIVE_MARGIN, true);
        let right = map_point(272.0, 36.0, FRAME, DISPLAY, ACTIVE_MARGIN, true);
        assert_eq!((left.0.round() as i32, left.1.round() as i32), (1919, 0), "left edge of the frame -> right of the screen");
        assert_eq!((right.0.round() as i32, right.1.round() as i32), (0, 0), "right edge -> left of the screen");
    }

    #[test]
    fn a_zero_margin_maps_the_frame_onto_the_whole_screen() {
        let (x, y) = map_point(320.0, 240.0, FRAME, DISPLAY, 0.0, false);
        assert_eq!((x.round() as i32, y.round() as i32), (1919, 1079), "no inset: {x}, {y}");
    }
}
