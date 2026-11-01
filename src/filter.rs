//! Exponential smoothing for the cursor.
//!
//! A fingertip is noisy: the landmark model moves it a few pixels per frame even when the hand is
//! held still, and the palm stage re-crops the hand every frame, which adds quantisation on top.
//! Feeding those coordinates straight to the OS produces a cursor that visibly vibrates and never
//! sits still on a target. An exponential moving average (EMA) keeps the recent average instead of
//! the newest sample, so small jitters cancel out while a real move still gets through:
//!
//! ```text
//! S_t = alpha * Y_t + (1 - alpha) * S_(t-1)
//! ```
//!
//! `alpha` is the weight of the newest measurement. High values track the hand closely but pass
//! jitter through; low values are glassy but lag behind.

/// Weight of the newest sample. 0.25 removes most fingertip jitter at ~30 FPS without the cursor
/// feeling detached from the hand.
pub const DEFAULT_ALPHA: f32 = 0.25;

/// Smooths a screen coordinate pair.
///
/// Each axis keeps only its own last average, so a hand that leaves the frame and comes back at a
/// different height does not drag the cursor diagonally through the old position.
#[derive(Copy, Clone, Debug)]
pub struct EmaFilter2D {
    prev_x: Option<f32>,
    prev_y: Option<f32>,
    alpha: f32,
}

impl Default for EmaFilter2D {
    fn default() -> Self {
        Self::new()
    }
}

impl EmaFilter2D {
    pub fn new() -> Self {
        Self::with_alpha(DEFAULT_ALPHA)
    }

    /// `alpha` outside 0..=1 is clamped: 0 freezes the cursor on the first sample, 1 disables
    /// smoothing entirely, and neither should be reachable by accident.
    pub fn with_alpha(alpha: f32) -> Self {
        Self { prev_x: None, prev_y: None, alpha: alpha.clamp(0.0, 1.0) }
    }

    /// Weight of the newest measurement.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Smooths one frame's coordinate. The first call after construction or [`reset`][Self::reset]
    /// returns the sample unchanged.
    pub fn filter(&mut self, x: f32, y: f32) -> (f32, f32) {
        // No history yet: the first measurement is the best estimate there is. Seeding this way is
        // what lets a reset filter snap to a re-acquired hand instead of sliding across the screen.
        let smoothed_x = match self.prev_x {
            Some(previous) => self.alpha * x + (1.0 - self.alpha) * previous,
            None => x,
        };
        let smoothed_y = match self.prev_y {
            Some(previous) => self.alpha * y + (1.0 - self.alpha) * previous,
            None => y,
        };
        self.prev_x = Some(smoothed_x);
        self.prev_y = Some(smoothed_y);
        (smoothed_x, smoothed_y)
    }

    /// Forgets the history, for when tracking drops out and the next sample belongs to a new hand
    /// (or the same hand somewhere else entirely).
    pub fn reset(&mut self) {
        self.prev_x = None;
        self.prev_y = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_sample_passes_through_untouched() {
        let mut filter = EmaFilter2D::new();
        assert_eq!(filter.filter(500.0, 300.0), (500.0, 300.0));
    }

    #[test]
    fn a_reset_reseeds_on_the_next_sample() {
        let mut filter = EmaFilter2D::new();
        filter.filter(500.0, 300.0);
        filter.reset();
        assert_eq!(filter.filter(100.0, 100.0), (100.0, 100.0), "no slide from the stale position");
    }

    #[test]
    fn a_step_is_followed_exponentially() {
        let mut filter = EmaFilter2D::with_alpha(0.25);
        filter.filter(0.0, 0.0);
        // S_t = 0.25 * 100 + 0.75 * S_(t-1), starting from 0.
        assert_eq!(filter.filter(100.0, 100.0), (25.0, 25.0));
        assert_eq!(filter.filter(100.0, 100.0), (43.75, 43.75));
        assert_eq!(filter.filter(100.0, 100.0), (57.8125, 57.8125));
    }

    #[test]
    fn jitter_around_a_stationary_target_is_attenuated() {
        let mut filter = EmaFilter2D::with_alpha(0.25);
        for _ in 0..30 {
            filter.filter(400.0, 400.0);
        }
        // Landmark noise of a few pixels around a settled value must not move the cursor much.
        let (x, y) = filter.filter(404.0, 396.0);
        assert!((x - 401.0).abs() < 1e-3, "x = {x}");
        assert!((y - 399.0).abs() < 1e-3, "y = {y}");
    }

    #[test]
    fn alpha_bounds_are_clamped() {
        assert_eq!(EmaFilter2D::with_alpha(2.0).alpha(), 1.0);
        assert_eq!(EmaFilter2D::with_alpha(-1.0).alpha(), 0.0);
        // alpha = 1 means no smoothing at all.
        let mut passthrough = EmaFilter2D::with_alpha(1.0);
        passthrough.filter(10.0, 10.0);
        assert_eq!(passthrough.filter(90.0, 20.0), (90.0, 20.0));
    }
}
