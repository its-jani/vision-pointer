//! Adaptive smoothing for the cursor.
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
//!
//! A fixed `alpha` forces a compromise: the value that kills tremor while hovering is the value
//! that drags the cursor a dozen pixels behind a fast sweep. The [`OneEuroFilter2D`] in this module
//! removes the compromise by measuring signal velocity (the rate of change of the samples, in
//! units per second) and raising `alpha` with it:
//!
//! ```text
//! cutoff = min_cutoff + beta * |velocity|
//! ```
//!
//! Hovering measures near zero velocity, so the cutoff stays at `min_cutoff` and the cursor is
//! glassy; a sweep across the display pushes the cutoff up and the filter almost disappears.
//! See [Casiez et al., 2012](https://cristal.univ-lille.fr/~guiron/pub/2012%20one%20euro.pdf).

#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

/// Weight of the newest sample. 0.25 removes most fingertip jitter at ~30 FPS without the cursor
/// feeling detached from the hand.
pub const DEFAULT_ALPHA: f32 = 0.25;
/// Cutoff (Hz) of the position filter at zero velocity. 1.0 Hz attenuates the few-pixel tremor a
/// stationary fingertip measures, while a deliberate move at >1 Hz comes through.
pub const DEFAULT_MIN_CUTOFF: f32 = 1.0;
/// Cutoff (Hz) added per screen-pixel-per-second of hand movement. The constant is from the 1-Euro
/// paper's mouse demonstration, whose units match a display-mapped cursor: a slow 100 px/s drift
/// raises the cutoff to 1.7 Hz, a 1000 px/s sweep to 8 Hz - nearly passthrough.
pub const DEFAULT_BETA: f32 = 0.007;
/// Cutoff (Hz) of the velocity estimator itself. The raw derivative amplifies high-frequency jitter
/// (differencing is a high-pass), so it gets its own low-pass at the same default as the position.
pub const DEFAULT_D_CUTOFF: f32 = 1.0;

/// Smooths a screen coordinate pair.
///
/// Each axis keeps only its own last average, so a hand that leaves the frame and comes back at a
/// different height does not drag the cursor diagonally through the old position.
/// Retained for unit-test regression; the production cursor path uses [`OneEuroFilter2D`].
#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub struct EmaFilter2D {
    prev_x: Option<f32>,
    prev_y: Option<f32>,
    alpha: f32,
}

#[allow(dead_code)]
impl Default for EmaFilter2D {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl EmaFilter2D {
    pub fn new() -> Self {
        Self::with_alpha(DEFAULT_ALPHA)
    }

    /// `alpha` outside 0..=1 is clamped: 0 freezes the cursor on the first sample, 1 disables
    /// smoothing entirely, and neither should be reachable by accident.
    pub fn with_alpha(alpha: f32) -> Self {
        Self {
            prev_x: None,
            prev_y: None,
            alpha: alpha.clamp(0.0, 1.0),
        }
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

/// One axis of exponential smoothing plus the derivative it feeds the velocity estimate.
#[derive(Copy, Clone, Debug)]
struct ExponentialSmoother {
    previous: Option<f32>,
    hat_previous: Option<f32>,
}

impl ExponentialSmoother {
    const fn new() -> Self {
        Self {
            previous: None,
            hat_previous: None,
        }
    }

    /// Low-pass `value` with `alpha` derived from `cutoff` and `dt`.
    fn filter(&mut self, value: f32, alpha: f32) -> f32 {
        let smoothed = match self.hat_previous {
            Some(previous) => alpha * value + (1.0 - alpha) * previous,
            None => value,
        };
        self.hat_previous = Some(smoothed);
        self.previous = Some(value);
        smoothed
    }

    fn reset(&mut self) {
        self.previous = None;
        self.hat_previous = None;
    }
}

/// Smooths a screen coordinate pair with a cutoff that follows the hand's speed.
///
/// Each axis runs an independent pair of smoothers (position and its derivative), which is what
/// lets a reset filter snap to a re-acquired hand instead of sliding through the old position.
#[derive(Copy, Clone, Debug)]
pub struct OneEuroFilter2D {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x: ExponentialSmoother,
    y: ExponentialSmoother,
    dx: ExponentialSmoother,
    dy: ExponentialSmoother,
    last_time: Option<Instant>,
}

impl Default for OneEuroFilter2D {
    fn default() -> Self {
        Self::new()
    }
}

impl OneEuroFilter2D {
    pub fn new() -> Self {
        Self::with_parameters(DEFAULT_MIN_CUTOFF, DEFAULT_BETA, DEFAULT_D_CUTOFF)
    }

    /// Negative cutoffs would invert the filter and a negative beta rewards jitter, so both are
    /// clamped to a non-negative floor instead of being rejected at the call site.
    pub fn with_parameters(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff: min_cutoff.max(0.0),
            beta: beta.max(0.0),
            d_cutoff: d_cutoff.max(0.0),
            x: ExponentialSmoother::new(),
            y: ExponentialSmoother::new(),
            dx: ExponentialSmoother::new(),
            dy: ExponentialSmoother::new(),
            last_time: None,
        }
    }

    /// EMA weight the adaptive cutoff corresponds to at `fps` frames per second and `speed` pixels
    /// per second. Exposed so callers can report the effective smoothing in familiar `alpha`
    /// terms and tests can check the cutoff transition without timing.
    #[allow(dead_code)]
    pub fn alpha_at(&self, fps: f32, speed: f32) -> f32 {
        let cutoff = self.min_cutoff + self.beta * speed.abs();
        alpha(cutoff, 1.0 / fps.max(f32::MIN_POSITIVE))
    }

    /// The base cutoff frequency in Hz (at zero velocity).
    pub fn min_cutoff(&self) -> f32 {
        self.min_cutoff
    }

    /// Smooths one frame's coordinate, stamped with the instant it was measured. The first call
    /// after construction or [`reset`][Self::reset] returns the sample unchanged.
    ///
    /// The timestamp rather than an assumed frame period sets `dt`, because the pipeline's real
    /// spacing wobbles with inference cost - and a caller that has no clock can stamp samples
    /// exactly `1 / fps` apart instead.
    pub fn filter(&mut self, x: f32, y: f32, timestamp: Instant) -> (f32, f32) {
        let dt = match self.last_time {
            Some(previous) => timestamp.saturating_duration_since(previous).as_secs_f32(),
            None => 0.0,
        };
        self.last_time = Some(timestamp);

        // A duplicate timestamp cannot yield a finite derivative; keep the previous average and
        // the previous clock instead of dividing by zero.
        if dt <= 0.0 && self.x.previous.is_some() {
            return (self.x.hat_previous.unwrap(), self.y.hat_previous.unwrap());
        }

        // Velocity is the derivative of the *raw* signal - differencing amplifies high-frequency
        // jitter, so the estimate is low-passed on its own before it raises the cutoff.
        let (vx, vy) = match (self.x.previous, self.y.previous) {
            (Some(px), Some(py)) => ((x - px) / dt, (y - py) / dt),
            _ => (0.0, 0.0),
        };
        let d_alpha = alpha(self.d_cutoff, dt);
        let vx = self.dx.filter(vx, d_alpha);
        let vy = self.dy.filter(vy, d_alpha);
        let speed = (vx * vx + vy * vy).sqrt();

        let p_alpha = alpha(self.min_cutoff + self.beta * speed, dt);
        (self.x.filter(x, p_alpha), self.y.filter(y, p_alpha))
    }

    /// Forgets the history, for the frame after a tracking dropout.
    pub fn reset(&mut self) {
        for smoother in [&mut self.x, &mut self.y, &mut self.dx, &mut self.dy] {
            smoother.reset();
        }
        self.last_time = None;
    }
}

/// Smoothing weight for a `cutoff`-Hz low-pass sampled every `dt` seconds.
///
/// The classic discretisation: `tau = 1 / (2 pi cutoff)`, `alpha = 1 / (1 + tau / dt)`. A cutoff at
/// or above the Nyquist rate of the sampling approaches alpha = 1 (passthrough); a low cutoff
/// approaches 0 and freezes the average.
fn alpha(cutoff: f32, dt: f32) -> f32 {
    if dt <= 0.0 {
        return 1.0;
    }
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff.max(0.0));
    (1.0 / (1.0 + tau / dt)).clamp(0.0, 1.0)
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
        assert_eq!(
            filter.filter(100.0, 100.0),
            (100.0, 100.0),
            "no slide from the stale position"
        );
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

    /// Stamps for a hypothetical `fps` that never sleeps: sample `n` sits at `n / fps` seconds.
    fn clock(fps: f32) -> impl FnMut(u32) -> Instant {
        let origin = Instant::now();
        let step = Duration::from_secs_f32(1.0 / fps);
        move |n: u32| origin + step * n
    }

    #[test]
    fn the_first_one_euro_sample_passes_through_untouched() {
        let mut filter = OneEuroFilter2D::new();
        let mut stamp = clock(30.0);
        assert_eq!(filter.filter(500.0, 300.0, stamp(0)), (500.0, 300.0));
    }

    #[test]
    fn a_one_euro_reset_reseeds_on_the_next_sample() {
        let mut filter = OneEuroFilter2D::new();
        let mut stamp = clock(30.0);
        filter.filter(500.0, 300.0, stamp(0));
        filter.filter(504.0, 296.0, stamp(1));
        filter.reset();
        assert_eq!(
            filter.filter(100.0, 100.0, stamp(2)),
            (100.0, 100.0),
            "no slide from the stale position"
        );
    }

    /// Steady-state lag (px) of a filter following a horizontal sweep at `px_per_frame` at 30 FPS.
    /// A first-order low-pass trails a constant-velocity signal by roughly `v * (1 - a) / a`, so
    /// lag at a given speed is a direct read of the cutoff the filter settled on.
    fn sweep_lag(mut filter: OneEuroFilter2D, px_per_frame: f32, frames: u32) -> f32 {
        let mut stamp = clock(30.0);
        let mut x = 0.0f32;
        for n in 0..frames {
            x = filter
                .filter(100.0 + px_per_frame * n as f32, 400.0, stamp(n))
                .0;
        }
        let hand = 100.0 + px_per_frame * (frames - 1) as f32;
        (hand - x).abs()
    }

    #[test]
    fn hovering_is_filtered_far_harder_than_a_sweep() {
        // The cutoff transition, against the fixed-alpha baseline. A slow 300 px/s drift trails
        // the hand by `v * (1 - a) / a` = 30 px with the old always-glassy alpha of 0.25; the
        // 1-Euro filter settles on a higher effective cutoff for the same motion and trails by
        // about half that. A fast 2700 px/s sweep still trails a few pixels - a first-order
        // filter always does - but its *relative* lag is what the user perceives as drag.
        let slow_lag = sweep_lag(OneEuroFilter2D::new(), 10.0, 30);
        let fast_lag = sweep_lag(OneEuroFilter2D::new(), 90.0, 14);
        assert!(
            (slow_lag - 15.5).abs() < 8.0,
            "300 px/s drift should trail ~16 px, lag = {slow_lag}"
        );
        assert!(
            (fast_lag - 23.6).abs() < 15.0,
            "2700 px/s sweep should trail tens of px, lag = {fast_lag}"
        );
        let baseline_slow = 10.0 * (1.0 - DEFAULT_ALPHA) / DEFAULT_ALPHA;
        assert!(
            slow_lag < baseline_slow,
            "the adaptive cutoff must beat fixed alpha at low speed: {slow_lag} vs {baseline_slow:.1}"
        );
        assert!(
            fast_lag / 2700.0 < slow_lag / 300.0,
            "lag per unit speed must fall as speed rises: {:.4} vs {:.4}",
            fast_lag / 2700.0,
            slow_lag / 300.0
        );
    }

    #[test]
    fn the_adaptive_cutoff_raises_alpha_with_speed() {
        let filter = OneEuroFilter2D::new();
        let slow = filter.alpha_at(30.0, 0.0);
        let fast = filter.alpha_at(30.0, 2700.0);
        assert!(
            slow < 0.2,
            "1 Hz cutoff at 30 FPS is heavy smoothing, alpha = {slow}"
        );
        assert!(
            fast > 0.75,
            "20 Hz cutoff at 30 FPS is close to passthrough, alpha = {fast}"
        );
        assert!(fast > slow, "cutoff must rise with velocity");
    }

    #[test]
    fn micro_tremor_around_a_stationary_target_is_attenuated() {
        let mut filter = OneEuroFilter2D::new();
        let mut stamp = clock(30.0);
        for n in 0..30 {
            // +/- 4 px alternating noise at a still target, the pattern the landmark model makes.
            let jitter = if n % 2 == 0 { 4.0 } else { -4.0 };
            filter.filter(400.0 + jitter, 400.0 - jitter, stamp(n));
        }
        let (x, y) = filter.filter(404.0, 396.0, stamp(30));
        assert!((x - 400.0).abs() < 3.0, "x = {x}");
        assert!((y - 400.0).abs() < 3.0, "y = {y}");
    }

    #[test]
    fn a_zero_beta_freezes_the_cutoff_at_the_minimum() {
        // beta = 0 makes speed irrelevant: the filter degrades to a fixed low-pass, which is the
        // knob for someone who wants the old always-glassy behaviour back.
        let filter = OneEuroFilter2D::with_parameters(1.0, 0.0, 1.0);
        assert_eq!(filter.alpha_at(30.0, 0.0), filter.alpha_at(30.0, 5000.0));
    }

    #[test]
    fn one_euro_parameters_are_clamped_to_a_usable_floor() {
        let filter = OneEuroFilter2D::with_parameters(-5.0, -1.0, -2.0);
        // Every parameter floored at 0: alpha_at must stay finite and inside 0..=1.
        let alpha = filter.alpha_at(30.0, 100.0);
        assert!((0.0..=1.0).contains(&alpha), "alpha = {alpha}");
    }
}
