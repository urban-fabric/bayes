//! Rejection sampling: exact draws from any density you can evaluate.
//!
//! The method is the oldest trick in the book and still the right answer when a
//! density has no invertible CDF. Put a scaled *envelope* over the target, draw
//! from the envelope, and keep the point with probability equal to the ratio of
//! the two heights. What survives is distributed **exactly** as the target — not
//! approximately, not asymptotically. There is no burn-in, no autocorrelation,
//! and no convergence to check, which is what distinguishes this from MCMC.
//!
//! The price is paid in wasted proposals, and the whole craft is in the
//! envelope. Two rules worth internalizing:
//!
//! * **The bound must actually bound.** If `target(x) > bound · proposal(x)`
//!   anywhere, the sampler silently draws from the wrong distribution — the
//!   region where the envelope dips below the target is under-represented and
//!   nothing in the output looks wrong. [`RejectionSampler`] therefore tracks
//!   whether the ratio ever exceeded 1 and will tell you.
//! * **Acceptance decays with mismatch.** The expected acceptance rate is
//!   `1/bound` when both are normalized, so an envelope five times too tall
//!   throws away four of every five proposals. In more than a couple of
//!   dimensions that decay is exponential, which is why rejection sampling is a
//!   univariate tool and MCMC exists at all.
//!
//! Work in the **log domain** where you can: `ln_target` and `ln_envelope`
//! rather than their exponentials. A density that underflows to zero still has a
//! perfectly finite logarithm, and the acceptance test is a comparison of logs
//! either way.

use bayes_core::BayesError;
use rand::{Rng, RngExt};

/// Outcome of a draw attempt, including how much work it took.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Draw {
    /// The accepted value.
    pub value: f64,
    /// How many proposals were made, including the accepted one.
    pub proposals: usize,
}

/// Exact rejection sampling from a log-density, given a log-envelope that
/// dominates it.
///
/// `ln_target` need not be normalized — rejection sampling never needs the
/// normalizing constant, which is the other reason it is useful for posteriors.
pub struct RejectionSampler<T, P, S> {
    ln_target: T,
    ln_envelope: P,
    propose: S,
    /// `ln` of the factor by which the envelope is scaled up.
    ln_bound: f64,
    /// The largest `ln_target - ln_envelope - ln_bound` seen. Above zero means
    /// the envelope failed and the draws are not from the target.
    worst_excess: f64,
    accepted: usize,
    proposed: usize,
}

impl<T, P, S> RejectionSampler<T, P, S>
where
    T: Fn(f64) -> f64,
    P: Fn(f64) -> f64,
    S: FnMut(&mut dyn RngCoreShim) -> f64,
{
    /// Construct a sampler. `bound` must satisfy
    /// `target(x) <= bound * envelope(x)` for every `x` in the support.
    pub fn new(ln_target: T, ln_envelope: P, propose: S, bound: f64) -> Result<Self, BayesError> {
        if !bound.is_finite() || bound <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "bound",
                value: bound,
            });
        }
        Ok(Self {
            ln_target,
            ln_envelope,
            propose,
            ln_bound: bound.ln(),
            worst_excess: f64::NEG_INFINITY,
            accepted: 0,
            proposed: 0,
        })
    }

    /// Draw one value, retrying until a proposal is accepted or `max_proposals`
    /// is exhausted.
    ///
    /// The cap is not optional paranoia: a badly chosen envelope can make
    /// acceptance arbitrarily rare, and an uncapped loop turns that into a hang
    /// rather than an error.
    pub fn draw<R: Rng + ?Sized>(&mut self, rng: &mut R, max_proposals: usize) -> Option<Draw> {
        let mut shim = RngShim { rng };
        for attempt in 1..=max_proposals {
            let x = (self.propose)(&mut shim);
            self.proposed += 1;
            let lt = (self.ln_target)(x);
            let le = (self.ln_envelope)(x);
            if !lt.is_finite() {
                // Zero target density here (or an invalid point): reject.
                continue;
            }
            // `excess > 0` means the envelope was breached at this point.
            let excess = lt - le - self.ln_bound;
            if excess > self.worst_excess {
                self.worst_excess = excess;
            }
            let u: f64 = shim.rng.random();
            if u.ln() <= excess {
                self.accepted += 1;
                return Some(Draw {
                    value: x,
                    proposals: attempt,
                });
            }
        }
        None
    }

    /// The observed acceptance rate, or `NaN` before the first proposal.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed == 0 {
            f64::NAN
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }

    /// Did the envelope ever fail to dominate the target?
    ///
    /// **Check this.** A breached envelope does not panic and does not look
    /// wrong in the output; it just biases the result in the region where the
    /// envelope dipped. A small positive tolerance absorbs rounding in the two
    /// log evaluations.
    pub fn envelope_was_breached(&self) -> bool {
        self.worst_excess > 1e-9
    }

    /// How far the envelope was breached, in log units. Zero or below is sound.
    pub fn worst_log_excess(&self) -> f64 {
        self.worst_excess
    }
}

/// Object-safe shim so a proposal closure can draw without being generic over
/// the RNG, which would make the sampler impossible to store.
pub trait RngCoreShim {
    /// A uniform on `[0, 1)`.
    fn uniform(&mut self) -> f64;
}

struct RngShim<'a, R: Rng + ?Sized> {
    rng: &'a mut R,
}

impl<R: Rng + ?Sized> RngCoreShim for RngShim<'_, R> {
    fn uniform(&mut self) -> f64 {
        self.rng.random()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// A truncated standard normal on `[-1, 2]`, sampled under a uniform
    /// envelope. The target is unnormalized, which rejection sampling does not
    /// mind.
    #[test]
    fn a_uniform_envelope_draws_a_truncated_normal() {
        let (lo, hi) = (-1.0_f64, 2.0_f64);
        // ln of exp(-x^2/2) with no normalizer; peak is 0 at x = 0.
        let ln_target = |x: f64| {
            if (lo..=hi).contains(&x) {
                -0.5 * x * x
            } else {
                f64::NEG_INFINITY
            }
        };
        let ln_envelope = |_x: f64| -(hi - lo).ln(); // uniform density on [lo, hi]
        // target <= 1 and envelope = 1/3, so bound = 3 suffices.
        let mut s = RejectionSampler::new(
            ln_target,
            ln_envelope,
            move |r: &mut dyn RngCoreShim| lo + (hi - lo) * r.uniform(),
            hi - lo,
        )
        .unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(20_260_105);
        let draws: Vec<f64> = (0..20_000)
            .map(|_| s.draw(&mut rng, 1_000).expect("acceptance is likely").value)
            .collect();

        assert!(
            !s.envelope_was_breached(),
            "excess {}",
            s.worst_log_excess()
        );
        assert!(draws.iter().all(|&x| (lo..=hi).contains(&x)));

        // Compare against the truncated normal's analytic mean. For a standard
        // normal truncated to [a,b] the mean is (phi(a) - phi(b)) / (Phi(b) - Phi(a)).
        let phi = |x: f64| (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let big_phi = |x: f64| 0.5 * (1.0 + erf(x / 2.0_f64.sqrt()));
        let expected = (phi(lo) - phi(hi)) / (big_phi(hi) - big_phi(lo));
        let got = draws.iter().sum::<f64>() / draws.len() as f64;
        assert!(
            (got - expected).abs() < 0.02,
            "sample mean {got} vs analytic {expected}"
        );
        // The acceptance rate should be near the mass ratio, not near zero.
        assert!(
            s.acceptance_rate() > 0.3,
            "acceptance {} is suspiciously low",
            s.acceptance_rate()
        );
    }

    /// A deliberately too-short envelope must be *reported*, not silently
    /// tolerated. This is the failure mode the flag exists for.
    #[test]
    fn a_breached_envelope_is_detected() {
        let ln_target = |x: f64| {
            if (0.0..=1.0).contains(&x) {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        };
        let ln_envelope = |_x: f64| 0.0;
        // bound = 0.5 is too small: target/envelope reaches 1, not 0.5.
        let mut s = RejectionSampler::new(
            ln_target,
            ln_envelope,
            |r: &mut dyn RngCoreShim| r.uniform(),
            0.5,
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..100 {
            let _ = s.draw(&mut rng, 10);
        }
        assert!(
            s.envelope_was_breached(),
            "a short envelope went unreported"
        );
        assert!(s.worst_log_excess() > 0.0);
    }

    #[test]
    fn an_impossible_target_gives_up_rather_than_hanging() {
        let mut s = RejectionSampler::new(
            |_x: f64| f64::NEG_INFINITY, // zero density everywhere
            |_x: f64| 0.0,
            |r: &mut dyn RngCoreShim| r.uniform(),
            1.0,
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        assert!(s.draw(&mut rng, 50).is_none());
        assert_eq!(s.acceptance_rate(), 0.0);
    }

    #[test]
    fn an_invalid_bound_is_rejected() {
        let t = |_x: f64| 0.0;
        let e = |_x: f64| 0.0;
        assert!(RejectionSampler::new(t, e, |r: &mut dyn RngCoreShim| r.uniform(), 0.0).is_err());
        assert!(RejectionSampler::new(t, e, |r: &mut dyn RngCoreShim| r.uniform(), -1.0).is_err());
        assert!(
            RejectionSampler::new(t, e, |r: &mut dyn RngCoreShim| r.uniform(), f64::NAN).is_err()
        );
    }

    /// Abramowitz-Stegun 7.1.26, good to ~1e-7 — ample for a test tolerance of
    /// 0.02 and avoids a dependency here.
    fn erf(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let y = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t
                * (-x * x).exp();
        sign * y
    }
}
