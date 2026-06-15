//! **Transformed distributions** — build new distributions by pushing a base
//! distribution through an invertible map (a [`Bijector`]). Sampling applies the
//! forward map; the density follows from the change-of-variables formula
//! `p_Y(y) = p_X(g⁻¹(y)) · |d g⁻¹/dy|`.
//!
//! This is the composition idea from TensorFlow Probability's bijectors: e.g.
//! `Transformed(Normal, Exp)` is the log-normal, `Transformed(d, Affine{..})`
//! shifts and scales any base distribution.

mod bijector;

pub use bijector::{Affine, Bijector, Exp, Sigmoid};

use rand::Rng;
use rand::distr::Distribution;

use crate::distribution::{Continuous, Moments};

/// A base distribution `D` pushed through a bijector `B`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transformed<D, B> {
    base: D,
    bijector: B,
}

impl<D, B> Transformed<D, B> {
    pub fn new(base: D, bijector: B) -> Self {
        Self { base, bijector }
    }

    pub fn base(&self) -> &D {
        &self.base
    }
    pub fn bijector(&self) -> &B {
        &self.bijector
    }
}

impl<D, B> Distribution<f64> for Transformed<D, B>
where
    D: Distribution<f64>,
    B: Bijector,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.bijector.forward(self.base.sample(rng))
    }
}

impl<D, B> Continuous for Transformed<D, B>
where
    D: Continuous,
    B: Bijector,
{
    fn pdf(&self, y: f64) -> f64 {
        self.ln_pdf(y).exp()
    }
    fn ln_pdf(&self, y: f64) -> f64 {
        let x = self.bijector.inverse(y);
        self.base.ln_pdf(x) + self.bijector.inverse_log_det_jacobian(y)
    }
}

/// Closed-form moments of an **affine-transformed** distribution `Y = aX + b`,
/// propagated exactly from the base by the linearity of expectation.
///
/// `E[Y] = a·E[X] + b` is `Moments.lean`'s `expectation_affine`
/// (`RiemannLean/Bayes/Moments.lean`); `Var[Y] = a²·Var[X]` is its corollary
/// applied to `(X − E[X])²`. The shape moments are scale/shift-equivariant:
/// skewness flips sign with `a` (odd), excess kurtosis is invariant (even),
/// the median and mode map through the monotone affine map, and the differential
/// entropy gains `ln|a|`.
impl<D: Moments> Moments for Transformed<D, Affine> {
    fn mean(&self) -> Option<f64> {
        Some(self.bijector.scale * self.base.mean()? + self.bijector.shift)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.bijector.scale.powi(2) * self.base.variance()?)
    }
    fn skewness(&self) -> Option<f64> {
        Some(self.bijector.scale.signum() * self.base.skewness()?)
    }
    fn kurtosis(&self) -> Option<f64> {
        self.base.kurtosis()
    }
    /// Differential entropy `H[Y] = H[X] + ln|a|`.
    fn entropy(&self) -> Option<f64> {
        Some(self.base.entropy()? + self.bijector.scale.abs().ln())
    }
    fn median(&self) -> Option<f64> {
        Some(self.bijector.scale * self.base.median()? + self.bijector.shift)
    }
    fn mode(&self) -> Option<f64> {
        Some(self.bijector.scale * self.base.mode()? + self.bijector.shift)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::{LogNormal, Normal};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn exp_of_normal_is_lognormal() {
        let t = Transformed::new(Normal::new(0.5, 1.2).unwrap(), Exp);
        let ln = LogNormal::new(0.5, 1.2).unwrap();
        for &y in &[0.1, 0.5, 1.0, 2.0, 5.0] {
            close(t.pdf(y), ln.pdf(y), 1e-10);
        }
    }

    #[test]
    fn affine_shifts_and_scales_a_normal() {
        let t = Transformed::new(
            Normal::new(0.0, 1.0).unwrap(),
            Affine::new(2.0, 3.0).unwrap(),
        );
        let target = Normal::new(3.0, 2.0).unwrap();
        for &y in &[-1.0, 1.0, 3.0, 5.0] {
            close(t.pdf(y), target.pdf(y), 1e-10);
        }
    }

    #[test]
    fn sampling_applies_the_forward_map() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let t = Transformed::new(Normal::new(0.0, 1.0).unwrap(), Exp);
        assert!((0..1000).all(|_| t.sample(&mut rng) > 0.0));
    }

    #[test]
    fn affine_moments_match_the_equivalent_normal() {
        use crate::distribution::Moments;
        // Y = 2·N(1, 1.5) + 3  ==  N(5, 3): moments propagate by expectation_affine.
        let t = Transformed::new(
            Normal::new(1.0, 1.5).unwrap(),
            Affine::new(2.0, 3.0).unwrap(),
        );
        let target = Normal::new(5.0, 3.0).unwrap();
        close(t.mean().unwrap(), target.mean().unwrap(), 1e-12);
        close(t.variance().unwrap(), target.variance().unwrap(), 1e-12);
        close(t.std_dev().unwrap(), target.std_dev().unwrap(), 1e-12);
        // Differential entropy gains ln|a|: H[N(1,1.5)] + ln 2 = H[N(5,3)].
        close(t.entropy().unwrap(), target.entropy().unwrap(), 1e-12);
        close(t.median().unwrap(), 5.0, 1e-12);
        close(t.mode().unwrap(), 5.0, 1e-12);
        // Normal is symmetric/mesokurtic, preserved under affine maps.
        close(t.skewness().unwrap(), 0.0, 1e-12);
        close(t.kurtosis().unwrap(), 0.0, 1e-12);
    }

    #[test]
    fn negative_scale_reflects_odd_moments_only() {
        use crate::distribution::{Exponential, Moments};
        // Reflect/shift an Exponential: Y = −X + 5. Mean shifts, variance is a²·Var,
        // skewness flips sign (odd), excess kurtosis is invariant (even).
        let base = Exponential::new(2.0).unwrap(); // mean 0.5, var 0.25, skew 2, kurt 6
        let t = Transformed::new(base, Affine::new(-1.0, 5.0).unwrap());
        close(t.mean().unwrap(), -base.mean().unwrap() + 5.0, 1e-12);
        close(t.variance().unwrap(), base.variance().unwrap(), 1e-12);
        close(t.skewness().unwrap(), -base.skewness().unwrap(), 1e-12);
        close(t.kurtosis().unwrap(), base.kurtosis().unwrap(), 1e-12);
    }
}
