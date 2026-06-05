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

use crate::distribution::Continuous;

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
        // ln p_Y(y) = ln p_X(g⁻¹(y)) + ln |d g⁻¹/dy|.
        let x = self.bijector.inverse(y);
        self.base.ln_pdf(x) + self.bijector.inverse_log_det_jacobian(y)
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
        // Transformed(Normal(μ,σ), Exp) ≡ LogNormal(μ,σ) — density matches.
        let t = Transformed::new(Normal::new(0.5, 1.2).unwrap(), Exp);
        let ln = LogNormal::new(0.5, 1.2).unwrap();
        for &y in &[0.1, 0.5, 1.0, 2.0, 5.0] {
            close(t.pdf(y), ln.pdf(y), 1e-10);
        }
    }

    #[test]
    fn affine_shifts_and_scales_a_normal() {
        // Transformed(Normal(0,1), 2·x + 3) ≡ Normal(3, 2).
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
        // exp(normal) is always positive.
        assert!((0..1000).all(|_| t.sample(&mut rng) > 0.0));
    }
}
