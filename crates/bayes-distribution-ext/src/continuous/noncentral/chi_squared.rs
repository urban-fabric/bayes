//! The non-central χ² distribution.

use super::series::{MixtureTerm, poisson_mixture};
use crate::{Continuous, ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use euclid_numeric::Real;

/// `P(χ²_{k+2j} ≤ x)` as `j` advances, by the incomplete-gamma recurrence
///
/// ```text
///     P(a + 1, z) = P(a, z) − z^a e^{−z} / Γ(a + 1)
/// ```
///
/// with `a = k/2 + j` and `z = x/2`. The subtracted bridge term has its own
/// one-multiply recurrence, so a walk of any length costs a single
/// `gamma_lr` — at `λ = 5000` that replaces about seven thousand of them.
struct LowerGamma {
    /// `x/2`.
    z: f64,
    /// `k/2`, the shape the index offsets from.
    k_half: f64,
    /// `a` at the index the state currently sits on.
    a: f64,
    /// `z^a e^{−z} / Γ(a + 1)`, the bridge between consecutive `a`.
    bridge: f64,
}

impl LowerGamma {
    /// `z^a e^{−z} / Γ(a+1)`, in the log domain so a large `a` cannot overflow
    /// the normalizer before the exponential pulls it back.
    fn bridge_at(&self, a: f64) -> f64 {
        (a * self.z.ln() - self.z - (a + 1.0).ln_gamma()).exp()
    }
}

impl MixtureTerm for LowerGamma {
    fn at(&mut self, j: usize) -> f64 {
        self.a = self.k_half + j as f64;
        self.bridge = self.bridge_at(self.a);
        self.z.gamma_lr(self.a)
    }

    fn up(&mut self, current: f64, _j: usize) -> f64 {
        // Going up subtracts, so clamp: `P` is decreasing in `a` and the series
        // is truncated by weight, not by this term, so an underflow to a hair
        // below zero must not become a negative contribution.
        let next = (current - self.bridge).max(0.0);
        self.bridge *= self.z / (self.a + 1.0);
        self.a += 1.0;
        next
    }

    fn down(&mut self, current: f64, _j: usize) -> f64 {
        // Going down adds, which cannot cancel.
        self.bridge *= self.a / self.z;
        self.a -= 1.0;
        (current + self.bridge).min(1.0)
    }
}

/// The central χ² density `x^{a−1} e^{−x/2} / (2^a Γ(a))` with `a = k/2 + j`.
///
/// Consecutive terms differ by the ratio `x / (2a)`, so unlike the CDF this
/// recurrence is purely multiplicative in both directions and cannot cancel.
struct CentralDensity {
    x: f64,
    k_half: f64,
    a: f64,
}

impl MixtureTerm for CentralDensity {
    fn at(&mut self, j: usize) -> f64 {
        self.a = self.k_half + j as f64;
        // In the log domain, for the same overflow reason as the bridge above.
        ((self.a - 1.0) * self.x.ln()
            - self.x / 2.0
            - self.a * std::f64::consts::LN_2
            - self.a.ln_gamma())
        .exp()
    }

    fn up(&mut self, current: f64, _j: usize) -> f64 {
        let next = current * self.x / (2.0 * self.a);
        self.a += 1.0;
        next
    }

    fn down(&mut self, current: f64, _j: usize) -> f64 {
        self.a -= 1.0;
        current * 2.0 * self.a / self.x
    }
}

/// The **non-central χ²** with `freedom > 0` and non-centrality `lambda >= 0`.
///
/// The distribution of `Σ (Z_i + μ_i)²` for independent standard normals, where
/// `lambda = Σ μ_i²`. It is what a chi-squared statistic follows when the null
/// is *false*, which makes it the distribution a power calculation needs.
///
/// At `lambda = 0` it is exactly [`crate::ChiSquared`], and the test suite
/// asserts that as an identity rather than a limit.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{ChiSquared, ContinuousCdf, Moments, NoncentralChiSquared};
///
/// // At zero non-centrality it *is* the central chi-squared, exactly.
/// let null = NoncentralChiSquared::new(3.0, 0.0)?;
/// assert!((null.cdf(2.5_f64) - ChiSquared::new(3.0)?.cdf(2.5_f64)).abs() < 1e-12);
///
/// // Non-centrality shifts the mean by exactly lambda, moving mass right and
/// // so lowering the CDF everywhere.
/// let alternative = NoncentralChiSquared::new(3.0, 5.0)?;
/// assert!((alternative.mean() - 8.0).abs() < 1e-12);
/// assert!(alternative.cdf(2.5_f64) < null.cdf(2.5_f64));
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoncentralChiSquared {
    freedom: f64,
    lambda: f64,
}

impl NoncentralChiSquared {
    pub fn new(freedom: f64, lambda: f64) -> Result<Self, BayesError> {
        validate::positive(&[("freedom", freedom)])?;
        validate::non_negative(&[("lambda", lambda)])?;
        Ok(Self { freedom, lambda })
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Support<f64> for NoncentralChiSquared {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf<f64> for NoncentralChiSquared {
    /// `Σ_j w_j · P(χ²_{k+2j} ≤ x)`, each central term a regularized lower
    /// incomplete gamma.
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return if x.is_nan() { f64::NAN } else { 0.0 };
        }
        if x.is_infinite() {
            return 1.0;
        }
        poisson_mixture(
            self.lambda,
            LowerGamma {
                z: x / 2.0,
                k_half: self.freedom / 2.0,
                a: 0.0,
                bridge: 0.0,
            },
        )
        .clamp(0.0, 1.0)
    }

    /// By bisection on the mixture CDF. There is no closed-form inverse, and the
    /// generic bisection searches an exponential coordinate, so it resolves the
    /// quantile to a relative precision the series itself cannot beat.
    fn inverse_cdf(&self, p: f64) -> f64 {
        bayes_kernels::inverse_cdf_bisect(|x| self.cdf(x), 0.0, f64::INFINITY, p)
    }
}

impl Continuous<f64> for NoncentralChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 || x.is_nan() {
            return if x.is_nan() { f64::NAN } else { 0.0 };
        }
        if x.is_infinite() {
            return 0.0;
        }
        if x == 0.0 {
            // The limit at the origin: finite only for freedom < 2, where the
            // central chi-squared density already diverges or is positive.
            return if self.freedom < 2.0 {
                f64::INFINITY
            } else if self.freedom == 2.0 {
                0.5 * (-self.lambda / 2.0).exp()
            } else {
                0.0
            };
        }
        poisson_mixture(
            self.lambda,
            CentralDensity {
                x,
                k_half: self.freedom / 2.0,
                a: 0.0,
            },
        )
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
}

impl Moments<f64> for NoncentralChiSquared {
    /// `k + λ`.
    fn mean(&self) -> f64 {
        self.freedom + self.lambda
    }
    /// `2(k + 2λ)`.
    fn variance(&self) -> f64 {
        2.0 * (self.freedom + 2.0 * self.lambda)
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// `2^{3/2}(k + 3λ) / (k + 2λ)^{3/2}`.
    fn skewness(&self) -> f64 {
        let d = self.freedom + 2.0 * self.lambda;
        2.0_f64.powf(1.5) * (self.freedom + 3.0 * self.lambda) / d.powf(1.5)
    }
    /// `12(k + 4λ) / (k + 2λ)²`.
    fn kurtosis(&self) -> f64 {
        let d = self.freedom + 2.0 * self.lambda;
        12.0 * (self.freedom + 4.0 * self.lambda) / (d * d)
    }
    /// No closed form; `NaN` by the trait's convention for an unavailable
    /// summary rather than a silently-bisected approximation.
    fn entropy(&self) -> f64 {
        f64::NAN
    }
    fn median(&self) -> f64 {
        self.inverse_cdf(0.5)
    }
    /// No closed form.
    fn mode(&self) -> f64 {
        f64::NAN
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// The recurrences against evaluating every central term from scratch.
    ///
    /// This is the only test that can see the difference: from the outside the
    /// two routes are the same function, and the stepped one is the only one
    /// fast enough to reach a large `λ`. So the slow route is kept here, where
    /// `λ` is small enough to afford it, and asserted term-for-term.
    ///
    /// The CDF is held to an absolute bound because it is a probability, and the
    /// density to a relative one because it ranges over many orders of
    /// magnitude. Both are far tighter than any wrong recurrence could slip
    /// through: a mis-stepped ratio is wrong by a factor, not by a part in
    /// `10^12`.
    #[test]
    fn the_recurrences_agree_with_evaluating_each_term_from_scratch() {
        for (k, lambda) in [
            (1.0, 0.5),
            (3.0, 2.0),
            (2.0, 20.0),
            (7.5, 60.0),
            (0.5, 150.0),
        ] {
            let (k_half, mean) = (k / 2.0, k + lambda);
            for i in 1..=40 {
                let x = mean * i as f64 / 20.0;
                let (z, ln_x) = (x / 2.0, x.ln());

                let direct_cdf = poisson_mixture(lambda, |j: usize| z.gamma_lr(k_half + j as f64));
                let stepped_cdf = poisson_mixture(
                    lambda,
                    LowerGamma {
                        z,
                        k_half,
                        a: 0.0,
                        bridge: 0.0,
                    },
                );
                assert!(
                    (direct_cdf - stepped_cdf).abs() < 1e-12,
                    "cdf({x}) for k={k}, lambda={lambda}: stepped {stepped_cdf} vs direct \
                     {direct_cdf}"
                );

                let direct_pdf = poisson_mixture(lambda, |j: usize| {
                    let a = k_half + j as f64;
                    ((a - 1.0) * ln_x - z - a * std::f64::consts::LN_2 - a.ln_gamma()).exp()
                });
                let stepped_pdf = poisson_mixture(lambda, CentralDensity { x, k_half, a: 0.0 });
                assert!(
                    (direct_pdf - stepped_pdf).abs() < 1e-12 * direct_pdf.max(f64::MIN_POSITIVE),
                    "pdf({x}) for k={k}, lambda={lambda}: stepped {stepped_pdf} vs direct \
                     {direct_pdf}"
                );
            }
        }
    }
}
