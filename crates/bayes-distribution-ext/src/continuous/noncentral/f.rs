//! The non-central F distribution.

use super::series::poisson_mixture;
use crate::{Continuous, ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use euclid_numeric::Real;

/// The **non-central F** with numerator freedom `d1`, denominator freedom `d2`,
/// and non-centrality `lambda >= 0`.
///
/// The ratio `(X/d1)/(Y/d2)` for `X ~ χ'²(d1, λ)` and an independent central
/// `Y ~ χ²(d2)`. This is the distribution an ANOVA F-statistic follows under the
/// alternative, so it is what turns "how many groups and how many per group"
/// into a power number.
///
/// At `lambda = 0` it is exactly [`crate::FisherSnedecor`].
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{ContinuousCdf, FisherSnedecor, NoncentralF};
///
/// // What an ANOVA F statistic follows when the null is false — which is what
/// // turns a design into a power number.
/// let central = FisherSnedecor::new(4.0, 20.0)?;
/// let critical = central.inverse_cdf(0.95_f64);
///
/// // Five groups of five, at a non-centrality of 6: a 38% chance of detection.
/// let alternative = NoncentralF::new(4.0, 20.0, 6.0)?;
/// assert!((1.0 - alternative.cdf(critical) - 0.379_301).abs() < 1e-4);
///
/// // With no effect at all the same calculation returns the test's own size.
/// let null = NoncentralF::new(4.0, 20.0, 0.0)?;
/// assert!((1.0 - null.cdf(critical) - 0.05).abs() < 1e-9);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoncentralF {
    d1: f64,
    d2: f64,
    lambda: f64,
}

impl NoncentralF {
    pub fn new(d1: f64, d2: f64, lambda: f64) -> Result<Self, BayesError> {
        validate::positive(&[("d1", d1), ("d2", d2)])?;
        validate::non_negative(&[("lambda", lambda)])?;
        Ok(Self { d1, d2, lambda })
    }

    pub fn d1(&self) -> f64 {
        self.d1
    }
    pub fn d2(&self) -> f64 {
        self.d2
    }
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Support<f64> for NoncentralF {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf<f64> for NoncentralF {
    /// `Σ_j w_j · I_y(d1/2 + j, d2/2)` with `y = d1·x / (d1·x + d2)`.
    ///
    /// The complement `1 − y = d2/(d1·x + d2)` is formed directly and the
    /// reflection `I_y(a,b) = 1 − I_{1−y}(b,a)` used above the midpoint, for the
    /// same reason the central F does it: `y` carries 1 only to within an ulp, so
    /// feeding a `y` near 1 straight to `beta_reg` loses the whole upper tail.
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return if x.is_nan() { f64::NAN } else { 0.0 };
        }
        if x.is_infinite() {
            return 1.0;
        }
        let d1x = self.d1 * x;
        let denom = d1x + self.d2;
        let y = d1x / denom;
        let y_complement = self.d2 / denom;
        let b = self.d2 / 2.0;
        poisson_mixture(self.lambda, |j| {
            let a = self.d1 / 2.0 + j as f64;
            if y <= 0.5 {
                y.beta_reg(a, b)
            } else {
                1.0 - y_complement.beta_reg(b, a)
            }
        })
        .clamp(0.0, 1.0)
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        bayes_kernels::inverse_cdf_bisect(|x| self.cdf(x), 0.0, f64::INFINITY, p)
    }
}

impl Continuous<f64> for NoncentralF {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 || x.is_nan() {
            return if x.is_nan() { f64::NAN } else { 0.0 };
        }
        if x.is_infinite() {
            return 0.0;
        }
        let (d1, d2) = (self.d1, self.d2);
        if x == 0.0 {
            return if d1 < 2.0 {
                f64::INFINITY
            } else if d1 == 2.0 {
                (-self.lambda / 2.0).exp()
            } else {
                0.0
            };
        }
        poisson_mixture(self.lambda, |j| {
            let a = d1 / 2.0 + j as f64;
            let b = d2 / 2.0;
            // The beta-prime density in the log domain.
            let ln_f = a * (d1 / d2).ln() + (a - 1.0) * x.ln()
                - (a + b) * (1.0 + d1 * x / d2).ln()
                - a.ln_beta(b);
            ln_f.exp()
        })
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
}

impl Moments<f64> for NoncentralF {
    /// `d2(d1 + λ) / (d1(d2 − 2))`, undefined for `d2 <= 2`.
    fn mean(&self) -> f64 {
        if self.d2 <= 2.0 {
            return f64::NAN;
        }
        self.d2 * (self.d1 + self.lambda) / (self.d1 * (self.d2 - 2.0))
    }
    /// `2(d2/d1)²·[(d1+λ)² + (d1+2λ)(d2−2)] / [(d2−2)²(d2−4)]`, undefined for
    /// `d2 <= 4`.
    fn variance(&self) -> f64 {
        if self.d2 <= 4.0 {
            return f64::NAN;
        }
        let (d1, d2, l) = (self.d1, self.d2, self.lambda);
        let ratio = d2 / d1;
        2.0 * ratio * ratio * ((d1 + l) * (d1 + l) + (d1 + 2.0 * l) * (d2 - 2.0))
            / ((d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0))
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    fn skewness(&self) -> f64 {
        f64::NAN
    }
    fn kurtosis(&self) -> f64 {
        f64::NAN
    }
    fn entropy(&self) -> f64 {
        f64::NAN
    }
    fn median(&self) -> f64 {
        self.inverse_cdf(0.5)
    }
    fn mode(&self) -> f64 {
        f64::NAN
    }
}
