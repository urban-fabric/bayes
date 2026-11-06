//! The non-central t distribution.

use crate::{Continuous, ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use euclid_numeric::Real;

/// Terms below this stop contributing to Lenth's series.
const TOLERANCE: f64 = 1e-15;
const MAX_TERMS: usize = 2_000;

/// The **non-central t** with `freedom > 0` and non-centrality `delta`.
///
/// The distribution of `(Z + δ) / √(V/ν)` for a standard normal `Z` and an
/// independent `V ~ χ²_ν`. This is the one a t-test power calculation actually
/// needs: under the alternative the t-statistic is non-central with
/// `δ = effect · √n`, so the exact power is a tail of this distribution rather
/// than the normal approximation usually substituted for it.
///
/// Unlike χ² and F, this is **not** a Poisson mixture. The CDF is Lenth's (1989)
/// series over regularized incomplete betas, and the density comes from the
/// exact recurrence
///
/// ```text
///     f_{ν,δ}(x) = (ν/x)·[ F_{ν+2,δ}(x·√(1 + 2/ν)) − F_{ν,δ}(x) ]
/// ```
///
/// which is a genuine identity rather than a numerical derivative — the
/// difference matters, because a numerical derivative shipped as a density
/// inherits a step size that nobody chose.
///
/// At `delta = 0` it is exactly [`crate::StudentT`].
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{ContinuousCdf, NoncentralT, StudentT};
///
/// // The exact power of a one-sample t-test. Under the alternative the
/// // statistic is non-central with delta = effect * sqrt(n), so the power is a
/// // tail of this distribution rather than of a normal approximation to it.
/// let n = 25.0_f64;
/// let delta = 0.6 * n.sqrt();
/// let alternative = NoncentralT::new(n - 1.0, delta)?;
/// let critical = StudentT::new(n - 1.0)?.inverse_cdf(0.975_f64);
/// let power = (1.0 - alternative.cdf(critical)) + alternative.cdf(-critical);
/// assert!((power - 0.820_722).abs() < 1e-4);
///
/// // At delta = 0 it is the central t, so the same expression returns alpha.
/// let null = NoncentralT::new(n - 1.0, 0.0)?;
/// let size = (1.0 - null.cdf(critical)) + null.cdf(-critical);
/// assert!((size - 0.05).abs() < 1e-9);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoncentralT {
    freedom: f64,
    delta: f64,
}

impl NoncentralT {
    pub fn new(freedom: f64, delta: f64) -> Result<Self, BayesError> {
        validate::positive(&[("freedom", freedom)])?;
        validate::finite(&[("delta", delta)])?;
        Ok(Self { freedom, delta })
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// `P(T <= t)` for a non-negative `t`, by Lenth's series.
    fn cdf_nonnegative(freedom: f64, delta: f64, t: f64) -> f64 {
        // `t²` overflows to infinity above about 1e154, which would make `y` a
        // NaN rather than the 1 it is converging to. Above the point where `y`
        // is 1 to within an ulp the series has nothing left to say, so answer
        // from the limit instead.
        if t * t + freedom == f64::INFINITY {
            return 1.0;
        }
        let y = t * t / (t * t + freedom);
        let half = freedom / 2.0;
        let normal_floor = standard_normal_cdf(-delta);
        if y <= 0.0 {
            return normal_floor;
        }
        let half_delta_sq = delta * delta / 2.0;
        let exp_term = (-half_delta_sq).exp();

        // p_j = e^{-δ²/2}(δ²/2)^j / j!   (even-order terms)
        // q_j = e^{-δ²/2} δ (δ²/2)^j / (√2 Γ(j + 3/2))   (odd-order terms)
        let mut p = exp_term;
        let mut q = exp_term * delta / (2.0_f64.sqrt() * 1.5_f64.ln_gamma().exp());
        let mut total = 0.0;
        for j in 0..MAX_TERMS {
            let jf = j as f64;
            let even = p * y.beta_reg(jf + 0.5, half);
            let odd = q * y.beta_reg(jf + 1.0, half);
            total += even + odd;
            if j > 0 && (even + odd).abs() < TOLERANCE * total.abs().max(1e-300) {
                break;
            }
            // Advance both coefficient families by their recurrences, which is
            // what keeps the gamma functions out of the loop.
            p *= half_delta_sq / (jf + 1.0);
            q *= half_delta_sq / (jf + 1.5);
        }
        (normal_floor + 0.5 * total).clamp(0.0, 1.0)
    }
}

/// `Φ(z)`, from the complementary error function.
fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (-z * std::f64::consts::FRAC_1_SQRT_2).erfc()
}

impl Support<f64> for NoncentralT {
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf<f64> for NoncentralT {
    fn cdf(&self, t: f64) -> f64 {
        if t.is_nan() {
            return f64::NAN;
        }
        if t == f64::NEG_INFINITY {
            return 0.0;
        }
        if t == f64::INFINITY {
            return 1.0;
        }
        if t >= 0.0 {
            Self::cdf_nonnegative(self.freedom, self.delta, t)
        } else {
            // The reflection: negating both the argument and the non-centrality
            // maps the lower tail onto the upper one, where the series converges.
            1.0 - Self::cdf_nonnegative(self.freedom, -self.delta, -t)
        }
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        bayes_kernels::inverse_cdf_bisect(|x| self.cdf(x), f64::NEG_INFINITY, f64::INFINITY, p)
    }
}

impl Continuous<f64> for NoncentralT {
    fn pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if !x.is_finite() {
            return 0.0;
        }
        let nu = self.freedom;
        if x == 0.0 {
            // The closed form at the origin, where the recurrence divides by x.
            let ln_f = ((nu + 1.0) / 2.0).ln_gamma()
                - 0.5 * (std::f64::consts::PI * nu).ln()
                - (nu / 2.0).ln_gamma()
                - self.delta * self.delta / 2.0;
            return ln_f.exp();
        }
        // f(x) = (ν/x)·[F_{ν+2,δ}(x√(1+2/ν)) − F_{ν,δ}(x)]
        let shifted = match Self::new(nu + 2.0, self.delta) {
            Ok(d) => d,
            Err(_) => return f64::NAN,
        };
        let scaled = x * (1.0 + 2.0 / nu).sqrt();
        let density = (nu / x) * (shifted.cdf(scaled) - self.cdf(x));
        density.max(0.0)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
}

impl Moments<f64> for NoncentralT {
    /// `δ·√(ν/2)·Γ((ν−1)/2)/Γ(ν/2)`, undefined for `ν <= 1`.
    fn mean(&self) -> f64 {
        if self.freedom <= 1.0 {
            return f64::NAN;
        }
        let nu = self.freedom;
        self.delta
            * (nu / 2.0).sqrt()
            * (((nu - 1.0) / 2.0).ln_gamma() - (nu / 2.0).ln_gamma()).exp()
    }
    /// `ν(1 + δ²)/(ν − 2) − mean²`, undefined for `ν <= 2`.
    fn variance(&self) -> f64 {
        if self.freedom <= 2.0 {
            return f64::NAN;
        }
        let nu = self.freedom;
        let m = self.mean();
        nu * (1.0 + self.delta * self.delta) / (nu - 2.0) - m * m
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
