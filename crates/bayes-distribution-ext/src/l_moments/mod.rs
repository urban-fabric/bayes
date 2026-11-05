//! **L-moments** — order-statistic moment analogs.
//!
//! L-moments are expectations of linear combinations of order statistics. They
//! exist whenever the mean does (so they describe heavy-tailed laws where the
//! ordinary [`skewness`](crate::Moments::skewness) / [`kurtosis`](crate::Moments::kurtosis)
//! diverge) and are far less sensitive to outliers:
//!
//! - `λ₁` — **L-location**, equal to the mean.
//! - `λ₂` — **L-scale**, a robust dispersion (`≥ 0`).
//! - `τ₃ = λ₃/λ₂` — **L-skewness**, in `[−1, 1]`.
//! - `τ₄ = λ₄/λ₂` — **L-kurtosis**.
//!
//! The implementations here are exact closed forms; they lean on the same
//! order-statistic structure already used by [`OrderStatistic`](crate::OrderStatistic).

use crate::{Exponential, Logistic, Normal, Uniform};

/// A family with closed-form L-moments.
pub trait LMoments {
    /// `λ₁`, the L-location (equal to the mean).
    fn l_location(&self) -> f64;
    /// `λ₂`, the L-scale (a nonnegative robust dispersion).
    fn l_scale(&self) -> f64;
    /// `τ₃ = λ₃/λ₂`, the L-skewness (in `[−1, 1]`).
    fn l_skewness(&self) -> f64;
    /// `τ₄ = λ₄/λ₂`, the L-kurtosis.
    fn l_kurtosis(&self) -> f64;
}

/// Uniform`(a, b)`: `λ₁=(a+b)/2`, `λ₂=(b−a)/6`, `τ₃=τ₄=0` (perfectly symmetric).
impl LMoments for Uniform {
    fn l_location(&self) -> f64 {
        0.5 * (self.lo() + self.hi())
    }
    fn l_scale(&self) -> f64 {
        (self.hi() - self.lo()) / 6.0
    }
    fn l_skewness(&self) -> f64 {
        0.0
    }
    fn l_kurtosis(&self) -> f64 {
        0.0
    }
}

/// Exponential(rate `λ`): `λ₁=1/λ`, `λ₂=1/(2λ)`, `τ₃=1/3`, `τ₄=1/6`.
impl LMoments for Exponential {
    fn l_location(&self) -> f64 {
        self.rate().recip()
    }
    fn l_scale(&self) -> f64 {
        0.5 * self.rate().recip()
    }
    fn l_skewness(&self) -> f64 {
        1.0 / 3.0
    }
    fn l_kurtosis(&self) -> f64 {
        1.0 / 6.0
    }
}

/// Normal(`μ, σ`): `λ₁=μ`, `λ₂=σ/√π`, `τ₃=0`, `τ₄=30·atan(√2)/π − 9 ≈ 0.1226`.
impl LMoments for Normal {
    fn l_location(&self) -> f64 {
        self.params().0
    }
    fn l_scale(&self) -> f64 {
        self.params().1 / std::f64::consts::PI.sqrt()
    }
    fn l_skewness(&self) -> f64 {
        0.0
    }
    fn l_kurtosis(&self) -> f64 {
        30.0 * std::f64::consts::SQRT_2.atan() / std::f64::consts::PI - 9.0
    }
}

/// Logistic(`μ, s`): `λ₁=μ`, `λ₂=s`, `τ₃=0`, `τ₄=1/6`.
impl LMoments for Logistic {
    fn l_location(&self) -> f64 {
        self.location()
    }
    fn l_scale(&self) -> f64 {
        self.scale()
    }
    fn l_skewness(&self) -> f64 {
        0.0
    }
    fn l_kurtosis(&self) -> f64 {
        1.0 / 6.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::Moments;
    use bayes_core_ext::assert_close;

    #[test]
    fn l_location_equals_the_mean() {
        assert_close!(Uniform::new(-1.0, 3.0).unwrap().l_location(), 1.0, 1e-12);
        let e = Exponential::new(2.0).unwrap();
        assert_close!(e.l_location(), e.mean(), 1e-12);
        assert_close!(Normal::new(5.0, 2.0).unwrap().l_location(), 5.0, 1e-12);
        assert_close!(Logistic::new(1.5, 0.5).unwrap().l_location(), 1.5, 1e-12);
    }

    #[test]
    fn l_scale_and_ratios_match_known_constants() {
        // Uniform width 4: L-scale = 4/6, no L-skew/kurt.
        let u = Uniform::new(2.0, 6.0).unwrap();
        assert_close!(u.l_scale(), 4.0 / 6.0, 1e-12);
        assert_close!(u.l_skewness(), 0.0, 1e-12);
        assert_close!(u.l_kurtosis(), 0.0, 1e-12);

        // Exponential: scale-free L-ratios 1/3 and 1/6.
        let e = Exponential::new(3.0).unwrap();
        assert_close!(e.l_scale(), 1.0 / 6.0, 1e-12);
        assert_close!(e.l_skewness(), 1.0 / 3.0, 1e-12);
        assert_close!(e.l_kurtosis(), 1.0 / 6.0, 1e-12);

        // Normal: L-scale σ/√π, L-kurtosis 30·atan(√2)/π − 9 ≈ 0.1226.
        let n = Normal::new(0.0, 1.0).unwrap();
        assert_close!(n.l_scale(), 1.0 / std::f64::consts::PI.sqrt(), 1e-12);
        assert_close!(n.l_skewness(), 0.0, 1e-12);
        assert_close!(n.l_kurtosis(), 0.122_601_719_540_892_16, 1e-12);

        // Logistic: L-scale = s, L-kurtosis 1/6.
        let l = Logistic::new(0.0, 2.0).unwrap();
        assert_close!(l.l_scale(), 2.0, 1e-12);
        assert_close!(l.l_kurtosis(), 1.0 / 6.0, 1e-12);
    }
}
