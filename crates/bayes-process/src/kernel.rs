//! Positive-semidefinite **covariance kernels** — the covariance functions that
//! parameterize a Gaussian process. A kernel maps a pair of input points (each a
//! `&[R]`, so inputs can be `d`-dimensional) to their covariance.
//!
//! Each kernel is generic over the euclid scalar `R: Real` (defaulting to `f64`), so the
//! same definition evaluates eagerly on `f64` and lowers onto a lazy carrier (a `maxwell`
//! tensor, a `dirac` column). The closed forms are delegated to `bayes_kernels`
//! correlation functions, shared with the distance-parameterized [`crate::gp`] kernels, and
//! the positive-definiteness domain guard (variance `≥ 0`, scales `> 0`, all finite) is
//! expressed on the euclid comparison/mask surface so it lowers to one branchless `select`
//! on every backend rather than a host `if`.

use bayes_kernels as kernels;
use euclid::{Boolean, Real};

/// A covariance function `k(a, b)`. Implementations must be symmetric
/// (`k(a,b) = k(b,a)`) and positive semidefinite. Generic over the carrier `R: Real`.
pub trait Kernel<R: Real = f64> {
    fn covariance(&self, a: &[R], b: &[R]) -> R;
}

/// `‖a − b‖²` as a carrier value. The length mismatch is a host-shape error (returns
/// `NaN`); non-finite *values* are left to propagate through the arithmetic, matching the
/// lazy-surface convention of guarding shape on the host and values via masks.
fn squared_distance<R: Real>(a: &[R], b: &[R]) -> R {
    if a.len() != b.len() {
        return R::nan();
    }
    a.iter()
        .cloned()
        .zip(b.iter().cloned())
        .map(|(x, y)| (x - y).square())
        .fold(R::zero(), |acc, d| acc + d)
}

/// A non-negative, finite parameter (a signal variance or a bias offset).
fn nonneg_finite<R: Real>(x: &R) -> R::Mask {
    x.is_finite().and(&x.ge(&R::zero()))
}

/// A strictly-positive, finite parameter (a length scale or a period).
fn positive_finite<R: Real>(x: &R) -> R::Mask {
    x.is_finite().and(&x.gt(&R::zero()))
}

/// The **radial basis function** (squared-exponential / Gaussian) kernel:
/// `σ² exp(-‖a-b‖² / (2ℓ²))`. The default smooth, stationary kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rbf<R: Real = f64> {
    pub variance: R,
    pub length_scale: R,
}

impl<R: Real> Rbf<R> {
    pub fn new(variance: R, length_scale: R) -> Self {
        Self {
            variance,
            length_scale,
        }
    }
}

impl<R: Real> Kernel<R> for Rbf<R> {
    fn covariance(&self, a: &[R], b: &[R]) -> R {
        let valid = nonneg_finite(&self.variance).and(&positive_finite(&self.length_scale));
        let distance = squared_distance(a, b).sqrt();
        let value = self.variance.clone()
            * kernels::squared_exponential_correlation(distance, self.length_scale.clone());
        value.mask_invalid(valid, R::nan())
    }
}

/// The smoothness order of a [`Matern`] kernel (half-integer `ν`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaternNu {
    /// `ν = 1/2` — the exponential kernel (rough, Ornstein–Uhlenbeck paths).
    Half,
    /// `ν = 3/2`.
    ThreeHalves,
    /// `ν = 5/2` — a common default for moderately smooth functions.
    FiveHalves,
}

/// The **Matérn** kernel at half-integer smoothness — less smooth than the RBF,
/// often a better model for physical processes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matern<R: Real = f64> {
    pub variance: R,
    pub length_scale: R,
    pub nu: MaternNu,
}

impl<R: Real> Matern<R> {
    pub fn new(variance: R, length_scale: R, nu: MaternNu) -> Self {
        Self {
            variance,
            length_scale,
            nu,
        }
    }
}

impl<R: Real> Kernel<R> for Matern<R> {
    fn covariance(&self, a: &[R], b: &[R]) -> R {
        let valid = nonneg_finite(&self.variance).and(&positive_finite(&self.length_scale));
        let distance = squared_distance(a, b).sqrt();
        let correlation = match self.nu {
            MaternNu::Half => kernels::exponential_correlation(distance, self.length_scale.clone()),
            MaternNu::ThreeHalves => {
                kernels::matern_32_correlation(distance, self.length_scale.clone())
            }
            MaternNu::FiveHalves => {
                kernels::matern_52_correlation(distance, self.length_scale.clone())
            }
        };
        let value = self.variance.clone() * correlation;
        value.mask_invalid(valid, R::nan())
    }
}

/// The **linear** (dot-product) kernel `σ² (a·b) + c` — Gaussian processes with
/// this kernel are Bayesian linear regression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Linear<R: Real = f64> {
    pub variance: R,
    pub bias: R,
}

impl<R: Real> Linear<R> {
    pub fn new(variance: R, bias: R) -> Self {
        Self { variance, bias }
    }
}

impl<R: Real> Kernel<R> for Linear<R> {
    fn covariance(&self, a: &[R], b: &[R]) -> R {
        let valid = nonneg_finite(&self.variance).and(&nonneg_finite(&self.bias));
        let dot = if a.len() != b.len() {
            R::nan()
        } else {
            a.iter()
                .cloned()
                .zip(b.iter().cloned())
                .map(|(x, y)| x * y)
                .fold(R::zero(), |acc, p| acc + p)
        };
        let value = self.variance.clone() * dot + self.bias.clone();
        value.mask_invalid(valid, R::nan())
    }
}

/// The **periodic** (exp-sine-squared) kernel
/// `σ² exp(-2 sin²(π‖a-b‖/p) / ℓ²)` — for processes with a repeating pattern.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Periodic<R: Real = f64> {
    pub variance: R,
    pub length_scale: R,
    pub period: R,
}

impl<R: Real> Periodic<R> {
    pub fn new(variance: R, length_scale: R, period: R) -> Self {
        Self {
            variance,
            length_scale,
            period,
        }
    }
}

impl<R: Real> Kernel<R> for Periodic<R> {
    fn covariance(&self, a: &[R], b: &[R]) -> R {
        let valid = nonneg_finite(&self.variance)
            .and(&positive_finite(&self.length_scale))
            .and(&positive_finite(&self.period));
        let distance = squared_distance(a, b).sqrt();
        let value = self.variance.clone()
            * kernels::periodic_correlation(
                distance,
                self.period.clone(),
                self.length_scale.clone(),
            );
        value.mask_invalid(valid, R::nan())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn rbf_peaks_at_zero_distance_and_is_symmetric() {
        let k = Rbf::new(2.0, 1.5);
        close(k.covariance(&[1.0], &[1.0]), 2.0, 1e-12);
        close(
            k.covariance(&[0.0], &[3.0]),
            k.covariance(&[3.0], &[0.0]),
            1e-15,
        );
        assert!(k.covariance(&[0.0], &[3.0]) < 2.0);
    }

    #[test]
    fn matern_half_is_the_exponential_kernel() {
        let k = Matern::new(1.0, 1.0, MaternNu::Half);
        close(k.covariance(&[0.0], &[2.0]), (-2.0f64).exp(), 1e-12);
    }

    #[test]
    fn periodic_repeats() {
        let k = Periodic::new(1.0, 1.0, 4.0);
        close(
            k.covariance(&[0.0], &[4.0]),
            k.covariance(&[0.0], &[0.0]),
            1e-12,
        );
    }

    #[test]
    fn invalid_kernel_inputs_are_nan() {
        assert!(
            Rbf::<f64>::new(1.0, 1.0)
                .covariance(&[0.0], &[0.0, 1.0])
                .is_nan()
        );
        assert!(Rbf::<f64>::new(1.0, 0.0).covariance(&[0.0], &[0.0]).is_nan());
        assert!(
            Periodic::<f64>::new(1.0, 1.0, 0.0)
                .covariance(&[0.0], &[0.0])
                .is_nan()
        );
        assert!(
            Linear::<f64>::new(-1.0, 0.0)
                .covariance(&[0.0], &[0.0])
                .is_nan()
        );
    }
}
