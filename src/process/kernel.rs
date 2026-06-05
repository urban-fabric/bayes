//! Positive-semidefinite **covariance kernels** — the covariance functions that
//! parameterize a Gaussian process. A kernel maps a pair of input points (each a
//! `&[f64]`, so inputs can be `d`-dimensional) to their covariance.

/// A covariance function `k(a, b)`. Implementations must be symmetric
/// (`k(a,b) = k(b,a)`) and positive semidefinite.
pub trait Kernel {
    fn covariance(&self, a: &[f64], b: &[f64]) -> f64;
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.iter().any(|x| !x.is_finite()) || b.iter().any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// The **radial basis function** (squared-exponential / Gaussian) kernel:
/// `σ² exp(-‖a-b‖² / (2ℓ²))`. The default smooth, stationary kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rbf {
    pub variance: f64,
    pub length_scale: f64,
}

impl Rbf {
    pub fn new(variance: f64, length_scale: f64) -> Self {
        Self {
            variance,
            length_scale,
        }
    }
}

impl Kernel for Rbf {
    fn covariance(&self, a: &[f64], b: &[f64]) -> f64 {
        if !self.variance.is_finite()
            || !self.length_scale.is_finite()
            || self.variance < 0.0
            || self.length_scale <= 0.0
        {
            return f64::NAN;
        }
        let d2 = squared_distance(a, b);
        self.variance * (-d2 / (2.0 * self.length_scale * self.length_scale)).exp()
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
pub struct Matern {
    pub variance: f64,
    pub length_scale: f64,
    pub nu: MaternNu,
}

impl Matern {
    pub fn new(variance: f64, length_scale: f64, nu: MaternNu) -> Self {
        Self {
            variance,
            length_scale,
            nu,
        }
    }
}

impl Kernel for Matern {
    fn covariance(&self, a: &[f64], b: &[f64]) -> f64 {
        if !self.variance.is_finite()
            || !self.length_scale.is_finite()
            || self.variance < 0.0
            || self.length_scale <= 0.0
        {
            return f64::NAN;
        }
        let r = squared_distance(a, b).sqrt() / self.length_scale;
        let shape = match self.nu {
            MaternNu::Half => (-r).exp(),
            MaternNu::ThreeHalves => {
                let s = 3.0_f64.sqrt() * r;
                (1.0 + s) * (-s).exp()
            }
            MaternNu::FiveHalves => {
                let s = 5.0_f64.sqrt() * r;
                (1.0 + s + s * s / 3.0) * (-s).exp()
            }
        };
        self.variance * shape
    }
}

/// The **linear** (dot-product) kernel `σ² (a·b) + c` — Gaussian processes with
/// this kernel are Bayesian linear regression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Linear {
    pub variance: f64,
    pub bias: f64,
}

impl Linear {
    pub fn new(variance: f64, bias: f64) -> Self {
        Self { variance, bias }
    }
}

impl Kernel for Linear {
    fn covariance(&self, a: &[f64], b: &[f64]) -> f64 {
        if !self.variance.is_finite()
            || !self.bias.is_finite()
            || self.variance < 0.0
            || self.bias < 0.0
        {
            return f64::NAN;
        }
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        self.variance * dot + self.bias
    }
}

/// The **periodic** (exp-sine-squared) kernel
/// `σ² exp(-2 sin²(π‖a-b‖/p) / ℓ²)` — for processes with a repeating pattern.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Periodic {
    pub variance: f64,
    pub length_scale: f64,
    pub period: f64,
}

impl Periodic {
    pub fn new(variance: f64, length_scale: f64, period: f64) -> Self {
        Self {
            variance,
            length_scale,
            period,
        }
    }
}

impl Kernel for Periodic {
    fn covariance(&self, a: &[f64], b: &[f64]) -> f64 {
        if !self.variance.is_finite()
            || !self.length_scale.is_finite()
            || !self.period.is_finite()
            || self.variance < 0.0
            || self.length_scale <= 0.0
            || self.period <= 0.0
        {
            return f64::NAN;
        }
        let d = squared_distance(a, b).sqrt();
        let s = (std::f64::consts::PI * d / self.period).sin();
        self.variance * (-2.0 * s * s / (self.length_scale * self.length_scale)).exp()
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
        close(k.covariance(&[1.0], &[1.0]), 2.0, 1e-12); // variance at d=0
        close(
            k.covariance(&[0.0], &[3.0]),
            k.covariance(&[3.0], &[0.0]),
            1e-15,
        );
        assert!(k.covariance(&[0.0], &[3.0]) < 2.0); // decays with distance
    }

    #[test]
    fn matern_half_is_the_exponential_kernel() {
        let k = Matern::new(1.0, 1.0, MaternNu::Half);
        close(k.covariance(&[0.0], &[2.0]), (-2.0f64).exp(), 1e-12);
    }

    #[test]
    fn periodic_repeats() {
        let k = Periodic::new(1.0, 1.0, 4.0);
        // Distance equal to a full period ⇒ same as distance 0.
        close(
            k.covariance(&[0.0], &[4.0]),
            k.covariance(&[0.0], &[0.0]),
            1e-12,
        );
    }

    #[test]
    fn invalid_kernel_inputs_are_nan() {
        assert!(Rbf::new(1.0, 1.0).covariance(&[0.0], &[0.0, 1.0]).is_nan());
        assert!(Rbf::new(1.0, 0.0).covariance(&[0.0], &[0.0]).is_nan());
        assert!(
            Periodic::new(1.0, 1.0, 0.0)
                .covariance(&[0.0], &[0.0])
                .is_nan()
        );
        assert!(Linear::new(-1.0, 0.0).covariance(&[0.0], &[0.0]).is_nan());
    }
}
