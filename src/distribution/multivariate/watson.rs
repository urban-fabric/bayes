//! Watson distribution on `S²`.
//!
//! The Watson law is the axial counterpart of von Mises-Fisher:
//! `p(x) ∝ exp(κ(μ·x)^2)`. It is antipodally symmetric (`x` and `-x` have the same
//! density), with `κ > 0` concentrating around the two poles `±μ` and `κ < 0`
//! concentrating around the great circle perpendicular to `μ`.
//!
//! This type intentionally covers the closed-form `S²` case only. Generic Watson /
//! Bingham normalizers are hypergeometric functions of matrix argument, which remain
//! outside Bayes' normalized distribution surface until they have a deliberate
//! closed-form special-function implementation.

use hausdorff::space::sphere::Sphere;
use ramanujan::DenseMatrix;
use rand::distr::Distribution;
use rand::{Rng, RngExt};

use super::BinghamPotential;
use crate::distribution::KlDivergence;
use crate::distribution::multivariate::{MultivariateContinuous, MultivariateMoments};
use crate::error::BayesError;
use crate::function::erf::erf;

use std::f64::consts::{PI, TAU};

const S2_DIM: usize = 3;
const UNIT_TOLERANCE: f64 = 1e-10;
const SERIES_RADIUS: f64 = 0.25;
const POSITIVE_ASYMPTOTIC_THRESHOLD: f64 = 80.0;
const MOMENT_UNIFORM: f64 = 1.0 / 3.0;

/// Closed-form Watson distribution on `S²`.
#[derive(Clone, Debug, PartialEq)]
pub struct WatsonS2<S> {
    axis: S,
    kappa: f64,
    ln_partition: f64,
    axial_second_moment: f64,
}

impl<S: Sphere> WatsonS2<S> {
    /// Build a Watson distribution on `S²` with axis `μ` and concentration `κ`.
    ///
    /// # Errors
    /// [`NonFinite`](BayesError::NonFinite) if `κ` is not finite and
    /// [`Malformed`](BayesError::Malformed) if `axis` is not a 3-dimensional sphere
    /// point.
    pub fn new(axis: S, kappa: f64) -> Result<Self, BayesError> {
        if axis.ambient_dim() != S2_DIM {
            return Err(BayesError::Malformed {
                what: "WatsonS2 needs a point on S²",
            });
        }
        if !kappa.is_finite() {
            return Err(BayesError::NonFinite {
                name: "kappa",
                value: kappa,
            });
        }
        Ok(Self {
            axis,
            kappa,
            ln_partition: ln_watson_s2_partition(kappa),
            axial_second_moment: axial_second_moment_from_kappa(kappa),
        })
    }

    /// Axial direction `μ`. The distribution does not distinguish `μ` from `-μ`.
    #[must_use]
    pub fn axis(&self) -> &S {
        &self.axis
    }

    /// Watson concentration `κ`: positive is bipolar, negative is girdle-like, zero is
    /// uniform.
    #[must_use]
    pub fn concentration(&self) -> f64 {
        self.kappa
    }

    /// Ambient dimension, always `3`.
    #[must_use]
    pub fn ambient_dim(&self) -> usize {
        S2_DIM
    }

    /// Whether `κ = 0`, making the distribution uniform on `S²`.
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        self.kappa == 0.0
    }

    /// Whether `κ > 0`, concentrating mass around the two antipodal poles `±μ`.
    #[must_use]
    pub fn is_bipolar(&self) -> bool {
        self.kappa > 0.0
    }

    /// Whether `κ < 0`, concentrating mass near the great circle perpendicular to `μ`.
    #[must_use]
    pub fn is_girdle(&self) -> bool {
        self.kappa < 0.0
    }

    /// Watson laws are antipodally symmetric: `p(x) = p(-x)`.
    #[must_use]
    pub fn is_antipodally_symmetric(&self) -> bool {
        true
    }

    /// Closed-form log partition function
    /// `ln Z(κ) = ln ∫_{S²} exp(κ(μ·x)^2) dx`.
    #[must_use]
    pub fn ln_partition(&self) -> f64 {
        self.ln_partition
    }

    /// Closed-form partition function.
    #[must_use]
    pub fn partition(&self) -> f64 {
        self.ln_partition.exp()
    }

    /// `E[(μ·X)^2] = d ln Z(κ) / dκ`.
    #[must_use]
    pub fn axial_second_moment(&self) -> f64 {
        self.axial_second_moment
    }

    /// Second moment along either equatorial tangent axis.
    #[must_use]
    pub fn equatorial_second_moment(&self) -> f64 {
        0.5 * (1.0 - self.axial_second_moment)
    }

    /// Natural Bingham concentration matrix `κ μ μᵀ`.
    #[must_use]
    pub fn concentration_matrix(&self) -> DenseMatrix {
        let mu = self.axis_vector();
        DenseMatrix::from_fn(S2_DIM, S2_DIM, |i, j| self.kappa * mu[i] * mu[j])
    }

    /// The same density represented as a Bingham potential with rank-one concentration
    /// matrix `κ μ μᵀ`.
    pub fn as_bingham_potential(&self) -> Result<BinghamPotential, BayesError> {
        BinghamPotential::new(self.concentration_matrix())
    }

    /// Second-moment matrix `E[XXᵀ]`. Since the mean is zero, this is also the
    /// covariance matrix.
    #[must_use]
    pub fn second_moment_matrix(&self) -> DenseMatrix {
        let mu = self.axis_vector();
        let parallel = self.axial_second_moment;
        let perpendicular = self.equatorial_second_moment();
        DenseMatrix::from_fn(S2_DIM, S2_DIM, |i, j| {
            let eye = if i == j { 1.0 } else { 0.0 };
            perpendicular * eye + (parallel - perpendicular) * mu[i] * mu[j]
        })
    }

    /// `E[(ν·X)^2]` for another unit axis `ν`.
    #[must_use]
    pub fn second_moment_along_axis(&self, axis: &S) -> f64 {
        if axis.ambient_dim() != S2_DIM {
            return f64::NAN;
        }
        let dot = self.axis.dot(axis);
        let perpendicular = self.equatorial_second_moment();
        perpendicular + (self.axial_second_moment - perpendicular) * dot * dot
    }

    /// Unnormalized log density `κ(μ·x)^2` at a unit ambient vector.
    #[must_use]
    pub fn ln_unnormalized_density(&self, x: &[f64]) -> f64 {
        let Ok(x) = validate_unit3(x) else {
            return f64::NEG_INFINITY;
        };
        let mu = self.axis_vector();
        let dot = dot3(&mu, &x);
        self.kappa * dot * dot
    }

    /// Normalized log density at a sphere point.
    #[must_use]
    pub fn ln_pdf_point(&self, x: &S) -> f64 {
        if x.ambient_dim() != S2_DIM {
            return f64::NEG_INFINITY;
        }
        let dot = self.axis.dot(x);
        self.kappa * dot * dot - self.ln_partition
    }

    /// Normalized density at a sphere point.
    #[must_use]
    pub fn pdf_point(&self, x: &S) -> f64 {
        self.ln_pdf_point(x).exp()
    }

    /// Log density ratio `ln p(x) - ln p(y)`, where the closed-form normalizer cancels.
    #[must_use]
    pub fn ln_density_difference(&self, x: &[f64], y: &[f64]) -> f64 {
        self.ln_unnormalized_density(x) - self.ln_unnormalized_density(y)
    }

    /// Density ratio `p(x)/p(y)`.
    #[must_use]
    pub fn density_ratio(&self, x: &[f64], y: &[f64]) -> f64 {
        self.ln_density_difference(x, y).exp()
    }

    /// Euclidean gradient of the unnormalized log density before projection to the
    /// sphere tangent space.
    pub fn euclidean_gradient(&self, x: &[f64]) -> Result<Vec<f64>, BayesError> {
        let x = validate_unit3(x)?;
        let mu = self.axis_vector();
        let dot = dot3(&mu, &x);
        Ok(mu.iter().map(|&m| 2.0 * self.kappa * dot * m).collect())
    }

    /// Riemannian gradient on `S²`.
    pub fn tangent_gradient(&self, x: &[f64]) -> Result<Vec<f64>, BayesError> {
        let x = validate_unit3(x)?;
        let grad = self.euclidean_gradient(&x)?;
        let radial = dot3_slice(&x, &grad);
        Ok(grad
            .iter()
            .zip(x.iter())
            .map(|(&g, &xi)| g - radial * xi)
            .collect())
    }

    /// Differential entropy in nats: `H = ln Z(κ) - κ E[(μ·X)^2]`.
    #[must_use]
    pub fn entropy_closed_form(&self) -> f64 {
        self.ln_partition - self.kappa * self.axial_second_moment
    }

    /// Cross-entropy `H(self, other) = -E_self[ln other(X)]`.
    #[must_use]
    pub fn cross_entropy(&self, other: &Self) -> f64 {
        other.ln_partition - other.kappa * self.second_moment_along_axis(&other.axis)
    }

    /// Product of two Watson densities when their axes are coaxial. For non-coaxial
    /// axes the product is a generic Bingham potential and is intentionally not
    /// normalized here.
    pub fn product_fusion(&self, other: &Self) -> Option<Self> {
        self.is_coaxial(other)
            .then(|| Self::new(self.axis.clone(), self.kappa + other.kappa).ok())
            .flatten()
    }

    /// Integral of the product of two normalized Watson densities when the axes are
    /// coaxial:
    /// `∫p₁p₂ = Z(κ₁+κ₂)/(Z(κ₁)Z(κ₂))`.
    #[must_use]
    pub fn ln_product_integral(&self, other: &Self) -> Option<f64> {
        self.product_fusion(other)
            .map(|fused| fused.ln_partition - self.ln_partition - other.ln_partition)
    }

    /// Maximum-likelihood fit to observed S2 directions.
    ///
    /// The axis comes from the eigensystem of the empirical scatter matrix. The
    /// concentration is the monotone inverse of the closed-form second-moment map
    /// `E[(μ·X)^2]`.
    pub fn fit(samples: &[S]) -> Result<Self, BayesError> {
        if samples.is_empty() {
            return Err(BayesError::Malformed {
                what: "WatsonS2 fit needs at least one sample",
            });
        }
        Self::fit_weighted(samples, &vec![1.0; samples.len()])
    }

    /// Weighted Watson MLE. Equal weights recover [`fit`](Self::fit).
    pub fn fit_weighted(samples: &[S], weights: &[f64]) -> Result<Self, BayesError> {
        if samples.is_empty() || samples.len() != weights.len() {
            return Err(BayesError::Malformed {
                what: "WatsonS2 weighted fit needs matching non-empty samples and weights",
            });
        }
        if samples.iter().any(|s| s.ambient_dim() != S2_DIM) {
            return Err(BayesError::Malformed {
                what: "WatsonS2 fit needs points on S²",
            });
        }
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(BayesError::Malformed {
                what: "WatsonS2 fit needs finite nonnegative weights",
            });
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(BayesError::Malformed {
                what: "WatsonS2 weighted fit needs positive total weight",
            });
        }

        let mut scatter = [[0.0; S2_DIM]; S2_DIM];
        let mut x = [0.0; S2_DIM];
        for (sample, &weight) in samples.iter().zip(weights) {
            sample.write_ambient(&mut x);
            for i in 0..S2_DIM {
                for j in 0..S2_DIM {
                    scatter[i][j] += weight * x[i] * x[j] / total;
                }
            }
        }
        let scatter = DenseMatrix::from_fn(S2_DIM, S2_DIM, |i, j| scatter[i][j]);
        let eigen = scatter
            .symmetric_eigen()
            .map_err(|_| BayesError::Malformed {
                what: "WatsonS2 scatter eigendecomposition failed",
            })?;

        let mut best: Option<(f64, usize, f64)> = None;
        for (index, &second_moment) in eigen.values.iter().enumerate() {
            let kappa = kappa_from_axial_second_moment(second_moment);
            let objective = kappa * second_moment - ln_watson_s2_partition(kappa);
            if best.is_none_or(|(best_objective, _, _)| objective > best_objective) {
                best = Some((objective, index, kappa));
            }
        }
        let (_, index, kappa) = best.unwrap_or_else(|| unreachable!());
        let axis = [
            eigen.vectors.get(0, index),
            eigen.vectors.get(1, index),
            eigen.vectors.get(2, index),
        ];
        Self::new(S::from_ambient(&axis), kappa)
    }

    fn axis_vector(&self) -> [f64; S2_DIM] {
        let mut axis = [0.0; S2_DIM];
        self.axis.write_ambient(&mut axis);
        axis
    }

    fn is_coaxial(&self, other: &Self) -> bool {
        (self.axis.dot(&other.axis).abs() - 1.0).abs() <= UNIT_TOLERANCE
    }

    fn sample_point<R: Rng + ?Sized>(&self, rng: &mut R) -> S {
        let axis = self.axis_vector();
        let (u, v) = tangent_basis(&axis);
        let t = sample_axial_cosine(self.kappa, rng);
        let radius = (1.0 - t * t).max(0.0).sqrt();
        let angle = TAU * rng.random::<f64>();
        let (sin, cos) = angle.sin_cos();
        let x = [
            t.mul_add(axis[0], radius * (cos * u[0] + sin * v[0])),
            t.mul_add(axis[1], radius * (cos * u[1] + sin * v[1])),
            t.mul_add(axis[2], radius * (cos * u[2] + sin * v[2])),
        ];
        S::from_ambient(&x)
    }
}

impl<S: Sphere> Distribution<S> for WatsonS2<S> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> S {
        self.sample_point(rng)
    }
}

impl<S: Sphere> MultivariateContinuous for WatsonS2<S> {
    fn dim(&self) -> usize {
        S2_DIM
    }

    /// Density of the direction of a finite non-zero ambient vector.
    fn ln_pdf(&self, x: &[f64]) -> f64 {
        if x.len() != S2_DIM || x.iter().any(|v| !v.is_finite()) || x.iter().all(|v| *v == 0.0) {
            return f64::NEG_INFINITY;
        }
        self.ln_pdf_point(&S::from_ambient(x))
    }
}

impl<S: Sphere> MultivariateMoments for WatsonS2<S> {
    fn mean_vector(&self) -> Vec<f64> {
        vec![0.0; S2_DIM]
    }

    fn covariance(&self) -> DenseMatrix {
        self.second_moment_matrix()
    }

    fn entropy(&self) -> Option<f64> {
        Some(self.entropy_closed_form())
    }
}

impl<S: Sphere> KlDivergence for WatsonS2<S> {
    fn kl_divergence(&self, other: &Self) -> f64 {
        self.cross_entropy(other) - self.entropy_closed_form()
    }
}

fn ln_watson_s2_partition(kappa: f64) -> f64 {
    (4.0 * PI).ln() + ln_uniform_quadratic_mgf(kappa)
}

fn ln_uniform_quadratic_mgf(kappa: f64) -> f64 {
    if kappa.abs() <= SERIES_RADIUS {
        return uniform_quadratic_mgf_series(kappa).ln();
    }
    if kappa < 0.0 {
        let a = (-kappa).sqrt();
        // ∫_{-1}^{1} e^{-a²t²}dt = √π erf(a)/a, while the uniform S2 marginal for
        // t = μ·X contributes a factor 1/2.
        return 0.5 * PI.ln() + erf(a).ln() - a.ln() - 2.0_f64.ln();
    }
    if kappa > POSITIVE_ASYMPTOTIC_THRESHOLD {
        // erfi asymptotic for ∫ e^{κt²}: E_uniform[e^{κT²}] =
        // e^κ/(2κ)·(1 + 1/(2κ) + 3/(4κ²) + 15/(8κ³) + ...).
        let inv = 1.0 / kappa;
        let series =
            1.0 + inv * (0.5 + inv * (0.75 + inv * (1.875 + inv * (6.5625 + inv * 29.53125))));
        return kappa - (2.0 * kappa).ln() + series.ln();
    }
    ln_positive_uniform_quadratic_mgf_series(kappa)
}

fn uniform_quadratic_mgf_series(kappa: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for n in 1..200 {
        let nf = n as f64;
        term *= kappa / nf * (2.0 * nf - 1.0) / (2.0 * nf + 1.0);
        sum += term;
        if term.abs() <= 1e-16 * sum.abs().max(1.0) {
            break;
        }
    }
    sum
}

fn ln_positive_uniform_quadratic_mgf_series(kappa: f64) -> f64 {
    let mut ln_sum = 0.0;
    let mut ln_term = 0.0;
    let ln_kappa = kappa.ln();
    for n in 1..100_000 {
        let nf = n as f64;
        ln_term += ln_kappa - nf.ln() + ((2.0 * nf - 1.0) / (2.0 * nf + 1.0)).ln();
        ln_sum = ln_add_exp(ln_sum, ln_term);
        if nf > kappa && ln_term < ln_sum - 40.0 {
            break;
        }
    }
    ln_sum
}

fn axial_second_moment_from_kappa(kappa: f64) -> f64 {
    if kappa.abs() <= SERIES_RADIUS {
        return axial_second_moment_series(kappa);
    }
    let ln_integral = ln_watson_s2_partition(kappa) - TAU.ln();
    let edge_over_integral = (kappa - ln_integral).exp();
    (edge_over_integral / kappa - 0.5 / kappa).clamp(0.0, 1.0)
}

fn axial_second_moment_series(kappa: f64) -> f64 {
    let mut sum0 = 1.0;
    let mut sum2 = MOMENT_UNIFORM;
    let mut term0 = 1.0;
    let mut term2 = MOMENT_UNIFORM;
    for n in 1..200 {
        let nf = n as f64;
        term0 *= kappa / nf * (2.0 * nf - 1.0) / (2.0 * nf + 1.0);
        term2 *= kappa / nf * (2.0 * nf + 1.0) / (2.0 * nf + 3.0);
        sum0 += term0;
        sum2 += term2;
        if term0.abs().max(term2.abs()) <= 1e-16 * sum0.abs().max(1.0) {
            break;
        }
    }
    (sum2 / sum0).clamp(0.0, 1.0)
}

fn kappa_from_axial_second_moment(second_moment: f64) -> f64 {
    let target = second_moment.clamp(1e-12, 1.0 - 1e-12);
    if (target - MOMENT_UNIFORM).abs() <= 1e-10 {
        return 0.0;
    }
    if target > MOMENT_UNIFORM {
        let mut low = 0.0;
        let mut high = 1.0;
        while axial_second_moment_from_kappa(high) < target && high < 1e12 {
            high *= 2.0;
        }
        for _ in 0..90 {
            let mid = 0.5 * (low + high);
            if axial_second_moment_from_kappa(mid) < target {
                low = mid;
            } else {
                high = mid;
            }
        }
        0.5 * (low + high)
    } else {
        let mut low = -1.0;
        let mut high = 0.0;
        while axial_second_moment_from_kappa(low) > target && low > -1e12 {
            low *= 2.0;
        }
        for _ in 0..90 {
            let mid = 0.5 * (low + high);
            if axial_second_moment_from_kappa(mid) < target {
                low = mid;
            } else {
                high = mid;
            }
        }
        0.5 * (low + high)
    }
}

fn sample_axial_cosine<R: Rng + ?Sized>(kappa: f64, rng: &mut R) -> f64 {
    if kappa == 0.0 {
        return 2.0 * rng.random::<f64>() - 1.0;
    }
    let max_log_density = kappa.max(0.0);
    loop {
        let t = 2.0 * rng.random::<f64>() - 1.0;
        let accept = (kappa * t * t - max_log_density).exp();
        if rng.random::<f64>() <= accept {
            return t;
        }
    }
}

fn tangent_basis(axis: &[f64; S2_DIM]) -> ([f64; S2_DIM], [f64; S2_DIM]) {
    let reference = if axis[0].abs() <= axis[1].abs() && axis[0].abs() <= axis[2].abs() {
        [1.0, 0.0, 0.0]
    } else if axis[1].abs() <= axis[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let u = normalize3(cross3(axis, &reference));
    let v = cross3(axis, &u);
    (u, v)
}

fn validate_unit3(x: &[f64]) -> Result<[f64; S2_DIM], BayesError> {
    if x.len() != S2_DIM || x.iter().any(|v| !v.is_finite()) {
        return Err(BayesError::Malformed {
            what: "point must be finite and 3-dimensional",
        });
    }
    let x = [x[0], x[1], x[2]];
    if (dot3(&x, &x) - 1.0).abs() > UNIT_TOLERANCE {
        return Err(BayesError::Malformed {
            what: "point must be unit length",
        });
    }
    Ok(x)
}

fn dot3(a: &[f64; S2_DIM], b: &[f64; S2_DIM]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dot3_slice(a: &[f64; S2_DIM], b: &[f64]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: &[f64; S2_DIM], b: &[f64; S2_DIM]) -> [f64; S2_DIM] {
    [
        a[1].mul_add(b[2], -a[2] * b[1]),
        a[2].mul_add(b[0], -a[0] * b[2]),
        a[0].mul_add(b[1], -a[1] * b[0]),
    ]
}

fn normalize3(x: [f64; S2_DIM]) -> [f64; S2_DIM] {
    let norm = dot3(&x, &x).sqrt();
    [x[0] / norm, x[1] / norm, x[2] / norm]
}

fn ln_add_exp(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use hausdorff::space::sphere::UnitVector;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn axis(x: &[f64; 3]) -> UnitVector {
        UnitVector::from_coords(x)
    }

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn closed_form_normalizer_covers_uniform_positive_and_negative() {
        let uniform = WatsonS2::new(axis(&[1.0, 0.0, 0.0]), 0.0).unwrap();
        close(uniform.partition(), 4.0 * PI, 1e-12);
        close(uniform.axial_second_moment(), 1.0 / 3.0, 1e-12);

        let negative = WatsonS2::new(axis(&[1.0, 0.0, 0.0]), -4.0).unwrap();
        let expected = PI * PI.sqrt() * erf(2.0);
        close(negative.partition(), expected, 1e-12);
        assert!(negative.axial_second_moment() < 1.0 / 3.0);

        let positive = WatsonS2::new(axis(&[1.0, 0.0, 0.0]), 3.0).unwrap();
        assert!(positive.partition() > 4.0 * PI);
        assert!(positive.axial_second_moment() > 1.0 / 3.0);
    }

    #[test]
    fn density_ratios_gradients_and_bingham_potential_agree() {
        let watson = WatsonS2::new(axis(&[1.0, 0.0, 0.0]), 4.0).unwrap();
        close(
            watson.ln_density_difference(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]),
            4.0,
            1e-12,
        );
        close(
            watson.density_ratio(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]),
            4.0_f64.exp(),
            1e-12,
        );
        let grad = watson.euclidean_gradient(&[1.0, 0.0, 0.0]).unwrap();
        close(grad[0], 8.0, 1e-12);
        let tangent = watson.tangent_gradient(&[1.0, 0.0, 0.0]).unwrap();
        close(tangent[0], 0.0, 1e-12);

        let bingham = watson.as_bingham_potential().unwrap();
        close(
            bingham.ln_unnormalized_density(&[0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0]),
            watson.ln_unnormalized_density(&[0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0]),
            1e-12,
        );
    }

    #[test]
    fn entropy_kl_moments_and_coaxial_product_are_closed_form() {
        use crate::distribution::{KlDivergence, MultivariateMoments};

        let p = WatsonS2::new(axis(&[1.0, 0.0, 0.0]), 3.0).unwrap();
        let q = WatsonS2::new(axis(&[0.0, 1.0, 0.0]), -2.0).unwrap();
        close(p.kl_divergence(&p), 0.0, 1e-12);
        assert!(p.kl_divergence(&q) > 0.0);
        close(p.entropy().unwrap(), p.entropy_closed_form(), 1e-12);
        close(p.mean_vector()[0], 0.0, 1e-12);
        close(p.covariance().get(0, 0), p.axial_second_moment(), 1e-12);
        close(
            p.covariance().get(1, 1),
            p.equatorial_second_moment(),
            1e-12,
        );

        let r = WatsonS2::new(axis(&[-1.0, 0.0, 0.0]), 2.0).unwrap();
        let fused = p.product_fusion(&r).unwrap();
        close(fused.concentration(), 5.0, 1e-12);
        let expected = fused.ln_partition() - p.ln_partition() - r.ln_partition();
        close(p.ln_product_integral(&r).unwrap(), expected, 1e-12);
        assert!(p.product_fusion(&q).is_none());
    }

    #[test]
    fn partition_derivative_matches_axial_second_moment() {
        for &kappa in &[-5.0, -0.1, 0.1, 4.0, 20.0] {
            let eps = 1e-5;
            let numerical = (ln_watson_s2_partition(kappa + eps)
                - ln_watson_s2_partition(kappa - eps))
                / (2.0 * eps);
            close(axial_second_moment_from_kappa(kappa), numerical, 1e-5);
        }
    }

    #[test]
    fn sampling_and_fit_recover_bipolar_axis() {
        let true_axis = axis(&[1.0, 0.0, 0.0]);
        let watson = WatsonS2::new(true_axis.clone(), 6.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(21);
        let samples: Vec<UnitVector> = (0..4_000).map(|_| watson.sample(&mut rng)).collect();
        let fitted = WatsonS2::fit(&samples).unwrap();
        assert!(fitted.axis().dot(&true_axis).abs() > 0.98);
        close(fitted.concentration(), 6.0, 0.35);
    }

    #[test]
    fn fit_recovers_girdle_axis() {
        let true_axis = axis(&[0.0, 0.0, 1.0]);
        let watson = WatsonS2::new(true_axis.clone(), -5.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(22);
        let samples: Vec<UnitVector> = (0..5_000).map(|_| watson.sample(&mut rng)).collect();
        let fitted = WatsonS2::fit(&samples).unwrap();
        assert!(fitted.axis().dot(&true_axis).abs() > 0.95);
        close(fitted.concentration(), -5.0, 0.4);
    }
}
