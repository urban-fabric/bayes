use super::*;

// ---- covariance kernels ---------------------------------------------------------------
//
// Stationary, isotropic correlation functions `ρ(d; ℓ)` for geostatistical Gaussian
// processes: `ρ(0) = 1`, decreasing in the separation `distance`, valid (positive-definite)
// covariances. The smoothness ladder runs exponential (Matérn ν = ½, rough) → Matérn 3/2 →
// Matérn 5/2 → squared-exponential (ν → ∞, smooth). Write-once over `Real`, so the GP
// covariance assembled from them runs on every backend (lifted out of `markov::gp::kernel`).
// A caller scales by a signal variance and adds a nugget.

/// Exponential correlation `ρ = exp(−d/ℓ)` — Matérn smoothness ν = ½.
pub fn exponential_correlation<T: Real>(distance: T, length_scale: T) -> T {
    (-(distance / length_scale)).exp()
}

/// Matérn-3/2 correlation `ρ = (1 + √3·d/ℓ)·exp(−√3·d/ℓ)` — once-differentiable fields.
pub fn matern_32_correlation<T: Real>(distance: T, length_scale: T) -> T {
    let s = T::scalar(3.0).sqrt() * (distance / length_scale);
    (T::one() + s.clone()) * (-s).exp()
}

/// Matérn-5/2 correlation `ρ = (1 + √5·d/ℓ + 5d²/3ℓ²)·exp(−√5·d/ℓ)` — twice-differentiable.
pub fn matern_52_correlation<T: Real>(distance: T, length_scale: T) -> T {
    let r = distance / length_scale;
    let s = T::scalar(5.0).sqrt() * r.clone();
    let quadratic = T::scalar(5.0 / 3.0) * r.square();
    (T::one() + s.clone() + quadratic) * (-s).exp()
}

/// Squared-exponential (RBF / Gaussian) correlation `ρ = exp(−½·(d/ℓ)²)` — Matérn ν → ∞.
pub fn squared_exponential_correlation<T: Real>(distance: T, length_scale: T) -> T {
    let r = distance / length_scale;
    (-(T::half() * r.square())).exp()
}

/// Periodic correlation `ρ = exp(−2·sin²(π·Δ/p)/ℓ²)` (1-D) — for cyclic inputs of period `p`.
pub fn periodic_correlation<T: Real>(delta: T, period: T, length_scale: T) -> T {
    let s = ((delta / period) * T::pi()).sin();
    (-(T::two() * s.square() / length_scale.square())).exp()
}

#[cfg(test)]
mod kernel_tests {
    use super::*;

    // ρ(0) = 1, ρ decreasing in d, ρ ≤ 1, ρ → 0 at large separation — on the f64 reference.
    fn assert_shape(rho: impl Fn(f64, f64) -> f64, name: &str) {
        assert!((rho(0.0, 1000.0) - 1.0).abs() < 1e-12, "{name}: ρ(0) != 1");
        assert!(
            rho(500.0, 1000.0) > rho(1500.0, 1000.0),
            "{name}: not decreasing"
        );
        assert!(rho(100.0, 1000.0) <= 1.0, "{name}: ρ > 1");
        assert!(rho(1.0e7, 1000.0) < 1e-6, "{name}: tail not → 0");
    }

    #[test]
    fn covariance_kernels_are_valid_correlations() {
        assert_shape(exponential_correlation::<f64>, "exponential");
        assert_shape(matern_32_correlation::<f64>, "matern32");
        assert_shape(matern_52_correlation::<f64>, "matern52");
        assert_shape(
            squared_exponential_correlation::<f64>,
            "squared_exponential",
        );
    }

    #[test]
    fn matern_matches_known_values() {
        // d = ℓ: matern32 = (1+√3)·e^{−√3}; squared_exp = e^{−½}.
        let s3 = 3.0_f64.sqrt();
        assert!((matern_32_correlation(1.0, 1.0) - (1.0 + s3) * (-s3).exp()).abs() < 1e-12);
        assert!((squared_exponential_correlation(1.0, 1.0) - (-0.5_f64).exp()).abs() < 1e-12);
    }
}
