use euclid_numeric::Real;

/// Wald / inverse-Gaussian log-density on `x > 0`.
pub fn wald_log_pdf_on_support<T: Real>(x: T, mean: T, shape: T) -> T {
    let delta = x.clone() - mean.clone();
    T::half() * (shape.clone().ln() - T::ln_tau() - T::scalar(3.0) * x.clone().ln())
        - shape * delta.clone() * delta / (T::two() * mean.clone() * mean * x)
}

/// Wald / inverse-Gaussian CDF.
pub fn wald_cdf<T: Real>(x: T, mu: T, lambda: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, mu.clone());
    // Distributed over the bracket rather than factored as `√(λ/x)·(x/μ ∓ 1)`. That form is
    // `0 · ∞` at `x = ∞` — `√(λ/x)` underflows to zero while `x/μ` overflows — so `cdf(∞)`
    // returned `NaN` instead of 1. Written as a difference of two separately-finite terms,
    // `√(λx)/μ` and `√(λ/x)`, the limit comes out as `∞` and `erfc` saturates correctly.
    let a = (lambda.clone() * safe_x.clone()).sqrt() / mu.clone()
        - (lambda.clone() / safe_x.clone()).sqrt();
    let b =
        (lambda.clone() * safe_x.clone()).sqrt() / mu.clone() + (lambda.clone() / safe_x).sqrt();
    let phi_a = T::half() * (-a * T::frac_1_sqrt_2()).erfc();
    let term2 = T::half()
        * (b.clone() * T::frac_1_sqrt_2()).erfcx()
        * (T::two() * lambda / mu - T::half() * b.clone() * b).exp();
    (phi_a + term2)
        .clamp(T::zero(), T::one())
        .mask_invalid(in_support, T::zero())
}

/// Wald / inverse-Gaussian skewness.
pub fn wald_skewness<T: Real>(mean: T, shape: T) -> T {
    T::scalar(3.0) * (mean / shape).sqrt()
}

/// Wald / inverse-Gaussian excess kurtosis.
pub fn wald_excess_kurtosis<T: Real>(mean: T, shape: T) -> T {
    T::scalar(15.0) * mean / shape
}

/// Wald / inverse-Gaussian KL divergence.
pub fn kl_wald<T: Real>(mean_self: T, shape_self: T, mean_other: T, shape_other: T) -> T {
    let one = T::one();
    T::half() * (shape_self.clone() / shape_other.clone()).ln()
        + shape_other.clone() * (mean_self.clone() - T::two() * mean_other.clone())
            / (T::two() * mean_other.clone() * mean_other)
        + T::half() * shape_other * (one.clone() / mean_self + one / shape_self)
        - T::half()
}

/// Wald / inverse-Gaussian squared Hellinger distance.
pub fn hellinger_sq_wald<T: Real>(mean_self: T, shape_self: T, mean_other: T, shape_other: T) -> T {
    let (m0, l0, m1, l1) = (mean_self, shape_self, mean_other, shape_other);
    let s = ((l0.clone() / (m0.clone() * m0.clone()) + l1.clone() / (m1.clone() * m1.clone()))
        * (l0.clone() + l1.clone()))
    .sqrt();
    let ln_bc = T::scalar(0.25) * (l0.clone() * l1.clone()).ln()
        + T::half() * (T::two() / (l0.clone() + l1.clone())).ln()
        + T::half() * (l0 / m0 + l1 / m1)
        - T::half() * s;
    crate::hellinger_sq_from_bc(ln_bc.exp())
}

/// Wald / inverse-Gaussian chi-squared divergence interior.
pub fn chi_squared_wald_on_support<T: Real>(
    mean_self: T,
    shape_self: T,
    mean_other: T,
    shape_other: T,
) -> T {
    let (m0, l0, m1, l1) = (mean_self, shape_self, mean_other, shape_other);
    let la = T::two() * l0.clone() - l1.clone();
    let aa =
        T::two() * l0.clone() / (m0.clone() * m0.clone()) - l1.clone() / (m1.clone() * m1.clone());
    let ln_int = l0.clone().ln() - T::half() * l1.clone().ln() + T::two() * l0 / m0
        - l1 / m1
        - T::half() * la.clone().ln()
        - (aa * la).sqrt();
    ln_int.exp() - T::one()
}

/// Wald / inverse-Gaussian Renyi divergence interior.
pub fn renyi_wald_on_support<T: Real>(
    mean_self: T,
    shape_self: T,
    mean_other: T,
    shape_other: T,
    alpha: T,
) -> T {
    let (m0, l0, m1, l1) = (mean_self, shape_self, mean_other, shape_other);
    let one_minus_alpha = T::one() - alpha.clone();
    let la = alpha.clone() * l0.clone() + one_minus_alpha.clone() * l1.clone();
    let aa = alpha.clone() * l0.clone() / (m0.clone() * m0.clone())
        + one_minus_alpha.clone() * l1.clone() / (m1.clone() * m1.clone());
    let ln_int = T::half()
        * (alpha.clone() * l0.clone().ln() + one_minus_alpha.clone() * l1.clone().ln())
        + alpha.clone() * l0 / m0
        + one_minus_alpha * l1 / m1
        - T::half() * la.clone().ln()
        - (aa * la).sqrt();
    ln_int / (alpha - T::one())
}

/// Wald / inverse-Gaussian log-survival.
///
/// # Why the `exp(2λ/μ)` factor disappears
///
/// The textbook survival is
///
/// ```text
///     1 − F(x) = Φ(−a) − e^{2λ/μ}·Φ(b),
///     a = √(λ/x)(x/μ − 1),   b = −√(λ/x)(x/μ + 1)
/// ```
///
/// and `e^{2λ/μ}` overflows on its own above `2λ/μ = 709`. A Wald with `μ = 1`
/// and `λ = 400` is an ordinary distribution, and that form cannot express its
/// survival at all.
///
/// The factor is spurious. Expanding both squares gives `b² − a² = 4λ/μ`
/// **exactly**, so
///
/// ```text
///     2λ/μ − b²/2 = −a²/2
/// ```
///
/// and the overflowing exponential cancels against the one hidden inside
/// `Φ(b)`. Taking both terms through `erfcx`, where `erfc(t) = erfcx(t)·e^{−t²}`,
/// factors the common `e^{−a²/2}` out and leaves nothing large anywhere:
///
/// ```text
///     1 − F(x) = ½·e^{−u²}·[erfcx(u) − erfcx(v)],   u = a/√2,  v = −b/√2
/// ```
///
/// [`wald_cdf`] already uses this identity for its own second term, which is
/// why the CDF never overflowed; only the survival was missing.
///
/// # The limit that remains
///
/// The bracket is a real cancellation: `erfcx(u) → 1/(u√π)` for large `u`, and
/// `(v − u)/v = 2μ/(x + μ)`, so roughly `log10(x/μ)` digits are lost. That puts
/// the floor near `x ≈ 1e16·μ`, past which it reports `−inf` — the honest
/// answer rather than a wrong one. Subtracting the CDF from 1 runs out sixteen
/// orders of magnitude earlier, a few `μ` from the mean.
pub fn wald_ln_sf<T: Real>(x: T, mu: T, lambda: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, mu.clone());
    // Distributed rather than factored as `√(λ/x)·(x/μ ∓ 1)`, for the same
    // `0 · ∞` reason spelled out in `wald_cdf`.
    let root = (lambda.clone() * safe_x.clone()).sqrt() / mu.clone();
    let reciprocal = (lambda / safe_x).sqrt();
    let u = (root.clone() - reciprocal.clone()) * T::frac_1_sqrt_2();
    let v = (root + reciprocal) * T::frac_1_sqrt_2();

    // At or above the mean, both terms carry `e^{−u²}`; factoring it out keeps
    // the difference at full precision and the exponential exact.
    let upper =
        -(u.clone() * u.clone()) + (T::half() * (u.clone().erfcx() - v.clone().erfcx())).ln();
    // Below the mean the survival is near 1, and `erfcx` of a very negative `u`
    // would overflow, so the leading term is taken as `erfc` directly.
    let lower = (T::half() * u.clone().erfc()
        - T::half() * (-(u.clone() * u.clone())).exp() * v.erfcx())
    .ln();
    T::select(&u.ge(&T::zero()), upper, lower).mask_invalid(in_support, T::zero())
}
