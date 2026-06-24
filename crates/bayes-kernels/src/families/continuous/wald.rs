use euclid::numeric::Real;

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
    let s = (lambda.clone() / safe_x.clone()).sqrt();
    let a = s.clone() * (safe_x.clone() / mu.clone() - T::one());
    let b = s * (safe_x / mu.clone() + T::one());
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
