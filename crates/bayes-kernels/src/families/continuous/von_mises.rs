use euclid::numeric::Real;

/// Von Mises log-density on `[−pi, pi)`.
pub fn von_mises_log_pdf<T: Real>(x: T, mean: T, kappa: T, log_normalizer: T) -> T {
    kappa * (x - mean).cos() - log_normalizer
}

/// von Mises entropy.
pub fn von_mises_entropy<T: Real>(kappa: T) -> T {
    let resultant = kappa.clone().bessel_i1e() / kappa.clone().bessel_i0e();
    T::ln_tau() + crate::ln_i0(kappa.clone()) - kappa * resultant
}

/// von Mises KL divergence.
pub fn kl_von_mises<T: Real>(mean_self: T, kappa_self: T, mean_other: T, kappa_other: T) -> T {
    let (mu0, k0, mu1, k1) = (mean_self, kappa_self, mean_other, kappa_other);
    let resultant = k0.clone().bessel_i1e() / k0.clone().bessel_i0e();
    crate::ln_i0(k1.clone()) - crate::ln_i0(k0.clone()) + resultant * (k0 - k1 * (mu0 - mu1).cos())
}

/// von Mises squared Hellinger distance.
pub fn hellinger_sq_von_mises<T: Real>(
    mean_self: T,
    kappa_self: T,
    mean_other: T,
    kappa_other: T,
) -> T {
    let (mu0, k0, mu1, k1) = (mean_self, kappa_self, mean_other, kappa_other);
    let cos = (mu0 - mu1).cos();
    let r = T::half()
        * (k0.clone() * k0.clone()
            + k1.clone() * k1.clone()
            + T::two() * k0.clone() * k1.clone() * cos)
            .sqrt();
    let ln_bc = crate::ln_i0(r) - T::half() * (crate::ln_i0(k0) + crate::ln_i0(k1));
    crate::hellinger_sq_from_bc(ln_bc.exp())
}

/// von Mises chi-squared divergence.
pub fn chi_squared_von_mises<T: Real>(
    mean_self: T,
    kappa_self: T,
    mean_other: T,
    kappa_other: T,
) -> T {
    let (mu0, k0, mu1, k1) = (mean_self, kappa_self, mean_other, kappa_other);
    let cos = (mu0 - mu1).cos();
    let r = (T::scalar(4.0) * k0.clone() * k0.clone() + k1.clone() * k1.clone()
        - T::scalar(4.0) * k0.clone() * k1.clone() * cos)
        .sqrt();
    (crate::ln_i0(r) + crate::ln_i0(k1) - T::two() * crate::ln_i0(k0)).exp() - T::one()
}

/// von Mises Renyi divergence.
pub fn renyi_von_mises<T: Real>(
    mean_self: T,
    kappa_self: T,
    mean_other: T,
    kappa_other: T,
    alpha: T,
) -> T {
    let (mu0, k0, mu1, k1) = (mean_self, kappa_self, mean_other, kappa_other);
    let cos = (mu0 - mu1).cos();
    let one_minus_alpha = T::one() - alpha.clone();
    let term0 = alpha.clone() * k0.clone();
    let term1 = one_minus_alpha.clone() * k1.clone();
    let r_alpha = (term0.clone() * term0
        + term1.clone() * term1
        + T::two() * alpha.clone() * one_minus_alpha.clone() * k0.clone() * k1.clone() * cos)
        .sqrt();
    (crate::ln_i0(r_alpha) - alpha.clone() * crate::ln_i0(k0) - one_minus_alpha * crate::ln_i0(k1))
        / (alpha - T::one())
}
