use super::*;

// --- closed-form χ² (Pearson) divergences ---
//
// `_on_support` forms carry the interior only; the `2λ₀>λ₁`-style convergence test
// (`+∞` otherwise) is the caller's branch.

/// Bernoulli χ²: `(p₀−p₁)² / (p₁(1−p₁))`.
pub fn chi_squared_bernoulli<T: LazyReal>(p_self: T, p_other: T) -> T {
    let diff = p_self - p_other.clone();
    diff.clone() * diff / (p_other.clone() * (T::one() - p_other))
}

/// Poisson χ²: `exp((λ₀−λ₁)²/λ₁) − 1`.
pub fn chi_squared_poisson<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    let diff = rate_self - rate_other.clone();
    (diff.clone() * diff / rate_other).exp() - T::one()
}

/// Gaussian χ² interior (`2s₁² > s₀²`): `s₁²/(s₀·√d)·exp((m₀−m₁)²/d) − 1`, `d=2s₁²−s₀²`.
pub fn chi_squared_normal_on_support<T: LazyReal>(
    mean_self: T,
    std_self: T,
    mean_other: T,
    std_other: T,
) -> T {
    let (m0, s0, m1, s1) = (mean_self, std_self, mean_other, std_other);
    let denom = T::two() * s1.clone() * s1.clone() - s0.clone() * s0.clone();
    let dm = m0 - m1;
    s1.clone() * s1 / (s0 * denom.clone().sqrt()) * (dm.clone() * dm / denom).exp() - T::one()
}

/// Exponential χ² interior (`2λ₀ > λ₁`): `λ₀²/(λ₁(2λ₀−λ₁)) − 1`.
pub fn chi_squared_exponential_on_support<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    let (l0, l1) = (rate_self, rate_other);
    l0.clone() * l0.clone() / (l1.clone() * (T::two() * l0 - l1)) - T::one()
}

/// Gamma χ² interior (`2a₀>a₁`, `2b₀>b₁`), via `ln_gamma`.
pub fn chi_squared_gamma_on_support<T: LazyReal>(
    shape_self: T,
    rate_self: T,
    shape_other: T,
    rate_other: T,
) -> T {
    let (a0, b0, a1, b1) = (shape_self, rate_self, shape_other, rate_other);
    let a = T::two() * a0.clone() - a1.clone();
    let b = T::two() * b0.clone() - b1.clone();
    let ln_term = T::two() * a0.clone() * b0.clone().ln() - T::two() * a0.ln_gamma()
        + a1.clone().ln_gamma()
        - a1 * b1.ln()
        + a.clone().ln_gamma()
        - a * b.ln();
    ln_term.exp() - T::one()
}

/// von Mises χ² (always finite): `I₀(R)·I₀(κ₁)/I₀(κ₀)² − 1`,
/// `R = √(4κ₀² + κ₁² − 4κ₀κ₁ cos Δµ)`.
pub fn chi_squared_von_mises<T: LazyReal>(
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
    (ln_i0(r) + ln_i0(k1) - T::two() * ln_i0(k0)).exp() - T::one()
}

/// Binomial χ² for a matched trial count `n`: `(1 + χ²_Bernoulli)^n − 1`, always
/// finite. A differing trial count is the caller's out-of-family branch.
pub fn chi_squared_binomial<T: LazyReal>(trials: T, p_self: T, p_other: T) -> T {
    let diff = p_self - p_other.clone();
    let chi_bernoulli = diff.clone() * diff / (p_other.clone() * (T::one() - p_other));
    (trials * (T::one() + chi_bernoulli).ln()).exp() - T::one()
}

/// Negative-binomial χ² interior for a matched number of successes `r`
/// (finite when `(1−p₀)² < 1−p₁`): `exp(r·(2 ln p₀ − ln p₁ − ln(1 − z))) − 1`,
/// `z = (1−p₀)²/(1−p₁)`. The convergence test is the caller's branch.
pub fn chi_squared_negative_binomial_on_support<T: LazyReal>(
    successes: T,
    p_self: T,
    p_other: T,
) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let z = q_self.clone() * q_self / (one.clone() - p_other.clone());
    (successes * (T::two() * p_self.ln() - p_other.ln() - (one - z).ln())).exp() - T::one()
}

/// Wald / inverse-Gaussian χ² interior (mean µ, shape λ; finite when `2λ₀ > λ₁` and
/// `2λ₀/µ₀² > λ₁/µ₁²`): `exp(ln λ₀ − ½ln λ₁ + 2λ₀/µ₀ − λ₁/µ₁ − ½ln la − √(aa·la)) − 1`,
/// `la = 2λ₀ − λ₁`, `aa = 2λ₀/µ₀² − λ₁/µ₁²`. The convergence test is the caller's branch.
pub fn chi_squared_wald_on_support<T: LazyReal>(
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

/// Uniform χ² on nested support: `w₁/w₀ − 1`, `w = b − a`. The containment test is
/// the caller's branch.
pub fn chi_squared_uniform_on_support<T: LazyReal>(width_self: T, width_other: T) -> T {
    width_other / width_self - T::one()
}

/// Discrete-uniform χ² on nested support: `n₁/n₀ − 1`, `n` the support count.
pub fn chi_squared_discrete_uniform_on_support<T: LazyReal>(count_self: T, count_other: T) -> T {
    count_other / count_self - T::one()
}
