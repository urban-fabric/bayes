use super::*;

// --- closed-form Rényi divergences `D_α` (α ≠ 1) ---

/// Bernoulli Rényi: `ln(p₀^α p₁^{1−α} + (1−p₀)^α (1−p₁)^{1−α}) / (α−1)`.
pub fn renyi_bernoulli<T: Real>(p_self: T, p_other: T, alpha: T) -> T {
    let one = T::one();
    let one_minus_alpha = one.clone() - alpha.clone();
    let term =
        |p: T, q: T| (alpha.clone() * p.ln()).exp() * (one_minus_alpha.clone() * q.ln()).exp();
    let s = term(p_self.clone(), p_other.clone()) + term(one.clone() - p_self, one - p_other);
    s.ln() / (alpha - T::one())
}

/// Gaussian Rényi interior (`σ_α² = α s₁² + (1−α)s₀² > 0`):
/// `α(m₀−m₁)²/(2σ_α²) + ((1−α)ln s₀ + α ln s₁ − ½ln σ_α²)/(α−1)`.
pub fn renyi_normal_on_support<T: Real>(
    mean_self: T,
    std_self: T,
    mean_other: T,
    std_other: T,
    alpha: T,
) -> T {
    let (m0, s0, m1, s1) = (mean_self, std_self, mean_other, std_other);
    let one_minus_alpha = T::one() - alpha.clone();
    let var_alpha =
        alpha.clone() * s1.clone() * s1.clone() + one_minus_alpha.clone() * s0.clone() * s0.clone();
    let dm = m0 - m1;
    let mean_term = alpha.clone() * dm.clone() * dm / (T::two() * var_alpha.clone());
    let log_term = (one_minus_alpha * s0.ln() + alpha.clone() * s1.ln()
        - T::half() * var_alpha.ln())
        / (alpha - T::one());
    mean_term + log_term
}

/// Gamma Rényi interior (`a_α, b_α > 0`), via `ln_gamma`.
pub fn renyi_gamma_on_support<T: Real>(
    shape_self: T,
    rate_self: T,
    shape_other: T,
    rate_other: T,
    alpha: T,
) -> T {
    let (a0, b0, a1, b1) = (shape_self, rate_self, shape_other, rate_other);
    let one_minus_alpha = T::one() - alpha.clone();
    let a_alpha = alpha.clone() * a0.clone() + one_minus_alpha.clone() * a1.clone();
    let b_alpha = alpha.clone() * b0.clone() + one_minus_alpha.clone() * b1.clone();
    let num = alpha.clone() * (a0.clone() * b0.ln() - a0.ln_gamma())
        + one_minus_alpha * (a1.clone() * b1.ln() - a1.ln_gamma())
        + a_alpha.clone().ln_gamma()
        - a_alpha * b_alpha.ln();
    num / (alpha - T::one())
}

/// von Mises Rényi interior:
/// `(ln I₀(R_α) − α ln I₀(κ₀) − (1−α) ln I₀(κ₁)) / (α−1)`,
/// `R_α = √((ακ₀)² + ((1−α)κ₁)² + 2α(1−α)κ₀κ₁ cos Δµ)`.
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
    (ln_i0(r_alpha) - alpha.clone() * ln_i0(k0) - one_minus_alpha * ln_i0(k1)) / (alpha - T::one())
}

/// Binomial Rényi for a matched trial count `n`: `n ·` the per-trial Bernoulli
/// Rényi, `n · ln(p₀^α p₁^{1−α} + (1−p₀)^α (1−p₁)^{1−α}) / (α−1)`. A differing trial
/// count is the caller's out-of-family branch.
pub fn renyi_binomial<T: Real>(trials: T, p_self: T, p_other: T, alpha: T) -> T {
    let one = T::one();
    let one_minus_alpha = one.clone() - alpha.clone();
    let term =
        |p: T, q: T| (alpha.clone() * p.ln()).exp() * (one_minus_alpha.clone() * q.ln()).exp();
    let s = term(p_self.clone(), p_other.clone()) + term(one.clone() - p_self, one - p_other);
    trials * s.ln() / (alpha - T::one())
}

/// Negative-binomial Rényi interior for matched successes `r`
/// (finite when `w = (1−p₀)^α (1−p₁)^{1−α} < 1`):
/// `r · (α ln p₀ + (1−α) ln p₁ − ln(1 − w)) / (α−1)`. The convergence test is the
/// caller's branch.
pub fn renyi_negative_binomial_on_support<T: Real>(
    successes: T,
    p_self: T,
    p_other: T,
    alpha: T,
) -> T {
    let one = T::one();
    let one_minus_alpha = one.clone() - alpha.clone();
    let w = (alpha.clone() * (one.clone() - p_self.clone()).ln()).exp()
        * (one_minus_alpha.clone() * (one.clone() - p_other.clone()).ln()).exp();
    successes * (alpha.clone() * p_self.ln() + one_minus_alpha * p_other.ln() - (one - w).ln())
        / (alpha - T::one())
}

/// Wald / inverse-Gaussian Rényi interior (mean µ, shape λ; finite when
/// `la = α λ₀ + (1−α) λ₁ > 0` and `aa = α λ₀/µ₀² + (1−α) λ₁/µ₁² > 0`):
/// `[½(α ln λ₀ + (1−α) ln λ₁) + α λ₀/µ₀ + (1−α) λ₁/µ₁ − ½ ln la − √(aa·la)] / (α−1)`.
/// The convergence test is the caller's branch.
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

/// Uniform Rényi interior: `ln(overlap · w₀^{−α} · w₁^{−(1−α)}) / (α−1)`, `w = b − a`.
/// The `+∞` boundary (`α>1` outside containment, or disjoint support) is the caller's
/// branch.
pub fn renyi_uniform_on_support<T: Real>(
    lo_self: T,
    hi_self: T,
    lo_other: T,
    hi_other: T,
    alpha: T,
) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone()))
        .max(T::zero());
    let width_self = hi_self - lo_self;
    let width_other = hi_other - lo_other;
    let one_minus_alpha = T::one() - alpha.clone();
    let s = overlap
        * (-alpha.clone() * width_self.ln()).exp()
        * (-one_minus_alpha * width_other.ln()).exp();
    s.ln() / (alpha - T::one())
}

/// Discrete-uniform Rényi interior: `ln(overlap · n₀^{−α} · n₁^{−(1−α)}) / (α−1)`,
/// integer overlap and `n = b − a + 1`. The `+∞` boundary is the caller's branch.
pub fn renyi_discrete_uniform_on_support<T: Real>(
    lo_self: T,
    hi_self: T,
    lo_other: T,
    hi_other: T,
    alpha: T,
) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone())
        + T::one())
    .max(T::zero());
    let n_self = hi_self - lo_self + T::one();
    let n_other = hi_other - lo_other + T::one();
    let one_minus_alpha = T::one() - alpha.clone();
    let s =
        overlap * (-alpha.clone() * n_self.ln()).exp() * (-one_minus_alpha * n_other.ln()).exp();
    s.ln() / (alpha - T::one())
}

// --- closed-form total variation ---

/// Bernoulli total variation: `|p₀ − p₁|`.
pub fn total_variation_bernoulli<T: Real>(p_self: T, p_other: T) -> T {
    (p_self - p_other).abs()
}
