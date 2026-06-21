use super::*;

// --- closed-form squared Hellinger distances `H² = 1 − BC` ---
//
// Each is built from the family Bhattacharyya coefficient `BC = ∫√(pq)`; the
// `(1 − BC).max(0)` only absorbs `O(ε)` rounding (`BC ≤ 1` is a theorem).

/// Gaussian Bhattacharyya coefficient `√(2 s₀ s₁/v)·exp(−¼(m₀−m₁)²/v)`, `v=s₀²+s₁²`.
fn gaussian_bc<T: LazyReal>(m0: T, s0: T, m1: T, s1: T) -> T {
    let v = s0.clone() * s0.clone() + s1.clone() * s1.clone();
    let dm = m0 - m1;
    (T::two() * s0 * s1 / v.clone()).sqrt() * (-T::scalar(0.25) * dm.clone() * dm / v).exp()
}

/// Gamma Bhattacharyya coefficient (shape `a`, rate `b`), via `ln_gamma`.
fn gamma_bc<T: LazyReal>(a0: T, b0: T, a1: T, b1: T) -> T {
    let a = T::half() * (a0.clone() + a1.clone());
    let b = T::half() * (b0.clone() + b1.clone());
    let ln_bc = a.clone().ln_gamma() + T::half() * (a0.clone() * b0.ln() + a1.clone() * b1.ln())
        - T::half() * (a0.ln_gamma() + a1.ln_gamma())
        - a * b.ln();
    ln_bc.exp()
}

/// Gaussian squared Hellinger (Normal, LogNormal on log-params, HalfNormal at m=0).
pub fn hellinger_sq_normal<T: LazyReal>(
    mean_self: T,
    std_self: T,
    mean_other: T,
    std_other: T,
) -> T {
    hellinger_sq_from_bc(gaussian_bc(mean_self, std_self, mean_other, std_other))
}

/// Exponential squared Hellinger: `BC = 2√(λ₀λ₁)/(λ₀+λ₁)`.
pub fn hellinger_sq_exponential<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    let (l0, l1) = (rate_self, rate_other);
    hellinger_sq_from_bc(T::two() * (l0.clone() * l1.clone()).sqrt() / (l0 + l1))
}

/// Poisson squared Hellinger: `BC = exp(−½(√λ₀ − √λ₁)²)`.
pub fn hellinger_sq_poisson<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    let d = rate_self.sqrt() - rate_other.sqrt();
    hellinger_sq_from_bc((-T::half() * d.clone() * d).exp())
}

/// Bernoulli squared Hellinger: `BC = √(p₀p₁) + √((1−p₀)(1−p₁))`.
pub fn hellinger_sq_bernoulli<T: LazyReal>(p_self: T, p_other: T) -> T {
    let one = T::one();
    let bc = (p_self.clone() * p_other.clone()).sqrt()
        + ((one.clone() - p_self) * (one - p_other)).sqrt();
    hellinger_sq_from_bc(bc)
}

/// Gamma squared Hellinger (Gamma, ChiSquared, InverseGamma).
pub fn hellinger_sq_gamma<T: LazyReal>(
    shape_self: T,
    rate_self: T,
    shape_other: T,
    rate_other: T,
) -> T {
    hellinger_sq_from_bc(gamma_bc(shape_self, rate_self, shape_other, rate_other))
}

/// Beta squared Hellinger:
/// `BC = exp(ln B(ā, b̄) − ½(ln B(a₀,b₀) + ln B(a₁,b₁)))`, `ā=(a₀+a₁)/2`.
pub fn hellinger_sq_beta<T: LazyReal>(
    alpha_self: T,
    beta_self: T,
    alpha_other: T,
    beta_other: T,
) -> T {
    let ln_beta = |a: T, b: T| a.clone().ln_gamma() + b.clone().ln_gamma() - (a + b).ln_gamma();
    let am = T::half() * (alpha_self.clone() + alpha_other.clone());
    let bm = T::half() * (beta_self.clone() + beta_other.clone());
    let ln_bc = ln_beta(am, bm)
        - T::half() * (ln_beta(alpha_self, beta_self) + ln_beta(alpha_other, beta_other));
    hellinger_sq_from_bc(ln_bc.exp())
}

/// Wald / inverse-Gaussian squared Hellinger (mean µ, shape λ).
pub fn hellinger_sq_wald<T: LazyReal>(
    mean_self: T,
    shape_self: T,
    mean_other: T,
    shape_other: T,
) -> T {
    let (m0, l0, m1, l1) = (mean_self, shape_self, mean_other, shape_other);
    let s = ((l0.clone() / (m0.clone() * m0.clone()) + l1.clone() / (m1.clone() * m1.clone()))
        * (l0.clone() + l1.clone()))
    .sqrt();
    let ln_bc = T::scalar(0.25) * (l0.clone() * l1.clone()).ln()
        + T::half() * (T::two() / (l0.clone() + l1.clone())).ln()
        + T::half() * (l0 / m0 + l1 / m1)
        - T::half() * s;
    hellinger_sq_from_bc(ln_bc.exp())
}

/// von Mises squared Hellinger: `BC = I₀(R)/√(I₀(κ₀)I₀(κ₁))`,
/// `R = ½√(κ₀² + κ₁² + 2κ₀κ₁ cos Δµ)`.
pub fn hellinger_sq_von_mises<T: LazyReal>(
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
    let ln_bc = ln_i0(r) - T::half() * (ln_i0(k0) + ln_i0(k1));
    hellinger_sq_from_bc(ln_bc.exp())
}

/// Binomial squared Hellinger for a matched trial count `n`: the Bhattacharyya
/// coefficient factorizes over the `n` iid Bernoulli trials, `BC = BC_Bernoulli^n`,
/// so `H² = 1 − BC`. A differing trial count is the caller's out-of-family branch.
pub fn hellinger_sq_binomial<T: LazyReal>(trials: T, p_self: T, p_other: T) -> T {
    let one = T::one();
    let bc_bernoulli = (p_self.clone() * p_other.clone()).sqrt()
        + ((one.clone() - p_self) * (one - p_other)).sqrt();
    hellinger_sq_from_bc((trials * bc_bernoulli.ln()).exp())
}

/// Negative-binomial squared Hellinger for a matched number of successes `r`:
/// `BC = [√(p₀p₁) / (1 − √((1−p₀)(1−p₁)))]^r`, so `H² = 1 − BC`.
pub fn hellinger_sq_negative_binomial<T: LazyReal>(successes: T, p_self: T, p_other: T) -> T {
    let one = T::one();
    let bc = (p_self.clone() * p_other.clone()).sqrt()
        / (one.clone() - ((one.clone() - p_self) * (one - p_other)).sqrt());
    hellinger_sq_from_bc((successes * bc.ln()).exp())
}

/// Uniform squared Hellinger: `BC = overlap / √(w₀·w₁)`, the overlap
/// `max(min(b₀,b₁) − max(a₀,a₁), 0)` and widths `w = b − a`. Bounded, so valid for
/// any two uniforms (the clamp absorbs disjoint support).
pub fn hellinger_sq_uniform<T: LazyReal>(lo_self: T, hi_self: T, lo_other: T, hi_other: T) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone()))
        .max(T::zero());
    let width_self = hi_self - lo_self;
    let width_other = hi_other - lo_other;
    hellinger_sq_from_bc(overlap / (width_self * width_other).sqrt())
}

/// Discrete-uniform squared Hellinger: `BC = overlap / √(n₀·n₁)`, integer overlap
/// `max(min(b₀,b₁) − max(a₀,a₁) + 1, 0)` and counts `n = b − a + 1`.
pub fn hellinger_sq_discrete_uniform<T: LazyReal>(
    lo_self: T,
    hi_self: T,
    lo_other: T,
    hi_other: T,
) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone())
        + T::one())
    .max(T::zero());
    let n_self = hi_self - lo_self + T::one();
    let n_other = hi_other - lo_other + T::one();
    hellinger_sq_from_bc(overlap / (n_self * n_other).sqrt())
}
