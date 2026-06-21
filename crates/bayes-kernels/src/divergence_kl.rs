use super::*;

// --- closed-form Kullback–Leibler divergences `D(self ‖ other)` ---
//
// Same-family closed forms in nats. The gamma/beta-shaped ones reach for the lazy
// `ln_gamma`/`digamma` rather than a caller-supplied normalizer, so no normalizer
// is threaded through. Forms whose divergence is `+∞` outside a support-containment
// region carry an `_on_support` suffix and the interior closed form only — the
// containment test is the caller's branch — which keeps them traceable.

/// Gaussian KL `D(N(m₀,s₀) ‖ N(m₁,s₁)) = ln(s₁/s₀) + (s₀² + (m₀−m₁)²)/(2 s₁²) − ½`.
///
/// Also the LogNormal KL on the log-space `(µ, σ)`, and the HalfNormal KL with both
/// means `0`.
pub fn kl_normal<T: LazyReal>(mean_self: T, std_self: T, mean_other: T, std_other: T) -> T {
    let dm = mean_self - mean_other;
    (std_other.clone() / std_self.clone()).ln()
        + (std_self.clone() * std_self + dm.clone() * dm)
            / (T::two() * std_other.clone() * std_other)
        - T::half()
}

/// Exponential KL: `ln(λ₀/λ₁) + λ₁/λ₀ − 1`.
pub fn kl_exponential<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    (rate_self.clone() / rate_other.clone()).ln() + rate_other / rate_self - T::one()
}

/// Poisson KL: `λ₀·ln(λ₀/λ₁) − λ₀ + λ₁`.
pub fn kl_poisson<T: LazyReal>(rate_self: T, rate_other: T) -> T {
    rate_self.clone() * (rate_self.clone() / rate_other.clone()).ln() - rate_self + rate_other
}

/// Geometric KL: `ln(p₀/p₁) + ((1−p₀)/p₀)·ln((1−p₀)/(1−p₁))`.
pub fn kl_geometric<T: LazyReal>(p_self: T, p_other: T) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let q_other = one - p_other.clone();
    (p_self.clone() / p_other).ln() + (q_self.clone() / p_self) * (q_self / q_other).ln()
}

/// Bernoulli KL on `0 < p < 1`: `p₀·ln(p₀/p₁) + (1−p₀)·ln((1−p₀)/(1−p₁))`.
pub fn kl_bernoulli_on_unit<T: LazyReal>(p_self: T, p_other: T) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let q_other = one - p_other.clone();
    p_self.clone() * (p_self / p_other).ln() + q_self.clone() * (q_self / q_other).ln()
}

/// Negative-binomial KL for a matched number of successes `r`:
/// `r·ln(p₀/p₁) + r·((1−p₀)/p₀)·ln((1−p₀)/(1−p₁))`.
pub fn kl_negative_binomial<T: LazyReal>(successes: T, p_self: T, p_other: T) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let q_other = one - p_other.clone();
    successes.clone() * (p_self.clone() / p_other).ln()
        + successes * (q_self.clone() / p_self) * (q_self / q_other).ln()
}

/// Binomial KL for a matched trial count `n`: `n · D_KL(Bernoulli(p₀) ‖ Bernoulli(p₁))`
/// — the binomial coefficient cancels in the log-ratio. A differing trial count is
/// out of family and is the caller's branch.
pub fn kl_binomial<T: LazyReal>(trials: T, p_self: T, p_other: T) -> T {
    trials * kl_bernoulli_on_unit(p_self, p_other)
}

/// Cauchy KL: `ln[((γ₀+γ₁)² + (x₀−x₁)²) / (4 γ₀ γ₁)]`.
pub fn kl_cauchy<T: LazyReal>(loc_self: T, scale_self: T, loc_other: T, scale_other: T) -> T {
    let scale_sum = scale_self.clone() + scale_other.clone();
    let dm = loc_self - loc_other;
    ((scale_sum.clone() * scale_sum + dm.clone() * dm)
        / (T::scalar(4.0) * scale_self * scale_other))
        .ln()
}

/// Rayleigh KL: `2·ln(σ₁/σ₀) + (σ₀/σ₁)² − 1`.
pub fn kl_rayleigh<T: LazyReal>(scale_self: T, scale_other: T) -> T {
    let ratio = scale_self.clone() / scale_other.clone();
    T::two() * (scale_other / scale_self).ln() + ratio.clone() * ratio - T::one()
}

/// Laplace KL: `ln(b₁/b₀) + (b₀·exp(−d/b₀) + d)/b₁ − 1`, `d = |m₀ − m₁|`.
pub fn kl_laplace<T: LazyReal>(loc_self: T, scale_self: T, loc_other: T, scale_other: T) -> T {
    let d = (loc_self - loc_other).abs();
    (scale_other.clone() / scale_self.clone()).ln()
        + (scale_self.clone() * (-(d.clone() / scale_self)).exp() + d) / scale_other
        - T::one()
}

/// Wald / inverse-Gaussian KL (mean µ, shape λ):
/// `½ln(λ₀/λ₁) + λ₁(µ₀−2µ₁)/(2µ₁²) + ½λ₁(1/µ₀ + 1/λ₀) − ½`.
pub fn kl_wald<T: LazyReal>(mean_self: T, shape_self: T, mean_other: T, shape_other: T) -> T {
    let one = T::one();
    T::half() * (shape_self.clone() / shape_other.clone()).ln()
        + shape_other.clone() * (mean_self.clone() - T::two() * mean_other.clone())
            / (T::two() * mean_other.clone() * mean_other)
        + T::half() * shape_other * (one.clone() / mean_self + one / shape_self)
        - T::half()
}

/// Gamma KL: `(a₀−a₁)·ψ(a₀) − ln Γ(a₀) + ln Γ(a₁) + a₁·ln(b₀/b₁) + a₀·(b₁−b₀)/b₀`.
///
/// Also the ChiSquared KL (shape `k/2`, rate `½`) and the InverseGamma KL (shape,
/// rate ≡ scale), both of which reduce to this form.
pub fn kl_gamma<T: LazyReal>(shape_self: T, rate_self: T, shape_other: T, rate_other: T) -> T {
    (shape_self.clone() - shape_other.clone()) * shape_self.clone().digamma()
        - shape_self.clone().ln_gamma()
        + shape_other.clone().ln_gamma()
        + shape_other * (rate_self.clone().ln() - rate_other.clone().ln())
        + shape_self * (rate_other - rate_self.clone()) / rate_self
}

/// Beta KL:
/// `ln B(a₁,b₁) − ln B(a₀,b₀) + (a₀−a₁)ψ(a₀) + (b₀−b₁)ψ(b₀) + (a₁−a₀+b₁−b₀)ψ(a₀+b₀)`.
pub fn kl_beta<T: LazyReal>(alpha_self: T, beta_self: T, alpha_other: T, beta_other: T) -> T {
    let sum_self = alpha_self.clone() + beta_self.clone();
    let ln_beta_self =
        alpha_self.clone().ln_gamma() + beta_self.clone().ln_gamma() - sum_self.clone().ln_gamma();
    let ln_beta_other = alpha_other.clone().ln_gamma() + beta_other.clone().ln_gamma()
        - (alpha_other.clone() + beta_other.clone()).ln_gamma();
    let psi_alpha = alpha_self.clone().digamma();
    let psi_beta = beta_self.clone().digamma();
    let psi_sum = sum_self.digamma();
    ln_beta_other - ln_beta_self
        + (alpha_self.clone() - alpha_other.clone()) * psi_alpha
        + (beta_self.clone() - beta_other.clone()) * psi_beta
        + (alpha_other - alpha_self + beta_other - beta_self) * psi_sum
}

/// Weibull KL (Bauckhage's closed form, shape `k`, scale `λ`):
/// `(ln k₀ − k₀ln λ₀) − (ln k₁ − k₁ln λ₁) + (k₀−k₁)(ln λ₀ − γ/k₀)
///  + (λ₀/λ₁)^{k₁}·Γ(1 + k₁/k₀) − 1`.
///
/// Also the Frechet KL through the `x ↦ 1/x` bijection (scales inverted).
pub fn kl_weibull<T: LazyReal>(shape_self: T, scale_self: T, shape_other: T, scale_other: T) -> T {
    let (k0, l0, k1, l1) = (shape_self, scale_self, shape_other, scale_other);
    let term0 = k0.clone().ln() - k0.clone() * l0.clone().ln();
    let term1 = k1.clone().ln() - k1.clone() * l1.clone().ln();
    let cross = (k0.clone() - k1.clone()) * (l0.clone().ln() - T::euler_gamma() / k0.clone());
    let ratio_pow = (k1.clone() * (l0 / l1).ln()).exp();
    let gamma_term = (T::one() + k1 / k0).ln_gamma().exp();
    term0 - term1 + cross + ratio_pow * gamma_term - T::one()
}

/// Gumbel KL (location µ, scale β):
/// `ln(β₁/β₀) + γ(β₀/β₁ − 1) + (µ₀−µ₁)/β₁ + exp((µ₁−µ₀)/β₁)·Γ(1 + β₀/β₁) − 1`.
pub fn kl_gumbel<T: LazyReal>(loc_self: T, scale_self: T, loc_other: T, scale_other: T) -> T {
    let (mu0, b0, mu1, b1) = (loc_self, scale_self, loc_other, scale_other);
    (b1.clone() / b0.clone()).ln()
        + T::euler_gamma() * (b0.clone() / b1.clone() - T::one())
        + (mu0.clone() - mu1.clone()) / b1.clone()
        + ((mu1 - mu0) / b1.clone()).exp() * (T::one() + b0 / b1).ln_gamma().exp()
        - T::one()
}

/// Pareto KL on nested support (`xₘ₀ ≥ xₘ₁`):
/// `ln(α₀/α₁) + α₁·ln(xₘ₀/xₘ₁) + α₁/α₀ − 1`.
pub fn kl_pareto_on_support<T: LazyReal>(
    minimum_self: T,
    shape_self: T,
    minimum_other: T,
    shape_other: T,
) -> T {
    let (xm0, a0, xm1, a1) = (minimum_self, shape_self, minimum_other, shape_other);
    (a0.clone() / a1.clone()).ln() + a1.clone() * (xm0 / xm1).ln() + a1 / a0 - T::one()
}

/// von Mises KL: `ln I₀(κ₁) − ln I₀(κ₀) + (I₁/I₀)(κ₀)·(κ₀ − κ₁·cos Δµ)`.
///
/// Reachable now that `cos` and the scaled Bessel `I₀ᵉ`/`I₁ᵉ` are on the lazy
/// surface: `ln I₀ = ln I₀ᵉ + |·|` and `I₁/I₀ = I₁ᵉ/I₀ᵉ`.
pub fn kl_von_mises<T: LazyReal>(mean_self: T, kappa_self: T, mean_other: T, kappa_other: T) -> T {
    let (mu0, k0, mu1, k1) = (mean_self, kappa_self, mean_other, kappa_other);
    let resultant = k0.clone().bessel_i1e() / k0.clone().bessel_i0e();
    ln_i0(k1.clone()) - ln_i0(k0.clone()) + resultant * (k0 - k1 * (mu0 - mu1).cos())
}

/// Uniform KL on nested support (`[a₀,b₀] ⊆ [a₁,b₁]`): `ln(w₁/w₀)`, `w = b − a`. The
/// containment test (`+∞` otherwise) is the caller's branch.
pub fn kl_uniform_on_support<T: LazyReal>(width_self: T, width_other: T) -> T {
    (width_other / width_self).ln()
}

/// Discrete-uniform KL on nested support: `ln(n₁/n₀)`, `n` the support count.
pub fn kl_discrete_uniform_on_support<T: LazyReal>(count_self: T, count_other: T) -> T {
    (count_other / count_self).ln()
}
