use crate::families::common::gaussian_bc;
use euclid_numeric::Real;

pub use crate::primitives::{
    complex_normal_log_prob_with_normalizer, complex_residual_norm_sqr, normal_log_kernel,
    normal_log_prob, normal_log_prob_with_constants, normal_log_prob_with_normalizer,
};

/// Normal log-density over the lazy real surface.
pub fn normal_log_pdf<T: Real>(x: T, mean: T, std_dev: T) -> T {
    normal_log_prob(x, mean, std_dev)
}

/// Normal CDF `Phi((x - mu) / sigma)`.
pub fn normal_cdf<T: Real>(x: T, mean: T, std_dev: T) -> T {
    T::half() * (-(x - mean) / (std_dev * T::sqrt_2())).erfc()
}

/// Normal log-CDF `ln Phi((x - mu) / sigma)` on the real line.
pub fn normal_ln_cdf<T: Real>(x: T, mean: T, std_dev: T) -> T {
    let z = (x - mean) / std_dev;
    let t = -(z.clone()) * T::frac_1_sqrt_2();
    let s = z.clone() * T::frac_1_sqrt_2();
    let left = -T::ln_2() + t.clone().erfcx().ln() - t.clone() * t;
    let right_ln_sf = -T::ln_2() + s.clone().erfcx().ln() - s.clone() * s;
    T::select(&z.le(&T::zero()), left, right_ln_sf.ln_1m_exp())
}

/// Normal log-survival `ln(1 - Phi((x - mu) / sigma))`.
pub fn normal_ln_sf<T: Real>(x: T, mean: T, std_dev: T) -> T {
    let z = (x - mean) / std_dev;
    let s = z.clone() * T::frac_1_sqrt_2();
    let t = -(z.clone()) * T::frac_1_sqrt_2();
    let right = -T::ln_2() + s.clone().erfcx().ln() - s.clone() * s;
    let left_ln_cdf = -T::ln_2() + t.clone().erfcx().ln() - t.clone() * t;
    T::select(&z.ge(&T::zero()), right, left_ln_cdf.ln_1m_exp())
}

/// Normal quantile, via `erfc_inv` on whichever tail keeps its argument exact.
///
/// # Why not `erf_inv(2p − 1)`
///
/// The textbook form destroys its own input. At `p = 1e-20` the expression
/// `2p − 1` is **exactly** `−1.0`, because `1 − 2e-20` is not representable —
/// so `erf_inv` is handed the endpoint of its domain and returns `−inf` for a
/// quantile whose true value is `−9.26`. The damage starts far earlier than the
/// hard failure: by `p = 1e-6` only about ten digits of the deviation from `−1`
/// survive the subtraction, and `erf_inv` is steep there, which showed up as a
/// `4e-7` error in the round trip `z → sf → isf` at `z = 6`.
///
/// `Φ⁻¹(p) = −√2·erfc_inv(2p)` is the same function without the subtraction.
/// Doubling is exact for every representable `p` — it only moves the exponent —
/// so the lower tail is good down to about `1e-308`. The upper tail is taken as
/// `+√2·erfc_inv(2(1 − p))`, which is the best available there: once a caller
/// has rounded `p` to `1 − 1e-20`, the information is already gone, and the way
/// to ask that question is the inverse *survival*, with the small argument
/// kept intact.
pub fn normal_quantile<T: Real>(p: T, mean: T, std_dev: T) -> T {
    let lower = -(T::two() * p.clone()).erfc_inv();
    let upper = (T::two() * (T::one() - p.clone())).erfc_inv();
    mean + std_dev * T::sqrt_2() * T::select(&p.le(&T::half()), lower, upper)
}

/// Normal entropy `0.5 * ln(2*pi*e*std_dev^2)`.
pub fn normal_entropy<T: Real>(std_dev: T) -> T {
    T::half() * (T::tau() * T::e() * std_dev.clone() * std_dev).ln()
}

/// Gaussian KL `D(N(m0,s0) || N(m1,s1))`.
pub fn kl_normal<T: Real>(mean_self: T, std_self: T, mean_other: T, std_other: T) -> T {
    let dm = mean_self - mean_other;
    (std_other.clone() / std_self.clone()).ln()
        + (std_self.clone() * std_self + dm.clone() * dm)
            / (T::two() * std_other.clone() * std_other)
        - T::half()
}

/// Gaussian squared Hellinger distance.
pub fn hellinger_sq_normal<T: Real>(mean_self: T, std_self: T, mean_other: T, std_other: T) -> T {
    crate::hellinger_sq_from_bc(gaussian_bc(mean_self, std_self, mean_other, std_other))
}

/// Gaussian chi-squared divergence interior, valid when the support/convergence guard holds.
pub fn chi_squared_normal_on_support<T: Real>(
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

/// Gaussian Renyi divergence interior.
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

/// Normal log marginal likelihood with known observation variance and a Normal prior.
pub fn normal_known_variance_log_marginal_likelihood<T: Real>(
    count: T,
    sum: T,
    sum_sq: T,
    prior_mean: T,
    prior_std: T,
    obs_std: T,
) -> T {
    let sigma2 = obs_std.clone() * obs_std;
    let lambda0 = T::one() / (prior_std.clone() * prior_std);
    let eta_n = lambda0.clone() * prior_mean.clone() + sum / sigma2.clone();
    let lambda_n = lambda0.clone() + count.clone() / sigma2.clone();
    let quad = sum_sq / sigma2.clone() + lambda0.clone() * prior_mean.clone() * prior_mean
        - eta_n.clone() * eta_n / lambda_n.clone();
    -T::half() * count * (T::tau() * sigma2).ln() + T::half() * (lambda0.ln() - lambda_n.ln())
        - T::half() * quad
}

/// Normal-Inverse-Gamma log marginal likelihood.
pub fn normal_inverse_gamma_log_marginal_likelihood<T: Real>(
    count: T,
    sum: T,
    sum_sq: T,
    prior_mean: T,
    prior_lambda: T,
    prior_alpha: T,
    prior_beta: T,
) -> T {
    let lambda_n = prior_lambda.clone() + count.clone();
    let lambda_mu_n = prior_lambda.clone() * prior_mean.clone() + sum;
    let alpha_n = prior_alpha.clone() + T::half() * count.clone();
    let sum_sq_nat = T::two() * prior_beta.clone()
        + prior_lambda.clone() * prior_mean.clone() * prior_mean
        + sum_sq;
    let beta_n = T::half() * (sum_sq_nat - lambda_mu_n.clone() * lambda_mu_n / lambda_n.clone());
    alpha_n.clone().ln_gamma() - prior_alpha.clone().ln_gamma() + prior_alpha * prior_beta.ln()
        - alpha_n * beta_n.ln()
        + T::half() * (prior_lambda.ln() - lambda_n.ln())
        - T::half() * count * T::ln_tau()
}
