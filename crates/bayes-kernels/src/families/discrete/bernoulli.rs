use euclid::numeric::Real;

pub use crate::families::continuous::beta::{
    beta_bernoulli_log_marginal_likelihood, beta_bernoulli_log_predictive,
};

/// Bernoulli log-mass for a probability parameter and relaxed label.
pub fn bernoulli_log_pmf<T: Real>(y: T, p: T) -> T {
    let one = T::one();
    y.clone() * p.clone().ln() + (one.clone() - y) * (one - p).ln()
}

/// Bernoulli log-mass parameterized by a logit.
pub fn bernoulli_logit_log_pmf<T: Real>(y: T, logit: T) -> T {
    y * logit.clone() - logit.ln_1p_exp()
}

/// Bernoulli skewness.
pub fn bernoulli_skewness<T: Real>(p: T) -> T {
    let q = T::one() - p.clone();
    (q.clone() - p.clone()) / (p * q).sqrt()
}

/// Bernoulli KL on `0 < p < 1`.
pub fn kl_bernoulli_on_unit<T: Real>(p_self: T, p_other: T) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let q_other = one - p_other.clone();
    p_self.clone() * (p_self / p_other).ln() + q_self.clone() * (q_self / q_other).ln()
}

/// Bernoulli squared Hellinger distance.
pub fn hellinger_sq_bernoulli<T: Real>(p_self: T, p_other: T) -> T {
    let one = T::one();
    let bc = (p_self.clone() * p_other.clone()).sqrt()
        + ((one.clone() - p_self) * (one - p_other)).sqrt();
    crate::hellinger_sq_from_bc(bc)
}

/// Bernoulli chi-squared divergence.
pub fn chi_squared_bernoulli<T: Real>(p_self: T, p_other: T) -> T {
    let diff = p_self - p_other.clone();
    diff.clone() * diff / (p_other.clone() * (T::one() - p_other))
}

/// Bernoulli Renyi divergence.
pub fn renyi_bernoulli<T: Real>(p_self: T, p_other: T, alpha: T) -> T {
    let one = T::one();
    let one_minus_alpha = one.clone() - alpha.clone();
    let term =
        |p: T, q: T| (alpha.clone() * p.ln()).exp() * (one_minus_alpha.clone() * q.ln()).exp();
    let s = term(p_self.clone(), p_other.clone()) + term(one.clone() - p_self, one - p_other);
    s.ln() / (alpha - T::one())
}

/// Bernoulli total variation.
pub fn total_variation_bernoulli<T: Real>(p_self: T, p_other: T) -> T {
    (p_self - p_other).abs()
}
