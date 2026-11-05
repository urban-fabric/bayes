use euclid_numeric::Real;

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

/// Bernoulli excess kurtosis `(1 − 6p(1−p)) / (p(1−p))`.
pub fn bernoulli_excess_kurtosis<T: Real>(p: T) -> T {
    let pq = p.clone() * (T::one() - p);
    (T::one() - T::scalar(6.0) * pq.clone()) / pq
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
/// Identical distributions give exactly `0`, including at the boundary.
///
/// `χ²(p ‖ p)` is zero by definition, but the formula is `(p−q)²/(q(1−q))` and
/// at `p = q = 0` both parts vanish: `chi_squared_bernoulli(0, 0)` was `NaN`,
/// a degenerate distribution reporting infinite disagreement with itself.
/// Selecting on the numerator fixes it without disturbing the genuinely
/// divergent case — `χ²(½ ‖ 0)` still has a zero denominator and a non-zero
/// numerator, so it is still `+inf`.
pub fn chi_squared_bernoulli<T: Real>(p_self: T, p_other: T) -> T {
    let diff = p_self - p_other.clone();
    let identical = diff.clone().abs().le(&T::zero());
    let value = diff.clone() * diff / (p_other.clone() * (T::one() - p_other));
    T::select(&identical, T::zero(), value)
}

/// Bernoulli Renyi divergence.
///
/// A category with `p = 0` contributes **nothing**, which is the same
/// convention the categorical form in `bayes-divergence` already applies. It
/// has to be stated rather than left to the arithmetic: the term is
/// `p^α q^{1−α}`, and at `p = q = 0` with `α = 2` that is `0 · ∞`, so
/// `renyi_bernoulli(0, 0, 2)` was `NaN` — a point mass reporting an undefined
/// divergence from itself.
///
/// The genuinely infinite case is untouched. With `p > 0`, `q = 0` and `α > 1`
/// the term is `p^α · 0^{negative} = ∞`, which is right: `q` rules out
/// something `p` asserts.
pub fn renyi_bernoulli<T: Real>(p_self: T, p_other: T, alpha: T) -> T {
    let one = T::one();
    let one_minus_alpha = one.clone() - alpha.clone();
    let term = |p: T, q: T| {
        let present = p.gt(&T::zero());
        let safe_p = T::select(&present, p, T::one());
        let value = (alpha.clone() * safe_p.ln()).exp() * (one_minus_alpha.clone() * q.ln()).exp();
        T::select(&present, value, T::zero())
    };
    let s = term(p_self.clone(), p_other.clone()) + term(one.clone() - p_self, one - p_other);
    s.ln() / (alpha - T::one())
}

/// Bernoulli total variation.
pub fn total_variation_bernoulli<T: Real>(p_self: T, p_other: T) -> T {
    (p_self - p_other).abs()
}
