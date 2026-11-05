use crate::families::common::ln_beta;
use euclid_numeric::Real;

/// Beta log-density on the **closed** `0 ≤ x ≤ 1`.
///
/// Written with [`Real::xlogy`] rather than as `(α−1)·ln(x)`, which is what extends it to the
/// closed interval. At `x = 0` with `α = 1` the direct form is `0 · (−∞) = NaN`, so the endpoints
/// had to be masked away — and the mask returned a density of *zero* there for every shape, which
/// is only the right limit when `α > 1`. At `α = 1` the density at zero is `β`, not `0`, so
/// `Beta(1, 1)` disagreed with the `Uniform(0, 1)` it is supposed to equal.
///
/// `xlogy` carries the `0·ln 0 = 0` convention, which delivers all three limits without a branch:
/// `+∞` for `α < 1`, `ln β` for `α = 1`, `−∞` for `α > 1`.
pub fn beta_log_pdf_on_unit<T: Real>(x: T, alpha: T, beta: T, ln_beta_ab: T) -> T {
    let one = T::one();
    (alpha - one.clone()).xlogy(x.clone()) + (beta - one.clone()).xlogy(one - x) - ln_beta_ab
}

/// Beta CDF on `0 <= x <= 1`.
pub fn beta_cdf_on_unit<T: Real>(x: T, alpha: T, beta: T) -> T {
    x.beta_reg(alpha, beta)
}

/// Beta variance `alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))`.
pub fn beta_variance<T: Real>(alpha: T, beta: T) -> T {
    let sum = alpha.clone() + beta.clone();
    (alpha * beta) / (sum.clone() * sum.clone() * (sum + T::one()))
}

/// Beta skewness.
pub fn beta_skewness<T: Real>(alpha: T, beta: T) -> T {
    let sum = alpha.clone() + beta.clone();
    T::two() * (beta.clone() - alpha.clone()) * (sum.clone() + T::one()).sqrt()
        / ((sum + T::two()) * (alpha * beta).sqrt())
}

/// Beta excess kurtosis
/// `6[(α−β)²(α+β+1) − αβ(α+β+2)] / [αβ(α+β+2)(α+β+3)]`.
pub fn beta_excess_kurtosis<T: Real>(alpha: T, beta: T) -> T {
    let sum = alpha.clone() + beta.clone();
    let diff = alpha.clone() - beta.clone();
    let prod = alpha * beta;
    let num = T::scalar(6.0)
        * (diff.clone() * diff * (sum.clone() + T::one())
            - prod.clone() * (sum.clone() + T::two()));
    let den = prod * (sum.clone() + T::two()) * (sum + T::scalar(3.0));
    num / den
}

/// Beta entropy.
pub fn beta_entropy<T: Real>(alpha: T, beta: T) -> T {
    let one = T::one();
    let sum = alpha.clone() + beta.clone();
    let ln_beta_ab = ln_beta(alpha.clone(), beta.clone());
    ln_beta_ab
        - (alpha.clone() - one.clone()) * alpha.digamma()
        - (beta.clone() - one) * beta.digamma()
        + (sum.clone() - T::two()) * sum.digamma()
}

/// Beta KL divergence.
pub fn kl_beta<T: Real>(alpha_self: T, beta_self: T, alpha_other: T, beta_other: T) -> T {
    let sum_self = alpha_self.clone() + beta_self.clone();
    let ln_beta_self = ln_beta(alpha_self.clone(), beta_self.clone());
    let ln_beta_other = ln_beta(alpha_other.clone(), beta_other.clone());
    let psi_alpha = alpha_self.clone().digamma();
    let psi_beta = beta_self.clone().digamma();
    let psi_sum = sum_self.digamma();
    ln_beta_other - ln_beta_self
        + (alpha_self.clone() - alpha_other.clone()) * psi_alpha
        + (beta_self.clone() - beta_other.clone()) * psi_beta
        + (alpha_other - alpha_self + beta_other - beta_self) * psi_sum
}

/// Beta squared Hellinger distance.
pub fn hellinger_sq_beta<T: Real>(alpha_self: T, beta_self: T, alpha_other: T, beta_other: T) -> T {
    let am = T::half() * (alpha_self.clone() + alpha_other.clone());
    let bm = T::half() * (beta_self.clone() + beta_other.clone());
    let ln_bc = ln_beta(am, bm)
        - T::half() * (ln_beta(alpha_self, beta_self) + ln_beta(alpha_other, beta_other));
    crate::hellinger_sq_from_bc(ln_bc.exp())
}

/// Beta-Bernoulli posterior-predictive log-probability for a relaxed label.
pub fn beta_bernoulli_log_predictive<T: Real>(y: T, alpha: T, beta: T) -> T {
    let total = alpha.clone() + beta.clone();
    y.clone() * (alpha / total.clone()).ln() + (T::one() - y) * (beta / total).ln()
}

/// Beta-Bernoulli log marginal likelihood for successes and failures.
pub fn beta_bernoulli_log_marginal_likelihood<T: Real>(
    alpha: T,
    beta: T,
    successes: T,
    failures: T,
) -> T {
    ln_beta(alpha.clone() + successes, beta.clone() + failures) - ln_beta(alpha, beta)
}

/// Beta-Binomial posterior-predictive log-probability.
pub fn beta_binomial_log_predictive<T: Real>(
    alpha: T,
    beta: T,
    successes: T,
    failures: T,
    ln_binomial: T,
) -> T {
    ln_binomial + ln_beta(alpha.clone() + successes, beta.clone() + failures) - ln_beta(alpha, beta)
}
