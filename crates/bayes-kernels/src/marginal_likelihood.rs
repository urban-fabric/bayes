use super::*;

// --- conjugate posterior-predictives and log marginal likelihoods ---
//
// Closed-form Bayesian evidence and posterior-predictives. The lazy `ln_gamma`
// supplies every `ln B`, `ln k!`, and `ln Γ`, so no normalizer is threaded in; each
// is branch-free over sufficient statistics or a single future datum.

/// Beta-Bernoulli posterior-predictive log-probability for a relaxed label
/// `y ∈ [0, 1]`: `y·ln(α/(α+β)) + (1−y)·ln(β/(α+β))`.
pub fn beta_bernoulli_log_predictive<T: LazyReal>(y: T, alpha: T, beta: T) -> T {
    let total = alpha.clone() + beta.clone();
    y.clone() * (alpha / total.clone()).ln() + (T::one() - y) * (beta / total).ln()
}

/// Beta-Bernoulli log marginal likelihood for `s` successes and `f` failures:
/// `ln B(α+s, β+f) − ln B(α, β)`.
pub fn beta_bernoulli_log_marginal_likelihood<T: LazyReal>(
    alpha: T,
    beta: T,
    successes: T,
    failures: T,
) -> T {
    let ln_beta = |a: T, b: T| a.clone().ln_gamma() + b.clone().ln_gamma() - (a + b).ln_gamma();
    ln_beta(alpha.clone() + successes, beta.clone() + failures) - ln_beta(alpha, beta)
}

/// Gamma-Poisson (negative-binomial) posterior-predictive log-mass for a `count`
/// over `exposure`:
/// `ln Γ(α+k) − ln Γ(α) − ln k! + α·ln β + k·ln e − (α+k)·ln(β+e)`.
pub fn gamma_poisson_log_predictive<T: LazyReal>(count: T, shape: T, rate: T, exposure: T) -> T {
    (shape.clone() + count.clone()).ln_gamma()
        - shape.clone().ln_gamma()
        - (count.clone() + T::one()).ln_gamma()
        + shape.clone() * rate.clone().ln()
        + count.clone() * exposure.clone().ln()
        - (shape + count) * (rate + exposure).ln()
}

/// Gamma-Exponential posterior-predictive log-density (a Lomax) on `x ≥ 0`:
/// `ln α + α·ln β − (α+1)·ln(β + x)`.
pub fn gamma_exponential_log_predictive<T: LazyReal>(x: T, shape: T, rate: T) -> T {
    shape.clone().ln() + shape.clone() * rate.clone().ln() - (shape + T::one()) * (rate + x).ln()
}

/// Normal log marginal likelihood with known observation variance and a Normal
/// prior on the mean, from the sufficient statistics `(n, Σx, Σx²)`.
///
/// The Gaussian integral from completing the square in `µ`:
/// `−½n·ln(2π σ²) + ½(ln λ₀ − ln λₙ) − ½·quad`, with `λ₀ = 1/τ₀²`, `λₙ = λ₀ + n/σ²`,
/// `ηₙ = λ₀µ₀ + Σx/σ²`, and `quad = Σx²/σ² + λ₀µ₀² − ηₙ²/λₙ`.
pub fn normal_known_variance_log_marginal_likelihood<T: LazyReal>(
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

/// Gamma-Poisson log marginal likelihood over count sufficient statistics
/// `(Σk, n)`, the rate integrated out:
/// `α·ln β − ln Γ(α) − Σ ln kᵢ! + ln Γ(α+Σk) − (α+Σk)·ln(β+n)`.
///
/// `ln_factorial_sum = Σ ln kᵢ!` is the data-combinatorial term and is caller
/// supplied (like the `ln_binomial` in [`binomial_log_pmf`]). For Poisson exposure
/// offsets, pass `n = Σ eᵢ` and add the caller's `Σ kᵢ·ln eᵢ` to the result.
pub fn gamma_poisson_log_marginal_likelihood<T: LazyReal>(
    shape: T,
    rate: T,
    count_sum: T,
    count_n: T,
    ln_factorial_sum: T,
) -> T {
    shape.clone() * rate.clone().ln() - shape.clone().ln_gamma() - ln_factorial_sum
        + (shape.clone() + count_sum.clone()).ln_gamma()
        - (shape + count_sum) * (rate + count_n).ln()
}

/// Gamma-Exponential log marginal likelihood over sufficient statistics `(n, Σx)`,
/// the rate integrated out:
/// `α·ln β − ln Γ(α) + ln Γ(α+n) − (α+n)·ln(β+Σx)`.
///
/// The same form carries the right-censored/exposure case: pass the failure count
/// for `n` and the total exposure for `sum`.
pub fn gamma_exponential_log_marginal_likelihood<T: LazyReal>(
    shape: T,
    rate: T,
    count_n: T,
    sum: T,
) -> T {
    shape.clone() * rate.clone().ln() - shape.clone().ln_gamma()
        + (shape.clone() + count_n.clone()).ln_gamma()
        - (shape + count_n) * (rate + sum).ln()
}

/// Beta-Binomial posterior-predictive log-probability for one future fixed-trial
/// binomial observation (`successes`, `failures = trials − successes`):
/// `ln C(trials, successes) + ln B(α+successes, β+failures) − ln B(α, β)`.
///
/// `ln_binomial = ln C(trials, successes)` is caller supplied.
pub fn beta_binomial_log_predictive<T: LazyReal>(
    alpha: T,
    beta: T,
    successes: T,
    failures: T,
    ln_binomial: T,
) -> T {
    let ln_beta = |a: T, b: T| a.clone().ln_gamma() + b.clone().ln_gamma() - (a + b).ln_gamma();
    ln_binomial + ln_beta(alpha.clone() + successes, beta.clone() + failures) - ln_beta(alpha, beta)
}

/// Normal-Inverse-Gamma log marginal likelihood (both mean and variance integrated
/// out) over sufficient statistics `(n, Σx, Σx²)` and the prior `(µ₀, λ₀, α₀, β₀)`:
/// `ln Γ(αₙ) − ln Γ(α₀) + α₀·ln β₀ − αₙ·ln βₙ + ½(ln λ₀ − ln λₙ) − ½n·ln(2π)`.
///
/// The posterior is recovered in the additive natural coordinates
/// `(λ, λµ, 2α, 2β+λµ²)`: `λₙ = λ₀+n`, `αₙ = α₀+n/2`, `λµₙ = λ₀µ₀+Σx`,
/// `2βₙ = (2β₀+λ₀µ₀²+Σx²) − λµₙ²/λₙ`. This is also the **Normal-Gamma** evidence —
/// the precision-parametrized twin shares the identical closed form.
pub fn normal_inverse_gamma_log_marginal_likelihood<T: LazyReal>(
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
