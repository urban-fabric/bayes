use super::*;

// --- closed-form CDFs over the lazy analysis surface ---
//
// `_on_support` kernels carry only the interior closed form: a CDF is `0` below its
// support and `1` above it, and that indicator is the caller's branch. The ones
// reaching for `erf` / the regularized incomplete gamma / beta were unreachable
// before the special functions joined the lazy surface; each lowers to the matching
// tensor op on a backend.

/// Normal CDF `Φ((x − µ)/σ) = ½·erfc(−(x − µ)/(σ·√2))`, on the whole real line.
pub fn normal_cdf<T: LazyReal>(x: T, mean: T, std_dev: T) -> T {
    T::half() * (-(x - mean) / (std_dev * T::sqrt_2())).erfc()
}

/// Half-normal CDF on `x ≥ 0`: `erf(x / (σ·√2))`.
pub fn half_normal_cdf_on_support<T: LazyReal>(x: T, scale: T) -> T {
    (x / (scale * T::sqrt_2())).erf()
}

/// Exponential CDF on `x ≥ 0`: `1 − exp(−rate·x)`.
pub fn exponential_cdf_on_support<T: LazyReal>(x: T, rate: T) -> T {
    T::one() - (-(rate * x)).exp()
}

/// Logistic CDF on the real line: `σ((x − location)/scale)`.
pub fn logistic_cdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    sigmoid((x - location) / scale)
}

/// Gumbel CDF on the real line: `exp(−exp(−z))`, `z = (x − location)/scale`.
pub fn gumbel_cdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    (-((-z).exp())).exp()
}

/// Weibull CDF on `x ≥ 0`: `1 − exp(−(x/scale)^shape)`.
pub fn weibull_cdf_on_support<T: LazyReal>(x: T, scale: T, shape: T) -> T {
    let powered = (shape * (x / scale).ln()).exp();
    T::one() - (-powered).exp()
}

/// Frechet CDF on `x > 0`: `exp(−(scale/x)^shape)`.
pub fn frechet_cdf_on_support<T: LazyReal>(x: T, scale: T, shape: T) -> T {
    (-((shape * (scale / x).ln()).exp())).exp()
}

/// Rayleigh CDF on `x ≥ 0`: `1 − exp(−x²/(2σ²))`.
pub fn rayleigh_cdf_on_support<T: LazyReal>(x: T, scale: T) -> T {
    T::one() - (-(x.clone() * x) / (T::two() * scale.clone() * scale)).exp()
}

/// Pareto CDF on `x ≥ minimum`: `1 − (minimum/x)^shape`.
pub fn pareto_cdf_on_support<T: LazyReal>(x: T, minimum: T, shape: T) -> T {
    T::one() - (shape * (minimum / x).ln()).exp()
}

/// Lomax CDF on `x ≥ 0`: `1 − (1 + x/scale)^(−shape)`.
pub fn lomax_cdf_on_support<T: LazyReal>(x: T, shape: T, scale: T) -> T {
    T::one() - (-(shape * (T::one() + x / scale).ln())).exp()
}

/// Gamma CDF on `x > 0`: the regularized lower incomplete gamma `P(shape, rate·x)`.
pub fn gamma_cdf_on_support<T: LazyReal>(x: T, shape: T, rate: T) -> T {
    (rate * x).gamma_lr(shape)
}

/// Chi-squared CDF on `x > 0`: the gamma CDF with `shape = freedom/2`, `rate = ½`.
pub fn chi_squared_cdf_on_support<T: LazyReal>(x: T, half_freedom: T) -> T {
    gamma_cdf_on_support(x, half_freedom, T::half())
}

/// Beta CDF on `0 ≤ x ≤ 1`: the regularized incomplete beta `Iₓ(α, β)`.
pub fn beta_cdf_on_unit<T: LazyReal>(x: T, alpha: T, beta: T) -> T {
    x.beta_reg(alpha, beta)
}

/// Student-t CDF on the real line.
///
/// Written branch-free as `½ + ½·sign(z)·(1 − I_w(ν/2, ½))`, with `z = (x−loc)/scale`
/// and `w = ν/(ν + z²)`, so it lowers to a `Sign` / `RegularizedIncompleteBeta` pair.
pub fn students_t_cdf<T: LazyReal>(x: T, location: T, scale: T, freedom: T) -> T {
    let z = (x - location) / scale;
    let w = freedom.clone() / (freedom.clone() + z.clone() * z.clone());
    let ix = w.beta_reg(freedom * T::half(), T::half());
    T::half() + T::half() * z.signum() * (T::one() - ix)
}

/// Fisher–Snedecor F CDF on `x > 0`: `I_{d1·x/(d1·x + d2)}(d1/2, d2/2)`.
pub fn fisher_snedecor_cdf_on_support<T: LazyReal>(x: T, d1: T, d2: T) -> T {
    let d1x = d1.clone() * x;
    (d1x.clone() / (d1x + d2.clone())).beta_reg(d1 * T::half(), d2 * T::half())
}

/// Poisson CDF at integer `k`: `Q(k + 1, λ)` (regularized upper incomplete gamma). `k`
/// is an integer count ([`LazyInteger`]).
pub fn poisson_cdf<I, T>(k: I, rate: T) -> T
where
    I: LazyInteger<Real = T>,
    T: LazyReal,
{
    rate.gamma_ur(k.to_real() + T::one())
}

/// Binomial CDF at `k < n`: `I_{1−p}(n − k, k + 1)`. The `successes` and `trials` counts
/// are integer-typed ([`LazyInteger`]).
pub fn binomial_cdf_on_support<I, T>(successes: I, trials: I, p: T) -> T
where
    I: LazyInteger<Real = T>,
    T: LazyReal,
{
    let s = successes.to_real();
    (T::one() - p).beta_reg(trials.to_real() - s.clone(), s + T::one())
}

/// Negative-binomial CDF at `k`: `I_p(r, k + 1)`. The `failures` count is integer-typed
/// ([`LazyInteger`]); `successes_until_stop` (`r`) stays real (continuous shape).
pub fn negative_binomial_cdf<I, T>(failures: I, successes_until_stop: T, p: T) -> T
where
    I: LazyInteger<Real = T>,
    T: LazyReal,
{
    p.beta_reg(successes_until_stop, failures.to_real() + T::one())
}

/// k-th order-statistic CDF of `sample_size` iid draws from a base CDF value
/// `u = F(x)`: the regularized incomplete beta `I_u(rank, sample_size − rank + 1)`.
/// `rank = 1` gives the minimum (`1 − (1−u)ⁿ`), `rank = n` the maximum (`uⁿ`).
pub fn order_statistic_cdf<T: LazyReal>(base_cdf: T, rank: T, sample_size: T) -> T {
    base_cdf.beta_reg(rank.clone(), sample_size - rank + T::one())
}

// --- stable log-CDF / log-survival tails ---
//
// Branch-free log-domain forms via the lazy `ln_1m_exp` / `ln_1p_exp`, accurate in
// the tails where `cdf().ln()` would lose precision.

/// Exponential log-CDF on `x ≥ 0`: `ln(1 − exp(−rate·x))` via `ln_1m_exp`.
pub fn exponential_ln_cdf_on_support<T: LazyReal>(x: T, rate: T) -> T {
    (-(rate * x)).ln_1m_exp()
}

/// Logistic log-CDF on the real line: `−softplus(−z)`, `z = (x − location)/scale`.
pub fn logistic_ln_cdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    -(-z).ln_1p_exp()
}

/// Logistic log-survival on the real line: `−softplus(z)`, `z = (x − location)/scale`.
pub fn logistic_ln_sf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    -z.ln_1p_exp()
}

/// Gumbel log-CDF on the real line: `−exp(−z)`, `z = (x − location)/scale`.
pub fn gumbel_ln_cdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    -(-z).exp()
}
