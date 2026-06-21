use super::*;
use euclid::numeric::LazyBool;

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

/// Cauchy CDF on the real line: `½ + atan((x − location)/scale)/π`.
pub fn cauchy_cdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    T::half() + ((x - location) / scale).atan() / T::pi()
}

/// Uniform CDF on `[lo, lo + width]`: the linear ramp `(x − lo)/width` clamped to `[0, 1]`.
pub fn uniform_cdf<T: LazyReal>(x: T, lo: T, width: T) -> T {
    ((x - lo) / width).clamp(T::zero(), T::one())
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

// --- full-domain CDFs / survivals (support indicator folded in via `select`) ---
//
// These wrap the `_on_support` interiors above with the below-support floor (or the
// piecewise tail) that the eager spine used to spell as a host `if`. The branch is now a
// branchless `select` / `mask_invalid` over the lazy surface, so the *whole* CDF — support
// boundary included — lowers to a backend's `Select` and runs unchanged on `f64`, Polars,
// and XLA. Where an interior would form a `NaN` off support (a `ln` / `sqrt` / `÷x`), the
// input is first scrubbed to an in-support dummy (the "double `where`"), so the discarded
// `select` arm stays finite for a backend that evaluates both.

/// Normal log-CDF `ln Φ((x−µ)/σ)` on the real line. The left tail (`z ≤ 0`) uses the stable
/// `erfcx` form; the right defers through `ln(1 − sf)`, so it stays accurate where `Φ`
/// underflows to `0`.
pub fn normal_ln_cdf<T: LazyReal>(x: T, mean: T, std_dev: T) -> T {
    let z = (x - mean) / std_dev;
    let t = -(z.clone()) * T::frac_1_sqrt_2();
    let s = z.clone() * T::frac_1_sqrt_2();
    let left = -T::ln_2() + t.clone().erfcx().ln() - t.clone() * t;
    let right_ln_sf = -T::ln_2() + s.clone().erfcx().ln() - s.clone() * s;
    T::select(&z.le(&T::zero()), left, right_ln_sf.ln_1m_exp())
}

/// Normal log-survival `ln(1 − Φ((x−µ)/σ))`, the mirror of [`normal_ln_cdf`].
pub fn normal_ln_sf<T: LazyReal>(x: T, mean: T, std_dev: T) -> T {
    let z = (x - mean) / std_dev;
    let s = z.clone() * T::frac_1_sqrt_2();
    let t = -(z.clone()) * T::frac_1_sqrt_2();
    let right = -T::ln_2() + s.clone().erfcx().ln() - s.clone() * s;
    let left_ln_cdf = -T::ln_2() + t.clone().erfcx().ln() - t.clone() * t;
    T::select(&z.ge(&T::zero()), right, left_ln_cdf.ln_1m_exp())
}

/// Exponential CDF on the real line: `0` for `x < 0`, else `1 − exp(−rate·x)`.
pub fn exponential_cdf<T: LazyReal>(x: T, rate: T) -> T {
    let in_support = x.ge(&T::zero());
    T::select(&in_support, exponential_cdf_on_support(x, rate), T::zero())
}

/// Exponential log-CDF: `−∞` for `x ≤ 0`, else `ln(1 − exp(−rate·x))` via `ln_1m_exp`.
pub fn exponential_ln_cdf<T: LazyReal>(x: T, rate: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    exponential_ln_cdf_on_support(safe_x, rate).mask_invalid(in_support, T::neg_infinity())
}

/// Exponential log-survival: `0` for `x ≤ 0`, else `−rate·x`.
pub fn exponential_ln_sf<T: LazyReal>(x: T, rate: T) -> T {
    let in_support = x.gt(&T::zero());
    T::select(
        &in_support,
        exponential_log_survival_on_support(x, rate),
        T::zero(),
    )
}

/// Gamma CDF: `0` for `x ≤ 0`, else the regularized lower incomplete gamma `P(shape, rate·x)`.
pub fn gamma_cdf<T: LazyReal>(x: T, shape: T, rate: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    gamma_cdf_on_support(safe_x, shape, rate).mask_invalid(in_support, T::zero())
}

/// Chi-squared CDF: `0` for `x ≤ 0`, else the gamma CDF with `shape = freedom/2`, `rate = ½`.
pub fn chi_squared_cdf<T: LazyReal>(x: T, half_freedom: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    chi_squared_cdf_on_support(safe_x, half_freedom).mask_invalid(in_support, T::zero())
}

/// Inverse-gamma CDF: `0` for `x ≤ 0`, else `Q(shape, scale/x)` (regularized upper gamma).
pub fn inverse_gamma_cdf<T: LazyReal>(x: T, shape: T, scale: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    (scale / safe_x)
        .gamma_ur(shape)
        .mask_invalid(in_support, T::zero())
}

/// Log-normal CDF: `0` for `x ≤ 0`, else the Normal CDF of `ln x`.
pub fn lognormal_cdf<T: LazyReal>(x: T, mean_log: T, std_log: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    normal_cdf(safe_x.ln(), mean_log, std_log).mask_invalid(in_support, T::zero())
}

/// Weibull CDF: `0` for `x < 0`, else `1 − exp(−(x/scale)^shape)`.
pub fn weibull_cdf<T: LazyReal>(x: T, scale: T, shape: T) -> T {
    let in_support = x.ge(&T::zero());
    let safe_x = T::select(&in_support, x, scale.clone());
    weibull_cdf_on_support(safe_x, scale, shape).mask_invalid(in_support, T::zero())
}

/// Rayleigh CDF: `0` for `x < 0`, else `1 − exp(−x²/(2σ²))`.
pub fn rayleigh_cdf<T: LazyReal>(x: T, scale: T) -> T {
    let in_support = x.ge(&T::zero());
    T::select(&in_support, rayleigh_cdf_on_support(x, scale), T::zero())
}

/// Half-normal CDF: `0` for `x < 0`, else `erf(x/(σ·√2))`.
pub fn half_normal_cdf<T: LazyReal>(x: T, scale: T) -> T {
    let in_support = x.ge(&T::zero());
    T::select(&in_support, half_normal_cdf_on_support(x, scale), T::zero())
}

/// Fréchet CDF: `0` for `x ≤ 0`, else `exp(−(scale/x)^shape)`.
pub fn frechet_cdf<T: LazyReal>(x: T, scale: T, shape: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, scale.clone());
    frechet_cdf_on_support(safe_x, scale, shape).mask_invalid(in_support, T::zero())
}

/// Fréchet log-CDF: `−∞` for `x ≤ 0`, else `−(scale/x)^shape`.
pub fn frechet_ln_cdf<T: LazyReal>(x: T, scale: T, shape: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, scale.clone());
    let ln_cdf = -((shape * (scale / safe_x).ln()).exp());
    ln_cdf.mask_invalid(in_support, T::neg_infinity())
}

/// Pareto CDF: `0` for `x < minimum`, else `1 − (minimum/x)^shape`.
pub fn pareto_cdf<T: LazyReal>(x: T, minimum: T, shape: T) -> T {
    let in_support = x.ge(&minimum);
    let safe_x = T::select(&in_support, x, minimum.clone());
    pareto_cdf_on_support(safe_x, minimum, shape).mask_invalid(in_support, T::zero())
}

/// Lomax CDF: `0` for `x < 0`, else `1 − (1 + x/scale)^(−shape)`.
pub fn lomax_cdf<T: LazyReal>(x: T, shape: T, scale: T) -> T {
    let in_support = x.ge(&T::zero());
    let safe_x = T::select(&in_support, x, T::zero());
    lomax_cdf_on_support(safe_x, shape, scale).mask_invalid(in_support, T::zero())
}

/// Lomax log-survival: `0` for `x ≤ 0`, else `−shape·ln(1 + x/scale)`.
pub fn lomax_ln_sf<T: LazyReal>(x: T, shape: T, scale: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::zero());
    let ln_sf = -(shape * (T::one() + safe_x / scale).ln());
    T::select(&in_support, ln_sf, T::zero())
}

/// Fisher–Snedecor F CDF: `0` for `x ≤ 0`, else `I_{d1·x/(d1·x+d2)}(d1/2, d2/2)`.
pub fn fisher_snedecor_cdf<T: LazyReal>(x: T, d1: T, d2: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    fisher_snedecor_cdf_on_support(safe_x, d1, d2).mask_invalid(in_support, T::zero())
}

/// Laplace CDF on the real line: `½·exp(z)` for `x < location`, else `1 − ½·exp(−z)`,
/// `z = (x − location)/scale`.
pub fn laplace_cdf<T: LazyReal>(x: T, location: T, scale: T) -> T {
    let z = (x.clone() - location.clone()) / scale;
    let left = T::half() * z.clone().exp();
    let right = T::one() - T::half() * (-z).exp();
    T::select(&x.lt(&location), left, right)
}

/// Wald (inverse-Gaussian) CDF: `0` for `x ≤ 0`, else the stable two-term `erfc`/`erfcx`
/// form, clamped to `[0, 1]`.
pub fn wald_cdf<T: LazyReal>(x: T, mu: T, lambda: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, mu.clone());
    let s = (lambda.clone() / safe_x.clone()).sqrt();
    let a = s.clone() * (safe_x.clone() / mu.clone() - T::one());
    let b = s * (safe_x / mu.clone() + T::one());
    let phi_a = T::half() * (-a * T::frac_1_sqrt_2()).erfc();
    let term2 = T::half()
        * (b.clone() * T::frac_1_sqrt_2()).erfcx()
        * (T::two() * lambda / mu - T::half() * b.clone() * b).exp();
    (phi_a + term2)
        .clamp(T::zero(), T::one())
        .mask_invalid(in_support, T::zero())
}

/// Triangular PDF on `[lo, hi]` with mode `c`: the rising/falling ramps, `0` outside.
pub fn triangular_pdf<T: LazyReal>(x: T, lo: T, hi: T, mode: T) -> T {
    let width = hi.clone() - lo.clone();
    let up = T::two() * (x.clone() - lo.clone()) / (width.clone() * (mode.clone() - lo.clone()));
    let down = T::two() * (hi.clone() - x.clone()) / (width * (hi.clone() - mode.clone()));
    let ramp = T::select(&x.lt(&mode), up, down);
    let inside = x.ge(&lo).and(&x.le(&hi));
    T::select(&inside, ramp, T::zero())
}

/// Triangular CDF on `[lo, hi]` with mode `c`: `0` below, the two quadratic pieces, `1` above.
pub fn triangular_cdf<T: LazyReal>(x: T, lo: T, hi: T, mode: T) -> T {
    let width = hi.clone() - lo.clone();
    let lower = (x.clone() - lo.clone()) * (x.clone() - lo.clone())
        / (width.clone() * (mode.clone() - lo.clone()));
    let upper = T::one()
        - (hi.clone() - x.clone()) * (hi.clone() - x.clone())
            / (width * (hi.clone() - mode.clone()));
    let body = T::select(&x.lt(&mode), lower, upper);
    let clamped = T::select(&x.ge(&hi), T::one(), body);
    T::select(&x.le(&lo), T::zero(), clamped)
}

/// Triangular quantile on `[lo, hi]` with mode `c`: the inverse of the two quadratic pieces.
/// `p` is taken on the closed unit interval — clamping it is the caller's branch.
pub fn triangular_quantile<T: LazyReal>(p: T, lo: T, hi: T, mode: T) -> T {
    let width = hi.clone() - lo.clone();
    let split = (mode.clone() - lo.clone()) / width.clone();
    let left = lo.clone() + (width.clone() * (mode.clone() - lo.clone()) * p.clone()).sqrt();
    let right = hi.clone() - (width * (hi - mode) * (T::one() - p.clone())).sqrt();
    T::select(&p.lt(&split), left, right)
}
