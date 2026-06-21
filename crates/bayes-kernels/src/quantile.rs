use super::*;

// --- closed-form quantiles (inverse CDFs) ---
//
// The inverse-CDF maps `p ∈ (0, 1)` to the support — a total formula on the open
// interval, so (as with `logit`) clamping `p` and the `p = 0`/`p = 1` boundary are
// the caller's branch. Variable powers `(·)^{1/shape}` go through
// `exp((1/shape)·ln(·))` because the lazy `powf` takes a *constant* exponent.

/// Exponential quantile: `−ln(1 − p) / rate`.
pub fn exponential_quantile<T: LazyReal>(p: T, rate: T) -> T {
    -((T::one() - p).ln()) / rate
}

/// Logistic quantile: `location + scale · ln(p / (1 − p))`.
pub fn logistic_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    location + scale * (p.clone() / (T::one() - p)).ln()
}

/// Cauchy quantile: `location + scale · tan(π·(p − ½))`.
pub fn cauchy_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    location + scale * (T::pi() * (p - T::half())).tan()
}

/// Laplace quantile: `location − scale · sign(q) · ln(1 − 2|q|)`, `q = p − ½`.
pub fn laplace_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    let q = p - T::half();
    location - scale * q.clone().signum() * (T::one() - T::two() * q.abs()).ln()
}

/// Uniform quantile: `lo + width · p`, with `width = hi − lo`.
pub fn uniform_quantile<T: LazyReal>(p: T, lo: T, width: T) -> T {
    lo + width * p
}

/// Weibull quantile: `scale · (−ln(1 − p))^{1/shape}`.
pub fn weibull_quantile<T: LazyReal>(p: T, scale: T, shape: T) -> T {
    let base = -((T::one() - p).ln());
    scale * ((T::one() / shape) * base.ln()).exp()
}

/// Gumbel quantile: `location − scale · ln(−ln p)`.
pub fn gumbel_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    location - scale * (-(p.ln())).ln()
}

/// Fréchet quantile: `scale · (−ln p)^{−1/shape}`.
pub fn frechet_quantile<T: LazyReal>(p: T, scale: T, shape: T) -> T {
    let base = -(p.ln());
    scale * ((-(T::one() / shape)) * base.ln()).exp()
}

/// Pareto quantile: `minimum · (1 − p)^{−1/shape}`.
pub fn pareto_quantile<T: LazyReal>(p: T, minimum: T, shape: T) -> T {
    let base = T::one() - p;
    minimum * ((-(T::one() / shape)) * base.ln()).exp()
}

/// Lomax quantile: `scale · ((1 − p)^{−1/shape} − 1)`.
pub fn lomax_quantile<T: LazyReal>(p: T, scale: T, shape: T) -> T {
    let base = T::one() - p;
    scale * (((-(T::one() / shape)) * base.ln()).exp() - T::one())
}

/// Rayleigh quantile: `scale · √(−2·ln(1 − p))`.
pub fn rayleigh_quantile<T: LazyReal>(p: T, scale: T) -> T {
    scale * (-T::two() * (T::one() - p).ln()).sqrt()
}

/// Normal quantile: `mean + std·√2·erf⁻¹(2p − 1)`.
pub fn normal_quantile<T: LazyReal>(p: T, mean: T, std_dev: T) -> T {
    mean + std_dev * T::sqrt_2() * (T::two() * p - T::one()).erf_inv()
}

/// Half-normal quantile: `scale·√2·erf⁻¹(p)`.
pub fn half_normal_quantile<T: LazyReal>(p: T, scale: T) -> T {
    scale * T::sqrt_2() * p.erf_inv()
}

/// Log-normal quantile: the Normal quantile on the log scale, exponentiated —
/// `exp(mean_log + σ_log·√2·erf⁻¹(2p − 1))`.
pub fn lognormal_quantile<T: LazyReal>(p: T, mean_log: T, std_log: T) -> T {
    normal_quantile(p, mean_log, std_log).exp()
}
