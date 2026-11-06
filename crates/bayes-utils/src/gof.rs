//! Goodness-of-fit probes for judging a **sampler** against the distribution it
//! claims to draw from.
//!
//! A sampler is the one part of a distribution that no identity can pin down:
//! the density, CDF, quantile and moments all constrain each other, but the
//! sampler is a separate algorithm — Marsaglia–Tsang for a gamma, inverse-CDF
//! for a Weibull, Box–Muller or ziggurat for a normal — and nothing in the
//! closed-form surface notices if it is wrong. It can only be checked
//! statistically.
//!
//! That makes the tolerance the hard part. A test that asserts "the sample mean
//! is within 0.01 of the true mean" is either vacuous or flaky depending on the
//! sample size, and tightening it until it passes is how a suite ends up
//! asserting the sampler's current bias. So every threshold here is **derived**:
//!
//! * [`ks_pvalue`] is the asymptotic Kolmogorov distribution, a pure series with
//!   no dependence on bayes at all — fully independent of what it judges.
//! * [`mean_tolerance`] is the central-limit standard error times a stated
//!   number of sigmas, so "how often may this fail by chance" is explicit.
//!
//! The chi-squared p-value does use bayes' own `ChiSquared`, which is mildly
//! circular. It is acceptable because that CDF is independently pinned by the
//! relationships suite — against `Gamma`, and against the `Poisson` tail bridge
//! — so it is not resting on itself.

use bayes_distribution_ext::{ChiSquared, ContinuousCdf};

/// The two-sided Kolmogorov–Smirnov statistic `sup|F_n(x) − F(x)|` of `samples`
/// against `cdf`.
///
/// Both one-sided gaps are taken at each order statistic, which is what makes
/// this the exact empirical supremum rather than an approximation of it: the
/// empirical CDF is a step function, so the largest deviation occurs at a step.
pub fn ks_statistic(samples: &[f64], cdf: impl Fn(f64) -> f64) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len() as f64;
    let mut d = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let f = cdf(x);
        let below = i as f64 / n; // F_n just before the step
        let above = (i + 1) as f64 / n; // F_n just after it
        d = d.max((f - below).abs()).max((above - f).abs());
    }
    d
}

/// The asymptotic p-value of a KS statistic: `Q(λ) = 2 Σ (−1)^{k−1} e^{−2k²λ²}`.
///
/// The Stephens correction `λ = (√n + 0.12 + 0.11/√n)·D` makes the asymptotic
/// form usable well below the sample sizes where the bare `√n·D` is honest.
/// This is a self-contained series: it shares no code with anything it judges.
pub fn ks_pvalue(d: f64, n: usize) -> f64 {
    if !d.is_finite() || n == 0 {
        return f64::NAN;
    }
    let rn = (n as f64).sqrt();
    let lambda = (rn + 0.12 + 0.11 / rn) * d;
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0;
    for k in 1..=100 {
        let term = (-2.0 * (k * k) as f64 * lambda * lambda).exp();
        sum += if k % 2 == 1 { term } else { -term };
        if term < 1e-18 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// Pearson's chi-squared goodness-of-fit p-value for binned counts.
///
/// Cells whose expected count falls below `min_expected` are pooled into their
/// neighbour, because Pearson's statistic is only chi-squared distributed when
/// no cell is nearly empty — and an unpooled tail cell is the classic way to get
/// a spuriously significant result out of a correct sampler. `degrees_of_freedom`
/// is the surviving cell count minus one.
pub fn chi_squared_gof_pvalue(observed: &[f64], expected: &[f64], min_expected: f64) -> f64 {
    if observed.len() != expected.len() || observed.is_empty() {
        return f64::NAN;
    }
    let (mut obs, mut exp) = (Vec::new(), Vec::new());
    let (mut obs_acc, mut exp_acc) = (0.0, 0.0);
    for (&o, &e) in observed.iter().zip(expected.iter()) {
        obs_acc += o;
        exp_acc += e;
        if exp_acc >= min_expected {
            obs.push(obs_acc);
            exp.push(exp_acc);
            obs_acc = 0.0;
            exp_acc = 0.0;
        }
    }
    if exp_acc > 0.0 {
        // Fold any remaining tail into the last surviving cell.
        if let (Some(o), Some(e)) = (obs.last_mut(), exp.last_mut()) {
            *o += obs_acc;
            *e += exp_acc;
        } else {
            obs.push(obs_acc);
            exp.push(exp_acc);
        }
    }
    if obs.len() < 2 {
        return f64::NAN;
    }
    let stat: f64 = obs
        .iter()
        .zip(exp.iter())
        .map(|(&o, &e)| (o - e) * (o - e) / e)
        .sum();
    let df = (obs.len() - 1) as f64;
    match ChiSquared::new(df) {
        Ok(chi) => ContinuousCdf::sf(&chi, stat),
        Err(_) => f64::NAN,
    }
}

/// The half-width a sample mean may legitimately miss the true mean by:
/// `sigmas · √(variance/n)`.
///
/// Stating it this way makes the false-failure rate explicit rather than
/// implicit in a hand-tuned constant. At `sigmas = 5` a correct sampler trips
/// this about once in 1.7 million runs per check.
pub fn mean_tolerance(variance: f64, n: usize, sigmas: f64) -> f64 {
    sigmas * (variance / n as f64).sqrt()
}

/// The same bound for a sample *variance*, whose standard error depends on the
/// fourth moment: `se ≈ σ²·√((κ + 2)/n)` for excess kurtosis `κ`.
///
/// Returns `None` when the kurtosis is not finite — a family whose fourth moment
/// does not exist has no usable bound, and saying so is better than inventing
/// one.
pub fn variance_tolerance(variance: f64, kurtosis: f64, n: usize, sigmas: f64) -> Option<f64> {
    if !kurtosis.is_finite() || !variance.is_finite() {
        return None;
    }
    Some(sigmas * variance * ((kurtosis + 2.0) / n as f64).sqrt())
}
