//! The continuous half of the canonical sweep.
//!
//! [`ContView`] collapses every continuous family's generic traits into one
//! object-safe surface, so a property check is written once rather than per
//! family — which is how a sweep ends up covering `Normal` thoroughly and
//! `Lomax` not at all. [`continuous_cases`] is the single parameter list every
//! suite iterates: one benign setting per family, then the settings that break
//! things.
//!
//! The two probes here detect `f64` limits **exactly** rather than
//! heuristically, so a test can skip a case it cannot judge instead of reporting
//! the format's limitation as a library defect.
use bayes_distribution_ext::{
    Beta, Cauchy, ChiSquared, Continuous, ContinuousCdf, Exponential, FisherSnedecor, Frechet,
    Gamma, Gumbel, HalfNormal, InverseGamma, Laplace, LogNormal, Logistic, Lomax, Moments, Normal,
    Pareto, Rayleigh, StudentT, Support, Triangular, Uniform, Wald, Weibull,
};
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// Continuous
// ---------------------------------------------------------------------------

/// One object-safe view over every continuous family.
pub trait ContView {
    fn pdf(&self, x: f64) -> f64;
    fn ln_pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
    fn sf(&self, x: f64) -> f64;
    fn ln_cdf(&self, x: f64) -> f64;
    fn ln_sf(&self, x: f64) -> f64;
    fn inverse_cdf(&self, p: f64) -> f64;
    /// The inverse survival. Absent from this view until the far-tail suite
    /// needed it, which is why no sweep had ever exercised it — and why its
    /// generic default, `inverse_cdf(1 - p)`, could return the support bound
    /// for every `p` below `1e-16` without anything noticing.
    fn isf(&self, p: f64) -> f64;
    fn lo(&self) -> f64;
    fn hi(&self) -> f64;
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn std_dev(&self) -> f64;
    fn entropy(&self) -> f64;
    fn median(&self) -> f64;
    fn skewness(&self) -> f64;
    fn kurtosis(&self) -> f64;
    /// One draw. The RNG type is concrete rather than generic so the trait stays
    /// object-safe, and seeded so a failing sweep reproduces exactly.
    fn sample(&self, rng: &mut ChaCha8Rng) -> f64;
}

impl<T> ContView for T
where
    T: Continuous<f64> + ContinuousCdf<f64> + Moments<f64> + Support<f64> + Distribution<f64>,
{
    fn pdf(&self, x: f64) -> f64 {
        Continuous::pdf(self, x)
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        Continuous::ln_pdf(self, x)
    }
    fn cdf(&self, x: f64) -> f64 {
        ContinuousCdf::cdf(self, x)
    }
    fn sf(&self, x: f64) -> f64 {
        ContinuousCdf::sf(self, x)
    }
    fn ln_cdf(&self, x: f64) -> f64 {
        ContinuousCdf::ln_cdf(self, x)
    }
    fn ln_sf(&self, x: f64) -> f64 {
        ContinuousCdf::ln_sf(self, x)
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        ContinuousCdf::inverse_cdf(self, p)
    }
    fn isf(&self, p: f64) -> f64 {
        ContinuousCdf::isf(self, p)
    }
    fn lo(&self) -> f64 {
        <T as Support<f64>>::min(self)
    }
    fn hi(&self) -> f64 {
        <T as Support<f64>>::max(self)
    }
    fn mean(&self) -> f64 {
        Moments::mean(self)
    }
    fn variance(&self) -> f64 {
        Moments::variance(self)
    }
    fn std_dev(&self) -> f64 {
        Moments::std_dev(self)
    }
    fn entropy(&self) -> f64 {
        Moments::entropy(self)
    }
    fn median(&self) -> f64 {
        Moments::median(self)
    }
    fn skewness(&self) -> f64 {
        Moments::skewness(self)
    }
    fn kurtosis(&self) -> f64 {
        Moments::kurtosis(self)
    }
    fn sample(&self, rng: &mut ChaCha8Rng) -> f64 {
        Distribution::sample(self, rng)
    }
}

/// A named continuous family at one parameter setting.
pub struct ContCase {
    pub name: String,
    pub dist: Box<dyn ContView>,
}

fn cont(name: &str, dist: impl ContView + 'static) -> ContCase {
    ContCase {
        name: name.to_string(),
        dist: Box::new(dist),
    }
}

/// Is this distribution's scale so small relative to its location that its own
/// quantiles are not representable in `f64`?
///
/// `Normal(1e12, 1e-6)` is: one ulp at `1e12` is `1.2e-4`, a hundred standard
/// deviations wide, so every quantile collapses onto the mean. Nothing an
/// implementation does can fix that, and a check that flags it is testing `f64`
/// rather than bayes.
pub fn quantiles_are_unrepresentable(d: &dyn ContView) -> bool {
    let (a, b) = (d.inverse_cdf(0.01), d.inverse_cdf(0.99));
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let ulp = a.abs().max(b.abs()) * f64::EPSILON;
    (b - a) <= 8.0 * ulp
}

/// Is the true `p`-quantile outside what `f64` can represent on this support?
///
/// `Gamma(1e-3, 1)` has `cdf(x) ≈ x^0.001`, so `cdf` is already `0.475` at the
/// smallest positive double and `Q(0.25) = 0.25^1000 = 1e-602` simply does not
/// exist as an `f64`. The probe is exact rather than heuristic: it asks whether
/// the CDF has already passed `p` at the first representable step inside the
/// support.
pub fn quantile_unrepresentable_at(d: &dyn ContView, p: f64) -> bool {
    let lo = d.lo();
    if lo.is_finite() && d.cdf(lo.next_up()) > p {
        return true;
    }
    let hi = d.hi();
    if hi.is_finite() && d.cdf(hi.next_down()) < p {
        return true;
    }
    false
}

/// Every continuous family at a benign setting, then at the settings that break
/// things. `VonMises` is absent because it implements no `ContinuousCdf` — the
/// circular CDF has no closed form — and is exercised separately.
pub fn continuous_cases() -> Vec<ContCase> {
    vec![
        // --- benign baselines ------------------------------------------------
        cont("Normal(0,1)", Normal::new(0.0, 1.0).unwrap()),
        cont("Normal(-3,0.5)", Normal::new(-3.0, 0.5).unwrap()),
        cont("Uniform(-1,2)", Uniform::new(-1.0, 2.0).unwrap()),
        cont("Exponential(1)", Exponential::new(1.0).unwrap()),
        cont("Exponential(7.5)", Exponential::new(7.5).unwrap()),
        cont("Gamma(2,3)", Gamma::new(2.0, 3.0).unwrap()),
        cont("Beta(2,5)", Beta::new(2.0, 5.0).unwrap()),
        cont("Beta(0.5,0.5)", Beta::new(0.5, 0.5).unwrap()),
        cont("LogNormal(0,1)", LogNormal::new(0.0, 1.0).unwrap()),
        cont("Laplace(1,2)", Laplace::new(1.0, 2.0).unwrap()),
        cont("Logistic(-1,0.3)", Logistic::new(-1.0, 0.3).unwrap()),
        cont("Cauchy(0,1)", Cauchy::new(0.0, 1.0).unwrap()),
        cont("ChiSquared(3)", ChiSquared::new(3.0).unwrap()),
        cont("StudentT(5)", StudentT::new(5.0).unwrap()),
        cont("Rayleigh(2)", Rayleigh::new(2.0).unwrap()),
        cont("HalfNormal(1.5)", HalfNormal::new(1.5).unwrap()),
        cont("Gumbel(0,1)", Gumbel::new(0.0, 1.0).unwrap()),
        cont("Frechet(3,2)", Frechet::new(3.0, 2.0).unwrap()),
        cont("Weibull(2,3)", Weibull::new(2.0, 3.0).unwrap()),
        cont("Pareto(1,4)", Pareto::new(1.0, 4.0).unwrap()),
        cont("Lomax(3,2)", Lomax::new(3.0, 2.0).unwrap()),
        cont("InverseGamma(4,2)", InverseGamma::new(4.0, 2.0).unwrap()),
        cont(
            "FisherSnedecor(6,9)",
            FisherSnedecor::new(6.0, 9.0).unwrap(),
        ),
        cont("Wald(1,3)", Wald::new(1.0, 3.0).unwrap()),
        cont(
            "Triangular(0,1,0.3)",
            Triangular::new(0.0, 1.0, 0.3).unwrap(),
        ),
        // Mode exactly at either endpoint: the degenerate right triangles.
        cont("Triangular(0,1,0)", Triangular::new(0.0, 1.0, 0.0).unwrap()),
        cont("Triangular(0,1,1)", Triangular::new(0.0, 1.0, 1.0).unwrap()),
        // --- tiny scales: everything must still normalize --------------------
        cont("Normal(0,1e-8)", Normal::new(0.0, 1e-8).unwrap()),
        cont("Uniform(0,1e-9)", Uniform::new(0.0, 1e-9).unwrap()),
        cont("Exponential(1e9)", Exponential::new(1e9).unwrap()),
        cont("Laplace(0,1e-9)", Laplace::new(0.0, 1e-9).unwrap()),
        cont("Cauchy(0,1e-9)", Cauchy::new(0.0, 1e-9).unwrap()),
        cont("Rayleigh(1e-9)", Rayleigh::new(1e-9).unwrap()),
        cont("HalfNormal(1e-9)", HalfNormal::new(1e-9).unwrap()),
        cont("Gumbel(0,1e-6)", Gumbel::new(0.0, 1e-6).unwrap()),
        // --- huge scales -----------------------------------------------------
        cont("Normal(0,1e10)", Normal::new(0.0, 1e10).unwrap()),
        cont("Exponential(1e-9)", Exponential::new(1e-9).unwrap()),
        cont("Uniform(-1e12,1e12)", Uniform::new(-1e12, 1e12).unwrap()),
        // --- a location so far out that the scale is below one ulp -----------
        cont("Normal(1e12,1e-6)", Normal::new(1e12, 1e-6).unwrap()),
        // --- extreme shapes ---------------------------------------------------
        cont("Gamma(1e-3,1)", Gamma::new(1e-3, 1.0).unwrap()),
        cont("Gamma(1e4,1)", Gamma::new(1e4, 1.0).unwrap()),
        cont("Beta(1e-2,1e-2)", Beta::new(1e-2, 1e-2).unwrap()),
        cont("Beta(1e3,1e3)", Beta::new(1e3, 1e3).unwrap()),
        cont("Beta(1,1)", Beta::new(1.0, 1.0).unwrap()),
        cont("Beta(1e3,1)", Beta::new(1e3, 1.0).unwrap()),
        cont("ChiSquared(1)", ChiSquared::new(1.0).unwrap()),
        cont("ChiSquared(1e4)", ChiSquared::new(1e4).unwrap()),
        cont("StudentT(1)", StudentT::new(1.0).unwrap()),
        cont("StudentT(2)", StudentT::new(2.0).unwrap()),
        cont("StudentT(3)", StudentT::new(3.0).unwrap()),
        cont("StudentT(1e6)", StudentT::new(1e6).unwrap()),
        cont("Weibull(1,0.3)", Weibull::new(1.0, 0.3).unwrap()),
        cont("Weibull(1,30)", Weibull::new(1.0, 30.0).unwrap()),
        cont("Pareto(1,0.5)", Pareto::new(1.0, 0.5).unwrap()),
        cont("Pareto(1,100)", Pareto::new(1.0, 100.0).unwrap()),
        cont("InverseGamma(0.5,1)", InverseGamma::new(0.5, 1.0).unwrap()),
        cont(
            "FisherSnedecor(1,1)",
            FisherSnedecor::new(1.0, 1.0).unwrap(),
        ),
        cont("Wald(1e-3,1e3)", Wald::new(1e-3, 1e3).unwrap()),
        cont("LogNormal(0,3)", LogNormal::new(0.0, 3.0).unwrap()),
        cont("Frechet(0.5,1)", Frechet::new(0.5, 1.0).unwrap()),
        cont("Lomax(0.5,1)", Lomax::new(0.5, 1.0).unwrap()),
    ]
}

// ---------------------------------------------------------------------------
// Discrete
// ---------------------------------------------------------------------------
