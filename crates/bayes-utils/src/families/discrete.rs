//! The discrete half of the canonical sweep.
//!
//! Two views rather than one: [`DiscView`] covers mass and CDF, while
//! [`DiscSampleView`] adds drawing. They are separate because `Bernoulli` draws
//! into `bool` and `DiscreteUniform` into `i64`, so neither can satisfy a single
//! blanket impl with the `u64` families — both are checked directly instead.
//!
//! [`discrete_cases`] includes the degenerate ends (`p = 0`, `p = 1`, `n = 0`, a
//! `Categorical` with an exact zero weight) because that is where a mass
//! function is most likely to divide by zero. [`discrete_sampler_cases`] omits
//! them deliberately: a chi-squared goodness-of-fit test on a point mass has
//! nothing to say.
use bayes_distribution_ext::Discrete as DiscreteMass;
use bayes_distribution_ext::{
    Bernoulli, BetaBinomial, Binomial, Categorical, DiscreteCdf, Geometric, Hypergeometric,
    Moments, NegativeBinomial, Poisson, Support,
};
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// Continuous
// ---------------------------------------------------------------------------

/// One object-safe view over the natural-count discrete families.
/// One object-safe view over the natural-count discrete families.
pub trait DiscView {
    fn pmf(&self, k: u64) -> f64;
    fn ln_pmf(&self, k: u64) -> f64;
    fn cdf(&self, k: u64) -> f64;
    fn sf(&self, k: u64) -> f64;
    fn inverse_cdf(&self, p: f64) -> u64;
    fn lo(&self) -> u64;
    fn hi(&self) -> u64;
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn entropy(&self) -> f64;
}

impl<T> DiscView for T
where
    T: DiscreteMass<u64> + DiscreteCdf<u64> + Moments<f64> + Support<u64>,
{
    fn pmf(&self, k: u64) -> f64 {
        DiscreteMass::pmf(self, k)
    }
    fn ln_pmf(&self, k: u64) -> f64 {
        DiscreteMass::ln_pmf(self, k)
    }
    fn cdf(&self, k: u64) -> f64 {
        DiscreteCdf::cdf(self, k)
    }
    fn sf(&self, k: u64) -> f64 {
        DiscreteCdf::sf(self, k)
    }
    fn inverse_cdf(&self, p: f64) -> u64 {
        DiscreteCdf::inverse_cdf(self, p)
    }
    fn lo(&self) -> u64 {
        <T as Support<u64>>::min(self)
    }
    fn hi(&self) -> u64 {
        <T as Support<u64>>::max(self)
    }
    fn mean(&self) -> f64 {
        Moments::mean(self)
    }
    fn variance(&self) -> f64 {
        Moments::variance(self)
    }
    fn entropy(&self) -> f64 {
        Moments::entropy(self)
    }
}

/// A named discrete family at one parameter setting.
pub struct DiscCase {
    pub name: String,
    pub dist: Box<dyn DiscView>,
}

fn disc(name: &str, dist: impl DiscView + 'static) -> DiscCase {
    DiscCase {
        name: name.to_string(),
        dist: Box::new(dist),
    }
}

/// Where to stop summing an unbounded support: the first `k` carrying all but
/// `1e-15` of the mass, capped so a pathological case cannot hang a sweep.
pub fn summation_limit(d: &dyn DiscView) -> u64 {
    const CAP: u64 = 400_000;
    let hi = d.hi();
    if hi <= CAP {
        return hi;
    }
    d.inverse_cdf(1.0 - 1e-15).saturating_add(16).min(CAP)
}

/// Every discrete family, including the degenerate ends: `p = 0` and `p = 1`,
/// `n = 0`, a mass concentrated almost entirely on zero, one sitting far from
/// the origin where the log-factorials are large, and a `Categorical` carrying
/// an exact zero weight.
pub fn discrete_cases() -> Vec<DiscCase> {
    vec![
        disc("Bernoulli(0.3)", Bernoulli::new(0.3).unwrap()),
        disc("Bernoulli(0)", Bernoulli::new(0.0).unwrap()),
        disc("Bernoulli(1)", Bernoulli::new(1.0).unwrap()),
        disc("Bernoulli(1e-300)", Bernoulli::new(1e-300).unwrap()),
        disc("Binomial(10,0.5)", Binomial::new(10, 0.5).unwrap()),
        disc("Binomial(1,0.25)", Binomial::new(1, 0.25).unwrap()),
        disc("Binomial(0,0.5)", Binomial::new(0, 0.5).unwrap()),
        disc("Binomial(10,0)", Binomial::new(10, 0.0).unwrap()),
        disc("Binomial(10,1)", Binomial::new(10, 1.0).unwrap()),
        disc("Binomial(1000,0.5)", Binomial::new(1000, 0.5).unwrap()),
        disc("Binomial(1000,1e-6)", Binomial::new(1000, 1e-6).unwrap()),
        disc(
            "Binomial(100000,0.999)",
            Binomial::new(100_000, 0.999).unwrap(),
        ),
        disc("Poisson(2.5)", Poisson::new(2.5).unwrap()),
        disc("Poisson(1e-9)", Poisson::new(1e-9).unwrap()),
        disc("Poisson(1e3)", Poisson::new(1e3).unwrap()),
        disc("Geometric(0.3)", Geometric::new(0.3).unwrap()),
        disc("Geometric(1)", Geometric::new(1.0).unwrap()),
        disc("Geometric(1e-3)", Geometric::new(1e-3).unwrap()),
        disc(
            "NegativeBinomial(5,0.5)",
            NegativeBinomial::new(5.0, 0.5).unwrap(),
        ),
        disc(
            "NegativeBinomial(0.5,0.5)",
            NegativeBinomial::new(0.5, 0.5).unwrap(),
        ),
        disc(
            "NegativeBinomial(1e-2,0.5)",
            NegativeBinomial::new(1e-2, 0.5).unwrap(),
        ),
        disc(
            "NegativeBinomial(5,0.99)",
            NegativeBinomial::new(5.0, 0.99).unwrap(),
        ),
        disc(
            "Hypergeometric(50,20,10)",
            Hypergeometric::new(50, 20, 10).unwrap(),
        ),
        disc(
            "Hypergeometric(10,10,5)",
            Hypergeometric::new(10, 10, 5).unwrap(),
        ),
        disc(
            "Hypergeometric(10,0,5)",
            Hypergeometric::new(10, 0, 5).unwrap(),
        ),
        disc(
            "Hypergeometric(10,5,10)",
            Hypergeometric::new(10, 5, 10).unwrap(),
        ),
        disc(
            "BetaBinomial(10,2,3)",
            BetaBinomial::new(10, 2.0, 3.0).unwrap(),
        ),
        disc(
            "BetaBinomial(10,1e-2,1e-2)",
            BetaBinomial::new(10, 1e-2, 1e-2).unwrap(),
        ),
        disc(
            "BetaBinomial(1,5,5)",
            BetaBinomial::new(1, 5.0, 5.0).unwrap(),
        ),
        disc(
            "BetaBinomial(200,0.5,0.5)",
            BetaBinomial::new(200, 0.5, 0.5).unwrap(),
        ),
        disc(
            "Categorical[0.2,0.5,0.3]",
            Categorical::new(&[0.2, 0.5, 0.3]).unwrap(),
        ),
        disc(
            "Categorical[1,0,1]",
            Categorical::new(&[1.0, 0.0, 1.0]).unwrap(),
        ),
        disc("Categorical[1]", Categorical::new(&[1.0]).unwrap()),
        disc(
            "Categorical[1e-12,1]",
            Categorical::new(&[1e-12, 1.0]).unwrap(),
        ),
        disc(
            "Categorical[unnormalized]",
            Categorical::new(&[2.0, 5.0, 3.0]).unwrap(),
        ),
    ]
}

/// The sampling half of [`DiscView`], for the families that draw into `u64`.
///
/// `Bernoulli` draws into `bool` and `DiscreteUniform` into `i64`, so neither
/// satisfies this bound; both are simple enough to check directly.
pub trait DiscSampleView: DiscView {
    fn sample(&self, rng: &mut ChaCha8Rng) -> u64;
}

impl<T> DiscSampleView for T
where
    T: DiscView + Distribution<u64>,
{
    fn sample(&self, rng: &mut ChaCha8Rng) -> u64 {
        Distribution::sample(self, rng)
    }
}

/// A named discrete family that can be sampled into `u64`.
pub struct DiscSampleCase {
    pub name: String,
    pub dist: Box<dyn DiscSampleView>,
}

fn disc_s(name: &str, dist: impl DiscSampleView + 'static) -> DiscSampleCase {
    DiscSampleCase {
        name: name.to_string(),
        dist: Box::new(dist),
    }
}

/// The `u64`-samplable discrete families, at parameters where a goodness-of-fit
/// test has enough mass per cell to be meaningful. Degenerate settings
/// (`p = 0`, `p = 1`, `n = 0`) are covered by the conformance sweep instead:
/// a chi-squared test on a point mass has nothing to say.
pub fn discrete_sampler_cases() -> Vec<DiscSampleCase> {
    vec![
        disc_s("Binomial(10,0.5)", Binomial::new(10, 0.5).unwrap()),
        disc_s("Binomial(10,0.05)", Binomial::new(10, 0.05).unwrap()),
        disc_s("Binomial(50,0.8)", Binomial::new(50, 0.8).unwrap()),
        disc_s("Poisson(0.5)", Poisson::new(0.5).unwrap()),
        disc_s("Poisson(2.5)", Poisson::new(2.5).unwrap()),
        disc_s("Poisson(20)", Poisson::new(20.0).unwrap()),
        disc_s("Geometric(0.3)", Geometric::new(0.3).unwrap()),
        disc_s("Geometric(0.8)", Geometric::new(0.8).unwrap()),
        disc_s(
            "NegativeBinomial(5,0.5)",
            NegativeBinomial::new(5.0, 0.5).unwrap(),
        ),
        disc_s(
            "NegativeBinomial(2,0.3)",
            NegativeBinomial::new(2.0, 0.3).unwrap(),
        ),
        disc_s(
            "Hypergeometric(50,20,10)",
            Hypergeometric::new(50, 20, 10).unwrap(),
        ),
        disc_s(
            "BetaBinomial(10,2,3)",
            BetaBinomial::new(10, 2.0, 3.0).unwrap(),
        ),
        disc_s(
            "BetaBinomial(20,0.5,0.5)",
            BetaBinomial::new(20, 0.5, 0.5).unwrap(),
        ),
        disc_s(
            "Categorical[0.2,0.5,0.3]",
            Categorical::new(&[0.2, 0.5, 0.3]).unwrap(),
        ),
        disc_s(
            "Categorical[1,0,1]",
            Categorical::new(&[1.0, 0.0, 1.0]).unwrap(),
        ),
    ]
}
