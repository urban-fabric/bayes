//! The **distribution trait spine** plus the concrete distributions.
//!
//! A distribution is an interface, not just a parameter bag: density/mass (PDF/PMF),
//! cumulative + inverse (CDF / quantile), support bounds, analytic moments, and
//! sampling. Those concerns are split into small traits so a distribution
//! implements exactly the ones it supports, and generic code can be written against
//! any of them.
//!
//! Sampling stays on [`rand::distr::Distribution`] (re-exported at the crate root),
//! so `rng.sample(&dist)` works for every distribution here.

pub mod combinator;
pub mod conjugate;
pub mod constraint;
pub mod continuous;
pub mod discrete;
mod divergence;
pub mod exponential_family;
pub mod functional;
pub mod multivariate;
mod numeric;
pub mod transformed;

pub use combinator::{Mixture, Truncated};
pub use conjugate::{
    BetaBernoulli, BetaBinomialModel, Conjugate, DirichletCategorical, DirichletMultinomial,
    GammaExponential, GammaPoisson, NormalInverseGamma, NormalKnownVariance,
};
pub use constraint::{Constrained, Constraint};
pub use continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel, HalfNormal,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh, StudentT,
    Triangular, Uniform, Uniform2D, VonMises, Wald, Weibull,
};
pub use discrete::{
    Bernoulli, BetaBinomial, Binomial, Categorical, DiscreteUniform, Geometric, Hypergeometric,
    NegativeBinomial, Poisson,
};
pub use divergence::KlDivergence;
pub use exponential_family::{CountSum, ExponentialFamily, MeanVar, SufficientStatistic, Trials};
pub use functional::{
    ContinuousCdfExt, ContinuousDensity, ContinuousDistributionExt, ContinuousFn,
    DiscreteDistributionExt, DiscreteFn, DiscreteMass,
};
pub use multivariate::{
    Dirichlet, Multinomial, MultivariateContinuous, MultivariateMoments, MultivariateNormal,
};
pub use transformed::{Affine, Bijector, Exp, Sigmoid, Transformed};

pub(crate) use continuous::normal;

/// Continuous probability **density** — the PDF `f(x)`.
pub trait Continuous<X = f64> {
    /// The density at `x`.
    fn pdf(&self, x: X) -> f64;
    /// `ln f(x)`. Override with a numerically stable closed form where possible.
    fn ln_pdf(&self, x: X) -> f64
    where
        X: Copy,
    {
        self.pdf(x).ln()
    }
}

/// Discrete probability **mass** — the PMF `P(X = k)`.
pub trait Discrete<X = u64> {
    /// The mass at `k`.
    fn pmf(&self, k: X) -> f64;
    /// `ln P(X = k)`. Override with a stable closed form where possible.
    fn ln_pmf(&self, k: X) -> f64
    where
        X: Copy,
    {
        self.pmf(k).ln()
    }
}

/// The support bounds — the infimum/supremum of the domain (`±∞` allowed).
pub trait Support<X = f64> {
    fn min(&self) -> X;
    fn max(&self) -> X;
}

/// Cumulative distribution, survival, and inverse (quantile) for a **continuous**
/// distribution.
pub trait ContinuousCdf<X = f64>: Support<X> {
    /// `P(X ≤ x)`.
    fn cdf(&self, x: X) -> f64;
    /// Survival `P(X > x) = 1 - cdf(x)`. Override for tail precision.
    fn sf(&self, x: X) -> f64
    where
        X: Copy,
    {
        1.0 - self.cdf(x)
    }
    /// The quantile / inverse CDF: the `x` with `cdf(x) = p`, `p ∈ [0, 1]`.
    fn inverse_cdf(&self, p: f64) -> X;
    /// `ln P(X ≤ x)`. The default `cdf(x).ln()` loses all precision once the CDF
    /// underflows to `0` in the left tail; override with a direct log form there.
    fn ln_cdf(&self, x: X) -> f64 {
        self.cdf(x).ln()
    }
    /// `ln P(X > x)`. Override for right-tail precision (same motivation as
    /// [`ln_cdf`](Self::ln_cdf)).
    fn ln_sf(&self, x: X) -> f64
    where
        X: Copy,
    {
        self.sf(x).ln()
    }
    /// The **inverse survival** function: the `x` with `sf(x) = p`. Default
    /// `inverse_cdf(1 - p)`; override when a direct form is more accurate for
    /// small `p` (extreme upper quantiles).
    fn isf(&self, p: f64) -> X {
        self.inverse_cdf(1.0 - p)
    }
}

/// Cumulative distribution, survival, and inverse for a **discrete** distribution.
pub trait DiscreteCdf<X = u64>: Support<X> {
    /// `P(X ≤ k)`.
    fn cdf(&self, k: X) -> f64;
    /// Survival `P(X > k) = 1 - cdf(k)`.
    fn sf(&self, k: X) -> f64
    where
        X: Copy,
    {
        1.0 - self.cdf(k)
    }
    /// The smallest `k` with `cdf(k) ≥ p`, `p ∈ [0, 1]`.
    fn inverse_cdf(&self, p: f64) -> X;
    /// `ln P(X ≤ k)`. Override for left-tail precision.
    fn ln_cdf(&self, k: X) -> f64 {
        self.cdf(k).ln()
    }
    /// `ln P(X > k)`. Override for right-tail precision.
    fn ln_sf(&self, k: X) -> f64
    where
        X: Copy,
    {
        self.sf(k).ln()
    }
    /// The **inverse survival** function: the smallest `k` with `sf(k) ≤ p`.
    /// Default `inverse_cdf(1 - p)`.
    fn isf(&self, p: f64) -> X {
        self.inverse_cdf(1.0 - p)
    }
}

/// Analytic **moments** and summary points — all optional, since some are
/// undefined for some parameters (e.g. the Cauchy mean/variance). `None` means
/// "does not exist / not implemented".
pub trait Moments {
    fn mean(&self) -> Option<f64> {
        None
    }
    fn variance(&self) -> Option<f64> {
        None
    }
    fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }
    /// Third standardized moment.
    fn skewness(&self) -> Option<f64> {
        None
    }
    /// Excess kurtosis (kurtosis − 3).
    fn kurtosis(&self) -> Option<f64> {
        None
    }
    /// Differential / Shannon entropy, in nats.
    fn entropy(&self) -> Option<f64> {
        None
    }
    fn median(&self) -> Option<f64> {
        None
    }
    fn mode(&self) -> Option<f64> {
        None
    }
}

/// Fit a distribution's parameters from a sample (MLE or method of moments).
/// `None` when the sample is too small or degenerate to fit.
pub trait Estimate: Sized {
    fn from_sample(data: &[f64]) -> Option<Self>;
}
