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
mod probability;
pub mod transformed;

pub use combinator::{Mixture, OrderStatistic, Truncated};
pub use conjugate::{
    BetaBernoulli, BetaBinomialModel, Conjugate, DirichletCategorical, DirichletMultinomial,
    GammaExponential, GammaPoisson, NaturalParams, NormalGamma, NormalInverseGamma,
    NormalInverseWishart, NormalKnownVariance, NormalWishart, precision_weighted_mean,
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
pub use divergence::{
    Alpha, Channel, ChiSquaredDivergence, ChiSquaredGen, Custom, Divergence, FGenerator,
    HellingerDistance, Jeffreys, JensenShannon, Kl, KlDivergence, RenyiDivergence, ReverseKl,
    SquaredHellinger, TotalVariation, TotalVariationGen, bhattacharyya_coefficient,
    conditional_entropy, cross_entropy, f_divergence, is_entropy_admissible, jensen_shannon,
    joint_entropy, max_entropy, mutual_information, normalized_entropy,
    normalized_mutual_information, shannon_entropy, variation_of_information,
};
pub use exponential_family::{CountSum, ExponentialFamily, MeanVar, SufficientStatistic, Trials};
pub use functional::{
    ContinuousCdfExt, ContinuousDensity, ContinuousDistributionExt, ContinuousFn,
    DiscreteDistributionExt, DiscreteFn, DiscreteMass,
};
pub use multivariate::{
    BinghamPotential, Dirichlet, DirichletMultinomialDistribution, InverseWishart, KentPotential,
    Multinomial, MultivariateContinuous, MultivariateMoments, MultivariateNormal,
    MultivariateStudentT, SphericalUniform, VonMisesFisher, WatsonS2, Wishart,
};
pub use probability::{
    PROBABILITY_TOLERANCE, brier_score, is_probability_vector,
    is_probability_vector_with_tolerance, is_row_stochastic, is_row_stochastic_matrix,
    is_row_stochastic_matrix_with_tolerance, is_row_stochastic_with_tolerance, log_loss, log_score,
    normalize_probability_rows, normalize_probability_weights, probability_sum,
    ranked_probability_score, spherical_score, validate_probability_vector,
    validate_row_stochastic, validate_row_stochastic_matrix,
    validate_row_stochastic_matrix_with_tolerance, validate_row_stochastic_with_tolerance,
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
///
/// [`mean`](Self::mean), [`variance`](Self::variance), and
/// [`moment2`](Self::moment2) are tied by the **computational-variance
/// identity** `Var(X) = E[X²] − E[X]²`, machine-checked in the `riemann` Lean
/// development (`RiemannLean/Bayes/Moments.lean`: `variance_eq_moment2_sub_sq`,
/// which also proves expectation is affine and variance nonnegative). The
/// identity is what lets `moment2` be derived for free from the other two —
/// and the same one-pass form underwrites the streaming variance in
/// [`statistics`](crate::statistics).
pub trait Moments {
    fn mean(&self) -> Option<f64> {
        None
    }
    fn variance(&self) -> Option<f64> {
        None
    }
    /// The raw **second moment** `E[X²] = Var(X) + E[X]²` — the
    /// computational-variance identity `Var(X) = E[X²] − E[X]²` rearranged.
    /// Provided for free from [`variance`](Self::variance) and
    /// [`mean`](Self::mean); `None` when either is undefined. The identity is
    /// proved generically in `riemann` (`Bayes/Moments.variance_eq_moment2_sub_sq`),
    /// so this default is correct for every family without per-distribution code.
    fn moment2(&self) -> Option<f64> {
        Some(self.variance()? + self.mean()?.powi(2))
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
