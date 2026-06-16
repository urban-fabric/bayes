//! Normal-prior conjugate models: **Normal–Normal** (known variance),
//! **Normal–Inverse-Gamma** (unknown mean and variance), and
//! **Normal-Gamma** (unknown mean and precision).

use std::f64::consts::PI;

use super::{Conjugate, NigNatural, Pair};
use crate::distribution::{Gamma, InverseGamma, Normal, StudentT};
use crate::error::BayesError;
use crate::function::ln_gamma;

/// **Normal–Normal (known variance).** A `Normal(μ₀, τ₀)` prior on the unknown mean
/// of a Normal likelihood whose standard deviation `obs_std` is known. The
/// posterior precision is `1/τ₀² + n/σ²` and the posterior mean is the
/// precision-weighted average of prior mean and sample mean.
///
/// The conjugacy mechanism — *why* the posterior is again Normal — is **completing the
/// square**, and the update formulas are exact algebra, machine-checked in `riemann`
/// `RiemannLean/Bayes/Conjugate.lean`: `normal_complete_square` (the prior + likelihood
/// exponent collapses to one Gaussian exponent in `θ`), `postPrec_additive` (the posterior
/// precision is additive in `n`, so streaming = batch), and `postMean_weighted` (the posterior
/// mean is the precision-weighted average). [`posterior_precision`](Self::posterior_precision),
/// [`shrinkage`](Self::shrinkage), and the free [`precision_weighted_mean`] expose that algebra.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalKnownVariance {
    prior: Normal,
    obs_std: f64,
}

impl NormalKnownVariance {
    /// `prior` is the prior over the mean; `obs_std` is the known likelihood σ.
    pub fn new(prior: Normal, obs_std: f64) -> Self {
        Self { prior, obs_std }
    }
    pub fn prior(&self) -> Normal {
        self.prior
    }

    /// The prior precision `1/τ₀²` (the prior's inverse variance).
    fn prior_precision(&self) -> f64 {
        let (_, tau0) = self.prior.params();
        1.0 / (tau0 * tau0)
    }

    /// The known observation precision `1/σ²`; `0` for a degenerate (`σ ≤ 0`) likelihood,
    /// which then contributes no information (matching [`statistic`](Conjugate::statistic)).
    fn obs_precision(&self) -> f64 {
        if self.obs_std <= 0.0 {
            0.0
        } else {
            1.0 / (self.obs_std * self.obs_std)
        }
    }

    /// **Posterior precision after `n` observations:** `1/τ₀² + n/σ²` — the prior precision
    /// plus `n` times the observation precision. Precision is *additive*, so splitting the data
    /// into any number of batches lands on the same posterior precision (riemann
    /// `postPrec_additive`) — the reason the streaming [`posterior`](Conjugate::posterior)
    /// equals the batch one. The posterior standard deviation is `1/√(posterior_precision)`.
    pub fn posterior_precision(&self, n: usize) -> f64 {
        self.prior_precision() + n as f64 * self.obs_precision()
    }

    /// **The shrinkage weight** `w = (n/σ²) / (1/τ₀² + n/σ²) ∈ [0, 1]` — the fraction of the way
    /// the posterior mean moves from the prior mean toward the sample mean, since the
    /// precision-weighted posterior mean is `μₙ = (1 − w)·μ₀ + w·x̄` (riemann `postMean_weighted`).
    /// `w → 0` with a sharp prior or little data (stay near μ₀); `w → 1` with many observations
    /// (follow the data). Returns `0` for the degenerate zero-precision case.
    pub fn shrinkage(&self, n: usize) -> f64 {
        let post = self.posterior_precision(n);
        if post <= 0.0 {
            return 0.0;
        }
        (n as f64 * self.obs_precision()) / post
    }

    /// Closed-form log marginal likelihood for observations from
    /// `Normal(μ, σ²)` with known `σ` and Normal prior over `μ`.
    ///
    /// This is the Gaussian integral obtained by completing the square in `μ`;
    /// no optimization or sampling is involved.
    pub fn ln_marginal_likelihood(&self, data: &[f64]) -> f64 {
        if self.obs_std <= 0.0 || data.iter().any(|x| !x.is_finite()) {
            return f64::NAN;
        }
        let (mu0, tau0) = self.prior.params();
        let sigma2 = self.obs_std * self.obs_std;
        let lambda0 = 1.0 / (tau0 * tau0);
        let eta0 = lambda0 * mu0;
        let n = data.len() as f64;
        let sum = data.iter().sum::<f64>();
        let sum_sq = data.iter().map(|x| x * x).sum::<f64>();
        let lambda_n = lambda0 + n / sigma2;
        let eta_n = eta0 + sum / sigma2;
        let quad = sum_sq / sigma2 + lambda0 * mu0 * mu0 - eta_n * eta_n / lambda_n;
        -0.5 * n * (2.0 * PI * sigma2).ln() + 0.5 * (lambda0.ln() - lambda_n.ln()) - 0.5 * quad
    }
}

/// **Precision-weighted (inverse-variance) pooling** of independent Gaussian estimates — the
/// algebra of the Normal–Normal update generalized to a list, and the operation behind sensor
/// fusion / Kalman measurement updates and fixed-effect (inverse-variance) meta-analysis. Each
/// estimate is a `(mean, precision)` pair (precision `= 1/variance`); the pooled belief is
///
/// - pooled **precision** `= Σ precisionᵢ` — information adds (riemann `postPrec_additive`);
/// - pooled **mean** `= (Σ precisionᵢ·meanᵢ) / Σ precisionᵢ` — precision-weighted
///   (`postMean_weighted`).
///
/// The pool is itself Gaussian (`normal_complete_square`) and independent of how the estimates
/// are grouped or ordered (precision additivity) — combining the prior with the data, or two
/// posteriors, gives the same answer. Non-positive precisions are skipped (an
/// infinitely-uncertain estimate carries no information). Returns `None` if no estimate has
/// positive precision.
pub fn precision_weighted_mean(estimates: &[(f64, f64)]) -> Option<(f64, f64)> {
    let mut total_precision = 0.0;
    let mut weighted_sum = 0.0;
    for &(mean, precision) in estimates {
        if precision > 0.0 {
            total_precision += precision;
            weighted_sum += precision * mean;
        }
    }
    (total_precision > 0.0).then_some((weighted_sum / total_precision, total_precision))
}

impl Conjugate for NormalKnownVariance {
    type Datum = f64;
    type Posterior = Normal;
    type Predictive = Normal;
    /// Natural coordinates `(precision, precision·mean)` — the space in which
    /// the Gaussian (known variance) update is additive.
    type Natural = Pair;

    /// One observation in natural coordinates: `T(x) = (1/σ², x/σ²)` for the
    /// known likelihood variance `σ²`. A non-positive `σ` contributes nothing
    /// (the model is degenerate; the posterior stays the prior).
    fn statistic(&self, &x: &f64) -> Pair {
        if self.obs_std <= 0.0 {
            return Pair::new(0.0, 0.0);
        }
        let prec = 1.0 / (self.obs_std * self.obs_std);
        Pair::new(prec, x * prec)
    }
    fn prior_natural(&self) -> Pair {
        let (mu0, tau0) = self.prior.params();
        let prec = 1.0 / (tau0 * tau0);
        Pair::new(prec, mu0 * prec)
    }
    fn posterior_from_natural(&self, p: Pair) -> Normal {
        if p.a <= 0.0 {
            return self.prior;
        }
        Normal::new(p.b / p.a, p.a.recip().sqrt()).unwrap_or(self.prior)
    }
    /// The posterior-predictive for a future observation:
    /// `Normal(μₙ, √(σₙ² + σ²))` — posterior uncertainty plus likelihood noise.
    fn posterior_predictive(&self, data: &[f64]) -> Normal {
        let post = self.posterior(data);
        let (m, s) = post.params();
        Normal::new(m, (s * s + self.obs_std * self.obs_std).sqrt()).unwrap_or(post)
    }
}

/// **Normal–Inverse-Gamma.** The conjugate prior for a Normal with *both* unknown
/// mean and variance: `μ | σ² ~ Normal(μ, σ²/λ)` and `σ² ~ InverseGamma(α, β)`. The
/// posterior is another NIG; the marginal of `μ` and the posterior-predictive are
/// location-scale [`StudentT`], and the marginal of `σ²` is [`InverseGamma`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalInverseGamma {
    mu: f64,
    lambda: f64,
    alpha: f64,
    beta: f64,
}

impl NormalInverseGamma {
    /// `mu` prior mean; `lambda` the prior pseudo-count on the mean; `alpha`/`beta`
    /// the Inverse-Gamma shape/scale on the variance (`lambda, alpha, beta > 0`).
    pub fn new(mu: f64, lambda: f64, alpha: f64, beta: f64) -> Result<Self, BayesError> {
        for (name, v) in [("lambda", lambda), ("alpha", alpha), ("beta", beta)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        if !mu.is_finite() {
            return Err(BayesError::NonFinite {
                name: "mu",
                value: mu,
            });
        }
        Ok(Self {
            mu,
            lambda,
            alpha,
            beta,
        })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// The marginal distribution of the variance `σ²`: `InverseGamma(α, β)`.
    pub fn variance_marginal(&self) -> InverseGamma {
        InverseGamma::new(self.alpha, self.beta).unwrap_or_else(|_| unreachable!())
    }
    /// The marginal distribution of the mean `μ`: Student-t with `ν = 2α`, location
    /// `μ`, scale `√(β/(αλ))`.
    pub fn mean_marginal(&self) -> StudentT {
        let scale = (self.beta / (self.alpha * self.lambda)).sqrt();
        StudentT::new_location_scale(2.0 * self.alpha, self.mu, scale)
            .unwrap_or_else(|_| unreachable!())
    }
    /// The (prior-)predictive of a future observation: Student-t with `ν = 2α`,
    /// location `μ`, scale `√(β(λ+1)/(αλ))`.
    pub fn predictive(&self) -> StudentT {
        let scale = (self.beta * (self.lambda + 1.0) / (self.alpha * self.lambda)).sqrt();
        StudentT::new_location_scale(2.0 * self.alpha, self.mu, scale)
            .unwrap_or_else(|_| unreachable!())
    }

    /// Closed-form log marginal likelihood for Normal observations with both
    /// unknown mean and variance integrated out under the Normal-Inverse-Gamma
    /// prior.
    pub fn ln_marginal_likelihood(&self, data: &[f64]) -> f64 {
        if data.iter().any(|x| !x.is_finite()) {
            return f64::NAN;
        }
        let post = self.posterior(data);
        let n = data.len() as f64;
        ln_gamma(post.alpha()) - ln_gamma(self.alpha) + self.alpha * self.beta.ln()
            - post.alpha() * post.beta().ln()
            + 0.5 * (self.lambda.ln() - post.lambda().ln())
            - 0.5 * n * (2.0 * PI).ln()
    }
}

/// **Normal-Gamma.** The conjugate prior for a Normal with unknown mean and
/// precision: `μ | τ ~ Normal(μ₀, 1/(λτ))` and `τ ~ Gamma(α, β)`, where `β` is
/// the Gamma rate. This is the precision-parameter twin of
/// [`NormalInverseGamma`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalGamma {
    mu: f64,
    lambda: f64,
    alpha: f64,
    beta: f64,
}

impl NormalGamma {
    /// `mu` prior mean; `lambda` the prior pseudo-count on the mean; `alpha`/`beta`
    /// the Gamma shape/rate on the precision (`lambda, alpha, beta > 0`).
    pub fn new(mu: f64, lambda: f64, alpha: f64, beta: f64) -> Result<Self, BayesError> {
        for (name, v) in [("lambda", lambda), ("alpha", alpha), ("beta", beta)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        if !mu.is_finite() {
            return Err(BayesError::NonFinite {
                name: "mu",
                value: mu,
            });
        }
        Ok(Self {
            mu,
            lambda,
            alpha,
            beta,
        })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// The marginal distribution of the precision `τ`: `Gamma(α, β)`.
    pub fn precision_marginal(&self) -> Gamma {
        Gamma::new(self.alpha, self.beta).unwrap_or_else(|_| unreachable!())
    }

    /// Conditional distribution of the mean `μ | τ`.
    pub fn mean_conditional(&self, precision: f64) -> Result<Normal, BayesError> {
        if !precision.is_finite() {
            return Err(BayesError::NonFinite {
                name: "precision",
                value: precision,
            });
        }
        if precision <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "precision",
                value: precision,
            });
        }
        Normal::new(self.mu, (self.lambda * precision).recip().sqrt())
    }

    /// The marginal distribution of the mean `μ`: Student-t with `ν = 2α`, location
    /// `μ`, scale `√(β/(αλ))`.
    pub fn mean_marginal(&self) -> StudentT {
        let scale = (self.beta / (self.alpha * self.lambda)).sqrt();
        StudentT::new_location_scale(2.0 * self.alpha, self.mu, scale)
            .unwrap_or_else(|_| unreachable!())
    }

    /// The (prior-)predictive of a future observation: Student-t with `ν = 2α`,
    /// location `μ`, scale `√(β(λ+1)/(αλ))`.
    pub fn predictive(&self) -> StudentT {
        let scale = (self.beta * (self.lambda + 1.0) / (self.alpha * self.lambda)).sqrt();
        StudentT::new_location_scale(2.0 * self.alpha, self.mu, scale)
            .unwrap_or_else(|_| unreachable!())
    }

    /// Closed-form log marginal likelihood for Normal observations with both
    /// unknown mean and precision integrated out under the Normal-Gamma prior.
    pub fn ln_marginal_likelihood(&self, data: &[f64]) -> f64 {
        if data.iter().any(|x| !x.is_finite()) {
            return f64::NAN;
        }
        let post = self.posterior(data);
        let n = data.len() as f64;
        ln_gamma(post.alpha()) - ln_gamma(self.alpha) + self.alpha * self.beta.ln()
            - post.alpha() * post.beta().ln()
            + 0.5 * (self.lambda.ln() - post.lambda().ln())
            - 0.5 * n * (2.0 * PI).ln()
    }
}

impl Conjugate for NormalInverseGamma {
    type Datum = f64;
    type Posterior = NormalInverseGamma;
    type Predictive = StudentT;
    /// Natural coordinates `(λ, λμ, 2α, 2β + λμ²)` — the space in which the NIG
    /// update is additive (riemann `normalInverseGamma_online_eq_batch`).
    type Natural = NigNatural;

    /// One observation: `T(x) = (1, x, 1, x²)` — the `λ` and `2α` counts both
    /// gain `1`, matching riemann's `nigStat x = (1, x, x²)`.
    fn statistic(&self, &x: &f64) -> NigNatural {
        NigNatural {
            lambda: 1.0,
            lambda_mu: x,
            two_alpha: 1.0,
            sum_sq: x * x,
        }
    }
    fn prior_natural(&self) -> NigNatural {
        NigNatural {
            lambda: self.lambda,
            lambda_mu: self.lambda * self.mu,
            two_alpha: 2.0 * self.alpha,
            sum_sq: 2.0 * self.beta + self.lambda * self.mu * self.mu,
        }
    }
    fn posterior_from_natural(&self, n: NigNatural) -> NormalInverseGamma {
        if n.lambda <= 0.0 {
            return *self;
        }
        let mu_n = n.lambda_mu / n.lambda;
        let alpha_n = n.two_alpha / 2.0;
        // β recovered from the natural sum-of-squares: 2β = (2β+λμ²) − λμ².
        let beta_n = (n.sum_sq - n.lambda_mu * n.lambda_mu / n.lambda) / 2.0;
        NormalInverseGamma::new(mu_n, n.lambda, alpha_n, beta_n).unwrap_or(*self)
    }
    /// The posterior-predictive after observing `data`: the posterior NIG's
    /// location-scale Student-t [`predictive`](Self::predictive).
    fn posterior_predictive(&self, data: &[f64]) -> StudentT {
        self.posterior(data).predictive()
    }
}

impl Conjugate for NormalGamma {
    type Datum = f64;
    type Posterior = NormalGamma;
    type Predictive = StudentT;
    /// Natural coordinates `(λ, λμ, 2α, 2β + λμ²)`, shared with
    /// [`NormalInverseGamma`]. The difference is interpretive: `β` is a Gamma
    /// rate for precision instead of an Inverse-Gamma scale for variance.
    type Natural = NigNatural;

    fn statistic(&self, &x: &f64) -> NigNatural {
        NigNatural {
            lambda: 1.0,
            lambda_mu: x,
            two_alpha: 1.0,
            sum_sq: x * x,
        }
    }

    fn prior_natural(&self) -> NigNatural {
        NigNatural {
            lambda: self.lambda,
            lambda_mu: self.lambda * self.mu,
            two_alpha: 2.0 * self.alpha,
            sum_sq: 2.0 * self.beta + self.lambda * self.mu * self.mu,
        }
    }

    fn posterior_from_natural(&self, n: NigNatural) -> NormalGamma {
        if n.lambda <= 0.0 {
            return *self;
        }
        let mu_n = n.lambda_mu / n.lambda;
        let alpha_n = n.two_alpha / 2.0;
        let beta_n = (n.sum_sq - n.lambda_mu * n.lambda_mu / n.lambda) / 2.0;
        NormalGamma::new(mu_n, n.lambda, alpha_n, beta_n).unwrap_or(*self)
    }

    fn posterior_predictive(&self, data: &[f64]) -> StudentT {
        self.posterior(data).predictive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::Moments;
    use crate::distribution::conjugate::NaturalParams;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn normal_known_variance_shrinks_toward_data() {
        let model = NormalKnownVariance::new(Normal::new(0.0, 10.0).unwrap(), 1.0);
        let data = [4.8, 5.0, 5.2, 4.9, 5.1];
        let post = model.posterior(&data);
        close(post.mean().unwrap(), 5.0, 0.02);
        assert!(post.variance().unwrap() < 1.0);
        let pred = model.posterior_predictive(&data);
        assert!(pred.variance().unwrap() > post.variance().unwrap());
        assert!(model.ln_marginal_likelihood(&data).is_finite());
        close(model.ln_marginal_likelihood(&[]), 0.0, 1e-12);
        assert!(model.ln_marginal_likelihood(&[f64::NAN]).is_nan());
    }

    #[test]
    fn posterior_precision_and_shrinkage_match_the_posterior() {
        // prior std 10 ⇒ prior precision 0.01; obs std 1 ⇒ obs precision 1.
        let model = NormalKnownVariance::new(Normal::new(0.0, 10.0).unwrap(), 1.0);
        let data = [4.8, 5.0, 5.2, 4.9, 5.1]; // n = 5
        // posterior precision = 0.01 + 5·1 = 5.01 (additive — postPrec_additive).
        close(model.posterior_precision(5), 5.01, 1e-12);
        // the posterior std is 1/√precision.
        close(
            model.posterior(&data).variance().unwrap(),
            1.0 / 5.01,
            1e-12,
        );
        // shrinkage w = 5/5.01 (almost all the way to the data); μₙ = (1−w)·μ₀ + w·x̄.
        let w = model.shrinkage(5);
        close(w, 5.0 / 5.01, 1e-12);
        let xbar = data.iter().sum::<f64>() / 5.0;
        close(
            model.posterior(&data).mean().unwrap(),
            (1.0 - w) * 0.0 + w * xbar,
            1e-12,
        );
        // no data ⇒ posterior precision is the prior precision, shrinkage 0.
        close(model.posterior_precision(0), 0.01, 1e-12);
        close(model.shrinkage(0), 0.0, 1e-12);
    }

    #[test]
    fn precision_weighted_mean_pools_and_matches_the_conjugate_update() {
        // Two equally-precise estimates average; precision adds.
        let (m, p) = precision_weighted_mean(&[(0.0, 1.0), (10.0, 1.0)]).unwrap();
        close(m, 5.0, 1e-12);
        close(p, 2.0, 1e-12);
        // Order- and grouping-independent (precision additivity / completing the square).
        let est = [(1.0, 2.0), (4.0, 1.0), (9.0, 0.5)];
        let direct = precision_weighted_mean(&est).unwrap();
        let mut rev = est;
        rev.reverse();
        let reversed = precision_weighted_mean(&rev).unwrap();
        close(direct.0, reversed.0, 1e-12);
        close(direct.1, reversed.1, 1e-12);
        let (ab_m, ab_p) = precision_weighted_mean(&est[..2]).unwrap();
        let grouped = precision_weighted_mean(&[(ab_m, ab_p), est[2]]).unwrap();
        close(direct.0, grouped.0, 1e-12);
        close(direct.1, grouped.1, 1e-12);
        // Pooling [prior, data…] reproduces the Normal–Normal posterior.
        let model = NormalKnownVariance::new(Normal::new(0.0, 10.0).unwrap(), 1.0);
        let data = [4.8, 5.0, 5.2, 4.9, 5.1];
        let mut estimates = vec![(0.0, 0.01)]; // prior (μ₀, 1/τ₀²)
        estimates.extend(data.iter().map(|&x| (x, 1.0))); // each datum (x, 1/σ²)
        let (pm, pp) = precision_weighted_mean(&estimates).unwrap();
        let post = model.posterior(&data);
        close(pm, post.mean().unwrap(), 1e-12);
        close(pp, model.posterior_precision(data.len()), 1e-12);
        // Degenerate inputs: no positive-precision estimate ⇒ None.
        assert!(precision_weighted_mean(&[]).is_none());
        assert!(precision_weighted_mean(&[(1.0, 0.0), (2.0, -1.0)]).is_none());
    }

    #[test]
    fn normal_inverse_gamma_posterior_and_predictive() {
        let model = NormalInverseGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
        let data = [1.0, 2.0, 3.0];
        let post = model.posterior(&data);
        close(post.mu(), 1.5, 1e-12);
        close(post.lambda(), 4.0, 1e-12);
        close(post.alpha(), 3.5, 1e-12);
        close(post.beta(), 3.5, 1e-12);
        let mu_t = post.mean_marginal();
        close(mu_t.freedom(), 7.0, 1e-12);
        close(mu_t.location(), 1.5, 1e-12);
        close(mu_t.scale(), 0.5, 1e-12);
        let pred = model.posterior_predictive(&data);
        close(pred.freedom(), 7.0, 1e-12);
        close(pred.scale(), 1.25_f64.sqrt(), 1e-12);
        let ig = post.variance_marginal();
        close(ig.shape(), 3.5, 1e-12);
        close(ig.scale(), 3.5, 1e-12);
        let evidence = ln_gamma(3.5) - ln_gamma(2.0) - 3.5 * 3.5_f64.ln()
            + 0.5 * (1.0_f64.ln() - 4.0_f64.ln())
            - 1.5 * (2.0 * PI).ln();
        close(model.ln_marginal_likelihood(&data), evidence, 1e-12);
        close(model.ln_marginal_likelihood(&[]), 0.0, 1e-12);
        assert!(model.ln_marginal_likelihood(&[f64::NAN]).is_nan());
    }

    #[test]
    fn normal_gamma_matches_precision_parameter_closed_form() {
        let model = NormalGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
        let data = [1.0, 2.0, 3.0];
        let post = model.posterior(&data);
        close(post.mu(), 1.5, 1e-12);
        close(post.lambda(), 4.0, 1e-12);
        close(post.alpha(), 3.5, 1e-12);
        close(post.beta(), 3.5, 1e-12);

        let precision = post.precision_marginal();
        close(precision.shape(), 3.5, 1e-12);
        close(precision.rate(), 3.5, 1e-12);
        close(post.mean_marginal().scale(), 0.5, 1e-12);
        close(
            post.mean_conditional(2.0).unwrap().variance().unwrap(),
            0.125,
            1e-12,
        );

        let pred = model.posterior_predictive(&data);
        close(pred.freedom(), 7.0, 1e-12);
        close(pred.location(), 1.5, 1e-12);
        close(pred.scale(), 1.25_f64.sqrt(), 1e-12);

        let mut natural = model.posterior_natural(&[1.0, 2.0]);
        natural = natural.combine(&model.statistic(&3.0));
        let online = model.posterior_from_natural(natural);
        close(online.mu(), post.mu(), 1e-12);
        close(online.beta(), post.beta(), 1e-12);

        let nig = NormalInverseGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
        close(
            model.ln_marginal_likelihood(&data),
            nig.ln_marginal_likelihood(&data),
            1e-12,
        );
        close(model.ln_marginal_likelihood(&[]), 0.0, 1e-12);
        assert!(model.ln_marginal_likelihood(&[f64::NAN]).is_nan());
        assert!(NormalGamma::new(0.0, 0.0, 2.0, 1.0).is_err());
        assert!(model.mean_conditional(0.0).is_err());
    }
}
