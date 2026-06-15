//! Normal-prior conjugate models: **Normal–Normal** (known variance) and
//! **Normal–Inverse-Gamma** (unknown mean and variance).

use super::{Conjugate, NigNatural, Pair};
use crate::distribution::{InverseGamma, Normal, StudentT};
use crate::error::BayesError;

/// **Normal–Normal (known variance).** A `Normal(μ₀, τ₀)` prior on the unknown mean
/// of a Normal likelihood whose standard deviation `obs_std` is known. The
/// posterior precision is `1/τ₀² + n/σ²` and the posterior mean is the
/// precision-weighted average of prior mean and sample mean.
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::Moments;

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
    }
}
