//! The Normal-Inverse-Gamma conjugate: unknown mean **and** unknown variance.
//!
//! The four-parameter prior `(mu0, kappa, alpha, beta)` is the one people get
//! wrong, because `kappa` is a *prior sample size* on the mean rather than a
//! variance: the prior says "the mean is `mu0`, and I know that about as well as
//! `kappa` observations would tell me". The update adds the sample size to
//! `kappa` and the sum of squares to `beta`, which is why both have to move
//! together.
use super::super::{Conjugate, NigNatural};
use crate::{Continuous, InverseGamma, StudentT};
use bayes_core_ext::BayesError;
use euclid_numeric::Real;

/// Half of `ln(2π)`, the per-observation Gaussian normalizer constant.
const HALF_LN_TWO_PI: f64 = 0.918_938_533_204_672_7;

/// **Normal–Inverse-Gamma hyperparameters** `NIG(μ, λ, α, β)`.
///
/// This is the conjugate prior — and posterior — for a Gaussian with both mean
/// and variance unknown. The generative reading is
/// `σ² ~ InverseGamma(α, β)`, `μ | σ² ~ Normal(μ, σ²/λ)`, where `λ` is the
/// mean's pseudo-count. It is a *joint* law over `(μ, σ²)`; rather than expose a
/// bivariate density it offers its univariate marginals and the univariate
/// posterior-predictive, all of which are families already in the crate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalInverseGamma {
    mu: f64,
    lambda: f64,
    alpha: f64,
    beta: f64,
}

impl NormalInverseGamma {
    /// Construct `NIG(μ, λ, α, β)`. Requires finite `μ` and strictly positive
    /// `λ`, `α`, `β`.
    pub fn new(mu: f64, lambda: f64, alpha: f64, beta: f64) -> Result<Self, BayesError> {
        if !mu.is_finite() {
            return Err(BayesError::NonFinite {
                name: "mu",
                value: mu,
            });
        }
        for (name, value) in [("lambda", lambda), ("alpha", alpha), ("beta", beta)] {
            if !value.is_finite() {
                return Err(BayesError::NonFinite { name, value });
            }
            if value <= 0.0 {
                return Err(BayesError::NotPositive { name, value });
            }
        }
        Ok(Self {
            mu,
            lambda,
            alpha,
            beta,
        })
    }

    /// Mean location hyperparameter `μ`.
    pub fn mu(&self) -> f64 {
        self.mu
    }
    /// Mean pseudo-count `λ`.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
    /// Inverse-gamma shape `α`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    /// Inverse-gamma scale `β`.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Marginal law of the variance `σ²`: `InverseGamma(α, β)`. Infallible — the
    /// hyperparameters are positive by construction.
    pub fn variance_marginal(&self) -> InverseGamma {
        InverseGamma::new(self.alpha, self.beta).unwrap_or_else(|_| unreachable!())
    }

    /// Marginal law of the mean `μ`, integrating out `σ²`:
    /// a Student-t with `2α` degrees of freedom, location `μ`, scale
    /// `√(β / (α·λ))`.
    pub fn mean_marginal(&self) -> StudentT {
        let scale = (self.beta / (self.alpha * self.lambda)).sqrt();
        StudentT::new_location_scale(2.0 * self.alpha, self.mu, scale)
            .unwrap_or_else(|_| unreachable!())
    }

    /// Posterior-predictive law of a future observation, integrating out both
    /// `μ` and `σ²`: a Student-t with `2α` degrees of freedom, location `μ`,
    /// scale `√(β·(λ+1) / (α·λ))`.
    pub fn predictive(&self) -> StudentT {
        let scale = (self.beta * (self.lambda + 1.0) / (self.alpha * self.lambda)).sqrt();
        StudentT::new_location_scale(2.0 * self.alpha, self.mu, scale)
            .unwrap_or_else(|_| unreachable!())
    }

    /// Log normalizer `ln Z = ln Γ(α) − α ln β − ½ ln λ` of the NIG density.
    ///
    /// The log marginal likelihood of a Gaussian sample is the difference of the
    /// posterior and prior normalizers, less the per-observation `½ ln(2π)`.
    fn ln_normalizer(&self) -> f64 {
        self.alpha.ln_gamma() - self.alpha * self.beta.ln() - 0.5 * self.lambda.ln()
    }
}

/// **Normal–Inverse-Gamma.** A `NIG(μ₀, λ₀, α₀, β₀)` prior on a Gaussian's
/// unknown `(mean, variance)`; the posterior is again NIG, formed by adding the
/// `(1, x, x²)` sufficient statistic of each observation to the prior's natural
/// hyperparameters. The posterior-predictive is a Student-t.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalInverseGammaModel {
    prior: NormalInverseGamma,
}

impl NormalInverseGammaModel {
    pub fn new(prior: NormalInverseGamma) -> Self {
        Self { prior }
    }
    pub fn prior(&self) -> NormalInverseGamma {
        self.prior
    }

    /// Closed-form prior/posterior predictive log density for one future
    /// observation under this NIG state.
    pub fn ln_predictive(&self, x: f64) -> f64 {
        self.prior.predictive().ln_pdf(x)
    }

    /// Closed-form posterior-predictive log density for one future observation
    /// after observing `data`.
    pub fn ln_posterior_predictive(&self, data: &[f64], x: f64) -> f64 {
        self.posterior(data).predictive().ln_pdf(x)
    }

    /// Closed-form log probability of a future batch after observing `data`.
    pub fn ln_posterior_predictive_likelihood(&self, data: &[f64], future: &[f64]) -> f64 {
        Self::new(self.posterior(data)).ln_marginal_likelihood(future)
    }

    /// Sequential predictive factorization `Σᵢ ln p(xᵢ | x₁,…,xᵢ₋₁)`.
    ///
    /// In exact arithmetic this equals [`ln_marginal_likelihood`](Self::ln_marginal_likelihood).
    pub fn ln_sequential_predictive(&self, data: &[f64]) -> f64 {
        let mut state = self.prior;
        let mut ln_prob = 0.0;
        for &x in data {
            ln_prob += state.predictive().ln_pdf(x);
            state = Self::new(state).posterior(&[x]);
        }
        ln_prob
    }

    /// Closed-form log marginal likelihood `ln p(data)` with both the mean and
    /// the variance integrated out:
    /// `ln Z(posterior) − ln Z(prior) − (n/2) ln(2π)`, where
    /// `ln Z = ln Γ(α) − α ln β − ½ ln λ`.
    pub fn ln_marginal_likelihood(&self, data: &[f64]) -> f64 {
        if data.iter().any(|x| !x.is_finite()) {
            return f64::NAN;
        }
        let n = data.len() as f64;
        let posterior = self.posterior(data);
        posterior.ln_normalizer() - self.prior.ln_normalizer() - n * HALF_LN_TWO_PI
    }
}

impl Conjugate for NormalInverseGammaModel {
    type Datum = f64;
    type Posterior = NormalInverseGamma;
    type Predictive = StudentT;
    type Natural = NigNatural;

    /// One Gaussian observation `x`: the `(n, Σx, Σx²)` sufficient statistic, in
    /// the additive natural coordinates `(λ, λμ, 2α, 2β + λμ²)`, contributes
    /// `(1, x, 1, x²)`.
    fn statistic(&self, &x: &f64) -> NigNatural {
        NigNatural {
            lambda: 1.0,
            lambda_mu: x,
            two_alpha: 1.0,
            sum_sq: x * x,
        }
    }
    fn prior_natural(&self) -> NigNatural {
        let p = self.prior;
        NigNatural {
            lambda: p.lambda,
            lambda_mu: p.lambda * p.mu,
            two_alpha: 2.0 * p.alpha,
            sum_sq: 2.0 * p.beta + p.lambda * p.mu * p.mu,
        }
    }
    fn posterior_from_natural(&self, n: NigNatural) -> NormalInverseGamma {
        let lambda = n.lambda;
        let mu = n.lambda_mu / lambda;
        let alpha = n.two_alpha / 2.0;
        // 2β = (2β + λμ²) − λμ² = sum_sq − (λμ)²/λ.
        let beta = (n.sum_sq - n.lambda_mu * n.lambda_mu / lambda) / 2.0;
        NormalInverseGamma {
            mu,
            lambda,
            alpha,
            beta,
        }
    }
    /// The posterior-predictive: a Student-t over the next observation.
    fn posterior_predictive(&self, data: &[f64]) -> StudentT {
        self.posterior(data).predictive()
    }
}
