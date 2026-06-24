//! Gaussian-prior conjugate models (univariate): **Normal–Normal** (known
//! observation variance) and **Normal–Inverse-Gamma** (unknown mean *and*
//! variance).
//!
//! Both observe scalar real data, so they live here alongside the other scalar
//! conjugates rather than in the matrix-coupled sibling crate `ito` (whose
//! `normal` model is the *multivariate*-Normal / Normal-Inverse-Wishart update).
//! The Normal–Inverse-Gamma update accumulates the `(n, Σx, Σx²)` sufficient
//! statistic in the [`NigNatural`] carrier defined by `bayes-distribution`,
//! exactly as named by riemann's `nigStat` (`Statistics/ConjugateFamilies.lean`);
//! the Normal–Normal precision-weighted update is the one verified by riemann's
//! `ExactArith/NormalPosterior.lean::posterior_is_conjugate`.

use super::{Conjugate, NigNatural, Pair};
use crate::{Continuous, InverseGamma, Normal, StudentT};
use bayes_core_ext::BayesError;
use euclid::Real;

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

/// **Normal–Normal (known variance).** A `Normal(μ₀, τ₀)` prior on the unknown
/// mean of a Gaussian whose observation standard deviation `obs_std` is known;
/// the posterior is again Normal, with precision the prior precision plus `n`
/// observation precisions, and mean the precision-weighted average of the prior
/// mean and the data. The posterior-predictive is Normal with the observation
/// variance added back. This is the update verified by riemann's
/// `ExactArith/NormalPosterior.lean`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalKnownVariance {
    prior: Normal,
    obs_std: f64,
}

impl NormalKnownVariance {
    pub fn new(prior: Normal, obs_std: f64) -> Self {
        Self { prior, obs_std }
    }
    pub fn prior(&self) -> Normal {
        self.prior
    }
    pub fn obs_std(&self) -> f64 {
        self.obs_std
    }

    fn obs_precision(&self) -> f64 {
        1.0 / (self.obs_std * self.obs_std)
    }

    /// Closed-form prior/posterior predictive log density for one future
    /// observation under this Normal state.
    pub fn ln_predictive(&self, x: f64) -> f64 {
        predictive(self.prior, self.obs_std).ln_pdf(x)
    }

    /// Closed-form posterior-predictive log density for one future observation
    /// after observing `data`.
    pub fn ln_posterior_predictive(&self, data: &[f64], x: f64) -> f64 {
        predictive(self.posterior(data), self.obs_std).ln_pdf(x)
    }

    /// Closed-form log probability of a future batch after observing `data`.
    pub fn ln_posterior_predictive_likelihood(&self, data: &[f64], future: &[f64]) -> f64 {
        Self::new(self.posterior(data), self.obs_std).ln_marginal_likelihood(future)
    }

    /// Sequential predictive factorization `Σᵢ ln p(xᵢ | x₁,…,xᵢ₋₁)`.
    ///
    /// In exact arithmetic this equals [`ln_marginal_likelihood`](Self::ln_marginal_likelihood).
    pub fn ln_sequential_predictive(&self, data: &[f64]) -> f64 {
        let mut state = self.prior;
        let mut ln_prob = 0.0;
        for &x in data {
            ln_prob += predictive(state, self.obs_std).ln_pdf(x);
            state = Self::new(state, self.obs_std).posterior(&[x]);
        }
        ln_prob
    }

    /// Closed-form log marginal likelihood `ln p(data)` with the mean integrated
    /// out under the known observation variance:
    ///
    /// `−½ n ln(2πσ²) + ½(ln Λ₀ − ln Λₙ) − ½ C`, where `Λ₀`, `Λₙ` are the prior
    /// and posterior precisions and
    /// `C = Λ₀ μ₀² + σ⁻² Σx² − Λₙ μₙ²` is the θ-free constant from completing the
    /// square (riemann `posterior_is_conjugate`).
    pub fn ln_marginal_likelihood(&self, data: &[f64]) -> f64 {
        if data.iter().any(|x| !x.is_finite()) {
            return f64::NAN;
        }
        let (mu0, tau0) = self.prior.params();
        let obs_prec = self.obs_precision();
        let prior_prec = 1.0 / (tau0 * tau0);
        let n = data.len() as f64;
        let sum_sq = data.iter().map(|&x| x * x).sum::<f64>();
        let post = self.posterior(data);
        let (post_mean, post_std) = post.params();
        let post_prec = 1.0 / (post_std * post_std);
        let c = prior_prec * mu0 * mu0 + obs_prec * sum_sq - post_prec * post_mean * post_mean;
        -0.5 * n * (2.0 * std::f64::consts::PI * self.obs_std * self.obs_std).ln()
            + 0.5 * (prior_prec.ln() - post_prec.ln())
            - 0.5 * c
    }
}

/// The Normal–Normal posterior-predictive: the posterior mean with the
/// observation variance added back into the spread. Infallible — the spread is
/// positive by construction.
fn predictive(posterior: Normal, obs_std: f64) -> Normal {
    let (mean, std) = posterior.params();
    Normal::new(mean, (std * std + obs_std * obs_std).sqrt()).unwrap_or(posterior)
}

impl Conjugate for NormalKnownVariance {
    type Datum = f64;
    type Posterior = Normal;
    type Predictive = Normal;
    type Natural = Pair;

    /// One observation `x` in precision coordinates: `(σ⁻², x·σ⁻²)` — adds one
    /// observation precision and its precision-weighted value.
    fn statistic(&self, &x: &f64) -> Pair {
        let obs_prec = self.obs_precision();
        Pair::new(obs_prec, x * obs_prec)
    }
    fn prior_natural(&self) -> Pair {
        let (mu0, tau0) = self.prior.params();
        let prior_prec = 1.0 / (tau0 * tau0);
        Pair::new(prior_prec, mu0 * prior_prec)
    }
    /// `a` is the posterior precision, `b` the precision-weighted mean, so the
    /// posterior is `Normal(b/a, √(1/a))`.
    fn posterior_from_natural(&self, p: Pair) -> Normal {
        Normal::new(p.b / p.a, p.a.recip().sqrt()).unwrap_or(self.prior)
    }
    /// The posterior-predictive: `Normal(μₙ, √(σₙ² + σ²))`.
    fn posterior_predictive(&self, data: &[f64]) -> Normal {
        predictive(self.posterior(data), self.obs_std)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::NaturalParams;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn nig_posterior_matches_textbook_update() {
        // NIG(0, 1, 1, 1) prior, data {2, 4}: n=2, Σx=6, Σx²=20.
        let model =
            NormalInverseGammaModel::new(NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0).unwrap());
        let data = [2.0, 4.0];
        let post = model.posterior(&data);
        // λ=λ₀+n=3, μ=(λ₀μ₀+Σx)/λ=2, α=α₀+n/2=2,
        // β=β₀+½[Σ(x-x̄)² + λ₀n(x̄-μ₀)²/λ]=1+½[2+6]=5.
        close(post.lambda(), 3.0, 1e-12);
        close(post.mu(), 2.0, 1e-12);
        close(post.alpha(), 2.0, 1e-12);
        close(post.beta(), 5.0, 1e-12);

        // Marginals are families already in the crate.
        let var_marg = post.variance_marginal();
        close(var_marg.shape(), 2.0, 1e-12);
        close(var_marg.scale(), 5.0, 1e-12);
        let mean_marg = post.mean_marginal();
        close(mean_marg.freedom(), 4.0, 1e-12);
        close(mean_marg.location(), 2.0, 1e-12);
        close(mean_marg.scale(), (5.0_f64 / (2.0 * 3.0)).sqrt(), 1e-12);

        // Predictive Student-t: ν=2α=4, loc=μ=2, scale²=β(λ+1)/(αλ)=20/6.
        let pred = model.posterior_predictive(&data);
        close(pred.freedom(), 4.0, 1e-12);
        close(pred.location(), 2.0, 1e-12);
        close(pred.scale(), (5.0_f64 * 4.0 / (2.0 * 3.0)).sqrt(), 1e-12);
    }

    #[test]
    fn nig_online_equals_batch_and_order_independent() {
        let model =
            NormalInverseGammaModel::new(NormalInverseGamma::new(0.5, 2.0, 3.0, 4.0).unwrap());
        let d1 = [1.0, -2.0, 0.5];
        let d2 = [3.0, 4.0];
        let all: Vec<f64> = d1.iter().chain(d2.iter()).copied().collect();

        let batch = model.posterior(&all);
        let mut online = model.posterior_natural(&d1);
        for x in d2 {
            online = online.combine(&model.statistic(&x));
        }
        let online = model.posterior_from_natural(online);
        close(online.mu(), batch.mu(), 1e-12);
        close(online.lambda(), batch.lambda(), 1e-12);
        close(online.alpha(), batch.alpha(), 1e-12);
        close(online.beta(), batch.beta(), 1e-12);

        let shuffled = [4.0, 0.5, 3.0, 1.0, -2.0];
        let permuted = model.posterior(&shuffled);
        close(permuted.mu(), batch.mu(), 1e-12);
        close(permuted.beta(), batch.beta(), 1e-12);
    }

    #[test]
    fn nig_sequential_predictive_equals_marginal_likelihood() {
        let model =
            NormalInverseGammaModel::new(NormalInverseGamma::new(0.0, 1.0, 2.0, 3.0).unwrap());
        let data = [1.5, -0.5, 2.0, 0.25];
        close(
            model.ln_sequential_predictive(&data),
            model.ln_marginal_likelihood(&data),
            1e-9,
        );
        // ln Z(post) − ln Z(prior) − (n/2) ln(2π), computed independently.
        let post = model.posterior(&data);
        let expect =
            (post.alpha().ln_gamma() - post.alpha() * post.beta().ln() - 0.5 * post.lambda().ln())
                - (2.0_f64.ln_gamma() - 2.0 * 3.0_f64.ln() - 0.5 * 1.0_f64.ln())
                - 0.5 * data.len() as f64 * (2.0 * std::f64::consts::PI).ln();
        close(model.ln_marginal_likelihood(&data), expect, 1e-12);
        assert!(model.ln_marginal_likelihood(&[f64::INFINITY]).is_nan());
    }

    #[test]
    fn nig_predictive_log_density_round_trips() {
        let model =
            NormalInverseGammaModel::new(NormalInverseGamma::new(0.0, 1.0, 2.0, 3.0).unwrap());
        let data = [1.0, 2.0];
        close(
            model.ln_predictive(0.7).exp(),
            model.posterior(&[]).predictive().pdf(0.7),
            1e-12,
        );
        close(
            model.ln_posterior_predictive(&data, 0.7),
            model.posterior(&data).predictive().ln_pdf(0.7),
            1e-12,
        );
        close(
            model.ln_posterior_predictive_likelihood(&data, &[0.3, 1.1]),
            NormalInverseGammaModel::new(model.posterior(&data))
                .ln_marginal_likelihood(&[0.3, 1.1]),
            1e-12,
        );
    }

    #[test]
    fn nig_constructor_rejects_nonpositive_scale() {
        assert!(NormalInverseGamma::new(0.0, 0.0, 1.0, 1.0).is_err());
        assert!(NormalInverseGamma::new(0.0, 1.0, -1.0, 1.0).is_err());
        assert!(NormalInverseGamma::new(0.0, 1.0, 1.0, 0.0).is_err());
        assert!(NormalInverseGamma::new(f64::NAN, 1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn normal_known_variance_posterior_is_precision_weighted() {
        // Normal(0, 1) prior, obs σ=1, data {2, 4}: prior_prec=1, n=2, Σx=6.
        let model = NormalKnownVariance::new(Normal::new(0.0, 1.0).unwrap(), 1.0);
        let data = [2.0, 4.0];
        let post = model.posterior(&data);
        let (mean, std) = post.params();
        // post_prec = 1 + 2 = 3, post_mean = (0·1 + 6·1)/3 = 2.
        close(mean, 2.0, 1e-12);
        close(1.0 / (std * std), 3.0, 1e-12);

        // Predictive adds back the observation variance: Normal(2, √(1/3 + 1)).
        let pred = model.posterior_predictive(&data);
        let (pmean, pstd) = pred.params();
        close(pmean, 2.0, 1e-12);
        close(pstd, (1.0_f64 / 3.0 + 1.0).sqrt(), 1e-12);
    }

    #[test]
    fn normal_known_variance_online_equals_batch_and_order_independent() {
        let model = NormalKnownVariance::new(Normal::new(1.0, 2.0).unwrap(), 1.5);
        let d1 = [0.5, 2.5];
        let d2 = [-1.0, 3.0, 0.0];
        let all: Vec<f64> = d1.iter().chain(d2.iter()).copied().collect();

        let batch = model.posterior(&all);
        let mut online = model.posterior_natural(&d1);
        for x in d2 {
            online = online.combine(&model.statistic(&x));
        }
        let online = model.posterior_from_natural(online);
        close(online.params().0, batch.params().0, 1e-12);
        close(online.params().1, batch.params().1, 1e-12);

        let shuffled = [3.0, 0.5, 0.0, 2.5, -1.0];
        let permuted = model.posterior(&shuffled);
        close(permuted.params().0, batch.params().0, 1e-12);
        close(permuted.params().1, batch.params().1, 1e-12);
    }

    #[test]
    fn normal_known_variance_sequential_predictive_equals_marginal_likelihood() {
        let model = NormalKnownVariance::new(Normal::new(0.0, 1.0).unwrap(), 1.0);
        let data = [2.0, 4.0];
        close(
            model.ln_sequential_predictive(&data),
            model.ln_marginal_likelihood(&data),
            1e-9,
        );
        // Hand value: −ln(2π) − ½ ln 3 − ½·8 = −1.837877 − 0.549306 − 4.
        let expect = -(2.0 * std::f64::consts::PI).ln() - 0.5 * 3.0_f64.ln() - 4.0;
        close(model.ln_marginal_likelihood(&data), expect, 1e-12);
        assert!(model.ln_marginal_likelihood(&[f64::NAN]).is_nan());
        close(
            model.ln_posterior_predictive(&data, 1.0),
            predictive(model.posterior(&data), 1.0).ln_pdf(1.0),
            1e-12,
        );
    }
}
