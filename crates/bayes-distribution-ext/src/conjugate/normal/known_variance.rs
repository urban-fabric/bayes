//! The Normal-Normal conjugate: unknown mean, known observation variance.
//!
//! The simplest conjugate in the library and the clearest illustration of what
//! conjugacy *is*: the posterior precision is the sum of the prior precision and
//! the data precision, and the posterior mean is the precision-weighted average
//! of the two. Everything else is bookkeeping.
use super::super::{Conjugate, Pair};
use crate::{Continuous, Normal};
use euclid_numeric::Real;

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
    use super::super::inverse_gamma::{NormalInverseGamma, NormalInverseGammaModel};
    use super::*;
    use crate::NaturalParams;
    use bayes_core_ext::assert_close;

    #[test]
    fn nig_posterior_matches_textbook_update() {
        // NIG(0, 1, 1, 1) prior, data {2, 4}: n=2, Σx=6, Σx²=20.
        let model =
            NormalInverseGammaModel::new(NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0).unwrap());
        let data = [2.0, 4.0];
        let post = model.posterior(&data);
        // λ=λ₀+n=3, μ=(λ₀μ₀+Σx)/λ=2, α=α₀+n/2=2,
        // β=β₀+½[Σ(x-x̄)² + λ₀n(x̄-μ₀)²/λ]=1+½[2+6]=5.
        assert_close!(post.lambda(), 3.0, 1e-12);
        assert_close!(post.mu(), 2.0, 1e-12);
        assert_close!(post.alpha(), 2.0, 1e-12);
        assert_close!(post.beta(), 5.0, 1e-12);

        // Marginals are families already in the crate.
        let var_marg = post.variance_marginal();
        assert_close!(var_marg.shape(), 2.0, 1e-12);
        assert_close!(var_marg.scale(), 5.0, 1e-12);
        let mean_marg = post.mean_marginal();
        assert_close!(mean_marg.freedom(), 4.0, 1e-12);
        assert_close!(mean_marg.location(), 2.0, 1e-12);
        assert_close!(mean_marg.scale(), (5.0_f64 / (2.0 * 3.0)).sqrt(), 1e-12);

        // Predictive Student-t: ν=2α=4, loc=μ=2, scale²=β(λ+1)/(αλ)=20/6.
        let pred = model.posterior_predictive(&data);
        assert_close!(pred.freedom(), 4.0, 1e-12);
        assert_close!(pred.location(), 2.0, 1e-12);
        assert_close!(pred.scale(), (5.0_f64 * 4.0 / (2.0 * 3.0)).sqrt(), 1e-12);
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
        assert_close!(online.mu(), batch.mu(), 1e-12);
        assert_close!(online.lambda(), batch.lambda(), 1e-12);
        assert_close!(online.alpha(), batch.alpha(), 1e-12);
        assert_close!(online.beta(), batch.beta(), 1e-12);

        let shuffled = [4.0, 0.5, 3.0, 1.0, -2.0];
        let permuted = model.posterior(&shuffled);
        assert_close!(permuted.mu(), batch.mu(), 1e-12);
        assert_close!(permuted.beta(), batch.beta(), 1e-12);
    }

    #[test]
    fn nig_sequential_predictive_equals_marginal_likelihood() {
        let model =
            NormalInverseGammaModel::new(NormalInverseGamma::new(0.0, 1.0, 2.0, 3.0).unwrap());
        let data = [1.5, -0.5, 2.0, 0.25];
        assert_close!(
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
        assert_close!(model.ln_marginal_likelihood(&data), expect, 1e-12);
        assert!(model.ln_marginal_likelihood(&[f64::INFINITY]).is_nan());
    }

    #[test]
    fn nig_predictive_log_density_round_trips() {
        let model =
            NormalInverseGammaModel::new(NormalInverseGamma::new(0.0, 1.0, 2.0, 3.0).unwrap());
        let data = [1.0, 2.0];
        assert_close!(
            model.ln_predictive(0.7).exp(),
            model.posterior(&[]).predictive().pdf(0.7),
            1e-12,
        );
        assert_close!(
            model.ln_posterior_predictive(&data, 0.7),
            model.posterior(&data).predictive().ln_pdf(0.7),
            1e-12,
        );
        assert_close!(
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
        assert_close!(mean, 2.0, 1e-12);
        assert_close!(1.0 / (std * std), 3.0, 1e-12);

        // Predictive adds back the observation variance: Normal(2, √(1/3 + 1)).
        let pred = model.posterior_predictive(&data);
        let (pmean, pstd) = pred.params();
        assert_close!(pmean, 2.0, 1e-12);
        assert_close!(pstd, (1.0_f64 / 3.0 + 1.0).sqrt(), 1e-12);
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
        assert_close!(online.params().0, batch.params().0, 1e-12);
        assert_close!(online.params().1, batch.params().1, 1e-12);

        let shuffled = [3.0, 0.5, 0.0, 2.5, -1.0];
        let permuted = model.posterior(&shuffled);
        assert_close!(permuted.params().0, batch.params().0, 1e-12);
        assert_close!(permuted.params().1, batch.params().1, 1e-12);
    }

    #[test]
    fn normal_known_variance_sequential_predictive_equals_marginal_likelihood() {
        let model = NormalKnownVariance::new(Normal::new(0.0, 1.0).unwrap(), 1.0);
        let data = [2.0, 4.0];
        assert_close!(
            model.ln_sequential_predictive(&data),
            model.ln_marginal_likelihood(&data),
            1e-9,
        );
        // Hand value: −ln(2π) − ½ ln 3 − ½·8 = −1.837877 − 0.549306 − 4.
        let expect = -(2.0 * std::f64::consts::PI).ln() - 0.5 * 3.0_f64.ln() - 4.0;
        assert_close!(model.ln_marginal_likelihood(&data), expect, 1e-12);
        assert!(model.ln_marginal_likelihood(&[f64::NAN]).is_nan());
        assert_close!(
            model.ln_posterior_predictive(&data, 1.0),
            predictive(model.posterior(&data), 1.0).ln_pdf(1.0),
            1e-12,
        );
    }
}
