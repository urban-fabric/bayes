//! **Fisher information, Cramér–Rao bounds, and Jeffreys priors.**
//!
//! For a one-parameter family the Fisher information
//! `I(θ) = E[(∂_θ ln f)²] = Var(score)` measures how sharply the log-likelihood
//! pins down `θ`. riemann's `Statistics/CramerRao.lean` proves the two facts this
//! module rests on: `fisher_eq_var_score` (the information *is* the score
//! variance, since the score is mean-zero) and `cramer_rao`
//! (`Var(θ̂)·I(θ) ≥ 1` for any unbiased estimator). So
//! [`cramer_rao_bound`](FisherInformation::cramer_rao_bound) is a *proved*
//! variance floor, not a heuristic.
//!
//! Each implementor documents **which** scalar parameter its information is about
//! — the family's natural inference target. Multi-parameter families whose
//! information is a matrix (Gamma/Beta jointly, multivariate Normal) are out of
//! scope: this layer is deliberately scalar.
//!
//! The Fisher information also yields the **Jeffreys prior** `π(θ) ∝ √I(θ)`, the
//! reparameterization-invariant objective prior. It is proper only for some
//! families; where it is — Bernoulli and Binomial both give `Beta(½, ½)` — this
//! module hands back the concrete distribution.

use crate::{Bernoulli, Beta, Binomial, Exponential, Geometric, Normal, Poisson, Rayleigh};

/// A one-parameter family with a closed-form Fisher information about its
/// canonical scalar parameter.
pub trait FisherInformation {
    /// Fisher information `I(θ)` **per observation**, evaluated at the current
    /// parameter. See each implementation for the parameter `θ` it refers to.
    fn fisher_information(&self) -> f64;

    /// Cramér–Rao lower bound on `Var(θ̂)` for any unbiased estimator of the
    /// canonical parameter from `n` iid observations: `1 / (n · I(θ))`.
    ///
    /// Proved a true lower bound in riemann `Statistics/CramerRao.lean`
    /// (`cramer_rao`); attained by efficient estimators (`cramer_rao_attained`).
    fn cramer_rao_bound(&self, n: usize) -> f64 {
        1.0 / (n as f64 * self.fisher_information())
    }
}

/// Fisher information about the success probability `p`: `1 / (p(1−p))`.
impl FisherInformation for Bernoulli {
    fn fisher_information(&self) -> f64 {
        let p = self.p();
        1.0 / (p * (1.0 - p))
    }
}

/// Fisher information about the success probability `p`, for a known trial
/// count `n`: `n / (p(1−p))`.
impl FisherInformation for Binomial {
    fn fisher_information(&self) -> f64 {
        let p = self.p();
        self.trials() as f64 / (p * (1.0 - p))
    }
}

/// Fisher information about the rate `λ`: `1 / λ`.
impl FisherInformation for Poisson {
    fn fisher_information(&self) -> f64 {
        1.0 / self.lambda()
    }
}

/// Fisher information about the success probability `p` (trial-count
/// parameterization): `1 / (p²(1−p))`.
impl FisherInformation for Geometric {
    fn fisher_information(&self) -> f64 {
        let p = self.p();
        1.0 / (p * p * (1.0 - p))
    }
}

/// Fisher information about the rate `λ`: `1 / λ²`.
impl FisherInformation for Exponential {
    fn fisher_information(&self) -> f64 {
        let rate = self.rate();
        1.0 / (rate * rate)
    }
}

/// Fisher information about the mean `μ` (observation variance `σ²` known):
/// `1 / σ²` — the precision the [`NormalKnownVariance`](crate::NormalKnownVariance)
/// conjugate accumulates.
impl FisherInformation for Normal {
    fn fisher_information(&self) -> f64 {
        let sigma = self.params().1;
        1.0 / (sigma * sigma)
    }
}

/// Fisher information about the scale `σ`: `4 / σ²`.
impl FisherInformation for Rayleigh {
    fn fisher_information(&self) -> f64 {
        let scale = self.scale();
        4.0 / (scale * scale)
    }
}

impl Bernoulli {
    /// The **Jeffreys prior** for the Bernoulli success probability: the proper
    /// `Beta(½, ½)`, since `π(p) ∝ √I(p) ∝ p^{−1/2}(1−p)^{−1/2}`. Pair it with
    /// [`BetaBernoulli`](crate::BetaBernoulli) for an objective-prior update.
    pub fn jeffreys_prior() -> Beta {
        Beta::new(0.5, 0.5).unwrap_or_else(|_| unreachable!())
    }
}

impl Binomial {
    /// The **Jeffreys prior** for the binomial success probability: the proper
    /// `Beta(½, ½)`, independent of the trial count.
    pub fn jeffreys_prior() -> Beta {
        Beta::new(0.5, 0.5).unwrap_or_else(|_| unreachable!())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::Moments;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn fisher_information_closed_forms() {
        close(
            Bernoulli::new(0.25).unwrap().fisher_information(),
            1.0 / (0.25 * 0.75),
            1e-12,
        );
        close(
            Binomial::new(10, 0.25).unwrap().fisher_information(),
            10.0 / (0.25 * 0.75),
            1e-12,
        );
        close(Poisson::new(4.0).unwrap().fisher_information(), 0.25, 1e-12);
        close(
            Geometric::new(0.5).unwrap().fisher_information(),
            1.0 / (0.25 * 0.5),
            1e-12,
        );
        close(
            Exponential::new(2.0).unwrap().fisher_information(),
            0.25,
            1e-12,
        );
        close(
            Normal::new(1.0, 2.0).unwrap().fisher_information(),
            0.25,
            1e-12,
        );
        close(
            Rayleigh::new(3.0).unwrap().fisher_information(),
            4.0 / 9.0,
            1e-12,
        );
    }

    #[test]
    fn cramer_rao_bound_matches_efficient_estimator_variance() {
        // Bernoulli MLE p̂ = mean is efficient: its variance p(1−p)/n equals the
        // bound 1/(n·I) exactly.
        let bern = Bernoulli::new(0.3).unwrap();
        let n = 100;
        let crlb = bern.cramer_rao_bound(n);
        close(crlb, 0.3 * 0.7 / n as f64, 1e-12);
        // The defining identity: bound · information · n = 1.
        close(crlb * bern.fisher_information() * n as f64, 1.0, 1e-12);

        // Poisson λ̂ = sample mean is efficient: variance λ/n = 1/(n·I).
        let pois = Poisson::new(4.0).unwrap();
        close(pois.cramer_rao_bound(50), 4.0 / 50.0, 1e-12);
    }

    #[test]
    fn jeffreys_priors_are_beta_half_half() {
        let jb = Bernoulli::jeffreys_prior();
        close(jb.alpha(), 0.5, 1e-12);
        close(jb.beta(), 0.5, 1e-12);
        // Symmetric about ½ with mean ½.
        close(jb.mean(), 0.5, 1e-12);
        let jbin = Binomial::jeffreys_prior();
        close(jbin.alpha(), 0.5, 1e-12);
        close(jbin.beta(), 0.5, 1e-12);
    }
}
