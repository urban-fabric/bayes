use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, Discrete, Moments};
use crate::error::BayesError;
use crate::function::stable::ln_sum_exp;

/// A finite **mixture distribution**: a weighted blend of `components`, all of the
/// same type `D`. Sampling picks a component by weight then samples it; the density
/// is the weighted sum of the components' densities.
#[derive(Clone, Debug, PartialEq)]
pub struct Mixture<D> {
    weights: Vec<f64>,
    cumulative: Vec<f64>,
    components: Vec<D>,
}

impl<D> Mixture<D> {
    /// Build from non-negative `weights` (normalized internally) and the same
    /// number of `components`.
    pub fn new(weights: Vec<f64>, components: Vec<D>) -> Result<Self, BayesError> {
        if components.is_empty() || weights.len() != components.len() {
            return Err(BayesError::Malformed {
                what: "mixture needs matching, non-empty weights and components",
            });
        }
        let mut sum = 0.0;
        for &w in &weights {
            if !w.is_finite() || w < 0.0 {
                return Err(BayesError::Malformed {
                    what: "mixture weights must be finite and non-negative",
                });
            }
            sum += w;
        }
        if sum <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "weight sum",
                value: sum,
            });
        }
        let weights: Vec<f64> = weights.iter().map(|&w| w / sum).collect();
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for &w in &weights {
            acc += w;
            cumulative.push(acc);
        }
        Ok(Self {
            weights,
            cumulative,
            components,
        })
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    pub fn components(&self) -> &[D] {
        &self.components
    }
}

impl<D: Moments> Mixture<D> {
    /// The **law-of-total-variance decomposition** `(within, between)` of the
    /// mixture variance: the *within-component* part `E[Var(X|Z)] = Σ wᵢ Var(Xᵢ)`
    /// (the weight-averaged component variance) and the *between-component* part
    /// `Var(E[X|Z]) = Σ wᵢ (μᵢ − μ)²` (the spread of component means about the
    /// mixture mean `μ`). Their sum is [`variance`](Moments::variance).
    ///
    /// The identity `variance = within + between` and its nonnegativity are proved
    /// in `riemann` (`RiemannLean/Bayes/Mixture.lean`: `law_of_total_variance`,
    /// `mixtureVar_nonneg`). `None` if any component lacks a mean or variance.
    pub fn variance_decomposition(&self) -> Option<(f64, f64)> {
        let mut mean = 0.0;
        for (&w, c) in self.weights.iter().zip(&self.components) {
            mean += w * c.mean()?;
        }
        let mut within = 0.0;
        let mut between = 0.0;
        for (&w, c) in self.weights.iter().zip(&self.components) {
            within += w * c.variance()?;
            between += w * (c.mean()? - mean).powi(2);
        }
        Some((within, between))
    }
}

impl<D, T> Distribution<T> for Mixture<D>
where
    D: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        let u: f64 = rng.random();
        let i = self
            .cumulative
            .iter()
            .position(|&c| u <= c)
            .unwrap_or(self.components.len() - 1);
        self.components[i].sample(rng)
    }
}

impl<D> Continuous for Mixture<D>
where
    D: Continuous,
{
    fn pdf(&self, x: f64) -> f64 {
        self.weights
            .iter()
            .zip(&self.components)
            .map(|(&w, c)| w * c.pdf(x))
            .sum()
    }
    /// `ln Σ wᵢ fᵢ(x)` via log-sum-exp over `ln wᵢ + ln fᵢ(x)` — stable when the
    /// component log-densities are very negative (the naive `pdf().ln()` would
    /// underflow each term to `0` before summing).
    fn ln_pdf(&self, x: f64) -> f64 {
        let terms: Vec<f64> = self
            .weights
            .iter()
            .zip(&self.components)
            .map(|(&w, c)| w.ln() + c.ln_pdf(x))
            .collect();
        ln_sum_exp(&terms)
    }
}

impl<D> Discrete<u64> for Mixture<D>
where
    D: Discrete<u64>,
{
    fn pmf(&self, k: u64) -> f64 {
        self.weights
            .iter()
            .zip(&self.components)
            .map(|(&w, c)| w * c.pmf(k))
            .sum()
    }
    /// `ln Σ wᵢ Pᵢ(k)` via log-sum-exp (see [`Continuous::ln_pdf`] for the mixture).
    fn ln_pmf(&self, k: u64) -> f64 {
        let terms: Vec<f64> = self
            .weights
            .iter()
            .zip(&self.components)
            .map(|(&w, c)| w.ln() + c.ln_pmf(k))
            .collect();
        ln_sum_exp(&terms)
    }
}

impl<D> Moments for Mixture<D>
where
    D: Moments,
{
    /// Mixture mean `Σ wᵢ μᵢ` (if every component has a mean).
    fn mean(&self) -> Option<f64> {
        let mut m = 0.0;
        for (&w, c) in self.weights.iter().zip(&self.components) {
            m += w * c.mean()?;
        }
        Some(m)
    }
    /// Mixture variance by the **law of total variance**, in the one-pass form
    /// `Σ wᵢ(σᵢ² + μᵢ²) − (Σ wᵢ μᵢ)²` — equal to the within + between split
    /// ([`variance_decomposition`](Mixture::variance_decomposition)) and always
    /// nonnegative, proved in `riemann` (`RiemannLean/Bayes/Mixture.lean`:
    /// `law_of_total_variance`, `mixtureVar_nonneg`).
    fn variance(&self) -> Option<f64> {
        let mut e_x2 = 0.0;
        for (&w, c) in self.weights.iter().zip(&self.components) {
            let mi = c.mean()?;
            let vi = c.variance()?;
            e_x2 += w * (vi + mi * mi);
        }
        let m = self.mean()?;
        Some(e_x2 - m * m)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::Normal;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn density_is_the_weighted_sum() {
        let m = Mixture::new(
            vec![0.3, 0.7],
            vec![
                Normal::new(-2.0, 1.0).unwrap(),
                Normal::new(3.0, 2.0).unwrap(),
            ],
        )
        .unwrap();
        let x = 0.5;
        let expect = 0.3 * Normal::new(-2.0, 1.0).unwrap().pdf(x)
            + 0.7 * Normal::new(3.0, 2.0).unwrap().pdf(x);
        close(m.pdf(x), expect, 1e-12);
    }

    #[test]
    fn moments_follow_the_mixture_laws() {
        let m = Mixture::new(
            vec![0.5, 0.5],
            vec![
                Normal::new(0.0, 1.0).unwrap(),
                Normal::new(4.0, 1.0).unwrap(),
            ],
        )
        .unwrap();
        close(m.mean().unwrap(), 2.0, 1e-12);
        close(m.variance().unwrap(), 5.0, 1e-12);
        // Law of total variance: within = ½·1 + ½·1 = 1, between = ½·2² + ½·2² = 4.
        let (within, between) = m.variance_decomposition().unwrap();
        close(within, 1.0, 1e-12);
        close(between, 4.0, 1e-12);
        // within + between == variance (law_of_total_variance).
        close(within + between, m.variance().unwrap(), 1e-12);
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let m = Mixture::new(
            vec![0.5, 0.5],
            vec![
                Normal::new(0.0, 1.0).unwrap(),
                Normal::new(4.0, 1.0).unwrap(),
            ],
        )
        .unwrap();
        let n = 60_000;
        let s: f64 = (0..n).map(|_| m.sample(&mut rng)).sum();
        close(s / n as f64, 2.0, 0.05);
    }
}
