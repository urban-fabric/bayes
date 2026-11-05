use crate::{Discrete, DiscreteCdf, Moments, Support};
use bayes_core::BayesError;
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

/// A categorical distribution over integer categories `{0, ..., k - 1}`.
///
/// The constructor accepts non-negative, not-all-zero weights and stores the
/// normalized probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct Categorical<R: Real = f64> {
    p: Vec<R>,
    cumulative: Vec<R>,
}

impl Categorical<f64> {
    pub fn new(weights: &[f64]) -> Result<Self, BayesError> {
        if weights.is_empty() {
            return Err(BayesError::Malformed {
                what: "categorical needs at least one category",
            });
        }
        let mut sum = 0.0;
        for &w in weights {
            if !w.is_finite() || w < 0.0 {
                return Err(BayesError::Malformed {
                    what: "weights must be finite and non-negative",
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

        let p: Vec<f64> = weights.iter().map(|&w| w / sum).collect();
        let mut cumulative = Vec::with_capacity(p.len());
        let mut acc = 0.0;
        for &pi in &p {
            acc += pi;
            cumulative.push(acc);
        }
        Ok(Self { p, cumulative })
    }
}

impl<R: Real> Categorical<R> {
    /// Build from already-normalized probabilities without host-side validation.
    ///
    /// This is the graph-friendly constructor for lazy [`Real`] backends: callers
    /// are responsible for ensuring nonnegative entries that sum to one.
    pub fn from_probabilities_unchecked(probabilities: Vec<R>) -> Self {
        let mut cumulative = Vec::with_capacity(probabilities.len());
        let mut acc = R::zero();
        for pi in &probabilities {
            acc = acc + pi.clone();
            cumulative.push(acc.clone());
        }
        Self {
            p: probabilities,
            cumulative,
        }
    }

    /// Normalize weights over the supplied [`Real`] carrier without host-side
    /// finite/nonnegative checks.
    pub fn from_weights_unchecked(weights: &[R]) -> Self {
        let sum = weights.iter().cloned().fold(R::zero(), |acc, w| acc + w);
        let probabilities = weights.iter().cloned().map(|w| w / sum.clone()).collect();
        Self::from_probabilities_unchecked(probabilities)
    }

    pub fn categories(&self) -> usize {
        self.p.len()
    }

    pub fn probabilities(&self) -> &[R] {
        &self.p
    }

    pub fn probability_at(&self, k: usize) -> R {
        self.p.get(k).cloned().unwrap_or_else(R::zero)
    }

    pub fn ln_probability_at(&self, k: usize) -> R {
        self.p
            .get(k)
            .cloned()
            .map(kernels::categorical_log_pmf)
            .unwrap_or_else(R::neg_infinity)
    }
}

impl Discrete<u64> for Categorical<f64> {
    fn pmf(&self, k: u64) -> f64 {
        self.probability_at(k as usize)
    }

    fn ln_pmf(&self, k: u64) -> f64 {
        self.ln_probability_at(k as usize)
    }
}

impl<R: Real> Support<u64> for Categorical<R> {
    fn min(&self) -> u64 {
        0
    }

    fn max(&self) -> u64 {
        (self.p.len() - 1) as u64
    }
}

impl DiscreteCdf<u64> for Categorical<f64> {
    fn cdf(&self, k: u64) -> f64 {
        let i = k as usize;
        if i >= self.cumulative.len() {
            1.0
        } else {
            self.cumulative[i]
        }
    }

    fn inverse_cdf(&self, p: f64) -> u64 {
        match self.cumulative.iter().position(|&c| c >= p) {
            Some(i) => i as u64,
            None => (self.p.len() - 1) as u64,
        }
    }
}

impl<R> Moments<R> for Categorical<R>
where
    R: Real + Compare,
{
    fn mean(&self) -> R {
        self.p
            .iter()
            .enumerate()
            .map(|(i, pi)| R::scalar(i as f64) * pi.clone())
            .fold(R::zero(), |acc, x| acc + x)
    }

    fn variance(&self) -> R {
        let m = self.mean();
        self.p
            .iter()
            .enumerate()
            .map(|(i, pi)| {
                let centered = R::scalar(i as f64) - m.clone();
                centered.clone() * centered * pi.clone()
            })
            .fold(R::zero(), |acc, x| acc + x)
    }

    fn std_dev(&self) -> R {
        self.variance().sqrt()
    }

    fn skewness(&self) -> R {
        R::nan()
    }

    fn kurtosis(&self) -> R {
        R::nan()
    }

    fn entropy(&self) -> R {
        -self
            .p
            .iter()
            .cloned()
            .map(|pi| pi.clone().xlogy(pi))
            .fold(R::zero(), |acc, x| acc + x)
    }

    fn median(&self) -> R {
        R::nan()
    }

    fn mode(&self) -> R {
        let Some(first) = self.p.first().cloned() else {
            return R::nan();
        };
        let mut best_probability = first;
        let mut best_index = R::zero();
        for (i, probability) in self.p.iter().cloned().enumerate().skip(1) {
            let replace = probability.gt(&best_probability);
            best_probability = R::select(&replace, probability, best_probability);
            best_index = R::select(&replace, R::scalar(i as f64), best_index);
        }
        best_index
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core::assert_close;

    #[test]
    fn normalizes_weights_and_pmf_cdf() {
        let c = Categorical::new(&[1.0, 3.0]).unwrap();
        assert_close!(c.pmf(0), 0.25, 1e-12);
        assert_close!(c.pmf(1), 0.75, 1e-12);
        assert_close!(c.cdf(0), 0.25, 1e-12);
        assert_close!(c.cdf(1), 1.0, 1e-12);
        assert_eq!(c.mode(), 1.0);
        assert_close!(c.mean(), 0.75, 1e-12);
    }

    #[test]
    fn rejects_malformed_weights() {
        assert!(Categorical::new(&[]).is_err());
        assert!(Categorical::new(&[0.0, 0.0]).is_err());
        assert!(Categorical::new(&[1.0, -1.0]).is_err());
        assert!(Categorical::new(&[1.0, f64::NAN]).is_err());
    }
}
