use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Discrete, DiscreteCdf, Moments, Support};
use bayes_error::BayesError;

/// The hypergeometric distribution: the number of successes in `draws` samples
/// taken **without replacement** from a population of `population` items
/// containing `successes` successes. Support `[max(0, draws+successes-population),
/// min(draws, successes)]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hypergeometric {
    population: u64,
    successes: u64,
    draws: u64,
}

impl Hypergeometric {
    pub fn new(population: u64, successes: u64, draws: u64) -> Result<Self, BayesError> {
        if successes > population {
            return Err(BayesError::Malformed {
                what: "successes must not exceed population",
            });
        }
        if draws > population {
            return Err(BayesError::Malformed {
                what: "draws must not exceed population",
            });
        }
        Ok(Self {
            population,
            successes,
            draws,
        })
    }

    pub fn population(&self) -> u64 {
        self.population
    }
    pub fn successes(&self) -> u64 {
        self.successes
    }
    pub fn draws(&self) -> u64 {
        self.draws
    }

    fn k_min(&self) -> u64 {
        (self.draws + self.successes).saturating_sub(self.population)
    }
    fn k_max(&self) -> u64 {
        self.draws.min(self.successes)
    }
}

impl Distribution<u64> for Hypergeometric {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        let mut remaining = self.population;
        let mut good = self.successes;
        let mut hits = 0;
        for _ in 0..self.draws {
            if remaining == 0 {
                break;
            }
            if rng.random::<f64>() < good as f64 / remaining as f64 {
                hits += 1;
                good -= 1;
            }
            remaining -= 1;
        }
        hits
    }
}

impl Support<u64> for Hypergeometric {
    fn min(&self) -> u64 {
        self.k_min()
    }
    fn max(&self) -> u64 {
        self.k_max()
    }
}

impl DiscreteCdf<u64> for Hypergeometric {
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.k_max() {
            return 1.0;
        }
        (self.k_min()..=k).map(|i| self.pmf(i)).sum()
    }
    fn inverse_cdf(&self, p: f64) -> u64 {
        let mut k = self.k_min();
        while k < self.k_max() && self.cdf(k) < p {
            k += 1;
        }
        k
    }
}

impl Moments for Hypergeometric {
    fn mean(&self) -> f64 {
        self.draws as f64 * self.successes as f64 / self.population as f64
    }
    fn variance(&self) -> f64 {
        let (n, k, d) = (
            self.population as f64,
            self.successes as f64,
            self.draws as f64,
        );
        if n <= 1.0 {
            return 0.0;
        }
        d * (k / n) * ((n - k) / n) * ((n - d) / (n - 1.0))
    }
    fn moment2(&self) -> f64 {
        let mean = self.mean();
        self.variance() + mean * mean
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn skewness(&self) -> f64 {
        f64::NAN
    }

    fn kurtosis(&self) -> f64 {
        f64::NAN
    }

    fn entropy(&self) -> f64 {
        f64::NAN
    }

    fn median(&self) -> f64 {
        f64::NAN
    }

    fn mode(&self) -> f64 {
        f64::NAN
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn pmf_sums_to_one_and_known_value() {
        let h = Hypergeometric::new(50, 5, 10).unwrap();
        let total: f64 = (h.min()..=h.max()).map(|k| h.pmf(k)).sum();
        close(total, 1.0, 1e-10);
        close(h.mean(), 1.0, 1e-12);
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let h = Hypergeometric::new(100, 30, 20).unwrap();
        let n = 40_000;
        let s: u64 = (0..n).map(|_| h.sample(&mut rng)).sum();
        close(s as f64 / n as f64, h.mean(), 0.05);
    }
}
