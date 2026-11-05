//! The sufficient-statistic summary types: [`MeanVar`] (real observations),
//! [`CountSum`] (non-negative integers), and [`Trials`] (booleans). Each is a small
//! `Copy` monoid under [`merge`](SufficientStatistic::merge).

use super::SufficientStatistic;

/// Welford mean / variance summary over real observations — the sufficient
/// statistic for the Normal, and the method-of-moments input for Gamma and Beta.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeanVar {
    n: f64,
    mean: f64,
    m2: f64,
}

impl MeanVar {
    pub fn count(&self) -> f64 {
        self.n
    }
    pub fn mean(&self) -> f64 {
        self.mean
    }
    /// MLE (biased, `÷n`) variance.
    pub fn population_variance(&self) -> f64 {
        if self.n > 0.0 {
            self.m2 / self.n
        } else {
            f64::NAN
        }
    }
    /// Unbiased (`÷(n−1)`) variance.
    pub fn sample_variance(&self) -> f64 {
        if self.n > 1.0 {
            self.m2 / (self.n - 1.0)
        } else {
            f64::NAN
        }
    }
}

impl SufficientStatistic for MeanVar {
    type Datum = f64;
    fn empty() -> Self {
        Self::default()
    }
    fn observe(x: f64) -> Self {
        Self {
            n: 1.0,
            mean: x,
            m2: 0.0,
        }
    }
    fn merge(&self, other: &Self) -> Self {
        if other.n == 0.0 {
            return *self;
        }
        if self.n == 0.0 {
            return *other;
        }
        let n = self.n + other.n;
        let delta = other.mean - self.mean;
        let mean = self.mean + delta * other.n / n;
        let m2 = self.m2 + other.m2 + delta * delta * self.n * other.n / n;
        Self { n, mean, m2 }
    }
}

/// Count + sum over non-negative integer observations — the sufficient statistic
/// for the Poisson.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CountSum {
    n: f64,
    sum: f64,
}

impl CountSum {
    pub fn count(&self) -> f64 {
        self.n
    }
    pub fn sum(&self) -> f64 {
        self.sum
    }
}

impl SufficientStatistic for CountSum {
    type Datum = u64;
    fn empty() -> Self {
        Self::default()
    }
    fn observe(k: u64) -> Self {
        Self {
            n: 1.0,
            sum: k as f64,
        }
    }
    fn merge(&self, other: &Self) -> Self {
        Self {
            n: self.n + other.n,
            sum: self.sum + other.sum,
        }
    }
}

/// Trials + successes over boolean observations — the sufficient statistic for the
/// Bernoulli (and the Beta–Bernoulli conjugate update).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Trials {
    n: f64,
    successes: f64,
}

impl Trials {
    pub fn count(&self) -> f64 {
        self.n
    }
    pub fn successes(&self) -> f64 {
        self.successes
    }
    pub fn failures(&self) -> f64 {
        self.n - self.successes
    }
}

impl SufficientStatistic for Trials {
    type Datum = bool;
    fn empty() -> Self {
        Self::default()
    }
    fn observe(b: bool) -> Self {
        Self {
            n: 1.0,
            successes: if b { 1.0 } else { 0.0 },
        }
    }
    fn merge(&self, other: &Self) -> Self {
        Self {
            n: self.n + other.n,
            successes: self.successes + other.successes,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;

    #[test]
    fn meanvar_merge_is_associative_and_matches_one_pass() {
        let data: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 0.7).sin() * 4.0 + 1.0)
            .collect();
        let whole = MeanVar::from_sample(&data);
        let a = MeanVar::from_sample(&data[..63]);
        let b = MeanVar::from_sample(&data[63..150]);
        let c = MeanVar::from_sample(&data[150..]);
        let left = a.merge(&b).merge(&c);
        let right = a.merge(&b.merge(&c));
        assert_close!(left.mean(), whole.mean(), 1e-12);
        assert_close!(
            left.population_variance(),
            whole.population_variance(),
            1e-10,
        );
        assert_close!(right.mean(), whole.mean(), 1e-12);
        assert_close!(
            right.population_variance(),
            whole.population_variance(),
            1e-10,
        );
        assert_eq!(MeanVar::empty().merge(&whole), whole);
    }
}
