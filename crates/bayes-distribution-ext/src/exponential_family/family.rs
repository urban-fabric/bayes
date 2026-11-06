//! [`ExponentialFamily`] impls for the supported distributions — each a closed-form
//! MLE or method-of-moments fit from the relevant sufficient statistic.

use super::ExponentialFamily;
use super::statistic::{CountSum, MeanVar, Trials};
use crate::{Bernoulli, Beta, Exponential, Gamma, Normal, Poisson};

impl ExponentialFamily for Normal {
    type Stat = MeanVar;
    fn from_sufficient(stat: &MeanVar) -> Option<Self> {
        if stat.count() < 2.0 {
            return None;
        }
        Normal::new(stat.mean(), stat.population_variance().sqrt()).ok()
    }
}

impl ExponentialFamily for Exponential {
    type Stat = MeanVar;
    fn from_sufficient(stat: &MeanVar) -> Option<Self> {
        if stat.count() < 1.0 || stat.mean() <= 0.0 {
            return None;
        }
        Exponential::new(stat.mean().recip()).ok()
    }
}

impl ExponentialFamily for Gamma {
    type Stat = MeanVar;
    fn from_sufficient(stat: &MeanVar) -> Option<Self> {
        let (m, v) = (stat.mean(), stat.population_variance());
        if stat.count() < 2.0 || m <= 0.0 || v <= 0.0 {
            return None;
        }
        Gamma::new(m * m / v, m / v).ok()
    }
}

impl ExponentialFamily for Beta {
    type Stat = MeanVar;
    fn from_sufficient(stat: &MeanVar) -> Option<Self> {
        let (m, v) = (stat.mean(), stat.population_variance());
        if stat.count() < 2.0 || !(0.0..=1.0).contains(&m) || v <= 0.0 {
            return None;
        }
        let c = m * (1.0 - m) / v - 1.0;
        if c <= 0.0 {
            return None;
        }
        Beta::new(m * c, (1.0 - m) * c).ok()
    }
}

impl ExponentialFamily for Poisson {
    type Stat = CountSum;
    fn from_sufficient(stat: &CountSum) -> Option<Self> {
        if stat.count() < 1.0 {
            return None;
        }
        Poisson::new(stat.sum() / stat.count()).ok()
    }
}

impl ExponentialFamily for Bernoulli {
    type Stat = Trials;
    fn from_sufficient(stat: &Trials) -> Option<Self> {
        if stat.count() < 1.0 {
            return None;
        }
        Bernoulli::new(stat.successes() / stat.count()).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::Moments;
    use crate::exponential_family::SufficientStatistic;
    use bayes_core_ext::assert_close;

    #[test]
    fn normal_fit_recovers_parameters() {
        let data = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let n = Normal::fit(&data).unwrap();
        assert_close!(n.mean(), 0.0, 1e-12);
        assert_close!(n.variance(), 2.0, 1e-12);
    }

    #[test]
    fn poisson_and_bernoulli_fit() {
        let counts = [2u64, 3, 5, 1, 4];
        assert_close!(Poisson::fit(&counts).unwrap().lambda(), 3.0, 1e-12);
        let flips = [true, true, false, true, false, false, true, true];
        assert_close!(Bernoulli::fit(&flips).unwrap().p(), 5.0 / 8.0, 1e-12);
    }

    #[test]
    fn fit_via_merged_statistics_equals_direct() {
        let data: Vec<f64> = (0..100).map(|i| 0.5 + 0.3 * (i as f64 / 100.0)).collect();
        let direct = Beta::fit(&data);
        let merged = Beta::from_sufficient(
            &MeanVar::from_sample(&data[..40]).merge(&MeanVar::from_sample(&data[40..])),
        );
        match (direct, merged) {
            (Some(a), Some(b)) => {
                assert_close!(a.alpha(), b.alpha(), 1e-9);
                assert_close!(a.beta(), b.beta(), 1e-9);
            }
            _ => panic!("expected both fits to succeed"),
        }
    }
}
