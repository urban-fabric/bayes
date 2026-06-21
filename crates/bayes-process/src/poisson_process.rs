//! The **Poisson (point) process** — random points in time. Homogeneous arrivals
//! come from exponential inter-arrival times; inhomogeneous arrivals (a
//! time-varying intensity) come from Lewis–Shedler thinning.

use rand::distr::Distribution;
use rand::{Rng, RngExt};

use bayes_distribution_ext::Exponential;
use bayes_error::BayesError;

/// A homogeneous Poisson process with constant `rate λ > 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoissonProcess {
    rate: f64,
}

impl PoissonProcess {
    pub fn new(rate: f64) -> Result<Self, BayesError> {
        if !rate.is_finite() {
            return Err(BayesError::NonFinite {
                name: "rate",
                value: rate,
            });
        }
        if rate <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "rate",
                value: rate,
            });
        }
        Ok(Self { rate })
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Arrival times in `[0, horizon]`, in increasing order. Inter-arrival times
    /// are `Exponential(rate)`; the count is `Poisson(rate · horizon)` in
    /// distribution.
    pub fn arrivals<R: Rng + ?Sized>(&self, horizon: f64, rng: &mut R) -> Vec<f64> {
        let mut out = Vec::new();
        let inter = match Exponential::new(self.rate) {
            Ok(e) => e,
            Err(_) => return out,
        };
        let mut t = 0.0;
        loop {
            t += inter.sample(rng);
            if t > horizon {
                break;
            }
            out.push(t);
        }
        out
    }
}

/// Arrival times of an **inhomogeneous** Poisson process with intensity `λ(t)` on
/// `[0, horizon]`, via thinning: propose at the dominating constant rate
/// `max_rate ≥ supₜ λ(t)`, then keep each proposal `t` with probability
/// `λ(t) / max_rate`.
pub fn inhomogeneous_arrivals<R, F>(
    horizon: f64,
    max_rate: f64,
    intensity: F,
    rng: &mut R,
) -> Vec<f64>
where
    R: Rng + ?Sized,
    F: Fn(f64) -> f64,
{
    let mut out = Vec::new();
    let proposal = match Exponential::new(max_rate) {
        Ok(e) => e,
        Err(_) => return out,
    };
    let mut t = 0.0;
    loop {
        t += proposal.sample(rng);
        if t > horizon {
            break;
        }
        if rng.random::<f64>() <= intensity(t) / max_rate {
            out.push(t);
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn arrivals_are_increasing_and_within_horizon() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let pp = PoissonProcess::new(2.0).unwrap();
        let a = pp.arrivals(10.0, &mut rng);
        assert!(a.windows(2).all(|w| w[0] <= w[1]));
        assert!(a.iter().all(|&t| t <= 10.0));
    }

    #[test]
    fn expected_count_matches_rate_times_horizon() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let (rate, horizon) = (3.0, 20.0);
        let pp = PoissonProcess::new(rate).unwrap();
        let trials = 2_000;
        let total: usize = (0..trials)
            .map(|_| pp.arrivals(horizon, &mut rng).len())
            .sum();
        let avg = total as f64 / trials as f64;
        assert!(
            (avg - rate * horizon).abs() < 2.0,
            "E[N] ≈ {}, got {avg}",
            rate * horizon
        );
    }

    #[test]
    fn thinning_drops_to_zero_intensity() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let a = inhomogeneous_arrivals(10.0, 5.0, |t| if t < 5.0 { 5.0 } else { 0.0 }, &mut rng);
        assert!(a.iter().all(|&t| t < 5.0));
    }
}
