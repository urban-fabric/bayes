//! Discrete-state, discrete-time **Markov chains** as *generative* processes:
//! sample state trajectories from a row-stochastic transition matrix, and compute
//! the stationary distribution. (This is the chain as a random process — distinct
//! from MCMC *inference*, which is out of scope for `bayes`.)
#![allow(clippy::needless_range_loop)]

use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::error::BayesError;

/// A finite-state Markov chain with a row-stochastic transition matrix `P`, where
/// `P[i][j] = P(next = j | current = i)`.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkovChain {
    transition: Vec<Vec<f64>>,
}

impl MarkovChain {
    /// Build from a transition matrix, validating that it is square and each row is
    /// a probability distribution (non-negative, summing to 1).
    pub fn new(transition: Vec<Vec<f64>>) -> Result<Self, BayesError> {
        let n = transition.len();
        if n == 0 {
            return Err(BayesError::Malformed {
                what: "transition matrix is empty",
            });
        }
        for row in &transition {
            if row.len() != n {
                return Err(BayesError::Malformed {
                    what: "transition matrix is not square",
                });
            }
            let mut sum = 0.0;
            for &p in row {
                if !p.is_finite() || p < 0.0 {
                    return Err(BayesError::Malformed {
                        what: "transition probabilities must be finite and non-negative",
                    });
                }
                sum += p;
            }
            if (sum - 1.0).abs() > 1e-9 {
                return Err(BayesError::Malformed {
                    what: "each transition row must sum to 1",
                });
            }
        }
        Ok(Self { transition })
    }

    /// The number of states.
    pub fn states(&self) -> usize {
        self.transition.len()
    }

    /// The next state sampled from row `current` of the transition matrix.
    fn step<R: Rng + ?Sized>(&self, current: usize, rng: &mut R) -> usize {
        let row = &self.transition[current];
        let u: f64 = rng.random();
        let mut acc = 0.0;
        for (j, &p) in row.iter().enumerate() {
            acc += p;
            if u <= acc {
                return j;
            }
        }
        row.len() - 1
    }

    /// A trajectory of `steps` transitions from `start` — a vector of `steps + 1`
    /// states (the first being `start`).
    pub fn trajectory<R: Rng + ?Sized>(
        &self,
        start: usize,
        steps: usize,
        rng: &mut R,
    ) -> Vec<usize> {
        let mut out = Vec::with_capacity(steps + 1);
        let mut s = start.min(self.states() - 1);
        out.push(s);
        for _ in 0..steps {
            s = self.step(s, rng);
            out.push(s);
        }
        out
    }

    /// The stationary distribution `π` (with `π P = π`) by power iteration from the
    /// uniform distribution. Converges for an irreducible aperiodic chain.
    pub fn stationary(&self) -> Vec<f64> {
        let n = self.states();
        let mut pi = vec![1.0 / n as f64; n];
        for _ in 0..1_000 {
            let mut next = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    next[j] += pi[i] * self.transition[i][j];
                }
            }
            let diff: f64 = next.iter().zip(&pi).map(|(a, b)| (a - b).abs()).sum();
            pi = next;
            if diff < 1e-14 {
                break;
            }
        }
        pi
    }
}

impl Distribution<usize> for MarkovChain {
    /// Sampling a `MarkovChain` directly draws a state from its stationary
    /// distribution.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let pi = self.stationary();
        let u: f64 = rng.random();
        let mut acc = 0.0;
        for (j, &p) in pi.iter().enumerate() {
            acc += p;
            if u <= acc {
                return j;
            }
        }
        pi.len() - 1
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
    fn rejects_non_stochastic_matrices() {
        assert!(MarkovChain::new(vec![vec![0.5, 0.4], vec![0.5, 0.5]]).is_err());
        assert!(MarkovChain::new(vec![vec![1.0, 0.0]]).is_err());
    }

    #[test]
    fn trajectory_stays_in_state_space() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mc = MarkovChain::new(vec![vec![0.2, 0.8], vec![0.6, 0.4]]).unwrap();
        let traj = mc.trajectory(0, 50, &mut rng);
        assert_eq!(traj.len(), 51);
        assert!(traj.iter().all(|&s| s < 2));
    }

    #[test]
    fn stationary_distribution_is_correct() {
        let mc = MarkovChain::new(vec![vec![0.2, 0.8], vec![0.6, 0.4]]).unwrap();
        let pi = mc.stationary();
        close(pi[0], 3.0 / 7.0, 1e-9);
        close(pi[1], 4.0 / 7.0, 1e-9);
        close(pi[0] + pi[1], 1.0, 1e-12);
    }
}
