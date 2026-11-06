//! Vose's alias method: draws from an arbitrary discrete distribution in **O(1)**.
//!
//! The obvious way to sample `n` outcomes with given probabilities is to build
//! the cumulative distribution and binary-search a uniform, which costs
//! `O(log n)` per draw — or, as `Categorical` did before this existed, to scan it
//! linearly for `O(n)`. Either is fine for a six-sided die and ruinous for a
//! 50,000-token vocabulary sampled a million times.
//!
//! The alias method removes the search entirely. The idea is that any
//! distribution over `n` outcomes can be rearranged into `n` equally likely
//! columns each holding **at most two** outcomes. A draw then costs one integer
//! pick plus one coin flip, whatever `n` is:
//!
//! ```text
//!     i = floor(n * u1)                      // which column
//!     return if u2 < prob[i] { i } else { alias[i] }
//! ```
//!
//! Building the table is `O(n)` by Vose's worklist construction: scale every
//! probability by `n`, split the outcomes into those now below 1 ("small") and
//! those at or above it ("large"), then repeatedly take one of each and use the
//! large one to top the small one up to exactly 1. Each pairing finalizes one
//! column and returns the large outcome to whichever list its remainder belongs
//! to, so the loop runs at most `n` times.

use bayes_core::BayesError;
use rand::{Rng, RngExt};

/// An alias table over `n` outcomes, sampling in constant time.
///
/// Build once, draw many times. Construction normalizes, so the input may be
/// unnormalized weights rather than probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct AliasTable {
    /// `prob[i]` is the chance of keeping column `i`'s primary outcome.
    prob: Vec<f64>,
    /// `alias[i]` is the outcome taken when column `i`'s coin flip fails.
    alias: Vec<usize>,
    /// The normalized probabilities, retained so [`AliasTable::pmf`] can answer
    /// without reconstructing them from the table.
    pmf: Vec<f64>,
}

impl AliasTable {
    /// Build a table from non-negative weights, which need not sum to 1.
    ///
    /// Rejects an empty input, a non-finite or negative weight, and a total of
    /// zero — all of which describe no distribution at all rather than a
    /// degenerate one.
    pub fn new(weights: &[f64]) -> Result<Self, BayesError> {
        if weights.is_empty() {
            return Err(BayesError::Malformed {
                what: "an alias table needs at least one outcome",
            });
        }
        let mut total = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            if !w.is_finite() {
                return Err(BayesError::NonFinite {
                    name: "weight",
                    value: w,
                });
            }
            if w < 0.0 {
                return Err(BayesError::NotPositive {
                    name: "weight",
                    value: w,
                });
            }
            total += w;
            let _ = i;
        }
        if total <= 0.0 {
            return Err(BayesError::Malformed {
                what: "alias table weights sum to zero",
            });
        }

        let n = weights.len();
        let pmf: Vec<f64> = weights.iter().map(|&w| w / total).collect();

        // Scale so the mean is exactly 1; a column is "small" below that and
        // "large" at or above it.
        let mut scaled: Vec<f64> = pmf.iter().map(|&p| p * n as f64).collect();
        let mut small: Vec<usize> = Vec::with_capacity(n);
        let mut large: Vec<usize> = Vec::with_capacity(n);
        for (i, &s) in scaled.iter().enumerate() {
            if s < 1.0 {
                small.push(i)
            } else {
                large.push(i)
            }
        }

        let mut prob = vec![1.0; n];
        let mut alias: Vec<usize> = (0..n).collect();
        // Each pass finalizes the small column and reduces the large one, so the
        // total work is linear even though the large index may be revisited.
        while let (Some(s), Some(l)) = (small.pop(), large.pop()) {
            prob[s] = scaled[s];
            alias[s] = l;
            // The large column gives away exactly the shortfall of the small one.
            scaled[l] -= 1.0 - scaled[s];
            if scaled[l] < 1.0 {
                small.push(l);
            } else {
                large.push(l);
            }
        }
        // Whatever remains is within rounding of 1 and keeps its own outcome.
        for i in small.into_iter().chain(large) {
            prob[i] = 1.0;
            alias[i] = i;
        }

        Ok(Self { prob, alias, pmf })
    }

    /// The number of outcomes.
    pub fn len(&self) -> usize {
        self.pmf.len()
    }

    /// Always false: an empty table cannot be constructed.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The normalized probability of outcome `k`, or 0 outside the support.
    pub fn pmf(&self, k: usize) -> f64 {
        self.pmf.get(k).copied().unwrap_or(0.0)
    }

    /// The normalized probabilities.
    pub fn probabilities(&self) -> &[f64] {
        &self.pmf
    }

    /// One draw, in constant time regardless of the number of outcomes.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let n = self.prob.len();
        // One uniform supplies both the column and the coin flip. Drawing the
        // column from a separate integer range would be marginally cleaner but
        // costs a second RNG call per draw, which is the whole budget here.
        let u: f64 = rng.random();
        let scaled = u * n as f64;
        let column = (scaled as usize).min(n - 1);
        let frac = scaled - column as f64;
        if frac < self.prob[column] {
            column
        } else {
            self.alias[column]
        }
    }

    /// `count` draws.
    pub fn sample_n<R: Rng + ?Sized>(&self, rng: &mut R, count: usize) -> Vec<usize> {
        (0..count).map(|_| self.sample(rng)).collect()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_table_reproduces_its_input_probabilities() {
        let t = AliasTable::new(&[2.0, 5.0, 3.0]).unwrap();
        assert!((t.pmf(0) - 0.2).abs() < 1e-15);
        assert!((t.pmf(1) - 0.5).abs() < 1e-15);
        assert!((t.pmf(2) - 0.3).abs() < 1e-15);
        assert_eq!(t.pmf(3), 0.0);
        assert!((t.probabilities().iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    /// Every column holds exactly one unit of probability, which is the
    /// invariant the construction exists to establish.
    #[test]
    fn every_column_is_filled_to_exactly_one() {
        for weights in [
            vec![1.0],
            vec![1.0, 1.0],
            vec![0.1, 0.9],
            vec![1.0, 0.0, 1.0],
            vec![1e-9, 1.0, 1e-9],
            (1..=50).map(|i| i as f64).collect::<Vec<_>>(),
        ] {
            let t = AliasTable::new(&weights).unwrap();
            let n = t.len();
            // Reconstruct each outcome's mass from the table and compare to the
            // normalized input: outcome k gets prob[k]/n from its own column
            // plus (1 - prob[j])/n from every column aliasing to it.
            let mut rebuilt = vec![0.0; n];
            for i in 0..n {
                rebuilt[i] += t.prob[i] / n as f64;
                rebuilt[t.alias[i]] += (1.0 - t.prob[i]) / n as f64;
            }
            for (k, &got) in rebuilt.iter().enumerate() {
                assert!(
                    (got - t.pmf(k)).abs() < 1e-12,
                    "weights {weights:?}: outcome {k} rebuilt as {got} but pmf is {}",
                    t.pmf(k)
                );
            }
        }
    }

    #[test]
    fn invalid_weights_are_rejected() {
        assert!(AliasTable::new(&[]).is_err());
        assert!(AliasTable::new(&[0.0, 0.0]).is_err());
        assert!(AliasTable::new(&[-1.0, 2.0]).is_err());
        assert!(AliasTable::new(&[f64::NAN, 1.0]).is_err());
        assert!(AliasTable::new(&[f64::INFINITY, 1.0]).is_err());
    }

    /// An outcome with zero weight must never be drawn.
    #[test]
    fn zero_weight_outcomes_are_never_drawn() {
        use rand::SeedableRng;
        let t = AliasTable::new(&[1.0, 0.0, 1.0, 0.0]).unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        for _ in 0..20_000 {
            let k = t.sample(&mut rng);
            assert!(k == 0 || k == 2, "drew {k}, which has zero weight");
        }
    }
}
