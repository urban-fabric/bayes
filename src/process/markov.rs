//! Discrete-state, discrete-time **Markov chains** as *generative* processes:
//! sample state trajectories from a row-stochastic transition matrix, and compute
//! the stationary distribution. (This is the chain as a random process — distinct
//! from MCMC *inference*, which is out of scope for `bayes`.)
#![allow(clippy::needless_range_loop)]

use ramanujan::DenseMatrix;
use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{validate_probability_vector, validate_row_stochastic_matrix};
use crate::error::BayesError;

const MARKOV_TOLERANCE: f64 = 1e-12;

/// A finite-state Markov chain with a row-stochastic transition matrix `P`, where
/// `P[i][j] = P(next = j | current = i)`.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkovChain {
    transition: DenseMatrix,
}

impl MarkovChain {
    /// Build from a transition matrix, validating that it is square and each row is
    /// a probability distribution (non-negative, summing to 1).
    pub fn new(transition: DenseMatrix) -> Result<Self, BayesError> {
        let n = transition.rows();
        if n == 0 {
            return Err(BayesError::Malformed {
                what: "transition matrix is empty",
            });
        }
        if transition.cols() != n {
            return Err(BayesError::Malformed {
                what: "transition matrix is not square",
            });
        }
        validate_row_stochastic_matrix(&transition)?;
        Ok(Self { transition })
    }

    /// The number of states.
    pub fn states(&self) -> usize {
        self.transition.rows()
    }

    /// The row-stochastic transition matrix.
    pub fn transition(&self) -> &DenseMatrix {
        &self.transition
    }

    /// The next state sampled from row `current` of the transition matrix.
    fn step<R: Rng + ?Sized>(&self, current: usize, rng: &mut R) -> usize {
        let row = self.transition.row(current);
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

    /// A stationary distribution `π` (with `π P = π`) computed by direct finite
    /// linear algebra.
    ///
    /// For a chain with one closed communicating class this is the unique
    /// stationary distribution. For reducible chains, the stationary distribution
    /// is not unique; this returns the equal-weight mixture over the closed
    /// communicating classes. No convergence loop is used.
    pub fn stationary(&self) -> Vec<f64> {
        let n = self.states();
        let classes = self.recurrent_classes();
        if classes.is_empty() {
            return Vec::new();
        }
        let mut pi = vec![0.0; n];
        let class_weight = 1.0 / classes.len() as f64;
        for class in &classes {
            let local = self.stationary_for_class(class);
            if local.len() != class.len() {
                return Vec::new();
            }
            for (&state, &p) in class.iter().zip(&local) {
                pi[state] = class_weight * p;
            }
        }
        pi
    }

    /// The `steps`-step transition matrix `P^steps`.
    ///
    /// `steps = 0` returns the identity matrix. This is the exact finite-chain
    /// evolution operator; iterative inference over model parameters belongs in
    /// the `markov` package.
    pub fn transition_power(&self, steps: usize) -> DenseMatrix {
        let n = self.states();
        let mut out = DenseMatrix::identity(n);
        for _ in 0..steps {
            out = multiply(&out, &self.transition);
        }
        out
    }

    /// Probability of being in state `to` after `steps` transitions from `from`.
    ///
    /// Returns `NaN` if either state index is out of range.
    pub fn transition_probability(&self, from: usize, to: usize, steps: usize) -> f64 {
        if from >= self.states() || to >= self.states() {
            return f64::NAN;
        }
        self.transition_power(steps).get(from, to)
    }

    /// Evolve an initial state distribution by `steps` transitions.
    ///
    /// Returns an empty vector if `initial` is not a probability vector with one
    /// entry per state.
    pub fn distribution_after(&self, initial: &[f64], steps: usize) -> Vec<f64> {
        let n = self.states();
        if initial.len() != n || validate_probability_vector(initial).is_err() {
            return Vec::new();
        }
        let mut dist = initial.to_vec();
        let mut next = vec![0.0; n];
        for _ in 0..steps {
            next.fill(0.0);
            for i in 0..n {
                for j in 0..n {
                    next[j] += dist[i] * self.transition.get(i, j);
                }
            }
            dist.copy_from_slice(&next);
        }
        dist
    }

    /// Distribution after `steps` transitions from a point-mass start state.
    ///
    /// Returns an empty vector when `start` is out of range.
    pub fn state_distribution(&self, start: usize, steps: usize) -> Vec<f64> {
        let n = self.states();
        if start >= n {
            return Vec::new();
        }
        let mut initial = vec![0.0; n];
        initial[start] = 1.0;
        self.distribution_after(&initial, steps)
    }

    /// Whether `state` is absorbing: `P[state,state] = 1`.
    pub fn is_absorbing_state(&self, state: usize) -> bool {
        if state >= self.states() {
            return false;
        }
        for j in 0..self.states() {
            let p = self.transition.get(state, j);
            if j == state {
                if (p - 1.0).abs() > MARKOV_TOLERANCE {
                    return false;
                }
            } else if p.abs() > MARKOV_TOLERANCE {
                return false;
            }
        }
        true
    }

    /// All absorbing states.
    pub fn absorbing_states(&self) -> Vec<usize> {
        (0..self.states())
            .filter(|&state| self.is_absorbing_state(state))
            .collect()
    }

    /// Communicating classes of the finite transition graph.
    ///
    /// States communicate when each is reachable from the other along positive
    /// transition-probability edges.
    pub fn communicating_classes(&self) -> Vec<Vec<usize>> {
        let n = self.states();
        let reach = self.reachability();
        let mut assigned = vec![false; n];
        let mut classes = Vec::new();
        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let class: Vec<usize> = (0..n).filter(|&j| reach[i][j] && reach[j][i]).collect();
            for &j in &class {
                assigned[j] = true;
            }
            classes.push(class);
        }
        classes
    }

    /// Closed communicating classes. In a finite Markov chain these are the
    /// recurrent classes.
    pub fn recurrent_classes(&self) -> Vec<Vec<usize>> {
        self.communicating_classes()
            .into_iter()
            .filter(|class| self.is_closed_class(class))
            .collect()
    }

    /// States in closed communicating classes.
    pub fn recurrent_states(&self) -> Vec<usize> {
        let mut states: Vec<usize> = self.recurrent_classes().into_iter().flatten().collect();
        states.sort_unstable();
        states
    }

    /// States outside closed communicating classes.
    pub fn transient_states(&self) -> Vec<usize> {
        let n = self.states();
        let mut recurrent = vec![false; n];
        for state in self.recurrent_states() {
            recurrent[state] = true;
        }
        (0..n).filter(|&state| !recurrent[state]).collect()
    }

    /// Fundamental matrix `N = (I - Q)⁻¹` over transient states.
    ///
    /// Rows and columns are ordered by [`transient_states`](Self::transient_states).
    /// Entry `N[i,j]` is the expected number of visits to transient state `j`
    /// before entering a recurrent class, when starting from transient state `i`.
    pub fn fundamental_matrix(&self) -> DenseMatrix {
        let transient = self.transient_states();
        if transient.is_empty() {
            return DenseMatrix::zeros(0, 0);
        }
        self.fundamental_matrix_for_states(&transient)
            .unwrap_or_else(|| DenseMatrix::zeros(0, 0))
    }

    /// Expected visits among transient states before entering a recurrent class.
    ///
    /// This is an alias for [`fundamental_matrix`](Self::fundamental_matrix), named
    /// for the direct interpretation of the entries.
    pub fn expected_transient_visit_counts(&self) -> DenseMatrix {
        self.fundamental_matrix()
    }

    /// Probability of ever hitting one of `targets`, for each starting state.
    ///
    /// The returned vector has one entry per state; target states have probability
    /// `1`, states that cannot reach a target have probability `0`. Returns an
    /// empty vector for an empty or out-of-range target set.
    pub fn hitting_probabilities(&self, targets: &[usize]) -> Vec<f64> {
        let Some(target) = self.target_mask(targets) else {
            return Vec::new();
        };
        let n = self.states();
        let can_reach = self.can_reach_targets(&target);
        let active: Vec<usize> = (0..n)
            .filter(|&state| !target[state] && can_reach[state])
            .collect();
        let mut out = vec![0.0; n];
        for state in 0..n {
            if target[state] {
                out[state] = 1.0;
            }
        }
        if active.is_empty() {
            return out;
        }
        let index = index_map(n, &active);
        let m = active.len();
        let a = DenseMatrix::from_fn(m, m, |r, c| {
            let i = active[r];
            let j = active[c];
            (if r == c { 1.0 } else { 0.0 }) - self.transition.get(i, j)
        });
        let b: Vec<f64> = active
            .iter()
            .map(|&i| {
                (0..n)
                    .filter(|&j| target[j])
                    .map(|j| self.transition.get(i, j))
                    .sum()
            })
            .collect();
        let Ok(lu) = a.lu() else {
            return Vec::new();
        };
        let solution = lu.solve(&b);
        for &state in &active {
            out[state] = clamp_probability(solution[index[state]]);
        }
        out
    }

    /// Expected hitting time to `targets`, for each starting state.
    ///
    /// Target states have time `0`; states that do not hit the target set almost
    /// surely have `+∞`. Returns an empty vector for an empty or out-of-range
    /// target set.
    pub fn expected_hitting_times(&self, targets: &[usize]) -> Vec<f64> {
        let Some(target) = self.target_mask(targets) else {
            return Vec::new();
        };
        let n = self.states();
        let hit = self.hitting_probabilities(targets);
        if hit.len() != n {
            return Vec::new();
        }
        let active: Vec<usize> = (0..n)
            .filter(|&state| !target[state] && (hit[state] - 1.0).abs() <= 1e-9)
            .collect();
        let mut out = vec![f64::INFINITY; n];
        for state in 0..n {
            if target[state] {
                out[state] = 0.0;
            }
        }
        if active.is_empty() {
            return out;
        }
        let index = index_map(n, &active);
        let m = active.len();
        let a = DenseMatrix::from_fn(m, m, |r, c| {
            let i = active[r];
            let j = active[c];
            (if r == c { 1.0 } else { 0.0 }) - self.transition.get(i, j)
        });
        let b = vec![1.0; m];
        let Ok(lu) = a.lu() else {
            return Vec::new();
        };
        let solution = lu.solve(&b);
        for &state in &active {
            out[state] = solution[index[state]].max(0.0);
        }
        out
    }

    /// Variance of the hitting time to `targets`, for each starting state.
    ///
    /// Target states have variance `0`; states that do not hit the target set
    /// almost surely have `+∞`. Returns an empty vector for an empty or
    /// out-of-range target set.
    pub fn hitting_time_variances(&self, targets: &[usize]) -> Vec<f64> {
        let Some(target) = self.target_mask(targets) else {
            return Vec::new();
        };
        let n = self.states();
        let means = self.expected_hitting_times(targets);
        if means.len() != n {
            return Vec::new();
        }
        let active: Vec<usize> = (0..n)
            .filter(|&state| !target[state] && means[state].is_finite())
            .collect();
        let mut out = vec![f64::INFINITY; n];
        for state in 0..n {
            if target[state] {
                out[state] = 0.0;
            }
        }
        if active.is_empty() {
            return out;
        }
        let Some(fundamental) = self.fundamental_matrix_for_states(&active) else {
            return Vec::new();
        };
        let m = active.len();
        let t: Vec<f64> = (0..m)
            .map(|i| (0..m).map(|j| fundamental.get(i, j)).sum())
            .collect();
        for (row, &state) in active.iter().enumerate() {
            let second_minus_mean = (0..m)
                .map(|j| 2.0 * fundamental.get(row, j) * t[j])
                .sum::<f64>()
                - t[row];
            out[state] = (second_minus_mean - t[row] * t[row]).max(0.0);
        }
        out
    }

    /// Absorption probabilities into each absorbing state.
    ///
    /// The returned matrix has one row per starting state and one column per
    /// absorbing state, in [`absorbing_states`](Self::absorbing_states) order.
    /// If the chain has no absorbing states, the result has zero columns.
    pub fn absorption_probabilities(&self) -> DenseMatrix {
        let absorbing = self.absorbing_states();
        let n = self.states();
        if absorbing.is_empty() {
            return DenseMatrix::zeros(n, 0);
        }
        let mut out = DenseMatrix::zeros(n, absorbing.len());
        for (col, &target) in absorbing.iter().enumerate() {
            let probs = self.hitting_probabilities(&[target]);
            if probs.len() != n {
                return DenseMatrix::zeros(0, 0);
            }
            for (row, &p) in probs.iter().enumerate() {
                out.set(row, col, p);
            }
        }
        out
    }

    /// Expected time to hit any absorbing state.
    ///
    /// If the chain has no absorbing states, every state has infinite expected
    /// absorption time.
    pub fn expected_absorption_times(&self) -> Vec<f64> {
        let absorbing = self.absorbing_states();
        if absorbing.is_empty() {
            return vec![f64::INFINITY; self.states()];
        }
        self.expected_hitting_times(&absorbing)
    }

    /// Variance of the time to hit any absorbing state.
    ///
    /// If the chain has no absorbing states, every state has infinite variance.
    pub fn absorption_time_variances(&self) -> Vec<f64> {
        let absorbing = self.absorbing_states();
        if absorbing.is_empty() {
            return vec![f64::INFINITY; self.states()];
        }
        self.hitting_time_variances(&absorbing)
    }

    /// Mean recurrence time for each state.
    ///
    /// Recurrent states use the closed form `1 / πᵢ` within their closed
    /// communicating class. Transient states have infinite mean recurrence time.
    pub fn mean_recurrence_times(&self) -> Vec<f64> {
        let n = self.states();
        let mut out = vec![f64::INFINITY; n];
        for class in self.recurrent_classes() {
            let pi = self.stationary_for_class(&class);
            if pi.len() != class.len() {
                return Vec::new();
            }
            for (&state, &p) in class.iter().zip(&pi) {
                out[state] = if p > 0.0 { 1.0 / p } else { f64::INFINITY };
            }
        }
        out
    }

    /// Whether the chain satisfies detailed balance with its direct stationary
    /// distribution.
    pub fn is_reversible(&self) -> bool {
        let pi = self.stationary();
        pi.len() == self.states() && self.is_reversible_with(&pi)
    }

    /// Whether `stationary` satisfies detailed balance:
    /// `πᵢ Pᵢⱼ = πⱼ Pⱼᵢ` for every state pair.
    pub fn is_reversible_with(&self, stationary: &[f64]) -> bool {
        let n = self.states();
        if stationary.len() != n || validate_probability_vector(stationary).is_err() {
            return false;
        }
        for i in 0..n {
            for j in 0..n {
                let lhs = stationary[i] * self.transition.get(i, j);
                let rhs = stationary[j] * self.transition.get(j, i);
                let scale = lhs.abs().max(rhs.abs()).max(1.0);
                if (lhs - rhs).abs() > 1e-9 * scale {
                    return false;
                }
            }
        }
        true
    }

    /// Time-reversed transition matrix under a full-support stationary
    /// distribution: `P*ᵢⱼ = πⱼ Pⱼᵢ / πᵢ`.
    ///
    /// The full-support requirement avoids ambiguity on zero-stationary-mass
    /// transient states.
    pub fn time_reversed(&self, stationary: &[f64]) -> Result<Self, BayesError> {
        let n = self.states();
        if stationary.len() != n || validate_probability_vector(stationary).is_err() {
            return Err(BayesError::Malformed {
                what: "stationary distribution must be a probability vector with one entry per state",
            });
        }
        if stationary.iter().any(|&p| p <= MARKOV_TOLERANCE) {
            return Err(BayesError::Malformed {
                what: "time reversal requires a full-support stationary distribution",
            });
        }
        let reversed = DenseMatrix::from_fn(n, n, |i, j| {
            stationary[j] * self.transition.get(j, i) / stationary[i]
        });
        MarkovChain::new(reversed)
    }

    /// Time-reversed chain using [`stationary`](Self::stationary).
    ///
    /// Reducible chains with zero-mass states in the selected stationary
    /// distribution return an error; call [`time_reversed`](Self::time_reversed)
    /// with an explicit full-support stationary distribution if appropriate.
    pub fn time_reversed_stationary(&self) -> Result<Self, BayesError> {
        self.time_reversed(&self.stationary())
    }

    /// Probability that the first return to `state` happens exactly after
    /// `steps` transitions.
    ///
    /// `steps = 0` returns `0` because first return is defined after leaving time
    /// zero. Returns `NaN` for an out-of-range state.
    pub fn first_return_probability(&self, state: usize, steps: usize) -> f64 {
        let n = self.states();
        if state >= n {
            return f64::NAN;
        }
        if steps == 0 {
            return 0.0;
        }
        if steps == 1 {
            return self.transition.get(state, state);
        }
        let others: Vec<usize> = (0..n).filter(|&j| j != state).collect();
        if others.is_empty() {
            return 0.0;
        }
        let q = DenseMatrix::from_fn(others.len(), others.len(), |r, c| {
            self.transition.get(others[r], others[c])
        });
        let q_power = matrix_power(&q, steps - 2);
        let mut prob = 0.0;
        for (r, &from) in others.iter().enumerate() {
            let start_to_from = self.transition.get(state, from);
            if start_to_from == 0.0 {
                continue;
            }
            for (c, &to) in others.iter().enumerate() {
                prob += start_to_from * q_power.get(r, c) * self.transition.get(to, state);
            }
        }
        prob
    }

    fn stationary_for_class(&self, class: &[usize]) -> Vec<f64> {
        let m = class.len();
        if m == 0 {
            return Vec::new();
        }
        if m == 1 {
            return vec![1.0];
        }
        let a = DenseMatrix::from_fn(m, m, |r, c| {
            if r == m - 1 {
                1.0
            } else {
                self.transition.get(class[c], class[r]) - if r == c { 1.0 } else { 0.0 }
            }
        });
        let mut b = vec![0.0; m];
        b[m - 1] = 1.0;
        let Ok(lu) = a.lu() else {
            return Vec::new();
        };
        normalize_signed_probability(lu.solve(&b))
    }

    fn fundamental_matrix_for_states(&self, states: &[usize]) -> Option<DenseMatrix> {
        let m = states.len();
        if m == 0 {
            return Some(DenseMatrix::zeros(0, 0));
        }
        let a = DenseMatrix::from_fn(m, m, |r, c| {
            (if r == c { 1.0 } else { 0.0 }) - self.transition.get(states[r], states[c])
        });
        a.lu().ok().map(|lu| lu.inverse())
    }

    fn reachability(&self) -> Vec<Vec<bool>> {
        let n = self.states();
        let mut reach = vec![vec![false; n]; n];
        for start in 0..n {
            let mut stack = vec![start];
            reach[start][start] = true;
            while let Some(i) = stack.pop() {
                for j in 0..n {
                    if !reach[start][j] && self.transition.get(i, j) > MARKOV_TOLERANCE {
                        reach[start][j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        reach
    }

    fn is_closed_class(&self, class: &[usize]) -> bool {
        let n = self.states();
        let mut in_class = vec![false; n];
        for &state in class {
            in_class[state] = true;
        }
        for &state in class {
            for j in 0..n {
                if !in_class[j] && self.transition.get(state, j) > MARKOV_TOLERANCE {
                    return false;
                }
            }
        }
        true
    }

    fn target_mask(&self, targets: &[usize]) -> Option<Vec<bool>> {
        let n = self.states();
        if targets.is_empty() || targets.iter().any(|&state| state >= n) {
            return None;
        }
        let mut target = vec![false; n];
        for &state in targets {
            target[state] = true;
        }
        Some(target)
    }

    fn can_reach_targets(&self, target: &[bool]) -> Vec<bool> {
        let reach = self.reachability();
        (0..self.states())
            .map(|state| (0..self.states()).any(|j| target[j] && reach[state][j]))
            .collect()
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

fn multiply(a: &DenseMatrix, b: &DenseMatrix) -> DenseMatrix {
    let rows = a.rows();
    let cols = b.cols();
    let inner = a.cols();
    let mut out = DenseMatrix::zeros(rows, cols);
    for i in 0..rows {
        for k in 0..inner {
            let aik = a.get(i, k);
            if aik == 0.0 {
                continue;
            }
            for j in 0..cols {
                out.set(i, j, out.get(i, j) + aik * b.get(k, j));
            }
        }
    }
    out
}

fn matrix_power(matrix: &DenseMatrix, steps: usize) -> DenseMatrix {
    let n = matrix.rows();
    let mut out = DenseMatrix::identity(n);
    for _ in 0..steps {
        out = multiply(&out, matrix);
    }
    out
}

fn index_map(n: usize, states: &[usize]) -> Vec<usize> {
    let mut index = vec![usize::MAX; n];
    for (pos, &state) in states.iter().enumerate() {
        index[state] = pos;
    }
    index
}

fn clamp_probability(p: f64) -> f64 {
    if !p.is_finite() {
        f64::NAN
    } else if p.abs() <= MARKOV_TOLERANCE {
        0.0
    } else if (p - 1.0).abs() <= MARKOV_TOLERANCE {
        1.0
    } else {
        p.clamp(0.0, 1.0)
    }
}

fn normalize_signed_probability(mut p: Vec<f64>) -> Vec<f64> {
    for pi in &mut p {
        if !pi.is_finite() || *pi < -1e-9 {
            return Vec::new();
        }
        if pi.abs() <= MARKOV_TOLERANCE {
            *pi = 0.0;
        }
    }
    let sum = p.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Vec::new();
    }
    for pi in &mut p {
        *pi /= sum;
    }
    p
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
        assert!(
            MarkovChain::new(DenseMatrix::from_rows(&[vec![0.5, 0.4], vec![0.5, 0.5]])).is_err()
        );
        assert!(MarkovChain::new(DenseMatrix::from_rows(&[vec![1.0, 0.0]])).is_err());
    }

    #[test]
    fn trajectory_stays_in_state_space() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mc =
            MarkovChain::new(DenseMatrix::from_rows(&[vec![0.2, 0.8], vec![0.6, 0.4]])).unwrap();
        let traj = mc.trajectory(0, 50, &mut rng);
        assert_eq!(traj.len(), 51);
        assert!(traj.iter().all(|&s| s < 2));
    }

    #[test]
    fn stationary_distribution_is_correct() {
        let mc =
            MarkovChain::new(DenseMatrix::from_rows(&[vec![0.2, 0.8], vec![0.6, 0.4]])).unwrap();
        let pi = mc.stationary();
        close(pi[0], 3.0 / 7.0, 1e-9);
        close(pi[1], 4.0 / 7.0, 1e-9);
        close(pi[0] + pi[1], 1.0, 1e-12);
    }

    #[test]
    fn finite_step_distributions_match_matrix_powers() {
        let mc =
            MarkovChain::new(DenseMatrix::from_rows(&[vec![0.2, 0.8], vec![0.6, 0.4]])).unwrap();

        let p2 = mc.transition_power(2);
        close(p2.get(0, 0), 0.52, 1e-12);
        close(p2.get(0, 1), 0.48, 1e-12);
        close(mc.transition_probability(0, 1, 2), 0.48, 1e-12);

        let dist = mc.distribution_after(&[1.0, 0.0], 2);
        close(dist[0], p2.get(0, 0), 1e-12);
        close(dist[1], p2.get(0, 1), 1e-12);
        assert_eq!(mc.state_distribution(0, 0), vec![1.0, 0.0]);
        assert!(mc.state_distribution(2, 1).is_empty());
        assert!(mc.distribution_after(&[0.5, 0.6], 1).is_empty());
        assert!(mc.transition_probability(2, 0, 1).is_nan());
    }

    #[test]
    fn finite_classification_finds_absorbing_recurrent_and_transient_states() {
        let mc = MarkovChain::new(DenseMatrix::from_rows(&[
            vec![0.5, 0.5, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.2, 0.3, 0.5],
        ]))
        .unwrap();

        assert!(mc.is_absorbing_state(1));
        assert_eq!(mc.absorbing_states(), vec![1]);
        assert_eq!(mc.recurrent_classes(), vec![vec![1]]);
        assert_eq!(mc.recurrent_states(), vec![1]);
        assert_eq!(mc.transient_states(), vec![0, 2]);
        assert_eq!(mc.communicating_classes(), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn hitting_and_absorption_quantities_are_direct_linear_solves() {
        let mc = MarkovChain::new(DenseMatrix::from_rows(&[
            vec![0.5, 0.5, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.2, 0.3, 0.5],
        ]))
        .unwrap();

        let hit = mc.hitting_probabilities(&[1]);
        close(hit[0], 1.0, 1e-12);
        close(hit[1], 1.0, 1e-12);
        close(hit[2], 1.0, 1e-12);

        let times = mc.expected_hitting_times(&[1]);
        close(times[0], 2.0, 1e-12);
        close(times[1], 0.0, 1e-12);
        close(times[2], 2.8, 1e-12);

        let absorption = mc.absorption_probabilities();
        assert_eq!((absorption.rows(), absorption.cols()), (3, 1));
        close(absorption.get(0, 0), 1.0, 1e-12);
        close(absorption.get(2, 0), 1.0, 1e-12);

        let absorption_times = mc.expected_absorption_times();
        close(absorption_times[0], 2.0, 1e-12);
        close(absorption_times[2], 2.8, 1e-12);
    }

    #[test]
    fn fundamental_matrix_counts_visits_and_absorption_variances() {
        let mc = MarkovChain::new(DenseMatrix::from_rows(&[
            vec![0.5, 0.5, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.2, 0.3, 0.5],
        ]))
        .unwrap();

        let fundamental = mc.fundamental_matrix();
        assert_eq!((fundamental.rows(), fundamental.cols()), (2, 2));
        close(fundamental.get(0, 0), 2.0, 1e-12);
        close(fundamental.get(0, 1), 0.0, 1e-12);
        close(fundamental.get(1, 0), 0.8, 1e-12);
        close(fundamental.get(1, 1), 2.0, 1e-12);

        let visits = mc.expected_transient_visit_counts();
        close(visits.get(1, 0), fundamental.get(1, 0), 1e-12);
        close(visits.get(1, 1), fundamental.get(1, 1), 1e-12);

        let hitting_variances = mc.hitting_time_variances(&[1]);
        close(hitting_variances[0], 2.0, 1e-12);
        close(hitting_variances[1], 0.0, 1e-12);
        close(hitting_variances[2], 3.76, 1e-12);

        let absorption_variances = mc.absorption_time_variances();
        close(absorption_variances[0], 2.0, 1e-12);
        close(absorption_variances[1], 0.0, 1e-12);
        close(absorption_variances[2], 3.76, 1e-12);
    }

    #[test]
    fn recurrence_reversibility_and_time_reversal_are_closed_form() {
        let mc =
            MarkovChain::new(DenseMatrix::from_rows(&[vec![0.2, 0.8], vec![0.6, 0.4]])).unwrap();

        let recurrence = mc.mean_recurrence_times();
        close(recurrence[0], 7.0 / 3.0, 1e-12);
        close(recurrence[1], 7.0 / 4.0, 1e-12);
        assert!(mc.is_reversible());
        assert!(mc.is_reversible_with(&[3.0 / 7.0, 4.0 / 7.0]));
        assert!(mc.time_reversed(&[0.0, 1.0]).is_err());

        let reversed = mc.time_reversed_stationary().unwrap();
        for i in 0..2 {
            for j in 0..2 {
                close(
                    reversed.transition().get(i, j),
                    mc.transition().get(i, j),
                    1e-12,
                );
            }
        }

        let cycle = MarkovChain::new(DenseMatrix::from_rows(&[
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0],
        ]))
        .unwrap();
        let pi = cycle.stationary();
        close(pi[0], 1.0 / 3.0, 1e-12);
        close(pi[1], 1.0 / 3.0, 1e-12);
        close(pi[2], 1.0 / 3.0, 1e-12);
        assert!(!cycle.is_reversible());

        let reversed_cycle = cycle.time_reversed_stationary().unwrap();
        close(reversed_cycle.transition().get(0, 2), 1.0, 1e-12);
        close(reversed_cycle.transition().get(1, 0), 1.0, 1e-12);
        close(reversed_cycle.transition().get(2, 1), 1.0, 1e-12);
    }

    #[test]
    fn unreachable_targets_have_zero_probability_and_infinite_hitting_time() {
        let mc = MarkovChain::new(DenseMatrix::from_rows(&[
            vec![1.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.5],
            vec![0.0, 0.0, 1.0],
        ]))
        .unwrap();

        let hit = mc.hitting_probabilities(&[2]);
        close(hit[0], 0.0, 1e-12);
        close(hit[1], 0.5, 1e-12);
        close(hit[2], 1.0, 1e-12);

        let times = mc.expected_hitting_times(&[2]);
        assert!(times[0].is_infinite());
        assert!(times[1].is_infinite());
        close(times[2], 0.0, 1e-12);
    }

    #[test]
    fn first_return_probability_excludes_earlier_returns() {
        let mc =
            MarkovChain::new(DenseMatrix::from_rows(&[vec![0.2, 0.8], vec![0.6, 0.4]])).unwrap();

        close(mc.first_return_probability(0, 0), 0.0, 1e-12);
        close(mc.first_return_probability(0, 1), 0.2, 1e-12);
        close(mc.first_return_probability(0, 2), 0.8 * 0.6, 1e-12);
        close(mc.first_return_probability(0, 3), 0.8 * 0.4 * 0.6, 1e-12);
        assert!(mc.first_return_probability(2, 1).is_nan());
    }
}
