//! **Mutual information** `I(X;Y) = KL(p_{XY} ‖ p_X ⊗ p_Y)` over a joint
//! distribution table.

use super::kl_term;

/// **Mutual information** `I(X;Y) = Σ_{i,k} J[i][k]·ln( J[i][k] / (p_X[i]·p_Y[k]) )`,
/// in nats, for a joint distribution `joint` (row `i` = outcome of `X`, column `k`
/// = outcome of `Y`; entries should sum to 1). Equivalently `KL(p_{XY} ‖ p_X ⊗ p_Y)`.
///
/// `I(X;Y) ≥ 0`, with equality iff `X` and `Y` are independent — proved in `riemann`
/// (`RiemannLean/Bayes/Divergence.lean`: `mutualInfo_nonneg`). The `0·ln 0 = 0`
/// convention is applied to empty cells.
pub fn mutual_information(joint: &[Vec<f64>]) -> f64 {
    if joint.is_empty() {
        return 0.0;
    }
    let cols = joint.iter().map(Vec::len).max().unwrap_or(0);
    let px: Vec<f64> = joint.iter().map(|row| row.iter().sum()).collect();
    let mut py = vec![0.0; cols];
    for row in joint {
        for (k, &v) in row.iter().enumerate() {
            py[k] += v;
        }
    }
    joint
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().map(move |(k, &j)| (i, k, j)))
        .map(|(i, k, j)| kl_term(j, px[i] * py[k]))
        .sum()
}
