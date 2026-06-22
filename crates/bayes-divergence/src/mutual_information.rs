//! **Joint-distribution information measures** over a joint distribution table — mutual
//! information `I(X;Y)`, joint entropy `H(X, Y)`, conditional entropy `H(Y|X)`, the
//! **variation of information** metric, and the **normalized mutual information**. The
//! full information diagram, each quantity backed by a machine-checked theorem in `riemann`
//! `RiemannLean/Bayes/Divergence/Information.lean`:
//! - `I(X;Y) = H(X) + H(Y) − H(X,Y)` (`mutualInfo_eq_entropies`);
//! - `H(X) ≤ H(X,Y)` and `H(Y) ≤ H(X,Y)` (`entropy_fst_le_joint` / `entropy_snd_le_joint`,
//!   i.e. conditional entropy `≥ 0`);
//! - `I(X;Y) ≤ H(X)` and `≤ H(Y)` (`mutualInfo_le_entropy_fst` / `_snd`).

#[cfg(test)]
use super::shannon_entropy;
use super::{kl_term, shannon_entropy_real};
use euclid::Real;

/// The row (X) and column (Y) marginals of a joint distribution table: `p_X[i] = Σ_k J[i][k]`
/// and `p_Y[k] = Σ_i J[i][k]`. Shared by the joint-distribution measures below.
#[cfg(test)]
fn marginals(joint: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let cols = joint.iter().map(Vec::len).max().unwrap_or(0);
    let px: Vec<f64> = joint.iter().map(|row| row.iter().sum()).collect();
    let mut py = vec![0.0; cols];
    for row in joint {
        for (k, &v) in row.iter().enumerate() {
            py[k] += v;
        }
    }
    (px, py)
}

/// Generic row and column marginals of a joint distribution table.
fn marginals_real<R: Real>(joint: &[Vec<R>]) -> (Vec<R>, Vec<R>) {
    let cols = joint.iter().map(Vec::len).max().unwrap_or(0);
    let px: Vec<R> = joint
        .iter()
        .map(|row| row.iter().cloned().fold(R::zero(), |sum, v| sum + v))
        .collect();
    let mut py = vec![R::zero(); cols];
    for row in joint {
        for (k, v) in row.iter().enumerate() {
            py[k] = py[k].clone() + v.clone();
        }
    }
    (px, py)
}

/// **Mutual information** `I(X;Y) = Σ_{i,k} J[i][k]·ln( J[i][k] / (p_X[i]·p_Y[k]) )`,
/// in nats, for a joint distribution `joint` (row `i` = outcome of `X`, column `k`
/// = outcome of `Y`; entries should sum to 1). Equivalently `KL(p_{XY} ‖ p_X ⊗ p_Y)`.
///
/// `I(X;Y) ≥ 0`, with equality iff `X` and `Y` are independent — proved in `riemann`
/// (`RiemannLean/Bayes/Divergence.lean`: `mutualInfo_nonneg`). The `0·ln 0 = 0`
/// convention is applied to empty cells.
pub fn mutual_information(joint: &[Vec<f64>]) -> f64 {
    mutual_information_real(joint)
}

/// **Mutual information** over any [`Real`] carrier.
pub fn mutual_information_real<R: Real>(joint: &[Vec<R>]) -> R {
    if joint.is_empty() {
        return R::zero();
    }
    let (px, py) = marginals_real(joint);
    joint
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().map(move |(k, j)| (i, k, j)))
        .map(|(i, k, j)| kl_term(j.clone(), px[i].clone() * py[k].clone()))
        .fold(R::zero(), |sum, term| sum + term)
}

/// **Joint entropy** `H(X, Y) = −Σ_{i,k} J[i][k]·ln J[i][k]` (in nats, with `0 ln 0 = 0`)
/// of a joint distribution `joint` (entries should sum to 1) — the total uncertainty in
/// the pair `(X, Y)`. As the Shannon entropy of the flattened table it is nonnegative and
/// at most `ln(rows·cols)` (`riemann`: `entropy_nonneg`, `entropy_le_log_card` over the
/// product support). Together with [`mutual_information`] it completes the information
/// diagram: `I(X;Y) = H(X) + H(Y) − H(X,Y) ≤ H(X,Y)`, and a conditional entropy
/// `H(Y|X) = H(X,Y) − H(X)` follows by the chain rule.
pub fn joint_entropy(joint: &[Vec<f64>]) -> f64 {
    joint_entropy_real(joint)
}

/// **Joint entropy** over any [`Real`] carrier.
pub fn joint_entropy_real<R: Real>(joint: &[Vec<R>]) -> R {
    -joint
        .iter()
        .flat_map(|row| row.iter().cloned())
        .map(|j| j.clone().xlogy(j))
        .fold(R::zero(), |sum, term| sum + term)
}

/// **Conditional entropy** `H(Y|X) = H(X,Y) − H(X)` (in nats) — the residual uncertainty in
/// the column variable `Y` once the row variable `X` is known. By the chain rule this is the
/// joint entropy minus the X-marginal entropy, and it is **nonnegative** because conditioning
/// cannot create uncertainty — `riemann`'s `entropy_fst_le_joint` (`H(X) ≤ H(X,Y)`). The
/// reverse direction `H(X|Y)` is this function on the transposed table. Returns `0` for an
/// empty table (any tiny negative from rounding is clamped, the proved floor being `0`).
pub fn conditional_entropy(joint: &[Vec<f64>]) -> f64 {
    conditional_entropy_real(joint)
}

/// **Conditional entropy** over any [`Real`] carrier.
pub fn conditional_entropy_real<R: Real>(joint: &[Vec<R>]) -> R {
    if joint.is_empty() {
        return R::zero();
    }
    let (px, _) = marginals_real(joint);
    (joint_entropy_real(joint) - shannon_entropy_real(&px)).max(R::zero())
}

/// The **variation of information** `VI(X, Y) = H(X|Y) + H(Y|X) = 2·H(X,Y) − H(X) − H(Y)`
/// (in nats) — a true **metric** on random variables / clusterings: symmetric, `0` iff the
/// two label the same partition, and the standard distance for *comparing clusterings*. Equal
/// to `H(X,Y) − I(X;Y)`. Nonnegative because both conditional entropies are (`riemann`:
/// `entropy_fst_le_joint` + `entropy_snd_le_joint`). Returns `0` for an empty table.
pub fn variation_of_information(joint: &[Vec<f64>]) -> f64 {
    variation_of_information_real(joint)
}

/// **Variation of information** over any [`Real`] carrier.
pub fn variation_of_information_real<R: Real>(joint: &[Vec<R>]) -> R {
    if joint.is_empty() {
        return R::zero();
    }
    let (px, py) = marginals_real(joint);
    (R::two() * joint_entropy_real(joint) - shannon_entropy_real(&px) - shannon_entropy_real(&py))
        .max(R::zero())
}

/// **Normalized mutual information** `NMI(X, Y) = I(X;Y) / min(H(X), H(Y))`, in `[0, 1]` — a
/// scale-free agreement score: `1` when one variable perfectly determines the other, `0` when
/// they are independent. The `min`-normalization is the one whose `≤ 1` bound is exactly
/// `riemann`'s `mutualInfo_le_entropy_fst` / `_snd` (`I ≤ min(H(X), H(Y))`). Returns `0` when
/// either marginal is (essentially) deterministic — `min(H(X), H(Y)) = 0`, so there is no
/// shared information to normalize (then `I = 0` too).
pub fn normalized_mutual_information(joint: &[Vec<f64>]) -> f64 {
    normalized_mutual_information_real(joint)
}

/// **Normalized mutual information** over any [`Real`] carrier.
pub fn normalized_mutual_information_real<R: Real>(joint: &[Vec<R>]) -> R {
    if joint.is_empty() {
        return R::zero();
    }
    let (px, py) = marginals_real(joint);
    let denom = shannon_entropy_real(&px).min(shannon_entropy_real(&py));
    let small = denom.le(&R::scalar(1e-12));
    let safe_denom = R::select(&small, R::one(), denom.clone());
    let value = (mutual_information_real(joint) / safe_denom)
        .max(R::zero())
        .min(R::one());
    R::select(&small, R::zero(), value)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn joint_entropy_matches_the_information_diagram() {
        // Independent fair bits: J = ¼ everywhere ⇒ H(X,Y) = ln 4, I(X;Y) = 0.
        let indep = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
        close(joint_entropy(&indep), 4.0_f64.ln(), 1e-12);
        close(mutual_information(&indep), 0.0, 1e-12);
        // Perfectly dependent: J on the diagonal ⇒ H(X,Y) = ln 2 = I(X;Y) (= H(X) = H(Y)).
        let dep = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        close(joint_entropy(&dep), 2.0_f64.ln(), 1e-12);
        // Mutual information never exceeds the joint entropy (I ≤ H(X,Y)).
        for j in [&indep, &dep] {
            assert!(mutual_information(j) <= joint_entropy(j) + 1e-12);
        }
        assert_eq!(joint_entropy(&[]), 0.0);
    }

    #[test]
    fn conditional_entropy_is_the_chain_rule_residual() {
        // Independent fair bits: H(Y|X) = H(Y) = ln 2 (knowing X tells nothing about Y).
        let indep = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
        close(conditional_entropy(&indep), 2.0_f64.ln(), 1e-12);
        // Perfectly dependent (diagonal): H(Y|X) = 0 (X determines Y).
        let dep = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        close(conditional_entropy(&dep), 0.0, 1e-12);
        // Chain rule: H(Y|X) = H(X,Y) − H(X), and the riemann floor H(Y|X) ≥ 0 holds.
        let skew = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let (px, _) = marginals(&skew);
        close(
            conditional_entropy(&skew),
            joint_entropy(&skew) - shannon_entropy(&px),
            1e-12,
        );
        assert!(conditional_entropy(&skew) >= 0.0);
        assert_eq!(conditional_entropy(&[]), 0.0);
    }

    #[test]
    fn variation_of_information_is_a_nonneg_metric() {
        let indep = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
        let dep = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        // VI = H(X,Y) − I(X;Y), and VI = 0 iff the partitions coincide.
        for j in [&indep, &dep] {
            close(
                variation_of_information(j),
                joint_entropy(j) - mutual_information(j),
                1e-12,
            );
            assert!(variation_of_information(j) >= 0.0);
        }
        // Identical labelings ⇒ VI = 0; independent fair bits ⇒ VI = 2·ln 2 (maximal here).
        close(variation_of_information(&dep), 0.0, 1e-12);
        close(variation_of_information(&indep), 2.0 * 2.0_f64.ln(), 1e-12);
    }

    #[test]
    fn normalized_mutual_information_is_in_unit_interval() {
        let indep = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
        let dep = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        // Independent ⇒ 0; a perfect (deterministic) coupling ⇒ 1.
        close(normalized_mutual_information(&indep), 0.0, 1e-12);
        close(normalized_mutual_information(&dep), 1.0, 1e-12);
        // Always in [0, 1] (I ≤ min(H(X), H(Y)) — mutualInfo_le_entropy_fst/_snd).
        let skew = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let nmi = normalized_mutual_information(&skew);
        assert!((0.0..=1.0).contains(&nmi));
        // A deterministic marginal (single nonzero row) ⇒ no normalizable shared info ⇒ 0.
        let degenerate = vec![vec![0.5, 0.5]];
        assert_eq!(normalized_mutual_information(&degenerate), 0.0);
        assert_eq!(normalized_mutual_information(&[]), 0.0);
    }
}
