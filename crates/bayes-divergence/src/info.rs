//! **Information measures over probability vectors** — Shannon entropy,
//! cross-entropy, Jensen–Shannon divergence, and the Bhattacharyya coefficient,
//! computed directly on `&[f64]` probability vectors (an empirical histogram, a
//! computed marginal, any discrete law) rather than a typed distribution.
//!
//! These complement the [`f_divergence`](super::f_divergence) umbrella: entropy
//! is not itself an f-divergence, but cross-entropy and Jensen–Shannon are built
//! from KL, and the Bhattacharyya coefficient is the affinity behind the squared
//! Hellinger distance. All in **nats** (natural log), matching the rest of the
//! divergence layer.
//!
//! Every guarantee is machine-checked in `riemann`
//! (`RiemannLean/Bayes/Divergence.lean`):
//! - `shannon_entropy` — `entropy_nonneg` (`H ≥ 0`) and **maximum entropy**
//!   `entropy_le_log_card` (`H(p) ≤ ln n`, the uniform is maximal).
//! - `max_entropy` / `normalized_entropy` / `is_entropy_admissible` — the `ln n`
//!   entropy ceiling, the `H/ln n ∈ [0, 1]` equitability index, and the admissibility
//!   certifier, all backed by `entropy_le_log_card` (+ `entropy_nonneg`).
//! - `cross_entropy` — `crossEntropy_eq_entropy_add_klDiv`
//!   (`H(p, q) = H(p) + KL(p ‖ q)`, so `H(p, q) ≥ H(p)` by Gibbs).
//! - `jensen_shannon` — `jsDiv_nonneg`, `jsDiv_symm`, and the bound
//!   `jsDiv_le_log2` (`0 ≤ JS ≤ ln 2`).
//! - `bhattacharyya_coefficient` — `bhattacharyya_nonneg`, `bhattacharyya_le_one`
//!   (`0 ≤ BC ≤ 1`), and `hellingerSq_eq_one_sub_bhattacharyya` (`H² = 1 − BC`).

#[cfg(test)]
use super::f_divergence;
use super::{Kl, f_divergence_real};
use euclid::Real;

/// **Shannon entropy** `H(p) = −Σ pᵢ ln pᵢ`, in nats (with the convention
/// `0 ln 0 = 0`). Nonnegative, and at most `ln n` over `n` categories — maximal
/// at the uniform distribution (`riemann`: `entropy_nonneg`, `entropy_le_log_card`).
///
/// ```
/// use bayes_divergence::shannon_entropy;
/// let h = shannon_entropy(&[0.25, 0.25, 0.25, 0.25]);
/// assert!((h - 4.0_f64.ln()).abs() < 1e-12);   // uniform over 4 ⇒ ln 4 (maximal)
/// assert_eq!(shannon_entropy(&[1.0, 0.0, 0.0]), 0.0);   // a point mass ⇒ 0
/// ```
pub fn shannon_entropy(p: &[f64]) -> f64 {
    shannon_entropy_real(p)
}

/// **Shannon entropy** over any [`Real`] carrier.
///
/// This is the numeric reduction only: validation of probability-vector shape and finiteness stays
/// in the eager host layer.
pub fn shannon_entropy_real<R: Real>(p: &[R]) -> R {
    -p.iter()
        .cloned()
        .map(|pi| pi.clone().xlogy(pi))
        .fold(R::zero(), |sum, term| sum + term)
}

/// **Maximum entropy** over `support_size` outcomes: `ln(n)`, the entropy of the
/// uniform distribution and the ceiling *every* distribution on `n` categories obeys
/// (`riemann`: `entropy_le_log_card`, `H(p) ≤ ln n`, the uniform maximal). Returns `0`
/// for `n ≤ 1` (a certain outcome carries no uncertainty).
///
/// ```
/// use bayes_divergence::{max_entropy, shannon_entropy};
/// assert!((max_entropy(4) - 4.0_f64.ln()).abs() < 1e-12);
/// // the uniform attains the ceiling; everything else is below it.
/// assert!(shannon_entropy(&[0.7, 0.2, 0.1]) <= max_entropy(3) + 1e-12);
/// ```
pub fn max_entropy(support_size: usize) -> f64 {
    max_entropy_real(support_size)
}

/// **Maximum entropy** over `support_size` outcomes in any [`Real`] carrier.
pub fn max_entropy_real<R: Real>(support_size: usize) -> R {
    if support_size <= 1 {
        return R::zero();
    }
    R::scalar(support_size as f64).ln()
}

/// **Normalized entropy** (the Shannon *equitability* / efficiency index):
/// `H(p) / ln(n)` over `n = p.len()` categories, in `[0, 1]`. It is `1` exactly at the
/// uniform distribution and `0` at a point mass — a scale-free measure of how *evenly*
/// mass is spread. Lands in `[0, 1]` by `riemann`'s `entropy_nonneg` (`H ≥ 0`) and
/// `entropy_le_log_card` (`H ≤ ln n`). Returns `0` for fewer than two categories (no
/// spread is possible) and `NaN` for an empty slice.
///
/// ```
/// use bayes_divergence::normalized_entropy;
/// assert!((normalized_entropy(&[0.25; 4]) - 1.0).abs() < 1e-12); // uniform ⇒ 1
/// assert_eq!(normalized_entropy(&[1.0, 0.0, 0.0]), 0.0);         // point mass ⇒ 0
/// ```
pub fn normalized_entropy(p: &[f64]) -> f64 {
    normalized_entropy_real(p)
}

/// **Normalized entropy** over any [`Real`] carrier.
pub fn normalized_entropy_real<R: Real>(p: &[R]) -> R {
    if p.is_empty() {
        return R::nan();
    }
    if p.len() <= 1 {
        return R::zero();
    }
    shannon_entropy_real(p) / max_entropy_real(p.len())
}

/// **Maximum-entropy admissibility check**: whether a computed entropy respects the
/// ceiling `H ≤ ln n` for `support_size` outcomes (`riemann`: `entropy_le_log_card`),
/// and is nonnegative (`entropy_nonneg`). A `false` flags an entropy above what *any*
/// distribution on `n` categories can reach — impossible, so it signals a miscounted
/// support or a units mismatch (e.g. bits read as nats). The information-theoretic
/// mirror of the variance cap `bayes_statistics::is_variance_admissible`.
/// A small tolerance absorbs floating-point slack at the uniform boundary.
pub fn is_entropy_admissible(entropy: f64, support_size: usize) -> bool {
    entropy >= 0.0 && entropy <= max_entropy(support_size) + 1e-9
}

/// **Cross-entropy** `H(p, q) = −Σ pᵢ ln qᵢ`, in nats — the expected surprise of
/// coding `p`-distributed data under model `q`. Equal to `H(p) + KL(p ‖ q)`
/// (`riemann`: `crossEntropy_eq_entropy_add_klDiv`), so it is never below the
/// entropy `H(p)`, with equality iff `p = q` (Gibbs). `+∞` when `q` puts zero
/// mass where `p` does not (the code cannot describe an outcome it deems
/// impossible).
///
/// ```
/// use bayes_divergence::{cross_entropy, shannon_entropy, f_divergence, Kl};
/// let p = [0.5, 0.5];
/// let q = [0.25, 0.75];
/// let lhs = cross_entropy(&p, &q);
/// let rhs = shannon_entropy(&p) + f_divergence(&p, &q, &Kl);
/// assert!((lhs - rhs).abs() < 1e-12);   // H(p,q) = H(p) + KL(p‖q)
/// ```
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    cross_entropy_real(p, q)
}

/// **Cross-entropy** over any [`Real`] carrier.
///
/// The shorter vector is zero-padded, matching [`cross_entropy`].
pub fn cross_entropy_real<R: Real>(p: &[R], q: &[R]) -> R {
    let n = p.len().max(q.len());
    (0..n)
        .map(|i| {
            let pi = p.get(i).cloned().unwrap_or_else(R::zero);
            let qi = q.get(i).cloned().unwrap_or_else(R::zero);
            -(pi.xlogy(qi))
        })
        .fold(R::zero(), |sum, term| sum + term)
}

/// The **Jensen–Shannon divergence** `JS(p, q) = ½ KL(p ‖ m) + ½ KL(q ‖ m)` with
/// the mixture `m = (p + q)/2`, in nats — the symmetric, bounded smoothing of KL.
/// Unlike KL it is symmetric, always finite, and lies in `[0, ln 2]`; its square
/// root is a metric (`riemann`: `jsDiv_nonneg`, `jsDiv_symm`, `jsDiv_le_log2`).
///
/// Built on the [`f_divergence`](super::f_divergence) umbrella with the [`Kl`]
/// generator against the mixture.
///
/// ```
/// use bayes_divergence::jensen_shannon;
/// // Disjoint supports saturate the upper bound ln 2.
/// let js = jensen_shannon(&[1.0, 0.0], &[0.0, 1.0]);
/// assert!((js - 2.0_f64.ln()).abs() < 1e-12);
/// assert_eq!(jensen_shannon(&[0.3, 0.7], &[0.3, 0.7]), 0.0);   // self ⇒ 0
/// ```
pub fn jensen_shannon(p: &[f64], q: &[f64]) -> f64 {
    jensen_shannon_real(p, q)
}

/// The **Jensen-Shannon divergence** over any [`Real`] carrier.
pub fn jensen_shannon_real<R: Real>(p: &[R], q: &[R]) -> R {
    let n = p.len().max(q.len());
    let m: Vec<R> = (0..n)
        .map(|i| {
            R::half()
                * (p.get(i).cloned().unwrap_or_else(R::zero)
                    + q.get(i).cloned().unwrap_or_else(R::zero))
        })
        .collect();
    R::half() * f_divergence_real(p, &m, &Kl) + R::half() * f_divergence_real(q, &m, &Kl)
}

/// The **Bhattacharyya coefficient** `BC(p, q) = Σ √(pᵢ qᵢ)` — the affinity
/// (overlap) between two discrete distributions, in `[0, 1]`: `1` iff `p = q`,
/// `0` for disjoint supports (`riemann`: `bhattacharyya_nonneg`,
/// `bhattacharyya_le_one`). The squared Hellinger distance is `1 − BC`
/// (`hellingerSq_eq_one_sub_bhattacharyya`), so the [`SquaredHellinger`] generator
/// and this affinity are two views of the same quantity.
///
/// [`SquaredHellinger`]: super::SquaredHellinger
///
/// ```
/// use bayes_divergence::{bhattacharyya_coefficient, f_divergence, SquaredHellinger};
/// let p = [0.5, 0.5];
/// let q = [0.25, 0.75];
/// let bc = bhattacharyya_coefficient(&p, &q);
/// let h2 = f_divergence(&p, &q, &SquaredHellinger);
/// assert!((h2 - (1.0 - bc)).abs() < 1e-12);   // H² = 1 − BC
/// ```
pub fn bhattacharyya_coefficient(p: &[f64], q: &[f64]) -> f64 {
    bhattacharyya_coefficient_real(p, q)
}

/// The **Bhattacharyya coefficient** over any [`Real`] carrier.
pub fn bhattacharyya_coefficient_real<R: Real>(p: &[R], q: &[R]) -> R {
    let n = p.len().max(q.len());
    (0..n)
        .map(|i| {
            let pi = p.get(i).cloned().unwrap_or_else(R::zero);
            let qi = q.get(i).cloned().unwrap_or_else(R::zero);
            (pi * qi).max(R::zero()).sqrt()
        })
        .fold(R::zero(), |sum, term| sum + term)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::super::{JensenShannon, SquaredHellinger};
    use super::*;
    use crate::{Bernoulli, Categorical};
    use std::f64::consts::LN_2;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    fn pairs() -> Vec<(Vec<f64>, Vec<f64>)> {
        vec![
            (vec![0.5, 0.5], vec![0.25, 0.75]),
            (vec![0.2, 0.3, 0.5], vec![0.5, 0.1, 0.4]),
            (vec![0.7, 0.2, 0.1], vec![0.34, 0.33, 0.33]),
        ]
    }

    #[test]
    fn entropy_is_maximal_at_the_uniform_and_zero_at_a_point_mass() {
        for n in [2usize, 3, 5, 8] {
            let u = vec![1.0 / n as f64; n];
            close(shannon_entropy(&u), (n as f64).ln(), 1e-12);
            // Any distribution over n categories is ≤ ln n (entropy_le_log_card).
            let skew = {
                let mut v = vec![0.1 / (n - 1) as f64; n];
                v[0] = 0.9;
                v
            };
            assert!(shannon_entropy(&skew) <= (n as f64).ln() + 1e-12);
        }
        let mut point = vec![0.0; 4];
        point[2] = 1.0;
        close(shannon_entropy(&point), 0.0, 1e-12);
    }

    #[test]
    fn max_entropy_is_the_uniform_ceiling() {
        for n in [2usize, 3, 5, 8] {
            let u = vec![1.0 / n as f64; n];
            // the uniform attains ln n exactly.
            close(shannon_entropy(&u), max_entropy(n), 1e-12);
            close(max_entropy(n), (n as f64).ln(), 1e-12);
        }
        // a certain / single outcome carries no uncertainty.
        assert_eq!(max_entropy(1), 0.0);
        assert_eq!(max_entropy(0), 0.0);
    }

    #[test]
    fn normalized_entropy_is_one_at_uniform_and_zero_at_a_point_mass() {
        for n in [2usize, 4, 7] {
            close(normalized_entropy(&vec![1.0 / n as f64; n]), 1.0, 1e-12);
        }
        // point mass ⇒ 0; intermediate spread ⇒ strictly inside (0, 1).
        assert_eq!(normalized_entropy(&[1.0, 0.0, 0.0]), 0.0);
        let h = normalized_entropy(&[0.7, 0.2, 0.1]);
        assert!((0.0..1.0).contains(&h));
        // degenerate inputs.
        assert!(normalized_entropy(&[]).is_nan());
        assert_eq!(normalized_entropy(&[1.0]), 0.0);
    }

    #[test]
    fn entropy_admissibility_flags_the_impossible() {
        // ln 3 is attainable on 3 categories; anything above is not.
        assert!(is_entropy_admissible(3.0_f64.ln(), 3));
        assert!(is_entropy_admissible(0.5, 3));
        assert!(!is_entropy_admissible(3.0_f64.ln() + 0.1, 3));
        // a point-mass entropy (0) is admissible even for a single outcome.
        assert!(is_entropy_admissible(0.0, 1));
        // a negative entropy is never valid; bits misread as nats (ln→log2) overflows.
        assert!(!is_entropy_admissible(-0.01, 3));
        assert!(!is_entropy_admissible((3.0_f64).log2(), 3)); // log2 3 ≈ 1.585 > ln 3 ≈ 1.099
    }

    #[test]
    fn cross_entropy_decomposes_into_entropy_plus_kl() {
        // crossEntropy_eq_entropy_add_klDiv, and H(p,p) = H(p).
        for (p, q) in pairs() {
            close(
                cross_entropy(&p, &q),
                shannon_entropy(&p) + f_divergence(&p, &q, &Kl),
                1e-12,
            );
            close(cross_entropy(&p, &p), shannon_entropy(&p), 1e-12);
            // Gibbs consequence: cross-entropy never below the entropy.
            assert!(cross_entropy(&p, &q) >= shannon_entropy(&p) - 1e-12);
        }
        // q impossible where p is possible ⇒ +∞.
        assert!(cross_entropy(&[0.5, 0.5], &[1.0, 0.0]).is_infinite());
    }

    #[test]
    fn jensen_shannon_is_symmetric_bounded_and_matches_the_per_family_form() {
        for (p, q) in pairs() {
            let js = jensen_shannon(&p, &q);
            close(js, jensen_shannon(&q, &p), 1e-12); // symmetric
            assert!((0.0..=LN_2 + 1e-12).contains(&js)); // 0 ≤ JS ≤ ln 2
            // Agrees with the per-family `JensenShannon` trait on Categorical.
            let cat = Categorical::new(&p)
                .unwrap()
                .js_divergence(&Categorical::new(&q).unwrap());
            close(js, cat, 1e-12);
        }
        // Self ⇒ 0; disjoint supports ⇒ ln 2.
        close(jensen_shannon(&[0.3, 0.7], &[0.3, 0.7]), 0.0, 1e-12);
        close(jensen_shannon(&[1.0, 0.0], &[0.0, 1.0]), LN_2, 1e-12);
        // Two-category vector agrees with Bernoulli (category 0 = "success").
        close(
            jensen_shannon(&[0.7, 0.3], &[0.4, 0.6]),
            Bernoulli::new(0.7).js_divergence(&Bernoulli::new(0.4)),
            1e-12,
        );
    }

    #[test]
    fn bhattacharyya_is_in_unit_interval_and_complements_hellinger() {
        for (p, q) in pairs() {
            let bc = bhattacharyya_coefficient(&p, &q);
            assert!((0.0..=1.0 + 1e-12).contains(&bc));
            // H² = 1 − BC (hellingerSq_eq_one_sub_bhattacharyya).
            close(f_divergence(&p, &q, &SquaredHellinger), 1.0 - bc, 1e-12);
        }
        close(
            bhattacharyya_coefficient(&[0.3, 0.7], &[0.3, 0.7]),
            1.0,
            1e-12,
        ); // self ⇒ 1
        close(
            bhattacharyya_coefficient(&[1.0, 0.0], &[0.0, 1.0]),
            0.0,
            1e-12,
        ); // disjoint ⇒ 0
    }
}
