//! The Halton sequence: the radical inverse in a different prime base per
//! dimension.

use bayes_core::BayesError;

/// The first primes, one per dimension.
const PRIMES: [u32; 32] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

/// A Halton sequence in `dimension` dimensions.
///
/// The construction is one idea applied per axis: write the index `n` in base
/// `b`, then reflect its digits about the radix point. Successive values in base
/// 2 go `1/2, 1/4, 3/4, 1/8, 5/8, …` — each new point lands in the largest
/// remaining gap, which is exactly the property that beats clumping.
///
/// **The known weakness.** Using a different prime per dimension means the
/// higher dimensions cycle slowly against each other, and pairs of
/// high-dimension coordinates show visible linear structure for the first few
/// hundred points. The standard fix, applied here, is to **skip** the first
/// `burn` points: discarding 64 or so removes the worst of it at negligible
/// cost. Beyond about 10 dimensions prefer [`super::Sobol`] regardless.
#[derive(Clone, Debug)]
pub struct Halton {
    bases: Vec<u32>,
    index: u64,
}

impl Halton {
    /// A sequence in `dimension` dimensions, starting after `burn` discarded
    /// points.
    ///
    /// `burn = 64` is a reasonable default; `burn = 0` gives the raw sequence,
    /// whose first point is the origin-adjacent `(1/2, 1/3, 1/5, …)`.
    pub fn new(dimension: usize, burn: u64) -> Result<Self, BayesError> {
        if dimension == 0 {
            return Err(BayesError::Malformed {
                what: "a Halton sequence needs at least one dimension",
            });
        }
        if dimension > PRIMES.len() {
            return Err(BayesError::Malformed {
                what: "Halton supports up to 32 dimensions with the tabulated primes",
            });
        }
        Ok(Self {
            bases: PRIMES[..dimension].to_vec(),
            // The sequence is conventionally 1-indexed: `vdc(0)` is 0, and a
            // point at the origin is useless for an inverse transform.
            index: burn.saturating_add(1),
        })
    }

    /// The number of dimensions.
    pub fn dimension(&self) -> usize {
        self.bases.len()
    }

    /// The next point in `[0, 1)^d`.
    pub fn next_point(&mut self) -> Vec<f64> {
        let p = self
            .bases
            .iter()
            .map(|&b| radical_inverse(self.index, b))
            .collect();
        // Saturating, because `+= 1` is a debug-mode panic at the top of the
        // range: `Halton::new(3, u64::MAX - 1)` constructs happily and then
        // aborts on the second point. The sequence has `1.8e19` points and no
        // computation will exhaust them, so saturating here is a formality —
        // but panicking on a `u64` a caller was allowed to pass is not.
        self.index = self.index.saturating_add(1);
        p
    }

    /// The next `count` points.
    pub fn points(&mut self, count: usize) -> Vec<Vec<f64>> {
        (0..count).map(|_| self.next_point()).collect()
    }

    /// The `n`-th point without advancing the sequence, so callers can split by
    /// index across threads or processes with no shared state.
    pub fn point_at(&self, n: u64) -> Vec<f64> {
        self.bases
            .iter()
            .map(|&b| radical_inverse(n + 1, b))
            .collect()
    }
}

/// The van der Corput radical inverse of `n` in base `b`.
///
/// Accumulated as a sum of `digit / b^k` rather than by building the fraction
/// and dividing once: the latter overflows the mantissa for large `n`, and the
/// sum is exact for every digit that matters.
fn radical_inverse(mut n: u64, base: u32) -> f64 {
    let b = base as f64;
    let mut result = 0.0;
    let mut denom = b;
    while n > 0 {
        let digit = (n % base as u64) as f64;
        result += digit / denom;
        n /= base as u64;
        denom *= b;
    }
    // The radical inverse is strictly below 1 for every `n`, but the sum need
    // not be: in base 2 an index with all 64 bits set gives `1 − 2^-64`, and
    // `2^-64` is far below the ulp of 1, so it rounds *up* to exactly `1.0` and
    // breaks the `[0, 1)` range this type documents. Rounding down to the
    // largest `f64` below 1 restores the strict inequality the mathematics has.
    //
    // It matters because the point of these sequences is to drive an inverse
    // transform, and `Q(1)` is the distribution's upper support bound — `+inf`
    // for a normal.
    result.min(1.0 - f64::EPSILON / 2.0)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// The base-2 radical inverse has a closed form for the first few indices,
    /// which pins the digit-reversal convention.
    #[test]
    fn the_base_two_sequence_bisects_the_interval() {
        assert!((radical_inverse(1, 2) - 0.5).abs() < 1e-15);
        assert!((radical_inverse(2, 2) - 0.25).abs() < 1e-15);
        assert!((radical_inverse(3, 2) - 0.75).abs() < 1e-15);
        assert!((radical_inverse(4, 2) - 0.125).abs() < 1e-15);
        assert!((radical_inverse(5, 2) - 0.625).abs() < 1e-15);
        // Base 3: 1/3, 2/3, 1/9, 4/9, 7/9.
        assert!((radical_inverse(1, 3) - 1.0 / 3.0).abs() < 1e-15);
        assert!((radical_inverse(2, 3) - 2.0 / 3.0).abs() < 1e-15);
        assert!((radical_inverse(3, 3) - 1.0 / 9.0).abs() < 1e-15);
    }

    #[test]
    fn points_stay_in_the_unit_cube() {
        let mut h = Halton::new(5, 0).unwrap();
        for _ in 0..10_000 {
            for v in h.next_point() {
                assert!((0.0..1.0).contains(&v), "{v} left [0,1)");
            }
        }
    }

    /// Indexed access must agree with sequential iteration, which is what makes
    /// splitting the sequence across workers safe.
    #[test]
    fn indexed_access_matches_sequential_iteration() {
        let mut seq = Halton::new(3, 0).unwrap();
        let indexed = Halton::new(3, 0).unwrap();
        for n in 0..500 {
            assert_eq!(seq.next_point(), indexed.point_at(n), "mismatch at {n}");
        }
    }

    #[test]
    fn invalid_dimensions_are_rejected() {
        assert!(Halton::new(0, 0).is_err());
        assert!(Halton::new(33, 0).is_err());
        assert!(Halton::new(32, 0).is_ok());
    }
}
