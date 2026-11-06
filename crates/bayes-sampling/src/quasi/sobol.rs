//! The Sobol sequence: a base-2 digital net driven by direction numbers.

use bayes_core::BayesError;

/// Primitive polynomial degrees and initial direction numbers for the first
/// dimensions, from Joe & Kuo's tables.
///
/// Each entry is `(degree, polynomial coefficients packed as bits, initial m
/// values)`. Only the first few dimensions are tabulated here; that bounds
/// [`Sobol::new`] rather than silently degrading, because a Sobol sequence built
/// from wrong direction numbers is not low-discrepancy and nothing about its
/// output would look wrong.
const DIRECTIONS: &[(u32, u32, &[u32])] = &[
    // Dimension 1 is the plain van der Corput sequence: no polynomial needed.
    (0, 0, &[1]),
    (1, 0, &[1]),
    (2, 1, &[1, 3]),
    (3, 1, &[1, 3, 1]),
    (3, 2, &[1, 1, 1]),
    (4, 1, &[1, 1, 3, 3]),
    (4, 4, &[1, 3, 5, 13]),
    (5, 2, &[1, 1, 5, 5, 17]),
    (5, 4, &[1, 1, 5, 5, 5]),
    (5, 7, &[1, 1, 7, 11, 19]),
    (5, 11, &[1, 1, 5, 1, 1]),
    (5, 13, &[1, 1, 1, 3, 11]),
    (5, 14, &[1, 3, 5, 5, 31]),
    (6, 1, &[1, 3, 3, 9, 7, 49]),
    (6, 13, &[1, 1, 1, 15, 21, 21]),
    (6, 16, &[1, 3, 1, 13, 27, 49]),
];

/// The number of bits of resolution: `2^32` distinct values per coordinate.
const BITS: u32 = 32;

/// A Sobol sequence in `dimension` dimensions.
///
/// Where Halton uses a different prime base per axis, Sobol stays in base 2 and
/// varies the *direction numbers* — a set of per-dimension bit patterns derived
/// from a primitive polynomial over GF(2). The `n`-th point is then a few XORs,
/// which makes it both faster than Halton and better behaved in moderate
/// dimension.
///
/// **Use counts that are powers of two.** A Sobol sequence is a digital net:
/// the first `2^k` points are exactly balanced across every dyadic box, and a
/// count that is not a power of two leaves that balance partly undone. Taking
/// 1000 points rather than 1024 measurably costs accuracy.
///
/// The first point of the raw sequence is the origin, which is a legitimate
/// member of the net but a poor first quadrature node; `burn` skips it by
/// default in the same way as [`super::Halton`].
#[derive(Clone, Debug)]
pub struct Sobol {
    /// `v[d][k]` is dimension `d`'s direction number for bit `k`, pre-scaled.
    v: Vec<Vec<u32>>,
    /// The running Gray-code state per dimension.
    x: Vec<u32>,
    index: u64,
}

impl Sobol {
    /// A sequence in `dimension` dimensions, skipping `burn` initial points.
    pub fn new(dimension: usize, burn: u64) -> Result<Self, BayesError> {
        if dimension == 0 {
            return Err(BayesError::Malformed {
                what: "a Sobol sequence needs at least one dimension",
            });
        }
        if dimension > DIRECTIONS.len() {
            return Err(BayesError::Malformed {
                what: "Sobol supports up to 16 dimensions with the tabulated direction numbers",
            });
        }
        // A 32-bit Sobol sequence addresses `2^32` points and no more: the
        // Gray code cycles after that, so index `2^32` *is* index 0. Asked for
        // a burn past the end, the closed form below would quietly return the
        // origin — the one point whose coordinates are all zero, and the worst
        // possible silent answer. Refusing says what happened.
        if burn >= 1_u64 << BITS {
            return Err(BayesError::Malformed {
                what: "Sobol addresses 2^32 points; a larger burn has no corresponding point",
            });
        }
        let mut v = Vec::with_capacity(dimension);
        for d in 0..dimension {
            v.push(direction_numbers(d));
        }
        // The burn is applied in closed form, not by iterating.
        //
        // The Gray-code recurrence `x_{n+1} = x_n ⊕ v[c(n)]` telescopes: with
        // `x_0 = 0`, the state at index `n` is the XOR of the direction numbers
        // whose bit is set in `gray(n) = n ⊕ (n >> 1)`. So reaching index `n`
        // costs 32 XORs rather than `n` of them.
        //
        // Iterating was not merely slow, it made the type's documented contract
        // false: "two processes can split the sequence by index with no
        // coordination" is only useful if arriving at an index is cheap, and
        // `Sobol::new(d, 1 << 48)` did not return. `Halton::new` has always
        // been `O(1)` here, which is what made the asymmetry visible.
        let gray = burn ^ (burn >> 1);
        let mut x = vec![0_u32; dimension];
        for (d, directions) in v.iter().enumerate() {
            let mut acc = 0_u32;
            for bit in 0..BITS {
                if gray >> bit & 1 == 1 {
                    acc ^= directions[bit as usize];
                }
            }
            x[d] = acc;
        }
        Ok(Self { v, x, index: burn })
    }

    /// The number of dimensions.
    pub fn dimension(&self) -> usize {
        self.v.len()
    }

    /// The next point in `[0, 1)^d`.
    ///
    /// Advanced by Gray code: consecutive indices differ in exactly one bit, so
    /// each step is one XOR per dimension rather than a full reconstruction.
    ///
    /// The current state is returned *before* advancing, so the first point is
    /// the **origin**. That is not an off-by-one to be tidied away: the origin is
    /// a member of the digital net, and dropping it breaks the stratification
    /// property that makes the sequence worth using — the first four points
    /// would then put two in one quarter of the interval and none in another.
    /// Skip it with `burn` if a quadrature rule needs an interior first node.
    ///
    /// Note also that Gray-code iteration visits the net in a *permuted* order:
    /// dimension one yields `0, 1/2, 3/4, 1/4, 3/8, …` rather than van der
    /// Corput's `1/2, 1/4, 3/4, …`. The balanced object is the **set** of the
    /// first `2^k` points, not the order they arrive in.
    pub fn next_point(&mut self) -> Vec<f64> {
        let point: Vec<f64> = self
            .x
            .iter()
            .map(|&xi| xi as f64 / 2.0_f64.powi(BITS as i32))
            .collect();
        // The bit that changes between index n and n+1 in Gray-code order is the
        // position of the lowest zero bit of n.
        let c = ((!self.index).trailing_zeros() as usize).min(BITS as usize - 1);
        for d in 0..self.v.len() {
            self.x[d] ^= self.v[d][c];
        }
        self.index += 1;
        point
    }

    /// The next `count` points. Prefer a power of two; see the type docs.
    pub fn points(&mut self, count: usize) -> Vec<Vec<f64>> {
        (0..count).map(|_| self.next_point()).collect()
    }
}

/// Build dimension `d`'s direction numbers, scaled so bit `k` sits at the top.
fn direction_numbers(d: usize) -> Vec<u32> {
    let (degree, poly, init) = DIRECTIONS[d];
    let mut v = vec![0_u32; BITS as usize];
    if d == 0 {
        // Dimension 1: v[k] = 2^(31-k), the van der Corput directions.
        for (k, slot) in v.iter_mut().enumerate() {
            *slot = 1 << (BITS as usize - 1 - k);
        }
        return v;
    }
    let s = degree as usize;
    for (k, slot) in v.iter_mut().enumerate().take(s.min(BITS as usize)) {
        *slot = init[k] << (BITS as usize - 1 - k);
    }
    // The recurrence from the primitive polynomial, over GF(2).
    for k in s..BITS as usize {
        v[k] = v[k - s] ^ (v[k - s] >> s);
        for j in 1..s {
            if (poly >> (s - 1 - j)) & 1 == 1 {
                v[k] ^= v[k - j];
            }
        }
    }
    v
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn points_stay_in_the_unit_cube() {
        let mut s = Sobol::new(8, 0).unwrap();
        for _ in 0..8_192 {
            for v in s.next_point() {
                assert!((0.0..1.0).contains(&v), "{v} left [0,1)");
            }
        }
    }

    /// Dimension one in **Gray-code order**, starting at the origin. This is a
    /// permutation of the van der Corput sequence, not that sequence itself —
    /// asserting the unpermuted order here was my first mistake, and it looks
    /// right until you check the stratification below.
    #[test]
    fn the_first_dimension_follows_gray_code_order() {
        let mut s = Sobol::new(1, 0).unwrap();
        let expected = [0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125];
        for (i, want) in expected.iter().enumerate() {
            let got = s.next_point()[0];
            assert!(
                (got - want).abs() < 1e-12,
                "point {i}: got {got}, want {want}"
            );
        }
        // And the set of the first eight is exactly the eighths, which is the
        // property that actually matters.
        let mut all: Vec<f64> = Sobol::new(1, 0)
            .unwrap()
            .points(8)
            .into_iter()
            .map(|p| p[0])
            .collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, v) in all.iter().enumerate() {
            assert!(
                (v - i as f64 / 8.0).abs() < 1e-12,
                "the set is not the eighths"
            );
        }
    }

    /// The defining property of a digital net: the first `2^k` points put
    /// exactly one point in each of the `2^k` equal sub-intervals, per
    /// dimension. This is what "low discrepancy" means concretely, and a wrong
    /// direction number breaks it while leaving the points looking uniform.
    #[test]
    fn each_power_of_two_block_is_perfectly_stratified() {
        for dim in 1..=6 {
            for k in 2..=10 {
                let n = 1_usize << k;
                let mut s = Sobol::new(dim, 0).unwrap();
                let pts = s.points(n);
                for d in 0..dim {
                    let mut buckets = vec![0_usize; n];
                    for p in &pts {
                        let b = ((p[d] * n as f64) as usize).min(n - 1);
                        buckets[b] += 1;
                    }
                    assert!(
                        buckets.iter().all(|&c| c == 1),
                        "dimension {d} of {dim} at n = {n} is not stratified: {:?}",
                        &buckets[..buckets.len().min(16)]
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_dimensions_are_rejected() {
        assert!(Sobol::new(0, 0).is_err());
        assert!(Sobol::new(17, 0).is_err());
        assert!(Sobol::new(16, 0).is_ok());
    }
}
