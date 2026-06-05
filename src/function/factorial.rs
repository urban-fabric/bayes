//! Combinatorics: `factorial`, `ln_factorial`, and the binomial coefficient — used
//! by the discrete distributions' PMFs (Poisson, Binomial, …).

use super::gamma::ln_gamma;

/// Exact factorials cached up to 20! (the largest that fits an `f64` exactly).
const SMALL: usize = 21;

/// `n!` as `f64`. Exact through 20!, then via `exp(ln_factorial)` (so it overflows
/// to `+∞` like the true factorial beyond ~170!).
pub fn factorial(n: u64) -> f64 {
    if (n as usize) < SMALL {
        let mut acc = 1.0_f64;
        for k in 2..=n {
            acc *= k as f64;
        }
        acc
    } else {
        ln_factorial(n).exp()
    }
}

/// `ln(n!) = ln Γ(n+1)`. Exact small sum below 21, Lanczos `ln_gamma` above.
pub fn ln_factorial(n: u64) -> f64 {
    if (n as usize) < SMALL {
        let mut s = 0.0_f64;
        for k in 2..=n {
            s += (k as f64).ln();
        }
        s
    } else {
        ln_gamma(n as f64 + 1.0)
    }
}

/// The binomial coefficient `C(n, k)` as `f64`.
pub fn binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    ln_binomial(n, k).exp().round()
}

/// `ln C(n, k)`.
pub fn ln_binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn factorials_exact() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3_628_800.0);
    }

    #[test]
    fn ln_factorial_matches() {
        for n in [0u64, 1, 5, 20, 25, 100] {
            let expect = factorial_ref(n);
            assert!((ln_factorial(n) - expect).abs() <= 1e-7 * expect.abs().max(1.0));
        }
    }

    fn factorial_ref(n: u64) -> f64 {
        (1..=n).map(|k| (k as f64).ln()).sum()
    }

    #[test]
    fn binomials() {
        assert_eq!(binomial(5, 0), 1.0);
        assert_eq!(binomial(5, 2), 10.0);
        assert_eq!(binomial(10, 5), 252.0);
        assert_eq!(binomial(3, 4), 0.0);
    }
}
