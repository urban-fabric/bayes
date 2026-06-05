//! Random walks and the Wiener process (Brownian motion) — the simplest
//! continuous-state stochastic processes, both built from Gaussian increments.

use rand::Rng;

use crate::distribution::normal::standard_normal_polar;

/// A Gaussian random walk: `xₜ₊₁ = xₜ + N(0, step_std²)`, starting at `start`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianRandomWalk {
    pub start: f64,
    pub step_std: f64,
}

impl GaussianRandomWalk {
    pub fn new(start: f64, step_std: f64) -> Self {
        Self { start, step_std }
    }

    /// A trajectory of `steps` increments — a vector of length `steps + 1`, the
    /// first element being `start`.
    pub fn trajectory<R: Rng + ?Sized>(&self, steps: usize, rng: &mut R) -> Vec<f64> {
        let mut path = Vec::with_capacity(steps + 1);
        let mut x = self.start;
        path.push(x);
        for _ in 0..steps {
            x += self.step_std * standard_normal_polar(rng);
            path.push(x);
        }
        path
    }
}

/// Sample a **Wiener process** (Brownian motion) path with diffusion `σ` at the
/// increasing `times`: `W(0) = 0` and `W(t) − W(s) ~ N(0, σ²(t − s))`. Returns one
/// value per time in `times` (which should be non-decreasing and start at ≥ 0).
pub fn wiener_path<R: Rng + ?Sized>(times: &[f64], diffusion: f64, rng: &mut R) -> Vec<f64> {
    let mut out = Vec::with_capacity(times.len());
    let mut w = 0.0;
    let mut prev = 0.0;
    for &t in times {
        let dt = (t - prev).max(0.0);
        w += diffusion * dt.sqrt() * standard_normal_polar(rng);
        out.push(w);
        prev = t;
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::statistics::SampleStats;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn trajectory_starts_at_start_and_has_right_length() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let w = GaussianRandomWalk::new(5.0, 1.0);
        let path = w.trajectory(10, &mut rng);
        assert_eq!(path.len(), 11);
        assert_eq!(path[0], 5.0);
    }

    #[test]
    fn wiener_variance_grows_linearly_in_time() {
        // Var[W(t)] = σ²·t. Estimate at t = 4 with σ = 1 across many paths.
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let times = [4.0];
        let ends: Vec<f64> = (0..20_000)
            .map(|_| wiener_path(&times, 1.0, &mut rng)[0])
            .collect();
        let var = ends.variance();
        assert!((var - 4.0).abs() < 0.2, "Var[W(4)] ≈ 4, got {var}");
    }
}
