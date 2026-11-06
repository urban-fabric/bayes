//! Gaussian-process kernels and the GP marginal-likelihood objective.
//!
//! The [`Kernel`] contract defines a covariance as a function of a (pre-computed) separation and
//! a flat hyperparameter vector `θ`, **generic over the backend scalar** `T: Real`. The same
//! kernel therefore lowers two ways from one definition: applied to a whole distance *matrix*
//! tensor it builds the covariance in one elementwise op (so it compiles to XLA / runs on
//! `maxwell`); applied per pair to `f64` it is the eager reference. Kernels delegate their closed
//! forms to `bayes_kernels` correlation functions and compose through [`Sum`] / [`Product`].
//!
//! These kernels are **numeric-only** (scalar, `T: Real`): one definition lowers
//! elementwise onto a distance *matrix* tensor (so the covariance assembly compiles to
//! XLA on `maxwell`) and per-pair onto `f64` (the eager reference). The matrix-coupled GP
//! marginal-likelihood objective that consumes these covariances lives in the backend
//! (`maxwell::dist` / `markov`), where the tensor is concrete.

use euclid_numeric::Real;

use bayes_kernels as kernels;

/// A covariance kernel `k(d; θ)`: covariance from a separation `distance` and a flat
/// hyperparameter vector `θ` (in declaration order). Generic over `T: Real`, so one
/// definition lowers elementwise on a distance tensor (XLA) and per pair on `f64` (eager).
///
/// Not object-safe (the `covariance` method is generic); kernels compose by *type*
/// ([`Sum`] / [`Product`]) and the GP code is monomorphized per backend.
pub trait Kernel {
    /// How many hyperparameters this kernel consumes from `θ`.
    fn hyper_count(&self) -> usize;
    /// Covariance `k(distance; θ)`.
    fn covariance<T: Real>(&self, distance: T, theta: &[T]) -> T;
}

/// Squared-exponential (RBF) kernel; `θ = [length_scale, variance]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rbf;
impl Kernel for Rbf {
    fn hyper_count(&self) -> usize {
        2
    }
    fn covariance<T: Real>(&self, distance: T, theta: &[T]) -> T {
        theta[1].clone() * kernels::squared_exponential_correlation(distance, theta[0].clone())
    }
}

/// Exponential (Matérn ν = ½) kernel; `θ = [length_scale, variance]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exponential;
impl Kernel for Exponential {
    fn hyper_count(&self) -> usize {
        2
    }
    fn covariance<T: Real>(&self, distance: T, theta: &[T]) -> T {
        theta[1].clone() * kernels::exponential_correlation(distance, theta[0].clone())
    }
}

/// Matérn-5/2 kernel; `θ = [length_scale, variance]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Matern52;
impl Kernel for Matern52 {
    fn hyper_count(&self) -> usize {
        2
    }
    fn covariance<T: Real>(&self, distance: T, theta: &[T]) -> T {
        theta[1].clone() * kernels::matern_52_correlation(distance, theta[0].clone())
    }
}

/// Matérn-3/2 kernel; `θ = [length_scale, variance]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Matern32;
impl Kernel for Matern32 {
    fn hyper_count(&self) -> usize {
        2
    }
    fn covariance<T: Real>(&self, distance: T, theta: &[T]) -> T {
        theta[1].clone() * kernels::matern_32_correlation(distance, theta[0].clone())
    }
}

/// Sum of two kernels — `θ` is the children's `θ` concatenated.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sum<A, B>(pub A, pub B);
impl<A: Kernel, B: Kernel> Kernel for Sum<A, B> {
    fn hyper_count(&self) -> usize {
        self.0.hyper_count() + self.1.hyper_count()
    }
    fn covariance<T: Real>(&self, distance: T, theta: &[T]) -> T {
        let (a, b) = theta.split_at(self.0.hyper_count());
        self.0.covariance(distance.clone(), a) + self.1.covariance(distance, b)
    }
}

/// Product of two kernels — `θ` is the children's `θ` concatenated.
#[derive(Clone, Copy, Debug, Default)]
pub struct Product<A, B>(pub A, pub B);
impl<A: Kernel, B: Kernel> Kernel for Product<A, B> {
    fn hyper_count(&self) -> usize {
        self.0.hyper_count() + self.1.hyper_count()
    }
    fn covariance<T: Real>(&self, distance: T, theta: &[T]) -> T {
        let (a, b) = theta.split_at(self.0.hyper_count());
        self.0.covariance(distance.clone(), a) * self.1.covariance(distance, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_covariance_scales_correlation_and_composes() {
        let close = |a: f64, b: f64| assert!((a - b).abs() < 1e-12, "{a} != {b}");
        let (d, ls, var) = (100.0_f64, 1000.0_f64, 2.0_f64);
        // Matérn-5/2 covariance = variance · correlation(d, length_scale).
        close(
            Matern52.covariance::<f64>(d, &[ls, var]),
            var * kernels::matern_52_correlation(d, ls),
        );
        // Sum concatenates θ and adds the children.
        let k = Sum(Matern52, Rbf);
        assert_eq!(k.hyper_count(), 4);
        let theta = [1000.0, 2.0, 800.0, 1.5];
        close(
            k.covariance::<f64>(d, &theta),
            Matern52.covariance::<f64>(d, &theta[0..2]) + Rbf.covariance::<f64>(d, &theta[2..4]),
        );
    }
}
