//! Gaussian-process kernels and the GP marginal-likelihood objective.
//!
//! The [`Kernel`] contract defines a covariance as a function of a (pre-computed) separation and
//! a flat hyperparameter vector `θ`, **generic over the backend scalar** `T: LazyReal`. The same
//! kernel therefore lowers two ways from one definition: applied to a whole distance *matrix*
//! tensor it builds the covariance in one elementwise op (so it compiles to XLA / runs on
//! `maxwell`); applied per pair to `f64` it is the eager reference. Kernels delegate their closed
//! forms to [`crate::lazy`]'s correlation functions and compose through [`Sum`] / [`Product`].
//!
//! [`neg_log_marginal_likelihood`] is the GP *fitting* objective: on a unified tensor backend
//! (`maxwell` / `lebesgue`) the covariance assembly *and* the marginal likelihood form one
//! differentiable graph in `θ` — markov's GP fit, compiled. The flat-`θ` shape matches the
//! optimizer's parameter vector, so the inference engine drives `θ` blind to the backend.

use euclid::array::LazyMatrix;
use euclid::numeric::LazyReal;

use crate::lazy;
use crate::lazy_matrix::gaussian_process_log_marginal_likelihood;

/// A covariance kernel `k(d; θ)`: covariance from a separation `distance` and a flat
/// hyperparameter vector `θ` (in declaration order). Generic over `T: LazyReal`, so one
/// definition lowers elementwise on a distance tensor (XLA) and per pair on `f64` (eager).
///
/// Not object-safe (the `covariance` method is generic); kernels compose by *type*
/// ([`Sum`] / [`Product`]) and the GP code is monomorphized per backend.
pub trait Kernel {
    /// How many hyperparameters this kernel consumes from `θ`.
    fn hyper_count(&self) -> usize;
    /// Covariance `k(distance; θ)`.
    fn covariance<T: LazyReal>(&self, distance: T, theta: &[T]) -> T;
}

/// Squared-exponential (RBF) kernel; `θ = [length_scale, variance]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rbf;
impl Kernel for Rbf {
    fn hyper_count(&self) -> usize {
        2
    }
    fn covariance<T: LazyReal>(&self, distance: T, theta: &[T]) -> T {
        theta[1].clone() * lazy::squared_exponential_correlation(distance, theta[0].clone())
    }
}

/// Exponential (Matérn ν = ½) kernel; `θ = [length_scale, variance]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exponential;
impl Kernel for Exponential {
    fn hyper_count(&self) -> usize {
        2
    }
    fn covariance<T: LazyReal>(&self, distance: T, theta: &[T]) -> T {
        theta[1].clone() * lazy::exponential_correlation(distance, theta[0].clone())
    }
}

/// Matérn-5/2 kernel; `θ = [length_scale, variance]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Matern52;
impl Kernel for Matern52 {
    fn hyper_count(&self) -> usize {
        2
    }
    fn covariance<T: LazyReal>(&self, distance: T, theta: &[T]) -> T {
        theta[1].clone() * lazy::matern_52_correlation(distance, theta[0].clone())
    }
}

/// Matérn-3/2 kernel; `θ = [length_scale, variance]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Matern32;
impl Kernel for Matern32 {
    fn hyper_count(&self) -> usize {
        2
    }
    fn covariance<T: LazyReal>(&self, distance: T, theta: &[T]) -> T {
        theta[1].clone() * lazy::matern_32_correlation(distance, theta[0].clone())
    }
}

/// Sum of two kernels — `θ` is the children's `θ` concatenated.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sum<A, B>(pub A, pub B);
impl<A: Kernel, B: Kernel> Kernel for Sum<A, B> {
    fn hyper_count(&self) -> usize {
        self.0.hyper_count() + self.1.hyper_count()
    }
    fn covariance<T: LazyReal>(&self, distance: T, theta: &[T]) -> T {
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
    fn covariance<T: LazyReal>(&self, distance: T, theta: &[T]) -> T {
        let (a, b) = theta.split_at(self.0.hyper_count());
        self.0.covariance(distance.clone(), a) * self.1.covariance(distance, b)
    }
}

/// The GP negative log marginal likelihood as a function of the kernel hyperparameters — the
/// fitting objective. `distances` is the fixed pairwise-separation matrix, `y` the centered
/// observations, `noise` the nugget τ², `eye` the identity (same backend). On a unified tensor
/// backend the covariance assembly `K = k(D; θ) + τ²I` and the marginal likelihood form one
/// graph, differentiable in `θ` and `noise` — so it compiles to XLA and runs on `maxwell`.
pub fn neg_log_marginal_likelihood<M, K>(
    kernel: &K,
    distances: M,
    y: M,
    theta: &[M],
    noise: M,
    eye: M,
) -> M
where
    K: Kernel,
    M: LazyReal + LazyMatrix<Scalar = M, Vector = M>,
{
    let covariance = kernel.covariance(distances, theta) + noise * eye;
    -gaussian_process_log_marginal_likelihood(y, covariance)
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
            var * lazy::matern_52_correlation(d, ls),
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
