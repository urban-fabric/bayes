//! Cross-checks: `bayes-kernels` closed-form kernels vs the concrete distribution
//! families. Relocated from `bayes-kernels` unit tests — the lazy kernels sit *below*
//! the families in the crate DAG, so this cross-check runs at the facade level
//! (`bayes`), where both the kernels and every family are visible.
#![allow(
    unused_imports,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]

// The lazy kernels are generic over `euclid::numeric::Real`; the cross-checks
// instantiate them at `f64` and call Real methods (e.g. `ln_1p_exp`) directly.
#[allow(unused_imports)]
use euclid::numeric::{Integer, Natural, Real};

use bayes::kernels::*;
// The scalar/lazy kernels are exercised end-to-end here against bayes' own concrete
// distributions (euclid-only families). The multivariate-matrix, conjugate-matrix,
// and S² directional families moved to the sibling crate `ito`; the cross-checks that
// these same kernels agree with `ito::MultivariateNormal` / `ito::VonMisesFisher` /
// `ito::Dirichlet` (etc.) live in ito's integration test
// `tests/bayes_kernels_on_eager.rs`.
use bayes::{
    ChiSquaredDivergence, HellingerDistance, KlDivergence, RenyiDivergence, TotalVariation,
};
use bayes_distribution::{
    Bernoulli, Beta, Binomial, Cauchy, DiscreteUniform, Exponential, Gamma, Geometric, Gumbel,
    Laplace, NegativeBinomial, Normal, Pareto, Poisson, Rayleigh, Uniform, VonMises, Wald, Weibull,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol + tol * b.abs(),
        "observed {a}, expected {b}"
    );
}

#[test]
fn lazy_real_chi_squared_renyi_tv_kernels_match_distribution() {
    let alpha = 0.5;

    let bernoulli = (Bernoulli::new(0.3), Bernoulli::new(0.7));
    close(
        chi_squared_bernoulli(0.3, 0.7),
        bernoulli.0.chi_squared_divergence(&bernoulli.1),
        1e-12,
    );
    close(
        renyi_bernoulli(0.3, 0.7, alpha),
        bernoulli.0.renyi_divergence(&bernoulli.1, alpha),
        1e-12,
    );
    close(
        total_variation_bernoulli(0.3, 0.7),
        bernoulli.0.total_variation(&bernoulli.1),
        1e-12,
    );

    let poisson = (Poisson::new(2.5), Poisson::new(4.0));
    close(
        chi_squared_poisson(2.5, 4.0),
        poisson.0.chi_squared_divergence(&poisson.1),
        1e-12,
    );

    // χ² / Rényi interiors in their convergence region.
    let normal = (Normal::new(0.0, 1.0), Normal::new(0.5, 1.2));
    close(
        chi_squared_normal_on_support(0.0, 1.0, 0.5, 1.2),
        normal.0.chi_squared_divergence(&normal.1),
        1e-12,
    );
    close(
        renyi_normal_on_support(0.0, 1.0, 0.5, 1.2, alpha),
        normal.0.renyi_divergence(&normal.1, alpha),
        1e-12,
    );

    let exponential = (Exponential::new(1.5), Exponential::new(1.0));
    close(
        chi_squared_exponential_on_support(1.5, 1.0),
        exponential.0.chi_squared_divergence(&exponential.1),
        1e-12,
    );

    let gamma = (Gamma::new(2.5, 1.2), Gamma::new(3.0, 0.8));
    close(
        chi_squared_gamma_on_support(2.5, 1.2, 3.0, 0.8),
        gamma.0.chi_squared_divergence(&gamma.1),
        1e-12,
    );
    close(
        renyi_gamma_on_support(2.5, 1.2, 3.0, 0.8, alpha),
        gamma.0.renyi_divergence(&gamma.1, alpha),
        1e-12,
    );

    // Binomial with a matched trial count n = 12 (always finite).
    let binomial = (Binomial::new(12_u64, 0.3), Binomial::new(12_u64, 0.5));
    close(
        chi_squared_binomial(12.0, 0.3, 0.5),
        binomial.0.chi_squared_divergence(&binomial.1),
        1e-12,
    );
    close(
        renyi_binomial(12.0, 0.3, 0.5, alpha),
        binomial.0.renyi_divergence(&binomial.1, alpha),
        1e-12,
    );

    // Negative-binomial interiors in their convergence region (matched r = 5).
    let negbin = (
        NegativeBinomial::new(5.0, 0.6),
        NegativeBinomial::new(5.0, 0.4),
    );
    close(
        chi_squared_negative_binomial_on_support(5.0, 0.6, 0.4),
        negbin.0.chi_squared_divergence(&negbin.1),
        1e-12,
    );
    close(
        renyi_negative_binomial_on_support(5.0, 0.6, 0.4, alpha),
        negbin.0.renyi_divergence(&negbin.1, alpha),
        1e-12,
    );

    // Wald / inverse-Gaussian interiors in their convergence region.
    let wald = (Wald::new(1.0, 3.0), Wald::new(1.2, 2.0));
    close(
        chi_squared_wald_on_support(1.0, 3.0, 1.2, 2.0),
        wald.0.chi_squared_divergence(&wald.1),
        1e-12,
    );
    close(
        renyi_wald_on_support(1.0, 3.0, 1.2, 2.0, alpha),
        wald.0.renyi_divergence(&wald.1, alpha),
        1e-12,
    );

    // Uniform / discrete-uniform χ² and Rényi on nested support.
    let uniform = (Uniform::new(-1.0, 2.0), Uniform::new(-2.0, 3.0));
    close(
        chi_squared_uniform_on_support(3.0, 5.0),
        uniform.0.chi_squared_divergence(&uniform.1),
        1e-12,
    );
    close(
        renyi_uniform_on_support(-1.0, 2.0, -2.0, 3.0, alpha),
        uniform.0.renyi_divergence(&uniform.1, alpha),
        1e-12,
    );
    let du = (
        DiscreteUniform::new(0_i64, 3),
        DiscreteUniform::new(-1_i64, 5),
    );
    close(
        chi_squared_discrete_uniform_on_support(4.0, 7.0),
        du.0.chi_squared_divergence(&du.1),
        1e-12,
    );
    close(
        renyi_discrete_uniform_on_support(0.0, 3.0, -1.0, 5.0, alpha),
        du.0.renyi_divergence(&du.1, alpha),
        1e-12,
    );
}

#[test]
fn lazy_real_von_mises_divergences_match_distribution() {
    // κ ≤ 15 keeps the distribution on its `i0(x).ln()` branch, so the scaled-
    // Bessel reformulation `ln I₀ᵉ + |x|` agrees to rounding.
    let vm = (VonMises::new(0.3, 2.0), VonMises::new(0.8, 1.5));
    close(
        kl_von_mises(0.3, 2.0, 0.8, 1.5),
        vm.0.kl_divergence(&vm.1),
        1e-9,
    );
    close(
        hellinger_sq_von_mises(0.3, 2.0, 0.8, 1.5),
        vm.0.hellinger_sq(&vm.1),
        1e-9,
    );
    close(
        chi_squared_von_mises(0.3, 2.0, 0.8, 1.5),
        vm.0.chi_squared_divergence(&vm.1),
        1e-9,
    );
    close(
        renyi_von_mises(0.3, 2.0, 0.8, 1.5, 0.5),
        vm.0.renyi_divergence(&vm.1, 0.5),
        1e-9,
    );
}

#[test]
fn lazy_real_kl_kernels_match_distribution() {
    // Gaussian family (Normal / LogNormal / HalfNormal share one closed form).
    let normal = (Normal::new(0.0, 1.0), Normal::new(1.0, 2.0));
    close(
        kl_normal(0.0, 1.0, 1.0, 2.0),
        normal.0.kl_divergence(&normal.1),
        1e-12,
    );

    let exponential = (Exponential::new(1.5), Exponential::new(0.7));
    close(
        kl_exponential(1.5, 0.7),
        exponential.0.kl_divergence(&exponential.1),
        1e-12,
    );

    let poisson = (Poisson::new(2.5), Poisson::new(4.0));
    close(
        kl_poisson(2.5, 4.0),
        poisson.0.kl_divergence(&poisson.1),
        1e-12,
    );

    let geometric = (Geometric::new(0.3), Geometric::new(0.6));
    close(
        kl_geometric(0.3, 0.6),
        geometric.0.kl_divergence(&geometric.1),
        1e-12,
    );

    let bernoulli = (Bernoulli::new(0.3), Bernoulli::new(0.7));
    close(
        kl_bernoulli_on_unit(0.3, 0.7),
        bernoulli.0.kl_divergence(&bernoulli.1),
        1e-12,
    );

    // Negative-binomial with matched successes r = 5.
    let negbin = (
        NegativeBinomial::new(5.0, 0.4),
        NegativeBinomial::new(5.0, 0.6),
    );
    close(
        kl_negative_binomial(5.0, 0.4, 0.6),
        negbin.0.kl_divergence(&negbin.1),
        1e-12,
    );

    let cauchy = (Cauchy::new(0.0, 1.0), Cauchy::new(0.5, 1.7));
    close(
        kl_cauchy(0.0, 1.0, 0.5, 1.7),
        cauchy.0.kl_divergence(&cauchy.1),
        1e-12,
    );

    let rayleigh = (Rayleigh::new(1.2), Rayleigh::new(2.0));
    close(
        kl_rayleigh(1.2, 2.0),
        rayleigh.0.kl_divergence(&rayleigh.1),
        1e-12,
    );

    let laplace = (Laplace::new(0.0, 1.0), Laplace::new(0.5, 1.7));
    close(
        kl_laplace(0.0, 1.0, 0.5, 1.7),
        laplace.0.kl_divergence(&laplace.1),
        1e-12,
    );

    // Wald::new(mu, shape).
    let wald = (Wald::new(1.0, 2.0), Wald::new(1.5, 3.0));
    close(
        kl_wald(1.0, 2.0, 1.5, 3.0),
        wald.0.kl_divergence(&wald.1),
        1e-12,
    );

    // Gamma family (Gamma / ChiSquared / InverseGamma share one closed form).
    let gamma = (Gamma::new(2.5, 1.2), Gamma::new(3.0, 0.8));
    close(
        kl_gamma(2.5, 1.2, 3.0, 0.8),
        gamma.0.kl_divergence(&gamma.1),
        1e-12,
    );

    let beta = (Beta::new(2.0, 5.0), Beta::new(3.0, 4.0));
    close(
        kl_beta(2.0, 5.0, 3.0, 4.0),
        beta.0.kl_divergence(&beta.1),
        1e-12,
    );

    // Weibull::new(scale, shape); the kernel takes (shape, scale).
    let weibull = (Weibull::new(1.4, 2.2), Weibull::new(1.0, 1.6));
    close(
        kl_weibull(2.2, 1.4, 1.6, 1.0),
        weibull.0.kl_divergence(&weibull.1),
        1e-12,
    );

    let gumbel = (Gumbel::new(0.0, 1.0), Gumbel::new(0.5, 1.5));
    close(
        kl_gumbel(0.0, 1.0, 0.5, 1.5),
        gumbel.0.kl_divergence(&gumbel.1),
        1e-12,
    );

    // Pareto::new(minimum, shape) with nested support (xm0 ≥ xm1).
    let pareto = (Pareto::new(3.0, 2.0), Pareto::new(2.0, 1.5));
    close(
        kl_pareto_on_support(3.0, 2.0, 2.0, 1.5),
        pareto.0.kl_divergence(&pareto.1),
        1e-12,
    );

    // Binomial with a matched trial count n = 12 (n × the Bernoulli KL).
    let binomial = (Binomial::new(12_u64, 0.3), Binomial::new(12_u64, 0.5));
    close(
        kl_binomial(12.0, 0.3, 0.5),
        binomial.0.kl_divergence(&binomial.1),
        1e-12,
    );

    // Uniform / discrete-uniform on nested support.
    let uniform = (Uniform::new(-1.0, 2.0), Uniform::new(-2.0, 3.0));
    close(
        kl_uniform_on_support(3.0, 5.0),
        uniform.0.kl_divergence(&uniform.1),
        1e-12,
    );
    let du = (
        DiscreteUniform::new(0_i64, 3),
        DiscreteUniform::new(-1_i64, 5),
    );
    close(
        kl_discrete_uniform_on_support(4.0, 7.0),
        du.0.kl_divergence(&du.1),
        1e-12,
    );
}
