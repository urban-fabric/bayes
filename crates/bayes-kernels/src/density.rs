//! Log-density and log-mass kernel view.
//!
//! Implementations live in [`crate::families`]. This module groups the density
//! and mass functions by purpose.

pub use crate::families::{
    bernoulli::{bernoulli_log_pmf, bernoulli_logit_log_pmf},
    beta::beta_log_pdf_on_unit,
    beta_binomial::beta_binomial_log_pmf,
    binomial::binomial_log_pmf,
    categorical::categorical_log_pmf,
    cauchy::cauchy_log_pdf,
    chi_squared::chi_squared_log_pdf_on_support,
    discrete_uniform::discrete_uniform_log_pmf_on_support,
    exponential::{exponential_log_pdf_on_support, exponential_log_survival_on_support},
    fisher_snedecor::fisher_snedecor_log_pdf_on_support,
    frechet::frechet_log_pdf_on_support,
    gamma::gamma_log_pdf_on_support,
    geometric::{geometric_failures_log_pmf, geometric_log_pmf},
    gumbel::gumbel_log_pdf,
    half_normal::half_normal_log_pdf_on_support,
    hypergeometric::hypergeometric_log_pmf,
    inverse_gamma::inverse_gamma_log_pdf_on_support,
    laplace::laplace_log_pdf,
    logistic::logistic_log_pdf,
    lognormal::lognormal_log_pdf_on_support,
    lomax::lomax_log_pdf_on_support,
    negative_binomial::negative_binomial_log_pmf,
    normal::normal_log_pdf,
    pareto::pareto_log_pdf_on_support,
    poisson::poisson_log_pmf,
    rayleigh::rayleigh_log_pdf_on_support,
    student_t::student_t_log_pdf,
    triangular::triangular_pdf,
    uniform::uniform_log_pdf_on_support,
    von_mises::von_mises_log_pdf,
    wald::wald_log_pdf_on_support,
    weibull::weibull_log_pdf_on_support,
};
