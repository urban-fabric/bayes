//! One table of kernels, so a property is written once rather than per family.
//!
//! The crate under test has no distribution types, so there is nothing
//! object-safe to sweep — this builds the equivalent out of closures. Each
//! entry carries only what the kernels actually provide: twelve families have a
//! quantile, nineteen have a log-survival, and the checks skip what is absent
//! rather than inventing it.

/// A kernel family, bound to one concrete parameter set.
pub(crate) struct Case {
    pub(crate) name: &'static str,
    pub(crate) cdf: Box<dyn Fn(f64) -> f64>,
    pub(crate) ln_sf: Box<dyn Fn(f64) -> f64>,
    /// Present only where the family has a closed-form quantile kernel.
    pub(crate) quantile: Option<Box<dyn Fn(f64) -> f64>>,
    /// The lower support bound, so a check knows where the mass starts.
    pub(crate) lo: f64,
}

fn case(
    name: &'static str,
    lo: f64,
    cdf: impl Fn(f64) -> f64 + 'static,
    ln_sf: impl Fn(f64) -> f64 + 'static,
) -> Case {
    Case {
        name,
        cdf: Box::new(cdf),
        ln_sf: Box::new(ln_sf),
        quantile: None,
        lo,
    }
}

impl Case {
    fn with_quantile(mut self, quantile: impl Fn(f64) -> f64 + 'static) -> Self {
        self.quantile = Some(Box::new(quantile));
        self
    }
}

/// Every continuous kernel family that has both a CDF and a log-survival.
///
/// The parameters are deliberately **not** all unit-scaled: a reduction or a
/// complement that holds at scale 1 can fail where a shape changes the branch
/// a series takes, and three of the families here are included twice for that
/// reason — once benign, once with a shape that makes a tail heavy enough to
/// have no variance.
pub(crate) fn cases() -> Vec<Case> {
    use bayes_kernels as k;
    vec![
        case(
            "normal(0,1)",
            f64::NEG_INFINITY,
            |x| k::normal_cdf(x, 0.0, 1.0),
            |x| k::normal_ln_sf(x, 0.0, 1.0),
        )
        .with_quantile(|p| k::normal_quantile(p, 0.0, 1.0)),
        case(
            "normal(-3,0.25)",
            f64::NEG_INFINITY,
            |x| k::normal_cdf(x, -3.0, 0.25),
            |x| k::normal_ln_sf(x, -3.0, 0.25),
        )
        .with_quantile(|p| k::normal_quantile(p, -3.0, 0.25)),
        case(
            "exponential(2.5)",
            0.0,
            |x| k::exponential_cdf(x, 2.5),
            |x| k::exponential_ln_sf(x, 2.5),
        )
        .with_quantile(|p| k::exponential_quantile(p, 2.5)),
        case(
            "weibull(3,1.7)",
            0.0,
            |x| k::weibull_cdf(x, 3.0, 1.7),
            |x| k::weibull_ln_sf(x, 3.0, 1.7),
        )
        .with_quantile(|p| k::weibull_quantile(p, 3.0, 1.7)),
        case(
            "weibull(1,0.4)",
            0.0,
            |x| k::weibull_cdf(x, 1.0, 0.4),
            |x| k::weibull_ln_sf(x, 1.0, 0.4),
        )
        .with_quantile(|p| k::weibull_quantile(p, 1.0, 0.4)),
        case(
            "pareto(2,1.3)",
            2.0,
            |x| k::pareto_cdf(x, 2.0, 1.3),
            |x| k::pareto_ln_sf(x, 2.0, 1.3),
        )
        .with_quantile(|p| k::pareto_quantile(p, 2.0, 1.3)),
        case(
            "rayleigh(1.4)",
            0.0,
            |x| k::rayleigh_cdf(x, 1.4),
            |x| k::rayleigh_ln_sf(x, 1.4),
        )
        .with_quantile(|p| k::rayleigh_quantile(p, 1.4)),
        case(
            "lognormal(0.5,1.2)",
            0.0,
            |x| k::lognormal_cdf(x, 0.5, 1.2),
            |x| k::lognormal_ln_sf(x, 0.5, 1.2),
        )
        .with_quantile(|p| k::lognormal_quantile(p, 0.5, 1.2)),
        case(
            "cauchy(0.5,1.5)",
            f64::NEG_INFINITY,
            |x| k::cauchy_cdf(x, 0.5, 1.5),
            |x| k::cauchy_ln_sf(x, 0.5, 1.5),
        )
        .with_quantile(|p| k::cauchy_quantile(p, 0.5, 1.5)),
        case(
            "laplace(-1,0.5)",
            f64::NEG_INFINITY,
            |x| k::laplace_cdf(x, -1.0, 0.5),
            |x| k::laplace_ln_sf(x, -1.0, 0.5),
        )
        .with_quantile(|p| k::laplace_quantile(p, -1.0, 0.5)),
        case(
            "logistic(2,0.8)",
            f64::NEG_INFINITY,
            |x| k::logistic_cdf(x, 2.0, 0.8),
            |x| k::logistic_ln_sf(x, 2.0, 0.8),
        )
        .with_quantile(|p| k::logistic_quantile(p, 2.0, 0.8)),
        case(
            "gumbel(0,1)",
            f64::NEG_INFINITY,
            |x| k::gumbel_cdf(x, 0.0, 1.0),
            |x| k::gumbel_ln_sf(x, 0.0, 1.0),
        )
        .with_quantile(|p| k::gumbel_quantile(p, 0.0, 1.0)),
        case(
            "frechet(2,3)",
            0.0,
            |x| k::frechet_cdf(x, 3.0, 2.0),
            |x| k::frechet_ln_sf(x, 3.0, 2.0),
        )
        .with_quantile(|p| k::frechet_quantile(p, 3.0, 2.0)),
        case(
            "lomax(3,2)",
            0.0,
            |x| k::lomax_cdf(x, 3.0, 2.0),
            |x| k::lomax_ln_sf(x, 3.0, 2.0),
        )
        .with_quantile(|p| k::lomax_quantile(p, 3.0, 2.0)),
        case(
            "half_normal(2)",
            0.0,
            |x| k::half_normal_cdf(x, 2.0),
            |x| k::half_normal_ln_sf(x, 2.0),
        )
        .with_quantile(|p| k::half_normal_quantile(p, 2.0)),
        // These kernels take **half** the degrees of freedom, so the names
        // below are the distributions and the arguments are the halves.
        case(
            "chi_squared(3 df)",
            0.0,
            |x| k::chi_squared_cdf(x, 1.5),
            |x| k::chi_squared_ln_sf(x, 1.5),
        ),
        case(
            "chi_squared(0.5 df)",
            0.0,
            |x| k::chi_squared_cdf(x, 0.25),
            |x| k::chi_squared_ln_sf(x, 0.25),
        ),
        case(
            "gamma(4,1.5)",
            0.0,
            |x| k::gamma_cdf(x, 4.0, 1.5),
            |x| k::gamma_ln_sf(x, 4.0, 1.5),
        ),
        case(
            "inverse_gamma(4,2)",
            0.0,
            |x| k::inverse_gamma_cdf(x, 4.0, 2.0),
            |x| k::inverse_gamma_ln_sf(x, 4.0, 2.0),
        ),
        case(
            "fisher_snedecor(6,9)",
            0.0,
            |x| k::fisher_snedecor_cdf(x, 6.0, 9.0),
            |x| k::fisher_snedecor_ln_sf(x, 6.0, 9.0),
        ),
        case(
            "students_t(5)",
            f64::NEG_INFINITY,
            |x| k::students_t_cdf(x, 0.0, 1.0, 5.0),
            |x| k::students_t_ln_sf(x, 0.0, 1.0, 5.0),
        ),
        case(
            "wald(1,3)",
            0.0,
            |x| k::wald_cdf(x, 1.0, 3.0),
            |x| k::wald_ln_sf(x, 1.0, 3.0),
        ),
        // The parameter set whose textbook survival overflows: `2λ/μ = 800`.
        case(
            "wald(1,400)",
            0.0,
            |x| k::wald_cdf(x, 1.0, 400.0),
            |x| k::wald_ln_sf(x, 1.0, 400.0),
        ),
    ]
}
