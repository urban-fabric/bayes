use super::*;

// ---- GLM likelihood score & curvature -------------------------------------------------
//
// First and (negative) second derivatives of the per-observation log-likelihood in the
// linear predictor η — the quantities IRLS / the Laplace Hessian / the GMRF field gradient
// need. Write-once over `LazyReal` (lifted from `markov::gmrf::likelihood`). `score` is
// `∂/∂η log p(y|η)`; `curvature` is `−∂²/∂η² log p(y|η) ≥ 0` (the IRLS weight). The forms are
// the clean mathematical ones — overflow guards (clamping η) belong in the eager caller.

/// Logistic link `σ(η) = 1/(1+e^{−η})`.
pub fn logistic<T: LazyReal>(eta: T) -> T {
    (T::scalar(1.0) + (-eta).exp()).recip()
}

/// Gaussian (identity-link) score `(y − η)/σ²`.
pub fn gaussian_score<T: LazyReal>(y: T, eta: T, variance: T) -> T {
    (y - eta) / variance
}

/// Gaussian curvature `1/σ²` (the constant IRLS weight).
pub fn gaussian_curvature<T: LazyReal>(variance: T) -> T {
    variance.recip()
}

/// Poisson (log-link) score `y − e^η`.
pub fn poisson_log_score<T: LazyReal>(y: T, eta: T) -> T {
    y - eta.exp()
}

/// Poisson (log-link) curvature `e^η` (the IRLS weight).
pub fn poisson_log_curvature<T: LazyReal>(eta: T) -> T {
    eta.exp()
}

/// Binomial (logit-link) score `y − trials·σ(η)`.
pub fn binomial_logit_score<T: LazyReal>(y: T, trials: T, eta: T) -> T {
    y - trials * logistic(eta)
}

/// Binomial (logit-link) curvature `trials·σ(η)·(1−σ(η))` (the IRLS weight).
pub fn binomial_logit_curvature<T: LazyReal>(trials: T, eta: T) -> T {
    let s = logistic(eta);
    trials * s.clone() * (T::scalar(1.0) - s)
}

/// Logistic sigmoid `1 / (1 + exp(-x))`.
pub fn sigmoid<T: LazyReal>(x: T) -> T {
    let one = T::one();
    one.clone() / (one + (-x).exp())
}

/// Logit `log(p / (1 - p))`, the inverse of [`sigmoid`] on `0 < p < 1`.
///
/// Written as `log(p) - log(1 - p)` so it composes from the lazy log directly.
/// Range membership is the caller's responsibility.
pub fn logit<T: LazyReal>(p: T) -> T {
    let one = T::one();
    p.clone().ln() - (one - p).ln()
}

#[cfg(test)]
mod glm_tests {
    use super::*;

    #[test]
    fn score_and_curvature_match_closed_form() {
        let close = |a: f64, b: f64| assert!((a - b).abs() < 1e-12, "{a} != {b}");
        close(gaussian_score(1.0, 0.5, 2.0), 0.25);
        close(gaussian_curvature(2.0), 0.5);
        close(poisson_log_score(3.0, 0.0), 2.0);
        close(poisson_log_curvature(0.0), 1.0);
        close(binomial_logit_score(2.0, 10.0, 0.0), -3.0); // 2 − 10·0.5
        close(binomial_logit_curvature(10.0, 0.0), 2.5); // 10·0.5·0.5
        close(logistic(0.0), 0.5);
    }

    #[test]
    fn distance_and_periodic() {
        let close = |a: f64, b: f64| assert!((a - b).abs() < 1e-12, "{a} != {b}");
        close(periodic_correlation(0.0, 1.0, 1.0), 1.0); // sin(0)=0 → ρ=1
    }
}
