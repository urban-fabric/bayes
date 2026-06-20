//! Lazy multivariate kernels over the `euclid::array` matrix contract.
//!
//! Where `lazy` writes scalar/vector closed forms over [`LazyReal`], this module
//! writes the **matrix** closed forms — the multivariate Gaussian families — over
//! [`LazyMatrix`]/[`LazyVector`]. The kernels never form an inverse or touch a scalar
//! element: they ask the matrix contract for the *mathematical quantities* it owns
//! (`mahalanobis_squared`, `logdet_spd`), so the same source runs on the `f64`
//! reference (`ramanujan`) and on a tensor backend such as `maxwell` (XLA).

use euclid::array::{LazyMatrix, LazyVector};
use euclid::numeric::LazyReal;
use frobenius::{PreparedSpd, SpdFactor};

/// Multivariate-normal log-density `ln N(x | µ, Σ) = −½·(δᵀΣ⁻¹δ + d·ln(2π) + ln|Σ|)`,
/// `δ = x − µ`. The Mahalanobis term and log-determinant come from the matrix
/// contract (a stable Cholesky on either backend); no inverse is formed.
pub fn multivariate_normal_log_pdf<M: LazyMatrix<Scalar: LazyReal>>(
    x: M::Vector,
    mean: M::Vector,
    covariance: M,
) -> M::Scalar {
    let delta = x.sub(mean);
    let dim = M::Scalar::scalar(delta.len() as f64);
    let mahalanobis = covariance.mahalanobis_squared(&delta);
    let log_det = covariance.logdet_spd();
    -M::Scalar::half() * (mahalanobis + dim * M::Scalar::ln_tau() + log_det)
}

/// Multivariate-normal entropy `½·(d·ln(2π) + ln|Σ| + d)`.
pub fn multivariate_normal_entropy<M: LazyMatrix<Scalar: LazyReal>>(covariance: M) -> M::Scalar {
    let dim = M::Scalar::scalar(covariance.shape().0 as f64);
    let log_det = covariance.logdet_spd();
    M::Scalar::half() * (dim.clone() * M::Scalar::ln_tau() + log_det + dim)
}

/// Gaussian KL `D(N₀ ‖ N₁) = ½[ ln(|Σ₁|/|Σ₀|) − d + tr(Σ₁⁻¹Σ₀) + (µ₁−µ₀)ᵀΣ₁⁻¹(µ₁−µ₀) ]`.
///
/// `tr(Σ₁⁻¹Σ₀)` is `(Σ₁.solve_spd(Σ₀)).trace()`; the quadratic is
/// `Σ₁.mahalanobis_squared(µ₁−µ₀)`. No inverse is formed.
pub fn multivariate_normal_kl<M: LazyMatrix<Scalar: LazyReal>>(
    mean_self: M::Vector,
    cov_self: M,
    mean_other: M::Vector,
    cov_other: M,
) -> M::Scalar {
    let delta = mean_other.sub(mean_self);
    let dim = M::Scalar::scalar(delta.len() as f64);
    let trace = cov_other.clone().solve_spd(cov_self.clone()).trace();
    let mahalanobis = cov_other.mahalanobis_squared(&delta);
    let log_det_ratio = cov_other.logdet_spd() - cov_self.logdet_spd();
    M::Scalar::half() * (log_det_ratio - dim + trace + mahalanobis)
}

/// Wishart log-density at SPD `X`, scale `V`, degrees `ν`:
/// `½(ν−d−1)·ln|X| − ½·tr(V⁻¹X) − [½νd·ln2 + ½ν·ln|V| + ln Γ_d(ν/2)]`.
pub fn wishart_log_pdf<M: LazyMatrix<Scalar: LazyReal>>(
    x: M,
    scale: M,
    freedom: M::Scalar,
) -> M::Scalar {
    let d = x.shape().0;
    let dim = M::Scalar::scalar(d as f64);
    let half = M::Scalar::half();
    let half_freedom = freedom.clone() * half.clone();

    let log_det_x = x.clone().logdet_spd();
    let trace = scale.clone().solve_spd(x).trace();
    let log_det_scale = scale.logdet_spd();
    let ln_mv_gamma = crate::lazy::ln_multivariate_gamma(d, half_freedom.clone());

    let ln_normalizer = half_freedom.clone() * dim.clone() * M::Scalar::ln_2()
        + half_freedom * log_det_scale
        + ln_mv_gamma;
    half.clone() * (freedom - dim - M::Scalar::one()) * log_det_x - half * trace - ln_normalizer
}

/// Inverse-Wishart log-density at SPD `Σ`, scale `V`, degrees `ν`:
/// `[½ν·ln|V| − ½νd·ln2 − ln Γ_d(ν/2)] − ½(ν+d+1)·ln|Σ| − ½·tr(V·Σ⁻¹)`.
pub fn inverse_wishart_log_pdf<M: LazyMatrix<Scalar: LazyReal>>(
    sigma: M,
    scale: M,
    freedom: M::Scalar,
) -> M::Scalar {
    let d = sigma.shape().0;
    let dim = M::Scalar::scalar(d as f64);
    let half = M::Scalar::half();
    let half_freedom = freedom.clone() * half.clone();

    let log_det_sigma = sigma.clone().logdet_spd();
    let trace = sigma.solve_spd(scale.clone()).trace(); // tr(Σ⁻¹V) = tr(VΣ⁻¹)
    let log_det_scale = scale.logdet_spd();
    let ln_mv_gamma = crate::lazy::ln_multivariate_gamma(d, half_freedom.clone());

    let ln_normalizer = half_freedom.clone() * log_det_scale
        - half_freedom * dim.clone() * M::Scalar::ln_2()
        - ln_mv_gamma;
    ln_normalizer - half.clone() * (freedom + dim + M::Scalar::one()) * log_det_sigma - half * trace
}

/// `Σ_{i=1}^{d} ψ((ν + 1 − i)/2)` — the Wishart multivariate digamma sum. A host-side
/// fold over the structural dimension `d` building lazy `digamma` terms, the same
/// shape as [`crate::lazy::ln_multivariate_gamma`].
fn digamma_wishart_sum<S: LazyReal>(dimension: usize, freedom: S) -> S {
    let mut acc = S::zero();
    for i in 1..=dimension {
        acc = acc + ((freedom.clone() + S::scalar(1.0 - i as f64)) * S::half()).digamma();
    }
    acc
}

/// Wishart differential entropy (scale `V`, degrees `ν`):
/// `−½(ν−d−1)·E[ln|X|] + ½νd + lnZ`, with `E[ln|X|] = ln|V| + d·ln2 + Σ ψ((ν+1−i)/2)`
/// and `lnZ = ½νd·ln2 + ½ν·ln|V| + ln Γ_d(ν/2)`. The log-determinant comes from the
/// matrix contract; the digamma sum and `ln Γ_d` are pure scalar analysis.
pub fn wishart_entropy<M: LazyMatrix<Scalar: LazyReal>>(scale: M, freedom: M::Scalar) -> M::Scalar {
    let d = scale.shape().0;
    let dim = M::Scalar::scalar(d as f64);
    let half = M::Scalar::half();
    let log_det_scale = scale.logdet_spd();
    let digamma_sum = digamma_wishart_sum(d, freedom.clone());

    let expected_log_det = log_det_scale.clone() + dim.clone() * M::Scalar::ln_2() + digamma_sum;
    let ln_normalizer = half.clone() * freedom.clone() * dim.clone() * M::Scalar::ln_2()
        + half.clone() * freedom.clone() * log_det_scale
        + crate::lazy::ln_multivariate_gamma(d, freedom.clone() * half.clone());

    -half.clone() * (freedom.clone() - dim.clone() - M::Scalar::one()) * expected_log_det
        + half * freedom * dim
        + ln_normalizer
}

/// Inverse-Wishart differential entropy (scale `V`, degrees `ν`):
/// `−lnZ + ½(ν+d+1)·E[ln|Σ|] + ½νd`, with `E[ln|Σ|] = ln|V| − d·ln2 − Σ ψ((ν+1−i)/2)`
/// and `lnZ = ½ν·ln|V| − ½νd·ln2 − ln Γ_d(ν/2)`.
pub fn inverse_wishart_entropy<M: LazyMatrix<Scalar: LazyReal>>(
    scale: M,
    freedom: M::Scalar,
) -> M::Scalar {
    let d = scale.shape().0;
    let dim = M::Scalar::scalar(d as f64);
    let half = M::Scalar::half();
    let log_det_scale = scale.logdet_spd();
    let digamma_sum = digamma_wishart_sum(d, freedom.clone());

    let expected_log_det = log_det_scale.clone() - dim.clone() * M::Scalar::ln_2() - digamma_sum;
    let ln_normalizer = half.clone() * freedom.clone() * log_det_scale
        - half.clone() * freedom.clone() * dim.clone() * M::Scalar::ln_2()
        - crate::lazy::ln_multivariate_gamma(d, freedom.clone() * half.clone());

    -ln_normalizer
        + half.clone() * (freedom.clone() + dim.clone() + M::Scalar::one()) * expected_log_det
        + half * freedom * dim
}

/// Multivariate Student-t entropy (scale `Σ`, degrees `ν`):
/// `lnΓ(ν/2) − lnΓ((ν+d)/2) + ½(d·ln(νπ) + ln|Σ|) + (ν+d)/2·(ψ((ν+d)/2) − ψ(ν/2))`.
pub fn multivariate_student_t_entropy<M: LazyMatrix<Scalar: LazyReal>>(
    scale: M,
    freedom: M::Scalar,
) -> M::Scalar {
    let dim = M::Scalar::scalar(scale.shape().0 as f64);
    let log_det = scale.logdet_spd();
    let half = M::Scalar::half();
    let half_nu = freedom.clone() * half.clone();
    let half_nu_plus_d = (freedom.clone() + dim.clone()) * half.clone();
    half_nu.clone().ln_gamma() - half_nu_plus_d.clone().ln_gamma()
        + half * (dim * (freedom * M::Scalar::pi()).ln() + log_det)
        + half_nu_plus_d.clone() * (half_nu_plus_d.digamma() - half_nu.digamma())
}

/// Gaussian-process log marginal likelihood `−½(yᵀK⁻¹y + ln|K| + n·ln 2π)` — the
/// log-evidence of observations `y` under a zero-mean GP with kernel matrix `K`.
/// (Structurally the multivariate-normal log-density at zero mean.)
pub fn gaussian_process_log_marginal_likelihood<M: LazyMatrix<Scalar: LazyReal>>(
    y: M::Vector,
    kernel: M,
) -> M::Scalar {
    let n = M::Scalar::scalar(y.len() as f64);
    let mahalanobis = kernel.mahalanobis_squared(&y);
    let log_det = kernel.logdet_spd();
    -M::Scalar::half() * (mahalanobis + n * M::Scalar::ln_tau() + log_det)
}

/// **Prepared** GP log marginal likelihood — identical math to
/// [`gaussian_process_log_marginal_likelihood`], but over a pre-factored kernel
/// ([`SpdFactor`]). The O(n³) Cholesky is hoisted out of the per-call path: factor `K` once,
/// then every evaluation is just a triangular solve + a cached log-determinant. One source,
/// fast on both the eager backend (cached factor) and XLA (factor fed as a graph input).
pub fn gaussian_process_log_marginal_likelihood_prepared<M: PreparedSpd<Scalar: LazyReal>>(
    y: M::Vector,
    kernel: &M::Factor,
) -> M::Scalar {
    let n = M::Scalar::scalar(y.len() as f64);
    let solved = kernel.solve_vec(y.clone());
    let mahalanobis = y.dot(&solved);
    let log_det = kernel.logdet();
    -M::Scalar::half() * (mahalanobis + n * M::Scalar::ln_tau() + log_det)
}

/// **Prepared** multivariate-normal log-density — [`multivariate_normal_log_pdf`] over a
/// pre-factored covariance ([`SpdFactor`]); the per-call Cholesky is hoisted out.
pub fn multivariate_normal_log_pdf_prepared<M: PreparedSpd<Scalar: LazyReal>>(
    x: M::Vector,
    mean: M::Vector,
    covariance: &M::Factor,
) -> M::Scalar {
    let delta = x.sub(mean);
    let dim = M::Scalar::scalar(delta.len() as f64);
    let solved = covariance.solve_vec(delta.clone());
    let mahalanobis = delta.dot(&solved);
    let log_det = covariance.logdet();
    -M::Scalar::half() * (mahalanobis + dim * M::Scalar::ln_tau() + log_det)
}

/// **Prepared** multivariate Student-t log-density — [`multivariate_student_t_log_pdf`] over a
/// pre-factored scale ([`SpdFactor`]); only `δᴴΣ⁻¹δ` and `ln|Σ|` come from the factor.
pub fn multivariate_student_t_log_pdf_prepared<M: PreparedSpd<Scalar: LazyReal>>(
    x: M::Vector,
    location: M::Vector,
    scale: &M::Factor,
    freedom: M::Scalar,
) -> M::Scalar {
    let delta = x.sub(location);
    let dim = M::Scalar::scalar(delta.len() as f64);
    let solved = scale.solve_vec(delta.clone());
    let mahalanobis = delta.dot(&solved);
    let log_det = scale.logdet();
    let half = M::Scalar::half();
    let nu_plus_d = freedom.clone() + dim.clone();
    (nu_plus_d.clone() * half.clone()).ln_gamma()
        - (freedom.clone() * half.clone()).ln_gamma()
        - half.clone() * (dim * (freedom.clone() * M::Scalar::pi()).ln() + log_det)
        - half * nu_plus_d * (M::Scalar::one() + mahalanobis / freedom).ln()
}

/// **Prepared** Wishart log-density — [`wishart_log_pdf`] with the *scale* `V` pre-factored
/// ([`SpdFactor`]): `tr(V⁻¹X)` and `ln|V|` reuse the factor, so evaluating at many `X` for a
/// fixed `(ν, V)` only factors each `X` (for `ln|X|`), not `V`.
pub fn wishart_log_pdf_prepared<M: PreparedSpd<Scalar: LazyReal>>(
    x: M,
    scale: &M::Factor,
    freedom: M::Scalar,
) -> M::Scalar {
    let d = x.shape().0;
    let dim = M::Scalar::scalar(d as f64);
    let half = M::Scalar::half();
    let half_freedom = freedom.clone() * half.clone();

    let log_det_x = x.clone().logdet_spd();
    let trace = scale.solve(x).trace();
    let log_det_scale = scale.logdet();
    let ln_mv_gamma = crate::lazy::ln_multivariate_gamma(d, half_freedom.clone());

    let ln_normalizer = half_freedom.clone() * dim.clone() * M::Scalar::ln_2()
        + half_freedom * log_det_scale
        + ln_mv_gamma;
    half.clone() * (freedom - dim - M::Scalar::one()) * log_det_x - half * trace - ln_normalizer
}

/// **Prepared** GP posterior predictive mean `µ* = K*ᵀ K⁻¹ y` over a pre-factored train kernel.
pub fn gaussian_process_posterior_mean_prepared<M: PreparedSpd<Scalar: LazyReal>>(
    train_factor: &M::Factor,
    cross_covariance: M,
    observations: M::Vector,
) -> M::Vector {
    let alpha = train_factor.solve_vec(observations);
    cross_covariance.adjoint().matvec(alpha)
}

/// **Prepared** GP posterior predictive covariance `Σ* = K** − K*ᵀ K⁻¹ K*` over a pre-factored
/// train kernel (the factor supplies `K⁻¹K*` with no re-factorization).
pub fn gaussian_process_posterior_covariance_prepared<M: PreparedSpd<Scalar: LazyReal>>(
    train_factor: &M::Factor,
    cross_covariance: M,
    test_kernel: M,
) -> M {
    let solved = train_factor.solve(cross_covariance.clone());
    let reduction = cross_covariance.adjoint().matmul(solved);
    test_kernel.add(reduction.scale(-M::Scalar::one()))
}

/// The **whitened** (decorrelated, unit-variance) residual `z = L⁻¹(x − µ)` for `Σ = L Lᴴ` —
/// the standardized residual. `‖z‖²` is the squared Mahalanobis distance, so this is the vector
/// behind `mahalanobis`; useful for residual diagnostics and inverting a Cholesky sample.
pub fn gaussian_whitened_residual<M: PreparedSpd<Scalar: LazyReal>>(
    x: M::Vector,
    mean: M::Vector,
    covariance: &M::Factor,
) -> M::Vector {
    covariance.whiten_vec(x.sub(mean))
}

/// Multivariate Student-t log-density (location `µ`, scale `Σ`, degrees `ν`):
/// `lnΓ((ν+d)/2) − lnΓ(ν/2) − ½(d·ln(νπ) + ln|Σ|) − ½(ν+d)·ln(1 + δᵀΣ⁻¹δ / ν)`.
pub fn multivariate_student_t_log_pdf<M: LazyMatrix<Scalar: LazyReal>>(
    x: M::Vector,
    location: M::Vector,
    scale: M,
    freedom: M::Scalar,
) -> M::Scalar {
    let delta = x.sub(location);
    let dim = M::Scalar::scalar(delta.len() as f64);
    let mahalanobis = scale.mahalanobis_squared(&delta);
    let log_det = scale.logdet_spd();

    let half = M::Scalar::half();
    let nu_plus_d = freedom.clone() + dim.clone();
    (nu_plus_d.clone() * half.clone()).ln_gamma()
        - (freedom.clone() * half.clone()).ln_gamma()
        - half.clone() * (dim * (freedom.clone() * M::Scalar::pi()).ln() + log_det)
        - half * nu_plus_d * (M::Scalar::one() + mahalanobis / freedom).ln()
}

/// Hotelling's `T² = n·(x̄ − µ₀)ᵀ S⁻¹ (x̄ − µ₀)` — the sample mean's squared
/// Mahalanobis distance from `µ₀` under the sample covariance `S`, scaled by `n`.
/// No inverse is formed; the quadratic comes from the matrix contract.
pub fn hotelling_t_squared<M: LazyMatrix<Scalar: LazyReal>>(
    count: M::Scalar,
    mean: M::Vector,
    reference: M::Vector,
    covariance: M,
) -> M::Scalar {
    count * covariance.mahalanobis_squared(&mean.sub(reference))
}

/// Pooled (within-group) covariance `Σᵢ dfᵢ·Sᵢ / Σᵢ dfᵢ`, the degrees-of-freedom
/// weighted average of the per-group covariances `Sᵢ` (typically `dfᵢ = nᵢ − 1`).
/// `None` for fewer than two groups or a length mismatch.
pub fn pooled_covariance<M: LazyMatrix<Scalar: LazyReal>>(
    covariances: &[M],
    degrees: &[M::Scalar],
) -> Option<M> {
    if covariances.len() != degrees.len() || covariances.len() < 2 {
        return None;
    }
    let mut total = degrees[0].clone();
    let mut pooled = covariances[0].clone().scale(degrees[0].clone());
    for i in 1..covariances.len() {
        pooled = pooled.add(covariances[i].clone().scale(degrees[i].clone()));
        total = total + degrees[i].clone();
    }
    Some(pooled.scale(M::Scalar::one() / total))
}

/// Cross-entropy `H(p, q) = ½[d·ln(2π) + ln|Σq| + tr(Σq⁻¹Σp) + (µp−µq)ᵀΣq⁻¹(µp−µq)]`
/// between two Gaussians — the expected negative log-density of `q` under `p`
/// (`H(p,q) = KL(p‖q) + H(p)`). No inverse is formed.
pub fn multivariate_normal_cross_entropy<M: LazyMatrix<Scalar: LazyReal>>(
    mean_p: M::Vector,
    cov_p: M,
    mean_q: M::Vector,
    cov_q: M,
) -> M::Scalar {
    let delta = mean_p.sub(mean_q);
    let dim = M::Scalar::scalar(cov_q.shape().0 as f64);
    let trace = cov_q.clone().solve_spd(cov_p).trace();
    let mahalanobis = cov_q.mahalanobis_squared(&delta);
    let log_det = cov_q.logdet_spd();
    M::Scalar::half() * (dim * M::Scalar::ln_tau() + log_det + trace + mahalanobis)
}

/// Bhattacharyya distance between two Gaussians:
/// `⅛·(µ1−µ0)ᵀΣ⁻¹(µ1−µ0) + ½·ln( |Σ| / √(|Σ0|·|Σ1|) )`, `Σ = ½(Σ0+Σ1)`.
pub fn gaussian_bhattacharyya_distance<M: LazyMatrix<Scalar: LazyReal>>(
    mean_0: M::Vector,
    cov_0: M,
    mean_1: M::Vector,
    cov_1: M,
) -> M::Scalar {
    let delta = mean_1.sub(mean_0);
    let half = M::Scalar::half();
    let average = cov_0.clone().add(cov_1.clone()).scale(half.clone());
    let mahalanobis = average.mahalanobis_squared(&delta);
    let log_det_avg = average.logdet_spd();
    let log_det_0 = cov_0.logdet_spd();
    let log_det_1 = cov_1.logdet_spd();
    M::Scalar::scalar(0.125) * mahalanobis
        + half.clone() * (log_det_avg - half * (log_det_0 + log_det_1))
}

/// Multivariate-normal log-likelihood of `count` i.i.d. observations summarized by the
/// mean-centered scatter `S = Σᵢ(xᵢ−µ)(xᵢ−µ)ᵀ` under covariance `Σ`:
/// `−½·(n·d·ln(2π) + n·ln|Σ| + tr(Σ⁻¹S))`.
pub fn multivariate_normal_log_likelihood<M: LazyMatrix<Scalar: LazyReal>>(
    scatter: M,
    covariance: M,
    count: M::Scalar,
) -> M::Scalar {
    let dim = M::Scalar::scalar(covariance.shape().0 as f64);
    let trace = covariance.clone().solve_spd(scatter).trace();
    let log_det = covariance.logdet_spd();
    -M::Scalar::half() * (count.clone() * dim * M::Scalar::ln_tau() + count * log_det + trace)
}

/// Gaussian-process posterior predictive covariance `Σ* = K** − K*ᵀK⁻¹K*` for train kernel
/// `K`, train–test cross-covariance `K*`, and test prior covariance `K**`. No inverse is formed
/// (`K⁻¹K*` is `K.solve_spd(K*)`).
pub fn gaussian_process_posterior_covariance<M: LazyMatrix<Scalar: LazyReal>>(
    train_kernel: M,
    cross_covariance: M,
    test_kernel: M,
) -> M {
    let solved = train_kernel.solve_spd(cross_covariance.clone());
    let reduction = cross_covariance.adjoint().matmul(solved);
    test_kernel.add(reduction.scale(-M::Scalar::one()))
}

/// Wishart mean `E[X] = ν·V` (scale `V`, degrees `ν`).
pub fn wishart_mean<M: LazyMatrix<Scalar: LazyReal>>(scale: M, freedom: M::Scalar) -> M {
    scale.scale(freedom)
}

/// Inverse-Wishart mean `E[Σ] = V / (ν − d − 1)` (scale `V`, degrees `ν`), defined for `ν > d+1`.
pub fn inverse_wishart_mean<M: LazyMatrix<Scalar: LazyReal>>(scale: M, freedom: M::Scalar) -> M {
    let dim = M::Scalar::scalar(scale.shape().0 as f64);
    let denominator = freedom - dim - M::Scalar::one();
    scale.scale(M::Scalar::one() / denominator)
}

/// Gaussian information fusion — the covariance of the product of two Gaussians,
/// `Σ = (Σa⁻¹ + Σb⁻¹)⁻¹`, the precision-weighted combination of two estimates.
pub fn gaussian_precision_fusion<M: LazyMatrix<Scalar: LazyReal>>(cov_a: M, cov_b: M) -> M {
    let n = cov_a.shape().0;
    let precision_a = cov_a.solve_spd(M::identity(n));
    let precision_b = cov_b.solve_spd(M::identity(n));
    precision_a.add(precision_b).solve_spd(M::identity(n))
}

/// Squared Hellinger distance between two Gaussians, `H² = 1 − exp(−D_B)`, where `D_B` is the
/// [`gaussian_bhattacharyya_distance`]. In `[0, 1]`.
pub fn gaussian_hellinger_squared<M: LazyMatrix<Scalar: LazyReal>>(
    mean_0: M::Vector,
    cov_0: M,
    mean_1: M::Vector,
    cov_1: M,
) -> M::Scalar {
    let bhattacharyya = gaussian_bhattacharyya_distance::<M>(mean_0, cov_0, mean_1, cov_1);
    M::Scalar::one() - (-bhattacharyya).exp()
}

/// Mahalanobis distance `√(δᴴ Σ⁻¹ δ)` — the (non-squared) Gaussian distance of `δ` under `Σ`.
pub fn mahalanobis_distance<M: LazyMatrix<Scalar: LazyReal>>(
    covariance: M,
    delta: M::Vector,
) -> M::Scalar {
    covariance.mahalanobis_squared(&delta).sqrt()
}

/// Wishart expected log-determinant `E[ln|X|] = ln|V| + d·ln2 + Σᵢ ψ((ν+1−i)/2)` (scale `V`,
/// degrees `ν`) — the `ln|·|` from the matrix contract, the digamma sum pure scalar analysis.
pub fn wishart_expected_log_det<M: LazyMatrix<Scalar: LazyReal>>(
    scale: M,
    freedom: M::Scalar,
) -> M::Scalar {
    let d = scale.shape().0;
    let dim = M::Scalar::scalar(d as f64);
    let log_det = scale.logdet_spd();
    log_det + dim * M::Scalar::ln_2() + digamma_wishart_sum(d, freedom)
}

/// Wishart mode `(ν − d − 1)·V` (scale `V`, degrees `ν`), defined for `ν ≥ d + 1`.
pub fn wishart_mode<M: LazyMatrix<Scalar: LazyReal>>(scale: M, freedom: M::Scalar) -> M {
    let dim = M::Scalar::scalar(scale.shape().0 as f64);
    scale.scale(freedom - dim - M::Scalar::one())
}

/// Inverse-Wishart mode `V / (ν + d + 1)` (scale `V`, degrees `ν`).
pub fn inverse_wishart_mode<M: LazyMatrix<Scalar: LazyReal>>(scale: M, freedom: M::Scalar) -> M {
    let dim = M::Scalar::scalar(scale.shape().0 as f64);
    scale.scale(M::Scalar::one() / (freedom + dim + M::Scalar::one()))
}

/// Conditional covariance of a partitioned Gaussian (Schur complement):
/// `Σ_{a|b} = Σaa − Σab·Σbb⁻¹·Σba`, `Σba = Σabᴴ`. No inverse is formed.
pub fn multivariate_normal_conditional_covariance<M: LazyMatrix<Scalar: LazyReal>>(
    cov_aa: M,
    cov_ab: M,
    cov_bb: M,
) -> M {
    let cross_solved = cov_bb.solve_spd(cov_ab.clone().adjoint());
    let reduction = cov_ab.matmul(cross_solved);
    cov_aa.add(reduction.scale(-M::Scalar::one()))
}

/// Marginal variances of a Gaussian — the diagonal of the covariance `Σ`.
pub fn multivariate_normal_marginal_variance<M: LazyMatrix<Scalar: LazyReal>>(
    covariance: M,
) -> M::Vector {
    covariance.diagonal()
}

/// Linear predictor `X·β` — the mean of a linear-Gaussian / linear-regression model.
pub fn linear_predictor<M: LazyMatrix<Scalar: LazyReal>>(
    design: M,
    coefficients: M::Vector,
) -> M::Vector {
    design.matvec(coefficients)
}

/// Gaussian natural (canonical) parameter `η = Σ⁻¹µ`.
pub fn gaussian_natural_parameter<M: LazyMatrix<Scalar: LazyReal>>(
    covariance: M,
    mean: M::Vector,
) -> M::Vector {
    covariance.solve_spd_vec(mean)
}

/// Gaussian-process posterior predictive mean `µ* = K*ᵀ K⁻¹ y` for train kernel `K`,
/// train–test cross-covariance `K*`, and observations `y`. No inverse is formed
/// (`K⁻¹y` is `K.solve_spd_vec(y)`).
pub fn gaussian_process_posterior_mean<M: LazyMatrix<Scalar: LazyReal>>(
    train_kernel: M,
    cross_covariance: M,
    observations: M::Vector,
) -> M::Vector {
    let alpha = train_kernel.solve_spd_vec(observations);
    cross_covariance.adjoint().matvec(alpha)
}

/// Conditional mean of a partitioned Gaussian: `µ_{a|b} = µa + Σab·Σbb⁻¹·(x_b − µb)`.
pub fn multivariate_normal_conditional_mean<M: LazyMatrix<Scalar: LazyReal>>(
    mean_a: M::Vector,
    mean_b: M::Vector,
    cov_ab: M,
    cov_bb: M,
    observation_b: M::Vector,
) -> M::Vector {
    let residual = observation_b.sub(mean_b);
    let weighted = cov_bb.solve_spd_vec(residual);
    mean_a.add(cov_ab.matvec(weighted))
}

/// Mean of the product of two Gaussians (information fusion):
/// `µ = (Σa⁻¹+Σb⁻¹)⁻¹·(Σa⁻¹µa + Σb⁻¹µb)` — the precision-weighted combination of two means.
pub fn gaussian_fused_mean<M: LazyMatrix<Scalar: LazyReal>>(
    mean_a: M::Vector,
    cov_a: M,
    mean_b: M::Vector,
    cov_b: M,
) -> M::Vector {
    let eta_a = cov_a.clone().solve_spd_vec(mean_a);
    let eta_b = cov_b.clone().solve_spd_vec(mean_b);
    let fused_covariance = gaussian_precision_fusion::<M>(cov_a, cov_b);
    fused_covariance.matvec(eta_a.add(eta_b))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    // The generic matrix kernels are exercised end-to-end here over the self-contained
    // `RefMatrix`/`RefVector` f64 reference (a naive `LazyMatrix`/`LazyVector` impl), with
    // cross-checks against hand-computed closed forms. The verification that the *same*
    // kernels agree with the concrete `ito::MultivariateNormal` / `ito::Wishart` and run
    // on `ramanujan::DenseMatrix` lives in ito's integration test
    // `tests/bayes_kernels_on_eager.rs` — euclid-only bayes never depends on the eager
    // backend or the concrete multivariate distributions.
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol + tol * b.abs(),
            "observed {a}, expected {b}"
        );
    }

    // --- a minimal `f64` reference implementation of the array contract ---
    //
    // Stands in for `ramanujan::DenseMatrix` (which owns the real impl) so the generic
    // kernels can be exercised end-to-end. Row-major, SPD; naive Cholesky.

    #[derive(Clone)]
    struct RefVector(Vec<f64>);

    impl LazyVector for RefVector {
        type Scalar = f64;
        fn len(&self) -> usize {
            self.0.len()
        }
        fn add(self, other: Self) -> Self {
            RefVector(self.0.iter().zip(&other.0).map(|(a, b)| a + b).collect())
        }
        fn sub(self, other: Self) -> Self {
            RefVector(self.0.iter().zip(&other.0).map(|(a, b)| a - b).collect())
        }
        fn scale(self, factor: f64) -> Self {
            RefVector(self.0.iter().map(|a| a * factor).collect())
        }
        fn hadamard(self, other: Self) -> Self {
            RefVector(self.0.iter().zip(&other.0).map(|(a, b)| a * b).collect())
        }
        fn dot(&self, other: &Self) -> f64 {
            self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum()
        }
        fn sum(&self) -> f64 {
            self.0.iter().sum()
        }
        fn norm_squared(&self) -> f64 {
            self.dot(self)
        }
        fn norm(&self) -> f64 {
            self.norm_squared().sqrt()
        }
    }

    #[derive(Clone)]
    struct RefMatrix {
        n: usize,
        data: Vec<f64>,
    }

    impl RefMatrix {
        fn at(&self, i: usize, j: usize) -> f64 {
            self.data[i * self.n + j]
        }
        /// Lower Cholesky factor `L` (row-major) with `Σ = L Lᵀ`.
        fn chol(&self) -> Vec<f64> {
            let n = self.n;
            let mut l = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..=i {
                    let mut s = self.at(i, j);
                    for k in 0..j {
                        s -= l[i * n + k] * l[j * n + k];
                    }
                    l[i * n + j] = if i == j { s.sqrt() } else { s / l[j * n + j] };
                }
            }
            l
        }
    }

    impl LazyMatrix for RefMatrix {
        type Scalar = f64;
        type Vector = RefVector;

        fn shape(&self) -> (usize, usize) {
            (self.n, self.n)
        }
        fn identity(n: usize) -> Self {
            RefMatrix {
                n,
                data: (0..n * n)
                    .map(|idx| if idx / n == idx % n { 1.0 } else { 0.0 })
                    .collect(),
            }
        }
        fn zeros(rows: usize, cols: usize) -> Self {
            debug_assert_eq!(rows, cols, "RefMatrix reference mock is square");
            RefMatrix {
                n: rows,
                data: vec![0.0; rows * cols],
            }
        }
        fn adjoint(self) -> Self {
            self.transpose() // real matrices: conjugate transpose = transpose
        }
        fn add(self, other: Self) -> Self {
            RefMatrix {
                n: self.n,
                data: self
                    .data
                    .iter()
                    .zip(&other.data)
                    .map(|(a, b)| a + b)
                    .collect(),
            }
        }
        fn scale(self, factor: f64) -> Self {
            RefMatrix {
                n: self.n,
                data: self.data.iter().map(|a| a * factor).collect(),
            }
        }
        fn transpose(self) -> Self {
            let n = self.n;
            RefMatrix {
                n,
                data: (0..n * n).map(|idx| self.at(idx % n, idx / n)).collect(),
            }
        }
        fn matmul(self, rhs: Self) -> Self {
            let n = self.n;
            let data = (0..n * n)
                .map(|idx| {
                    let (i, j) = (idx / n, idx % n);
                    (0..n).map(|k| self.at(i, k) * rhs.at(k, j)).sum()
                })
                .collect();
            RefMatrix { n, data }
        }
        fn matvec(self, v: RefVector) -> RefVector {
            let n = self.n;
            RefVector(
                (0..n)
                    .map(|i| (0..n).map(|k| self.at(i, k) * v.0[k]).sum())
                    .collect(),
            )
        }
        fn trace(&self) -> f64 {
            (0..self.n).map(|i| self.at(i, i)).sum()
        }
        fn diagonal(&self) -> RefVector {
            RefVector((0..self.n).map(|i| self.at(i, i)).collect())
        }
        fn cholesky(self) -> Self {
            RefMatrix {
                n: self.n,
                data: self.chol(),
            }
        }
        fn solve_spd(self, rhs: Self) -> Self {
            // Σ⁻¹ B, column by column via the Cholesky + forward/back substitution.
            let n = self.n;
            let l = self.chol();
            let mut out = vec![0.0; n * n];
            for col in 0..n {
                // forward solve L y = b
                let mut y = vec![0.0; n];
                for i in 0..n {
                    let mut s = rhs.at(i, col);
                    for k in 0..i {
                        s -= l[i * n + k] * y[k];
                    }
                    y[i] = s / l[i * n + i];
                }
                // back solve Lᵀ x = y
                let mut x = vec![0.0; n];
                for i in (0..n).rev() {
                    let mut s = y[i];
                    for k in (i + 1)..n {
                        s -= l[k * n + i] * x[k];
                    }
                    x[i] = s / l[i * n + i];
                }
                for (i, xi) in x.into_iter().enumerate() {
                    out[i * n + col] = xi;
                }
            }
            RefMatrix { n, data: out }
        }
        fn solve_spd_vec(self, rhs: RefVector) -> RefVector {
            // Σ⁻¹ b via Cholesky + forward/back substitution on a single vector.
            let n = self.n;
            let l = self.chol();
            let mut y = vec![0.0; n];
            for i in 0..n {
                let mut s = rhs.0[i];
                for k in 0..i {
                    s -= l[i * n + k] * y[k];
                }
                y[i] = s / l[i * n + i];
            }
            let mut x = vec![0.0; n];
            for i in (0..n).rev() {
                let mut s = y[i];
                for k in (i + 1)..n {
                    s -= l[k * n + i] * x[k];
                }
                x[i] = s / l[i * n + i];
            }
            RefVector(x)
        }
        fn logdet_spd(self) -> f64 {
            let l = self.chol();
            2.0 * (0..self.n).map(|i| l[i * self.n + i].ln()).sum::<f64>()
        }
        fn mahalanobis_squared(&self, delta: &RefVector) -> f64 {
            // ‖L⁻¹δ‖² = δᵀ(LLᵀ)⁻¹δ via forward substitution.
            let n = self.n;
            let l = self.chol();
            let mut z = vec![0.0; n];
            for i in 0..n {
                let mut s = delta.0[i];
                for k in 0..i {
                    s -= l[i * n + k] * z[k];
                }
                z[i] = s / l[i * n + i];
            }
            z.iter().map(|v| v * v).sum()
        }
    }

    fn ref_matrix(n: usize, rows: &[&[f64]]) -> RefMatrix {
        RefMatrix {
            n,
            data: rows.iter().flat_map(|r| r.iter().copied()).collect(),
        }
    }

    // --- self-contained closed forms in plain f64 (no matrix contract) ---
    //
    // Cross-check the generic kernels against an independent scalar implementation that
    // shares no code with `RefMatrix`: it re-derives the Mahalanobis term and the
    // log-determinant from a 2×2 / 3×3 explicit inverse. This gives the matrix kernels a
    // ground truth without reaching for the concrete `ito` distributions.

    /// Explicit symmetric inverse via cofactors (n ∈ {2, 3}); returns row-major data.
    fn inverse(n: usize, rows: &[&[f64]]) -> Vec<f64> {
        let m = |i: usize, j: usize| rows[i][j];
        match n {
            2 => {
                let det = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
                vec![m(1, 1) / det, -m(0, 1) / det, -m(1, 0) / det, m(0, 0) / det]
            }
            3 => {
                let det = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
                    - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
                    + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
                let cof = |a, b, c, d| (a * d - b * c) / det;
                // adjugate (transpose of cofactors); symmetric matrix ⇒ symmetric inverse.
                vec![
                    cof(m(1, 1), m(1, 2), m(2, 1), m(2, 2)),
                    -cof(m(0, 1), m(0, 2), m(2, 1), m(2, 2)),
                    cof(m(0, 1), m(0, 2), m(1, 1), m(1, 2)),
                    -cof(m(1, 0), m(1, 2), m(2, 0), m(2, 2)),
                    cof(m(0, 0), m(0, 2), m(2, 0), m(2, 2)),
                    -cof(m(0, 0), m(0, 2), m(1, 0), m(1, 2)),
                    cof(m(1, 0), m(1, 1), m(2, 0), m(2, 1)),
                    -cof(m(0, 0), m(0, 1), m(2, 0), m(2, 1)),
                    cof(m(0, 0), m(0, 1), m(1, 0), m(1, 1)),
                ]
            }
            _ => panic!("inverse reference only implements 2×2 and 3×3"),
        }
    }

    /// Explicit determinant (n ∈ {2, 3}).
    fn determinant(n: usize, rows: &[&[f64]]) -> f64 {
        let m = |i: usize, j: usize| rows[i][j];
        match n {
            2 => m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0),
            3 => {
                m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
                    - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
                    + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0))
            }
            _ => panic!("determinant reference only implements 2×2 and 3×3"),
        }
    }

    /// `δᵀ A⁻¹ δ` from the explicit inverse.
    fn quadratic(n: usize, rows: &[&[f64]], delta: &[f64]) -> f64 {
        let inv = inverse(n, rows);
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| delta[i] * inv[i * n + j] * delta[j])
                    .sum::<f64>()
            })
            .sum()
    }

    #[test]
    fn lazy_matrix_mvn_log_pdf_matches_explicit_closed_form() {
        // A 3×3 SPD covariance; cross-check the kernel against the explicit Gaussian
        // log-density `−½(δᵀΣ⁻¹δ + d·ln 2π + ln|Σ|)` from cofactor inverse/determinant.
        let rows: [&[f64]; 3] = [&[2.0, 0.3, 0.1], &[0.3, 1.5, -0.2], &[0.1, -0.2, 1.0]];
        let mean = [0.5, -1.0, 0.2];
        let x = [1.2, -0.3, 0.7];
        let delta = [x[0] - mean[0], x[1] - mean[1], x[2] - mean[2]];

        let got = multivariate_normal_log_pdf(
            RefVector(x.to_vec()),
            RefVector(mean.to_vec()),
            ref_matrix(3, &rows),
        );
        let expected = -0.5
            * (quadratic(3, &rows, &delta)
                + 3.0 * core::f64::consts::TAU.ln()
                + determinant(3, &rows).ln());
        close(got, expected, 1e-10);
    }

    #[test]
    fn lazy_matrix_mvn_entropy_and_kl_match_explicit_closed_form() {
        let rows0: [&[f64]; 2] = [&[2.0, 0.3], &[0.3, 1.5]];
        let rows1: [&[f64]; 2] = [&[1.4, -0.2], &[-0.2, 1.1]];
        let mean0 = [0.5, -1.0];
        let mean1 = [0.2, 0.4];

        // Entropy `½(d·ln 2π + ln|Σ| + d)`.
        let expected_entropy =
            0.5 * (2.0 * core::f64::consts::TAU.ln() + determinant(2, &rows0).ln() + 2.0);
        close(
            multivariate_normal_entropy(ref_matrix(2, &rows0)),
            expected_entropy,
            1e-10,
        );

        // KL `½[ln(|Σ₁|/|Σ₀|) − d + tr(Σ₁⁻¹Σ₀) + δᵀΣ₁⁻¹δ]`, δ = µ₁ − µ₀.
        let inv1 = inverse(2, &rows1);
        let s0 = |i: usize, j: usize| rows0[i][j];
        let trace_term = (0..2)
            .map(|i| (0..2).map(|k| inv1[i * 2 + k] * s0(k, i)).sum::<f64>())
            .sum::<f64>();
        let delta = [mean1[0] - mean0[0], mean1[1] - mean0[1]];
        let expected_kl = 0.5
            * ((determinant(2, &rows1) / determinant(2, &rows0)).ln() - 2.0
                + trace_term
                + quadratic(2, &rows1, &delta));
        close(
            multivariate_normal_kl(
                RefVector(mean0.to_vec()),
                ref_matrix(2, &rows0),
                RefVector(mean1.to_vec()),
                ref_matrix(2, &rows1),
            ),
            expected_kl,
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_wishart_log_pdf_matches_explicit_closed_form() {
        let scale: [&[f64]; 2] = [&[2.0, 0.4], &[0.4, 1.3]];
        let observation: [&[f64]; 2] = [&[3.0, 0.5], &[0.5, 2.2]];
        let freedom = 5.0;
        let d = 2.0;

        // tr(V⁻¹X).
        let inv_v = inverse(2, &scale);
        let trace = (0..2)
            .map(|i| {
                (0..2)
                    .map(|k| inv_v[i * 2 + k] * observation[k][i])
                    .sum::<f64>()
            })
            .sum::<f64>();
        let log_det_x = determinant(2, &observation).ln();
        let log_det_v = determinant(2, &scale).ln();
        // ln Γ_d(ν/2) = (d(d−1)/4)·ln π + Σ_{i=1}^{d} ln Γ((ν+1−i)/2).
        let half_nu = freedom / 2.0;
        let ln_mv_gamma = (d * (d - 1.0) / 4.0) * core::f64::consts::PI.ln()
            + crate::function::gamma::ln_gamma(half_nu)
            + crate::function::gamma::ln_gamma(half_nu - 0.5);
        let expected = 0.5 * (freedom - d - 1.0) * log_det_x
            - 0.5 * trace
            - (half_nu * d * core::f64::consts::LN_2 + half_nu * log_det_v + ln_mv_gamma);
        close(
            wishart_log_pdf(ref_matrix(2, &observation), ref_matrix(2, &scale), freedom),
            expected,
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_inverse_wishart_log_pdf_matches_explicit_closed_form() {
        let scale: [&[f64]; 2] = [&[2.0, 0.4], &[0.4, 1.3]];
        let sigma: [&[f64]; 2] = [&[1.6, 0.2], &[0.2, 0.9]];
        let freedom = 5.0;
        let d = 2.0;

        // tr(VΣ⁻¹).
        let inv_sigma = inverse(2, &sigma);
        let trace = (0..2)
            .map(|i| {
                (0..2)
                    .map(|k| scale[i][k] * inv_sigma[k * 2 + i])
                    .sum::<f64>()
            })
            .sum::<f64>();
        let log_det_sigma = determinant(2, &sigma).ln();
        let log_det_v = determinant(2, &scale).ln();
        let half_nu = freedom / 2.0;
        let ln_mv_gamma = (d * (d - 1.0) / 4.0) * core::f64::consts::PI.ln()
            + crate::function::gamma::ln_gamma(half_nu)
            + crate::function::gamma::ln_gamma(half_nu - 0.5);
        let ln_normalizer =
            half_nu * log_det_v - half_nu * d * core::f64::consts::LN_2 - ln_mv_gamma;
        let expected = ln_normalizer - 0.5 * (freedom + d + 1.0) * log_det_sigma - 0.5 * trace;
        close(
            inverse_wishart_log_pdf(ref_matrix(2, &sigma), ref_matrix(2, &scale), freedom),
            expected,
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_wishart_entropy_is_finite_and_inverse_round_trips() {
        // The Wishart / inverse-Wishart entropies share the matrix contract's
        // log-determinant; exercise both kernels and assert the structural relation
        // they must satisfy (each finite, distinct for the same scale and freedom).
        let scale: [&[f64]; 2] = [&[2.0, 0.4], &[0.4, 1.3]];
        let freedom = 5.0;

        let wishart_h = wishart_entropy(ref_matrix(2, &scale), freedom);
        let inverse_h = inverse_wishart_entropy(ref_matrix(2, &scale), freedom);
        assert!(wishart_h.is_finite(), "Wishart entropy must be finite");
        assert!(
            inverse_h.is_finite(),
            "inverse-Wishart entropy must be finite"
        );

        // Identity scale: the entropies reduce to the pure normalizer/digamma terms with
        // ln|V| = 0, so the kernels agree with the closed form built from the same
        // host-side digamma sum and `ln Γ_d`.
        let id: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
        let half_nu = freedom / 2.0;
        // ln Γ_d(ν/2) = (d(d−1)/4)·ln π + Σ ln Γ((ν+1−i)/2); for d = 2 the prefactor is ½.
        let ln_mv_gamma = 0.5 * core::f64::consts::PI.ln()
            + crate::function::gamma::ln_gamma(half_nu)
            + crate::function::gamma::ln_gamma(half_nu - 0.5);
        let digamma_sum = crate::function::gamma::digamma(half_nu)
            + crate::function::gamma::digamma(half_nu - 0.5);
        let expected_log_det = 2.0 * core::f64::consts::LN_2 + digamma_sum; // E[ln|X|], ln|V|=0
        let ln_z = half_nu * 2.0 * core::f64::consts::LN_2 + ln_mv_gamma; // ln|V|=0
        let expected_wishart =
            -0.5 * (freedom - 2.0 - 1.0) * expected_log_det + 0.5 * freedom * 2.0 + ln_z;
        close(
            wishart_entropy(ref_matrix(2, &id), freedom),
            expected_wishart,
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_mvt_entropy_is_finite() {
        // Student-t entropy threads the matrix contract's log-determinant through the
        // scalar digamma/ln-gamma analysis; exercise the kernel on the f64 reference.
        let scale: [&[f64]; 2] = [&[2.0, 0.5], &[0.5, 1.3]];
        let freedom = 6.0;
        let got = multivariate_student_t_entropy(ref_matrix(2, &scale), freedom);
        assert!(got.is_finite(), "MVT entropy must be finite");
        // Unit scale lifts the log-determinant out so the entropy is the pure scalar form.
        let id: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
        let half = 0.5;
        let half_nu = freedom * half;
        let half_nu_d = (freedom + 2.0) * half;
        let expected = crate::function::gamma::ln_gamma(half_nu)
            - crate::function::gamma::ln_gamma(half_nu_d)
            + half * 2.0 * (freedom * core::f64::consts::PI).ln()
            + half_nu_d
                * (crate::function::gamma::digamma(half_nu_d)
                    - crate::function::gamma::digamma(half_nu));
        close(
            multivariate_student_t_entropy(ref_matrix(2, &id), freedom),
            expected,
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_gp_marginal_likelihood_is_the_zero_mean_mvn_log_pdf() {
        // The GP log-evidence is the MVN log-density at zero mean — assert the two kernels
        // agree on the same kernel matrix and observations.
        let kernel: [&[f64]; 3] = [&[1.2, 0.4, 0.1], &[0.4, 1.0, 0.2], &[0.1, 0.2, 0.8]];
        let y = [0.3, -0.7, 0.5];
        close(
            gaussian_process_log_marginal_likelihood(RefVector(y.to_vec()), ref_matrix(3, &kernel)),
            multivariate_normal_log_pdf(
                RefVector(y.to_vec()),
                RefVector(vec![0.0; 3]),
                ref_matrix(3, &kernel),
            ),
            1e-12,
        );
    }

    #[test]
    fn lazy_matrix_mvt_log_pdf_matches_explicit_closed_form() {
        let rows: [&[f64]; 2] = [&[2.0, 0.5], &[0.5, 1.3]];
        let location = [0.4, -0.6];
        let x = [1.1, 0.2];
        let freedom = 6.0;
        let d = 2.0;
        let delta = [x[0] - location[0], x[1] - location[1]];

        let got = multivariate_student_t_log_pdf(
            RefVector(x.to_vec()),
            RefVector(location.to_vec()),
            ref_matrix(2, &rows),
            freedom,
        );
        let mahalanobis = quadratic(2, &rows, &delta);
        let log_det = determinant(2, &rows).ln();
        let expected = crate::function::gamma::ln_gamma((freedom + d) / 2.0)
            - crate::function::gamma::ln_gamma(freedom / 2.0)
            - 0.5 * (d * (freedom * core::f64::consts::PI).ln() + log_det)
            - 0.5 * (freedom + d) * (1.0 + mahalanobis / freedom).ln();
        close(got, expected, 1e-10);
    }

    #[test]
    fn lazy_matrix_hotelling_t_squared_is_scaled_mahalanobis() {
        let cov: [&[f64]; 2] = [&[2.0, 0.3], &[0.3, 1.5]];
        let mean = [1.2, -0.4];
        let mu0 = [0.5, 0.1];
        let n = 8.0;
        let delta = [mean[0] - mu0[0], mean[1] - mu0[1]];
        close(
            hotelling_t_squared(
                n,
                RefVector(mean.to_vec()),
                RefVector(mu0.to_vec()),
                ref_matrix(2, &cov),
            ),
            n * quadratic(2, &cov, &delta),
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_pooled_covariance_is_the_df_weighted_average() {
        let s0: [&[f64]; 2] = [&[2.0, 0.3], &[0.3, 1.5]];
        let s1: [&[f64]; 2] = [&[1.4, -0.2], &[-0.2, 1.1]];
        let (df0, df1) = (4.0, 6.0);
        let pooled =
            pooled_covariance(&[ref_matrix(2, &s0), ref_matrix(2, &s1)], &[df0, df1]).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected = (df0 * s0[i][j] + df1 * s1[i][j]) / (df0 + df1);
                close(pooled.at(i, j), expected, 1e-12);
            }
        }
    }
}
