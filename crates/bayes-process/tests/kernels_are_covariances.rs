//! **Covariance kernels**: are they actually covariances?
//!
//! A kernel is not just a similarity function. To be a covariance it must be
//! symmetric and **positive semi-definite**: `vᵀKv ≥ 0` for every vector `v`
//! and every set of input points. That is the property a Gaussian process
//! needs and the one a plausible-looking formula can fail — a Matérn with the
//! wrong polynomial prefactor, or a periodic kernel missing a square, is still
//! smooth, still decays, still looks like a kernel, and produces a covariance
//! matrix with a negative eigenvalue.
//!
//! PSD is also the hardest property to get by inspection, so it is checked
//! directly here: Gram matrices on deliberately awkward point sets, contracted
//! against many vectors including ones designed to find a negative direction.

#![allow(clippy::panic, clippy::unwrap_used)]

use bayes_process::{Kernel, Linear, Matern, MaternNu, Periodic, Rbf};
use bayes_utils::Failures;

/// Point sets chosen to stress a kernel rather than flatter it: duplicated
/// points (which make the Gram matrix singular), tight clusters, and a wide
/// spread that pushes a stationary kernel to the floor of its range.
fn point_sets() -> Vec<Vec<Vec<f64>>> {
    vec![
        vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
        // Duplicates: the Gram matrix is singular, so PSD must hold with a
        // zero eigenvalue rather than tipping negative.
        vec![vec![1.0], vec![1.0], vec![1.0], vec![2.0]],
        // A tight cluster, where every entry is near the variance.
        vec![vec![0.0], vec![1e-8], vec![2e-8], vec![3e-8]],
        // A wide spread, where a stationary kernel is near zero off-diagonal.
        vec![vec![-1e4], vec![0.0], vec![1e4]],
        // Two dimensions, including a repeated coordinate.
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ],
        vec![
            vec![0.5, -0.5, 2.0],
            vec![-1.0, 0.25, 0.0],
            vec![2.0, 2.0, 2.0],
        ],
    ]
}

/// Contraction vectors, including the alternating and near-null directions that
/// find a negative eigenvalue if one exists.
fn directions(n: usize) -> Vec<Vec<f64>> {
    let mut out = vec![
        vec![1.0; n],
        (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
        (0..n).map(|i| i as f64 - n as f64 / 2.0).collect(),
    ];
    // Each coordinate direction, and each opposed pair — a negative
    // eigenvalue in a singular Gram matrix usually shows up in a difference.
    for i in 0..n {
        let mut e = vec![0.0; n];
        e[i] = 1.0;
        out.push(e);
        for j in (i + 1)..n {
            let mut d = vec![0.0; n];
            d[i] = 1.0;
            d[j] = -1.0;
            out.push(d);
        }
    }
    out
}

/// A named kernel, bound to one hyperparameter set, with its variance.
type KernelCase = (&'static str, Box<dyn Fn(&[f64], &[f64]) -> f64>, f64);

fn kernels() -> Vec<KernelCase> {
    vec![
        (
            "Rbf(2, 1.5)",
            Box::new(|a: &[f64], b: &[f64]| Rbf::new(2.0, 1.5).covariance(a, b)),
            2.0,
        ),
        (
            "Rbf(1, 0.01)",
            Box::new(|a: &[f64], b: &[f64]| Rbf::new(1.0, 0.01).covariance(a, b)),
            1.0,
        ),
        (
            "Matern32(1.5, 2)",
            Box::new(|a: &[f64], b: &[f64]| {
                Matern::new(1.5, 2.0, MaternNu::ThreeHalves).covariance(a, b)
            }),
            1.5,
        ),
        (
            "Matern52(1.5, 2)",
            Box::new(|a: &[f64], b: &[f64]| {
                Matern::new(1.5, 2.0, MaternNu::FiveHalves).covariance(a, b)
            }),
            1.5,
        ),
        (
            "Periodic(1, 1, 2pi)",
            Box::new(|a: &[f64], b: &[f64]| {
                Periodic::new(1.0, 1.0, std::f64::consts::TAU).covariance(a, b)
            }),
            1.0,
        ),
    ]
}

/// `vᵀKv ≥ 0` for every direction, on every point set.
///
/// The tolerance scales with the Gram matrix's own magnitude, because that is
/// what the rounding error of the contraction scales with — a `4×4` matrix of
/// entries near 2 cannot be contracted to better than a few ulps of 32.
#[test]
fn every_kernel_is_positive_semi_definite() {
    let mut f = Failures::default();
    for (name, k, variance) in kernels() {
        for points in point_sets() {
            let n = points.len();
            let gram: Vec<Vec<f64>> = points
                .iter()
                .map(|a| points.iter().map(|b| k(a, b)).collect())
                .collect();
            for v in directions(n) {
                let quadratic: f64 = (0..n)
                    .flat_map(|i| (0..n).map(move |j| (i, j)))
                    .map(|(i, j)| v[i] * gram[i][j] * v[j])
                    .sum();
                let magnitude: f64 = v.iter().map(|x| x.abs()).sum::<f64>().powi(2) * variance;
                f.check(quadratic >= -1e-12 * magnitude.max(1.0), || {
                    format!(
                        "{name}: v^T K v = {quadratic:.6e} for v = {v:?} on {n} points — a \
                         covariance cannot be negative in any direction"
                    )
                });
            }
        }
    }
    f.finish("positive semi-definiteness");
}

/// A kernel is symmetric, equals its variance on the diagonal, and never
/// exceeds it off the diagonal.
///
/// The last is Cauchy–Schwarz: `|k(x,y)| ≤ √(k(x,x)k(y,y))`, which for a
/// stationary kernel is the variance. A kernel that exceeded it would not be
/// PSD, so this is a cheap necessary condition that localizes the failure to a
/// single pair rather than a whole matrix.
#[test]
fn every_kernel_is_symmetric_and_bounded_by_its_variance() {
    let mut f = Failures::default();
    for (name, k, variance) in kernels() {
        for points in point_sets() {
            for a in &points {
                f.close(
                    || format!("{name}: k(x, x) at {a:?}"),
                    k(a, a),
                    variance,
                    1e-12 * variance,
                );
                for b in &points {
                    f.close(
                        || format!("{name}: symmetry at {a:?}, {b:?}"),
                        k(a, b),
                        k(b, a),
                        1e-15 * variance,
                    );
                    f.check(k(a, b).abs() <= variance * (1.0 + 1e-12), || {
                        format!(
                            "{name}: |k({a:?}, {b:?})| = {} exceeds the variance {variance}",
                            k(a, b).abs()
                        )
                    });
                }
            }
        }
    }
    f.finish("symmetry and bounds");
}

/// The stationary kernels decay monotonically, and the Matérn family's
/// ordering against the RBF **reverses** with distance.
///
/// Near zero a smoother kernel stays higher — `RBF ≥ Matérn(5/2) ≥
/// Matérn(3/2)`, since smoothness means slower initial decorrelation. In the
/// tail the order is exactly opposite, because a rougher kernel has a *heavier*
/// tail: the RBF decays like `exp(−d²/2ℓ²)` while Matérn(3/2) decays like
/// `exp(−√3·d/ℓ)`, and an exponential eventually dominates a Gaussian.
///
/// Asserting the near-zero ordering everywhere is the natural mistake, and it
/// fails from about `2ℓ` outward. Checking both regimes *and* that a crossover
/// exists between them is the sharper test — a Matérn with a wrong prefactor
/// loses one of the two orderings while staying smooth, decaying and PSD.
#[test]
fn the_stationary_kernels_decay_and_cross_over() {
    let mut f = Failures::default();
    let (variance, length) = (1.0, 1.0);
    let rbf = Rbf::new(variance, length);
    let m32 = Matern::new(variance, length, MaternNu::ThreeHalves);
    let m52 = Matern::new(variance, length, MaternNu::FiveHalves);
    let mut previous = (variance, variance, variance);
    for step in 0..40 {
        let d = f64::from(step) * 0.25;
        let point = [d];
        let origin = [0.0];
        let (r, a, b) = (
            rbf.covariance(&origin, &point),
            m32.covariance(&origin, &point),
            m52.covariance(&origin, &point),
        );
        f.check(r <= previous.0 + 1e-15, || {
            format!("Rbf rose at distance {d}: {} -> {r}", previous.0)
        });
        f.check(a <= previous.1 + 1e-15, || {
            format!("Matern32 rose at distance {d}: {} -> {a}", previous.1)
        });
        f.check(b <= previous.2 + 1e-15, || {
            format!("Matern52 rose at distance {d}: {} -> {b}", previous.2)
        });
        // Near zero: smoother stays higher.
        if d <= 1.0 {
            f.check(a <= b + 1e-12, || {
                format!("near zero at {d}, Matern32 {a:.12} should not exceed Matern52 {b:.12}")
            });
            f.check(b <= r + 1e-12, || {
                format!("near zero at {d}, Matern52 {b:.12} should not exceed Rbf {r:.12}")
            });
        }
        // In the tail: rougher stays higher, the exact reverse.
        if d >= 3.0 && a > 1e-12 {
            f.check(b <= a + 1e-12, || {
                format!("in the tail at {d}, Matern52 {b:.12} should not exceed Matern32 {a:.12}")
            });
            f.check(r <= b + 1e-12, || {
                format!("in the tail at {d}, Rbf {r:.12} should not exceed Matern52 {b:.12}")
            });
        }
        previous = (r, a, b);
    }

    // And the crossover is real, not an artefact of where the sampling stops.
    let near = [0.5_f64];
    let far = [4.0_f64];
    f.check(
        rbf.covariance(&[0.0], &near) > m32.covariance(&[0.0], &near)
            && rbf.covariance(&[0.0], &far) < m32.covariance(&[0.0], &far),
        || "the Rbf / Matern32 ordering should reverse between 0.5 and 4 length scales".into(),
    );
    f.finish("decay and crossover");
}

/// A periodic kernel repeats, exactly.
///
/// `k(x, x + n·period) = variance` for every integer `n`. This is the only
/// property that distinguishes it from the other stationary kernels, and the
/// one a missing factor of two inside the sine destroys — which otherwise
/// leaves a perfectly well-behaved PSD kernel with the wrong period.
#[test]
fn the_periodic_kernel_repeats_at_its_period() {
    let mut f = Failures::default();
    let period = 2.5;
    let k = Periodic::new(1.3, 0.7, period);
    for n in 1..=6 {
        let shifted = [f64::from(n) * period];
        f.close(
            || format!("Periodic at {n} periods"),
            k.covariance(&[0.0], &shifted),
            1.3,
            1e-10,
        );
        // And a half-period away is the minimum of the cycle.
        let half = [(f64::from(n) - 0.5) * period];
        f.check(k.covariance(&[0.0], &half) < 1.3, || {
            format!(
                "Periodic at {n} and a half periods is {}, which should be below the variance",
                k.covariance(&[0.0], &half)
            )
        });
    }
    f.finish("periodicity");
}

/// A linear kernel is not stationary, and that is the point.
///
/// `k(x, y) = variance·⟨x, y⟩ + bias` grows with the inputs, so none of the
/// stationary assumptions above apply to it. What must still hold is PSD —
/// checked separately below, since the bound used in the general test is a
/// stationarity assumption.
#[test]
fn the_linear_kernel_is_an_inner_product_and_still_positive_semi_definite() {
    let mut f = Failures::default();
    let k = Linear::new(2.0, 0.5);
    // Bilinearity in the first argument.
    for (a, b, c) in [
        (1.0_f64, 2.0_f64, 3.0_f64),
        (-1.0, 0.5, 4.0),
        (0.0, 0.0, 1.0),
    ] {
        let left = k.covariance(&[a + b], &[c]);
        let right = k.covariance(&[a], &[c]) + k.covariance(&[b], &[c]) - 0.5;
        f.close(
            || format!("linear additivity at ({a}, {b}, {c})"),
            left,
            right,
            1e-12,
        );
    }
    // PSD on its own terms.
    for points in point_sets() {
        let n = points.len();
        let gram: Vec<Vec<f64>> = points
            .iter()
            .map(|a| points.iter().map(|b| k.covariance(a, b)).collect())
            .collect();
        for v in directions(n) {
            let quadratic: f64 = (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| v[i] * gram[i][j] * v[j])
                .sum();
            let scale: f64 = gram
                .iter()
                .flatten()
                .map(|x| x.abs())
                .fold(1.0_f64, f64::max);
            f.check(quadratic >= -1e-10 * scale * n as f64, || {
                format!("Linear: v^T K v = {quadratic:.6e} for v = {v:?}")
            });
        }
    }
    f.finish("linear kernel");
}
