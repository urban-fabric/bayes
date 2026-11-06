//! Numerical probes, with no knowledge of bayes.
//!
//! The quadrature here is deliberately not one routine. [`simpson`] is a plain
//! uniform-grid rule, fine when the integrand is smooth over a known interval.
//! [`integrate_unit`] exists because integrating *in quantile space* — which is
//! how a moment or an entropy gets checked against the analytic form — puts
//! endpoint singularities exactly where a uniform grid cannot see them: `Q(u)`
//! behaves like `-ln(1-u)` for an exponential tail and `(1-u)^(-1/α)` for a
//! Pareto one. The substitution it applies turns those into smooth, rapidly
//! decaying integrands.

/// A named density/CDF pair for table-driven sweeps.
///
/// The `dyn` closures let one loop cover families that share no trait bound
/// beyond the two functions actually used, which is what lets a table hold a
/// `Normal` next to a `Wald`.
pub type DensityCase = (
    String,
    Box<dyn Fn(f64) -> f64>,
    Box<dyn Fn(f64) -> f64>,
    f64,
    f64,
);

/// Collects every failure in a sweep so one bad case does not mask the rest.
///
/// A sweep that aborts on its first failure hides however many follow it, which
/// matters most exactly when something systemic has broken: the difference
/// between "one family is wrong" and "thirteen families are wrong in the same
/// way" is the difference between a typo and a defect in a shared kernel.
#[derive(Default)]
pub struct Failures {
    msgs: Vec<String>,
    checked: usize,
}

impl Failures {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a predicate, building the message only on failure.
    pub fn check(&mut self, ok: bool, msg: impl FnOnce() -> String) {
        self.checked += 1;
        if !ok {
            self.msgs.push(msg());
        }
    }

    /// Assert two quantities agree to a relative tolerance, reporting both
    /// values and the achieved error when they do not.
    pub fn close(&mut self, label: impl FnOnce() -> String, a: f64, b: f64, tol: f64) {
        let err = (a - b).abs() / b.abs().max(1.0);
        let ok = err <= tol;
        self.check(ok, || {
            format!(
                "{}: {a:.15e} vs {b:.15e} (rel {err:.3e} > {tol:.0e})",
                label()
            )
        });
    }

    /// Assert agreement on an absolute scale supplied by the caller.
    ///
    /// Use this when the natural yardstick is not the compared value — a mean of
    /// zero on a distribution with standard deviation `1e10` should be judged
    /// against `1e10`, not against `1`.
    pub fn close_on_scale(
        &mut self,
        label: impl FnOnce() -> String,
        a: f64,
        b: f64,
        tol: f64,
        scale: f64,
    ) {
        let ok = (a - b).abs() <= tol * scale;
        self.check(ok, || {
            format!(
                "{}: {a:.15e} vs {b:.15e} (|diff| {:.3e} > {tol:.0e}·{scale:.3e})",
                label(),
                (a - b).abs()
            )
        });
    }

    pub fn failures(&self) -> usize {
        self.msgs.len()
    }

    /// Panic with every failure, or return quietly if there were none.
    pub fn finish(self, label: &str) {
        if !self.msgs.is_empty() {
            panic!(
                "{label}: {} of {} checks failed:\n  {}",
                self.msgs.len(),
                self.checked,
                self.msgs.join("\n  ")
            );
        }
    }
}

/// Composite Simpson over `[a, b]`; `n` must be even.
///
/// Non-finite samples are skipped rather than propagated, so an integrable
/// endpoint singularity contributes approximately nothing instead of poisoning
/// the whole sum with a `NaN`.
pub fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    assert!(n.is_multiple_of(2), "Simpson needs an even interval count");
    if b <= a || !(a.is_finite() && b.is_finite()) {
        return 0.0;
    }
    let h = (b - a) / n as f64;
    let mut acc = 0.0;
    for i in 0..=n {
        let y = f(a + h * i as f64);
        if !y.is_finite() {
            continue;
        }
        let w = if i == 0 || i == n {
            1.0
        } else if i.is_multiple_of(2) {
            2.0
        } else {
            4.0
        };
        acc += w * y;
    }
    acc * h / 3.0
}

/// `∫₀¹ g(u) du` via the substitution `u = (1 + tanh s)/2`, which clusters
/// abscissae exponentially at both endpoints.
///
/// `du = sech²(s)/2 ds` decays like `e^{-2s}`, which tames any endpoint growth
/// lighter than that into a smooth integrand. That is what makes it possible to
/// check `E[X] = ∫₀¹ Q(u) du` for a Pareto or a Student's t, where a uniform grid
/// in `u` puts almost no points where almost all of the variation is.
pub fn integrate_unit(g: impl Fn(f64) -> f64, n: usize) -> f64 {
    const HALF_WIDTH: f64 = 20.0; // tanh(20) is within 4e-18 of 1
    simpson(
        |s| {
            let t = s.tanh();
            let u = 0.5 * (1.0 + t);
            if u <= 0.0 || u >= 1.0 {
                return 0.0;
            }
            let weight = 0.5 * (1.0 - t * t); // sech²(s)/2
            let v = g(u);
            if v.is_finite() { v * weight } else { 0.0 }
        },
        -HALF_WIDTH,
        HALF_WIDTH,
        n,
    )
}

/// A five-point central difference, accurate to `O(h⁴)`.
pub fn derivative(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)
}

/// A five-point second central difference, accurate to `O(h⁴)`.
pub fn second_derivative(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (-f(x + 2.0 * h) + 16.0 * f(x + h) - 30.0 * f(x) + 16.0 * f(x - h) - f(x - 2.0 * h))
        / (12.0 * h * h)
}

/// Relative error against `b`, floored at 1 so a comparison to zero does not
/// divide by it.
pub fn rel_error(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

/// A grid of points spanning a family's interesting range, taken from its own
/// quantiles so the comparison happens where the mass actually is.
pub fn quantile_grid(q: impl Fn(f64) -> f64) -> Vec<f64> {
    [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99]
        .iter()
        .map(|&p| q(p))
        .filter(|x| x.is_finite())
        .collect()
}
