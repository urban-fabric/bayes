//! Ways to build new maps from old, with the Jacobian carried along.
//!
//! Neither of these differentiates anything. Composition chains the pieces'
//! Jacobians by the chain rule; inversion flips one by the inverse function
//! theorem. So a primitive only ever has to know its own derivative once, and
//! every map reachable from it inherits a correct Jacobian.

use super::Bijector;

/// `Outer ∘ Inner`: apply `inner` first, then `outer`.
///
/// # The Jacobian
///
/// With `g = outer ∘ inner`, the inverse is `g⁻¹(y) = inner⁻¹(outer⁻¹(y))`, so
/// the chain rule gives
///
/// ```text
///     d g⁻¹      d inner⁻¹        d outer⁻¹
///     ───── (y) = ───────── (u) · ───────── (y),      u = outer⁻¹(y)
///      dy            du               dy
/// ```
///
/// and taking logs turns the product into the **sum** of the two log-dets. The
/// only thing to get right is that `inner`'s term is evaluated at `u`, the
/// intermediate point, not at `y` — which is the mistake this type exists to
/// make impossible to repeat by hand at every call site.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Affine, Bijector, Exp, LogNormal, Continuous, Normal, Transformed};
///
/// // A log-normal, assembled rather than named: scale and shift a standard
/// // normal, then exponentiate.
/// let built = Transformed::new(
///     Normal::new(0.0, 1.0)?,
///     Affine::new(1.2, 0.5)?.then(Exp),
/// );
/// let named = LogNormal::new(0.5, 1.2)?;
/// assert!((built.pdf(2.0_f64) - named.pdf(2.0_f64)).abs() < 1e-12);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Compose<Inner, Outer> {
    inner: Inner,
    outer: Outer,
}

impl<Inner, Outer> Compose<Inner, Outer> {
    pub fn new(inner: Inner, outer: Outer) -> Self {
        Self { inner, outer }
    }

    pub fn inner(&self) -> &Inner {
        &self.inner
    }
    pub fn outer(&self) -> &Outer {
        &self.outer
    }
}

impl<Inner: Bijector, Outer: Bijector> Bijector for Compose<Inner, Outer> {
    fn forward(&self, x: f64) -> f64 {
        self.outer.forward(self.inner.forward(x))
    }

    fn inverse(&self, y: f64) -> f64 {
        self.inner.inverse(self.outer.inverse(y))
    }

    fn inverse_log_det_jacobian(&self, y: f64) -> f64 {
        let intermediate = self.outer.inverse(y);
        self.inner.inverse_log_det_jacobian(intermediate) + self.outer.inverse_log_det_jacobian(y)
    }
}

/// The map that undoes `B`, with the Jacobian flipped to match.
///
/// # The Jacobian
///
/// For `h = B⁻¹` the inverse is `h⁻¹ = B`, so the term needed is
/// `ln|dB(y)/dy|` — a *forward* derivative, which [`Bijector`] does not expose.
/// The inverse function theorem supplies it anyway: `dB(y)/dy` is the
/// reciprocal of `dB⁻¹(z)/dz` at `z = B(y)`, so
///
/// ```text
///     h.inverse_log_det_jacobian(y) = −B.inverse_log_det_jacobian(B(y))
/// ```
///
/// Both the negation and the evaluation at `B(y)` rather than `y` are load
/// bearing; dropping either gives a density that still integrates to something
/// finite and is still wrong.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Bijector, Exp};
///
/// // Inverting Exp gives the logarithm.
/// let log = Exp.inverted();
/// assert!((log.forward(std::f64::consts::E) - 1.0).abs() < 1e-12);
/// // And inverting twice is the identity.
/// assert!((Exp.inverted().inverted().forward(2.0) - Exp.forward(2.0)).abs() < 1e-12);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Inverse<B> {
    inner: B,
}

impl<B> Inverse<B> {
    pub fn new(inner: B) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &B {
        &self.inner
    }
}

impl<B: Bijector> Bijector for Inverse<B> {
    fn forward(&self, x: f64) -> f64 {
        self.inner.inverse(x)
    }

    fn inverse(&self, y: f64) -> f64 {
        self.inner.forward(y)
    }

    fn inverse_log_det_jacobian(&self, y: f64) -> f64 {
        -self.inner.inverse_log_det_jacobian(self.inner.forward(y))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::super::primitives::{Affine, Exp, Identity, Sigmoid, Softplus};
    use super::super::probe::behaves_like_a_bijector;
    use super::*;
    use crate::{Continuous, LogNormal, Normal, Transformed};

    #[test]
    fn composites_and_inverses_carry_the_jacobian_they_claim() {
        let reals = [-6.0, -1.5, -0.25, 0.0, 0.25, 1.5, 6.0];
        let affine = Affine::new(1.7, -0.4).unwrap();

        behaves_like_a_bijector("Affine.then(Exp)", &affine.then(Exp), &reals);
        behaves_like_a_bijector("Exp.inverted()", &Exp.inverted(), &[0.5, 1.0, 2.0, 30.0]);
        behaves_like_a_bijector("Affine.then(Sigmoid)", &affine.then(Sigmoid), &reals);
        behaves_like_a_bijector("Affine.then(Softplus)", &affine.then(Softplus), &reals);
        // Three deep, mixing scales: the Jacobian has to accumulate across all
        // of them and be evaluated at the right intermediate point each time.
        behaves_like_a_bijector(
            "Affine.then(Exp).then(Affine)",
            &affine.then(Exp).then(Affine::new(0.3, 2.0).unwrap()),
            &reals,
        );
        behaves_like_a_bijector(
            "Softplus.inverted().then(Sigmoid)",
            &Softplus.inverted().then(Sigmoid),
            &[0.1, 0.5, 1.0, 4.0],
        );
    }

    /// `Identity` is the unit on both sides, and inverting twice is a no-op.
    /// These are the algebra's laws; a `Compose` that evaluated its inner
    /// Jacobian at the wrong point would still satisfy the forward half, so the
    /// densities are compared too.
    #[test]
    fn the_algebra_has_the_laws_an_algebra_should() {
        let base = Normal::new(0.3, 1.4).unwrap();
        let plain = Transformed::new(base, Exp);
        let left = Transformed::new(base, Identity.then(Exp));
        let right = Transformed::new(base, Exp.then(Identity));
        let twice = Transformed::new(base, Exp.inverted().inverted());
        for &y in &[0.2, 0.7, 1.0, 3.0, 11.0] {
            let want = plain.ln_pdf(y);
            assert!(
                (left.ln_pdf(y) - want).abs() < 1e-12,
                "Identity ∘ Exp at {y}"
            );
            assert!(
                (right.ln_pdf(y) - want).abs() < 1e-12,
                "Exp ∘ Identity at {y}"
            );
            assert!(
                (twice.ln_pdf(y) - want).abs() < 1e-12,
                "Exp inverted twice at {y}"
            );
        }
    }

    /// Composition is associative, including in the Jacobian.
    ///
    /// The `y` values are obtained by pushing `x` forward rather than written
    /// down, because this particular composite has image `(-inf, 1.1)` — the
    /// negative outer scale flips the positive half-line — and a hand-picked
    /// `y = 2` would compare two `NaN`s and call it a disagreement.
    #[test]
    fn composition_is_associative() {
        let a = Affine::new(1.3, 0.2).unwrap();
        let c = Affine::new(-0.6, 1.1).unwrap();
        let left = a.then(Exp).then(c);
        let right = a.then(Exp.then(c));
        for &x in &[-3.0, -0.5, 0.0, 0.5, 3.0] {
            assert!((left.forward(x) - right.forward(x)).abs() < 1e-12);
            let y = left.forward(x);
            assert!(
                (left.inverse_log_det_jacobian(y) - right.inverse_log_det_jacobian(y)).abs()
                    < 1e-12,
                "log-det disagreed at y={y}"
            );
        }
    }

    /// An assembled log-normal against the named one — the check that says the
    /// accumulated Jacobian is the right *number*, not merely self-consistent.
    #[test]
    fn an_assembled_lognormal_matches_the_named_one() {
        for &(mu, sigma) in &[(0.0, 1.0), (0.5, 1.2), (-1.3, 0.4)] {
            let built = Transformed::new(
                Normal::new(0.0, 1.0).unwrap(),
                Affine::new(sigma, mu).unwrap().then(Exp),
            );
            let named = LogNormal::new(mu, sigma).unwrap();
            for &y in &[0.05, 0.3, 1.0, 2.5, 9.0] {
                assert!(
                    (built.ln_pdf(y) - named.ln_pdf(y)).abs() < 1e-10,
                    "LogNormal({mu}, {sigma}) at {y}: assembled {} vs named {}",
                    built.ln_pdf(y),
                    named.ln_pdf(y)
                );
            }
        }
    }

    /// Pushing a normal through `Sigmoid` gives the logit-normal, whose density
    /// has a known closed form. This is a second independent target for the
    /// composed Jacobian, on a bounded support rather than a half-line.
    #[test]
    fn a_logit_normal_matches_its_closed_form() {
        let (mu, sigma) = (0.4, 0.9);
        let t = Transformed::new(Normal::new(mu, sigma).unwrap(), Sigmoid);
        for &y in &[0.05_f64, 0.25, 0.5, 0.75, 0.95] {
            let logit = (y / (1.0 - y)).ln();
            let closed = (-((logit - mu) / sigma).powi(2) / 2.0).exp()
                / (sigma * (2.0 * std::f64::consts::PI).sqrt() * y * (1.0 - y));
            assert!(
                (t.pdf(y) - closed).abs() < 1e-10,
                "logit-normal at {y}: {} vs {closed}",
                t.pdf(y)
            );
        }
        // And it is a density on (0, 1): integrate it.
        let n = 20_000;
        let mass: f64 = (1..n).map(|i| t.pdf(i as f64 / n as f64) / n as f64).sum();
        assert!((mass - 1.0).abs() < 1e-4, "logit-normal mass is {mass}");
    }
}
