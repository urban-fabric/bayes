//! Invertible scalar maps for [`Transformed`](crate::Transformed) distributions.
//!
//! A [`Bijector`] provides the forward map, its inverse, and the log of the
//! absolute derivative of the *inverse* — the Jacobian term in the change of
//! variables `p_Y(y) = p_X(g⁻¹(y))·|d g⁻¹/dy|`. All maps here are strictly
//! monotone, so that density is well-defined.
//!
//! | module | holds |
//! |--------|-------|
//! | [`primitives`] | the maps themselves: [`Identity`], [`Affine`], [`Exp`], [`Sigmoid`], [`Softplus`] |
//! | [`algebra`] | ways to build new maps from old: [`Compose`], [`Inverse`] |
//!
//! # Why the Jacobian accumulates rather than being re-derived
//!
//! The reason to separate these two modules is that [`algebra`] never needs to
//! know a derivative. Composition chains them by the chain rule and inversion
//! flips one by the inverse function theorem, so a composite map's Jacobian is
//! assembled from its parts' — which means a new primitive gets composition and
//! inversion for free, and a composite cannot disagree with the pieces it was
//! built from. The suites in both modules check exactly that, by comparing
//! every `inverse_log_det_jacobian` against a numerical derivative of the
//! corresponding `inverse`.

pub mod algebra;
pub mod primitives;

pub use algebra::{Compose, Inverse};
pub use primitives::{Affine, Exp, Identity, Sigmoid, Softplus};

/// A strictly monotone, invertible map `g: ℝ → S`.
pub trait Bijector {
    /// `y = g(x)`.
    fn forward(&self, x: f64) -> f64;
    /// `x = g⁻¹(y)`.
    fn inverse(&self, y: f64) -> f64;
    /// `ln |d g⁻¹(y) / dy|` — the Jacobian correction for `p_Y(y)`.
    fn inverse_log_det_jacobian(&self, y: f64) -> f64;

    /// `other ∘ self`: apply `self` first, then `other`.
    ///
    /// Reads in application order, so the log-normal is
    /// `Affine{scale: σ, shift: μ}.then(Exp)` — scale and shift a standard
    /// normal, then exponentiate.
    fn then<Outer: Bijector>(self, other: Outer) -> Compose<Self, Outer>
    where
        Self: Sized,
    {
        Compose::new(self, other)
    }

    /// The map that undoes this one, with the Jacobian flipped accordingly.
    fn inverted(self) -> Inverse<Self>
    where
        Self: Sized,
    {
        Inverse::new(self)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
pub(crate) mod probe {
    use super::Bijector;
    use bayes_utils::derivative;

    /// `inverse` really undoes `forward`, and `inverse_log_det_jacobian` really
    /// is the log of `|d inverse / dy|`.
    ///
    /// The second half is the one worth having. A log-det is easy to get wrong
    /// by a sign or a reciprocal and nothing else in the crate would notice —
    /// the density would simply be a different, still-plausible density. Here it
    /// is held against a five-point numerical derivative of the inverse map the
    /// same bijector supplies, so the two can only agree if both are right.
    /// The points are given in the **domain**, and each is pushed forward to get
    /// a `y`, which guarantees every `y` tested is inside the map's image. A `y`
    /// chosen by hand is easy to put outside it — `Affine{scale: -0.6}` after an
    /// `Exp` has image `(-inf, 1.1)`, so testing at `y = 2` compares two `NaN`s
    /// and reports a disagreement that is the test's fault.
    pub(crate) fn behaves_like_a_bijector(name: &str, b: &impl Bijector, points: &[f64]) {
        for &x in points {
            let y = b.forward(x);
            assert!(y.is_finite(), "{name}: forward({x}) = {y} is not finite");
            let back = b.inverse(y);
            assert!(
                (back - x).abs() < 1e-8 * x.abs().max(1.0),
                "{name}: inverse(forward({x})) = {back}"
            );

            // The step is derived from the map's own local scale, not from `y`.
            // A fixed fraction of `y` fails wherever the inverse is steep: at
            // `y = 0.99994` a sigmoid's inverse has slope 10600, so a step of
            // `1e-5` spans most of the remaining distance to the boundary and
            // the five-point rule reports a slope 0.1% off — an artefact of the
            // probe, not an error in the Jacobian. Setting `h` so the inverse
            // moves by about `1e-4` keeps the difference well-conditioned
            // wherever the map happens to be steep or flat.
            //
            // Using the claimed slope to pick the step is safe: it only sets the
            // scale of a difference whose *value* is then compared against the
            // claim, so a badly wrong Jacobian still fails loudly.
            let slope = b.inverse_log_det_jacobian(y).exp();
            let h = 1e-4 / slope.max(1e-8);
            let numeric = derivative(|t| b.inverse(t), y, h).abs();
            assert!(
                (slope - numeric).abs() < 1e-6 * numeric.max(1e-6),
                "{name}: at y={y}, exp(inverse_log_det_jacobian) = {slope} but \
                 |d inverse/dy| = {numeric}"
            );
        }
    }
}
