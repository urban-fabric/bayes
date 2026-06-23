//! **Characteristic functions** `φ(t) = E[e^{itX}]`.
//!
//! The characteristic function is the Fourier transform of the law and the
//! complex completion of the [`MomentGenerating`](crate::MomentGenerating)
//! surface: where the MGF `E[e^{tX}]` diverges (heavy tails), the characteristic
//! function still **always exists** and satisfies `|φ(t)| ≤ 1`. That makes it the
//! only transform that reaches families like the Cauchy, whose MGF is undefined
//! everywhere but `0`.
//!
//! Values are returned as `(re, im)` pairs — a self-contained eager scalar API,
//! no complex-number type required. `φ(0) = (1, 0)` for every family, and the
//! derivative at the origin recovers the mean: `φ'(0) = i·E[X]`.

use crate::{
    Bernoulli, Binomial, Cauchy, ChiSquared, Exponential, Gamma, Geometric, Laplace,
    NegativeBinomial, Normal, Poisson, Uniform,
};

/// `r·(cos θ, sin θ)` — a complex number from modulus and angle.
fn polar(modulus: f64, angle: f64) -> (f64, f64) {
    (modulus * angle.cos(), modulus * angle.sin())
}

/// Complex division `a / b`, both as `(re, im)`.
fn cdiv(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

/// Raise a complex number to a real power via its polar form.
fn cpowf(z: (f64, f64), exponent: f64) -> (f64, f64) {
    let modulus = (z.0 * z.0 + z.1 * z.1).sqrt().powf(exponent);
    polar(modulus, exponent * z.1.atan2(z.0))
}

/// A family with a closed-form characteristic function.
pub trait CharacteristicFunction {
    /// `φ(t) = E[e^{itX}]`, returned as `(real, imaginary)`. Always defined, with
    /// modulus at most one.
    fn characteristic_function(&self, t: f64) -> (f64, f64);

    /// `|φ(t)|`, the modulus of the characteristic function (`≤ 1`).
    fn cf_modulus(&self, t: f64) -> f64 {
        let (re, im) = self.characteristic_function(t);
        (re * re + im * im).sqrt()
    }
}

/// `φ(t) = exp(iμt − ½σ²t²)`.
impl CharacteristicFunction for Normal {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        let (mu, sigma) = self.params();
        polar((-0.5 * sigma * sigma * t * t).exp(), mu * t)
    }
}

/// `φ(t) = exp(ix₀t − γ|t|)` — defined despite the Cauchy having no MGF.
impl CharacteristicFunction for Cauchy {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        polar((-self.scale() * t.abs()).exp(), self.location() * t)
    }
}

/// `φ(t) = λ / (λ − it)`.
impl CharacteristicFunction for Exponential {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        let rate = self.rate();
        cdiv((rate, 0.0), (rate, -t))
    }
}

/// `φ(t) = (1 − it/β)^{−k}` (shape `k`, rate `β`).
impl CharacteristicFunction for Gamma {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        let (shape, rate) = (self.shape(), self.rate());
        cpowf((1.0, -t / rate), -shape)
    }
}

/// `φ(t) = (e^{itb} − e^{ita}) / (it(b−a))`.
impl CharacteristicFunction for Uniform {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        let (a, b) = (self.lo(), self.hi());
        if t == 0.0 {
            return (1.0, 0.0);
        }
        let d = t * (b - a);
        (
            ((t * b).sin() - (t * a).sin()) / d,
            ((t * a).cos() - (t * b).cos()) / d,
        )
    }
}

/// `φ(t) = e^{iμt} / (1 + b²t²)` (location `μ`, scale `b`).
impl CharacteristicFunction for Laplace {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        let (mu, b) = (self.location(), self.scale());
        let denom = 1.0 + b * b * t * t;
        ((mu * t).cos() / denom, (mu * t).sin() / denom)
    }
}

/// `φ(t) = (1 − 2it)^{−k/2}` (`k` degrees of freedom).
impl CharacteristicFunction for ChiSquared {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        cpowf((1.0, -2.0 * t), -0.5 * self.freedom())
    }
}

/// `φ(t) = exp(λ(e^{it} − 1))`.
impl CharacteristicFunction for Poisson {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        let lambda = self.lambda();
        polar((lambda * (t.cos() - 1.0)).exp(), lambda * t.sin())
    }
}

/// `φ(t) = (1 − p) + p·e^{it}`.
impl CharacteristicFunction for Bernoulli {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        let p = self.p();
        (1.0 - p + p * t.cos(), p * t.sin())
    }
}

/// `φ(t) = ((1 − p) + p·e^{it})^n`.
impl CharacteristicFunction for Binomial {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        let p = self.p();
        cpowf((1.0 - p + p * t.cos(), p * t.sin()), self.trials() as f64)
    }
}

/// `φ(t) = p·e^{it} / (1 − (1−p)·e^{it})` (trial-count parameterization).
impl CharacteristicFunction for Geometric {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        let p = self.p();
        let denom = (1.0 - (1.0 - p) * t.cos(), -(1.0 - p) * t.sin());
        cdiv(polar(p, t), denom)
    }
}

/// `φ(t) = (p / (1 − (1−p)·e^{it}))^r`.
impl CharacteristicFunction for NegativeBinomial {
    fn characteristic_function(&self, t: f64) -> (f64, f64) {
        let p = self.p();
        let denom = (1.0 - (1.0 - p) * t.cos(), -(1.0 - p) * t.sin());
        cpowf(cdiv((p, 0.0), denom), self.successes())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::Moments;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    fn cf_origin_is_one(d: &dyn CharacteristicFunction) {
        let (re, im) = d.characteristic_function(0.0);
        close(re, 1.0, 1e-12);
        close(im, 0.0, 1e-12);
    }

    #[test]
    fn cf_is_one_at_origin_and_bounded_by_one() {
        let conts: [&dyn CharacteristicFunction; 7] = [
            &Normal::new(2.0, 1.5).unwrap(),
            &Cauchy::new(0.5, 2.0).unwrap(),
            &Exponential::new(1.3).unwrap(),
            &Gamma::new(2.0, 0.7).unwrap(),
            &Uniform::new(-1.0, 3.0).unwrap(),
            &Laplace::new(0.0, 1.0).unwrap(),
            &ChiSquared::new(5.0).unwrap(),
        ];
        let discs: [&dyn CharacteristicFunction; 5] = [
            &Poisson::new(3.0).unwrap(),
            &Bernoulli::new(0.4).unwrap(),
            &Binomial::new(6, 0.3).unwrap(),
            &Geometric::new(0.5).unwrap(),
            &NegativeBinomial::new(3.0, 0.6).unwrap(),
        ];
        for d in conts {
            cf_origin_is_one(d);
            for &t in &[0.3, 1.0, -2.5, 4.0] {
                assert!(d.cf_modulus(t) <= 1.0 + 1e-12);
            }
        }
        for d in discs {
            cf_origin_is_one(d);
            for &t in &[0.3, 1.0, -2.5, 4.0] {
                assert!(d.cf_modulus(t) <= 1.0 + 1e-12);
            }
        }
    }

    #[test]
    fn cf_closed_forms_match_known_values() {
        // Standard normal: φ(t) = e^{−t²/2}, purely real.
        let (re, im) = Normal::new(0.0, 1.0).unwrap().characteristic_function(1.0);
        close(re, (-0.5_f64).exp(), 1e-12);
        close(im, 0.0, 1e-12);

        // Cauchy(0,1): φ(t) = e^{−|t|}.
        close(
            Cauchy::new(0.0, 1.0).unwrap().characteristic_function(2.0).0,
            (-2.0_f64).exp(),
            1e-12,
        );

        // Exponential(1): φ(1) = 1/(1−i) = (1+i)/2.
        let (re, im) = Exponential::new(1.0).unwrap().characteristic_function(1.0);
        close(re, 0.5, 1e-12);
        close(im, 0.5, 1e-12);

        // Bernoulli(½) at t = π: 1 − p + p·cos π = 0, im = p·sin π = 0.
        let (re, im) = Bernoulli::new(0.5)
            .unwrap()
            .characteristic_function(std::f64::consts::PI);
        close(re, 0.0, 1e-12);
        close(im, 0.0, 1e-12);
    }

    #[test]
    fn cf_derivative_at_origin_recovers_the_mean() {
        // φ(h) ≈ 1 + i·E[X]·h, so Im φ(h)/h → mean.
        let check = |d: &dyn CharacteristicFunction, mean: f64| {
            let h = 1e-5;
            let im = d.characteristic_function(h).1;
            close(im / h, mean, 1e-4);
        };
        let g = Gamma::new(3.0, 2.0).unwrap();
        check(&g, g.mean());
        let po = Poisson::new(4.0).unwrap();
        check(&po, po.mean());
        let nb = NegativeBinomial::new(3.0, 0.6).unwrap();
        check(&nb, nb.mean());
    }
}
