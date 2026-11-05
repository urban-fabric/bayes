//! Adaptive rejection sampling for log-concave densities (Gilks & Wild, 1992).
//!
//! Plain rejection sampling needs you to supply a good envelope, and for an
//! irregular density you usually cannot: you do not know where the mode is, how
//! heavy the tails are, or what constant makes the bound valid. Choose badly and
//! you either bias the result (envelope too low) or throw away 99% of proposals
//! (envelope too high).
//!
//! Adaptive rejection sampling removes that choice. If `h(x) = ln f(x)` is
//! **concave** — which covers the normal, exponential, gamma with shape ≥ 1,
//! beta with both shapes ≥ 1, Weibull with shape ≥ 1, logistic, Gumbel, and most
//! posteriors from those likelihoods — then two facts follow for free:
//!
//! * the **tangent** to `h` at any point lies *above* `h` everywhere, so a set
//!   of tangents forms an upper hull, and `exp` of a piecewise-linear hull is a
//!   piecewise-exponential density that can be sampled exactly;
//! * the **chord** between two points lies *below* `h` between them, so chords
//!   give a lower hull and a free "squeeze" test that accepts most proposals
//!   without ever evaluating `h`.
//!
//! The adaptive part is the pay-off: every rejected point is *added* to the hull,
//! which tightens it exactly where the fit was worst. Acceptance climbs toward 1
//! as the sampler runs, so the expensive density evaluations stop almost
//! entirely after the first few dozen draws. Draws are **exact** throughout —
//! the hull affects only how often a proposal is kept, never the distribution of
//! what is kept.
//!
//! The concavity requirement is not checkable in general, so it is the caller's
//! to honour. [`AdaptiveRejection`] does detect the observable consequence: if a
//! proposal is found above the supposed upper hull, the density was not
//! log-concave and [`AdaptiveRejection::hull_was_breached`] reports it rather
//! than quietly returning biased draws.

use bayes_core::BayesError;
use rand::{Rng, RngExt};

/// A point on the hull: an abscissa with the log-density and its slope there.
#[derive(Clone, Copy, Debug)]
struct Node {
    x: f64,
    h: f64,
    dh: f64,
}

/// Adaptive rejection sampler for a log-concave `ln_density`.
pub struct AdaptiveRejection<H, D> {
    ln_density: H,
    ln_density_prime: D,
    nodes: Vec<Node>,
    lo: f64,
    hi: f64,
    /// Points where consecutive tangents cross, one fewer than the nodes.
    crossings: Vec<f64>,
    breached: bool,
    evaluations: usize,
    proposals: usize,
    accepted: usize,
}

impl<H, D> AdaptiveRejection<H, D>
where
    H: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    /// Build a sampler over `(lo, hi)` seeded with the given abscissae.
    ///
    /// The seeds must bracket the mode: on an unbounded side the hull can only
    /// be integrated if the outermost tangent slopes back toward the interior,
    /// so at least one seed must have positive slope when `lo` is `-inf` and at
    /// least one negative slope when `hi` is `+inf`. That is a real requirement
    /// rather than an implementation detail — without it the envelope has
    /// infinite mass and no sampler exists.
    pub fn new(
        ln_density: H,
        ln_density_prime: D,
        lo: f64,
        hi: f64,
        seeds: &[f64],
    ) -> Result<Self, BayesError> {
        if hi <= lo || !(lo.is_finite() || lo.is_infinite()) {
            return Err(BayesError::Malformed {
                what: "adaptive rejection needs a non-empty interval",
            });
        }
        if seeds.len() < 2 {
            return Err(BayesError::Malformed {
                what: "adaptive rejection needs at least two seed abscissae",
            });
        }
        let mut nodes: Vec<Node> = Vec::with_capacity(seeds.len());
        for &x in seeds {
            if !(x > lo && x < hi) {
                return Err(BayesError::Malformed {
                    what: "a seed abscissa lies outside the interval",
                });
            }
            let h = ln_density(x);
            let dh = ln_density_prime(x);
            if !h.is_finite() || !dh.is_finite() {
                return Err(BayesError::NonFinite {
                    name: "ln_density at a seed",
                    value: h,
                });
            }
            nodes.push(Node { x, h, dh });
        }
        nodes.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        if lo.is_infinite() && nodes[0].dh <= 0.0 {
            return Err(BayesError::Malformed {
                what: "on an unbounded left tail the leftmost seed must have positive slope",
            });
        }
        if hi.is_infinite() && nodes[nodes.len() - 1].dh >= 0.0 {
            return Err(BayesError::Malformed {
                what: "on an unbounded right tail the rightmost seed must have negative slope",
            });
        }

        let mut s = Self {
            ln_density,
            ln_density_prime,
            nodes,
            lo,
            hi,
            crossings: Vec::new(),
            breached: false,
            evaluations: 0,
            proposals: 0,
            accepted: 0,
        };
        s.rebuild_crossings();
        Ok(s)
    }

    /// Where the tangents at consecutive nodes cross. Concavity guarantees the
    /// crossing lies between the two abscissae.
    fn rebuild_crossings(&mut self) {
        self.crossings.clear();
        for w in self.nodes.windows(2) {
            let (a, b) = (w[0], w[1]);
            let z = if (a.dh - b.dh).abs() < 1e-300 {
                // Parallel tangents: the nodes agree on slope, so put the
                // crossing midway and let the (identical) hulls coincide.
                0.5 * (a.x + b.x)
            } else {
                (b.h - a.h - b.x * b.dh + a.x * a.dh) / (a.dh - b.dh)
            };
            self.crossings.push(z.clamp(a.x, b.x));
        }
    }

    /// The upper hull at `x`: the tangent from whichever node owns this segment.
    fn upper(&self, x: f64) -> f64 {
        let i = self.segment_of(x);
        let n = self.nodes[i];
        n.h + n.dh * (x - n.x)
    }

    /// Which tangent governs `x`.
    fn segment_of(&self, x: f64) -> usize {
        self.crossings.partition_point(|&z| z < x)
    }

    /// The lower hull at `x`: the chord between the bracketing nodes, or
    /// `-inf` outside the node range where no chord exists.
    fn lower(&self, x: f64) -> f64 {
        let k = self.nodes.len();
        if x < self.nodes[0].x || x > self.nodes[k - 1].x {
            return f64::NEG_INFINITY;
        }
        let j = self.nodes.partition_point(|n| n.x < x);
        if j == 0 {
            return self.nodes[0].h;
        }
        let (a, b) = (self.nodes[j - 1], self.nodes[j]);
        if (b.x - a.x).abs() < 1e-300 {
            return a.h;
        }
        a.h + (x - a.x) * (b.h - a.h) / (b.x - a.x)
    }

    /// The boundaries of segment `i`, which is `[z_{i-1}, z_i]` with the outer
    /// ends being the interval bounds.
    fn segment_bounds(&self, i: usize) -> (f64, f64) {
        let left = if i == 0 {
            self.lo
        } else {
            self.crossings[i - 1]
        };
        let right = if i == self.crossings.len() {
            self.hi
        } else {
            self.crossings[i]
        };
        (left, right)
    }

    /// `ln` of the mass the upper hull puts on segment `i`.
    ///
    /// The exponential integral is written relative to the segment's own peak so
    /// the intermediate `exp` cannot overflow: a hull whose log-height is 700
    /// has finite mass but `exp(700)` does not.
    fn ln_segment_mass(&self, i: usize) -> f64 {
        let n = self.nodes[i];
        let (a, b) = self.segment_bounds(i);
        let (ua, ub) = (n.h + n.dh * (a - n.x), n.h + n.dh * (b - n.x));
        let peak = ua.max(ub);
        if n.dh.abs() < 1e-12 {
            // Flat: the mass is height times width.
            return peak + (b - a).ln();
        }
        // integral exp(u) dx over [a,b] = (exp(ub) - exp(ua)) / dh
        // = exp(peak) * (exp(ub - peak) - exp(ua - peak)) / dh
        let diff = ((ub - peak).exp() - (ua - peak).exp()) / n.dh;
        if diff <= 0.0 {
            return f64::NEG_INFINITY;
        }
        peak + diff.ln()
    }

    /// Draw from the piecewise-exponential upper hull.
    fn sample_hull<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let k = self.nodes.len();
        let ln_masses: Vec<f64> = (0..k).map(|i| self.ln_segment_mass(i)).collect();
        let max = ln_masses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = ln_masses.iter().map(|&m| (m - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut target: f64 = rng.random::<f64>() * total;
        let mut i = k - 1;
        for (j, &w) in weights.iter().enumerate() {
            if target < w {
                i = j;
                break;
            }
            target -= w;
        }

        // Inverse-CDF within the chosen exponential segment.
        let n = self.nodes[i];
        let (a, b) = self.segment_bounds(i);
        let u: f64 = rng.random();
        if n.dh.abs() < 1e-12 {
            return a + u * (b - a);
        }
        // On [a,b] with density proportional to exp(dh * x):
        //   x = a + ln(1 + u*(exp(dh*(b-a)) - 1)) / dh
        // written through `exp_m1`/`ln_1p` so the small-`dh` case does not cancel,
        // and with the infinite-width case handled by its limit.
        let width = b - a;
        if !width.is_finite() {
            // One side unbounded; concavity plus the constructor's slope checks
            // guarantee the open side runs downhill, so the limit is clean.
            return if b.is_infinite() {
                a - (1.0_f64 - u).ln() / n.dh.abs()
            } else {
                b + (1.0_f64 - u).ln() / n.dh.abs()
            };
        }
        let e = (n.dh * width).exp_m1();
        a + (u * e).ln_1p() / n.dh
    }

    /// Draw one value, adapting the hull on each rejection.
    pub fn draw<R: Rng + ?Sized>(&mut self, rng: &mut R, max_proposals: usize) -> Option<f64> {
        for _ in 0..max_proposals {
            let x = self.sample_hull(rng);
            self.proposals += 1;
            if !(x > self.lo && x < self.hi) || !x.is_finite() {
                continue;
            }
            let u: f64 = rng.random();
            let ln_u = u.ln();
            let up = self.upper(x);

            // Squeeze test: accept without touching the density at all.
            if ln_u <= self.lower(x) - up {
                self.accepted += 1;
                return Some(x);
            }

            // Otherwise pay for one density evaluation.
            let hx = (self.ln_density)(x);
            self.evaluations += 1;
            if hx > up + 1e-8 {
                // The "upper" hull is below the density: not log-concave.
                self.breached = true;
            }
            let accept = ln_u <= hx - up;

            // Adapt regardless of the verdict: the hull is now known to be loose
            // here, and tightening it is what makes later draws cheap.
            if hx.is_finite() && self.nodes.len() < 64 {
                let dh = (self.ln_density_prime)(x);
                if dh.is_finite() {
                    let pos = self.nodes.partition_point(|n| n.x < x);
                    if self.nodes.iter().all(|n| (n.x - x).abs() > 1e-12) {
                        self.nodes.insert(pos, Node { x, h: hx, dh });
                        self.rebuild_crossings();
                    }
                }
            }

            if accept {
                self.accepted += 1;
                return Some(x);
            }
        }
        None
    }

    /// Was a proposal found above the upper hull? If so the density was not
    /// log-concave and the draws are not from it.
    pub fn hull_was_breached(&self) -> bool {
        self.breached
    }

    /// How many density evaluations the squeeze test did **not** avoid.
    pub fn density_evaluations(&self) -> usize {
        self.evaluations
    }

    /// The observed acceptance rate.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposals == 0 {
            f64::NAN
        } else {
            self.accepted as f64 / self.proposals as f64
        }
    }

    /// How many abscissae the hull currently carries.
    pub fn hull_size(&self) -> usize {
        self.nodes.len()
    }
}
