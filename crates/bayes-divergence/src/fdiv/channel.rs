use super::*;

/// A **row-stochastic channel** (Markov kernel) `K`: `inputs` rows, `outputs`
/// columns, each row a probability vector. Post-processing a distribution `p` is
/// the pushforward `(pK)ₖ = Σᵢ pᵢ Kᵢₖ` ([`apply`](Self::apply)).
///
/// The **data-processing inequality** is the reason channels matter here: for
/// *every* f-divergence,
///
/// ```text
/// D_f(pK ‖ qK) ≤ D_f(p ‖ q)
/// ```
///
/// — proved in `riemann` (`RiemannLean/Bayes/Divergence.lean`:
/// `fDiv_dataProcessing`, with the KL specialization `klDiv_dataProcessing` and
/// the deterministic two-group case `klDiv_ge_binary`). No channel — coarse-
/// graining, marginalizing, adding noise, any Markov post-processing — can
/// increase the divergence between two distributions; it only ever loses
/// discriminating information.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    /// `rows[i]` is the output distribution given input `i`; each sums to `1`.
    rows: Vec<Vec<f64>>,
    outputs: usize,
}

impl Channel {
    /// Build a channel from its rows, validating that it is rectangular and
    /// row-stochastic (each row finite, non-negative, summing to `1`).
    pub fn new(rows: Vec<Vec<f64>>) -> Result<Self, BayesError> {
        let outputs = rows.first().map(Vec::len).unwrap_or(0);
        validate_row_stochastic(&rows)?;
        Ok(Self { rows, outputs })
    }

    /// A deterministic **coarse-graining** channel: input `i` is sent to group
    /// `assignment[i]` among `groups` outputs (a 0/1 kernel). Merging categories
    /// is a channel, so the DPI specializes to "coarse-graining only lowers the
    /// divergence" — the `riemann` two-group case is `klDiv_ge_binary`.
    pub fn coarse_grain(assignment: &[usize], groups: usize) -> Result<Self, BayesError> {
        if assignment.is_empty() || groups == 0 {
            return Err(BayesError::Malformed {
                what: "coarse-graining needs at least one input and one group",
            });
        }
        let mut rows = Vec::with_capacity(assignment.len());
        for &g in assignment {
            if g >= groups {
                return Err(BayesError::Malformed {
                    what: "coarse-graining group index out of range",
                });
            }
            let mut row = vec![0.0; groups];
            row[g] = 1.0;
            rows.push(row);
        }
        Ok(Self {
            rows,
            outputs: groups,
        })
    }

    /// The number of input categories (rows).
    pub fn inputs(&self) -> usize {
        self.rows.len()
    }

    /// The number of output categories (columns).
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// The pushforward `pK` — the output distribution `(pK)ₖ = Σᵢ pᵢ Kᵢₖ`. The
    /// input `p` is read positionally over the channel's inputs (zero-padded /
    /// truncated to [`inputs`](Self::inputs)); the result has
    /// [`outputs`](Self::outputs) entries and is itself a probability vector
    /// when `p` is.
    pub fn apply(&self, p: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.outputs];
        for (i, row) in self.rows.iter().enumerate() {
            let pi = p.get(i).copied().unwrap_or(0.0);
            if pi == 0.0 {
                continue;
            }
            for (k, &kik) in row.iter().enumerate() {
                out[k] += pi * kik;
            }
        }
        out
    }
}
