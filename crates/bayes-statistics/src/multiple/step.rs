//! The ordering machinery the corrections share, and the dispatch enum.

/// Indices of `values` in ascending order, ties broken by index for
/// reproducibility.
pub(super) fn order_ascending(values: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

/// Indices of `values` in descending order.
pub(super) fn order_descending(values: &[f64]) -> Vec<usize> {
    let mut idx = order_ascending(values);
    idx.reverse();
    idx
}

/// Put `sorted[k]` back at position `order[k]`, recovering the input ordering.
pub(super) fn scatter(order: &[usize], sorted: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; order.len()];
    for (k, &i) in order.iter().enumerate() {
        out[i] = sorted[k];
    }
    out
}

/// Which correction to apply, for code that chooses at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Correction {
    /// FWER, no dependence assumption, most conservative.
    Bonferroni,
    /// FWER, independence assumed.
    Sidak,
    /// FWER, step-down Bonferroni; dominates [`Correction::Bonferroni`].
    Holm,
    /// FWER, step-down Šidák.
    HolmSidak,
    /// FDR under independence or positive dependence.
    BenjaminiHochberg,
    /// FDR under arbitrary dependence.
    BenjaminiYekutieli,
}

impl Correction {
    /// Adjusted p-values under this correction.
    pub fn adjust(self, p_values: &[f64]) -> Vec<f64> {
        use super::adjust as a;
        match self {
            Self::Bonferroni => a::bonferroni(p_values),
            Self::Sidak => a::sidak(p_values),
            Self::Holm => a::holm(p_values),
            Self::HolmSidak => a::holm_sidak(p_values),
            Self::BenjaminiHochberg => a::benjamini_hochberg(p_values),
            Self::BenjaminiYekutieli => a::benjamini_yekutieli(p_values),
        }
    }

    /// Does this control the family-wise error rate rather than the FDR?
    pub fn controls_fwer(self) -> bool {
        matches!(
            self,
            Self::Bonferroni | Self::Sidak | Self::Holm | Self::HolmSidak
        )
    }
}

/// Which hypotheses to reject at level `alpha` under `correction`.
///
/// Returns a mask in the input order. Comparing the *adjusted* p-value to the
/// original `alpha` is the right move and the reason these functions return
/// adjusted values rather than a threshold: it means the same `alpha` the study
/// was designed around stays the number in the code.
pub fn reject_at(p_values: &[f64], alpha: f64, correction: Correction) -> Vec<bool> {
    if alpha.is_nan() || !(0.0..=1.0).contains(&alpha) {
        return vec![false; p_values.len()];
    }
    correction
        .adjust(p_values)
        .iter()
        .map(|&q| q <= alpha)
        .collect()
}
