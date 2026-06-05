//! Effect sizes and association measures that complement hypothesis-test p-values.

use super::summary::{mean, variance};

fn finite_sample(data: &[f64], min_len: usize) -> bool {
    data.len() >= min_len && data.iter().all(|x| x.is_finite())
}

/// Pooled sample standard deviation for two independent samples.
pub fn pooled_std_dev(a: &[f64], b: &[f64]) -> f64 {
    if !finite_sample(a, 2) || !finite_sample(b, 2) {
        return f64::NAN;
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let sp2 = ((na - 1.0) * variance(a) + (nb - 1.0) * variance(b)) / (na + nb - 2.0);
    sp2.sqrt()
}

/// Cohen's `d`: difference in means divided by the pooled sample standard deviation.
pub fn cohen_d(a: &[f64], b: &[f64]) -> f64 {
    let sp = pooled_std_dev(a, b);
    if sp <= 0.0 || sp.is_nan() {
        return f64::NAN;
    }
    (mean(a) - mean(b)) / sp
}

/// Hedges' `g`: small-sample bias-corrected Cohen's `d`.
pub fn hedges_g(a: &[f64], b: &[f64]) -> f64 {
    let d = cohen_d(a, b);
    let df = (a.len() + b.len()).saturating_sub(2) as f64;
    if d.is_nan() || df <= 1.0 {
        return f64::NAN;
    }
    d * (1.0 - 3.0 / (4.0 * df - 1.0))
}

/// Glass's `Δ`: mean difference divided by the control group's standard deviation.
pub fn glass_delta(treatment: &[f64], control: &[f64]) -> f64 {
    if !finite_sample(treatment, 1) || !finite_sample(control, 2) {
        return f64::NAN;
    }
    let s = variance(control).sqrt();
    if s <= 0.0 || s.is_nan() {
        return f64::NAN;
    }
    (mean(treatment) - mean(control)) / s
}

fn group_sums(groups: &[&[f64]]) -> Option<(f64, f64, f64, f64)> {
    if groups.len() < 2 || groups.iter().any(|g| g.is_empty() || !finite_sample(g, 1)) {
        return None;
    }
    let total_n = groups.iter().map(|g| g.len()).sum::<usize>();
    let grand_mean = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / total_n as f64;
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for group in groups {
        let m = mean(group);
        ss_between += group.len() as f64 * (m - grand_mean).powi(2);
        ss_within += group.iter().map(|&x| (x - m).powi(2)).sum::<f64>();
    }
    Some((
        ss_between,
        ss_within,
        ss_between + ss_within,
        total_n as f64,
    ))
}

/// Eta-squared effect size for one-way ANOVA: `SS_between / SS_total`.
pub fn eta_squared(groups: &[&[f64]]) -> f64 {
    let Some((ss_between, _, ss_total, _)) = group_sums(groups) else {
        return f64::NAN;
    };
    if ss_total <= 0.0 {
        return f64::NAN;
    }
    ss_between / ss_total
}

/// Omega-squared effect size for one-way ANOVA.
pub fn omega_squared(groups: &[&[f64]]) -> f64 {
    let Some((ss_between, ss_within, ss_total, total_n)) = group_sums(groups) else {
        return f64::NAN;
    };
    let k = groups.len() as f64;
    let df_between = k - 1.0;
    let df_within = total_n - k;
    if df_within <= 0.0 || ss_total <= 0.0 {
        return f64::NAN;
    }
    let ms_within = ss_within / df_within;
    ((ss_between - df_between * ms_within) / (ss_total + ms_within)).max(0.0)
}

/// Sample odds ratio for a 2x2 table `[[a, b], [c, d]]`: `ad / bc`.
pub fn odds_ratio(a: u64, b: u64, c: u64, d: u64) -> f64 {
    let numerator = a as f64 * d as f64;
    let denominator = b as f64 * c as f64;
    if denominator == 0.0 {
        return if numerator > 0.0 {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    numerator / denominator
}

fn risk(events: u64, non_events: u64) -> f64 {
    let n = events + non_events;
    if n == 0 {
        f64::NAN
    } else {
        events as f64 / n as f64
    }
}

/// Risk ratio for two groups represented as `(events, non_events)`.
pub fn risk_ratio(events_a: u64, non_events_a: u64, events_b: u64, non_events_b: u64) -> f64 {
    let ra = risk(events_a, non_events_a);
    let rb = risk(events_b, non_events_b);
    if ra.is_nan() || rb.is_nan() {
        return f64::NAN;
    }
    if rb == 0.0 {
        return if ra > 0.0 { f64::INFINITY } else { f64::NAN };
    }
    ra / rb
}

/// Risk difference for two groups represented as `(events, non_events)`.
pub fn risk_difference(events_a: u64, non_events_a: u64, events_b: u64, non_events_b: u64) -> f64 {
    let ra = risk(events_a, non_events_a);
    let rb = risk(events_b, non_events_b);
    if ra.is_nan() || rb.is_nan() {
        f64::NAN
    } else {
        ra - rb
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn standardized_mean_effects_have_expected_direction() {
        let a = [4.0, 5.0, 6.0, 7.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        assert!(cohen_d(&a, &b) > 1.0);
        assert!(hedges_g(&a, &b) < cohen_d(&a, &b));
        assert!(glass_delta(&a, &b) > 1.0);
    }

    #[test]
    fn anova_effect_sizes_are_bounded() {
        let g1 = [1.0, 2.0, 3.0];
        let g2 = [10.0, 11.0, 12.0];
        let eta = eta_squared(&[&g1, &g2]);
        let omega = omega_squared(&[&g1, &g2]);
        assert!(eta > 0.9 && eta <= 1.0);
        assert!(omega > 0.8 && omega <= eta);
    }

    #[test]
    fn table_effects_match_definitions() {
        close(odds_ratio(10, 5, 4, 8), 4.0, 1e-12);
        close(risk_ratio(10, 10, 5, 15), 2.0, 1e-12);
        close(risk_difference(10, 10, 5, 15), 0.25, 1e-12);
    }
}
