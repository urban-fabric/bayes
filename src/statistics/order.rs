//! Order statistics — quantiles and the summaries built on them. All sort a copy
//! of the data (the input slice is left untouched), using linear interpolation
//! between order statistics (the common "type 7" estimator).

/// Sort a copy ascending, dropping NaNs (which have no order).
fn sorted(data: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// The `tau`-quantile (`tau ∈ [0, 1]`) by linear interpolation between order
/// statistics. `NaN` for an empty sample.
pub fn quantile(data: &[f64], tau: f64) -> f64 {
    let v = sorted(data);
    quantile_sorted(&v, tau)
}

/// Quantile of an already-sorted slice (internal — avoids re-sorting).
fn quantile_sorted(v: &[f64], tau: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    if v.len() == 1 {
        return v[0];
    }
    let tau = tau.clamp(0.0, 1.0);
    let h = (v.len() as f64 - 1.0) * tau;
    let lo = h.floor() as usize;
    let frac = h - lo as f64;
    if lo + 1 < v.len() {
        v[lo] + frac * (v[lo + 1] - v[lo])
    } else {
        v[lo]
    }
}

/// The median (the 0.5-quantile).
pub fn median(data: &[f64]) -> f64 {
    quantile(data, 0.5)
}

/// The `p`-th percentile, `p ∈ [0, 100]`.
pub fn percentile(data: &[f64], p: f64) -> f64 {
    quantile(data, p / 100.0)
}

/// The interquartile range `Q3 − Q1`.
pub fn iqr(data: &[f64]) -> f64 {
    let v = sorted(data);
    quantile_sorted(&v, 0.75) - quantile_sorted(&v, 0.25)
}

/// Tukey's five-number summary `(min, Q1, median, Q3, max)`.
pub fn five_number_summary(data: &[f64]) -> (f64, f64, f64, f64, f64) {
    let v = sorted(data);
    if v.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    (
        v[0],
        quantile_sorted(&v, 0.25),
        quantile_sorted(&v, 0.5),
        quantile_sorted(&v, 0.75),
        v[v.len() - 1],
    )
}

/// Median absolute deviation `median(|xᵢ − median(x)|)` — a robust spread estimate.
pub fn median_abs_deviation(data: &[f64]) -> f64 {
    let m = median(data);
    if m.is_nan() {
        return f64::NAN;
    }
    let dev: Vec<f64> = data.iter().map(|&x| (x - m).abs()).collect();
    median(&dev)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn median_odd_and_even() {
        close(median(&[3.0, 1.0, 2.0]), 2.0, 1e-12);
        close(median(&[1.0, 2.0, 3.0, 4.0]), 2.5, 1e-12);
    }

    #[test]
    fn quantiles_interpolate() {
        let d = [1.0, 2.0, 3.0, 4.0, 5.0];
        close(quantile(&d, 0.0), 1.0, 1e-12);
        close(quantile(&d, 1.0), 5.0, 1e-12);
        close(quantile(&d, 0.5), 3.0, 1e-12);
        close(percentile(&d, 25.0), 2.0, 1e-12);
        close(iqr(&d), 2.0, 1e-12); // Q3=4, Q1=2
    }

    #[test]
    fn five_number_and_mad() {
        let (mn, q1, med, q3, mx) = five_number_summary(&[7.0, 1.0, 3.0, 5.0, 9.0]);
        close(mn, 1.0, 1e-12);
        close(q1, 3.0, 1e-12);
        close(med, 5.0, 1e-12);
        close(q3, 7.0, 1e-12);
        close(mx, 9.0, 1e-12);
        // MAD of {1,3,5,7,9}: median 5, |dev| {4,2,0,2,4}, median 2.
        close(median_abs_deviation(&[1.0, 3.0, 5.0, 7.0, 9.0]), 2.0, 1e-12);
    }

    #[test]
    fn input_is_not_mutated() {
        let d = [3.0, 1.0, 2.0];
        let _ = median(&d);
        assert_eq!(d, [3.0, 1.0, 2.0]); // unchanged — we sort a copy
    }
}
