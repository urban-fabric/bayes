//! Rank-based correlation: **Spearman's ρ** and **Kendall's τ-b** — robust,
//! distribution-free measures of monotone association (the Pearson coefficient is
//! in [`super::summary::correlation`]).

use super::summary::correlation as pearson;

fn valid_table(table: &[Vec<f64>]) -> Option<(usize, usize)> {
    let r = table.len();
    let c = table.first().map_or(0, Vec::len);
    if r < 2
        || c < 2
        || table.iter().any(|row| row.len() != c)
        || table.iter().flatten().any(|x| !x.is_finite() || *x < 0.0)
    {
        return None;
    }
    Some((r, c))
}

/// Fractional ranks of `data` (1-based), assigning the **average rank** to ties.
pub fn ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        data[a]
            .partial_cmp(&data[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && data[idx[j]] == data[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &k in &idx[i..j] {
            out[k] = avg;
        }
        i = j;
    }
    out
}

/// Spearman's rank correlation `ρ` — the Pearson correlation of the ranks.
/// `NaN` for mismatched lengths or fewer than 2 points.
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return f64::NAN;
    }
    pearson(&ranks(a), &ranks(b))
}

/// Kendall's `τ-b` rank correlation, with tie correction. `NaN` for mismatched
/// lengths or fewer than 2 points.
pub fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n != b.len() || n < 2 {
        return f64::NAN;
    }
    let (mut concordant, mut discordant) = (0i64, 0i64);
    let (mut ties_a, mut ties_b) = (0i64, 0i64);
    for i in 0..n {
        for j in (i + 1)..n {
            let da = (a[i] - a[j]).signum();
            let db = (b[i] - b[j]).signum();
            if a[i] == a[j] && b[i] == b[j] {
            } else if a[i] == a[j] {
                ties_a += 1;
            } else if b[i] == b[j] {
                ties_b += 1;
            } else if (da * db) > 0.0 {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let n0 = (n * (n - 1) / 2) as f64;
    let denom = ((n0 - ties_a as f64) * (n0 - ties_b as f64)).sqrt();
    if denom == 0.0 {
        return f64::NAN;
    }
    (concordant - discordant) as f64 / denom
}

/// **Cramér's V** — association strength for a contingency `table` (rows × cols of
/// counts), in `[0, 1]`: `√(χ² / (n·(min(r, c) − 1)))`. The nominal-variable analogue
/// of a correlation coefficient. `NaN` for a degenerate table.
pub fn cramers_v(table: &[Vec<f64>]) -> f64 {
    let Some((r, c)) = valid_table(table) else {
        return f64::NAN;
    };
    let row_sums: Vec<f64> = table.iter().map(|row| row.iter().sum()).collect();
    let col_sums: Vec<f64> = (0..c)
        .map(|j| table.iter().map(|row| row[j]).sum())
        .collect();
    let total: f64 = row_sums.iter().sum();
    if total <= 0.0 {
        return f64::NAN;
    }
    let mut chi2 = 0.0;
    for i in 0..r {
        for j in 0..c {
            let e = row_sums[i] * col_sums[j] / total;
            if e > 0.0 {
                let diff = table[i][j] - e;
                chi2 += diff * diff / e;
            }
        }
    }
    let k = r.min(c) as f64;
    (chi2 / (total * (k - 1.0))).sqrt()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_distribution_ext::assert_close;

    #[test]
    fn perfect_monotone_relationships() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 20.0, 30.0, 40.0, 50.0];
        assert_close!(spearman(&x, &y), 1.0, 1e-12);
        assert_close!(kendall_tau(&x, &y), 1.0, 1e-12);
        let z = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert_close!(spearman(&x, &z), -1.0, 1e-12);
        assert_close!(kendall_tau(&x, &z), -1.0, 1e-12);
    }

    #[test]
    fn spearman_beats_pearson_on_monotone_nonlinear() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|v| v.powi(3)).collect();
        assert_close!(spearman(&x, &y), 1.0, 1e-12);
    }

    #[test]
    fn ranks_average_ties() {
        let r = ranks(&[10.0, 20.0, 20.0, 40.0]);
        assert_eq!(r, vec![1.0, 2.5, 2.5, 4.0]);
    }

    #[test]
    fn cramers_v_bounds() {
        assert_close!(cramers_v(&[vec![10.0, 0.0], vec![0.0, 10.0]]), 1.0, 1e-12);
        assert_close!(cramers_v(&[vec![10.0, 20.0], vec![20.0, 40.0]]), 0.0, 1e-12);
    }
}
