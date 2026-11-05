//! Multiple-comparison corrections.
//!
//! Run twenty independent tests at `alpha = 0.05` on data where nothing is going
//! on and you expect one significant result. Run a thousand and you expect
//! fifty. That is not a subtlety — it is the single most common way a study
//! reports a finding that is not there, and the reason a correction is not
//! optional once you are testing more than one hypothesis.
//!
//! Two different things can be controlled, and choosing between them is the
//! whole decision:
//!
//! **Family-wise error rate (FWER)** — the probability of *any* false positive
//! among all the tests. [`bonferroni`], [`holm`], [`sidak`] and [`holm_sidak`]
//! control this. Use it when a single false claim is expensive: a drug approval,
//! a published effect, a production rollout.
//!
//! **False discovery rate (FDR)** — the expected *proportion* of false positives
//! among the tests you called significant. [`benjamini_hochberg`] and
//! [`benjamini_yekutieli`] control this. Use it when you are screening: ten
//! thousand genes, a feature sweep, anything where the output is a shortlist for
//! follow-up and a few false leads are acceptable.
//!
//! FDR is the less conservative of the two by a wide margin at large `m`, and
//! that is the point rather than a defect. Bonferroni at `m = 10000` and
//! `alpha = 0.05` demands `p < 5e-6` of every test; Benjamini-Hochberg will
//! accept a `p` of `0.01` if enough of its neighbours are smaller still.
//!
//! Every function here returns **adjusted p-values**, directly comparable to the
//! original `alpha`, rather than an adjusted threshold. That is the more useful
//! form: it composes, it survives being put in a table, and it does not require
//! the reader to remember which `alpha` was in force.

mod adjust;
mod step;

pub use adjust::{benjamini_hochberg, benjamini_yekutieli, bonferroni, holm, holm_sidak, sidak};
pub use step::{Correction, reject_at};
