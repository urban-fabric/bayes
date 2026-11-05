//! Conjugate updates: against Bayes' rule, and against their own algebra.
//!
//! | module | asks |
//! |--------|------|
//! | [`vs_bayes_rule`] | does the shortcut match the definition it shortcuts? |
//! | [`algebraic`] | is the update associative, and does the posterior concentrate? |

mod algebraic;
mod vs_bayes_rule;
