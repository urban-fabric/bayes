//! Adversarial inputs to the statistics surface.
//!
//! Two questions, split across the modules below: does a reduction stay accurate
//! when the numbers get hostile, and does it *decline* gracefully when the input
//! carries no information at all.
//!
//! | module | asks |
//! |--------|------|
//! | [`stability`] | does accuracy survive cancellation and extreme scales? |
//! | [`degenerate`] | is degenerate input `NaN` rather than a panic? |
//! | [`structures`] | do intervals, weights and designs decline when they must? |

mod degenerate;
mod stability;
mod structures;
