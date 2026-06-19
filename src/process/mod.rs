//! **Probabilistic processes** — the generative half of `bayes`: sampling whole
//! stochastic processes, not just i.i.d. draws.
//!
//! * [`GaussianRandomWalk`] / [`wiener_path`] — random walks and Brownian motion.
//! * [`PoissonProcess`] — random points in time (homogeneous + inhomogeneous).
//! * The covariance [`Kernel`]s (RBF, Matérn, periodic, linear) used by Gaussian
//!   processes.
//!
//! These are *generative* — they define and sample random processes. Inference
//! over them (MCMC, VI) is out of scope and belongs to a higher layer. The
//! matrix-coupled processes — `GaussianProcess` (GP regression) and `MarkovChain`
//! (transition-matrix trajectories) — live in the sibling crate `ito`, built on
//! these kernels.

mod kernel;
mod poisson_process;
mod random_walk;

pub use kernel::{Kernel, Linear, Matern, MaternNu, Periodic, Rbf};
pub use poisson_process::{PoissonProcess, inhomogeneous_arrivals};
pub use random_walk::{GaussianRandomWalk, wiener_path};
