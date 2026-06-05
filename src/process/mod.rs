//! **Probabilistic processes** — the generative half of `bayes`: sampling whole
//! stochastic processes, not just i.i.d. draws.
//!
//! * [`GaussianProcess`] — distributions over functions, defined by a covariance
//!   [`Kernel`]; sample from the prior or condition on data (GP regression).
//! * [`GaussianRandomWalk`] / [`wiener_path`] — random walks and Brownian motion.
//! * [`PoissonProcess`] — random points in time (homogeneous + inhomogeneous).
//! * [`MarkovChain`] — state trajectories from a transition matrix.
//!
//! These are *generative* — they define and sample random processes. Inference
//! over them (MCMC, VI) is out of scope and belongs to a higher layer.

mod gaussian;
mod kernel;
mod markov;
mod poisson_process;
mod random_walk;

pub use gaussian::{GaussianProcess, GpPosterior};
pub use kernel::{Kernel, Linear, Matern, MaternNu, Periodic, Rbf};
pub use markov::MarkovChain;
pub use poisson_process::{PoissonProcess, inhomogeneous_arrivals};
pub use random_walk::{GaussianRandomWalk, wiener_path};
