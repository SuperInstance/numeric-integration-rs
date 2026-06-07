//! Romberg integration using Richardson extrapolation.

#![allow(clippy::needless_range_loop)]
//!
//! Starts with the trapezoidal rule and applies extrapolation to
//! achieve high-order accuracy rapidly.

use crate::trapezoidal;

/// Integrate `f` over [a, b] using Romberg integration with `levels` extrapolation steps.
///
/// The Romberg method builds a table where each level uses successively
/// more refined trapezoidal estimates combined with Richardson extrapolation.
///
/// # Arguments
///
/// * `f` — The integrand
/// * `a` — Lower bound
/// * `b` — Upper bound
/// * `levels` — Number of extrapolation levels (≥ 1)
pub fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, levels: usize) -> f64 {
    if levels == 0 || a == b {
        return 0.0;
    }

    // Build Romberg table
    let n = levels;
    let mut r = vec![vec![0.0; n]; n];

    // First column: trapezoidal estimates with 2^k segments
    for k in 0..n {
        let segs = 2usize.pow(k as u32).max(1);
        r[k][0] = trapezoidal::integrate(&f, a, b, segs);
    }

    // Richardson extrapolation
    for j in 1..n {
        let four_j = 4.0_f64.powi(j as i32);
        for k in j..n {
            r[k][j] = (four_j * r[k][j - 1] - r[k - 1][j - 1]) / (four_j - 1.0);
        }
    }

    r[n - 1][n - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_romberg_sin() {
        let val = integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 8);
        assert!((val - 2.0).abs() < 1e-10, "got {}", val);
    }
}
