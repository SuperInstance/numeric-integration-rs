//! Adaptive Simpson's quadrature.

#![allow(clippy::too_many_arguments)]
//!
//! Recursively subdivides the integration interval, using Simpson's rule
//! with an error estimate to decide when a subinterval is sufficiently accurate.

/// Integrate `f` over [a, b] using adaptive Simpson's quadrature.
///
/// Recursively refines subintervals until the estimated error falls below `tol`.
///
/// # Arguments
///
/// * `f` — The integrand
/// * `a` — Lower bound
/// * `b` — Upper bound
/// * `tol` — Error tolerance
pub fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64) -> f64 {
    let fa = f(a);
    let fb = f(b);
    let m = (a + b) / 2.0;
    let fm = f(m);
    let whole = simpson_basic(fa, fm, fb, a, b);
    adaptive_step(&f, a, b, fa, fm, fb, whole, tol, 50)
}

/// Basic Simpson's rule on [a, b] given function values at a, m, b.
fn simpson_basic(fa: f64, fm: f64, fb: f64, a: f64, b: f64) -> f64 {
    (b - a) / 6.0 * (fa + 4.0 * fm + fb)
}

/// Recursive adaptive step.
fn adaptive_step<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fm: f64,
    fb: f64,
    whole: f64,
    tol: f64,
    max_depth: usize,
) -> f64 {
    let m = (a + b) / 2.0;
    let m1 = (a + m) / 2.0;
    let m2 = (m + b) / 2.0;
    let fm1 = f(m1);
    let fm2 = f(m2);
    let left = simpson_basic(fa, fm1, fm, a, m);
    let right = simpson_basic(fm, fm2, fb, m, b);
    let combined = left + right;

    let err = (combined - whole) / 15.0; // Richardson error estimate

    if max_depth == 0 || err.abs() < tol {
        combined + err // Error correction
    } else {
        let left_val = adaptive_step(f, a, m, fa, fm1, fm, left, tol / 2.0, max_depth - 1);
        let right_val = adaptive_step(f, m, b, fm, fm2, fb, right, tol / 2.0, max_depth - 1);
        left_val + right_val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_zero_width() {
        let val = integrate(|x| x, 0.0, 0.0, 1e-10);
        assert!(val.abs() < 1e-15);
    }

    #[test]
    fn test_adaptive_exp() {
        let val = integrate(|x| x.exp(), 0.0, 1.0, 1e-10);
        assert!((val - (std::f64::consts::E - 1.0)).abs() < 1e-8, "got {}", val);
    }
}
