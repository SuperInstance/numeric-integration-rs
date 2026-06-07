//! Trapezoidal rule for numerical integration.
//!
//! Approximates ∫ₐᵇ f(x) dx by dividing [a, b] into `n` segments
//! and summing trapezoid areas.

/// Integrate `f` over [a, b] using the composite trapezoidal rule with `n` segments.
///
/// # Arguments
///
/// * `f` — The integrand
/// * `a` — Lower bound
/// * `b` — Upper bound
/// * `n` — Number of segments (must be ≥ 1)
///
/// # Panics
///
/// Panics if `n` is 0.
pub fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    assert!(n > 0, "n must be >= 1");
    let h = (b - a) / n as f64;
    let mut sum = (f(a) + f(b)) / 2.0;
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    sum * h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trap_sin() {
        // ∫₀^π sin(x) dx = 2
        let val = integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 1000);
        assert!((val - 2.0).abs() < 1e-4, "got {}", val);
    }
}
