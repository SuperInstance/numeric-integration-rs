//! Simpson's 1/3 rule for numerical integration.
//!
//! Uses quadratic interpolation between triplets of points for higher
//! accuracy than the trapezoidal rule. Exact for polynomials up to degree 3.

/// Integrate `f` over [a, b] using composite Simpson's 1/3 rule with `n` segments.
///
/// `n` must be even. If `n` is odd, it will be incremented by 1.
///
/// # Arguments
///
/// * `f` — The integrand
/// * `a` — Lower bound
/// * `b` — Upper bound
/// * `n` — Number of segments (will be rounded up to even)
pub fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = if n.is_multiple_of(2) { n } else { n + 1 };
    if n == 0 {
        return 0.0;
    }
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(x);
    }
    sum * h / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simpson_exp() {
        // ∫₀¹ eˣ dx = e - 1
        let val = integrate(|x| x.exp(), 0.0, 1.0, 100);
        assert!((val - (std::f64::consts::E - 1.0)).abs() < 1e-8, "got {}", val);
    }
}
