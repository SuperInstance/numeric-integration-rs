//! Gaussian quadrature using Legendre polynomial nodes and weights.
//!
//! Approximates ∫ₐᵇ f(x) dx by transforming to [-1, 1] and evaluating
//! at Gauss-Legendre nodes. Exact for polynomials up to degree 2n−1.

/// Integrate `f` over [a, b] using n-point Gauss-Legendre quadrature.
///
/// # Arguments
///
/// * `f` — The integrand
/// * `a` — Lower bound
/// * `b` — Upper bound
/// * `n` — Number of quadrature points (1–5 supported; higher values use n=5)
pub fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let (nodes, weights) = gauss_legendre(n);
    let mid = (a + b) / 2.0;
    let half = (b - a) / 2.0;

    let mut sum = 0.0;
    for i in 0..nodes.len() {
        let x = mid + half * nodes[i];
        sum += weights[i] * f(x);
    }
    sum * half
}

/// Return Gauss-Legendre nodes and weights for n points on [-1, 1].
///
/// Supports n = 1 through 5 with hardcoded values for maximum accuracy.
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (
            vec![0.0],
            vec![2.0],
        ),
        2 => {
            let s = (1.0_f64 / 3.0).sqrt();
            (vec![-s, s], vec![1.0, 1.0])
        }
        3 => {
            let s = (3.0_f64 / 5.0).sqrt();
            (
                vec![-s, 0.0, s],
                vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0],
            )
        }
        4 => {
            // Standard 4-point Gauss-Legendre nodes and weights
            (
                vec![
                    -0.8611363115940526,
                    -0.3399810435848563,
                    0.3399810435848563,
                    0.8611363115940526,
                ],
                vec![
                    0.3478548451374538,
                    0.6521451548625461,
                    0.6521451548625461,
                    0.3478548451374538,
                ],
            )
        }
        _ => {
            // 5-point (and default for n > 5)
            (
                vec![
                    -0.906_179_845_938_664,
                    -0.5384693101056831,
                    0.0,
                    0.5384693101056831,
                    0.906_179_845_938_664,
                ],
                vec![
                    0.2369268850561891,
                    0.4786286704993665,
                    0.5688888888888889,
                    0.4786286704993665,
                    0.2369268850561891,
                ],
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_1point_linear() {
        // 1-point is exact for linear: ∫₀¹ x dx = 0.5
        let val = integrate(|x| x, 0.0, 1.0, 1);
        assert!((val - 0.5).abs() < 1e-10, "got {}", val);
    }

    #[test]
    fn test_gauss_2point_cubic() {
        // 2-point exact up to degree 3: ∫₀¹ x³ dx = 0.25
        let val = integrate(|x| x * x * x, 0.0, 1.0, 2);
        assert!((val - 0.25).abs() < 1e-10, "got {}", val);
    }
}
