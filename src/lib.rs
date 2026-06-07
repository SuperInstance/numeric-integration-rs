//! # numeric-integration
//!
//! Pure-Rust numerical integration methods: trapezoidal rule, Simpson's rule,
//! Romberg integration, Gaussian quadrature, and adaptive quadrature.
//!
//! All methods operate on closures `F: Fn(f64) -> f64` for maximum flexibility.
//!
//! ## Modules
//!
//! - [`trapezoidal`] — Trapezoidal rule integration
//! - [`simpson`] — Simpson's 1/3 rule
//! - [`romberg`] — Romberg extrapolation integration
//! - [`gaussian`] — Gaussian quadrature (Legendre nodes)
//! - [`adaptive`] — Adaptive Simpson's quadrature

pub mod trapezoidal;
pub mod simpson;
pub mod romberg;
pub mod gaussian;
pub mod adaptive;

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    // ── Trapezoidal tests ────────────────────────────────────────────────────

    #[test]
    fn test_trap_constant() {
        // ∫₀¹ 5 dx = 5
        let val = trapezoidal::integrate(|_| 5.0, 0.0, 1.0, 10);
        assert!((val - 5.0).abs() < TOL, "got {}", val);
    }

    #[test]
    fn test_trap_linear() {
        // ∫₀¹ x dx = 0.5
        let val = trapezoidal::integrate(|x| x, 0.0, 1.0, 100);
        assert!((val - 0.5).abs() < TOL, "got {}", val);
    }

    #[test]
    fn test_trap_quadratic() {
        // ∫₀¹ x² dx = 1/3
        let val = trapezoidal::integrate(|x| x * x, 0.0, 1.0, 1000);
        assert!((val - 1.0 / 3.0).abs() < 1e-4, "got {}", val);
    }

    #[test]
    fn test_trap_zero_width() {
        let val = trapezoidal::integrate(|x| x, 0.0, 0.0, 10);
        assert!(val.abs() < 1e-15);
    }

    #[test]
    fn test_trap_error_decreases() {
        let f = |x: f64| x * x * x;
        let v10 = trapezoidal::integrate(&f, 0.0, 1.0, 10);
        let v100 = trapezoidal::integrate(&f, 0.0, 1.0, 100);
        let v1000 = trapezoidal::integrate(&f, 0.0, 1.0, 1000);
        let exact = 0.25;
        let e10 = (v10 - exact).abs();
        let e100 = (v100 - exact).abs();
        let e1000 = (v1000 - exact).abs();
        assert!(e1000 < e100, "errors: {}, {}, {}", e10, e100, e1000);
        assert!(e100 < e10, "errors: {}, {}, {}", e10, e100, e1000);
    }

    // ── Simpson tests ────────────────────────────────────────────────────────

    #[test]
    fn test_simpson_constant() {
        let val = simpson::integrate(|_| 3.0, 0.0, 2.0, 100);
        assert!((val - 6.0).abs() < TOL, "got {}", val);
    }

    #[test]
    fn test_simpson_quadratic_exact() {
        // Simpson's rule is exact for cubics!
        // ∫₀¹ x³ dx = 0.25
        let val = simpson::integrate(|x| x * x * x, 0.0, 1.0, 100);
        assert!((val - 0.25).abs() < TOL, "got {}", val);
    }

    #[test]
    fn test_simpson_polynomial_exact() {
        // ∫₀² (x³ - 2x² + x) dx = [x⁴/4 - 2x³/3 + x²/2]₀² = 4 - 16/3 + 2 = -2/3
        let val = simpson::integrate(|x| x*x*x - 2.0*x*x + x, 0.0, 2.0, 100);
        let exact = 4.0 - 16.0 / 3.0 + 2.0;
        assert!((val - exact).abs() < TOL, "got {} expected {}", val, exact);
    }

    #[test]
    fn test_simpson_pi_circle() {
        // ∫₀¹ 4/(1+x²) dx = π
        let val = simpson::integrate(|x| 4.0 / (1.0 + x * x), 0.0, 1.0, 1000);
        assert!((val - std::f64::consts::PI).abs() < 1e-8, "got {}", val);
    }

    #[test]
    fn test_simpson_error_decreases() {
        let f = |x: f64| x.sin();
        let v10 = simpson::integrate(&f, 0.0, std::f64::consts::PI, 10);
        let v100 = simpson::integrate(&f, 0.0, std::f64::consts::PI, 100);
        let exact = 2.0;
        assert!((v100 - exact).abs() < (v10 - exact).abs());
    }

    // ── Romberg tests ────────────────────────────────────────────────────────

    #[test]
    fn test_romberg_constant() {
        let val = romberg::integrate(|_| 7.0, 0.0, 1.0, 5);
        assert!((val - 7.0).abs() < TOL, "got {}", val);
    }

    #[test]
    fn test_romberg_polynomial() {
        // ∫₀¹ x⁴ dx = 1/5
        let val = romberg::integrate(|x| x.powi(4), 0.0, 1.0, 10);
        assert!((val - 0.2).abs() < 1e-8, "got {}", val);
    }

    #[test]
    fn test_romberg_pi() {
        let val = romberg::integrate(|x| 4.0 / (1.0 + x * x), 0.0, 1.0, 10);
        assert!((val - std::f64::consts::PI).abs() < 1e-10, "got {}", val);
    }

    #[test]
    fn test_romberg_convergence() {
        let f = |x: f64| x.exp();
        let val = romberg::integrate(&f, 0.0, 1.0, 8);
        let exact = std::f64::consts::E - 1.0;
        assert!((val - exact).abs() < 1e-10, "got {} expected {}", val, exact);
    }

    // ── Gaussian quadrature tests ────────────────────────────────────────────

    #[test]
    fn test_gauss_constant() {
        let val = gaussian::integrate(|_| 3.0, 0.0, 1.0, 3);
        assert!((val - 3.0).abs() < TOL, "got {}", val);
    }

    #[test]
    fn test_gauss_linear() {
        let val = gaussian::integrate(|x| 2.0 * x, 0.0, 1.0, 2);
        assert!((val - 1.0).abs() < TOL, "got {}", val);
    }

    #[test]
    fn test_gauss_cubic_exact() {
        // Gauss-Legendre with n≥2 is exact for polynomials up to degree 2n-1
        // With n=3, exact up to degree 5
        // ∫₀¹ x³ dx = 0.25
        let val = gaussian::integrate(|x| x * x * x, 0.0, 1.0, 2);
        assert!((val - 0.25).abs() < 1e-10, "got {}", val);
    }

    #[test]
    fn test_gauss_quintic_exact() {
        // With n=3, should be exact for degree 5
        // ∫₀¹ x⁵ dx = 1/6
        let val = gaussian::integrate(|x| x.powi(5), 0.0, 1.0, 3);
        assert!((val - 1.0 / 6.0).abs() < 1e-10, "got {}", val);
    }

    #[test]
    fn test_gauss_pi() {
        let val = gaussian::integrate(|x| 4.0 / (1.0 + x * x), 0.0, 1.0, 5);
        assert!((val - std::f64::consts::PI).abs() < 1e-7, "got {}", val);
    }

    #[test]
    fn test_gauss_negative_interval() {
        let val = gaussian::integrate(|x| x, 1.0, 0.0, 3);
        assert!((val - (-0.5)).abs() < TOL, "got {}", val);
    }

    // ── Adaptive tests ───────────────────────────────────────────────────────

    #[test]
    fn test_adaptive_constant() {
        let val = adaptive::integrate(|_| 5.0, 0.0, 1.0, 1e-10);
        assert!((val - 5.0).abs() < TOL, "got {}", val);
    }

    #[test]
    fn test_adaptive_polynomial() {
        // ∫₀¹ x² dx = 1/3
        let val = adaptive::integrate(|x| x * x, 0.0, 1.0, 1e-10);
        assert!((val - 1.0 / 3.0).abs() < TOL, "got {}", val);
    }

    #[test]
    fn test_adaptive_pi() {
        let val = adaptive::integrate(|x| 4.0 / (1.0 + x * x), 0.0, 1.0, 1e-10);
        assert!((val - std::f64::consts::PI).abs() < 1e-8, "got {}", val);
    }

    #[test]
    fn test_adaptive_sharp_function() {
        // Gaussian bump: ∫₋₁¹ exp(-100x²) dx — should converge
        let val = adaptive::integrate(|x| (-100.0 * x * x).exp(), -1.0, 1.0, 1e-8);
        // Exact value ≈ sqrt(π/100) ≈ 0.1772...
        let exact = (std::f64::consts::PI / 100.0).sqrt();
        assert!((val - exact).abs() < 1e-4, "got {} expected {}", val, exact);
    }

    #[test]
    fn test_adaptive_sin() {
        // ∫₀^π sin(x) dx = 2
        let val = adaptive::integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-10);
        assert!((val - 2.0).abs() < 1e-6, "got {}", val);
    }

    // ── Cross-method comparison tests ────────────────────────────────────────

    #[test]
    fn test_all_methods_agree_on_polynomial() {
        // ∫₀² 3x² dx = 8
        let f = |x: f64| 3.0 * x * x;
        let t = trapezoidal::integrate(&f, 0.0, 2.0, 1000);
        let s = simpson::integrate(&f, 0.0, 2.0, 100);
        let r = romberg::integrate(&f, 0.0, 2.0, 10);
        let g = gaussian::integrate(&f, 0.0, 2.0, 5);
        let a = adaptive::integrate(&f, 0.0, 2.0, 1e-10);
        let exact = 8.0;
        assert!((t - exact).abs() < 1e-2, "trap: {}", t);
        assert!((s - exact).abs() < 1e-6, "simpson: {}", s);
        assert!((r - exact).abs() < 1e-6, "romberg: {}", r);
        assert!((g - exact).abs() < 1e-6, "gauss: {}", g);
        assert!((a - exact).abs() < 1e-6, "adaptive: {}", a);
    }
}
