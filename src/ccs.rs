//! Drift-time and collision cross section calibration.
//!
//! Two separate things live here:
//!
//! * The **drift axis**, which every file has: a frame's drift scans are evenly
//!   spaced by `frm-dt-period` milliseconds, so scan `s` arrives at
//!   `s * period`. See [`DriftAxis`].
//!
//! * The **CCS calibration**, stored as JSON under the global `cal-ccs`
//!   attribute, which converts arrival time to collision cross section. Files
//!   acquired without a CCS calibration simply omit it.
//!
//! The CCS model is
//!
//! ```text
//! CCS(at, mz, z) = P(at) * z / sqrt(mu),   mu = mz * gas_mass / (mz + gas_mass)
//! ```
//!
//! where `P` is the stored polynomial in arrival time (milliseconds, lowest-order
//! coefficient first). Note that the reduced mass is computed from **m/z**, not
//! from the neutral mass `mz * z` — physically surprising, but it is what the
//! vendor SDK does, and a reader that "corrects" it would disagree with every
//! CCS the instrument software reports.

use serde::Deserialize;

/// Evenly spaced drift-scan arrival times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftAxis {
    /// Time between consecutive drift scans, in milliseconds (`frm-dt-period`).
    pub period_ms: f64,
    /// Number of drift scans in the frame.
    pub n_scans: usize,
}

impl DriftAxis {
    /// Arrival time of a drift scan, in milliseconds. Scan 0 arrives at 0.
    #[inline]
    pub fn arrival_time_ms(&self, scan: usize) -> f64 {
        scan as f64 * self.period_ms
    }

    /// Arrival times of every scan.
    pub fn arrival_times_ms(&self) -> Vec<f64> {
        (0..self.n_scans).map(|s| self.arrival_time_ms(s)).collect()
    }

    /// The drift scan whose arrival time is nearest `t_ms`, clamped to the frame.
    pub fn scan_at(&self, t_ms: f64) -> usize {
        if self.period_ms <= 0.0 || t_ms <= 0.0 {
            return 0;
        }
        let s = (t_ms / self.period_ms).round() as usize;
        s.min(self.n_scans.saturating_sub(1))
    }
}

/// A CCS calibration, as stored in the global `cal-ccs` attribute.
#[derive(Debug, Clone, PartialEq)]
pub struct CcsCalibration {
    /// Polynomial in arrival time (ms), lowest-order coefficient first.
    pub coefficients: Vec<f64>,
    /// Lower bound of the calibrated CCS range.
    pub min_ccs: f64,
    /// Upper bound of the calibrated CCS range.
    pub max_ccs: f64,
    /// Earliest valid arrival time, in milliseconds. A validity bound only: it
    /// does *not* shift the polynomial's argument.
    pub at_surfing: f64,
    /// Drift gas mass in Daltons (N2 is 28.0134).
    pub gas_mass: f64,
    /// Drift gas as named in the calibration, if present.
    pub gas_type: Option<String>,
    /// Calibration format version, if present.
    pub version: Option<String>,
}

#[derive(Deserialize)]
struct CcsCalJson {
    #[serde(default)]
    coefficients: Vec<f64>,
    #[serde(default)]
    min: f64,
    #[serde(default)]
    max: f64,
    #[serde(default)]
    at_surfing: f64,
    #[serde(default, rename = "gas mass")]
    gas_mass: Option<f64>,
    #[serde(default, rename = "Mass Flow.gas type")]
    gas_type: Option<String>,
    #[serde(default)]
    version: Option<String>,
}

/// Mass of N2, the default drift gas.
pub const GAS_MASS_N2: f64 = 28.0134;

/// Whether a calibration predates the coefficient-scaling change.
///
/// Observed against the SDK: absent, empty and `0.0.1` take the legacy path;
/// `0.9.9`, `1.0.0`, `1.0.1` and `2.5.0` do not. A version that is present must
/// be `X.Y.Z` — the SDK rejects `1.0` and `1` outright.
fn is_legacy_version(version: Option<&str>) -> bool {
    let Some(v) = version.map(str::trim).filter(|v| !v.is_empty()) else {
        return true;
    };
    let mut parts = v.split('.').map(|p| p.parse::<u32>());
    match (parts.next(), parts.next()) {
        (Some(Ok(major)), Some(Ok(minor))) => major == 0 && minor == 0,
        _ => false,
    }
}

impl CcsCalibration {
    /// Build from coefficients already in **lowest-order-first** order, i.e. the
    /// order the SDK's own `GetCCSCoefficients` and de novo constructor use.
    pub fn from_coefficients(coefficients: Vec<f64>, gas_mass: f64) -> Self {
        Self {
            coefficients,
            min_ccs: 0.0,
            max_ccs: f64::INFINITY,
            at_surfing: 0.0,
            gas_mass,
            gas_type: None,
            version: None,
        }
    }

    /// Parse the global `cal-ccs` attribute.
    ///
    /// Two things about the stored form are easy to get wrong, and both silently
    /// produce plausible-looking but wrong cross sections:
    ///
    /// 1. The JSON `coefficients` are **highest-order first** — the reverse of
    ///    the order the SDK's API hands them back. They are reversed here.
    /// 2. Calibrations written by early software (version `0.0.x`, or with no
    ///    version at all) store coefficients that omit a `sqrt(gas_mass)`
    ///    factor, which the SDK multiplies back in on load.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let c: CcsCalJson = serde_json::from_str(json)?;
        let gas_mass = c.gas_mass.unwrap_or(GAS_MASS_N2);

        // Stored highest-order first; we keep lowest-order first internally.
        let mut coefficients = c.coefficients;
        coefficients.reverse();

        if is_legacy_version(c.version.as_deref()) {
            let scale = gas_mass.sqrt();
            for v in &mut coefficients {
                *v *= scale;
            }
        }

        Ok(Self {
            coefficients,
            min_ccs: c.min,
            max_ccs: c.max,
            at_surfing: c.at_surfing,
            gas_mass,
            gas_type: c.gas_type,
            version: c.version,
        })
    }

    /// Degree of the stored polynomial.
    pub fn degree(&self) -> usize {
        self.coefficients.len().saturating_sub(1)
    }

    /// The calibrated CCS range.
    pub fn calibrated_range(&self) -> (f64, f64) {
        (self.min_ccs, self.max_ccs)
    }

    /// `sqrt(mu) / z` — divide a reduced CCS by this to get the real one.
    ///
    /// Matches the SDK's `ReductionFactor`.
    #[inline]
    pub fn reduction_factor(&self, mz: f64, z: i32) -> f64 {
        let mu = mz * self.gas_mass / (mz + self.gas_mass);
        mu.sqrt() / z as f64
    }

    /// The polynomial evaluated at an arrival time, i.e. the reduced CCS.
    #[inline]
    pub fn reduced_ccs(&self, at_ms: f64) -> f64 {
        self.coefficients
            .iter()
            .rev()
            .fold(0.0, |acc, &c| acc * at_ms + c)
    }

    /// Arrival time (ms) -> CCS in square angstroms.
    #[inline]
    pub fn arrival_time_to_ccs(&self, at_ms: f64, mz: f64, z: i32) -> f64 {
        self.reduced_ccs(at_ms) / self.reduction_factor(mz, z)
    }

    /// Scale a CCS into reduced units (multiply by the reduction factor).
    #[inline]
    pub fn reduce_ccs(&self, ccs: f64, mz: f64, z: i32) -> f64 {
        ccs * self.reduction_factor(mz, z)
    }

    /// Scale a reduced CCS back to real units.
    #[inline]
    pub fn unreduce_ccs(&self, reduced: f64, mz: f64, z: i32) -> f64 {
        reduced / self.reduction_factor(mz, z)
    }

    /// CCS -> arrival time in milliseconds.
    ///
    /// Degree 1 inverts in closed form. Higher degrees are solved numerically,
    /// since the SDK's own polynomials are not analytically invertible above
    /// cubic and it uses a root search there too.
    /// Returns `None` for non-finite or unphysical inputs, and for a calibration
    /// whose polynomial the solver cannot invert to a converged root.
    pub fn ccs_to_arrival_time(&self, ccs: f64, mz: f64, z: i32) -> Option<f64> {
        if !ccs.is_finite() || !mz.is_finite() || mz <= 0.0 || z == 0 || !self.gas_mass.is_finite()
        {
            return None;
        }
        let target = self.reduce_ccs(ccs, mz, z);
        if !target.is_finite() {
            return None;
        }
        match self.coefficients.len() {
            0 => None,
            1 => None, // constant polynomial: no arrival time maps to a CCS
            2 => {
                let (c0, c1) = (self.coefficients[0], self.coefficients[1]);
                if c1 == 0.0 {
                    None
                } else {
                    Some((target - c0) / c1)
                }
            }
            _ => self.solve_reduced(target),
        }
    }

    /// Newton with a bisection fallback, over a bracket that starts at the
    /// calibration's own validity floor.
    fn solve_reduced(&self, target: f64) -> Option<f64> {
        let f = |t: f64| self.reduced_ccs(t) - target;

        // Bracket by expanding from at_surfing until the sign flips.
        let lo0 = self.at_surfing.max(0.0);
        let mut lo = lo0;
        let mut hi = (lo0 + 1.0).max(1.0);
        let mut flo = f(lo);
        let mut fhi = f(hi);
        let mut expansions = 0;
        while flo.signum() == fhi.signum() {
            hi *= 2.0;
            fhi = f(hi);
            expansions += 1;
            if expansions > 60 {
                return None;
            }
        }

        // Newton from the bracket midpoint, falling back to bisection whenever a
        // step would leave the bracket or the derivative vanishes.
        let mut t = 0.5 * (lo + hi);
        for _ in 0..100 {
            let ft = f(t);
            if ft.abs() < 1e-12 {
                return Some(t);
            }
            if ft.signum() == flo.signum() {
                lo = t;
                flo = ft;
            } else {
                hi = t;
                fhi = ft;
            }
            let d = self.derivative(t);
            let next = if d == 0.0 { f64::NAN } else { t - ft / d };
            t = if next.is_finite() && next > lo && next < hi {
                next
            } else {
                0.5 * (lo + hi)
            };
            if (hi - lo).abs() < 1e-13 {
                break;
            }
        }
        let _ = fhi;
        // Only hand back a root the iteration actually converged to. Returning the
        // last iterate regardless would report a confident wrong arrival time for a
        // calibration this solver cannot invertturn.
        let residual = f(t).abs();
        let scale = target.abs().max(1.0);
        if t.is_finite() && residual <= 1e-9 * scale {
            Some(t)
        } else {
            None
        }
    }

    #[inline]
    fn derivative(&self, at_ms: f64) -> f64 {
        self.coefficients
            .iter()
            .enumerate()
            .skip(1)
            .map(|(k, &c)| c * k as f64 * at_ms.powi(k as i32 - 1))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Coefficients here are lowest-order first, matching the SDK's de novo
    /// constructor (which is the opposite of the stored JSON order).
    fn cal(coefficients: &[f64], gas_mass: f64) -> CcsCalibration {
        CcsCalibration {
            coefficients: coefficients.to_vec(),
            min_ccs: 0.0,
            max_ccs: 1.0e5,
            at_surfing: 0.0,
            gas_mass,
            gas_type: None,
            version: None,
        }
    }

    /// Reference values produced by the vendor SDK's EyeOnCcsCalibration,
    /// constructed de novo from these coefficients.
    #[test]
    fn arrival_time_to_ccs_matches_sdk() {
        let cases: &[(&[f64], f64, f64, f64, i32, f64)] = &[
            (&[0.0, 1.0], 28.0134, 10.0, 622.0, 1, 1.931448137929251),
            (&[10.0, 2.0], 28.0134, 7.5, 922.0, 2, 9.589290999832132),
            (&[5.0, 0.0, 1.0], 28.0134, 3.0, 100.0, 1, 2.9927665465998845),
            (
                &[1.5, 3.25, -0.05, 0.001],
                28.0134,
                12.5,
                1500.5,
                3,
                20.746752939925237,
            ),
            (&[0.0, 1.0], 4.0026, 10.0, 622.0, 1, 5.014432396574801),
            (&[2.0, 0.5], 39.948, 25.0, 300.25, 2, 4.883990913777644),
        ];
        for &(coef, gas, at, mz, z, expect) in cases {
            let got = cal(coef, gas).arrival_time_to_ccs(at, mz, z);
            assert!(
                (got - expect).abs() < 1e-12,
                "coef={coef:?} gas={gas} at={at} mz={mz} z={z}: got {got}, want {expect}"
            );
        }
    }

    #[test]
    fn reduction_factor_matches_sdk() {
        let c = cal(&[0.0, 1.0], GAS_MASS_N2);
        for &(mz, z, expect) in &[
            (622.0, 1, 5.177462342178768),
            (922.0, 2, 2.607074913091869),
            (1500.5, 3, 1.7480145015950959),
        ] {
            let got = c.reduction_factor(mz, z);
            assert!((got - expect).abs() < 1e-12, "got {got}, want {expect}");
        }
    }

    #[test]
    fn ccs_to_arrival_time_inverts() {
        for (coef, gas, at, mz, z) in [
            (vec![0.0, 1.0], 28.0134, 10.0, 622.0, 1),
            (vec![10.0, 2.0], 28.0134, 7.5, 922.0, 2),
            (vec![5.0, 0.0, 1.0], 28.0134, 3.0, 100.0, 1),
            (vec![1.5, 3.25, -0.05, 0.001], 28.0134, 12.5, 1500.5, 3),
        ] {
            let c = cal(&coef, gas);
            let ccs = c.arrival_time_to_ccs(at, mz, z);
            let back = c.ccs_to_arrival_time(ccs, mz, z).expect("invertible");
            assert!((back - at).abs() < 1e-6, "coef={coef:?}: {back} != {at}");
        }
    }

    #[test]
    fn inverse_rejects_unphysical_input() {
        let c = cal(&[0.0, 1.0], GAS_MASS_N2);
        assert_eq!(c.ccs_to_arrival_time(f64::NAN, 622.0, 1), None);
        assert_eq!(c.ccs_to_arrival_time(100.0, f64::INFINITY, 1), None);
        assert_eq!(c.ccs_to_arrival_time(100.0, 0.0, 1), None, "m/z must be positive");
        assert_eq!(c.ccs_to_arrival_time(100.0, 622.0, 0), None, "charge 0 is not an ion");
    }

    #[test]
    fn inverse_reports_failure_rather_than_a_wrong_root() {
        // A constant polynomial has no arrival time mapping to a given CCS; the
        // solver must say so instead of returning its last iterate.
        let c = cal(&[5.0], GAS_MASS_N2);
        assert_eq!(c.ccs_to_arrival_time(1.0, 622.0, 1), None);
    }

    #[test]
    fn parses_the_stored_json_shape() {
        let json = r#"{"peaks": [], "coefficients": [1.5, 3.25],
                       "min": 120.0, "max": 400.0, "degree": 1,
                       "at_surfing": 2.5, "ccaps": 0,
                       "Mass Flow.gas type": "N2", "gas mass": 28.0134,
                       "version": "1.0.0"}"#;
        let c = CcsCalibration::from_json(json).unwrap();
        // Stored highest-order first, held lowest-order first.
        assert_eq!(c.coefficients, vec![3.25, 1.5]);
        assert_eq!(c.calibrated_range(), (120.0, 400.0));
        assert_eq!(c.at_surfing, 2.5);
        assert_eq!(c.gas_mass, 28.0134);
        assert_eq!(c.gas_type.as_deref(), Some("N2"));
        assert_eq!(c.degree(), 1);
    }

    /// The SDK reversing the stored coefficients, and rescaling the legacy ones,
    /// are both invisible until the numbers come out wrong. Values here were read
    /// back from the SDK parsing these exact JSON blobs out of a real file.
    #[test]
    fn from_json_matches_sdk_file_path() {
        let cases: &[(&str, f64, f64, i32, f64)] = &[
            (
                r#"{"coefficients":[12.5,3.75],"min":100.0,"max":900.0,"degree":1,
                    "at_surfing":0.5,"gas mass":28.0134,"version":"1.0.0"}"#,
                1.435, 622.0, 1, 4.188828149134063,
            ),
            // Same calibration, legacy version: coefficients scale by sqrt(gas mass).
            (
                r#"{"coefficients":[12.5,3.75],"min":100.0,"max":900.0,"degree":1,
                    "at_surfing":0.5,"gas mass":28.0134,"version":"0.0.1"}"#,
                1.435, 622.0, 1, 22.1704983149298,
            ),
            (
                r#"{"coefficients":[1.5,3.25,-0.05,0.001],"min":100.0,"max":900.0,"degree":3,
                    "at_surfing":0.0,"gas mass":28.0134,"version":"1.0.0"}"#,
                12.5, 1500.5, 3, 1966.1598899001049,
            ),
            (
                r#"{"coefficients":[2.0,0.5],"min":100.0,"max":900.0,"degree":1,
                    "at_surfing":0.0,"gas mass":4.0026,"version":"2.0.0"}"#,
                10.0, 922.0, 2, 20.53777556972605,
            ),
            // No version at all: also the legacy path.
            (
                r#"{"coefficients":[2.0,0.5],"min":100.0,"max":900.0,"degree":1,
                    "at_surfing":0.0,"gas mass":39.948}"#,
                25.0, 300.25, 2, 107.50922812278505,
            ),
        ];
        for &(json, at, mz, z, expect) in cases {
            let c = CcsCalibration::from_json(json).unwrap();
            let got = c.arrival_time_to_ccs(at, mz, z);
            let rel = (got - expect).abs() / expect;
            assert!(rel < 1e-12, "json={json}\n got {got}, want {expect}");
        }
    }

    #[test]
    fn legacy_version_detection() {
        assert!(is_legacy_version(None));
        assert!(is_legacy_version(Some("")));
        assert!(is_legacy_version(Some("0.0.1")));
        assert!(!is_legacy_version(Some("0.9.9")));
        assert!(!is_legacy_version(Some("1.0.0")));
        assert!(!is_legacy_version(Some("2.5.0")));
    }

    #[test]
    fn at_surfing_does_not_shift_the_polynomial() {
        // Confirmed against the SDK: changing at_surfing leaves CCS unchanged.
        let mut a = cal(&[0.0, 1.0], GAS_MASS_N2);
        let mut b = a.clone();
        a.at_surfing = 0.0;
        b.at_surfing = 2.0;
        assert_eq!(
            a.arrival_time_to_ccs(10.0, 622.0, 1),
            b.arrival_time_to_ccs(10.0, 622.0, 1)
        );
    }

    #[test]
    fn drift_axis_matches_trigger_timestamps() {
        // frame 600 of 200S-100ngHeLa: frm-dt-period, and the arrival times its
        // per-scan trigger timestamps imply.
        let axis = DriftAxis { period_ms: 0.11958356190006342, n_scans: 3345 };
        assert_eq!(axis.arrival_time_ms(0), 0.0);
        assert!((axis.arrival_time_ms(1000) - 119.583560).abs() < 1e-5);
        assert!((axis.arrival_time_ms(3344) - 399.887432).abs() < 1e-5);
        assert_eq!(axis.scan_at(119.583562), 1000);
        assert_eq!(axis.scan_at(-1.0), 0);
        assert_eq!(axis.scan_at(1.0e9), 3344);
    }
}
