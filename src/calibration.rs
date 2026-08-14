//! Time-of-flight mass calibration.
//!
//! The stored calibration is a "traditional" TOF fit (`sqrt(m/z)` linear in drift
//! time) plus a polynomial residual correction expressed in **ppm**. Reproducing
//! the vendor SDK exactly requires all three pieces; the base fit alone is off by
//! up to ~4 ppm, which is not good enough for proteomics.

use serde::Deserialize;

/// A TOF calibration for one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct TofCalibration {
    /// Slope of the traditional fit: `sqrt(m/z) = slope * (t_us - intercept)`.
    pub slope: f64,
    /// Intercept, in microseconds.
    pub intercept: f64,
    /// Polynomial coefficients of the mass residual, ascending powers of `t_us`,
    /// evaluating to a correction in ppm. Empty when the file is uncorrected.
    pub residual_terms: Vec<f64>,
    /// Digitiser sampling rate in samples/second (2e9 on current hardware).
    pub sample_rate: f64,
}

#[derive(Deserialize)]
struct TraditionalCal {
    slope: f64,
    intercept: f64,
    #[serde(default)]
    mz_residual_terms: Vec<f64>,
}

impl TofCalibration {
    /// Parse the `cal-ms-traditional` frame attribute.
    pub fn from_json(json: &str, sample_rate: f64) -> Result<Self, serde_json::Error> {
        let c: TraditionalCal = serde_json::from_str(json)?;
        Ok(Self {
            slope: c.slope,
            intercept: c.intercept,
            residual_terms: c.mz_residual_terms,
            sample_rate,
        })
    }

    /// TOF bin index -> drift time in microseconds.
    #[inline]
    pub fn index_to_micros(&self, index: u64) -> f64 {
        index as f64 * (1.0e6 / self.sample_rate)
    }

    /// The mass error implied by the residual fit, in ppm, at a given drift time.
    ///
    /// Equivalent to the SDK's `TofCalibration::TofError`.
    #[inline]
    pub fn tof_error_ppm(&self, t_us: f64) -> f64 {
        // Horner, ascending powers.
        self.residual_terms
            .iter()
            .rev()
            .fold(0.0, |acc, &c| acc * t_us + c)
    }

    /// Drift time (microseconds) -> m/z.
    #[inline]
    pub fn micros_to_mz(&self, t_us: f64) -> f64 {
        let root = self.slope * (t_us - self.intercept);
        let raw = root * root;
        if self.residual_terms.is_empty() {
            raw
        } else {
            raw * (1.0 - self.tof_error_ppm(t_us) / 1.0e6)
        }
    }

    /// TOF bin index -> m/z.
    #[inline]
    pub fn index_to_mz(&self, index: u64) -> f64 {
        self.micros_to_mz(self.index_to_micros(index))
    }

    /// Fill `out` with the m/z for each index. Cheaper than calling
    /// [`Self::index_to_mz`] in a loop from a binding layer.
    pub fn index_to_mz_buffer(&self, indices: &[u64], out: &mut Vec<f64>) {
        out.clear();
        out.reserve(indices.len());
        out.extend(indices.iter().map(|&i| self.index_to_mz(i)));
    }

    /// m/z -> drift time in microseconds.
    ///
    /// The residual correction makes this non-analytic, so invert the base fit and
    /// refine. Two Newton steps converge to well below one sample period.
    pub fn mz_to_micros(&self, mz: f64) -> f64 {
        let mut t = mz.sqrt() / self.slope + self.intercept;
        if self.residual_terms.is_empty() {
            return t;
        }
        for _ in 0..4 {
            let f = self.micros_to_mz(t) - mz;
            // df/dt, central difference at a scale far below the sample period.
            let h = 1.0e-6;
            let d = (self.micros_to_mz(t + h) - self.micros_to_mz(t - h)) / (2.0 * h);
            if d == 0.0 {
                break;
            }
            let step = f / d;
            t -= step;
            if step.abs() < 1.0e-12 {
                break;
            }
        }
        t
    }

    /// m/z -> TOF bin index: the largest bin whose m/z does not exceed `mz`.
    ///
    /// This truncates rather than rounds, matching the vendor SDK. The small
    /// epsilon absorbs the inversion's numerical noise (a millionth of a bin, far
    /// below anything physically meaningful) so that
    /// `mz_to_index(index_to_mz(i)) == i` holds exactly, as it does in the SDK.
    pub fn mz_to_index(&self, mz: f64) -> u64 {
        let t = self.mz_to_micros(mz);
        let idx = t * self.sample_rate / 1.0e6;
        if idx <= 0.0 {
            0
        } else {
            (idx + 1.0e-6).floor() as u64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values lifted from the vendor SDK running on
    /// `200S-100ngHeLa-14.19.00.mbi` (MassIVE MSV000099577).
    fn reference() -> TofCalibration {
        TofCalibration {
            slope: 0.3478131086072915,
            intercept: 0.10579624799291579,
            residual_terms: vec![
                1.8229826931044388,
                -0.6138856370823927,
                0.020957953832513733,
                -0.00024019397491639355,
                8.989476042960146e-07,
            ],
            sample_rate: 2.0e9,
        }
    }

    #[test]
    fn matches_sdk_index_to_mz() {
        let cal = reference();
        // (index, SDK IndexToMz)
        let cases = [
            (40000u64, 47.87917811),
            (76479, 175.91827611),
            (100000, 301.15665694),
            (143621, 621.99493436),
            (146624, 648.31687165),
            (180000, 977.58646517),
            (232991, 1638.78045647),
        ];
        for (idx, expect) in cases {
            let got = cal.index_to_mz(idx);
            let ppm = (got - expect).abs() / expect * 1.0e6;
            assert!(ppm < 1.0e-3, "index {idx}: got {got}, want {expect} ({ppm} ppm)");
        }
    }

    #[test]
    fn tof_error_matches_sdk() {
        let cal = reference();
        // SDK TofCalibration::TofError at t = index * 0.0005 us
        assert!((cal.tof_error_ppm(20.0) - -3.849269).abs() < 1e-5);
        assert!((cal.tof_error_ppm(38.2395) - -2.514303).abs() < 1e-5);
        assert!((cal.tof_error_ppm(116.4955) - 0.555807).abs() < 1e-5);
    }

    #[test]
    fn mz_to_index_truncates_like_sdk() {
        let cal = reference();
        // The SDK returns the largest bin with m/z <= input, so 622.0 -- which
        // falls between bin 143621 (621.99493) and 143622 (622.00361) -- is 143621.
        assert_eq!(cal.mz_to_index(622.0), 143621);
        assert_eq!(cal.mz_to_index(622.0036), 143621);
        assert_eq!(cal.mz_to_index(622.004), 143622);
    }

    #[test]
    fn mz_to_index_round_trips() {
        let cal = reference();
        for idx in [1000u64, 40000, 76479, 100000, 143621, 180000, 232991] {
            assert_eq!(cal.mz_to_index(cal.index_to_mz(idx)), idx, "index {idx}");
        }
    }

    #[test]
    fn uncorrected_calibration_skips_residual() {
        let mut cal = reference();
        cal.residual_terms.clear();
        let t = 38.2395;
        let root = cal.slope * (t - cal.intercept);
        assert_eq!(cal.micros_to_mz(t), root * root);
    }
}
