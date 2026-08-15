//! Python (PyO3) bindings for `mobilionmbi`. Exposes the pure-Rust reader as
//! `mobilionmbi_connector.MbiFile`, with numpy-backed frame data and dict-shaped
//! metadata.
//!
//! Arrays are moved out of Rust into numpy rather than copied: `into_pyarray`
//! hands the `Vec`'s allocation to numpy, so a frame costs one decode and no
//! marshalling pass.

use std::collections::BTreeMap;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mobilionmbi::{Error as MbiError, Frame, MbiFile, TofCalibration};

fn to_pyerr(e: MbiError) -> PyErr {
    match e {
        MbiError::FrameOutOfRange { .. } => PyIndexError::new_err(e.to_string()),
        MbiError::Hdf5(_) => PyIOError::new_err(e.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// A TOF mass calibration.
#[pyclass(name = "Calibration")]
#[derive(Clone)]
pub struct PyCalibration {
    inner: TofCalibration,
}

#[pymethods]
impl PyCalibration {
    #[getter]
    fn slope(&self) -> f64 {
        self.inner.slope
    }
    #[getter]
    fn intercept(&self) -> f64 {
        self.inner.intercept
    }
    #[getter]
    fn residual_terms(&self) -> Vec<f64> {
        self.inner.residual_terms.clone()
    }
    #[getter]
    fn sample_rate(&self) -> f64 {
        self.inner.sample_rate
    }

    /// TOF bin indices -> m/z, vectorised over a numpy array.
    fn index_to_mz<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'py, u64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let idx = indices.as_slice()?;
        let mut out = Vec::new();
        py.detach(|| self.inner.index_to_mz_buffer(idx, &mut out));
        Ok(out.into_pyarray(py))
    }

    /// m/z -> TOF bin index (truncating, as the vendor SDK does).
    fn mz_to_index<'py>(
        &self,
        py: Python<'py>,
        mz: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let mz = mz.as_slice()?;
        let out: Vec<u64> = py.detach(|| mz.iter().map(|&m| self.inner.mz_to_index(m)).collect());
        Ok(out.into_pyarray(py))
    }

    /// Drift time in microseconds -> m/z.
    fn micros_to_mz(&self, t_us: f64) -> f64 {
        self.inner.micros_to_mz(t_us)
    }

    /// The residual fit's mass error, in ppm, at a given drift time.
    fn tof_error_ppm(&self, t_us: f64) -> f64 {
        self.inner.tof_error_ppm(t_us)
    }

    fn __repr__(&self) -> String {
        format!(
            "Calibration(slope={}, intercept={}, residual_terms={})",
            self.inner.slope,
            self.inner.intercept,
            self.inner.residual_terms.len()
        )
    }
}

/// One frame: a sparse `drift scan x TOF index` plane.
#[pyclass(name = "Frame")]
pub struct PyFrame {
    inner: Option<Frame>,
    index: usize,
    n_rows: usize,
    n_cols: usize,
    nnz: usize,
}

impl PyFrame {
    fn frame(&self) -> PyResult<&Frame> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("frame data already consumed"))
    }
}

#[pymethods]
impl PyFrame {
    #[getter]
    fn index(&self) -> usize {
        self.index
    }
    #[getter]
    fn n_rows(&self) -> usize {
        self.n_rows
    }
    #[getter]
    fn n_cols(&self) -> usize {
        self.n_cols
    }
    #[getter]
    fn nnz(&self) -> usize {
        self.nnz
    }
    #[getter]
    fn total_intensity(&self) -> PyResult<i64> {
        Ok(self.frame()?.total_intensity())
    }

    fn __len__(&self) -> usize {
        self.nnz
    }
    fn __repr__(&self) -> String {
        format!(
            "Frame({}, {} points, {}x{})",
            self.index, self.nnz, self.n_rows, self.n_cols
        )
    }

    /// `(data, indices, indptr, n_rows, n_cols)` — scipy `csr_matrix` layout.
    ///
    /// This consumes the frame's buffers, moving them into numpy without a copy.
    fn csr<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<u64>>,
        usize,
        usize,
    )> {
        let f = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("frame data already consumed"))?;
        Ok((
            f.data.into_pyarray(py),
            f.indices.into_pyarray(py),
            f.indptr.into_pyarray(py),
            f.n_rows,
            f.n_cols,
        ))
    }

    /// `(rows, cols, values, n_rows, n_cols)` — drift bin, TOF index, intensity.
    fn coo<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<i32>>,
        usize,
        usize,
    )> {
        let f = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("frame data already consumed"))?;
        let rows = py.detach(|| {
            let mut rows = Vec::with_capacity(f.data.len());
            for row in 0..f.n_rows {
                let (a, b) = (f.indptr[row] as usize, f.indptr[row + 1] as usize);
                rows.extend(std::iter::repeat(row as u64).take(b - a));
            }
            rows
        });
        Ok((
            rows.into_pyarray(py),
            f.indices.into_pyarray(py),
            f.data.into_pyarray(py),
            f.n_rows,
            f.n_cols,
        ))
    }

    /// Indices of drift scans holding at least one non-zero point.
    fn nonzero_scan_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u64>>> {
        Ok(self.frame()?.nonzero_scan_indices().into_pyarray(py))
    }
}

/// An open `.mbi` file.
#[pyclass(name = "MbiFile")]
pub struct PyMbiFile {
    inner: MbiFile,
    path: String,
}

fn map_to_dict<'py>(py: Python<'py>, map: &BTreeMap<String, String>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in map {
        d.set_item(k, v)?;
    }
    Ok(d)
}

#[pymethods]
impl PyMbiFile {
    #[new]
    fn new(py: Python<'_>, path: String) -> PyResult<Self> {
        let p = path.clone();
        let inner = py.detach(|| MbiFile::open(&p)).map_err(to_pyerr)?;
        Ok(Self { inner, path })
    }

    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }
    #[getter]
    fn sample_rate(&self) -> f64 {
        self.inner.sample_rate()
    }
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    fn __len__(&self) -> usize {
        self.inner.n_frames()
    }
    fn __repr__(&self) -> String {
        format!("MbiFile('{}', frames={})", self.path, self.inner.n_frames())
    }

    /// File-level metadata, as stored (values are raw strings, some JSON).
    fn global_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        map_to_dict(py, self.inner.global_metadata())
    }

    /// Per-frame metadata, as stored.
    fn frame_metadata<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyDict>> {
        let md = self.inner.frame_metadata(index).map_err(to_pyerr)?;
        map_to_dict(py, &md)
    }

    /// The mass calibration for a frame.
    fn calibration(&self, index: usize) -> PyResult<PyCalibration> {
        Ok(PyCalibration {
            inner: self.inner.calibration(index).map_err(to_pyerr)?,
        })
    }

    /// Read one frame. Frame indices are **1-based**, as in the vendor API.
    fn frame(&self, py: Python<'_>, index: usize) -> PyResult<PyFrame> {
        let f = py.detach(|| self.inner.frame(index)).map_err(to_pyerr)?;
        Ok(PyFrame {
            index: f.index,
            n_rows: f.n_rows,
            n_cols: f.n_cols,
            nnz: f.nnz(),
            inner: Some(f),
        })
    }

    /// Retention time of every frame.
    fn retention_times<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rt = py.detach(|| self.inner.retention_times()).map_err(to_pyerr)?;
        Ok(rt.into_pyarray(py))
    }
}

#[pymodule]
fn mobilionmbi_connector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMbiFile>()?;
    m.add_class::<PyFrame>()?;
    m.add_class::<PyCalibration>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
