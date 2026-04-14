//! YomiToku searchable-PDF generation via subprocess.
//!
//! Mirrors the approach taken by the original C# implementation
//! (`SuperBookTools/Basic/SuperPdfUtil.cs` → `PdfYomitokuLib.PerformOcrAsync`):
//! shell out to the `yomitoku` CLI with `-f pdf` and let YomiToku itself emit
//! a searchable PDF with the text layer correctly positioned. This avoids
//! re-implementing the coordinate math and font handling in Rust.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum YomiTokuPdfError {
    #[error("Input PDF not found: {0}")]
    InputNotFound(PathBuf),

    #[error("yomitoku CLI not found at {0}")]
    CliNotFound(PathBuf),

    #[error("yomitoku failed (exit {code}): {stderr}")]
    ExecutionFailed { code: i32, stderr: String },

    #[error("yomitoku produced no PDF in {0}")]
    NoOutput(PathBuf),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, YomiTokuPdfError>;

/// Which compute device to ask YomiToku to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cuda,
    Mps,
    Cpu,
}

impl Device {
    fn as_arg(self) -> &'static str {
        match self {
            Device::Cuda => "cuda",
            Device::Mps => "mps",
            Device::Cpu => "cpu",
        }
    }

    /// Pick a reasonable default for the current platform.
    pub fn autodetect(gpu_requested: bool) -> Self {
        if !gpu_requested {
            return Device::Cpu;
        }
        if cfg!(target_os = "macos") {
            Device::Mps
        } else {
            Device::Cuda
        }
    }
}

/// Options for a single YomiToku PDF conversion.
#[derive(Debug, Clone)]
pub struct Options {
    pub device: Device,
    pub dpi: u32,
    /// Seconds before the subprocess is killed.
    pub timeout: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            device: Device::autodetect(true),
            dpi: 300,
            timeout: Duration::from_secs(5 * 3600),
        }
    }
}

/// Invoke `yomitoku -f pdf` on `input_pdf` and return the path of the
/// produced searchable PDF inside `output_dir`.
///
/// `venv_path` should point at the Python virtualenv holding the
/// installed YomiToku package (the CLI lives at `<venv>/bin/yomitoku`).
pub fn ocr_pdf(
    venv_path: &Path,
    input_pdf: &Path,
    output_dir: &Path,
    options: &Options,
) -> Result<PathBuf> {
    if !input_pdf.is_file() {
        return Err(YomiTokuPdfError::InputNotFound(input_pdf.to_path_buf()));
    }

    std::fs::create_dir_all(output_dir)?;

    let cli = venv_path.join("bin").join("yomitoku");
    if !cli.is_file() {
        return Err(YomiTokuPdfError::CliNotFound(cli));
    }

    // Flags mirror the original C# invocation:
    //   yomitoku <input> -f pdf -o <out> -d <device>
    //            --dpi <dpi> --ignore_line_break --combine
    //            --figure --figure_letter --ignore_meta
    let output = Command::new(&cli)
        .arg(input_pdf)
        .arg("-f")
        .arg("pdf")
        .arg("-o")
        .arg(output_dir)
        .arg("-d")
        .arg(options.device.as_arg())
        .arg("--dpi")
        .arg(options.dpi.to_string())
        .arg("--ignore_line_break")
        .arg("--combine")
        .arg("--figure")
        .arg("--figure_letter")
        .arg("--ignore_meta")
        .output()?;

    if !output.status.success() {
        return Err(YomiTokuPdfError::ExecutionFailed {
            code: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        });
    }

    find_generated_pdf(output_dir)
}

fn find_generated_pdf(output_dir: &Path) -> Result<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(output_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s.eq_ignore_ascii_case("pdf"))
                    .unwrap_or(false)
        })
        .collect();

    // Pick the newest PDF so repeat runs in the same directory don't pick
    // up stale output.
    candidates.sort_by_key(|p| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .ok()
    });

    candidates
        .pop()
        .ok_or_else(|| YomiTokuPdfError::NoOutput(output_dir.to_path_buf()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_autodetect_respects_gpu_flag() {
        assert_eq!(Device::autodetect(false), Device::Cpu);
    }

    #[test]
    fn device_arg_strings() {
        assert_eq!(Device::Cuda.as_arg(), "cuda");
        assert_eq!(Device::Mps.as_arg(), "mps");
        assert_eq!(Device::Cpu.as_arg(), "cpu");
    }

    #[test]
    fn input_not_found_errors() {
        let tmp = std::env::temp_dir().join("superbook_yomitoku_pdf_test_in");
        let out = std::env::temp_dir().join("superbook_yomitoku_pdf_test_out");
        let venv = std::env::temp_dir().join("superbook_yomitoku_pdf_test_venv");
        let err = ocr_pdf(&venv, &tmp.join("missing.pdf"), &out, &Options::default())
            .unwrap_err();
        assert!(matches!(err, YomiTokuPdfError::InputNotFound(_)));
    }
}
