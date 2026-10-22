//! Model registry: where the ONNX assets live, how to obtain them, and how to open sessions.
//!
//! Nothing is downloaded at build time and nothing is downloaded implicitly - the user has to
//! run `fetch-models` (or set `VISION_POINTER_MODELS` to a directory that already has them).

use anyhow::{Context, Result, anyhow, bail};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::ValueType;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// One downloadable model asset.
pub struct ModelSpec {
    pub label: &'static str,
    pub file_name: &'static str,
    pub urls: &'static [&'static str],
}

/// Stage 1: locates palms in the frame and gives the 7 points used to orient the hand crop.
pub const PALM: ModelSpec = ModelSpec {
    label: "palm detection",
    file_name: "palm_detection_mediapipe_2023feb.onnx",
    urls: &[
        "https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb.onnx",
        "https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb_int8.onnx",
    ],
};

/// Stage 2: consumes the rotated, cropped hand and returns the 21 landmarks.
pub const LANDMARK: ModelSpec = ModelSpec {
    label: "hand landmark",
    file_name: "handpose_estimation_mediapipe_2023feb.onnx",
    urls: &[
        "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx",
        "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb_int8.onnx",
    ],
};

/// Directory holding the `.onnx` assets: `$VISION_POINTER_MODELS` if set, else `./models`.
pub fn models_dir() -> PathBuf {
    match std::env::var_os("VISION_POINTER_MODELS") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("models"),
    }
}

pub fn path_for(spec: &ModelSpec) -> PathBuf {
    models_dir().join(spec.file_name)
}

/// Specs that are not present on disk yet.
pub fn missing_models() -> Vec<&'static ModelSpec> {
    [(&PALM as &'static ModelSpec), (&LANDMARK as &'static ModelSpec)]
        .into_iter()
        .filter(|spec| !path_for(spec).is_file())
        .collect()
}

pub fn require_models() -> Result<()> {
    let missing = missing_models();
    if missing.is_empty() {
        return Ok(());
    }
    let names = missing
        .iter()
        .map(|spec| format!("  - {} ({})", spec.label, path_for(spec).display()))
        .collect::<Vec<_>>()
        .join("\n");
    bail!(
        "Missing model file(s):\n{names}\n\nDownload them with:\n  cargo run --release -- fetch-models\n\
         or place the files in {} (override with VISION_POINTER_MODELS).",
        models_dir().display()
    )
}

/// Downloads any missing model into the models directory. Already-present files are kept.
pub fn fetch_models(force: bool) -> Result<usize> {
    let dir = models_dir();
    fs::create_dir_all(&dir).with_context(|| format!("Failed to create {}", dir.display()))?;

    let mut fetched = 0;
    for spec in [&PALM, &LANDMARK] {
        let dest = dir.join(spec.file_name);
        if dest.is_file() && !force {
            println!(
                "{}: already present ({} MB) - use --force to re-download",
                spec.label,
                size_mb(&dest)
            );
            continue;
        }
        download(spec, &dest)?;
        fetched += 1;
    }
    Ok(fetched)
}

fn size_mb(path: &Path) -> f64 {
    fs::metadata(path).map_or(0.0, |m| m.len() as f64 / (1024.0 * 1024.0))
}

fn download(spec: &ModelSpec, dest: &Path) -> Result<()> {
    let mut last_error = None;
    for url in spec.urls {
        println!("{}: fetching {}", spec.label, url);
        match transfer(url, dest)
            .with_context(|| format!("Failed to download {} for {}", url, spec.label))
        {
            Ok(bytes) => {
                println!(
                    "{}: saved {} ({:.1} MB)",
                    spec.label,
                    dest.display(),
                    bytes as f64 / (1024.0 * 1024.0)
                );
                return Ok(());
            }
            Err(why) => {
                eprintln!("{}: {}", spec.label, why);
                last_error = Some(why);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("No download URLs configured for {}", spec.label)))
}

/// Writes to a temporary file first so an interrupted download never leaves a truncated model.
fn transfer(url: &str, dest: &Path) -> Result<u64> {
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("HTTP request failed ({url})"))?;
    if response.status() != 200 {
        bail!("Unexpected HTTP status {} from {url}", response.status());
    }

    let temp_path = dest.with_extension("onnx.part");
    let mut temp = fs::File::create(&temp_path)
        .with_context(|| format!("Failed to create {}", temp_path.display()))?;
    let mut reader = response.into_reader();
    let written = std::io::copy(&mut reader, &mut temp)
        .with_context(|| format!("Failed to write {}", temp_path.display()))?;
    drop(temp);

    if written < 4096 {
        let _ = fs::remove_file(&temp_path);
        bail!("Downloaded payload is too small to be a model ({written} bytes)");
    }
    fs::rename(&temp_path, dest)
        .with_context(|| format!("Failed to move model into {}", dest.display()))?;
    Ok(written)
}

/// Opens an ORT environment. Exactly one may exist per process, so callers create this once.
pub fn init_environment() -> Result<Environment> {
    ort::init()
        .with_name("gesture-daemon")
        .build()
        .context("Failed to initialise ONNX Runtime")
}

pub fn open_session(environment: &Environment, spec: &ModelSpec) -> Result<Session> {
    let path = path_for(spec);
    Session::builder(environment)
        .context("Failed to create a session builder")?
        .commit_from_file(&path)
        .with_context(|| format!("Failed to load {} from {}", spec.label, path.display()))
}

/// Prints the tensor contract of each model: names, element types and shapes.
/// Useful when a conversion exposes outputs under different names than expected.
pub fn describe_models(environment: &Environment) -> Result<()> {
    for spec in [&PALM, &LANDMARK] {
        let path = path_for(spec);
        if !path.is_file() {
            println!("{}: {} (missing)", spec.label, path.display());
            continue;
        }
        let session = open_session(environment, spec)?;
        println!("{}: {}", spec.label, path.display());
        print_io("  input", session.inputs())?;
        print_io("  output", session.outputs())?;
        let _ = session;
    }
    Ok(())
}

fn print_io(kind: &str, outlets: &[ort::value::Outlet]) -> Result<()> {
    for outlet in outlets {
        let (dtype, shape) = match outlet.dtype() {
            ValueType::Tensor { ty, shape, .. } => (format!("{ty:?}"), format!("{:?}", shape.dimensions())),
            other => (format!("{other:?}"), String::from("-")),
        };
        println!("  {kind}: {} : {dtype} {shape}", outlet.name());
    }
    Ok(())
}

/// Reads a whole file into memory; used by the self-test path only.
pub fn read_file(path: &Path) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    fs::File::open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?
        .read_to_end(&mut buffer)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    Ok(buffer)
}
