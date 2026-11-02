use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use tauri::Emitter;
use tracing::{error, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub item_id: String,
    pub percentage: f32,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub phase: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperModelOption {
    pub id: String,
    pub name: String,
    pub tag: String,
    pub description: String,
    pub is_downloaded: bool,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub vad_ready: bool,
    pub pyannote_ready: bool,
    pub speaker_id_ready: bool,
    pub whisper_ready: bool,
    pub all_ready: bool,
    pub models_dir: String,
    pub vad_model_path: String,
    pub pyannote_model_path: String,
    pub speaker_id_model_path: String,
    pub whisper_encoder_path: String,
    pub whisper_decoder_path: String,
    pub whisper_tokens_path: String,
    pub active_whisper_model: String,
    pub available_whisper_models: Vec<WhisperModelOption>,
}

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub models_dir: PathBuf,
    pub vad_path: PathBuf,
    pub pyannote_path: PathBuf,
    pub speaker_id_path: PathBuf,
    pub whisper_encoder: PathBuf,
    pub whisper_decoder: PathBuf,
    pub whisper_tokens: PathBuf,
    pub active_whisper_variant: String,
}

pub fn dir_has_models(dir: &Path) -> bool {
    if !dir.exists() || !dir.is_dir() {
        return false;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().map(|e| e == "onnx").unwrap_or(false) {
                return true;
            }
            if p.is_dir() {
                if let Ok(sub) = fs::read_dir(&p) {
                    for s in sub.flatten() {
                        let sp = s.path();
                        if sp.is_file() && sp.extension().map(|e| e == "onnx").unwrap_or(false) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

pub fn resolve_models_dir(app_data_dir: Option<PathBuf>) -> PathBuf {
    // 1. Check ./models next to executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let local_models = exe_dir.join("models");
            if dir_has_models(&local_models) {
                return local_models;
            }
            // Also check parent directories of exe (e.g. workspace when running target/release/live-transcriber.exe)
            let mut curr = exe_dir.parent();
            for _ in 0..4 {
                if let Some(p) = curr {
                    let cand = p.join("models");
                    if dir_has_models(&cand) {
                        return cand;
                    }
                    curr = p.parent();
                } else {
                    break;
                }
            }
        }
    }

    // 2. Check workspace models dir
    let cwd_models = PathBuf::from("models");
    if dir_has_models(&cwd_models) {
        if let Ok(abs) = cwd_models.canonicalize() {
            return abs;
        }
        return cwd_models;
    }

    // 3. Check app_data_dir/models
    if let Some(ref data_dir) = app_data_dir {
        let app_models = data_dir.join("models");
        if dir_has_models(&app_models) {
            return app_models;
        }
    }

    // 4. Fallback to existing directory
    if cwd_models.exists() {
        if let Ok(abs) = cwd_models.canonicalize() {
            return abs;
        }
        return cwd_models;
    }

    // 5. Fallback to app_data_dir/models
    if let Some(data_dir) = app_data_dir {
        let app_models = data_dir.join("models");
        let _ = fs::create_dir_all(&app_models);
        return app_models;
    }

    let default_dir = PathBuf::from("./models");
    let _ = fs::create_dir_all(&default_dir);
    default_dir
}

pub fn is_valid_file(p: &Path) -> bool {
    p.is_file() && fs::metadata(p).map(|m| m.len() > 1000).unwrap_or(false)
}

pub fn find_whisper_files_in_dir(dir: &Path) -> Option<(PathBuf, PathBuf, PathBuf)> {
    if !dir.exists() || !dir.is_dir() {
        return None;
    }
    let mut encoder = None;
    let mut decoder = None;
    let mut tokens = None;

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !is_valid_file(&path) {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
            if name.ends_with(".onnx") {
                if name.contains("encoder") {
                    if name.contains("int8") || encoder.is_none() {
                        encoder = Some(path.clone());
                    }
                } else if name.contains("decoder") {
                    if name.contains("int8") || decoder.is_none() {
                        decoder = Some(path.clone());
                    }
                }
            } else if (name.contains("tokens") || name == "tokens.txt") && name.ends_with(".txt") {
                tokens = Some(path.clone());
            }
        }
    }

    if let (Some(enc), Some(dec), Some(tok)) = (encoder, decoder, tokens) {
        Some((enc, dec, tok))
    } else {
        None
    }
}

pub fn find_whisper_files_for_variant(models_dir: &Path, variant: &str) -> Option<(PathBuf, PathBuf, PathBuf)> {
    // 1. Try canonical folder names
    let canonical_folder = match variant {
        "turbo" => "sherpa-onnx-whisper-turbo",
        "small.en" => "sherpa-onnx-whisper-small.en",
        "base.en" => "sherpa-onnx-whisper-base.en",
        "large-v3" => "sherpa-onnx-whisper-large-v3",
        "tiny.en" => "sherpa-onnx-whisper-tiny.en",
        _ => variant,
    };
    let canonical_path = models_dir.join(canonical_folder);
    if let Some(files) = find_whisper_files_in_dir(&canonical_path) {
        return Some(files);
    }

    // 2. Search any subdirectory in models_dir whose name contains variant keyword
    let key = match variant {
        "turbo" => "turbo",
        "small.en" => "small",
        "base.en" => "base",
        "large-v3" => "large",
        "tiny.en" => "tiny",
        other => other,
    };

    if let Ok(entries) = fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let dirname = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                if dirname.contains(key) {
                    if let Some(files) = find_whisper_files_in_dir(&path) {
                        return Some(files);
                    }
                }
            }
        }
    }

    // 3. Search direct files in models_dir (in case files were extracted directly into models_dir)
    let mut encoder = None;
    let mut decoder = None;
    let mut tokens = None;

    if let Ok(entries) = fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !is_valid_file(&path) {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
            if name.contains(key) {
                if name.ends_with(".onnx") {
                    if name.contains("encoder") {
                        if name.contains("int8") || encoder.is_none() {
                            encoder = Some(path.clone());
                        }
                    } else if name.contains("decoder") {
                        if name.contains("int8") || decoder.is_none() {
                            decoder = Some(path.clone());
                        }
                    }
                } else if name.contains("tokens") && name.ends_with(".txt") {
                    tokens = Some(path.clone());
                }
            } else if tokens.is_none() && (name == "tokens.txt" || name.ends_with("-tokens.txt")) {
                tokens = Some(path.clone());
            }
        }
    }

    if let (Some(enc), Some(dec), Some(tok)) = (encoder, decoder, tokens) {
        Some((enc, dec, tok))
    } else {
        None
    }
}

pub fn get_available_whisper_models(models_dir: &Path, active_variant: &str) -> Vec<WhisperModelOption> {
    let variants = [
        (
            "turbo",
            "Whisper Large-v3-Turbo",
            "Fast & Accurate - Recommended",
            "High accuracy with fast 4-layer decoder (~809M params, 8x faster than Large-v3)",
            "sherpa-onnx-whisper-turbo",
        ),
        (
            "small.en",
            "Whisper Small.en",
            "Ultra-Fast / Low CPU",
            "Ultra-fast real-time inference with solid conversational accuracy (~244M params)",
            "sherpa-onnx-whisper-small.en",
        ),
        (
            "base.en",
            "Whisper Base.en",
            "Standard Conversational",
            "Standard conversational English model (~74M params)",
            "sherpa-onnx-whisper-base.en",
        ),
        (
            "large-v3",
            "Whisper Large-v3",
            "Max Accuracy (32-Layer Heavy)",
            "Maximum transcription accuracy across complex accents (32 decoder layers, heavy CPU load on long conversations)",
            "sherpa-onnx-whisper-large-v3",
        ),
        (
            "tiny.en",
            "Whisper Tiny.en",
            "Ultra-Fast Fallback",
            "Ultra-low latency and low resource usage (~39M params)",
            "sherpa-onnx-whisper-tiny.en",
        ),
    ];

    variants
        .iter()
        .map(|(id, name, tag, desc, _folder)| {
            let is_downloaded = find_whisper_files_for_variant(models_dir, id).is_some();
            let is_active = *id == active_variant;
            WhisperModelOption {
                id: id.to_string(),
                name: name.to_string(),
                tag: tag.to_string(),
                description: desc.to_string(),
                is_downloaded,
                is_active,
            }
        })
        .collect()
}

pub fn get_model_paths_with_variant(models_dir: &Path, active_variant: Option<&str>) -> ModelPaths {
    let vad_path = models_dir.join("silero_vad.onnx");

    // Check CAM++ or 3D-Speaker
    let cam_path = models_dir.join("wespeaker_en_voxceleb_cam++.onnx");
    let speaker_id_path = if cam_path.exists() {
        cam_path
    } else {
        let alt = models_dir.join("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx");
        if alt.exists() {
            alt
        } else {
            cam_path
        }
    };

    // Determine whisper model files
    let variants_order = match active_variant {
        Some("turbo") => vec!["turbo", "small.en", "base.en", "tiny.en", "large-v3"],
        Some("small.en") => vec!["small.en", "turbo", "base.en", "tiny.en", "large-v3"],
        Some("base.en") => vec!["base.en", "turbo", "small.en", "tiny.en", "large-v3"],
        Some("large-v3") => vec!["large-v3", "turbo", "small.en", "base.en", "tiny.en"],
        Some("tiny.en") => vec!["tiny.en", "turbo", "small.en", "base.en", "large-v3"],
        _ => vec!["turbo", "small.en", "base.en", "tiny.en", "large-v3"],
    };

    let mut found_whisper = None;
    let mut resolved_variant = active_variant.unwrap_or("turbo").to_string();

    // Check if the requested active variant is downloaded first
    if let Some(req) = active_variant {
        if let Some(files) = find_whisper_files_for_variant(models_dir, req) {
            found_whisper = Some(files);
            resolved_variant = req.to_string();
        }
    }

    // If not found, check fallbacks in variants_order
    if found_whisper.is_none() {
        for v in &variants_order {
            if let Some(files) = find_whisper_files_for_variant(models_dir, v) {
                found_whisper = Some(files);
                resolved_variant = (*v).to_string();
                break;
            }
        }
    }

    let (whisper_encoder, whisper_decoder, whisper_tokens) = if let Some(files) = found_whisper {
        files
    } else {
        // Direct files fallback
        (
            models_dir.join("tiny.en-encoder.int8.onnx"),
            models_dir.join("tiny.en-decoder.int8.onnx"),
            models_dir.join("tiny.en-tokens.txt"),
        )
    };

    // Check Pyannote Segmentation 3.0
    let pyannote_direct = models_dir.join("pyannote_segmentation_3_0.onnx");
    let pyannote_nested = models_dir.join("sherpa-onnx-pyannote-segmentation-3-0").join("model.onnx");
    let pyannote_path = if pyannote_direct.exists() {
        pyannote_direct
    } else if pyannote_nested.exists() {
        pyannote_nested
    } else {
        pyannote_direct
    };

    ModelPaths {
        models_dir: models_dir.to_path_buf(),
        vad_path,
        pyannote_path,
        speaker_id_path,
        whisper_encoder,
        whisper_decoder,
        whisper_tokens,
        active_whisper_variant: resolved_variant,
    }
}

pub fn get_model_paths(models_dir: &Path) -> ModelPaths {
    get_model_paths_with_variant(models_dir, None)
}

pub fn check_models_status(models_dir: &Path, active_variant: Option<&str>) -> ModelStatus {
    let paths = get_model_paths_with_variant(models_dir, active_variant);
    let vad_ready = paths.vad_path.exists() && fs::metadata(&paths.vad_path).map(|m| m.len() > 1000).unwrap_or(false);
    let pyannote_ready = paths.pyannote_path.exists() && fs::metadata(&paths.pyannote_path).map(|m| m.len() > 1000).unwrap_or(false);
    let speaker_id_ready = paths.speaker_id_path.exists() && fs::metadata(&paths.speaker_id_path).map(|m| m.len() > 1000).unwrap_or(false);
    let whisper_ready = paths.whisper_encoder.exists()
        && paths.whisper_decoder.exists()
        && paths.whisper_tokens.exists();

    let all_ready = vad_ready && pyannote_ready && speaker_id_ready && whisper_ready;
    let available_whisper_models = get_available_whisper_models(models_dir, &paths.active_whisper_variant);

    ModelStatus {
        vad_ready,
        pyannote_ready,
        speaker_id_ready,
        whisper_ready,
        all_ready,
        models_dir: models_dir.to_string_lossy().to_string(),
        vad_model_path: paths.vad_path.to_string_lossy().to_string(),
        pyannote_model_path: paths.pyannote_path.to_string_lossy().to_string(),
        speaker_id_model_path: paths.speaker_id_path.to_string_lossy().to_string(),
        whisper_encoder_path: paths.whisper_encoder.to_string_lossy().to_string(),
        whisper_decoder_path: paths.whisper_decoder.to_string_lossy().to_string(),
        whisper_tokens_path: paths.whisper_tokens.to_string_lossy().to_string(),
        active_whisper_model: paths.active_whisper_variant,
        available_whisper_models,
    }
}

pub fn download_file(url: &str, dest_path: &Path) -> Result<()> {
    download_file_with_progress(url, dest_path, "default-file", None)
}

pub fn download_file_with_progress(
    url: &str,
    dest_path: &Path,
    item_id: &str,
    app_handle: Option<&tauri::AppHandle>,
) -> Result<()> {
    info!("Downloading {} to {}", url, dest_path.display());
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let temp_path = dest_path.with_extension("downloading");
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("Failed to GET {}", url))?;

    let total_bytes: u64 = resp
        .header("content-length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut reader = resp.into_reader();
    let mut file = File::create(&temp_path)?;
    let mut buffer = [0u8; 65536];
    let mut downloaded: u64 = 0;
    let mut last_emit = std::time::Instant::now();

    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        downloaded += n as u64;

        if let Some(handle) = app_handle {
            if last_emit.elapsed() > std::time::Duration::from_millis(80) || (total_bytes > 0 && downloaded == total_bytes) {
                last_emit = std::time::Instant::now();
                let pct = if total_bytes > 0 {
                    (downloaded as f32 / total_bytes as f32 * 98.0).clamp(0.0, 98.0)
                } else {
                    ((downloaded as f32 / (downloaded + 5_000_000) as f32) * 98.0).clamp(0.0, 98.0)
                };

                let _ = handle.emit(
                    "download-progress",
                    DownloadProgress {
                        item_id: item_id.to_string(),
                        percentage: pct,
                        downloaded_bytes: downloaded,
                        total_bytes,
                        phase: "downloading".to_string(),
                        message: format!("{:.0}%", pct),
                    },
                );
            }
        }
    }

    file.flush()?;
    drop(file);

    fs::rename(&temp_path, dest_path)?;
    info!("Finished downloading to {}", dest_path.display());

    if let Some(handle) = app_handle {
        let _ = handle.emit(
            "download-progress",
            DownloadProgress {
                item_id: item_id.to_string(),
                percentage: 100.0,
                downloaded_bytes: downloaded,
                total_bytes,
                phase: "completed".to_string(),
                message: "Ready ✓".to_string(),
            },
        );
    }

    Ok(())
}

pub fn download_and_extract_tar_bz2(url: &str, target_dir: &Path) -> Result<()> {
    download_and_extract_tar_bz2_with_progress(url, target_dir, "archive", None)
}

pub fn download_and_extract_tar_bz2_with_progress(
    url: &str,
    target_dir: &Path,
    item_id: &str,
    app_handle: Option<&tauri::AppHandle>,
) -> Result<()> {
    info!("Downloading archive {}...", url);
    fs::create_dir_all(target_dir)?;

    let temp_archive = target_dir.join(format!("{}.tar.bz2.downloading", item_id.replace('/', "_")));

    // 1. Download file with progress up to 85%
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("Failed to download archive from {}", url))?;

    let total_bytes: u64 = resp
        .header("content-length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut reader = resp.into_reader();
    let mut file = File::create(&temp_archive)?;
    let mut buffer = [0u8; 65536];
    let mut downloaded: u64 = 0;
    let mut last_emit = std::time::Instant::now();

    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        downloaded += n as u64;

        if let Some(handle) = app_handle {
            if last_emit.elapsed() > std::time::Duration::from_millis(80) || (total_bytes > 0 && downloaded == total_bytes) {
                last_emit = std::time::Instant::now();
                let pct = if total_bytes > 0 {
                    (downloaded as f32 / total_bytes as f32 * 85.0).clamp(0.0, 85.0)
                } else {
                    ((downloaded as f32 / (downloaded + 20_000_000) as f32) * 85.0).clamp(0.0, 85.0)
                };

                let _ = handle.emit(
                    "download-progress",
                    DownloadProgress {
                        item_id: item_id.to_string(),
                        percentage: pct,
                        downloaded_bytes: downloaded,
                        total_bytes,
                        phase: "downloading".to_string(),
                        message: format!("{:.0}%", pct),
                    },
                );
            }
        }
    }

    file.flush()?;
    drop(file);

    // 2. Extraction phase (85% to 99%)
    if let Some(handle) = app_handle {
        let _ = handle.emit(
            "download-progress",
            DownloadProgress {
                item_id: item_id.to_string(),
                percentage: 88.0,
                downloaded_bytes: downloaded,
                total_bytes,
                phase: "extracting".to_string(),
                message: "Extracting...".to_string(),
            },
        );
    }

    info!("Extracting archive to {}", target_dir.display());
    let arc_file = File::open(&temp_archive)?;
    let decompressor = bzip2_rs::DecoderReader::new(BufReader::new(arc_file));
    let mut archive = tar::Archive::new(decompressor);
    archive.set_overwrite(true);
    archive.set_preserve_permissions(false);
    archive.unpack(target_dir)?;
    let _ = fs::remove_file(&temp_archive);

    info!("Extraction completed.");

    if let Some(handle) = app_handle {
        let _ = handle.emit(
            "download-progress",
            DownloadProgress {
                item_id: item_id.to_string(),
                percentage: 100.0,
                downloaded_bytes: downloaded,
                total_bytes,
                phase: "completed".to_string(),
                message: "Ready ✓".to_string(),
            },
        );
    }

    Ok(())
}

pub fn download_whisper_variant(models_dir: &Path, variant: &str) -> Result<()> {
    download_whisper_variant_with_progress(models_dir, variant, None)
}

pub fn download_whisper_variant_with_progress(
    models_dir: &Path,
    variant: &str,
    app_handle: Option<&tauri::AppHandle>,
) -> Result<()> {
    fs::create_dir_all(models_dir)?;
    if find_whisper_files_for_variant(models_dir, variant).is_some() {
        info!("Whisper variant '{}' already exists on disk, skipping download", variant);
        if let Some(handle) = app_handle {
            let _ = handle.emit(
                "download-progress",
                DownloadProgress {
                    item_id: format!("whisper-{}", variant),
                    percentage: 100.0,
                    downloaded_bytes: 0,
                    total_bytes: 0,
                    phase: "completed".to_string(),
                    message: "Ready ✓".to_string(),
                },
            );
        }
        return Ok(());
    }

    let archive_name = match variant {
        "turbo" => "sherpa-onnx-whisper-turbo.tar.bz2",
        "small.en" => "sherpa-onnx-whisper-small.en.tar.bz2",
        "base.en" => "sherpa-onnx-whisper-base.en.tar.bz2",
        "large-v3" => "sherpa-onnx-whisper-large-v3.tar.bz2",
        "tiny.en" => "sherpa-onnx-whisper-tiny.en.tar.bz2",
        _ => "sherpa-onnx-whisper-turbo.tar.bz2",
    };
    let url = format!(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{}",
        archive_name
    );
    let item_id = format!("whisper-{}", variant);
    info!("Downloading Whisper variant '{}' from {}", variant, url);
    download_and_extract_tar_bz2_with_progress(&url, models_dir, &item_id, app_handle)?;
    Ok(())
}

pub fn ensure_default_models(models_dir: &Path, active_variant: Option<&str>) -> Result<ModelStatus> {
    ensure_default_models_with_progress(models_dir, active_variant, None)
}

pub fn ensure_default_models_with_progress(
    models_dir: &Path,
    active_variant: Option<&str>,
    app_handle: Option<&tauri::AppHandle>,
) -> Result<ModelStatus> {
    fs::create_dir_all(models_dir)?;
    let paths = get_model_paths_with_variant(models_dir, active_variant);

    // 1. Silero VAD
    if !paths.vad_path.exists() {
        let vad_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";
        if let Err(e) = download_file_with_progress(vad_url, &paths.vad_path, "default-models", app_handle) {
            error!("Failed to download VAD model: {:#}", e);
        }
    }

    // 2. Pyannote Segmentation 3.0
    if !paths.pyannote_path.exists() {
        let pyannote_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";
        if let Err(e) = download_and_extract_tar_bz2_with_progress(pyannote_url, models_dir, "default-models", app_handle) {
            error!("Failed to download Pyannote model: {:#}", e);
        } else {
            let extracted = models_dir.join("sherpa-onnx-pyannote-segmentation-3-0").join("model.onnx");
            let target = models_dir.join("pyannote_segmentation_3_0.onnx");
            if extracted.exists() && !target.exists() {
                let _ = fs::copy(&extracted, &target);
            }
        }
    }

    // 3. Speaker Diarization / Embedding
    if !paths.speaker_id_path.exists() {
        let speaker_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-re-id-models/wespeaker_en_voxceleb_CAM++.onnx";
        if let Err(e) = download_file_with_progress(speaker_url, &paths.speaker_id_path, "default-models", app_handle) {
            error!("Failed to download Speaker ID model: {:#}", e);
        }
    }

    // 4. Whisper ASR (Ensure preferred active variant or working fallback is downloaded)
    let preferred = active_variant.unwrap_or("turbo");
    if find_whisper_files_for_variant(models_dir, preferred).is_none() {
        info!("Preferred Whisper model '{}' is not present. Downloading...", preferred);
        if let Err(e) = download_whisper_variant_with_progress(models_dir, preferred, app_handle) {
            error!("Failed to download preferred Whisper model '{}': {:#}. Falling back to tiny.en...", preferred, e);
            if find_whisper_files_for_variant(models_dir, "tiny.en").is_none() {
                let _ = download_whisper_variant_with_progress(models_dir, "tiny.en", app_handle);
            }
        }
    }

    Ok(check_models_status(models_dir, active_variant))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_whisper_files_for_variant_mock() {
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let temp_dir = std::env::temp_dir().join(format!("test_models_{}", ts));
        let tiny_dir = temp_dir.join("sherpa-onnx-whisper-tiny.en");
        fs::create_dir_all(&tiny_dir).unwrap();

        // Write dummy files (> 1000 bytes)
        let dummy = vec![0u8; 1200];
        fs::write(tiny_dir.join("tiny.en-encoder.int8.onnx"), &dummy).unwrap();
        fs::write(tiny_dir.join("tiny.en-decoder.int8.onnx"), &dummy).unwrap();
        fs::write(tiny_dir.join("tiny.en-tokens.txt"), &dummy).unwrap();

        let found = find_whisper_files_for_variant(&temp_dir, "tiny.en");
        assert!(found.is_some(), "Expected tiny.en to be found");

        let missing = find_whisper_files_for_variant(&temp_dir, "turbo");
        assert!(missing.is_none(), "Expected turbo to not be found");

        let options = get_available_whisper_models(&temp_dir, "tiny.en");
        let tiny_opt = options.iter().find(|o| o.id == "tiny.en").unwrap();
        assert!(tiny_opt.is_downloaded);
        assert!(tiny_opt.is_active);

        let turbo_opt = options.iter().find(|o| o.id == "turbo").unwrap();
        assert!(!turbo_opt.is_downloaded);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_find_whisper_files_direct_folder_matching() {
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let temp_dir = std::env::temp_dir().join(format!("test_models_custom_{}", ts));
        let custom_dir = temp_dir.join("whisper-turbo");
        fs::create_dir_all(&custom_dir).unwrap();

        let dummy = vec![0u8; 1200];
        fs::write(custom_dir.join("turbo-encoder.int8.onnx"), &dummy).unwrap();
        fs::write(custom_dir.join("turbo-decoder.int8.onnx"), &dummy).unwrap();
        fs::write(custom_dir.join("tokens.txt"), &dummy).unwrap();

        let found = find_whisper_files_for_variant(&temp_dir, "turbo");
        assert!(found.is_some(), "Expected turbo to be discovered in custom named folder");

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
