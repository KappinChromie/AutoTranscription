use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter};
use tracing::{info, warn};

use crate::models::DownloadProgress;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceOption {
    pub id: String, // "cpu" | "cuda" | "directml"
    pub name: String,
    pub tag: String,
    pub description: String,
    pub is_downloaded: bool,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceStatus {
    pub active_device: String,
    pub is_downloaded: bool,
    pub available_devices: Vec<DeviceOption>,
    pub runtimes_dir: String,
}

pub fn resolve_runtimes_dir(app_data_dir: Option<PathBuf>) -> PathBuf {
    // 1. Check local runtimes dir next to executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let local_runtimes = exe_dir.join("runtimes");
            if local_runtimes.exists() {
                return local_runtimes;
            }
            if let Some(parent) = exe_dir.parent() {
                let cand = parent.join("runtimes");
                if cand.exists() {
                    return cand;
                }
            }
        }
    }

    // 2. Check app_data_dir/runtimes
    if let Some(ref data_dir) = app_data_dir {
        let app_runtimes = data_dir.join("runtimes");
        let _ = fs::create_dir_all(&app_runtimes);
        return app_runtimes;
    }

    // 3. Fallback to ./runtimes
    let cwd_runtimes = PathBuf::from("runtimes");
    let _ = fs::create_dir_all(&cwd_runtimes);
    cwd_runtimes
}

pub fn is_device_downloaded(device: &str, runtimes_dir: &Path) -> bool {
    match device {
        "cpu" => true,
        "cuda" => {
            let cuda_dir = runtimes_dir.join("cuda");
            if !cuda_dir.exists() {
                return false;
            }
            cuda_dir.join("onnxruntime_providers_cuda.dll").exists()
                || cuda_dir.join("onnxruntime.dll").exists()
        }
        "directml" => {
            let dml_dir = runtimes_dir.join("directml");
            if !dml_dir.exists() {
                return false;
            }
            dml_dir.join("DirectML.dll").exists()
                || dml_dir.join("onnxruntime_providers_dml.dll").exists()
                || dml_dir.join("onnxruntime.dll").exists()
        }
        _ => false,
    }
}

pub fn get_available_devices(runtimes_dir: &Path, active_device: &str) -> Vec<DeviceOption> {
    let devices = [
        (
            "cpu",
            "CPU (AVX2 Multi-threaded)",
            "Default / 12 Cores",
            "High-speed multithreaded CPU inference with dynamic intra-op parallelism",
        ),
        (
            "cuda",
            "NVIDIA GPU (CUDA / Tensor Cores)",
            "RTX / GTX Accelerated",
            "Hardware-accelerated matrix multiplication using NVIDIA CUDA & Tensor Cores",
        ),
        (
            "directml",
            "AMD / Intel GPU (DirectML)",
            "DirectX 12 Acceleration",
            "Hardware acceleration across AMD Radeon, Intel Arc, and Windows DirectX 12 GPUs",
        ),
    ];

    devices
        .iter()
        .map(|(id, name, tag, desc)| {
            let is_downloaded = is_device_downloaded(id, runtimes_dir);
            let is_active = *id == active_device;
            DeviceOption {
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

pub fn check_device_status(runtimes_dir: &Path, active_device: &str) -> DeviceStatus {
    let is_downloaded = is_device_downloaded(active_device, runtimes_dir);
    let available_devices = get_available_devices(runtimes_dir, active_device);

    DeviceStatus {
        active_device: active_device.to_string(),
        is_downloaded,
        available_devices,
        runtimes_dir: runtimes_dir.to_string_lossy().to_string(),
    }
}

pub fn activate_runtime(device: &str, runtimes_dir: &Path) {
    if device == "cpu" {
        return;
    }

    let dev_dir = runtimes_dir.join(device);
    if !dev_dir.exists() {
        return;
    }

    info!("Activating GPU runtime for '{}' at {}", device, dev_dir.display());

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = dev_dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        unsafe {
            extern "system" {
                fn SetDllDirectoryW(lpPathName: *const u16) -> i32;
            }
            let res = SetDllDirectoryW(wide.as_ptr());
            if res == 0 {
                warn!("SetDllDirectoryW failed for {}", dev_dir.display());
            }
        }
    }

    if let Ok(curr_path) = std::env::var("PATH") {
        let dir_str = dev_dir.to_string_lossy();
        if !curr_path.contains(&*dir_str) {
            std::env::set_var("PATH", format!("{};{}", dir_str, curr_path));
        }
    }
}

pub fn download_device_runtime(
    device: &str,
    runtimes_dir: &Path,
    app_handle: &AppHandle,
) -> Result<()> {
    if device == "cpu" {
        return Ok(());
    }

    let target_dir = runtimes_dir.join(device);
    fs::create_dir_all(&target_dir)?;

    let item_id = format!("device-{}", device);

    let (url, fallback_url) = match device {
        "cuda" => (
            "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu.Windows/1.19.2",
            "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-gpu-1.19.2.zip",
        ),
        "directml" => (
            "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.19.2",
            "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-1.19.2.zip",
        ),
        _ => anyhow::bail!("Unsupported device runtime: {}", device),
    };

    info!("Starting download of GPU libraries for '{}' from {}", device, url);

    let _ = app_handle.emit(
        "download-progress",
        DownloadProgress {
            item_id: item_id.clone(),
            percentage: 0.0,
            downloaded_bytes: 0,
            total_bytes: 0,
            phase: "downloading".to_string(),
            message: "0%".to_string(),
        },
    );

    let temp_zip = target_dir.join("runtime_archive.zip.downloading");
    let dl_res = download_file_streaming(url, &temp_zip, &item_id, app_handle)
        .or_else(|_| download_file_streaming(fallback_url, &temp_zip, &item_id, app_handle));

    dl_res.with_context(|| format!("Failed to download runtime for {}", device))?;

    let _ = app_handle.emit(
        "download-progress",
        DownloadProgress {
            item_id: item_id.clone(),
            percentage: 85.0,
            downloaded_bytes: 0,
            total_bytes: 0,
            phase: "extracting".to_string(),
            message: "Extracting GPU Libraries...".to_string(),
        },
    );

    extract_zip_dlls(&temp_zip, &target_dir, &item_id, app_handle)?;
    let _ = fs::remove_file(&temp_zip);

    let _ = app_handle.emit(
        "download-progress",
        DownloadProgress {
            item_id: item_id.clone(),
            percentage: 100.0,
            downloaded_bytes: 0,
            total_bytes: 0,
            phase: "completed".to_string(),
            message: "Ready ✓".to_string(),
        },
    );

    info!("Successfully installed GPU runtime for '{}' in {}", device, target_dir.display());
    Ok(())
}

fn download_file_streaming(
    url: &str,
    dest_path: &Path,
    item_id: &str,
    app_handle: &AppHandle,
) -> Result<()> {
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("Failed to GET {}", url))?;

    let total_bytes: u64 = resp
        .header("content-length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut reader = resp.into_reader();
    let mut file = File::create(dest_path)?;
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

        if last_emit.elapsed() > std::time::Duration::from_millis(80) || (total_bytes > 0 && downloaded == total_bytes) {
            last_emit = std::time::Instant::now();
            let pct = if total_bytes > 0 {
                ((downloaded as f32 / total_bytes as f32) * 85.0).clamp(0.0, 85.0)
            } else {
                ((downloaded as f32 / (downloaded + 5_000_000) as f32) * 85.0).clamp(0.0, 85.0)
            };

            let _ = app_handle.emit(
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

    file.flush()?;
    Ok(())
}

fn extract_zip_dlls(
    zip_path: &Path,
    target_dir: &Path,
    item_id: &str,
    app_handle: &AppHandle,
) -> Result<()> {
    let file = File::open(zip_path)?;
    let reader = BufReader::new(file);
    let mut archive = zip::ZipArchive::new(reader)?;
    let total_entries = archive.len();

    let mut last_emit = std::time::Instant::now();

    for i in 0..total_entries {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();

        let is_dll = name.to_lowercase().ends_with(".dll");
        if is_dll && !name.ends_with('/') {
            let filename = Path::new(&name)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            let outpath = target_dir.join(&*filename);
            let mut outfile = File::create(&outpath)?;
            std::io::copy(&mut entry, &mut outfile)?;
            outfile.flush()?;
        }

        if last_emit.elapsed() > std::time::Duration::from_millis(100) || i + 1 == total_entries {
            last_emit = std::time::Instant::now();
            let extract_pct = 85.0 + ((i as f32 / total_entries.max(1) as f32) * 14.0);
            let _ = app_handle.emit(
                "download-progress",
                DownloadProgress {
                    item_id: item_id.to_string(),
                    percentage: extract_pct.clamp(85.0, 99.0),
                    downloaded_bytes: 0,
                    total_bytes: 0,
                    phase: "extracting".to_string(),
                    message: "Extracting...".to_string(),
                },
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_options_and_status() {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!("test_runtimes_{}", ts));
        let _ = fs::create_dir_all(&temp_dir);

        let status = check_device_status(&temp_dir, "cpu");
        assert_eq!(status.active_device, "cpu");
        assert!(status.is_downloaded); // CPU is always downloaded

        let devices = get_available_devices(&temp_dir, "cpu");
        assert_eq!(devices.len(), 3);
        let cpu_dev = devices.iter().find(|d| d.id == "cpu").unwrap();
        assert!(cpu_dev.is_downloaded);
        assert!(cpu_dev.is_active);

        let cuda_dev = devices.iter().find(|d| d.id == "cuda").unwrap();
        assert!(!cuda_dev.is_downloaded);

        // Simulate mock cuda dll
        let cuda_dir = temp_dir.join("cuda");
        let _ = fs::create_dir_all(&cuda_dir);
        let _ = fs::write(cuda_dir.join("onnxruntime_providers_cuda.dll"), b"dummy");

        let updated_devices = get_available_devices(&temp_dir, "cuda");
        let updated_cuda = updated_devices.iter().find(|d| d.id == "cuda").unwrap();
        assert!(updated_cuda.is_downloaded);

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
