use crate::audio_capture::{
    enumerate_audio_devices, enumerate_running_applications, start_app_capture_thread,
    start_mic_capture_thread, AppProcessInfo, AudioDeviceInfo, CaptureMode,
};
use crate::diarization::SpeakerProfile;
use crate::dsp_vad::start_dsp_vad_thread;
use crate::models::{
    check_models_status, get_model_paths_with_variant, resolve_models_dir,
    ModelStatus, WhisperModelOption,
};
use crate::state::{ActiveSession, AppState};
use crate::transcriber::{start_transcriber_thread, TranscriptEntry};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    pub mic_device_id: Option<String>,
    pub capture_mode: String, // "app_pid" | "global"
    pub target_pid: Option<u32>,
    pub target_app_name: String,
    pub similarity_threshold: f32,
    pub device: Option<String>, // "cpu" | "cuda" | "directml"
}

#[tauri::command]
pub fn get_audio_devices() -> Result<Vec<AudioDeviceInfo>, String> {
    enumerate_audio_devices().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_running_applications() -> Result<Vec<AppProcessInfo>, String> {
    enumerate_running_applications().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_models_status(
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<ModelStatus, String> {
    let app_data = app_handle.path().app_data_dir().ok();
    let dir = resolve_models_dir(app_data);
    let active_variant = state.active_whisper_variant.read().clone();
    let status = check_models_status(&dir, Some(&active_variant));

    // If currently selected model is missing but another valid model exists on disk, auto-sync active model
    let is_active_downloaded = crate::models::find_whisper_files_for_variant(&dir, &active_variant).is_some();
    if !is_active_downloaded && crate::models::find_whisper_files_for_variant(&dir, &status.active_whisper_model).is_some() {
        *state.active_whisper_variant.write() = status.active_whisper_model.clone();
    }

    Ok(status)
}

#[tauri::command]
pub async fn download_models(
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<ModelStatus, String> {
    let app_data = app_handle.path().app_data_dir().ok();
    let dir = resolve_models_dir(app_data);
    let active_variant = state.active_whisper_variant.read().clone();
    let app_handle_clone = app_handle.clone();
    tauri::async_runtime::spawn_blocking(move || {
        crate::models::ensure_default_models_with_progress(
            &dir,
            Some(&active_variant),
            Some(&app_handle_clone),
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn start_transcription(
    config: SessionConfig,
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<(), String> {
    info!("Starting dual transcription session with config: {:?}", config);

    // Stop existing session if running
    stop_internal_session(&state);

    // Update similarity threshold
    state.registry.set_threshold(config.similarity_threshold);

    // Update active device & activate runtime directory for GPU providers
    let device = config.device.clone().unwrap_or_else(|| state.active_device.read().clone());
    *state.active_device.write() = device.clone();

    let app_data = app_handle.path().app_data_dir().ok();
    let runtimes_dir = crate::runtimes::resolve_runtimes_dir(app_data.clone());
    crate::runtimes::activate_runtime(&device, &runtimes_dir);

    let is_running = Arc::new(AtomicBool::new(true));

    // Allocate lock-free SPSC ring buffers: 48kHz * 5 seconds = 240,000 samples
    let (mic_producer, mic_consumer) = rtrb::RingBuffer::<f32>::new(240_000);
    let (app_producer, app_consumer) = rtrb::RingBuffer::<f32>::new(240_000);

    // Create crossbeam channel for SpeechSegment transmission
    let (segment_tx, segment_rx) = crossbeam_channel::unbounded();

    // Model paths with active variant
    let models_dir = resolve_models_dir(app_data);
    let active_variant = state.active_whisper_variant.read().clone();
    let model_paths = get_model_paths_with_variant(&models_dir, Some(&active_variant));

    // 1. Spawn Track A Mic Capture Thread
    let mic_thread = start_mic_capture_thread(
        config.mic_device_id.clone(),
        mic_producer,
        is_running.clone(),
    )
    .map_err(|e| format!("Failed to start Mic capture thread: {:#}", e))?;

    // 2. Spawn Track B App Loopback Capture Thread
    let capture_mode = if config.capture_mode == "app_pid" && config.target_pid.is_some() {
        CaptureMode::ApplicationPid(config.target_pid.unwrap())
    } else {
        CaptureMode::GlobalLoopback
    };

    let app_thread = start_app_capture_thread(
        capture_mode,
        app_producer,
        is_running.clone(),
    )
    .map_err(|e| format!("Failed to start App capture thread: {:#}", e))?;

    // 3. Spawn DSP & VAD Thread
    let dsp_thread = start_dsp_vad_thread(
        mic_consumer,
        app_consumer,
        segment_tx,
        app_handle.clone(),
        model_paths.vad_path.clone(),
        model_paths.pyannote_path.clone(),
        is_running.clone(),
        config.target_app_name.clone(),
        config.target_pid,
    )
    .map_err(|e| format!("Failed to start DSP/VAD thread: {:#}", e))?;

    // 4. Spawn ML Diarization & Transcription Worker
    let transcriber_thread = start_transcriber_thread(
        segment_rx,
        app_handle.clone(),
        model_paths,
        state.registry.clone(),
        is_running.clone(),
        state.history.clone(),
        state.custom_vocabulary.clone(),
        device,
    )
    .map_err(|e| format!("Failed to start Transcriber thread: {:#}", e))?;

    let session = ActiveSession {
        is_running,
        mic_thread: Some(mic_thread),
        app_thread: Some(app_thread),
        dsp_thread: Some(dsp_thread),
        transcriber_thread: Some(transcriber_thread),
    };

    *state.session.write() = Some(session);
    info!("All dual capture and inference threads started.");
    Ok(())
}

#[tauri::command]
pub async fn stop_transcription(state: State<'_, AppState>) -> Result<(), String> {
    info!("Stopping transcription session");
    let session = {
        let mut session_guard = state.session.write();
        session_guard.take()
    };

    if let Some(session) = session {
        session.is_running.store(false, Ordering::SeqCst);
        tauri::async_runtime::spawn_blocking(move || {
            if let Some(t) = session.mic_thread {
                let _ = t.join();
            }
            if let Some(t) = session.app_thread {
                let _ = t.join();
            }
            if let Some(t) = session.dsp_thread {
                let _ = t.join();
            }
            if let Some(t) = session.transcriber_thread {
                let _ = t.join();
            }
            info!("All capture and inference threads terminated cleanly in background.");
        });
    }

    Ok(())
}

fn stop_internal_session(state: &AppState) {
    let mut session_guard = state.session.write();
    if let Some(session) = session_guard.take() {
        session.is_running.store(false, Ordering::SeqCst);
        if let Some(t) = session.mic_thread {
            let _ = t.join();
        }
        if let Some(t) = session.app_thread {
            let _ = t.join();
        }
        if let Some(t) = session.dsp_thread {
            let _ = t.join();
        }
        if let Some(t) = session.transcriber_thread {
            let _ = t.join();
        }
    }
}

#[tauri::command]
pub fn rename_speaker(
    speaker_id: String,
    new_name: String,
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<Vec<SpeakerProfile>, String> {
    let updated = state
        .registry
        .rename_speaker(&speaker_id, &new_name)
        .ok_or_else(|| "Speaker ID not found".to_string())?;

    // Retroactively update historical entries
    let mut history = state.history.write();
    for entry in history.iter_mut() {
        if entry.speaker_id == speaker_id {
            entry.speaker_name = new_name.clone();
        }
    }

    let _ = app_handle.emit("speakers-updated", &updated);
    Ok(updated)
}

#[tauri::command]
pub fn merge_speakers(
    source_speaker_id: String,
    target_speaker_id: String,
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<Vec<SpeakerProfile>, String> {
    let updated = state
        .registry
        .merge_speakers(&source_speaker_id, &target_speaker_id)
        .ok_or_else(|| "Failed to merge speakers".to_string())?;

    // Find target display name and color
    let (tgt_name, tgt_color) = updated
        .iter()
        .find(|p| p.id == target_speaker_id)
        .map(|p| (p.display_name.clone(), p.color_hex.clone()))
        .unwrap_or_else(|| ("Person".to_string(), "#3b82f6".to_string()));

    // Retroactively re-map historical entries
    let mut history = state.history.write();
    for entry in history.iter_mut() {
        if entry.speaker_id == source_speaker_id {
            entry.speaker_id = target_speaker_id.clone();
            entry.speaker_name = tgt_name.clone();
            entry.speaker_color = tgt_color.clone();
        }
    }

    let _ = app_handle.emit("speakers-updated", &updated);
    Ok(updated)
}

#[tauri::command]
pub fn lock_speaker(
    speaker_id: String,
    locked: bool,
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<Vec<SpeakerProfile>, String> {
    let updated = state
        .registry
        .lock_speaker(&speaker_id, locked)
        .ok_or_else(|| "Speaker ID not found".to_string())?;

    let _ = app_handle.emit("speakers-updated", &updated);
    Ok(updated)
}

#[tauri::command]
pub fn clear_session(state: State<'_, AppState>, app_handle: AppHandle) -> Result<(), String> {
    state.history.write().clear();
    state.registry.clear();
    let profiles = state.registry.get_profiles();
    let _ = app_handle.emit("speakers-updated", &profiles);
    Ok(())
}

#[tauri::command]
pub fn get_transcript_history(state: State<'_, AppState>) -> Result<Vec<TranscriptEntry>, String> {
    Ok(state.history.read().clone())
}

#[tauri::command]
pub fn get_speakers(state: State<'_, AppState>) -> Result<Vec<SpeakerProfile>, String> {
    Ok(state.registry.get_profiles())
}

#[tauri::command]
pub fn set_diarization_threshold(
    threshold: f32,
    state: State<'_, AppState>,
) -> Result<f32, String> {
    state.registry.set_threshold(threshold);
    Ok(state.registry.get_threshold())
}

#[tauri::command]
pub fn get_diarization_threshold(state: State<'_, AppState>) -> Result<f32, String> {
    Ok(state.registry.get_threshold())
}

#[tauri::command]
pub fn export_transcript(format: String, state: State<'_, AppState>) -> Result<String, String> {
    let history = state.history.read();
    let mut output = String::new();

    if format.to_lowercase() == "markdown" || format.to_lowercase() == "md" {
        output.push_str("# Live Transcript Export\n\n");
        for entry in history.iter() {
            let context = if entry.source_type == "mic" {
                format!("*(to {})*", entry.app_name)
            } else {
                format!("*({})*", entry.app_name)
            };
            output.push_str(&format!(
                "- `[{}]` **{}** {}: {}\n",
                entry.timestamp_str, entry.speaker_name, context, entry.text
            ));
        }
    } else {
        // Plain text format matching specification
        for entry in history.iter() {
            let label = if entry.source_type == "mic" {
                format!("Me (to {})", entry.app_name)
            } else {
                entry.speaker_name.clone()
            };
            output.push_str(&format!("{}: {}\n", label, entry.text));
        }
    }

    Ok(output)
}

#[tauri::command]
pub fn get_available_whisper_models(
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<Vec<WhisperModelOption>, String> {
    let app_data = app_handle.path().app_data_dir().ok();
    let dir = resolve_models_dir(app_data);
    let active_variant = state.active_whisper_variant.read().clone();
    Ok(crate::models::get_available_whisper_models(&dir, &active_variant))
}

#[tauri::command]
pub fn set_active_whisper_model(
    variant: String,
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<ModelStatus, String> {
    *state.active_whisper_variant.write() = variant.clone();
    let app_data = app_handle.path().app_data_dir().ok();
    let dir = resolve_models_dir(app_data);
    Ok(check_models_status(&dir, Some(&variant)))
}

#[tauri::command]
pub async fn download_whisper_model(
    variant: String,
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<ModelStatus, String> {
    let app_data = app_handle.path().app_data_dir().ok();
    let dir = resolve_models_dir(app_data);
    let var_clone = variant.clone();
    let dir_clone = dir.clone();
    let app_handle_clone = app_handle.clone();

    tauri::async_runtime::spawn_blocking(move || {
        crate::models::download_whisper_variant_with_progress(
            &dir_clone,
            &var_clone,
            Some(&app_handle_clone),
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())??;

    *state.active_whisper_variant.write() = variant.clone();
    Ok(check_models_status(&dir, Some(&variant)))
}

#[tauri::command]
pub fn get_available_devices(
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<Vec<crate::runtimes::DeviceOption>, String> {
    let app_data = app_handle.path().app_data_dir().ok();
    let runtimes_dir = crate::runtimes::resolve_runtimes_dir(app_data);
    let active_device = state.active_device.read().clone();
    Ok(crate::runtimes::get_available_devices(&runtimes_dir, &active_device))
}

#[tauri::command]
pub fn get_device_status(
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<crate::runtimes::DeviceStatus, String> {
    let app_data = app_handle.path().app_data_dir().ok();
    let runtimes_dir = crate::runtimes::resolve_runtimes_dir(app_data);
    let active_device = state.active_device.read().clone();
    Ok(crate::runtimes::check_device_status(&runtimes_dir, &active_device))
}

#[tauri::command]
pub fn set_active_device(
    device: String,
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<crate::runtimes::DeviceStatus, String> {
    *state.active_device.write() = device.clone();
    let app_data = app_handle.path().app_data_dir().ok();
    let runtimes_dir = crate::runtimes::resolve_runtimes_dir(app_data);
    Ok(crate::runtimes::check_device_status(&runtimes_dir, &device))
}

#[tauri::command]
pub async fn download_device_runtime(
    device: String,
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<crate::runtimes::DeviceStatus, String> {
    let app_data = app_handle.path().app_data_dir().ok();
    let runtimes_dir = crate::runtimes::resolve_runtimes_dir(app_data);
    let dev_clone = device.clone();
    let dir_clone = runtimes_dir.clone();
    let handle_clone = app_handle.clone();

    tauri::async_runtime::spawn_blocking(move || {
        crate::runtimes::download_device_runtime(&dev_clone, &dir_clone, &handle_clone)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())??;

    *state.active_device.write() = device.clone();
    Ok(crate::runtimes::check_device_status(&runtimes_dir, &device))
}

#[tauri::command]
pub fn get_custom_vocabulary(state: State<'_, AppState>) -> Result<Vec<String>, String> {
    Ok(state.custom_vocabulary.read().clone())
}

#[tauri::command]
pub fn set_custom_vocabulary(
    vocabulary: Vec<String>,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let sanitized: Vec<String> = vocabulary
        .into_iter()
        .map(|w| w.trim().to_string())
        .filter(|w| !w.is_empty())
        .collect();
    *state.custom_vocabulary.write() = sanitized;
    Ok(())
}
