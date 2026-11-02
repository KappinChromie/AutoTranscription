pub mod audio_capture;
pub mod commands;
pub mod diarization;
pub mod dsp_vad;
pub mod models;
pub mod runtimes;
pub mod segmentation;
pub mod state;
pub mod transcriber;

use commands::*;
use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            get_audio_devices,
            get_running_applications,
            get_models_status,
            download_models,
            start_transcription,
            stop_transcription,
            rename_speaker,
            merge_speakers,
            lock_speaker,
            clear_session,
            get_transcript_history,
            get_speakers,
            set_diarization_threshold,
            get_diarization_threshold,
            export_transcript,
            get_available_whisper_models,
            set_active_whisper_model,
            download_whisper_model,
            get_available_devices,
            get_device_status,
            set_active_device,
            download_device_runtime,
            get_custom_vocabulary,
            set_custom_vocabulary
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
