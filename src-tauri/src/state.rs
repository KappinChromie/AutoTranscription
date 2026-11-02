use crate::diarization::SpeakerRegistry;
use crate::transcriber::TranscriptEntry;
use parking_lot::RwLock;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::JoinHandle;

pub struct ActiveSession {
    pub is_running: Arc<AtomicBool>,
    pub mic_thread: Option<JoinHandle<()>>,
    pub app_thread: Option<JoinHandle<()>>,
    pub dsp_thread: Option<JoinHandle<()>>,
    pub transcriber_thread: Option<JoinHandle<()>>,
}

pub struct AppState {
    pub session: RwLock<Option<ActiveSession>>,
    pub registry: Arc<SpeakerRegistry>,
    pub history: Arc<RwLock<Vec<TranscriptEntry>>>,
    pub active_whisper_variant: RwLock<String>,
    pub active_device: RwLock<String>,
    pub custom_vocabulary: Arc<RwLock<Vec<String>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            session: RwLock::new(None),
            registry: Arc::new(SpeakerRegistry::new(0.58)),
            history: Arc::new(RwLock::new(Vec::new())),
            active_whisper_variant: RwLock::new("turbo".into()),
            active_device: RwLock::new("cpu".into()),
            custom_vocabulary: Arc::new(RwLock::new(Vec::new())),
        }
    }
}
