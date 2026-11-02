use crate::diarization::{Diarizer, SpeakerMatch, SpeakerRegistry, TranscriptRevision};
use crate::dsp_vad::{AudioSource, SpeechSegment};
use crate::models::ModelPaths;
use crossbeam_channel::Receiver;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStatus {
    pub requested: String,
    pub actual: String,
    pub is_fallback: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub id: u64,
    pub timestamp_str: String,
    pub source_type: String, // "mic" | "app"
    pub app_name: String,
    pub speaker_id: String,
    pub speaker_name: String,
    pub speaker_color: String,
    pub text: String,
    pub confidence: f32,
    pub duration_ms: u64,
}

fn format_timestamp(ms: u64) -> String {
    let secs = (ms / 1000) % 86400;
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, s)
}

/// Detects subword and character-level repetition loops in a single token
pub fn has_subword_loop(word: &str) -> bool {
    let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
    let c: Vec<char> = clean.chars().collect();
    let n = c.len();
    if n > 25 {
        return true;
    }
    for unit in 1..=4 {
        let min_run = if unit == 1 { 5 } else { 4 };
        for start in 0..n {
            let mut run = 1;
            while start + (run + 1) * unit <= n
                && c[start..start + unit] == c[start + run * unit..start + (run + 1) * unit]
            {
                run += 1;
            }
            if run >= min_run {
                return true;
            }
        }
    }
    false
}

/// Advanced Text Normalizer & Hallucination Filter:
/// - Strips phantom Whisper artifacts ([BLANK_AUDIO], [music], ♪, etc.)
/// - Detects single-token subword repetition loops (has_subword_loop)
/// - Detects multi-ngram repetition loops and compression ratio threshold (>2.4)
/// - Normalizes repeated punctuation (".." -> ".") and capitalizes first alphanumeric letter
pub fn clean_and_normalize_transcript(raw: &str) -> Option<String> {
    let mut text = raw.trim().to_string();
    if text.is_empty() {
        debug!("Filter discarded segment: empty raw input");
        return None;
    }

    // 1. Strip phantom Whisper hallucination tokens
    let phantom_patterns = [
        "[blank_audio]", "(blank_audio)", "[BLANK_AUDIO]", "(BLANK_AUDIO)",
        "[music]", "(music)", "[applause]", "(applause)",
        "[laughter]", "(laughter)", "♪", "♫",
        "subtitles by", "subtitle by", "thanks for watching", "thanks for watching!",
        "thank you for watching", "thank you very much.", "mbc 뉴스", "mbc news",
    ];

    for pattern in &phantom_patterns {
        text = text.replace(pattern, "");
        text = text.replace(&pattern.to_uppercase(), "");
        let mut chars = pattern.chars();
        if let Some(first) = chars.next() {
            let cap = first.to_uppercase().to_string() + chars.as_str();
            text = text.replace(&cap, "");
        }
    }

    text = text.trim().to_string();
    if text.is_empty() {
        debug!("Filter discarded segment: empty after phantom token removal: \"{}\"", raw);
        return None;
    }

    // 2. Repetition Loop Detection
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        debug!("Filter discarded segment: no words found after tokenization: \"{}\"", raw);
        return None;
    }

    // 2a. Single-word character/sub-word repetition loop (e.g. "pdpdpdpdpd" or "aaaaaaaa")
    for &w in &words {
        if has_subword_loop(w) {
            debug!("Filter discarded segment: sub-word repetition loop in token '{}': \"{}\"", w, raw);
            return None;
        }
    }

    // 2b. Single-word consecutive repetition (e.g. "you you you you")
    let mut consecutive_word_repeats = 0;
    let mut prev_clean = String::new();
    for &w in &words {
        let clean_w = w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        if !clean_w.is_empty() && clean_w == prev_clean {
            consecutive_word_repeats += 1;
            if consecutive_word_repeats >= 3 {
                debug!("Filter discarded segment: consecutive word repeat ('{}' x {}): \"{}\"", clean_w, consecutive_word_repeats + 1, raw);
                return None;
            }
        } else {
            prev_clean = clean_w;
            consecutive_word_repeats = 0;
        }
    }

    // 2c. Compression ratio threshold: if total words >= 6 and ratio > 2.4
    if words.len() >= 6 {
        let mut unique = HashSet::new();
        for &w in &words {
            unique.insert(w.to_lowercase());
        }
        let ratio = words.len() as f32 / unique.len() as f32;
        if ratio > 2.4 {
            debug!("Filter discarded segment: high compression ratio ({:.2} > 2.4, {} words): \"{}\"", ratio, words.len(), raw);
            return None;
        }
    }

    // 3. Punctuation Cleanup
    while text.contains("..") {
        text = text.replace("..", ".");
    }
    while text.contains(",,") {
        text = text.replace(",,", ",");
    }
    while text.contains("??") {
        text = text.replace("??", "?");
    }
    while text.contains("!!") {
        text = text.replace("!!", "!");
    }

    // Trim lone leading/trailing punctuation artifacts
    text = text.trim_start_matches(|c: char| c == '.' || c == ',' || c == '-' || c == ' ' || c == ';').to_string();
    text = text.trim().to_string();
    if text.is_empty() {
        debug!("Filter discarded segment: empty after punctuation trim: \"{}\"", raw);
        return None;
    }

    // If text has no alphanumeric characters, discard
    if !text.chars().any(|c| c.is_alphanumeric()) {
        debug!("Filter discarded segment: no alphanumeric characters: \"{}\"", raw);
        return None;
    }

    // 4. Capitalization (start of utterance and after sentence delimiters)
    let mut chars: Vec<char> = text.chars().collect();
    let mut capitalize_next = true;
    for c in chars.iter_mut() {
        if *c == '.' || *c == '?' || *c == '!' {
            capitalize_next = true;
        } else if c.is_alphanumeric() {
            if capitalize_next {
                *c = c.to_uppercase().next().unwrap_or(*c);
                capitalize_next = false;
            }
        }
    }
    let formatted: String = chars.into_iter().collect();

    Some(formatted)
}

/// Applies custom vocabulary dictionary biasing to align proper nouns and technical terms
pub fn apply_custom_vocabulary_biasing(text: &str, vocab: &[String]) -> String {
    if vocab.is_empty() || text.is_empty() {
        return text.to_string();
    }
    let mut words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
    for vocab_word in vocab {
        let clean_vocab = vocab_word.trim();
        if clean_vocab.is_empty() {
            continue;
        }
        let lower_vocab = clean_vocab.to_lowercase();
        for w in &mut words {
            let prefix: String = w.chars().take_while(|c| !c.is_alphanumeric()).collect();
            let suffix: String = w.chars().rev().take_while(|c| !c.is_alphanumeric()).collect::<String>().chars().rev().collect();
            let core = w.trim_matches(|c: char| !c.is_alphanumeric());
            if core.eq_ignore_ascii_case(&lower_vocab) {
                *w = format!("{}{}{}", prefix, clean_vocab, suffix);
            }
        }
    }
    words.join(" ")
}

/// Advanced Whisper Recognizer with Hotwords and Deterministic Greedy Search
pub struct AdvancedWhisperRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
    pub actual_provider: String,
}

impl AdvancedWhisperRecognizer {
    pub fn new(
        encoder_path: &Path,
        decoder_path: &Path,
        tokens_path: &Path,
        provider: &str,
    ) -> anyhow::Result<Self> {
        info!("Whisper Recognizer requested provider: '{}'", provider);
        let encoder_cstr = std::ffi::CString::new(encoder_path.to_string_lossy().as_bytes())?;
        let decoder_cstr = std::ffi::CString::new(decoder_path.to_string_lossy().as_bytes())?;
        let tokens_cstr = std::ffi::CString::new(tokens_path.to_string_lossy().as_bytes())?;
        let language_cstr = std::ffi::CString::new("en")?;
        let task_cstr = std::ffi::CString::new("transcribe")?;
        let provider_cstr = std::ffi::CString::new(provider)?;
        let decoding_method_cstr = std::ffi::CString::new("greedy_search")?; // temperature = 0.0
        let empty_cstr = std::ffi::CString::new("")?;

        let whisper_config = sherpa_rs_sys::SherpaOnnxOfflineWhisperModelConfig {
            encoder: encoder_cstr.as_ptr(),
            decoder: decoder_cstr.as_ptr(),
            language: language_cstr.as_ptr(),
            task: task_cstr.as_ptr(),
            tail_paddings: 0,
        };

        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let whisper_threads = (cpus / 2).clamp(2, 6) as i32;
        info!("Initializing Whisper Recognizer with requested provider '{}' and {} threads (total cores: {})", provider, whisper_threads, cpus);

        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                whisper: whisper_config,
                debug: 0,
                num_threads: whisper_threads,
                provider: provider_cstr.as_ptr(),
                tokens: tokens_cstr.as_ptr(),
                bpe_vocab: empty_cstr.as_ptr(),
                model_type: empty_cstr.as_ptr(),
                modeling_unit: empty_cstr.as_ptr(),
                nemo_ctc: std::mem::zeroed(),
                paraformer: std::mem::zeroed(),
                tdnn: std::mem::zeroed(),
                telespeech_ctc: std::mem::zeroed(),
                transducer: std::mem::zeroed(),
                fire_red_asr: std::mem::zeroed(),
                sense_voice: std::mem::zeroed(),
                moonshine: std::mem::zeroed(),
                dolphin: std::mem::zeroed(),
                zipformer_ctc: std::mem::zeroed(),
                canary: std::mem::zeroed(),
            }
        };

        let hr = sherpa_rs_sys::SherpaOnnxHomophoneReplacerConfig {
            dict_dir: empty_cstr.as_ptr(),
            lexicon: empty_cstr.as_ptr(),
            rule_fsts: empty_cstr.as_ptr(),
        };

        let recognizer_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 512,
                },
                model_config,
                decoding_method: decoding_method_cstr.as_ptr(),
                max_active_paths: 0,
                hotwords_file: empty_cstr.as_ptr(),
                hotwords_score: 0.0,
                lm_config: std::mem::zeroed(),
                rule_fars: empty_cstr.as_ptr(),
                rule_fsts: empty_cstr.as_ptr(),
                blank_penalty: 0.0,
                hr,
            }
        };

        let mut actual_provider = provider.to_string();
        let mut recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config) };
        if recognizer.is_null() && provider != "cpu" {
            warn!("⚠️ EXECUTION PROVIDER FALLBACK: Requested execution provider '{}' failed to initialize! Falling back to 'cpu'.", provider);
            let cpu_cstr = std::ffi::CString::new("cpu")?;
            let mut cpu_model_config = model_config;
            cpu_model_config.provider = cpu_cstr.as_ptr();
            let mut cpu_rec_config = recognizer_config;
            cpu_rec_config.model_config = cpu_model_config;
            recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&cpu_rec_config) };
            if !recognizer.is_null() {
                actual_provider = "cpu".to_string();
            }
        }

        if recognizer.is_null() {
            anyhow::bail!("Failed to create SherpaOnnx offline recognizer");
        }

        info!("Whisper Recognizer initialized successfully: requested='{}', actual='{}'", provider, actual_provider);
        Ok(Self { recognizer, actual_provider })
    }

    pub fn transcribe(&mut self, samples_16k: &[f32]) -> String {
        unsafe {
            // Whisper models in sherpa-onnx only support SherpaOnnxCreateOfflineStream.
            // Calling SherpaOnnxCreateOfflineStreamWithHotwords aborts via exit(-1).
            let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);

            if stream.is_null() {
                return String::new();
            }

            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                16000,
                samples_16k.as_ptr(),
                samples_16k.len() as i32,
            );

            sherpa_rs_sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(stream);

            if result_ptr.is_null() {
                sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
                return String::new();
            }

            let raw_result = result_ptr.read();
            let text = if !raw_result.text.is_null() {
                std::ffi::CStr::from_ptr(raw_result.text).to_string_lossy().to_string()
            } else {
                String::new()
            };

            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);

            text
        }
    }
}

unsafe impl Send for AdvancedWhisperRecognizer {}
unsafe impl Sync for AdvancedWhisperRecognizer {}

impl Drop for AdvancedWhisperRecognizer {
    fn drop(&mut self) {
        unsafe {
            if !self.recognizer.is_null() {
                sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
            }
        }
    }
}

pub fn start_transcriber_thread(
    segment_receiver: Receiver<SpeechSegment>,
    app_handle: AppHandle,
    model_paths: ModelPaths,
    registry: Arc<SpeakerRegistry>,
    is_running: Arc<AtomicBool>,
    history_entries: Arc<RwLock<Vec<TranscriptEntry>>>,
    custom_vocabulary: Arc<RwLock<Vec<String>>>,
    device: String,
) -> anyhow::Result<thread::JoinHandle<()>> {
    let handle = thread::Builder::new()
        .name("ml-inference-transcriber".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            info!("ML Inference & Transcription Worker started with device '{}'.", device);

            // Initialize Advanced Whisper Recognizer
            let (mut recognizer, actual_provider) = if model_paths.whisper_encoder.exists()
                && model_paths.whisper_decoder.exists()
                && model_paths.whisper_tokens.exists()
            {
                match AdvancedWhisperRecognizer::new(
                    &model_paths.whisper_encoder,
                    &model_paths.whisper_decoder,
                    &model_paths.whisper_tokens,
                    &device,
                ) {
                    Ok(rec) => {
                        let actual = rec.actual_provider.clone();
                        (Some(rec), actual)
                    }
                    Err(e) => {
                        warn!("Failed to init AdvancedWhisperRecognizer with device '{}': {:?}", device, e);
                        (None, "none".to_string())
                    }
                }
            } else {
                warn!("Whisper model files not found on disk.");
                (None, "none".to_string())
            };

            let is_fallback = device != "cpu" && actual_provider == "cpu";
            info!(
                "Execution provider status: requested='{}', actual='{}', is_fallback={}",
                device, actual_provider, is_fallback
            );
            let _ = app_handle.emit("provider-status", ProviderStatus {
                requested: device.clone(),
                actual: actual_provider.clone(),
                is_fallback,
            });

            // Initialize Diarizer
            let mut diarizer = Diarizer::new(&model_paths.speaker_id_path, registry.clone(), Some(&actual_provider));

            let mut prev_speakers_count = registry.get_profiles().len();
            let mut track_b_seg_count: usize = 0;

            while is_running.load(Ordering::Relaxed) {
                let segment = match segment_receiver.recv_timeout(Duration::from_millis(100)) {
                    Ok(seg) => seg,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                };

                let remaining_in_queue = segment_receiver.len();
                info!(
                    "Segment #{} pulled from queue (remaining queued: {}) | Source: {:?} | Duration: {}ms",
                    segment.id, remaining_in_queue, segment.source, segment.duration_ms
                );

                if !is_running.load(Ordering::Relaxed) {
                    break;
                }

                // Queue backlog mitigation: If transcriber falls behind, skip short non-critical fragments
                let queue_len = segment_receiver.len();
                if queue_len > 3 && segment.duration_ms < 1000 {
                    warn!("Transcriber queue backlog ({} segments), skipping short fragment ({}ms) to catch up", queue_len, segment.duration_ms);
                    continue;
                }

                // 1. Speaker Identification / Diarization
                let mut segment_revisions: Vec<TranscriptRevision> = Vec::new();
                let (source_type, app_name, speaker_match) = match &segment.source {
                    AudioSource::Microphone => {
                        // Track A: Hardcoded zero-overhead bypass to "Me"
                        let match_res = SpeakerMatch {
                            speaker_id: "me".into(),
                            display_name: "Me".into(),
                            color_hex: "#10b981".into(),
                            confidence: 1.0,
                        };
                        ("mic".to_string(), "Hardware Mic".to_string(), match_res)
                    }
                    AudioSource::Application { pid: _, app_name } => {
                        track_b_seg_count += 1;
                        let clean_dur_ms = (segment.clean_samples_16k.len() as u64 * 1000) / 16000;
                        let (match_res, revs) = diarizer.process_track_b(
                            segment.id,
                            &segment.clean_samples_16k,
                            clean_dur_ms,
                        );
                        segment_revisions.extend(revs);

                        // Periodically trigger rolling AHC clustering
                        if track_b_seg_count % 8 == 0 {
                            let ahc_revs = registry.run_rolling_ahc();
                            segment_revisions.extend(ahc_revs);
                        }

                        ("app".to_string(), app_name.clone(), match_res)
                    }
                };

                // Process retroactive revisions if any
                if !segment_revisions.is_empty() {
                    let mut history = history_entries.write();
                    for rev in &segment_revisions {
                        for item in history.iter_mut() {
                            if item.id == rev.id {
                                item.speaker_id = rev.speaker_id.clone();
                                item.speaker_name = rev.speaker_name.clone();
                                item.speaker_color = rev.speaker_color.clone();
                            }
                        }
                    }
                    drop(history);
                    let _ = app_handle.emit("transcript-revisions", &segment_revisions);
                }

                // Check if speaker roster updated and emit event
                let current_speakers = registry.get_profiles();
                if current_speakers.len() != prev_speakers_count || !segment_revisions.is_empty() {
                    prev_speakers_count = current_speakers.len();
                    let _ = app_handle.emit("speakers-updated", &current_speakers);
                }

                if !is_running.load(Ordering::Relaxed) {
                    break;
                }

                // 2. Speech-to-Text Transcription with Whisper
                let t_start = std::time::Instant::now();
                let raw_text = if let Some(ref mut rec) = recognizer {
                    rec.transcribe(&segment.samples_16k)
                } else {
                    format!("[Audio detected: {:.1}s]", segment.duration_ms as f32 / 1000.0)
                };
                let inference_ms = t_start.elapsed().as_millis() as u64;
                let rtf = if segment.duration_ms > 0 {
                    inference_ms as f32 / segment.duration_ms as f32
                } else {
                    0.0
                };
                info!(
                    "ASR Inferred Segment #{} | Duration: {}ms | Inference Time: {}ms | RTF: {:.2}x | Raw output: \"{}\"",
                    segment.id,
                    segment.duration_ms,
                    inference_ms,
                    rtf,
                    raw_text.trim()
                );

                // 3. Text Normalization, Vocabulary Biasing & Hallucination Removal
                let normalized_text = match clean_and_normalize_transcript(&raw_text) {
                    Some(t) => t,
                    None => continue,
                };
                let vocab_list = custom_vocabulary.read().clone();
                let cleaned_text = apply_custom_vocabulary_biasing(&normalized_text, &vocab_list);

                let timestamp_str = format_timestamp(segment.started_at_ms);

                let entry = TranscriptEntry {
                    id: segment.id,
                    timestamp_str,
                    source_type,
                    app_name,
                    speaker_id: speaker_match.speaker_id,
                    speaker_name: speaker_match.display_name,
                    speaker_color: speaker_match.color_hex,
                    text: cleaned_text,
                    confidence: speaker_match.confidence,
                    duration_ms: segment.duration_ms,
                };

                // Store in history
                history_entries.write().push(entry.clone());

                // Emit Tauri event to frontend
                let _ = app_handle.emit("transcript-entry", &entry);
            }

            // Drain remaining receiver queue on shutdown so memory is promptly freed
            while let Ok(_) = segment_receiver.try_recv() {}
            info!("ML Inference & Transcription Worker shut down cleanly.");
        })?;

    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_phantom_tokens() {
        let raw = "[BLANK_AUDIO] hello there [music] ♪";
        let cleaned = clean_and_normalize_transcript(raw);
        assert_eq!(cleaned, Some("Hello there".to_string()));
    }

    #[test]
    fn test_clean_punctuation_and_capitalization() {
        let raw = "hello world.. how are you??";
        let cleaned = clean_and_normalize_transcript(raw);
        assert_eq!(cleaned, Some("Hello world. How are you?".to_string()));
    }

    #[test]
    fn test_filter_empty_or_only_phantom() {
        let raw = "[BLANK_AUDIO]  [music] ♪";
        let cleaned = clean_and_normalize_transcript(raw);
        assert_eq!(cleaned, None);
    }

    #[test]
    fn test_filter_repetition_loops() {
        let raw = "you know you know you know you know you know you know you know";
        let cleaned = clean_and_normalize_transcript(raw);
        assert_eq!(cleaned, None, "Repetitive ngram hallucination loop should be filtered");
    }

    #[test]
    fn test_apply_custom_vocabulary_biasing() {
        let text = "Hey tyler, join discord and test wasapi!";
        let vocab = vec!["Tyler".to_string(), "Discord".to_string(), "WASAPI".to_string()];
        let biased = apply_custom_vocabulary_biasing(text, &vocab);
        assert_eq!(biased, "Hey Tyler, join Discord and test WASAPI!");
    }

    #[test]
    fn test_subword_repetition_loops_rejected() {
        assert!(has_subword_loop("pdpdpdpdpd"));
        assert!(has_subword_loop("aaaaaaaa"));
        assert!(has_subword_loop("pdpdpdpdpd!"));
        assert_eq!(clean_and_normalize_transcript("pdpdpdpdpd"), None);
        assert_eq!(clean_and_normalize_transcript("then he said aaaaaaaa and left"), None);
    }

    #[test]
    fn test_normal_words_and_sentences_pass() {
        assert!(!has_subword_loop("hello"));
        assert!(!has_subword_loop("stuff"));
        assert!(!has_subword_loop("mississippi"));
        assert!(!has_subword_loop("bookkeeper"));

        assert_eq!(
            clean_and_normalize_transcript("Hello stuff mississippi bookkeeper"),
            Some("Hello stuff mississippi bookkeeper".to_string())
        );
    }
}
