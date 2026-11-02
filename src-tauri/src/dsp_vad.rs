use anyhow::Result;
use crossbeam_channel::Sender;
use serde::{Deserialize, Serialize};
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use crate::segmentation::PyannoteSegmenter;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};
use tracing::{info, warn};

static NEXT_SEGMENT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AudioSource {
    #[serde(rename = "mic")]
    Microphone,
    #[serde(rename = "app")]
    Application {
        pid: Option<u32>,
        app_name: String,
    },
}

#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub id: u64,
    pub source: AudioSource,
    pub samples_16k: Vec<f32>,
    pub clean_samples_16k: Vec<f32>,
    pub started_at_ms: u64,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioLevels {
    pub mic_rms: f32,
    pub app_rms: f32,
    pub mic_speaking: bool,
    pub app_speaking: bool,
}

/// Simple 3:1 anti-aliasing decimation filter (48kHz -> 16kHz)
pub fn decimate_48k_to_16k(input: &[f32]) -> Vec<f32> {
    let out_len = input.len() / 3;
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let idx = i * 3;
        let s0 = input[idx];
        let s1 = input[idx + 1];
        let s2 = input[idx + 2];
        let filtered = 0.25 * s0 + 0.5 * s1 + 0.25 * s2;
        output.push(filtered.clamp(-1.0, 1.0));
    }
    output
}

pub fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt().min(1.0)
}

#[derive(Debug, Clone)]
pub struct CircularAudioBuffer {
    buffer: Vec<f32>,
    capacity: usize,
}

impl CircularAudioBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push_slice(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);
        if self.buffer.len() > self.capacity {
            let excess = self.buffer.len() - self.capacity;
            self.buffer.drain(0..excess);
        }
    }

    pub fn get_last(&self, count: usize) -> Vec<f32> {
        let available = self.buffer.len();
        let take = count.min(available);
        self.buffer[available - take..].to_vec()
    }
}

/// Silence gate threshold for RMS audio conditioning in dBFS:
/// -45 dBFS = 10^(-45 / 20) ≈ 0.005623 linear
pub const SILENCE_GATE_RMS_THRESHOLD: f32 = 0.005623;

/// Maximum boost gain applied during audio conditioning (+12 dB = 10^(12 / 20) ≈ 3.981072 linear).
/// Prevents low-level background noise from being amplified to full scale.
pub const MAX_CONDITIONING_GAIN: f32 = 3.981072;

/// Target peak amplitude (-1.0 dBFS = 10^(-1.0 / 20) ≈ 0.89125 linear)
pub const TARGET_PEAK_AMPLITUDE: f32 = 0.89125;

/// Dynamic Gain Leveling & Soft RMS AGC:
/// - Discards segments below SILENCE_GATE_RMS_THRESHOLD as background noise/silence
/// - Normalizes peak amplitude towards TARGET_PEAK_AMPLITUDE, capped at MAX_CONDITIONING_GAIN (+12 dB)
/// - Preserves gentle soft tanh limiting above 0.70
pub fn condition_audio_for_asr(samples: &mut Vec<f32>) -> bool {
    if samples.is_empty() {
        return false;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();

    // Discard segments below silence threshold (default: -45 dBFS / 0.005623)
    if rms < SILENCE_GATE_RMS_THRESHOLD {
        return false;
    }

    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max_abs > 1e-4 {
        let uncapped_gain = TARGET_PEAK_AMPLITUDE / max_abs;
        // Cap maximum applied gain to +12 dB (3.981x) to prevent amplifying low-level noise
        let gain = uncapped_gain.min(MAX_CONDITIONING_GAIN);

        for s in samples.iter_mut() {
            let scaled = *s * gain;
            // Gentle tanh soft limiter above 0.70
            *s = if scaled.abs() > 0.70 {
                let sign = scaled.signum();
                let excess = scaled.abs() - 0.70;
                (0.70 + (excess * 1.5).tanh() * (TARGET_PEAK_AMPLITUDE - 0.70)) * sign
            } else {
                scaled
            };
        }
    }

    true
}

pub fn prepare_speech_segment(
    raw_samples: Vec<f32>,
    pre_roll: &[f32],
    clean_samples: Vec<f32>,
    id: u64,
    source: AudioSource,
    started_at_ms: u64,
) -> Option<SpeechSegment> {
    // 1. Prepend up to 250ms (4000 samples) of pre-roll
    let mut padded = Vec::with_capacity(pre_roll.len() + raw_samples.len());
    padded.extend_from_slice(pre_roll);
    padded.extend_from_slice(&raw_samples);

    // 2. Condition audio (discard if RMS < -45 dBFS, peak normalize with soft tanh AGC)
    if !condition_audio_for_asr(&mut padded) {
        return None;
    }

    let duration_ms = (padded.len() as u64 * 1000) / 16000;
    Some(SpeechSegment {
        id,
        source,
        samples_16k: padded,
        clean_samples_16k: clean_samples,
        started_at_ms,
        duration_ms,
    })
}

/// Fallback energy-based VAD state machine when ONNX Silero VAD is not loaded
struct FallbackVad {
    threshold: f32,
    is_speaking: bool,
    active_samples: Vec<f32>,
    silence_frames: usize,
    pre_roll: Vec<f32>,
    start_time_ms: u64,
}

impl FallbackVad {
    fn new(threshold: f32) -> Self {
        Self {
            threshold,
            is_speaking: false,
            active_samples: Vec::new(),
            silence_frames: 0,
            pre_roll: Vec::with_capacity(4800), // 300ms at 16k
            start_time_ms: 0,
        }
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn process(&mut self, chunk_16k: &[f32]) -> Option<(Vec<f32>, u64, u64)> {
        let rms = calculate_rms(chunk_16k);
        let speaking = rms >= self.threshold;

        if speaking {
            if !self.is_speaking {
                self.is_speaking = true;
                self.start_time_ms = Self::now_ms().saturating_sub(300);
                self.active_samples.clear();
                self.active_samples.extend_from_slice(&self.pre_roll);
            }
            self.active_samples.extend_from_slice(chunk_16k);
            self.silence_frames = 0;

            // Max segment duration: 12 seconds = 192,000 samples
            if self.active_samples.len() >= 192_000 {
                let duration_ms = (self.active_samples.len() as u64 * 1000) / 16000;
                let samples = std::mem::take(&mut self.active_samples);
                self.is_speaking = false;
                self.pre_roll.clear();
                return Some((samples, self.start_time_ms, duration_ms));
            }
        } else {
            // Keep 300ms pre-roll rolling buffer
            if !self.is_speaking {
                self.pre_roll.extend_from_slice(chunk_16k);
                if self.pre_roll.len() > 4800 {
                    let excess = self.pre_roll.len() - 4800;
                    self.pre_roll.drain(0..excess);
                }
            } else {
                self.active_samples.extend_from_slice(chunk_16k);
                self.silence_frames += 1;
                // 450ms silence hangover: at 32ms frames (512 samples), ~14 frames
                if self.silence_frames >= 14 {
                    self.is_speaking = false;
                    let duration_ms = (self.active_samples.len() as u64 * 1000) / 16000;
                    let samples = std::mem::take(&mut self.active_samples);
                    self.pre_roll.clear();
                    if duration_ms >= 250 {
                        return Some((samples, self.start_time_ms, duration_ms));
                    }
                }
            }
        }

        None
    }
}

pub fn start_dsp_vad_thread(
    mut mic_consumer: rtrb::Consumer<f32>,
    mut app_consumer: rtrb::Consumer<f32>,
    segment_sender: Sender<SpeechSegment>,
    app_handle: AppHandle,
    vad_model_path: PathBuf,
    pyannote_model_path: PathBuf,
    is_running: Arc<AtomicBool>,
    target_app_name: String,
    target_pid: Option<u32>,
) -> Result<thread::JoinHandle<()>> {
    let handle = thread::Builder::new()
        .name("dsp-vad-thread".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            info!("DSP & VAD Processing Thread started.");

            // Initialize Pyannote Powerset Segmenter if model exists
            let mut pyannote_segmenter = if pyannote_model_path.exists() {
                match PyannoteSegmenter::new(&pyannote_model_path) {
                    Ok(seg) => Some(seg),
                    Err(e) => {
                        warn!("Failed to load Pyannote Segmenter: {:?}", e);
                        None
                    }
                }
            } else {
                None
            };

            // Attempt to load Silero VAD from ONNX model
            let mut mic_silero = if vad_model_path.exists() {
                let cfg = SileroVadConfig {
                    model: vad_model_path.to_string_lossy().to_string(),
                    min_silence_duration: 0.45,
                    min_speech_duration: 0.25,
                    max_speech_duration: 12.0,
                    threshold: 0.5,
                    sample_rate: 16000,
                    window_size: 512,
                    num_threads: Some(1),
                    provider: Some("cpu".into()),
                    debug: false,
                };
                SileroVad::new(cfg, 30.0).map_err(|e| warn!("Failed to init Mic SileroVad: {:?}", e)).ok()
            } else {
                None
            };

            let mut app_silero = if vad_model_path.exists() {
                let cfg = SileroVadConfig {
                    model: vad_model_path.to_string_lossy().to_string(),
                    min_silence_duration: 0.45,
                    min_speech_duration: 0.25,
                    max_speech_duration: 12.0,
                    threshold: 0.5,
                    sample_rate: 16000,
                    window_size: 512,
                    num_threads: Some(1),
                    provider: Some("cpu".into()),
                    debug: false,
                };
                SileroVad::new(cfg, 30.0).map_err(|e| warn!("Failed to init App SileroVad: {:?}", e)).ok()
            } else {
                None
            };

            let mut mic_fallback = FallbackVad::new(0.015);
            let mut app_fallback = FallbackVad::new(0.015);

            let mut mic_pre_roll = CircularAudioBuffer::new(4800); // 300ms at 16kHz
            let mut app_pre_roll = CircularAudioBuffer::new(4800); // 300ms at 16kHz

            let mut mic_raw_buf: Vec<f32> = Vec::with_capacity(48000);
            let mut app_raw_buf: Vec<f32> = Vec::with_capacity(48000);

            let mut last_emit_time = Instant::now();
            let mut latest_mic_rms = 0.0f32;
            let mut latest_app_rms = 0.0f32;

            while is_running.load(Ordering::Relaxed) {
                // Sleep ~10ms to allow capture ring buffers to accumulate samples
                thread::sleep(Duration::from_millis(10));

                // 1. Drain Mic Consumer
                while let Ok(sample) = mic_consumer.pop() {
                    mic_raw_buf.push(sample);
                }

                // 2. Drain App Consumer
                while let Ok(sample) = app_consumer.pop() {
                    app_raw_buf.push(sample);
                }

                // Resample available mic samples (in multiples of 3 for 48k -> 16k)
                let mic_usable_len = (mic_raw_buf.len() / 3) * 3;
                let mic_16k = if mic_usable_len > 0 {
                    let chunk: Vec<f32> = mic_raw_buf.drain(0..mic_usable_len).collect();
                    decimate_48k_to_16k(&chunk)
                } else {
                    Vec::new()
                };

                // Resample available app samples
                let app_usable_len = (app_raw_buf.len() / 3) * 3;
                let app_16k = if app_usable_len > 0 {
                    let chunk: Vec<f32> = app_raw_buf.drain(0..app_usable_len).collect();
                    decimate_48k_to_16k(&chunk)
                } else {
                    Vec::new()
                };

                if !mic_16k.is_empty() {
                    mic_pre_roll.push_slice(&mic_16k);
                    let rms = calculate_rms(&mic_16k);
                    latest_mic_rms = latest_mic_rms * 0.7 + rms * 0.3;
                } else {
                    latest_mic_rms *= 0.9;
                }

                if !app_16k.is_empty() {
                    app_pre_roll.push_slice(&app_16k);
                    let rms = calculate_rms(&app_16k);
                    latest_app_rms = latest_app_rms * 0.7 + rms * 0.3;
                } else {
                    latest_app_rms *= 0.9;
                }

                let mut mic_speaking = false;
                let mut app_speaking = false;

                // Process Mic VAD
                if !mic_16k.is_empty() {
                    if let Some(ref mut vad) = mic_silero {
                        vad.accept_waveform(mic_16k.clone());
                        mic_speaking = vad.is_speech();
                        while !vad.is_empty() {
                            let seg = vad.front();
                            let duration_ms = (seg.samples.len() as u64 * 1000) / 16000;
                            if duration_ms >= 500 {
                                let id = NEXT_SEGMENT_ID.fetch_add(1, Ordering::SeqCst);
                                let started_at_ms = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis() as u64
                                    - duration_ms;

                                let pre = mic_pre_roll.get_last(4000); // 250ms pre-roll
                                let clean = seg.samples.clone();
                                if let Some(segment) = prepare_speech_segment(
                                    seg.samples,
                                    &pre,
                                    clean,
                                    id,
                                    AudioSource::Microphone,
                                    started_at_ms,
                                ) {
                                    let _ = segment_sender.send(segment);
                                }
                            }
                            vad.pop();
                        }
                    } else {
                        // Fallback energy VAD
                        mic_speaking = latest_mic_rms > 0.02;
                        if let Some((samples, started_at_ms, _duration_ms)) = mic_fallback.process(&mic_16k) {
                            let id = NEXT_SEGMENT_ID.fetch_add(1, Ordering::SeqCst);
                            let pre = mic_pre_roll.get_last(4000);
                            let clean = samples.clone();
                            if let Some(segment) = prepare_speech_segment(
                                samples,
                                &pre,
                                clean,
                                id,
                                AudioSource::Microphone,
                                started_at_ms,
                            ) {
                                let _ = segment_sender.send(segment);
                            }
                        }
                    }
                }

                // Process App VAD with Pyannote Powerset Neural Segmentation
                if !app_16k.is_empty() {
                    if let Some(ref mut vad) = app_silero {
                        vad.accept_waveform(app_16k.clone());
                        app_speaking = vad.is_speech();
                        while !vad.is_empty() {
                            let seg = vad.front();
                            let duration_ms = (seg.samples.len() as u64 * 1000) / 16000;
                            if duration_ms >= 500 {
                                let base_started_at_ms = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis() as u64
                                    - duration_ms;

                                let pre = app_pre_roll.get_last(4000); // 250ms pre-roll

                                if let Some(ref mut segmenter) = pyannote_segmenter {
                                    match segmenter.split_and_extract_clean(&seg.samples) {
                                        Ok(sub_segs) => {
                                            for (idx, sub) in sub_segs.into_iter().enumerate() {
                                                let sub_len = sub.end_sample - sub.start_sample;
                                                let sub_dur = (sub_len as u64 * 1000) / 16000;
                                                if sub_dur >= 500 {
                                                    let sub_id = NEXT_SEGMENT_ID.fetch_add(1, Ordering::SeqCst);
                                                    let sub_start = base_started_at_ms + (sub.start_sample as u64 * 1000) / 16000;
                                                    let sub_slice = seg.samples[sub.start_sample..sub.end_sample].to_vec();
                                                    let sub_pre = if idx == 0 { &pre[..] } else { &[][..] };

                                                    if let Some(segment) = prepare_speech_segment(
                                                        sub_slice,
                                                        sub_pre,
                                                        sub.clean_samples,
                                                        sub_id,
                                                        AudioSource::Application {
                                                            pid: target_pid,
                                                            app_name: target_app_name.clone(),
                                                        },
                                                        sub_start,
                                                    ) {
                                                        let _ = segment_sender.send(segment);
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            warn!("Pyannote segmentation error: {:?}", e);
                                            let id = NEXT_SEGMENT_ID.fetch_add(1, Ordering::SeqCst);
                                            let clean = seg.samples.clone();
                                            if let Some(segment) = prepare_speech_segment(
                                                seg.samples,
                                                &pre,
                                                clean,
                                                id,
                                                AudioSource::Application {
                                                    pid: target_pid,
                                                    app_name: target_app_name.clone(),
                                                },
                                                base_started_at_ms,
                                            ) {
                                                let _ = segment_sender.send(segment);
                                            }
                                        }
                                    }
                                } else {
                                    let id = NEXT_SEGMENT_ID.fetch_add(1, Ordering::SeqCst);
                                    let clean = seg.samples.clone();
                                    if let Some(segment) = prepare_speech_segment(
                                        seg.samples,
                                        &pre,
                                        clean,
                                        id,
                                        AudioSource::Application {
                                            pid: target_pid,
                                            app_name: target_app_name.clone(),
                                        },
                                        base_started_at_ms,
                                    ) {
                                        let _ = segment_sender.send(segment);
                                    }
                                }
                            }
                            vad.pop();
                        }
                    } else {
                        // Fallback energy VAD
                        app_speaking = latest_app_rms > 0.02;
                        if let Some((samples, started_at_ms, _duration_ms)) = app_fallback.process(&app_16k) {
                            let pre = app_pre_roll.get_last(4000);
                            if let Some(ref mut segmenter) = pyannote_segmenter {
                                match segmenter.split_and_extract_clean(&samples) {
                                    Ok(sub_segs) => {
                                        for (idx, sub) in sub_segs.into_iter().enumerate() {
                                            let sub_len = sub.end_sample - sub.start_sample;
                                            let sub_dur = (sub_len as u64 * 1000) / 16000;
                                            if sub_dur >= 500 {
                                                let sub_id = NEXT_SEGMENT_ID.fetch_add(1, Ordering::SeqCst);
                                                let sub_start = started_at_ms + (sub.start_sample as u64 * 1000) / 16000;
                                                let sub_slice = samples[sub.start_sample..sub.end_sample].to_vec();
                                                let sub_pre = if idx == 0 { &pre[..] } else { &[][..] };

                                                if let Some(segment) = prepare_speech_segment(
                                                    sub_slice,
                                                    sub_pre,
                                                    sub.clean_samples,
                                                    sub_id,
                                                    AudioSource::Application {
                                                        pid: target_pid,
                                                        app_name: target_app_name.clone(),
                                                    },
                                                    sub_start,
                                                ) {
                                                    let _ = segment_sender.send(segment);
                                                }
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        let id = NEXT_SEGMENT_ID.fetch_add(1, Ordering::SeqCst);
                                        let clean = samples.clone();
                                        if let Some(segment) = prepare_speech_segment(
                                            samples,
                                            &pre,
                                            clean,
                                            id,
                                            AudioSource::Application {
                                                pid: target_pid,
                                                app_name: target_app_name.clone(),
                                            },
                                            started_at_ms,
                                        ) {
                                            let _ = segment_sender.send(segment);
                                        }
                                    }
                                }
                            } else {
                                let id = NEXT_SEGMENT_ID.fetch_add(1, Ordering::SeqCst);
                                let clean = samples.clone();
                                if let Some(segment) = prepare_speech_segment(
                                    samples,
                                    &pre,
                                    clean,
                                    id,
                                    AudioSource::Application {
                                        pid: target_pid,
                                        app_name: target_app_name.clone(),
                                    },
                                    started_at_ms,
                                ) {
                                    let _ = segment_sender.send(segment);
                                }
                            }
                        }
                    }
                }

                // Emit audio-levels event at 15 Hz (~66ms)
                if last_emit_time.elapsed() >= Duration::from_millis(66) {
                    last_emit_time = Instant::now();
                    let levels = AudioLevels {
                        mic_rms: latest_mic_rms,
                        app_rms: latest_app_rms,
                        mic_speaking,
                        app_speaking,
                    };
                    let _ = app_handle.emit("audio-levels", &levels);
                }
            }

            info!("DSP & VAD Processing Thread stopped.");
        })?;

    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_audio_buffer() {
        let mut buf = CircularAudioBuffer::new(5);
        buf.push_slice(&[1.0, 2.0, 3.0]);
        let last2 = buf.get_last(2);
        assert_eq!(last2, vec![2.0, 3.0]);

        buf.push_slice(&[4.0, 5.0, 6.0]);
        // Buffer capacity is 5, so contents are [2.0, 3.0, 4.0, 5.0, 6.0]
        let last3 = buf.get_last(3);
        assert_eq!(last3, vec![4.0, 5.0, 6.0]);
        let all5 = buf.get_last(5);
        assert_eq!(all5, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_condition_audio_silence_discard() {
        // Very low amplitude (silence, RMS < -45 dBFS)
        let mut silence = vec![0.0001f32; 1600];
        let accepted = condition_audio_for_asr(&mut silence);
        assert!(!accepted, "Quiet silence must be discarded");
    }

    #[test]
    fn test_condition_audio_soft_agc_normalization() {
        // Moderate voice (peak around 0.20)
        let mut speech = Vec::new();
        for i in 0..1600 {
            speech.push(0.20 * (i as f32 * 0.1).sin());
        }
        let accepted = condition_audio_for_asr(&mut speech);
        assert!(accepted, "Speech should be retained and normalized");
        let max_amp = speech.iter().copied().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(max_amp > 0.7, "Peak amplitude should be boosted towards target level");
        assert!(max_amp <= 1.0, "Limiter must not allow clipping above 1.0");
    }

    #[test]
    fn test_condition_audio_gain_cap_at_12db() {
        // Very quiet noise segment just above silence threshold (peak 0.02, RMS ~ 0.014 > 0.0056)
        let mut quiet_noise = Vec::new();
        for i in 0..1600 {
            quiet_noise.push(0.02 * (i as f32 * 0.1).sin());
        }
        let accepted = condition_audio_for_asr(&mut quiet_noise);
        assert!(accepted, "Should pass above silence threshold");
        let max_amp = quiet_noise.iter().copied().fold(0.0f32, |m, s| m.max(s.abs()));
        // Max gain is capped at 3.981072 (+12 dB), so peak cannot exceed 0.02 * 3.981072 ≈ 0.0796
        assert!(max_amp <= 0.08, "Quiet audio gain must be capped at +12 dB, got max_amp: {}", max_amp);
    }
}
