use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rubato::{Fft, FixedSync, Resampler};
use rubato::audioadapter_buffers::direct::InterleavedSlice;

// ============================================================================
// WAV & DSP Helpers (Independent of Tauri & DirectML)
// ============================================================================

const SILENCE_GATE_RMS_THRESHOLD: f32 = 0.005623413; // -45 dBFS
const MAX_LINEAR_GAIN: f32 = 3.981072; // +12 dB max gain

pub fn has_subword_loop(word: &str) -> bool {
    let c: Vec<char> = word.chars().collect();
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

pub fn clean_and_normalize_transcript(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }

    for word in &words {
        let cleaned: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if !cleaned.is_empty() && has_subword_loop(&cleaned) {
            return None;
        }
    }

    if words.len() >= 4 {
        let first = words[0].to_lowercase();
        if words.iter().all(|w| w.to_lowercase() == first) {
            return None;
        }
    }

    Some(trimmed.to_string())
}

pub fn condition_audio_for_asr(samples: &mut [f32]) -> bool {
    if samples.is_empty() {
        return false;
    }

    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();

    if rms < SILENCE_GATE_RMS_THRESHOLD {
        return false;
    }

    let max_peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let target_peak = 0.89125; // -1 dBFS
    let raw_gain = if max_peak > 0.001 {
        target_peak / max_peak
    } else {
        1.0
    };
    let gain = raw_gain.clamp(0.1, MAX_LINEAR_GAIN);

    for s in samples.iter_mut() {
        let amplified = *s * gain;
        *s = amplified.tanh();
    }

    true
}

pub struct StreamingResampler48kTo16k {
    resampler: Fft<f32>,
    in_frames: usize,
    out_frames: usize,
    chunk_out_buf: Vec<f32>,
}

impl StreamingResampler48kTo16k {
    pub fn new() -> Result<Self, String> {
        let resampler = Fft::<f32>::new(48000, 16000, 480, 1, FixedSync::Both)
            .map_err(|e| format!("Failed to create Rubato Fft resampler: {:#}", e))?;
        let in_frames = resampler.input_frames_next();
        let out_frames = resampler.output_frames_next();
        Ok(Self {
            resampler,
            in_frames,
            out_frames,
            chunk_out_buf: vec![0.0f32; out_frames],
        })
    }

    pub fn process_from_raw(&mut self, raw_48k: &mut Vec<f32>) -> Vec<f32> {
        let chunk_size = self.in_frames;
        let num_complete_chunks = raw_48k.len() / chunk_size;
        let mut output_16k = Vec::with_capacity(num_complete_chunks * self.out_frames);

        for i in 0..num_complete_chunks {
            let start = i * chunk_size;
            let end = start + chunk_size;
            let chunk_slice = &raw_48k[start..end];
            let in_adapter = match InterleavedSlice::new(chunk_slice, 1, self.in_frames) {
                Ok(a) => a,
                Err(_) => break,
            };
            let mut out_adapter = match InterleavedSlice::new_mut(&mut self.chunk_out_buf, 1, self.out_frames) {
                Ok(a) => a,
                Err(_) => break,
            };
            if let Ok((_nin, nout)) = self.resampler.process_into_buffer(&in_adapter, &mut out_adapter, None) {
                output_16k.extend_from_slice(&self.chunk_out_buf[..nout]);
            }
        }
        output_16k
    }
}

pub fn write_wav_16k_mono(path: &Path, samples: &[f32]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    let num_samples = samples.len() as u32;
    let byte_rate: u32 = 16000 * 2;
    let block_align = 2u16;
    let bits_per_sample = 16u16;
    let data_chunk_size = num_samples * 2;
    let chunk_size = 36 + data_chunk_size;

    file.write_all(b"RIFF")?;
    file.write_all(&chunk_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&16000u32.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    file.write_all(b"data")?;
    file.write_all(&data_chunk_size.to_le_bytes())?;
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let pcm_16 = (clamped * 32767.0) as i16;
        file.write_all(&pcm_16.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open WAV file: {:#}", e))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| format!("Failed to read WAV file: {:#}", e))?;

    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("Invalid WAV file (missing RIFF/WAVE header)".into());
    }

    let mut pos = 12;
    let mut channels = 1u16;
    let mut sample_rate = 16000u32;
    let mut bits_per_sample = 16u16;
    let mut audio_format = 1u16;
    let mut audio_bytes = &[][..];

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_len = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let chunk_data_start = pos + 8;
        let chunk_data_end = (chunk_data_start + chunk_len).min(data.len());

        if chunk_id == b"fmt " && chunk_len >= 16 {
            audio_format = u16::from_le_bytes(data[chunk_data_start..chunk_data_start + 2].try_into().unwrap());
            channels = u16::from_le_bytes(data[chunk_data_start + 2..chunk_data_start + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(data[chunk_data_start + 4..chunk_data_start + 8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(data[chunk_data_start + 14..chunk_data_start + 16].try_into().unwrap());
        } else if chunk_id == b"data" {
            audio_bytes = &data[chunk_data_start..chunk_data_end];
            break;
        }

        pos = chunk_data_end;
        if chunk_len % 2 != 0 {
            pos += 1;
        }
    }

    if audio_bytes.is_empty() {
        return Err("No data chunk found in WAV".into());
    }

    let mut samples = Vec::new();
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let frame_size = bytes_per_sample * (channels as usize);

    for frame in audio_bytes.chunks_exact(frame_size) {
        let mut sample_val = 0.0f32;
        for ch in 0..(channels as usize) {
            let offset = ch * bytes_per_sample;
            let ch_val = if audio_format == 1 {
                if bits_per_sample == 16 {
                    let s = i16::from_le_bytes(frame[offset..offset + 2].try_into().unwrap());
                    s as f32 / 32768.0
                } else if bits_per_sample == 32 {
                    let s = i32::from_le_bytes(frame[offset..offset + 4].try_into().unwrap());
                    s as f32 / 2147483648.0
                } else {
                    0.0
                }
            } else if audio_format == 3 && bits_per_sample == 32 {
                f32::from_le_bytes(frame[offset..offset + 4].try_into().unwrap())
            } else {
                0.0
            };
            sample_val += ch_val;
        }
        samples.push(sample_val / channels as f32);
    }

    Ok((samples, sample_rate))
}

fn segment_audio_energy(samples_16k: &[f32]) -> Vec<(usize, usize, Vec<f32>)> {
    let frame_len = 1600; // 100ms frames
    let energy_threshold = SILENCE_GATE_RMS_THRESHOLD;
    let mut speech_frames: Vec<bool> = Vec::new();

    for chunk in samples_16k.chunks(frame_len) {
        let sum_sq: f32 = chunk.iter().map(|&s| s * s).sum();
        let rms = (sum_sq / chunk.len() as f32).sqrt();
        speech_frames.push(rms >= energy_threshold);
    }

    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut start_frame = 0;
    let mut silence_count = 0;
    let hangover_frames = 5; // 500ms hangover

    for (i, &is_speech) in speech_frames.iter().enumerate() {
        if is_speech {
            if !in_speech {
                in_speech = true;
                start_frame = i.saturating_sub(2); // 200ms pre-roll
            }
            silence_count = 0;
        } else if in_speech {
            silence_count += 1;
            if silence_count >= hangover_frames || i == speech_frames.len() - 1 {
                let end_frame = i;
                let start_idx = start_frame * frame_len;
                let end_idx = (end_frame * frame_len).min(samples_16k.len());

                let mut seg_audio = samples_16k[start_idx..end_idx].to_vec();
                if condition_audio_for_asr(&mut seg_audio) {
                    segments.push((start_idx, end_idx, seg_audio));
                }
                in_speech = false;
                silence_count = 0;
            }
        }
    }

    if segments.is_empty() && !samples_16k.is_empty() {
        let mut full = samples_16k.to_vec();
        if condition_audio_for_asr(&mut full) {
            segments.push((0, samples_16k.len(), full));
        }
    }

    segments
}

// ============================================================================
// Model Discovery & Inference Wrapper
// ============================================================================

fn resolve_models_dir() -> PathBuf {
    if let Ok(env_dir) = std::env::var("TRANSCRIBER_MODELS_DIR") {
        let p = PathBuf::from(env_dir);
        if p.exists() {
            return p;
        }
    }
    let local = PathBuf::from("models");
    if local.exists() {
        return local.canonicalize().unwrap_or(local);
    }
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()));
    if let Some(d) = exe_dir {
        let m = d.join("models");
        if m.exists() {
            return m;
        }
        let m_up = d.join("../models");
        if m_up.exists() {
            return m_up;
        }
    }
    PathBuf::from("models")
}

fn find_model_files(models_dir: &Path, variant: &str) -> Option<(PathBuf, PathBuf, Option<PathBuf>, PathBuf, bool)> {
    if variant == "parakeet-tdt" {
        let canonical = models_dir.join("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8");
        let dir = if canonical.exists() { canonical } else { models_dir.to_path_buf() };
        let mut enc = None;
        let mut dec = None;
        let mut join = None;
        let mut tok = None;

        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                let name = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                if name.contains("encoder") && name.ends_with(".onnx") {
                    enc = Some(p);
                } else if name.contains("decoder") && name.ends_with(".onnx") {
                    dec = Some(p);
                } else if name.contains("joiner") && name.ends_with(".onnx") {
                    join = Some(p);
                } else if name.contains("tokens") && name.ends_with(".txt") {
                    tok = Some(p);
                }
            }
        }
        if let (Some(e), Some(d), Some(j), Some(t)) = (enc, dec, join, tok) {
            return Some((e, d, Some(j), t, true));
        }
    } else {
        // Whisper variants
        let mut target_dir = models_dir.to_path_buf();
        if let Ok(entries) = fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    let dname = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                    if dname.contains(variant) || (variant == "turbo" && dname.contains("turbo")) {
                        target_dir = p;
                        break;
                    }
                }
            }
        }
        let mut enc = None;
        let mut dec = None;
        let mut tok = None;

        if let Ok(entries) = fs::read_dir(&target_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                let name = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                if name.contains("encoder") && name.ends_with(".onnx") {
                    enc = Some(p);
                } else if name.contains("decoder") && name.ends_with(".onnx") {
                    dec = Some(p);
                } else if name.contains("tokens") && name.ends_with(".txt") {
                    tok = Some(p);
                }
            }
        }
        if let (Some(e), Some(d), Some(t)) = (enc, dec, tok) {
            return Some((e, d, None, t, false));
        }
    }
    None
}

struct StandaloneRecognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
    pub actual_provider: String,
}

impl Drop for StandaloneRecognizer {
    fn drop(&mut self) {
        if !self.recognizer.is_null() {
            unsafe {
                sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
            }
        }
    }
}

impl StandaloneRecognizer {
    pub fn new(
        encoder: &Path,
        decoder: &Path,
        joiner: Option<&Path>,
        tokens: &Path,
        is_transducer: bool,
        provider: &str,
    ) -> Result<Self, String> {
        let encoder_cstr = std::ffi::CString::new(encoder.to_string_lossy().as_bytes()).unwrap();
        let decoder_cstr = std::ffi::CString::new(decoder.to_string_lossy().as_bytes()).unwrap();
        let tokens_cstr = std::ffi::CString::new(tokens.to_string_lossy().as_bytes()).unwrap();
        let provider_cstr = std::ffi::CString::new(provider).unwrap();
        let decoding_method = std::ffi::CString::new("greedy_search").unwrap();
        let empty_cstr = std::ffi::CString::new("").unwrap();

        let (whisper_cfg, transducer_cfg, model_type_cstr, feat_dim) = if is_transducer {
            let join = joiner.ok_or("Joiner required for transducer")?;
            let join_cstr = std::ffi::CString::new(join.to_string_lossy().as_bytes()).unwrap();
            let trans = sherpa_rs_sys::SherpaOnnxOfflineTransducerModelConfig {
                encoder: encoder_cstr.as_ptr(),
                decoder: decoder_cstr.as_ptr(),
                joiner: join_cstr.into_raw(),
            };
            (
                unsafe { std::mem::zeroed::<sherpa_rs_sys::SherpaOnnxOfflineWhisperModelConfig>() },
                trans,
                std::ffi::CString::new("nemo_transducer").unwrap(),
                80,
            )
        } else {
            let lang = std::ffi::CString::new("en").unwrap();
            let task = std::ffi::CString::new("transcribe").unwrap();
            let whisp = sherpa_rs_sys::SherpaOnnxOfflineWhisperModelConfig {
                encoder: encoder_cstr.as_ptr(),
                decoder: decoder_cstr.as_ptr(),
                language: lang.into_raw(),
                task: task.into_raw(),
                tail_paddings: 0,
            };
            (
                whisp,
                unsafe { std::mem::zeroed::<sherpa_rs_sys::SherpaOnnxOfflineTransducerModelConfig>() },
                empty_cstr.clone(),
                512,
            )
        };

        let model_cfg = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                whisper: whisper_cfg,
                transducer: transducer_cfg,
                debug: 0,
                num_threads: 4,
                provider: provider_cstr.as_ptr(),
                tokens: tokens_cstr.as_ptr(),
                bpe_vocab: empty_cstr.as_ptr(),
                model_type: model_type_cstr.as_ptr(),
                modeling_unit: empty_cstr.as_ptr(),
                nemo_ctc: std::mem::zeroed(),
                paraformer: std::mem::zeroed(),
                tdnn: std::mem::zeroed(),
                telespeech_ctc: std::mem::zeroed(),
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

        let offline_cfg = unsafe {
            sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: feat_dim,
                },
                model_config: model_cfg,
                decoding_method: decoding_method.as_ptr(),
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
        let mut recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&offline_cfg) };
        if recognizer.is_null() && provider != "cpu" {
            println!("Execution provider '{}' failed to initialize! Falling back to 'cpu'.", provider);
            let cpu_cstr = std::ffi::CString::new("cpu").unwrap();
            let mut cpu_model_cfg = model_cfg;
            cpu_model_cfg.provider = cpu_cstr.as_ptr();
            let mut cpu_rec_cfg = offline_cfg;
            cpu_rec_cfg.model_config = cpu_model_cfg;
            recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&cpu_rec_cfg) };
            if !recognizer.is_null() {
                actual_provider = "cpu".to_string();
            }
        }

        if recognizer.is_null() {
            return Err("Failed to create Sherpa recognizer (check model files & provider)".into());
        }

        Ok(Self { recognizer, actual_provider })
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> String {
        unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            if stream.is_null() {
                return String::new();
            }

            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                16000,
                samples.as_ptr(),
                samples.len() as i32,
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

// ============================================================================
// CLI Entry Point
// ============================================================================

fn print_usage() {
    println!("Usage: ab_compare <input_wav_path> [options]");
    println!();
    println!("Options:");
    println!("  --dump-segments <dir>    Dump each conditioned 16kHz speech segment as WAV to <dir>");
    println!("  --output <file>          Output side-by-side comparison report text file (default: ab_comparison_report.txt)");
    println!("  --provider <cpu|cuda>    Execution provider (default: cpu)");
    println!("  --help                   Show this help message");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.contains(&"--help".to_string()) {
        print_usage();
        return;
    }

    let input_wav = &args[1];
    let mut dump_dir: Option<PathBuf> = None;
    let mut report_path = PathBuf::from("ab_comparison_report.txt");
    let mut provider = "cpu".to_string();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--dump-segments" => {
                if i + 1 < args.len() {
                    dump_dir = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    report_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--provider" => {
                if i + 1 < args.len() {
                    provider = args[i + 1].clone();
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    println!("================================================================================");
    println!("              A/B TRANSCRIPTION PIPELINE COMPARISON HARNESS                     ");
    println!("================================================================================");
    println!("Input WAV:      {}", input_wav);
    println!("Provider:       {}", provider);
    if let Some(ref d) = dump_dir {
        println!("Dump Segments:  {}", d.display());
    }
    println!("Output Report:  {}", report_path.display());
    println!("--------------------------------------------------------------------------------");

    // 1. Load Audio
    let (raw_samples, sample_rate) = match read_wav(Path::new(input_wav)) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Error loading WAV: {}", e);
            return;
        }
    };

    let duration_secs = raw_samples.len() as f32 / sample_rate as f32;
    println!("Loaded audio: {:.2}s ({} samples at {} Hz)", duration_secs, raw_samples.len(), sample_rate);

    // 2. Resample to 16kHz if needed
    let samples_16k = if sample_rate == 48000 {
        println!("Resampling 48kHz -> 16kHz via Rubato StreamingResampler...");
        let mut resampler = StreamingResampler48kTo16k::new().expect("Failed to initialize Rubato resampler");
        let mut raw = raw_samples;
        resampler.process_from_raw(&mut raw)
    } else if sample_rate == 16000 {
        raw_samples
    } else {
        eprintln!("Unsupported sample rate: {} Hz (expected 16000 or 48000)", sample_rate);
        return;
    };

    println!("Conditioning and segmenting audio...");
    let segments = segment_audio_energy(&samples_16k);
    println!("Identified {} speech segment(s).", segments.len());

    if let Some(ref d) = dump_dir {
        fs::create_dir_all(d).expect("Failed to create segment dump directory");
        for (idx, (_, _, seg_audio)) in segments.iter().enumerate() {
            let seg_file = d.join(format!("segment_{:03}.wav", idx + 1));
            write_wav_16k_mono(&seg_file, seg_audio).expect("Failed to write dumped segment WAV");
        }
        println!("Dumped {} post-conditioned 16kHz segment WAV files to {}", segments.len(), d.display());
    }

    // 3. Resolve Models
    let models_dir = resolve_models_dir();
    println!("Models directory: {}", models_dir.display());

    let mut recognizers: Vec<(String, StandaloneRecognizer)> = Vec::new();

    // Check Parakeet TDT
    if let Some((enc, dec, join, tok, is_transducer)) = find_model_files(&models_dir, "parakeet-tdt") {
        println!("Found Parakeet TDT model files. Initializing...");
        match StandaloneRecognizer::new(&enc, &dec, join.as_deref(), &tok, is_transducer, &provider) {
            Ok(rec) => {
                let name = format!("Parakeet-TDT-0.6B ({})", rec.actual_provider);
                recognizers.push((name, rec));
            }
            Err(e) => eprintln!("Failed to initialize Parakeet-TDT: {}", e),
        }
    }

    // Check Whisper Variants
    for variant in &["tiny.en", "base.en", "small.en", "distil-small.en", "distil-large-v3", "large-v3-turbo", "turbo"] {
        if let Some((enc, dec, join, tok, is_transducer)) = find_model_files(&models_dir, variant) {
            let label = format!("Whisper-{}", variant);
            if !recognizers.iter().any(|(n, _)| n.starts_with(&label)) {
                println!("Found {} model files. Initializing...", label);
                match StandaloneRecognizer::new(&enc, &dec, join.as_deref(), &tok, is_transducer, &provider) {
                    Ok(rec) => {
                        let name = format!("{} ({})", label, rec.actual_provider);
                        recognizers.push((name, rec));
                    }
                    Err(e) => eprintln!("Failed to initialize {}: {}", variant, e),
                }
            }
        }
    }

    if recognizers.is_empty() {
        eprintln!("No models could be loaded from {}. Download models in the UI or check path.", models_dir.display());
        return;
    }

    println!("Loaded {} model(s) for comparison.", recognizers.len());
    println!("================================================================================");

    let report_file = File::create(&report_path).expect("Failed to create report output file");
    let mut writer = BufWriter::new(report_file);

    writeln!(writer, "A/B MODEL COMPARISON REPORT").unwrap();
    writeln!(writer, "Audio File: {}", input_wav).unwrap();
    writeln!(writer, "Duration: {:.2}s", duration_secs).unwrap();
    writeln!(writer, "Total Segments: {}", segments.len()).unwrap();
    writeln!(writer, "Provider: {}", provider).unwrap();
    writeln!(writer, "================================================================================").unwrap();

    let mut model_total_infer_ms = vec![0u64; recognizers.len()];
    let total_audio_ms = (duration_secs * 1000.0) as u64;

    for (seg_idx, (start_idx, end_idx, audio)) in segments.iter().enumerate() {
        let seg_start_s = *start_idx as f32 / 16000.0;
        let seg_end_s = *end_idx as f32 / 16000.0;
        let dur_ms = (audio.len() as u64 * 1000) / 16000;

        println!("\n--- Segment {:02} [{:.2}s - {:.2}s] (Duration: {}ms) ---", seg_idx + 1, seg_start_s, seg_end_s, dur_ms);
        writeln!(writer, "\n--- Segment {:02} [{:.2}s - {:.2}s] (Duration: {}ms) ---", seg_idx + 1, seg_start_s, seg_end_s, dur_ms).unwrap();

        for (m_idx, (name, recognizer)) in recognizers.iter_mut().enumerate() {
            let t0 = Instant::now();
            let raw_text = recognizer.transcribe(audio);
            let infer_ms = t0.elapsed().as_millis() as u64;
            model_total_infer_ms[m_idx] += infer_ms;
            let rtf = infer_ms as f32 / dur_ms.max(1) as f32;

            let clean_text = clean_and_normalize_transcript(&raw_text);
            let display_text = clean_text.unwrap_or_else(|| "[REJECTED BY REPETITION/SILENCE FILTER]".into());

            println!("  [{}] RTF: {:.3} (Inference: {}ms)", name, rtf, infer_ms);
            println!("    Text: \"{}\"", display_text);

            writeln!(writer, "  [{}] RTF: {:.3} (Inference: {}ms)", name, rtf, infer_ms).unwrap();
            writeln!(writer, "    \"{}\"", display_text).unwrap();
        }
    }

    println!("\n================================================================================");
    println!("                           SUMMARY BENCHMARK RESULTS                            ");
    println!("================================================================================");
    writeln!(writer, "\n================================================================================").unwrap();
    writeln!(writer, "                           SUMMARY BENCHMARK RESULTS                            ").unwrap();
    writeln!(writer, "================================================================================").unwrap();

    for (m_idx, (name, _)) in recognizers.iter().enumerate() {
        let total_infer = model_total_infer_ms[m_idx];
        let overall_rtf = total_infer as f32 / total_audio_ms.max(1) as f32;
        let summary_line = format!(
            "{:<25} | Total Infer: {:>6}ms | Audio: {:>6}ms | Overall RTF: {:.3}",
            name, total_infer, total_audio_ms, overall_rtf
        );
        println!("{}", summary_line);
        writeln!(writer, "{}", summary_line).unwrap();
    }

    println!("================================================================================");
    println!("Comparison report written to: {}", report_path.display());
}
