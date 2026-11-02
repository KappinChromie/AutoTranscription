use anyhow::{Context, Result};
use ndarray::Array3;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;
use tracing::info;

pub const SAMPLES_PER_FRAME: usize = 270; // 16.875ms at 16kHz
pub const MIN_TURN_FLIP_FRAMES: usize = 18; // ~303ms at 16.875ms per frame

#[derive(Debug, Clone)]
pub struct SubSegment {
    pub start_sample: usize,
    pub end_sample: usize,
    pub dominant_speaker: usize,
    pub clean_samples: Vec<f32>,
}

pub struct PyannoteSegmenter {
    session: Session,
}

impl PyannoteSegmenter {
    pub fn new(model_path: &Path) -> Result<Self> {
        let env = ort::init().build()?;
        let session = Session::builder(&env)
            .map_err(|e| anyhow::anyhow!("Session builder error: {:?}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("Threads error: {:?}", e))?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load Pyannote ONNX model from {}", model_path.display()))?;

        info!("Pyannote Segmentation 3.0 model loaded successfully.");
        Ok(Self { session })
    }

    /// Run inference on a 16kHz mono audio buffer and return powerset-decoded frame outputs.
    /// Output is [num_frames, 7].
    pub fn infer_powerset(&mut self, samples_16k: &[f32]) -> Result<Vec<[f32; 7]>> {
        let num_samples = samples_16k.len();
        if num_samples < 800 {
            return Ok(Vec::new());
        }

        let array = Array3::from_shape_vec((1, 1, num_samples), samples_16k.to_vec())?;
        let tensor_ref = TensorRef::from_array_view(&array)?;
        let outputs = self.session.run(ort::inputs![tensor_ref])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let num_frames = data.len() / 7;
        let mut frames = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let offset = i * 7;
            let slice = &data[offset..offset + 7];

            // Pyannote 3.0 output is already LogSoftmax: convert log-probabilities to linear probabilities via exp()
            let mut probs = [0.0f32; 7];
            for c in 0..7 {
                probs[c] = slice[c].exp();
            }
            frames.push(probs);
        }

        Ok(frames)
    }

    /// Process audio and split at speaker change turns.
    /// Returns vector of SubSegment with audio slices and clean single-speaker samples.
    pub fn split_and_extract_clean(
        &mut self,
        samples_16k: &[f32],
    ) -> Result<Vec<SubSegment>> {
        let total_samples = samples_16k.len();
        if total_samples < 4000 {
            // Less than 250ms: return single segment
            return Ok(vec![SubSegment {
                start_sample: 0,
                end_sample: total_samples,
                dominant_speaker: 0,
                clean_samples: samples_16k.to_vec(),
            }]);
        }

        let frames = self.infer_powerset(samples_16k)?;
        let num_frames = frames.len();
        if num_frames == 0 {
            return Ok(vec![SubSegment {
                start_sample: 0,
                end_sample: total_samples,
                dominant_speaker: 0,
                clean_samples: samples_16k.to_vec(),
            }]);
        }

        // For each frame, decode to:
        // spk_0 = p[1] + p[4] + p[5]
        // spk_1 = p[2] + p[4] + p[6]
        // spk_2 = p[3] + p[5] + p[6]
        // is_overlap = (p[4] + p[5] + p[6] > 0.35) || count(spk_k > 0.5) > 1
        // dominant_speaker = argmax(spk_k) if spk_k > 0.5 and !is_overlap
        let mut frame_spk: Vec<Option<usize>> = Vec::with_capacity(num_frames);
        let mut frame_is_overlap: Vec<bool> = Vec::with_capacity(num_frames);

        for p in &frames {
            let spk0 = p[1] + p[4] + p[5];
            let spk1 = p[2] + p[4] + p[6];
            let spk2 = p[3] + p[5] + p[6];

            let active_count = (if spk0 > 0.5 { 1 } else { 0 })
                + (if spk1 > 0.5 { 1 } else { 0 })
                + (if spk2 > 0.5 { 1 } else { 0 });

            let overlap = (p[4] + p[5] + p[6] > 0.35) || active_count > 1;
            frame_is_overlap.push(overlap);

            if !overlap {
                let max_spk = if spk0 >= spk1 && spk0 >= spk2 {
                    if spk0 > 0.5 { Some(0) } else { None }
                } else if spk1 >= spk0 && spk1 >= spk2 {
                    if spk1 > 0.5 { Some(1) } else { None }
                } else {
                    if spk2 > 0.5 { Some(2) } else { None }
                };
                frame_spk.push(max_spk);
            } else {
                frame_spk.push(None);
            }
        }

        // Find turn switch points (where dominant speaker flips for >= 18 frames / ~300ms)
        let mut split_frame_indices: Vec<(usize, usize)> = Vec::new(); // (split_frame, current_speaker)
        let mut current_speaker: Option<usize> = None;

        let mut candidate_speaker: Option<usize> = None;
        let mut candidate_run_start = 0;
        let mut candidate_count = 0;

        for (f_idx, &spk) in frame_spk.iter().enumerate() {
            if let Some(s) = spk {
                if current_speaker.is_none() {
                    current_speaker = Some(s);
                } else if Some(s) != current_speaker {
                    // Possible speaker switch
                    if candidate_speaker == Some(s) {
                        candidate_count += 1;
                        if candidate_count >= MIN_TURN_FLIP_FRAMES {
                            // Turn flip confirmed!
                            let dominant = current_speaker.unwrap_or(0);
                            split_frame_indices.push((candidate_run_start, dominant));
                            current_speaker = Some(s);
                            candidate_speaker = None;
                            candidate_count = 0;
                        }
                    } else {
                        candidate_speaker = Some(s);
                        candidate_run_start = f_idx;
                        candidate_count = 1;
                    }
                } else {
                    // Same as current speaker
                    candidate_speaker = None;
                    candidate_count = 0;
                }
            }
        }

        // Build list of segments (from split_frame_indices)
        let mut segments = Vec::new();
        let mut last_sample_idx = 0;
        let mut last_frame_idx = 0;

        for (split_frame, speaker_id) in split_frame_indices {
            let split_sample = (split_frame * SAMPLES_PER_FRAME).min(total_samples);
            if split_sample > last_sample_idx + 3200 { // at least 200ms
                let clean = Self::collect_clean_samples(
                    samples_16k,
                    last_sample_idx,
                    split_sample,
                    last_frame_idx,
                    split_frame,
                    speaker_id,
                    &frame_spk,
                    &frame_is_overlap,
                );

                segments.push(SubSegment {
                    start_sample: last_sample_idx,
                    end_sample: split_sample,
                    dominant_speaker: speaker_id,
                    clean_samples: clean,
                });

                last_sample_idx = split_sample;
                last_frame_idx = split_frame;
            }
        }

        // Final remaining segment
        let remaining_speaker = current_speaker.unwrap_or(0);
        let clean = Self::collect_clean_samples(
            samples_16k,
            last_sample_idx,
            total_samples,
            last_frame_idx,
            num_frames,
            remaining_speaker,
            &frame_spk,
            &frame_is_overlap,
        );

        segments.push(SubSegment {
            start_sample: last_sample_idx,
            end_sample: total_samples,
            dominant_speaker: remaining_speaker,
            clean_samples: clean,
        });

        Ok(segments)
    }

    fn collect_clean_samples(
        full_audio: &[f32],
        start_samp: usize,
        end_samp: usize,
        start_f: usize,
        end_f: usize,
        target_spk: usize,
        frame_spk: &[Option<usize>],
        frame_is_overlap: &[bool],
    ) -> Vec<f32> {
        let mut clean = Vec::new();
        let total = full_audio.len();

        for f in start_f..end_f {
            if f < frame_spk.len() && !frame_is_overlap[f] && frame_spk[f] == Some(target_spk) {
                let f_start = (f * SAMPLES_PER_FRAME).min(total);
                let f_end = (f_start + SAMPLES_PER_FRAME).min(total);
                if f_start >= start_samp && f_end <= end_samp && f_start < f_end {
                    clean.extend_from_slice(&full_audio[f_start..f_end]);
                }
            }
        }

        // If clean samples are too sparse (< 4800 samples = 300ms), fallback to the non-overlapped part of the segment
        if clean.len() < 4800 {
            for f in start_f..end_f {
                if f < frame_is_overlap.len() && !frame_is_overlap[f] {
                    let f_start = (f * SAMPLES_PER_FRAME).min(total);
                    let f_end = (f_start + SAMPLES_PER_FRAME).min(total);
                    if f_start >= start_samp && f_end <= end_samp && f_start < f_end {
                        clean.extend_from_slice(&full_audio[f_start..f_end]);
                    }
                }
            }
        }

        // If still too short, fallback to the segment audio directly
        if clean.len() < 3200 {
            clean = full_audio[start_samp..end_samp].to_vec();
        }

        // Normalize clean audio
        normalize_clean_audio(&mut clean);

        clean
    }
}

pub fn normalize_clean_audio(samples: &mut Vec<f32>) {
    if samples.is_empty() {
        return;
    }

    // Check pre-normalized RMS energy
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();

    // Discard segments below -45 dBFS (0.0056)
    if rms < 0.0056 {
        samples.clear();
        return;
    }

    // Peak RMS normalization to -1.0 dBFS (peak = 0.89125)
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max_abs > 1e-4 {
        let target_peak = 0.89125f32; // -1.0 dBFS
        let factor = target_peak / max_abs;
        for s in samples.iter_mut() {
            *s = (*s * factor).clamp(-1.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_log_probs_to_linear_exp() {
        // True linear probabilities summing to 1.0
        let true_probs = [0.70f32, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03];
        let log_probs: Vec<f32> = true_probs.iter().map(|&p| p.ln()).collect();

        let mut linear_probs = [0.0f32; 7];
        for c in 0..7 {
            linear_probs[c] = log_probs[c].exp();
        }

        for c in 0..7 {
            assert!((linear_probs[c] - true_probs[c]).abs() < 1e-5);
        }
    }
}
