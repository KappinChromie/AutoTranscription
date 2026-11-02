use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

const COLOR_PALETTE: &[&str] = &[
    "#3b82f6", // Blue
    "#8b5cf6", // Purple
    "#f59e0b", // Amber
    "#ec4899", // Pink
    "#06b6d4", // Cyan
    "#f97316", // Orange
    "#a855f7", // Violet
    "#14b8a6", // Teal
    "#6366f1", // Indigo
    "#e11d48", // Rose
    "#84cc16", // Lime
];

pub const MIN_NEW_SPEAKER_SPEECH_MS: u64 = 4000; // 4 seconds accumulated clean speech
pub const MIN_EMBEDDING_DURATION_MS: u64 = 1000; // 1 second minimum for new candidates
pub const AHC_DISTANCE_THRESHOLD: f32 = 0.48; // ~0.52 cosine similarity (kodama dissimilarity: 1.0 - sim)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerProfile {
    pub id: String,
    pub display_name: String,
    pub centroid: Vec<f32>,
    pub sample_count: u32,
    pub total_speech_ms: u64,
    pub color_hex: String,
    pub locked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerMatch {
    pub speaker_id: String,
    pub display_name: String,
    pub color_hex: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptRevision {
    pub id: u64,
    pub speaker_id: String,
    pub speaker_name: String,
    pub speaker_color: String,
}

#[derive(Debug, Clone)]
pub struct PendingCandidate {
    pub candidate_id: usize,
    pub temp_name: String,
    pub centroid: Vec<f32>,
    pub accumulated_clean_speech_ms: u64,
    pub segment_ids: Vec<u64>,
    pub color_hex: String,
}

#[derive(Debug, Clone)]
pub struct HistorySegment {
    pub segment_id: u64,
    pub embedding: Vec<f32>,
    pub duration_ms: u64,
    pub assigned_speaker_id: String,
}

pub fn l2_normalize(vec: &mut [f32]) {
    let norm = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-6 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// Disjoint Set (Union-Find) for Dendrogram cutting
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a != root_b {
            self.parent[root_b] = root_a;
        }
    }
}

pub struct SpeakerRegistry {
    profiles: RwLock<Vec<SpeakerProfile>>,
    pending_pool: RwLock<Vec<PendingCandidate>>,
    history_segments: RwLock<Vec<HistorySegment>>,
    candidate_counter: RwLock<usize>,
    speaker_counter: RwLock<usize>,
    similarity_threshold: RwLock<f32>,
    last_remote_speaker: RwLock<Option<(String, Instant)>>,
}

impl SpeakerRegistry {
    pub fn new(threshold: f32) -> Self {
        let me_profile = SpeakerProfile {
            id: "me".into(),
            display_name: "Me".into(),
            centroid: Vec::new(),
            sample_count: 0,
            total_speech_ms: 0,
            color_hex: "#10b981".into(), // Emerald green
            locked: true,
        };

        Self {
            profiles: RwLock::new(vec![me_profile]),
            pending_pool: RwLock::new(Vec::new()),
            history_segments: RwLock::new(Vec::with_capacity(300)),
            candidate_counter: RwLock::new(0),
            speaker_counter: RwLock::new(0),
            similarity_threshold: RwLock::new(threshold.clamp(0.45, 0.80)),
            last_remote_speaker: RwLock::new(None),
        }
    }

    pub fn set_threshold(&self, threshold: f32) {
        *self.similarity_threshold.write() = threshold.clamp(0.45, 0.80);
    }

    pub fn get_threshold(&self) -> f32 {
        *self.similarity_threshold.read()
    }

    pub fn get_profiles(&self) -> Vec<SpeakerProfile> {
        self.profiles.read().clone()
    }

    pub fn lock_speaker(&self, speaker_id: &str, locked: bool) -> Option<Vec<SpeakerProfile>> {
        let mut profiles = self.profiles.write();
        if let Some(p) = profiles.iter_mut().find(|p| p.id == speaker_id) {
            p.locked = locked;
            Some(profiles.clone())
        } else {
            None
        }
    }

    /// Match or enroll incoming Track B speech segment with 4-second probation
    pub fn match_or_enroll(
        &self,
        segment_id: u64,
        mut embedding_opt: Option<Vec<f32>>,
        clean_duration_ms: u64,
    ) -> (SpeakerMatch, Vec<TranscriptRevision>) {
        let mut revisions = Vec::new();

        let embedding = match embedding_opt.as_mut() {
            Some(emb) => {
                l2_normalize(emb);
                emb
            }
            None => {
                // No clean embedding: assign unknown or best guess if only 1 remote profile exists
                let profiles = self.profiles.read();
                let remote_profiles: Vec<_> = profiles.iter().filter(|p| p.id != "me").collect();
                if remote_profiles.len() == 1 {
                    let p = remote_profiles[0];
                    return (
                        SpeakerMatch {
                            speaker_id: p.id.clone(),
                            display_name: p.display_name.clone(),
                            color_hex: p.color_hex.clone(),
                            confidence: 0.6,
                        },
                        revisions,
                    );
                }
                return (
                    SpeakerMatch {
                        speaker_id: "unknown".into(),
                        display_name: "Unknown / Noise".into(),
                        color_hex: "#94a3b8".into(),
                        confidence: 0.3,
                    },
                    revisions,
                );
            }
        };

        let threshold = *self.similarity_threshold.read();
        let mut profiles = self.profiles.write();

        // Conversational Recency Continuity Bonus:
        // If a confirmed remote speaker spoke within the last 4.0 seconds, grant a modest +0.05
        // similarity boost to that speaker to prevent momentary acoustic variations from fracturing turns.
        let last_remote = self.last_remote_speaker.read().clone();

        // 1. Check similarity against all confirmed remote profiles
        let mut best_confirmed_idx = None;
        let mut best_confirmed_effective_sim = -1.0f32;
        let mut best_confirmed_raw_sim = -1.0f32;

        for (idx, p) in profiles.iter().enumerate() {
            if p.id == "me" || p.centroid.is_empty() {
                continue;
            }
            let raw_sim = cosine_similarity(embedding, &p.centroid);
            let is_recent = if let Some((ref r_id, r_time)) = last_remote {
                r_id == &p.id && r_time.elapsed() < Duration::from_millis(4000)
            } else {
                false
            };
            let effective_sim = if is_recent {
                (raw_sim + 0.05).min(1.0)
            } else {
                raw_sim
            };

            if effective_sim > best_confirmed_effective_sim {
                best_confirmed_effective_sim = effective_sim;
                best_confirmed_raw_sim = raw_sim;
                best_confirmed_idx = Some(idx);
            }
        }

        // Matched an existing confirmed speaker profile
        if let Some(idx) = best_confirmed_idx {
            if best_confirmed_effective_sim >= threshold {
                let profile = &mut profiles[idx];
                profile.sample_count += 1;
                profile.total_speech_ms += clean_duration_ms;

                // Centroid Update Rule (EMA update only if segment >= 1200ms and not locked)
                if clean_duration_ms >= 1200 && !profile.locked {
                    let alpha = 0.85f32;
                    for (c, &e) in profile.centroid.iter_mut().zip(embedding.iter()) {
                        *c = alpha * (*c) + (1.0 - alpha) * e;
                    }
                    l2_normalize(&mut profile.centroid);
                }

                let speaker_id = profile.id.clone();
                let display_name = profile.display_name.clone();
                let color_hex = profile.color_hex.clone();
                *self.last_remote_speaker.write() = Some((speaker_id.clone(), Instant::now()));
                drop(profiles);

                // Add to history segments for rolling AHC
                if clean_duration_ms >= 1000 {
                    let mut history = self.history_segments.write();
                    if history.len() >= 250 {
                        history.remove(0);
                    }
                    history.push(HistorySegment {
                        segment_id,
                        embedding: embedding.clone(),
                        duration_ms: clean_duration_ms,
                        assigned_speaker_id: speaker_id.clone(),
                    });
                }

                return (
                    SpeakerMatch {
                        speaker_id,
                        display_name,
                        color_hex,
                        confidence: best_confirmed_raw_sim.clamp(0.0, 1.0),
                    },
                    revisions,
                );
            }
        }

        // 2. Utterance did NOT match confirmed speakers:
        // Short utterance (< 1000ms): do not spawn new speaker
        if clean_duration_ms < MIN_EMBEDDING_DURATION_MS {
            // First check if raw similarity is >= 0.48 to best confirmed profile
            if let Some(idx) = best_confirmed_idx {
                if best_confirmed_raw_sim >= 0.48 {
                    let profile = &profiles[idx];
                    *self.last_remote_speaker.write() = Some((profile.id.clone(), Instant::now()));
                    return (
                        SpeakerMatch {
                            speaker_id: profile.id.clone(),
                            display_name: profile.display_name.clone(),
                            color_hex: profile.color_hex.clone(),
                            confidence: best_confirmed_raw_sim.max(0.4),
                        },
                        revisions,
                    );
                }
            }

            // Next check if a remote speaker spoke in the last 3.5s (conversational reply/interjection)
            if let Some((ref r_id, r_time)) = last_remote {
                if r_time.elapsed() < Duration::from_millis(3500) {
                    if let Some(profile) = profiles.iter().find(|p| &p.id == r_id) {
                        return (
                            SpeakerMatch {
                                speaker_id: profile.id.clone(),
                                display_name: profile.display_name.clone(),
                                color_hex: profile.color_hex.clone(),
                                confidence: 0.5,
                            },
                            revisions,
                        );
                    }
                }
            }

            return (
                SpeakerMatch {
                    speaker_id: "unknown".into(),
                    display_name: "Unknown / Short".into(),
                    color_hex: "#94a3b8".into(),
                    confidence: 0.4,
                },
                revisions,
            );
        }

        // 3. Utterance >= 1000ms: Voice Probation in PendingSpeakerPool
        let mut pending_pool = self.pending_pool.write();

        // Check against existing pending candidates
        let mut best_pending_idx = None;
        let mut best_pending_sim = -1.0f32;

        for (idx, cand) in pending_pool.iter().enumerate() {
            let sim = cosine_similarity(embedding, &cand.centroid);
            if sim > best_pending_sim {
                best_pending_sim = sim;
                best_pending_idx = Some(idx);
            }
        }

        if let Some(cand_idx) = best_pending_idx {
            if best_pending_sim >= 0.58 {
                let cand = &mut pending_pool[cand_idx];
                cand.accumulated_clean_speech_ms += clean_duration_ms;
                cand.segment_ids.push(segment_id);

                // Update candidate centroid
                let alpha = 0.80f32;
                for (c, &e) in cand.centroid.iter_mut().zip(embedding.iter()) {
                    *c = alpha * (*c) + (1.0 - alpha) * e;
                }
                l2_normalize(&mut cand.centroid);

                let accumulated_ms = cand.accumulated_clean_speech_ms;
                let cand_centroid = cand.centroid.clone();
                let cand_segment_ids = cand.segment_ids.clone();
                let cand_color = cand.color_hex.clone();
                let cand_temp_id = format!("candidate_{}", cand.candidate_id);
                let cand_temp_name = cand.temp_name.clone();

                // Check promotion threshold: >= 4000ms accumulated clean speech
                if accumulated_ms >= MIN_NEW_SPEAKER_SPEECH_MS {
                    // PRE-PROMOTION RE-CHECK: Before minting Person N, verify if candidate centroid
                    // actually matches an existing confirmed profile with relaxed threshold (>= 0.54)
                    let mut matched_confirmed_idx = None;
                    let mut matched_confirmed_sim = -1.0f32;

                    for (p_idx, p) in profiles.iter().enumerate() {
                        if p.id == "me" || p.centroid.is_empty() {
                            continue;
                        }
                        let sim = cosine_similarity(&cand_centroid, &p.centroid);
                        if sim > matched_confirmed_sim {
                            matched_confirmed_sim = sim;
                            matched_confirmed_idx = Some(p_idx);
                        }
                    }

                    if let Some(p_idx) = matched_confirmed_idx {
                        if matched_confirmed_sim >= 0.54 {
                            let profile = &mut profiles[p_idx];
                            profile.sample_count += cand_segment_ids.len() as u32;
                            profile.total_speech_ms += accumulated_ms;

                            // Update confirmed centroid with accumulated candidate centroid
                            if !profile.locked {
                                let alpha = 0.70f32;
                                for (c, &e) in profile.centroid.iter_mut().zip(cand_centroid.iter()) {
                                    *c = alpha * (*c) + (1.0 - alpha) * e;
                                }
                                l2_normalize(&mut profile.centroid);
                            }

                            let id = profile.id.clone();
                            let display_name = profile.display_name.clone();
                            let color = profile.color_hex.clone();

                            // Generate retroactive revisions for probation segments
                            for &prev_seg_id in &cand_segment_ids {
                                revisions.push(TranscriptRevision {
                                    id: prev_seg_id,
                                    speaker_id: id.clone(),
                                    speaker_name: display_name.clone(),
                                    speaker_color: color.clone(),
                                });
                            }

                            *self.last_remote_speaker.write() = Some((id.clone(), Instant::now()));
                            pending_pool.remove(cand_idx);

                            info!(
                                "Pre-promotion: Merged candidate into existing profile {} ({}) (similarity: {:.2})",
                                display_name, id, matched_confirmed_sim
                            );

                            return (
                                SpeakerMatch {
                                    speaker_id: id,
                                    display_name,
                                    color_hex: color,
                                    confidence: matched_confirmed_sim,
                                },
                                revisions,
                            );
                        }
                    }

                    // Utterance is genuinely a new speaker: PROMOTE TO CONFIRMED SPEAKER!
                    let mut counter = self.speaker_counter.write();
                    *counter += 1;
                    let num = *counter;

                    let color_idx = (num - 1) % COLOR_PALETTE.len();
                    let color = COLOR_PALETTE[color_idx].to_string();
                    let id = format!("speaker_{}", num);
                    let display_name = format!("Person {}", num);

                    let new_profile = SpeakerProfile {
                        id: id.clone(),
                        display_name: display_name.clone(),
                        centroid: cand_centroid,
                        sample_count: cand_segment_ids.len() as u32,
                        total_speech_ms: accumulated_ms,
                        color_hex: color.clone(),
                        locked: false,
                    };

                    profiles.push(new_profile);

                    // Generate retroactive transcript revisions for all segments of this promoted candidate
                    for &prev_seg_id in &cand_segment_ids {
                        revisions.push(TranscriptRevision {
                            id: prev_seg_id,
                            speaker_id: id.clone(),
                            speaker_name: display_name.clone(),
                            speaker_color: color.clone(),
                        });
                    }

                    *self.last_remote_speaker.write() = Some((id.clone(), Instant::now()));
                    pending_pool.remove(cand_idx);

                    info!(
                        "Promoted candidate to confirmed profile: {} ({}) with {}ms speech",
                        display_name, id, accumulated_ms
                    );

                    return (
                        SpeakerMatch {
                            speaker_id: id,
                            display_name,
                            color_hex: color,
                            confidence: 1.0,
                        },
                        revisions,
                    );
                } else {
                    // Still in probation
                    return (
                        SpeakerMatch {
                            speaker_id: cand_temp_id,
                            display_name: cand_temp_name,
                            color_hex: cand_color,
                            confidence: best_pending_sim,
                        },
                        revisions,
                    );
                }
            }
        }

        // New candidate entering probation
        let mut cand_counter = self.candidate_counter.write();
        *cand_counter += 1;
        let c_num = *cand_counter;

        let color_idx = (c_num - 1) % COLOR_PALETTE.len();
        let color = COLOR_PALETTE[color_idx].to_string();
        let temp_name = format!("Speaker Candidate {}", c_num);

        let new_candidate = PendingCandidate {
            candidate_id: c_num,
            temp_name: temp_name.clone(),
            centroid: embedding.clone(),
            accumulated_clean_speech_ms: clean_duration_ms,
            segment_ids: vec![segment_id],
            color_hex: color.clone(),
        };

        pending_pool.push(new_candidate);

        (
            SpeakerMatch {
                speaker_id: format!("candidate_{}", c_num),
                display_name: temp_name,
                color_hex: color,
                confidence: 0.6,
            },
            revisions,
        )
    }

    /// Rolling Agglomerative Hierarchical Clustering (AHC) using kodama
    pub fn run_rolling_ahc(&self) -> Vec<TranscriptRevision> {
        let mut profiles = self.profiles.write();
        let mut history = self.history_segments.write();
        let mut revisions = Vec::new();

        // 1. Auto-consolidate any duplicate confirmed profiles that drifted close (similarity >= 0.58)
        self.consolidate_duplicate_profiles_internal(&mut profiles, &mut history, &mut revisions);

        let n = history.len();
        if n < 6 {
            return revisions;
        }

        // Build condensed distance vector: length n*(n-1)/2
        let condensed_len = n * (n - 1) / 2;
        let mut condensed = Vec::with_capacity(condensed_len);

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(&history[i].embedding, &history[j].embedding);
                let dist = (1.0 - sim).clamp(0.0, 2.0);
                condensed.push(dist);
            }
        }

        // Run kodama average linkage
        let dendrogram = kodama::linkage(&mut condensed, n, kodama::Method::Average);

        // Cut dendrogram at AHC_DISTANCE_THRESHOLD using UnionFind
        let mut uf = UnionFind::new(2 * n);
        for (step_idx, step) in dendrogram.steps().iter().enumerate() {
            let new_node = n + step_idx;
            if step.dissimilarity <= AHC_DISTANCE_THRESHOLD {
                uf.union(step.cluster1, step.cluster2);
                uf.union(step.cluster1, new_node);
            }
        }

        // Group observations by cluster ID
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let cid = uf.find(i);
            clusters.entry(cid).or_default().push(i);
        }

        for (_cid, obs_indices) in clusters {
            if obs_indices.is_empty() {
                continue;
            }

            // Compute cluster centroid
            let emb_dim = history[obs_indices[0]].embedding.len();
            let mut cluster_centroid = vec![0.0f32; emb_dim];
            let mut total_duration = 0u64;

            for &idx in &obs_indices {
                total_duration += history[idx].duration_ms;
                for (c, &val) in cluster_centroid.iter_mut().zip(history[idx].embedding.iter()) {
                    *c += val;
                }
            }
            l2_normalize(&mut cluster_centroid);

            // Check if cluster is anchored to a locked speaker profile
            let mut target_speaker: Option<(String, String, String)> = None;

            for &idx in &obs_indices {
                let assigned = &history[idx].assigned_speaker_id;
                if let Some(locked_prof) = profiles.iter().find(|p| p.id == *assigned && p.locked) {
                    target_speaker = Some((
                        locked_prof.id.clone(),
                        locked_prof.display_name.clone(),
                        locked_prof.color_hex.clone(),
                    ));
                    break;
                }
            }

            // If not anchored to a locked profile, find best matching confirmed profile
            if target_speaker.is_none() {
                let mut best_sim = -1.0f32;
                let mut best_prof = None;

                for p in profiles.iter() {
                    if p.id == "me" || p.centroid.is_empty() {
                        continue;
                    }
                    let sim = cosine_similarity(&cluster_centroid, &p.centroid);
                    if sim > best_sim {
                        best_sim = sim;
                        best_prof = Some(p);
                    }
                }

                if let Some(p) = best_prof {
                    if best_sim >= 0.52 {
                        target_speaker = Some((p.id.clone(), p.display_name.clone(), p.color_hex.clone()));
                    } else if best_sim >= 0.45 && total_duration < 4000 {
                        // Fold micro-cluster into nearest confirmed cluster
                        target_speaker = Some((p.id.clone(), p.display_name.clone(), p.color_hex.clone()));
                    }
                }
            }

            // If target speaker resolved, verify if any segment needs revision
            if let Some((spk_id, spk_name, spk_color)) = target_speaker {
                for &idx in &obs_indices {
                    if history[idx].assigned_speaker_id != spk_id {
                        history[idx].assigned_speaker_id = spk_id.clone();
                        revisions.push(TranscriptRevision {
                            id: history[idx].segment_id,
                            speaker_id: spk_id.clone(),
                            speaker_name: spk_name.clone(),
                            speaker_color: spk_color.clone(),
                        });
                    }
                }
            }
        }

        revisions
    }

    /// Internal helper to consolidate duplicate speaker profiles that have drifted close
    fn consolidate_duplicate_profiles_internal(
        &self,
        profiles: &mut Vec<SpeakerProfile>,
        history: &mut Vec<HistorySegment>,
        revisions: &mut Vec<TranscriptRevision>,
    ) {
        let n = profiles.len();
        if n <= 2 {
            return;
        }

        let mut merge_pair = None;
        for i in 1..n {
            if profiles[i].locked || profiles[i].centroid.is_empty() {
                continue;
            }
            for j in (i + 1)..n {
                if profiles[j].locked || profiles[j].centroid.is_empty() {
                    continue;
                }
                let sim = cosine_similarity(&profiles[i].centroid, &profiles[j].centroid);
                if sim >= 0.58 {
                    merge_pair = Some((i, j, sim));
                    break;
                }
            }
            if merge_pair.is_some() {
                break;
            }
        }

        if let Some((tgt_idx, src_idx, sim)) = merge_pair {
            let src_id = profiles[src_idx].id.clone();
            let tgt_id = profiles[tgt_idx].id.clone();
            let tgt_name = profiles[tgt_idx].display_name.clone();
            let tgt_color = profiles[tgt_idx].color_hex.clone();

            info!(
                "Auto-consolidating duplicate profile {} ({}) into {} ({}) (similarity: {:.2})",
                profiles[src_idx].display_name, src_id, tgt_name, tgt_id, sim
            );

            let src_count = profiles[src_idx].sample_count as f32;
            let tgt_count = profiles[tgt_idx].sample_count as f32;
            let total_count = src_count + tgt_count;
            let src_speech = profiles[src_idx].total_speech_ms;
            let src_centroid = profiles[src_idx].centroid.clone();

            let tgt_profile = &mut profiles[tgt_idx];
            if total_count > 0.0 && !src_centroid.is_empty() && !tgt_profile.centroid.is_empty() {
                for (t, s) in tgt_profile.centroid.iter_mut().zip(src_centroid.iter()) {
                    *t = (*t * tgt_count + *s * src_count) / total_count;
                }
                l2_normalize(&mut tgt_profile.centroid);
            }
            tgt_profile.sample_count = (total_count as u32).max(1);
            tgt_profile.total_speech_ms += src_speech;

            // Remove source profile
            profiles.remove(src_idx);

            // Update history segments
            for seg in history.iter_mut() {
                if seg.assigned_speaker_id == src_id {
                    seg.assigned_speaker_id = tgt_id.clone();
                    revisions.push(TranscriptRevision {
                        id: seg.segment_id,
                        speaker_id: tgt_id.clone(),
                        speaker_name: tgt_name.clone(),
                        speaker_color: tgt_color.clone(),
                    });
                }
            }
        }
    }

    pub fn rename_speaker(&self, speaker_id: &str, new_name: &str) -> Option<Vec<SpeakerProfile>> {
        let mut profiles = self.profiles.write();
        if let Some(p) = profiles.iter_mut().find(|p| p.id == speaker_id) {
            p.display_name = new_name.trim().to_string();
            Some(profiles.clone())
        } else {
            None
        }
    }

    pub fn merge_speakers(&self, source_id: &str, target_id: &str) -> Option<Vec<SpeakerProfile>> {
        if source_id == target_id || source_id == "me" || target_id == "me" {
            return None;
        }

        let mut profiles = self.profiles.write();
        let src_idx = profiles.iter().position(|p| p.id == source_id)?;
        let tgt_idx = profiles.iter().position(|p| p.id == target_id)?;

        let src_count = profiles[src_idx].sample_count as f32;
        let tgt_count = profiles[tgt_idx].sample_count as f32;
        let src_speech = profiles[src_idx].total_speech_ms;
        let total_count = src_count + tgt_count;

        let src_centroid = profiles[src_idx].centroid.clone();
        let tgt_profile = &mut profiles[tgt_idx];

        // Weighted centroid average
        if total_count > 0.0 && !src_centroid.is_empty() && !tgt_profile.centroid.is_empty() {
            for (t, s) in tgt_profile.centroid.iter_mut().zip(src_centroid.iter()) {
                *t = (*t * tgt_count + *s * src_count) / total_count;
            }
            l2_normalize(&mut tgt_profile.centroid);
        }
        tgt_profile.sample_count = (total_count as u32).max(1);
        tgt_profile.total_speech_ms += src_speech;

        // Remove source profile
        profiles.remove(src_idx);
        Some(profiles.clone())
    }

    pub fn clear(&self) {
        let mut profiles = self.profiles.write();
        profiles.clear();
        profiles.push(SpeakerProfile {
            id: "me".into(),
            display_name: "Me".into(),
            centroid: Vec::new(),
            sample_count: 0,
            total_speech_ms: 0,
            color_hex: "#10b981".into(),
            locked: true,
        });
        self.pending_pool.write().clear();
        self.history_segments.write().clear();
        *self.candidate_counter.write() = 0;
        *self.speaker_counter.write() = 0;
        *self.last_remote_speaker.write() = None;
    }
}

pub struct Diarizer {
    extractor: Option<EmbeddingExtractor>,
    registry: Arc<SpeakerRegistry>,
}

impl Diarizer {
    pub fn new(model_path: &Path, registry: Arc<SpeakerRegistry>, provider: Option<&str>) -> Self {
        let prov = provider.unwrap_or("cpu");
        let extractor = if model_path.exists() {
            let config = ExtractorConfig {
                model: model_path.to_string_lossy().to_string(),
                provider: Some(prov.into()),
                num_threads: Some(1),
                debug: false,
            };
            match EmbeddingExtractor::new(config) {
                Ok(ext) => {
                    info!("Speaker Embedding Extractor loaded successfully with provider '{}'.", prov);
                    Some(ext)
                }
                Err(e) => {
                    warn!("Failed to load speaker embedding model with provider '{}': {:?}", prov, e);
                    if prov != "cpu" {
                        let fallback_config = ExtractorConfig {
                            model: model_path.to_string_lossy().to_string(),
                            provider: Some("cpu".into()),
                            num_threads: Some(1),
                            debug: false,
                        };
                        EmbeddingExtractor::new(fallback_config).ok()
                    } else {
                        None
                    }
                }
            }
        } else {
            None
        };

        Self { extractor, registry }
    }

    /// Extract speaker embedding with 1.5s sliding sub-windows (0.75s hop)
    pub fn extract_embedding(&mut self, clean_samples: &[f32]) -> Option<Vec<f32>> {
        let ext = self.extractor.as_mut()?;
        if clean_samples.len() < 8000 {
            // Less than 0.5s: too short for reliable embedding
            return None;
        }

        // Sub-windowing parameters: 1.5s window = 24,000 samples, 0.75s hop = 12,000 samples
        const SUB_WINDOW_LEN: usize = 24_000;
        const SUB_WINDOW_HOP: usize = 12_000;

        if clean_samples.len() >= 28_800 { // >= 1.8s
            let mut sub_embeddings = Vec::new();
            let mut start = 0;

            while start + SUB_WINDOW_LEN <= clean_samples.len() {
                let window = clean_samples[start..start + SUB_WINDOW_LEN].to_vec();
                if let Ok(mut emb) = ext.compute_speaker_embedding(window, 16000) {
                    l2_normalize(&mut emb);
                    sub_embeddings.push(emb);
                }
                start += SUB_WINDOW_HOP;
            }

            if !sub_embeddings.is_empty() {
                let dim = sub_embeddings[0].len();
                let mut mean_emb = vec![0.0f32; dim];
                for emb in &sub_embeddings {
                    for (m, &e) in mean_emb.iter_mut().zip(emb.iter()) {
                        *m += e;
                    }
                }
                l2_normalize(&mut mean_emb);
                return Some(mean_emb);
            }
        }

        // Direct single embedding for samples between 0.5s and 1.8s
        if let Ok(mut emb) = ext.compute_speaker_embedding(clean_samples.to_vec(), 16000) {
            l2_normalize(&mut emb);
            return Some(emb);
        }

        None
    }

    pub fn process_track_b(
        &mut self,
        segment_id: u64,
        clean_samples: &[f32],
        clean_duration_ms: u64,
    ) -> (SpeakerMatch, Vec<TranscriptRevision>) {
        let embedding = self.extract_embedding(clean_samples);
        self.registry.match_or_enroll(segment_id, embedding, clean_duration_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controlled_vector(dim: usize, range: std::ops::Range<usize>) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        for i in range {
            v[i] = 1.0;
        }
        l2_normalize(&mut v);
        v
    }

    #[test]
    fn test_recency_bias_and_continuity() {
        let registry = SpeakerRegistry::new(0.58);
        let emb1 = controlled_vector(200, 0..100);

        // 1. Promote first candidate by accumulating 4000ms
        let (match1, _) = registry.match_or_enroll(1, Some(emb1.clone()), 2000);
        assert!(match1.speaker_id.starts_with("candidate_"));

        let (match2, revs) = registry.match_or_enroll(2, Some(emb1.clone()), 2000);
        assert_eq!(match2.speaker_id, "speaker_1");
        assert_eq!(revs.len(), 2);

        // 2. A vector with 55 overlapping elements (raw sim = 0.55 < 0.58)
        let mut drifted = vec![0.0f32; 200];
        for i in 0..55 {
            drifted[i] = 1.0;
        }
        for i in 100..145 {
            drifted[i] = 1.0;
        }
        l2_normalize(&mut drifted);
        let raw_sim = cosine_similarity(&emb1, &drifted);
        assert!((raw_sim - 0.55).abs() < 0.01);

        // Should match speaker_1 due to recency continuity boost (+0.05 brings effective sim to 0.60 >= 0.58)
        let (match3, _) = registry.match_or_enroll(3, Some(drifted), 1500);
        assert_eq!(match3.speaker_id, "speaker_1");
    }

    #[test]
    fn test_pre_promotion_folds_into_confirmed() {
        let registry = SpeakerRegistry::new(0.58);
        let emb1 = controlled_vector(200, 0..100);

        // Enroll speaker_1 with 4000ms
        registry.match_or_enroll(1, Some(emb1.clone()), 4000);

        // Clear recency so turn 2 does not get the +0.05 recency boost
        *registry.last_remote_speaker.write() = None;

        // Now simulate a turn with raw similarity = 0.55.
        // Since raw sim 0.55 < 0.58 and no recency boost, it enters probation!
        let mut emb2 = vec![0.0f32; 200];
        for i in 0..55 {
            emb2[i] = 1.0;
        }
        for i in 100..145 {
            emb2[i] = 1.0;
        }
        l2_normalize(&mut emb2);
        let raw_sim = cosine_similarity(&emb1, &emb2);
        assert!((raw_sim - 0.55).abs() < 0.01);

        let (cand_match1, _) = registry.match_or_enroll(2, Some(emb2.clone()), 2000);
        assert!(cand_match1.speaker_id.starts_with("candidate_"));

        // After accumulating another 2000ms (total 4000ms), candidate centroid has sim = 0.55 >= 0.54 against speaker_1:
        // Pre-promotion should fold candidate into speaker_1 instead of creating speaker_2!
        let (cand_match2, revs) = registry.match_or_enroll(3, Some(emb2.clone()), 2000);
        assert_eq!(cand_match2.speaker_id, "speaker_1");
        assert_eq!(registry.get_profiles().len(), 2); // "me" + "speaker_1" (NO speaker_2!)
        assert!(!revs.is_empty());
    }

    #[test]
    fn test_auto_consolidation_merges_duplicate_profiles() {
        let registry = SpeakerRegistry::new(0.58);
        let emb1 = controlled_vector(200, 0..100);
        // Vector with 65 overlapping elements (sim = 0.65 >= 0.58)
        let mut emb2 = vec![0.0f32; 200];
        for i in 0..65 {
            emb2[i] = 1.0;
        }
        for i in 100..135 {
            emb2[i] = 1.0;
        }
        l2_normalize(&mut emb2);

        // Force two profiles into registry
        {
            let mut profiles = registry.profiles.write();
            profiles.push(SpeakerProfile {
                id: "speaker_1".into(),
                display_name: "Person 1".into(),
                centroid: emb1,
                sample_count: 5,
                total_speech_ms: 5000,
                color_hex: "#3b82f6".into(),
                locked: false,
            });
            profiles.push(SpeakerProfile {
                id: "speaker_2".into(),
                display_name: "Person 2".into(),
                centroid: emb2,
                sample_count: 4,
                total_speech_ms: 4000,
                color_hex: "#8b5cf6".into(),
                locked: false,
            });
        }

        assert_eq!(registry.get_profiles().len(), 3); // "me" + "speaker_1" + "speaker_2"

        // Running rolling AHC triggers auto-consolidation of duplicate profiles
        registry.run_rolling_ahc();

        // One duplicate should be consolidated into the other
        assert_eq!(registry.get_profiles().len(), 2); // "me" + 1 consolidated speaker!
    }
}
