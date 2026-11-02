# Dual-Stream Live Transcription & Speaker Diarization App

Native Windows desktop application in **Rust + Tauri v2** that captures two independent audio streams simultaneously in real time:
1. **Track A (Hardware Microphone):** Captures physical microphone via Windows WASAPI, detects speech via Voice Activity Detection (VAD), transcribes locally, and tags speaker as **`Me`** (zero speaker-identification overhead).
2. **Track B (Target Application / System Loopback):** Captures audio from a user-selected application process (`AUDIOCLIENT_ACTIVATION_PARAMS` / `AudioClient::new_application_loopback_client` with `include_tree = true`) or global desktop audio loopback. Extracts ONNX speaker embeddings, runs online cosine-similarity clustering (`Person 1`, `Person 2`), transcribes speech, and merges everything into a live transcript feed.

---

## Architecture & Threading Model

```
[Track A: WASAPI Mic Thread] ──(rtrb SPSC)──┐
                                            ▼
                                  [DSP & VAD Thread]
                                  (48kHz->16kHz Decimation, Silero VAD)
                                            ▲
[Track B: WASAPI App Thread] ──(rtrb SPSC)──┘
                                            │
                       Emits SpeechSegment { source, samples_16k, timestamp }
                                            │ (crossbeam channel)
                                            ▼
                            [ML Inference & Diarization Worker]
                             ├─ If Source == Mic:
                             │    1. Speaker = "Me" (id: "me", confidence: 1.0)
                             │    2. Local Whisper ASR
                             └─ If Source == Application(PID):
                                  1. Extract ONNX Speaker Embedding
                                  2. Match/Create Cluster in SpeakerRegistry
                                  3. Local Whisper ASR
                                            │
                       Emits Tauri Event: "transcript-entry"
                                            ▼
                                [Tauri v2 Frontend UI]
```

### Strict Real-Time Audio Guarantees
- No resampling, VAD, ONNX inference, Whisper transcription, or mutex locking inside WASAPI capture threads.
- WASAPI capture threads only read hardware packet buffers and push raw `f32` samples into lock-free SPSC ring buffers (`rtrb`).
- No non-`Send` COM pointers cross thread boundaries.

---

## Build & Release Artifacts

- Standalone Executable: `src-tauri/target/release/live-transcriber.exe`
- NSIS Setup Installer: `src-tauri/target/release/bundle/nsis/DualStream Transcriber_0.1.0_x64-setup.exe`
- MSI Installer: `src-tauri/target/release/bundle/msi/DualStream Transcriber_0.1.0_x64_en-US.msi`

### Commands
```powershell
# Development mode
cargo tauri dev

# Production build
cargo tauri build
```
