# AutoTranscription

[![Built with Tauri](https://img.shields.io/badge/Built%20with-Tauri%20v2-24C8D8.svg)](https://tauri.app/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue.svg)](https://microsoft.com)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-green.svg)](#license)

**AutoTranscription** is a high-performance, real-time, privacy-first Windows desktop application for live audio transcription and speaker diarization. Built with **Rust** and **Tauri v2**, it captures and transcribes two independent audio streams simultaneously (your physical microphone and any target application's audio output) with local machine learning models running on CPU or GPU.

---

## Key Highlights

- **Dual-Stream Isolation (Microphone + Application Loopback):**
  - **Track A (Hardware Microphone):** Captures your physical microphone via low-latency Windows WASAPI, detects speech via Silero VAD, and immediately attributes text to **`Me`** without any speaker-identification latency or ambiguity.
  - **Track B (Target Application / System Loopback):** Captures audio directly from a selected running application process (e.g., Discord, Zoom, Microsoft Teams, Chrome) using Windows WASAPI process-loopback activation (`AUDIOCLIENT_ACTIVATION_PARAMS`), or from global desktop audio.
- **Local Speaker Diarization:**
  - Extracts speaker embeddings in real time using ONNX models (CAM++ / 3D-Speaker).
  - Clusters incoming application voices dynamically using Online Agglomerative Hierarchical Clustering (`Person 1`, `Person 2`, etc.).
  - Rename, merge, or lock speaker profiles directly in the UI.
- **Multiple State-of-the-Art ASR Models:**
  - **OpenAI Whisper variants:** `tiny.en`, `base.en`, `small.en`, `distil-small.en`, `distil-large-v3`, `large-v3-turbo` running via `sherpa-onnx`.
  - **NVIDIA Parakeet TDT 0.6B (Int8 Fast-Conformer):** Offline NeMo transducer path with blazing-fast Real-Time Factor (RTF $< 0.10$ on CPU).
- **Hardened DSP & Audio Conditioning:**
  - **Rubato 5.0 Streaming FFT Resampler:** Ultra-clean 48 kHz $\to$ 16 kHz conversion with continuous filter state across chunks (zero clicks or boundary artifacts).
  - **Gain-Capped Conditioning (+12 dB AGC Limit):** Eliminates Whisper noise hallucinations by preventing low-level room noise/fan hum from being amplified into full-scale noise.
  - **Silence Gating ($-45\text{ dBFS}$):** Drops non-speech frames before inference.
  - **Character & Subword Repetition Filters:** Rejects repetition hallucination loops (e.g., `"pdpdpdpd"`, `"aaaa"`).
- **Flexible Hardware Acceleration:**
  - Automatic detection and support for **CPU**, **DirectML** (DirectX 12 on any modern GPU), and **CUDA** with transparent fallback and UI status indicators.
- **Built-in Model Downloader & A/B Benchmark Harness:**
  - Download models directly inside the application with live percentage progress tracking.
  - Standalone `ab_compare` CLI tool to benchmark different models on audio files side-by-side.

---

## Architecture & Threading Model

The application enforces strict real-time audio constraints to prevent buffer underruns, UI stalls, or audio glitches:

```
[Track A: WASAPI Mic Thread] ────────(rtrb lock-free SPSC)───────┐
                                                                 ▼
                                                       [DSP & VAD Worker]
                                                       ├─ Rubato 48k->16k Resampling
                                                       ├─ AGC (+12 dB cap) & Silence Gate
                                                       └─ Silero VAD Segmentation
                                                                 ▲
[Track B: WASAPI App Loopback Thread] ─(rtrb lock-free SPSC)─────┘
                                                                 │
                                           Emits SpeechSegment { source, samples_16k, timestamp }
                                                                 │ (crossbeam-channel)
                                                                 ▼
                                                  [ASR & Diarization Worker]
                                                   ├─ If Source == Mic:
                                                   │    1. Speaker = "Me" (confidence: 1.0)
                                                   │    2. Sherpa Whisper / Parakeet ASR
                                                   └─ If Source == Application:
                                                        1. Extract ONNX Speaker Embedding
                                                        2. Match / Create Cluster in Registry
                                                        3. Sherpa Whisper / Parakeet ASR
                                                                 │
                                                       Emits "transcript-entry"
                                                                 ▼
                                                       [Tauri v2 Frontend UI]
```

- **Lock-Free Capture:** WASAPI capture threads never lock mutexes, allocate memory, or perform inference; audio samples are pushed immediately to `rtrb` single-producer single-consumer ring buffers.
- **Background Workers:** VAD segmentation, model inference, and clustering execute asynchronously on dedicated worker threads.

---

## Prerequisites

Before building or running the project, make sure the following are installed on Windows:

1. **Rust Toolchain (1.75 or later):**
   - Install via [rustup.rs](https://rustup.rs/):
     ```powershell
     winget install Rustlang.Rustup
     # Or run the rustup-init installer and select default MSVC toolchain
     ```
2. **Node.js & npm (v18 or later):**
   - Download from [nodejs.org](https://nodejs.org/) or install via winget:
     ```powershell
     winget install OpenJS.NodeJS
     ```
3. **C++ Build Tools for Visual Studio:**
   - Visual Studio 2022 Community or Build Tools with the **Desktop development with C++** workload and the **Windows 10/11 SDK** selected.
4. **GPU Drivers (Optional, for GPU acceleration):**
   - NVIDIA GPU: Current NVIDIA Game Ready / Studio drivers (CUDA/DirectML).
   - AMD / Intel GPU: Current vendor drivers with DirectX 12 support (DirectML).

---

## Installation & Setup

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/KappinChromie/AutoTranscription.git
   cd AutoTranscription
   ```

2. **Install frontend dependencies:**
   ```powershell
   npm install
   ```

3. **Verify the Rust environment:**
   ```powershell
   cargo check --manifest-path src-tauri/Cargo.toml
   ```

---

## Running the Application

### Development Mode

Run the frontend dev server and the Tauri desktop app with live reload:
```powershell
npm run tauri dev
```

### Production Build

To build an optimized, standalone Windows installer and executable:
```powershell
npm run tauri build
```
The output artifacts will be located in:
- **Standalone executable:** `src-tauri/target/release/live-transcriber.exe`
- **NSIS Setup installer:** `src-tauri/target/release/bundle/nsis/DualStream Transcriber_*.exe`
- **MSI installer:** `src-tauri/target/release/bundle/msi/DualStream Transcriber_*.msi`

---

## Model Setup

AutoTranscription uses local models stored in a `models/` directory.

### In-App Download (Recommended)
Launch the application. On first startup, click the **Download** button in the top navigation or settings drawer to fetch the required base models:
- **Silero VAD** (Voice Activity Detection)
- **3D-Speaker / CAM++** (Speaker Embedding Extraction)
- **Default Whisper Model** (`whisper-base.en` or `whisper-large-v3-turbo`)
- **NVIDIA Parakeet TDT 0.6B** (Optional ultra-fast alternative)

### Custom Models Directory
By default, the application looks for `models/` in the workspace or application data folder. You can override this location using an environment variable:
```powershell
$env:TRANSCRIBER_MODELS_DIR = "D:\AI\speech_models"
npm run tauri dev
```

---

## Benchmarking & Diagnostics

### A/B Comparison CLI Harness (`ab_compare`)
AutoTranscription includes a standalone benchmarking tool to test and compare models on any recorded WAV file:

```powershell
# Run side-by-side comparison on a WAV file:
cargo run --manifest-path src-tauri/Cargo.toml --bin ab_compare -- "path\to\recording.wav" [options]

# Options:
#   --dump-segments <dir>    Dump each conditioned 16kHz speech segment as WAV to <dir>
#   --output <file>          Output side-by-side comparison report text file (default: ab_comparison_report.txt)
#   --provider <cpu|cuda>    Execution provider (default: cpu)
#   --help                   Show help message
```

### Live Audio Segment Dumper
To inspect the exact post-conditioned 16 kHz audio segments fed into the speech recognizer during a live session:
```powershell
$env:DUMP_SPEECH_SEGMENTS_DIR = "C:\transcription_debug_segments"
npm run tauri dev
```
Segments will be automatically written as `segment_0001_mic.wav`, `segment_0002_app.wav`, etc.

---

## Project Structure

```
AutoTranscription/
├── index.html                 # Main frontend HTML template
├── package.json               # Node dependencies & Vite scripts
├── vite.config.ts             # Vite build configuration
├── src/                       # Frontend application
│   ├── main.ts                # UI controller, event listeners & Tauri invoke calls
│   └── styles.css             # Modern dark-mode styling & layout
└── src-tauri/                 # Rust backend
    ├── Cargo.toml             # Rust dependencies & build profiles
    ├── tauri.conf.json        # Tauri v2 window and security settings
    ├── src/
    │   ├── main.rs            # Application entry point
    │   ├── lib.rs             # Tauri command registrations & event dispatch
    │   ├── audio_capture.rs   # Windows WASAPI microphone & process loopback capture
    │   ├── dsp_vad.rs         # Rubato resampler, AGC, silence gate, Silero VAD
    │   ├── transcriber.rs     # Sherpa Whisper & Parakeet recognizers, repetition filters
    │   ├── diarization.rs     # Speaker embeddings & Agglomerative Hierarchical Clustering
    │   ├── models.rs          # Model registry, discovery, and HTTP downloader
    │   ├── runtimes.rs        # DirectML / CUDA runtime management
    │   ├── state.rs           # Thread-safe application state
    │   ├── commands.rs        # Tauri IPC handlers invoked by frontend
    │   └── bin/
    │       └── ab_compare.rs  # Standalone A/B comparison CLI tool
```

---

## Troubleshooting

- **"localhost refused to connect" on startup:**
  - This occurs if the desktop window launches before the Vite development server finishes binding port `1420`. Run `npm run build` or launch via `npm run tauri dev`.
- **Microphone or System Audio not capturing:**
  - Ensure Windows microphone permissions are granted in *Windows Settings $\to$ Privacy & Security $\to$ Microphone*.
  - For application capture, ensure the target process is actively generating audio.
- **CUDA fallback warning:**
  - If CUDA execution is requested but the required CUDA libraries or driver are not detected, the app logs a warning and automatically falls back to CPU execution. You can check the active provider pill in the bottom status bar.

---

## License

This project is licensed under the [MIT License](LICENSE) or Apache-2.0 License.
Model weights (Whisper, Parakeet, Silero, CAM++) are subject to their respective upstream licenses.
