import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

interface AudioDeviceInfo {
  id: string;
  name: string;
  is_default: boolean;
}

interface AppProcessInfo {
  pid: number;
  name: string;
  title: string;
  window_title?: string;
  has_window: boolean;
  icon_hint: string;
}

interface SpeakerProfile {
  id: string;
  display_name: string;
  centroid: number[];
  sample_count: number;
  total_speech_ms: number;
  color_hex: string;
  locked: boolean;
}

interface TranscriptRevision {
  id: number;
  speaker_id: string;
  speaker_name: string;
  speaker_color: string;
}

interface TranscriptEntry {
  id: number;
  timestamp_str: string;
  source_type: "mic" | "app";
  app_name: string;
  speaker_id: string;
  speaker_name: string;
  speaker_color: string;
  text: string;
  confidence: number;
  duration_ms: number;
}

interface AudioLevels {
  mic_rms: number;
  app_rms: number;
  mic_speaking: boolean;
  app_speaking: boolean;
}

interface WhisperModelOption {
  id: string;
  name: string;
  tag: string;
  description: string;
  is_downloaded: boolean;
  is_active: boolean;
}

interface ModelStatus {
  vad_ready: boolean;
  pyannote_ready: boolean;
  speaker_id_ready: boolean;
  whisper_ready: boolean;
  all_ready: boolean;
  models_dir: string;
  vad_model_path: string;
  pyannote_model_path: string;
  speaker_id_model_path: string;
  whisper_encoder_path: string;
  whisper_decoder_path: string;
  whisper_tokens_path: string;
  active_whisper_model: string;
  available_whisper_models: WhisperModelOption[];
}

interface DeviceOption {
  id: string; // "cpu" | "cuda" | "directml"
  name: string;
  tag: string;
  description: string;
  is_downloaded: boolean;
  is_active: boolean;
}

interface DeviceStatus {
  active_device: string;
  is_downloaded: boolean;
  available_devices: DeviceOption[];
  runtimes_dir: string;
}

interface ProviderStatus {
  requested: string;
  actual: string;
  is_fallback: boolean;
}

interface DownloadProgress {
  item_id: string;
  percentage: number;
  downloaded_bytes: number;
  total_bytes: number;
  phase: string;
  message: string;
}

// State
let isRecording = false;
let currentSpeakers: SpeakerProfile[] = [];
let entries: TranscriptEntry[] = [];
let userScrolledUp = false;

// Whisper & Vocabulary State
let availableWhisperModels: WhisperModelOption[] = [];

// Device Acceleration State
let availableDevices: DeviceOption[] = [];
let activeDevice = "cpu";
let isDownloadingModels = false;
let isDownloadingVariant = false;
let isDownloadingDevice = false;

// Selected App Target State
let selectedAppMode: "global" | "app_pid" = "global";
let selectedPid: number | null = null;
let selectedAppName = "Global Desktop Audio";

// App Selector Modal State
let runningApps: AppProcessInfo[] = [];
let activeFilter = "all";
let searchQuery = "";

// DOM Elements
const micSelect = document.getElementById("mic-select") as HTMLSelectElement;
const btnRefreshApps = document.getElementById("btn-refresh-apps") as HTMLButtonElement;
const slider = document.getElementById("similarity-slider") as HTMLInputElement;
const sliderVal = document.getElementById("similarity-value") as HTMLElement;
const btnToggleCapture = document.getElementById("btn-toggle-capture") as HTMLButtonElement;
const captureBtnText = document.getElementById("capture-btn-text") as HTMLElement;
const iconPlay = document.querySelector(".icon-play") as HTMLElement;
const iconStop = document.querySelector(".icon-stop") as HTMLElement;
const sessionStatus = document.getElementById("session-status") as HTMLElement;
const statusText = document.getElementById("status-text") as HTMLElement;

const micMeterBar = document.getElementById("mic-meter-bar") as HTMLElement;
const appMeterBar = document.getElementById("app-meter-bar") as HTMLElement;
const micVadLed = document.getElementById("mic-vad-led") as HTMLElement;
const appVadLed = document.getElementById("app-vad-led") as HTMLElement;

const modelStatusLabel = document.getElementById("model-status-label") as HTMLElement;
const btnDownloadModels = document.getElementById("btn-download-models") as HTMLButtonElement;

// Accuracy & Tuning Bar Elements
const whisperModelSelect = document.getElementById("whisper-model-select") as HTMLSelectElement;
const whisperStatusPill = document.getElementById("whisper-status-pill") as HTMLElement;
const btnDownloadVariant = document.getElementById("btn-download-variant") as HTMLButtonElement;
const deviceSelect = document.getElementById("device-select") as HTMLSelectElement;
const deviceStatusPill = document.getElementById("device-status-pill") as HTMLElement;
const btnDownloadDeviceRuntime = document.getElementById("btn-download-device-runtime") as HTMLButtonElement;
const customVocabInput = document.getElementById("custom-vocab-input") as HTMLInputElement;
const btnSaveVocab = document.getElementById("btn-save-vocab") as HTMLButtonElement;
const vocabStatusPill = document.getElementById("vocab-status-pill") as HTMLElement;

// App Selector Elements
const appSelectorTrigger = document.getElementById("app-selector-trigger") as HTMLElement;
const selectedAppIconEl = document.getElementById("selected-app-icon") as HTMLElement;
const selectedAppNameEl = document.getElementById("selected-app-name") as HTMLElement;
const selectedAppSubEl = document.getElementById("selected-app-sub") as HTMLElement;
const btnOpenAppModal = document.getElementById("btn-open-app-modal") as HTMLButtonElement;

// App Modal Elements
const appModalOverlay = document.getElementById("app-modal-overlay") as HTMLElement;
const btnCloseAppModal = document.getElementById("btn-close-app-modal") as HTMLButtonElement;
const appSearchInput = document.getElementById("app-search-input") as HTMLInputElement;
const btnClearSearch = document.getElementById("btn-clear-search") as HTMLButtonElement;
const filterChips = document.querySelectorAll<HTMLButtonElement>(".filter-chips .chip");
const modalAppList = document.getElementById("modal-app-list") as HTMLElement;

// Transcript Elements
const transcriptFeed = document.getElementById("transcript-feed") as HTMLElement;
const entryCountBadge = document.getElementById("entry-count-badge") as HTMLElement;
const btnJumpLive = document.getElementById("btn-jump-live") as HTMLButtonElement;
const btnClearTranscript = document.getElementById("btn-clear-transcript") as HTMLButtonElement;

// Speaker Roster Elements
const speakerList = document.getElementById("speaker-list") as HTMLElement;
const speakerCountBadge = document.getElementById("speaker-count-badge") as HTMLElement;
const btnCopyTranscript = document.getElementById("btn-copy-transcript") as HTMLButtonElement;
const copyBtnText = document.getElementById("copy-btn-text") as HTMLElement;
const btnExportTxt = document.getElementById("btn-export-txt") as HTMLButtonElement;
const btnExportMd = document.getElementById("btn-export-md") as HTMLButtonElement;
const btnResetSpeakers = document.getElementById("btn-reset-speakers") as HTMLButtonElement;

// Initialization
async function init() {
  initDiarizationThreshold();
  await loadAudioDevices();
  await loadRunningApplications();
  await loadWhisperModels();
  await loadDevices();
  await checkModels();
  await initCustomVocabulary();
  await loadSpeakers();
  await loadHistory();
  setupEventListeners();
  setupAppSelectorModal();
  setupTauriListeners();
}

function initDiarizationThreshold() {
  const saved = localStorage.getItem("dualstream_diarization_threshold");
  if (saved) {
    const val = parseFloat(saved);
    if (!isNaN(val) && val >= 0.45 && val <= 0.80) {
      slider.value = saved;
      sliderVal.textContent = val.toFixed(2);
      invoke("set_diarization_threshold", { threshold: val }).catch(console.error);
      return;
    }
  }
  slider.value = "0.58";
  sliderVal.textContent = "0.58";
  invoke("set_diarization_threshold", { threshold: 0.58 }).catch(console.error);
}

async function loadAudioDevices() {
  try {
    const devices = await invoke<AudioDeviceInfo[]>("get_audio_devices");
    micSelect.innerHTML = "";

    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
    defaultOpt.textContent = "Default Microphone";
    micSelect.appendChild(defaultOpt);

    devices.forEach((dev) => {
      const opt = document.createElement("option");
      opt.value = dev.id;
      opt.textContent = `${dev.name}${dev.is_default ? " (System Default)" : ""}`;
      if (dev.is_default) {
        opt.selected = true;
      }
      micSelect.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed to load audio devices:", err);
  }
}

async function loadRunningApplications() {
  btnRefreshApps.textContent = "Refreshing...";
  try {
    const apps = await invoke<AppProcessInfo[]>("get_running_applications");
    runningApps = apps;
    renderModalAppList();
  } catch (err) {
    console.error("Failed to load applications:", err);
  } finally {
    btnRefreshApps.textContent = "↻ Refresh";
  }
}

function getAppEmoji(iconHint: string): string {
  switch (iconHint) {
    case "discord":
      return "🎧";
    case "spotify":
      return "🎵";
    case "chrome":
    case "firefox":
    case "edge":
    case "brave":
      return "🌐";
    case "teams":
    case "zoom":
    case "slack":
    case "telegram":
      return "👥";
    case "steam":
      return "🎮";
    case "vlc":
      return "🎬";
    case "obs":
      return "📹";
    default:
      return "🪟";
  }
}

function selectTargetApp(mode: "global" | "app_pid", pid: number | null, name: string, sub: string, icon: string) {
  selectedAppMode = mode;
  selectedPid = pid;
  selectedAppName = name;

  selectedAppIconEl.textContent = icon;
  selectedAppNameEl.textContent = name;
  selectedAppSubEl.textContent = sub;

  closeAppModal();
  renderModalAppList();
}

function openAppModal() {
  appModalOverlay.classList.remove("hidden");
  appSearchInput.value = "";
  searchQuery = "";
  btnClearSearch.classList.add("hidden");
  renderModalAppList();
  setTimeout(() => appSearchInput.focus(), 50);
}

function closeAppModal() {
  appModalOverlay.classList.add("hidden");
}

function setupAppSelectorModal() {
  appSelectorTrigger.addEventListener("click", () => {
    openAppModal();
  });

  btnOpenAppModal.addEventListener("click", (e) => {
    e.stopPropagation();
    openAppModal();
  });

  btnCloseAppModal.addEventListener("click", () => {
    closeAppModal();
  });

  appModalOverlay.addEventListener("click", (e) => {
    if (e.target === appModalOverlay) {
      closeAppModal();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !appModalOverlay.classList.contains("hidden")) {
      closeAppModal();
    }
  });

  appSearchInput.addEventListener("input", () => {
    searchQuery = appSearchInput.value.trim().toLowerCase();
    if (searchQuery.length > 0) {
      btnClearSearch.classList.remove("hidden");
    } else {
      btnClearSearch.classList.add("hidden");
    }
    renderModalAppList();
  });

  btnClearSearch.addEventListener("click", () => {
    appSearchInput.value = "";
    searchQuery = "";
    btnClearSearch.classList.add("hidden");
    renderModalAppList();
    appSearchInput.focus();
  });

  filterChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      filterChips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      activeFilter = chip.dataset.filter || "all";
      renderModalAppList();
    });
  });
}

function renderModalAppList() {
  modalAppList.innerHTML = "";

  // 1. Global Audio Hero Card (always visible unless search excludes it)
  const matchesGlobalSearch =
    searchQuery === "" ||
    "global desktop audio system sound loopback".includes(searchQuery);

  const matchesGlobalFilter =
    activeFilter === "all" || activeFilter === "global";

  if (matchesGlobalSearch && matchesGlobalFilter) {
    const isSelected = selectedAppMode === "global";
    const globalCard = document.createElement("div");
    globalCard.className = `app-item-card hero-global ${isSelected ? "is-selected" : ""}`;
    globalCard.innerHTML = `
      <div class="app-item-icon" style="background: rgba(59, 130, 246, 0.2); color: #60a5fa;">🔊</div>
      <div class="app-item-content">
        <div class="app-item-top">
          <span class="app-item-title">Global Desktop Audio (System Loopback)</span>
          <span class="app-item-pid" style="color: #60a5fa;">Default</span>
        </div>
        <span class="app-item-desc">Captures all computer sounds, games, meetings, and browser audio simultaneously</span>
      </div>
      <button class="app-item-btn ${isSelected ? "btn-selected" : ""}">
        ${isSelected ? "Selected ✓" : "Select"}
      </button>
    `;
    globalCard.addEventListener("click", () => {
      selectTargetApp(
        "global",
        null,
        "Global Desktop Audio",
        "Captures all system sounds & meetings",
        "🔊"
      );
    });
    modalAppList.appendChild(globalCard);
  }

  if (activeFilter === "global") {
    return;
  }

  // Filter apps
  const filteredApps = runningApps.filter((app) => {
    // Search match
    const q = searchQuery;
    const nameMatch = app.name.toLowerCase().includes(q);
    const titleMatch = app.title.toLowerCase().includes(q);
    const windowMatch = app.window_title ? app.window_title.toLowerCase().includes(q) : false;
    const searchPass = q === "" || nameMatch || titleMatch || windowMatch;
    if (!searchPass) return false;

    // Filter chip match
    if (activeFilter === "windows") {
      return app.has_window;
    }
    if (activeFilter === "voice") {
      return ["discord", "teams", "zoom", "slack", "telegram"].includes(app.icon_hint);
    }
    if (activeFilter === "browsers") {
      return ["chrome", "firefox", "edge", "brave"].includes(app.icon_hint);
    }
    return true;
  });

  if (filteredApps.length === 0 && !matchesGlobalSearch) {
    const noResults = document.createElement("div");
    noResults.className = "modal-no-results";
    noResults.innerHTML = `No applications matching "<strong>${escapeHtml(searchQuery)}</strong>" found.<br>Try refreshing running applications.`;
    modalAppList.appendChild(noResults);
    return;
  }

  // Split into "Active Windows" and "Other Running Apps"
  const windowApps = filteredApps.filter((a) => a.has_window);
  const backgroundApps = filteredApps.filter((a) => !a.has_window);

  if (windowApps.length > 0) {
    const secTitle = document.createElement("div");
    secTitle.className = "app-section-title";
    secTitle.textContent = `Active Windows (${windowApps.length})`;
    modalAppList.appendChild(secTitle);

    windowApps.forEach((app) => {
      modalAppList.appendChild(createAppCardElement(app));
    });
  }

  if (backgroundApps.length > 0 && activeFilter !== "windows") {
    const secTitle = document.createElement("div");
    secTitle.className = "app-section-title";
    secTitle.textContent = `Other Applications (${backgroundApps.length})`;
    modalAppList.appendChild(secTitle);

    backgroundApps.forEach((app) => {
      modalAppList.appendChild(createAppCardElement(app));
    });
  }
}

function createAppCardElement(app: AppProcessInfo): HTMLElement {
  const isSelected = selectedAppMode === "app_pid" && selectedPid === app.pid;
  const emoji = getAppEmoji(app.icon_hint);

  const card = document.createElement("div");
  card.className = `app-item-card ${isSelected ? "is-selected" : ""}`;

  const cleanName = app.name.replace(/\.exe$/i, "");
  const windowDisplay = app.window_title
    ? `<span class="app-item-window">🪟 ${escapeHtml(app.window_title)}</span>`
    : `<span class="app-item-desc">${escapeHtml(app.title)}</span>`;

  card.innerHTML = `
    <div class="app-item-icon">${emoji}</div>
    <div class="app-item-content">
      <div class="app-item-top">
        <span class="app-item-title">${escapeHtml(cleanName)}</span>
        <span class="app-item-pid">PID: ${app.pid}</span>
      </div>
      ${windowDisplay}
    </div>
    <button class="app-item-btn ${isSelected ? "btn-selected" : ""}">
      ${isSelected ? "Selected ✓" : "Select"}
    </button>
  `;

  card.addEventListener("click", () => {
    const displaySub = app.window_title
      ? `${app.window_title} (PID: ${app.pid})`
      : `${app.name} (PID: ${app.pid})`;

    selectTargetApp("app_pid", app.pid, cleanName, displaySub, emoji);
  });

  return card;
}

function setButtonProgress(btn: HTMLButtonElement, percentage: number | null, text: string) {
  btn.classList.add("downloading");
  btn.disabled = true;
  const spinner = btn.querySelector(".btn-spinner");
  if (spinner) spinner.classList.remove("hidden");
  const label = btn.querySelector(".btn-label");
  if (label) {
    label.textContent = percentage !== null ? `${Math.round(percentage)}% ${text}`.trim() : text;
  } else {
    btn.textContent = percentage !== null ? `${Math.round(percentage)}% ${text}`.trim() : text;
  }
}

function resetButtonProgress(btn: HTMLButtonElement, defaultText: string) {
  btn.classList.remove("downloading");
  btn.disabled = false;
  const spinner = btn.querySelector(".btn-spinner");
  if (spinner) spinner.classList.add("hidden");
  const label = btn.querySelector(".btn-label");
  if (label) {
    label.textContent = defaultText;
  } else {
    btn.textContent = defaultText;
  }
}

async function checkModels() {
  try {
    const status = await invoke<ModelStatus>("get_models_status");
    if (status.available_whisper_models && status.available_whisper_models.length > 0) {
      availableWhisperModels = status.available_whisper_models;
      // Refresh the option texts in-place without altering the user's dropdown choice
      const currentSelected = whisperModelSelect.value;
      for (const opt of Array.from(whisperModelSelect.options)) {
        const m = availableWhisperModels.find((mod) => mod.id === opt.value);
        if (m) {
          opt.textContent = `${m.name} (${m.tag})${m.is_downloaded ? " ✓" : " [needs download]"}`;
        }
      }
      if (currentSelected) {
        whisperModelSelect.value = currentSelected;
      }
      updateWhisperModelUi();
    }
    if (status.all_ready) {
      modelStatusLabel.textContent = `AI Models Ready (${status.active_whisper_model} • CAM++ • Pyannote 3.0)`;
      modelStatusLabel.classList.add("ready");
      btnDownloadModels.classList.add("hidden");
    } else {
      modelStatusLabel.textContent = "Local models not downloaded";
      modelStatusLabel.classList.remove("ready");
      btnDownloadModels.classList.remove("hidden");
    }
  } catch (err) {
    console.error("Failed to check models:", err);
  }
}

async function downloadModels() {
  isDownloadingModels = true;
  setButtonProgress(btnDownloadModels, 0, "Downloading (~95MB)...");
  modelStatusLabel.textContent = "Downloading Whisper & Diarization models...";
  try {
    const status = await invoke<ModelStatus>("download_models");
    await loadWhisperModels();
    await checkModels();
    if (status.all_ready) {
      modelStatusLabel.textContent = `AI Models Ready (${status.active_whisper_model} • CAM++ • Pyannote 3.0)`;
      modelStatusLabel.classList.add("ready");
      btnDownloadModels.classList.add("hidden");
    } else {
      modelStatusLabel.textContent = "Partial download completed. Ready with fallbacks.";
    }
  } catch (err) {
    console.error("Failed to download models:", err);
    modelStatusLabel.textContent = "Model download error. Ready with fallbacks.";
    resetButtonProgress(btnDownloadModels, "Download AI Models (~95MB)");
  } finally {
    isDownloadingModels = false;
    btnDownloadModels.disabled = false;
  }
}

async function loadWhisperModels(preferredSelection?: string) {
  try {
    const currentVal = preferredSelection || whisperModelSelect.value;
    const models = await invoke<WhisperModelOption[]>("get_available_whisper_models");
    availableWhisperModels = models;
    whisperModelSelect.innerHTML = "";

    let hasSelected = false;
    models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.name} (${m.tag})${m.is_downloaded ? " ✓" : " [needs download]"}`;
      if (currentVal ? m.id === currentVal : m.is_active) {
        opt.selected = true;
        hasSelected = true;
      }
      whisperModelSelect.appendChild(opt);
    });

    if (!hasSelected && models.length > 0) {
      const best = models.find((m) => m.is_active) || models.find((m) => m.is_downloaded) || models[0];
      whisperModelSelect.value = best.id;
    }

    updateWhisperModelUi();
  } catch (err) {
    console.error("Failed to load whisper models:", err);
  }
}

function updateWhisperModelUi() {
  const current = availableWhisperModels.find((m) => m.id === whisperModelSelect.value);
  if (!current) return;

  if (current.is_downloaded) {
    whisperStatusPill.textContent = "Ready ✓";
    whisperStatusPill.className = "status-pill ready";
    btnDownloadVariant.classList.add("hidden");
    btnDownloadVariant.disabled = false;
  } else {
    whisperStatusPill.textContent = "Not Downloaded";
    whisperStatusPill.className = "status-pill missing";
    btnDownloadVariant.classList.remove("hidden");
    btnDownloadVariant.disabled = false;
    const shortName = current.name.replace("Whisper ", "");
    resetButtonProgress(btnDownloadVariant, `Download ${shortName}`);
  }
}

async function onWhisperModelChanged() {
  if (isRecording) {
    alert("Please stop recording before switching models.");
    return;
  }
  const selectedId = whisperModelSelect.value;
  const current = availableWhisperModels.find((m) => m.id === selectedId);
  if (!current) return;

  updateWhisperModelUi();

  if (current.is_downloaded) {
    try {
      await invoke("set_active_whisper_model", { variant: selectedId });
      await checkModels();
    } catch (err) {
      console.error("Failed to set active whisper model:", err);
    }
  }
}

async function downloadActiveWhisperModel() {
  const selectedId = whisperModelSelect.value;
  const current = availableWhisperModels.find((m) => m.id === selectedId);
  if (!current) return;

  isDownloadingVariant = true;
  setButtonProgress(btnDownloadVariant, 0, "Starting download...");
  whisperModelSelect.disabled = true;
  whisperStatusPill.textContent = "Downloading archive...";
  whisperStatusPill.className = "status-pill downloading";

  try {
    await invoke<ModelStatus>("download_whisper_model", { variant: selectedId });
    await loadWhisperModels(selectedId);
    await checkModels();

    whisperStatusPill.textContent = "Ready ✓";
    whisperStatusPill.className = "status-pill ready";
    btnDownloadVariant.classList.add("hidden");
  } catch (err) {
    console.error("Failed to download whisper model variant:", err);
    whisperStatusPill.textContent = "Download Error";
    whisperStatusPill.className = "status-pill missing";
    const shortName = current.name.replace("Whisper ", "");
    resetButtonProgress(btnDownloadVariant, `Download ${shortName}`);
    alert(`Failed to download model: ${err}`);
  } finally {
    isDownloadingVariant = false;
    btnDownloadVariant.disabled = false;
    whisperModelSelect.disabled = false;
    updateWhisperModelUi();
  }
}

async function loadDevices(preferredSelection?: string) {
  try {
    const currentVal = preferredSelection || deviceSelect.value || "cpu";
    const devices = await invoke<DeviceOption[]>("get_available_devices");
    availableDevices = devices;
    deviceSelect.innerHTML = "";

    let hasSelected = false;
    devices.forEach((d) => {
      const opt = document.createElement("option");
      opt.value = d.id;
      opt.textContent = `${d.name} (${d.tag})${d.is_downloaded ? " ✓" : " [needs download]"}`;
      if (currentVal ? d.id === currentVal : d.is_active) {
        opt.selected = true;
        hasSelected = true;
        activeDevice = d.id;
      }
      deviceSelect.appendChild(opt);
    });

    if (!hasSelected && devices.length > 0) {
      deviceSelect.value = "cpu";
      activeDevice = "cpu";
    }

    updateDeviceUi();
  } catch (err) {
    console.error("Failed to load device acceleration options:", err);
  }
}

function updateDeviceUi() {
  const current = availableDevices.find((d) => d.id === deviceSelect.value);
  if (!current) return;

  if (current.id === "cpu" || current.is_downloaded) {
    deviceStatusPill.textContent = "Ready ✓";
    deviceStatusPill.className = "status-pill ready";
    btnDownloadDeviceRuntime.classList.add("hidden");
    btnDownloadDeviceRuntime.disabled = false;
  } else {
    deviceStatusPill.textContent = "Not Downloaded";
    deviceStatusPill.className = "status-pill missing";
    btnDownloadDeviceRuntime.classList.remove("hidden");
    btnDownloadDeviceRuntime.disabled = false;
    resetButtonProgress(btnDownloadDeviceRuntime, `Download ${current.tag}`);
  }
}

async function onDeviceChanged() {
  if (isRecording) {
    alert("Please stop recording before switching acceleration devices.");
    deviceSelect.value = activeDevice;
    return;
  }
  const selectedId = deviceSelect.value;
  activeDevice = selectedId;
  updateDeviceUi();

  const current = availableDevices.find((d) => d.id === selectedId);
  if (!current) return;

  try {
    await invoke("set_active_device", { device: selectedId });
  } catch (err) {
    console.error("Failed to set active device:", err);
  }
}

async function downloadActiveDeviceRuntime() {
  const selectedId = deviceSelect.value;
  if (selectedId === "cpu") return;
  const current = availableDevices.find((d) => d.id === selectedId);
  if (!current) return;

  isDownloadingDevice = true;
  setButtonProgress(btnDownloadDeviceRuntime, 0, "Starting download...");
  deviceSelect.disabled = true;
  deviceStatusPill.textContent = "Downloading...";
  deviceStatusPill.className = "status-pill downloading";

  try {
    await invoke<DeviceStatus>("download_device_runtime", { device: selectedId });
    await loadDevices(selectedId);
    deviceStatusPill.textContent = "Ready ✓";
    deviceStatusPill.className = "status-pill ready";
    btnDownloadDeviceRuntime.classList.add("hidden");
  } catch (err) {
    console.error("Failed to download GPU runtime libraries:", err);
    deviceStatusPill.textContent = "Download Error";
    deviceStatusPill.className = "status-pill missing";
    resetButtonProgress(btnDownloadDeviceRuntime, `Retry ${current.tag}`);
    alert(`Failed to download GPU runtime libraries: ${err}`);
  } finally {
    isDownloadingDevice = false;
    deviceSelect.disabled = false;
    updateDeviceUi();
  }
}

async function initCustomVocabulary() {
  const saved = localStorage.getItem("dualstream_custom_vocab");
  if (saved) {
    customVocabInput.value = saved;
    const words = saved.split(",").map((s) => s.trim()).filter((s) => s.length > 0);
    try {
      await invoke("set_custom_vocabulary", { vocabulary: words });
    } catch (err) {
      console.error("Failed to initialize custom vocabulary:", err);
    }
  } else {
    try {
      const vocab = await invoke<string[]>("get_custom_vocabulary");
      if (vocab && vocab.length > 0) {
        customVocabInput.value = vocab.join(", ");
      }
    } catch (err) {
      console.error("Failed to fetch custom vocabulary:", err);
    }
  }
}

async function saveCustomVocabulary() {
  const text = customVocabInput.value.trim();
  const words = text
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  localStorage.setItem("dualstream_custom_vocab", text);

  try {
    await invoke("set_custom_vocabulary", { vocabulary: words });
    vocabStatusPill.textContent = `Synced (${words.length} words)`;
    vocabStatusPill.className = "status-pill saved";
    setTimeout(() => {
      vocabStatusPill.textContent = "Synced";
    }, 2500);
  } catch (err) {
    console.error("Failed to save custom vocabulary:", err);
    vocabStatusPill.textContent = "Sync Error";
    vocabStatusPill.className = "status-pill missing";
  }
}

async function loadSpeakers() {
  try {
    const speakers = await invoke<SpeakerProfile[]>("get_speakers");
    currentSpeakers = speakers;
    renderSpeakerRoster();
  } catch (err) {
    console.error("Failed to load speakers:", err);
  }
}

async function loadHistory() {
  try {
    const history = await invoke<TranscriptEntry[]>("get_transcript_history");
    entries = history;
    renderTranscriptFeed();
  } catch (err) {
    console.error("Failed to load transcript history:", err);
  }
}

function setupEventListeners() {
  btnRefreshApps.addEventListener("click", () => {
    loadRunningApplications();
  });

  slider.addEventListener("input", () => {
    const val = parseFloat(slider.value);
    sliderVal.textContent = val.toFixed(2);
    localStorage.setItem("dualstream_diarization_threshold", slider.value);
    invoke("set_diarization_threshold", { threshold: val }).catch(console.error);
  });

  btnDownloadModels.addEventListener("click", () => {
    downloadModels();
  });

  whisperModelSelect.addEventListener("change", () => {
    onWhisperModelChanged();
  });

  btnDownloadVariant.addEventListener("click", () => {
    downloadActiveWhisperModel();
  });

  deviceSelect.addEventListener("change", () => {
    onDeviceChanged();
  });

  btnDownloadDeviceRuntime.addEventListener("click", () => {
    downloadActiveDeviceRuntime();
  });

  btnSaveVocab.addEventListener("click", () => {
    saveCustomVocabulary();
  });

  customVocabInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      saveCustomVocabulary();
    }
  });

  customVocabInput.addEventListener("blur", () => {
    saveCustomVocabulary();
  });

  btnToggleCapture.addEventListener("click", () => {
    toggleCapture();
  });

  transcriptFeed.addEventListener("scroll", () => {
    const threshold = 60;
    const isAtBottom =
      transcriptFeed.scrollHeight - transcriptFeed.scrollTop - transcriptFeed.clientHeight <= threshold;
    userScrolledUp = !isAtBottom;
    if (userScrolledUp) {
      btnJumpLive.classList.remove("hidden");
    } else {
      btnJumpLive.classList.add("hidden");
    }
  });

  btnJumpLive.addEventListener("click", () => {
    userScrolledUp = false;
    transcriptFeed.scrollTo({ top: transcriptFeed.scrollHeight, behavior: "smooth" });
    btnJumpLive.classList.add("hidden");
  });

  btnClearTranscript.addEventListener("click", async () => {
    if (confirm("Clear live transcript history?")) {
      try {
        await invoke("clear_session");
        entries = [];
        renderTranscriptFeed();
        await loadSpeakers();
      } catch (err) {
        console.error("Failed to clear transcript:", err);
      }
    }
  });

  btnCopyTranscript.addEventListener("click", async () => {
    try {
      const text = await invoke<string>("export_transcript", { format: "text" });
      await navigator.clipboard.writeText(text);
      copyBtnText.textContent = "Copied!";
      setTimeout(() => {
        copyBtnText.textContent = "Copy Transcript";
      }, 2000);
    } catch (err) {
      console.error("Failed to copy transcript:", err);
    }
  });

  btnExportTxt.addEventListener("click", async () => {
    downloadTranscriptFile("text", "transcript.txt", "text/plain");
  });

  btnExportMd.addEventListener("click", async () => {
    downloadTranscriptFile("markdown", "transcript.md", "text/markdown");
  });

  btnResetSpeakers.addEventListener("click", async () => {
    if (confirm("Reset learned voice profiles to default?")) {
      try {
        await invoke("clear_session");
        await loadSpeakers();
      } catch (err) {
        console.error("Failed to reset speakers:", err);
      }
    }
  });
}

async function downloadTranscriptFile(format: string, filename: string, mimeType: string) {
  try {
    const text = await invoke<string>("export_transcript", { format });
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error("Failed to export transcript:", err);
  }
}

async function toggleCapture() {
  if (!isRecording) {
    const micId = micSelect.value || null;

    const config = {
      mic_device_id: micId,
      capture_mode: selectedAppMode,
      target_pid: selectedPid,
      target_app_name: selectedAppName,
      similarity_threshold: parseFloat(slider.value),
      device: deviceSelect.value || activeDevice || "cpu",
    };

    try {
      await invoke("start_transcription", { config });
      isRecording = true;
      updateRecordingUI(true);
    } catch (err) {
      console.error("Failed to start transcription:", err);
      alert(`Could not start capture: ${err}`);
    }
  } else {
    try {
      btnToggleCapture.disabled = true;
      captureBtnText.textContent = "Stopping...";
      await invoke("stop_transcription");
      isRecording = false;
      updateRecordingUI(false);
    } catch (err) {
      console.error("Failed to stop transcription:", err);
    } finally {
      btnToggleCapture.disabled = false;
    }
  }
}

function updateRecordingUI(recording: boolean) {
  whisperModelSelect.disabled = recording;
  btnDownloadVariant.disabled = recording;
  deviceSelect.disabled = recording;
  btnDownloadDeviceRuntime.disabled = recording;
  micSelect.disabled = recording;
  btnOpenAppModal.disabled = recording;

  if (recording) {
    btnToggleCapture.className = "btn-capture btn-stop";
    captureBtnText.textContent = "Stop Capture";
    iconPlay.classList.add("hidden");
    iconStop.classList.remove("hidden");
    sessionStatus.className = "session-status live";
    statusText.textContent = "Live Recording & Diarizing";
  } else {
    btnToggleCapture.className = "btn-capture btn-start";
    captureBtnText.textContent = "Start Live Capture";
    iconPlay.classList.remove("hidden");
    iconStop.classList.add("hidden");
    sessionStatus.className = "session-status idle";
    statusText.textContent = "Ready to Record";
    micMeterBar.style.width = "0%";
    appMeterBar.style.width = "0%";
    micVadLed.classList.remove("active-mic");
    appVadLed.classList.remove("active-app");
  }
}

function setupTauriListeners() {
  // 1. Audio Levels (15 Hz)
  listen<AudioLevels>("audio-levels", (event) => {
    if (!isRecording) return;
    const { mic_rms, app_rms, mic_speaking, app_speaking } = event.payload;

    const micPct = Math.min(100, Math.pow(mic_rms * 4.0, 0.7) * 100);
    const appPct = Math.min(100, Math.pow(app_rms * 4.0, 0.7) * 100);

    micMeterBar.style.width = `${micPct}%`;
    appMeterBar.style.width = `${appPct}%`;

    if (mic_speaking) {
      micVadLed.classList.add("active-mic");
    } else {
      micVadLed.classList.remove("active-mic");
    }

    if (app_speaking) {
      appVadLed.classList.add("active-app");
    } else {
      appVadLed.classList.remove("active-app");
    }
  });

  // 2. Transcript Segment
  listen<TranscriptEntry>("transcript-entry", (event) => {
    const entry = event.payload;
    entries.push(entry);
    appendTranscriptEntry(entry);
    updateSpeakerCounts();
  });

  // 3. Speakers Updated
  listen<SpeakerProfile[]>("speakers-updated", (event) => {
    currentSpeakers = event.payload;
    renderSpeakerRoster();
    refreshTranscriptSpeakerNames();
  });

  // 4. Retroactive Transcript Revisions
  listen<TranscriptRevision[]>("transcript-revisions", (event) => {
    const revisions = event.payload;
    if (!revisions || revisions.length === 0) return;

    const revMap = new Map<number, TranscriptRevision>();
    revisions.forEach((r) => revMap.set(r.id, r));

    // Update in-memory entries array
    entries.forEach((entry) => {
      if (revMap.has(entry.id)) {
        const rev = revMap.get(entry.id)!;
        entry.speaker_id = rev.speaker_id;
        entry.speaker_name = rev.speaker_name;
        entry.speaker_color = rev.speaker_color;
      }
    });

    // Update existing DOM rows
    const rows = transcriptFeed.querySelectorAll<HTMLElement>(".transcript-row");
    rows.forEach((row) => {
      const segId = row.dataset.segmentId ? parseInt(row.dataset.segmentId, 10) : null;
      if (segId !== null && revMap.has(segId)) {
        const rev = revMap.get(segId)!;
        row.dataset.speakerId = rev.speaker_id;
        row.style.setProperty("--speaker-color", rev.speaker_color);
        const badge = row.querySelector<HTMLElement>(".speaker-badge");
        if (badge) {
          badge.textContent = rev.speaker_name;
          badge.style.backgroundColor = rev.speaker_color;
        }
      }
    });

    updateSpeakerCounts();
  });

  // 5. Active Execution Provider Status
  listen<ProviderStatus>("provider-status", (event) => {
    const { requested, actual, is_fallback } = event.payload;
    if (is_fallback) {
      deviceStatusPill.textContent = `${actual.toUpperCase()} (${requested.toUpperCase()} fallback)`;
      deviceStatusPill.className = "status-pill warning";
      statusText.textContent = `Live Recording (Fallback to ${actual.toUpperCase()})`;
    } else {
      deviceStatusPill.textContent = `Active: ${actual.toUpperCase()}`;
      deviceStatusPill.className = "status-pill ready";
      statusText.textContent = `Live Recording (${actual.toUpperCase()})`;
    }
  });

  // 6. Download Progress Streaming
  listen<DownloadProgress>("download-progress", (event) => {
    const p = event.payload;
    const pct = Math.round(p.percentage);
    const phaseText = p.phase === "extracting" ? "Extracting" : "Downloading";

    // 1. Device Runtime downloads
    if (isDownloadingDevice && p.item_id.startsWith("device-")) {
      const deviceId = p.item_id.replace("device-", "");
      if (deviceSelect.value === deviceId) {
        setButtonProgress(btnDownloadDeviceRuntime, pct, `${phaseText}...`);
        deviceStatusPill.textContent = `${pct}% ${phaseText}`;
        deviceStatusPill.className = "status-pill downloading";
      }
    }

    // 2. Whisper variant downloads
    if (isDownloadingVariant && p.item_id.startsWith("whisper-")) {
      const variantId = p.item_id.replace("whisper-", "");
      if (whisperModelSelect.value === variantId) {
        setButtonProgress(btnDownloadVariant, pct, `${phaseText}...`);
        whisperStatusPill.textContent = `${pct}% ${phaseText}`;
        whisperStatusPill.className = "status-pill downloading";
      }
    }

    // 3. General AI models download button
    if (isDownloadingModels) {
      setButtonProgress(btnDownloadModels, pct, `${phaseText}...`);
      modelStatusLabel.textContent = `${phaseText} ${p.item_id} (${pct}%)...`;
    }
  });
}

function renderTranscriptFeed() {
  transcriptFeed.innerHTML = "";
  if (entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.id = "empty-state";
    empty.innerHTML = `
      <div class="empty-icon">🎙️</div>
      <h3>Awaiting Audio Input</h3>
      <p>Select your microphone and target application above, then click <strong>Start Live Capture</strong>.</p>
      <p class="empty-sub">Speech from your microphone is automatically tagged as <strong>Me</strong>. Incoming application audio is separated by voice embeddings.</p>
    `;
    transcriptFeed.appendChild(empty);
    entryCountBadge.textContent = "0 utterances";
    return;
  }

  entries.forEach((entry) => {
    appendTranscriptEntry(entry, false);
  });

  entryCountBadge.textContent = `${entries.length} utterance${entries.length === 1 ? "" : "s"}`;
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
}

function appendTranscriptEntry(entry: TranscriptEntry, autoScroll = true) {
  const empty = document.getElementById("empty-state");
  if (empty) {
    empty.remove();
  }

  const row = document.createElement("div");
  row.className = `transcript-row ${entry.source_type === "mic" ? "row-mic" : "row-app"}`;
  row.dataset.speakerId = entry.speaker_id;
  row.dataset.segmentId = String(entry.id);
  row.style.setProperty("--speaker-color", entry.speaker_color);

  const contextTag =
    entry.source_type === "mic" ? `<span class="context-tag">(to ${escapeHtml(entry.app_name)})</span>` : "";

  const durationSec = (entry.duration_ms / 1000).toFixed(1);

  row.innerHTML = `
    <div class="row-meta">
      <span class="meta-time">${escapeHtml(entry.timestamp_str)}</span>
      <span class="speaker-badge" style="background-color: ${entry.speaker_color};">
        ${escapeHtml(entry.speaker_name)}
      </span>
      ${contextTag}
      <span class="duration-tag">${durationSec}s</span>
    </div>
    <div class="row-text">${escapeHtml(entry.text)}</div>
  `;

  transcriptFeed.appendChild(row);
  entryCountBadge.textContent = `${entries.length} utterance${entries.length === 1 ? "" : "s"}`;

  if (autoScroll && !userScrolledUp) {
    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
  }
}

function refreshTranscriptSpeakerNames() {
  const speakerMap = new Map(currentSpeakers.map((s) => [s.id, s]));
  const rows = transcriptFeed.querySelectorAll<HTMLElement>(".transcript-row");
  rows.forEach((row) => {
    const spkId = row.dataset.speakerId;
    if (spkId && speakerMap.has(spkId)) {
      const spk = speakerMap.get(spkId)!;
      const badge = row.querySelector<HTMLElement>(".speaker-badge");
      if (badge) {
        badge.textContent = spk.display_name;
        badge.style.backgroundColor = spk.color_hex;
      }
      row.style.setProperty("--speaker-color", spk.color_hex);
    }
  });
}

function updateSpeakerCounts() {
  renderSpeakerRoster();
}

function renderSpeakerRoster() {
  speakerList.innerHTML = "";
  speakerCountBadge.textContent = `${currentSpeakers.length} speaker${currentSpeakers.length === 1 ? "" : "s"}`;

  const countMap: Record<string, number> = {};
  entries.forEach((e) => {
    countMap[e.speaker_id] = (countMap[e.speaker_id] || 0) + 1;
  });

  currentSpeakers.forEach((spk) => {
    const card = document.createElement("div");
    card.className = "speaker-card";

    const utteranceCount = countMap[spk.id] || spk.sample_count || 0;
    const speechSec = spk.total_speech_ms ? ` (${(spk.total_speech_ms / 1000).toFixed(1)}s)` : "";

    const cardHeader = document.createElement("div");
    cardHeader.className = "speaker-card-header";
    cardHeader.innerHTML = `
      <div class="speaker-color-dot" style="background-color: ${spk.color_hex};"></div>
      <span class="speaker-name-display">${escapeHtml(spk.display_name)}</span>
      <span class="speaker-utterances">${utteranceCount} utt${speechSec}</span>
    `;

    if (spk.id !== "me") {
      const btnLock = document.createElement("button");
      btnLock.className = `btn-lock-speaker ${spk.locked ? "locked" : ""}`;
      btnLock.title = spk.locked
        ? "Voiceprint is locked (click to unlock)"
        : "Lock voiceprint to prevent merging or re-clustering";
      btnLock.innerHTML = spk.locked ? "🔒 Locked" : "🔓 Lock";
      btnLock.addEventListener("click", async () => {
        try {
          await invoke("lock_speaker", { speakerId: spk.id, locked: !spk.locked });
        } catch (err) {
          console.error("Failed to toggle voice lock:", err);
        }
      });
      cardHeader.appendChild(btnLock);
    }

    card.appendChild(cardHeader);

    // Inline Rename Box
    const renameBox = document.createElement("div");
    renameBox.className = "rename-box";

    const renameInput = document.createElement("input");
    renameInput.className = "rename-input";
    renameInput.value = spk.display_name;
    renameInput.placeholder = "Rename speaker...";

    const btnSave = document.createElement("button");
    btnSave.className = "btn-save-name";
    btnSave.textContent = "Save";

    const handleSave = async () => {
      const newName = renameInput.value.trim();
      if (newName && newName !== spk.display_name) {
        try {
          await invoke("rename_speaker", { speakerId: spk.id, newName });
        } catch (err) {
          console.error("Failed to rename speaker:", err);
        }
      }
    };

    btnSave.addEventListener("click", handleSave);
    renameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") handleSave();
    });

    renameBox.appendChild(renameInput);
    renameBox.appendChild(btnSave);
    card.appendChild(renameBox);

    // Merge Box (Only for remote speakers, not "me")
    if (spk.id !== "me") {
      const otherSpeakers = currentSpeakers.filter((s) => s.id !== spk.id && s.id !== "me");
      if (otherSpeakers.length > 0) {
        const mergeBox = document.createElement("div");
        mergeBox.className = "merge-box";

        const mergeSelect = document.createElement("select");
        mergeSelect.className = "merge-select";

        otherSpeakers.forEach((other) => {
          const opt = document.createElement("option");
          opt.value = other.id;
          opt.textContent = `Merge into ${other.display_name}`;
          mergeSelect.appendChild(opt);
        });

        const btnMerge = document.createElement("button");
        btnMerge.className = "btn-merge";
        btnMerge.textContent = "Merge";

        btnMerge.addEventListener("click", async () => {
          const targetId = mergeSelect.value;
          if (confirm(`Merge "${spk.display_name}" into selected speaker?`)) {
            try {
              await invoke("merge_speakers", {
                sourceSpeakerId: spk.id,
                targetSpeakerId: targetId,
              });
            } catch (err) {
              console.error("Failed to merge speakers:", err);
            }
          }
        });

        mergeBox.appendChild(mergeSelect);
        mergeBox.appendChild(btnMerge);
        card.appendChild(mergeBox);
      }
    }

    speakerList.appendChild(card);
  });
}

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

window.addEventListener("DOMContentLoaded", () => {
  init();
});
