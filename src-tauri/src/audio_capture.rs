use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use tracing::{error, info, warn};
use wasapi::{
    AudioClient, DeviceEnumerator, Direction, SampleType, StreamMode, WaveFormat,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDeviceInfo {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppProcessInfo {
    pub pid: u32,
    pub name: String,
    pub title: String,
    pub window_title: Option<String>,
    pub has_window: bool,
    pub icon_hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "mode", content = "pid")]
pub enum CaptureMode {
    ApplicationPid(u32),
    GlobalLoopback,
}

/// Enumerate available hardware recording microphones via WASAPI.
pub fn enumerate_audio_devices() -> Result<Vec<AudioDeviceInfo>> {
    let _ = wasapi::initialize_mta().ok();

    let enumerator = wasapi::DeviceEnumerator::new()
        .map_err(|e| anyhow!("Failed to create DeviceEnumerator: {:?}", e))?;

    let default_id = enumerator
        .get_default_device(&Direction::Capture)
        .ok()
        .and_then(|d| d.get_id().ok())
        .unwrap_or_default();

    let collection = enumerator
        .get_device_collection(&Direction::Capture)
        .map_err(|e| anyhow!("Failed to access audio capture devices: {:?}", e))?;

    let count = collection
        .get_nbr_devices()
        .map_err(|e| anyhow!("Failed to get device count: {:?}", e))?;

    let mut devices = Vec::new();

    for i in 0..count {
        if let Ok(device) = collection.get_device_at_index(i) {
            let id = device.get_id().unwrap_or_default();
            let name = device
                .get_friendlyname()
                .unwrap_or_else(|_| format!("Audio Device {}", i + 1));
            let is_default = !default_id.is_empty() && id == default_id;

            devices.push(AudioDeviceInfo {
                id,
                name,
                is_default,
            });
        }
    }

    // Sort default device first, then alphabetical
    devices.sort_by(|a, b| {
        b.is_default
            .cmp(&a.is_default)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });

    Ok(devices)
}

#[link(name = "user32")]
extern "system" {
    fn EnumWindows(lpEnumFunc: Option<unsafe extern "system" fn(isize, isize) -> i32>, lParam: isize) -> i32;
    fn GetWindowThreadProcessId(hWnd: isize, lpdwProcessId: *mut u32) -> u32;
    fn IsWindowVisible(hWnd: isize) -> i32;
    fn GetWindowTextW(hWnd: isize, lpString: *mut u16, nMaxCount: i32) -> i32;
    fn GetWindowTextLengthW(hWnd: isize) -> i32;
}

unsafe extern "system" fn enum_windows_callback(hwnd: isize, lparam: isize) -> i32 {
    let map = &mut *(lparam as *mut HashMap<u32, String>);
    if IsWindowVisible(hwnd) != 0 {
        let len = GetWindowTextLengthW(hwnd);
        if len > 0 {
            let mut buf = vec![0u16; (len + 1) as usize];
            let read = GetWindowTextW(hwnd, buf.as_mut_ptr(), len + 1);
            if read > 0 {
                let title = String::from_utf16_lossy(&buf[..read as usize]);
                let trimmed = title.trim();
                if !trimmed.is_empty()
                    && trimmed != "Default IME"
                    && trimmed != "MSCTFIME UI"
                    && trimmed != "Program Manager"
                {
                    let mut pid = 0u32;
                    GetWindowThreadProcessId(hwnd, &mut pid);
                    if pid > 0 && !map.contains_key(&pid) {
                        map.insert(pid, trimmed.to_string());
                    }
                }
            }
        }
    }
    1
}

pub fn get_window_titles() -> HashMap<u32, String> {
    let mut map: HashMap<u32, String> = HashMap::new();
    unsafe {
        EnumWindows(Some(enum_windows_callback), &mut map as *mut _ as isize);
    }
    map
}

fn detect_icon_hint(name: &str) -> String {
    let lower = name.to_lowercase();
    if lower.contains("discord") {
        "discord".into()
    } else if lower.contains("spotify") {
        "spotify".into()
    } else if lower.contains("chrome") {
        "chrome".into()
    } else if lower.contains("firefox") {
        "firefox".into()
    } else if lower.contains("msedge") || lower.contains("edge") {
        "edge".into()
    } else if lower.contains("brave") {
        "brave".into()
    } else if lower.contains("teams") {
        "teams".into()
    } else if lower.contains("zoom") {
        "zoom".into()
    } else if lower.contains("slack") {
        "slack".into()
    } else if lower.contains("telegram") {
        "telegram".into()
    } else if lower.contains("steam") {
        "steam".into()
    } else if lower.contains("vlc") {
        "vlc".into()
    } else if lower.contains("obs") {
        "obs".into()
    } else {
        "generic".into()
    }
}

/// Enumerate user running applications, filtering system background noise and deduplicating
/// multi-process apps to their root parent PID.
pub fn enumerate_running_applications() -> Result<Vec<AppProcessInfo>> {
    let window_titles = get_window_titles();
    let mut sys = sysinfo::System::new_all();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);

    let noisy_names: HashSet<&'static str> = [
        "svchost.exe",
        "conhost.exe",
        "runtimebroker.exe",
        "registry",
        "system",
        "smss.exe",
        "csrss.exe",
        "wininit.exe",
        "services.exe",
        "lsass.exe",
        "fontdrvhost.exe",
        "dwm.exe",
        "spoolsv.exe",
        "sihost.exe",
        "taskhostw.exe",
        "explorer.exe",
        "searchhost.exe",
        "startmenuexperiencehost.exe",
        "shellexperiencehost.exe",
        "ctfmon.exe",
        "searchindexer.exe",
        "audiodg.exe",
        "wlanext.exe",
        "securityhealthservice.exe",
        "securityhealthsystray.exe",
        "smartscreen.exe",
        "msmpeng.exe",
        "compattelrunner.exe",
        "searchapp.exe",
        "textinputhost.exe",
        "dllhost.exe",
        "systemsettings.exe",
        "applicationframehost.exe",
    ]
    .into_iter()
    .collect();

    let mut grouped: HashMap<String, Vec<(u32, Option<u32>)>> = HashMap::new();

    for (pid, process) in sys.processes() {
        let raw_name = process.name().to_string_lossy().to_string();
        let lower_name = raw_name.to_lowercase();

        if noisy_names.contains(lower_name.as_str()) {
            continue;
        }

        if lower_name.starts_with("system") || lower_name.starts_with("host") {
            continue;
        }

        let pid_u32 = pid.as_u32();
        let parent_u32 = process.parent().map(|p| p.as_u32());

        grouped.entry(raw_name).or_default().push((pid_u32, parent_u32));
    }

    let mut result = Vec::new();

    for (name, procs) in grouped {
        if procs.is_empty() {
            continue;
        }

        let pids_in_group: HashSet<u32> = procs.iter().map(|(p, _)| *p).collect();

        // Choose root PID: prefer process whose parent is NOT inside the same group
        let root_pid = procs
            .iter()
            .find(|(_, parent)| match parent {
                Some(p) => !pids_in_group.contains(p),
                None => true,
            })
            .map(|(p, _)| *p)
            .unwrap_or_else(|| procs.iter().map(|(p, _)| *p).min().unwrap());

        let count = procs.len();
        let icon_hint = detect_icon_hint(&name);

        // Find if any PID in group has a visible window title
        let window_title = procs.iter().find_map(|(p, _)| window_titles.get(p).cloned());
        let has_window = window_title.is_some();

        let title = if let Some(ref wt) = window_title {
            if wt.eq_ignore_ascii_case(&name) || wt.eq_ignore_ascii_case(name.trim_end_matches(".exe")) {
                name.clone()
            } else {
                format!("{} — {}", name.trim_end_matches(".exe"), wt)
            }
        } else if count > 1 {
            format!("{} ({} processes)", name, count)
        } else {
            name.clone()
        };

        result.push(AppProcessInfo {
            pid: root_pid,
            name,
            title,
            window_title,
            has_window,
            icon_hint,
        });
    }

    // Sort: apps with visible windows first, then by name
    result.sort_by(|a, b| {
        b.has_window
            .cmp(&a.has_window)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });

    Ok(result)
}

/// Spawns the Track A physical microphone capture thread.
/// Runs lock-free: reads WASAPI buffer frames and pushes mono f32 samples to rtrb::Producer.
pub fn start_mic_capture_thread(
    device_id: Option<String>,
    mut producer: rtrb::Producer<f32>,
    is_running: Arc<AtomicBool>,
) -> Result<thread::JoinHandle<()>> {
    let handle = thread::Builder::new()
        .name("track-a-mic-capture".into())
        .spawn(move || {
            let _ = wasapi::initialize_mta().ok();

            let enumerator = match DeviceEnumerator::new() {
                Ok(e) => e,
                Err(e) => {
                    error!("Failed to create DeviceEnumerator for Mic: {:?}", e);
                    return;
                }
            };

            let device = if let Some(ref id) = device_id {
                enumerator.get_device(id)
            } else {
                enumerator.get_default_device(&Direction::Capture)
            };

            let device = match device {
                Ok(d) => d,
                Err(e) => {
                    error!("Failed to open Mic device: {:?}", e);
                    return;
                }
            };

            let mut audio_client = match device.get_iaudioclient() {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to get Mic IAudioClient: {:?}", e);
                    return;
                }
            };

            let desired_format = WaveFormat::new(32, 32, &SampleType::Float, 48000, 2, None);
            let blockalign = desired_format.get_blockalign() as usize;

            let (_, min_time) = audio_client.get_device_period().unwrap_or((100000, 50000));
            let mode = StreamMode::EventsShared {
                autoconvert: true,
                buffer_duration_hns: min_time,
            };

            if let Err(e) = audio_client.initialize_client(&desired_format, &Direction::Capture, &mode) {
                error!("Failed to initialize Mic audio client: {:?}", e);
                return;
            }

            let h_event = match audio_client.set_get_eventhandle() {
                Ok(h) => h,
                Err(e) => {
                    error!("Failed to set Mic event handle: {:?}", e);
                    return;
                }
            };

            let capture_client = match audio_client.get_audiocaptureclient() {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to get Mic capture client: {:?}", e);
                    return;
                }
            };

            if let Err(e) = audio_client.start_stream() {
                error!("Failed to start Mic audio stream: {:?}", e);
                return;
            }

            info!("Track A (Mic) capture thread active.");
            let mut sample_queue: VecDeque<u8> = VecDeque::with_capacity(32768);

            while is_running.load(Ordering::Relaxed) {
                let _ = h_event.wait_for_event(40);

                if !is_running.load(Ordering::Relaxed) {
                    break;
                }

                if let Ok(Some(packets)) = capture_client.get_next_packet_size() {
                    if packets > 0 {
                        let _ = capture_client.read_from_device_to_deque(&mut sample_queue);
                    }
                }

                while sample_queue.len() >= blockalign {
                    let b0 = sample_queue.pop_front().unwrap();
                    let b1 = sample_queue.pop_front().unwrap();
                    let b2 = sample_queue.pop_front().unwrap();
                    let b3 = sample_queue.pop_front().unwrap();
                    let left = f32::from_le_bytes([b0, b1, b2, b3]);

                    let right = if blockalign >= 8 {
                        let b4 = sample_queue.pop_front().unwrap();
                        let b5 = sample_queue.pop_front().unwrap();
                        let b6 = sample_queue.pop_front().unwrap();
                        let b7 = sample_queue.pop_front().unwrap();
                        f32::from_le_bytes([b4, b5, b6, b7])
                    } else {
                        left
                    };

                    for _ in 8..blockalign {
                        sample_queue.pop_front();
                    }

                    let mono = (left + right) * 0.5;
                    let _ = producer.push(mono);
                }
            }

            let _ = audio_client.stop_stream();
            info!("Track A (Mic) capture thread terminated.");
        })?;

    Ok(handle)
}

/// Spawns the Track B application / global loopback capture thread.
/// Runs lock-free: reads WASAPI buffer frames and pushes mono f32 samples to rtrb::Producer.
pub fn start_app_capture_thread(
    mode: CaptureMode,
    mut producer: rtrb::Producer<f32>,
    is_running: Arc<AtomicBool>,
) -> Result<thread::JoinHandle<()>> {
    let handle = thread::Builder::new()
        .name("track-b-app-capture".into())
        .spawn(move || {
            let _ = wasapi::initialize_mta().ok();
            let desired_format = WaveFormat::new(32, 32, &SampleType::Float, 48000, 2, None);
            let blockalign = desired_format.get_blockalign() as usize;

            let mut audio_client = match &mode {
                CaptureMode::ApplicationPid(pid) => {
                    info!("Initializing WASAPI Process Loopback for PID {}", pid);
                    match AudioClient::new_application_loopback_client(*pid, true) {
                        Ok(client) => client,
                        Err(e) => {
                            warn!("Process loopback failed for PID {}: {:?}. Falling back to Global Loopback.", pid, e);
                            match get_global_loopback_client() {
                                Ok(c) => c,
                                Err(err) => {
                                    error!("Failed fallback to global loopback: {:?}", err);
                                    return;
                                }
                            }
                        }
                    }
                }
                CaptureMode::GlobalLoopback => {
                    info!("Initializing WASAPI Global Desktop Audio Loopback");
                    match get_global_loopback_client() {
                        Ok(c) => c,
                        Err(err) => {
                            error!("Failed to create global loopback client: {:?}", err);
                            return;
                        }
                    }
                }
            };

            let stream_mode = StreamMode::EventsShared {
                autoconvert: true,
                buffer_duration_hns: 0,
            };

            if let Err(e) = audio_client.initialize_client(&desired_format, &Direction::Capture, &stream_mode) {
                error!("Failed to initialize App audio client: {:?}", e);
                return;
            }

            let h_event = match audio_client.set_get_eventhandle() {
                Ok(h) => h,
                Err(e) => {
                    error!("Failed to set App event handle: {:?}", e);
                    return;
                }
            };

            let capture_client = match audio_client.get_audiocaptureclient() {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to get App capture client: {:?}", e);
                    return;
                }
            };

            if let Err(e) = audio_client.start_stream() {
                error!("Failed to start App audio stream: {:?}", e);
                return;
            }

            info!("Track B (App Loopback) capture thread active.");
            let mut sample_queue: VecDeque<u8> = VecDeque::with_capacity(32768);

            while is_running.load(Ordering::Relaxed) {
                let _ = h_event.wait_for_event(40);

                if !is_running.load(Ordering::Relaxed) {
                    break;
                }

                if let Ok(Some(packets)) = capture_client.get_next_packet_size() {
                    if packets > 0 {
                        let _ = capture_client.read_from_device_to_deque(&mut sample_queue);
                    }
                }

                while sample_queue.len() >= blockalign {
                    let b0 = sample_queue.pop_front().unwrap();
                    let b1 = sample_queue.pop_front().unwrap();
                    let b2 = sample_queue.pop_front().unwrap();
                    let b3 = sample_queue.pop_front().unwrap();
                    let left = f32::from_le_bytes([b0, b1, b2, b3]);

                    let right = if blockalign >= 8 {
                        let b4 = sample_queue.pop_front().unwrap();
                        let b5 = sample_queue.pop_front().unwrap();
                        let b6 = sample_queue.pop_front().unwrap();
                        let b7 = sample_queue.pop_front().unwrap();
                        f32::from_le_bytes([b4, b5, b6, b7])
                    } else {
                        left
                    };

                    for _ in 8..blockalign {
                        sample_queue.pop_front();
                    }

                    let mono = (left + right) * 0.5;
                    let _ = producer.push(mono);
                }
            }

            let _ = audio_client.stop_stream();
            info!("Track B (App Loopback) capture thread terminated.");
        })?;

    Ok(handle)
}

fn get_global_loopback_client() -> Result<AudioClient> {
    let enumerator = DeviceEnumerator::new()
        .map_err(|e| anyhow!("Failed to create DeviceEnumerator: {:?}", e))?;
    let device = enumerator
        .get_default_device(&Direction::Render)
        .map_err(|e| anyhow!("Failed to get default Render device: {:?}", e))?;
    let client = device
        .get_iaudioclient()
        .map_err(|e| anyhow!("Failed to get IAudioClient: {:?}", e))?;
    Ok(client)
}
