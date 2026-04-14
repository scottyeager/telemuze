mod sni;
mod x11;

use std::sync::mpsc;

use tracing::warn;

#[derive(Clone, Copy, PartialEq)]
pub enum TrayStatus {
    Idle,
    Sleeping,
    Listening,
    Dictating,
    Recording,
    Processing,
    RecordingProcessing,
}

#[derive(Clone)]
pub enum TrayHandle {
    X11(x11::X11Handle),
    Sni(sni::SniHandle),
}

impl TrayHandle {
    pub fn update(&self, status: TrayStatus) {
        match self {
            Self::X11(h) => h.update(status),
            Self::Sni(h) => h.update(status),
        }
    }

    pub fn shutdown(&self) {
        match self {
            Self::X11(h) => h.shutdown(),
            Self::Sni(h) => h.shutdown(),
        }
    }
}

pub fn spawn_tray(
    tx: mpsc::SyncSender<crate::Event>,
    initial: TrayStatus,
) -> anyhow::Result<TrayHandle> {
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
    let x11_present = std::env::var_os("DISPLAY").is_some();

    if wayland {
        match sni::spawn(tx.clone(), initial) {
            Ok(h) => return Ok(TrayHandle::Sni(h)),
            Err(e) if x11_present => {
                warn!("SNI tray unavailable ({e}); falling back to X11");
            }
            Err(e) => return Err(e),
        }
    }

    x11::spawn(tx, initial).map(TrayHandle::X11)
}
