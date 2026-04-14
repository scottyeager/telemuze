use std::sync::mpsc;

use ksni::blocking::{Handle, TrayMethods};
use ksni::menu::StandardItem;
use ksni::{Icon, MenuItem};

use super::TrayStatus;
use crate::{Event, IpcCommand};

const ICON_SIZE: i32 = 48;

const ICON_IDLE: &[u8] = include_bytes!("../../assets/idle_lg.rgba");
const ICON_SLEEPING: &[u8] = include_bytes!("../../assets/sleeping_lg.rgba");
const ICON_LISTENING: &[u8] = include_bytes!("../../assets/listening_lg.rgba");
const ICON_DICTATING: &[u8] = include_bytes!("../../assets/dictating_lg.rgba");
const ICON_RECORDING: &[u8] = include_bytes!("../../assets/recording_lg.rgba");
const ICON_PROCESSING: &[u8] = include_bytes!("../../assets/processing_lg.rgba");
const ICON_RECORDING_PROCESSING: &[u8] = include_bytes!("../../assets/recording_processing_lg.rgba");

fn icon_rgba(status: TrayStatus) -> &'static [u8] {
    match status {
        TrayStatus::Idle => ICON_IDLE,
        TrayStatus::Sleeping => ICON_SLEEPING,
        TrayStatus::Listening => ICON_LISTENING,
        TrayStatus::Dictating => ICON_DICTATING,
        TrayStatus::Recording => ICON_RECORDING,
        TrayStatus::Processing => ICON_PROCESSING,
        TrayStatus::RecordingProcessing => ICON_RECORDING_PROCESSING,
    }
}

// ARGB32, network byte order: [A, R, G, B] per pixel.
fn rgba_to_argb(rgba: &[u8]) -> Vec<u8> {
    let mut out = rgba.to_vec();
    for p in out.chunks_exact_mut(4) {
        p.rotate_right(1);
    }
    out
}

struct TelemuzeTray {
    status: TrayStatus,
    tx: mpsc::SyncSender<Event>,
}

impl TelemuzeTray {
    fn send(&self, cmd: IpcCommand) {
        let _ = self.tx.send(Event::Ipc(cmd));
    }
}

impl ksni::Tray for TelemuzeTray {
    fn id(&self) -> String {
        "telemuze".into()
    }

    fn title(&self) -> String {
        "Telemuze".into()
    }

    fn icon_pixmap(&self) -> Vec<Icon> {
        vec![Icon {
            width: ICON_SIZE,
            height: ICON_SIZE,
            data: rgba_to_argb(icon_rgba(self.status)),
        }]
    }

    fn activate(&mut self, _x: i32, _y: i32) {
        self.send(IpcCommand::Toggle);
    }

    fn menu(&self) -> Vec<MenuItem<Self>> {
        vec![
            StandardItem {
                label: "Copy Last".into(),
                activate: Box::new(|this: &mut Self| this.send(IpcCommand::CopyLast)),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: "Toggle".into(),
                activate: Box::new(|this: &mut Self| this.send(IpcCommand::Toggle)),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: "Exit".into(),
                activate: Box::new(|this: &mut Self| this.send(IpcCommand::Stop)),
                ..Default::default()
            }
            .into(),
        ]
    }
}

#[derive(Clone)]
pub struct SniHandle {
    handle: Handle<TelemuzeTray>,
}

impl SniHandle {
    pub fn update(&self, status: TrayStatus) {
        self.handle.update(move |t| {
            if t.status != status {
                t.status = status;
            }
        });
    }

    pub fn shutdown(&self) {
        let _ = self.handle.shutdown();
    }
}

pub fn spawn(
    tx: mpsc::SyncSender<Event>,
    initial: TrayStatus,
) -> anyhow::Result<SniHandle> {
    let tray = TelemuzeTray { status: initial, tx };
    let handle = tray
        .spawn()
        .map_err(|e| anyhow::anyhow!("SNI tray: {e}"))?;
    Ok(SniHandle { handle })
}
