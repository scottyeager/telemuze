use std::sync::mpsc;
use std::sync::Mutex;
use x11rb::connection::Connection;
use x11rb::protocol::xproto::*;
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;
use x11rb::{atom_manager, COPY_DEPTH_FROM_PARENT};

use crate::IpcCommand;

const SYSTEM_TRAY_REQUEST_DOCK: u32 = 0;
const XEMBED_MAPPED: u32 = 1;
const ICON_SIZE: u16 = 24;

atom_manager! {
    pub Atoms: AtomsCookie {
        _NET_SYSTEM_TRAY_OPCODE,
        _NET_SYSTEM_TRAY_S0,
        _XEMBED_INFO,
        MANAGER,
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum TrayStatus {
    Idle,
    Listening,
    Recording,
    Processing,
}

/// RGB for each status.
fn status_rgb(status: TrayStatus) -> (u16, u16, u16) {
    match status {
        TrayStatus::Idle => (0x88, 0x88, 0x88),      // gray
        TrayStatus::Listening => (0x4C, 0xAF, 0x50), // green
        TrayStatus::Recording => (0xF4, 0x43, 0x36), // red
        TrayStatus::Processing => (0x21, 0x96, 0xF3), // blue
    }
}

/// Allocate an X color pixel value.
fn alloc_pixel(
    conn: &RustConnection,
    colormap: Colormap,
    r: u16,
    g: u16,
    b: u16,
) -> Result<u32, Box<dyn std::error::Error>> {
    let reply = conn
        .alloc_color(colormap, r * 257, g * 257, b * 257)?
        .reply()?;
    Ok(reply.pixel)
}

/// Draw the status icon into a pixmap, then copy to window.
fn draw_icon(
    conn: &RustConnection,
    pixmap: Pixmap,
    win: Window,
    gc: Gcontext,
    bg_pixel: u32,
    fg_pixel: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Fill background
    conn.change_gc(gc, &ChangeGCAux::new().foreground(bg_pixel))?;
    conn.poly_fill_rectangle(
        pixmap,
        gc,
        &[Rectangle {
            x: 0,
            y: 0,
            width: ICON_SIZE,
            height: ICON_SIZE,
        }],
    )?;

    // Draw filled circle
    conn.change_gc(gc, &ChangeGCAux::new().foreground(fg_pixel))?;
    let margin: i16 = 4;
    let diameter = ICON_SIZE as i16 - margin * 2;
    conn.poly_fill_arc(
        pixmap,
        gc,
        &[Arc {
            x: margin,
            y: margin,
            width: diameter as u16,
            height: diameter as u16,
            angle1: 0,
            angle2: 360 * 64,
        }],
    )?;

    // Copy pixmap to window (no expose generation)
    conn.copy_area(pixmap, win, gc, 0, 0, 0, 0, ICON_SIZE, ICON_SIZE)?;
    conn.flush()?;
    Ok(())
}

pub struct TrayHandle {
    status: std::sync::Arc<Mutex<TrayStatus>>,
    /// Signal the tray thread to repaint.
    repaint_conn: std::sync::Arc<RustConnection>,
    win: Window,
}

impl Clone for TrayHandle {
    fn clone(&self) -> Self {
        Self {
            status: std::sync::Arc::clone(&self.status),
            repaint_conn: std::sync::Arc::clone(&self.repaint_conn),
            win: self.win,
        }
    }
}

impl TrayHandle {
    pub fn update(&self, status: TrayStatus) {
        let mut s = self.status.lock().unwrap();
        if *s != status {
            *s = status;
            drop(s);
            // Send a synthetic expose event to wake the tray thread
            let _ = self.repaint_conn.send_event(
                false,
                self.win,
                EventMask::EXPOSURE,
                ExposeEvent {
                    response_type: x11rb::protocol::xproto::EXPOSE_EVENT,
                    sequence: 0,
                    window: self.win,
                    x: 0,
                    y: 0,
                    width: ICON_SIZE,
                    height: ICON_SIZE,
                    count: 0,
                },
            );
            let _ = self.repaint_conn.flush();
        }
    }

    pub fn shutdown(&self) {
        let _ = self.repaint_conn.destroy_window(self.win);
        let _ = self.repaint_conn.flush();
    }
}

pub fn spawn_tray(
    tx: mpsc::SyncSender<crate::Event>,
    initial: TrayStatus,
) -> anyhow::Result<TrayHandle> {
    let (conn, screen_num) =
        RustConnection::connect(None).map_err(|e| anyhow::anyhow!("X11 connect: {e}"))?;
    let screen = conn.setup().roots[screen_num].clone();
    let atoms = Atoms::new(&conn)?.reply()?;

    // Find the system tray manager
    let tray_owner = conn
        .get_selection_owner(atoms._NET_SYSTEM_TRAY_S0)?
        .reply()?
        .owner;
    if tray_owner == x11rb::NONE {
        anyhow::bail!("no system tray manager found");
    }

    // Create the icon window
    let win = conn.generate_id()?;
    conn.create_window(
        COPY_DEPTH_FROM_PARENT,
        win,
        screen.root,
        0,
        0,
        ICON_SIZE,
        ICON_SIZE,
        0,
        WindowClass::INPUT_OUTPUT,
        screen.root_visual,
        &CreateWindowAux::default()
            .background_pixel(screen.black_pixel)
            .event_mask(
                EventMask::EXPOSURE
                    | EventMask::STRUCTURE_NOTIFY
                    | EventMask::BUTTON_PRESS,
            ),
    )?;

    // Create an off-screen pixmap for double buffering
    let pixmap = conn.generate_id()?;
    conn.create_pixmap(
        screen.root_depth,
        pixmap,
        win,
        ICON_SIZE,
        ICON_SIZE,
    )?;

    // Set _XEMBED_INFO: version 0, flags XEMBED_MAPPED
    conn.change_property32(
        PropMode::REPLACE,
        win,
        atoms._XEMBED_INFO,
        atoms._XEMBED_INFO,
        &[0, XEMBED_MAPPED],
    )?;

    // Send SYSTEM_TRAY_REQUEST_DOCK to the tray manager
    conn.send_event(
        false,
        tray_owner,
        EventMask::NO_EVENT,
        ClientMessageEvent {
            response_type: CLIENT_MESSAGE_EVENT,
            format: 32,
            sequence: 0,
            window: tray_owner,
            type_: atoms._NET_SYSTEM_TRAY_OPCODE,
            data: ClientMessageData::from([
                x11rb::CURRENT_TIME,
                SYSTEM_TRAY_REQUEST_DOCK,
                win,
                0u32,
                0u32,
            ]),
        },
    )?;
    conn.flush()?;

    // Create a GC for drawing
    let gc = conn.generate_id()?;
    conn.create_gc(gc, win, &CreateGCAux::default())?;

    // Pre-allocate color pixels for all statuses
    let colormap = screen.default_colormap;
    let bg_pixel = screen.black_pixel;
    let pixels: Vec<(TrayStatus, u32)> = [
        TrayStatus::Idle,
        TrayStatus::Listening,
        TrayStatus::Recording,
        TrayStatus::Processing,
    ]
    .iter()
    .map(|&s| {
        let (r, g, b) = status_rgb(s);
        let px = alloc_pixel(&conn, colormap, r, g, b).unwrap_or(screen.white_pixel);
        (s, px)
    })
    .collect();

    // Open a second connection for sending repaint signals from other threads
    let (repaint_conn, _) =
        RustConnection::connect(None).map_err(|e| anyhow::anyhow!("X11 connect: {e}"))?;

    let status = std::sync::Arc::new(Mutex::new(initial));
    let handle = TrayHandle {
        status: std::sync::Arc::clone(&status),
        repaint_conn: std::sync::Arc::new(repaint_conn),
        win,
    };

    // Spawn the event loop thread
    std::thread::spawn(move || {
        let pixel_for = |s: TrayStatus| -> u32 {
            pixels
                .iter()
                .find(|(st, _)| *st == s)
                .map(|(_, px)| *px)
                .unwrap_or(screen.white_pixel)
        };

        loop {
            let event = match conn.wait_for_event() {
                Ok(e) => e,
                Err(_) => break,
            };
            match event {
                Event::Expose(ev) if ev.count == 0 => {
                    let s = *status.lock().unwrap();
                    let _ = draw_icon(&conn, pixmap, win, gc, bg_pixel, pixel_for(s));
                }
                Event::ButtonPress(ev) => {
                    if ev.detail == 1 {
                        let _ = tx.send(crate::Event::Ipc(IpcCommand::Toggle));
                    } else if ev.detail == 3 {
                        let _ = tx.send(crate::Event::Ipc(IpcCommand::Stop));
                    }
                }
                Event::DestroyNotify(ev) if ev.window == win => {
                    break;
                }
                _ => {}
            }
        }
    });

    Ok(handle)
}
