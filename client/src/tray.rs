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

// ── Software-rendered icons ───────────────────────────────────────────────

/// RGB color for each status.
fn status_rgb(status: TrayStatus) -> (u8, u8, u8) {
    match status {
        TrayStatus::Idle => (0x88, 0x88, 0x88),
        TrayStatus::Listening => (0x4C, 0xAF, 0x50),
        TrayStatus::Recording => (0xF4, 0x43, 0x36),
        TrayStatus::Processing => (0x42, 0xA5, 0xF5),
    }
}

/// Blend foreground onto background with alpha (0.0–1.0).
fn blend(fg: u8, bg: u8, a: f32) -> u8 {
    (fg as f32 * a + bg as f32 * (1.0 - a)) as u8
}

/// Smoothstep for anti-aliased edges: returns 1.0 inside, 0.0 outside,
/// smooth gradient over ~1px at the boundary.
fn aa_fill(dist_from_edge: f32) -> f32 {
    (dist_from_edge + 0.5).clamp(0.0, 1.0)
}

/// Smoothstep for a ring/stroke: 1.0 on the stroke, 0.0 elsewhere.
fn aa_ring(dist_from_center: f32, radius: f32, half_width: f32) -> f32 {
    let outer = aa_fill(radius + half_width - dist_from_center);
    let inner = aa_fill(dist_from_center - (radius - half_width));
    outer * inner
}

/// Render icon to a BGRA pixel buffer (X11 ZPixmap, little-endian 32bpp).
/// Renders at the given width/height so the icon is always perfectly centered.
fn render_icon(status: TrayStatus, bg: (u8, u8, u8), w: u16, h: u16) -> Vec<u8> {
    let fw = w as f32;
    let fh = h as f32;
    let cx = fw / 2.0;
    let cy = fh / 2.0;
    // Use the smaller axis so the icon fits
    let s = fw.min(fh);
    let (fr, fg, fb) = status_rgb(status);
    let (bgr, bgg, bgb) = bg;
    let mut data = Vec::with_capacity((w as usize) * (h as usize) * 4);

    for y in 0..h {
        for x in 0..w {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let dx = px - cx;
            let dy = py - cy;
            let dist = (dx * dx + dy * dy).sqrt();

            let a = match status {
                TrayStatus::Listening => {
                    let radius = s / 2.0 - 3.0;
                    aa_fill(radius - dist)
                }
                TrayStatus::Recording => {
                    let radius = s / 2.0 - 2.0;
                    aa_fill(radius - dist)
                }
                TrayStatus::Idle => {
                    let radius = s / 2.0 - 4.0;
                    aa_ring(dist, radius, 1.8)
                }
                TrayStatus::Processing => {
                    let outer = aa_ring(dist, s / 2.0 - 3.5, 1.5);
                    let inner = aa_fill(2.5 - dist);
                    1.0 - (1.0 - outer) * (1.0 - inner)
                }
            };

            let r = blend(fr, bgr, a);
            let g = blend(fg, bgg, a);
            let b = blend(fb, bgb, a);

            // X11 ZPixmap 32bpp little-endian: B G R pad
            data.push(b);
            data.push(g);
            data.push(r);
            data.push(0xFF);
        }
    }
    data
}

// ── X11 tray handle ──────────────────────────────────────────────────────

pub struct TrayHandle {
    status: std::sync::Arc<Mutex<TrayStatus>>,
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

// ── Spawn tray ───────────────────────────────────────────────────────────

pub fn spawn_tray(
    tx: mpsc::SyncSender<crate::Event>,
    initial: TrayStatus,
) -> anyhow::Result<TrayHandle> {
    let (conn, screen_num) =
        RustConnection::connect(None).map_err(|e| anyhow::anyhow!("X11 connect: {e}"))?;
    let screen = conn.setup().roots[screen_num].clone();
    let atoms = Atoms::new(&conn)?.reply()?;
    let depth = screen.root_depth;

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

    // Pixmap for double buffering — will be recreated on resize
    let pixmap = conn.generate_id()?;
    conn.create_pixmap(depth, pixmap, win, ICON_SIZE, ICON_SIZE)?;

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

    let bg = (0u8, 0u8, 0u8);

    // Open a second connection for sending repaint signals from other threads
    let (repaint_conn, _) =
        RustConnection::connect(None).map_err(|e| anyhow::anyhow!("X11 connect: {e}"))?;

    let status = std::sync::Arc::new(Mutex::new(initial));
    let handle = TrayHandle {
        status: std::sync::Arc::clone(&status),
        repaint_conn: std::sync::Arc::new(repaint_conn),
        win,
    };

    std::thread::spawn(move || {
        // Track actual window size (tray manager may resize us)
        let mut win_w = ICON_SIZE;
        let mut win_h = ICON_SIZE;
        let mut cur_pixmap = pixmap;
        let mut pm_w = ICON_SIZE;
        let mut pm_h = ICON_SIZE;

        let ensure_pixmap =
            |cur: &mut Pixmap,
             pw: &mut u16,
             ph: &mut u16,
             ww: u16,
             wh: u16|
             -> Result<(), Box<dyn std::error::Error>> {
                if ww != *pw || wh != *ph {
                    conn.free_pixmap(*cur)?;
                    let new_pm = conn.generate_id()?;
                    conn.create_pixmap(depth, new_pm, win, ww, wh)?;
                    *cur = new_pm;
                    *pw = ww;
                    *ph = wh;
                }
                Ok(())
            };

        loop {
            let event = match conn.wait_for_event() {
                Ok(e) => e,
                Err(_) => break,
            };
            match event {
                Event::ConfigureNotify(ev) => {
                    win_w = ev.width;
                    win_h = ev.height;
                }
                Event::Expose(ev) if ev.count == 0 => {
                    let s = *status.lock().unwrap();
                    let _ = ensure_pixmap(
                        &mut cur_pixmap,
                        &mut pm_w,
                        &mut pm_h,
                        win_w,
                        win_h,
                    );
                    let data = render_icon(s, bg, win_w, win_h);
                    let _ = conn.put_image(
                        ImageFormat::Z_PIXMAP,
                        cur_pixmap,
                        gc,
                        win_w,
                        win_h,
                        0,
                        0,
                        0,
                        depth,
                        &data,
                    );
                    let _ = conn.copy_area(
                        cur_pixmap, win, gc, 0, 0, 0, 0, win_w, win_h,
                    );
                    let _ = conn.flush();
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
