//! Proof of concept: extend an X11 desktop onto an OpenDisplay receiver.
//!
//! A disconnected output is enabled with xrandr at the receiver's size. A
//! capture thread grabs it through MIT-SHM when X Damage reports a change and
//! pipes raw frames to ffmpeg (VA-API or libx264) for encoding only, so every
//! access unit carries its real capture time. Input comes back over XTest on its own thread, and the
//! cursor travels as protocol sprite + position (§6.2/6.3), drawn by the
//! receiver. Not a backend: no rotation handling, no reconnect.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::Parser;
use opendisplay_proto::control::{CursorImg, StreamConfig, TouchPhase};
use opendisplay_proto::video::AccessUnit;
use opendisplay_session::Now;
use opendisplay_session::sender::{SenderAction, SenderConfig, SenderEvent, SenderSession};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::mpsc;
use tracing::{info, warn};
use x11rb::connection::Connection;
use x11rb::protocol::Event;
use x11rb::protocol::damage::{self, ConnectionExt as _};
use x11rb::protocol::shm::ConnectionExt as _;
use x11rb::protocol::xfixes::ConnectionExt as _;
use x11rb::protocol::xproto::ImageFormat;
use x11rb::protocol::xtest::ConnectionExt as _;

#[derive(Parser, Debug)]
struct Args {
    /// Receiver address, e.g. 192.168.1.20:9000.
    #[arg(long)]
    connect: SocketAddr,
    /// Disconnected output to turn into the virtual display.
    #[arg(long, default_value = "HDMI-1")]
    output: String,
    /// Divide the receiver's panel pixels by this for the X11 mode (X11 has
    /// no per-output scaling; 1 = native and tiny, 3 = iPhone points).
    #[arg(long, default_value_t = 2.0)]
    div: f64,
    #[arg(long, default_value_t = 60)]
    fps: u32,
    /// Keyframe interval in frames (the encoder cannot be asked for an IDR).
    #[arg(long, default_value_t = 30)]
    gop: u32,
    /// Finger travel, in video pixels, per mouse-wheel notch.
    #[arg(long, default_value_t = 12.0)]
    scroll_step: f64,
    /// Flip the scroll direction.
    #[arg(long)]
    invert_scroll: bool,
    /// Video encoder.
    #[arg(long, value_enum, default_value = "vaapi")]
    encoder: EncoderKind,
    /// Capture every tick instead of only when X reports damage.
    #[arg(long)]
    no_damage: bool,
    /// Stop after this many seconds.
    #[arg(long)]
    duration: Option<f64>,
}

fn wall_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

fn now(start: Instant) -> Now {
    Now::new(start.elapsed(), wall_ms())
}

fn sh(cmd: &str, args: &[&str]) -> Result<String> {
    let out = Command::new(cmd)
        .args(args)
        .output()
        .with_context(|| format!("running {cmd}"))?;
    if !out.status.success() {
        bail!(
            "{cmd} {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

// ---------------------------------------------------------------------------
// Virtual output (xrandr).

#[derive(Clone, Copy, Debug)]
struct Rect {
    x: i32,
    y: i32,
    w: u32,
    h: u32,
}

/// The xrandr output we enabled; turned off and its mode removed on drop.
struct VirtualOutput {
    output: String,
    mode: String,
    rect: Rect,
}

impl VirtualOutput {
    fn create(output: &str, want_w: u32, want_h: u32, fps: u32) -> Result<Self> {
        let cvt = sh(
            "cvt",
            &[&want_w.to_string(), &want_h.to_string(), &fps.to_string()],
        )?;
        let line = cvt
            .lines()
            .find_map(|l| l.strip_prefix("Modeline "))
            .context("cvt printed no modeline")?;
        let mut parts = line.split_whitespace();
        let name = parts
            .next()
            .context("modeline name")?
            .trim_matches('"')
            .to_string();
        let timings: Vec<&str> = parts.collect();
        let (w, h) = name
            .split('_')
            .next()
            .and_then(|s| s.split_once('x'))
            .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
            .context("mode size")?;
        // Right of the rightmost monitor, top-aligned with it.
        let mons = sh("xrandr", &["--listmonitors"])?;
        let (mut x, mut y) = (0i32, 0i32);
        for l in mons.lines().skip(1) {
            // " 1: +DP-1-2 1920/527x1080/296+1920+0  DP-1-2"
            let Some(geom) = l.split_whitespace().nth(2) else {
                continue;
            };
            let mut it = geom.split('+');
            let (Some(size), Some(px), Some(py)) = (it.next(), it.next(), it.next()) else {
                continue;
            };
            let mw: i32 = size.split('/').next().unwrap_or("0").parse().unwrap_or(0);
            let (px, py): (i32, i32) = (px.parse().unwrap_or(0), py.parse().unwrap_or(0));
            if px + mw > x {
                x = px + mw;
                y = py;
            }
        }
        let mut args = vec!["--newmode", name.as_str()];
        args.extend(timings.iter().copied());
        sh("xrandr", &args)?;
        let me = VirtualOutput {
            output: output.into(),
            mode: name.clone(),
            rect: Rect { x, y, w, h },
        };
        sh("xrandr", &["--addmode", output, &name])?;
        sh(
            "xrandr",
            &[
                "--output",
                output,
                "--mode",
                &name,
                "--pos",
                &format!("{x}x{y}"),
            ],
        )?;
        info!("virtual display {output}: {w}x{h} at +{x}+{y}");
        Ok(me)
    }
}

impl Drop for VirtualOutput {
    fn drop(&mut self) {
        let _ = sh("xrandr", &["--output", &self.output, "--off"]);
        let _ = sh("xrandr", &["--delmode", &self.output, &self.mode]);
        let _ = sh("xrandr", &["--rmmode", &self.mode]);
        info!("virtual display {} removed", self.output);
    }
}

// ---------------------------------------------------------------------------
// Capture + encode.

struct Frame {
    unit: AccessUnit,
    cap_ms: f64,
}

enum CursorMsg {
    Image(CursorImg),
    Pos { x: f64, y: f64, visible: bool },
}

#[derive(Default)]
struct CaptureStats {
    frames: u64,
    grab_ms: f64,
    damaged: u64,
    refresh: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum EncoderKind {
    /// Intel/AMD hardware encoder through VA-API.
    Vaapi,
    /// libx264 on the CPU.
    X264,
}

struct Pipeline {
    child: Child,
    stop: Arc<AtomicBool>,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn ffmpeg_command(r: Rect, fps: u32, gop: u32, enc: EncoderKind, device: &str) -> Command {
    let mut c = Command::new("ffmpeg");
    // No input probing (and no -fflags nobuffer, which undoes this): probing reads one frame ahead, which holds every
    // frame back until the next one arrives (fatal with damage-driven capture).
    c.args([
        "-loglevel",
        "error",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
    ]);
    if enc == EncoderKind::Vaapi {
        c.args(["-vaapi_device", device]);
    }
    c.args(["-f", "rawvideo", "-pix_fmt", "bgr0"])
        .args(["-video_size", &format!("{}x{}", r.w, r.h)])
        .args(["-framerate", &fps.to_string(), "-i", "pipe:0"]);
    match enc {
        EncoderKind::Vaapi => {
            // Upload BGRX and convert to NV12 on the GPU, not with swscale.
            c.args(["-vf", "hwupload,scale_vaapi=format=nv12"])
                .args(["-c:v", "h264_vaapi", "-async_depth", "1"])
                .args(["-rc_mode", "VBR", "-b:v", "6M", "-maxrate", "12M"]);
        }
        EncoderKind::X264 => {
            c.args([
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
            ])
            .args(["-pix_fmt", "yuv420p", "-x264-params", "repeat-headers=1"]);
        }
    }
    c.args(["-bf", "0", "-g", &gop.to_string()]).args([
        "-fps_mode",
        "passthrough",
        "-flush_packets",
        "1",
        "-f",
        "flv",
        "-flvflags",
        "no_duration_filesize",
        "pipe:1",
    ]);
    c
}

/// A MIT-SHM segment the X server writes captured frames into.
struct ShmBuf {
    seg: u32,
    ptr: *mut libc::c_void,
    len: usize,
}

// The mapping is only touched from the capture thread.
unsafe impl Send for ShmBuf {}

impl ShmBuf {
    fn new(conn: &impl Connection, len: usize) -> Result<Self> {
        use std::os::fd::AsRawFd;
        conn.shm_query_version()?
            .reply()
            .context("MIT-SHM missing")?;
        let seg = conn.generate_id()?;
        let reply = conn.shm_create_segment(seg, len as u32, false)?.reply()?;
        let fd = reply.shm_fd;
        // SAFETY: mapping a fresh shared segment of exactly `len` bytes.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            bail!("mmap of the shm segment failed");
        }
        Ok(ShmBuf { seg, ptr, len })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: mapped for `len` bytes until drop; the server has finished
        // writing once the shm_get_image reply arrived.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Drop for ShmBuf {
    fn drop(&mut self) {
        // SAFETY: unmapping what new() mapped.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}

fn overlaps(a: &x11rb::protocol::xproto::Rectangle, r: Rect) -> bool {
    let (ax, ay) = (a.x as i32, a.y as i32);
    ax < r.x + r.w as i32
        && ax + a.width as i32 > r.x
        && ay < r.y + r.h as i32
        && ay + a.height as i32 > r.y
}

#[allow(clippy::too_many_arguments)]
fn start_pipeline(
    r: Rect,
    fps: u32,
    gop: u32,
    enc: EncoderKind,
    use_damage: bool,
    want_frames: Arc<AtomicBool>,
    frames: mpsc::Sender<Frame>,
    cursor: mpsc::UnboundedSender<CursorMsg>,
    stats: Arc<Mutex<CaptureStats>>,
) -> Result<Pipeline> {
    let mut child = ffmpeg_command(r, fps, gop, enc, "/dev/dri/renderD128")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .context("starting ffmpeg")?;
    let mut stdin = child.stdin.take().context("ffmpeg stdin")?;
    let mut stdout = child.stdout.take().context("ffmpeg stdout")?;
    let stop = Arc::new(AtomicBool::new(false));
    // Capture times, one per frame written; with no B-frames and no
    // lookahead both encoders emit exactly one access unit per input frame.
    let stamps: Arc<Mutex<VecDeque<f64>>> = Arc::new(Mutex::new(VecDeque::new()));

    let (conn, screen) = x11rb::connect(None).context("connecting to X")?;
    let root = conn.setup().roots[screen].root;
    conn.xfixes_query_version(5, 0)?
        .reply()
        .context("XFixes missing")?;
    let shm = ShmBuf::new(&conn, (r.w * r.h * 4) as usize)?;
    let damage = if use_damage {
        conn.damage_query_version(1, 1)?
            .reply()
            .context("Damage missing")?;
        let d = conn.generate_id()?;
        conn.damage_create(d, root, damage::ReportLevel::BOUNDING_BOX)?;
        conn.flush()?;
        Some(d)
    } else {
        None
    };
    let stop_c = stop.clone();
    let stamps_c = stamps.clone();
    std::thread::spawn(move || {
        let period = Duration::from_secs_f64(1.0 / fps as f64);
        // Safety net when nothing reports damage: a few frames a second.
        let refresh = Duration::from_millis(250);
        let mut next = Instant::now();
        let mut last_frame = Instant::now() - refresh;
        let mut serial = u32::MAX;
        let mut cursor_err_logged = false;
        while !stop_c.load(Ordering::Relaxed) {
            let mut dirty = damage.is_none();
            if let Some(d) = damage {
                while let Ok(Some(ev)) = conn.poll_for_event() {
                    if let Event::DamageNotify(n) = ev {
                        dirty |= overlaps(&n.area, r);
                    }
                }
                // Start every tick from an empty region so the next change
                // inside our rect always raises a fresh notify.
                let _ = conn.damage_subtract(d, x11rb::NONE, x11rb::NONE);
            }
            let forced = want_frames.load(Ordering::Relaxed);
            let stale = last_frame.elapsed() >= refresh;
            if dirty || forced || stale {
                let t0 = Instant::now();
                let cap = wall_ms();
                let ok = conn
                    .shm_get_image(
                        root,
                        r.x as i16,
                        r.y as i16,
                        r.w as u16,
                        r.h as u16,
                        !0,
                        ImageFormat::Z_PIXMAP.into(),
                        shm.seg,
                        0,
                    )
                    .ok()
                    .and_then(|c| c.reply().ok())
                    .is_some();
                if !ok {
                    return;
                }
                let grab = t0.elapsed().as_secs_f64() * 1000.0;
                stamps_c.lock().unwrap().push_back(cap);
                if stdin.write_all(shm.bytes()).is_err() {
                    return;
                }
                last_frame = Instant::now();
                let mut s = stats.lock().unwrap();
                s.frames += 1;
                s.grab_ms += grab;
                if dirty && damage.is_some() {
                    s.damaged += 1;
                } else if !forced {
                    s.refresh += 1;
                }
            }
            // Cursor: position every tick, sprite when it changes.
            let reply = conn
                .xfixes_get_cursor_image()
                .map_err(anyhow::Error::from)
                .and_then(|c| c.reply().map_err(anyhow::Error::from));
            if let Err(e) = &reply {
                if !cursor_err_logged {
                    cursor_err_logged = true;
                    warn!("cursor image unavailable: {e:#}");
                }
            }
            if let Ok(c) = reply {
                if c.cursor_serial != serial && c.width > 0 && c.height > 0 {
                    serial = c.cursor_serial;
                    let rgba = argb_premul_to_rgba(&c.cursor_image);
                    let png = encode_png(&rgba, c.width as u32, c.height as u32);
                    let _ = cursor.send(CursorMsg::Image(CursorImg {
                        nw: c.width as f64 / r.w as f64,
                        nh: c.height as f64 / r.h as f64,
                        ax: c.xhot as f64 / c.width as f64,
                        ay: c.yhot as f64 / c.height as f64,
                        png: base64(&png),
                    }));
                }
                let (px, py) = (c.x as i32 - r.x, c.y as i32 - r.y);
                let visible = px >= 0 && py >= 0 && px < r.w as i32 && py < r.h as i32;
                let _ = cursor.send(CursorMsg::Pos {
                    x: px as f64 / r.w as f64,
                    y: py as f64 / r.h as f64,
                    visible,
                });
            }
            next += period;
            let now = Instant::now();
            if next > now {
                std::thread::sleep(next - now);
            } else {
                next = now; // fell behind: do not try to catch up
            }
        }
    });

    // FLV tags carry their length, so each access unit is handed on the
    // moment its last byte arrives instead of when the next one starts.
    std::thread::spawn(move || {
        let mut flv = FlvReader::default();
        let mut header = [0u8; 13];
        if stdout.read_exact(&mut header).is_err() || &header[..3] != b"FLV" {
            return;
        }
        loop {
            let mut th = [0u8; 11];
            if stdout.read_exact(&mut th).is_err() {
                return;
            }
            let size = u32::from_be_bytes([0, th[1], th[2], th[3]]) as usize;
            let mut data = vec![0u8; size + 4]; // + PreviousTagSize
            if stdout.read_exact(&mut data).is_err() {
                return;
            }
            if th[0] != 9 {
                continue; // not video
            }
            if let Some(unit) = flv.video_tag(&data[..size]) {
                let cap_ms = stamps.lock().unwrap().pop_front().unwrap_or_else(wall_ms);
                if frames.blocking_send(Frame { unit, cap_ms }).is_err() {
                    return;
                }
            }
        }
    });
    Ok(Pipeline { child, stop })
}

/// Turns FLV AVC video tags back into Annex B access units.
#[derive(Default)]
struct FlvReader {
    len_size: usize,
    sps: Vec<Vec<u8>>,
    pps: Vec<Vec<u8>>,
}

impl FlvReader {
    fn video_tag(&mut self, d: &[u8]) -> Option<AccessUnit> {
        if d.len() < 5 || d[0] & 0x0f != 7 {
            return None; // not AVC
        }
        let body = &d[5..];
        match d[1] {
            0 => {
                self.parse_config(body);
                None
            }
            1 => {
                let mut nals: Vec<&[u8]> = Vec::new();
                let mut i = 0;
                let ls = self.len_size.max(1);
                while i + ls <= body.len() {
                    let n = body[i..i + ls]
                        .iter()
                        .fold(0usize, |a, &b| (a << 8) | b as usize);
                    i += ls;
                    if n == 0 || i + n > body.len() {
                        break;
                    }
                    nals.push(&body[i..i + n]);
                    i += n;
                }
                let is_idr = nals.iter().any(|n| n[0] & 0x1f == 5);
                let has_sps = nals.iter().any(|n| n[0] & 0x1f == 7);
                let mut annexb = Vec::with_capacity(body.len() + 64);
                let mut push = |n: &[u8]| {
                    annexb.extend_from_slice(&[0, 0, 0, 1]);
                    annexb.extend_from_slice(n);
                };
                if is_idr && !has_sps {
                    self.sps.iter().chain(self.pps.iter()).for_each(|n| push(n));
                }
                for n in nals {
                    if n[0] & 0x1f != 9 {
                        push(n); // drop AUDs
                    }
                }
                Some(AccessUnit { annexb, is_idr })
            }
            _ => None,
        }
    }

    /// AVCDecoderConfigurationRecord.
    fn parse_config(&mut self, c: &[u8]) {
        if c.len() < 7 {
            return;
        }
        self.len_size = (c[4] & 3) as usize + 1;
        self.sps.clear();
        self.pps.clear();
        let mut i = 6;
        let take = |i: &mut usize| -> Option<Vec<u8>> {
            let n = u16::from_be_bytes([*c.get(*i)?, *c.get(*i + 1)?]) as usize;
            let v = c.get(*i + 2..*i + 2 + n)?.to_vec();
            *i += 2 + n;
            Some(v)
        };
        for _ in 0..(c[5] & 0x1f) {
            if let Some(v) = take(&mut i) {
                self.sps.push(v);
            }
        }
        let Some(&np) = c.get(i) else { return };
        i += 1;
        for _ in 0..np {
            if let Some(v) = take(&mut i) {
                self.pps.push(v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Input (XTest, own thread and connection).

enum Input {
    Move(i16, i16),
    Button(u8, bool),
}

fn start_input() -> Result<std::sync::mpsc::Sender<Input>> {
    let (conn, screen) = x11rb::connect(None).context("connecting to X for input")?;
    let root = conn.setup().roots[screen].root;
    conn.xtest_get_version(2, 2)?
        .reply()
        .context("XTest extension missing")?;
    let (tx, rx) = std::sync::mpsc::channel::<Input>();
    std::thread::spawn(move || {
        const MOTION: u8 = 6;
        const PRESS: u8 = 4;
        const RELEASE: u8 = 5;
        while let Ok(first) = rx.recv() {
            // Coalesce a run of moves into the last one.
            let mut pending = vec![first];
            while let Ok(more) = rx.try_recv() {
                pending.push(more);
            }
            let mut out: Vec<Input> = Vec::with_capacity(pending.len());
            for ev in pending {
                if let (Input::Move(..), Some(Input::Move(..))) = (&ev, out.last()) {
                    out.pop();
                }
                out.push(ev);
            }
            for ev in out {
                let _ = match ev {
                    Input::Move(x, y) => conn.xtest_fake_input(MOTION, 0, 0, root, x, y, 0),
                    Input::Button(b, down) => conn.xtest_fake_input(
                        if down { PRESS } else { RELEASE },
                        b,
                        0,
                        root,
                        0,
                        0,
                        0,
                    ),
                };
            }
            let _ = conn.flush();
        }
    });
    Ok(tx)
}

// ---------------------------------------------------------------------------
// Cursor sprite helpers (same encoding as od-sender's wayland/cursor.rs).

fn argb_premul_to_rgba(px: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(px.len() * 4);
    for &p in px {
        let a = (p >> 24) as u8;
        let un = |c: u32| -> u8 {
            let c = (c & 0xff) as u16;
            if a == 0 {
                0
            } else {
                ((c * 255 + a as u16 / 2) / a as u16).min(255) as u8
            }
        };
        out.extend_from_slice(&[un(p >> 16), un(p >> 8), un(p), a]);
    }
    out
}

fn crc32(parts: &[&[u8]]) -> u32 {
    let mut crc = !0u32;
    for part in parts {
        for &b in *part {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xedb8_8320
                } else {
                    crc >> 1
                };
            }
        }
    }
    !crc
}

fn encode_png(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut raw = Vec::with_capacity((width * height * 4 + height) as usize);
    for row in rgba.chunks_exact(width as usize * 4) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let idat = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 9);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    for (kind, data) in [
        (b"IHDR", &ihdr[..]),
        (b"IDAT", &idat[..]),
        (b"IEND", &[][..]),
    ] {
        png.extend_from_slice(&(data.len() as u32).to_be_bytes());
        png.extend_from_slice(kind);
        png.extend_from_slice(data);
        png.extend_from_slice(&crc32(&[kind, data]).to_be_bytes());
    }
    png
}

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_target(false)
        .init();
    let a = Args::parse();
    let start = Instant::now();

    let mut stream = TcpStream::connect(a.connect)
        .await
        .context("connecting to receiver")?;
    stream.set_nodelay(true)?;
    info!("connected to {}", a.connect);
    let mut sess = SenderSession::new(SenderConfig::default());
    sess.handle(now(start), SenderEvent::Connected);
    let udp = UdpSocket::bind("0.0.0.0:0").await?;
    let mut buf = vec![0u8; 64 * 1024];

    let hello = loop {
        let n = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut buf))
            .await
            .context("no hello within 5 s")??;
        if n == 0 {
            bail!("receiver closed before hello");
        }
        for act in sess.handle(now(start), SenderEvent::Bytes(&buf[..n])) {
            if let SenderAction::Send(b) = act {
                stream.write_all(&b).await?;
            }
        }
        if let Some(h) = sess.hello() {
            break h.clone();
        }
    };
    let (pw, ph) = match &hello.panel {
        Some(p) if p.pixels_wide.is_some() && p.pixels_high.is_some() => {
            (p.pixels_wide.unwrap(), p.pixels_high.unwrap())
        }
        _ => (hello.pixels_wide, hello.pixels_high),
    };
    info!(
        "receiver {:?}: panel {pw}x{ph}, scale {}",
        hello.device, hello.scale
    );
    if pw == 0 || ph == 0 {
        bail!("receiver reported no panel size");
    }
    let want_w = ((pw as f64 / a.div).round() as u32).max(320);
    let want_h = ((ph as f64 / a.div).round() as u32).max(200);

    let vout = VirtualOutput::create(&a.output, want_w, want_h, a.fps)?;
    let r = vout.rect;
    if let SenderAction::Send(b) = sess.stream_config(StreamConfig {
        codec: "h264".into(),
        width: r.w,
        height: r.h,
        frames_per_second: a.fps,
    }) {
        stream.write_all(&b).await?;
    }
    let input = start_input()?;
    let (ftx, mut frx) = mpsc::channel::<Frame>(4);
    let (ctx, mut crx) = mpsc::unbounded_channel::<CursorMsg>();
    let cap_stats = Arc::new(Mutex::new(CaptureStats::default()));
    // Set while the session waits for an IDR: capture every tick so the
    // encoder reaches its next keyframe even on an idle screen.
    let want_frames = Arc::new(AtomicBool::new(true));
    let _pipe = start_pipeline(
        r,
        a.fps,
        a.gop,
        a.encoder,
        !a.no_damage,
        want_frames.clone(),
        ftx,
        ctx,
        cap_stats.clone(),
    )?;
    info!(
        "encoder {:?}, capture {}",
        a.encoder,
        if a.no_damage {
            "every tick"
        } else {
            "on damage"
        }
    );

    let mut tick = tokio::time::interval(Duration::from_millis(100));
    let mut stats = tokio::time::interval(Duration::from_secs(5));
    stats.reset();
    let (mut frames, mut bytes, mut lag_sum) = (0u64, 0u64, 0f64);
    let mut window = Instant::now();
    let mut scroll_acc = (0f64, 0f64);
    let mut pressed = false;
    let mut last_cursor: Option<(f64, f64, bool)> = None;
    let (mut cur_sent, mut cur_img) = (0u64, 0u64);
    let to_screen = |x: f64, y: f64| {
        let sx = r.x + (x.clamp(0.0, 1.0) * (r.w as f64 - 1.0)).round() as i32;
        let sy = r.y + (y.clamp(0.0, 1.0) * (r.h as f64 - 1.0)).round() as i32;
        (sx as i16, sy as i16)
    };
    let deadline = a.duration.map(|d| start + Duration::from_secs_f64(d));

    'outer: loop {
        let actions: Vec<SenderAction> = tokio::select! {
            _ = tokio::signal::ctrl_c() => { info!("interrupted"); break; }
            res = stream.read(&mut buf) => {
                let n = res.context("read")?;
                if n == 0 { info!("receiver closed the connection"); break; }
                sess.handle(now(start), SenderEvent::Bytes(&buf[..n]))
            }
            _ = tick.tick() => {
                if deadline.is_some_and(|d| Instant::now() >= d) { info!("duration reached"); break; }
                sess.handle(now(start), SenderEvent::Tick)
            }
            f = frx.recv() => {
                let Some(f) = f else { warn!("encoder stopped"); break; };
                let n = now(start);
                let out = sess.push_encoded_frame(n, &f.unit.annexb, f.unit.is_idr, Some(f.cap_ms));
                if !out.is_empty() {
                    frames += 1;
                    lag_sum += n.wall_ms - f.cap_ms;
                }
                out
            }
            c = crx.recv() => match c {
                Some(CursorMsg::Image(img)) => { cur_img += 1; vec![sess.cursor_image(img)] }
                Some(CursorMsg::Pos { x, y, visible }) => {
                    // Only send changes; hidden positions are not worth repeating.
                    let cur = (x, y, visible);
                    if last_cursor == Some(cur) || (!visible && last_cursor.is_some_and(|l| !l.2)) {
                        vec![]
                    } else {
                        last_cursor = Some(cur);
                        let out = sess.cursor(now(start), x, y, visible);
                        cur_sent += out.len() as u64;
                        out
                    }
                }
                None => vec![],
            },
            _ = stats.tick() => {
                let el = window.elapsed().as_secs_f64();
                let (cf, gm, dmg, rf) = {
                    let mut s = cap_stats.lock().unwrap();
                    let v = (s.frames, if s.frames > 0 { s.grab_ms / s.frames as f64 } else { 0.0 }, s.damaged, s.refresh);
                    *s = CaptureStats::default();
                    v
                };
                info!(
                    "captured {:.1} fps ({dmg} damage, {rf} refresh; grab {:.1} ms), sent {:.1} fps, {:.1} Mbit/s, capture->send {:.1} ms, cursor {} ({cur_sent} pos msgs, {cur_img} sprites)",
                    cf as f64 / el, gm,
                    frames as f64 / el,
                    bytes as f64 * 8.0 / el / 1e6,
                    if frames > 0 { lag_sum / frames as f64 } else { 0.0 },
                    if sess.cursor_udp_port().is_some() { "udp" } else { "tcp" },
                );
                frames = 0; bytes = 0; lag_sum = 0.0; window = Instant::now();
                vec![]
            }
        };
        for act in actions {
            match act {
                SenderAction::Send(b) => {
                    bytes += b.len() as u64;
                    stream.write_all(&b).await.context("write")?;
                }
                SenderAction::SendCursorDatagram(d) => {
                    if let Some(port) = sess.cursor_udp_port() {
                        let _ = udp.send_to(&d, (a.connect.ip(), port)).await;
                    }
                }
                SenderAction::Touch(t) => {
                    let (sx, sy) = to_screen(t.x, t.y);
                    let _ = input.send(Input::Move(sx, sy));
                    match t.phase {
                        TouchPhase::Began if !pressed => {
                            pressed = true;
                            let _ = input.send(Input::Button(1, true));
                        }
                        TouchPhase::Ended | TouchPhase::Cancelled if pressed => {
                            pressed = false;
                            let _ = input.send(Input::Button(1, false));
                        }
                        _ => {}
                    }
                }
                SenderAction::Scroll(s) => {
                    let sign = if a.invert_scroll { -1.0 } else { 1.0 };
                    scroll_acc.0 += sign * s.dx;
                    scroll_acc.1 += sign * s.dy;
                    // Vertical: 4 up / 5 down. Horizontal: 6 left / 7 right.
                    for (acc, pos, neg) in
                        [(&mut scroll_acc.1, 4u8, 5u8), (&mut scroll_acc.0, 6, 7)]
                    {
                        while acc.abs() >= a.scroll_step {
                            let b = if *acc > 0.0 { pos } else { neg };
                            let _ = input.send(Input::Button(b, true));
                            let _ = input.send(Input::Button(b, false));
                            *acc -= a.scroll_step.copysign(*acc);
                        }
                    }
                }
                SenderAction::Stats(v) => info!("phone stats {v}"),
                SenderAction::RequestKeyframe => {
                    info!("keyframe requested (next within {} frames)", a.gop)
                }
                SenderAction::ReceiverSleeping => info!("receiver sleeping"),
                SenderAction::ReceiverClosing => info!("receiver closing"),
                SenderAction::Close(why) => {
                    warn!("closing: {why:?}");
                    break 'outer;
                }
                SenderAction::Warn(m) => warn!("{m}"),
                _ => {}
            }
        }
        want_frames.store(sess.needs_idr(), Ordering::Relaxed);
        if !sess.is_connected() {
            break;
        }
    }
    if pressed {
        let _ = input.send(Input::Button(1, false));
    }
    // Give the input thread a moment to flush the release.
    tokio::time::sleep(Duration::from_millis(50)).await;
    Ok(())
}
