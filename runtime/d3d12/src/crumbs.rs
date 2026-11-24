//! LOCAL: GPU progress markers around compute work. After a device loss, the
//! entries whose start marker landed but whose end marker did not were running.
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

const N: u32 = 1 << 17;

struct Info {
    seq: u32,
    list: usize,
    kind: i32,
    what: String,
    detail: String,
}

struct Ring {
    base: usize,
    va: u64,
    next: AtomicU32,
    info: Mutex<Vec<Option<Info>>>,
    _buffer: ID3D12Resource,
}

static RING: OnceLock<Option<Ring>> = OnceLock::new();
static SHADERS: LazyLock<Mutex<HashMap<usize, String>>> = LazyLock::new(Default::default);

pub(crate) fn name(pipeline: *mut core::ffi::c_void, what: String) {
    SHADERS.lock().unwrap_or_else(|e| e.into_inner()).insert(pipeline as usize, what);
}

pub(crate) fn shader(pipeline: *mut core::ffi::c_void) -> String {
    SHADERS.lock().unwrap_or_else(|e| e.into_inner()).get(&(pipeline as usize)).cloned().unwrap_or_else(|| "?".into())
}

/// Name each stand-in raster pipeline the title binds, once.
pub(crate) fn null_bound(pipeline: *mut core::ffi::c_void) {
    static SEEN: LazyLock<Mutex<std::collections::HashSet<String>>> = LazyLock::new(Default::default);
    let name = shader(pipeline);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() < 400 && seen.insert(name.clone()) { crate::diagnostics::log(&format!("NULL-BOUND {name}")); }
}

pub(crate) fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("TUXBOX_CRUMBS").is_some());
    *ON
}
fn ring(list: &ID3D12GraphicsCommandList) -> Option<&'static Ring> {
    if !on() { return None; }
    RING.get_or_init(|| unsafe {
        let mut device: Option<ID3D12Device> = None;
        list.GetDevice(&mut device).ok()?;
        let device = device?;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: u64::from(N) * 8, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer).ok()?;
        let buffer = buffer?;
        let mut base = core::ptr::null_mut();
        buffer.Map(0, None, Some(&mut base)).ok()?;
        core::ptr::write_bytes(base.cast::<u8>(), 0, N as usize * 8);
        let watched = device.as_raw() as usize;
        core::mem::forget(device.clone());
        std::thread::spawn(move || {
            let raw = watched as *mut core::ffi::c_void;
            let device = ID3D12Device::from_raw_borrowed(&raw).expect("device");
            let mut seen = HashMap::new();
            for round in 0u32.. {
                std::thread::sleep(std::time::Duration::from_millis(250));
                if device.GetDeviceRemovedReason().is_err() { dump(); return; }
                if round % 2 == 0 { poll(&mut seen); }
            }
        });
        crate::diagnostics::log("CRUMBS markers on");
        Some(Ring {
            base: base as usize, va: buffer.GetGPUVirtualAddress(), next: AtomicU32::new(0),
            info: Mutex::new((0..N).map(|_| None).collect()), _buffer: buffer,
        })
    }).as_ref()
}

/// Extra facts about a started piece of work, shown only if it never finishes.
pub(crate) fn detail(seq: Option<u32>, text: impl FnOnce() -> String) {
    let (Some(seq), Some(Some(ring))) = (seq, RING.get()) else { return };
    if let Some(entry) = &mut ring.info.lock().unwrap_or_else(|e| e.into_inner())[(seq % N) as usize] && entry.seq == seq {
        entry.detail = text();
    }
}

fn mark(list: &ID3D12GraphicsCommandList, dest: u64, value: u32, mode: D3D12_WRITEBUFFERIMMEDIATE_MODE) {
    if let Ok(list) = list.cast::<ID3D12GraphicsCommandList2>() {
        let parameter = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: dest, Value: value };
        unsafe { list.WriteBufferImmediate(1, &parameter, Some(&mode)) };
    }
}

/// Marks the start of one piece of work; pass the result to `end`.
pub(crate) fn begin(list: &ID3D12GraphicsCommandList, what: String) -> Option<u32> {
    let ring = ring(list)?;
    let seq = ring.next.fetch_add(1, Ordering::Relaxed) + 1;
    let slot = seq % N;
    ring.info.lock().unwrap_or_else(|e| e.into_inner())[slot as usize] =
        Some(Info { seq, list: list.as_raw() as usize, kind: unsafe { list.GetType() }.0, what, detail: String::new() });
    mark(list, ring.va + u64::from(slot) * 8, seq, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN);
    Some(seq)
}

pub(crate) fn end(list: &ID3D12GraphicsCommandList, seq: Option<u32>) {
    let (Some(seq), Some(Some(ring))) = (seq, RING.get()) else { return };
    mark(list, ring.va + u64::from(seq % N) * 8 + 4, seq, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT);
}

/// Logs work that has been running for several polls.
fn poll(seen: &mut HashMap<u32, u32>) {
    let Some(Some(ring)) = RING.get() else { return };
    let info = ring.info.lock().unwrap_or_else(|e| e.into_inner());
    let word = |slot: usize, part: usize| unsafe { ((ring.base + slot * 8 + part * 4) as *const u32).read_volatile() };
    let mut now = HashMap::new();
    let mut stuck = false;
    for (slot, entry) in info.iter().enumerate() {
        let Some(entry) = entry else { continue };
        if word(slot, 0) != entry.seq || word(slot, 1) == entry.seq { continue; }
        let polls = seen.get(&entry.seq).copied().unwrap_or(0) + 1;
        now.insert(entry.seq, polls);
        if polls == 2 {
            crate::diagnostics::log(&format!("CRUMBS stalled #{} list={:x} kind={} {} {}", entry.seq, entry.list, entry.kind, entry.what, entry.detail));
            if crate::viewkinds::wanted(entry.what.split("CS=").nth(1).unwrap_or("")) { stuck = true; }
        }
    }
    *seen = now;
    drop(info);
    if stuck { crate::viewkinds::stalled(); }
}

// Trace: a file `trace` in the audit directory writes everything recorded during
// the next two frames, with the values of indirect arguments, to trace.txt.
const ARGS: u32 = 1 << 15;
struct Values { base: usize, buffer: ID3D12Resource, next: AtomicU32 }
static VALUES: OnceLock<Option<Values>> = OnceLock::new();
static TRACING: AtomicU32 = AtomicU32::new(0);
static FIRST: AtomicU32 = AtomicU32::new(0);
static LAST: AtomicU32 = AtomicU32::new(0);

/// While tracing, copies the count and the first three words of indirect arguments.
pub(crate) fn arguments(list: &ID3D12GraphicsCommandList, args: &ID3D12Resource, offset: u64, count: Option<&ID3D12Resource>, count_offset: u64) -> u32 {
    if !matches!(TRACING.load(Ordering::Relaxed), 1..=2) { return 0; }
    let Some(values) = VALUES.get_or_init(|| unsafe {
        let mut device: Option<ID3D12Device> = None;
        list.GetDevice(&mut device).ok()?;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: u64::from(ARGS) * 16, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        device?.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer).ok()?;
        let buffer = buffer?;
        let mut base = core::ptr::null_mut();
        buffer.Map(0, None, Some(&mut base)).ok()?;
        core::ptr::write_bytes(base.cast::<u8>(), 0xfe, ARGS as usize * 16);
        Some(Values { base: base as usize, buffer, next: AtomicU32::new(0) })
    }) else { return 0 };
    let n = values.next.fetch_add(1, Ordering::Relaxed) + 1;
    if n >= ARGS { return 0; }
    unsafe {
        let d = args.GetDesc().Width;
        list.CopyBufferRegion(&values.buffer, u64::from(n) * 16 + 4, args, offset, 12.min(d - offset));
        if let Some(count) = count { list.CopyBufferRegion(&values.buffer, u64::from(n) * 16, count, count_offset, 4); }
    }
    n
}

pub(crate) fn trace(dir: &std::path::Path) {
    let Some(Some(ring)) = RING.get() else { return };
    let stage = TRACING.load(Ordering::Relaxed);
    if stage == 0 {
        if std::fs::rename(dir.join("trace"), dir.join("tracing")).is_err() { return; }
        FIRST.store(ring.next.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    if stage == 2 { LAST.store(ring.next.load(Ordering::Relaxed), Ordering::Relaxed); }
    if stage < 6 { TRACING.store(stage + 1, Ordering::Relaxed); return; }
    TRACING.store(0, Ordering::Relaxed);
    let (first, last) = (FIRST.load(Ordering::Relaxed), LAST.load(Ordering::Relaxed));
    let info = ring.info.lock().unwrap_or_else(|e| e.into_inner());
    let mut entries: Vec<&Info> = info.iter().flatten().filter(|e| e.seq > first && e.seq <= last).collect();
    entries.sort_by_key(|e| e.seq);
    let mut text = String::new();
    for e in entries {
        let done = unsafe { ((ring.base + (e.seq % N) as usize * 8 + 4) as *const u32).read_volatile() } == e.seq;
        let args = e.what.split("INDIRECT@").nth(1).and_then(|t| t.split(' ').next()?.parse::<usize>().ok()).filter(|n| *n != 0);
        let words = args.and_then(|n| Some((VALUES.get()?.as_ref()?.base, n))).map_or(String::new(), |(base, n)|
            format!("{:x?}", unsafe { core::ptr::read_volatile((base + n * 16) as *const [u32; 4]) }));
        text += &format!("{} {:x} {} {} {words} {} {}\n", e.seq, e.list, e.kind, if done { "done" } else { "-" }, e.what, e.detail);
    }
    let _ = std::fs::write(dir.join("trace.txt"), text);
    let _ = std::fs::remove_file(dir.join("tracing"));
}

pub(crate) fn dump() {
    let Some(Some(ring)) = RING.get() else { return };
    let info = ring.info.lock().unwrap_or_else(|e| e.into_inner());
    let word = |slot: usize, part: usize| unsafe { ((ring.base + slot * 8 + part * 4) as *const u32).read_volatile() };
    let (mut open, mut done, mut waiting): (Vec<&Info>, Vec<&Info>, usize) = Default::default();
    for (slot, entry) in info.iter().enumerate() {
        let Some(entry) = entry else { continue };
        match (word(slot, 0) == entry.seq, word(slot, 1) == entry.seq) {
            (true, false) => open.push(entry),
            (true, true) => done.push(entry),
            _ => waiting += 1,
        }
    }
    open.sort_by_key(|e| e.seq);
    done.sort_by_key(|e| e.seq);
    crate::diagnostics::log(&format!("CRUMBS after device loss: running={} finished={} not started={waiting}", open.len(), done.len()));
    for e in open.iter().take(60) {
        crate::diagnostics::log(&format!("CRUMBS running #{} list={:x} kind={} {} {}", e.seq, e.list, e.kind, e.what, e.detail));
    }
    for e in done.iter().rev().take(25) {
        crate::diagnostics::log(&format!("CRUMBS finished #{} list={:x} kind={} {}", e.seq, e.list, e.kind, e.what));
    }
    // What was queued right after the running work, in the same lists.
    for e in open.iter().take(8) {
        for next in info.iter().flatten().filter(|n| n.list == e.list && n.seq > e.seq && n.seq < e.seq + 400).take(3) {
            crate::diagnostics::log(&format!("CRUMBS after #{}: #{} {}", e.seq, next.seq, next.what));
        }
    }
}
