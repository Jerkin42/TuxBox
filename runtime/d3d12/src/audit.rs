//! LOCAL: dump every GPU-written image once, a few per present. Put a file
//! named `go` in TUXBOX_AUDIT to start; it is renamed when the dump begins.
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

#[derive(Clone, Copy)]
struct Entry { address: u64, kind: &'static str }
static LIVE: Mutex<BTreeMap<usize, Entry>> = Mutex::new(BTreeMap::new());
static QUEUE: Mutex<VecDeque<(usize, Entry)>> = Mutex::new(VecDeque::new());
static DIR: LazyLock<Option<PathBuf>> = LazyLock::new(|| std::env::var_os("TUXBOX_AUDIT").map(PathBuf::from));

pub(crate) fn add(native: &ID3D12Resource, address: u64, kind: &'static str) {
    if DIR.is_none() { return; }
    let d = unsafe { native.GetDesc() };
    let written = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
    if d.Dimension.0 < 3 || d.Width < 64 || d.SampleDesc.Count != 1 || d.Flags.0 & written == 0 { return; }
    LIVE.lock().unwrap_or_else(|e| e.into_inner()).insert(native.as_raw() as usize, Entry { address, kind });
}

pub(crate) fn remove(native: &ID3D12Resource) {
    if DIR.is_none() { return; }
    LIVE.lock().unwrap_or_else(|e| e.into_inner()).remove(&(native.as_raw() as usize));
}

static VIEWS: LazyLock<Mutex<std::collections::HashMap<usize, String>>> = LazyLock::new(Default::default);
/// Names what a render target or depth view shows.
pub(crate) fn view(handle: usize, resource: Option<&ID3D12Resource>, mip: u32) {
    if !crate::crumbs::on() { return; }
    let text = resource.map_or("null".into(), |r| format!("{}{}", label(r), if mip == 0 { String::new() } else { format!("m{mip}") }));
    VIEWS.lock().unwrap_or_else(|e| e.into_inner()).insert(handle, text);
}
pub(crate) fn view_label(handle: usize) -> String {
    VIEWS.lock().unwrap_or_else(|e| e.into_inner()).get(&handle).cloned().unwrap_or_else(|| "?".into())
}

pub(crate) fn label(native: &ID3D12Resource) -> String {
    let d = unsafe { native.GetDesc() };
    let at = LIVE.lock().unwrap_or_else(|e| e.into_inner()).get(&(native.as_raw() as usize)).map_or(0, |e| e.address);
    format!("{at:x}:{}x{}:f{}", d.Width, d.Height, d.Format.0)
}

pub(crate) struct Shot {
    buffer: ID3D12Resource, _source: ID3D12Resource,
    layout: D3D12_PLACED_SUBRESOURCE_FOOTPRINT, rows: u32, row_bytes: u64, name: String,
}

pub(crate) fn record(list: &ID3D12GraphicsCommandList) -> Vec<Shot> {
    let Some(dir) = DIR.as_ref() else { return Vec::new() };
    crate::crumbs::trace(dir);
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if queue.is_empty() {
        if std::fs::rename(dir.join("go"), dir.join("started")).is_err() { return Vec::new(); }
        queue.extend(LIVE.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|(p, e)| (*p, *e)));
        crate::diagnostics::log(&format!("AUDIT of {} images", queue.len()));
    }
    let mut shots = Vec::new();
    while shots.len() < 3 && let Some((raw, entry)) = queue.pop_front() {
        let source = {
            let live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
            if !live.contains_key(&raw) { continue; }
            let p = raw as *mut core::ffi::c_void;
            // SAFETY: the owner removes its entry under this lock before it releases the image.
            unsafe { ID3D12Resource::from_raw_borrowed(&p) }.expect("image").clone()
        };
        if let Some(shot) = unsafe { copy(list, source, entry) } { shots.push(shot); }
    }
    if queue.is_empty() { let _ = std::fs::rename(dir.join("started"), dir.join("done")); }
    shots
}

unsafe fn copy(list: &ID3D12GraphicsCommandList, source: ID3D12Resource, entry: Entry) -> Option<Shot> {
    let mut device = None::<ID3D12Device>;
    unsafe { list.GetDevice(&mut device).ok()? };
    let device = device?;
    let d = unsafe { source.GetDesc() };
    let (mut layout, mut rows, mut row_bytes, mut total) = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), 0, 0, 0);
    unsafe { device.GetCopyableFootprints(&d, 0, 1, 0, Some(&mut layout), Some(&mut rows), Some(&mut row_bytes), Some(&mut total)) };
    if total == u64::MAX || total > 1 << 28 { return None; }
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default()
    };
    let mut buffer = None::<ID3D12Resource>;
    unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer).ok()? };
    let buffer = buffer?;
    let barrier = |before, after| D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
            pResource: std::mem::ManuallyDrop::new(Some(source.clone())), Subresource: 0, StateBefore: before, StateAfter: after,
        }) },
    };
    unsafe {
        // The image's real layout is the title's business; this is a debug copy.
        let mut to = [barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)];
        let mut back = [barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)];
        list.ResourceBarrier(&to);
        let mut dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(buffer.clone())), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: layout },
        };
        let mut src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(source.clone())), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.ResourceBarrier(&back);
        for b in to.iter_mut().chain(back.iter_mut()) { std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource); }
        std::mem::ManuallyDrop::drop(&mut dst.pResource);
        std::mem::ManuallyDrop::drop(&mut src.pResource);
    }
    let name = format!("{}_{:x}_{}x{}x{}_f{}_c{}_fl{:x}_{:x}", entry.kind, entry.address, d.Width, d.Height, d.DepthOrArraySize,
        d.Format.0, layout.Footprint.Format.0, d.Flags.0, source.as_raw() as usize);
    Some(Shot { buffer, _source: source, layout, rows, row_bytes, name })
}

impl Shot {
    /// After the GPU finished: write every n-th texel of the first slice.
    pub(crate) fn save(&self) {
        let Some(dir) = DIR.as_ref() else { return };
        let mut mapped = core::ptr::null_mut();
        if unsafe { self.buffer.Map(0, None, Some(&mut mapped)) }.is_err() || mapped.is_null() { return; }
        let f = self.layout.Footprint;
        let (pitch, rows, width) = (f.RowPitch as usize, self.rows as usize, f.Width as usize);
        let bytes = (self.row_bytes as usize / width).max(1);
        let data = unsafe { core::slice::from_raw_parts(mapped.cast::<u8>(), pitch * (rows - 1) + self.row_bytes as usize) };
        let step = width.div_ceil(640).max(1);
        let mut out = Vec::new();
        let (mut w, mut h) = (0, 0);
        for y in (0..rows).step_by(step) {
            h += 1;
            w = 0;
            for x in (0..width).step_by(step) {
                w += 1;
                out.extend_from_slice(&data[y * pitch + x * bytes..][..bytes]);
            }
        }
        unsafe { self.buffer.Unmap(0, None) };
        let _ = std::fs::write(dir.join(format!("{}_{w}x{h}x{bytes}.bin", self.name)), out);
    }
}
