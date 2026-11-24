//! LOCAL: live GPU allocations by kind (`TUXBOX_VRAM=1`), logged every 15 s.
use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{LazyLock, Mutex};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, HRESULT, IUnknown, Interface};

const KEY: GUID = GUID::from_u128(0x5d0a6c1e_77b4_4f1c_9b52_0d5c7a9e6b31);
/// kind -> (live, live bytes, ever created)
static LIVE: LazyLock<Mutex<HashMap<String, (i64, i64, u64)>>> = LazyLock::new(Default::default);

fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        let on = std::env::var_os("TUXBOX_VRAM").is_some();
        if on {
            std::thread::spawn(|| loop {
                std::thread::sleep(std::time::Duration::from_secs(15));
                report();
            });
        }
        on
    });
    *ON
}

fn report() {
    let live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    let mut totals: HashMap<&str, (i64, i64, u64)> = HashMap::new();
    for (key, v) in live.iter() {
        let t = totals.entry(key.split(' ').next().unwrap_or("")).or_default();
        *t = (t.0 + v.0, t.1 + v.1, t.2 + v.2);
    }
    let mut totals: Vec<_> = totals.into_iter().collect();
    totals.sort_by_key(|t| -t.1.1);
    let mib = |b: i64| b >> 20;
    crate::diagnostics::log(&format!("VRAM {}", totals.iter().map(|(k, v)| format!("{k}={}MiB/{}(of {})", mib(v.1), v.0, v.2)).collect::<Vec<_>>().join(" ")));
    let mut keys: Vec<_> = live.iter().collect();
    keys.sort_by_key(|k| -k.1.1);
    crate::diagnostics::log(&format!("VRAM-TOP {}", keys.iter().take(14).map(|(k, v)| format!("[{k}: {}MiB x{} of {}]", mib(v.1), v.0, v.2)).collect::<Vec<_>>().join(" ")));
}

#[repr(C)]
struct Sentinel {
    vtable: &'static [usize; 3],
    count: AtomicU32,
    kind: String,
    bytes: i64,
}
static VTABLE: LazyLock<[usize; 3]> = LazyLock::new(|| [query as *const () as usize, add as *const () as usize, release as *const () as usize]);

unsafe extern "system" fn query(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    unsafe {
        if *iid == IUnknown::IID {
            add(this);
            *out = this;
            return HRESULT(0);
        }
        *out = core::ptr::null_mut();
    }
    HRESULT(0x8000_4002u32 as i32)
}
unsafe extern "system" fn add(this: *mut c_void) -> u32 {
    unsafe { (*this.cast::<Sentinel>()).count.fetch_add(1, Ordering::Relaxed) + 1 }
}
unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    let left = unsafe { (*this.cast::<Sentinel>()).count.fetch_sub(1, Ordering::AcqRel) } - 1;
    if left == 0 {
        let gone = unsafe { Box::from_raw(this.cast::<Sentinel>()) };
        if let Some(v) = LIVE.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&gone.kind) {
            v.0 -= 1;
            v.1 -= gone.bytes;
        }
    }
    left
}

/// Counts `bytes` under `kind` (first word: group) until `object` is destroyed.
pub(crate) fn held(object: &ID3D12Object, kind: impl FnOnce() -> String, bytes: u64) {
    if !on() { return; }
    let (kind, bytes) = (kind(), bytes as i64);
    {
        let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        let v = live.entry(kind.clone()).or_default();
        *v = (v.0 + 1, v.1 + bytes, v.2 + 1);
    }
    let raw = Box::into_raw(Box::new(Sentinel { vtable: &VTABLE, count: AtomicU32::new(1), kind, bytes }));
    unsafe {
        let sentinel = IUnknown::from_raw(raw.cast());
        let _ = object.SetPrivateDataInterface(&KEY, &sentinel);
    }
}

/// A resource owning its memory.
pub(crate) fn resource(resource: &ID3D12Resource, group: &str) {
    if !on() { return; }
    unsafe {
        let desc = resource.GetDesc();
        let mut device: Option<ID3D12Device> = None;
        let bytes = match resource.GetDevice(&mut device) {
            Ok(()) => device.map_or(0, |d| d.GetResourceAllocationInfo(0, &[desc]).SizeInBytes),
            Err(_) => 0,
        };
        let mut heap = D3D12_HEAP_PROPERTIES::default();
        let _ = resource.GetHeapProperties(Some(&mut heap), None);
        let Ok(object) = resource.cast::<ID3D12Object>() else { return };
        held(&object, || if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
            format!("{group}/t{} buffer {}", heap.Type.0, bytes.next_power_of_two() >> 10)
        } else {
            format!("{group}/t{} {}x{}x{}m{} f{} fl{:x}", heap.Type.0, desc.Width, desc.Height, desc.DepthOrArraySize, desc.MipLevels, desc.Format.0, desc.Flags.0)
        }, bytes);
    }
}
