//! LOCAL: what each descriptor holds, to check the skinning shaders' inputs.
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{LazyLock, Mutex};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Kind {
    pub tag: &'static str,
    pub guest: u64,
    pub bytes: u64,
    pub format: i32,
    pub dim: i32,
    pub first: u64,
    pub count: u32,
    pub stride: u32,
    pub native: u64,
}
pub(crate) const NONE: Kind = Kind { tag: "?", guest: 0, bytes: 0, format: 0, dim: 0, first: 0, count: 0, stride: 0, native: 0 };

static KINDS: LazyLock<Mutex<HashMap<usize, Kind>>> = LazyLock::new(Default::default);
static LINES: AtomicU32 = AtomicU32::new(0);
static SEEN: LazyLock<Mutex<HashMap<String, u32>>> = LazyLock::new(Default::default);

pub(crate) fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("TUXBOX_VIEWKINDS").is_some());
    *ON
}
pub(crate) fn note(handle: usize, kind: Kind) {
    if !on() { return; }
    KINDS.lock().unwrap_or_else(|e| e.into_inner()).insert(handle, kind);
}
pub(crate) fn copy(dst: usize, src: usize, count: u32, stride: usize) {
    if !on() { return; }
    let mut map = KINDS.lock().unwrap_or_else(|e| e.into_inner());
    for i in 0..count as usize {
        match map.get(&(src + i * stride)).copied() {
            Some(kind) => { map.insert(dst + i * stride, kind); }
            None => { map.remove(&(dst + i * stride)); }
        }
    }
}
fn get(handle: usize) -> Option<Kind> { KINDS.lock().unwrap_or_else(|e| e.into_inner()).get(&handle).copied() }

fn read(address: u64, out: &mut [u32]) -> bool {
    use windows::Win32::System::{Diagnostics::Debug::ReadProcessMemory, Threading::GetCurrentProcess};
    let mut copied = 0;
    address != 0 && unsafe { ReadProcessMemory(GetCurrentProcess(), address as *const _, out.as_mut_ptr().cast(), size_of_val(out), Some(&mut copied)) }.is_ok()
        && copied == size_of_val(out)
}
const WANTED: [&str; 4] = ["2ea1ba38", "727f287e", "76289ae7", "e19219f0"];
pub(crate) fn wanted(name: &str) -> bool { on() && WANTED.iter().any(|w| name.starts_with(w)) }

fn show(k: Option<Kind>) -> String {
    k.map_or("UNWRITTEN".into(), |k| format!("{} guest={:#x} bytes={:#x} fmt={} dim={} first={} n={} stride={} native={:#x}",
        k.tag, k.guest, k.bytes, k.format, k.dim, k.first, k.count, k.stride, k.native))
}


#[derive(Clone)]
struct Job { name: String, records: Kind, base: u32, count: u32, heap: std::ops::Range<usize>, stride: usize }
static JOBS: LazyLock<Mutex<std::collections::VecDeque<Job>>> = LazyLock::new(Default::default);
// Slots whose copied record no longer resolves, and why.
static LOST: LazyLock<Mutex<HashMap<usize, &'static str>>> = LazyLock::new(Default::default);
pub(crate) fn lost(handle: usize, why: Option<&'static str>) {
    let mut map = LOST.lock().unwrap_or_else(|e| e.into_inner());
    match why { Some(why) => { map.insert(handle, why); } None => { map.remove(&handle); } }
}

/// Remember what a job shader was bound to, for `stalled`.
pub(crate) fn job(name: &str, slots: &[(usize, u32, usize)], heap: std::ops::Range<usize>, stride: usize, base: u32, count: u32) {
    let Some(records) = slots.iter().find(|s| (s.0, s.1) == (0, 14)).and_then(|s| get(s.2)) else { return };
    let mut jobs = JOBS.lock().unwrap_or_else(|e| e.into_inner());
    if jobs.len() >= 400 { jobs.pop_front(); }
    jobs.push_back(Job { name: name[..8].into(), records, base, count, heap, stride });
}

/// The GPU is stuck: print the records the recent jobs index, and what they name.
pub(crate) fn stalled() {
    static DONE: AtomicU32 = AtomicU32::new(0);
    if DONE.fetch_add(1, Ordering::Relaxed) > 2 { return; }
    let jobs: Vec<Job> = JOBS.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect();
    crate::diagnostics::log(&format!("JOBS stalled, {} recent jobs, {} lost slots", jobs.len(), LOST.lock().unwrap_or_else(|e| e.into_inner()).len()));
    if let Some(job) = jobs.last() {
        let k = job.records;
        let mut tags: std::collections::BTreeMap<String, (u32, usize)> = Default::default();
        for (&handle, v) in KINDS.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            if v.native == k.native || (v.guest != 0 && v.guest < k.guest + k.bytes && k.guest < v.guest + v.bytes) {
                let e = tags.entry(show(Some(*v))).or_default();
                e.0 += 1; e.1 = handle.wrapping_sub(job.heap.start) / job.stride;
            }
        }
        for (view, (n, slot)) in tags.iter().take(30) { crate::diagnostics::log(&format!("JOBS same memory: {n} of {view}, e.g. slot {slot:#x}")); }
    }
    let mut seen = std::collections::BTreeSet::new();
    for job in jobs.iter().rev() {
        let k = job.records;
        for at in job.base..job.base.saturating_add(job.count.min(64)) {
            if !seen.insert((k.native, at)) { continue; }
            let mut record = [0u32; 44];
            let offset = (k.first + u64::from(at)) * 176;
            if offset + 176 > k.bytes || !read(k.guest + offset, &mut record) { continue; }
            if seen.len() > 90 { break; }
            crate::diagnostics::log(&format!("JOBS {} record {at}/{}+{} {:x?}", job.name, job.base, job.count, record));
            for w in [14usize, 15, 18, 19] {
                let index = record[w] & 0xff_ffff;
                let target = job.heap.start + index as usize * job.stride;
                let lost = LOST.lock().unwrap_or_else(|e| e.into_inner()).get(&target).copied().unwrap_or("");
                crate::diagnostics::log(&format!("JOBS    w{w} {:#x} -> {} {lost}", record[w], if target < job.heap.end { show(get(target)) } else { "beyond heap".into() }));
            }
        }
    }
}
