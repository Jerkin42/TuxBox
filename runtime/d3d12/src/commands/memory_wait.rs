//! Translate waits on host-visible Series fence words at a native list boundary.
//! A wait after rendering commands ends the native list. The next command opens
//! another one with the guest's bindings (native_list, Recording::resume).
//!
//! Each boundary is also an ExecuteCommandLists boundary, where D3D12 decays
//! buffers, simultaneous-access textures, read-only states reached by promotion
//! and everything on a copy queue to COMMON. Those promote again on use. Explicit
//! states of other textures, enhanced layouts and acceleration structures persist.
//! A later guest barrier may name the state from before the decay, as it already
//! can across guest lists; the fence pair at the boundary flushes the writes.
//! The game can write markers and sample timestamps before waiting. Keep these
//! prefix operations ordered without disturbing bindings on the main command list.
use super::*;
use std::sync::atomic::{AtomicI32, AtomicU64};
use std::sync::mpsc;
use std::time::Duration;
use windows::Win32::Foundation::HANDLE;
use windows::core::Result;

pub(super) enum Packet {
    Marker(D3D12_WRITEBUFFERIMMEDIATE_PARAMETER),
    Timestamp(ID3D12QueryHeap, u32, ID3D12Resource, u64),
}

impl Packet {
    fn emit(&self, list: &ID3D12GraphicsCommandList) -> Result<()> {
        unsafe {
            match self {
                Self::Marker(write) => {
                    list.cast::<ID3D12GraphicsCommandList2>()?
                        .WriteBufferImmediate(1, write, None);
                }
                Self::Timestamp(heap, index, dest, offset) => {
                    list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, *index);
                    list.ResolveQueryData(
                        heap,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        *index,
                        1,
                        dest,
                        *offset,
                    );
                }
            }
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct Prefix {
    pending: Vec<Packet>,
    in_body: bool,
    pub(super) ops: Vec<(&'static str, u64, u64)>, // LOCAL: fence trace
    // The number of native lists that execute before each wait.
    pub(super) segments: Vec<(usize, Arc<Segment>)>,
}

impl Prefix {
    pub(super) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub(super) fn push(&mut self, list: &ID3D12GraphicsCommandList, packet: Packet) -> Result<()> {
        if self.in_body {
            return packet.emit(list);
        }
        self.pending.try_reserve(1).map_err(|_| E_OUTOFMEMORY)?;
        self.pending.push(packet);
        Ok(())
    }

    pub(super) fn finish(&mut self, list: &ID3D12GraphicsCommandList) -> Result<()> {
        if self.in_body {
            return Ok(());
        }
        self.in_body = true;
        for packet in self.pending.drain(..) {
            packet.emit(list)?;
        }
        Ok(())
    }
}

// LOCAL: last value recorded for each fence word, to explain a wait that stalls.
static WRITES: std::sync::Mutex<Option<std::collections::HashMap<u64, (u64, u64)>>> = std::sync::Mutex::new(None);
pub(super) fn note_write(address: u64, value: u64) {
    let mut writes = WRITES.lock().unwrap_or_else(|e| e.into_inner());
    let map = writes.get_or_insert_default();
    if map.len() < 65536 || map.contains_key(&address) {
        let entry = map.entry(address).or_default();
        *entry = (value, entry.1 + 1);
    }
}

// LOCAL: every submitted list that touches fence words, dumped when a wait stalls.
pub(super) struct Submitted { pub(super) queue: usize, pub(super) kind: i32, pub(super) list: usize, pub(super) error: i32,
    pub(super) outcome: &'static str, pub(super) ops: Vec<(&'static str, u64, u64)> }
static SUBMITTED: std::sync::Mutex<std::collections::VecDeque<(u64, Submitted)>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
static SEQUENCE: AtomicU64 = AtomicU64::new(0);
pub(super) fn note_submit(entry: Submitted) {
    if entry.ops.is_empty() && entry.outcome == "ok" { return; }
    let n = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut all = SUBMITTED.lock().unwrap_or_else(|e| e.into_inner());
    if all.len() >= 600 { all.pop_front(); }
    all.push_back((n, entry));
}
fn dump_submitted() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let all = SUBMITTED.lock().unwrap_or_else(|e| e.into_inner());
        for (n, s) in all.iter() {
            let ops: Vec<String> = s.ops.iter().take(40).map(|(k, a, v)| format!("{k}:{a:x}={v:x}")).collect();
            crate::diagnostics::log(&format!("FENCETRACE #{n} q={:x} kind={} list={:x} err={:#x} {} {}",
                s.queue, s.kind, s.list, s.error, s.outcome, ops.join(" ")));
        }
    });
}

struct Word {
    resource: ID3D12Resource,
    mapped: usize,
    address: u64,
}

impl Drop for Word {
    fn drop(&mut self) {
        unsafe { self.resource.Unmap(0, Some(&D3D12_RANGE::default())) };
    }
}

impl Word {
    fn map(list: &List, address: u64) -> Result<Self> {
        if address == 0 || !address.is_multiple_of(8) {
            return Err(E_INVALIDARG.into());
        }
        let (buffer, gpu) =
            unsafe { device::command_buffer_address(parent(&list.parent), address, 8)? };
        let resource = buffer.ok_or(E_NOTIMPL)?;
        let offset = gpu
            .checked_sub(unsafe { resource.GetGPUVirtualAddress() })
            .ok_or(E_INVALIDARG)? as usize;
        let range = D3D12_RANGE {
            Begin: offset,
            End: offset.checked_add(8).ok_or(E_INVALIDARG)?,
        };
        let mut base = ptr::null_mut();
        unsafe { resource.Map(0, Some(&range), Some(&mut base))? };
        if base.is_null() {
            unsafe { resource.Unmap(0, Some(&D3D12_RANGE::default())) };
            return Err(E_FAIL.into());
        }
        Ok(Self {
            resource,
            mapped: base.addr() + offset,
            address,
        })
    }

    fn reached(&self, value: u64) -> bool {
        // X64 aligned 64-bit load. The COM resource and its CPU mapping stay alive
        // until all queued waits release their segment.
        unsafe { AtomicU64::from_ptr(self.mapped as *mut u64).load(Ordering::Acquire) >= value }
    }
}

pub(super) struct Segment {
    prefix: Option<(ID3D12CommandList, ID3D12CommandAllocator)>,
    word: Word,
    target: u64,
}

pub(super) unsafe extern "system" fn wait_value64(
    list: &List,
    address: u64,
    comparison: u32,
    target: u64,
    flags: u32,
) {
    let mut state = recording(list);
    if !state.ensure_open(&list.native) {
        return;
    }
    let result = (|| -> Result<()> {
        if comparison != 7 || flags > 1 {
            return Err(E_NOTIMPL.into());
        }
        let word = Word::map(list, address)?;
        if target == 0 {
            return Ok(());
        }
        let in_body = state.prefix.in_body;
        let refused = if !in_body {
            None
        } else if state.open_queries != 0 {
            Some("inside a query")
        } else if state.split_barriers != 0 {
            Some("between the halves of a split barrier")
        } else {
            None
        };
        state.prefix.ops.push((if refused.is_some() { "DROPWAIT" } else if in_body { "midwait" } else { "wait" }, address, target));
        if let Some(reason) = refused {
            crate::diagnostics::recording_error(format_args!(
                "GPU memory wait after rendering commands, {reason}"
            ));
            return Err(E_NOTIMPL.into());
        }
        state
            .prefix
            .segments
            .try_reserve(1)
            .map_err(|_| E_OUTOFMEMORY)?;
        if in_body {
            // Later markers and waits queue up as at the start of a list until
            // a command needs the continuation.
            unsafe { list.native.Close()? };
            state.parts.parted = true;
            state.prefix.in_body = false;
            static FIRST: std::sync::Once = std::sync::Once::new();
            crate::diagnostics::log_once(&FIRST, || format!(
                "split a kind {} list at a GPU memory wait", unsafe { list.native.GetType() }.0
            ));
        }
        let prefix = if state.prefix.pending.is_empty() {
            None
        } else {
            let mut device: Option<ID3D12Device> = None;
            unsafe {
                list.native.GetDevice(&mut device)?;
            }
            let device = device.ok_or(E_FAIL)?;
            let kind = unsafe { list.native.GetType() };
            let allocator: ID3D12CommandAllocator = unsafe { device.CreateCommandAllocator(kind)? };
            let native: ID3D12GraphicsCommandList =
                unsafe { device.CreateCommandList(0, kind, &allocator, None)? };
            for packet in state.prefix.pending.drain(..) {
                packet.emit(&native)?;
            }
            unsafe {
                native.Close()?;
            }
            Some((native.into(), allocator))
        };
        let before = state.parts.closed();
        state.prefix.segments.push((before, Arc::new(Segment {
            prefix,
            word,
            target,
        })));
        Ok(())
    })();
    if let Err(error) = result {
        crate::diagnostics::recording_error(format_args!(
            "64-bit GPU memory wait {address:#x} >= {target:#x} ({:?})",
            error.code()
        ));
        state.error = error.code();
    }
}

struct Job {
    fence: ID3D12Fence,
    segment: Arc<Segment>,
    device: ID3D12Device,
}

pub(super) struct Worker(mpsc::Sender<Job>);

impl Worker {
    fn new(error: Arc<AtomicI32>) -> Result<Self> {
        let (sender, jobs) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("TuxBox GPU memory waits".into())
            .spawn(move || {
                for job in jobs {
                    let result = (|| -> Result<()> {
                        let span = crate::timing::span("wait.prefix");
                        unsafe { job.fence.SetEventOnCompletion(1, HANDLE::default())? };
                        drop(span);
                        let span = crate::timing::span("wait.word");
                        let (mut rounds, mut reports) = (0u32, 0u32);
                        while !job.segment.word.reached(job.segment.target) {
                            rounds += 1;
                            if rounds % 3000 == 0 && reports < 4 {
                                reports += 1;
                                dump_submitted();
                                let word = &job.segment.word;
                                let now = unsafe { AtomicU64::from_ptr(word.mapped as *mut u64).load(Ordering::Acquire) };
                                let last = WRITES.lock().unwrap_or_else(|e| e.into_inner())
                                    .as_ref().and_then(|w| w.get(&word.address).copied());
                                crate::diagnostics::log(&format!(
                                    "STUCK wait {:#x} >= {:#x}: holds {now:#x}, last recorded write (value, count) {last:x?}",
                                    word.address, job.segment.target));
                            }
                            unsafe {
                                job.device.GetDeviceRemovedReason()?;
                            }
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        drop(span);
                        unsafe { job.fence.Signal(2)?; }
                        crate::device::placement::TITLE_WAITS.fetch_sub(1, Ordering::AcqRel);
                        static FIRST: std::sync::Once = std::sync::Once::new();
                        crate::diagnostics::log_once(&FIRST, || format!(
                            "GPU memory wait completed target={:#x}", job.segment.target
                        ));
                        Ok(())
                    })();
                    if let Err(cause) = result {
                        let _ = error.compare_exchange(
                            S_OK.0,
                            cause.code().0,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        );
                        // Fail the device rather than releasing work behind a broken gate.
                        if let Ok(device) = job.device.cast::<ID3D12Device5>() {
                            unsafe { device.RemoveDevice() };
                        }
                        break;
                    }
                }
            })
            .map_err(|_| E_OUTOFMEMORY)?;
        Ok(Self(sender))
    }
}

pub(super) type Splits = Vec<(usize, Arc<Segment>)>;

pub(super) unsafe fn submit(queue: &Queue, host: &[*mut c_void], splits: Splits) -> Result<()> {
    let run = |lists: &[*mut c_void]| {
        if !lists.is_empty() {
            unsafe {
                (queue.native.vtable().ExecuteCommandLists)(
                    queue.native.as_raw(),
                    lists.len() as u32,
                    lists.as_ptr(),
                )
            };
        }
    };
    let mut start = 0;
    for (index, segment) in splits {
        run(&host[start..index]);
        start = index;
        let mut device: Option<ID3D12Device> = None;
        unsafe {
            queue.native.GetDevice(&mut device)?;
        }
        let device = device.ok_or(E_FAIL)?;
        let fence = unsafe { device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)? };
        let worker = match queue.state.wait_worker.get() {
            Some(worker) => worker,
            None => {
                let new = Worker::new(queue.state.placement.error.clone())?;
                queue.state.wait_worker.get_or_init(|| new)
            }
        };
        if let Some((prefix, _)) = &segment.prefix {
            run(&[prefix.as_raw()]);
        }
        // Fence 1 retires the entire prefix before the CPU reads the guest word.
        // Fence 2 holds the suffix on the GPU without blocking the guest submission thread.
        unsafe {
            queue.native.Signal(&fence, 1)?;
        }
        crate::device::placement::TITLE_WAITS.fetch_add(1, Ordering::AcqRel);
        worker
            .0
            .send(Job {
                fence: fence.clone(),
                segment,
                device,
            })
            .map_err(|_| E_FAIL)?;
        unsafe {
            queue.native.Wait(&fence, 2)?;
        }
    }
    run(&host[start..]);
    Ok(())
}
