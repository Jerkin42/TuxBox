//! A writable Xbox descriptor arena can exceed the shader-visible heap limit.
//! Store all native descriptors in a CPU heap and materialize bounded root
//! tables in a shader-visible heap when the command list references them.
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, Weak, atomic::{AtomicU64, Ordering}},
};
use windows::Win32::Foundation::{E_INVALIDARG, E_OUTOFMEMORY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Result;

use super::title_memory::GuestMemory;
mod index_map;
pub(crate) use index_map::index_map;

#[derive(Default)]
pub(crate) struct Registry {
    arenas: Mutex<BTreeMap<usize, Weak<Arena>>>,
    count: std::sync::atomic::AtomicUsize,
    next_version: AtomicU64,
    identity_map: Mutex<Option<ID3D12Resource>>,
    snapshots: Mutex<Option<Snapshots>>,
}

/// A title copies a descriptor's record out of a staging cell and then reuses
/// the cell. Keep the host descriptor each recent record named, so the copies
/// still resolve.
struct Snapshots {
    heap: ID3D12DescriptorHeap,
    stride: usize,
    records: Vec<(usize, u64)>,
    imported: super::placement::DescriptorRange<super::views::DescriptorBacking>,
    placed: super::placement::DescriptorRange<super::placement::PlacedView>,
    scenes: super::placement::DescriptorRange<super::blas::SceneView>,
}
impl Snapshots {
    const COUNT: usize = 1 << 19;
    fn create(arena: &Arena) -> Option<Self> {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: Self::COUNT as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0,
        };
        let heap: ID3D12DescriptorHeap = unsafe { arena.device.CreateDescriptorHeap(&desc) }.ok()?;
        Some(Self {
            imported: arena.imported.sibling(&arena.device, &heap, &desc)?,
            placed: arena.placed.sibling(&arena.device, &heap, &desc)?,
            scenes: arena.scenes.sibling(&arena.device, &heap, &desc)?,
            stride: arena.stride, records: vec![(0, 0); Self::COUNT], heap,
        })
    }
    fn slot(version: u64) -> usize { version as usize % Self::COUNT }
    fn handle(&self, slot: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() }.ptr + slot * self.stride }
    }
    fn get(&self, record: InlineDescriptor) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        let slot = Self::slot(record.version);
        (self.records[slot] == (record.source, record.version)).then(|| self.handle(slot))
    }
}
impl Registry {
    pub(crate) fn find(&self, address: usize) -> Option<Arc<Arena>> {
        if self.count.load(std::sync::atomic::Ordering::Acquire) == 0 {
            return None;
        }
        let entries = self.arenas.lock().unwrap_or_else(|e| e.into_inner());
        let arena = entries.range(..=address).next_back()?.1.upgrade()?;
        (address - arena.base() < arena.count as usize * arena.stride).then_some(arena)
    }
    /// LOCAL: see `Arena::sync_copied`. Throttled; measures whether it matters.
    pub(crate) fn sync_copied(&self) {
        static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let full = {
            let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
            let due = last.is_none_or(|t| t.elapsed() >= std::time::Duration::from_millis(5));
            if due { *last = Some(std::time::Instant::now()); }
            due
        };
        let arenas: Vec<_> = self.arenas.lock().unwrap_or_else(|e| e.into_inner()).values().filter_map(Weak::upgrade).collect();
        for arena in arenas { arena.sync_copied(full); }
    }
    pub(super) fn copy_source(&self, handle: D3D12_CPU_DESCRIPTOR_HANDLE) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let Some(arena) = self.find(handle.ptr) else { return handle; };
        if arena.inline && (handle.ptr - arena.base()).is_multiple_of(arena.stride) {
            if let Some(source) = arena.resolve_inline(arena.inline_record(handle.ptr)) {
                return source;
            }
        }
        arena.cpu(handle.ptr)
    }


}

// Only the opaque record lives in guest-writable memory. The host descriptor
// stays in a CPU-only heap and is never edited by the guest's 32-byte copies.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct InlineDescriptor {
    source: usize,
    version: u64,
    reserved: [u64; 2],
}
const PAGE: usize = 128;

/// The record each native slot holds. The slots that coincide with the title's
/// are dense; aliases above them are not.
#[derive(Default)]
struct Resident { dense: Vec<Option<InlineDescriptor>>, rest: BTreeMap<usize, InlineDescriptor> }
impl Resident {
    fn new(dense: u32) -> Self { Self { dense: vec![None; dense as usize], rest: BTreeMap::new() } }
    fn get(&self, slot: usize) -> Option<InlineDescriptor> {
        match self.dense.get(slot) {
            Some(record) => *record,
            None => self.rest.get(&slot).copied(),
        }
    }
    fn insert(&mut self, slot: usize, record: InlineDescriptor) {
        match self.dense.get_mut(slot) {
            Some(entry) => *entry = Some(record),
            None => { self.rest.insert(slot, record); }
        }
    }
    fn remove(&mut self, slot: usize) {
        match self.dense.get_mut(slot) {
            Some(entry) => *entry = None,
            None => { self.rest.remove(&slot); }
        }
    }
    fn forget_aliases(&mut self) { self.rest.clear(); }
}
const INLINE_BYTES: usize = size_of::<InlineDescriptor>();

pub(crate) struct Arena {
    memory: GuestMemory,
    registry: Arc<Registry>,
    imported: super::placement::DescriptorRange<super::views::DescriptorBacking>,
    placed: super::placement::DescriptorRange<super::placement::PlacedView>,
    scenes: super::placement::DescriptorRange<super::blas::SceneView>,
    staging: ID3D12DescriptorHeap,
    visible: ID3D12DescriptorHeap,
    device: ID3D12Device,
    stride: usize,
    count: u32,
    limit: u32,
    inline: bool,
    // Guest and native slots coincide below this, so shader-supplied heap
    // indices are valid. Tables that reach above it use aliases.
    identity: u32,
    authored: Mutex<BTreeMap<usize, u64>>,
    // Guest ranges can overlap. Keep stable slots for every table that has
    // been referenced; a CPU descriptor update is published to each alias.
    tables: Mutex<Tables>,
    // The records the last scan saw, to find the title's own copies.
    shadow: Mutex<(Vec<InlineDescriptor>, Vec<u64>, u64)>,
}
#[derive(Default)]
struct Tables {
    used: u32,
    ranges: BTreeMap<(u32, u32), u32>,
    direct: index_map::Direct,
    resident_records: Resident,
}
impl Drop for Arena {
    fn drop(&mut self) {
        let mut entries = self.registry.arenas.lock().unwrap_or_else(|e| e.into_inner());
        if entries
            .get(&self.base())
            .is_some_and(|entry| std::ptr::eq(entry.as_ptr(), self))
        {
            entries.remove(&self.base());
            self.registry
                .count
                .store(entries.len(), std::sync::atomic::Ordering::Release);
        }
    }
}
impl Arena {
    pub(super) fn create(
        owner: &super::Device,
        desc: &D3D12_DESCRIPTOR_HEAP_DESC,
        mut limit: u32,
    ) -> Result<Arc<Self>> {
        let registry = &owner.state.descriptor_arenas;
        let device = &owner.native;
        let cpu = D3D12_DESCRIPTOR_HEAP_DESC {
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            ..*desc
        };
        let gpu = D3D12_DESCRIPTOR_HEAP_DESC {
            NumDescriptors: limit,
            ..*desc
        };
        let staging: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&cpu)? };
        let visible = if desc.Flags == D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE {
            let full = D3D12_DESCRIPTOR_HEAP_DESC { ..*desc };
            match unsafe { device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&full) } {
                Ok(heap) => { limit = limit.max(desc.NumDescriptors); heap }
                Err(error) => {
                    crate::diagnostics::log(&format!("HEAP full size {} refused: {:?}", desc.NumDescriptors, error.code()));
                    unsafe { device.CreateDescriptorHeap(&gpu)? }
                }
            }
        } else {
            staging.clone()
        };
        let stride = unsafe { device.GetDescriptorHandleIncrementSize(desc.Type) } as usize;
        let bytes = stride
            .checked_mul(desc.NumDescriptors as usize)
            .ok_or(E_OUTOFMEMORY)?;
        let identity = if !owner.state.device11_queue || stride != INLINE_BYTES || std::env::var_os("TUXBOX_NO_IDENTITY_HEAP").is_some() {
            0
        } else if desc.NumDescriptors <= limit {
            desc.NumDescriptors
        } else {
            limit - limit / 8
        };
        crate::diagnostics::log(&format!("HEAP arena count={} limit={limit} visible={} stride={stride} identity={identity}",
            desc.NumDescriptors, desc.Flags == D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE));
        let memory = GuestMemory::allocate(bytes, 0x0480_0004)?;
        let arena = Arc::new(Self {
            imported: owner.state.imported_views.heap(device, &staging, &cpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV).ok_or(E_INVALIDARG)?,
            placed: owner
                .state
                .placements
                .views
                .heap(
                    device,
                    &staging,
                    &cpu,
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                )
                .ok_or(E_INVALIDARG)?,
            scenes: owner
                .state
                .blas
                .views
                .heap(
                    device,
                    &staging,
                    &cpu,
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                )
                .ok_or(E_INVALIDARG)?,
            memory,
            registry: registry.clone(),
            staging,
            visible,
            device: device.clone(),
            stride,
            count: desc.NumDescriptors,
            limit,
            inline: owner.state.device11_queue && stride >= INLINE_BYTES && stride.is_multiple_of(INLINE_BYTES),
            identity,
            authored: Mutex::default(),
            tables: Mutex::new(Tables { used: identity, resident_records: Resident::new(if desc.Flags == D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE { identity } else { 0 }), ..Default::default() }),
            shadow: Mutex::default(),
        });
        let mut entries = registry.arenas.lock().unwrap_or_else(|e| e.into_inner());
        entries.insert(arena.base(), Arc::downgrade(&arena));
        registry
            .count
            .store(entries.len(), std::sync::atomic::Ordering::Release);
        Ok(arena)
    }
    pub(crate) fn is_identity(&self) -> bool { self.identity != 0 }

    /// LOCAL: publish records the title copied itself into slots no table names.
    fn sync_copied(&self, full: bool) {
        if self.identity == 0 || self.visible == self.staging { return; }
        let started = std::time::Instant::now();
        let (mut copied, mut pages) = (0u64, 0u64);
        // Compare whole pages with what the last scan saw. Recording threads
        // resolve tables meanwhile, so hold their lock only to write a page.
        let mut shadow = self.shadow.lock().unwrap_or_else(|e| e.into_inner());
        let (shadow, warm, calls) = &mut *shadow;
        if shadow.is_empty() {
            shadow.resize(self.count as usize, InlineDescriptor::default());
            warm.resize((self.count as usize).div_ceil(PAGE), 0);
        }
        *calls += 1;
        let start = unsafe { self.visible.GetCPUDescriptorHandleForHeapStart() };
        for (page, seen) in shadow.chunks_mut(PAGE).enumerate() {
            // Between full passes, look at the pages that changed lately.
            if !full && (warm[page] == 0 || *calls - warm[page] > 2000) { continue; }
            let first = page * PAGE;
            let cells = (self.base() + first * INLINE_BYTES) as *const InlineDescriptor;
            // The title writes this committed memory at any time; a torn record
            // differs from the next scan's and is taken then.
            let same = unsafe { std::slice::from_raw_parts(cells.cast::<u64>(), seen.len() * 4) }
                == unsafe { std::slice::from_raw_parts(seen.as_ptr().cast::<u64>(), seen.len() * 4) };
            if same { continue; }
            pages += 1;
            warm[page] = *calls;
            let mut tables = self.tables.lock().unwrap_or_else(|e| e.into_inner());
            for (index, old) in seen.iter_mut().enumerate() {
                let slot = first + index;
                let cell = self.base() + slot * INLINE_BYTES;
                let record = self.inline_record(cell);
                if record == *old { continue; }
                *old = record;
                // A record names a descriptor cell; skip the title's own data cheaply.
                if record.source < 0x10000 || record.source == cell || record.source & 15 != 0
                    || record.version == 0 || record.reserved != [0; 2] || slot >= self.identity as usize
                    || tables.resident_records.get(slot) == Some(record) { continue; }
                let Some(from) = self.resolve_inline(record) else { continue; };
                self.copy_native_view(D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start.ptr + slot * self.stride }, from);
                tables.resident_records.insert(slot, record);
                copied += 1;
            }
        }
        static TOTAL: AtomicU64 = AtomicU64::new(0);
        static CALLS: AtomicU64 = AtomicU64::new(0);
        static NANOS: AtomicU64 = AtomicU64::new(0);
        static PAGES: AtomicU64 = AtomicU64::new(0);
        TOTAL.fetch_add(copied, Ordering::Relaxed);
        PAGES.fetch_add(pages, Ordering::Relaxed);
        NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if CALLS.fetch_add(1, Ordering::Relaxed) % 1000 == 999 {
            crate::diagnostics::log(&format!("HEAP sync last 1000 calls: published={} changed pages={} mean {} us",
                TOTAL.swap(0, Ordering::Relaxed), PAGES.swap(0, Ordering::Relaxed), NANOS.swap(0, Ordering::Relaxed) / 1_000_000));
        }
    }

    pub(crate) fn uses_inline_records(&self) -> bool { self.inline }

    pub(crate) fn base(&self) -> usize {
        self.memory.address
    }
    pub(crate) fn contains(&self, address: u64) -> bool {
        address
            .checked_sub(self.base() as u64)
            .is_some_and(|offset| offset < self.count as u64 * self.stride as u64)
    }
    pub(crate) fn visible(&self) -> &ID3D12DescriptorHeap {
        &self.visible
    }
    pub(crate) fn cpu(&self, address: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let base = unsafe { self.staging.GetCPUDescriptorHandleForHeapStart() };
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + (address - self.base()),
        }
    }
    fn inline_record(&self, address: usize) -> InlineDescriptor {
        // The arena owns this committed memory. Bounds are checked by callers
        // for the whole record, and the title synchronizes descriptor writes.
        unsafe { (address as *const InlineDescriptor).read_unaligned() }
    }

    pub(super) fn author(&self, address: usize, count: u32) {
        if !self.inline || !self.contains(address as u64)
            || !(address - self.base()).is_multiple_of(self.stride)
            || count as usize > self.count as usize - (address - self.base()) / self.stride
        {
            return;
        }
        let mut versions = self.authored.lock().unwrap_or_else(|e| e.into_inner());
        for slot in 0..count as usize {
            let source = address + slot * self.stride;
            let version = self.registry.next_version.fetch_add(1, Ordering::Relaxed) + 1;
            versions.insert(source, version);
            let record = InlineDescriptor { source, version, reserved: [0; 2] };
            // The caller validated the destination span through the heap API.
            unsafe { (source as *mut InlineDescriptor).write_unaligned(record) };
            let mut snapshots = self.registry.snapshots.lock().unwrap_or_else(|e| e.into_inner());
            if snapshots.is_none() { *snapshots = Snapshots::create(self); }
            if let Some(kept) = snapshots.as_mut() {
                let at = Snapshots::slot(version);
                kept.records[at] = (source, version);
                self.copy_native_view(kept.handle(at), self.cpu(source));
            }
        }
    }

    fn resolve_inline(&self, record: InlineDescriptor) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        if record.source == 0 || record.version == 0 { return None; }
        // Match the snapshot first: its contents cannot change under the copy.
        if let Some(kept) = self.registry.snapshots.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|s| s.get(record)) {
            return Some(kept);
        }
        let source = self.registry.find(record.source)?;
        let versions = source.authored.lock().unwrap_or_else(|e| e.into_inner());
        (versions.get(&record.source) == Some(&record.version)).then(|| source.cpu(record.source))
    }

    fn copy_native_view(&self, to: D3D12_CPU_DESCRIPTOR_HANDLE, from: D3D12_CPU_DESCRIPTOR_HANDLE) {
        let stride = self.stride;
        self.imported.copy_to(stride, to.ptr, from.ptr, 1);
        self.placed.copy_to(stride, to.ptr, from.ptr, 1);
        self.scenes.copy_to(stride, to.ptr, from.ptr, 1);
        crate::diagnostics::display::copy_texture_views(to.ptr, from.ptr, 1, stride);
        unsafe { self.device.CopyDescriptorsSimple(1, to, from, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
    }

    fn copy(&self, src: u32, dst: u32, count: u32) {
        let from = self.cpu(self.base() + src as usize * self.stride);
        let start = unsafe { self.visible.GetCPUDescriptorHandleForHeapStart() };
        let to = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + dst as usize * self.stride,
        };
        self.imported.copy_to(self.stride, to.ptr, from.ptr, count);
        self.placed.copy_to(self.stride, to.ptr, from.ptr, count);
        self.scenes.copy_to(self.stride, to.ptr, from.ptr, count);
        crate::diagnostics::display::copy_texture_views(to.ptr, from.ptr, count, self.stride);
        unsafe {
            self.device.CopyDescriptorsSimple(
                count,
                to,
                from,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            )
        };
    }
    pub(crate) fn publish(&self, address: usize, count: u32) {
        let first = ((address - self.base()) / self.stride) as u32;
        let end = first.saturating_add(count);
        if end > self.count || count == 0 {
            return;
        }
        let mut tables = self.tables.lock().unwrap_or_else(|e| e.into_inner());
        if self.identity != 0 && self.visible != self.staging {
            static BLOCKS: AtomicU64 = AtomicU64::new(0);
            for block in first >> 15..=(end - 1) >> 15 {
                let bit = 1u64 << block.min(63);
                if BLOCKS.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
                    crate::diagnostics::log(&format!("HEAP block {:#x} of {:#x} first written (slots {first:#x}..{end:#x})", block << 15, self.count));
                }
            }
            let low = end.min(self.identity);
            if first < low {
                self.copy(first, first, low - first);
                for slot in first..low { tables.resident_records.remove(slot as usize); }
            }
        }
        let Tables { direct, used, ranges, resident_records, .. } = &mut *tables;
        if self.identity == 0 { direct.publish(self, first..end, used); }
        for (&(start, length), &resident) in ranges.iter() {
            if self.inline && self.stride != INLINE_BYTES {
                let offset = (first as usize).saturating_sub(start as usize) * self.stride / INLINE_BYTES;
                if first >= start && offset < length as usize {
                    let count = u32::try_from((length as usize - offset).div_ceil(self.stride / INLINE_BYTES)).unwrap_or(0).min(count);
                    for step in 0..count as usize {
                        let slot = resident as usize + offset + step * (self.stride / INLINE_BYTES);
                        self.copy(first + step as u32, slot as u32, 1);
                        resident_records.remove(slot);
                    }
                }
                continue;
            }
            let lo = start.max(first);
            let hi = (start + length).min(end);
            if lo < hi {
                let destination = resident + lo - start;
                self.copy(lo, destination, hi - lo);
                if self.inline {
                    for slot in destination..destination + hi - lo {
                        resident_records.remove(slot as usize);
                    }
                }
            }
        }
    }
    pub(crate) fn gpu(&self, address: u64, length: u32) -> Result<D3D12_GPU_DESCRIPTOR_HANDLE> {
        self.gpu_table(address, length, &[0..length])
    }

    pub(crate) fn resident_gpu(&self, address: u64, length: u32) -> Option<D3D12_GPU_DESCRIPTOR_HANDLE> {
        let offset = (address as usize).checked_sub(self.base())?;
        let first = u32::try_from(offset / self.stride).ok()?;
        if first.checked_add(length)? <= self.identity { return Some(self.gpu_handle(first)); }
        let tables = self.tables.lock().unwrap_or_else(|e| e.into_inner());
        let slot = tables.ranges.get(&(first, length)).copied().or_else(|| tables.direct.contiguous(first, length))?;
        Some(self.gpu_handle(slot))
    }

    fn gpu_handle(&self, slot: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let base = unsafe { self.visible.GetGPUDescriptorHandleForHeapStart() }.ptr;
        D3D12_GPU_DESCRIPTOR_HANDLE { ptr: base + u64::from(slot) * self.stride as u64 }
    }

    pub(crate) fn gpu_table(&self, address: u64, length: u32, active: &[std::ops::Range<u32>]) -> Result<D3D12_GPU_DESCRIPTOR_HANDLE> {
        let offset = (address as usize)
            .checked_sub(self.base())
            .ok_or(E_INVALIDARG)?;
        if length == 0 || !offset.is_multiple_of(self.stride) {
            return Err(E_INVALIDARG.into());
        }
        let first = u32::try_from(offset / self.stride).map_err(|_| E_INVALIDARG)?;
        if first.checked_add(length).is_none_or(|end| end > self.count) {
            return Err(E_INVALIDARG.into());
        }
        let mut tables = self.tables.lock().unwrap_or_else(|e| e.into_inner());
        let key = (first, length);
        let resident = if first + length <= self.identity {
            first
        } else if let Some(index) = tables.ranges.get(&key) {
            *index
        } else if !self.inline && let Some(index) = tables.direct.contiguous(first, length) {
            index
        } else {
            if self.identity != 0 && tables.used.checked_add(length).is_none_or(|end| end > self.limit) {
                // LOCAL: reuse the aliases of long-retired frames from the start.
                static WRAPS: AtomicU64 = AtomicU64::new(0);
                let n = WRAPS.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() { crate::diagnostics::log(&format!("HEAP aliases recycled ({n} so far, {} tables)", tables.ranges.len())); }
                tables.used = self.identity;
                tables.ranges.clear();
                tables.resident_records.forget_aliases();
            }
            let index = tables.used;
            let end = index
                .checked_add(length)
                .filter(|&n| n <= self.limit)
                .ok_or(E_OUTOFMEMORY)?;
            if !self.inline {
                self.copy(first, index, length);
            }
            tables.ranges.insert(key, index);
            tables.used = end;
            index
        };
        if self.inline {
            let start = unsafe { self.visible.GetCPUDescriptorHandleForHeapStart() };
            let mut materialized = 0;
            for index in active.iter().flat_map(|range| range.clone().take_while(|&i| i < length)) {
                let index = index as usize;
                let cell = address as usize + index * INLINE_BYTES;
                let record = self.inline_record(cell);
                let slot = resident as usize + index;
                if tables.resident_records.get(slot) == Some(record) {
                    continue;
                }
                let destination = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start.ptr + slot * self.stride };
                if let Some(source) = self.resolve_inline(record) {
                    self.copy_native_view(destination, source);
                    materialized += usize::from(record.source != cell || INLINE_BYTES != self.stride);
                } else {
                    // LOCAL: the title keeps constants in the same memory, so a declared
                    // entry that the shader never reads can hold anything.
                    if record != InlineDescriptor::default() {
                        static ONCE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        let n = ONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if n.is_power_of_two() { crate::diagnostics::log(&format!(
                            "UNRESOLVED table entry {index}/{length} at slot {:#x}, source {:#x} version {} ({n} so far)",
                            first as usize + index, record.source, record.version)); }
                    }
                    // Inactive slots have no resource. Use a valid null SRV
                    // rather than copying an uninitialized host descriptor.
                    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
                    let null = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        ..Default::default()
                    };
                    self.imported.copy_to(self.stride, destination.ptr, 0, 1);
                    self.placed.copy_to(self.stride, destination.ptr, 0, 1);
                    self.scenes.copy_to(self.stride, destination.ptr, 0, 1);
                    crate::diagnostics::display::texture_view(destination.ptr, None, None);
                    crate::viewkinds::note(destination.ptr, crate::viewkinds::Kind { tag: "NULL-TEXTURE", ..crate::viewkinds::NONE });
                    unsafe { self.device.CreateShaderResourceView(None, Some(&null), destination) };
                }
                tables.resident_records.insert(slot, record);
            }
            if materialized != 0 {
                static FIRST: std::sync::Once = std::sync::Once::new();
                crate::diagnostics::log_once(&FIRST, || format!("materialized {materialized} CPU-authored descriptor records (table length {length}, native stride {})", self.stride));
            }
        }
        Ok(self.gpu_handle(resident))
    }
}

/// # Safety
/// The owner must be the controlling unknown of the live guest device.
pub(crate) unsafe fn sync_copied(owner: &windows::core::IUnknown) {
    use windows::core::Interface;
    let device = unsafe { &*owner.as_raw().cast::<super::Device>() };
    device.state.descriptor_arenas.sync_copied();
}

// Keep the logical slot and publish descriptor writes after the host call.
pub(super) struct Write {
    arena: Option<Arc<Arena>>,
    address: usize,
}
impl Write {
    pub(super) fn new(registry: &Registry, handle: &mut D3D12_CPU_DESCRIPTOR_HANDLE) -> Self {
        let address = handle.ptr;
        let arena = registry.find(address);
        if let Some(a) = &arena {
            *handle = a.cpu(address);
        }
        Self { arena, address }
    }
}
impl Drop for Write {
    fn drop(&mut self) {
        if let Some(arena) = &self.arena {
            arena.author(self.address, 1);
            arena.publish(self.address, 1);
        }
    }
}
