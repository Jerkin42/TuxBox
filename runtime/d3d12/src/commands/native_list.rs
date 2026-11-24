//! The desktop lists behind one guest list. A guest wait in the middle of a
//! recording ends the current desktop list; later commands continue in another.
use super::enhanced_barriers;
use core::ops::Deref;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, OnceLock};
use windows::Win32::Foundation::{E_FAIL, E_OUTOFMEMORY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, Result};

/// One desktop list and the newer interfaces of that same object.
pub(super) struct Part {
    pub(super) list: ID3D12GraphicsCommandList,
    extended: OnceLock<Option<ID3D12GraphicsCommandList1>>,
    enhanced_barriers: OnceLock<Option<ID3D12GraphicsCommandList7>>,
    mesh: OnceLock<Option<ID3D12GraphicsCommandList6>>,
    pub(super) raytracing: Option<ID3D12GraphicsCommandList4>,
}

impl Part {
    fn new(list: ID3D12GraphicsCommandList, raytracing: bool) -> Self {
        Self {
            raytracing: raytracing.then(|| list.cast().ok()).flatten(),
            list,
            extended: OnceLock::new(),
            enhanced_barriers: OnceLock::new(),
            mesh: OnceLock::new(),
        }
    }

    pub(super) fn extended(&self) -> Option<&ID3D12GraphicsCommandList1> {
        self.extended.get_or_init(|| self.list.cast().ok()).as_ref()
    }

    pub(super) fn enhanced_barriers(&self) -> Option<&ID3D12GraphicsCommandList7> {
        self.enhanced_barriers
            .get_or_init(|| enhanced_barriers(&self.list))
            .as_ref()
    }

    pub(super) fn mesh(&self) -> Option<&ID3D12GraphicsCommandList6> {
        self.mesh.get_or_init(|| self.list.cast().ok()).as_ref()
    }
}

/// Dereferences to the list that receives commands now. Only the thread holding
/// the recording lock changes it. Object identity (private data, name, type)
/// always belongs to the primary.
pub(super) struct Native {
    primary: Arc<Part>,
    current: AtomicPtr<Part>,
}

impl Native {
    pub(super) fn new(list: ID3D12GraphicsCommandList, raytracing: bool) -> Self {
        let primary = Arc::new(Part::new(list, raytracing));
        let current = AtomicPtr::new(Arc::as_ptr(&primary).cast_mut());
        Self { primary, current }
    }

    pub(super) fn primary(&self) -> &Part {
        &self.primary
    }

    pub(super) fn part(&self) -> &Part {
        // SAFETY: the primary lives as long as self. Every other part is owned by
        // the recording of the same proxy, which replaces only parts that are not
        // current and is dropped together with this field.
        unsafe { &*self.current.load(Ordering::Acquire) }
    }
}

impl Deref for Native {
    type Target = ID3D12GraphicsCommandList;
    fn deref(&self) -> &Self::Target {
        &self.part().list
    }
}

impl<'a> From<&'a Native> for &'a ID3D12Object {
    fn from(native: &'a Native) -> Self {
        (&native.primary.list).into()
    }
}

/// Continuations after the primary. The desktop objects are reused across Reset.
#[derive(Default)]
pub(super) struct Parts {
    pool: Vec<Arc<Part>>,
    used: usize,
    /// The current list is closed and the next one has not been opened.
    pub(super) parted: bool,
}

impl Parts {
    pub(super) fn rewind(&mut self, native: &Native) {
        native
            .current
            .store(Arc::as_ptr(&native.primary).cast_mut(), Ordering::Release);
        self.used = 0;
        self.parted = false;
    }

    /// Open the next list on the guest's allocator. The preceding list is closed,
    /// so the allocator still has a single open list, and the guest cannot reset
    /// it before the whole recording has completed.
    pub(super) fn open(&mut self, native: &Native, allocator: &ID3D12CommandAllocator) -> Result<()> {
        let pooled = self
            .pool
            .get(self.used)
            .filter(|part| unsafe { part.list.Reset(allocator, None) }.is_ok());
        let part = match pooled {
            Some(part) => part.clone(),
            None => {
                let mut device: Option<ID3D12Device> = None;
                unsafe { native.primary.list.GetDevice(&mut device)? };
                let kind = unsafe { native.primary.list.GetType() };
                let list: ID3D12GraphicsCommandList =
                    unsafe { device.ok_or(E_FAIL)?.CreateCommandList(0, kind, allocator, None)? };
                let part = Arc::new(Part::new(list, native.primary.raytracing.is_some()));
                match self.pool.get_mut(self.used) {
                    Some(slot) => *slot = part.clone(),
                    None => {
                        self.pool.try_reserve(1).map_err(|_| E_OUTOFMEMORY)?;
                        self.pool.push(part.clone());
                    }
                }
                part
            }
        };
        self.used += 1;
        self.parted = false;
        native
            .current
            .store(Arc::as_ptr(&part).cast_mut(), Ordering::Release);
        Ok(())
    }

    /// Desktop lists that are already closed at this point of the recording.
    pub(super) fn closed(&self) -> usize {
        self.used + usize::from(self.parted)
    }

    /// All lists of a closed recording in execution order.
    pub(super) fn lists<'a>(&'a self, native: &'a Native) -> impl Iterator<Item = &'a ID3D12GraphicsCommandList> {
        core::iter::once(&native.primary)
            .chain(&self.pool[..self.used])
            .map(|part| &part.list)
    }
}
