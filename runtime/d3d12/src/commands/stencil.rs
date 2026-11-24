//! Snapshot immediate CPU descriptors and surround an emulated stencil-export
//! draw with copies of just the stencil plane. Both resources return to their
//! incoming state before control is returned to the guest command list.
//! The same snapshot binds the targets again after a mid-list wait.
use super::{CopyLocation, TransitionBarrier};
use crate::device::stencil::{Mirror, Surface, cpu_heap};
use core::ptr;
use std::sync::Arc;
use windows::Win32::Foundation::E_NOTIMPL;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, Result};

pub(super) struct Targets {
    device: ID3D12Device,
    host_rtv: ID3D12DescriptorHeap,
    _host_dsv: ID3D12DescriptorHeap,
    handles: [D3D12_CPU_DESCRIPTOR_HANDLE; 8],
    count: usize,
    dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    bound: bool,
    dsv_bound: bool,
    surface: Option<Arc<Surface>>,
}

impl Targets {
    pub(super) fn new(device: &ID3D12Device) -> Result<Self> {
        let host_rtv = cpu_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 8)?;
        let host_dsv = cpu_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1)?;
        let dsv = unsafe { host_dsv.GetCPUDescriptorHandleForHeapStart() };
        Ok(Self {
            device: device.clone(),
            host_rtv,
            _host_dsv: host_dsv,
            handles: Default::default(),
            count: 0,
            dsv,
            bound: false,
            dsv_bound: false,
            surface: None,
        })
    }

    pub(super) fn unbind(&mut self) {
        self.surface = None;
        self.bound = false;
    }

    pub(super) unsafe fn snapshot(
        &mut self,
        surface: Option<Arc<Surface>>,
        count: u32,
        rtvs: *const D3D12_CPU_DESCRIPTOR_HANDLE,
        contiguous: bool,
        dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    ) {
        let stride = unsafe {
            self.device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
        } as usize;
        let first = unsafe { self.host_rtv.GetCPUDescriptorHandleForHeapStart() }.ptr;
        for i in 0..count as usize {
            let handle = if contiguous && unsafe { (*rtvs).ptr } != 0 {
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: unsafe { (*rtvs).ptr } + i * stride,
                }
            } else {
                unsafe { *rtvs.add(if contiguous { 0 } else { i }) }
            };
            if handle.ptr == 0 {
                self.handles[i] = handle;
                continue;
            }
            let copy = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: first + i * stride,
            };
            unsafe {
                self.device
                    .CopyDescriptorsSimple(1, copy, handle, D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
            };
            self.handles[i] = copy;
        }
        let dsv = dsv.filter(|dsv| dsv.ptr != 0);
        if let Some(dsv) = dsv {
            unsafe {
                self.device
                    .CopyDescriptorsSimple(1, self.dsv, dsv, D3D12_DESCRIPTOR_HEAP_TYPE_DSV)
            };
        }
        self.surface = surface;
        self.count = count as usize;
        self.bound = true;
        self.dsv_bound = dsv.is_some();
    }

    pub(super) unsafe fn rebind(&self, list: &ID3D12GraphicsCommandList) {
        if self.bound {
            unsafe {
                list.OMSetRenderTargets(
                    self.count as u32,
                    (self.count != 0).then_some(self.handles.as_ptr()),
                    false,
                    self.dsv_bound.then_some(core::ptr::from_ref(&self.dsv)),
                );
            }
        }
    }

    pub(super) fn draw(
        &self,
        list: &ID3D12GraphicsCommandList,
        slot: u32,
        draw: impl FnOnce(),
    ) -> Result<()> {
        if self.count != slot as usize || self.count >= 8 {
            return Err(E_NOTIMPL.into());
        }
        let surface = self.surface.as_ref().ok_or(E_NOTIMPL)?;
        let mirror = surface.mirror(&self.device)?;
        let mut handles = self.handles;
        handles[self.count] = mirror.rtv();
        copy_plane(list, &surface.depth, mirror, true);
        unsafe {
            list.OMSetRenderTargets(slot + 1, Some(handles.as_ptr()), false, Some(&self.dsv));
        }
        draw();
        unsafe {
            list.OMSetRenderTargets(slot, Some(self.handles.as_ptr()), false, Some(&self.dsv));
        }
        copy_plane(list, &surface.depth, mirror, false);
        Ok(())
    }
}

fn copy_plane(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    mirror: &Mirror,
    save: bool,
) {
    let image = |res: &ID3D12Resource, sub| CopyLocation {
        resource: res.as_raw(),
        kind: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        data: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: sub,
        },
    };
    let a = image(depth, 1);
    let b = image(&mirror.color, 0);
    let through = CopyLocation {
        resource: mirror.transfer.as_raw(),
        kind: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        data: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: mirror.layout,
        },
    };
    let depth_state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    let rtv_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    let (src, src_state, dst, dst_state) = if save {
        (&a, depth_state, &b, rtv_state)
    } else {
        (&b, rtv_state, &a, depth_state)
    };
    let transition = |loc: &CopyLocation, sub, before, after| {
        let b = TransitionBarrier {
            resource: loc.resource,
            kind: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            subresource: sub,
            before,
            after,
        };
        unsafe {
            (list.vtable().ResourceBarrier)(list.as_raw(), 1, ptr::from_ref(&b).cast());
        }
    };
    let src_sub = unsafe { src.data.SubresourceIndex };
    let dst_sub = unsafe { dst.data.SubresourceIndex };
    let cs = D3D12_RESOURCE_STATE_COPY_SOURCE;
    let cd = D3D12_RESOURCE_STATE_COPY_DEST;
    transition(src, src_sub, src_state, cs);
    transition(dst, dst_sub, dst_state, cd);
    let copy = |dst: &CopyLocation, src: &CopyLocation| unsafe {
        (list.vtable().CopyTextureRegion)(
            list.as_raw(),
            ptr::from_ref(dst).cast(),
            0,
            0,
            0,
            ptr::from_ref(src).cast(),
            ptr::null(),
        )
    };
    copy(&through, src);
    transition(&through, 0, cd, cs);
    copy(dst, &through);
    transition(src, src_sub, cs, src_state);
    transition(dst, dst_sub, cd, dst_state);
    transition(&through, 0, cs, cd);
}
