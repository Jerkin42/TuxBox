// bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
//
// Written for TuxBox after docs/BCPACK_SPEC.md and src/bc1.rs (`assemble`) of
// bcpack-rs 5bf5cc56ed56f1ababf1f9592300bd182ff90bd8, used under Apache-2.0.
//! BCPack BC1: 12 rANS lanes of one symbol per block, rebuilt into 8-byte
//! BC1 blocks.
//!
//! | Lane | Width | Symbol |
//! | --- | --- | --- |
//! | 0 | 8 | back reference: `d = unzigzag(v) mod 256` blocks back, 0 for none |
//! | 1 | 8 | unused, 0 in all retail data bcpack-rs measured |
//! | 2-7 | 5 6 5 5 6 5 | zigzag residuals of R0 G0 B0 R1 G1 B1 |
//! | 8-11 | 8 | index codebook symbols of the 2x2 quadrants TL TR BL BR |
//!
//! A block's endpoint fields are those of the block `d` back plus the
//! residuals, each modulo its width. A reference before the first block reads
//! slot `(i - d) mod 256` of a fixed window. The window and the quadrant
//! codebook are not in the stream and are not published: the values used here
//! were fitted on retail data ([`crate::bc1_tables`]), so a decoded block can
//! differ from the console's.
use crate::bc1_tables::{QUADS, WINDOW};
use crate::lanes;
use crate::models;
use crate::rans::{Error, Models};
use std::sync::LazyLock;

/// Symbol widths of the 12 lanes.
pub const WIDTHS: [u8; 12] = [8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8];
/// Widths of the endpoint fields R0 G0 B0 R1 G1 B1.
pub const FIELD_BITS: [u8; 6] = [5, 6, 5, 5, 6, 5];
/// Bytes of one BC1 block.
pub const BLOCK_BYTES: usize = 8;

static MODELS: LazyLock<Models> = LazyLock::new(models::silicon_rans);

/// Index-word bits each codebook symbol sets, per quadrant. Pixel (x, y) is
/// at bits `2 * (4 * y + x)`.
static QUAD_BITS: LazyLock<[[u32; 256]; 4]> = LazyLock::new(|| {
    let mut bits = [[0u32; 256]; 4];
    for (quadrant, (ox, oy)) in [(0, 0), (2, 0), (0, 2), (2, 2)].into_iter().enumerate() {
        for (symbol, codes) in QUADS.iter().enumerate() {
            for (k, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                bits[quadrant][symbol] |=
                    u32::from(codes[k] & 3) << (2 * (4 * (oy + dy) + ox + dx));
            }
        }
    }
    bits
});

/// What a decode relied on that is not exact.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Report {
    /// References that read the fitted window.
    pub window_references: usize,
    /// Blocks whose reference chain reaches a window slot with no fit.
    pub unfitted_blocks: usize,
}

/// Lane 0 symbol to the reference distance.
pub fn distance(v: u16) -> usize {
    let v = usize::from(v & 0xff);
    if v & 1 == 0 { v >> 1 } else { 255 - (v >> 1) }
}

/// 0, 1, 2, 3, 4 to 0, -1, 1, -2, 2.
pub fn unzigzag(v: u16) -> i32 {
    let v = i32::from(v);
    if v & 1 != 0 { -((v + 1) >> 1) } else { v >> 1 }
}

/// Decodes a BCPack BC1 stream into `out`, which holds `out.len() / 8`
/// blocks in stream order. Only zero bytes may follow the stream.
pub fn decode(data: &[u8], out: &mut [u8]) -> Result<Report, Error> {
    if out.is_empty() || !out.len().is_multiple_of(BLOCK_BYTES) {
        return Err(Error::BadAlphabet);
    }
    let blocks = out.len() / BLOCK_BYTES;
    let lanes = lanes::decode_slice(data, &WIDTHS, blocks, &MODELS)?;
    Ok(assemble(&lanes, out))
}

/// Rebuilds BC1 blocks from the 12 decoded lanes.
pub fn assemble(lanes: &lanes::Symbols, out: &mut [u8]) -> Report {
    let quads = &*QUAD_BITS;
    let mut fields = vec![[0u8; 6]; lanes.per_lane];
    let mut unfitted = vec![false; lanes.per_lane];
    let mut report = Report::default();
    for (i, block) in out
        .chunks_exact_mut(BLOCK_BYTES)
        .enumerate()
        .take(lanes.per_lane)
    {
        let d = distance(lanes.lane(0)[i]);
        let (prediction, inherited) = if d == 0 {
            ([0; 6], false)
        } else if d <= i {
            (fields[i - d], unfitted[i - d])
        } else {
            report.window_references += 1;
            let slot = (i + 256 - d % 256) % 256;
            WINDOW[slot].map_or(([0; 6], true), |f| (f, false))
        };
        let mut e = [0u8; 6];
        for k in 0..6 {
            let mask = (1i32 << FIELD_BITS[k]) - 1;
            e[k] = ((i32::from(prediction[k]) + unzigzag(lanes.lane(2 + k)[i])) & mask) as u8;
        }
        fields[i] = e;
        unfitted[i] = inherited;
        report.unfitted_blocks += usize::from(inherited);
        let c0 = u16::from(e[0]) << 11 | u16::from(e[1]) << 5 | u16::from(e[2]);
        let c1 = u16::from(e[3]) << 11 | u16::from(e[4]) << 5 | u16::from(e[5]);
        let mut index = 0u32;
        for (q, bits) in quads.iter().enumerate() {
            index |= bits[usize::from(lanes.lane(8 + q)[i] & 0xff)];
        }
        block[0..2].copy_from_slice(&c0.to_le_bytes());
        block[2..4].copy_from_slice(&c1.to_le_bytes());
        block[4..8].copy_from_slice(&index.to_le_bytes());
    }
    report
}

#[cfg(test)]
#[path = "tests/bc1.rs"]
mod tests;
