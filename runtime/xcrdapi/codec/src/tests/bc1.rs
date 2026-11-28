use super::*;
use crate::encode::{Random, stream};

/// The lane symbols that encode `blocks` with the given references.
fn symbols(blocks: &[[u8; 8]], references: &[usize]) -> Vec<Vec<u16>> {
    let mut code = [0u8; 256];
    for (symbol, codes) in QUADS.iter().enumerate() {
        let key = codes.iter().fold(0usize, |k, &c| k << 2 | usize::from(c));
        code[key] = symbol as u8;
    }
    let fields = |b: &[u8; 8]| {
        let c0 = u16::from_le_bytes([b[0], b[1]]);
        let c1 = u16::from_le_bytes([b[2], b[3]]);
        [
            (c0 >> 11) as u8,
            (c0 >> 5 & 63) as u8,
            (c0 & 31) as u8,
            (c1 >> 11) as u8,
            (c1 >> 5 & 63) as u8,
            (c1 & 31) as u8,
        ]
    };
    let mut lanes = vec![Vec::new(); 12];
    for (i, (block, &d)) in blocks.iter().zip(references).enumerate() {
        let prediction = if d == 0 {
            [0; 6]
        } else if d <= i {
            fields(&blocks[i - d])
        } else {
            WINDOW[(i + 256 - d) % 256].unwrap()
        };
        lanes[0].push(if d < 128 {
            2 * d as u16
        } else {
            2 * (255 - d as u16) + 1
        });
        lanes[1].push(0);
        for (k, (&f, &p)) in fields(block).iter().zip(&prediction).enumerate() {
            let size = 1i32 << FIELD_BITS[k];
            let mut r = (i32::from(f) - i32::from(p)).rem_euclid(size);
            if r >= size / 2 {
                r -= size;
            }
            lanes[2 + k].push(if r >= 0 {
                2 * r as u16
            } else {
                (-2 * r - 1) as u16
            });
        }
        let index = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        for (q, (ox, oy)) in [(0, 0), (2, 0), (0, 2), (2, 2)].into_iter().enumerate() {
            let key = [(0, 0), (1, 0), (0, 1), (1, 1)]
                .iter()
                .fold(0usize, |k, (dx, dy)| {
                    k << 2 | (index >> (2 * (4 * (oy + dy) + ox + dx)) & 3) as usize
                });
            lanes[8 + q].push(u16::from(code[key]));
        }
    }
    lanes
}

#[test]
fn the_quadrant_codebook_is_a_bijection() {
    let mut seen = [false; 256];
    for codes in &QUADS {
        let key = codes.iter().fold(0usize, |k, &c| k << 2 | usize::from(c));
        assert!(!seen[key]);
        seen[key] = true;
    }
}

#[test]
fn random_blocks_round_trip() {
    let mut random = Random(0xbc1);
    for (count, longest) in [(1usize, 1usize), (64, 150), (700, 150), (3000, 2000)] {
        let blocks: Vec<[u8; 8]> = (0..count).map(|_| random.next().to_le_bytes()).collect();
        let references: Vec<usize> = (0..count)
            .map(|i| match random.below(4) {
                0 => 0,
                1 => 1 + random.below(i.min(255) as u64 + 1) as usize % 255,
                // Before the first block: only slots that have a fit.
                _ => loop {
                    let d = 1 + random.below(255) as usize;
                    if d <= i || WINDOW[(i + 256 - d) % 256].is_some() {
                        break d;
                    }
                },
            })
            .collect();
        let data = stream(
            &WIDTHS,
            &symbols(&blocks, &references),
            longest,
            &mut random,
        );
        let mut out = vec![0u8; count * BLOCK_BYTES];
        let report = decode(&data, &mut out).unwrap();
        assert_eq!(out, blocks.concat());
        let early = references
            .iter()
            .enumerate()
            .filter(|&(i, &d)| d > i)
            .count();
        assert_eq!(
            report,
            Report {
                window_references: early,
                unfitted_blocks: 0
            }
        );
    }
}

#[test]
fn sizes_that_are_not_whole_blocks_are_refused() {
    assert_eq!(decode(&[0; 64], &mut []), Err(Error::BadAlphabet));
    assert_eq!(decode(&[0; 64], &mut [0; 12]), Err(Error::BadAlphabet));
}

/// Four blocks the storage guest decodes through DirectStorage: block 1
/// repeats block 0's endpoints, block 3 reads window slot 255.
fn guest_fixture() -> (Vec<u8>, Vec<u8>) {
    let blocks: Vec<[u8; 8]> = [
        0x5555_aaaa_07e0_f800u64,
        0x0123_4567_07e0_f800,
        0xffff_0000_001f_ffff,
        0x1b1b_e4e4_8410_4208,
    ]
    .iter()
    .map(|b| b.to_le_bytes())
    .collect();
    let references = [0, 1, 0, 4];
    let mut random = Random(0x6e57);
    let data = stream(&WIDTHS, &symbols(&blocks, &references), 2, &mut random);
    (data, blocks.concat())
}

#[test]
fn the_guest_fixture_decodes() {
    let (data, blocks) = guest_fixture();
    if std::env::var_os("PRINT_FIXTURE").is_some() {
        let hex = |b: &[u8]| {
            b.iter()
                .map(|v| format!("0x{v:02x}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!(
            "stream {} bytes\n{}\nblocks\n{}",
            data.len(),
            hex(&data),
            hex(&blocks)
        );
    }
    assert_eq!(data, GUEST_STREAM);
    let mut out = vec![0u8; blocks.len()];
    decode(&data, &mut out).unwrap();
    assert_eq!(out, blocks);
}

/// The same bytes as `bc1` in runtime/xcrdapi/guest.cpp.
const GUEST_STREAM: &[u8] = &[
    0x1d, 0x18, 0x1f, 0x1d, 0x1b, 0x10, 0x15, 0x13, 0x03, 0x03, 0x05, 0x03, 0x03, 0x04, 0x03, 0x03,
    0x17, 0x19, 0x01, 0x01, 0x02, 0x04, 0x10, 0x0c, 0x8d, 0xd5, 0x00, 0x38, 0x8d, 0x01, 0x36, 0xf2,
    0x36, 0x00, 0xff, 0x98, 0x85, 0x00, 0x1c, 0x06, 0x3b, 0x24, 0xe1, 0x81, 0x54, 0x02, 0x48, 0xd5,
    0x00, 0x05, 0x18, 0x14, 0x1e, 0x15, 0x13, 0x00, 0x11, 0x13, 0x1f, 0x04, 0x03, 0x05, 0x04, 0x19,
    0x03, 0x03, 0x05, 0x0f, 0x4c, 0x1c, 0x27, 0x03, 0x2d, 0x19, 0x01, 0xf9, 0xeb, 0x36, 0x76, 0x02,
    0x82, 0x70, 0x01, 0x02, 0x01, 0xf9, 0x02, 0xd0, 0x04, 0x09, 0x20, 0x3e, 0xe5, 0x35, 0x7b, 0x56,
    0xc1, 0x01, 0x02, 0x0a, 0x00, 0x03, 0x51, 0x11, 0x3d, 0x04, 0xa6, 0xa9, 0x10, 0xf0, 0x73, 0x7b,
    0x5d, 0x00, 0xc5, 0x19, 0x13, 0x17, 0x1a, 0x1d, 0x1a, 0x19, 0x14, 0x17, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x00, 0x64, 0x06, 0x05, 0x1f, 0x02, 0x05, 0x4f, 0x1e, 0x83, 0x0a, 0x0b, 0x72,
    0x20, 0x13, 0x26, 0x03, 0x02, 0x53, 0x6b, 0xfb, 0x01, 0x30, 0x07, 0x04, 0x5d, 0xdb, 0xa8, 0x8c,
    0x01, 0x23, 0xb8, 0xfc, 0xdd, 0x05, 0x43, 0x13, 0xcb, 0xff, 0xba, 0x0f, 0x24, 0x16, 0x15, 0x16,
    0x1d, 0x11, 0x18, 0x17, 0x00, 0x03, 0x03, 0x04, 0x05, 0x03, 0x04, 0x04, 0x19, 0x3a, 0x3b, 0x01,
    0x01, 0x1d, 0x09, 0x05, 0x04, 0x83, 0x33, 0x20, 0x00, 0xca, 0xd8, 0x72, 0x40, 0x2c, 0x02, 0xf2,
    0xff, 0x86, 0xff, 0xff, 0x00, 0xa0, 0x78, 0x70, 0xe0, 0x50, 0xf6, 0xe3, 0xff, 0x00, 0x00, 0x00,
    0x00, 0xfc, 0x1d, 0x18, 0x18, 0x00, 0x00, 0x05, 0x03, 0x03, 0x10, 0x01, 0x28, 0x28, 0x04, 0x00,
    0x81, 0x81, 0x01, 0xff, 0x59, 0x52, 0x00, 0x24, 0x04, 0xb2, 0x22, 0x00, 0x00, 0x00, 0x00,
];
