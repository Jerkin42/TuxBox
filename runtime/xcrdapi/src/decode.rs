//! The hardware decode stages of a request, in the documented order: zlib
//! (RFC 1950), then BCPack. Swizzle is not implemented.
use std::borrow::Cow;
use tuxbox_xcrdapi_codec::bc1;

/// `ZlibDecompress`, bit 0 of the request options.
pub(crate) const ZLIB: u64 = 1;
/// `BcpackMode`, bits 1-4: the BC format number, 0 for none.
pub(crate) const BCPACK: u64 = 0xf << 1;
const BC1: u64 = 1;

/// True when this provider decodes the BCPack mode in `options`.
pub(crate) fn supported(options: u64) -> bool {
    matches!(bcpack_mode(options), 0 | BC1)
}

fn bcpack_mode(options: u64) -> u64 {
    (options & BCPACK) >> 1
}

/// Runs the stages `options` selects over `input`. `capacity` is the
/// destination size; `intermediate` is the zlib output size when BCPack
/// follows zlib.
pub(crate) fn run(
    input: &[u8],
    options: u64,
    capacity: usize,
    intermediate: usize,
) -> Result<Cow<'_, [u8]>, ()> {
    let mode = bcpack_mode(options);
    let packed = if options & ZLIB != 0 {
        let limit = if mode == 0 { capacity } else { intermediate };
        Cow::Owned(
            miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(input, limit)
                .map_err(|_| ())?,
        )
    } else {
        Cow::Borrowed(input)
    };
    match mode {
        0 => Ok(packed),
        BC1 => {
            let mut blocks = Vec::new();
            blocks.try_reserve_exact(capacity).map_err(|_| ())?;
            blocks.resize(capacity, 0);
            bc1::decode(&packed, &mut blocks).map_err(|_| ())?;
            Ok(Cow::Owned(blocks))
        }
        _ => Err(()),
    }
}
