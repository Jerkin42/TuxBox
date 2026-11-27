//! Decoders for the Xbox DirectStorage hardware stages that have public
//! descriptions. Platform-independent, so they are tested on any host.
//!
//! BCPack itself is not here yet. Its entropy layer is the rANS fragment
//! decoder in [`rans`]; the block layer above it is not publicly documented.
pub mod rans;
