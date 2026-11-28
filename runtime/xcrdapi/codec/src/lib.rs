//! Decoders for the Xbox DirectStorage hardware stages. Platform-independent,
//! so they are tested on any host.
//!
//! - [`rans`]: one lane of the MDU rANS decoder (Microsoft's silicon-rans);
//! - [`lanes`]: several lanes sharing one stream, as BCPack uses them;
//! - [`bc1`]: BCPack BC1 lanes rebuilt into BC1 blocks.
//!
//! The other BCPack formats are not decoded: their constant tables have no
//! source TuxBox can use.
pub mod bc1;
mod bc1_tables;
pub mod lanes;
pub mod models;
pub mod rans;

#[cfg(test)]
#[path = "tests/encode.rs"]
mod encode;
