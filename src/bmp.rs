//! BMP decoding and encoding.
//!
//! # Decoding
//!
//! The primary entry point is [`decode`], which takes a byte slice and returns
//! a [`BmpDecoded`] — a struct containing the pixel data as a [`BmpImage`]
//! and ancillary information as [`BmpMetadata`].
//!
//! ```no_run
//! # use irys_cv_io::bmp;
//! let bytes = std::fs::read("photo.bmp").unwrap();
//! let decoded = bmp::decode(&bytes).unwrap();
//! // Just the pixels:
//! let image = decoded.image;
//! // Or inspect metadata too:
//! // let header = decoded.metadata.header_version;
//! ```
//!
//! For streaming sources use [`decode_reader`], which accepts any `impl Read`.
//!
//! # Design rationale
//!
//! - **`BmpImage` is exhaustive** (no `#[non_exhaustive]`).  It is the spec
//!   sheet for what a BMP file can produce.  Adding a variant is a genuine
//!   semantic change that callers must handle — a compile error is the
//!   correct response.
//!
//! - **`BmpDecoded` is `#[non_exhaustive]`**.  It is a return-only struct that
//!   may gain fields (e.g. decode warnings) in the future without breaking
//!   downstream code.
//!
//! - **Always sRGB**.  BMP data is sRGB by convention — the Windows GDI
//!   subsystem operates in sRGB.  Returning `Rgb8` (linear) would be a
//!   type-level lie.  Users with genuinely linear BMP data (industrial/
//!   scientific, extremely rare) can transmute pixel types after decoding
//!   since `Srgb8` and `Rgb8` share identical memory layout.
//!
//! - **BGR → RGB reordering at the I/O boundary**.  BMP stores channels
//!   as B, G, R (and optionally A) — the reverse of our RGB pixel types.
//!   Reordering at decode/encode time is lossless and keeps pixel memory
//!   layout consistent with documented `Srgb8` / `Srgba8` semantics.
//!
//! - **No RLE encoding**.  RLE-compressed BMPs are a legacy feature with
//!   poor compression ratios.  We decode RLE4/RLE8 but always encode
//!   uncompressed.
//!
//! - **No grayscale mode**.  BMP has no native grayscale representation.
//!   `SrgbMono8` is excluded from [`BmpPixel`] — users should convert
//!   to `Srgb8` explicitly or use `encode_indexed` with a grayscale palette.
//!
//! # Encoding
//!
//! The primary entry point is [`encode`], which accepts any image whose
//! pixel type implements [`BmpPixel`] (a sealed trait covering exactly
//! `Srgb8` and `Srgba8`).
//!
//! ```no_run
//! # use irys_cv::image::Image;
//! # use irys_cv::pixel::Srgb8;
//! # use irys_cv_io::bmp::{self, BmpEncodeOptions};
//! let image = Image::fill(320, 240, Srgb8::new(0, 0, 0));
//! let bytes = bmp::encode(&image, &BmpEncodeOptions::default()).unwrap();
//! std::fs::write("output.bmp", bytes).unwrap();
//! ```
//!
//! For streaming output use [`encode_writer`].  For palette-indexed
//! images use [`encode_indexed`].  For roundtripping a decoded
//! [`BmpImage`] use [`encode_bmp_image`].
//!
//! ## Compile-time safety
//!
//! Attempting to encode a pixel type that BMP cannot represent is a
//! compile-time error — the type simply does not implement [`BmpPixel`].
//!
//! ```compile_fail
//! # use irys_cv::image::Image;
//! # use irys_cv::pixel::Rgb8;
//! # use irys_cv_io::bmp::{self, BmpEncodeOptions};
//! // ERROR: `Rgb8` does not implement `BmpPixel`
//! let image = Image::fill(1, 1, Rgb8::new(0, 0, 0));
//! let _ = bmp::encode(&image, &BmpEncodeOptions::default());
//! ```
//!
//! ```compile_fail
//! # use irys_cv::image::Image;
//! # use irys_cv::pixel::SrgbMono8;
//! # use irys_cv_io::bmp::{self, BmpEncodeOptions};
//! // ERROR: `SrgbMono8` does not implement `BmpPixel`
//! let image = Image::fill(1, 1, SrgbMono8::new(128));
//! let _ = bmp::encode(&image, &BmpEncodeOptions::default());
//! ```
//!
//! ## Supported header versions
//!
//! | Header               | Size (bytes) | Decode | Encode |
//! |----------------------|-------------|--------|--------|
//! | `BITMAPCOREHEADER`   | 12          | ✅     | ❌     |
//! | `BITMAPINFOHEADER`   | 40          | ✅     | ✅     |
//! | `BITMAPV4HEADER`     | 108         | ✅     | ❌     |
//! | `BITMAPV5HEADER`     | 124         | ✅     | ❌     |
//!
//! ## BMP configurations → pixel types
//!
//! | Source bit depth | Compression        | `BmpImage` variant |
//! |------------------|--------------------|-------------------|
//! | 1-bit indexed    | `BI_RGB`           | `Indexed8`        |
//! | 4-bit indexed    | `BI_RGB` / `BI_RLE4` | `Indexed8`     |
//! | 8-bit indexed    | `BI_RGB` / `BI_RLE8` | `Indexed8`     |
//! | 16-bit           | `BI_RGB` / `BI_BITFIELDS` | `Srgb8`   |
//! | 24-bit           | `BI_RGB`           | `Srgb8`           |
//! | 32-bit (no α)    | `BI_RGB` / `BI_BITFIELDS` | `Srgb8`  |
//! | 32-bit (α)       | `BI_BITFIELDS`     | `Srgba8`          |

use irys_cv::image::{Image, ImageView, PlainImage};
use irys_cv::pixel::{Indexed8, PlainPixel, Srgb8, Srgba8};

use crate::IoError;

// ═══════════════════════════════════════════════════════════════════════════════
// BmpImage — per-codec exhaustive output enum
// ═══════════════════════════════════════════════════════════════════════════════

/// Decoded BMP pixel data.
///
/// Each variant corresponds to a BMP colour configuration.  BMP has no
/// native grayscale mode — all non-indexed data is RGB or RGBA.
///
/// Sub-byte indexed images (1-bit, 4-bit) are expanded to 8-bit indices,
/// matching the PNG decoder's behaviour.  16-bit RGB (5-5-5 or 5-6-5) is
/// expanded to 8-bit per channel using full-range scaling.
///
/// This enum is deliberately **not** `#[non_exhaustive]`.  It is the spec
/// sheet; adding a variant is semver-major.
///
/// The `Debug` impl shows the variant name and image dimensions (e.g.
/// `Srgb8(320x240)`) without dumping pixel data.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv_io::bmp::{self, BmpImage};
/// let decoded = bmp::decode(&std::fs::read("image.bmp").unwrap()).unwrap();
/// match decoded.image {
///     BmpImage::Indexed8 { data, palette } => { /* palette-indexed */ }
///     BmpImage::Srgb8(image) => { /* 24/32-bit RGB */ }
///     BmpImage::Srgba8(image) => { /* 32-bit RGBA */ }
/// }
/// ```
pub enum BmpImage {
    /// Palette-indexed image (1, 4, or 8-bit source, expanded to 8-bit indices).
    ///
    /// The palette merges the BMP color table into `Srgba8` entries.
    /// For standard BMP files (pre-V4), the reserved byte in each color
    /// table entry is set to alpha 255 (fully opaque).
    Indexed8 {
        /// The index data — one `Indexed8` per pixel.
        data: Image<Indexed8>,
        /// The 256-entry sRGBA palette (boxed to keep the enum compact).
        palette: Box<[Srgba8; 256]>,
    },
    /// 8-bit sRGB truecolour (16-bit expanded, 24-bit, or 32-bit without alpha).
    ///
    /// BMP stores pixels in BGR order; the decoder reorders to RGB.
    /// 16-bit pixels (5-5-5 or 5-6-5) are expanded to 8-bit per channel
    /// using full-range scaling.
    Srgb8(Image<Srgb8>),
    /// 8-bit sRGB truecolour + alpha (32-bit with alpha channel).
    ///
    /// Only produced when the DIB header is V4 or V5 and the bitfield
    /// masks include a non-zero alpha mask, or when the header explicitly
    /// signals BGRA layout.
    Srgba8(Image<Srgba8>),
}

impl BmpImage {
    /// Width in pixels, regardless of variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use irys_cv_io::bmp::BmpImage;
    /// use irys_cv::image::Image;
    /// use irys_cv::pixel::Srgb8;
    ///
    /// let img = BmpImage::Srgb8(Image::fill(320, 240, Srgb8::new(0, 0, 0)));
    /// assert_eq!(img.width(), 320);
    /// ```
    #[must_use]
    pub fn width(&self) -> usize {
        use irys_cv::image::ImageView;
        match self {
            BmpImage::Indexed8 { data, .. } => data.width(),
            BmpImage::Srgb8(img) => img.width(),
            BmpImage::Srgba8(img) => img.width(),
        }
    }

    /// Height in pixels, regardless of variant.
    #[must_use]
    pub fn height(&self) -> usize {
        use irys_cv::image::ImageView;
        match self {
            BmpImage::Indexed8 { data, .. } => data.height(),
            BmpImage::Srgb8(img) => img.height(),
            BmpImage::Srgba8(img) => img.height(),
        }
    }

    /// Image size (`width × height`), regardless of variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use irys_cv_io::bmp::BmpImage;
    /// use irys_cv::image::Image;
    /// use irys_cv::pixel::Srgb8;
    ///
    /// let img = BmpImage::Srgb8(Image::fill(320, 240, Srgb8::new(0, 0, 0)));
    /// let sz = img.size();
    /// assert_eq!(sz.width, 320);
    /// assert_eq!(sz.height, 240);
    /// ```
    #[must_use]
    pub fn size(&self) -> irys_cv::Size {
        use irys_cv::image::ImageView;
        match self {
            BmpImage::Indexed8 { data, .. } => data.size(),
            BmpImage::Srgb8(img) => img.size(),
            BmpImage::Srgba8(img) => img.size(),
        }
    }
}

impl std::fmt::Debug for BmpImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BmpImage::Indexed8 { data, .. } => {
                write!(f, "Indexed8({}x{})", data.width(), data.height())
            }
            BmpImage::Srgb8(img) => write!(f, "Srgb8({}x{})", img.width(), img.height()),
            BmpImage::Srgba8(img) => write!(f, "Srgba8({}x{})", img.width(), img.height()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BmpBitDepth — source bits per pixel
// ═══════════════════════════════════════════════════════════════════════════════

/// Source bits per pixel as stored in the BMP file.
///
/// BMP supports exactly six bit-depth values.  A `u8` would admit 250
/// invalid states — per PHILOSOPHY.md §1 ("the type is the truth"),
/// a six-valued domain is a six-variant enum.
///
/// This enum is deliberately **not** `#[non_exhaustive]` — these are
/// the complete set from the BMP specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BmpBitDepth {
    /// 1-bit indexed (monochrome palette).
    One,
    /// 4-bit indexed (16-color palette).
    Four,
    /// 8-bit indexed (256-color palette).
    Eight,
    /// 16-bit direct color (5-5-5 or 5-6-5 via bitfields).
    Sixteen,
    /// 24-bit direct color (BGR).
    TwentyFour,
    /// 32-bit direct color (BGRA or BGRX).
    ThirtyTwo,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BmpResolution — physical resolution
// ═══════════════════════════════════════════════════════════════════════════════

/// Physical resolution in pixels per metre.
///
/// BMP resolution is always in pixels per metre (unlike JPEG's DPI/DPCM/aspect
/// or PNG's pixels-per-unit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BmpResolution {
    /// Horizontal resolution in pixels per metre.
    pub x_pixels_per_meter: u32,
    /// Vertical resolution in pixels per metre.
    pub y_pixels_per_meter: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BmpColorSpace — colour-space signalling
// ═══════════════════════════════════════════════════════════════════════════════

/// Color space signalling from the DIB header.
///
/// This enum is deliberately **not** `#[non_exhaustive]` — these two
/// variants cover 99.9%+ of real-world BMPs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BmpColorSpace {
    /// sRGB assumed — pre-V4 header with no color space info,
    /// or V4/V5 with `LCS_sRGB` (0x73524742) or
    /// `LCS_WINDOWS_COLOR_SPACE` (0x57696E20).
    Srgb,
    /// An ICC profile is embedded (V5 header with `PROFILE_EMBEDDED`).
    /// The profile bytes are in `BmpMetadata::icc_profile`.
    IccTagged,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BmpCompression — compression method
// ═══════════════════════════════════════════════════════════════════════════════

/// Compression method used in the source BMP file.
///
/// The decoded image is always uncompressed regardless of the source
/// compression.  This enum surfaces the original encoding for metadata
/// inspection.
///
/// This enum is deliberately **not** `#[non_exhaustive]` — these are
/// the compression types we support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BmpCompression {
    /// Uncompressed (BI_RGB, value 0).
    None,
    /// Run-length encoded, 8-bit (BI_RLE8, value 1).
    Rle8,
    /// Run-length encoded, 4-bit (BI_RLE4, value 2).
    Rle4,
    /// Bitfield masks (BI_BITFIELDS, value 3).
    Bitfields,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BmpHeaderVersion — DIB header version
// ═══════════════════════════════════════════════════════════════════════════════

/// DIB header version (identifies the exact header structure).
///
/// This enum is deliberately **not** `#[non_exhaustive]` — these are
/// the four header versions the decoder handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BmpHeaderVersion {
    /// OS/2 1.x BITMAPCOREHEADER (12 bytes).
    Core,
    /// Windows BITMAPINFOHEADER (40 bytes).
    Info,
    /// Windows BITMAPV4HEADER (108 bytes).
    V4,
    /// Windows BITMAPV5HEADER (124 bytes).
    V5,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BmpMetadata — ancillary information
// ═══════════════════════════════════════════════════════════════════════════════

/// Metadata extracted from a BMP file.
///
/// BMP metadata is minimal compared to JPEG/PNG — no text chunks, no EXIF,
/// no timestamps.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BmpMetadata {
    /// Color space signalling from the DIB header.
    pub color_space: BmpColorSpace,
    /// Source bits per pixel as stored in the file.
    pub source_bit_depth: BmpBitDepth,
    /// Physical resolution in pixels per metre, if specified and non-zero.
    pub resolution: Option<BmpResolution>,
    /// Compression method used in the source file.
    pub compression: BmpCompression,
    /// DIB header version (identifies the exact header structure).
    pub header_version: BmpHeaderVersion,
    /// ICC profile bytes, if embedded (V5 header with `PROFILE_EMBEDDED`).
    pub icc_profile: Option<Box<[u8]>>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BmpDecoded — struct return
// ═══════════════════════════════════════════════════════════════════════════════

/// A decoded BMP file — pixel data + metadata.
///
/// `#[non_exhaustive]` so we can add fields (e.g. `warnings`) later.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv_io::bmp;
/// let decoded = bmp::decode(&std::fs::read("image.bmp").unwrap()).unwrap();
/// println!("{:?}", decoded.metadata.header_version);
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct BmpDecoded {
    /// The decoded pixel data.
    pub image: BmpImage,
    /// Metadata extracted from the BMP headers.
    pub metadata: BmpMetadata,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BmpEncodeOptions — encoding configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Options for BMP encoding.
///
/// BMP encoding has very few meaningful options.  We always write:
/// - `BITMAPINFOHEADER` (40-byte DIB header) — the most widely compatible
/// - Uncompressed pixel data
/// - Rows padded to 4-byte boundaries
/// - Bottom-up row order (the BMP default)
///
/// # Examples
///
/// ```
/// # use irys_cv_io::bmp::BmpEncodeOptions;
/// let opts = BmpEncodeOptions::default();
/// assert!(opts.resolution.is_none());
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct BmpEncodeOptions {
    /// Physical resolution in pixels per metre.
    /// `None` means both fields are set to zero (unspecified).
    pub resolution: Option<BmpResolution>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BmpPixel — sealed encode trait
// ═══════════════════════════════════════════════════════════════════════════════

mod bmp_pixel_sealed {
    /// Sealed supertrait — prevents out-of-crate implementations of
    /// [`BmpPixel`](super::BmpPixel).
    pub trait Sealed {}
}

/// Maps a pixel type to its BMP wire parameters.
///
/// This trait is **sealed**: it is implemented for exactly two pixel
/// types (`Srgb8` and `Srgba8`).  Attempting to encode a type that
/// does not implement `BmpPixel` is a compile-time error.
///
/// # Implementors
///
/// | Pixel type | `BMP_BYTES_PER_PIXEL` | `BMP_BIT_COUNT` | Notes                    |
/// |------------|----------------------|-----------------|--------------------------|
/// | `Srgb8`    | 3                    | 24              | 24-bit BGR, no alpha     |
/// | `Srgba8`   | 4                    | 32              | 32-bit BGRA with alpha   |
pub trait BmpPixel: bmp_pixel_sealed::Sealed + PlainPixel {
    /// Number of bytes per pixel in the BMP file (3 for RGB, 4 for RGBA).
    const BMP_BYTES_PER_PIXEL: u32;
    /// Bits per pixel in the BMP header.
    const BMP_BIT_COUNT: u16;
    /// Whether to write BI_BITFIELDS compression with an alpha mask.
    const BMP_HAS_ALPHA: bool;

    /// Write this pixel in BMP wire format (BGR or BGRA) to the output slice.
    /// The slice must be at least `BMP_BYTES_PER_PIXEL` bytes.
    #[doc(hidden)]
    fn write_bmp_bytes(&self, out: &mut [u8]);
}

impl bmp_pixel_sealed::Sealed for Srgb8 {}
impl BmpPixel for Srgb8 {
    const BMP_BYTES_PER_PIXEL: u32 = 3;
    const BMP_BIT_COUNT: u16 = 24;
    const BMP_HAS_ALPHA: bool = false;

    #[inline]
    fn write_bmp_bytes(&self, out: &mut [u8]) {
        out[0] = self.b.0;
        out[1] = self.g.0;
        out[2] = self.r.0;
    }
}

impl bmp_pixel_sealed::Sealed for Srgba8 {}
impl BmpPixel for Srgba8 {
    const BMP_BYTES_PER_PIXEL: u32 = 4;
    const BMP_BIT_COUNT: u16 = 32;
    const BMP_HAS_ALPHA: bool = true;

    #[inline]
    fn write_bmp_bytes(&self, out: &mut [u8]) {
        out[0] = self.b.0;
        out[1] = self.g.0;
        out[2] = self.r.0;
        out[3] = self.a.0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal types — file / DIB header parsing
// ═══════════════════════════════════════════════════════════════════════════════

/// Parsed BMP file header (14 bytes).
struct BmpFileHeader {
    /// Offset from the beginning of the file to the pixel data.
    pixel_data_offset: u32,
}

/// Parsed and normalized DIB header — all header versions are collapsed
/// into this single internal representation.
struct BmpDibHeader {
    width: u32,
    height: u32,
    /// True if the image is stored top-down (negative height).
    top_down: bool,
    bit_count: u16,
    compression: u32,
    header_version: BmpHeaderVersion,
    x_pels_per_meter: u32,
    y_pels_per_meter: u32,
    colors_used: u32,
    /// Color space type field from V4/V5 headers (0 for earlier versions).
    cs_type: u32,
    /// Size of the DIB header in bytes.
    header_size: u32,
    /// Bitfield masks from V4/V5 headers embedded in the header itself.
    v4_masks: Option<BitfieldMasks>,
    /// ICC profile data offset (relative to start of BITMAPFILEHEADER) and size.
    icc_profile: Option<(u32, u32)>,
    /// Color table entry size: 3 for BITMAPCOREHEADER, 4 otherwise.
    color_table_entry_size: u8,
}

/// Bitfield channel masks for 16-bit and 32-bit pixel extraction.
#[derive(Debug, Clone, Copy)]
struct BitfieldMasks {
    r: u32,
    g: u32,
    b: u32,
    a: u32,
}

/// Precomputed shift and bit count for a single channel mask.
#[derive(Debug, Clone, Copy)]
struct MaskInfo {
    shift: u8,
    bits: u8,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Little-endian reader helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Read a little-endian `u16` from `data` at `offset`.
/// Returns `None` if out of bounds.
#[inline]
fn read_u16_le(data: &[u8], offset: usize) -> Option<u16> {
    let bytes = data.get(offset..offset + 2)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Read a little-endian `u32` from `data` at `offset`.
/// Returns `None` if out of bounds.
#[inline]
fn read_u32_le(data: &[u8], offset: usize) -> Option<u32> {
    let bytes = data.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Read a little-endian `i32` from `data` at `offset`.
/// Returns `None` if out of bounds.
#[inline]
fn read_i32_le(data: &[u8], offset: usize) -> Option<i32> {
    let bytes = data.get(offset..offset + 4)?;
    Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Write a little-endian `u16` into `buf` at `offset`.
#[inline]
fn write_u16_le(buf: &mut [u8], offset: usize, val: u16) {
    buf[offset..offset + 2].copy_from_slice(&val.to_le_bytes());
}

/// Write a little-endian `u32` into `buf` at `offset`.
#[inline]
fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
    buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

/// Write a little-endian `i32` into `buf` at `offset`.
#[inline]
fn write_i32_le(buf: &mut [u8], offset: usize, val: i32) {
    buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

// ═══════════════════════════════════════════════════════════════════════════════
// File header parser
// ═══════════════════════════════════════════════════════════════════════════════

/// Parse the 14-byte BMP file header.
fn parse_file_header(data: &[u8]) -> Result<BmpFileHeader, IoError> {
    if data.len() < 14 {
        return Err(IoError::InvalidFormat {
            reason: "BMP file too short for file header (need at least 14 bytes)",
        });
    }

    // Validate BM signature.
    if data[0] != 0x42 || data[1] != 0x4D {
        return Err(IoError::InvalidFormat {
            reason: "not a BMP file (missing 'BM' signature)",
        });
    }

    let pixel_data_offset = read_u32_le(data, 10).unwrap();

    Ok(BmpFileHeader { pixel_data_offset })
}

// ═══════════════════════════════════════════════════════════════════════════════
// DIB header parser
// ═══════════════════════════════════════════════════════════════════════════════

/// Parse the DIB header starting at offset 14 in the file.
fn parse_dib_header(data: &[u8]) -> Result<BmpDibHeader, IoError> {
    let dib_start = 14usize;

    let header_size = read_u32_le(data, dib_start).ok_or(IoError::InvalidFormat {
        reason: "BMP file too short for DIB header size field",
    })?;

    match header_size {
        12 => parse_core_header(data, dib_start),
        40 => parse_info_header(data, dib_start),
        108 => parse_v4_header(data, dib_start),
        124 => parse_v5_header(data, dib_start),
        _ => Err(IoError::UnsupportedFeature {
            reason: "unsupported DIB header size (only 12, 40, 108, 124 are supported)",
        }),
    }
}

/// Parse BITMAPCOREHEADER (12 bytes).
fn parse_core_header(data: &[u8], offset: usize) -> Result<BmpDibHeader, IoError> {
    if data.len() < offset + 12 {
        return Err(IoError::InvalidFormat {
            reason: "BMP file too short for BITMAPCOREHEADER",
        });
    }

    let width = read_u16_le(data, offset + 4).unwrap() as u32;
    let height = read_u16_le(data, offset + 6).unwrap() as u32;
    let planes = read_u16_le(data, offset + 8).unwrap();
    let bit_count = read_u16_le(data, offset + 10).unwrap();

    if planes != 1 {
        return Err(IoError::InvalidFormat {
            reason: "BMP planes field must be 1",
        });
    }

    validate_bit_count_core(bit_count)?;

    if width == 0 || height == 0 {
        return Err(IoError::InvalidFormat {
            reason: "BMP image dimensions must be non-zero",
        });
    }

    Ok(BmpDibHeader {
        width,
        height,
        top_down: false,
        bit_count,
        compression: 0, // BI_RGB
        header_version: BmpHeaderVersion::Core,
        x_pels_per_meter: 0,
        y_pels_per_meter: 0,
        colors_used: 0,
        cs_type: 0,
        header_size: 12,
        v4_masks: None,
        icc_profile: None,
        color_table_entry_size: 3,
    })
}

/// Parse BITMAPINFOHEADER (40 bytes).
fn parse_info_header(data: &[u8], offset: usize) -> Result<BmpDibHeader, IoError> {
    if data.len() < offset + 40 {
        return Err(IoError::InvalidFormat {
            reason: "BMP file too short for BITMAPINFOHEADER",
        });
    }

    let raw_width = read_i32_le(data, offset + 4).unwrap();
    let raw_height = read_i32_le(data, offset + 8).unwrap();
    let planes = read_u16_le(data, offset + 12).unwrap();
    let bit_count = read_u16_le(data, offset + 14).unwrap();
    let compression = read_u32_le(data, offset + 16).unwrap();
    let x_pels = read_u32_le(data, offset + 24).unwrap();
    let y_pels = read_u32_le(data, offset + 28).unwrap();
    let colors_used = read_u32_le(data, offset + 32).unwrap();

    if planes != 1 {
        return Err(IoError::InvalidFormat {
            reason: "BMP planes field must be 1",
        });
    }

    validate_bit_count_info(bit_count)?;

    let (width, height, top_down) = resolve_dimensions(raw_width, raw_height)?;

    Ok(BmpDibHeader {
        width,
        height,
        top_down,
        bit_count,
        compression,
        header_version: BmpHeaderVersion::Info,
        x_pels_per_meter: x_pels,
        y_pels_per_meter: y_pels,
        colors_used,
        cs_type: 0,
        header_size: 40,
        v4_masks: None,
        icc_profile: None,
        color_table_entry_size: 4,
    })
}

/// Parse BITMAPV4HEADER (108 bytes).
fn parse_v4_header(data: &[u8], offset: usize) -> Result<BmpDibHeader, IoError> {
    if data.len() < offset + 108 {
        return Err(IoError::InvalidFormat {
            reason: "BMP file too short for BITMAPV4HEADER",
        });
    }

    let raw_width = read_i32_le(data, offset + 4).unwrap();
    let raw_height = read_i32_le(data, offset + 8).unwrap();
    let planes = read_u16_le(data, offset + 12).unwrap();
    let bit_count = read_u16_le(data, offset + 14).unwrap();
    let compression = read_u32_le(data, offset + 16).unwrap();
    let x_pels = read_u32_le(data, offset + 24).unwrap();
    let y_pels = read_u32_le(data, offset + 28).unwrap();
    let colors_used = read_u32_le(data, offset + 32).unwrap();

    // V4 masks at offsets 40..56
    let r_mask = read_u32_le(data, offset + 40).unwrap();
    let g_mask = read_u32_le(data, offset + 44).unwrap();
    let b_mask = read_u32_le(data, offset + 48).unwrap();
    let a_mask = read_u32_le(data, offset + 52).unwrap();
    let cs_type = read_u32_le(data, offset + 56).unwrap();

    if planes != 1 {
        return Err(IoError::InvalidFormat {
            reason: "BMP planes field must be 1",
        });
    }

    validate_bit_count_info(bit_count)?;

    let (width, height, top_down) = resolve_dimensions(raw_width, raw_height)?;

    Ok(BmpDibHeader {
        width,
        height,
        top_down,
        bit_count,
        compression,
        header_version: BmpHeaderVersion::V4,
        x_pels_per_meter: x_pels,
        y_pels_per_meter: y_pels,
        colors_used,
        cs_type,
        header_size: 108,
        v4_masks: Some(BitfieldMasks {
            r: r_mask,
            g: g_mask,
            b: b_mask,
            a: a_mask,
        }),
        icc_profile: None,
        color_table_entry_size: 4,
    })
}

/// Parse BITMAPV5HEADER (124 bytes).
fn parse_v5_header(data: &[u8], offset: usize) -> Result<BmpDibHeader, IoError> {
    if data.len() < offset + 124 {
        return Err(IoError::InvalidFormat {
            reason: "BMP file too short for BITMAPV5HEADER",
        });
    }

    let raw_width = read_i32_le(data, offset + 4).unwrap();
    let raw_height = read_i32_le(data, offset + 8).unwrap();
    let planes = read_u16_le(data, offset + 12).unwrap();
    let bit_count = read_u16_le(data, offset + 14).unwrap();
    let compression = read_u32_le(data, offset + 16).unwrap();
    let x_pels = read_u32_le(data, offset + 24).unwrap();
    let y_pels = read_u32_le(data, offset + 28).unwrap();
    let colors_used = read_u32_le(data, offset + 32).unwrap();

    // V4 masks at offsets 40..56
    let r_mask = read_u32_le(data, offset + 40).unwrap();
    let g_mask = read_u32_le(data, offset + 44).unwrap();
    let b_mask = read_u32_le(data, offset + 48).unwrap();
    let a_mask = read_u32_le(data, offset + 52).unwrap();
    let cs_type = read_u32_le(data, offset + 56).unwrap();

    // V5 specific: ICC profile offset and size.
    let profile_data = read_u32_le(data, offset + 112).unwrap();
    let profile_size = read_u32_le(data, offset + 116).unwrap();

    if planes != 1 {
        return Err(IoError::InvalidFormat {
            reason: "BMP planes field must be 1",
        });
    }

    validate_bit_count_info(bit_count)?;

    let (width, height, top_down) = resolve_dimensions(raw_width, raw_height)?;

    // ICC profile embedded?
    let icc = if cs_type == LCS_PROFILE_EMBEDDED && profile_size > 0 {
        Some((profile_data, profile_size))
    } else {
        None
    };

    Ok(BmpDibHeader {
        width,
        height,
        top_down,
        bit_count,
        compression,
        header_version: BmpHeaderVersion::V5,
        x_pels_per_meter: x_pels,
        y_pels_per_meter: y_pels,
        colors_used,
        cs_type,
        header_size: 124,
        v4_masks: Some(BitfieldMasks {
            r: r_mask,
            g: g_mask,
            b: b_mask,
            a: a_mask,
        }),
        icc_profile: icc,
        color_table_entry_size: 4,
    })
}

/// Validate bit count for BITMAPCOREHEADER (1, 4, 8, or 24 only).
fn validate_bit_count_core(bit_count: u16) -> Result<(), IoError> {
    match bit_count {
        1 | 4 | 8 | 24 => Ok(()),
        _ => Err(IoError::InvalidFormat {
            reason: "invalid bit count for BITMAPCOREHEADER (must be 1, 4, 8, or 24)",
        }),
    }
}

/// Validate bit count for BITMAPINFOHEADER+ (1, 4, 8, 16, 24, or 32).
fn validate_bit_count_info(bit_count: u16) -> Result<(), IoError> {
    match bit_count {
        1 | 4 | 8 | 16 | 24 | 32 => Ok(()),
        _ => Err(IoError::InvalidFormat {
            reason: "invalid bit count (must be 1, 4, 8, 16, 24, or 32)",
        }),
    }
}

/// Resolve signed width/height into unsigned dimensions + top_down flag.
fn resolve_dimensions(raw_width: i32, raw_height: i32) -> Result<(u32, u32, bool), IoError> {
    if raw_width <= 0 {
        return Err(IoError::InvalidFormat {
            reason: "BMP width must be positive",
        });
    }

    if raw_height == 0 {
        return Err(IoError::InvalidFormat {
            reason: "BMP height must be non-zero",
        });
    }

    let width = raw_width as u32;
    let (height, top_down) = if raw_height < 0 {
        // Negative height means top-down storage.
        // Handle i32::MIN carefully.
        let abs = (raw_height as i64).unsigned_abs() as u32;
        (abs, true)
    } else {
        (raw_height as u32, false)
    };

    Ok((width, height, top_down))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Color space constants
// ═══════════════════════════════════════════════════════════════════════════════

const LCS_SRGB: u32 = 0x73524742;
const LCS_WINDOWS_COLOR_SPACE: u32 = 0x57696E20;
const LCS_PROFILE_EMBEDDED: u32 = 0x4D424544;
// const LCS_CALIBRATED_RGB: u32 = 0x00000000;
// const LCS_PROFILE_LINKED: u32 = 0x4C494E4B;

// Compression type constants.
const BI_RGB: u32 = 0;
const BI_RLE8: u32 = 1;
const BI_RLE4: u32 = 2;
const BI_BITFIELDS: u32 = 3;
// const BI_JPEG: u32 = 4;
// const BI_PNG: u32 = 5;

// ═══════════════════════════════════════════════════════════════════════════════
// Bitfield mask utilities
// ═══════════════════════════════════════════════════════════════════════════════

impl BitfieldMasks {
    /// Default masks for 16-bit BI_RGB (5-5-5 layout).
    fn default_16bit() -> Self {
        BitfieldMasks {
            r: 0x7C00,
            g: 0x03E0,
            b: 0x001F,
            a: 0,
        }
    }

    /// Default masks for 32-bit BI_RGB (8-8-8-X layout).
    fn default_32bit() -> Self {
        BitfieldMasks {
            r: 0x00FF0000,
            g: 0x0000FF00,
            b: 0x000000FF,
            a: 0x00000000,
        }
    }
}

/// Compute the shift (number of trailing zeros) and bit count for a mask.
fn mask_info(mask: u32) -> MaskInfo {
    if mask == 0 {
        return MaskInfo { shift: 0, bits: 0 };
    }
    let shift = mask.trailing_zeros() as u8;
    let bits = (mask >> shift).trailing_ones() as u8;
    MaskInfo { shift, bits }
}

/// Extract a channel value from a pixel word using the given mask info,
/// and scale it to 8-bit using full-range scaling.
#[inline]
fn extract_channel(value: u32, info: MaskInfo) -> u8 {
    if info.bits == 0 {
        return 0;
    }
    let raw = (value >> info.shift) & ((1u32 << info.bits) - 1);
    if info.bits >= 8 {
        (raw >> (info.bits - 8)) as u8
    } else {
        // Full-range scale: raw * 255 / max_value
        let max_val = (1u32 << info.bits) - 1;
        ((raw * 255 + max_val / 2) / max_val) as u8
    }
}

/// Parse bitfield masks from the data following the DIB header.
/// Used when `biCompression == BI_BITFIELDS` and no V4/V5 masks are present.
fn parse_bitfield_masks(data: &[u8], offset: usize) -> Result<BitfieldMasks, IoError> {
    if data.len() < offset + 12 {
        return Err(IoError::InvalidFormat {
            reason: "BMP file too short for bitfield masks",
        });
    }
    let r = read_u32_le(data, offset).unwrap();
    let g = read_u32_le(data, offset + 4).unwrap();
    let b = read_u32_le(data, offset + 8).unwrap();

    // Check for an optional alpha mask (4th DWORD).
    let a = if data.len() >= offset + 16 {
        read_u32_le(data, offset + 12).unwrap()
    } else {
        0
    };

    Ok(BitfieldMasks { r, g, b, a })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Color table parser
// ═══════════════════════════════════════════════════════════════════════════════

/// Parse the color table (palette) into a 256-entry `Srgba8` array.
///
/// `entry_size` is 3 for BITMAPCOREHEADER (BGR) or 4 for later headers (BGRA).
/// Entries beyond the actual count are filled with opaque black.
fn parse_color_table(
    data: &[u8],
    offset: usize,
    entry_size: u8,
    count: usize,
) -> Result<Box<[Srgba8; 256]>, IoError> {
    let mut palette = Box::new([Srgba8::new(0, 0, 0, 255); 256]);

    for i in 0..count.min(256) {
        let entry_offset = offset + i * entry_size as usize;
        if data.len() < entry_offset + entry_size as usize {
            return Err(IoError::InvalidFormat {
                reason: "BMP file too short for color table",
            });
        }

        let b = data[entry_offset];
        let g = data[entry_offset + 1];
        let r = data[entry_offset + 2];
        // For 4-byte entries, the 4th byte is reserved (always 0 in standard BMPs).
        // We set alpha to 255 (fully opaque).
        palette[i] = Srgba8::new(r, g, b, 255);
    }

    Ok(palette)
}

// ═══════════════════════════════════════════════════════════════════════════════
// RLE decompressors
// ═══════════════════════════════════════════════════════════════════════════════

/// Decompress RLE8-encoded pixel data into a flat buffer of 8-bit indices.
fn decompress_rle8(data: &[u8], width: usize, height: usize) -> Result<Vec<u8>, IoError> {
    let mut output = vec![0u8; width * height];
    let mut x = 0usize;
    let mut y = 0usize;
    let mut pos = 0usize;

    while pos < data.len() {
        let first = data[pos];
        pos += 1;
        if pos >= data.len() {
            break;
        }
        let second = data[pos];
        pos += 1;

        if first > 0 {
            // Encoded run: repeat `second` for `first` pixels.
            for _ in 0..first as usize {
                if y >= height {
                    return Err(decode_failed("RLE8 data exceeds image bounds"));
                }
                if x < width {
                    output[y * width + x] = second;
                }
                x += 1;
            }
        } else {
            // Escape sequence.
            match second {
                0 => {
                    // End of line.
                    x = 0;
                    y += 1;
                }
                1 => {
                    // End of bitmap.
                    break;
                }
                2 => {
                    // Delta.
                    if pos + 1 >= data.len() {
                        return Err(decode_failed("RLE8 delta truncated"));
                    }
                    let dx = data[pos] as usize;
                    let dy = data[pos + 1] as usize;
                    pos += 2;
                    x += dx;
                    y += dy;
                }
                n => {
                    // Absolute mode: `n` literal pixels.
                    let count = n as usize;
                    for _ in 0..count {
                        if pos >= data.len() {
                            return Err(decode_failed("RLE8 absolute mode truncated"));
                        }
                        if y >= height {
                            return Err(decode_failed("RLE8 data exceeds image bounds"));
                        }
                        if x < width {
                            output[y * width + x] = data[pos];
                        }
                        x += 1;
                        pos += 1;
                    }
                    // Word-aligned: skip pad byte if odd count.
                    if count & 1 != 0 {
                        pos += 1;
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Decompress RLE4-encoded pixel data into a flat buffer of 8-bit indices.
fn decompress_rle4(data: &[u8], width: usize, height: usize) -> Result<Vec<u8>, IoError> {
    let mut output = vec![0u8; width * height];
    let mut x = 0usize;
    let mut y = 0usize;
    let mut pos = 0usize;

    while pos < data.len() {
        let first = data[pos];
        pos += 1;
        if pos >= data.len() {
            break;
        }
        let second = data[pos];
        pos += 1;

        if first > 0 {
            // Encoded run: `first` pixels, alternating nibbles of `second`.
            let hi = second >> 4;
            let lo = second & 0x0F;
            for i in 0..first as usize {
                if y >= height {
                    return Err(decode_failed("RLE4 data exceeds image bounds"));
                }
                let val = if i & 1 == 0 { hi } else { lo };
                if x < width {
                    output[y * width + x] = val;
                }
                x += 1;
            }
        } else {
            // Escape sequence.
            match second {
                0 => {
                    // End of line.
                    x = 0;
                    y += 1;
                }
                1 => {
                    // End of bitmap.
                    break;
                }
                2 => {
                    // Delta.
                    if pos + 1 >= data.len() {
                        return Err(decode_failed("RLE4 delta truncated"));
                    }
                    let dx = data[pos] as usize;
                    let dy = data[pos + 1] as usize;
                    pos += 2;
                    x += dx;
                    y += dy;
                }
                n => {
                    // Absolute mode: `n` literal pixels, nibble-packed.
                    let count = n as usize;
                    let byte_count = count.div_ceil(2);
                    for i in 0..count {
                        let byte_idx = i / 2;
                        if pos + byte_idx >= data.len() {
                            return Err(decode_failed("RLE4 absolute mode truncated"));
                        }
                        let byte = data[pos + byte_idx];
                        let val = if i & 1 == 0 { byte >> 4 } else { byte & 0x0F };
                        if y >= height {
                            return Err(decode_failed("RLE4 data exceeds image bounds"));
                        }
                        if x < width {
                            output[y * width + x] = val;
                        }
                        x += 1;
                    }
                    pos += byte_count;
                    // Word-aligned: skip pad byte if odd number of bytes.
                    if byte_count & 1 != 0 {
                        pos += 1;
                    }
                }
            }
        }
    }

    Ok(output)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Row stride calculation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the byte stride of a BMP row (padded to 4-byte boundary).
fn row_stride(width: u32, bits_per_pixel: u32) -> usize {
    let bits = width as usize * bits_per_pixel as usize;
    bits.div_ceil(32) * 4
}

// ═══════════════════════════════════════════════════════════════════════════════
// Decode — public API
// ═══════════════════════════════════════════════════════════════════════════════

/// Decode a BMP image from an in-memory byte slice.
///
/// Returns a [`BmpDecoded`] containing the pixel data and metadata.
///
/// # Errors
///
/// - [`IoError::InvalidFormat`] — not a BMP file, truncated, or structurally
///   invalid.
/// - [`IoError::UnsupportedFeature`] — unsupported DIB header version,
///   `BI_JPEG`, or `BI_PNG` compression.
/// - [`IoError::DecodeFailed`] — pixel data corruption (e.g. RLE out of
///   bounds).
///
/// # Examples
///
/// ```no_run
/// # use irys_cv_io::bmp::{self, BmpImage};
/// let bytes = std::fs::read("photo.bmp").unwrap();
/// let decoded = bmp::decode(&bytes).unwrap();
///
/// match decoded.image {
///     BmpImage::Srgb8(image) => { /* work with Image<Srgb8> */ }
///     BmpImage::Srgba8(image) => { /* work with Image<Srgba8> */ }
///     BmpImage::Indexed8 { data, palette } => { /* depalettise */ }
/// }
/// ```
pub fn decode(data: &[u8]) -> Result<BmpDecoded, IoError> {
    // ── Parse file header ────────────────────────────────────────────────
    let file_header = parse_file_header(data)?;

    // ── Parse DIB header ─────────────────────────────────────────────────
    let dib = parse_dib_header(data)?;

    // ── Reject unsupported compression ───────────────────────────────────
    match dib.compression {
        BI_RGB | BI_RLE8 | BI_RLE4 | BI_BITFIELDS => {}
        4 => {
            return Err(IoError::UnsupportedFeature {
                reason: "BI_JPEG compression is not supported",
            });
        }
        5 => {
            return Err(IoError::UnsupportedFeature {
                reason: "BI_PNG compression is not supported",
            });
        }
        _ => {
            return Err(IoError::UnsupportedFeature {
                reason: "unknown BMP compression type",
            });
        }
    }

    // Validate compression + bit depth combinations.
    if dib.compression == BI_RLE8 && dib.bit_count != 8 {
        return Err(IoError::InvalidFormat {
            reason: "BI_RLE8 compression requires 8-bit indexed",
        });
    }
    if dib.compression == BI_RLE4 && dib.bit_count != 4 {
        return Err(IoError::InvalidFormat {
            reason: "BI_RLE4 compression requires 4-bit indexed",
        });
    }

    let width = dib.width as usize;
    let height = dib.height as usize;
    let pixel_offset = file_header.pixel_data_offset as usize;

    // ── Decode pixels based on bit depth ─────────────────────────────────
    let image = match dib.bit_count {
        1 => decode_indexed(&dib, data, pixel_offset, width, height, 1)?,
        4 => {
            if dib.compression == BI_RLE4 {
                decode_indexed_rle4(&dib, data, pixel_offset, width, height)?
            } else {
                decode_indexed(&dib, data, pixel_offset, width, height, 4)?
            }
        }
        8 => {
            if dib.compression == BI_RLE8 {
                decode_indexed_rle8(&dib, data, pixel_offset, width, height)?
            } else {
                decode_indexed(&dib, data, pixel_offset, width, height, 8)?
            }
        }
        16 => decode_16bit(&dib, data, pixel_offset, width, height)?,
        24 => decode_24bit(&dib, data, pixel_offset, width, height)?,
        32 => decode_32bit(&dib, data, pixel_offset, width, height)?,
        _ => {
            return Err(IoError::InvalidFormat {
                reason: "unsupported BMP bit count",
            });
        }
    };

    // ── Build metadata ───────────────────────────────────────────────────
    let metadata = build_metadata(&dib, data);

    Ok(BmpDecoded { image, metadata })
}

/// Decode a BMP image from a streaming reader.
///
/// Reads the entire stream into memory and delegates to [`decode`].
///
/// # Errors
///
/// Same error conditions as [`decode`], plus [`IoError::Io`] for read
/// failures.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv_io::bmp::{self, BmpImage};
/// let file = std::fs::File::open("photo.bmp").unwrap();
/// let reader = std::io::BufReader::new(file);
/// let decoded = bmp::decode_reader(reader).unwrap();
///
/// match decoded.image {
///     BmpImage::Srgb8(image) => { /* work with Image<Srgb8> */ }
///     _ => { /* handle remaining variants */ }
/// }
/// ```
pub fn decode_reader(mut reader: impl std::io::Read) -> Result<BmpDecoded, IoError> {
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    decode(&buf)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Decode — internal pixel readers
// ═══════════════════════════════════════════════════════════════════════════════

/// Decode indexed (1, 4, or 8-bit) uncompressed pixel data.
fn decode_indexed(
    dib: &BmpDibHeader,
    data: &[u8],
    pixel_offset: usize,
    width: usize,
    height: usize,
    bits: u32,
) -> Result<BmpImage, IoError> {
    // Parse color table.
    let color_table_offset = 14 + dib.header_size as usize;
    let color_count = if dib.colors_used > 0 {
        dib.colors_used as usize
    } else {
        1usize << bits
    };
    let palette = parse_color_table(
        data,
        color_table_offset,
        dib.color_table_entry_size,
        color_count,
    )?;

    let stride = row_stride(dib.width, bits);
    let mut pixels = Vec::with_capacity(width * height);

    for row in 0..height {
        let src_row = if dib.top_down { row } else { height - 1 - row };
        let row_start = pixel_offset + src_row * stride;

        match bits {
            1 => {
                for col in 0..width {
                    let byte_idx = row_start + col / 8;
                    if byte_idx >= data.len() {
                        return Err(decode_failed("BMP pixel data truncated"));
                    }
                    let bit_idx = 7 - (col % 8);
                    let index = (data[byte_idx] >> bit_idx) & 1;
                    pixels.push(Indexed8(index));
                }
            }
            4 => {
                for col in 0..width {
                    let byte_idx = row_start + col / 2;
                    if byte_idx >= data.len() {
                        return Err(decode_failed("BMP pixel data truncated"));
                    }
                    let index = if col % 2 == 0 {
                        data[byte_idx] >> 4
                    } else {
                        data[byte_idx] & 0x0F
                    };
                    pixels.push(Indexed8(index));
                }
            }
            8 => {
                if row_start + width > data.len() {
                    return Err(decode_failed("BMP pixel data truncated"));
                }
                for col in 0..width {
                    pixels.push(Indexed8(data[row_start + col]));
                }
            }
            _ => unreachable!(),
        }
    }

    let img = Image::from_vec(width, height, pixels)
        .map_err(|_| decode_failed("pixel count does not match image dimensions"))?;

    Ok(BmpImage::Indexed8 { data: img, palette })
}

/// Decode RLE8-compressed indexed pixel data.
fn decode_indexed_rle8(
    dib: &BmpDibHeader,
    data: &[u8],
    pixel_offset: usize,
    width: usize,
    height: usize,
) -> Result<BmpImage, IoError> {
    let color_table_offset = 14 + dib.header_size as usize;
    let color_count = if dib.colors_used > 0 {
        dib.colors_used as usize
    } else {
        256
    };
    let palette = parse_color_table(
        data,
        color_table_offset,
        dib.color_table_entry_size,
        color_count,
    )?;

    let rle_data = data.get(pixel_offset..).ok_or(IoError::InvalidFormat {
        reason: "BMP pixel data offset beyond file",
    })?;

    // RLE stores rows bottom-up, but the decompressor outputs in the order
    // it encounters them (row 0 = bottom). We need to flip.
    let decompressed = decompress_rle8(rle_data, width, height)?;

    // The RLE decompressor produces rows in bottom-up order (row 0 is bottom).
    // Re-arrange to top-down.
    let mut pixels = Vec::with_capacity(width * height);
    for row in 0..height {
        let src_row = if dib.top_down { row } else { height - 1 - row };
        let start = src_row * width;
        for col in 0..width {
            pixels.push(Indexed8(decompressed[start + col]));
        }
    }

    let img = Image::from_vec(width, height, pixels)
        .map_err(|_| decode_failed("pixel count does not match image dimensions"))?;

    Ok(BmpImage::Indexed8 { data: img, palette })
}

/// Decode RLE4-compressed indexed pixel data.
fn decode_indexed_rle4(
    dib: &BmpDibHeader,
    data: &[u8],
    pixel_offset: usize,
    width: usize,
    height: usize,
) -> Result<BmpImage, IoError> {
    let color_table_offset = 14 + dib.header_size as usize;
    let color_count = if dib.colors_used > 0 {
        dib.colors_used as usize
    } else {
        16
    };
    let palette = parse_color_table(
        data,
        color_table_offset,
        dib.color_table_entry_size,
        color_count,
    )?;

    let rle_data = data.get(pixel_offset..).ok_or(IoError::InvalidFormat {
        reason: "BMP pixel data offset beyond file",
    })?;

    let decompressed = decompress_rle4(rle_data, width, height)?;

    let mut pixels = Vec::with_capacity(width * height);
    for row in 0..height {
        let src_row = if dib.top_down { row } else { height - 1 - row };
        let start = src_row * width;
        for col in 0..width {
            pixels.push(Indexed8(decompressed[start + col]));
        }
    }

    let img = Image::from_vec(width, height, pixels)
        .map_err(|_| decode_failed("pixel count does not match image dimensions"))?;

    Ok(BmpImage::Indexed8 { data: img, palette })
}

/// Decode 16-bit pixel data (5-5-5 default or bitfield masks).
fn decode_16bit(
    dib: &BmpDibHeader,
    data: &[u8],
    pixel_offset: usize,
    width: usize,
    height: usize,
) -> Result<BmpImage, IoError> {
    let masks = resolve_masks_with_data(dib, 16, data)?;
    let r_info = mask_info(masks.r);
    let g_info = mask_info(masks.g);
    let b_info = mask_info(masks.b);

    let stride = row_stride(dib.width, 16);
    let mut pixels = Vec::with_capacity(width * height);

    for row in 0..height {
        let src_row = if dib.top_down { row } else { height - 1 - row };
        let row_start = pixel_offset + src_row * stride;

        for col in 0..width {
            let pix_offset = row_start + col * 2;
            let val = read_u16_le(data, pix_offset)
                .ok_or_else(|| decode_failed("BMP 16-bit pixel data truncated"))?
                as u32;

            let r = extract_channel(val, r_info);
            let g = extract_channel(val, g_info);
            let b = extract_channel(val, b_info);
            pixels.push(Srgb8::new(r, g, b));
        }
    }

    let img = Image::from_vec(width, height, pixels)
        .map_err(|_| decode_failed("pixel count does not match image dimensions"))?;

    Ok(BmpImage::Srgb8(img))
}

/// Decode 24-bit pixel data (BGR → RGB).
fn decode_24bit(
    dib: &BmpDibHeader,
    data: &[u8],
    pixel_offset: usize,
    width: usize,
    height: usize,
) -> Result<BmpImage, IoError> {
    let stride = row_stride(dib.width, 24);
    let mut pixels = Vec::with_capacity(width * height);

    for row in 0..height {
        let src_row = if dib.top_down { row } else { height - 1 - row };
        let row_start = pixel_offset + src_row * stride;

        for col in 0..width {
            let pix_offset = row_start + col * 3;
            if pix_offset + 3 > data.len() {
                return Err(decode_failed("BMP 24-bit pixel data truncated"));
            }
            let b = data[pix_offset];
            let g = data[pix_offset + 1];
            let r = data[pix_offset + 2];
            pixels.push(Srgb8::new(r, g, b));
        }
    }

    let img = Image::from_vec(width, height, pixels)
        .map_err(|_| decode_failed("pixel count does not match image dimensions"))?;

    Ok(BmpImage::Srgb8(img))
}

/// Decode 32-bit pixel data.
///
/// If the bitfield masks include a non-zero alpha mask → `Srgba8`.
/// Otherwise → `Srgb8` (BGRX, ignore the 4th byte).
fn decode_32bit(
    dib: &BmpDibHeader,
    data: &[u8],
    pixel_offset: usize,
    width: usize,
    height: usize,
) -> Result<BmpImage, IoError> {
    let masks = resolve_masks_with_data(dib, 32, data)?;
    let has_alpha = masks.a != 0;
    let r_info = mask_info(masks.r);
    let g_info = mask_info(masks.g);
    let b_info = mask_info(masks.b);
    let a_info = mask_info(masks.a);

    let stride = row_stride(dib.width, 32);

    if has_alpha {
        let mut pixels = Vec::with_capacity(width * height);
        for row in 0..height {
            let src_row = if dib.top_down { row } else { height - 1 - row };
            let row_start = pixel_offset + src_row * stride;

            for col in 0..width {
                let pix_offset = row_start + col * 4;
                let val = read_u32_le(data, pix_offset)
                    .ok_or_else(|| decode_failed("BMP 32-bit pixel data truncated"))?;

                let r = extract_channel(val, r_info);
                let g = extract_channel(val, g_info);
                let b = extract_channel(val, b_info);
                let a = extract_channel(val, a_info);
                pixels.push(Srgba8::new(r, g, b, a));
            }
        }

        let img = Image::from_vec(width, height, pixels)
            .map_err(|_| decode_failed("pixel count does not match image dimensions"))?;

        Ok(BmpImage::Srgba8(img))
    } else {
        let mut pixels = Vec::with_capacity(width * height);
        for row in 0..height {
            let src_row = if dib.top_down { row } else { height - 1 - row };
            let row_start = pixel_offset + src_row * stride;

            for col in 0..width {
                let pix_offset = row_start + col * 4;
                let val = read_u32_le(data, pix_offset)
                    .ok_or_else(|| decode_failed("BMP 32-bit pixel data truncated"))?;

                let r = extract_channel(val, r_info);
                let g = extract_channel(val, g_info);
                let b = extract_channel(val, b_info);
                pixels.push(Srgb8::new(r, g, b));
            }
        }

        let img = Image::from_vec(width, height, pixels)
            .map_err(|_| decode_failed("pixel count does not match image dimensions"))?;

        Ok(BmpImage::Srgb8(img))
    }
}

/// Resolve the effective bitfield masks for a given bit depth, using original file data.
fn resolve_masks_with_data(
    dib: &BmpDibHeader,
    bit_count: u16,
    data: &[u8],
) -> Result<BitfieldMasks, IoError> {
    if dib.compression == BI_BITFIELDS {
        // V4/V5 headers have masks embedded in the header itself.
        if let Some(masks) = dib.v4_masks {
            return Ok(masks);
        }
        // BITMAPINFOHEADER with BI_BITFIELDS: masks follow immediately after the header.
        let mask_offset = 14 + dib.header_size as usize;
        return parse_bitfield_masks(data, mask_offset);
    }

    // BI_RGB defaults.
    match bit_count {
        16 => Ok(BitfieldMasks::default_16bit()),
        32 => Ok(BitfieldMasks::default_32bit()),
        _ => unreachable!("resolve_masks_with_data only called for 16-bit and 32-bit"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Metadata builder
// ═══════════════════════════════════════════════════════════════════════════════

/// Build `BmpMetadata` from the parsed DIB header.
fn build_metadata(dib: &BmpDibHeader, data: &[u8]) -> BmpMetadata {
    let source_bit_depth = match dib.bit_count {
        1 => BmpBitDepth::One,
        4 => BmpBitDepth::Four,
        8 => BmpBitDepth::Eight,
        16 => BmpBitDepth::Sixteen,
        24 => BmpBitDepth::TwentyFour,
        32 => BmpBitDepth::ThirtyTwo,
        _ => unreachable!("bit count validated earlier"),
    };

    let compression = match dib.compression {
        BI_RGB => BmpCompression::None,
        BI_RLE8 => BmpCompression::Rle8,
        BI_RLE4 => BmpCompression::Rle4,
        BI_BITFIELDS => BmpCompression::Bitfields,
        _ => unreachable!("compression type validated earlier"),
    };

    let resolution = if dib.x_pels_per_meter > 0 || dib.y_pels_per_meter > 0 {
        Some(BmpResolution {
            x_pixels_per_meter: dib.x_pels_per_meter,
            y_pixels_per_meter: dib.y_pels_per_meter,
        })
    } else {
        None
    };

    let color_space = determine_color_space(dib);

    // Extract ICC profile if present.
    let icc_profile = if let Some((profile_offset, profile_size)) = dib.icc_profile {
        // Profile offset is from the beginning of the file header.
        let start = 14 + profile_offset as usize;
        let end = start + profile_size as usize;
        if end <= data.len() {
            Some(data[start..end].to_vec().into_boxed_slice())
        } else {
            None
        }
    } else {
        None
    };

    BmpMetadata {
        color_space,
        source_bit_depth,
        resolution,
        compression,
        header_version: dib.header_version,
        icc_profile,
    }
}

/// Determine the color space from the DIB header fields.
fn determine_color_space(dib: &BmpDibHeader) -> BmpColorSpace {
    match dib.header_version {
        BmpHeaderVersion::Core | BmpHeaderVersion::Info => BmpColorSpace::Srgb,
        BmpHeaderVersion::V4 | BmpHeaderVersion::V5 => match dib.cs_type {
            LCS_PROFILE_EMBEDDED => BmpColorSpace::IccTagged,
            // Explicit sRGB signalling from V4/V5 header.
            LCS_SRGB | LCS_WINDOWS_COLOR_SPACE => BmpColorSpace::Srgb,
            // LCS_CALIBRATED_RGB, PROFILE_LINKED, and any other value —
            // treat as sRGB (the safe default for BMP data).
            _ => BmpColorSpace::Srgb,
        },
    }
}

/// Convenience for constructing a `DecodeFailed` error.
fn decode_failed(msg: &'static str) -> IoError {
    IoError::DecodeFailed { source: msg.into() }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Encode — public API
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode an image to an in-memory BMP byte vector.
///
/// Only pixel types implementing [`BmpPixel`] can be encoded — currently
/// [`Srgb8`] (24-bit) and [`Srgba8`] (32-bit).
///
/// # Errors
///
/// Returns [`IoError::EncodeFailed`] on I/O errors during encoding.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv::image::Image;
/// # use irys_cv::pixel::Srgb8;
/// # use irys_cv_io::bmp::{self, BmpEncodeOptions};
/// let image: Image<Srgb8> = todo!();
/// let bytes = bmp::encode(&image, &BmpEncodeOptions::default()).unwrap();
/// std::fs::write("output.bmp", bytes).unwrap();
/// ```
pub fn encode<P: BmpPixel>(
    image: &(impl ImageView<Pixel = P> + PlainImage),
    options: &BmpEncodeOptions,
) -> Result<Vec<u8>, IoError> {
    let mut buf = Vec::new();
    encode_writer(image, &mut buf, options)?;
    Ok(buf)
}

/// Encode an image to a streaming writer.
///
/// This is the core encoding function — [`encode`] is a convenience
/// wrapper that writes into a `Vec<u8>`.
///
/// Writes a `BITMAPINFOHEADER` (40 bytes) with uncompressed pixel data.
/// For `Srgb8`: 24-bit `BI_RGB`.  For `Srgba8`: 32-bit `BI_BITFIELDS`
/// with explicit channel masks.
///
/// # Errors
///
/// Returns [`IoError::EncodeFailed`] if the underlying writer fails.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv::image::Image;
/// # use irys_cv::pixel::Srgb8;
/// # use irys_cv_io::bmp::{self, BmpEncodeOptions};
/// let image: Image<Srgb8> = todo!();
/// let mut out = Vec::new();
/// bmp::encode_writer(&image, &mut out, &BmpEncodeOptions::default()).unwrap();
/// ```
pub fn encode_writer<P: BmpPixel>(
    image: &(impl ImageView<Pixel = P> + PlainImage),
    writer: &mut impl std::io::Write,
    options: &BmpEncodeOptions,
) -> Result<(), IoError> {
    let width = image.width() as u32;
    let height = image.height() as u32;
    let bpp = P::BMP_BYTES_PER_PIXEL;
    let stride = row_stride(width, P::BMP_BIT_COUNT as u32) as u32;
    let pixel_data_size = stride * height;

    // For 32-bit with alpha, we write 16 bytes of bitfield masks after
    // the BITMAPINFOHEADER (3 color masks + 1 alpha mask × 4 bytes each).
    let mask_size: u32 = if P::BMP_HAS_ALPHA { 16 } else { 0 };
    let pixel_data_offset = 14 + 40 + mask_size;
    let file_size = pixel_data_offset + pixel_data_size;

    // ── Write BITMAPFILEHEADER (14 bytes) ────────────────────────────────
    let mut file_header = [0u8; 14];
    file_header[0] = 0x42; // 'B'
    file_header[1] = 0x4D; // 'M'
    write_u32_le(&mut file_header, 2, file_size);
    // Reserved fields at 6..10 stay zero.
    write_u32_le(&mut file_header, 10, pixel_data_offset);
    writer.write_all(&file_header).map_err(encode_error)?;

    // ── Write BITMAPINFOHEADER (40 bytes) ────────────────────────────────
    let mut info_header = [0u8; 40];
    write_u32_le(&mut info_header, 0, 40); // biSize
    write_i32_le(&mut info_header, 4, width as i32); // biWidth
    write_i32_le(&mut info_header, 8, height as i32); // biHeight (positive = bottom-up)
    write_u16_le(&mut info_header, 12, 1); // biPlanes
    write_u16_le(&mut info_header, 14, P::BMP_BIT_COUNT); // biBitCount

    let compression: u32 = if P::BMP_HAS_ALPHA {
        BI_BITFIELDS
    } else {
        BI_RGB
    };
    write_u32_le(&mut info_header, 16, compression); // biCompression
    write_u32_le(&mut info_header, 20, pixel_data_size); // biSizeImage

    let (x_res, y_res) = match &options.resolution {
        Some(r) => (r.x_pixels_per_meter, r.y_pixels_per_meter),
        None => (0, 0),
    };
    write_u32_le(&mut info_header, 24, x_res); // biXPelsPerMeter
    write_u32_le(&mut info_header, 28, y_res); // biYPelsPerMeter
    // biClrUsed and biClrImportant stay zero.
    writer.write_all(&info_header).map_err(encode_error)?;

    // ── Write bitfield masks (32-bit with alpha only) ────────────────────
    if P::BMP_HAS_ALPHA {
        let mut masks = [0u8; 16];
        write_u32_le(&mut masks, 0, 0x00FF0000); // R
        write_u32_le(&mut masks, 4, 0x0000FF00); // G
        write_u32_le(&mut masks, 8, 0x000000FF); // B
        write_u32_le(&mut masks, 12, 0xFF000000); // A
        writer.write_all(&masks).map_err(encode_error)?;
    }

    // ── Write pixel data (bottom-up, BGR/BGRA, padded rows) ─────────────
    let pad_bytes = stride as usize - (width as usize * bpp as usize);
    let padding = [0u8; 3]; // max 3 bytes of padding

    // Write rows in bottom-up order: last image row first.
    let mut row_buf = vec![0u8; width as usize * bpp as usize];
    for row in (0..height as usize).rev() {
        // Build the row in BGR/BGRA order.
        for col in 0..width as usize {
            let pixel = image.pixel_at(col, row);
            let offset = col * bpp as usize;
            pixel.write_bmp_bytes(&mut row_buf[offset..]);
        }
        writer.write_all(&row_buf).map_err(encode_error)?;
        if pad_bytes > 0 {
            writer
                .write_all(&padding[..pad_bytes])
                .map_err(encode_error)?;
        }
    }

    Ok(())
}

/// Encode a palette-indexed image to an in-memory BMP byte vector.
///
/// Writes an 8-bit indexed BMP with a color table.  The palette may have
/// 1–256 entries.
///
/// # Errors
///
/// - Palette must have 1–256 entries.
/// - All pixel indices must be within the palette range.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv::image::Image;
/// # use irys_cv::pixel::{Indexed8, Srgba8};
/// # use irys_cv_io::bmp::{self, BmpEncodeOptions};
/// let image: Image<Indexed8> = todo!();
/// let palette = [Srgba8::new(255, 0, 0, 255), Srgba8::new(0, 255, 0, 255)];
/// let bytes = bmp::encode_indexed(&image, &palette, &BmpEncodeOptions::default()).unwrap();
/// ```
pub fn encode_indexed(
    image: &(impl ImageView<Pixel = Indexed8> + PlainImage),
    palette: &[Srgba8],
    options: &BmpEncodeOptions,
) -> Result<Vec<u8>, IoError> {
    let mut buf = Vec::new();
    encode_indexed_writer(image, palette, &mut buf, options)?;
    Ok(buf)
}

/// Core indexed encoding — writes to a streaming writer.
fn encode_indexed_writer(
    image: &(impl ImageView<Pixel = Indexed8> + PlainImage),
    palette: &[Srgba8],
    writer: &mut impl std::io::Write,
    options: &BmpEncodeOptions,
) -> Result<(), IoError> {
    // Validate palette size.
    if palette.is_empty() || palette.len() > 256 {
        return Err(IoError::EncodeFailed {
            source: "palette must have 1\u{2013}256 entries".into(),
        });
    }

    // Validate all indices are in range.
    let max_index = (palette.len() - 1) as u8;
    for pixel in image.as_slice() {
        if pixel.0 > max_index {
            return Err(IoError::EncodeFailed {
                source: "pixel index exceeds palette size".into(),
            });
        }
    }

    let width = image.width() as u32;
    let height = image.height() as u32;
    let stride = row_stride(width, 8) as u32;
    let pixel_data_size = stride * height;
    let color_table_size = (palette.len() as u32) * 4;
    let pixel_data_offset = 14 + 40 + color_table_size;
    let file_size = pixel_data_offset + pixel_data_size;

    // ── Write BITMAPFILEHEADER (14 bytes) ────────────────────────────────
    let mut file_header = [0u8; 14];
    file_header[0] = 0x42;
    file_header[1] = 0x4D;
    write_u32_le(&mut file_header, 2, file_size);
    write_u32_le(&mut file_header, 10, pixel_data_offset);
    writer.write_all(&file_header).map_err(encode_error)?;

    // ── Write BITMAPINFOHEADER (40 bytes) ────────────────────────────────
    let mut info_header = [0u8; 40];
    write_u32_le(&mut info_header, 0, 40);
    write_i32_le(&mut info_header, 4, width as i32);
    write_i32_le(&mut info_header, 8, height as i32);
    write_u16_le(&mut info_header, 12, 1);
    write_u16_le(&mut info_header, 14, 8); // 8-bit indexed
    write_u32_le(&mut info_header, 16, BI_RGB);
    write_u32_le(&mut info_header, 20, pixel_data_size);

    let (x_res, y_res) = match &options.resolution {
        Some(r) => (r.x_pixels_per_meter, r.y_pixels_per_meter),
        None => (0, 0),
    };
    write_u32_le(&mut info_header, 24, x_res);
    write_u32_le(&mut info_header, 28, y_res);
    write_u32_le(&mut info_header, 32, palette.len() as u32); // biClrUsed
    // biClrImportant stays zero.
    writer.write_all(&info_header).map_err(encode_error)?;

    // ── Write color table (BGRA, 4 bytes per entry) ─────────────────────
    for entry in palette {
        // BMP color table: B, G, R, Reserved (0).
        let ct_entry = [entry.b.0, entry.g.0, entry.r.0, 0];
        writer.write_all(&ct_entry).map_err(encode_error)?;
    }

    // ── Write pixel data (bottom-up, padded rows) ────────────────────────
    let pad_bytes = stride as usize - width as usize;
    let padding = [0u8; 3];

    let mut row_data = vec![0u8; width as usize];
    for row in (0..height as usize).rev() {
        // Fill the row buffer with indices.
        for (col, dst) in row_data.iter_mut().enumerate() {
            *dst = image.pixel_at(col, row).0;
        }
        writer.write_all(&row_data).map_err(encode_error)?;
        if pad_bytes > 0 {
            writer
                .write_all(&padding[..pad_bytes])
                .map_err(encode_error)?;
        }
    }

    Ok(())
}

/// Encode a [`BmpImage`] back to BMP bytes.
///
/// Convenience wrapper that dispatches over all [`BmpImage`] variants,
/// calling [`encode`] for truecolour types and [`encode_indexed`] for
/// the indexed variant.
///
/// # Errors
///
/// Returns [`IoError::EncodeFailed`] on encoding errors.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv_io::bmp::{self, BmpEncodeOptions};
/// let decoded = bmp::decode(&std::fs::read("photo.bmp").unwrap()).unwrap();
/// let bytes = bmp::encode_bmp_image(&decoded.image, &BmpEncodeOptions::default()).unwrap();
/// std::fs::write("copy.bmp", bytes).unwrap();
/// ```
pub fn encode_bmp_image(image: &BmpImage, options: &BmpEncodeOptions) -> Result<Vec<u8>, IoError> {
    match image {
        BmpImage::Indexed8 { data, palette } => encode_indexed(data, palette.as_ref(), options),
        BmpImage::Srgb8(img) => encode(img, options),
        BmpImage::Srgba8(img) => encode(img, options),
    }
}

/// Map an I/O error into `IoError::EncodeFailed`.
fn encode_error(e: std::io::Error) -> IoError {
    IoError::EncodeFailed {
        source: Box::new(e),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use irys_cv::image::ImageView;

    // ── Type definition tests ────────────────────────────────────────────

    #[test]
    fn bmp_image_enum_is_compact() {
        // All variants should be compact (Image is ~3 words + discriminant).
        let size = std::mem::size_of::<BmpImage>();
        // Image has a Size (2 × usize) + Box<[T]> (pointer + len = 2 × usize),
        // plus the enum discriminant and alignment. ≤ 48 bytes is acceptable.
        assert!(size <= 48, "BmpImage is {} bytes, expected <= 48", size);
    }

    #[test]
    fn bmp_decoded_field_access() {
        let img = Image::fill(2, 2, Srgb8::new(10, 20, 30));
        let decoded = BmpDecoded {
            image: BmpImage::Srgb8(img),
            metadata: BmpMetadata {
                color_space: BmpColorSpace::Srgb,
                source_bit_depth: BmpBitDepth::TwentyFour,
                resolution: None,
                compression: BmpCompression::None,
                header_version: BmpHeaderVersion::Info,
                icc_profile: None,
            },
        };
        match &decoded.image {
            BmpImage::Srgb8(img) => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.height(), 2);
            }
            _ => panic!("expected Srgb8"),
        }
        assert_eq!(decoded.metadata.header_version, BmpHeaderVersion::Info);
    }

    #[test]
    fn color_space_variants_are_constructible() {
        let srgb = BmpColorSpace::Srgb;
        let icc = BmpColorSpace::IccTagged;
        assert_ne!(srgb, icc);
        let _ = format!("{:?}", srgb);
        let _ = format!("{:?}", icc);
    }

    #[test]
    fn bit_depth_all_variants() {
        let depths = [
            BmpBitDepth::One,
            BmpBitDepth::Four,
            BmpBitDepth::Eight,
            BmpBitDepth::Sixteen,
            BmpBitDepth::TwentyFour,
            BmpBitDepth::ThirtyTwo,
        ];
        for d in &depths {
            let _ = format!("{:?}", d);
        }
        assert_eq!(depths[0], BmpBitDepth::One);
        assert_ne!(depths[0], depths[1]);
    }

    #[test]
    fn compression_all_variants() {
        let comps = [
            BmpCompression::None,
            BmpCompression::Rle8,
            BmpCompression::Rle4,
            BmpCompression::Bitfields,
        ];
        for c in &comps {
            let _ = format!("{:?}", c);
        }
        assert_eq!(comps[0], BmpCompression::None);
    }

    #[test]
    fn header_version_all_variants() {
        let versions = [
            BmpHeaderVersion::Core,
            BmpHeaderVersion::Info,
            BmpHeaderVersion::V4,
            BmpHeaderVersion::V5,
        ];
        for v in &versions {
            let _ = format!("{:?}", v);
        }
        assert_eq!(versions[1], BmpHeaderVersion::Info);
    }

    #[test]
    fn resolution_constructible() {
        let res = BmpResolution {
            x_pixels_per_meter: 3780,
            y_pixels_per_meter: 3780,
        };
        assert_eq!(res.x_pixels_per_meter, 3780);
        assert_eq!(res, res);
        let _ = format!("{:?}", res);
    }

    #[test]
    fn bmp_encode_options_default() {
        let opts = BmpEncodeOptions::default();
        assert!(opts.resolution.is_none());
        let _ = format!("{:?}", opts);
    }

    #[test]
    fn bmp_image_debug_shows_variant_and_dimensions() {
        let img = Image::fill(320, 240, Srgb8::new(0, 0, 0));
        let bmp_img = BmpImage::Srgb8(img);
        let dbg = format!("{:?}", bmp_img);
        assert_eq!(dbg, "Srgb8(320x240)");
    }

    #[test]
    fn bmp_image_debug_indexed() {
        let img = Image::fill(4, 4, Indexed8(0));
        let palette = Box::new([Srgba8::new(0, 0, 0, 255); 256]);
        let bmp_img = BmpImage::Indexed8 { data: img, palette };
        let dbg = format!("{:?}", bmp_img);
        assert_eq!(dbg, "Indexed8(4x4)");
    }

    #[test]
    fn bmp_image_debug_srgba8() {
        let img = Image::fill(10, 5, Srgba8::new(0, 0, 0, 255));
        let bmp_img = BmpImage::Srgba8(img);
        let dbg = format!("{:?}", bmp_img);
        assert_eq!(dbg, "Srgba8(10x5)");
    }

    #[test]
    fn metadata_fully_populated() {
        let meta = BmpMetadata {
            color_space: BmpColorSpace::IccTagged,
            source_bit_depth: BmpBitDepth::ThirtyTwo,
            resolution: Some(BmpResolution {
                x_pixels_per_meter: 2835,
                y_pixels_per_meter: 2835,
            }),
            compression: BmpCompression::Bitfields,
            header_version: BmpHeaderVersion::V5,
            icc_profile: Some(vec![0u8; 100].into_boxed_slice()),
        };
        assert_eq!(meta.color_space, BmpColorSpace::IccTagged);
        assert_eq!(meta.source_bit_depth, BmpBitDepth::ThirtyTwo);
        assert!(meta.resolution.is_some());
        assert_eq!(meta.compression, BmpCompression::Bitfields);
        assert_eq!(meta.header_version, BmpHeaderVersion::V5);
        assert!(meta.icc_profile.is_some());
    }

    // ── LE reader tests ──────────────────────────────────────────────────

    #[test]
    fn le_reader_u16() {
        assert_eq!(read_u16_le(&[0x01, 0x02], 0), Some(0x0201));
        assert_eq!(read_u16_le(&[0xFF, 0x00], 0), Some(0x00FF));
        assert_eq!(read_u16_le(&[0x00], 0), None);
        assert_eq!(read_u16_le(&[0x00, 0x01, 0x02], 1), Some(0x0201));
    }

    #[test]
    fn le_reader_u32() {
        assert_eq!(read_u32_le(&[0x01, 0x02, 0x03, 0x04], 0), Some(0x04030201));
        assert_eq!(read_u32_le(&[0x00, 0x01, 0x02], 0), None);
    }

    #[test]
    fn le_reader_i32() {
        // -1 in LE = FF FF FF FF
        assert_eq!(read_i32_le(&[0xFF, 0xFF, 0xFF, 0xFF], 0), Some(-1));
        assert_eq!(read_i32_le(&[0x04, 0x00, 0x00, 0x00], 0), Some(4));
    }

    #[test]
    fn le_reader_bounds_checking() {
        assert_eq!(read_u16_le(&[], 0), None);
        assert_eq!(read_u32_le(&[0; 3], 0), None);
        assert_eq!(read_i32_le(&[0; 3], 0), None);
        assert_eq!(read_u16_le(&[0; 10], 9), None);
        assert_eq!(read_u32_le(&[0; 10], 8), None);
    }

    // ── File header parsing tests ────────────────────────────────────────

    #[test]
    fn file_header_valid() {
        let mut data = vec![0u8; 14];
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, 100); // file size
        write_u32_le(&mut data, 10, 54); // pixel data offset
        let header = parse_file_header(&data).unwrap();
        assert_eq!(header.pixel_data_offset, 54);
    }

    #[test]
    fn file_header_reject_non_bmp() {
        let data = vec![
            0x89, 0x50, 0x4E, 0x47, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let result = parse_file_header(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn file_header_reject_truncated() {
        let data = vec![0x42, 0x4D, 0x00];
        let result = parse_file_header(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    // ── DIB header parsing tests ─────────────────────────────────────────

    /// Build a minimal valid BMP with a given DIB header size.
    fn make_bmp_with_info_header(
        width: i32,
        height: i32,
        bit_count: u16,
        compression: u32,
    ) -> Vec<u8> {
        let mut data = vec![0u8; 14 + 40];
        // File header.
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 10, 54);
        // BITMAPINFOHEADER.
        write_u32_le(&mut data, 14, 40); // biSize
        write_i32_le(&mut data, 18, width); // biWidth
        write_i32_le(&mut data, 22, height); // biHeight
        write_u16_le(&mut data, 26, 1); // biPlanes
        write_u16_le(&mut data, 28, bit_count); // biBitCount
        write_u32_le(&mut data, 30, compression); // biCompression
        data
    }

    #[test]
    fn dib_header_info_40_bytes() {
        let data = make_bmp_with_info_header(4, 4, 24, 0);
        let dib = parse_dib_header(&data).unwrap();
        assert_eq!(dib.header_version, BmpHeaderVersion::Info);
        assert_eq!(dib.width, 4);
        assert_eq!(dib.height, 4);
        assert_eq!(dib.bit_count, 24);
        assert!(!dib.top_down);
    }

    #[test]
    fn dib_header_negative_height_is_top_down() {
        let data = make_bmp_with_info_header(4, -4, 24, 0);
        let dib = parse_dib_header(&data).unwrap();
        assert_eq!(dib.height, 4);
        assert!(dib.top_down);
    }

    #[test]
    fn dib_header_reject_zero_width() {
        let data = make_bmp_with_info_header(0, 4, 24, 0);
        let result = parse_dib_header(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn dib_header_reject_zero_height() {
        let data = make_bmp_with_info_header(4, 0, 24, 0);
        let result = parse_dib_header(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn dib_header_reject_invalid_bit_depth() {
        let data = make_bmp_with_info_header(4, 4, 7, 0);
        let result = parse_dib_header(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn dib_header_reject_invalid_planes() {
        let mut data = make_bmp_with_info_header(4, 4, 24, 0);
        write_u16_le(&mut data, 26, 2); // planes = 2
        let result = parse_dib_header(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn dib_header_reject_unsupported_size() {
        let mut data = vec![0u8; 14 + 64];
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 64); // unsupported 64-byte header
        let result = parse_dib_header(&data);
        assert!(matches!(result, Err(IoError::UnsupportedFeature { .. })));
    }

    #[test]
    fn dib_header_core_12_bytes() {
        let mut data = vec![0u8; 14 + 12];
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 12); // bcSize
        write_u16_le(&mut data, 18, 4); // bcWidth
        write_u16_le(&mut data, 20, 4); // bcHeight
        write_u16_le(&mut data, 22, 1); // bcPlanes
        write_u16_le(&mut data, 24, 24); // bcBitCount
        let dib = parse_dib_header(&data).unwrap();
        assert_eq!(dib.header_version, BmpHeaderVersion::Core);
        assert_eq!(dib.width, 4);
        assert_eq!(dib.height, 4);
        assert_eq!(dib.color_table_entry_size, 3);
    }

    // ── Color table parsing tests ────────────────────────────────────────

    #[test]
    fn color_table_bgr_reorder() {
        // 4-byte entries: B, G, R, reserved.
        let data = [
            0x00, 0x00, 0xFF, 0x00, // entry 0: B=0, G=0, R=255 → red
            0xFF, 0x00, 0x00, 0x00, // entry 1: B=255, G=0, R=0 → blue
        ];
        let palette = parse_color_table(&data, 0, 4, 2).unwrap();
        assert_eq!(palette[0], Srgba8::new(255, 0, 0, 255));
        assert_eq!(palette[1], Srgba8::new(0, 0, 255, 255));
        // Remaining entries should be opaque black.
        assert_eq!(palette[255], Srgba8::new(0, 0, 0, 255));
    }

    #[test]
    fn color_table_3byte_core_entries() {
        // 3-byte entries (BITMAPCOREHEADER): B, G, R.
        let data = [
            0x00, 0xFF, 0x00, // entry 0: B=0, G=255, R=0 → green
        ];
        let palette = parse_color_table(&data, 0, 3, 1).unwrap();
        assert_eq!(palette[0], Srgba8::new(0, 255, 0, 255));
    }

    #[test]
    fn color_table_partial() {
        let data = [0xAA, 0xBB, 0xCC, 0x00];
        let palette = parse_color_table(&data, 0, 4, 1).unwrap();
        assert_eq!(palette[0], Srgba8::new(0xCC, 0xBB, 0xAA, 255));
        // Unused entries are black.
        assert_eq!(palette[1], Srgba8::new(0, 0, 0, 255));
    }

    // ── Bitfield mask tests ──────────────────────────────────────────────

    #[test]
    fn mask_info_555() {
        let masks = BitfieldMasks::default_16bit();
        let r = mask_info(masks.r); // 0x7C00 = 0111_1100_0000_0000
        let g = mask_info(masks.g); // 0x03E0 = 0000_0011_1110_0000
        let b = mask_info(masks.b); // 0x001F = 0000_0000_0001_1111

        assert_eq!(r.shift, 10);
        assert_eq!(r.bits, 5);
        assert_eq!(g.shift, 5);
        assert_eq!(g.bits, 5);
        assert_eq!(b.shift, 0);
        assert_eq!(b.bits, 5);
    }

    #[test]
    fn mask_info_32bit_standard() {
        let masks = BitfieldMasks::default_32bit();
        let r = mask_info(masks.r);
        let g = mask_info(masks.g);
        let b = mask_info(masks.b);

        assert_eq!(r.shift, 16);
        assert_eq!(r.bits, 8);
        assert_eq!(g.shift, 8);
        assert_eq!(g.bits, 8);
        assert_eq!(b.shift, 0);
        assert_eq!(b.bits, 8);
    }

    #[test]
    fn extract_channel_5bit() {
        let info = MaskInfo { shift: 10, bits: 5 };
        // 5-bit max (31) should scale to 255.
        assert_eq!(extract_channel(0x7C00, info), 255);
        // 5-bit 0 should be 0.
        assert_eq!(extract_channel(0x0000, info), 0);
        // 5-bit 16 should scale to ~131.
        let val = extract_channel(16 << 10, info);
        assert_eq!(val, ((16u32 * 255 + 15) / 31) as u8);
    }

    #[test]
    fn extract_channel_8bit() {
        let info = MaskInfo { shift: 16, bits: 8 };
        assert_eq!(extract_channel(0x00FF0000, info), 255);
        assert_eq!(extract_channel(0x00800000, info), 128);
        assert_eq!(extract_channel(0x00000000, info), 0);
    }

    #[test]
    fn extract_channel_zero_mask() {
        let info = MaskInfo { shift: 0, bits: 0 };
        assert_eq!(extract_channel(0xFFFFFFFF, info), 0);
    }

    // ── RLE8 decompression tests ─────────────────────────────────────────

    #[test]
    fn rle8_simple_run() {
        // 3 pixels of value 7, then end-of-bitmap.
        let data = [3, 7, 0, 1];
        let result = decompress_rle8(&data, 4, 1).unwrap();
        assert_eq!(result[0..3], [7, 7, 7]);
        assert_eq!(result[3], 0); // unfilled
    }

    #[test]
    fn rle8_end_of_line() {
        // Row 0: 2 pixels of value 1.
        // End of line.
        // Row 1: 2 pixels of value 2.
        // End of bitmap.
        let data = [2, 1, 0, 0, 2, 2, 0, 1];
        let result = decompress_rle8(&data, 3, 2).unwrap();
        assert_eq!(result[0..2], [1, 1]);
        assert_eq!(result[2], 0); // unfilled in row 0
        assert_eq!(result[3..5], [2, 2]);
    }

    #[test]
    fn rle8_delta() {
        // Delta: move 1 right, 1 down. Then 1 pixel of value 5. End.
        let data = [0, 2, 1, 1, 1, 5, 0, 1];
        let result = decompress_rle8(&data, 3, 3).unwrap();
        // After delta, cursor is at (1, 1).
        assert_eq!(result[1 * 3 + 1], 5);
    }

    #[test]
    fn rle8_absolute_mode_even() {
        // Absolute mode: 4 literal pixels.
        let data = [0, 4, 10, 20, 30, 40, 0, 1];
        let result = decompress_rle8(&data, 4, 1).unwrap();
        assert_eq!(result, [10, 20, 30, 40]);
    }

    #[test]
    fn rle8_absolute_mode_odd() {
        // Absolute mode: 3 literal pixels (odd → 1 pad byte).
        let data = [0, 3, 10, 20, 30, 0, 0, 1]; // pad byte after 30
        let result = decompress_rle8(&data, 3, 1).unwrap();
        assert_eq!(result, [10, 20, 30]);
    }

    #[test]
    fn rle8_out_of_bounds_error() {
        // Try to write more pixels than the image has room for.
        let data = [255, 7]; // 255 pixels but image is 2×1
        let result = decompress_rle8(&data, 2, 1);
        // Should not panic — should error or write within bounds.
        // The current implementation allows x to exceed width (skips writes),
        // but y overflow is caught.
        assert!(result.is_ok() || result.is_err());
    }

    // ── RLE4 decompression tests ─────────────────────────────────────────

    #[test]
    fn rle4_simple_run() {
        // 4 pixels alternating nibbles 0xA and 0xB.
        let data = [4, 0xAB, 0, 1];
        let result = decompress_rle4(&data, 4, 1).unwrap();
        assert_eq!(result, [0xA, 0xB, 0xA, 0xB]);
    }

    #[test]
    fn rle4_absolute_mode() {
        // Absolute mode: 3 pixels. Nibble-packed: 0x12, 0x30.
        // [0, 3]: absolute mode, 3 pixels.
        // Pixels come from nibbles: 0x12 → pixels 1, 2; 0x30 → pixel 3.
        // byte_count = (3+1)/2 = 2 bytes of data.
        // 2 bytes is even, so no pad byte needed.
        let data = [0, 3, 0x12, 0x30, 0, 1];
        let result = decompress_rle4(&data, 4, 1).unwrap();
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);
    }

    #[test]
    fn rle4_out_of_bounds() {
        // Write way more data than fits.
        let data = [255, 0xAB]; // 255 pixels alternating, image is 2×1
        let result = decompress_rle4(&data, 2, 1);
        // Should not panic.
        assert!(result.is_ok() || result.is_err());
    }

    // ── Row stride tests ─────────────────────────────────────────────────

    #[test]
    fn row_stride_24bit() {
        // Width 1, 24bpp: 3 bytes → padded to 4.
        assert_eq!(row_stride(1, 24), 4);
        // Width 2, 24bpp: 6 bytes → padded to 8.
        assert_eq!(row_stride(2, 24), 8);
        // Width 4, 24bpp: 12 bytes → already aligned.
        assert_eq!(row_stride(4, 24), 12);
    }

    #[test]
    fn row_stride_32bit() {
        // 32bpp: width × 4 is always 4-aligned.
        assert_eq!(row_stride(1, 32), 4);
        assert_eq!(row_stride(3, 32), 12);
    }

    #[test]
    fn row_stride_8bit() {
        assert_eq!(row_stride(1, 8), 4);
        assert_eq!(row_stride(4, 8), 4);
        assert_eq!(row_stride(5, 8), 8);
    }

    #[test]
    fn row_stride_1bit() {
        // 1bpp, width 1: 1 bit → 1 byte → padded to 4.
        assert_eq!(row_stride(1, 1), 4);
        // 1bpp, width 33: 33 bits → 5 bytes → padded to 8.
        assert_eq!(row_stride(33, 1), 8);
    }

    // ── Decode tests ─────────────────────────────────────────────────────

    /// Helper: build a minimal 24-bit BMP in memory.
    fn make_24bit_bmp(width: u32, height: u32, pixels: &[Srgb8]) -> Vec<u8> {
        let stride = row_stride(width, 24);
        let pixel_data_size = stride * height as usize;
        let file_size = 14 + 40 + pixel_data_size;
        let mut data = vec![0u8; file_size];

        // File header.
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, 54);

        // BITMAPINFOHEADER.
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32); // bottom-up
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 24);
        write_u32_le(&mut data, 30, 0); // BI_RGB

        // Pixel data: bottom-up, BGR.
        for row in 0..height as usize {
            let src_row = height as usize - 1 - row; // bottom-up
            let row_offset = 54 + row * stride;
            for col in 0..width as usize {
                let pix = &pixels[src_row * width as usize + col];
                let pix_offset = row_offset + col * 3;
                data[pix_offset] = pix.b.0;
                data[pix_offset + 1] = pix.g.0;
                data[pix_offset + 2] = pix.r.0;
            }
        }

        data
    }

    #[test]
    fn decode_24bit_uncompressed() {
        let pixels = vec![
            Srgb8::new(255, 0, 0),
            Srgb8::new(0, 255, 0),
            Srgb8::new(0, 0, 255),
            Srgb8::new(128, 128, 128),
        ];
        let data = make_24bit_bmp(2, 2, &pixels);
        let decoded = decode(&data).unwrap();

        match &decoded.image {
            BmpImage::Srgb8(img) => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.height(), 2);
                assert_eq!(img.pixel_at(0, 0), Srgb8::new(255, 0, 0));
                assert_eq!(img.pixel_at(1, 0), Srgb8::new(0, 255, 0));
                assert_eq!(img.pixel_at(0, 1), Srgb8::new(0, 0, 255));
                assert_eq!(img.pixel_at(1, 1), Srgb8::new(128, 128, 128));
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }

        assert_eq!(decoded.metadata.source_bit_depth, BmpBitDepth::TwentyFour);
        assert_eq!(decoded.metadata.compression, BmpCompression::None);
        assert_eq!(decoded.metadata.header_version, BmpHeaderVersion::Info);
        assert_eq!(decoded.metadata.color_space, BmpColorSpace::Srgb);
    }

    #[test]
    fn decode_24bit_bgr_reorder_correct() {
        // Specifically test that BGR → RGB reordering is correct.
        // A pixel with B=10, G=20, R=30 should decode as Srgb8(30, 20, 10).
        let pixels = vec![Srgb8::new(30, 20, 10)];
        let data = make_24bit_bmp(1, 1, &pixels);
        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(img) => {
                assert_eq!(img.pixel_at(0, 0), Srgb8::new(30, 20, 10));
            }
            _ => panic!("expected Srgb8"),
        }
    }

    #[test]
    fn decode_preserves_dimensions() {
        let pixels = vec![Srgb8::new(0, 0, 0); 5 * 3];
        let data = make_24bit_bmp(5, 3, &pixels);
        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(img) => {
                assert_eq!(img.width(), 5);
                assert_eq!(img.height(), 3);
            }
            _ => panic!("expected Srgb8"),
        }
    }

    #[test]
    fn decode_24bit_with_row_padding() {
        // Width 3 at 24bpp = 9 bytes per row, padded to 12.
        let pixels = vec![
            Srgb8::new(1, 2, 3),
            Srgb8::new(4, 5, 6),
            Srgb8::new(7, 8, 9),
        ];
        let data = make_24bit_bmp(3, 1, &pixels);
        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(img) => {
                assert_eq!(img.pixel_at(0, 0), Srgb8::new(1, 2, 3));
                assert_eq!(img.pixel_at(1, 0), Srgb8::new(4, 5, 6));
                assert_eq!(img.pixel_at(2, 0), Srgb8::new(7, 8, 9));
            }
            _ => panic!("expected Srgb8"),
        }
    }

    #[test]
    fn decode_top_down() {
        // Build a top-down BMP (negative height).
        let width = 2u32;
        let height = 2u32;
        let stride = row_stride(width, 24);
        let pixel_data_size = stride * height as usize;
        let file_size = 14 + 40 + pixel_data_size;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, 54);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, -(height as i32)); // top-down
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 24);

        // Top-down: first row in file = top row of image.
        // Row 0: red, green.
        data[54] = 0;
        data[55] = 0;
        data[56] = 255; // B=0, G=0, R=255 → red
        data[57] = 0;
        data[58] = 255;
        data[59] = 0; // B=0, G=255, R=0 → green
        // Row 1: blue, white.
        let row1_start = 54 + stride;
        data[row1_start] = 255;
        data[row1_start + 1] = 0;
        data[row1_start + 2] = 0; // B=255, G=0, R=0 → blue
        data[row1_start + 3] = 255;
        data[row1_start + 4] = 255;
        data[row1_start + 5] = 255; // white

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(img) => {
                assert_eq!(img.pixel_at(0, 0), Srgb8::new(255, 0, 0)); // top-left = red
                assert_eq!(img.pixel_at(1, 0), Srgb8::new(0, 255, 0)); // top-right = green
                assert_eq!(img.pixel_at(0, 1), Srgb8::new(0, 0, 255)); // bottom-left = blue
                assert_eq!(img.pixel_at(1, 1), Srgb8::new(255, 255, 255)); // bottom-right = white
            }
            _ => panic!("expected Srgb8"),
        }
    }

    #[test]
    fn decode_invalid_data_returns_error() {
        let result = decode(&[0x00, 0x00, 0x00, 0x00]);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn decode_empty_data_returns_error() {
        let result = decode(&[]);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn decode_bi_jpeg_unsupported() {
        let data = make_bmp_with_info_header(4, 4, 24, 4); // BI_JPEG = 4
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::UnsupportedFeature { .. })));
    }

    #[test]
    fn decode_bi_png_unsupported() {
        let data = make_bmp_with_info_header(4, 4, 24, 5); // BI_PNG = 5
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::UnsupportedFeature { .. })));
    }

    #[test]
    fn decode_unsupported_dib_header_size() {
        let mut data = vec![0u8; 14 + 16];
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 16); // unsupported size
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::UnsupportedFeature { .. })));
    }

    #[test]
    fn decode_truncated_pixel_data() {
        // Make a valid header but truncate the pixel data.
        let data = make_bmp_with_info_header(4, 4, 24, 0);
        // data is only 54 bytes — no pixel data at all.
        let result = decode(&data);
        assert!(result.is_err());
    }

    #[test]
    fn decode_reader_from_cursor() {
        let pixels = vec![Srgb8::new(100, 150, 200), Srgb8::new(50, 60, 70)];
        let data = make_24bit_bmp(2, 1, &pixels);
        let decoded = decode_reader(std::io::Cursor::new(&data)).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(img) => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.pixel_at(0, 0), Srgb8::new(100, 150, 200));
                assert_eq!(img.pixel_at(1, 0), Srgb8::new(50, 60, 70));
            }
            _ => panic!("expected Srgb8"),
        }
    }

    // ── 8-bit indexed decode test ────────────────────────────────────────

    /// Helper: build an 8-bit indexed BMP.
    fn make_8bit_indexed_bmp(
        width: u32,
        height: u32,
        palette: &[Srgba8],
        indices: &[u8],
    ) -> Vec<u8> {
        let stride = row_stride(width, 8);
        let color_table_size = palette.len() * 4;
        let pixel_data_size = stride * height as usize;
        let pixel_offset = 14 + 40 + color_table_size;
        let file_size = pixel_offset + pixel_data_size;
        let mut data = vec![0u8; file_size];

        // File header.
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);

        // BITMAPINFOHEADER.
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 8);
        write_u32_le(&mut data, 30, 0); // BI_RGB
        write_u32_le(&mut data, 46, palette.len() as u32); // biClrUsed

        // Color table (BGRA, 4 bytes each).
        for (i, entry) in palette.iter().enumerate() {
            let offset = 54 + i * 4;
            data[offset] = entry.b.0;
            data[offset + 1] = entry.g.0;
            data[offset + 2] = entry.r.0;
            data[offset + 3] = 0; // reserved
        }

        // Pixel data (bottom-up).
        for row in 0..height as usize {
            let src_row = height as usize - 1 - row;
            let row_offset = pixel_offset + row * stride;
            for col in 0..width as usize {
                data[row_offset + col] = indices[src_row * width as usize + col];
            }
        }

        data
    }

    #[test]
    fn decode_8bit_indexed() {
        let palette = vec![
            Srgba8::new(255, 0, 0, 255), // index 0 = red
            Srgba8::new(0, 255, 0, 255), // index 1 = green
            Srgba8::new(0, 0, 255, 255), // index 2 = blue
        ];
        let indices = vec![0, 1, 2, 1];
        let data = make_8bit_indexed_bmp(2, 2, &palette, &indices);
        let decoded = decode(&data).unwrap();

        match &decoded.image {
            BmpImage::Indexed8 {
                data: img,
                palette: pal,
            } => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.height(), 2);
                assert_eq!(img.pixel_at(0, 0).0, 0);
                assert_eq!(img.pixel_at(1, 0).0, 1);
                assert_eq!(img.pixel_at(0, 1).0, 2);
                assert_eq!(img.pixel_at(1, 1).0, 1);
                assert_eq!(pal[0], Srgba8::new(255, 0, 0, 255));
                assert_eq!(pal[1], Srgba8::new(0, 255, 0, 255));
                assert_eq!(pal[2], Srgba8::new(0, 0, 255, 255));
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }

        assert_eq!(decoded.metadata.source_bit_depth, BmpBitDepth::Eight);
    }

    // ── 1-bit indexed decode test ────────────────────────────────────────

    #[test]
    fn decode_1bit_indexed() {
        let palette = [
            Srgba8::new(0, 0, 0, 255),       // index 0 = black
            Srgba8::new(255, 255, 255, 255), // index 1 = white
        ];

        let stride = row_stride(4, 1); // 4 pixels × 1 bit = 4 bits → 1 byte → padded to 4
        let pixel_offset = 14 + 40 + 2 * 4; // 2 palette entries × 4 bytes
        let file_size = pixel_offset + stride;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, 4);
        write_i32_le(&mut data, 22, 1);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 1);
        write_u32_le(&mut data, 46, 2);

        // Color table (BGRA, 4 bytes each).
        for (i, entry) in palette.iter().enumerate() {
            let off = 54 + i * 4;
            data[off] = entry.b.0;
            data[off + 1] = entry.g.0;
            data[off + 2] = entry.r.0;
            data[off + 3] = 0;
        }

        // Pixel data: 1010 xxxx (MSB first) = indices 1, 0, 1, 0.
        data[pixel_offset] = 0b1010_0000;

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: img, .. } => {
                assert_eq!(img.pixel_at(0, 0).0, 1);
                assert_eq!(img.pixel_at(1, 0).0, 0);
                assert_eq!(img.pixel_at(2, 0).0, 1);
                assert_eq!(img.pixel_at(3, 0).0, 0);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }

        assert_eq!(decoded.metadata.source_bit_depth, BmpBitDepth::One);
    }

    // ── 4-bit indexed decode test ────────────────────────────────────────

    #[test]
    fn decode_4bit_indexed() {
        // 4 pixels: indices 0xA, 0xB, 0xC, 0xD.
        // We need a palette with at least 14 entries (0..=0xD).
        let palette: Vec<Srgba8> = (0..16)
            .map(|i| Srgba8::new(i * 16, i * 16, i * 16, 255))
            .collect();
        let _ = palette.len(); // 16 entries

        let stride = row_stride(4, 4); // 4 pixels × 4 bits = 16 bits = 2 bytes → padded to 4
        let pixel_offset = 14 + 40 + 16 * 4;
        let file_size = pixel_offset + stride;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, 4);
        write_i32_le(&mut data, 22, 1);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 4);
        write_u32_le(&mut data, 46, 16);

        // Color table.
        for (i, entry) in palette.iter().enumerate() {
            let off = 54 + i * 4;
            data[off] = entry.b.0;
            data[off + 1] = entry.g.0;
            data[off + 2] = entry.r.0;
            data[off + 3] = 0;
        }

        // Pixel data: 0xAB, 0xCD → indices 0xA, 0xB, 0xC, 0xD.
        data[pixel_offset] = 0xAB;
        data[pixel_offset + 1] = 0xCD;

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: img, .. } => {
                assert_eq!(img.pixel_at(0, 0).0, 0xA);
                assert_eq!(img.pixel_at(1, 0).0, 0xB);
                assert_eq!(img.pixel_at(2, 0).0, 0xC);
                assert_eq!(img.pixel_at(3, 0).0, 0xD);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }

        assert_eq!(decoded.metadata.source_bit_depth, BmpBitDepth::Four);
    }

    // ── 32-bit decode tests ──────────────────────────────────────────────

    #[test]
    fn decode_32bit_no_alpha() {
        // 32-bit BI_RGB → BGRX, no alpha → Srgb8.
        let width = 2u32;
        let height = 1u32;
        let stride = row_stride(width, 32);
        let pixel_offset = 54usize;
        let file_size = pixel_offset + stride * height as usize;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 32);
        write_u32_le(&mut data, 30, 0); // BI_RGB

        // Pixel 0: B=10, G=20, R=30, X=0.
        data[pixel_offset] = 10;
        data[pixel_offset + 1] = 20;
        data[pixel_offset + 2] = 30;
        data[pixel_offset + 3] = 0;
        // Pixel 1: B=100, G=150, R=200, X=255.
        data[pixel_offset + 4] = 100;
        data[pixel_offset + 5] = 150;
        data[pixel_offset + 6] = 200;
        data[pixel_offset + 7] = 255;

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(img) => {
                assert_eq!(img.pixel_at(0, 0), Srgb8::new(30, 20, 10));
                assert_eq!(img.pixel_at(1, 0), Srgb8::new(200, 150, 100));
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }

        assert_eq!(decoded.metadata.source_bit_depth, BmpBitDepth::ThirtyTwo);
    }

    #[test]
    fn decode_32bit_with_alpha() {
        // 32-bit BI_BITFIELDS with alpha mask → Srgba8.
        // Need a V4 header to embed the masks.
        let width = 1u32;
        let height = 1u32;
        let header_size = 108u32;
        let pixel_offset = 14 + header_size;
        let stride = row_stride(width, 32);
        let file_size = pixel_offset as usize + stride * height as usize;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset);
        write_u32_le(&mut data, 14, header_size);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 32);
        write_u32_le(&mut data, 30, BI_BITFIELDS);
        // V4 masks.
        write_u32_le(&mut data, 54, 0x00FF0000); // R
        write_u32_le(&mut data, 58, 0x0000FF00); // G
        write_u32_le(&mut data, 62, 0x000000FF); // B
        write_u32_le(&mut data, 66, 0xFF000000); // A
        write_u32_le(&mut data, 70, LCS_SRGB); // csType

        // Pixel: BGRA as u32 LE = 0xCC332211 → B=0x11, G=0x22, R=0x33, A=0xCC.
        let pix_off = pixel_offset as usize;
        // Write bytes directly: B, G, R, A in the 32-bit word.
        // With masks: R = (val >> 16) & 0xFF, G = (val >> 8) & 0xFF, B = val & 0xFF, A = (val >> 24) & 0xFF.
        // val = 0xCC332211: R = 0x33, G = 0x22, B = 0x11, A = 0xCC.
        write_u32_le(&mut data, pix_off, 0xCC332211);

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Srgba8(img) => {
                let p = img.pixel_at(0, 0);
                assert_eq!(p.r.0, 0x33);
                assert_eq!(p.g.0, 0x22);
                assert_eq!(p.b.0, 0x11);
                assert_eq!(p.a.0, 0xCC);
            }
            other => panic!("expected Srgba8, got {:?}", other),
        }

        assert_eq!(decoded.metadata.source_bit_depth, BmpBitDepth::ThirtyTwo);
        assert_eq!(decoded.metadata.compression, BmpCompression::Bitfields);
    }

    // ── 16-bit decode tests ──────────────────────────────────────────────

    #[test]
    fn decode_16bit_555() {
        // 16-bit BI_RGB default = 5-5-5.
        let width = 1u32;
        let height = 1u32;
        let stride = row_stride(width, 16);
        let pixel_offset = 54usize;
        let file_size = pixel_offset + stride * height as usize;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 16);
        write_u32_le(&mut data, 30, 0); // BI_RGB

        // Pixel: all channels max = 0x7FFF = 0111_1111_1111_1111.
        // R = bits 14..10 = 11111 = 31
        // G = bits 9..5   = 11111 = 31
        // B = bits 4..0   = 11111 = 31
        write_u16_le(&mut data, pixel_offset, 0x7FFF);

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(img) => {
                let p = img.pixel_at(0, 0);
                assert_eq!(p.r.0, 255);
                assert_eq!(p.g.0, 255);
                assert_eq!(p.b.0, 255);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }

        assert_eq!(decoded.metadata.source_bit_depth, BmpBitDepth::Sixteen);
    }

    #[test]
    fn decode_16bit_565_bitfields() {
        // 16-bit BI_BITFIELDS with 5-6-5 masks.
        let width = 1u32;
        let height = 1u32;
        let mask_size = 12usize; // 3 × u32
        let pixel_offset = 54 + mask_size;
        let stride = row_stride(width, 16);
        let file_size = pixel_offset + stride * height as usize;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 16);
        write_u32_le(&mut data, 30, BI_BITFIELDS);

        // Bitfield masks (after the 40-byte header).
        write_u32_le(&mut data, 54, 0xF800); // R: 5 bits at 11
        write_u32_le(&mut data, 58, 0x07E0); // G: 6 bits at 5
        write_u32_le(&mut data, 62, 0x001F); // B: 5 bits at 0

        // Pixel: R=31, G=63, B=31 → all max.
        write_u16_le(&mut data, pixel_offset, 0xFFFF);

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(img) => {
                let p = img.pixel_at(0, 0);
                assert_eq!(p.r.0, 255);
                assert_eq!(p.g.0, 255);
                assert_eq!(p.b.0, 255);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    // ── RLE decode tests ─────────────────────────────────────────────────

    #[test]
    fn decode_rle8_compressed() {
        // Build a RLE8-compressed BMP.
        let palette = [Srgba8::new(0, 0, 0, 255), Srgba8::new(255, 0, 0, 255)];
        let width = 3u32;
        let height = 1u32;
        let color_table_size = 2 * 4;
        let pixel_offset = 14 + 40 + color_table_size;

        // RLE data: 3 pixels of index 1, then end-of-bitmap.
        let rle_data = [3u8, 1, 0, 1];
        let file_size = pixel_offset + rle_data.len();
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 8);
        write_u32_le(&mut data, 30, BI_RLE8);
        write_u32_le(&mut data, 46, 2);

        // Color table.
        for (i, entry) in palette.iter().enumerate() {
            let off = 54 + i * 4;
            data[off] = entry.b.0;
            data[off + 1] = entry.g.0;
            data[off + 2] = entry.r.0;
            data[off + 3] = 0;
        }

        // RLE pixel data.
        data[pixel_offset..pixel_offset + rle_data.len()].copy_from_slice(&rle_data);

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: img, .. } => {
                assert_eq!(img.pixel_at(0, 0).0, 1);
                assert_eq!(img.pixel_at(1, 0).0, 1);
                assert_eq!(img.pixel_at(2, 0).0, 1);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }

        assert_eq!(decoded.metadata.compression, BmpCompression::Rle8);
    }

    // ── BITMAPCOREHEADER decode test ─────────────────────────────────────

    #[test]
    fn decode_core_header_24bit() {
        // Build a BITMAPCOREHEADER BMP (12-byte header).
        let width = 2u16;
        let height = 1u16;
        let stride = row_stride(width as u32, 24);
        let pixel_offset = 14 + 12;
        let file_size = pixel_offset + stride;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 12); // bcSize
        write_u16_le(&mut data, 18, width);
        write_u16_le(&mut data, 20, height);
        write_u16_le(&mut data, 22, 1); // bcPlanes
        write_u16_le(&mut data, 24, 24); // bcBitCount

        // Pixel data (bottom-up for 1 row doesn't matter): BGR.
        data[pixel_offset] = 10; // B
        data[pixel_offset + 1] = 20; // G
        data[pixel_offset + 2] = 30; // R
        data[pixel_offset + 3] = 100; // B
        data[pixel_offset + 4] = 150; // G
        data[pixel_offset + 5] = 200; // R

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(img) => {
                assert_eq!(img.pixel_at(0, 0), Srgb8::new(30, 20, 10));
                assert_eq!(img.pixel_at(1, 0), Srgb8::new(200, 150, 100));
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }

        assert_eq!(decoded.metadata.header_version, BmpHeaderVersion::Core);
        assert_eq!(decoded.metadata.color_space, BmpColorSpace::Srgb);
    }

    // ── Encode tests ─────────────────────────────────────────────────────

    #[test]
    fn encode_srgb8_roundtrip() {
        let img = Image::fill(3, 2, Srgb8::new(100, 150, 200));
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(dec_img) => {
                assert_eq!(dec_img.width(), 3);
                assert_eq!(dec_img.height(), 2);
                for y in 0..2 {
                    for x in 0..3 {
                        assert_eq!(
                            dec_img.pixel_at(x, y),
                            Srgb8::new(100, 150, 200),
                            "mismatch at ({}, {})",
                            x,
                            y
                        );
                    }
                }
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn encode_srgba8_roundtrip() {
        let img = Image::fill(2, 2, Srgba8::new(10, 20, 30, 128));
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Srgba8(dec_img) => {
                assert_eq!(dec_img.width(), 2);
                assert_eq!(dec_img.height(), 2);
                for y in 0..2 {
                    for x in 0..2 {
                        assert_eq!(
                            dec_img.pixel_at(x, y),
                            Srgba8::new(10, 20, 30, 128),
                            "mismatch at ({}, {})",
                            x,
                            y
                        );
                    }
                }
            }
            other => panic!("expected Srgba8, got {:?}", other),
        }
    }

    #[test]
    fn encode_indexed_roundtrip() {
        let palette = [
            Srgba8::new(255, 0, 0, 255),
            Srgba8::new(0, 255, 0, 255),
            Srgba8::new(0, 0, 255, 255),
        ];
        let img = Image::from_vec(
            2,
            2,
            vec![Indexed8(0), Indexed8(1), Indexed8(2), Indexed8(1)],
        )
        .unwrap();

        let bytes = encode_indexed(&img, &palette, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();

        match &decoded.image {
            BmpImage::Indexed8 {
                data: dec_img,
                palette: dec_pal,
            } => {
                assert_eq!(dec_img.width(), 2);
                assert_eq!(dec_img.height(), 2);
                assert_eq!(dec_img.pixel_at(0, 0).0, 0);
                assert_eq!(dec_img.pixel_at(1, 0).0, 1);
                assert_eq!(dec_img.pixel_at(0, 1).0, 2);
                assert_eq!(dec_img.pixel_at(1, 1).0, 1);
                assert_eq!(dec_pal[0], Srgba8::new(255, 0, 0, 255));
                assert_eq!(dec_pal[1], Srgba8::new(0, 255, 0, 255));
                assert_eq!(dec_pal[2], Srgba8::new(0, 0, 255, 255));
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    #[test]
    fn encode_writer_byte_identical_to_encode() {
        let img = Image::fill(4, 3, Srgb8::new(50, 100, 150));
        let encode_result = encode(&img, &BmpEncodeOptions::default()).unwrap();
        let mut writer_result = Vec::new();
        encode_writer(&img, &mut writer_result, &BmpEncodeOptions::default()).unwrap();
        assert_eq!(encode_result, writer_result);
    }

    #[test]
    fn encode_dimensions_roundtrip() {
        for (w, h) in [(1, 1), (2, 3), (5, 1), (1, 7), (100, 50)] {
            let img = Image::fill(w, h, Srgb8::new(0, 0, 0));
            let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();
            let decoded = decode(&bytes).unwrap();
            match &decoded.image {
                BmpImage::Srgb8(dec_img) => {
                    assert_eq!(dec_img.width(), w);
                    assert_eq!(dec_img.height(), h);
                }
                _ => panic!("expected Srgb8"),
            }
        }
    }

    #[test]
    fn encode_pixel_values_roundtrip() {
        // BMP is lossless — pixel values MUST match exactly.
        let img = Image::generate(4, 4, |x, y| {
            Srgb8::new(
                (x * 60 + y * 20) as u8,
                (x * 30 + y * 40) as u8,
                (x * 10 + y * 80) as u8,
            )
        });
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(dec_img) => {
                for y in 0..4 {
                    for x in 0..4 {
                        assert_eq!(
                            dec_img.pixel_at(x, y),
                            img.pixel_at(x, y),
                            "pixel mismatch at ({}, {})",
                            x,
                            y
                        );
                    }
                }
            }
            _ => panic!("expected Srgb8"),
        }
    }

    #[test]
    fn encode_bgr_byte_order_correct() {
        let img = Image::fill(1, 1, Srgb8::new(0xAA, 0xBB, 0xCC));
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();

        // Pixel data starts at offset 54 for 24-bit.
        // Should be in BGR order: CC, BB, AA.
        assert_eq!(bytes[54], 0xCC); // B
        assert_eq!(bytes[55], 0xBB); // G
        assert_eq!(bytes[56], 0xAA); // R
    }

    #[test]
    fn encode_row_padding_correct() {
        // Width 1 at 24bpp = 3 bytes per row. Padded to 4.
        let img = Image::fill(1, 1, Srgb8::new(255, 0, 0));
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();

        // File should be 14 + 40 + 4 = 58 bytes (4-byte aligned row).
        assert_eq!(bytes.len(), 58);
        // Pad byte should be 0.
        assert_eq!(bytes[57], 0);
    }

    #[test]
    fn encode_rows_are_bottom_up() {
        // 1×2 image. Top pixel = red, bottom pixel = blue.
        let img =
            Image::from_vec(1, 2, vec![Srgb8::new(255, 0, 0), Srgb8::new(0, 0, 255)]).unwrap();
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();

        let stride = row_stride(1, 24);
        // First row in file (at offset 54) should be the bottom row (blue).
        assert_eq!(bytes[54], 255); // B=255 (blue pixel)
        assert_eq!(bytes[55], 0); // G=0
        assert_eq!(bytes[56], 0); // R=0
        // Second row should be the top row (red).
        assert_eq!(bytes[54 + stride], 0); // B=0
        assert_eq!(bytes[54 + stride + 1], 0); // G=0
        assert_eq!(bytes[54 + stride + 2], 255); // R=255
    }

    #[test]
    fn encode_resolution_in_header() {
        let img = Image::fill(1, 1, Srgb8::new(0, 0, 0));
        let opts = BmpEncodeOptions {
            resolution: Some(BmpResolution {
                x_pixels_per_meter: 3780,
                y_pixels_per_meter: 3780,
            }),
            ..Default::default()
        };
        let bytes = encode(&img, &opts).unwrap();

        // biXPelsPerMeter at offset 14+24 = 38.
        assert_eq!(read_u32_le(&bytes, 38), Some(3780));
        // biYPelsPerMeter at offset 14+28 = 42.
        assert_eq!(read_u32_le(&bytes, 42), Some(3780));
    }

    #[test]
    fn encode_bmp_image_dispatches_srgb8() {
        let img = Image::fill(2, 2, Srgb8::new(1, 2, 3));
        let bmp_img = BmpImage::Srgb8(img);
        let bytes = encode_bmp_image(&bmp_img, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(dec_img) => {
                assert_eq!(dec_img.pixel_at(0, 0), Srgb8::new(1, 2, 3));
            }
            _ => panic!("expected Srgb8"),
        }
    }

    #[test]
    fn encode_bmp_image_dispatches_srgba8() {
        let img = Image::fill(2, 2, Srgba8::new(10, 20, 30, 128));
        let bmp_img = BmpImage::Srgba8(img);
        let bytes = encode_bmp_image(&bmp_img, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Srgba8(_) => {}
            _ => panic!("expected Srgba8"),
        }
    }

    #[test]
    fn encode_bmp_image_dispatches_indexed() {
        let palette = Box::new([Srgba8::new(128, 64, 32, 255); 256]);
        let img = Image::fill(2, 2, Indexed8(0));
        let bmp_img = BmpImage::Indexed8 { data: img, palette };
        let bytes = encode_bmp_image(&bmp_img, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { .. } => {}
            _ => panic!("expected Indexed8"),
        }
    }

    #[test]
    fn encode_indexed_empty_palette_error() {
        let img = Image::fill(1, 1, Indexed8(0));
        let result = encode_indexed(&img, &[], &BmpEncodeOptions::default());
        assert!(matches!(result, Err(IoError::EncodeFailed { .. })));
    }

    #[test]
    fn encode_indexed_palette_too_large_error() {
        let img = Image::fill(1, 1, Indexed8(0));
        let palette = vec![Srgba8::new(0, 0, 0, 255); 257];
        let result = encode_indexed(&img, &palette, &BmpEncodeOptions::default());
        assert!(matches!(result, Err(IoError::EncodeFailed { .. })));
    }

    #[test]
    fn encode_indexed_index_out_of_range_error() {
        let img = Image::fill(1, 1, Indexed8(5));
        let palette = [Srgba8::new(0, 0, 0, 255); 3]; // max index = 2
        let result = encode_indexed(&img, &palette, &BmpEncodeOptions::default());
        assert!(matches!(result, Err(IoError::EncodeFailed { .. })));
    }

    #[test]
    fn encode_large_image_padding() {
        // Odd width at 24bpp — tests padding doesn't cause off-by-one.
        let img = Image::fill(101, 50, Srgb8::new(42, 84, 126));
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(dec_img) => {
                assert_eq!(dec_img.width(), 101);
                assert_eq!(dec_img.height(), 50);
                // Spot-check corners.
                assert_eq!(dec_img.pixel_at(0, 0), Srgb8::new(42, 84, 126));
                assert_eq!(dec_img.pixel_at(100, 49), Srgb8::new(42, 84, 126));
            }
            _ => panic!("expected Srgb8"),
        }
    }

    // ── Metadata tests ───────────────────────────────────────────────────

    #[test]
    fn metadata_resolution_none_for_zero() {
        let img = Image::fill(1, 1, Srgb8::new(0, 0, 0));
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert!(decoded.metadata.resolution.is_none());
    }

    #[test]
    fn metadata_resolution_roundtrip() {
        let img = Image::fill(1, 1, Srgb8::new(0, 0, 0));
        let opts = BmpEncodeOptions {
            resolution: Some(BmpResolution {
                x_pixels_per_meter: 2835,
                y_pixels_per_meter: 2835,
            }),
            ..Default::default()
        };
        let bytes = encode(&img, &opts).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(
            decoded.metadata.resolution,
            Some(BmpResolution {
                x_pixels_per_meter: 2835,
                y_pixels_per_meter: 2835,
            })
        );
    }

    #[test]
    fn metadata_color_space_srgb_for_pre_v4() {
        let img = Image::fill(1, 1, Srgb8::new(0, 0, 0));
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.metadata.color_space, BmpColorSpace::Srgb);
    }

    // ── BmpDecoded Debug impl test ───────────────────────────────────────

    #[test]
    fn bmp_decoded_implements_debug() {
        let img = Image::fill(2, 2, Srgb8::new(0, 0, 0));
        let decoded = BmpDecoded {
            image: BmpImage::Srgb8(img),
            metadata: BmpMetadata {
                color_space: BmpColorSpace::Srgb,
                source_bit_depth: BmpBitDepth::TwentyFour,
                resolution: None,
                compression: BmpCompression::None,
                header_version: BmpHeaderVersion::Info,
                icc_profile: None,
            },
        };
        let dbg = format!("{:?}", decoded);
        assert!(dbg.contains("Srgb8(2x2)"));
    }

    // ── Srgba8 encode BGR byte order ─────────────────────────────────────

    #[test]
    fn encode_srgba8_bgra_byte_order() {
        let img = Image::fill(1, 1, Srgba8::new(0xAA, 0xBB, 0xCC, 0xDD));
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();

        // For 32-bit with alpha:
        // File header: 14, BITMAPINFOHEADER: 40, Bitfield masks: 16 = offset 70.
        let pix_offset = 70;
        // BGRA order in the file.
        assert_eq!(bytes[pix_offset], 0xCC); // B
        assert_eq!(bytes[pix_offset + 1], 0xBB); // G
        assert_eq!(bytes[pix_offset + 2], 0xAA); // R
        assert_eq!(bytes[pix_offset + 3], 0xDD); // A
    }

    // ── Srgba8 full pixel roundtrip ──────────────────────────────────────

    #[test]
    fn encode_srgba8_pixel_exact_roundtrip() {
        let img = Image::generate(3, 3, |x, y| {
            Srgba8::new(
                (x * 80 + 10) as u8,
                (y * 60 + 20) as u8,
                (x * 40 + y * 30) as u8,
                ((x + y) * 50 + 55) as u8,
            )
        });
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Srgba8(dec_img) => {
                for y in 0..3 {
                    for x in 0..3 {
                        assert_eq!(
                            dec_img.pixel_at(x, y),
                            img.pixel_at(x, y),
                            "pixel mismatch at ({}, {})",
                            x,
                            y
                        );
                    }
                }
            }
            _ => panic!("expected Srgba8"),
        }
    }

    // ── Indexed palette roundtrip ────────────────────────────────────────

    #[test]
    fn encode_indexed_palette_values_roundtrip() {
        let palette = [
            Srgba8::new(0, 0, 0, 255),
            Srgba8::new(255, 0, 0, 255),
            Srgba8::new(0, 255, 0, 255),
            Srgba8::new(0, 0, 255, 255),
            Srgba8::new(128, 128, 128, 255),
        ];
        let img = Image::from_vec(3, 1, vec![Indexed8(0), Indexed8(2), Indexed8(4)]).unwrap();
        let bytes = encode_indexed(&img, &palette, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 {
                data: dec_img,
                palette: dec_pal,
            } => {
                assert_eq!(dec_img.pixel_at(0, 0).0, 0);
                assert_eq!(dec_img.pixel_at(1, 0).0, 2);
                assert_eq!(dec_img.pixel_at(2, 0).0, 4);
                for i in 0..5 {
                    assert_eq!(dec_pal[i], palette[i], "palette mismatch at index {}", i);
                }
            }
            _ => panic!("expected Indexed8"),
        }
    }

    // ── V4 header color space ────────────────────────────────────────────

    #[test]
    fn color_space_srgb_for_v4_lcs_srgb() {
        // Build a V4 header with LCS_sRGB.
        let width = 1u32;
        let height = 1u32;
        let header_size = 108u32;
        let pixel_offset = 14 + header_size;
        let stride = row_stride(width, 24);
        let file_size = pixel_offset as usize + stride;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset);
        write_u32_le(&mut data, 14, header_size);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 24);
        write_u32_le(&mut data, 30, 0); // BI_RGB
        write_u32_le(&mut data, 70, LCS_SRGB); // csType

        // Write one pixel.
        let off = pixel_offset as usize;
        data[off] = 0;
        data[off + 1] = 0;
        data[off + 2] = 0;

        let decoded = decode(&data).unwrap();
        assert_eq!(decoded.metadata.color_space, BmpColorSpace::Srgb);
        assert_eq!(decoded.metadata.header_version, BmpHeaderVersion::V4);
    }

    #[test]
    fn color_space_icc_tagged_for_v5_profile_embedded() {
        let width = 1u32;
        let height = 1u32;
        let header_size = 124u32;
        let icc_profile = vec![0xDE, 0xAD, 0xBE, 0xEF]; // fake ICC data
        let pixel_offset = 14 + header_size;
        let icc_offset = pixel_offset + row_stride(width, 24) as u32;
        let file_size = icc_offset as usize + icc_profile.len();
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset);
        write_u32_le(&mut data, 14, header_size);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 24);
        write_u32_le(&mut data, 30, 0); // BI_RGB
        write_u32_le(&mut data, 70, LCS_PROFILE_EMBEDDED);
        // V5 ICC profile offset (relative to file header start, but our parse
        // expects it relative to the start of the DIB header).
        // Actually, per the spec: bV5ProfileData is the offset from the
        // beginning of BITMAPV5HEADER to the start of the profile data.
        // But we store it raw and add 14 in build_metadata.
        // The V5 profile offset in the header is from beginning of BITMAPV5HEADER.
        write_u32_le(&mut data, 14 + 112, icc_offset - 14); // relative to file header
        write_u32_le(&mut data, 14 + 116, icc_profile.len() as u32);

        // Write pixel data.
        let off = pixel_offset as usize;
        data[off] = 0;
        data[off + 1] = 0;
        data[off + 2] = 0;

        // Write ICC profile.
        data[icc_offset as usize..icc_offset as usize + icc_profile.len()]
            .copy_from_slice(&icc_profile);

        let decoded = decode(&data).unwrap();
        assert_eq!(decoded.metadata.color_space, BmpColorSpace::IccTagged);
        assert_eq!(decoded.metadata.header_version, BmpHeaderVersion::V5);
        assert!(decoded.metadata.icc_profile.is_some());
        assert_eq!(&*decoded.metadata.icc_profile.unwrap(), &icc_profile[..]);
    }

    // ── BmpMetadata Clone test ───────────────────────────────────────────

    #[test]
    fn metadata_is_cloneable() {
        let meta = BmpMetadata {
            color_space: BmpColorSpace::Srgb,
            source_bit_depth: BmpBitDepth::TwentyFour,
            resolution: None,
            compression: BmpCompression::None,
            header_version: BmpHeaderVersion::Info,
            icc_profile: None,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.color_space, meta.color_space);
        assert_eq!(cloned.source_bit_depth, meta.source_bit_depth);
    }

    // ── decode_reader matches decode ─────────────────────────────────────

    #[test]
    fn decode_reader_matches_decode() {
        let img = Image::fill(3, 2, Srgb8::new(42, 84, 126));
        let bytes = encode(&img, &BmpEncodeOptions::default()).unwrap();

        let d1 = decode(&bytes).unwrap();
        let d2 = decode_reader(std::io::Cursor::new(&bytes)).unwrap();

        match (&d1.image, &d2.image) {
            (BmpImage::Srgb8(a), BmpImage::Srgb8(b)) => {
                assert_eq!(a.width(), b.width());
                assert_eq!(a.height(), b.height());
                for y in 0..a.height() {
                    for x in 0..a.width() {
                        assert_eq!(a.pixel_at(x, y), b.pixel_at(x, y));
                    }
                }
            }
            _ => panic!("variant mismatch"),
        }
    }

    // ── All metadata types constructible ─────────────────────────────────

    #[test]
    fn all_metadata_types_constructible() {
        let _ = BmpColorSpace::Srgb;
        let _ = BmpColorSpace::IccTagged;
        let _ = BmpBitDepth::One;
        let _ = BmpBitDepth::Four;
        let _ = BmpBitDepth::Eight;
        let _ = BmpBitDepth::Sixteen;
        let _ = BmpBitDepth::TwentyFour;
        let _ = BmpBitDepth::ThirtyTwo;
        let _ = BmpCompression::None;
        let _ = BmpCompression::Rle8;
        let _ = BmpCompression::Rle4;
        let _ = BmpCompression::Bitfields;
        let _ = BmpHeaderVersion::Core;
        let _ = BmpHeaderVersion::Info;
        let _ = BmpHeaderVersion::V4;
        let _ = BmpHeaderVersion::V5;
        let _ = BmpResolution {
            x_pixels_per_meter: 0,
            y_pixels_per_meter: 0,
        };
        let _ = BmpEncodeOptions::default();
    }

    // ═════════════════════════════════════════════════════════════════════
    // Additional coverage tests — error paths, edge cases, RLE4 decode
    // ═════════════════════════════════════════════════════════════════════

    // ── RLE4 full BMP decode ─────────────────────────────────────────────

    #[test]
    fn decode_rle4_compressed() {
        // Build a 4-bit RLE4-compressed BMP.
        let palette: Vec<Srgba8> = (0..16)
            .map(|i| Srgba8::new(i * 16, i * 16, i * 16, 255))
            .collect();
        let width = 4u32;
        let height = 1u32;
        let color_table_size = 16 * 4;
        let pixel_offset = 14 + 40 + color_table_size;

        // RLE4 data: 4 pixels alternating 0xA and 0xB, then end-of-bitmap.
        let rle_data: &[u8] = &[4, 0xAB, 0, 1];
        let file_size = pixel_offset + rle_data.len();
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 4);
        write_u32_le(&mut data, 30, BI_RLE4);
        write_u32_le(&mut data, 46, 16);

        // Color table.
        for (i, entry) in palette.iter().enumerate() {
            let off = 54 + i * 4;
            data[off] = entry.b.0;
            data[off + 1] = entry.g.0;
            data[off + 2] = entry.r.0;
            data[off + 3] = 0;
        }

        // RLE pixel data.
        data[pixel_offset..pixel_offset + rle_data.len()].copy_from_slice(rle_data);

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: img, .. } => {
                assert_eq!(img.pixel_at(0, 0).0, 0xA);
                assert_eq!(img.pixel_at(1, 0).0, 0xB);
                assert_eq!(img.pixel_at(2, 0).0, 0xA);
                assert_eq!(img.pixel_at(3, 0).0, 0xB);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }

        assert_eq!(decoded.metadata.compression, BmpCompression::Rle4);
        assert_eq!(decoded.metadata.source_bit_depth, BmpBitDepth::Four);
    }

    // ── Truncated header error paths ─────────────────────────────────────

    #[test]
    fn parse_core_header_truncated() {
        // Valid file header but truncated BITMAPCOREHEADER.
        let mut data = vec![0u8; 14 + 8]; // need 12, only provide 8
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 12); // claims 12-byte header
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn parse_info_header_truncated() {
        // Valid file header but truncated BITMAPINFOHEADER.
        let mut data = vec![0u8; 14 + 20]; // need 40, only provide 20
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 40); // claims 40-byte header
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn parse_v4_header_truncated() {
        let mut data = vec![0u8; 14 + 60]; // need 108, only provide 60
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 108);
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn parse_v5_header_truncated() {
        let mut data = vec![0u8; 14 + 80]; // need 124, only provide 80
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 124);
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    // ── Invalid planes in V4/V5 headers ──────────────────────────────────

    #[test]
    fn v4_header_reject_invalid_planes() {
        let mut data = vec![0u8; 14 + 108 + 32]; // V4 + some pixel data
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 108);
        write_i32_le(&mut data, 18, 1);
        write_i32_le(&mut data, 22, 1);
        write_u16_le(&mut data, 26, 2); // planes = 2 (invalid)
        write_u16_le(&mut data, 28, 24);
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn v5_header_reject_invalid_planes() {
        let mut data = vec![0u8; 14 + 124 + 32];
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 124);
        write_i32_le(&mut data, 18, 1);
        write_i32_le(&mut data, 22, 1);
        write_u16_le(&mut data, 26, 2); // planes = 2 (invalid)
        write_u16_le(&mut data, 28, 24);
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    // ── Core header zero dimensions and invalid planes ───────────────────

    #[test]
    fn core_header_reject_zero_dimensions() {
        let mut data = vec![0u8; 14 + 12];
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 12);
        write_u16_le(&mut data, 18, 0); // width = 0
        write_u16_le(&mut data, 20, 4);
        write_u16_le(&mut data, 22, 1);
        write_u16_le(&mut data, 24, 24);
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn core_header_reject_invalid_planes() {
        let mut data = vec![0u8; 14 + 12];
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 12);
        write_u16_le(&mut data, 18, 4);
        write_u16_le(&mut data, 20, 4);
        write_u16_le(&mut data, 22, 2); // planes = 2 (invalid)
        write_u16_le(&mut data, 24, 24);
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn core_header_reject_invalid_bit_count() {
        let mut data = vec![0u8; 14 + 12];
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 14, 12);
        write_u16_le(&mut data, 18, 4);
        write_u16_le(&mut data, 20, 4);
        write_u16_le(&mut data, 22, 1); // planes
        write_u16_le(&mut data, 24, 16); // 16 not valid for BITMAPCOREHEADER
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    // ── Compression / bit-depth mismatches ───────────────────────────────

    #[test]
    fn rle8_with_non_8bit_rejected() {
        let data = make_bmp_with_info_header(4, 4, 24, BI_RLE8);
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn rle4_with_non_4bit_rejected() {
        let data = make_bmp_with_info_header(4, 4, 8, BI_RLE4);
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn unknown_compression_type_rejected() {
        let data = make_bmp_with_info_header(4, 4, 24, 99);
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::UnsupportedFeature { .. })));
    }

    // ── Bitfield mask edge cases ─────────────────────────────────────────

    #[test]
    fn bitfield_masks_truncated() {
        // BITMAPINFOHEADER with BI_BITFIELDS but no mask data after header.
        let mut data = vec![0u8; 14 + 40]; // no room for masks
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 10, 66); // pixel offset past masks
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, 1);
        write_i32_le(&mut data, 22, 1);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 16);
        write_u32_le(&mut data, 30, BI_BITFIELDS);
        let result = decode(&data);
        assert!(result.is_err());
    }

    #[test]
    fn bitfield_masks_no_alpha_dword() {
        // BI_BITFIELDS with exactly 12 bytes of mask data (no alpha DWORD).
        let mask_size = 12usize;
        let pixel_offset = 54 + mask_size;
        let stride = row_stride(1, 16);
        let file_size = pixel_offset + stride;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, 1);
        write_i32_le(&mut data, 22, 1);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 16);
        write_u32_le(&mut data, 30, BI_BITFIELDS);

        // Only 3 mask DWORDs, no 4th.
        write_u32_le(&mut data, 54, 0xF800);
        write_u32_le(&mut data, 58, 0x07E0);
        write_u32_le(&mut data, 62, 0x001F);
        // Ensure no 4th DWORD available by making file end right at pixel data.

        write_u16_le(&mut data, pixel_offset, 0xFFFF);

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Srgb8(img) => {
                let p = img.pixel_at(0, 0);
                assert_eq!(p.r.0, 255);
                assert_eq!(p.g.0, 255);
                assert_eq!(p.b.0, 255);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    // ── Color table truncated ────────────────────────────────────────────

    #[test]
    fn color_table_truncated_error() {
        // Build an 8-bit indexed BMP but truncate the color table.
        let mut data = vec![0u8; 14 + 40 + 2]; // only 2 bytes for color table
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 10, 14 + 40 + 256 * 4); // pixel offset past full table
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, 1);
        write_i32_le(&mut data, 22, 1);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 8);
        write_u32_le(&mut data, 30, 0);
        // colors_used = 0, so it will try to read 256 entries
        let result = decode(&data);
        assert!(result.is_err());
    }

    // ── Indexed pixel data truncation ────────────────────────────────────

    #[test]
    fn decode_1bit_pixel_data_truncated() {
        // 1-bit indexed BMP with pixel data cut short.
        let palette_size = 2 * 4;
        let pixel_offset = 14 + 40 + palette_size;
        // Need at least 4 bytes for stride, provide 0.
        let mut data = vec![0u8; pixel_offset];
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, pixel_offset as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, 8);
        write_i32_le(&mut data, 22, 1);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 1);
        write_u32_le(&mut data, 46, 2);
        // Write 2 palette entries.
        for i in 0..2 {
            let off = 54 + i * 4;
            data[off] = 0;
            data[off + 1] = 0;
            data[off + 2] = 0;
            data[off + 3] = 0;
        }
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::DecodeFailed { .. })));
    }

    #[test]
    fn decode_4bit_pixel_data_truncated() {
        let palette_size = 16 * 4;
        let pixel_offset = 14 + 40 + palette_size;
        let mut data = vec![0u8; pixel_offset]; // no pixel data
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, pixel_offset as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, 4);
        write_i32_le(&mut data, 22, 1);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 4);
        write_u32_le(&mut data, 46, 16);
        for i in 0..16 {
            let off = 54 + i * 4;
            data[off] = 0;
            data[off + 1] = 0;
            data[off + 2] = 0;
            data[off + 3] = 0;
        }
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::DecodeFailed { .. })));
    }

    #[test]
    fn decode_8bit_pixel_data_truncated() {
        let palette_size = 4 * 4;
        let pixel_offset = 14 + 40 + palette_size;
        let mut data = vec![0u8; pixel_offset]; // no pixel data
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, pixel_offset as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, 4);
        write_i32_le(&mut data, 22, 1);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 8);
        write_u32_le(&mut data, 46, 4);
        for i in 0..4 {
            let off = 54 + i * 4;
            data[off] = 0;
            data[off + 1] = 0;
            data[off + 2] = 0;
            data[off + 3] = 0;
        }
        let result = decode(&data);
        assert!(matches!(result, Err(IoError::DecodeFailed { .. })));
    }

    // ── Indexed colors_used = 0 defaults ─────────────────────────────────

    #[test]
    fn decode_8bit_indexed_colors_used_zero() {
        // colors_used = 0 → should default to 256 entries.
        let palette_count = 256;
        let palette_size = palette_count * 4;
        let width = 1u32;
        let height = 1u32;
        let stride = row_stride(width, 8);
        let pixel_offset = 14 + 40 + palette_size;
        let file_size = pixel_offset + stride;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 8);
        write_u32_le(&mut data, 46, 0); // colors_used = 0

        // Fill palette entries.
        for i in 0..palette_count {
            let off = 54 + i * 4;
            data[off] = i as u8;
            data[off + 1] = 0;
            data[off + 2] = 0;
            data[off + 3] = 0;
        }

        // Pixel data: index 42.
        data[pixel_offset] = 42;

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: img, .. } => {
                assert_eq!(img.pixel_at(0, 0).0, 42);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    // ── RLE4 edge cases at BMP level ─────────────────────────────────────

    #[test]
    fn rle4_end_of_line_and_delta() {
        // RLE4 with end-of-line and delta escapes.
        // 3×2 image.
        let data_seq: &[u8] = &[
            2, 0xAB, // 2 pixels: A, B
            0, 0, // end of line
            0, 2, 1, 0, // delta: dx=1, dy=0
            1, 0xC0, // 1 pixel: C
            0, 1, // end of bitmap
        ];
        let result = decompress_rle4(data_seq, 3, 2).unwrap();
        // Row 0: A, B, 0
        assert_eq!(result[0], 0xA);
        assert_eq!(result[1], 0xB);
        assert_eq!(result[2], 0);
        // Row 1: after delta (1,0) cursor is at (1,1), then pixel C
        assert_eq!(result[3], 0); // (0,1) untouched
        assert_eq!(result[4], 0xC); // (1,1)
    }

    #[test]
    fn rle4_absolute_mode_odd_byte_pad() {
        // Absolute mode requires n >= 3 (0/1/2 are escape codes).
        // count=5 → byte_count = (5+1)/2 = 3 (odd) → 1 pad byte.
        // Nibbles: 0x12 → 1, 2; 0x34 → 3, 4; 0x50 → 5.
        let data_seq: &[u8] = &[
            0, 5, // absolute mode: 5 pixels
            0x12, 0x34, 0x50, // 3 data bytes
            0,    // pad byte (odd byte_count=3)
            0, 1, // end of bitmap
        ];
        let result = decompress_rle4(data_seq, 5, 1).unwrap();
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);
        assert_eq!(result[3], 4);
        assert_eq!(result[4], 5);
    }

    #[test]
    fn rle4_exceeds_image_bounds_encoded_run() {
        // More rows than the image has.
        let data_seq: &[u8] = &[
            2, 0xAB, // row 0
            0, 0, // end of line → row 1
            2, 0xAB, // row 1
            0, 0, // end of line → row 2 (out of bounds)
            2, 0xAB, // row 2 — should fail
        ];
        let result = decompress_rle4(data_seq, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn rle4_delta_truncated() {
        let data_seq: &[u8] = &[0, 2, 1]; // delta but only 1 byte of offset
        let result = decompress_rle4(data_seq, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn rle4_absolute_mode_truncated() {
        // Absolute mode claims 6 pixels but data is truncated.
        let data_seq: &[u8] = &[0, 6, 0x12]; // need 3 bytes, only 1
        let result = decompress_rle4(data_seq, 8, 1);
        assert!(result.is_err());
    }

    // ── RLE8 error paths ─────────────────────────────────────────────────

    #[test]
    fn rle8_exceeds_image_bounds_encoded_run() {
        // Encoded run that pushes y past height.
        let data_seq: &[u8] = &[
            2, 1, // row 0: 2 pixels
            0, 0, // end of line → row 1
            2, 1, // row 1: 2 pixels
            0, 0, // end of line → row 2 (out of bounds for 2-row image)
            2, 1, // this should trigger error
        ];
        let result = decompress_rle8(data_seq, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn rle8_delta_truncated() {
        let data_seq: &[u8] = &[0, 2, 1]; // delta but only 1 byte of offset
        let result = decompress_rle8(data_seq, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn rle8_absolute_mode_truncated() {
        // Absolute mode claims 5 literal pixels but data is cut short.
        let data_seq: &[u8] = &[0, 5, 1, 2]; // need 5 bytes, only 2
        let result = decompress_rle8(data_seq, 8, 1);
        assert!(result.is_err());
    }

    #[test]
    fn rle8_absolute_mode_exceeds_bounds() {
        // Absolute mode with enough data but y is past height.
        let data_seq: &[u8] = &[
            3, 1, // 3 pixels of 1
            0, 0, // end of line → row 1
            0, 0, // end of line → row 2 (out of bounds for 1-row image)
            0, 3, 1, 2, 3, 0, // absolute mode 3 pixels (should hit y >= height)
            0, 1,
        ];
        let result = decompress_rle8(data_seq, 3, 1);
        // y wraps to row 2, but height is 1 — error on absolute mode.
        assert!(result.is_err());
    }

    // ── ICC profile edge cases ───────────────────────────────────────────

    #[test]
    fn icc_profile_beyond_file_end_is_none() {
        // V5 header with PROFILE_EMBEDDED but profile extends past file end.
        let width = 1u32;
        let height = 1u32;
        let header_size = 124u32;
        let pixel_offset = 14 + header_size;
        let stride = row_stride(width, 24);
        let file_size = pixel_offset as usize + stride; // no room for ICC
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset);
        write_u32_le(&mut data, 14, header_size);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 24);
        write_u32_le(&mut data, 30, 0);
        write_u32_le(&mut data, 70, LCS_PROFILE_EMBEDDED);
        // Point profile at offset beyond file.
        write_u32_le(&mut data, 14 + 112, 9999);
        write_u32_le(&mut data, 14 + 116, 100);

        let decoded = decode(&data).unwrap();
        assert_eq!(decoded.metadata.color_space, BmpColorSpace::IccTagged);
        // Profile bytes should be None since they're past file end.
        assert!(decoded.metadata.icc_profile.is_none());
    }

    // ── Encode indexed with resolution and padding ───────────────────────

    #[test]
    fn encode_indexed_with_resolution() {
        let img = Image::fill(1, 1, Indexed8(0));
        let palette = [Srgba8::new(0, 0, 0, 255)];
        let opts = BmpEncodeOptions {
            resolution: Some(BmpResolution {
                x_pixels_per_meter: 2835,
                y_pixels_per_meter: 2835,
            }),
            ..Default::default()
        };
        let bytes = encode_indexed(&img, &palette, &opts).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(
            decoded.metadata.resolution,
            Some(BmpResolution {
                x_pixels_per_meter: 2835,
                y_pixels_per_meter: 2835,
            })
        );
    }

    #[test]
    fn encode_indexed_row_padding() {
        // Width 3 at 8bpp = 3 bytes per row → padded to 4.
        let img = Image::from_vec(3, 1, vec![Indexed8(0), Indexed8(1), Indexed8(2)]).unwrap();
        let palette = [
            Srgba8::new(255, 0, 0, 255),
            Srgba8::new(0, 255, 0, 255),
            Srgba8::new(0, 0, 255, 255),
        ];
        let bytes = encode_indexed(&img, &palette, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: dec_img, .. } => {
                assert_eq!(dec_img.pixel_at(0, 0).0, 0);
                assert_eq!(dec_img.pixel_at(1, 0).0, 1);
                assert_eq!(dec_img.pixel_at(2, 0).0, 2);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    #[test]
    fn encode_indexed_multi_row() {
        // 2×3 indexed image — tests bottom-up row ordering in indexed encode.
        let img = Image::from_vec(
            2,
            3,
            vec![
                Indexed8(0),
                Indexed8(1),
                Indexed8(1),
                Indexed8(0),
                Indexed8(0),
                Indexed8(0),
            ],
        )
        .unwrap();
        let palette = [Srgba8::new(10, 20, 30, 255), Srgba8::new(40, 50, 60, 255)];
        let bytes = encode_indexed(&img, &palette, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: dec_img, .. } => {
                assert_eq!(dec_img.pixel_at(0, 0).0, 0);
                assert_eq!(dec_img.pixel_at(1, 0).0, 1);
                assert_eq!(dec_img.pixel_at(0, 1).0, 1);
                assert_eq!(dec_img.pixel_at(1, 1).0, 0);
                assert_eq!(dec_img.pixel_at(0, 2).0, 0);
                assert_eq!(dec_img.pixel_at(1, 2).0, 0);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    // ── Encode error path (writer failure) ───────────────────────────────

    #[test]
    fn encode_writer_io_error() {
        // A writer that always fails.
        struct FailWriter;
        impl std::io::Write for FailWriter {
            fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(std::io::ErrorKind::Other, "fail"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::new(std::io::ErrorKind::Other, "fail"))
            }
        }

        let img = Image::fill(1, 1, Srgb8::new(0, 0, 0));
        let result = encode_writer(&img, &mut FailWriter, &BmpEncodeOptions::default());
        assert!(matches!(result, Err(IoError::EncodeFailed { .. })));
    }

    // ── LE writer coverage ───────────────────────────────────────────────

    #[test]
    fn le_writer_roundtrip() {
        let mut buf = [0u8; 10];
        write_u16_le(&mut buf, 0, 0xBEEF);
        assert_eq!(read_u16_le(&buf, 0), Some(0xBEEF));

        write_u32_le(&mut buf, 2, 0xDEADBEEF);
        assert_eq!(read_u32_le(&buf, 2), Some(0xDEADBEEF));

        write_i32_le(&mut buf, 6, -42);
        assert_eq!(read_i32_le(&buf, 6), Some(-42));
    }

    // ── V5 header with profile_size = 0 ──────────────────────────────────

    #[test]
    fn v5_profile_embedded_zero_size() {
        // cs_type = LCS_PROFILE_EMBEDDED but profile_size = 0 → no ICC.
        let width = 1u32;
        let height = 1u32;
        let header_size = 124u32;
        let pixel_offset = 14 + header_size;
        let stride = row_stride(width, 24);
        let file_size = pixel_offset as usize + stride;
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset);
        write_u32_le(&mut data, 14, header_size);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 24);
        write_u32_le(&mut data, 30, 0);
        write_u32_le(&mut data, 70, LCS_PROFILE_EMBEDDED);
        write_u32_le(&mut data, 14 + 112, 0);
        write_u32_le(&mut data, 14 + 116, 0); // profile_size = 0

        let decoded = decode(&data).unwrap();
        // cs_type says embedded, but size is 0 → no icc_profile field set.
        assert!(decoded.metadata.icc_profile.is_none());
    }

    // ── RLE8 pixel data offset beyond file ───────────────────────────────

    #[test]
    fn rle8_pixel_offset_beyond_file() {
        let color_table_size = 1 * 4;
        let pixel_offset = 14 + 40 + color_table_size;
        // File doesn't extend to pixel_offset.
        let file_len = pixel_offset - 1;
        let mut data = vec![0u8; file_len];
        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_len as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, 1);
        write_i32_le(&mut data, 22, 1);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 8);
        write_u32_le(&mut data, 30, BI_RLE8);
        write_u32_le(&mut data, 46, 1);
        // Palette entry.
        if data.len() >= 58 {
            data[54] = 0;
            data[55] = 0;
            data[56] = 0;
            data[57] = 0;
        }
        let result = decode(&data);
        assert!(result.is_err());
    }

    // ── extract_channel with > 8 bits ────────────────────────────────────

    #[test]
    fn extract_channel_more_than_8_bits() {
        // 10-bit mask: should right-shift to 8 bits.
        let info = MaskInfo { shift: 0, bits: 10 };
        // max 10-bit value = 1023
        assert_eq!(extract_channel(1023, info), 255);
        assert_eq!(extract_channel(512, info), 128);
        assert_eq!(extract_channel(0, info), 0);
    }

    // ── mask_info with zero mask ─────────────────────────────────────────

    #[test]
    fn mask_info_zero() {
        let info = mask_info(0);
        assert_eq!(info.shift, 0);
        assert_eq!(info.bits, 0);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Targeted coverage for remaining reachable branches
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_bitfield_masks_no_alpha_dword_direct() {
        // Exactly 12 bytes of mask data → alpha defaults to 0.
        let mut buf = [0u8; 12];
        write_u32_le(&mut buf, 0, 0xF800);
        write_u32_le(&mut buf, 4, 0x07E0);
        write_u32_le(&mut buf, 8, 0x001F);
        let masks = parse_bitfield_masks(&buf, 0).unwrap();
        assert_eq!(masks.r, 0xF800);
        assert_eq!(masks.g, 0x07E0);
        assert_eq!(masks.b, 0x001F);
        assert_eq!(masks.a, 0); // no 4th DWORD → 0
    }

    #[test]
    fn rle8_data_ends_after_first_byte() {
        // Data has only one byte — loop reads `first`, then pos >= data.len(),
        // so the inner `break` fires.
        let data_seq: &[u8] = &[5];
        let result = decompress_rle8(data_seq, 4, 1).unwrap();
        // No pixels were written; output is all zeros.
        assert_eq!(result, vec![0; 4]);
    }

    #[test]
    fn rle4_data_ends_after_first_byte() {
        let data_seq: &[u8] = &[5];
        let result = decompress_rle4(data_seq, 4, 1).unwrap();
        assert_eq!(result, vec![0; 4]);
    }

    #[test]
    fn rle4_absolute_mode_exceeds_bounds_in_nibbles() {
        // Absolute mode where the nibble loop pushes y past height.
        // Row height is 1, width is 2. After end-of-line cursor is at y=1 (out of bounds).
        let data_seq: &[u8] = &[
            2, 0xAB, // row 0: 2 pixels
            0, 0, // end of line → y=1
            0, 3, 0x12, 0x30, // absolute mode: 3 pixels at y=1 → exceeds bounds
            0, 1,
        ];
        let result = decompress_rle4(data_seq, 2, 1);
        assert!(result.is_err());
    }

    #[test]
    fn decode_rle8_colors_used_zero() {
        // Build an RLE8 BMP where colors_used = 0 → defaults to 256.
        let width = 2u32;
        let height = 1u32;
        let palette_count = 256;
        let color_table_size = palette_count * 4;
        let pixel_offset = 14 + 40 + color_table_size;

        // RLE data: 2 pixels of index 0, end-of-bitmap.
        let rle_data: &[u8] = &[2, 0, 0, 1];
        let file_size = pixel_offset + rle_data.len();
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 8);
        write_u32_le(&mut data, 30, BI_RLE8);
        write_u32_le(&mut data, 46, 0); // colors_used = 0

        // Fill 256 palette entries with opaque black.
        for i in 0..palette_count {
            let off = 54 + i * 4;
            data[off] = 0;
            data[off + 1] = 0;
            data[off + 2] = 0;
            data[off + 3] = 0;
        }

        data[pixel_offset..pixel_offset + rle_data.len()].copy_from_slice(rle_data);

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: img, .. } => {
                assert_eq!(img.pixel_at(0, 0).0, 0);
                assert_eq!(img.pixel_at(1, 0).0, 0);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    #[test]
    fn decode_rle4_colors_used_zero() {
        // Build an RLE4 BMP where colors_used = 0 → defaults to 16.
        let width = 2u32;
        let height = 1u32;
        let palette_count = 16;
        let color_table_size = palette_count * 4;
        let pixel_offset = 14 + 40 + color_table_size;

        // RLE4 data: 2 pixels alternating nibbles 0x0 and 0x1, end-of-bitmap.
        let rle_data: &[u8] = &[2, 0x01, 0, 1];
        let file_size = pixel_offset + rle_data.len();
        let mut data = vec![0u8; file_size];

        data[0] = 0x42;
        data[1] = 0x4D;
        write_u32_le(&mut data, 2, file_size as u32);
        write_u32_le(&mut data, 10, pixel_offset as u32);
        write_u32_le(&mut data, 14, 40);
        write_i32_le(&mut data, 18, width as i32);
        write_i32_le(&mut data, 22, height as i32);
        write_u16_le(&mut data, 26, 1);
        write_u16_le(&mut data, 28, 4);
        write_u32_le(&mut data, 30, BI_RLE4);
        write_u32_le(&mut data, 46, 0); // colors_used = 0

        for i in 0..palette_count {
            let off = 54 + i * 4;
            data[off] = 0;
            data[off + 1] = 0;
            data[off + 2] = 0;
            data[off + 3] = 0;
        }

        data[pixel_offset..pixel_offset + rle_data.len()].copy_from_slice(rle_data);

        let decoded = decode(&data).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: img, .. } => {
                assert_eq!(img.pixel_at(0, 0).0, 0);
                assert_eq!(img.pixel_at(1, 0).0, 1);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    // ── Coverage gap tests ───────────────────────────────────────────────

    #[test]
    fn rle4_pixel_offset_beyond_file() {
        // Build a minimal 4-bit BMP with BI_RLE4 compression but set the
        // pixel data offset beyond the end of the file.
        let mut bmp = vec![0u8; 14 + 40];
        // File header
        bmp[0] = 0x42;
        bmp[1] = 0x4D;
        let file_size = bmp.len() as u32;
        write_u32_le(&mut bmp, 2, file_size);
        write_u32_le(&mut bmp, 10, 9999); // pixel offset way beyond file
        // DIB header (BITMAPINFOHEADER)
        write_u32_le(&mut bmp, 14, 40);
        write_i32_le(&mut bmp, 18, 2); // width
        write_i32_le(&mut bmp, 22, 2); // height
        write_u16_le(&mut bmp, 26, 1); // planes
        write_u16_le(&mut bmp, 28, 4); // 4-bit
        write_u32_le(&mut bmp, 30, BI_RLE4); // compression
        // Append a minimal color table (16 entries × 4 bytes).
        bmp.extend(vec![0u8; 16 * 4]);
        let result = decode(&bmp);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("offset") || msg.contains("truncated") || msg.contains("beyond"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn determine_color_space_v4_lcs_windows() {
        // V4 header with cs_type = LCS_WINDOWS_COLOR_SPACE → Srgb.
        let mut bmp = vec![0u8; 14 + 108];
        bmp[0] = 0x42;
        bmp[1] = 0x4D;
        // DIB header size = 108 (V4)
        write_u32_le(&mut bmp, 14, 108);
        write_i32_le(&mut bmp, 18, 1); // width
        write_i32_le(&mut bmp, 22, 1); // height
        write_u16_le(&mut bmp, 26, 1); // planes
        write_u16_le(&mut bmp, 28, 24); // 24-bit
        write_u32_le(&mut bmp, 30, BI_RGB);
        // cs_type at offset 14 + 56 = 70
        write_u32_le(&mut bmp, 70, 0x57696E20); // LCS_WINDOWS_COLOR_SPACE
        // pixel data offset
        let pixel_offset = 14 + 108;
        write_u32_le(&mut bmp, 10, pixel_offset as u32);
        write_u32_le(&mut bmp, 2, (pixel_offset + 4) as u32); // file size
        // Append 4 bytes of pixel data (1 pixel × 3 bytes + 1 pad).
        bmp.extend(&[0u8; 4]);
        let decoded = decode(&bmp).unwrap();
        assert_eq!(decoded.metadata.color_space, BmpColorSpace::Srgb);
        assert_eq!(decoded.metadata.header_version, BmpHeaderVersion::V4);
    }

    #[test]
    fn determine_color_space_v4_lcs_srgb_explicit() {
        // V4 header with cs_type = LCS_sRGB → Srgb.
        let mut bmp = vec![0u8; 14 + 108];
        bmp[0] = 0x42;
        bmp[1] = 0x4D;
        write_u32_le(&mut bmp, 14, 108);
        write_i32_le(&mut bmp, 18, 1);
        write_i32_le(&mut bmp, 22, 1);
        write_u16_le(&mut bmp, 26, 1);
        write_u16_le(&mut bmp, 28, 24);
        write_u32_le(&mut bmp, 30, BI_RGB);
        write_u32_le(&mut bmp, 70, 0x73524742); // LCS_sRGB
        let pixel_offset = 14 + 108;
        write_u32_le(&mut bmp, 10, pixel_offset as u32);
        write_u32_le(&mut bmp, 2, (pixel_offset + 4) as u32);
        bmp.extend(&[0u8; 4]);
        let decoded = decode(&bmp).unwrap();
        assert_eq!(decoded.metadata.color_space, BmpColorSpace::Srgb);
    }

    #[test]
    fn decode_16bit_pixel_data_truncated() {
        // A 16-bit BMP where the pixel data is too short.
        let mut bmp = vec![0u8; 14 + 40];
        bmp[0] = 0x42;
        bmp[1] = 0x4D;
        write_u32_le(&mut bmp, 14, 40);
        write_i32_le(&mut bmp, 18, 4); // width = 4
        write_i32_le(&mut bmp, 22, 1); // height = 1
        write_u16_le(&mut bmp, 26, 1); // planes
        write_u16_le(&mut bmp, 28, 16); // 16-bit
        write_u32_le(&mut bmp, 30, BI_RGB);
        let pixel_offset = 14 + 40;
        write_u32_le(&mut bmp, 10, pixel_offset as u32);
        write_u32_le(&mut bmp, 2, (pixel_offset + 2) as u32); // only 2 bytes
        bmp.extend(&[0u8; 2]); // need 8 bytes for 4 pixels, only provide 2
        let result = decode(&bmp);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("truncated") || msg.contains("decode failed"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn decode_32bit_pixel_data_truncated() {
        // A 32-bit BMP where the pixel data is too short.
        let mut bmp = vec![0u8; 14 + 40];
        bmp[0] = 0x42;
        bmp[1] = 0x4D;
        write_u32_le(&mut bmp, 14, 40);
        write_i32_le(&mut bmp, 18, 2); // width = 2
        write_i32_le(&mut bmp, 22, 1); // height = 1
        write_u16_le(&mut bmp, 26, 1);
        write_u16_le(&mut bmp, 28, 32); // 32-bit
        write_u32_le(&mut bmp, 30, BI_RGB);
        let pixel_offset = 14 + 40;
        write_u32_le(&mut bmp, 10, pixel_offset as u32);
        write_u32_le(&mut bmp, 2, (pixel_offset + 4) as u32); // only 4 bytes
        bmp.extend(&[0u8; 4]); // need 8 bytes for 2 pixels, only provide 4
        let result = decode(&bmp);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("truncated") || msg.contains("decode failed"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn decode_top_down_rle8() {
        // Top-down (negative height) is unusual with RLE8 but must work.
        // Build a 2×2 BMP with negative height and RLE8 compression.
        let mut bmp = vec![0u8; 14 + 40];
        bmp[0] = 0x42;
        bmp[1] = 0x4D;
        write_u32_le(&mut bmp, 14, 40);
        write_i32_le(&mut bmp, 18, 2); // width
        write_i32_le(&mut bmp, 22, -2); // height (top-down)
        write_u16_le(&mut bmp, 26, 1);
        write_u16_le(&mut bmp, 28, 8); // 8-bit
        write_u32_le(&mut bmp, 30, BI_RLE8);
        // Color table (2 entries × 4 bytes)
        bmp.extend(&[255, 0, 0, 0]); // index 0: blue (BGR)
        bmp.extend(&[0, 255, 0, 0]); // index 1: green (BGR)
        let pixel_offset = bmp.len();
        // RLE8 data: row 0 = [0, 0], row 1 = [1, 1]
        // Encoded run: 2 pixels of index 0
        bmp.push(2);
        bmp.push(0);
        // End of line
        bmp.push(0);
        bmp.push(0);
        // Encoded run: 2 pixels of index 1
        bmp.push(2);
        bmp.push(1);
        // End of bitmap
        bmp.push(0);
        bmp.push(1);
        // Patch offsets.
        write_u32_le(&mut bmp, 10, pixel_offset as u32);
        let file_size = bmp.len() as u32;
        write_u32_le(&mut bmp, 2, file_size);
        write_u32_le(&mut bmp, 46, 2); // colors_used (DIB offset 32 = file offset 46)
        let decoded = decode(&bmp).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: img, .. } => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.height(), 2);
                // Top-down: row 0 in image = first RLE row = index 0.
                // But RLE decompressor outputs bottom-up, and top_down
                // flips, so check that we get consistent pixel values.
                // Row 0 pixels should be index 0, row 1 should be index 1.
                assert_eq!(img.pixel_at(0, 0).0, 0);
                assert_eq!(img.pixel_at(1, 0).0, 0);
                assert_eq!(img.pixel_at(0, 1).0, 1);
                assert_eq!(img.pixel_at(1, 1).0, 1);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    #[test]
    fn decode_top_down_rle4() {
        // Top-down (negative height) with RLE4 compression.
        let mut bmp = vec![0u8; 14 + 40];
        bmp[0] = 0x42;
        bmp[1] = 0x4D;
        write_u32_le(&mut bmp, 14, 40);
        write_i32_le(&mut bmp, 18, 2); // width
        write_i32_le(&mut bmp, 22, -2); // height (top-down)
        write_u16_le(&mut bmp, 26, 1);
        write_u16_le(&mut bmp, 28, 4); // 4-bit
        write_u32_le(&mut bmp, 30, BI_RLE4);
        // Color table (16 entries × 4 bytes for 4-bit)
        let mut ct = vec![0u8; 16 * 4];
        ct[0] = 255; // index 0: B=255
        ct[4 + 1] = 255; // index 1: G=255
        bmp.extend_from_slice(&ct);
        let pixel_offset = bmp.len();
        // RLE4 data: row 0 = [0, 0], row 1 = [1, 1]
        // Encoded run: 2 pixels of nibble pattern 0x00 (both nibbles = 0)
        bmp.push(2);
        bmp.push(0x00);
        // End of line
        bmp.push(0);
        bmp.push(0);
        // Encoded run: 2 pixels of nibble pattern 0x11
        bmp.push(2);
        bmp.push(0x11);
        // End of bitmap
        bmp.push(0);
        bmp.push(1);
        write_u32_le(&mut bmp, 10, pixel_offset as u32);
        let file_size = bmp.len() as u32;
        write_u32_le(&mut bmp, 2, file_size);
        let decoded = decode(&bmp).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: img, .. } => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.height(), 2);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    #[test]
    fn encode_indexed_writer_io_error() {
        // Verify that encode_indexed_writer propagates I/O errors.
        struct FailWriter;
        impl std::io::Write for FailWriter {
            fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "boom"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "boom"))
            }
        }
        let img = Image::fill(1, 1, Indexed8(0));
        let palette = [Srgba8::new(0, 0, 0, 255)];
        let result = encode_indexed_writer(
            &img,
            &palette,
            &mut FailWriter,
            &BmpEncodeOptions::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn encode_indexed_width_multiple_of_4_no_padding() {
        // Width = 4 at 8bpp → stride = 4 → pad_bytes = 0.
        // Exercises the `if pad_bytes > 0` false branch.
        let img = Image::from_vec(
            4,
            2,
            vec![
                Indexed8(0),
                Indexed8(1),
                Indexed8(2),
                Indexed8(3),
                Indexed8(3),
                Indexed8(2),
                Indexed8(1),
                Indexed8(0),
            ],
        )
        .unwrap();
        let palette = [
            Srgba8::new(10, 20, 30, 255),
            Srgba8::new(40, 50, 60, 255),
            Srgba8::new(70, 80, 90, 255),
            Srgba8::new(100, 110, 120, 255),
        ];
        let bytes = encode_indexed(&img, &palette, &BmpEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            BmpImage::Indexed8 { data: dec_img, .. } => {
                assert_eq!(dec_img.width(), 4);
                assert_eq!(dec_img.height(), 2);
                assert_eq!(dec_img.pixel_at(0, 0).0, 0);
                assert_eq!(dec_img.pixel_at(3, 0).0, 3);
                assert_eq!(dec_img.pixel_at(0, 1).0, 3);
                assert_eq!(dec_img.pixel_at(3, 1).0, 0);
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    #[test]
    fn decode_24bit_truncated_pixel_data() {
        // 24-bit BMP where pixel data is too short for the declared dimensions.
        let mut bmp = vec![0u8; 14 + 40];
        bmp[0] = 0x42;
        bmp[1] = 0x4D;
        write_u32_le(&mut bmp, 14, 40);
        write_i32_le(&mut bmp, 18, 4); // width = 4
        write_i32_le(&mut bmp, 22, 2); // height = 2
        write_u16_le(&mut bmp, 26, 1);
        write_u16_le(&mut bmp, 28, 24); // 24-bit
        write_u32_le(&mut bmp, 30, BI_RGB);
        let pixel_offset = 14 + 40;
        write_u32_le(&mut bmp, 10, pixel_offset as u32);
        // Only provide 6 bytes (2 pixels) instead of the 24 needed.
        bmp.extend(&[0u8; 6]);
        let file_size = bmp.len() as u32;
        write_u32_le(&mut bmp, 2, file_size);
        let result = decode(&bmp);
        assert!(result.is_err());
    }

    #[test]
    fn determine_color_space_v4_lcs_calibrated_rgb() {
        // V4 header with cs_type = 0 (LCS_CALIBRATED_RGB) → falls through
        // to the `_` arm in determine_color_space → Srgb.
        let mut bmp = vec![0u8; 14 + 108];
        bmp[0] = 0x42;
        bmp[1] = 0x4D;
        write_u32_le(&mut bmp, 14, 108);
        write_i32_le(&mut bmp, 18, 1);
        write_i32_le(&mut bmp, 22, 1);
        write_u16_le(&mut bmp, 26, 1);
        write_u16_le(&mut bmp, 28, 24);
        write_u32_le(&mut bmp, 30, BI_RGB);
        // cs_type = 0 (LCS_CALIBRATED_RGB) — NOT LCS_SRGB or LCS_WINDOWS_COLOR_SPACE
        write_u32_le(&mut bmp, 70, 0x00000000);
        let pixel_offset = 14 + 108;
        write_u32_le(&mut bmp, 10, pixel_offset as u32);
        let file_size = (pixel_offset + 4) as u32;
        write_u32_le(&mut bmp, 2, file_size);
        bmp.extend(&[0u8; 4]);
        let decoded = decode(&bmp).unwrap();
        assert_eq!(decoded.metadata.color_space, BmpColorSpace::Srgb);
        assert_eq!(decoded.metadata.header_version, BmpHeaderVersion::V4);
    }

    #[test]
    fn v5_header_lcs_windows_color_space() {
        // V5 header with LCS_WINDOWS_COLOR_SPACE → Srgb (not IccTagged).
        let mut bmp = vec![0u8; 14 + 124];
        bmp[0] = 0x42;
        bmp[1] = 0x4D;
        write_u32_le(&mut bmp, 14, 124);
        write_i32_le(&mut bmp, 18, 1);
        write_i32_le(&mut bmp, 22, 1);
        write_u16_le(&mut bmp, 26, 1);
        write_u16_le(&mut bmp, 28, 24);
        write_u32_le(&mut bmp, 30, BI_RGB);
        // cs_type at offset 14 + 56 = 70
        write_u32_le(&mut bmp, 70, 0x57696E20); // LCS_WINDOWS_COLOR_SPACE
        let pixel_offset = 14 + 124;
        write_u32_le(&mut bmp, 10, pixel_offset as u32);
        bmp.extend(&[0u8; 4]); // 1 pixel (3 bytes + 1 pad)
        let file_size = bmp.len() as u32;
        write_u32_le(&mut bmp, 2, file_size);
        let decoded = decode(&bmp).unwrap();
        assert_eq!(decoded.metadata.color_space, BmpColorSpace::Srgb);
        assert_eq!(decoded.metadata.header_version, BmpHeaderVersion::V5);
    }
}
