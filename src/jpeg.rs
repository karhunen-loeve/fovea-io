//! JPEG decoding and encoding.
//!
//! # Decoding
//!
//! The primary entry point is [`decode`], which takes a byte slice and returns
//! a [`JpegDecoded`] — a struct containing the pixel data as a [`JpegImage`]
//! and ancillary information as [`JpegMetadata`].
//!
//! For streaming sources, use [`decode_reader`] which accepts any `impl Read`.
//! Both paths produce identical results — `decode_reader` buffers the stream
//! internally and delegates to `decode`.
//!
//! | JPEG configuration         | [`JpegImage`] variant | Pixel type      |
//! |----------------------------|-----------------------|-----------------|
//! | 8-bit grayscale            | `SrgbMono8`           | [`SrgbMono8`]   |
//! | 12-bit grayscale (ext.)    | `SrgbMono16`          | [`SrgbMono16`]  |
//! | 8-bit RGB                  | `Srgb8`               | [`Srgb8`]       |
//! | CMYK                       | —                     | rejected        |
//!
//! # Encoding
//!
//! Use [`encode`] (to `Vec<u8>`) or [`encode_writer`] (to any `impl Write`)
//! to produce JPEG data.  Only pixel types implementing the sealed
//! [`JpegPixel`] trait can be encoded — currently [`SrgbMono8`] and
//! [`Srgb8`].  Attempting to encode other types (e.g. `Rgb8`, `Srgba8`)
//! is a compile-time error.
//!
//! [`encode_jpeg_image`] dispatches over [`JpegImage`] variants for
//! convenient roundtripping.
//!
//! # EXIF parsing
//!
//! A curated set of ~15 EXIF tags is parsed into [`JpegExifInfo`] using a
//! built-in TIFF IFD parser with no external dependencies.  Tags extracted:
//!
//! - **IFD0:** orientation, make, model, software, datetime
//! - **EXIF sub-IFD:** exposure time, f-number, ISO, focal length,
//!   datetime original
//! - **GPS sub-IFD:** latitude, longitude, altitude (converted to decimal
//!   degrees / metres)
//!
//! Raw EXIF bytes are also retained in [`JpegMetadata::raw_exif`] for users
//! who need vendor-specific or niche tags via a dedicated EXIF library.
//!
//! # Design rationale
//!
//! - **`JpegImage` is exhaustive** (no `#[non_exhaustive]`).  It is the spec
//!   sheet for what a JPEG file can produce.  Adding a variant (e.g. CMYK)
//!   is a genuine semantic change that callers must handle — a compile error
//!   is the correct response.
//!
//! - **`JpegDecoded` is `#[non_exhaustive]`**.  It is a return-only struct
//!   that may gain fields (e.g. decode warnings) in the future without
//!   breaking downstream code.
//!
//! - **Always sRGB.**  JPEG is defined as sRGB by JFIF 1.02.  Even when an
//!   ICC profile is embedded, decoded sample values are gamma-encoded.
//!   Returning linear pixel types would be a type-level lie — the compiler
//!   would then allow linear-math operations on gamma-encoded data,
//!   producing silently wrong results.
//!
//! - **No CMYK.**  CMYK → sRGB conversion requires an ICC profile and a
//!   colour management engine.  Naive conversion is incorrect for most
//!   real-world CMYK profiles.  CMYK JPEGs are rejected with
//!   [`IoError::UnsupportedFeature`].
//!
//! - **EXIF is parsed into typed fields**, not returned as an opaque blob.
//!   Raw EXIF bytes are also retained for users who need vendor-specific or
//!   niche tags.
//!
//! - **Orientation is NOT auto-applied.**  EXIF orientation is metadata, not
//!   pixel data.  Auto-applying it would mean the returned image silently
//!   differs from the raw JPEG samples — violating explicit data layout.
//!
//! - **`Srgba8` is excluded from [`JpegPixel`].**  JPEG has no alpha
//!   channel.  Silently dropping alpha would be a lossy conversion hidden
//!   inside the I/O layer, violating design principle §4 (conversions are
//!   named).  Strip alpha explicitly before encoding.

use crate::IoError;
use fovea::image::{Image, ImageView};
use fovea::pixel::{Srgb8, SrgbMono8, SrgbMono16};

// ═══════════════════════════════════════════════════════════════════════════════
// JpegImage — per-codec exhaustive output enum
// ═══════════════════════════════════════════════════════════════════════════════

/// Decoded JPEG pixel data.
///
/// Each variant corresponds to a JPEG pixel format decoded by `jpeg-decoder`.
/// All variants use sRGB pixel types because JPEG is inherently gamma-encoded
/// (JFIF 1.02 defines JPEG as sRGB).
///
/// | `jpeg-decoder` format | Channels | Variant      | Pixel type    |
/// |-----------------------|----------|--------------|---------------|
/// | `PixelFormat::L8`     | 1 × u8   | `SrgbMono8`  | `SrgbMono8`   |
/// | `PixelFormat::L16`    | 1 × u16  | `SrgbMono16` | `SrgbMono16`  |
/// | `PixelFormat::RGB24`  | 3 × u8   | `Srgb8`      | `Srgb8`       |
/// | `PixelFormat::CMYK32` | 4 × u8   | —            | rejected      |
///
/// This enum is deliberately **not** `#[non_exhaustive]`.  It is the spec
/// sheet; adding a variant is semver-major.
///
/// The `Debug` impl shows the variant name and image dimensions (e.g.
/// `Srgb8(320x240)`) without dumping pixel data.
///
/// # Examples
///
/// ```
/// # use fovea::image::Image;
/// # use fovea::pixel::{Srgb8, SrgbMono8, SrgbMono16};
/// use fovea_io::jpeg::JpegImage;
///
/// // Construct each variant:
/// let mono8 = JpegImage::SrgbMono8(Image::fill(2, 2, SrgbMono8::new(128)));
/// let mono16 = JpegImage::SrgbMono16(Image::fill(4, 3, SrgbMono16::new(1000)));
/// let rgb = JpegImage::Srgb8(Image::fill(320, 240, Srgb8::new(0, 0, 0)));
///
/// // Debug shows variant + dimensions, not pixel data:
/// assert_eq!(format!("{:?}", mono8), "SrgbMono8(2x2)");
/// assert_eq!(format!("{:?}", mono16), "SrgbMono16(4x3)");
/// assert_eq!(format!("{:?}", rgb), "Srgb8(320x240)");
/// ```
pub enum JpegImage {
    /// 8-bit sRGB grayscale (JFIF luminance-only).
    SrgbMono8(Image<SrgbMono8>),
    /// 16-bit sRGB grayscale (12-bit JPEG extended, decoded to 16-bit).
    SrgbMono16(Image<SrgbMono16>),
    /// 8-bit sRGB truecolour (the overwhelmingly common case).
    Srgb8(Image<Srgb8>),
}

impl JpegImage {
    /// Width in pixels, regardless of variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use fovea_io::jpeg::JpegImage;
    /// use fovea::image::Image;
    /// use fovea::pixel::Srgb8;
    ///
    /// let img = JpegImage::Srgb8(Image::fill(320, 240, Srgb8::new(0, 0, 0)));
    /// assert_eq!(img.width(), 320);
    /// ```
    #[must_use]
    pub fn width(&self) -> usize {
        use fovea::image::ImageView;
        match self {
            JpegImage::SrgbMono8(img) => img.width(),
            JpegImage::SrgbMono16(img) => img.width(),
            JpegImage::Srgb8(img) => img.width(),
        }
    }

    /// Height in pixels, regardless of variant.
    #[must_use]
    pub fn height(&self) -> usize {
        use fovea::image::ImageView;
        match self {
            JpegImage::SrgbMono8(img) => img.height(),
            JpegImage::SrgbMono16(img) => img.height(),
            JpegImage::Srgb8(img) => img.height(),
        }
    }

    /// Image size (`width × height`), regardless of variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use fovea_io::jpeg::JpegImage;
    /// use fovea::image::Image;
    /// use fovea::pixel::Srgb8;
    ///
    /// let img = JpegImage::Srgb8(Image::fill(320, 240, Srgb8::new(0, 0, 0)));
    /// let sz = img.size();
    /// assert_eq!(sz.width, 320);
    /// assert_eq!(sz.height, 240);
    /// ```
    #[must_use]
    pub fn size(&self) -> fovea::Size {
        use fovea::image::ImageView;
        match self {
            JpegImage::SrgbMono8(img) => img.size(),
            JpegImage::SrgbMono16(img) => img.size(),
            JpegImage::Srgb8(img) => img.size(),
        }
    }
}

impl std::fmt::Debug for JpegImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JpegImage::SrgbMono8(img) => {
                write!(f, "SrgbMono8({}x{})", img.width(), img.height())
            }
            JpegImage::SrgbMono16(img) => {
                write!(f, "SrgbMono16({}x{})", img.width(), img.height())
            }
            JpegImage::Srgb8(img) => write!(f, "Srgb8({}x{})", img.width(), img.height()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// JpegExifInfo — curated EXIF tag fields
// ═══════════════════════════════════════════════════════════════════════════════

/// A curated set of EXIF tags parsed into typed Rust fields.
///
/// The selection covers tags that matter for image processing, display
/// correctness, provenance, and scientific/industrial workflows.  All fields
/// are `Option<T>` because any EXIF tag may be absent or malformed.
///
/// # Design choices
///
/// - **Rationals stay as `(u32, u32)` tuples** for exposure, f-number, focal
///   length.  Users who want `f64` can compute `num as f64 / den as f64`.
///
/// - **GPS coordinates are `f64` decimal degrees.**  `f64` gives
///   sub-millimetre precision and matches standard geospatial representations.
///
/// - **Timestamps stay as `String`.** EXIF timestamps follow `"YYYY:MM:DD
///   HH:MM:SS"`.  A `String` preserves the exact EXIF value.
///
/// # Examples
///
/// ```ignore
/// // Construction requires being inside the defining crate due to
/// // #[non_exhaustive].  In practice, obtain via `jpeg::decode()`.
/// use fovea_io::jpeg::JpegExifInfo;
///
/// let info = JpegExifInfo {
///     orientation: Some(1),
///     datetime: Some("2025:01:15 12:30:00".to_string()),
///     camera_make: Some("Canon".to_string()),
///     ..Default::default()
/// };
/// assert_eq!(info.orientation, Some(1));
/// assert_eq!(info.camera_make.as_deref(), Some("Canon"));
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JpegExifInfo {
    // ── Display correctness ──────────────────────────────────────────
    /// EXIF orientation tag (0x0112).  Values 1–8.
    /// Describes how the stored pixel grid relates to the intended
    /// display orientation (rotation + mirror).
    pub orientation: Option<u8>,

    // ── Timestamps ───────────────────────────────────────────────────
    /// File modification time (IFD0 tag 0x0132).
    /// Format: `"YYYY:MM:DD HH:MM:SS"` per EXIF spec.
    pub datetime: Option<String>,
    /// Original capture time (EXIF sub-IFD tag 0x9003).
    /// This is the shutter-release moment — more reliable than `datetime`.
    pub datetime_original: Option<String>,

    // ── Camera identification ────────────────────────────────────────
    /// Camera manufacturer (IFD0 tag 0x010F).
    pub camera_make: Option<String>,
    /// Camera model (IFD0 tag 0x0110).
    pub camera_model: Option<String>,
    /// Software that produced the file (IFD0 tag 0x0131).
    pub software: Option<String>,

    // ── Exposure parameters (scientific / industrial) ────────────────
    /// Exposure time in seconds as a rational (tag 0x829A).
    /// E.g. `(1, 250)` means 1/250 s.
    pub exposure_time: Option<(u32, u32)>,
    /// F-number as a rational (tag 0x829D).
    /// E.g. `(28, 10)` means f/2.8.
    pub f_number: Option<(u32, u32)>,
    /// ISO speed (tag 0x8827).
    pub iso_speed: Option<u16>,
    /// Focal length in mm as a rational (tag 0x920A).
    /// E.g. `(50, 1)` means 50 mm.
    pub focal_length: Option<(u32, u32)>,

    // ── Geospatial ───────────────────────────────────────────────────
    /// GPS latitude in decimal degrees.
    /// Positive = North, negative = South.
    /// Converted from EXIF DMS rationals + N/S reference.
    pub gps_latitude: Option<f64>,
    /// GPS longitude in decimal degrees.
    /// Positive = East, negative = West.
    /// Converted from EXIF DMS rationals + E/W reference.
    pub gps_longitude: Option<f64>,
    /// GPS altitude in metres.
    /// Positive = above sea level, negative = below.
    /// Converted from EXIF rational + altitude reference byte.
    pub gps_altitude: Option<f64>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// JpegColorSpace — what the colour-space markers told us
// ═══════════════════════════════════════════════════════════════════════════════

/// Colour-space signalling extracted from a JPEG file.
///
/// JPEG has fewer colour-space signalling mechanisms than PNG.  The pixel
/// type in [`JpegImage`] already encodes the transfer-function assumption
/// (always sRGB).  This enum signals whether an ICC profile is present.
///
/// This enum is deliberately **not** `#[non_exhaustive]` — it is a spec
/// sheet.  Adding Adobe RGB signalling (APP14 marker) would be a genuine
/// semantic change requiring a semver-major bump.
///
/// # Examples
///
/// ```
/// use fovea_io::jpeg::JpegColorSpace;
///
/// let cs = JpegColorSpace::Srgb;
/// assert_eq!(cs, JpegColorSpace::Srgb);
/// assert_ne!(cs, JpegColorSpace::IccTagged);
///
/// // Copy semantics:
/// let cs2 = cs;
/// assert_eq!(cs, cs2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegColorSpace {
    /// Standard JFIF — sRGB assumed (no embedded ICC profile).
    Srgb,
    /// An ICC profile is present (see `JpegMetadata::icc_profile`).
    IccTagged,
}

// ═══════════════════════════════════════════════════════════════════════════════
// JpegPixelDensity — JFIF APP0 pixel density
// ═══════════════════════════════════════════════════════════════════════════════

/// Pixel density information from the JFIF APP0 marker.
///
/// JFIF defines three kinds of density information: DPI, dots per centimetre,
/// and unitless aspect ratio.
///
/// # Examples
///
/// ```
/// use fovea_io::jpeg::JpegPixelDensity;
///
/// let dpi = JpegPixelDensity::Dpi { x: 300, y: 300 };
/// let dpcm = JpegPixelDensity::Dpcm { x: 118, y: 118 };
/// let aspect = JpegPixelDensity::AspectRatio { x: 1, y: 1 };
///
/// assert_eq!(dpi, JpegPixelDensity::Dpi { x: 300, y: 300 });
/// assert_ne!(dpi, dpcm);
///
/// // Copy semantics:
/// let dpi2 = dpi;
/// assert_eq!(dpi, dpi2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegPixelDensity {
    /// Pixels per inch.
    Dpi {
        /// Horizontal density.
        x: u16,
        /// Vertical density.
        y: u16,
    },
    /// Pixels per centimetre.
    Dpcm {
        /// Horizontal density.
        x: u16,
        /// Vertical density.
        y: u16,
    },
    /// Aspect ratio only (unitless).
    AspectRatio {
        /// Horizontal aspect.
        x: u16,
        /// Vertical aspect.
        y: u16,
    },
}

// ═══════════════════════════════════════════════════════════════════════════════
// JpegBitDepth — source sample precision
// ═══════════════════════════════════════════════════════════════════════════════

/// Source bit depth of a JPEG file.
///
/// JPEG supports exactly two sample precisions: 8-bit (baseline/progressive)
/// and 12-bit (extended).  A `u8` field would admit 254 invalid states.
/// Per design principle §1 (types are the spec), a two-valued domain is a
/// two-variant enum.
///
/// # Examples
///
/// ```
/// use fovea_io::jpeg::JpegBitDepth;
///
/// let depth = JpegBitDepth::Eight;
/// assert_eq!(depth, JpegBitDepth::Eight);
/// assert_ne!(depth, JpegBitDepth::Twelve);
///
/// // Copy semantics:
/// let depth2 = depth;
/// assert_eq!(depth, depth2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegBitDepth {
    /// Baseline / progressive JPEG — 8-bit samples.
    Eight,
    /// JPEG Extended — 12-bit samples (decoded to 16-bit by `jpeg-decoder`).
    Twelve,
}

// ═══════════════════════════════════════════════════════════════════════════════
// JpegMetadata — ancillary information beyond pixel data
// ═══════════════════════════════════════════════════════════════════════════════

/// Ancillary metadata extracted from a JPEG file.
///
/// Returned as part of [`JpegDecoded`].  Fields are optional — only populated
/// when the corresponding markers exist in the file.
///
/// # Examples
///
/// ```ignore
/// // Construction requires being inside the defining crate due to
/// // #[non_exhaustive].  In practice, obtain via `jpeg::decode()`.
/// use fovea_io::jpeg::{JpegMetadata, JpegColorSpace, JpegBitDepth};
///
/// let meta = JpegMetadata {
///     exif: None,
///     raw_exif: None,
///     icc_profile: None,
///     pixel_density: None,
///     comments: vec!["Hello, world!".to_string()],
///     source_bit_depth: JpegBitDepth::Eight,
///     color_space: JpegColorSpace::Srgb,
/// };
/// assert_eq!(meta.source_bit_depth, JpegBitDepth::Eight);
/// assert_eq!(meta.comments.len(), 1);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct JpegMetadata {
    /// Parsed EXIF information (curated tag subset).
    pub exif: Option<JpegExifInfo>,
    /// Raw EXIF bytes from the APP1 marker, for users who need niche tags.
    pub raw_exif: Option<Box<[u8]>>,
    /// ICC profile bytes from the APP2 marker.
    pub icc_profile: Option<Box<[u8]>>,
    /// Pixel density from the JFIF APP0 marker.
    pub pixel_density: Option<JpegPixelDensity>,
    /// Comment strings from COM markers.
    pub comments: Vec<String>,
    /// Source bit depth — either 8-bit (baseline/progressive) or 12-bit (extended).
    pub source_bit_depth: JpegBitDepth,
    /// Colour-space signalling.
    pub color_space: JpegColorSpace,
}

// ═══════════════════════════════════════════════════════════════════════════════
// JpegDecoded — the complete decode result
// ═══════════════════════════════════════════════════════════════════════════════

/// The complete result of decoding a JPEG file.
///
/// Contains both the pixel data ([`JpegImage`]) and ancillary metadata
/// ([`JpegMetadata`]).  This struct is `#[non_exhaustive]` so that new
/// fields (e.g. decode warnings) can be added without a semver-major bump.
///
/// # Examples
///
/// ```ignore
/// // Construction requires being inside the defining crate due to
/// // #[non_exhaustive].  In practice, obtain via `jpeg::decode()`.
/// # use fovea::image::Image;
/// # use fovea::pixel::Srgb8;
/// use fovea_io::jpeg::{JpegDecoded, JpegImage, JpegMetadata, JpegColorSpace, JpegBitDepth};
///
/// let decoded = JpegDecoded {
///     image: JpegImage::Srgb8(Image::fill(1, 1, Srgb8::new(0, 0, 0))),
///     metadata: JpegMetadata {
///         exif: None,
///         raw_exif: None,
///         icc_profile: None,
///         pixel_density: None,
///         comments: vec![],
///         source_bit_depth: JpegBitDepth::Eight,
///         color_space: JpegColorSpace::Srgb,
///     },
/// };
/// // Fields are directly accessible:
/// let _img = decoded.image;
/// let _meta = decoded.metadata;
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct JpegDecoded {
    /// The decoded pixel data.
    pub image: JpegImage,
    /// Ancillary metadata (EXIF, ICC, comments, …).
    pub metadata: JpegMetadata,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TIFF / EXIF internals — private helpers for parsing EXIF APP1 data
// ═══════════════════════════════════════════════════════════════════════════════

/// Byte order of a TIFF stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteOrder {
    /// Little-endian (`II` — Intel).
    Little,
    /// Big-endian (`MM` — Motorola).
    Big,
}

/// Bounds-checked reader over a TIFF byte stream with a fixed byte order.
///
/// All offset-based accessors return `Option<T>` — out-of-bounds reads
/// produce `None` rather than panicking.  No unsafe code.
struct TiffReader<'a> {
    data: &'a [u8],
    order: ByteOrder,
}

impl<'a> TiffReader<'a> {
    /// Create a new reader over `data` with the given byte order.
    fn new(data: &'a [u8], order: ByteOrder) -> Self {
        Self { data, order }
    }

    /// Total length of the underlying data.
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.data.len()
    }

    /// Read a single byte at `offset`.
    fn u8_at(&self, offset: usize) -> Option<u8> {
        self.data.get(offset).copied()
    }

    /// Read a 16-bit unsigned integer at `offset`, respecting byte order.
    fn u16_at(&self, offset: usize) -> Option<u16> {
        let bytes: &[u8] = self.data.get(offset..offset.checked_add(2)?)?;
        let arr: [u8; 2] = [bytes[0], bytes[1]];
        Some(match self.order {
            ByteOrder::Little => u16::from_le_bytes(arr),
            ByteOrder::Big => u16::from_be_bytes(arr),
        })
    }

    /// Read a 32-bit unsigned integer at `offset`, respecting byte order.
    fn u32_at(&self, offset: usize) -> Option<u32> {
        let bytes: &[u8] = self.data.get(offset..offset.checked_add(4)?)?;
        let arr: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        Some(match self.order {
            ByteOrder::Little => u32::from_le_bytes(arr),
            ByteOrder::Big => u32::from_be_bytes(arr),
        })
    }

    /// Read a TIFF RATIONAL (two consecutive u32s: numerator, denominator)
    /// at `offset`, respecting byte order.
    fn rational_at(&self, offset: usize) -> Option<(u32, u32)> {
        let num = self.u32_at(offset)?;
        let den = self.u32_at(offset.checked_add(4)?)?;
        Some((num, den))
    }
}

// ── IFD entry reader (task 2.2) ──────────────────────────────────────────────

/// Maximum number of IFD entries we'll read.  Prevents DoS on malformed
/// data that claims millions of entries.
const MAX_IFD_ENTRIES: u16 = 1000;

/// A raw IFD entry parsed from a TIFF stream.
///
/// If the value fits in 4 bytes (based on type × count), `value_or_offset`
/// holds the value directly (left-aligned in the 4-byte field).  Otherwise
/// it is a byte offset into the TIFF stream where the value is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IfdEntry {
    /// EXIF/TIFF tag number (e.g. 0x0112 = Orientation).
    tag: u16,
    /// TIFF data type (1=BYTE, 2=ASCII, 3=SHORT, 4=LONG, 5=RATIONAL, …).
    tiff_type: u16,
    /// Number of values of the given type.
    count: u32,
    /// The value itself (if it fits in 4 bytes) or an offset to the value.
    value_or_offset: u32,
}

/// Read all IFD entries starting at `ifd_offset` in the TIFF stream.
///
/// Returns an empty `Vec` if the offset is out of bounds or the entry
/// count is zero.  Caps the entry count at [`MAX_IFD_ENTRIES`].
fn read_ifd_entries(reader: &TiffReader<'_>, ifd_offset: usize) -> Vec<IfdEntry> {
    let count = match reader.u16_at(ifd_offset) {
        Some(c) => c.min(MAX_IFD_ENTRIES),
        None => return Vec::new(),
    };

    let mut entries = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        // Each IFD entry is 12 bytes, starting 2 bytes after ifd_offset.
        let base = match ifd_offset
            .checked_add(2)
            .and_then(|b| b.checked_add(i.checked_mul(12)?))
        {
            Some(b) => b,
            None => break,
        };

        let tag = match reader.u16_at(base) {
            Some(v) => v,
            None => break,
        };
        let tiff_type = match reader.u16_at(base + 2) {
            Some(v) => v,
            None => break,
        };
        let count = match reader.u32_at(base + 4) {
            Some(v) => v,
            None => break,
        };
        let value_or_offset = match reader.u32_at(base + 8) {
            Some(v) => v,
            None => break,
        };

        entries.push(IfdEntry {
            tag,
            tiff_type,
            count,
            value_or_offset,
        });
    }

    entries
}

// ── ASCII string reader (task 2.3) ───────────────────────────────────────────

/// Read an ASCII string value from the TIFF stream.
///
/// `offset` is the byte offset where the string data starts.
/// `count` is the total byte length including the trailing NUL (per TIFF spec).
///
/// - Trims trailing NUL bytes.
/// - Accepts valid UTF-8 as-is (covers both ASCII and UTF-8 encoded strings).
/// - Falls back to Latin-1 → UTF-8 conversion for non-UTF-8 data (common in
///   real-world EXIF from European cameras).
///
/// Returns `None` if the range is out of bounds or the result is empty after
/// trimming.
fn read_ascii(reader: &TiffReader<'_>, offset: usize, count: u32) -> Option<String> {
    let count = count as usize;
    if count == 0 {
        return None;
    }
    let end = offset.checked_add(count)?;
    let bytes = reader.data.get(offset..end)?;

    // Trim trailing NUL bytes (there may be more than one in malformed data).
    let trimmed = match bytes.iter().rposition(|&b| b != 0) {
        Some(last) => &bytes[..=last],
        None => return None, // all NULs or empty
    };

    if trimmed.is_empty() {
        return None;
    }

    // Try UTF-8 first (covers ASCII and UTF-8).
    if let Ok(s) = std::str::from_utf8(trimmed) {
        return Some(s.to_string());
    }

    // Fall back to Latin-1: each byte maps to its Unicode code point.
    let s: String = trimmed.iter().map(|&b| b as char).collect();
    Some(s)
}

/// Read an ASCII string from an IFD entry, handling both inline (≤ 4 bytes)
/// and offset-referenced values.
///
/// TIFF type 2 = ASCII.  If `count ≤ 4`, the bytes live in the
/// `value_or_offset` field itself (left-aligned, in stream byte order).
/// Otherwise, `value_or_offset` is an offset into the TIFF data.
fn read_ifd_ascii(
    reader: &TiffReader<'_>,
    count: u32,
    value_or_offset: u32,
    entry_value_offset: usize,
) -> Option<String> {
    if count <= 4 {
        // Value is stored inline in the 4-byte value_or_offset field.
        // `entry_value_offset` points to the raw bytes of that field in the stream.
        read_ascii(reader, entry_value_offset, count)
    } else {
        read_ascii(reader, value_or_offset as usize, count)
    }
}

// ── GPS coordinate converter (task 2.4) ──────────────────────────────────────

/// TIFF type constants used when interpreting IFD entry types.
const TIFF_TYPE_BYTE: u16 = 1;
const TIFF_TYPE_ASCII: u16 = 2;
const TIFF_TYPE_SHORT: u16 = 3;
#[allow(dead_code)]
const TIFF_TYPE_LONG: u16 = 4;
const TIFF_TYPE_RATIONAL: u16 = 5;

/// Convert GPS DMS (degrees/minutes/seconds) rationals to decimal degrees.
///
/// Each component is a TIFF RATIONAL `(numerator, denominator)`.
/// `ref_char` is the ASCII reference: `b'N'`/`b'S'` for latitude,
/// `b'E'`/`b'W'` for longitude.
///
/// Returns `None` if:
/// - Any denominator is zero
/// - Minutes ≥ 60 or seconds ≥ 60
/// - Degrees ≥ 360
/// - `ref_char` is not one of `N`, `S`, `E`, `W`
fn dms_to_decimal(
    degrees: (u32, u32),
    minutes: (u32, u32),
    seconds: (u32, u32),
    ref_char: u8,
) -> Option<f64> {
    // Validate denominators.
    if degrees.1 == 0 || minutes.1 == 0 || seconds.1 == 0 {
        return None;
    }

    let deg = degrees.0 as f64 / degrees.1 as f64;
    let min = minutes.0 as f64 / minutes.1 as f64;
    let sec = seconds.0 as f64 / seconds.1 as f64;

    // Range validation.
    if deg >= 360.0 || min >= 60.0 || sec >= 60.0 {
        return None;
    }

    let sign = match ref_char {
        b'N' | b'E' => 1.0,
        b'S' | b'W' => -1.0,
        _ => return None,
    };

    Some(sign * (deg + min / 60.0 + sec / 3600.0))
}

// ── Top-level EXIF parser (task 2.5) ─────────────────────────────────────────

/// Parse an APP1 EXIF payload into a [`JpegExifInfo`].
///
/// `raw` is the full APP1 payload starting with `Exif\0\0`.
///
/// Returns `Some(JpegExifInfo)` if the TIFF header is valid (even if all
/// tag fields end up `None`).  Returns `None` if the header is invalid
/// or the data is not EXIF.
///
/// # Best-effort parsing
///
/// Individual malformed tags are silently skipped — a corrupt orientation
/// tag does not prevent GPS data from being read.  Only a fundamentally
/// invalid TIFF header causes the entire parse to fail.
#[allow(dead_code)]
fn parse_exif(raw: &[u8]) -> Option<JpegExifInfo> {
    // Validate Exif header prefix.
    if raw.len() < 14 {
        return None;
    }
    if &raw[0..6] != b"Exif\0\0" {
        return None;
    }
    parse_tiff_exif(&raw[6..])
}

/// Parse EXIF data from a raw TIFF stream (no `Exif\0\0` prefix).
///
/// This is the shared implementation used by both [`parse_exif`] (which
/// strips the 6-byte prefix first) and the decode path (where
/// `jpeg_decoder::Decoder::exif_data()` returns data starting at the
/// TIFF header directly).
fn parse_tiff_exif(tiff: &[u8]) -> Option<JpegExifInfo> {
    if tiff.len() < 8 {
        // Need at least: 2 (byte order) + 2 (magic) + 4 (IFD0 offset)
        return None;
    }

    // ── Detect byte order ────────────────────────────────────────────
    let order = match &tiff[0..2] {
        b"II" => ByteOrder::Little,
        b"MM" => ByteOrder::Big,
        _ => return None,
    };

    let reader = TiffReader::new(tiff, order);

    // ── Validate TIFF magic number (42) ──────────────────────────────
    let magic = reader.u16_at(2)?;
    if magic != 42 {
        return None;
    }

    // ── Read IFD0 offset ─────────────────────────────────────────────
    let ifd0_offset = reader.u32_at(4)? as usize;

    let mut info = JpegExifInfo::default();
    let mut exif_ifd_offset: Option<usize> = None;
    let mut gps_ifd_offset: Option<usize> = None;

    // ── Walk IFD0 ────────────────────────────────────────────────────
    let ifd0_entries = read_ifd_entries(&reader, ifd0_offset);
    for (i, entry) in ifd0_entries.iter().enumerate() {
        let IfdEntry {
            tag,
            tiff_type,
            count,
            value_or_offset,
        } = *entry;
        // Byte offset of the value/offset field for this entry in the TIFF stream.
        let entry_val_off = ifd0_offset + 2 + i * 12 + 8;

        match tag {
            // Orientation (SHORT, count=1)
            0x0112 => {
                if tiff_type == TIFF_TYPE_SHORT && count == 1 {
                    if let Some(v) = reader.u16_at(entry_val_off) {
                        if (1..=8).contains(&v) {
                            info.orientation = Some(v as u8);
                        }
                    }
                }
            }
            // Make (ASCII)
            0x010F => {
                if tiff_type == TIFF_TYPE_ASCII {
                    info.camera_make =
                        read_ifd_ascii(&reader, count, value_or_offset, entry_val_off);
                }
            }
            // Model (ASCII)
            0x0110 => {
                if tiff_type == TIFF_TYPE_ASCII {
                    info.camera_model =
                        read_ifd_ascii(&reader, count, value_or_offset, entry_val_off);
                }
            }
            // Software (ASCII)
            0x0131 => {
                if tiff_type == TIFF_TYPE_ASCII {
                    info.software = read_ifd_ascii(&reader, count, value_or_offset, entry_val_off);
                }
            }
            // DateTime (ASCII, 20 bytes)
            0x0132 => {
                if tiff_type == TIFF_TYPE_ASCII {
                    info.datetime = read_ifd_ascii(&reader, count, value_or_offset, entry_val_off);
                }
            }
            // ExifIFDPointer (LONG)
            0x8769 => {
                if count == 1 {
                    exif_ifd_offset = Some(value_or_offset as usize);
                }
            }
            // GPSInfoPointer (LONG)
            0x8825 => {
                if count == 1 {
                    gps_ifd_offset = Some(value_or_offset as usize);
                }
            }
            _ => {}
        }
    }

    // ── Walk EXIF sub-IFD ────────────────────────────────────────────
    if let Some(exif_off) = exif_ifd_offset {
        let exif_entries = read_ifd_entries(&reader, exif_off);
        for (i, entry) in exif_entries.iter().enumerate() {
            let IfdEntry {
                tag,
                tiff_type,
                count,
                value_or_offset,
            } = *entry;
            let entry_val_off = exif_off + 2 + i * 12 + 8;

            match tag {
                // ExposureTime (RATIONAL)
                0x829A => {
                    if tiff_type == TIFF_TYPE_RATIONAL && count == 1 {
                        info.exposure_time = reader.rational_at(value_or_offset as usize);
                    }
                }
                // FNumber (RATIONAL)
                0x829D => {
                    if tiff_type == TIFF_TYPE_RATIONAL && count == 1 {
                        info.f_number = reader.rational_at(value_or_offset as usize);
                    }
                }
                // ISOSpeedRatings (SHORT)
                0x8827 => {
                    if tiff_type == TIFF_TYPE_SHORT && count == 1 {
                        info.iso_speed = reader.u16_at(entry_val_off);
                    }
                }
                // DateTimeOriginal (ASCII)
                0x9003 => {
                    if tiff_type == TIFF_TYPE_ASCII {
                        info.datetime_original =
                            read_ifd_ascii(&reader, count, value_or_offset, entry_val_off);
                    }
                }
                // FocalLength (RATIONAL)
                0x920A => {
                    if tiff_type == TIFF_TYPE_RATIONAL && count == 1 {
                        info.focal_length = reader.rational_at(value_or_offset as usize);
                    }
                }
                _ => {}
            }
        }
    }

    // ── Walk GPS sub-IFD ─────────────────────────────────────────────
    if let Some(gps_off) = gps_ifd_offset {
        let gps_entries = read_ifd_entries(&reader, gps_off);

        // Collect raw GPS tag values; we need all of them before conversion.
        let mut lat_ref: Option<u8> = None;
        let mut lat_dms: Option<[(u32, u32); 3]> = None;
        let mut lon_ref: Option<u8> = None;
        let mut lon_dms: Option<[(u32, u32); 3]> = None;
        let mut alt_ref: Option<u8> = None;
        let mut alt_rational: Option<(u32, u32)> = None;

        for (i, entry) in gps_entries.iter().enumerate() {
            let IfdEntry {
                tag,
                tiff_type,
                count,
                value_or_offset,
            } = *entry;
            let entry_val_off = gps_off + 2 + i * 12 + 8;

            match tag {
                // GPSLatitudeRef (ASCII, 2 bytes: "N\0" or "S\0")
                0x0001 => {
                    if tiff_type == TIFF_TYPE_ASCII && count == 2 {
                        lat_ref = reader.u8_at(entry_val_off);
                    }
                }
                // GPSLatitude (3 × RATIONAL = 24 bytes, always offset-referenced)
                0x0002 => {
                    if tiff_type == TIFF_TYPE_RATIONAL && count == 3 {
                        let off = value_or_offset as usize;
                        if let (Some(d), Some(m), Some(s)) = (
                            reader.rational_at(off),
                            reader.rational_at(off + 8),
                            reader.rational_at(off + 16),
                        ) {
                            lat_dms = Some([d, m, s]);
                        }
                    }
                }
                // GPSLongitudeRef (ASCII, 2 bytes: "E\0" or "W\0")
                0x0003 => {
                    if tiff_type == TIFF_TYPE_ASCII && count == 2 {
                        lon_ref = reader.u8_at(entry_val_off);
                    }
                }
                // GPSLongitude (3 × RATIONAL)
                0x0004 => {
                    if tiff_type == TIFF_TYPE_RATIONAL && count == 3 {
                        let off = value_or_offset as usize;
                        if let (Some(d), Some(m), Some(s)) = (
                            reader.rational_at(off),
                            reader.rational_at(off + 8),
                            reader.rational_at(off + 16),
                        ) {
                            lon_dms = Some([d, m, s]);
                        }
                    }
                }
                // GPSAltitudeRef (BYTE, count=1: 0 = above, 1 = below sea level)
                0x0005 => {
                    if tiff_type == TIFF_TYPE_BYTE && count == 1 {
                        alt_ref = reader.u8_at(entry_val_off);
                    }
                }
                // GPSAltitude (RATIONAL)
                0x0006 => {
                    if tiff_type == TIFF_TYPE_RATIONAL && count == 1 {
                        alt_rational = reader.rational_at(value_or_offset as usize);
                    }
                }
                _ => {}
            }
        }

        // Convert GPS latitude.
        if let (Some(ref_ch), Some(dms)) = (lat_ref, lat_dms) {
            info.gps_latitude = dms_to_decimal(dms[0], dms[1], dms[2], ref_ch);
        }

        // Convert GPS longitude.
        if let (Some(ref_ch), Some(dms)) = (lon_ref, lon_dms) {
            info.gps_longitude = dms_to_decimal(dms[0], dms[1], dms[2], ref_ch);
        }

        // Convert GPS altitude.
        if let Some((num, den)) = alt_rational {
            if den != 0 {
                let alt = num as f64 / den as f64;
                let sign = match alt_ref {
                    Some(1) => -1.0,
                    _ => 1.0, // 0 or absent → above sea level
                };
                info.gps_altitude = Some(sign * alt);
            }
        }
    }

    Some(info)
}

// ═══════════════════════════════════════════════════════════════════════════════
// JPEG marker scanners — raw byte scanning for markers not exposed by
// jpeg-decoder (COM comments, JFIF APP0 pixel density)
// ═══════════════════════════════════════════════════════════════════════════════

/// Scan raw JPEG bytes for COM (comment) markers and extract their text.
///
/// COM markers are `FF FE` followed by a 2-byte big-endian length (which
/// includes the 2 length bytes themselves).  Scanning stops at the SOS
/// marker (`FF DA`) — there's no need to scan entropy-coded data.
///
/// Text is decoded as UTF-8 when valid, with a lossless Latin-1 fallback
/// for non-UTF-8 data — consistent with the EXIF ASCII tag decoder (see
/// D12).  This avoids the information loss of `from_utf8_lossy` which
/// replaces non-UTF-8 bytes with U+FFFD.
fn scan_com_markers(data: &[u8]) -> Vec<String> {
    let mut comments = Vec::new();
    // JPEG files start with FF D8.  Skip to the first marker.
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return comments;
    }
    let mut pos = 2;

    while pos + 1 < data.len() {
        // Scan for marker prefix.
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }

        // Skip fill bytes (consecutive 0xFF).
        while pos + 1 < data.len() && data[pos + 1] == 0xFF {
            pos += 1;
        }
        if pos + 1 >= data.len() {
            break;
        }

        let marker = data[pos + 1];
        pos += 2; // skip past FF XX

        match marker {
            // SOS — stop scanning (rest is entropy data).
            0xDA => break,
            // Standalone markers with no payload.
            0x00 | 0x01 | 0xD0..=0xD7 => continue,
            // COM marker (0xFE).
            0xFE => {
                if pos + 2 > data.len() {
                    break;
                }
                let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if len < 2 {
                    break; // invalid length
                }
                let payload_len = len - 2;
                let payload_start = pos + 2;
                if payload_start + payload_len > data.len() {
                    break; // truncated
                }
                let payload = &data[payload_start..payload_start + payload_len];
                let text = if let Ok(s) = std::str::from_utf8(payload) {
                    s.to_string()
                } else {
                    // Latin-1 fallback: each byte maps to its Unicode code point.
                    payload.iter().map(|&b| b as char).collect()
                };
                comments.push(text);
                pos = payload_start + payload_len;
            }
            // Any other marker with a length field — skip over it.
            _ => {
                if pos + 2 > data.len() {
                    break;
                }
                let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if len < 2 || pos + len > data.len() {
                    break;
                }
                pos += len;
            }
        }
    }

    comments
}

/// Scan raw JPEG bytes for the JFIF APP0 marker and extract pixel density.
///
/// JFIF APP0 layout after `FF E0`:
/// - 2 bytes: length (BE, includes itself)
/// - 5 bytes: identifier `JFIF\0`
/// - 2 bytes: version (major, minor)
/// - 1 byte: density units (0 = aspect, 1 = DPI, 2 = DPCM)
/// - 2 bytes: X density (BE)
/// - 2 bytes: Y density (BE)
///
/// Returns `None` if no valid JFIF APP0 is found.
fn scan_jfif_density(data: &[u8]) -> Option<JpegPixelDensity> {
    // JPEG must start with FF D8.
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut pos = 2;

    // The JFIF APP0 marker should be the very first marker after SOI,
    // but we scan a few markers in case there's padding.
    while pos + 1 < data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        // Skip fill bytes.
        while pos + 1 < data.len() && data[pos + 1] == 0xFF {
            pos += 1;
        }
        if pos + 1 >= data.len() {
            break;
        }

        let marker = data[pos + 1];
        pos += 2;

        match marker {
            0xDA => break, // SOS — stop
            0x00 | 0x01 | 0xD0..=0xD7 => continue,
            0xE0 => {
                // APP0 — check if it's JFIF.
                if pos + 2 > data.len() {
                    return None;
                }
                let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if len < 2 || pos + len > data.len() {
                    return None;
                }
                let segment = &data[pos + 2..pos + len];
                // Check JFIF identifier: "JFIF\0" (5 bytes) + version (2) + units (1) + Xd (2) + Yd (2) = 12 bytes minimum
                if segment.len() >= 12 && &segment[0..5] == b"JFIF\0" {
                    let units = segment[7];
                    let x_density = u16::from_be_bytes([segment[8], segment[9]]);
                    let y_density = u16::from_be_bytes([segment[10], segment[11]]);
                    return match units {
                        1 => Some(JpegPixelDensity::Dpi {
                            x: x_density,
                            y: y_density,
                        }),
                        2 => Some(JpegPixelDensity::Dpcm {
                            x: x_density,
                            y: y_density,
                        }),
                        _ => Some(JpegPixelDensity::AspectRatio {
                            x: x_density,
                            y: y_density,
                        }),
                    };
                }
                pos += len;
            }
            _ => {
                if pos + 2 > data.len() {
                    break;
                }
                let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if len < 2 || pos + len > data.len() {
                    break;
                }
                pos += len;
            }
        }
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════════
// Decoding — public API (Phase 3)
// ═══════════════════════════════════════════════════════════════════════════════

/// Map a `jpeg_decoder::Error` into [`IoError`].
fn decode_error(e: jpeg_decoder::Error) -> IoError {
    match e {
        jpeg_decoder::Error::Io(io) => IoError::Io(io),
        other => IoError::DecodeFailed {
            source: Box::new(other),
        },
    }
}

/// Decode a JPEG image from an in-memory byte slice.
///
/// Returns a [`JpegDecoded`] containing the pixel data as a [`JpegImage`]
/// and ancillary metadata as a [`JpegMetadata`].
///
/// This is the preferred entry point when the entire file is available in
/// memory.  It scans the raw bytes for COM comment markers and JFIF APP0
/// pixel density *before* handing the data to `jpeg-decoder`, giving a
/// richer [`JpegMetadata`] than [`decode_reader`].
///
/// # Errors
///
/// - [`IoError::DecodeFailed`] — the data is not a valid JPEG or is
///   corrupt beyond recovery.
/// - [`IoError::UnsupportedFeature`] — the JPEG uses CMYK colour, which
///   this library deliberately rejects (see design decision D7).
/// - [`IoError::Io`] — an I/O error during decoding (unlikely from bytes).
///
/// # Examples
///
/// ```no_run
/// # use fovea_io::jpeg::{self, JpegImage};
/// let bytes = std::fs::read("photo.jpg").unwrap();
/// let decoded = jpeg::decode(&bytes).unwrap();
///
/// match decoded.image {
///     JpegImage::Srgb8(image) => { /* work with Image<Srgb8> */ }
///     JpegImage::SrgbMono8(image) => { /* 8-bit grayscale */ }
///     JpegImage::SrgbMono16(image) => { /* 12-bit extended, decoded to 16 */ }
/// }
/// ```
pub fn decode(data: &[u8]) -> Result<JpegDecoded, IoError> {
    // ── Pre-scan raw bytes for markers the decoder doesn't expose ─────
    let comments = scan_com_markers(data);
    let pixel_density = scan_jfif_density(data);

    // ── Run the JPEG decoder ─────────────────────────────────────────
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    let pixels = decoder.decode().map_err(decode_error)?;
    let info = decoder.info().ok_or_else(|| IoError::DecodeFailed {
        source: "jpeg decoder produced no image info after successful decode".into(),
    })?;

    let width = info.width as usize;
    let height = info.height as usize;

    // ── Convert pixels → JpegImage ───────────────────────────────────
    let (image, source_bit_depth) = pixels_to_image(pixels, width, height, info.pixel_format)?;

    // ── Build metadata ───────────────────────────────────────────────
    let metadata = build_metadata(&decoder, source_bit_depth, comments, pixel_density);

    Ok(JpegDecoded { image, metadata })
}

/// Decode a JPEG image from a streaming reader.
///
/// Buffers the entire stream into memory and then delegates to [`decode`],
/// so both paths produce identical [`JpegDecoded`] results — including
/// COM comments and JFIF pixel density.  `jpeg-decoder` already buffers
/// the entire stream internally for decoding, so buffering up-front adds
/// no net memory cost.
///
/// # Errors
///
/// Same error conditions as [`decode`], plus [`IoError::Io`] for read
/// failures.
///
/// # Examples
///
/// ```no_run
/// # use fovea_io::jpeg::{self, JpegImage};
/// let file = std::fs::File::open("photo.jpg").unwrap();
/// let reader = std::io::BufReader::new(file);
/// let decoded = jpeg::decode_reader(reader).unwrap();
///
/// match decoded.image {
///     JpegImage::Srgb8(image) => { /* work with Image<Srgb8> */ }
///     _ => { /* handle remaining variants */ }
/// }
/// ```
pub fn decode_reader(mut reader: impl std::io::Read) -> Result<JpegDecoded, IoError> {
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    decode(&buf)
}

/// Convert raw decoder output to a [`JpegImage`], returning the image and
/// the source bit depth.
fn pixels_to_image(
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    pixel_format: jpeg_decoder::PixelFormat,
) -> Result<(JpegImage, JpegBitDepth), IoError> {
    match pixel_format {
        jpeg_decoder::PixelFormat::L8 => {
            let img = Image::from_raw_bytes(width, height, pixels).map_err(|_| {
                IoError::DecodeFailed {
                    source: "pixel count does not match image dimensions (L8)".into(),
                }
            })?;
            Ok((JpegImage::SrgbMono8(img), JpegBitDepth::Eight))
        }
        jpeg_decoder::PixelFormat::L16 => {
            // jpeg-decoder stores 16-bit luminance as native-endian u16 pairs.
            let pixel_vec: Vec<SrgbMono16> = pixels
                .chunks_exact(2)
                .map(|b| SrgbMono16::new(u16::from_ne_bytes([b[0], b[1]])))
                .collect();
            let img =
                Image::from_vec(width, height, pixel_vec).map_err(|_| IoError::DecodeFailed {
                    source: "pixel count does not match image dimensions (L16)".into(),
                })?;
            Ok((JpegImage::SrgbMono16(img), JpegBitDepth::Twelve))
        }
        jpeg_decoder::PixelFormat::RGB24 => {
            let img = Image::from_raw_bytes(width, height, pixels).map_err(|_| {
                IoError::DecodeFailed {
                    source: "pixel count does not match image dimensions (RGB24)".into(),
                }
            })?;
            Ok((JpegImage::Srgb8(img), JpegBitDepth::Eight))
        }
        jpeg_decoder::PixelFormat::CMYK32 => Err(IoError::UnsupportedFeature {
            reason: "CMYK JPEG is not supported — convert to RGB before loading",
        }),
    }
}

/// Build a [`JpegMetadata`] from decoder state and pre-scanned data.
fn build_metadata<R: std::io::Read>(
    decoder: &jpeg_decoder::Decoder<R>,
    source_bit_depth: JpegBitDepth,
    comments: Vec<String>,
    pixel_density: Option<JpegPixelDensity>,
) -> JpegMetadata {
    // ── EXIF ─────────────────────────────────────────────────────────
    // jpeg-decoder returns EXIF data starting at the TIFF header (no
    // `Exif\0\0` prefix), so we use `parse_tiff_exif` directly.
    let raw_exif_data = decoder.exif_data();
    let exif = raw_exif_data.and_then(parse_tiff_exif);
    let raw_exif = raw_exif_data.map(|d| d.to_vec().into_boxed_slice());

    // ── ICC profile ──────────────────────────────────────────────────
    let icc_profile = decoder.icc_profile().map(|v| v.into_boxed_slice());

    // ── Colour space ─────────────────────────────────────────────────
    let color_space = if icc_profile.is_some() {
        JpegColorSpace::IccTagged
    } else {
        JpegColorSpace::Srgb
    };

    JpegMetadata {
        exif,
        raw_exif,
        icc_profile,
        pixel_density,
        comments,
        source_bit_depth,
        color_space,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Encoding — types
// ═══════════════════════════════════════════════════════════════════════════════

/// Chroma subsampling factor for JPEG encoding.
///
/// Controls the trade-off between colour fidelity and file size.
/// `F1x1` (4:4:4) preserves full chroma resolution; `F2x2` (4:2:0) halves
/// both dimensions, producing the smallest files.
///
/// # Examples
///
/// ```
/// use fovea_io::jpeg::JpegSamplingFactor;
///
/// let factor = JpegSamplingFactor::F1x1;
/// assert_eq!(factor, JpegSamplingFactor::F1x1);
/// assert_ne!(factor, JpegSamplingFactor::F2x2);
///
/// // Copy semantics:
/// let factor2 = factor;
/// assert_eq!(factor, factor2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegSamplingFactor {
    /// 4:4:4 — no chroma subsampling.  Best quality, largest file.
    F1x1,
    /// 4:2:2 — horizontal chroma halved.
    F2x1,
    /// 4:2:0 — both dimensions halved.  Smallest file.
    F2x2,
}

/// Options for JPEG encoding.
///
/// `#[non_exhaustive]` so that new fields (e.g. optimise Huffman tables)
/// can be added without a semver-major bump.
///
/// # Examples
///
/// ```
/// use fovea_io::jpeg::{JpegEncodeOptions, JpegSamplingFactor};
///
/// // Use defaults (quality 85, encoder-default sampling, not progressive):
/// let opts = JpegEncodeOptions::default();
/// assert_eq!(opts.quality, 85);
/// assert!(!opts.progressive);
///
/// // Custom options via mutation:
/// let mut opts = JpegEncodeOptions::default();
/// opts.quality = 95;
/// opts.sampling_factor = Some(JpegSamplingFactor::F1x1);
/// opts.progressive = true;
/// assert_eq!(opts.quality, 95);
/// ```
//
// Note: ICC profile (`APP2`) embedding and `COM` comment emission are
// intentionally **not** part of the encode options yet. The previous
// `icc_profile` / `comments` fields silently dropped their data (the
// encoder never wrote them), which violates design principles §4 and §7–§8
// (no silent data loss; I/O must preserve format fidelity). Because the
// struct is `#[non_exhaustive]`, those fields can be reintroduced with a
// real implementation later without a semver-major bump.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct JpegEncodeOptions {
    /// Quality factor (1–100).  Default: 85.
    pub quality: u8,
    /// Chroma subsampling.  Default: `None` (encoder default, typically 4:2:0).
    pub sampling_factor: Option<JpegSamplingFactor>,
    /// Progressive encoding.  Default: `false`.
    pub progressive: bool,
}

impl Default for JpegEncodeOptions {
    fn default() -> Self {
        Self {
            quality: 85,
            sampling_factor: None,
            progressive: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// JpegPixel — sealed encode trait (Phase 4, task 4.3)
// ═══════════════════════════════════════════════════════════════════════════════

mod jpeg_pixel_sealed {
    /// Sealed supertrait — prevents out-of-crate implementations of
    /// [`JpegPixel`](super::JpegPixel).
    pub trait Sealed {}
}

/// Maps a pixel type to its `jpeg-encoder` colour type.
///
/// This trait is **sealed**: it is implemented for exactly the pixel types
/// that JPEG can represent without lossy conversion.  Attempting to encode
/// a type that does not implement `JpegPixel` is a compile-time error.
///
/// # Implementors
///
/// | Pixel type  | `jpeg_encoder::ColorType` | Notes              |
/// |-------------|---------------------------|--------------------|
/// | `SrgbMono8` | `Luma`                    | Grayscale JPEG     |
/// | `Srgb8`     | `Rgb`                     | Standard RGB JPEG  |
///
/// `Srgba8` is deliberately **excluded**.  JPEG does not support alpha;
/// encoding `Srgba8` would silently discard the alpha channel — a lossy
/// conversion that violates design principle §4.  Strip alpha explicitly
/// before encoding.
///
/// `Rgb8` and `Mono8` (linear) are excluded because JPEG is sRGB by
/// definition.  Encoding linear data as sRGB would be a type-level lie.
///
/// # Compile-time enforcement
///
/// ```compile_fail
/// # use fovea::image::Image;
/// # use fovea::pixel::Rgb8;
/// # use fovea_io::jpeg::{self, JpegEncodeOptions};
/// // ERROR: `Rgb8` does not implement `JpegPixel`.
/// // JPEG is sRGB — encode `Srgb8` instead.
/// let img = Image::fill(1, 1, Rgb8::new(0, 0, 0));
/// let _ = jpeg::encode(&img, &JpegEncodeOptions::default());
/// ```
///
/// ```compile_fail
/// # use fovea::image::Image;
/// # use fovea::pixel::Mono8;
/// # use fovea_io::jpeg::{self, JpegEncodeOptions};
/// // ERROR: `Mono8` does not implement `JpegPixel`.
/// let img = Image::fill(1, 1, Mono8::new(0));
/// let _ = jpeg::encode(&img, &JpegEncodeOptions::default());
/// ```
///
/// ```compile_fail
/// # use fovea::image::Image;
/// # use fovea::pixel::Srgba8;
/// # use fovea_io::jpeg::{self, JpegEncodeOptions};
/// // ERROR: `Srgba8` does not implement `JpegPixel`.
/// // JPEG has no alpha channel — strip alpha first.
/// let img = Image::fill(1, 1, Srgba8::new(0, 0, 0, 255));
/// let _ = jpeg::encode(&img, &JpegEncodeOptions::default());
/// ```
///
/// ```compile_fail
/// # use fovea::image::Image;
/// # use fovea::pixel::Rgb16;
/// # use fovea_io::jpeg::{self, JpegEncodeOptions};
/// // ERROR: `Rgb16` does not implement `JpegPixel`.
/// let img = Image::fill(1, 1, Rgb16::new(0, 0, 0));
/// let _ = jpeg::encode(&img, &JpegEncodeOptions::default());
/// ```
pub trait JpegPixel: jpeg_pixel_sealed::Sealed + fovea::pixel::PlainPixel {
    /// `jpeg-encoder` colour type for this pixel.
    const JPEG_COLOR_TYPE: jpeg_encoder::ColorType;
}

macro_rules! impl_jpeg_pixel {
    ($ty:ty, $color:expr) => {
        impl jpeg_pixel_sealed::Sealed for $ty {}
        impl JpegPixel for $ty {
            const JPEG_COLOR_TYPE: jpeg_encoder::ColorType = $color;
        }
    };
}

impl_jpeg_pixel!(SrgbMono8, jpeg_encoder::ColorType::Luma);
impl_jpeg_pixel!(Srgb8, jpeg_encoder::ColorType::Rgb);

// ═══════════════════════════════════════════════════════════════════════════════
// Encoding — public API (Phase 4)
// ═══════════════════════════════════════════════════════════════════════════════

/// Map a `jpeg_encoder::EncodingError` into [`IoError`].
fn encode_error(e: jpeg_encoder::EncodingError) -> IoError {
    match e {
        jpeg_encoder::EncodingError::IoError(io) => IoError::Io(io),
        other => IoError::EncodeFailed {
            source: Box::new(other),
        },
    }
}

/// Encode an image to an in-memory JPEG byte vector.
///
/// Only pixel types that implement [`JpegPixel`] can be encoded — currently
/// [`SrgbMono8`] and [`Srgb8`].  Attempting to encode other types (e.g.
/// `Rgb8`, `Srgba8`) is a compile-time error.
///
/// # Errors
///
/// - [`IoError::EncodeFailed`] — the encoder reports an error (e.g. quality
///   out of range after clamping).
/// - [`IoError::Io`] — an I/O error during encoding (unlikely to `Vec<u8>`).
///
/// # Examples
///
/// ```no_run
/// # use fovea::image::Image;
/// # use fovea::pixel::Srgb8;
/// # use fovea_io::jpeg::{self, JpegEncodeOptions};
/// let image = Image::fill(320, 240, Srgb8::new(128, 64, 32));
/// let bytes = jpeg::encode(&image, &JpegEncodeOptions::default()).unwrap();
/// std::fs::write("output.jpg", bytes).unwrap();
/// ```
pub fn encode<P: JpegPixel>(
    image: &(impl fovea::image::ImageView<Pixel = P> + fovea::image::PlainImage),
    options: &JpegEncodeOptions,
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
/// # Errors
///
/// - [`IoError::EncodeFailed`] — the encoder reports an error.
/// - [`IoError::Io`] — the underlying writer fails.
///
/// # Examples
///
/// ```no_run
/// # use fovea::image::Image;
/// # use fovea::pixel::Srgb8;
/// # use fovea_io::jpeg::{self, JpegEncodeOptions};
/// let image = Image::fill(320, 240, Srgb8::new(128, 64, 32));
/// let mut out = Vec::new();
/// jpeg::encode_writer(&image, &mut out, &JpegEncodeOptions::default()).unwrap();
/// ```
pub fn encode_writer<P: JpegPixel>(
    image: &(impl fovea::image::ImageView<Pixel = P> + fovea::image::PlainImage),
    writer: impl std::io::Write,
    options: &JpegEncodeOptions,
) -> Result<(), IoError> {
    // ── JPEG spec dimension limit ────────────────────────────────────
    // The SOFn marker encodes width and height as 16-bit unsigned
    // big-endian values (ITU-T T.81 §B.2.2). Anything above 65535 cannot
    // be represented in a spec-compliant JPEG. A naked `as u16` cast
    // silently wraps (e.g. 65_536 → 0), which the encoder then rejects
    // as a zero-dimension image — a confusing error for the caller.
    // Validate up front and surface a clear, codec-agnostic error.
    if image.width() > u16::MAX as usize || image.height() > u16::MAX as usize {
        return Err(IoError::UnsupportedFeature {
            reason: "JPEG dimensions exceed 65535 (u16::MAX) per ITU-T T.81 §B.2.2",
        });
    }
    let width = image.width() as u16;
    let height = image.height() as u16;

    // Clamp quality to valid range (1–100).
    let quality = options.quality.clamp(1, 100);

    let mut encoder = jpeg_encoder::Encoder::new(writer, quality);

    // ── Sampling factor ──────────────────────────────────────────────
    if let Some(sf) = options.sampling_factor {
        let sampling = match sf {
            JpegSamplingFactor::F1x1 => jpeg_encoder::SamplingFactor::F_1_1,
            JpegSamplingFactor::F2x1 => jpeg_encoder::SamplingFactor::F_2_1,
            JpegSamplingFactor::F2x2 => jpeg_encoder::SamplingFactor::F_2_2,
        };
        encoder.set_sampling_factor(sampling);
    }

    // ── Progressive mode ─────────────────────────────────────────────
    if options.progressive {
        encoder.set_progressive(true);
    }

    // ── Write image data ─────────────────────────────────────────────
    let bytes: &[u8] = image.as_bytes();
    encoder
        .encode(bytes, width, height, P::JPEG_COLOR_TYPE)
        .map_err(encode_error)?;

    Ok(())
}

/// Encode a [`JpegImage`] back to JPEG bytes.
///
/// Convenience wrapper that dispatches over all [`JpegImage`] variants.
/// Useful for roundtripping and generic tooling.
///
/// Note: `SrgbMono16` (12-bit JPEG) cannot be re-encoded because
/// `jpeg-encoder` only supports 8-bit encoding.  This variant returns
/// [`IoError::UnsupportedFeature`].
///
/// # Errors
///
/// - [`IoError::EncodeFailed`] — the encoder reports an error.
/// - [`IoError::UnsupportedFeature`] — the image is `SrgbMono16` (16-bit
///   data cannot be encoded to baseline JPEG).
///
/// # Examples
///
/// ```no_run
/// # use fovea_io::jpeg::{self, JpegEncodeOptions};
/// let decoded = jpeg::decode(&std::fs::read("photo.jpg").unwrap()).unwrap();
/// let bytes = jpeg::encode_jpeg_image(&decoded.image, &JpegEncodeOptions::default()).unwrap();
/// std::fs::write("copy.jpg", bytes).unwrap();
/// ```
pub fn encode_jpeg_image(
    image: &JpegImage,
    options: &JpegEncodeOptions,
) -> Result<Vec<u8>, IoError> {
    match image {
        JpegImage::SrgbMono8(img) => encode(img, options),
        JpegImage::SrgbMono16(_) => Err(IoError::UnsupportedFeature {
            reason: "16-bit grayscale (12-bit JPEG) cannot be re-encoded to baseline JPEG",
        }),
        JpegImage::Srgb8(img) => encode(img, options),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fovea::image::ImageView;
    use std::mem;

    // ── Minimal JPEG builder (for decode tests) ──────────────────────

    /// Build a minimal valid JFIF JPEG byte stream.
    ///
    /// This generates a baseline JPEG with the given dimensions, pixel
    /// format (grayscale or RGB), and optional APP0/COM/APP1 segments.
    /// We use `jpeg-encoder` to produce valid entropy data, then
    /// optionally prepend extra markers.
    fn build_jpeg_rgb(width: u16, height: u16, r: u8, g: u8, b: u8) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let mut pixels = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
        let mut buf = Vec::new();
        let encoder = jpeg_encoder::Encoder::new(&mut buf, 90);
        encoder
            .encode(&pixels, width, height, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        buf
    }

    fn build_jpeg_gray(width: u16, height: u16, value: u8) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let pixels = vec![value; w * h];
        let mut buf = Vec::new();
        let encoder = jpeg_encoder::Encoder::new(&mut buf, 90);
        encoder
            .encode(&pixels, width, height, jpeg_encoder::ColorType::Luma)
            .unwrap();
        buf
    }

    /// Inject a COM marker into an existing JPEG byte stream right after SOI.
    fn inject_com_marker(jpeg: &[u8], text: &str) -> Vec<u8> {
        assert!(jpeg.len() >= 2 && jpeg[0] == 0xFF && jpeg[1] == 0xD8);
        let text_bytes = text.as_bytes();
        let seg_len = (text_bytes.len() + 2) as u16; // +2 for the length field
        let mut out = Vec::with_capacity(jpeg.len() + 4 + text_bytes.len());
        out.extend_from_slice(&jpeg[..2]); // SOI
        out.push(0xFF);
        out.push(0xFE); // COM marker
        out.extend_from_slice(&seg_len.to_be_bytes());
        out.extend_from_slice(text_bytes);
        out.extend_from_slice(&jpeg[2..]); // rest of JPEG
        out
    }

    /// Inject a JFIF APP0 segment with specified density into a JPEG.
    fn inject_jfif_app0(jpeg: &[u8], units: u8, x: u16, y: u16) -> Vec<u8> {
        assert!(jpeg.len() >= 2 && jpeg[0] == 0xFF && jpeg[1] == 0xD8);
        let mut segment = Vec::new();
        segment.extend_from_slice(b"JFIF\0"); // identifier
        segment.push(1); // version major
        segment.push(2); // version minor
        segment.push(units);
        segment.extend_from_slice(&x.to_be_bytes());
        segment.extend_from_slice(&y.to_be_bytes());
        segment.push(0); // thumbnail width
        segment.push(0); // thumbnail height
        let seg_len = (segment.len() + 2) as u16;
        let mut out = Vec::with_capacity(jpeg.len() + 4 + segment.len());
        out.extend_from_slice(&jpeg[..2]); // SOI
        out.push(0xFF);
        out.push(0xE0); // APP0 marker
        out.extend_from_slice(&seg_len.to_be_bytes());
        out.extend_from_slice(&segment);
        out.extend_from_slice(&jpeg[2..]); // rest
        out
    }

    // ── JpegImage — enum compactness ─────────────────────────────────────

    /// All three variants are `Image<T>` (thin pointer + size), so the
    /// enum should stay compact.
    #[test]
    fn jpeg_image_enum_is_compact() {
        let srgb8_size = mem::size_of::<Image<Srgb8>>();
        let mono8_size = mem::size_of::<Image<SrgbMono8>>();
        let mono16_size = mem::size_of::<Image<SrgbMono16>>();
        let enum_size = mem::size_of::<JpegImage>();

        // The enum should be at most the size of the largest variant + tag + padding.
        // All variants are the same underlying shape, so the enum shouldn't be
        // drastically larger than any single variant.
        let max_variant = srgb8_size.max(mono8_size).max(mono16_size);
        assert!(
            enum_size <= max_variant + 16,
            "JpegImage enum is unexpectedly large: {enum_size} bytes \
             (max variant is {max_variant} bytes)"
        );
    }

    // ── JpegImage — Debug impl ───────────────────────────────────────────

    #[test]
    fn jpeg_image_debug_srgb_mono8() {
        let img = JpegImage::SrgbMono8(Image::fill(10, 20, SrgbMono8::new(0)));
        assert_eq!(format!("{:?}", img), "SrgbMono8(10x20)");
    }

    #[test]
    fn jpeg_image_debug_srgb_mono16() {
        let img = JpegImage::SrgbMono16(Image::fill(5, 15, SrgbMono16::new(0)));
        assert_eq!(format!("{:?}", img), "SrgbMono16(5x15)");
    }

    #[test]
    fn jpeg_image_debug_srgb8() {
        let img = JpegImage::Srgb8(Image::fill(320, 240, Srgb8::new(0, 0, 0)));
        assert_eq!(format!("{:?}", img), "Srgb8(320x240)");
    }

    #[test]
    fn jpeg_image_debug_1x1() {
        let img = JpegImage::Srgb8(Image::fill(1, 1, Srgb8::new(0, 0, 0)));
        assert_eq!(format!("{:?}", img), "Srgb8(1x1)");
    }

    #[test]
    fn jpeg_image_debug_all_variants() {
        // Ensure all branches of the Debug match are covered.
        let variants: Vec<JpegImage> = vec![
            JpegImage::SrgbMono8(Image::fill(1, 1, SrgbMono8::new(0))),
            JpegImage::SrgbMono16(Image::fill(2, 3, SrgbMono16::new(0))),
            JpegImage::Srgb8(Image::fill(4, 5, Srgb8::new(0, 0, 0))),
        ];
        let expected = ["SrgbMono8(1x1)", "SrgbMono16(2x3)", "Srgb8(4x5)"];
        for (v, e) in variants.iter().zip(expected.iter()) {
            assert_eq!(format!("{:?}", v), *e);
        }
    }

    // ── JpegExifInfo — constructibility and default ──────────────────────

    #[test]
    fn jpeg_exif_info_default_all_none() {
        let info = JpegExifInfo::default();
        assert_eq!(info.orientation, None);
        assert_eq!(info.datetime, None);
        assert_eq!(info.datetime_original, None);
        assert_eq!(info.camera_make, None);
        assert_eq!(info.camera_model, None);
        assert_eq!(info.software, None);
        assert_eq!(info.exposure_time, None);
        assert_eq!(info.f_number, None);
        assert_eq!(info.iso_speed, None);
        assert_eq!(info.focal_length, None);
        assert_eq!(info.gps_latitude, None);
        assert_eq!(info.gps_longitude, None);
        assert_eq!(info.gps_altitude, None);
    }

    #[test]
    fn jpeg_exif_info_fully_populated() {
        let info = JpegExifInfo {
            orientation: Some(6),
            datetime: Some("2025:01:15 12:30:00".to_string()),
            datetime_original: Some("2025:01:15 12:29:59".to_string()),
            camera_make: Some("Canon".to_string()),
            camera_model: Some("EOS R5".to_string()),
            software: Some("Lightroom 13.0".to_string()),
            exposure_time: Some((1, 250)),
            f_number: Some((28, 10)),
            iso_speed: Some(400),
            focal_length: Some((50, 1)),
            gps_latitude: Some(48.8566),
            gps_longitude: Some(2.3522),
            gps_altitude: Some(35.0),
        };
        assert_eq!(info.orientation, Some(6));
        assert_eq!(info.datetime.as_deref(), Some("2025:01:15 12:30:00"));
        assert_eq!(
            info.datetime_original.as_deref(),
            Some("2025:01:15 12:29:59")
        );
        assert_eq!(info.camera_make.as_deref(), Some("Canon"));
        assert_eq!(info.camera_model.as_deref(), Some("EOS R5"));
        assert_eq!(info.software.as_deref(), Some("Lightroom 13.0"));
        assert_eq!(info.exposure_time, Some((1, 250)));
        assert_eq!(info.f_number, Some((28, 10)));
        assert_eq!(info.iso_speed, Some(400));
        assert_eq!(info.focal_length, Some((50, 1)));
        assert!((info.gps_latitude.unwrap() - 48.8566).abs() < 1e-10);
        assert!((info.gps_longitude.unwrap() - 2.3522).abs() < 1e-10);
        assert!((info.gps_altitude.unwrap() - 35.0).abs() < 1e-10);
    }

    #[test]
    fn jpeg_exif_info_partial_fields() {
        let info = JpegExifInfo {
            orientation: Some(1),
            camera_make: Some("Nikon".to_string()),
            ..Default::default()
        };
        assert_eq!(info.orientation, Some(1));
        assert_eq!(info.camera_make.as_deref(), Some("Nikon"));
        assert_eq!(info.camera_model, None);
        assert_eq!(info.exposure_time, None);
        assert_eq!(info.gps_latitude, None);
    }

    #[test]
    fn jpeg_exif_info_clone() {
        let info = JpegExifInfo {
            orientation: Some(3),
            datetime: Some("2025:06:01 08:00:00".to_string()),
            ..Default::default()
        };
        let cloned = info.clone();
        assert_eq!(info, cloned);
    }

    #[test]
    fn jpeg_exif_info_debug() {
        let info = JpegExifInfo::default();
        let dbg = format!("{:?}", info);
        assert!(dbg.contains("JpegExifInfo"));
        assert!(dbg.contains("orientation: None"));
    }

    #[test]
    fn jpeg_exif_info_partial_eq() {
        let a = JpegExifInfo {
            orientation: Some(1),
            ..Default::default()
        };
        let b = JpegExifInfo {
            orientation: Some(1),
            ..Default::default()
        };
        let c = JpegExifInfo {
            orientation: Some(2),
            ..Default::default()
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn jpeg_exif_info_gps_negative_values() {
        // South latitude, West longitude, below sea level.
        let info = JpegExifInfo {
            gps_latitude: Some(-33.8688),
            gps_longitude: Some(-151.2093),
            gps_altitude: Some(-10.5),
            ..Default::default()
        };
        assert!(info.gps_latitude.unwrap() < 0.0);
        assert!(info.gps_longitude.unwrap() < 0.0);
        assert!(info.gps_altitude.unwrap() < 0.0);
    }

    #[test]
    fn jpeg_exif_info_exposure_rational_precision() {
        // Ensure rationals preserve exact fractions.
        let info = JpegExifInfo {
            exposure_time: Some((1, 8000)),
            f_number: Some((14, 10)),
            focal_length: Some((200, 1)),
            ..Default::default()
        };
        let (num, den) = info.exposure_time.unwrap();
        assert_eq!(num, 1);
        assert_eq!(den, 8000);
        // User can compute f64:
        let exposure_secs = num as f64 / den as f64;
        assert!((exposure_secs - 0.000125).abs() < 1e-10);
    }

    // ── JpegColorSpace — variants and traits ─────────────────────────────

    #[test]
    fn jpeg_color_space_variants_constructible() {
        let srgb = JpegColorSpace::Srgb;
        let icc = JpegColorSpace::IccTagged;
        assert_eq!(srgb, JpegColorSpace::Srgb);
        assert_eq!(icc, JpegColorSpace::IccTagged);
        assert_ne!(srgb, icc);
    }

    #[test]
    fn jpeg_color_space_is_copy() {
        let cs = JpegColorSpace::Srgb;
        let cs2 = cs;
        assert_eq!(cs, cs2);
    }

    #[test]
    fn jpeg_color_space_debug() {
        assert_eq!(format!("{:?}", JpegColorSpace::Srgb), "Srgb");
        assert_eq!(format!("{:?}", JpegColorSpace::IccTagged), "IccTagged");
    }

    #[test]
    fn jpeg_color_space_clone() {
        let cs = JpegColorSpace::IccTagged;
        let cloned = cs.clone();
        assert_eq!(cs, cloned);
    }

    // ── JpegPixelDensity — variants and traits ───────────────────────────

    #[test]
    fn jpeg_pixel_density_dpi() {
        let d = JpegPixelDensity::Dpi { x: 300, y: 300 };
        match d {
            JpegPixelDensity::Dpi { x, y } => {
                assert_eq!(x, 300);
                assert_eq!(y, 300);
            }
            _ => panic!("expected Dpi"),
        }
    }

    #[test]
    fn jpeg_pixel_density_dpcm() {
        let d = JpegPixelDensity::Dpcm { x: 118, y: 118 };
        match d {
            JpegPixelDensity::Dpcm { x, y } => {
                assert_eq!(x, 118);
                assert_eq!(y, 118);
            }
            _ => panic!("expected Dpcm"),
        }
    }

    #[test]
    fn jpeg_pixel_density_aspect_ratio() {
        let d = JpegPixelDensity::AspectRatio { x: 1, y: 2 };
        match d {
            JpegPixelDensity::AspectRatio { x, y } => {
                assert_eq!(x, 1);
                assert_eq!(y, 2);
            }
            _ => panic!("expected AspectRatio"),
        }
    }

    #[test]
    fn jpeg_pixel_density_is_copy() {
        let d = JpegPixelDensity::Dpi { x: 72, y: 72 };
        let d2 = d;
        assert_eq!(d, d2);
    }

    #[test]
    fn jpeg_pixel_density_debug() {
        let d = JpegPixelDensity::Dpi { x: 300, y: 300 };
        let dbg = format!("{:?}", d);
        assert!(dbg.contains("Dpi"));
        assert!(dbg.contains("300"));
    }

    #[test]
    fn jpeg_pixel_density_eq() {
        let a = JpegPixelDensity::Dpi { x: 300, y: 300 };
        let b = JpegPixelDensity::Dpi { x: 300, y: 300 };
        let c = JpegPixelDensity::Dpi { x: 72, y: 72 };
        let d = JpegPixelDensity::Dpcm { x: 300, y: 300 };
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn jpeg_pixel_density_non_square() {
        let d = JpegPixelDensity::Dpi { x: 300, y: 600 };
        match d {
            JpegPixelDensity::Dpi { x, y } => {
                assert_eq!(x, 300);
                assert_eq!(y, 600);
            }
            _ => panic!("expected Dpi"),
        }
    }

    #[test]
    fn jpeg_pixel_density_all_variants_debug() {
        // Cover all Debug arms.
        let _ = format!("{:?}", JpegPixelDensity::Dpi { x: 1, y: 1 });
        let _ = format!("{:?}", JpegPixelDensity::Dpcm { x: 1, y: 1 });
        let _ = format!("{:?}", JpegPixelDensity::AspectRatio { x: 1, y: 1 });
    }

    // ── JpegMetadata — constructibility ──────────────────────────────────

    fn make_minimal_metadata() -> JpegMetadata {
        JpegMetadata {
            exif: None,
            raw_exif: None,
            icc_profile: None,
            pixel_density: None,
            comments: vec![],
            source_bit_depth: JpegBitDepth::Eight,
            color_space: JpegColorSpace::Srgb,
        }
    }

    #[test]
    fn jpeg_metadata_minimal() {
        let meta = make_minimal_metadata();
        assert!(meta.exif.is_none());
        assert!(meta.raw_exif.is_none());
        assert!(meta.icc_profile.is_none());
        assert!(meta.pixel_density.is_none());
        assert!(meta.comments.is_empty());
        assert_eq!(meta.source_bit_depth, JpegBitDepth::Eight);
        assert_eq!(meta.color_space, JpegColorSpace::Srgb);
    }

    #[test]
    fn jpeg_metadata_fully_populated() {
        let exif = JpegExifInfo {
            orientation: Some(1),
            camera_make: Some("Sony".to_string()),
            ..Default::default()
        };
        let meta = JpegMetadata {
            exif: Some(exif),
            raw_exif: Some(vec![0x45, 0x78, 0x69, 0x66].into_boxed_slice()),
            icc_profile: Some(vec![0u8; 128].into_boxed_slice()),
            pixel_density: Some(JpegPixelDensity::Dpi { x: 300, y: 300 }),
            comments: vec!["test comment".to_string(), "another".to_string()],
            source_bit_depth: JpegBitDepth::Twelve,
            color_space: JpegColorSpace::IccTagged,
        };
        assert!(meta.exif.is_some());
        assert_eq!(meta.exif.as_ref().unwrap().orientation, Some(1));
        assert!(meta.raw_exif.is_some());
        assert!(meta.icc_profile.is_some());
        assert_eq!(meta.icc_profile.as_ref().unwrap().len(), 128);
        assert_eq!(
            meta.pixel_density,
            Some(JpegPixelDensity::Dpi { x: 300, y: 300 })
        );
        assert_eq!(meta.comments.len(), 2);
        assert_eq!(meta.comments[0], "test comment");
        assert_eq!(meta.source_bit_depth, JpegBitDepth::Twelve);
        assert_eq!(meta.color_space, JpegColorSpace::IccTagged);
    }

    #[test]
    fn jpeg_metadata_with_12bit_depth() {
        let meta = JpegMetadata {
            source_bit_depth: JpegBitDepth::Twelve,
            ..make_minimal_metadata()
        };
        assert_eq!(meta.source_bit_depth, JpegBitDepth::Twelve);
    }

    #[test]
    fn jpeg_metadata_clone() {
        let meta = JpegMetadata {
            exif: Some(JpegExifInfo {
                orientation: Some(3),
                ..Default::default()
            }),
            comments: vec!["hello".to_string()],
            ..make_minimal_metadata()
        };
        let cloned = meta.clone();
        assert_eq!(
            cloned.exif.as_ref().unwrap().orientation,
            meta.exif.as_ref().unwrap().orientation
        );
        assert_eq!(cloned.comments, meta.comments);
        assert_eq!(cloned.source_bit_depth, meta.source_bit_depth);
        assert_eq!(cloned.color_space, meta.color_space);
    }

    #[test]
    fn jpeg_metadata_debug() {
        let meta = make_minimal_metadata();
        let dbg = format!("{:?}", meta);
        assert!(dbg.contains("JpegMetadata"));
        assert!(dbg.contains("source_bit_depth"));
    }

    #[test]
    fn jpeg_metadata_icc_profile_boxed_slice() {
        // Verify ICC profile uses boxed slice, not Vec.
        let profile_data: Box<[u8]> = vec![1, 2, 3, 4].into_boxed_slice();
        let meta = JpegMetadata {
            icc_profile: Some(profile_data),
            ..make_minimal_metadata()
        };
        assert_eq!(meta.icc_profile.as_ref().unwrap().len(), 4);
        assert_eq!(meta.icc_profile.as_ref().unwrap()[0], 1);
    }

    #[test]
    fn jpeg_metadata_raw_exif_boxed_slice() {
        let raw: Box<[u8]> = vec![0x45, 0x78, 0x69, 0x66, 0x00, 0x00].into_boxed_slice();
        let meta = JpegMetadata {
            raw_exif: Some(raw),
            ..make_minimal_metadata()
        };
        assert_eq!(meta.raw_exif.as_ref().unwrap().len(), 6);
    }

    #[test]
    fn jpeg_metadata_multiple_comments() {
        let meta = JpegMetadata {
            comments: vec![
                "comment 1".to_string(),
                "comment 2".to_string(),
                "comment 3".to_string(),
            ],
            ..make_minimal_metadata()
        };
        assert_eq!(meta.comments.len(), 3);
        assert_eq!(meta.comments[2], "comment 3");
    }

    // ── JpegDecoded — constructibility ───────────────────────────────────

    #[test]
    fn jpeg_decoded_field_access() {
        let decoded = JpegDecoded {
            image: JpegImage::Srgb8(Image::fill(1, 1, Srgb8::new(0, 0, 0))),
            metadata: make_minimal_metadata(),
        };
        // Fields are directly accessible:
        let _img = decoded.image;
        let _meta = decoded.metadata;
    }

    #[test]
    fn jpeg_decoded_with_srgb_mono8() {
        let decoded = JpegDecoded {
            image: JpegImage::SrgbMono8(Image::fill(10, 10, SrgbMono8::new(128))),
            metadata: make_minimal_metadata(),
        };
        match &decoded.image {
            JpegImage::SrgbMono8(img) => {
                assert_eq!(img.width(), 10);
                assert_eq!(img.height(), 10);
            }
            _ => panic!("expected SrgbMono8"),
        }
    }

    #[test]
    fn jpeg_decoded_with_srgb_mono16() {
        let decoded = JpegDecoded {
            image: JpegImage::SrgbMono16(Image::fill(5, 5, SrgbMono16::new(1024))),
            metadata: JpegMetadata {
                source_bit_depth: JpegBitDepth::Twelve,
                ..make_minimal_metadata()
            },
        };
        match &decoded.image {
            JpegImage::SrgbMono16(img) => {
                assert_eq!(img.width(), 5);
                assert_eq!(img.height(), 5);
            }
            _ => panic!("expected SrgbMono16"),
        }
        assert_eq!(decoded.metadata.source_bit_depth, JpegBitDepth::Twelve);
    }

    #[test]
    fn jpeg_decoded_with_full_metadata() {
        let img = Image::fill(1, 1, Srgb8::new(0, 0, 0));
        let decoded = JpegDecoded {
            image: JpegImage::Srgb8(img),
            metadata: JpegMetadata {
                exif: Some(JpegExifInfo {
                    orientation: Some(6),
                    camera_make: Some("TestCam".to_string()),
                    ..Default::default()
                }),
                raw_exif: Some(vec![0xAA, 0xBB].into_boxed_slice()),
                icc_profile: Some(vec![0xCC, 0xDD].into_boxed_slice()),
                pixel_density: Some(JpegPixelDensity::Dpi { x: 300, y: 300 }),
                comments: vec!["photo comment".to_string()],
                source_bit_depth: JpegBitDepth::Eight,
                color_space: JpegColorSpace::IccTagged,
            },
        };
        match &decoded.image {
            JpegImage::Srgb8(img) => {
                assert_eq!(img.width(), 1);
                assert_eq!(img.height(), 1);
            }
            _ => panic!("expected Srgb8"),
        }
        let exif = decoded.metadata.exif.as_ref().unwrap();
        assert_eq!(exif.orientation, Some(6));
        assert_eq!(exif.camera_make.as_deref(), Some("TestCam"));
        assert_eq!(decoded.metadata.color_space, JpegColorSpace::IccTagged);
    }

    #[test]
    fn jpeg_decoded_debug() {
        let decoded = JpegDecoded {
            image: JpegImage::Srgb8(Image::fill(2, 2, Srgb8::new(0, 0, 0))),
            metadata: make_minimal_metadata(),
        };
        let dbg = format!("{:?}", decoded);
        assert!(dbg.contains("JpegDecoded"));
        assert!(dbg.contains("Srgb8(2x2)"));
        assert!(dbg.contains("JpegMetadata"));
    }

    // ── Exhaustive matching — compile-time guarantee ─────────────────────

    /// Verify that JpegImage supports exhaustive matching (no wildcard needed).
    #[test]
    fn jpeg_image_exhaustive_match() {
        let img = JpegImage::Srgb8(Image::fill(1, 1, Srgb8::new(0, 0, 0)));
        let name = match &img {
            JpegImage::SrgbMono8(_) => "SrgbMono8",
            JpegImage::SrgbMono16(_) => "SrgbMono16",
            JpegImage::Srgb8(_) => "Srgb8",
        };
        assert_eq!(name, "Srgb8");
    }

    /// Verify that JpegColorSpace supports exhaustive matching.
    #[test]
    fn jpeg_color_space_exhaustive_match() {
        let cs = JpegColorSpace::Srgb;
        let name = match cs {
            JpegColorSpace::Srgb => "Srgb",
            JpegColorSpace::IccTagged => "IccTagged",
        };
        assert_eq!(name, "Srgb");
    }

    /// Verify that JpegPixelDensity supports exhaustive matching.
    #[test]
    fn jpeg_pixel_density_exhaustive_match() {
        let d = JpegPixelDensity::Dpi { x: 72, y: 72 };
        let kind = match d {
            JpegPixelDensity::Dpi { .. } => "dpi",
            JpegPixelDensity::Dpcm { .. } => "dpcm",
            JpegPixelDensity::AspectRatio { .. } => "aspect",
        };
        assert_eq!(kind, "dpi");
    }

    // ── Type sizes ───────────────────────────────────────────────────────

    #[test]
    fn jpeg_color_space_is_small() {
        // Two-variant enum with no data — should be 1 byte.
        assert!(mem::size_of::<JpegColorSpace>() <= 2);
    }

    #[test]
    fn jpeg_pixel_density_is_small() {
        // Three variants each with two u16 fields + discriminant.
        // Should be compact (≤8 bytes).
        assert!(mem::size_of::<JpegPixelDensity>() <= 8);
    }

    #[test]
    fn jpeg_metadata_is_reasonable_size() {
        // JpegMetadata contains heap-allocated optionals; the struct itself
        // should be reasonably sized (no large inline data).
        let size = mem::size_of::<JpegMetadata>();
        // Should be well under 512 bytes for the struct itself.
        // The struct carries several Option<Box<[u8]>>, Option<JpegExifInfo>,
        // Vec<String>, etc. — all heap-allocated, but the struct shell with
        // its fat pointers and discriminants can reach ~280 bytes on 64-bit.
        assert!(
            size < 512,
            "JpegMetadata is {size} bytes — expected under 512"
        );
    }

    // ── Variant dispatch helpers ─────────────────────────────────────────

    /// Helper to get variant name as a string — useful for testing.
    fn variant_name(img: &JpegImage) -> &'static str {
        match img {
            JpegImage::SrgbMono8(_) => "SrgbMono8",
            JpegImage::SrgbMono16(_) => "SrgbMono16",
            JpegImage::Srgb8(_) => "Srgb8",
        }
    }

    #[test]
    fn variant_name_covers_all() {
        assert_eq!(
            variant_name(&JpegImage::SrgbMono8(Image::fill(1, 1, SrgbMono8::new(0)))),
            "SrgbMono8"
        );
        assert_eq!(
            variant_name(&JpegImage::SrgbMono16(Image::fill(
                1,
                1,
                SrgbMono16::new(0)
            ))),
            "SrgbMono16"
        );
        assert_eq!(
            variant_name(&JpegImage::Srgb8(Image::fill(1, 1, Srgb8::new(0, 0, 0)))),
            "Srgb8"
        );
    }

    // ── Image dimensions through JpegImage ───────────────────────────────

    #[test]
    fn jpeg_image_preserves_dimensions_mono8() {
        let img = JpegImage::SrgbMono8(Image::fill(100, 200, SrgbMono8::new(42)));
        match &img {
            JpegImage::SrgbMono8(i) => {
                assert_eq!(i.width(), 100);
                assert_eq!(i.height(), 200);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn jpeg_image_preserves_dimensions_mono16() {
        let img = JpegImage::SrgbMono16(Image::fill(50, 75, SrgbMono16::new(1000)));
        match &img {
            JpegImage::SrgbMono16(i) => {
                assert_eq!(i.width(), 50);
                assert_eq!(i.height(), 75);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn jpeg_image_preserves_dimensions_srgb8() {
        let img = JpegImage::Srgb8(Image::fill(1920, 1080, Srgb8::new(128, 64, 32)));
        match &img {
            JpegImage::Srgb8(i) => {
                assert_eq!(i.width(), 1920);
                assert_eq!(i.height(), 1080);
            }
            _ => panic!("wrong variant"),
        }
    }

    // ── Pixel data access through JpegImage ──────────────────────────────

    #[test]
    fn jpeg_image_pixel_data_accessible() {
        let img = JpegImage::Srgb8(Image::generate(3, 2, |x, y| {
            Srgb8::new(x as u8, y as u8, 0)
        }));
        match &img {
            JpegImage::Srgb8(i) => {
                assert_eq!(i.get(0, 0).unwrap().r.0, 0);
                assert_eq!(i.get(0, 0).unwrap().g.0, 0);
                assert_eq!(i.get(2, 1).unwrap().r.0, 2);
                assert_eq!(i.get(2, 1).unwrap().g.0, 1);
            }
            _ => panic!("wrong variant"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // TiffReader tests (task 2.1)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn tiff_reader_u8_at() {
        let data = [0xAA, 0xBB, 0xCC];
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(r.u8_at(0), Some(0xAA));
        assert_eq!(r.u8_at(1), Some(0xBB));
        assert_eq!(r.u8_at(2), Some(0xCC));
        assert_eq!(r.u8_at(3), None);
    }

    #[test]
    fn tiff_reader_u16_little_endian() {
        let data = [0x34, 0x12]; // LE: 0x1234
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(r.u16_at(0), Some(0x1234));
    }

    #[test]
    fn tiff_reader_u16_big_endian() {
        let data = [0x12, 0x34]; // BE: 0x1234
        let r = TiffReader::new(&data, ByteOrder::Big);
        assert_eq!(r.u16_at(0), Some(0x1234));
    }

    #[test]
    fn tiff_reader_u16_out_of_bounds() {
        let data = [0x12];
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(r.u16_at(0), None);
    }

    #[test]
    fn tiff_reader_u16_at_offset() {
        let data = [0x00, 0x00, 0x78, 0x56]; // LE at offset 2: 0x5678
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(r.u16_at(2), Some(0x5678));
    }

    #[test]
    fn tiff_reader_u32_little_endian() {
        let data = [0x78, 0x56, 0x34, 0x12]; // LE: 0x12345678
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(r.u32_at(0), Some(0x12345678));
    }

    #[test]
    fn tiff_reader_u32_big_endian() {
        let data = [0x12, 0x34, 0x56, 0x78]; // BE: 0x12345678
        let r = TiffReader::new(&data, ByteOrder::Big);
        assert_eq!(r.u32_at(0), Some(0x12345678));
    }

    #[test]
    fn tiff_reader_u32_out_of_bounds() {
        let data = [0x12, 0x34, 0x56];
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(r.u32_at(0), None);
    }

    #[test]
    fn tiff_reader_u32_partial_overlap() {
        // 5 bytes — u32 at offset 0 works, at offset 2 doesn't
        let data = [0x01, 0x02, 0x03, 0x04, 0x05];
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert!(r.u32_at(0).is_some());
        assert_eq!(r.u32_at(2), None);
    }

    #[test]
    fn tiff_reader_rational_little_endian() {
        // num = 1 (LE), den = 250 (LE)
        let mut data = vec![];
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&250u32.to_le_bytes());
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(r.rational_at(0), Some((1, 250)));
    }

    #[test]
    fn tiff_reader_rational_big_endian() {
        let mut data = vec![];
        data.extend_from_slice(&1u32.to_be_bytes());
        data.extend_from_slice(&250u32.to_be_bytes());
        let r = TiffReader::new(&data, ByteOrder::Big);
        assert_eq!(r.rational_at(0), Some((1, 250)));
    }

    #[test]
    fn tiff_reader_rational_out_of_bounds() {
        // Only 7 bytes — need 8 for a rational
        let data = [0u8; 7];
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(r.rational_at(0), None);
    }

    #[test]
    fn tiff_reader_rational_at_offset() {
        // Pad 4 bytes, then rational at offset 4
        let mut data = vec![0u8; 4];
        data.extend_from_slice(&50u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(r.rational_at(4), Some((50, 1)));
    }

    #[test]
    fn tiff_reader_empty_data() {
        let data: [u8; 0] = [];
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(r.u8_at(0), None);
        assert_eq!(r.u16_at(0), None);
        assert_eq!(r.u32_at(0), None);
        assert_eq!(r.rational_at(0), None);
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn tiff_reader_len() {
        let data = [1, 2, 3, 4, 5];
        let r = TiffReader::new(&data, ByteOrder::Big);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn tiff_reader_u16_max_offset_no_overflow() {
        // Ensure we don't panic on usize near-overflow
        let data = [0xFFu8; 4];
        let r = TiffReader::new(&data, ByteOrder::Little);
        // Very large offset — should return None, not panic
        assert_eq!(r.u16_at(usize::MAX), None);
        assert_eq!(r.u32_at(usize::MAX), None);
        assert_eq!(r.rational_at(usize::MAX), None);
    }

    #[test]
    fn byte_order_debug_and_eq() {
        let le = ByteOrder::Little;
        let be = ByteOrder::Big;
        assert_eq!(le, ByteOrder::Little);
        assert_ne!(le, be);
        assert_eq!(format!("{:?}", le), "Little");
        assert_eq!(format!("{:?}", be), "Big");
        // Copy
        let le2 = le;
        assert_eq!(le, le2);
    }

    // ═════════════════════════════════════════════════════════════════════
    // IFD entry reader tests (task 2.2)
    // ═════════════════════════════════════════════════════════════════════

    /// Build a minimal IFD blob at offset 0: count (2 bytes) + N entries (12 bytes each).
    fn build_ifd_le(entries: &[(u16, u16, u32, u32)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for &(tag, typ, count, val) in entries {
            buf.extend_from_slice(&tag.to_le_bytes());
            buf.extend_from_slice(&typ.to_le_bytes());
            buf.extend_from_slice(&count.to_le_bytes());
            buf.extend_from_slice(&val.to_le_bytes());
        }
        buf
    }

    fn build_ifd_be(entries: &[(u16, u16, u32, u32)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(entries.len() as u16).to_be_bytes());
        for &(tag, typ, count, val) in entries {
            buf.extend_from_slice(&tag.to_be_bytes());
            buf.extend_from_slice(&typ.to_be_bytes());
            buf.extend_from_slice(&count.to_be_bytes());
            buf.extend_from_slice(&val.to_be_bytes());
        }
        buf
    }

    #[test]
    fn read_ifd_single_entry_le() {
        let data = build_ifd_le(&[(0x010F, 2, 6, 100)]);
        let r = TiffReader::new(&data, ByteOrder::Little);
        let entries = read_ifd_entries(&r, 0);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0],
            IfdEntry {
                tag: 0x010F,
                tiff_type: 2,
                count: 6,
                value_or_offset: 100
            }
        );
    }

    #[test]
    fn read_ifd_single_entry_be() {
        let data = build_ifd_be(&[(0x010F, 2, 6, 100)]);
        let r = TiffReader::new(&data, ByteOrder::Big);
        let entries = read_ifd_entries(&r, 0);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0],
            IfdEntry {
                tag: 0x010F,
                tiff_type: 2,
                count: 6,
                value_or_offset: 100
            }
        );
    }

    #[test]
    fn read_ifd_multiple_entries() {
        let data = build_ifd_le(&[(0x010F, 2, 6, 100), (0x0110, 2, 10, 200), (0x0112, 3, 1, 6)]);
        let r = TiffReader::new(&data, ByteOrder::Little);
        let entries = read_ifd_entries(&r, 0);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].tag, 0x010F);
        assert_eq!(entries[1].tag, 0x0110);
        assert_eq!(entries[2].tag, 0x0112);
        assert_eq!(entries[2].value_or_offset, 6); // orientation value
    }

    #[test]
    fn read_ifd_zero_entries() {
        let data = [0u8, 0]; // count = 0
        let r = TiffReader::new(&data, ByteOrder::Little);
        let entries = read_ifd_entries(&r, 0);
        assert!(entries.is_empty());
    }

    #[test]
    fn read_ifd_empty_data() {
        let data: [u8; 0] = [];
        let r = TiffReader::new(&data, ByteOrder::Little);
        let entries = read_ifd_entries(&r, 0);
        assert!(entries.is_empty());
    }

    #[test]
    fn read_ifd_out_of_bounds_offset() {
        let data = build_ifd_le(&[(0x010F, 2, 6, 100)]);
        let r = TiffReader::new(&data, ByteOrder::Little);
        let entries = read_ifd_entries(&r, 9999);
        assert!(entries.is_empty());
    }

    #[test]
    fn read_ifd_truncated_entry() {
        // Build 2-entry IFD but truncate after the first entry
        let mut data = build_ifd_le(&[(0x010F, 2, 6, 100), (0x0110, 2, 10, 200)]);
        data.truncate(2 + 12 + 6); // count + 1 full entry + partial second
        let r = TiffReader::new(&data, ByteOrder::Little);
        let entries = read_ifd_entries(&r, 0);
        // Should get only the first complete entry
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn read_ifd_capped_at_max() {
        // Craft data that claims 2000 entries but only has bytes for a few
        let mut data = Vec::new();
        data.extend_from_slice(&2000u16.to_le_bytes()); // claim 2000
        // Only provide 2 entries worth of data
        for _ in 0..2 {
            data.extend_from_slice(&0x0100u16.to_le_bytes());
            data.extend_from_slice(&3u16.to_le_bytes());
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&42u32.to_le_bytes());
        }
        let r = TiffReader::new(&data, ByteOrder::Little);
        let entries = read_ifd_entries(&r, 0);
        // Capped at MAX_IFD_ENTRIES (1000), but only 2 entries' worth of data,
        // so we should get exactly 2 before hitting out-of-bounds.
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn read_ifd_at_nonzero_offset() {
        // 8 bytes of padding, then the IFD
        let mut data = vec![0u8; 8];
        data.extend_from_slice(&build_ifd_le(&[(0x0112, 3, 1, 1)]));
        let r = TiffReader::new(&data, ByteOrder::Little);
        let entries = read_ifd_entries(&r, 8);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0],
            IfdEntry {
                tag: 0x0112,
                tiff_type: 3,
                count: 1,
                value_or_offset: 1
            }
        );
    }

    #[test]
    fn read_ifd_preserves_value_or_offset() {
        // SHORT type (3), count 1 — value fits inline (value_or_offset = 6)
        // LONG type (4), count 1 — value fits inline (value_or_offset = 99999)
        // ASCII type (2), count 20 — value is an offset (value_or_offset = 500)
        let data = build_ifd_le(&[
            (0x0112, 3, 1, 6),
            (0x8769, 4, 1, 99999),
            (0x010F, 2, 20, 500),
        ]);
        let r = TiffReader::new(&data, ByteOrder::Little);
        let entries = read_ifd_entries(&r, 0);
        assert_eq!(
            entries[0],
            IfdEntry {
                tag: 0x0112,
                tiff_type: 3,
                count: 1,
                value_or_offset: 6
            }
        );
        assert_eq!(
            entries[1],
            IfdEntry {
                tag: 0x8769,
                tiff_type: 4,
                count: 1,
                value_or_offset: 99999
            }
        );
        assert_eq!(
            entries[2],
            IfdEntry {
                tag: 0x010F,
                tiff_type: 2,
                count: 20,
                value_or_offset: 500
            }
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // ASCII string reader tests (task 2.3)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn read_ascii_simple() {
        let data = b"Canon\0";
        let r = TiffReader::new(data, ByteOrder::Little);
        assert_eq!(read_ascii(&r, 0, 6), Some("Canon".to_string()));
    }

    #[test]
    fn read_ascii_no_nul() {
        // Some malformed EXIF may not NUL-terminate
        let data = b"Nikon";
        let r = TiffReader::new(data, ByteOrder::Little);
        assert_eq!(read_ascii(&r, 0, 5), Some("Nikon".to_string()));
    }

    #[test]
    fn read_ascii_multiple_trailing_nuls() {
        let data = b"Test\0\0\0";
        let r = TiffReader::new(data, ByteOrder::Little);
        assert_eq!(read_ascii(&r, 0, 7), Some("Test".to_string()));
    }

    #[test]
    fn read_ascii_all_nuls() {
        let data = [0u8; 4];
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(read_ascii(&r, 0, 4), None);
    }

    #[test]
    fn read_ascii_empty_count() {
        let data = b"Hello\0";
        let r = TiffReader::new(data, ByteOrder::Little);
        assert_eq!(read_ascii(&r, 0, 0), None);
    }

    #[test]
    fn read_ascii_out_of_bounds() {
        let data = b"Hi\0";
        let r = TiffReader::new(data, ByteOrder::Little);
        assert_eq!(read_ascii(&r, 0, 10), None);
    }

    #[test]
    fn read_ascii_at_offset() {
        let mut data = vec![0u8; 10];
        data.extend_from_slice(b"Sony\0");
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(read_ascii(&r, 10, 5), Some("Sony".to_string()));
    }

    #[test]
    fn read_ascii_datetime_format() {
        let data = b"2025:01:15 12:30:00\0";
        let r = TiffReader::new(data, ByteOrder::Little);
        let result = read_ascii(&r, 0, 20);
        assert_eq!(result, Some("2025:01:15 12:30:00".to_string()));
    }

    #[test]
    fn read_ascii_utf8_passthrough() {
        // UTF-8 encoded string (German umlaut)
        let data = "Müller\0".as_bytes();
        let r = TiffReader::new(data, ByteOrder::Little);
        let result = read_ascii(&r, 0, data.len() as u32);
        assert_eq!(result, Some("Müller".to_string()));
    }

    #[test]
    fn read_ascii_latin1_fallback() {
        // Latin-1: 0xFC = ü, 0xE4 = ä — not valid UTF-8 as a sequence
        let data = [0x4Du8, 0xFC, 0x6C, 0x6C, 0x65, 0x72, 0x00]; // "Müller\0" in Latin-1
        let r = TiffReader::new(&data, ByteOrder::Little);
        let result = read_ascii(&r, 0, 7);
        assert_eq!(result, Some("Müller".to_string()));
    }

    #[test]
    fn read_ascii_single_char() {
        let data = b"A\0";
        let r = TiffReader::new(data, ByteOrder::Little);
        assert_eq!(read_ascii(&r, 0, 2), Some("A".to_string()));
    }

    #[test]
    fn read_ascii_single_nul() {
        let data = [0u8];
        let r = TiffReader::new(&data, ByteOrder::Little);
        assert_eq!(read_ascii(&r, 0, 1), None);
    }

    #[test]
    fn read_ifd_ascii_inline_short_string() {
        // An ASCII value with count <= 4 is stored inline in the value_or_offset field.
        // "OK\0" = 3 bytes, stored inline starting at the entry's value offset.
        let mut data = Vec::new();
        // Simulate: the 4-byte value field contains "OK\0\0"
        data.extend_from_slice(b"OK\0\0");
        let r = TiffReader::new(&data, ByteOrder::Little);
        // entry_value_offset = 0, pointing to the "OK\0\0" bytes
        let result = read_ifd_ascii(&r, 3, 0 /* ignored for inline */, 0);
        assert_eq!(result, Some("OK".to_string()));
    }

    #[test]
    fn read_ifd_ascii_offset_referenced() {
        // count > 4, so value_or_offset is an offset.
        let mut data = vec![0u8; 50];
        // Place "Canon EOS R5\0" at offset 30
        let s = b"Canon EOS R5\0";
        data[30..30 + s.len()].copy_from_slice(s);
        let r = TiffReader::new(&data, ByteOrder::Little);
        let result = read_ifd_ascii(&r, 13, 30, 0 /* ignored for offset */);
        assert_eq!(result, Some("Canon EOS R5".to_string()));
    }

    #[test]
    fn read_ifd_ascii_inline_4_bytes_exact() {
        // Exactly 4 bytes inline: "RGB\0"
        let data = b"RGB\0";
        let r = TiffReader::new(data, ByteOrder::Little);
        let result = read_ifd_ascii(&r, 4, 0, 0);
        assert_eq!(result, Some("RGB".to_string()));
    }

    #[test]
    fn read_ifd_ascii_inline_1_byte() {
        // Single byte inline: "N\0" but count=2 ≤ 4
        let data = b"N\0\0\0";
        let r = TiffReader::new(data, ByteOrder::Little);
        let result = read_ifd_ascii(&r, 2, 0, 0);
        assert_eq!(result, Some("N".to_string()));
    }

    // ═════════════════════════════════════════════════════════════════════
    // parse_tiff_exif tests (refactored internal entry point)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_tiff_exif_empty() {
        assert_eq!(parse_tiff_exif(b""), None);
    }

    #[test]
    fn parse_tiff_exif_too_short() {
        assert_eq!(parse_tiff_exif(b"II\x2a\x00"), None);
    }

    #[test]
    fn parse_tiff_exif_valid_le_zero_entries() {
        // Minimal valid TIFF: II, magic 42, IFD0 offset=8
        // At TIFF offset 8: entry count = 0, next-IFD = 0
        let mut data = vec![0u8; 14];
        data[0..2].copy_from_slice(b"II");
        data[2..4].copy_from_slice(&42u16.to_le_bytes());
        data[4..8].copy_from_slice(&8u32.to_le_bytes());
        // IFD count = 0
        data[8..10].copy_from_slice(&0u16.to_le_bytes());
        // next-IFD = 0
        data[10..14].copy_from_slice(&0u32.to_le_bytes());
        let info = parse_tiff_exif(&data).unwrap();
        assert_eq!(info, JpegExifInfo::default());
    }

    #[test]
    fn parse_tiff_exif_matches_parse_exif() {
        // Verify that parse_exif("Exif\0\0" ++ tiff) == parse_tiff_exif(tiff)
        let data = build_exif_le(&[(0x0112, 3, 1, 3)], &[]);
        let from_exif = parse_exif(&data).unwrap();
        let from_tiff = parse_tiff_exif(&data[6..]).unwrap();
        assert_eq!(from_exif, from_tiff);
    }

    // ═════════════════════════════════════════════════════════════════════
    // dms_to_decimal tests (task 2.4)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn dms_to_decimal_north_latitude() {
        // 40° 26' 46" N → 40.446111...
        let result = dms_to_decimal((40, 1), (26, 1), (46, 1), b'N');
        let val = result.unwrap();
        assert!((val - 40.44611111).abs() < 1e-6, "got {val}");
    }

    #[test]
    fn dms_to_decimal_south_latitude() {
        // 33° 51' 54" S → -33.865
        let result = dms_to_decimal((33, 1), (51, 1), (54, 1), b'S');
        let val = result.unwrap();
        assert!(val < 0.0, "South should be negative");
        assert!((val - (-33.865)).abs() < 1e-6, "got {val}");
    }

    #[test]
    fn dms_to_decimal_east_longitude() {
        // 79° 58' 56" E → 79.98222...
        let result = dms_to_decimal((79, 1), (58, 1), (56, 1), b'E');
        let val = result.unwrap();
        assert!(val > 0.0);
        assert!((val - 79.98222222).abs() < 1e-6, "got {val}");
    }

    #[test]
    fn dms_to_decimal_west_longitude() {
        // 73° 59' 11" W → -73.98638...
        let result = dms_to_decimal((73, 1), (59, 1), (11, 1), b'W');
        let val = result.unwrap();
        assert!(val < 0.0, "West should be negative");
        assert!((val - (-73.98638888)).abs() < 1e-6, "got {val}");
    }

    #[test]
    fn dms_to_decimal_zero_coordinates() {
        let result = dms_to_decimal((0, 1), (0, 1), (0, 1), b'N');
        assert_eq!(result, Some(0.0));
    }

    #[test]
    fn dms_to_decimal_fractional_seconds() {
        // Fractional seconds via rational: 46.5 seconds = (93, 2)
        // 40° 26' 46.5" N → 40.44625
        let result = dms_to_decimal((40, 1), (26, 1), (93, 2), b'N');
        let val = result.unwrap();
        assert!((val - 40.44625).abs() < 1e-8, "got {val}");
    }

    #[test]
    fn dms_to_decimal_fractional_degrees() {
        // Some GPS stores everything in degrees: (40446111, 1000000), (0,1), (0,1)
        let result = dms_to_decimal((40446111, 1000000), (0, 1), (0, 1), b'N');
        let val = result.unwrap();
        assert!((val - 40.446111).abs() < 1e-6, "got {val}");
    }

    #[test]
    fn dms_to_decimal_zero_denominator_degrees() {
        assert_eq!(dms_to_decimal((40, 0), (26, 1), (46, 1), b'N'), None);
    }

    #[test]
    fn dms_to_decimal_zero_denominator_minutes() {
        assert_eq!(dms_to_decimal((40, 1), (26, 0), (46, 1), b'N'), None);
    }

    #[test]
    fn dms_to_decimal_zero_denominator_seconds() {
        assert_eq!(dms_to_decimal((40, 1), (26, 1), (46, 0), b'N'), None);
    }

    #[test]
    fn dms_to_decimal_degrees_out_of_range() {
        // 360° is invalid
        assert_eq!(dms_to_decimal((360, 1), (0, 1), (0, 1), b'N'), None);
        assert_eq!(dms_to_decimal((500, 1), (0, 1), (0, 1), b'E'), None);
    }

    #[test]
    fn dms_to_decimal_minutes_out_of_range() {
        assert_eq!(dms_to_decimal((40, 1), (60, 1), (0, 1), b'N'), None);
        assert_eq!(dms_to_decimal((40, 1), (99, 1), (0, 1), b'N'), None);
    }

    #[test]
    fn dms_to_decimal_seconds_out_of_range() {
        assert_eq!(dms_to_decimal((40, 1), (26, 1), (60, 1), b'N'), None);
        assert_eq!(dms_to_decimal((40, 1), (26, 1), (120, 1), b'N'), None);
    }

    #[test]
    fn dms_to_decimal_invalid_ref_char() {
        assert_eq!(dms_to_decimal((40, 1), (26, 1), (46, 1), b'X'), None);
        assert_eq!(dms_to_decimal((40, 1), (26, 1), (46, 1), b'n'), None);
        assert_eq!(dms_to_decimal((40, 1), (26, 1), (46, 1), 0), None);
    }

    #[test]
    fn dms_to_decimal_max_valid() {
        // 359° 59' 59" N — boundary: just under 360
        let result = dms_to_decimal((359, 1), (59, 1), (59, 1), b'N');
        let val = result.unwrap();
        assert!(val < 360.0);
        assert!(val > 359.99);
    }

    // ═════════════════════════════════════════════════════════════════════
    // parse_exif tests (tasks 2.5 / 2.6)
    // ═════════════════════════════════════════════════════════════════════

    // ── Test helpers for building synthetic EXIF data ─────────────────

    /// Write a u16 in little-endian at the given offset.
    fn put_u16_le(buf: &mut [u8], offset: usize, val: u16) {
        buf[offset..offset + 2].copy_from_slice(&val.to_le_bytes());
    }

    /// Write a u32 in little-endian at the given offset.
    fn put_u32_le(buf: &mut [u8], offset: usize, val: u32) {
        buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    }

    /// Write a u16 in big-endian at the given offset.
    fn put_u16_be(buf: &mut [u8], offset: usize, val: u16) {
        buf[offset..offset + 2].copy_from_slice(&val.to_be_bytes());
    }

    /// Write a u32 in big-endian at the given offset.
    fn put_u32_be(buf: &mut [u8], offset: usize, val: u32) {
        buf[offset..offset + 4].copy_from_slice(&val.to_be_bytes());
    }

    /// Build a minimal valid EXIF blob (little-endian) with the given
    /// IFD0 entries placed after the header.  `extra` bytes are appended
    /// after the IFD for offset-referenced values.
    ///
    /// Layout:
    /// - [0..6]: `Exif\0\0`
    /// - [6..8]: `II` (LE byte order)
    /// - [8..10]: magic 42
    /// - [10..14]: IFD0 offset = 8 (relative to TIFF start, i.e. byte 14 in raw)
    /// - [14..16]: entry count
    /// - [16..]: 12-byte IFD entries
    /// - After entries: 4-byte next-IFD pointer (0)
    /// - Then: `extra` bytes for offset-referenced data
    ///
    /// Returns `(raw_exif_bytes, tiff_data_start_in_raw, ifd_entries_start_in_raw)`.
    fn build_exif_le(entries: &[(u16, u16, u32, u32)], extra: &[u8]) -> Vec<u8> {
        let ifd0_tiff_offset: u32 = 8; // IFD0 starts at offset 8 within the TIFF stream
        let entry_count = entries.len();
        let ifd_size = 2 + entry_count * 12 + 4; // count + entries + next-IFD pointer
        let tiff_size = 8 + ifd_size + extra.len(); // header + IFD + extra
        let total = 6 + tiff_size; // Exif\0\0 + TIFF

        let mut buf = vec![0u8; total];

        // Exif header
        buf[0..6].copy_from_slice(b"Exif\0\0");

        // TIFF header (starts at raw offset 6)
        buf[6..8].copy_from_slice(b"II"); // little-endian
        put_u16_le(&mut buf, 8, 42); // magic
        put_u32_le(&mut buf, 10, ifd0_tiff_offset); // IFD0 offset

        // IFD0 (starts at raw offset 14, TIFF offset 8)
        let ifd_raw_start = 14;
        put_u16_le(&mut buf, ifd_raw_start, entry_count as u16);

        for (i, &(tag, typ, count, val)) in entries.iter().enumerate() {
            let base = ifd_raw_start + 2 + i * 12;
            put_u16_le(&mut buf, base, tag);
            put_u16_le(&mut buf, base + 2, typ);
            put_u32_le(&mut buf, base + 4, count);
            put_u32_le(&mut buf, base + 8, val);
        }

        // next-IFD pointer = 0 (no more IFDs)
        let next_ifd_off = ifd_raw_start + 2 + entry_count * 12;
        put_u32_le(&mut buf, next_ifd_off, 0);

        // Extra data for offset-referenced values
        let extra_start = next_ifd_off + 4;
        buf[extra_start..extra_start + extra.len()].copy_from_slice(extra);

        buf
    }

    /// Build a minimal valid EXIF blob (big-endian) analogous to `build_exif_le`.
    fn build_exif_be(entries: &[(u16, u16, u32, u32)], extra: &[u8]) -> Vec<u8> {
        let ifd0_tiff_offset: u32 = 8;
        let entry_count = entries.len();
        let ifd_size = 2 + entry_count * 12 + 4;
        let tiff_size = 8 + ifd_size + extra.len();
        let total = 6 + tiff_size;

        let mut buf = vec![0u8; total];

        buf[0..6].copy_from_slice(b"Exif\0\0");
        buf[6..8].copy_from_slice(b"MM"); // big-endian
        put_u16_be(&mut buf, 8, 42);
        put_u32_be(&mut buf, 10, ifd0_tiff_offset);

        let ifd_raw_start = 14;
        put_u16_be(&mut buf, ifd_raw_start, entry_count as u16);

        for (i, &(tag, typ, count, val)) in entries.iter().enumerate() {
            let base = ifd_raw_start + 2 + i * 12;
            put_u16_be(&mut buf, base, tag);
            put_u16_be(&mut buf, base + 2, typ);
            put_u32_be(&mut buf, base + 4, count);
            put_u32_be(&mut buf, base + 8, val);
        }

        let next_ifd_off = ifd_raw_start + 2 + entry_count * 12;
        put_u32_be(&mut buf, next_ifd_off, 0);

        let extra_start = next_ifd_off + 4;
        buf[extra_start..extra_start + extra.len()].copy_from_slice(extra);

        buf
    }

    /// Compute the TIFF offset of the "extra" region that follows the IFD
    /// in our synthetic EXIF blobs.
    ///
    /// The TIFF stream starts at raw offset 6.
    /// IFD0 starts at TIFF offset 8.
    /// IFD0 = 2 (count) + entries*12 + 4 (next-IFD pointer).
    /// Extra starts right after.
    fn extra_tiff_offset(entry_count: usize) -> u32 {
        // TIFF offset of IFD0 = 8
        // IFD0 size = 2 + entry_count * 12 + 4
        (8 + 2 + entry_count * 12 + 4) as u32
    }

    // ── Basic EXIF parse tests ───────────────────────────────────────

    #[test]
    fn parse_exif_completely_empty() {
        assert_eq!(parse_exif(b""), None);
    }

    #[test]
    fn parse_exif_too_short() {
        assert_eq!(parse_exif(b"Exif\0\0II"), None);
    }

    #[test]
    fn parse_exif_wrong_header() {
        let mut data = build_exif_le(&[], &[]);
        data[0..4].copy_from_slice(b"JFIF");
        assert_eq!(parse_exif(&data), None);
    }

    #[test]
    fn parse_exif_wrong_magic() {
        let mut data = build_exif_le(&[], &[]);
        // Overwrite magic 42 with 99
        put_u16_le(&mut data, 8, 99);
        assert_eq!(parse_exif(&data), None);
    }

    #[test]
    fn parse_exif_wrong_byte_order() {
        let mut data = build_exif_le(&[], &[]);
        data[6..8].copy_from_slice(b"XX");
        assert_eq!(parse_exif(&data), None);
    }

    #[test]
    fn parse_exif_valid_header_zero_entries() {
        let data = build_exif_le(&[], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info, JpegExifInfo::default());
    }

    #[test]
    fn parse_exif_valid_header_zero_entries_be() {
        let data = build_exif_be(&[], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info, JpegExifInfo::default());
    }

    // ── Orientation ──────────────────────────────────────────────────

    #[test]
    fn parse_exif_orientation_le() {
        // Orientation (0x0112), SHORT (3), count=1, value=6
        // In LE, a SHORT value=6 stored inline: low 16 bits = 6 → u32 = 6
        let data = build_exif_le(&[(0x0112, 3, 1, 6)], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.orientation, Some(6));
    }

    #[test]
    fn parse_exif_orientation_be() {
        // In BE, a SHORT value=6 inline: the u16 is at bytes [8..10] of the entry,
        // which we read with u16_at(entry_val_off). The 4-byte value field has
        // the SHORT in the first 2 bytes. We store as u32 in BE:
        // 6 as u16 in first 2 bytes → u32 = 6 << 16 = 0x00060000
        // But our builder writes the u32 as BE, so put_u32_be(6 << 16).
        // When we reader.u16_at(entry_val_off), it reads the first 2 bytes
        // of the value field as BE u16 → 6.
        let data = build_exif_be(&[(0x0112, 3, 1, 6 << 16)], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.orientation, Some(6));
    }

    #[test]
    fn parse_exif_orientation_all_valid_values() {
        for v in 1u16..=8 {
            let data = build_exif_le(&[(0x0112, 3, 1, v as u32)], &[]);
            let info = parse_exif(&data).unwrap();
            assert_eq!(info.orientation, Some(v as u8), "orientation {v}");
        }
    }

    #[test]
    fn parse_exif_orientation_zero_invalid() {
        let data = build_exif_le(&[(0x0112, 3, 1, 0)], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.orientation, None);
    }

    #[test]
    fn parse_exif_orientation_9_invalid() {
        let data = build_exif_le(&[(0x0112, 3, 1, 9)], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.orientation, None);
    }

    #[test]
    fn parse_exif_orientation_wrong_type_ignored() {
        // Orientation with wrong TIFF type (LONG instead of SHORT) → skip
        let data = build_exif_le(&[(0x0112, 4, 1, 6)], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.orientation, None);
    }

    // ── ASCII fields (Make, Model, Software, DateTime) ───────────────

    #[test]
    fn parse_exif_make_inline() {
        // Make (0x010F), ASCII (2), count=4 (≤4 → inline), value="Hi!\0"
        // The 4-byte value field contains "Hi!\0" = [0x48, 0x69, 0x21, 0x00]
        // LE u32 of those bytes: 0x00216948
        let val = u32::from_le_bytes([b'H', b'i', b'!', 0]);
        let data = build_exif_le(&[(0x010F, 2, 4, val)], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.camera_make, Some("Hi!".to_string()));
    }

    #[test]
    fn parse_exif_model_offset_referenced() {
        // Model (0x0110), ASCII (2), count=6, offset → extra region
        let extra_off = extra_tiff_offset(1);
        let model_bytes = b"Canon\0";
        let data = build_exif_le(&[(0x0110, 2, 6, extra_off)], model_bytes);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.camera_model, Some("Canon".to_string()));
    }

    #[test]
    fn parse_exif_software_offset_referenced() {
        let extra_off = extra_tiff_offset(1);
        let sw_bytes = b"Lightroom 6.0\0";
        let data = build_exif_le(&[(0x0131, 2, 14, extra_off)], sw_bytes);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.software, Some("Lightroom 6.0".to_string()));
    }

    #[test]
    fn parse_exif_datetime_offset_referenced() {
        let extra_off = extra_tiff_offset(1);
        let dt_bytes = b"2025:01:15 12:30:00\0";
        let data = build_exif_le(&[(0x0132, 2, 20, extra_off)], dt_bytes);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.datetime, Some("2025:01:15 12:30:00".to_string()));
    }

    #[test]
    fn parse_exif_ascii_wrong_type_ignored() {
        // Make with wrong type (SHORT instead of ASCII) → skip
        let data = build_exif_le(&[(0x010F, 3, 1, 42)], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.camera_make, None);
    }

    // ── Multiple IFD0 fields ─────────────────────────────────────────

    #[test]
    fn parse_exif_multiple_ifd0_fields() {
        // Build extra: make at extra_off, model at extra_off+6, datetime at extra_off+12
        let make_str = b"Nikon\0";
        let model_str = b"D850\0\0"; // pad to 6 bytes for alignment
        let dt_str = b"2024:06:01 09:15:30\0";
        let mut extra = Vec::new();
        extra.extend_from_slice(make_str); // offset 0
        extra.extend_from_slice(model_str); // offset 6
        extra.extend_from_slice(dt_str); // offset 12

        let base_off = extra_tiff_offset(4); // 4 entries
        let entries = [
            (0x010F, 2u16, 6u32, base_off), // Make
            (0x0110, 2, 6, base_off + 6),   // Model
            (0x0112, 3, 1, 1u32),           // Orientation = 1
            (0x0132, 2, 20, base_off + 12), // DateTime
        ];
        let data = build_exif_le(&entries, &extra);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.camera_make, Some("Nikon".to_string()));
        assert_eq!(info.camera_model, Some("D850".to_string()));
        assert_eq!(info.orientation, Some(1));
        assert_eq!(info.datetime, Some("2024:06:01 09:15:30".to_string()));
    }

    // ── EXIF sub-IFD ─────────────────────────────────────────────────

    /// Build a complete EXIF blob with an IFD0 → EXIF sub-IFD chain (LE).
    ///
    /// Layout:
    /// - Exif\0\0 (6 bytes)
    /// - TIFF header (8 bytes at raw 6..14)
    /// - IFD0 (at TIFF offset 8):
    ///   - 1 entry: ExifIFDPointer (0x8769) pointing to the sub-IFD
    ///   - next-IFD = 0
    /// - EXIF sub-IFD (placed after IFD0 + next-IFD-ptr):
    ///   - `exif_entries` IFD entries
    ///   - next-IFD = 0
    /// - Extra data after the sub-IFD
    fn build_exif_with_sub_ifd_le(
        exif_entries: &[(u16, u16, u32, u32)],
        extra: &[u8],
    ) -> (Vec<u8>, u32) {
        // IFD0: 1 entry (ExifIFDPointer)
        let ifd0_tiff_off: u32 = 8;
        let ifd0_size = 2 + 1 * 12 + 4; // 18 bytes
        let exif_ifd_tiff_off = ifd0_tiff_off + ifd0_size as u32;

        let exif_entry_count = exif_entries.len();
        let exif_ifd_size = 2 + exif_entry_count * 12 + 4;
        let extra_tiff_off = exif_ifd_tiff_off + exif_ifd_size as u32;

        let tiff_size = 8 + ifd0_size + exif_ifd_size + extra.len();
        let total = 6 + tiff_size;
        let mut buf = vec![0u8; total];

        // Exif header
        buf[0..6].copy_from_slice(b"Exif\0\0");
        buf[6..8].copy_from_slice(b"II");
        put_u16_le(&mut buf, 8, 42);
        put_u32_le(&mut buf, 10, ifd0_tiff_off);

        // IFD0
        let ifd0_raw = 6 + ifd0_tiff_off as usize;
        put_u16_le(&mut buf, ifd0_raw, 1); // 1 entry
        // Entry: ExifIFDPointer (0x8769), LONG (4), count=1, value = exif_ifd_tiff_off
        let e0 = ifd0_raw + 2;
        put_u16_le(&mut buf, e0, 0x8769);
        put_u16_le(&mut buf, e0 + 2, 4); // LONG
        put_u32_le(&mut buf, e0 + 4, 1); // count
        put_u32_le(&mut buf, e0 + 8, exif_ifd_tiff_off);
        // next-IFD = 0
        put_u32_le(&mut buf, e0 + 12, 0);

        // EXIF sub-IFD
        let exif_raw = 6 + exif_ifd_tiff_off as usize;
        put_u16_le(&mut buf, exif_raw, exif_entry_count as u16);
        for (i, &(tag, typ, count, val)) in exif_entries.iter().enumerate() {
            let base = exif_raw + 2 + i * 12;
            put_u16_le(&mut buf, base, tag);
            put_u16_le(&mut buf, base + 2, typ);
            put_u32_le(&mut buf, base + 4, count);
            put_u32_le(&mut buf, base + 8, val);
        }
        // next-IFD = 0
        let next_ptr = exif_raw + 2 + exif_entry_count * 12;
        put_u32_le(&mut buf, next_ptr, 0);

        // Extra data
        let extra_raw = 6 + extra_tiff_off as usize;
        buf[extra_raw..extra_raw + extra.len()].copy_from_slice(extra);

        (buf, extra_tiff_off)
    }

    #[test]
    fn parse_exif_sub_ifd_exposure_time() {
        // ExposureTime (0x829A), RATIONAL, count=1, offset → extra
        // Value: 1/250 = (1, 250)
        let mut extra = [0u8; 8];
        put_u32_le(&mut extra, 0, 1); // numerator
        put_u32_le(&mut extra, 4, 250); // denominator

        let (data, extra_off) = build_exif_with_sub_ifd_le(
            &[(0x829A, 5, 1, 0)], // placeholder offset, fix below
            &extra,
        );
        // Fix the offset in the sub-IFD entry's value_or_offset field
        let mut data = data;
        // The sub-IFD entry's value_or_offset is at:
        // raw = 6 + exif_ifd_tiff_off + 2 + 0 * 12 + 8
        let ifd0_size = 2 + 1 * 12 + 4;
        let exif_ifd_tiff_off = 8 + ifd0_size as usize;
        let val_off_raw = 6 + exif_ifd_tiff_off + 2 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        assert_eq!(info.exposure_time, Some((1, 250)));
    }

    #[test]
    fn parse_exif_sub_ifd_fnumber() {
        let mut extra = [0u8; 8];
        put_u32_le(&mut extra, 0, 28); // numerator
        put_u32_le(&mut extra, 4, 10); // denominator → f/2.8

        let (mut data, extra_off) = build_exif_with_sub_ifd_le(&[(0x829D, 5, 1, 0)], &extra);
        let ifd0_size = 2 + 1 * 12 + 4;
        let exif_ifd_tiff_off = 8 + ifd0_size;
        let val_off_raw = 6 + exif_ifd_tiff_off + 2 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        assert_eq!(info.f_number, Some((28, 10)));
    }

    #[test]
    fn parse_exif_sub_ifd_iso_speed() {
        // ISOSpeedRatings (0x8827), SHORT (3), count=1, value=400
        let (data, _) = build_exif_with_sub_ifd_le(&[(0x8827, 3, 1, 400)], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.iso_speed, Some(400));
    }

    #[test]
    fn parse_exif_sub_ifd_focal_length() {
        let mut extra = [0u8; 8];
        put_u32_le(&mut extra, 0, 50); // numerator
        put_u32_le(&mut extra, 4, 1); // denominator → 50mm

        let (mut data, extra_off) = build_exif_with_sub_ifd_le(&[(0x920A, 5, 1, 0)], &extra);
        let ifd0_size = 2 + 1 * 12 + 4;
        let exif_ifd_tiff_off = 8 + ifd0_size;
        let val_off_raw = 6 + exif_ifd_tiff_off + 2 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        assert_eq!(info.focal_length, Some((50, 1)));
    }

    #[test]
    fn parse_exif_sub_ifd_datetime_original() {
        let dt = b"2024:12:25 08:00:00\0";
        let (mut data, extra_off) = build_exif_with_sub_ifd_le(&[(0x9003, 2, 20, 0)], dt);
        let ifd0_size = 2 + 1 * 12 + 4;
        let exif_ifd_tiff_off = 8 + ifd0_size;
        let val_off_raw = 6 + exif_ifd_tiff_off + 2 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        assert_eq!(
            info.datetime_original,
            Some("2024:12:25 08:00:00".to_string())
        );
    }

    #[test]
    fn parse_exif_missing_exif_sub_ifd() {
        // IFD0 has no ExifIFDPointer → all EXIF sub-IFD fields remain None
        let data = build_exif_le(&[(0x0112, 3, 1, 1)], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.orientation, Some(1));
        assert_eq!(info.exposure_time, None);
        assert_eq!(info.f_number, None);
        assert_eq!(info.iso_speed, None);
        assert_eq!(info.focal_length, None);
        assert_eq!(info.datetime_original, None);
    }

    #[test]
    fn parse_exif_multiple_exif_sub_ifd_entries() {
        // Two entries in EXIF sub-IFD: ISO and a RATIONAL focal length
        let mut extra = [0u8; 8];
        put_u32_le(&mut extra, 0, 85); // focal length numerator
        put_u32_le(&mut extra, 4, 1); // focal length denominator

        let (mut data, extra_off) = build_exif_with_sub_ifd_le(
            &[
                (0x8827, 3, 1, 200), // ISO 200
                (0x920A, 5, 1, 0),   // focal length → fix offset
            ],
            &extra,
        );
        // Fix focal length offset (entry index 1)
        let ifd0_size = 2 + 1 * 12 + 4;
        let exif_ifd_tiff_off = 8 + ifd0_size;
        let val_off_raw = 6 + exif_ifd_tiff_off + 2 + 1 * 12 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        assert_eq!(info.iso_speed, Some(200));
        assert_eq!(info.focal_length, Some((85, 1)));
    }

    // ── GPS sub-IFD ──────────────────────────────────────────────────

    /// Build a complete EXIF blob with IFD0 → GPS sub-IFD chain (LE).
    fn build_exif_with_gps_ifd_le(
        gps_entries: &[(u16, u16, u32, u32)],
        extra: &[u8],
    ) -> (Vec<u8>, u32) {
        let ifd0_tiff_off: u32 = 8;
        let ifd0_size = 2 + 1 * 12 + 4; // 1 entry (GPSInfoPointer)
        let gps_ifd_tiff_off = ifd0_tiff_off + ifd0_size as u32;

        let gps_entry_count = gps_entries.len();
        let gps_ifd_size = 2 + gps_entry_count * 12 + 4;
        let extra_tiff_off = gps_ifd_tiff_off + gps_ifd_size as u32;

        let tiff_size = 8 + ifd0_size + gps_ifd_size + extra.len();
        let total = 6 + tiff_size;
        let mut buf = vec![0u8; total];

        buf[0..6].copy_from_slice(b"Exif\0\0");
        buf[6..8].copy_from_slice(b"II");
        put_u16_le(&mut buf, 8, 42);
        put_u32_le(&mut buf, 10, ifd0_tiff_off);

        // IFD0: 1 entry (GPSInfoPointer)
        let ifd0_raw = 6 + ifd0_tiff_off as usize;
        put_u16_le(&mut buf, ifd0_raw, 1);
        let e0 = ifd0_raw + 2;
        put_u16_le(&mut buf, e0, 0x8825);
        put_u16_le(&mut buf, e0 + 2, 4); // LONG
        put_u32_le(&mut buf, e0 + 4, 1);
        put_u32_le(&mut buf, e0 + 8, gps_ifd_tiff_off);
        put_u32_le(&mut buf, e0 + 12, 0); // next-IFD

        // GPS sub-IFD
        let gps_raw = 6 + gps_ifd_tiff_off as usize;
        put_u16_le(&mut buf, gps_raw, gps_entry_count as u16);
        for (i, &(tag, typ, count, val)) in gps_entries.iter().enumerate() {
            let base = gps_raw + 2 + i * 12;
            put_u16_le(&mut buf, base, tag);
            put_u16_le(&mut buf, base + 2, typ);
            put_u32_le(&mut buf, base + 4, count);
            put_u32_le(&mut buf, base + 8, val);
        }
        let next_ptr = gps_raw + 2 + gps_entry_count * 12;
        put_u32_le(&mut buf, next_ptr, 0);

        // Extra
        let extra_raw = 6 + extra_tiff_off as usize;
        buf[extra_raw..extra_raw + extra.len()].copy_from_slice(extra);

        (buf, extra_tiff_off)
    }

    #[test]
    fn parse_exif_gps_latitude_north() {
        // GPS: 40° 26' 46" N
        // Needs: LatRef (0x0001), Latitude (0x0002)
        let mut extra = [0u8; 24];
        // Degrees: 40/1
        put_u32_le(&mut extra, 0, 40);
        put_u32_le(&mut extra, 4, 1);
        // Minutes: 26/1
        put_u32_le(&mut extra, 8, 26);
        put_u32_le(&mut extra, 12, 1);
        // Seconds: 46/1
        put_u32_le(&mut extra, 16, 46);
        put_u32_le(&mut extra, 20, 1);

        // LatRef = 'N' inline as ASCII count=2: "N\0"
        let lat_ref_val = u32::from_le_bytes([b'N', 0, 0, 0]);

        let (mut data, extra_off) = build_exif_with_gps_ifd_le(
            &[
                (0x0001, 2, 2, lat_ref_val), // GPSLatitudeRef
                (0x0002, 5, 3, 0),           // GPSLatitude → fix offset
            ],
            &extra,
        );
        // Fix latitude offset (entry index 1, value_or_offset)
        let ifd0_size = 2 + 1 * 12 + 4;
        let gps_ifd_tiff_off = (8 + ifd0_size) as usize;
        let val_off_raw = 6 + gps_ifd_tiff_off + 2 + 1 * 12 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        assert!(info.gps_latitude.is_some());
        let lat = info.gps_latitude.unwrap();
        assert!((lat - 40.44611111).abs() < 1e-6, "got {lat}");
    }

    #[test]
    fn parse_exif_gps_latitude_south_negative() {
        let mut extra = [0u8; 24];
        put_u32_le(&mut extra, 0, 33);
        put_u32_le(&mut extra, 4, 1);
        put_u32_le(&mut extra, 8, 51);
        put_u32_le(&mut extra, 12, 1);
        put_u32_le(&mut extra, 16, 54);
        put_u32_le(&mut extra, 20, 1);

        let lat_ref_val = u32::from_le_bytes([b'S', 0, 0, 0]);

        let (mut data, extra_off) =
            build_exif_with_gps_ifd_le(&[(0x0001, 2, 2, lat_ref_val), (0x0002, 5, 3, 0)], &extra);
        let ifd0_size = 2 + 1 * 12 + 4;
        let gps_ifd_tiff_off = (8 + ifd0_size) as usize;
        let val_off_raw = 6 + gps_ifd_tiff_off + 2 + 1 * 12 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        let lat = info.gps_latitude.unwrap();
        assert!(lat < 0.0, "South should be negative, got {lat}");
        assert!((lat - (-33.865)).abs() < 1e-6, "got {lat}");
    }

    #[test]
    fn parse_exif_gps_longitude_west_negative() {
        let mut extra = [0u8; 24];
        put_u32_le(&mut extra, 0, 73);
        put_u32_le(&mut extra, 4, 1);
        put_u32_le(&mut extra, 8, 59);
        put_u32_le(&mut extra, 12, 1);
        put_u32_le(&mut extra, 16, 11);
        put_u32_le(&mut extra, 20, 1);

        let lon_ref_val = u32::from_le_bytes([b'W', 0, 0, 0]);

        let (mut data, extra_off) =
            build_exif_with_gps_ifd_le(&[(0x0003, 2, 2, lon_ref_val), (0x0004, 5, 3, 0)], &extra);
        let ifd0_size = 2 + 1 * 12 + 4;
        let gps_ifd_tiff_off = (8 + ifd0_size) as usize;
        let val_off_raw = 6 + gps_ifd_tiff_off + 2 + 1 * 12 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        let lon = info.gps_longitude.unwrap();
        assert!(lon < 0.0, "West should be negative, got {lon}");
        assert!((lon - (-73.98638888)).abs() < 1e-6, "got {lon}");
    }

    #[test]
    fn parse_exif_gps_altitude_above_sea_level() {
        // Altitude: 100/1 m, AltRef=0 (above)
        let mut extra = [0u8; 8];
        put_u32_le(&mut extra, 0, 100);
        put_u32_le(&mut extra, 4, 1);

        let (mut data, extra_off) = build_exif_with_gps_ifd_le(
            &[
                (0x0005, 1, 1, 0), // GPSAltitudeRef = 0 (above)
                (0x0006, 5, 1, 0), // GPSAltitude → fix
            ],
            &extra,
        );
        let ifd0_size = 2 + 1 * 12 + 4;
        let gps_ifd_tiff_off = (8 + ifd0_size) as usize;
        let val_off_raw = 6 + gps_ifd_tiff_off + 2 + 1 * 12 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        let alt = info.gps_altitude.unwrap();
        assert!((alt - 100.0).abs() < 1e-6, "got {alt}");
    }

    #[test]
    fn parse_exif_gps_altitude_below_sea_level() {
        let mut extra = [0u8; 8];
        put_u32_le(&mut extra, 0, 50);
        put_u32_le(&mut extra, 4, 1);

        let (mut data, extra_off) = build_exif_with_gps_ifd_le(
            &[
                (0x0005, 1, 1, 1), // GPSAltitudeRef = 1 (below)
                (0x0006, 5, 1, 0),
            ],
            &extra,
        );
        let ifd0_size = 2 + 1 * 12 + 4;
        let gps_ifd_tiff_off = (8 + ifd0_size) as usize;
        let val_off_raw = 6 + gps_ifd_tiff_off + 2 + 1 * 12 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        let alt = info.gps_altitude.unwrap();
        assert!((alt - (-50.0)).abs() < 1e-6, "got {alt}");
    }

    #[test]
    fn parse_exif_gps_altitude_no_ref_defaults_above() {
        // No GPSAltitudeRef → defaults to above sea level (positive)
        let mut extra = [0u8; 8];
        put_u32_le(&mut extra, 0, 200);
        put_u32_le(&mut extra, 4, 1);

        let (mut data, extra_off) = build_exif_with_gps_ifd_le(
            &[
                (0x0006, 5, 1, 0), // GPSAltitude only, no AltRef
            ],
            &extra,
        );
        let ifd0_size = 2 + 1 * 12 + 4;
        let gps_ifd_tiff_off = (8 + ifd0_size) as usize;
        let val_off_raw = 6 + gps_ifd_tiff_off + 2 + 0 * 12 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        assert_eq!(info.gps_altitude, Some(200.0));
    }

    #[test]
    fn parse_exif_missing_gps_sub_ifd() {
        // IFD0 has orientation but no GPSInfoPointer → GPS fields None
        let data = build_exif_le(&[(0x0112, 3, 1, 1)], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.gps_latitude, None);
        assert_eq!(info.gps_longitude, None);
        assert_eq!(info.gps_altitude, None);
    }

    #[test]
    fn parse_exif_gps_lat_without_ref_produces_none() {
        // GPSLatitude present but no GPSLatitudeRef → lat remains None
        let mut extra = [0u8; 24];
        put_u32_le(&mut extra, 0, 40);
        put_u32_le(&mut extra, 4, 1);
        put_u32_le(&mut extra, 8, 26);
        put_u32_le(&mut extra, 12, 1);
        put_u32_le(&mut extra, 16, 46);
        put_u32_le(&mut extra, 20, 1);

        let (mut data, extra_off) = build_exif_with_gps_ifd_le(
            &[
                (0x0002, 5, 3, 0), // GPSLatitude only, no ref
            ],
            &extra,
        );
        let ifd0_size = 2 + 1 * 12 + 4;
        let gps_ifd_tiff_off = (8 + ifd0_size) as usize;
        let val_off_raw = 6 + gps_ifd_tiff_off + 2 + 0 * 12 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        assert_eq!(info.gps_latitude, None);
    }

    #[test]
    fn parse_exif_gps_full_coordinates() {
        // Full GPS: lat 40°26'46"N, lon 73°59'11"W, alt 10m above
        // Extra layout: lat DMS (24 bytes), lon DMS (24 bytes), alt RATIONAL (8 bytes)
        let mut extra = [0u8; 56];
        // Latitude: 40° 26' 46" at offset 0
        put_u32_le(&mut extra, 0, 40);
        put_u32_le(&mut extra, 4, 1);
        put_u32_le(&mut extra, 8, 26);
        put_u32_le(&mut extra, 12, 1);
        put_u32_le(&mut extra, 16, 46);
        put_u32_le(&mut extra, 20, 1);
        // Longitude: 73° 59' 11" at offset 24
        put_u32_le(&mut extra, 24, 73);
        put_u32_le(&mut extra, 28, 1);
        put_u32_le(&mut extra, 32, 59);
        put_u32_le(&mut extra, 36, 1);
        put_u32_le(&mut extra, 40, 11);
        put_u32_le(&mut extra, 44, 1);
        // Altitude: 10/1 at offset 48
        put_u32_le(&mut extra, 48, 10);
        put_u32_le(&mut extra, 52, 1);

        let lat_ref = u32::from_le_bytes([b'N', 0, 0, 0]);
        let lon_ref = u32::from_le_bytes([b'W', 0, 0, 0]);

        let (mut data, extra_off) = build_exif_with_gps_ifd_le(
            &[
                (0x0001, 2, 2, lat_ref), // LatRef
                (0x0002, 5, 3, 0),       // Lat → fix
                (0x0003, 2, 2, lon_ref), // LonRef
                (0x0004, 5, 3, 0),       // Lon → fix
                (0x0005, 1, 1, 0),       // AltRef = 0
                (0x0006, 5, 1, 0),       // Alt → fix
            ],
            &extra,
        );
        let ifd0_size = 2 + 1 * 12 + 4;
        let gps_ifd_tiff_off = (8 + ifd0_size) as usize;
        // Fix Latitude offset (entry 1)
        let lat_val_off = 6 + gps_ifd_tiff_off + 2 + 1 * 12 + 8;
        put_u32_le(&mut data, lat_val_off, extra_off + 0);
        // Fix Longitude offset (entry 3)
        let lon_val_off = 6 + gps_ifd_tiff_off + 2 + 3 * 12 + 8;
        put_u32_le(&mut data, lon_val_off, extra_off + 24);
        // Fix Altitude offset (entry 5)
        let alt_val_off = 6 + gps_ifd_tiff_off + 2 + 5 * 12 + 8;
        put_u32_le(&mut data, alt_val_off, extra_off + 48);

        let info = parse_exif(&data).unwrap();

        let lat = info.gps_latitude.unwrap();
        assert!((lat - 40.44611111).abs() < 1e-6, "lat={lat}");

        let lon = info.gps_longitude.unwrap();
        assert!(lon < 0.0);
        assert!((lon - (-73.98638888)).abs() < 1e-6, "lon={lon}");

        let alt = info.gps_altitude.unwrap();
        assert!((alt - 10.0).abs() < 1e-6, "alt={alt}");
    }

    // ── Robustness / edge-case tests ─────────────────────────────────

    #[test]
    fn parse_exif_truncated_after_tiff_header() {
        // Valid Exif header + byte order + magic, but IFD0 offset points past end
        let mut data = vec![0u8; 14];
        data[0..6].copy_from_slice(b"Exif\0\0");
        data[6..8].copy_from_slice(b"II");
        put_u16_le(&mut data, 8, 42);
        put_u32_le(&mut data, 10, 200); // offset past end
        // Should fail gracefully when trying to read IFD count
        let info = parse_exif(&data).unwrap();
        assert_eq!(info, JpegExifInfo::default());
    }

    #[test]
    fn parse_exif_malformed_orientation_other_fields_still_parse() {
        // Orientation has wrong TIFF type (LONG), but Make is valid
        let extra_off = extra_tiff_offset(2);
        let make = b"Sony\0\0"; // 6 bytes
        let data = build_exif_le(
            &[
                (0x0112, 4, 1, 6), // wrong type for orientation
                (0x010F, 2, 6, extra_off),
            ],
            make,
        );
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.orientation, None, "bad orientation should be None");
        assert_eq!(
            info.camera_make,
            Some("Sony".to_string()),
            "Make should still parse"
        );
    }

    #[test]
    fn parse_exif_unknown_tags_ignored() {
        // Tags that we don't recognize should be silently skipped
        let data = build_exif_le(
            &[
                (0xFFFF, 3, 1, 42),  // unknown tag
                (0x0112, 3, 1, 3),   // valid orientation
                (0xABCD, 4, 1, 999), // unknown tag
            ],
            &[],
        );
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.orientation, Some(3));
    }

    #[test]
    fn parse_exif_not_exif_data() {
        // Random bytes that don't start with Exif\0\0
        assert_eq!(parse_exif(b"This is not EXIF data at all"), None);
    }

    #[test]
    fn parse_exif_jfif_not_exif() {
        assert_eq!(parse_exif(b"JFIF\0\0IIQQ"), None);
    }

    #[test]
    fn parse_exif_big_endian_full_parse() {
        // Test a BE EXIF with orientation and make
        let make_val = u32::from_be_bytes([b'X', 0, 0, 0]);
        let data = build_exif_be(
            &[
                (0x0112, 3, 1, 1 << 16),  // orientation=1 as BE SHORT inline
                (0x010F, 2, 1, make_val), // Make="X" — but count=1 is just the NUL issue
            ],
            &[],
        );
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.orientation, Some(1));
    }

    #[test]
    fn parse_exif_big_endian_make_offset() {
        let extra_off = extra_tiff_offset(1);
        let make = b"Fujifilm\0";
        // Build BE: offset values must also be BE-encoded
        let data = build_exif_be(&[(0x010F, 2, 9, extra_off)], make);
        // The extra offset is already BE-encoded by build_exif_be
        let _ = &data; // verify it compiles
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.camera_make, Some("Fujifilm".to_string()));
    }

    #[test]
    fn parse_exif_gps_altitude_fractional() {
        // Altitude: 1234/10 = 123.4m above sea level
        let mut extra = [0u8; 8];
        put_u32_le(&mut extra, 0, 1234);
        put_u32_le(&mut extra, 4, 10);

        let (mut data, extra_off) =
            build_exif_with_gps_ifd_le(&[(0x0005, 1, 1, 0), (0x0006, 5, 1, 0)], &extra);
        let ifd0_size = 2 + 1 * 12 + 4;
        let gps_ifd_tiff_off = (8 + ifd0_size) as usize;
        let val_off_raw = 6 + gps_ifd_tiff_off + 2 + 1 * 12 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        let alt = info.gps_altitude.unwrap();
        assert!((alt - 123.4).abs() < 1e-6, "got {alt}");
    }

    #[test]
    fn parse_exif_gps_altitude_zero_denominator_produces_none() {
        // Altitude with zero denominator → no altitude
        let mut extra = [0u8; 8];
        put_u32_le(&mut extra, 0, 100);
        put_u32_le(&mut extra, 4, 0); // zero denominator!

        let (mut data, extra_off) = build_exif_with_gps_ifd_le(&[(0x0006, 5, 1, 0)], &extra);
        let ifd0_size = 2 + 1 * 12 + 4;
        let gps_ifd_tiff_off = (8 + ifd0_size) as usize;
        let val_off_raw = 6 + gps_ifd_tiff_off + 2 + 0 * 12 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        assert_eq!(info.gps_altitude, None);
    }

    #[test]
    fn parse_exif_exposure_time_round_trip() {
        // Verify specific rational values survive the round trip
        let mut extra = [0u8; 8];
        put_u32_le(&mut extra, 0, 1);
        put_u32_le(&mut extra, 4, 8000); // 1/8000 sec

        let (mut data, extra_off) = build_exif_with_sub_ifd_le(&[(0x829A, 5, 1, 0)], &extra);
        let ifd0_size = 2 + 1 * 12 + 4;
        let exif_ifd_tiff_off = 8 + ifd0_size;
        let val_off_raw = 6 + exif_ifd_tiff_off + 2 + 8;
        put_u32_le(&mut data, val_off_raw, extra_off);

        let info = parse_exif(&data).unwrap();
        assert_eq!(info.exposure_time, Some((1, 8000)));
    }

    #[test]
    fn parse_exif_iso_speed_max_u16() {
        let (data, _) = build_exif_with_sub_ifd_le(&[(0x8827, 3, 1, u16::MAX as u32)], &[]);
        let info = parse_exif(&data).unwrap();
        assert_eq!(info.iso_speed, Some(u16::MAX));
    }

    #[test]
    fn parse_exif_does_not_panic_on_fuzz_like_data() {
        // Various degenerate inputs that must not panic
        let inputs: &[&[u8]] = &[
            b"Exif\0\0II\x2a\x00\x08\x00\x00\x00",
            b"Exif\0\0MM\x00\x2a\x00\x00\x00\x08",
            b"Exif\0\0II\x2a\x00\xff\xff\xff\xff",
            b"Exif\0\0II\x2a\x00\x08\x00\x00\x00\xff\xff",
        ];
        for input in inputs {
            // Just ensure no panic — result can be Some or None
            let _ = parse_exif(input);
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // scan_com_markers tests (task 3.4)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn scan_com_no_markers() {
        let jpeg = build_jpeg_rgb(2, 2, 128, 128, 128);
        // The encoder may or may not insert COM markers.
        // At minimum, verify it doesn't panic and returns a Vec.
        let _ = scan_com_markers(&jpeg);
    }

    #[test]
    fn scan_com_single_comment() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let jpeg = inject_com_marker(&jpeg, "Hello, world!");
        let comments = scan_com_markers(&jpeg);
        assert!(comments.contains(&"Hello, world!".to_string()));
    }

    #[test]
    fn scan_com_multiple_comments() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let jpeg = inject_com_marker(&jpeg, "First");
        let jpeg = inject_com_marker(&jpeg, "Second");
        let comments = scan_com_markers(&jpeg);
        assert!(comments.contains(&"First".to_string()));
        assert!(comments.contains(&"Second".to_string()));
    }

    #[test]
    fn scan_com_empty_data() {
        assert!(scan_com_markers(&[]).is_empty());
    }

    #[test]
    fn scan_com_not_jpeg() {
        assert!(scan_com_markers(b"This is not JPEG").is_empty());
    }

    #[test]
    fn scan_com_truncated() {
        // SOI + COM marker but truncated length
        let data = [0xFF, 0xD8, 0xFF, 0xFE, 0x00];
        let comments = scan_com_markers(&data);
        assert!(comments.is_empty());
    }

    #[test]
    fn scan_com_latin1_fallback() {
        let jpeg = build_jpeg_rgb(1, 1, 0, 0, 0);
        // Build a COM marker with invalid UTF-8 bytes (Latin-1 characters)
        let mut injected = Vec::new();
        injected.extend_from_slice(&jpeg[..2]); // SOI
        injected.push(0xFF);
        injected.push(0xFE); // COM
        let text = [0xC0, 0xC1, 0xFE, 0xFF]; // invalid UTF-8, valid Latin-1
        let seg_len = (text.len() + 2) as u16;
        injected.extend_from_slice(&seg_len.to_be_bytes());
        injected.extend_from_slice(&text);
        injected.extend_from_slice(&jpeg[2..]);
        let comments = scan_com_markers(&injected);
        assert_eq!(comments.len(), 1);
        // Latin-1 fallback: each byte maps to its Unicode code point.
        // 0xC0 = 'À', 0xC1 = 'Á', 0xFE = 'þ', 0xFF = 'ÿ'
        assert_eq!(comments[0], "\u{00C0}\u{00C1}\u{00FE}\u{00FF}");
        // Must NOT contain U+FFFD replacement characters.
        assert!(!comments[0].contains('\u{FFFD}'));
    }

    // ═════════════════════════════════════════════════════════════════════
    // scan_jfif_density tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn scan_jfif_density_dpi() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let jpeg = inject_jfif_app0(&jpeg, 1, 300, 300);
        let density = scan_jfif_density(&jpeg);
        assert_eq!(density, Some(JpegPixelDensity::Dpi { x: 300, y: 300 }));
    }

    #[test]
    fn scan_jfif_density_dpcm() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let jpeg = inject_jfif_app0(&jpeg, 2, 118, 118);
        let density = scan_jfif_density(&jpeg);
        assert_eq!(density, Some(JpegPixelDensity::Dpcm { x: 118, y: 118 }));
    }

    #[test]
    fn scan_jfif_density_aspect_ratio() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let jpeg = inject_jfif_app0(&jpeg, 0, 1, 1);
        let density = scan_jfif_density(&jpeg);
        assert_eq!(density, Some(JpegPixelDensity::AspectRatio { x: 1, y: 1 }));
    }

    #[test]
    fn scan_jfif_density_no_app0() {
        // A JPEG without our injected APP0 may or may not have one from the encoder.
        // Test with empty data → should be None.
        assert_eq!(scan_jfif_density(&[]), None);
        assert_eq!(scan_jfif_density(b"not jpeg"), None);
    }

    #[test]
    fn scan_jfif_density_non_square() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let jpeg = inject_jfif_app0(&jpeg, 1, 300, 600);
        let density = scan_jfif_density(&jpeg);
        assert_eq!(density, Some(JpegPixelDensity::Dpi { x: 300, y: 600 }));
    }

    // ═════════════════════════════════════════════════════════════════════
    // decode / decode_reader tests (task 3.6)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn decode_rgb_8bit() {
        let jpeg = build_jpeg_rgb(4, 3, 200, 100, 50);
        let decoded = decode(&jpeg).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(img) => {
                assert_eq!(img.width(), 4);
                assert_eq!(img.height(), 3);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
        assert_eq!(decoded.metadata.source_bit_depth, JpegBitDepth::Eight);
        assert_eq!(decoded.metadata.color_space, JpegColorSpace::Srgb);
    }

    #[test]
    fn decode_grayscale_8bit() {
        let jpeg = build_jpeg_gray(4, 3, 128);
        let decoded = decode(&jpeg).unwrap();
        match &decoded.image {
            JpegImage::SrgbMono8(img) => {
                assert_eq!(img.width(), 4);
                assert_eq!(img.height(), 3);
            }
            other => panic!("expected SrgbMono8, got {:?}", other),
        }
        assert_eq!(decoded.metadata.source_bit_depth, JpegBitDepth::Eight);
    }

    #[test]
    fn decode_preserves_dimensions_rgb() {
        for &(w, h) in &[(1, 1), (2, 2), (8, 8), (16, 9), (100, 50)] {
            let jpeg = build_jpeg_rgb(w, h, 0, 0, 0);
            let decoded = decode(&jpeg).unwrap();
            match &decoded.image {
                JpegImage::Srgb8(img) => {
                    assert_eq!(img.width(), w as usize, "width mismatch for {w}x{h}");
                    assert_eq!(img.height(), h as usize, "height mismatch for {w}x{h}");
                }
                other => panic!("expected Srgb8 for {w}x{h}, got {:?}", other),
            }
        }
    }

    #[test]
    fn decode_preserves_dimensions_gray() {
        for &(w, h) in &[(1, 1), (8, 8), (16, 9)] {
            let jpeg = build_jpeg_gray(w, h, 0);
            let decoded = decode(&jpeg).unwrap();
            match &decoded.image {
                JpegImage::SrgbMono8(img) => {
                    assert_eq!(img.width(), w as usize);
                    assert_eq!(img.height(), h as usize);
                }
                other => panic!("expected SrgbMono8 for {w}x{h}, got {:?}", other),
            }
        }
    }

    #[test]
    fn decode_invalid_data_returns_error() {
        let result = decode(b"this is not a jpeg");
        assert!(result.is_err());
    }

    #[test]
    fn decode_empty_data_returns_error() {
        let result = decode(b"");
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_jpeg_returns_error() {
        let jpeg = build_jpeg_rgb(4, 4, 100, 100, 100);
        let truncated = &jpeg[..jpeg.len() / 2];
        let result = decode(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn decode_debug_output_shows_variant_and_dimensions() {
        let jpeg = build_jpeg_rgb(8, 6, 0, 0, 0);
        let decoded = decode(&jpeg).unwrap();
        let dbg = format!("{:?}", decoded.image);
        assert_eq!(dbg, "Srgb8(8x6)");
    }

    #[test]
    fn decode_debug_output_gray() {
        let jpeg = build_jpeg_gray(3, 2, 0);
        let decoded = decode(&jpeg).unwrap();
        let dbg = format!("{:?}", decoded.image);
        assert_eq!(dbg, "SrgbMono8(3x2)");
    }

    #[test]
    fn decode_metadata_no_exif() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let decoded = decode(&jpeg).unwrap();
        // A minimal encoder-produced JPEG typically has no EXIF.
        // (The encoder may or may not produce exif data.)
        // Just verify the metadata is accessible.
        let _ = &decoded.metadata.exif;
        let _ = &decoded.metadata.raw_exif;
    }

    #[test]
    fn decode_metadata_color_space_srgb_no_icc() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let decoded = decode(&jpeg).unwrap();
        // No ICC profile embedded by default → Srgb
        if decoded.metadata.icc_profile.is_none() {
            assert_eq!(decoded.metadata.color_space, JpegColorSpace::Srgb);
        }
    }

    #[test]
    fn decode_with_com_marker() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let jpeg = inject_com_marker(&jpeg, "Test comment");
        let decoded = decode(&jpeg).unwrap();
        assert!(
            decoded
                .metadata
                .comments
                .contains(&"Test comment".to_string())
        );
    }

    #[test]
    fn decode_with_jfif_density() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let jpeg = inject_jfif_app0(&jpeg, 1, 72, 72);
        let decoded = decode(&jpeg).unwrap();
        // Our injected APP0 should be picked up.
        // Note: the encoder may also inject its own APP0, and ours comes first.
        assert_eq!(
            decoded.metadata.pixel_density,
            Some(JpegPixelDensity::Dpi { x: 72, y: 72 })
        );
    }

    #[test]
    fn decode_reader_rgb_8bit() {
        let jpeg = build_jpeg_rgb(4, 3, 200, 100, 50);
        let decoded = decode_reader(std::io::Cursor::new(&jpeg)).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(img) => {
                assert_eq!(img.width(), 4);
                assert_eq!(img.height(), 3);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn decode_reader_gray_8bit() {
        let jpeg = build_jpeg_gray(4, 3, 128);
        let decoded = decode_reader(std::io::Cursor::new(&jpeg)).unwrap();
        match &decoded.image {
            JpegImage::SrgbMono8(img) => {
                assert_eq!(img.width(), 4);
                assert_eq!(img.height(), 3);
            }
            other => panic!("expected SrgbMono8, got {:?}", other),
        }
    }

    #[test]
    fn decode_reader_has_com_and_density() {
        // After R2 fixup, decode_reader buffers internally and delegates
        // to decode(), so COM comments and JFIF density are available.
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let jpeg_with_com = inject_com_marker(&jpeg, "visible");
        let jpeg_with_both = inject_jfif_app0(&jpeg_with_com, 1, 150, 150);
        let decoded = decode_reader(std::io::Cursor::new(&jpeg_with_both)).unwrap();
        // Comments should be populated from the reader path.
        assert!(decoded.metadata.comments.contains(&"visible".to_string()));
        // Density should be populated from the reader path.
        assert_eq!(
            decoded.metadata.pixel_density,
            Some(JpegPixelDensity::Dpi { x: 150, y: 150 })
        );
    }

    #[test]
    fn decode_reader_matches_decode_image() {
        // Verify both paths produce the same image variant and dimensions.
        let jpeg = build_jpeg_rgb(8, 6, 100, 150, 200);
        let d1 = decode(&jpeg).unwrap();
        let d2 = decode_reader(std::io::Cursor::new(&jpeg)).unwrap();
        match (&d1.image, &d2.image) {
            (JpegImage::Srgb8(a), JpegImage::Srgb8(b)) => {
                assert_eq!(a.width(), b.width());
                assert_eq!(a.height(), b.height());
            }
            _ => panic!("variant mismatch"),
        }
    }

    #[test]
    fn decode_reader_invalid_data_returns_error() {
        let result = decode_reader(std::io::Cursor::new(b"not jpeg"));
        assert!(result.is_err());
    }

    #[test]
    fn decode_reader_empty_data_returns_error() {
        let result = decode_reader(std::io::Cursor::new(b""));
        assert!(result.is_err());
    }

    #[test]
    fn decode_pixel_values_approximately_correct() {
        // JPEG is lossy, so we can't test exact values, but we can check
        // that a solid-colour image decodes to approximately the right values.
        let jpeg = build_jpeg_rgb(8, 8, 200, 100, 50);
        let decoded = decode(&jpeg).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(img) => {
                let px = img.get(4, 4).unwrap();
                // JPEG compression introduces some error; allow ±10
                assert!((px.r.0 as i16 - 200).unsigned_abs() < 10, "red: {}", px.r.0);
                assert!(
                    (px.g.0 as i16 - 100).unsigned_abs() < 10,
                    "green: {}",
                    px.g.0
                );
                assert!((px.b.0 as i16 - 50).unsigned_abs() < 10, "blue: {}", px.b.0);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn decode_pixel_values_gray_approximately_correct() {
        let jpeg = build_jpeg_gray(8, 8, 180);
        let decoded = decode(&jpeg).unwrap();
        match &decoded.image {
            JpegImage::SrgbMono8(img) => {
                let px = img.get(4, 4).unwrap();
                assert!(
                    (px.0.0 as i16 - 180).unsigned_abs() < 10,
                    "gray: {}",
                    px.0.0
                );
            }
            other => panic!("expected SrgbMono8, got {:?}", other),
        }
    }

    #[test]
    fn decode_1x1_rgb() {
        let jpeg = build_jpeg_rgb(1, 1, 255, 0, 0);
        let decoded = decode(&jpeg).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(img) => {
                assert_eq!(img.width(), 1);
                assert_eq!(img.height(), 1);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn decode_1x1_gray() {
        let jpeg = build_jpeg_gray(1, 1, 42);
        let decoded = decode(&jpeg).unwrap();
        match &decoded.image {
            JpegImage::SrgbMono8(img) => {
                assert_eq!(img.width(), 1);
                assert_eq!(img.height(), 1);
            }
            other => panic!("expected SrgbMono8, got {:?}", other),
        }
    }

    #[test]
    fn decode_metadata_source_bit_depth_8_for_rgb() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let decoded = decode(&jpeg).unwrap();
        assert_eq!(decoded.metadata.source_bit_depth, JpegBitDepth::Eight);
    }

    #[test]
    fn decode_metadata_source_bit_depth_8_for_gray() {
        let jpeg = build_jpeg_gray(2, 2, 0);
        let decoded = decode(&jpeg).unwrap();
        assert_eq!(decoded.metadata.source_bit_depth, JpegBitDepth::Eight);
    }

    #[test]
    fn decode_error_maps_io_error() {
        // Construct a jpeg_decoder Io error and verify mapping
        let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "test");
        let jpeg_err = jpeg_decoder::Error::Io(io_err);
        let mapped = decode_error(jpeg_err);
        match mapped {
            IoError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof),
            other => panic!("expected IoError::Io, got {:?}", other),
        }
    }

    #[test]
    fn decode_error_maps_format_error() {
        let jpeg_err = jpeg_decoder::Error::Format("bad format".to_string());
        let mapped = decode_error(jpeg_err);
        match mapped {
            IoError::DecodeFailed { .. } => {} // expected
            other => panic!("expected IoError::DecodeFailed, got {:?}", other),
        }
    }

    #[test]
    fn decode_multiple_com_markers() {
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let jpeg = inject_com_marker(&jpeg, "Comment A");
        let jpeg = inject_com_marker(&jpeg, "Comment B");
        let decoded = decode(&jpeg).unwrap();
        assert!(decoded.metadata.comments.len() >= 2);
        assert!(decoded.metadata.comments.contains(&"Comment A".to_string()));
        assert!(decoded.metadata.comments.contains(&"Comment B".to_string()));
    }

    #[test]
    fn decode_exhaustive_image_match() {
        // Verify all JpegImage variants can be matched exhaustively
        let jpeg = build_jpeg_rgb(2, 2, 0, 0, 0);
        let decoded = decode(&jpeg).unwrap();
        match decoded.image {
            JpegImage::SrgbMono8(_) => {}
            JpegImage::SrgbMono16(_) => {}
            JpegImage::Srgb8(_) => {}
        }
    }

    #[test]
    fn decode_large_image() {
        // Test with a larger image to ensure no panics on buffer sizing.
        let jpeg = build_jpeg_rgb(256, 256, 100, 200, 50);
        let decoded = decode(&jpeg).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(img) => {
                assert_eq!(img.width(), 256);
                assert_eq!(img.height(), 256);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // IfdEntry struct tests (R5 fixup)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn ifd_entry_struct_fields() {
        let entry = IfdEntry {
            tag: 0x0112,
            tiff_type: 3,
            count: 1,
            value_or_offset: 6,
        };
        assert_eq!(entry.tag, 0x0112);
        assert_eq!(entry.tiff_type, 3);
        assert_eq!(entry.count, 1);
        assert_eq!(entry.value_or_offset, 6);
    }

    #[test]
    fn ifd_entry_debug() {
        let entry = IfdEntry {
            tag: 0x010F,
            tiff_type: 2,
            count: 5,
            value_or_offset: 100,
        };
        let dbg = format!("{:?}", entry);
        assert!(dbg.contains("tag"));
        assert!(dbg.contains("tiff_type"));
    }

    #[test]
    fn ifd_entry_eq() {
        let a = IfdEntry {
            tag: 1,
            tiff_type: 2,
            count: 3,
            value_or_offset: 4,
        };
        let b = IfdEntry {
            tag: 1,
            tiff_type: 2,
            count: 3,
            value_or_offset: 4,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn ifd_entry_copy() {
        let a = IfdEntry {
            tag: 1,
            tiff_type: 2,
            count: 3,
            value_or_offset: 4,
        };
        let b = a;
        assert_eq!(a, b);
    }

    // ═════════════════════════════════════════════════════════════════════
    // JpegSamplingFactor tests (task 4.1)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn jpeg_sampling_factor_variants() {
        let f1 = JpegSamplingFactor::F1x1;
        let f2 = JpegSamplingFactor::F2x1;
        let f3 = JpegSamplingFactor::F2x2;
        assert_ne!(f1, f2);
        assert_ne!(f2, f3);
        assert_ne!(f1, f3);
    }

    #[test]
    fn jpeg_sampling_factor_is_copy() {
        let f = JpegSamplingFactor::F1x1;
        let f2 = f;
        assert_eq!(f, f2);
    }

    #[test]
    fn jpeg_sampling_factor_debug() {
        assert_eq!(format!("{:?}", JpegSamplingFactor::F1x1), "F1x1");
        assert_eq!(format!("{:?}", JpegSamplingFactor::F2x1), "F2x1");
        assert_eq!(format!("{:?}", JpegSamplingFactor::F2x2), "F2x2");
    }

    // ═════════════════════════════════════════════════════════════════════
    // JpegEncodeOptions tests (task 4.2)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn jpeg_encode_options_default() {
        let opts = JpegEncodeOptions::default();
        assert_eq!(opts.quality, 85);
        assert_eq!(opts.sampling_factor, None);
        assert!(!opts.progressive);
    }

    #[test]
    fn jpeg_encode_options_custom() {
        let opts = JpegEncodeOptions {
            quality: 95,
            sampling_factor: Some(JpegSamplingFactor::F1x1),
            progressive: true,
        };
        assert_eq!(opts.quality, 95);
        assert_eq!(opts.sampling_factor, Some(JpegSamplingFactor::F1x1));
        assert!(opts.progressive);
    }

    #[test]
    fn jpeg_encode_options_debug() {
        let opts = JpegEncodeOptions::default();
        let dbg = format!("{:?}", opts);
        assert!(dbg.contains("quality"));
        assert!(dbg.contains("85"));
    }

    #[test]
    fn jpeg_encode_options_clone() {
        let opts = JpegEncodeOptions {
            quality: 50,
            sampling_factor: Some(JpegSamplingFactor::F2x2),
            progressive: true,
        };
        let opts2 = opts.clone();
        assert_eq!(opts2.quality, 50);
        assert_eq!(opts2.sampling_factor, Some(JpegSamplingFactor::F2x2));
        assert!(opts2.progressive);
    }

    // ═════════════════════════════════════════════════════════════════════
    // JpegPixel trait tests (task 4.3)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn jpeg_pixel_srgb_mono8() {
        // SrgbMono8 implements JpegPixel with Luma colour type
        assert_eq!(SrgbMono8::JPEG_COLOR_TYPE, jpeg_encoder::ColorType::Luma);
    }

    #[test]
    fn jpeg_pixel_srgb8() {
        // Srgb8 implements JpegPixel with Rgb colour type
        assert_eq!(Srgb8::JPEG_COLOR_TYPE, jpeg_encoder::ColorType::Rgb);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Encode tests (tasks 4.4–4.8)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn encode_srgb8_roundtrip() {
        let img = Image::fill(8, 8, Srgb8::new(200, 100, 50));
        let bytes = encode(&img, &JpegEncodeOptions::default()).unwrap();
        // Verify it decodes back to an Srgb8 image with correct dimensions.
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(dec_img) => {
                assert_eq!(dec_img.width(), 8);
                assert_eq!(dec_img.height(), 8);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn encode_srgb_mono8_roundtrip() {
        let img = Image::fill(8, 8, SrgbMono8::new(128));
        let bytes = encode(&img, &JpegEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            JpegImage::SrgbMono8(dec_img) => {
                assert_eq!(dec_img.width(), 8);
                assert_eq!(dec_img.height(), 8);
            }
            other => panic!("expected SrgbMono8, got {:?}", other),
        }
    }

    #[test]
    fn encode_quality_affects_size() {
        let img = Image::fill(64, 64, Srgb8::new(128, 64, 32));
        let low = encode(
            &img,
            &JpegEncodeOptions {
                quality: 10,
                ..Default::default()
            },
        )
        .unwrap();
        let high = encode(
            &img,
            &JpegEncodeOptions {
                quality: 100,
                ..Default::default()
            },
        )
        .unwrap();
        // Higher quality should produce a larger file.
        assert!(
            high.len() > low.len(),
            "high quality ({}) should be larger than low quality ({})",
            high.len(),
            low.len()
        );
    }

    #[test]
    fn encode_writer_byte_identical_to_encode() {
        let img = Image::fill(4, 4, Srgb8::new(100, 150, 200));
        let opts = JpegEncodeOptions::default();
        let bytes_encode = encode(&img, &opts).unwrap();
        let mut bytes_writer = Vec::new();
        encode_writer(&img, &mut bytes_writer, &opts).unwrap();
        assert_eq!(bytes_encode, bytes_writer);
    }

    #[test]
    fn encode_dimensions_roundtrip() {
        for &(w, h) in &[(1u16, 1), (2, 2), (8, 6), (16, 9), (100, 50)] {
            let img = Image::fill(w as usize, h as usize, Srgb8::new(0, 0, 0));
            let bytes = encode(&img, &JpegEncodeOptions::default()).unwrap();
            let decoded = decode(&bytes).unwrap();
            match &decoded.image {
                JpegImage::Srgb8(dec_img) => {
                    assert_eq!(dec_img.width(), w as usize, "width for {w}x{h}");
                    assert_eq!(dec_img.height(), h as usize, "height for {w}x{h}");
                }
                other => panic!("expected Srgb8 for {w}x{h}, got {:?}", other),
            }
        }
    }

    #[test]
    fn encode_progressive_accepted() {
        let img = Image::fill(8, 8, Srgb8::new(0, 0, 0));
        let opts = JpegEncodeOptions {
            progressive: true,
            ..Default::default()
        };
        let bytes = encode(&img, &opts).unwrap();
        // Verify the result is a valid JPEG.
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(_) => {}
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn encode_sampling_factor_f1x1() {
        let img = Image::fill(8, 8, Srgb8::new(0, 0, 0));
        let opts = JpegEncodeOptions {
            sampling_factor: Some(JpegSamplingFactor::F1x1),
            ..Default::default()
        };
        let bytes = encode(&img, &opts).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(_) => {}
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn encode_sampling_factor_f2x1() {
        let img = Image::fill(8, 8, Srgb8::new(0, 0, 0));
        let opts = JpegEncodeOptions {
            sampling_factor: Some(JpegSamplingFactor::F2x1),
            ..Default::default()
        };
        let bytes = encode(&img, &opts).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(_) => {}
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn encode_sampling_factor_f2x2() {
        let img = Image::fill(8, 8, Srgb8::new(0, 0, 0));
        let opts = JpegEncodeOptions {
            sampling_factor: Some(JpegSamplingFactor::F2x2),
            ..Default::default()
        };
        let bytes = encode(&img, &opts).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(_) => {}
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn encode_jpeg_image_srgb8() {
        let img = JpegImage::Srgb8(Image::fill(4, 4, Srgb8::new(128, 64, 32)));
        let bytes = encode_jpeg_image(&img, &JpegEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(dec_img) => {
                assert_eq!(dec_img.width(), 4);
                assert_eq!(dec_img.height(), 4);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn encode_jpeg_image_srgb_mono8() {
        let img = JpegImage::SrgbMono8(Image::fill(4, 4, SrgbMono8::new(200)));
        let bytes = encode_jpeg_image(&img, &JpegEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            JpegImage::SrgbMono8(_) => {}
            other => panic!("expected SrgbMono8, got {:?}", other),
        }
    }

    #[test]
    fn encode_jpeg_image_srgb_mono16_unsupported() {
        let img = JpegImage::SrgbMono16(Image::fill(2, 2, SrgbMono16::new(4000)));
        let result = encode_jpeg_image(&img, &JpegEncodeOptions::default());
        match result {
            Err(IoError::UnsupportedFeature { .. }) => {} // expected
            other => panic!("expected UnsupportedFeature, got {:?}", other),
        }
    }

    #[test]
    fn encode_pixel_values_approximately_correct() {
        // JPEG is lossy, but with quality 100, values should be close.
        let img = Image::fill(8, 8, Srgb8::new(200, 100, 50));
        let opts = JpegEncodeOptions {
            quality: 100,
            ..Default::default()
        };
        let bytes = encode(&img, &opts).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(dec_img) => {
                let px = dec_img.get(4, 4).unwrap();
                assert!((px.r.0 as i16 - 200).unsigned_abs() < 5, "red: {}", px.r.0);
                assert!(
                    (px.g.0 as i16 - 100).unsigned_abs() < 5,
                    "green: {}",
                    px.g.0
                );
                assert!((px.b.0 as i16 - 50).unsigned_abs() < 5, "blue: {}", px.b.0);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn encode_error_maps_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "test");
        let enc_err = jpeg_encoder::EncodingError::IoError(io_err);
        let mapped = encode_error(enc_err);
        match mapped {
            IoError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::BrokenPipe),
            other => panic!("expected IoError::Io, got {:?}", other),
        }
    }

    #[test]
    fn encode_error_maps_encoding_error() {
        // BadImageData is a non-IO encoding error.
        let enc_err = jpeg_encoder::EncodingError::BadImageData {
            length: 10,
            required: 12,
        };
        let mapped = encode_error(enc_err);
        match mapped {
            IoError::EncodeFailed { .. } => {} // expected
            other => panic!("expected IoError::EncodeFailed, got {:?}", other),
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // P1-3: JPEG spec limits dimensions to u16 in the SOFn marker.
    // Casting `as u16` silently wraps; the encoder must validate and
    // return Err instead.
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn encode_rejects_width_exceeding_u16_max() {
        // 65_536 wide ⇒ exceeds JPEG's 16-bit width field by 1.
        // Mono8 ⇒ 64 KiB allocation, cheap for a test.
        let img: Image<SrgbMono8> = Image::fill(u16::MAX as usize + 1, 1, SrgbMono8::new(0));
        let result = encode(&img, &JpegEncodeOptions::default());
        match result {
            Err(IoError::UnsupportedFeature { reason }) => {
                assert!(
                    reason.contains("65535") || reason.contains("dimensions"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected UnsupportedFeature, got {:?}", other),
        }
    }

    #[test]
    fn encode_rejects_height_exceeding_u16_max() {
        let img: Image<SrgbMono8> = Image::fill(1, u16::MAX as usize + 1, SrgbMono8::new(0));
        let result = encode(&img, &JpegEncodeOptions::default());
        assert!(
            matches!(result, Err(IoError::UnsupportedFeature { .. })),
            "expected UnsupportedFeature, got {:?}",
            result
        );
    }

    #[test]
    fn encode_accepts_max_valid_dimension() {
        // 65_535 (= u16::MAX) is the largest spec-compliant width.
        // Use 1px tall ⇒ 64 KiB, encodes quickly.
        let img: Image<SrgbMono8> = Image::fill(u16::MAX as usize, 1, SrgbMono8::new(42));
        let result = encode(&img, &JpegEncodeOptions::default());
        assert!(
            result.is_ok(),
            "expected Ok for max valid width, got {:?}",
            result.err()
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // Roundtrip integration tests (Phase 5)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn roundtrip_srgb8_encode_decode() {
        let img = Image::fill(16, 12, Srgb8::new(100, 150, 200));
        let bytes = encode(&img, &JpegEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            JpegImage::Srgb8(dec_img) => {
                assert_eq!(dec_img.width(), 16);
                assert_eq!(dec_img.height(), 12);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
        assert_eq!(decoded.metadata.source_bit_depth, JpegBitDepth::Eight);
    }

    #[test]
    fn roundtrip_srgb_mono8_encode_decode() {
        let img = Image::fill(16, 12, SrgbMono8::new(180));
        let bytes = encode(&img, &JpegEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match &decoded.image {
            JpegImage::SrgbMono8(dec_img) => {
                assert_eq!(dec_img.width(), 16);
                assert_eq!(dec_img.height(), 12);
            }
            other => panic!("expected SrgbMono8, got {:?}", other),
        }
    }

    #[test]
    fn roundtrip_via_encode_jpeg_image() {
        let orig = JpegImage::Srgb8(Image::fill(8, 8, Srgb8::new(50, 100, 150)));
        let bytes = encode_jpeg_image(&orig, &JpegEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match (&orig, &decoded.image) {
            (JpegImage::Srgb8(a), JpegImage::Srgb8(b)) => {
                assert_eq!(a.width(), b.width());
                assert_eq!(a.height(), b.height());
            }
            _ => panic!("variant mismatch"),
        }
    }

    #[test]
    fn decode_reader_matches_decode_fully() {
        // After R2 fixup, decode_reader should produce identical results
        // to decode, including comments and pixel density.
        let jpeg = build_jpeg_rgb(8, 6, 100, 150, 200);
        let jpeg = inject_com_marker(&jpeg, "reader test");
        let jpeg = inject_jfif_app0(&jpeg, 1, 72, 72);
        let d1 = decode(&jpeg).unwrap();
        let d2 = decode_reader(std::io::Cursor::new(&jpeg)).unwrap();
        // Image dimensions and variant match.
        match (&d1.image, &d2.image) {
            (JpegImage::Srgb8(a), JpegImage::Srgb8(b)) => {
                assert_eq!(a.width(), b.width());
                assert_eq!(a.height(), b.height());
            }
            _ => panic!("variant mismatch"),
        }
        // Metadata matches.
        assert_eq!(d1.metadata.comments, d2.metadata.comments);
        assert_eq!(d1.metadata.pixel_density, d2.metadata.pixel_density);
        assert_eq!(d1.metadata.source_bit_depth, d2.metadata.source_bit_depth);
        assert_eq!(d1.metadata.color_space, d2.metadata.color_space);
    }
}
