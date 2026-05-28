//! PNG decoding and encoding.
//!
//! # Decoding
//!
//! The primary entry point is [`decode`], which takes a byte slice and returns
//! a [`PngDecoded`] — a struct containing the pixel data as a [`PngImage`]
//! and ancillary information as [`PngMetadata`].
//!
//! ```no_run
//! # use irys_cv_io::png;
//! let bytes = std::fs::read("photo.png").unwrap();
//! let decoded = png::decode(&bytes).unwrap();
//! // Just the pixels:
//! let image = decoded.image;
//! // Or inspect metadata too:
//! // let color_space = &decoded.metadata.color_space;
//! ```
//!
//! For streaming sources use [`decode_reader`], which accepts any `impl Read`.
//!
//! # Design rationale
//!
//! - **`PngImage` is exhaustive** (no `#[non_exhaustive]`).  It is the spec
//!   sheet for what a PNG file can produce.  Adding a variant (e.g. `Pq16` for
//!   HDR) is a genuine semantic change that callers must handle — a compile
//!   error is the correct response.
//!
//! - **`PngDecoded` is `#[non_exhaustive]`**.  It is a return-only struct that
//!   may gain fields (e.g. decode warnings) in the future without breaking
//!   downstream code.
//!
//! - **Transfer-function detection**.  The decoder inspects colour-space
//!   metadata chunks (`cICP` > `iCCP` > `sRGB` > `gAMA`+`cHRM`) to decide
//!   whether pixel data is sRGB-encoded or linear.  The pixel type returned
//!   in [`PngImage`] encodes this assumption so the compiler can enforce
//!   correct usage (e.g. rejecting linear-math operations on gamma-encoded
//!   data).  For untagged PNGs (no colour-space chunk), 8-bit defaults to
//!   sRGB and 16-bit defaults to linear, following widespread convention.
//!
//! - **8-bit sRGB → `SrgbMono8`/`SrgbMonoA8`/`Srgb8`/`Srgba8`**.  These
//!   types do not implement `LinearSpace`, so the compiler rejects
//!   linear-math operations on gamma-encoded data (ADR-0007).  The sRGB
//!   transfer function is defined per-channel and applies identically to
//!   grayscale images.
//!
//! - **8-bit linear → `Mono8`/`MonoA8`/`Rgb8`/`Rgba8`**.  Used when the
//!   PNG signals linear transfer (e.g. `gAMA` = 1.0, `cICP` TF 8).
//!   These types implement `LinearSpace`.
//!
//! - **16-bit linear → `Mono16`/`MonoA16`/`Rgb16`/`Rgba16`**.  The default
//!   for untagged 16-bit PNGs.  These types have `LinearSpace`.
//!
//! - **16-bit sRGB → `SrgbMono16`/`SrgbMonoA16`/`Srgb16`/`Srgba16`**.
//!   Used when the PNG carries sRGB/iCCP/non-linear gamma metadata with
//!   16-bit data (common in Photoshop, GIMP, Lightroom exports).  These
//!   types do not implement `LinearSpace`.
//!
//! - **Indexed → `Image<Indexed8>` + `Box<[Srgba8; 256]>`**.  The palette
//!   is boxed to keep the enum compact (~40 bytes vs ~1064 bytes inline).
//!   Palette entries are `Srgba8` because PNG palettes are sRGB (PLTE + tRNS
//!   merged into RGBA).
//!
//! - **Metadata is always returned**.  The `png` crate parses all ancillary
//!   chunks before image data regardless, so constructing [`PngMetadata`] is
//!   essentially free.  A single [`decode`] function replaces the previous
//!   `decode` / `decode_with_metadata` split — less API surface, no
//!   "I should have called the other function" moments.
//!
//! # Encoding
//!
//! The primary entry point is [`encode`], which accepts any image whose
//! pixel type implements [`PngPixel`] (a sealed trait covering exactly
//! the pixel types that PNG can represent).
//!
//! ```no_run
//! # use irys_cv::image::Image;
//! # use irys_cv::pixel::Srgb8;
//! # use irys_cv_io::png::{self, PngEncodeOptions};
//! let image = Image::fill(320, 240, Srgb8::new(0, 0, 0));
//! let bytes = png::encode(&image, &PngEncodeOptions::default()).unwrap();
//! ```
//!
//! Colour-space chunks (`sRGB`, `gAMA`) are written **automatically**
//! based on the pixel type's transfer function — the type is the source
//! of truth.  Additional metadata (text chunks, timestamps, pixel
//! dimensions, compression) is passed via [`PngEncodeOptions`].
//!
//! For streaming output use [`encode_writer`].  For palette-indexed
//! images use [`encode_indexed`].  For roundtripping a decoded
//! [`PngImage`] use [`encode_png_image`].
//!
//! ## Compile-time safety
//!
//! Attempting to encode a pixel type that PNG cannot represent (e.g.
//! `RgbF32`, `Bgr8`) is a compile-time error — the type simply does
//! not implement [`PngPixel`].
//!
//! ```compile_fail
//! # use irys_cv::image::Image;
//! # use irys_cv::pixel::RgbF32;
//! # use irys_cv_io::png::{self, PngEncodeOptions};
//! // ERROR: `RgbF32` does not implement `PngPixel`
//! let image = Image::fill(1, 1, RgbF32 { r: 0.0, g: 0.0, b: 0.0 });
//! let _ = png::encode(&image, &PngEncodeOptions::default());
//! ```

use irys_cv::image::{Image, ImageView, PlainImage};
use irys_cv::pixel::{
    Indexed8, Mono8, Mono16, MonoA8, MonoA16, PlainPixel, Rgb8, Rgb16, Rgba8, Rgba16, Srgb8,
    Srgb16, SrgbMono8, SrgbMono16, SrgbMonoA8, SrgbMonoA16, Srgba8, Srgba16,
};

use crate::IoError;

// ═══════════════════════════════════════════════════════════════════════════════
// PngImage — per-codec exhaustive output enum
// ═══════════════════════════════════════════════════════════════════════════════

/// Decoded PNG pixel data.
///
/// Each variant corresponds to a PNG colour type × bit depth × transfer
/// function combination.  The pixel type encodes the transfer-function
/// assumption so the compiler can enforce correct usage (e.g. rejecting
/// linear-math operations on gamma-encoded data).
///
/// Transfer function detection uses colour-space metadata chunks in PNG
/// priority order (`cICP` > `iCCP` > `sRGB` > `gAMA`+`cHRM`).  For
/// untagged PNGs (no colour-space chunk): 8-bit defaults to sRGB, 16-bit
/// defaults to linear.  See [`TransferAssumption`] (crate-private).
///
/// | PNG colour type  | Bit depth | Transfer | Variant        | Pixel type      |
/// |------------------|-----------|----------|----------------|-----------------|
/// | Grayscale        | 1/2/4/8   | sRGB*    | `SrgbMono8`    | `SrgbMono8`     |
/// | Grayscale        | 1/2/4/8   | linear   | `Mono8`        | `Mono8`         |
/// | Grayscale        | 16        | sRGB*    | `SrgbMono16`   | `SrgbMono16`    |
/// | Grayscale        | 16        | linear   | `Mono16`       | `Mono16`        |
/// | Grayscale+Alpha  | 8         | sRGB*    | `SrgbMonoA8`   | `SrgbMonoA8`    |
/// | Grayscale+Alpha  | 8         | linear   | `MonoA8`       | `MonoA8`        |
/// | Grayscale+Alpha  | 16        | sRGB*    | `SrgbMonoA16`  | `SrgbMonoA16`   |
/// | Grayscale+Alpha  | 16        | linear   | `MonoA16`      | `MonoA16`       |
/// | Truecolour       | 8         | sRGB*    | `Srgb8`        | `Srgb8`         |
/// | Truecolour       | 8         | linear   | `Rgb8`         | `Rgb8`          |
/// | Truecolour       | 16        | sRGB*    | `Srgb16`       | `Srgb16`        |
/// | Truecolour       | 16        | linear   | `Rgb16`        | `Rgb16`         |
/// | Truecolour+Alpha | 8         | sRGB*    | `Srgba8`       | `Srgba8`        |
/// | Truecolour+Alpha | 8         | linear   | `Rgba8`        | `Rgba8`         |
/// | Truecolour+Alpha | 16        | sRGB*    | `Srgba16`      | `Srgba16`       |
/// | Truecolour+Alpha | 16        | linear   | `Rgba16`       | `Rgba16`        |
/// | Indexed          | 1/2/4/8   | n/a      | `Indexed8`     | `Indexed8`      |
///
/// \* "sRGB" includes: explicit `sRGB` chunk, `iCCP`, gamma ≈ 0.45455,
///   sRGB/BT.709 cICP transfer functions.  For untagged PNGs (no
///   colour-space chunk): 8-bit defaults to sRGB, 16-bit defaults to
///   linear.
///
/// Sub-byte grayscale (1/2/4-bit) is expanded to 8-bit with full-range
/// scaling (`1-bit: 1→255`, `2-bit: ×85`, `4-bit: ×17`).
///
/// Sub-byte indexed (1/2/4-bit) is unpacked to one byte per index without
/// depalettising or value-scaling.
///
/// This enum is deliberately **not** `#[non_exhaustive]`.  It is the spec
/// sheet; adding a variant is semver-major.
///
/// The `Debug` impl shows the variant name and image dimensions (e.g.
/// `Srgb8(320x240)`) without dumping pixel data.
impl std::fmt::Debug for PngImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PngImage::SrgbMono8(img) => write!(f, "SrgbMono8({}x{})", img.width(), img.height()),
            PngImage::SrgbMonoA8(img) => write!(f, "SrgbMonoA8({}x{})", img.width(), img.height()),
            PngImage::Srgb8(img) => write!(f, "Srgb8({}x{})", img.width(), img.height()),
            PngImage::Srgba8(img) => write!(f, "Srgba8({}x{})", img.width(), img.height()),
            PngImage::Mono8(img) => write!(f, "Mono8({}x{})", img.width(), img.height()),
            PngImage::MonoA8(img) => write!(f, "MonoA8({}x{})", img.width(), img.height()),
            PngImage::Rgb8(img) => write!(f, "Rgb8({}x{})", img.width(), img.height()),
            PngImage::Rgba8(img) => write!(f, "Rgba8({}x{})", img.width(), img.height()),
            PngImage::SrgbMono16(img) => write!(f, "SrgbMono16({}x{})", img.width(), img.height()),
            PngImage::SrgbMonoA16(img) => {
                write!(f, "SrgbMonoA16({}x{})", img.width(), img.height())
            }
            PngImage::Srgb16(img) => write!(f, "Srgb16({}x{})", img.width(), img.height()),
            PngImage::Srgba16(img) => write!(f, "Srgba16({}x{})", img.width(), img.height()),
            PngImage::Mono16(img) => write!(f, "Mono16({}x{})", img.width(), img.height()),
            PngImage::MonoA16(img) => write!(f, "MonoA16({}x{})", img.width(), img.height()),
            PngImage::Rgb16(img) => write!(f, "Rgb16({}x{})", img.width(), img.height()),
            PngImage::Rgba16(img) => write!(f, "Rgba16({}x{})", img.width(), img.height()),
            PngImage::Indexed8 { data, .. } => {
                write!(f, "Indexed8({}x{})", data.width(), data.height())
            }
        }
    }
}

pub enum PngImage {
    // ── 8-bit sRGB (gamma-encoded, no `LinearSpace`) ─────────────────────
    /// 8-bit sRGB grayscale (sub-byte depths expanded with full-range scaling).
    SrgbMono8(Image<SrgbMono8>),
    /// 8-bit sRGB grayscale + alpha.
    SrgbMonoA8(Image<SrgbMonoA8>),
    /// 8-bit sRGB truecolour (gamma-encoded, no `LinearSpace`).
    Srgb8(Image<Srgb8>),
    /// 8-bit sRGB truecolour + alpha (gamma-encoded, no `LinearSpace`).
    Srgba8(Image<Srgba8>),

    // ── 8-bit linear (has `LinearSpace`) ─────────────────────────────────
    /// 8-bit linear grayscale (sub-byte depths expanded with full-range scaling).
    Mono8(Image<Mono8>),
    /// 8-bit linear grayscale + alpha.
    MonoA8(Image<MonoA8>),
    /// 8-bit linear truecolour.
    Rgb8(Image<Rgb8>),
    /// 8-bit linear truecolour + alpha.
    Rgba8(Image<Rgba8>),

    // ── 16-bit sRGB (gamma-encoded, no `LinearSpace`) ────────────────────
    /// 16-bit sRGB grayscale.
    SrgbMono16(Image<SrgbMono16>),
    /// 16-bit sRGB grayscale + alpha.
    SrgbMonoA16(Image<SrgbMonoA16>),
    /// 16-bit sRGB truecolour.
    Srgb16(Image<Srgb16>),
    /// 16-bit sRGB truecolour + alpha.
    Srgba16(Image<Srgba16>),

    // ── 16-bit linear (has `LinearSpace`) ────────────────────────────────
    /// 16-bit linear grayscale.
    Mono16(Image<Mono16>),
    /// 16-bit linear grayscale + alpha.
    MonoA16(Image<MonoA16>),
    /// 16-bit linear truecolour.
    Rgb16(Image<Rgb16>),
    /// 16-bit linear truecolour + alpha.
    Rgba16(Image<Rgba16>),
    /// Palette-indexed image with a boxed 256-entry sRGBA palette.
    ///
    /// The palette merges `PLTE` (RGB) and `tRNS` (per-entry alpha) into
    /// `Srgba8` entries.  Entries beyond the actual palette size have
    /// `alpha = 255` and `r = g = b = 0`.
    ///
    /// To depalettise, move the palette out of the box and build a
    /// [`Depalettize`](irys_cv::transform::Depalettize) strategy:
    ///
    /// ```ignore
    /// let PngImage::Indexed8 { data, palette } = decoded.image else { /* … */ };
    /// let strategy = Depalettize::new(*palette); // move out of Box, no copy
    /// let rgb = convert_image(&data, &strategy);
    /// ```
    Indexed8 {
        /// The index data — one `Indexed8` per pixel.
        data: Image<Indexed8>,
        /// The 256-entry sRGBA palette (boxed to keep the enum compact).
        palette: Box<[Srgba8; 256]>,
    },
}

impl PngImage {
    /// Width in pixels, regardless of variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use irys_cv_io::png::PngImage;
    /// use irys_cv::image::Image;
    /// use irys_cv::pixel::Srgb8;
    ///
    /// let img = PngImage::Srgb8(Image::fill(320, 240, Srgb8::new(0, 0, 0)));
    /// assert_eq!(img.width(), 320);
    /// ```
    #[must_use]
    pub fn width(&self) -> usize {
        use irys_cv::image::ImageView;
        match self {
            PngImage::SrgbMono8(img) => img.width(),
            PngImage::SrgbMonoA8(img) => img.width(),
            PngImage::Srgb8(img) => img.width(),
            PngImage::Srgba8(img) => img.width(),
            PngImage::Mono8(img) => img.width(),
            PngImage::MonoA8(img) => img.width(),
            PngImage::Rgb8(img) => img.width(),
            PngImage::Rgba8(img) => img.width(),
            PngImage::SrgbMono16(img) => img.width(),
            PngImage::SrgbMonoA16(img) => img.width(),
            PngImage::Srgb16(img) => img.width(),
            PngImage::Srgba16(img) => img.width(),
            PngImage::Mono16(img) => img.width(),
            PngImage::MonoA16(img) => img.width(),
            PngImage::Rgb16(img) => img.width(),
            PngImage::Rgba16(img) => img.width(),
            PngImage::Indexed8 { data, .. } => data.width(),
        }
    }

    /// Height in pixels, regardless of variant.
    #[must_use]
    pub fn height(&self) -> usize {
        use irys_cv::image::ImageView;
        match self {
            PngImage::SrgbMono8(img) => img.height(),
            PngImage::SrgbMonoA8(img) => img.height(),
            PngImage::Srgb8(img) => img.height(),
            PngImage::Srgba8(img) => img.height(),
            PngImage::Mono8(img) => img.height(),
            PngImage::MonoA8(img) => img.height(),
            PngImage::Rgb8(img) => img.height(),
            PngImage::Rgba8(img) => img.height(),
            PngImage::SrgbMono16(img) => img.height(),
            PngImage::SrgbMonoA16(img) => img.height(),
            PngImage::Srgb16(img) => img.height(),
            PngImage::Srgba16(img) => img.height(),
            PngImage::Mono16(img) => img.height(),
            PngImage::MonoA16(img) => img.height(),
            PngImage::Rgb16(img) => img.height(),
            PngImage::Rgba16(img) => img.height(),
            PngImage::Indexed8 { data, .. } => data.height(),
        }
    }

    /// Image size (`width × height`), regardless of variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use irys_cv_io::png::PngImage;
    /// use irys_cv::image::Image;
    /// use irys_cv::pixel::Srgb8;
    ///
    /// let img = PngImage::Srgb8(Image::fill(320, 240, Srgb8::new(0, 0, 0)));
    /// let sz = img.size();
    /// assert_eq!(sz.width, 320);
    /// assert_eq!(sz.height, 240);
    /// ```
    #[must_use]
    pub fn size(&self) -> irys_cv::Size {
        // Reuse width/height to avoid a 17-arm match here. The width/height
        // dispatches are inlined by the compiler.
        irys_cv::Size::new(self.width(), self.height())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PngColorSpace — what the colour-space chunks told us
// ═══════════════════════════════════════════════════════════════════════════════

/// Colour-space information extracted from PNG metadata chunks.
///
/// Determined by chunk priority: `cICP` > `iCCP` > `sRGB` > `gAMA`+`cHRM`.
///
/// The pixel type in [`PngImage`] already encodes the *transfer function*
/// assumption.  This enum carries the *rest* — primaries, exact gamma, ICC
/// profile, rendering intent — so users who need colour-accurate workflows
/// can act on it.
#[derive(Debug, Clone)]
pub enum PngColorSpace {
    /// No colour-space chunks present.  Untagged PNG is sRGB by convention.
    Unknown,

    /// An explicit `sRGB` chunk was present.
    Srgb {
        /// The rendering intent (0–3 per PNG spec).
        rendering_intent: u8,
    },

    /// A `gAMA` chunk (and optionally `cHRM`) was present, but no higher-
    /// priority chunk overrides it.
    Gamma {
        /// The file gamma as a floating-point value (e.g. 0.45455 for sRGB).
        gamma: f64,
        /// Optional chromaticity values from `cHRM`, in the order:
        /// `[white_x, white_y, red_x, red_y, green_x, green_y, blue_x, blue_y]`.
        chromaticities: Option<[f64; 8]>,
    },

    /// An `iCCP` chunk was present — an embedded ICC profile.
    IccProfile {
        /// The raw ICC profile bytes (decompressed).  Boxed because there is
        /// no need for a growable `Vec` — the profile is a fixed blob from
        /// the file.
        profile: Box<[u8]>,
    },

    /// A `cICP` chunk was present (PNG Third Edition).
    Cicp {
        /// Colour primaries (e.g. 1 = BT.709/sRGB, 12 = Display P3).
        colour_primaries: u8,
        /// Transfer characteristics (e.g. 13 = sRGB, 16 = PQ, 18 = HLG).
        transfer_characteristics: u8,
        /// Matrix coefficients (should be 0 for RGB).
        matrix_coefficients: u8,
        /// Video full range flag.
        video_full_range: bool,
    },
}

// ═══════════════════════════════════════════════════════════════════════════════
// PngTextChunk — tEXt / zTXt / iTXt normalised into one type
// ═══════════════════════════════════════════════════════════════════════════════

/// A text chunk extracted from a PNG file.
///
/// PNG defines three text chunk types — `tEXt` (Latin-1), `zTXt`
/// (compressed Latin-1), and `iTXt` (UTF-8 with optional language tag).
/// All three are normalised into this struct with the text content
/// decoded/decompressed into a UTF-8 `String`.
///
/// Standard keywords defined by the PNG spec include: `Title`, `Author`,
/// `Description`, `Copyright`, `Creation Time`, `Software`, `Disclaimer`,
/// `Warning`, `Source`, `Comment`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PngTextChunk {
    /// The keyword identifying this text entry (e.g. `"Title"`, `"Author"`).
    /// Always 1–79 bytes of Latin-1 printable characters per the PNG spec.
    pub keyword: String,

    /// The text content (decompressed if from `zTXt`/`iTXt`).
    pub text: String,

    /// Language tag from `iTXt` (BCP 47), if present.
    /// Always `None` for `tEXt` and `zTXt` chunks.
    pub language: Option<String>,

    /// Translated keyword from `iTXt`, if present.
    /// Always `None` for `tEXt` and `zTXt` chunks.
    pub translated_keyword: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PngTimestamp — the tIME chunk
// ═══════════════════════════════════════════════════════════════════════════════

/// Last-modification timestamp from the PNG `tIME` chunk.
///
/// Represents UTC time.  The PNG spec allows `second = 60` for leap seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PngTimestamp {
    /// Four-digit year (e.g. 2025).
    pub year: u16,
    /// Month of the year (1–12).
    pub month: u8,
    /// Day of the month (1–31).
    pub day: u8,
    /// Hour (0–23).
    pub hour: u8,
    /// Minute (0–59).
    pub minute: u8,
    /// Second (0–60, allowing for leap seconds per the PNG spec).
    pub second: u8,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PngMetadata — ancillary information beyond pixel data
// ═══════════════════════════════════════════════════════════════════════════════

/// Ancillary metadata extracted from a PNG file.
///
/// Returned as part of [`PngDecoded`].  Fields are optional — only populated
/// when the corresponding chunks exist in the file.
#[derive(Debug, Clone)]
pub struct PngMetadata {
    /// Colour-space information derived from colour-related chunks.
    pub color_space: PngColorSpace,

    /// The original PNG bit depth before any expansion (1, 2, 4, 8, or 16).
    pub source_bit_depth: u8,

    /// The significant bits from the `sBIT` chunk, if present.
    /// Length matches the number of channels in the source colour type
    /// (1 for grayscale, 2 for grayscale+alpha, 3 for RGB, 4 for RGBA).
    /// Boxed because the data is a fixed blob — no need for a growable `Vec`.
    pub significant_bits: Option<Box<[u8]>>,

    /// Physical pixel dimensions from the `pHYs` chunk, if present.
    /// `(pixels_per_unit_x, pixels_per_unit_y, unit)` where unit is
    /// 0 = unknown, 1 = metre.
    pub pixel_dimensions: Option<(u32, u32, u8)>,

    /// Text entries from `tEXt`, `zTXt`, and `iTXt` chunks.
    ///
    /// Chunks are grouped by type (`tEXt`, then `zTXt`, then `iTXt`),
    /// not by the order they appear in the file.
    pub text_chunks: Vec<PngTextChunk>,

    /// Last-modification timestamp from the `tIME` chunk, if present.
    pub last_modified: Option<PngTimestamp>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PngDecoded — the complete decode result
// ═══════════════════════════════════════════════════════════════════════════════

/// The complete result of decoding a PNG file.
///
/// Contains both the pixel data ([`PngImage`]) and ancillary metadata
/// ([`PngMetadata`]).  This is always returned as a whole — metadata
/// construction is essentially free because the `png` crate parses all
/// ancillary chunks before image data regardless.
///
/// This struct is `#[non_exhaustive]` so that new fields (e.g. decode
/// warnings) can be added in the future without a semver-major bump.
/// Users access fields directly:
///
/// ```no_run
/// # use irys_cv_io::png::{self, PngImage};
/// let decoded = png::decode(&std::fs::read("photo.png").unwrap()).unwrap();
///
/// // Just the pixels — discard metadata implicitly:
/// let image = decoded.image;
///
/// // Or inspect both:
/// // let color_space = &decoded.metadata.color_space;
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct PngDecoded {
    /// The decoded pixel data.
    pub image: PngImage,
    /// Ancillary metadata (colour space, text, timestamps, …).
    pub metadata: PngMetadata,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TransferAssumption — private detection logic for Q6+Q7
// ═══════════════════════════════════════════════════════════════════════════════

/// Transfer function assumption derived from PNG colour-space metadata.
///
/// A single boolean `is_linear` is insufficient because the default for
/// untagged PNGs differs by bit depth: 8-bit untagged is sRGB by
/// convention, 16-bit untagged is linear by convention.  This three-variant
/// enum separates "explicitly tagged" from "unknown" so each decode arm
/// can resolve `Unknown` based on its bit depth.
///
/// This enum is private to the decoder — it is not part of the public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransferAssumption {
    /// Metadata says linear (cICP TF 8, gAMA ≈ 1.0).
    Linear,
    /// Metadata says non-linear (sRGB chunk, iCCP, gAMA ≈ 0.45455,
    /// sRGB/BT.709 cICP TFs).
    Srgb,
    /// No colour-space chunk — bit-depth-dependent default
    /// (8-bit → sRGB, 16-bit → linear).
    Unknown,
}

impl TransferAssumption {
    /// Compute the transfer assumption from a [`PngColorSpace`].
    ///
    /// Follows PNG Third Edition §4.3 chunk priority:
    ///   `cICP` > `iCCP` > `sRGB` > `gAMA`+`cHRM` > `Unknown`
    ///
    /// The priority is already encoded in `PngColorSpace` by
    /// [`extract_color_space`], so we only need to interpret the variant.
    fn from_color_space(cs: &PngColorSpace) -> Self {
        match cs {
            // cICP with linear transfer (code point 8 per ITU-T H.273).
            PngColorSpace::Cicp {
                transfer_characteristics: 8,
                ..
            } => TransferAssumption::Linear,

            // cICP with sRGB or BT.709 transfer (code points 1 and 13).
            PngColorSpace::Cicp {
                transfer_characteristics: 1 | 13,
                ..
            } => TransferAssumption::Srgb,

            // PQ (16) and HLG (18) are rejected by Q2 before this
            // function is called.  Any remaining cICP TF is treated as
            // non-linear (sRGB-like) — a safe default that avoids the
            // type-level lie of claiming the data is linear.
            PngColorSpace::Cicp { .. } => TransferAssumption::Srgb,

            // Explicit sRGB chunk.
            PngColorSpace::Srgb { .. } => TransferAssumption::Srgb,

            // gAMA ≈ 1.0 means linear.  The stored value is the file
            // gamma as f64; 1.0 is exactly linear.  Use a tolerance
            // band to catch minor rounding from other encoders.
            PngColorSpace::Gamma { gamma, .. } if (*gamma - 1.0).abs() < 0.001 => {
                TransferAssumption::Linear
            }

            // Any other gamma value is non-linear.
            PngColorSpace::Gamma { .. } => TransferAssumption::Srgb,

            // iCCP — almost always gamma-encoded; parsing ICC internals
            // to detect a linear profile is out of scope.
            PngColorSpace::IccProfile { .. } => TransferAssumption::Srgb,

            // No colour-space chunk.  Convention differs by bit depth:
            //   8-bit → sRGB,  16-bit → linear.
            PngColorSpace::Unknown => TransferAssumption::Unknown,
        }
    }

    /// Resolve `Unknown` based on bit depth: 8-bit → sRGB, 16-bit → linear.
    fn is_linear(self, bit_depth: png::BitDepth) -> bool {
        match self {
            TransferAssumption::Linear => true,
            TransferAssumption::Srgb => false,
            TransferAssumption::Unknown => bit_depth == png::BitDepth::Sixteen,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Decode — public API
// ═══════════════════════════════════════════════════════════════════════════════

/// Decode a PNG image from an in-memory byte slice.
///
/// Returns a [`PngDecoded`] whose [`image`](PngDecoded::image) variant is
/// determined by the IHDR colour type and bit depth, and whose
/// [`metadata`](PngDecoded::metadata) carries colour-space information,
/// text chunks, timestamps, and other ancillary data.
///
/// # Errors
///
/// Returns [`IoError`] if:
/// - the input is not a valid PNG file ([`IoError::InvalidFormat`])
/// - the PNG uses an unsupported feature ([`IoError::UnsupportedFeature`])
/// - the decoder encounters corruption ([`IoError::DecodeFailed`])
///
/// [`IoError::InvalidFormat`]: crate::IoError::InvalidFormat
/// [`IoError::UnsupportedFeature`]: crate::IoError::UnsupportedFeature
/// [`IoError::DecodeFailed`]: crate::IoError::DecodeFailed
///
/// # Examples
///
/// ```no_run
/// # use irys_cv_io::png::{self, PngImage};
/// let bytes = std::fs::read("photo.png").unwrap();
/// let decoded = png::decode(&bytes).unwrap();
///
/// match decoded.image {
///     PngImage::Srgb8(image) => { /* work with Image<Srgb8> */ }
///     PngImage::Srgba8(image) => { /* work with Image<Srgba8> */ }
///     PngImage::Indexed8 { data, palette } => { /* depalettise */ }
///     _ => { /* handle remaining variants */ }
/// }
/// ```
pub fn decode(data: &[u8]) -> Result<PngDecoded, IoError> {
    let time = parse_time_chunk(data);
    let mut decoded = decode_reader(std::io::Cursor::new(data))?;
    decoded.metadata.last_modified = time;
    Ok(decoded)
}

/// Decode a PNG image from a streaming reader.
///
/// Behaves identically to [`decode`] but reads from any `impl Read`
/// instead of requiring the entire file in memory.  Useful when the
/// PNG source is a network stream, a decompressor, or any other
/// non-seekable reader.
///
/// # Errors
///
/// Same error conditions as [`decode`].
///
/// # Examples
///
/// ```no_run
/// # use irys_cv_io::png::{self, PngImage};
/// let file = std::fs::File::open("photo.png").unwrap();
/// let reader = std::io::BufReader::new(file);
/// let decoded = png::decode_reader(reader).unwrap();
///
/// match decoded.image {
///     PngImage::Srgb8(image) => { /* work with Image<Srgb8> */ }
///     _ => { /* handle remaining variants */ }
/// }
/// ```
pub fn decode_reader(reader: impl std::io::Read) -> Result<PngDecoded, IoError> {
    use irys_cv::pixel::PlainPixel;
    use png::{BitDepth, ColorType, Transformations};

    // ── Configure decoder ────────────────────────────────────────────────
    //
    // Use IDENTITY (no transforms) so we have full control over sample
    // expansion and byte-order conversion:
    //   - Sub-byte grayscale: manual unpack + full-range scale to 0..=255
    //   - Sub-byte indexed: manual unpack to one index byte per pixel
    //     (no value scaling, no depalettising)
    //   - 16-bit: big-endian bytes, converted via `from_bytes_be`
    //
    // We intentionally avoid EXPAND (which depalettises indexed images
    // and adds alpha from tRNS) and STRIP_16 (which discards 16-bit
    // precision).
    let mut decoder = png::Decoder::new(reader);
    decoder.set_transformations(Transformations::IDENTITY);

    // ── Read header + ancillary chunks ───────────────────────────────────
    let mut png_reader = decoder.read_info().map_err(decode_error)?;
    let info = png_reader.info();

    // Capture IHDR values before reading pixel data (info borrows reader).
    let width = info.width as usize;
    let height = info.height as usize;
    let color_type = info.color_type;
    let bit_depth = info.bit_depth;
    let source_bit_depth = bit_depth as u8;

    // ── Extract metadata (cheap — just copying scalars / cloning Cows) ──
    let metadata = extract_metadata(info, source_bit_depth);

    // ── Reject unsupported transfer functions (Q2) ───────────────────────
    //
    // PQ (16) and HLG (18) transfer functions require tone-mapping that
    // we do not implement.  Reject them early — before the pixel-type
    // decision logic — so callers get a clear error instead of silently
    // wrong pixel types.
    if let PngColorSpace::Cicp {
        transfer_characteristics: 16 | 18,
        ..
    } = &metadata.color_space
    {
        return Err(IoError::UnsupportedFeature {
            reason: "PQ (16) and HLG (18) transfer functions are not supported",
        });
    }

    // ── Determine transfer function assumption (Q6+Q7) ───────────────────
    let transfer = TransferAssumption::from_color_space(&metadata.color_space);

    // ── Read frame data ──────────────────────────────────────────────────
    let mut buf = vec![0u8; png_reader.output_buffer_size()];
    let frame_info = png_reader.next_frame(&mut buf).map_err(decode_error)?;
    let line_size = frame_info.line_size;
    buf.truncate(frame_info.buffer_size());

    // ── Convert raw bytes → PngImage ─────────────────────────────────────
    //
    // The `is_linear` flag is resolved once per image from the transfer
    // assumption and the effective bit depth.  Sub-byte grayscale is
    // expanded to 8-bit, so it uses `BitDepth::Eight` for the resolution.
    let image = match (color_type, bit_depth) {
        // ── Grayscale (sub-byte: 1/2/4-bit) ──────────────────────────────
        (ColorType::Grayscale, BitDepth::One | BitDepth::Two | BitDepth::Four) => {
            // Sub-byte samples are packed MSB-first in each byte.
            // Unpack to one value per pixel, then scale to full 0..=255.
            // The expanded result is 8-bit, so resolve transfer as 8-bit.
            let is_linear = transfer.is_linear(BitDepth::Eight);
            let bits_per_sample = source_bit_depth as usize;
            let scale = match bit_depth {
                BitDepth::One => 255u8, // 0→0, 1→255
                BitDepth::Two => 85u8,  // 0→0, 1→85, 2→170, 3→255
                BitDepth::Four => 17u8, // 0→0, …, 15→255
                _ => unreachable!(),
            };
            let unpacked = unpack_sub_byte(&buf, width, height, line_size, bits_per_sample);
            if is_linear {
                let pixels: Vec<Mono8> = unpacked
                    .into_iter()
                    .map(|v| Mono8::new(v.saturating_mul(scale)))
                    .collect();
                let img =
                    Image::from_vec(width, height, pixels).map_err(|_| IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    })?;
                PngImage::Mono8(img)
            } else {
                let pixels: Vec<SrgbMono8> = unpacked
                    .into_iter()
                    .map(|v| SrgbMono8::new(v.saturating_mul(scale)))
                    .collect();
                let img =
                    Image::from_vec(width, height, pixels).map_err(|_| IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    })?;
                PngImage::SrgbMono8(img)
            }
        }

        // ── Grayscale 8-bit ──────────────────────────────────────────────
        (ColorType::Grayscale, BitDepth::Eight) => {
            // One byte per pixel — zero-copy cast via PlainPixel layout
            // guarantee.  Mono8 and SrgbMono8 have identical layout.
            if transfer.is_linear(bit_depth) {
                let img = Image::from_raw_bytes(width, height, buf).map_err(|_| {
                    IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    }
                })?;
                PngImage::Mono8(img)
            } else {
                let img = Image::from_raw_bytes(width, height, buf).map_err(|_| {
                    IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    }
                })?;
                PngImage::SrgbMono8(img)
            }
        }

        // ── Grayscale 16-bit ─────────────────────────────────────────────
        (ColorType::Grayscale, BitDepth::Sixteen) => {
            // Big-endian 16-bit samples.  Use from_bytes_be for correct
            // endianness handling on all hosts.
            if transfer.is_linear(bit_depth) {
                let pixels: Vec<Mono16> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        Mono16::from_bytes_be(c).expect("chunk size guaranteed by chunks_exact")
                    })
                    .collect();
                let img =
                    Image::from_vec(width, height, pixels).map_err(|_| IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    })?;
                PngImage::Mono16(img)
            } else {
                let pixels: Vec<SrgbMono16> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        SrgbMono16::from_bytes_be(c).expect("chunk size guaranteed by chunks_exact")
                    })
                    .collect();
                let img =
                    Image::from_vec(width, height, pixels).map_err(|_| IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    })?;
                PngImage::SrgbMono16(img)
            }
        }

        // ── Grayscale + Alpha 8-bit ──────────────────────────────────────
        (ColorType::GrayscaleAlpha, BitDepth::Eight) => {
            // 2 bytes per pixel: (v, a) — zero-copy cast via PlainPixel.
            // MonoA8 and SrgbMonoA8 have identical layout.
            if transfer.is_linear(bit_depth) {
                let img = Image::from_raw_bytes(width, height, buf).map_err(|_| {
                    IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    }
                })?;
                PngImage::MonoA8(img)
            } else {
                let img = Image::from_raw_bytes(width, height, buf).map_err(|_| {
                    IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    }
                })?;
                PngImage::SrgbMonoA8(img)
            }
        }

        // ── Grayscale + Alpha 16-bit ─────────────────────────────────────
        (ColorType::GrayscaleAlpha, BitDepth::Sixteen) => {
            // 4 bytes per pixel (big-endian): v_hi v_lo a_hi a_lo.
            if transfer.is_linear(bit_depth) {
                let pixels: Vec<MonoA16> = buf
                    .chunks_exact(4)
                    .map(|c| {
                        MonoA16::from_bytes_be(c).expect("chunk size guaranteed by chunks_exact")
                    })
                    .collect();
                let img =
                    Image::from_vec(width, height, pixels).map_err(|_| IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    })?;
                PngImage::MonoA16(img)
            } else {
                let pixels: Vec<SrgbMonoA16> = buf
                    .chunks_exact(4)
                    .map(|c| {
                        SrgbMonoA16::from_bytes_be(c)
                            .expect("chunk size guaranteed by chunks_exact")
                    })
                    .collect();
                let img =
                    Image::from_vec(width, height, pixels).map_err(|_| IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    })?;
                PngImage::SrgbMonoA16(img)
            }
        }

        // ── Truecolour RGB 8-bit ─────────────────────────────────────────
        (ColorType::Rgb, BitDepth::Eight) => {
            // 3 bytes per pixel: (r, g, b).
            // Zero-copy cast via PlainPixel layout guarantee.
            // Rgb8 and Srgb8 have identical layout.
            if transfer.is_linear(bit_depth) {
                let img = Image::from_raw_bytes(width, height, buf).map_err(|_| {
                    IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    }
                })?;
                PngImage::Rgb8(img)
            } else {
                let img = Image::from_raw_bytes(width, height, buf).map_err(|_| {
                    IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    }
                })?;
                PngImage::Srgb8(img)
            }
        }

        // ── Truecolour RGB 16-bit ────────────────────────────────────────
        (ColorType::Rgb, BitDepth::Sixteen) => {
            // 6 bytes per pixel (big-endian).
            if transfer.is_linear(bit_depth) {
                let pixels: Vec<Rgb16> = buf
                    .chunks_exact(6)
                    .map(|c| {
                        Rgb16::from_bytes_be(c).expect("chunk size guaranteed by chunks_exact")
                    })
                    .collect();
                let img =
                    Image::from_vec(width, height, pixels).map_err(|_| IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    })?;
                PngImage::Rgb16(img)
            } else {
                let pixels: Vec<Srgb16> = buf
                    .chunks_exact(6)
                    .map(|c| {
                        Srgb16::from_bytes_be(c).expect("chunk size guaranteed by chunks_exact")
                    })
                    .collect();
                let img =
                    Image::from_vec(width, height, pixels).map_err(|_| IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    })?;
                PngImage::Srgb16(img)
            }
        }

        // ── Truecolour + Alpha RGBA 8-bit ────────────────────────────────
        (ColorType::Rgba, BitDepth::Eight) => {
            // 4 bytes per pixel: (r, g, b, a) — zero-copy cast via PlainPixel.
            // Rgba8 and Srgba8 have identical layout.
            if transfer.is_linear(bit_depth) {
                let img = Image::from_raw_bytes(width, height, buf).map_err(|_| {
                    IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    }
                })?;
                PngImage::Rgba8(img)
            } else {
                let img = Image::from_raw_bytes(width, height, buf).map_err(|_| {
                    IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    }
                })?;
                PngImage::Srgba8(img)
            }
        }

        // ── Truecolour + Alpha RGBA 16-bit ───────────────────────────────
        (ColorType::Rgba, BitDepth::Sixteen) => {
            // 8 bytes per pixel (big-endian).
            if transfer.is_linear(bit_depth) {
                let pixels: Vec<Rgba16> = buf
                    .chunks_exact(8)
                    .map(|c| {
                        Rgba16::from_bytes_be(c).expect("chunk size guaranteed by chunks_exact")
                    })
                    .collect();
                let img =
                    Image::from_vec(width, height, pixels).map_err(|_| IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    })?;
                PngImage::Rgba16(img)
            } else {
                let pixels: Vec<Srgba16> = buf
                    .chunks_exact(8)
                    .map(|c| {
                        Srgba16::from_bytes_be(c).expect("chunk size guaranteed by chunks_exact")
                    })
                    .collect();
                let img =
                    Image::from_vec(width, height, pixels).map_err(|_| IoError::DecodeFailed {
                        source: "pixel count does not match image dimensions".into(),
                    })?;
                PngImage::Srgba16(img)
            }
        }

        // ── Indexed (palette) ────────────────────────────────────────────
        (ColorType::Indexed, BitDepth::One | BitDepth::Two | BitDepth::Four) => {
            // Sub-byte indices packed MSB-first.  Unpack to one index byte
            // per pixel WITHOUT value-scaling (indices are not intensities).
            let bits_per_sample = source_bit_depth as usize;
            let unpacked = unpack_sub_byte(&buf, width, height, line_size, bits_per_sample);

            let palette_info = png_reader.info();
            let palette = build_palette(palette_info)?;

            let pixels: Vec<Indexed8> = unpacked.into_iter().map(Indexed8).collect();
            let img =
                Image::from_vec(width, height, pixels).map_err(|_| IoError::DecodeFailed {
                    source: "pixel count does not match image dimensions".into(),
                })?;
            PngImage::Indexed8 { data: img, palette }
        }

        (ColorType::Indexed, BitDepth::Eight) => {
            // One index byte per pixel — zero-copy cast via PlainPixel.
            let palette_info = png_reader.info();
            let palette = build_palette(palette_info)?;

            let img =
                Image::from_raw_bytes(width, height, buf).map_err(|_| IoError::DecodeFailed {
                    source: "pixel count does not match image dimensions".into(),
                })?;
            PngImage::Indexed8 { data: img, palette }
        }

        // ── Unsupported combinations ─────────────────────────────────────
        _ => {
            return Err(IoError::UnsupportedFeature {
                reason: "unsupported PNG colour type / bit depth combination",
            });
        }
    };

    Ok(PngDecoded { image, metadata })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Map a `png::DecodingError` into `IoError`.
///
/// We inspect the error kind to produce a more specific `IoError` variant
/// where possible — `Io` for underlying I/O errors, and `DecodeFailed`
/// for everything else (format violations, parameter errors, limits).
fn decode_error(e: png::DecodingError) -> IoError {
    // The png crate's DecodingError has IoError, Format, Parameter,
    // and LimitsExceeded variants.  We only special-case IoError to
    // preserve the std::io::Error; all others become DecodeFailed
    // with the original error boxed as the source.
    if let png::DecodingError::IoError(io) = e {
        IoError::Io(io)
    } else {
        IoError::DecodeFailed {
            source: Box::new(e),
        }
    }
}

/// Unpack sub-byte samples (1/2/4-bit) from packed scanline data.
///
/// PNG stores sub-byte samples MSB-first within each byte. Each scanline
/// is `line_size` bytes; pixels beyond `width` in the last byte of each
/// row are padding and must be discarded.
///
/// Returns exactly `width * height` unpacked sample values.
fn unpack_sub_byte(
    buf: &[u8],
    width: usize,
    height: usize,
    line_size: usize,
    bits_per_sample: usize,
) -> Vec<u8> {
    debug_assert!(
        bits_per_sample == 1 || bits_per_sample == 2 || bits_per_sample == 4,
        "unpack_sub_byte only handles 1, 2, or 4 bits per sample"
    );

    let samples_per_byte = 8 / bits_per_sample;
    let mask: u8 = (1u8 << bits_per_sample) - 1; // 1→1, 2→3, 4→15

    let mut pixels = Vec::with_capacity(width * height);

    for y in 0..height {
        let row_start = y * line_size;
        let row = &buf[row_start..row_start + line_size];

        let mut col = 0;
        for &byte in row {
            // Extract samples from MSB to LSB.
            for i in 0..samples_per_byte {
                if col >= width {
                    break;
                }
                let shift = 8 - bits_per_sample * (i + 1);
                let sample = (byte >> shift) & mask;
                pixels.push(sample);
                col += 1;
            }
        }
    }

    pixels
}

/// Extract colour-space information from PNG `Info`, respecting the
/// priority order defined by PNG Third Edition §4.3 Table 1:
///
///   cICP  >  iCCP  >  sRGB  >  gAMA+cHRM  >  Unknown
fn extract_color_space(info: &png::Info<'_>) -> PngColorSpace {
    // Priority 1: cICP (Coding Independent Code Points)
    if let Some(cicp) = &info.coding_independent_code_points {
        return PngColorSpace::Cicp {
            colour_primaries: cicp.color_primaries,
            transfer_characteristics: cicp.transfer_function,
            matrix_coefficients: cicp.matrix_coefficients,
            video_full_range: cicp.is_video_full_range_image,
        };
    }

    // Priority 2: iCCP (embedded ICC profile)
    if let Some(icc) = &info.icc_profile {
        return PngColorSpace::IccProfile {
            profile: icc.to_vec().into_boxed_slice(),
        };
    }

    // Priority 3: sRGB chunk
    if let Some(intent) = &info.srgb {
        return PngColorSpace::Srgb {
            rendering_intent: *intent as u8,
        };
    }

    // Priority 4: gAMA (optionally with cHRM)
    // Use `source_gamma` which merges gAMA and sRGB-derived gamma.
    // But since we already checked sRGB above, at this point
    // source_gamma can only come from an actual gAMA chunk.
    if let Some(gamma) = &info.gama_chunk {
        let chromaticities = info.chrm_chunk.map(|c| {
            [
                c.white.0.into_value() as f64,
                c.white.1.into_value() as f64,
                c.red.0.into_value() as f64,
                c.red.1.into_value() as f64,
                c.green.0.into_value() as f64,
                c.green.1.into_value() as f64,
                c.blue.0.into_value() as f64,
                c.blue.1.into_value() as f64,
            ]
        });
        return PngColorSpace::Gamma {
            gamma: gamma.into_value() as f64,
            chromaticities,
        };
    }

    PngColorSpace::Unknown
}

/// Extract text chunks from all three PNG text chunk types, normalising
/// them into [`PngTextChunk`].
///
/// Compressed text (zTXt, iTXt) is decompressed via the `png` crate's
/// `get_text()` method.  If decompression fails for a chunk, that chunk
/// is silently skipped (text is ancillary, non-critical data).
fn extract_text_chunks(info: &png::Info<'_>) -> Vec<PngTextChunk> {
    let mut chunks = Vec::new();

    // tEXt — uncompressed Latin-1
    for t in &info.uncompressed_latin1_text {
        chunks.push(PngTextChunk {
            keyword: t.keyword.clone(),
            text: t.text.clone(),
            language: None,
            translated_keyword: None,
        });
    }

    // zTXt — compressed Latin-1
    for z in &info.compressed_latin1_text {
        // get_text() decompresses if needed.
        if let Ok(text) = z.get_text() {
            chunks.push(PngTextChunk {
                keyword: z.keyword.clone(),
                text,
                language: None,
                translated_keyword: None,
            });
        }
    }

    // iTXt — UTF-8, optionally compressed, with language tag
    for i in &info.utf8_text {
        if let Ok(text) = i.get_text() {
            chunks.push(PngTextChunk {
                keyword: i.keyword.clone(),
                text,
                language: if i.language_tag.is_empty() {
                    None
                } else {
                    Some(i.language_tag.clone())
                },
                translated_keyword: if i.translated_keyword.is_empty() {
                    None
                } else {
                    Some(i.translated_keyword.clone())
                },
            });
        }
    }

    chunks
}

/// Build full metadata struct from PNG `Info`.
fn extract_metadata(info: &png::Info<'_>, source_bit_depth: u8) -> PngMetadata {
    let color_space = extract_color_space(info);

    let significant_bits = info
        .sbit
        .as_ref()
        .map(|sbit| sbit.to_vec().into_boxed_slice());

    let pixel_dimensions = info.pixel_dims.map(|pd| {
        let unit = match pd.unit {
            png::Unit::Meter => 1u8,
            png::Unit::Unspecified => 0u8,
        };
        (pd.xppu, pd.yppu, unit)
    });

    let text_chunks = extract_text_chunks(info);

    // The `png` crate 0.17 does not parse the tIME chunk into a
    // structured field.  For the streaming `decode_reader` path we
    // cannot access raw bytes, so `last_modified` stays `None` here.
    // The `decode(&[u8])` entry point patches it in afterwards via
    // `parse_time_chunk`.
    let last_modified = None;

    PngMetadata {
        color_space,
        source_bit_depth,
        significant_bits,
        pixel_dimensions,
        text_chunks,
        last_modified,
    }
}

/// Build a 256-entry `Srgba8` palette from `PLTE` (RGB triples) and
/// `tRNS` (per-entry alpha), merging them into a single array.
///
/// Entries beyond the actual palette size get `r = g = b = 0, a = 255`.
/// tRNS entries beyond PLTE length are ignored.
fn build_palette(info: &png::Info<'_>) -> Result<Box<[Srgba8; 256]>, IoError> {
    let plte = info.palette.as_ref().ok_or(IoError::InvalidFormat {
        reason: "indexed PNG missing PLTE chunk",
    })?;

    // PLTE contains RGB triples (3 bytes each).
    let entry_count = plte.len() / 3;

    // tRNS for indexed images: one alpha byte per palette entry.
    let trns = info.trns.as_deref().unwrap_or(&[]);

    let mut palette = Box::new([Srgba8::new(0, 0, 0, 255); 256]);
    for i in 0..entry_count.min(256) {
        let r = plte[i * 3];
        let g = plte[i * 3 + 1];
        let b = plte[i * 3 + 2];
        let a = if i < trns.len() { trns[i] } else { 255 };
        palette[i] = Srgba8::new(r, g, b, a);
    }

    Ok(palette)
}

/// Parse the `tIME` chunk from raw PNG bytes.
///
/// Walks the PNG chunk structure looking for a `tIME` chunk (which is
/// always exactly 7 bytes of data) and parses it into a [`PngTimestamp`].
///
/// Returns `None` if the input is too short, has no `tIME` chunk, or if
/// the chunk has an unexpected length.
fn parse_time_chunk(data: &[u8]) -> Option<PngTimestamp> {
    // PNG signature is 8 bytes; minimum chunk overhead is 12 bytes
    // (4 length + 4 type + 0 data + 4 CRC).
    if data.len() < 8 {
        return None;
    }
    let mut pos = 8; // skip PNG signature

    while pos + 12 <= data.len() {
        let length =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let chunk_type = &data[pos + 4..pos + 8];

        if chunk_type == b"tIME" {
            // tIME chunk is exactly 7 bytes: year(2) month(1) day(1)
            // hour(1) minute(1) second(1).
            if length == 7 && pos + 8 + 7 <= data.len() {
                let d = &data[pos + 8..];
                return Some(PngTimestamp {
                    year: u16::from_be_bytes([d[0], d[1]]),
                    month: d[2],
                    day: d[3],
                    hour: d[4],
                    minute: d[5],
                    second: d[6],
                });
            }
            return None; // tIME chunk with wrong length
        }

        // Advance to next chunk: 4 (length) + 4 (type) + data + 4 (CRC).
        pos = pos.checked_add(12 + length)?;
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════════
// TransferTag — what colour-space chunks the encoder writes
// ═══════════════════════════════════════════════════════════════════════════════

/// What colour-space chunks the encoder must write for a given pixel type.
///
/// This enum is an implementation detail of the [`PngPixel`] trait.  It is
/// `pub` only because Rust requires associated-constant types on public
/// traits to be public.  **Do not match on or construct this type** — it is
/// not part of the stable API.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferTag {
    /// Write an `sRGB` chunk (rendering intent = Perceptual).
    Srgb,
    /// Write a `gAMA` chunk with gamma = 1.0 (100 000).
    Linear,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PngPixel — sealed trait mapping pixel types to PNG wire parameters
// ═══════════════════════════════════════════════════════════════════════════════

mod png_pixel_sealed {
    /// Sealed supertrait — prevents out-of-crate implementations of
    /// [`PngPixel`](super::PngPixel).
    pub trait Sealed {}
}

/// Maps a pixel type to its PNG colour type, bit depth, and transfer tag.
///
/// This trait is **sealed**: it is implemented for exactly the 17 pixel
/// types that PNG can represent (the 16 non-indexed types plus
/// [`Indexed8`]).  Attempting to encode a type that does not implement
/// `PngPixel` (e.g. `RgbF32`, `Bgr8`) is a compile-time error.
///
/// # Associated constants
///
/// | Constant         | Meaning                                           |
/// |------------------|---------------------------------------------------|
/// | `PNG_COLOR_TYPE` | PNG colour type for the IHDR chunk.                |
/// | `PNG_BIT_DEPTH`  | PNG bit depth for the IHDR chunk (Eight or Sixteen). |
/// | `TRANSFER`       | What colour-space chunks the encoder must write.   |
///
/// # Implementors
///
/// | Pixel type    | Colour type      | Bit depth | Transfer |
/// |---------------|------------------|-----------|----------|
/// | `SrgbMono8`   | `Grayscale`      | `Eight`   | sRGB     |
/// | `SrgbMonoA8`  | `GrayscaleAlpha` | `Eight`   | sRGB     |
/// | `Srgb8`       | `Rgb`            | `Eight`   | sRGB     |
/// | `Srgba8`      | `Rgba`           | `Eight`   | sRGB     |
/// | `Mono8`       | `Grayscale`      | `Eight`   | Linear   |
/// | `MonoA8`      | `GrayscaleAlpha` | `Eight`   | Linear   |
/// | `Rgb8`        | `Rgb`            | `Eight`   | Linear   |
/// | `Rgba8`       | `Rgba`           | `Eight`   | Linear   |
/// | `SrgbMono16`  | `Grayscale`      | `Sixteen` | sRGB     |
/// | `SrgbMonoA16` | `GrayscaleAlpha` | `Sixteen` | sRGB     |
/// | `Srgb16`      | `Rgb`            | `Sixteen` | sRGB     |
/// | `Srgba16`     | `Rgba`           | `Sixteen` | sRGB     |
/// | `Mono16`      | `Grayscale`      | `Sixteen` | Linear   |
/// | `MonoA16`     | `GrayscaleAlpha` | `Sixteen` | Linear   |
/// | `Rgb16`       | `Rgb`            | `Sixteen` | Linear   |
/// | `Rgba16`      | `Rgba`           | `Sixteen` | Linear   |
/// | `Indexed8`    | `Indexed`        | `Eight`   | sRGB     |
pub trait PngPixel: png_pixel_sealed::Sealed + PlainPixel {
    /// PNG colour type for the IHDR chunk.
    const PNG_COLOR_TYPE: png::ColorType;

    /// PNG bit depth for the IHDR chunk (`Eight` or `Sixteen`).
    const PNG_BIT_DEPTH: png::BitDepth;

    /// What colour-space chunks the encoder must write.
    #[doc(hidden)]
    const TRANSFER: TransferTag;
}

// ── Macro to reduce impl boilerplate ─────────────────────────────────────────

macro_rules! impl_png_pixel {
    ($ty:ty, $color:expr, $depth:expr, $transfer:expr) => {
        impl png_pixel_sealed::Sealed for $ty {}
        impl PngPixel for $ty {
            const PNG_COLOR_TYPE: png::ColorType = $color;
            const PNG_BIT_DEPTH: png::BitDepth = $depth;
            const TRANSFER: TransferTag = $transfer;
        }
    };
}

// ── 8-bit sRGB ───────────────────────────────────────────────────────────────

impl_png_pixel!(
    SrgbMono8,
    png::ColorType::Grayscale,
    png::BitDepth::Eight,
    TransferTag::Srgb
);
impl_png_pixel!(
    SrgbMonoA8,
    png::ColorType::GrayscaleAlpha,
    png::BitDepth::Eight,
    TransferTag::Srgb
);
impl_png_pixel!(
    Srgb8,
    png::ColorType::Rgb,
    png::BitDepth::Eight,
    TransferTag::Srgb
);
impl_png_pixel!(
    Srgba8,
    png::ColorType::Rgba,
    png::BitDepth::Eight,
    TransferTag::Srgb
);

// ── 8-bit linear ─────────────────────────────────────────────────────────────

impl_png_pixel!(
    Mono8,
    png::ColorType::Grayscale,
    png::BitDepth::Eight,
    TransferTag::Linear
);
impl_png_pixel!(
    MonoA8,
    png::ColorType::GrayscaleAlpha,
    png::BitDepth::Eight,
    TransferTag::Linear
);
impl_png_pixel!(
    Rgb8,
    png::ColorType::Rgb,
    png::BitDepth::Eight,
    TransferTag::Linear
);
impl_png_pixel!(
    Rgba8,
    png::ColorType::Rgba,
    png::BitDepth::Eight,
    TransferTag::Linear
);

// ── 16-bit sRGB ──────────────────────────────────────────────────────────────

impl_png_pixel!(
    SrgbMono16,
    png::ColorType::Grayscale,
    png::BitDepth::Sixteen,
    TransferTag::Srgb
);
impl_png_pixel!(
    SrgbMonoA16,
    png::ColorType::GrayscaleAlpha,
    png::BitDepth::Sixteen,
    TransferTag::Srgb
);
impl_png_pixel!(
    Srgb16,
    png::ColorType::Rgb,
    png::BitDepth::Sixteen,
    TransferTag::Srgb
);
impl_png_pixel!(
    Srgba16,
    png::ColorType::Rgba,
    png::BitDepth::Sixteen,
    TransferTag::Srgb
);

// ── 16-bit linear ────────────────────────────────────────────────────────────

impl_png_pixel!(
    Mono16,
    png::ColorType::Grayscale,
    png::BitDepth::Sixteen,
    TransferTag::Linear
);
impl_png_pixel!(
    MonoA16,
    png::ColorType::GrayscaleAlpha,
    png::BitDepth::Sixteen,
    TransferTag::Linear
);
impl_png_pixel!(
    Rgb16,
    png::ColorType::Rgb,
    png::BitDepth::Sixteen,
    TransferTag::Linear
);
impl_png_pixel!(
    Rgba16,
    png::ColorType::Rgba,
    png::BitDepth::Sixteen,
    TransferTag::Linear
);

// ── Indexed ──────────────────────────────────────────────────────────────────

impl_png_pixel!(
    Indexed8,
    png::ColorType::Indexed,
    png::BitDepth::Eight,
    TransferTag::Srgb
);

// ═══════════════════════════════════════════════════════════════════════════════
// PngFilterStrategy — PNG row filter selection
// ═══════════════════════════════════════════════════════════════════════════════

/// PNG row filter strategy for encoding.
///
/// Each scanline can be filtered before compression to improve deflate
/// ratios.  This enum mirrors the strategies supported by the `png` crate.
///
/// When passed as `None` in [`PngEncodeOptions`], the encoder uses the
/// `png` crate's default (adaptive filtering).
///
/// # Variants
///
/// | Variant    | Filter byte | Description                     |
/// |------------|-------------|---------------------------------|
/// | `None`     | 0           | No filtering — raw bytes.       |
/// | `Sub`      | 1           | Difference from the left pixel. |
/// | `Up`       | 2           | Difference from the pixel above.|
/// | `Avg`      | 3           | Average of left and above.      |
/// | `Paeth`    | 4           | Paeth predictor.                |
/// | `Adaptive` | —           | Choose the best filter per row. |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngFilterStrategy {
    /// No filtering — raw bytes (filter byte 0).
    None,
    /// Difference from the byte to the left (filter byte 1).
    Sub,
    /// Difference from the byte above (filter byte 2).
    Up,
    /// Average of the byte to the left and the byte above (filter byte 3).
    Avg,
    /// Paeth predictor (filter byte 4).
    Paeth,
    /// Choose the best filter independently for each row.
    Adaptive,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PngEncodeOptions — user-controlled encoding parameters
// ═══════════════════════════════════════════════════════════════════════════════

/// User-controlled encoding parameters for PNG output.
///
/// Colour-space chunks (`sRGB`, `gAMA`) are **not** exposed here — they
/// are derived exclusively from the pixel type's [`PngPixel::TRANSFER`]
/// constant.  This struct controls only the things the user legitimately
/// decides: metadata, compression, and filtering.
///
/// All fields default to `None` / empty, which means "use the `png`
/// crate's built-in defaults."  The most common usage is simply
/// `&PngEncodeOptions::default()`.
///
/// # Examples
///
/// ```
/// use irys_cv_io::png::PngEncodeOptions;
///
/// // All defaults — no text, no timestamp, default compression.
/// let opts = PngEncodeOptions::default();
/// assert!(opts.text_chunks.is_empty());
/// assert!(opts.compression_level.is_none());
/// ```
///
/// ```
/// use irys_cv_io::png::{PngEncodeOptions, PngFilterStrategy, PngTimestamp};
///
/// let mut opts = PngEncodeOptions::default();
/// opts.compression_level = Some(9);
/// opts.filter = Some(PngFilterStrategy::Paeth);
/// opts.last_modified = Some(PngTimestamp {
///     year: 2025, month: 6, day: 25,
///     hour: 12, minute: 0, second: 0,
/// });
/// assert_eq!(opts.compression_level, Some(9));
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct PngEncodeOptions {
    /// Text chunks to write (`tEXt` / `iTXt`).
    ///
    /// Each entry becomes a text chunk in the encoded PNG.  The encoder
    /// writes `iTXt` when [`PngTextChunk::language`] is `Some`, otherwise
    /// `tEXt`.
    pub text_chunks: Vec<PngTextChunk>,

    /// Physical pixel dimensions (`pHYs` chunk).
    ///
    /// `(pixels_per_unit_x, pixels_per_unit_y, unit)` where `unit` is
    /// `0` = unknown, `1` = metre.  `None` means no `pHYs` chunk is
    /// written.  Matches the representation used by [`PngMetadata`].
    pub pixel_dimensions: Option<(u32, u32, u8)>,

    /// Last-modification timestamp (`tIME` chunk).
    ///
    /// `None` means no `tIME` chunk is written.
    pub last_modified: Option<PngTimestamp>,

    /// Deflate compression level (`0` = none, `1` = fast, `9` = best).
    ///
    /// `None` uses the `png` crate's default (level 6).
    pub compression_level: Option<u8>,

    /// PNG filter strategy.
    ///
    /// `None` uses the `png` crate's default (adaptive filtering).
    pub filter: Option<PngFilterStrategy>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Encoding — public API
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode an image to an in-memory PNG byte vector.
///
/// Colour-space chunks are written automatically based on the pixel
/// type's transfer function.  Use [`PngEncodeOptions`] for text chunks,
/// timestamps, compression, etc.
///
/// # Errors
///
/// Returns [`IoError::EncodeFailed`] if the `png` crate reports an error.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv::image::Image;
/// # use irys_cv::pixel::Srgb8;
/// # use irys_cv_io::png::{self, PngEncodeOptions};
/// let image: Image<Srgb8> = todo!();
/// let bytes = png::encode(&image, &PngEncodeOptions::default()).unwrap();
/// std::fs::write("output.png", bytes).unwrap();
/// ```
pub fn encode<P: PngPixel>(
    image: &(impl ImageView<Pixel = P> + PlainImage),
    options: &PngEncodeOptions,
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
/// Colour-space chunks (`sRGB` or `gAMA`) are written automatically
/// based on `P::TRANSFER`.  Optional metadata (text chunks, `pHYs`,
/// `tIME`) and compression settings come from [`PngEncodeOptions`].
///
/// # Errors
///
/// Returns [`IoError::EncodeFailed`] if the `png` crate reports an error,
/// or [`IoError::Io`] if the underlying writer fails.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv::image::Image;
/// # use irys_cv::pixel::Srgb8;
/// # use irys_cv_io::png::{self, PngEncodeOptions};
/// let image: Image<Srgb8> = todo!();
/// let mut out = Vec::new();
/// png::encode_writer(&image, &mut out, &PngEncodeOptions::default()).unwrap();
/// ```
pub fn encode_writer<P: PngPixel>(
    image: &(impl ImageView<Pixel = P> + PlainImage),
    writer: impl std::io::Write,
    options: &PngEncodeOptions,
) -> Result<(), IoError> {
    let width = image.width() as u32;
    let height = image.height() as u32;

    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(P::PNG_COLOR_TYPE);
    encoder.set_depth(P::PNG_BIT_DEPTH);

    // ── Colour-space tagging (Q2) ────────────────────────────────────────
    match P::TRANSFER {
        TransferTag::Srgb => {
            encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        }
        TransferTag::Linear => {
            encoder.set_source_gamma(png::ScaledFloat::new(1.0));
        }
    }

    // ── Compression (Q4) ─────────────────────────────────────────────────
    apply_compression(&mut encoder, options);

    // ── Pixel dimensions (pHYs) ──────────────────────────────────────────
    apply_pixel_dimensions(&mut encoder, options);

    // ── Text chunks ──────────────────────────────────────────────────────
    apply_text_chunks(&mut encoder, options)?;

    let mut png_writer = encoder.write_header().map_err(encode_error)?;

    // ── tIME chunk (written after header, before IDAT) ───────────────────
    write_time_chunk(&mut png_writer, options)?;

    // ── Write pixel data ─────────────────────────────────────────────────
    match P::PNG_BIT_DEPTH {
        png::BitDepth::Eight => {
            // Zero-copy: as_bytes() returns &[u8] in native order,
            // which is correct for single-byte channels.
            let bytes: &[u8] = image.as_bytes();
            png_writer.write_image_data(bytes).map_err(encode_error)?;
        }
        png::BitDepth::Sixteen => {
            // Must convert to big-endian.  as_bytes_be() allocates on
            // little-endian hosts but is correct on all hosts.
            let bytes = image.as_bytes_be();
            png_writer.write_image_data(&bytes).map_err(encode_error)?;
        }
        _ => unreachable!("PngPixel only uses Eight or Sixteen"),
    }

    Ok(())
}

/// Encode a palette-indexed image to an in-memory PNG byte vector.
///
/// `palette` is a slice of 1–256 [`Srgba8`] entries.  The encoder writes
/// `PLTE` (RGB) and `tRNS` (alpha, if any entry has `alpha < 255`).
/// An `sRGB` chunk is always written because palette entries are sRGB.
///
/// # Errors
///
/// - [`IoError::EncodeFailed`] if `palette` is empty or has more than 256
///   entries, or if any pixel index `≥ palette.len()`.
/// - [`IoError::EncodeFailed`] if the `png` crate reports an error.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv::image::Image;
/// # use irys_cv::pixel::{Indexed8, Srgba8};
/// # use irys_cv_io::png::{self, PngEncodeOptions};
/// let image: Image<Indexed8> = todo!();
/// let palette = [Srgba8::new(255, 0, 0, 255), Srgba8::new(0, 255, 0, 255)];
/// let bytes = png::encode_indexed(&image, &palette, &PngEncodeOptions::default()).unwrap();
/// ```
pub fn encode_indexed(
    image: &(impl ImageView<Pixel = Indexed8> + PlainImage),
    palette: &[Srgba8],
    options: &PngEncodeOptions,
) -> Result<Vec<u8>, IoError> {
    let mut buf = Vec::new();
    encode_indexed_writer(image, palette, &mut buf, options)?;
    Ok(buf)
}

/// Core indexed encoding — writes to a streaming writer.
fn encode_indexed_writer(
    image: &(impl ImageView<Pixel = Indexed8> + PlainImage),
    palette: &[Srgba8],
    writer: impl std::io::Write,
    options: &PngEncodeOptions,
) -> Result<(), IoError> {
    // ── Validate palette size ────────────────────────────────────────────
    if palette.is_empty() || palette.len() > 256 {
        return Err(IoError::EncodeFailed {
            source: "palette must have 1\u{2013}256 entries".into(),
        });
    }

    // ── Validate all indices are in range ─────────────────────────────────
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

    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);

    // ── PLTE (RGB) ───────────────────────────────────────────────────────
    let plte: Vec<u8> = palette.iter().flat_map(|c| [c.r.0, c.g.0, c.b.0]).collect();
    encoder.set_palette(plte);

    // ── tRNS (alpha, if any entry has alpha < 255) ───────────────────────
    let has_transparency = palette.iter().any(|c| c.a.0 < 255);
    if has_transparency {
        // tRNS for indexed: one alpha byte per palette entry,
        // truncated after the last non-255 entry.
        let mut trns: Vec<u8> = palette.iter().map(|c| c.a.0).collect();
        while trns.last() == Some(&255) {
            trns.pop();
        }
        encoder.set_trns(trns);
    }

    // Palette entries are always sRGB.
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);

    // ── Compression ──────────────────────────────────────────────────────
    apply_compression(&mut encoder, options);

    // ── Pixel dimensions (pHYs) ──────────────────────────────────────────
    apply_pixel_dimensions(&mut encoder, options);

    // ── Text chunks ──────────────────────────────────────────────────────
    apply_text_chunks(&mut encoder, options)?;

    let mut png_writer = encoder.write_header().map_err(encode_error)?;

    // ── tIME chunk ───────────────────────────────────────────────────────
    write_time_chunk(&mut png_writer, options)?;

    // ── Write pixel data ─────────────────────────────────────────────────
    // Indexed8 is 1 byte per pixel — zero-copy via as_bytes().
    let bytes: &[u8] = image.as_bytes();
    png_writer.write_image_data(bytes).map_err(encode_error)?;

    Ok(())
}

/// Encode a [`PngImage`] back to PNG bytes.
///
/// Convenience wrapper that dispatches over all [`PngImage`] variants,
/// calling [`encode`] for the 16 non-indexed types and [`encode_indexed`]
/// for the indexed variant.  Useful for roundtripping and generic tooling.
///
/// Colour-space chunks are written automatically based on the pixel type
/// stored in each variant.
///
/// # Errors
///
/// Returns [`IoError::EncodeFailed`] if the `png` crate reports an error.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv_io::png::{self, PngEncodeOptions};
/// let decoded = png::decode(&std::fs::read("photo.png").unwrap()).unwrap();
/// let bytes = png::encode_png_image(&decoded.image, &PngEncodeOptions::default()).unwrap();
/// std::fs::write("copy.png", bytes).unwrap();
/// ```
pub fn encode_png_image(image: &PngImage, options: &PngEncodeOptions) -> Result<Vec<u8>, IoError> {
    match image {
        PngImage::SrgbMono8(img) => encode(img, options),
        PngImage::SrgbMonoA8(img) => encode(img, options),
        PngImage::Srgb8(img) => encode(img, options),
        PngImage::Srgba8(img) => encode(img, options),
        PngImage::Mono8(img) => encode(img, options),
        PngImage::MonoA8(img) => encode(img, options),
        PngImage::Rgb8(img) => encode(img, options),
        PngImage::Rgba8(img) => encode(img, options),
        PngImage::SrgbMono16(img) => encode(img, options),
        PngImage::SrgbMonoA16(img) => encode(img, options),
        PngImage::Srgb16(img) => encode(img, options),
        PngImage::Srgba16(img) => encode(img, options),
        PngImage::Mono16(img) => encode(img, options),
        PngImage::MonoA16(img) => encode(img, options),
        PngImage::Rgb16(img) => encode(img, options),
        PngImage::Rgba16(img) => encode(img, options),
        PngImage::Indexed8 { data, palette } => encode_indexed(data, palette.as_ref(), options),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Encoding — internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Map a `png::EncodingError` into [`IoError`].
fn encode_error(e: png::EncodingError) -> IoError {
    use png::EncodingError;
    match e {
        EncodingError::IoError(io) => IoError::Io(io),
        other => IoError::EncodeFailed {
            source: Box::new(other),
        },
    }
}

/// Apply compression level and filter strategy from [`PngEncodeOptions`]
/// to a `png::Encoder`.
fn apply_compression<W: std::io::Write>(encoder: &mut png::Encoder<W>, options: &PngEncodeOptions) {
    if let Some(level) = options.compression_level {
        let compression = match level {
            0..=2 => png::Compression::Fast,
            3..=6 => png::Compression::Default,
            _ => png::Compression::Best,
        };
        encoder.set_compression(compression);
    }

    if let Some(ref filter) = options.filter {
        match filter {
            PngFilterStrategy::Adaptive => {
                encoder.set_adaptive_filter(png::AdaptiveFilterType::Adaptive);
            }
            other => {
                let ft = match other {
                    PngFilterStrategy::None => png::FilterType::NoFilter,
                    PngFilterStrategy::Sub => png::FilterType::Sub,
                    PngFilterStrategy::Up => png::FilterType::Up,
                    PngFilterStrategy::Avg => png::FilterType::Avg,
                    PngFilterStrategy::Paeth => png::FilterType::Paeth,
                    PngFilterStrategy::Adaptive => unreachable!(),
                };
                encoder.set_filter(ft);
            }
        }
    }
}

/// Apply physical pixel dimensions (`pHYs` chunk) from [`PngEncodeOptions`]
/// to a `png::Encoder`.
fn apply_pixel_dimensions<W: std::io::Write>(
    encoder: &mut png::Encoder<W>,
    options: &PngEncodeOptions,
) {
    if let Some((xppu, yppu, unit)) = options.pixel_dimensions {
        encoder.set_pixel_dims(Some(png::PixelDimensions {
            xppu,
            yppu,
            unit: if unit == 1 {
                png::Unit::Meter
            } else {
                png::Unit::Unspecified
            },
        }));
    }
}

/// Apply text chunks (`tEXt` / `iTXt`) from [`PngEncodeOptions`] to a
/// `png::Encoder`.
///
/// Note: the `png` crate's `add_itxt_chunk` convenience method only sets
/// keyword + text.  `language_tag` and `translated_keyword` are not
/// forwarded — see the TODO for a `write_chunk`-based solution.
fn apply_text_chunks<W: std::io::Write>(
    encoder: &mut png::Encoder<W>,
    options: &PngEncodeOptions,
) -> Result<(), IoError> {
    for chunk in &options.text_chunks {
        if chunk.language.is_some() {
            encoder
                .add_itxt_chunk(chunk.keyword.clone(), chunk.text.clone())
                .map_err(encode_error)?;
        } else {
            encoder
                .add_text_chunk(chunk.keyword.clone(), chunk.text.clone())
                .map_err(encode_error)?;
        }
    }
    Ok(())
}

/// Write a `tIME` chunk to a `png::Writer` if [`PngEncodeOptions`] has a
/// timestamp.
fn write_time_chunk<W: std::io::Write>(
    writer: &mut png::Writer<W>,
    options: &PngEncodeOptions,
) -> Result<(), IoError> {
    if let Some(ref ts) = options.last_modified {
        let mut time_data = [0u8; 7];
        time_data[0..2].copy_from_slice(&ts.year.to_be_bytes());
        time_data[2] = ts.month;
        time_data[3] = ts.day;
        time_data[4] = ts.hour;
        time_data[5] = ts.minute;
        time_data[6] = ts.second;
        writer
            .write_chunk(png::chunk::tIME, &time_data)
            .map_err(encode_error)?;
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use irys_cv::image::ContiguousImage;
    use std::mem;
    use std::num::Saturating;

    /// The enum should stay compact — palette is boxed, everything else is
    /// `Image<T>` which is `Size` + `Box<[T]>`.  The enum itself should
    /// not blow up because of the indexed variant.
    #[test]
    fn png_image_enum_is_compact() {
        let indexed_variant_size =
            mem::size_of::<Image<Indexed8>>() + mem::size_of::<Box<[Srgba8; 256]>>();
        let mono_variant_size = mem::size_of::<Image<SrgbMono8>>();

        // Indexed variant should not be drastically larger than a simple variant.
        // Box<[Srgba8; 256]> is one pointer (8 bytes on 64-bit), not 1024 bytes.
        assert!(
            indexed_variant_size < mono_variant_size + 64,
            "Indexed8 variant is too large ({indexed_variant_size} bytes), \
             suggests the palette is inline instead of boxed"
        );
    }

    /// Verify that `PngDecoded` fields are accessible and the struct
    /// can be destructured.
    #[test]
    fn png_decoded_field_access() {
        let decoded = PngDecoded {
            image: PngImage::SrgbMono8(Image::generate(1, 1, |_, _| SrgbMono8::new(42))),
            metadata: PngMetadata {
                color_space: PngColorSpace::Unknown,
                source_bit_depth: 8,
                significant_bits: None,
                pixel_dimensions: None,
                text_chunks: vec![],
                last_modified: None,
            },
        };

        // Fields are directly accessible:
        let _img = decoded.image;
        let _meta = decoded.metadata;
    }

    /// Verify that all PngColorSpace variants are constructible.
    #[test]
    fn color_space_variants_are_constructible() {
        let _ = PngColorSpace::Unknown;

        let _ = PngColorSpace::Srgb {
            rendering_intent: 0,
        };

        let _ = PngColorSpace::Gamma {
            gamma: 0.45455,
            chromaticities: None,
        };

        let _ = PngColorSpace::IccProfile {
            profile: vec![0u8; 128].into_boxed_slice(),
        };

        let _ = PngColorSpace::Cicp {
            colour_primaries: 1,
            transfer_characteristics: 13,
            matrix_coefficients: 0,
            video_full_range: true,
        };
    }

    /// Verify PngTextChunk construction and equality.
    #[test]
    fn text_chunk_is_constructible() {
        let chunk = PngTextChunk {
            keyword: "Title".into(),
            text: "Test Image".into(),
            language: None,
            translated_keyword: None,
        };
        let clone = chunk.clone();
        assert_eq!(chunk, clone);
        assert_eq!(chunk.keyword, "Title");
        assert_eq!(chunk.text, "Test Image");
        assert!(chunk.language.is_none());
        assert!(chunk.translated_keyword.is_none());
    }

    /// Verify PngTimestamp is Copy and constructible.
    #[test]
    fn timestamp_is_constructible_and_copy() {
        let ts = PngTimestamp {
            year: 2025,
            month: 1,
            day: 15,
            hour: 12,
            minute: 30,
            second: 45,
        };
        let copy = ts; // Copy
        assert_eq!(ts, copy);
    }

    /// The PNG spec allows second = 60 for leap seconds.
    #[test]
    fn timestamp_allows_leap_second() {
        let ts = PngTimestamp {
            year: 2016,
            month: 12,
            day: 31,
            hour: 23,
            minute: 59,
            second: 60,
        };
        assert_eq!(ts.second, 60);
    }

    /// Verify PngMetadata can be fully populated.
    #[test]
    fn metadata_fully_populated() {
        let meta = PngMetadata {
            color_space: PngColorSpace::Srgb {
                rendering_intent: 1,
            },
            source_bit_depth: 8,
            significant_bits: Some(vec![8, 8, 8].into_boxed_slice()),
            pixel_dimensions: Some((3780, 3780, 1)),
            text_chunks: vec![PngTextChunk {
                keyword: "Software".into(),
                text: "irys-cv-io".into(),
                language: None,
                translated_keyword: None,
            }],
            last_modified: Some(PngTimestamp {
                year: 2025,
                month: 1,
                day: 1,
                hour: 0,
                minute: 0,
                second: 0,
            }),
        };

        assert_eq!(meta.source_bit_depth, 8);
        assert!(meta.significant_bits.is_some());
        assert!(meta.pixel_dimensions.is_some());
        assert_eq!(meta.text_chunks.len(), 1);
        assert!(meta.last_modified.is_some());
    }

    /// Verify PngMetadata with all optional fields as None.
    #[test]
    fn metadata_minimal() {
        let meta = PngMetadata {
            color_space: PngColorSpace::Unknown,
            source_bit_depth: 1,
            significant_bits: None,
            pixel_dimensions: None,
            text_chunks: vec![],
            last_modified: None,
        };
        assert!(meta.significant_bits.is_none());
        assert!(meta.pixel_dimensions.is_none());
        assert!(meta.text_chunks.is_empty());
        assert!(meta.last_modified.is_none());
    }

    /// ICC profiles are boxed slices, not Vecs.
    #[test]
    fn icc_profile_is_boxed_slice() {
        let profile_data = vec![0u8; 1024];
        let cs = PngColorSpace::IccProfile {
            profile: profile_data.into_boxed_slice(),
        };
        if let PngColorSpace::IccProfile { profile } = cs {
            assert_eq!(profile.len(), 1024);
        } else {
            panic!("expected IccProfile variant");
        }
    }

    // ── Decode integration tests ────────────────────────────────────────

    /// Helper: encode a raw image buffer into an in-memory PNG.
    fn encode_png(
        width: u32,
        height: u32,
        color_type: png::ColorType,
        bit_depth: png::BitDepth,
        data: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, width, height);
            encoder.set_color(color_type);
            encoder.set_depth(bit_depth);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(data).unwrap();
        }
        out
    }

    /// Helper: encode a raw image buffer with palette into an in-memory PNG.
    fn encode_indexed_png(
        width: u32,
        height: u32,
        bit_depth: png::BitDepth,
        palette: &[u8],
        trns: Option<&[u8]>,
        data: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, width, height);
            encoder.set_color(png::ColorType::Indexed);
            encoder.set_depth(bit_depth);
            encoder.set_palette(palette);
            if let Some(t) = trns {
                encoder.set_trns(t);
            }
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(data).unwrap();
        }
        out
    }

    /// Helper: encode PNG with an sRGB chunk.
    fn encode_png_with_srgb(
        width: u32,
        height: u32,
        color_type: png::ColorType,
        bit_depth: png::BitDepth,
        intent: png::SrgbRenderingIntent,
        data: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, width, height);
            encoder.set_color(color_type);
            encoder.set_depth(bit_depth);
            encoder.set_source_srgb(intent);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(data).unwrap();
        }
        out
    }

    #[test]
    fn decode_grayscale_8bit() {
        use irys_cv::image::ImageView;
        // 2×2 grayscale 8-bit image: pixel values 10, 20, 30, 40
        let data = [10u8, 20, 30, 40];
        let png_data = encode_png(2, 2, png::ColorType::Grayscale, png::BitDepth::Eight, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::SrgbMono8(img) => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.height(), 2);
                assert_eq!(img.get(0, 0).unwrap(), SrgbMono8::new(10));
                assert_eq!(img.get(1, 0).unwrap(), SrgbMono8::new(20));
                assert_eq!(img.get(0, 1).unwrap(), SrgbMono8::new(30));
                assert_eq!(img.get(1, 1).unwrap(), SrgbMono8::new(40));
            }
            other => panic!("expected SrgbMono8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_grayscale_16bit() {
        use irys_cv::image::ImageView;
        // 2×1 grayscale 16-bit: big-endian values 1000, 60000
        let mut data = Vec::new();
        data.extend_from_slice(&1000u16.to_be_bytes());
        data.extend_from_slice(&60000u16.to_be_bytes());
        let png_data = encode_png(
            2,
            1,
            png::ColorType::Grayscale,
            png::BitDepth::Sixteen,
            &data,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Mono16(img) => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.height(), 1);
                assert_eq!(img.get(0, 0).unwrap(), Mono16::new(1000));
                assert_eq!(img.get(1, 0).unwrap(), Mono16::new(60000));
            }
            other => panic!("expected Mono16, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_grayscale_1bit_full_range_scaling() {
        use irys_cv::image::ImageView;
        // 8×1, 1-bit: alternating 0 and 1 packed into one byte = 0b01010101 = 0x55
        let data = [0x55u8];
        let png_data = encode_png(8, 1, png::ColorType::Grayscale, png::BitDepth::One, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::SrgbMono8(img) => {
                assert_eq!(img.width(), 8);
                // 1-bit: 0→0, 1→255
                assert_eq!(img.get(0, 0).unwrap(), SrgbMono8::new(0));
                assert_eq!(img.get(1, 0).unwrap(), SrgbMono8::new(255));
                assert_eq!(img.get(2, 0).unwrap(), SrgbMono8::new(0));
                assert_eq!(img.get(3, 0).unwrap(), SrgbMono8::new(255));
            }
            other => panic!("expected SrgbMono8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_grayscale_4bit_full_range_scaling() {
        use irys_cv::image::ImageView;
        // 2×1, 4-bit: values 0 and 15 packed into 0x0F
        let data = [0x0Fu8];
        let png_data = encode_png(2, 1, png::ColorType::Grayscale, png::BitDepth::Four, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::SrgbMono8(img) => {
                // 4-bit: 0→0, 15→255 (×17)
                assert_eq!(img.get(0, 0).unwrap(), SrgbMono8::new(0));
                assert_eq!(img.get(1, 0).unwrap(), SrgbMono8::new(255));
            }
            other => panic!("expected SrgbMono8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_rgb_8bit() {
        use irys_cv::image::ImageView;
        // 1×1 RGB: (100, 150, 200)
        let data = [100u8, 150, 200];
        let png_data = encode_png(1, 1, png::ColorType::Rgb, png::BitDepth::Eight, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Srgb8(img) => {
                assert_eq!(img.width(), 1);
                assert_eq!(img.height(), 1);
                let px = img.get(0, 0).unwrap();
                assert_eq!(px, Srgb8::new(100, 150, 200));
            }
            other => panic!("expected Srgb8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_rgba_8bit() {
        use irys_cv::image::ImageView;
        // 1×1 RGBA: (10, 20, 30, 128)
        let data = [10u8, 20, 30, 128];
        let png_data = encode_png(1, 1, png::ColorType::Rgba, png::BitDepth::Eight, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Srgba8(img) => {
                assert_eq!(img.get(0, 0).unwrap(), Srgba8::new(10, 20, 30, 128));
            }
            other => panic!("expected Srgba8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_rgb_16bit_endianness() {
        use irys_cv::image::ImageView;
        // 1×1 RGB 16-bit: (0x1234, 0x5678, 0x9ABC)
        let mut data = Vec::new();
        data.extend_from_slice(&0x1234u16.to_be_bytes());
        data.extend_from_slice(&0x5678u16.to_be_bytes());
        data.extend_from_slice(&0x9ABCu16.to_be_bytes());
        let png_data = encode_png(1, 1, png::ColorType::Rgb, png::BitDepth::Sixteen, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Rgb16(img) => {
                let px = img.get(0, 0).unwrap();
                assert_eq!(px, Rgb16::new(0x1234, 0x5678, 0x9ABC));
            }
            other => panic!("expected Rgb16, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_rgba_16bit_endianness() {
        use irys_cv::image::ImageView;
        // 1×1 RGBA 16-bit
        let mut data = Vec::new();
        data.extend_from_slice(&100u16.to_be_bytes());
        data.extend_from_slice(&200u16.to_be_bytes());
        data.extend_from_slice(&300u16.to_be_bytes());
        data.extend_from_slice(&400u16.to_be_bytes());
        let png_data = encode_png(1, 1, png::ColorType::Rgba, png::BitDepth::Sixteen, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Rgba16(img) => {
                let px = img.get(0, 0).unwrap();
                assert_eq!(px, Rgba16::new(100, 200, 300, 400));
            }
            other => panic!("expected Rgba16, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_grayscale_alpha_8bit() {
        use irys_cv::image::ImageView;
        // 2×1 GA 8-bit: (100, 255), (200, 128)
        let data = [100u8, 255, 200, 128];
        let png_data = encode_png(
            2,
            1,
            png::ColorType::GrayscaleAlpha,
            png::BitDepth::Eight,
            &data,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::SrgbMonoA8(img) => {
                assert_eq!(img.width(), 2);
                let px0 = img.get(0, 0).unwrap();
                let px1 = img.get(1, 0).unwrap();
                assert_eq!(px0.v.0, 100);
                assert_eq!(px0.a.0, 255);
                assert_eq!(px1.v.0, 200);
                assert_eq!(px1.a.0, 128);
            }
            other => panic!("expected SrgbMonoA8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_grayscale_alpha_16bit() {
        use irys_cv::image::ImageView;
        // 1×1 GA 16-bit
        let mut data = Vec::new();
        data.extend_from_slice(&1000u16.to_be_bytes());
        data.extend_from_slice(&50000u16.to_be_bytes());
        let png_data = encode_png(
            1,
            1,
            png::ColorType::GrayscaleAlpha,
            png::BitDepth::Sixteen,
            &data,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::MonoA16(img) => {
                let px = img.get(0, 0).unwrap();
                assert_eq!(px.v.0, 1000);
                assert_eq!(px.a.0, 50000);
            }
            other => panic!("expected MonoA16, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_indexed_8bit() {
        use irys_cv::image::ImageView;
        // 2×1 indexed with a 3-entry palette
        let palette = [
            255u8, 0, 0, // entry 0: red
            0, 255, 0, // entry 1: green
            0, 0, 255, // entry 2: blue
        ];
        let data = [0u8, 2]; // pixel 0 → red, pixel 1 → blue
        let png_data = encode_indexed_png(2, 1, png::BitDepth::Eight, &palette, None, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Indexed8 {
                data: img,
                palette: pal,
            } => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.get(0, 0).unwrap().0, 0);
                assert_eq!(img.get(1, 0).unwrap().0, 2);
                // Palette entry 0 should be red, fully opaque
                assert_eq!(pal[0], Srgba8::new(255, 0, 0, 255));
                assert_eq!(pal[2], Srgba8::new(0, 0, 255, 255));
            }
            other => panic!("expected Indexed8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_indexed_with_trns() {
        let palette = [255u8, 0, 0, 0, 255, 0];
        let trns = [128u8, 64];
        let data = [0u8, 1];
        let png_data = encode_indexed_png(2, 1, png::BitDepth::Eight, &palette, Some(&trns), &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Indexed8 { palette: pal, .. } => {
                assert_eq!(pal[0], Srgba8::new(255, 0, 0, 128));
                assert_eq!(pal[1], Srgba8::new(0, 255, 0, 64));
            }
            other => panic!("expected Indexed8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_srgb_chunk_detected() {
        let data = [42u8, 43, 44];
        let png_data = encode_png_with_srgb(
            1,
            1,
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            png::SrgbRenderingIntent::Perceptual,
            &data,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.metadata.color_space {
            PngColorSpace::Srgb {
                rendering_intent: 0,
            } => {}
            other => panic!("expected Srgb {{ rendering_intent: 0 }}, got {other:?}"),
        }
    }

    #[test]
    fn decode_no_color_chunks_gives_unknown() {
        let data = [42u8];
        let png_data = encode_png(1, 1, png::ColorType::Grayscale, png::BitDepth::Eight, &data);
        let decoded = decode(&png_data).unwrap();
        assert!(
            matches!(decoded.metadata.color_space, PngColorSpace::Unknown),
            "expected Unknown, got {:?}",
            decoded.metadata.color_space
        );
    }

    #[test]
    fn decode_preserves_dimensions() {
        use irys_cv::image::ImageView;
        let data = vec![0u8; 17 * 31 * 3]; // 17×31 RGB
        let png_data = encode_png(17, 31, png::ColorType::Rgb, png::BitDepth::Eight, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Srgb8(img) => {
                assert_eq!(img.width(), 17);
                assert_eq!(img.height(), 31);
            }
            other => panic!("expected Srgb8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_invalid_data_returns_error() {
        let result = decode(b"not a png file");
        assert!(result.is_err());
    }

    #[test]
    fn decode_empty_data_returns_error() {
        let result = decode(b"");
        assert!(result.is_err());
    }

    #[test]
    fn decode_reader_from_cursor() {
        use irys_cv::image::ImageView;
        let data = [10u8, 20, 30];
        let png_data = encode_png(1, 1, png::ColorType::Rgb, png::BitDepth::Eight, &data);
        let cursor = std::io::Cursor::new(png_data);
        let decoded = decode_reader(cursor).unwrap();

        match &decoded.image {
            PngImage::Srgb8(img) => {
                assert_eq!(img.get(0, 0).unwrap(), Srgb8::new(10, 20, 30));
            }
            other => panic!("expected Srgb8, got {:?}", variant_name(other)),
        }
    }

    /// Encode → decode roundtrip for RGB 8-bit: all pixels must be exact.
    #[test]
    fn roundtrip_rgb8_pixel_exact() {
        use irys_cv::image::ImageView;
        // 4×2 image with distinct pixel values
        let data: Vec<u8> = (0..24).collect(); // 4×2×3 = 24 bytes
        let png_data = encode_png(4, 2, png::ColorType::Rgb, png::BitDepth::Eight, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Srgb8(img) => {
                assert_eq!(img.width(), 4);
                assert_eq!(img.height(), 2);
                for y in 0..2 {
                    for x in 0..4 {
                        let idx = (y * 4 + x) * 3;
                        let expected = Srgb8::new(data[idx], data[idx + 1], data[idx + 2]);
                        let actual = img.get(x, y).unwrap();
                        assert_eq!(
                            actual, expected,
                            "pixel ({x},{y}): expected {expected:?}, got {actual:?}"
                        );
                    }
                }
            }
            other => panic!("expected Srgb8, got {:?}", variant_name(other)),
        }
    }

    /// Encode → decode roundtrip for RGB 16-bit.
    #[test]
    fn roundtrip_rgb16_pixel_exact() {
        use irys_cv::image::ImageView;
        let pixels_native: Vec<(u16, u16, u16)> = vec![(0, 32768, 65535), (1000, 2000, 3000)];

        // Encode as big-endian bytes (PNG 16-bit format).
        let mut data = Vec::new();
        for (r, g, b) in &pixels_native {
            data.extend_from_slice(&r.to_be_bytes());
            data.extend_from_slice(&g.to_be_bytes());
            data.extend_from_slice(&b.to_be_bytes());
        }

        let png_data = encode_png(2, 1, png::ColorType::Rgb, png::BitDepth::Sixteen, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Rgb16(img) => {
                assert_eq!(img.width(), 2);
                let px0 = img.get(0, 0).unwrap();
                let px1 = img.get(1, 0).unwrap();
                assert_eq!(px0, Rgb16::new(0, 32768, 65535));
                assert_eq!(px1, Rgb16::new(1000, 2000, 3000));
            }
            other => panic!("expected Rgb16, got {:?}", variant_name(other)),
        }
    }

    /// Encode → decode roundtrip for Mono16.
    #[test]
    fn roundtrip_mono16_pixel_exact() {
        use irys_cv::image::ImageView;
        let values: Vec<u16> = vec![0, 1, 32768, 65534, 65535];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }

        let png_data = encode_png(
            5,
            1,
            png::ColorType::Grayscale,
            png::BitDepth::Sixteen,
            &data,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Mono16(img) => {
                assert_eq!(img.width(), 5);
                for (i, &v) in values.iter().enumerate() {
                    assert_eq!(
                        img.get(i, 0).unwrap(),
                        Mono16::new(v),
                        "pixel {i}: expected {v}"
                    );
                }
            }
            other => panic!("expected Mono16, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_grayscale_2bit_full_range_scaling() {
        use irys_cv::image::ImageView;
        // 4×1, 2-bit: values 0, 1, 2, 3 packed into one byte = 0b00_01_10_11 = 0x1B
        let data = [0x1Bu8];
        let png_data = encode_png(4, 1, png::ColorType::Grayscale, png::BitDepth::Two, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::SrgbMono8(img) => {
                // 2-bit: ×85 → 0, 85, 170, 255
                assert_eq!(img.get(0, 0).unwrap(), SrgbMono8::new(0));
                assert_eq!(img.get(1, 0).unwrap(), SrgbMono8::new(85));
                assert_eq!(img.get(2, 0).unwrap(), SrgbMono8::new(170));
                assert_eq!(img.get(3, 0).unwrap(), SrgbMono8::new(255));
            }
            other => panic!("expected SrgbMono8, got {:?}", variant_name(other)),
        }
    }

    /// Helper to get a variant name string for error messages.
    fn variant_name(img: &PngImage) -> &'static str {
        match img {
            PngImage::SrgbMono8(_) => "SrgbMono8",
            PngImage::SrgbMonoA8(_) => "SrgbMonoA8",
            PngImage::Srgb8(_) => "Srgb8",
            PngImage::Srgba8(_) => "Srgba8",
            PngImage::Mono8(_) => "Mono8",
            PngImage::MonoA8(_) => "MonoA8",
            PngImage::Rgb8(_) => "Rgb8",
            PngImage::Rgba8(_) => "Rgba8",
            PngImage::SrgbMono16(_) => "SrgbMono16",
            PngImage::SrgbMonoA16(_) => "SrgbMonoA16",
            PngImage::Srgb16(_) => "Srgb16",
            PngImage::Srgba16(_) => "Srgba16",
            PngImage::Mono16(_) => "Mono16",
            PngImage::MonoA16(_) => "MonoA16",
            PngImage::Rgb16(_) => "Rgb16",
            PngImage::Rgba16(_) => "Rgba16",
            PngImage::Indexed8 { .. } => "Indexed8",
        }
    }

    // ── TransferAssumption unit tests ────────────────────────────────────

    #[test]
    fn transfer_assumption_cicp_linear() {
        let cs = PngColorSpace::Cicp {
            colour_primaries: 1,
            transfer_characteristics: 8,
            matrix_coefficients: 0,
            video_full_range: true,
        };
        let ta = TransferAssumption::from_color_space(&cs);
        assert_eq!(ta, TransferAssumption::Linear);
        assert!(ta.is_linear(png::BitDepth::Eight));
        assert!(ta.is_linear(png::BitDepth::Sixteen));
    }

    #[test]
    fn transfer_assumption_cicp_srgb() {
        for tf in [1u8, 13] {
            let cs = PngColorSpace::Cicp {
                colour_primaries: 1,
                transfer_characteristics: tf,
                matrix_coefficients: 0,
                video_full_range: true,
            };
            let ta = TransferAssumption::from_color_space(&cs);
            assert_eq!(ta, TransferAssumption::Srgb, "TF={tf} should be Srgb");
            assert!(!ta.is_linear(png::BitDepth::Eight));
            assert!(!ta.is_linear(png::BitDepth::Sixteen));
        }
    }

    #[test]
    fn transfer_assumption_cicp_other_is_srgb() {
        // PQ (16) and HLG (18) are rejected by Q2 before reaching
        // TransferAssumption, but the function itself maps them to Srgb.
        // All other non-standard TFs also fall into Srgb (non-linear).
        for tf in [16u8, 18, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17] {
            let cs = PngColorSpace::Cicp {
                colour_primaries: 1,
                transfer_characteristics: tf,
                matrix_coefficients: 0,
                video_full_range: true,
            };
            let ta = TransferAssumption::from_color_space(&cs);
            assert_eq!(ta, TransferAssumption::Srgb, "TF={tf} should be Srgb");
        }
    }

    #[test]
    fn transfer_assumption_srgb_chunk() {
        let cs = PngColorSpace::Srgb {
            rendering_intent: 0,
        };
        let ta = TransferAssumption::from_color_space(&cs);
        assert_eq!(ta, TransferAssumption::Srgb);
        assert!(!ta.is_linear(png::BitDepth::Eight));
        assert!(!ta.is_linear(png::BitDepth::Sixteen));
    }

    #[test]
    fn transfer_assumption_gamma_linear() {
        // Exactly 1.0
        let cs = PngColorSpace::Gamma {
            gamma: 1.0,
            chromaticities: None,
        };
        assert_eq!(
            TransferAssumption::from_color_space(&cs),
            TransferAssumption::Linear
        );

        // Within tolerance: 0.9995
        let cs = PngColorSpace::Gamma {
            gamma: 0.9995,
            chromaticities: None,
        };
        assert_eq!(
            TransferAssumption::from_color_space(&cs),
            TransferAssumption::Linear
        );

        // Within tolerance: 1.0005
        let cs = PngColorSpace::Gamma {
            gamma: 1.0005,
            chromaticities: None,
        };
        assert_eq!(
            TransferAssumption::from_color_space(&cs),
            TransferAssumption::Linear
        );
    }

    #[test]
    fn transfer_assumption_gamma_nonlinear() {
        // sRGB gamma
        let cs = PngColorSpace::Gamma {
            gamma: 0.45455,
            chromaticities: None,
        };
        assert_eq!(
            TransferAssumption::from_color_space(&cs),
            TransferAssumption::Srgb
        );

        // gamma 2.2 (display gamma, not quite sRGB but non-linear)
        let cs = PngColorSpace::Gamma {
            gamma: 2.2,
            chromaticities: None,
        };
        assert_eq!(
            TransferAssumption::from_color_space(&cs),
            TransferAssumption::Srgb
        );
    }

    #[test]
    fn transfer_assumption_icc_profile() {
        let cs = PngColorSpace::IccProfile {
            profile: vec![0u8; 64].into_boxed_slice(),
        };
        assert_eq!(
            TransferAssumption::from_color_space(&cs),
            TransferAssumption::Srgb
        );
    }

    #[test]
    fn transfer_assumption_unknown() {
        let ta = TransferAssumption::from_color_space(&PngColorSpace::Unknown);
        assert_eq!(ta, TransferAssumption::Unknown);
    }

    #[test]
    fn transfer_assumption_is_linear_resolution() {
        // Unknown + 8-bit → sRGB (not linear)
        assert!(!TransferAssumption::Unknown.is_linear(png::BitDepth::Eight));
        assert!(!TransferAssumption::Unknown.is_linear(png::BitDepth::Four));
        assert!(!TransferAssumption::Unknown.is_linear(png::BitDepth::Two));
        assert!(!TransferAssumption::Unknown.is_linear(png::BitDepth::One));

        // Unknown + 16-bit → linear
        assert!(TransferAssumption::Unknown.is_linear(png::BitDepth::Sixteen));
    }

    // ── Linear 8-bit decode tests (Q6) ──────────────────────────────────

    /// Helper: encode PNG with a gAMA chunk.
    fn encode_png_with_gamma(
        width: u32,
        height: u32,
        color_type: png::ColorType,
        bit_depth: png::BitDepth,
        gamma: f32,
        data: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, width, height);
            encoder.set_color(color_type);
            encoder.set_depth(bit_depth);
            encoder.set_source_gamma(png::ScaledFloat::new(gamma));
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(data).unwrap();
        }
        out
    }

    #[test]
    fn decode_linear_grayscale_8bit_via_gamma() {
        use irys_cv::image::ImageView;
        // 2×2 grayscale 8-bit with gAMA = 1.0 (linear).
        let data = [10u8, 20, 30, 40];
        let png_data = encode_png_with_gamma(
            2,
            2,
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            1.0,
            &data,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Mono8(img) => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.height(), 2);
                assert_eq!(img.get(0, 0).unwrap(), Mono8::new(10));
                assert_eq!(img.get(1, 0).unwrap(), Mono8::new(20));
                assert_eq!(img.get(0, 1).unwrap(), Mono8::new(30));
                assert_eq!(img.get(1, 1).unwrap(), Mono8::new(40));
            }
            other => panic!("expected Mono8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_linear_rgb_8bit_via_gamma() {
        use irys_cv::image::ImageView;
        let data = [100u8, 150, 200];
        let png_data =
            encode_png_with_gamma(1, 1, png::ColorType::Rgb, png::BitDepth::Eight, 1.0, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Rgb8(img) => {
                assert_eq!(img.get(0, 0).unwrap(), Rgb8::new(100, 150, 200));
            }
            other => panic!("expected Rgb8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_linear_rgba_8bit_via_gamma() {
        use irys_cv::image::ImageView;
        let data = [10u8, 20, 30, 128];
        let png_data =
            encode_png_with_gamma(1, 1, png::ColorType::Rgba, png::BitDepth::Eight, 1.0, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Rgba8(img) => {
                assert_eq!(img.get(0, 0).unwrap(), Rgba8::new(10, 20, 30, 128));
            }
            other => panic!("expected Rgba8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_linear_grayscale_alpha_8bit_via_gamma() {
        use irys_cv::image::ImageView;
        let data = [100u8, 255, 200, 128];
        let png_data = encode_png_with_gamma(
            2,
            1,
            png::ColorType::GrayscaleAlpha,
            png::BitDepth::Eight,
            1.0,
            &data,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::MonoA8(img) => {
                assert_eq!(img.width(), 2);
                let px0 = img.get(0, 0).unwrap();
                let px1 = img.get(1, 0).unwrap();
                assert_eq!(px0.v.0, 100);
                assert_eq!(px0.a.0, 255);
                assert_eq!(px1.v.0, 200);
                assert_eq!(px1.a.0, 128);
            }
            other => panic!("expected MonoA8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_sub_byte_grayscale_linear_1bit() {
        use irys_cv::image::ImageView;
        // 8×1, 1-bit with linear gamma: alternating 0 and 1
        let data = [0x55u8]; // 0b01010101
        let png_data = encode_png_with_gamma(
            8,
            1,
            png::ColorType::Grayscale,
            png::BitDepth::One,
            1.0,
            &data,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Mono8(img) => {
                assert_eq!(img.width(), 8);
                assert_eq!(img.get(0, 0).unwrap(), Mono8::new(0));
                assert_eq!(img.get(1, 0).unwrap(), Mono8::new(255));
                assert_eq!(img.get(2, 0).unwrap(), Mono8::new(0));
                assert_eq!(img.get(3, 0).unwrap(), Mono8::new(255));
            }
            other => panic!("expected Mono8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_sub_byte_grayscale_linear_4bit() {
        use irys_cv::image::ImageView;
        // 2×1, 4-bit with linear gamma: values 0 and 15
        let data = [0x0Fu8];
        let png_data = encode_png_with_gamma(
            2,
            1,
            png::ColorType::Grayscale,
            png::BitDepth::Four,
            1.0,
            &data,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Mono8(img) => {
                assert_eq!(img.get(0, 0).unwrap(), Mono8::new(0));
                assert_eq!(img.get(1, 0).unwrap(), Mono8::new(255));
            }
            other => panic!("expected Mono8, got {:?}", variant_name(other)),
        }
    }

    // ── sRGB 8-bit: ensure existing behaviour is preserved ──────────────

    #[test]
    fn decode_srgb_grayscale_8bit_with_srgb_chunk() {
        use irys_cv::image::ImageView;
        let data = [42u8];
        let png_data = encode_png_with_srgb(
            1,
            1,
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            png::SrgbRenderingIntent::Perceptual,
            &data,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::SrgbMono8(img) => {
                assert_eq!(img.get(0, 0).unwrap(), SrgbMono8::new(42));
            }
            other => panic!("expected SrgbMono8, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_untagged_8bit_defaults_to_srgb() {
        // No colour-space chunks + 8-bit → sRGB by convention.
        let data = [42u8];
        let png_data = encode_png(1, 1, png::ColorType::Grayscale, png::BitDepth::Eight, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::SrgbMono8(_) => {}
            other => panic!("expected SrgbMono8, got {:?}", variant_name(other)),
        }
    }

    // ── 16-bit sRGB decode tests (Q7) ───────────────────────────────────

    #[test]
    fn decode_16bit_rgb_with_srgb_chunk_produces_srgb16() {
        use irys_cv::image::ImageView;
        let mut raw = Vec::new();
        raw.extend_from_slice(&0x1234u16.to_be_bytes());
        raw.extend_from_slice(&0x5678u16.to_be_bytes());
        raw.extend_from_slice(&0x9ABCu16.to_be_bytes());
        let png_data = encode_png_with_srgb(
            1,
            1,
            png::ColorType::Rgb,
            png::BitDepth::Sixteen,
            png::SrgbRenderingIntent::Perceptual,
            &raw,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Srgb16(img) => {
                let px = img.get(0, 0).unwrap();
                assert_eq!(px, Srgb16::new(0x1234, 0x5678, 0x9ABC));
            }
            other => panic!("expected Srgb16, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_16bit_rgba_with_srgb_chunk_produces_srgba16() {
        use irys_cv::image::ImageView;
        let mut raw = Vec::new();
        raw.extend_from_slice(&100u16.to_be_bytes());
        raw.extend_from_slice(&200u16.to_be_bytes());
        raw.extend_from_slice(&300u16.to_be_bytes());
        raw.extend_from_slice(&400u16.to_be_bytes());
        let png_data = encode_png_with_srgb(
            1,
            1,
            png::ColorType::Rgba,
            png::BitDepth::Sixteen,
            png::SrgbRenderingIntent::Perceptual,
            &raw,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Srgba16(img) => {
                let px = img.get(0, 0).unwrap();
                assert_eq!(px, Srgba16::new(100, 200, 300, 400));
            }
            other => panic!("expected Srgba16, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_16bit_grayscale_with_srgb_chunk_produces_srgb_mono16() {
        use irys_cv::image::ImageView;
        let raw = 32768u16.to_be_bytes();
        let png_data = encode_png_with_srgb(
            1,
            1,
            png::ColorType::Grayscale,
            png::BitDepth::Sixteen,
            png::SrgbRenderingIntent::Perceptual,
            &raw,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::SrgbMono16(img) => {
                assert_eq!(img.get(0, 0).unwrap(), SrgbMono16::new(32768));
            }
            other => panic!("expected SrgbMono16, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_16bit_grayscale_alpha_with_srgb_chunk_produces_srgb_mono_a16() {
        use irys_cv::image::ImageView;
        let mut raw = Vec::new();
        raw.extend_from_slice(&1000u16.to_be_bytes());
        raw.extend_from_slice(&50000u16.to_be_bytes());
        let png_data = encode_png_with_srgb(
            1,
            1,
            png::ColorType::GrayscaleAlpha,
            png::BitDepth::Sixteen,
            png::SrgbRenderingIntent::Perceptual,
            &raw,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::SrgbMonoA16(img) => {
                let px = img.get(0, 0).unwrap();
                assert_eq!(px.v.0, 1000);
                assert_eq!(px.a.0, 50000);
            }
            other => panic!("expected SrgbMonoA16, got {:?}", variant_name(other)),
        }
    }

    // ── 16-bit untagged / linear tests ──────────────────────────────────

    #[test]
    fn decode_16bit_untagged_defaults_to_linear() {
        // No colour-space chunk + 16-bit → linear by convention.
        let raw = 12345u16.to_be_bytes();
        let png_data = encode_png(
            1,
            1,
            png::ColorType::Grayscale,
            png::BitDepth::Sixteen,
            &raw,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Mono16(_) => {}
            other => panic!("expected Mono16, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_16bit_rgb_with_gamma_1_produces_rgb16() {
        use irys_cv::image::ImageView;
        let mut raw = Vec::new();
        raw.extend_from_slice(&100u16.to_be_bytes());
        raw.extend_from_slice(&200u16.to_be_bytes());
        raw.extend_from_slice(&300u16.to_be_bytes());
        let png_data =
            encode_png_with_gamma(1, 1, png::ColorType::Rgb, png::BitDepth::Sixteen, 1.0, &raw);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Rgb16(img) => {
                let px = img.get(0, 0).unwrap();
                assert_eq!(px, Rgb16::new(100, 200, 300));
            }
            other => panic!("expected Rgb16, got {:?}", variant_name(other)),
        }
    }

    #[test]
    fn decode_16bit_rgb_with_nonlinear_gamma_produces_srgb16() {
        use irys_cv::image::ImageView;
        let mut raw = Vec::new();
        raw.extend_from_slice(&100u16.to_be_bytes());
        raw.extend_from_slice(&200u16.to_be_bytes());
        raw.extend_from_slice(&300u16.to_be_bytes());
        let png_data = encode_png_with_gamma(
            1,
            1,
            png::ColorType::Rgb,
            png::BitDepth::Sixteen,
            0.45455, // sRGB-like gamma
            &raw,
        );
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Srgb16(img) => {
                let px = img.get(0, 0).unwrap();
                assert_eq!(px, Srgb16::new(100, 200, 300));
            }
            other => panic!("expected Srgb16, got {:?}", variant_name(other)),
        }
    }

    // ── Q2: PQ / HLG rejection tests ────────────────────────────────────

    /// Compute PNG CRC32 over the given bytes (type + data).
    ///
    /// Uses the standard CRC-32/ISO-HDLC polynomial (0xEDB88320, reflected).
    /// This is a test-only helper — not performance-critical.
    fn png_crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFF_FFFF;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
            }
        }
        crc ^ 0xFFFF_FFFF
    }

    /// Inject a cICP chunk into a raw PNG byte stream, right after IHDR.
    ///
    /// cICP chunk data is 4 bytes:
    ///   colour_primaries(1), transfer_characteristics(1),
    ///   matrix_coefficients(1), video_full_range(1).
    fn inject_cicp_chunk(
        png_bytes: &[u8],
        colour_primaries: u8,
        transfer_characteristics: u8,
        matrix_coefficients: u8,
        video_full_range: bool,
    ) -> Vec<u8> {
        // End of IHDR chunk: 8 sig + 4 len + 4 type + 13 data + 4 crc = 33.
        let ihdr_end = 8 + 4 + 4 + 13 + 4;
        assert!(png_bytes.len() > ihdr_end, "PNG too short for IHDR");

        let cicp_data = [
            colour_primaries,
            transfer_characteristics,
            matrix_coefficients,
            u8::from(video_full_range),
        ];
        let cicp_type = b"cICP";
        let length_bytes = 4u32.to_be_bytes();

        // CRC covers type + data.
        let mut crc_input = Vec::with_capacity(8);
        crc_input.extend_from_slice(cicp_type);
        crc_input.extend_from_slice(&cicp_data);
        let crc_bytes = png_crc32(&crc_input).to_be_bytes();

        let mut out = Vec::with_capacity(png_bytes.len() + 16);
        out.extend_from_slice(&png_bytes[..ihdr_end]);
        out.extend_from_slice(&length_bytes);
        out.extend_from_slice(cicp_type);
        out.extend_from_slice(&cicp_data);
        out.extend_from_slice(&crc_bytes);
        out.extend_from_slice(&png_bytes[ihdr_end..]);
        out
    }

    /// Inject a tIME chunk into a raw PNG byte stream, right after IHDR.
    fn inject_time_chunk(
        png_bytes: &[u8],
        year: u16,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
    ) -> Vec<u8> {
        let ihdr_end = 8 + 4 + 4 + 13 + 4;
        assert!(png_bytes.len() > ihdr_end, "PNG too short for IHDR");

        let year_bytes = year.to_be_bytes();
        let time_data = [
            year_bytes[0],
            year_bytes[1],
            month,
            day,
            hour,
            minute,
            second,
        ];
        let time_type = b"tIME";
        let length_bytes = 7u32.to_be_bytes();

        // CRC covers type + data.
        let mut crc_input = Vec::with_capacity(11);
        crc_input.extend_from_slice(time_type);
        crc_input.extend_from_slice(&time_data);
        let crc_bytes = png_crc32(&crc_input).to_be_bytes();

        let mut out = Vec::with_capacity(png_bytes.len() + 19);
        out.extend_from_slice(&png_bytes[..ihdr_end]);
        out.extend_from_slice(&length_bytes);
        out.extend_from_slice(time_type);
        out.extend_from_slice(&time_data);
        out.extend_from_slice(&crc_bytes);
        out.extend_from_slice(&png_bytes[ihdr_end..]);
        out
    }

    #[test]
    fn decode_cicp_pq_returns_unsupported() {
        // Construct a minimal PNG, then inject a cICP chunk with TF = 16 (PQ).
        let base = encode_png(1, 1, png::ColorType::Rgb, png::BitDepth::Eight, &[0, 0, 0]);
        let patched = inject_cicp_chunk(&base, 1, 16, 0, true);
        let result = decode(&patched);
        assert!(result.is_err(), "PQ should be rejected");
        let err = result.unwrap_err();
        match err {
            IoError::UnsupportedFeature { reason } => {
                assert!(
                    reason.contains("PQ") || reason.contains("16"),
                    "error should mention PQ: {reason}"
                );
            }
            other => panic!("expected UnsupportedFeature, got: {other}"),
        }
    }

    #[test]
    fn decode_cicp_hlg_returns_unsupported() {
        let base = encode_png(1, 1, png::ColorType::Rgb, png::BitDepth::Eight, &[0, 0, 0]);
        let patched = inject_cicp_chunk(&base, 1, 18, 0, true);
        let result = decode(&patched);
        assert!(result.is_err(), "HLG should be rejected");
        let err = result.unwrap_err();
        match err {
            IoError::UnsupportedFeature { reason } => {
                assert!(
                    reason.contains("HLG") || reason.contains("18"),
                    "error should mention HLG: {reason}"
                );
            }
            other => panic!("expected UnsupportedFeature, got: {other}"),
        }
    }

    #[test]
    fn decode_cicp_srgb_tf_is_accepted() {
        // cICP with TF = 13 (sRGB) should NOT be rejected.
        let base = encode_png(1, 1, png::ColorType::Rgb, png::BitDepth::Eight, &[0, 0, 0]);
        let patched = inject_cicp_chunk(&base, 1, 13, 0, true);
        let result = decode(&patched);
        assert!(result.is_ok(), "sRGB cICP (TF=13) should be accepted");
    }

    #[test]
    fn decode_cicp_linear_tf_is_accepted() {
        // cICP with TF = 8 (linear) should NOT be rejected.
        let base = encode_png(1, 1, png::ColorType::Rgb, png::BitDepth::Eight, &[0, 0, 0]);
        let patched = inject_cicp_chunk(&base, 1, 8, 0, true);
        let result = decode(&patched);
        assert!(result.is_ok(), "linear cICP (TF=8) should be accepted");
    }

    // ── tIME parsing tests ──────────────────────────────────────────────

    #[test]
    fn parse_time_chunk_extracts_timestamp() {
        let base = encode_png(1, 1, png::ColorType::Grayscale, png::BitDepth::Eight, &[42]);
        let patched = inject_time_chunk(&base, 2025, 7, 15, 10, 30, 45);
        let ts = parse_time_chunk(&patched);
        assert!(ts.is_some(), "should find tIME chunk");
        let ts = ts.unwrap();
        assert_eq!(ts.year, 2025);
        assert_eq!(ts.month, 7);
        assert_eq!(ts.day, 15);
        assert_eq!(ts.hour, 10);
        assert_eq!(ts.minute, 30);
        assert_eq!(ts.second, 45);
    }

    #[test]
    fn parse_time_chunk_none_when_absent() {
        let base = encode_png(1, 1, png::ColorType::Grayscale, png::BitDepth::Eight, &[42]);
        assert!(parse_time_chunk(&base).is_none());
    }

    #[test]
    fn parse_time_chunk_too_short() {
        assert!(parse_time_chunk(&[]).is_none());
        assert!(parse_time_chunk(&[0; 7]).is_none());
    }

    #[test]
    fn decode_populates_last_modified() {
        let base = encode_png(1, 1, png::ColorType::Grayscale, png::BitDepth::Eight, &[42]);
        let patched = inject_time_chunk(&base, 2024, 12, 25, 0, 0, 60);
        let decoded = decode(&patched).unwrap();
        let ts = decoded
            .metadata
            .last_modified
            .expect("should have timestamp");
        assert_eq!(ts.year, 2024);
        assert_eq!(ts.month, 12);
        assert_eq!(ts.day, 25);
        assert_eq!(ts.hour, 0);
        assert_eq!(ts.minute, 0);
        assert_eq!(ts.second, 60); // leap second
    }

    #[test]
    fn decode_reader_does_not_populate_last_modified() {
        // decode_reader cannot access raw bytes, so last_modified stays None.
        let base = encode_png(1, 1, png::ColorType::Grayscale, png::BitDepth::Eight, &[42]);
        let patched = inject_time_chunk(&base, 2025, 1, 1, 0, 0, 0);
        let decoded = decode_reader(std::io::Cursor::new(patched)).unwrap();
        assert!(
            decoded.metadata.last_modified.is_none(),
            "decode_reader should not populate last_modified"
        );
    }

    // ── Debug impl tests ────────────────────────────────────────────────

    #[test]
    fn png_image_debug_shows_variant_and_dimensions() {
        let img = PngImage::Srgb8(Image::generate(320, 240, |_, _| Srgb8::new(0, 0, 0)));
        let dbg = format!("{:?}", img);
        assert_eq!(dbg, "Srgb8(320x240)");
    }

    #[test]
    fn png_image_debug_indexed() {
        let img = PngImage::Indexed8 {
            data: Image::generate(16, 8, |_, _| Indexed8(0)),
            palette: Box::new([Srgba8::new(0, 0, 0, 255); 256]),
        };
        let dbg = format!("{:?}", img);
        assert_eq!(dbg, "Indexed8(16x8)");
    }

    #[test]
    fn png_image_debug_all_variants() {
        // Verify Debug doesn't panic for any variant and covers all 17 arms.
        let variants: Vec<PngImage> = vec![
            PngImage::SrgbMono8(Image::generate(1, 1, |_, _| SrgbMono8::new(0))),
            PngImage::SrgbMonoA8(Image::generate(1, 1, |_, _| SrgbMonoA8::new(0, 255))),
            PngImage::Srgb8(Image::generate(1, 1, |_, _| Srgb8::new(0, 0, 0))),
            PngImage::Srgba8(Image::generate(1, 1, |_, _| Srgba8::new(0, 0, 0, 255))),
            PngImage::Mono8(Image::generate(2, 3, |_, _| Mono8::new(0))),
            PngImage::MonoA8(Image::generate(1, 1, |_, _| MonoA8::new(0, 255))),
            PngImage::Rgb8(Image::generate(1, 1, |_, _| Rgb8::new(0, 0, 0))),
            PngImage::Rgba8(Image::generate(1, 1, |_, _| Rgba8::new(0, 0, 0, 255))),
            PngImage::SrgbMono16(Image::generate(4, 5, |_, _| SrgbMono16::new(0))),
            PngImage::SrgbMonoA16(Image::generate(1, 1, |_, _| SrgbMonoA16::new(0, 65535))),
            PngImage::Srgb16(Image::generate(1, 1, |_, _| Srgb16::new(0, 0, 0))),
            PngImage::Srgba16(Image::generate(1, 1, |_, _| Srgba16::new(0, 0, 0, 65535))),
            PngImage::Mono16(Image::generate(6, 7, |_, _| Mono16::new(0))),
            PngImage::MonoA16(Image::generate(1, 1, |_, _| MonoA16::new(0, 65535))),
            PngImage::Rgb16(Image::generate(8, 9, |_, _| Rgb16::new(0, 0, 0))),
            PngImage::Rgba16(Image::generate(1, 1, |_, _| Rgba16::new(0, 0, 0, 65535))),
            PngImage::Indexed8 {
                data: Image::generate(1, 1, |_, _| Indexed8(0)),
                palette: Box::new([Srgba8::new(0, 0, 0, 255); 256]),
            },
        ];
        assert_eq!(variants.len(), 17, "must cover all 17 PngImage variants");
        for v in &variants {
            let dbg = format!("{v:?}");
            assert!(!dbg.is_empty());
        }
    }

    #[test]
    fn png_decoded_implements_debug() {
        let decoded = PngDecoded {
            image: PngImage::Mono8(Image::generate(1, 1, |_, _| Mono8::new(0))),
            metadata: PngMetadata {
                color_space: PngColorSpace::Unknown,
                source_bit_depth: 8,
                significant_bits: None,
                pixel_dimensions: None,
                text_chunks: vec![],
                last_modified: None,
            },
        };
        // Just verify it compiles and doesn't panic.
        let dbg = format!("{:?}", decoded);
        assert!(!dbg.is_empty());
    }

    // ── Variant name coverage ───────────────────────────────────────────

    #[test]
    fn variant_name_covers_all() {
        // Ensure variant_name returns a distinct non-empty string for
        // every variant.  We construct one of each.
        let variants: Vec<PngImage> = vec![
            PngImage::SrgbMono8(Image::generate(1, 1, |_, _| SrgbMono8::new(0))),
            PngImage::SrgbMonoA8(Image::generate(1, 1, |_, _| SrgbMonoA8::new(0, 0))),
            PngImage::Srgb8(Image::generate(1, 1, |_, _| Srgb8::new(0, 0, 0))),
            PngImage::Srgba8(Image::generate(1, 1, |_, _| Srgba8::new(0, 0, 0, 0))),
            PngImage::Mono8(Image::generate(1, 1, |_, _| Mono8::new(0))),
            PngImage::MonoA8(Image::generate(1, 1, |_, _| MonoA8::new(0, 0))),
            PngImage::Rgb8(Image::generate(1, 1, |_, _| Rgb8::new(0, 0, 0))),
            PngImage::Rgba8(Image::generate(1, 1, |_, _| Rgba8::new(0, 0, 0, 0))),
            PngImage::SrgbMono16(Image::generate(1, 1, |_, _| SrgbMono16::new(0))),
            PngImage::SrgbMonoA16(Image::generate(1, 1, |_, _| SrgbMonoA16::new(0, 0))),
            PngImage::Srgb16(Image::generate(1, 1, |_, _| Srgb16::new(0, 0, 0))),
            PngImage::Srgba16(Image::generate(1, 1, |_, _| Srgba16::new(0, 0, 0, 0))),
            PngImage::Mono16(Image::generate(1, 1, |_, _| Mono16::new(0))),
            PngImage::MonoA16(Image::generate(1, 1, |_, _| MonoA16::new(0, 0))),
            PngImage::Rgb16(Image::generate(1, 1, |_, _| Rgb16::new(0, 0, 0))),
            PngImage::Rgba16(Image::generate(1, 1, |_, _| Rgba16::new(0, 0, 0, 0))),
            PngImage::Indexed8 {
                data: Image::generate(1, 1, |_, _| Indexed8(0)),
                palette: Box::new([Srgba8::new(0, 0, 0, 255); 256]),
            },
        ];

        let names: Vec<&str> = variants.iter().map(|v| variant_name(v)).collect();

        // All names are non-empty.
        for name in &names {
            assert!(!name.is_empty());
        }

        // All names are distinct (no two variants map to the same string).
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            names.len(),
            "duplicate variant names detected: {names:?}"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TransferTag
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn transfer_tag_variants() {
        // Ensure both variants exist and are distinct.
        assert_ne!(TransferTag::Srgb, TransferTag::Linear);
    }

    #[test]
    fn transfer_tag_is_copy() {
        let t = TransferTag::Srgb;
        let t2 = t; // Copy
        assert_eq!(t, t2);
    }

    #[test]
    fn transfer_tag_debug() {
        assert_eq!(format!("{:?}", TransferTag::Srgb), "Srgb");
        assert_eq!(format!("{:?}", TransferTag::Linear), "Linear");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PngPixel — associated constants
    // ═══════════════════════════════════════════════════════════════════════

    /// Verify every 8-bit sRGB type has the correct PNG wire parameters.
    #[test]
    fn png_pixel_8bit_srgb() {
        assert_eq!(SrgbMono8::PNG_COLOR_TYPE, png::ColorType::Grayscale);
        assert_eq!(SrgbMono8::PNG_BIT_DEPTH, png::BitDepth::Eight);
        assert_eq!(SrgbMono8::TRANSFER, TransferTag::Srgb);

        assert_eq!(SrgbMonoA8::PNG_COLOR_TYPE, png::ColorType::GrayscaleAlpha);
        assert_eq!(SrgbMonoA8::PNG_BIT_DEPTH, png::BitDepth::Eight);
        assert_eq!(SrgbMonoA8::TRANSFER, TransferTag::Srgb);

        assert_eq!(Srgb8::PNG_COLOR_TYPE, png::ColorType::Rgb);
        assert_eq!(Srgb8::PNG_BIT_DEPTH, png::BitDepth::Eight);
        assert_eq!(Srgb8::TRANSFER, TransferTag::Srgb);

        assert_eq!(Srgba8::PNG_COLOR_TYPE, png::ColorType::Rgba);
        assert_eq!(Srgba8::PNG_BIT_DEPTH, png::BitDepth::Eight);
        assert_eq!(Srgba8::TRANSFER, TransferTag::Srgb);
    }

    /// Verify every 8-bit linear type has the correct PNG wire parameters.
    #[test]
    fn png_pixel_8bit_linear() {
        assert_eq!(Mono8::PNG_COLOR_TYPE, png::ColorType::Grayscale);
        assert_eq!(Mono8::PNG_BIT_DEPTH, png::BitDepth::Eight);
        assert_eq!(Mono8::TRANSFER, TransferTag::Linear);

        assert_eq!(MonoA8::PNG_COLOR_TYPE, png::ColorType::GrayscaleAlpha);
        assert_eq!(MonoA8::PNG_BIT_DEPTH, png::BitDepth::Eight);
        assert_eq!(MonoA8::TRANSFER, TransferTag::Linear);

        assert_eq!(Rgb8::PNG_COLOR_TYPE, png::ColorType::Rgb);
        assert_eq!(Rgb8::PNG_BIT_DEPTH, png::BitDepth::Eight);
        assert_eq!(Rgb8::TRANSFER, TransferTag::Linear);

        assert_eq!(Rgba8::PNG_COLOR_TYPE, png::ColorType::Rgba);
        assert_eq!(Rgba8::PNG_BIT_DEPTH, png::BitDepth::Eight);
        assert_eq!(Rgba8::TRANSFER, TransferTag::Linear);
    }

    /// Verify every 16-bit sRGB type has the correct PNG wire parameters.
    #[test]
    fn png_pixel_16bit_srgb() {
        assert_eq!(SrgbMono16::PNG_COLOR_TYPE, png::ColorType::Grayscale);
        assert_eq!(SrgbMono16::PNG_BIT_DEPTH, png::BitDepth::Sixteen);
        assert_eq!(SrgbMono16::TRANSFER, TransferTag::Srgb);

        assert_eq!(SrgbMonoA16::PNG_COLOR_TYPE, png::ColorType::GrayscaleAlpha);
        assert_eq!(SrgbMonoA16::PNG_BIT_DEPTH, png::BitDepth::Sixteen);
        assert_eq!(SrgbMonoA16::TRANSFER, TransferTag::Srgb);

        assert_eq!(Srgb16::PNG_COLOR_TYPE, png::ColorType::Rgb);
        assert_eq!(Srgb16::PNG_BIT_DEPTH, png::BitDepth::Sixteen);
        assert_eq!(Srgb16::TRANSFER, TransferTag::Srgb);

        assert_eq!(Srgba16::PNG_COLOR_TYPE, png::ColorType::Rgba);
        assert_eq!(Srgba16::PNG_BIT_DEPTH, png::BitDepth::Sixteen);
        assert_eq!(Srgba16::TRANSFER, TransferTag::Srgb);
    }

    /// Verify every 16-bit linear type has the correct PNG wire parameters.
    #[test]
    fn png_pixel_16bit_linear() {
        assert_eq!(Mono16::PNG_COLOR_TYPE, png::ColorType::Grayscale);
        assert_eq!(Mono16::PNG_BIT_DEPTH, png::BitDepth::Sixteen);
        assert_eq!(Mono16::TRANSFER, TransferTag::Linear);

        assert_eq!(MonoA16::PNG_COLOR_TYPE, png::ColorType::GrayscaleAlpha);
        assert_eq!(MonoA16::PNG_BIT_DEPTH, png::BitDepth::Sixteen);
        assert_eq!(MonoA16::TRANSFER, TransferTag::Linear);

        assert_eq!(Rgb16::PNG_COLOR_TYPE, png::ColorType::Rgb);
        assert_eq!(Rgb16::PNG_BIT_DEPTH, png::BitDepth::Sixteen);
        assert_eq!(Rgb16::TRANSFER, TransferTag::Linear);

        assert_eq!(Rgba16::PNG_COLOR_TYPE, png::ColorType::Rgba);
        assert_eq!(Rgba16::PNG_BIT_DEPTH, png::BitDepth::Sixteen);
        assert_eq!(Rgba16::TRANSFER, TransferTag::Linear);
    }

    /// Verify Indexed8 maps to Indexed colour type, 8-bit, sRGB transfer.
    #[test]
    fn png_pixel_indexed8() {
        assert_eq!(Indexed8::PNG_COLOR_TYPE, png::ColorType::Indexed);
        assert_eq!(Indexed8::PNG_BIT_DEPTH, png::BitDepth::Eight);
        assert_eq!(Indexed8::TRANSFER, TransferTag::Srgb);
    }

    /// All 17 PngPixel implementors are accounted for — one assertion per
    /// type ensures the macro expanded for each.  This test is a compile-
    /// time guard: if an impl is missing, the associated constant access
    /// fails to compile.
    #[test]
    fn png_pixel_all_17_types_compile() {
        // We don't care about values here — the 5 tests above check those.
        // This test just proves all 17 impls exist by touching each one.
        let _types: [png::ColorType; 17] = [
            SrgbMono8::PNG_COLOR_TYPE,
            SrgbMonoA8::PNG_COLOR_TYPE,
            Srgb8::PNG_COLOR_TYPE,
            Srgba8::PNG_COLOR_TYPE,
            Mono8::PNG_COLOR_TYPE,
            MonoA8::PNG_COLOR_TYPE,
            Rgb8::PNG_COLOR_TYPE,
            Rgba8::PNG_COLOR_TYPE,
            SrgbMono16::PNG_COLOR_TYPE,
            SrgbMonoA16::PNG_COLOR_TYPE,
            Srgb16::PNG_COLOR_TYPE,
            Srgba16::PNG_COLOR_TYPE,
            Mono16::PNG_COLOR_TYPE,
            MonoA16::PNG_COLOR_TYPE,
            Rgb16::PNG_COLOR_TYPE,
            Rgba16::PNG_COLOR_TYPE,
            Indexed8::PNG_COLOR_TYPE,
        ];
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PngFilterStrategy
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn png_filter_strategy_all_variants() {
        // All six variants are distinct and Debug-formattable.
        let variants = [
            PngFilterStrategy::None,
            PngFilterStrategy::Sub,
            PngFilterStrategy::Up,
            PngFilterStrategy::Avg,
            PngFilterStrategy::Paeth,
            PngFilterStrategy::Adaptive,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b, "{:?} should differ from {:?}", a, b);
                }
            }
        }
    }

    #[test]
    fn png_filter_strategy_is_copy() {
        let f = PngFilterStrategy::Paeth;
        let f2 = f; // Copy
        assert_eq!(f, f2);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PngEncodeOptions
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn png_encode_options_default() {
        let opts = PngEncodeOptions::default();
        assert!(opts.text_chunks.is_empty());
        assert!(opts.pixel_dimensions.is_none());
        assert!(opts.last_modified.is_none());
        assert!(opts.compression_level.is_none());
        assert!(opts.filter.is_none());
    }

    #[test]
    fn png_encode_options_with_all_fields() {
        let opts = PngEncodeOptions {
            text_chunks: vec![PngTextChunk {
                keyword: "Title".into(),
                text: "Test".into(),
                language: None,
                translated_keyword: None,
            }],
            pixel_dimensions: Some((300, 300, 1)),
            last_modified: Some(PngTimestamp {
                year: 2025,
                month: 6,
                day: 25,
                hour: 14,
                minute: 30,
                second: 0,
            }),
            compression_level: Some(9),
            filter: Some(PngFilterStrategy::Adaptive),
        };
        assert_eq!(opts.text_chunks.len(), 1);
        assert_eq!(opts.text_chunks[0].keyword, "Title");
        assert_eq!(opts.pixel_dimensions, Some((300, 300, 1)));
        assert_eq!(opts.compression_level, Some(9));
        assert_eq!(opts.filter, Some(PngFilterStrategy::Adaptive));
        assert!(opts.last_modified.is_some());
    }

    #[test]
    fn png_encode_options_debug() {
        let opts = PngEncodeOptions::default();
        let dbg = format!("{:?}", opts);
        assert!(dbg.contains("PngEncodeOptions"));
    }

    #[test]
    fn png_encode_options_clone() {
        let mut opts = PngEncodeOptions::default();
        opts.compression_level = Some(3);
        opts.filter = Some(PngFilterStrategy::Sub);
        let opts2 = opts.clone();
        assert_eq!(opts2.compression_level, Some(3));
        assert_eq!(opts2.filter, Some(PngFilterStrategy::Sub));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Encoding — roundtrip tests
    // ═══════════════════════════════════════════════════════════════════════

    /// Helper: roundtrip encode→decode for any PngPixel type.
    /// Returns the decoded PngImage so the caller can match the variant.
    fn roundtrip_encode<P: PngPixel + std::fmt::Debug + PartialEq>(
        pixels: Vec<P>,
        width: usize,
        height: usize,
    ) -> PngDecoded {
        let image = Image::from_vec(width, height, pixels).unwrap();
        let opts = PngEncodeOptions::default();
        let bytes = encode(&image, &opts).unwrap();
        decode(&bytes).unwrap()
    }

    #[test]
    fn encode_roundtrip_srgb_mono8() {
        let pixels = vec![SrgbMono8::new(0), SrgbMono8::new(128), SrgbMono8::new(255)];
        let decoded = roundtrip_encode(pixels.clone(), 3, 1);
        match decoded.image {
            PngImage::SrgbMono8(img) => {
                assert_eq!(img.width(), 3);
                assert_eq!(img.as_slice(), &pixels);
            }
            other => panic!("expected SrgbMono8, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_srgb_mono_a8() {
        let pixels = vec![SrgbMonoA8 {
            v: Saturating(100),
            a: Saturating(200),
        }];
        let decoded = roundtrip_encode(pixels.clone(), 1, 1);
        match decoded.image {
            PngImage::SrgbMonoA8(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected SrgbMonoA8, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_srgb8() {
        let pixels = vec![Srgb8::new(10, 20, 30), Srgb8::new(200, 100, 50)];
        let decoded = roundtrip_encode(pixels.clone(), 2, 1);
        match decoded.image {
            PngImage::Srgb8(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_srgba8() {
        let pixels = vec![Srgba8::new(10, 20, 30, 128)];
        let decoded = roundtrip_encode(pixels.clone(), 1, 1);
        match decoded.image {
            PngImage::Srgba8(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected Srgba8, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_mono8() {
        let pixels = vec![Mono8::new(0), Mono8::new(42), Mono8::new(255)];
        let image = Image::from_vec(3, 1, pixels.clone()).unwrap();
        // Mono8 is linear — must tag with gAMA=1.0 so decoder sees Linear.
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Mono8(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected Mono8, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_mono_a8() {
        let pixels = vec![MonoA8 {
            v: Saturating(50),
            a: Saturating(200),
        }];
        let image = Image::from_vec(1, 1, pixels.clone()).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::MonoA8(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected MonoA8, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_rgb8() {
        let pixels = vec![Rgb8::new(100, 150, 200)];
        let image = Image::from_vec(1, 1, pixels.clone()).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Rgb8(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected Rgb8, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_rgba8() {
        let pixels = vec![Rgba8::new(10, 20, 30, 128)];
        let image = Image::from_vec(1, 1, pixels.clone()).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Rgba8(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected Rgba8, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_srgb_mono16() {
        let pixels = vec![
            SrgbMono16::new(0),
            SrgbMono16::new(1000),
            SrgbMono16::new(65535),
        ];
        let decoded = roundtrip_encode(pixels.clone(), 3, 1);
        match decoded.image {
            PngImage::SrgbMono16(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected SrgbMono16, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_srgb_mono_a16() {
        let pixels = vec![SrgbMonoA16 {
            v: Saturating(5000),
            a: Saturating(60000),
        }];
        let decoded = roundtrip_encode(pixels.clone(), 1, 1);
        match decoded.image {
            PngImage::SrgbMonoA16(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected SrgbMonoA16, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_srgb16() {
        let pixels = vec![Srgb16::new(100, 200, 300)];
        let decoded = roundtrip_encode(pixels.clone(), 1, 1);
        match decoded.image {
            PngImage::Srgb16(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected Srgb16, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_srgba16() {
        let pixels = vec![Srgba16::new(100, 200, 300, 400)];
        let decoded = roundtrip_encode(pixels.clone(), 1, 1);
        match decoded.image {
            PngImage::Srgba16(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected Srgba16, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_mono16() {
        let pixels = vec![Mono16::new(0), Mono16::new(32768), Mono16::new(65535)];
        let image = Image::from_vec(3, 1, pixels.clone()).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Mono16(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected Mono16, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_mono_a16() {
        let pixels = vec![MonoA16 {
            v: Saturating(1234),
            a: Saturating(5678),
        }];
        let image = Image::from_vec(1, 1, pixels.clone()).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::MonoA16(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected MonoA16, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_rgb16() {
        let pixels = vec![Rgb16::new(0x0102, 0x0304, 0x0506)];
        let image = Image::from_vec(1, 1, pixels.clone()).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Rgb16(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected Rgb16, got {:?}", other),
        }
    }

    #[test]
    fn encode_roundtrip_rgba16() {
        let pixels = vec![Rgba16::new(100, 200, 300, 400)];
        let image = Image::from_vec(1, 1, pixels.clone()).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Rgba16(img) => assert_eq!(img.as_slice(), &pixels),
            other => panic!("expected Rgba16, got {:?}", other),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Encoding — colour-space tagging
    // ═══════════════════════════════════════════════════════════════════════

    /// sRGB 8-bit: encoder must write sRGB chunk, decoder must detect sRGB.
    #[test]
    fn encode_colour_space_srgb_8bit() {
        let image = Image::from_vec(1, 1, vec![Srgb8::new(42, 84, 126)]).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert!(
            matches!(decoded.metadata.color_space, PngColorSpace::Srgb { .. }),
            "expected Srgb, got {:?}",
            decoded.metadata.color_space
        );
    }

    /// Linear 8-bit: encoder must write gAMA=1.0, decoder must detect linear.
    #[test]
    fn encode_colour_space_linear_8bit() {
        let image = Image::from_vec(1, 1, vec![Rgb8::new(42, 84, 126)]).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.metadata.color_space {
            PngColorSpace::Gamma { gamma, .. } => {
                assert!(
                    (gamma - 1.0).abs() < 0.001,
                    "expected gamma ≈ 1.0, got {}",
                    gamma
                );
            }
            other => panic!("expected Gamma, got {:?}", other),
        }
    }

    /// sRGB 16-bit: CRITICAL — without sRGB chunk, 16-bit defaults to linear.
    /// The encoder must write sRGB chunk to preserve the transfer function.
    #[test]
    fn encode_colour_space_srgb_16bit() {
        let image = Image::from_vec(1, 1, vec![Srgb16::new(100, 200, 300)]).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert!(
            matches!(decoded.metadata.color_space, PngColorSpace::Srgb { .. }),
            "expected Srgb, got {:?}",
            decoded.metadata.color_space
        );
        // Must decode as Srgb16, not Rgb16.
        assert!(
            matches!(decoded.image, PngImage::Srgb16(_)),
            "expected Srgb16 variant, got {:?}",
            decoded.image
        );
    }

    /// Linear 16-bit: encoder writes gAMA=1.0.
    #[test]
    fn encode_colour_space_linear_16bit() {
        let image = Image::from_vec(1, 1, vec![Rgb16::new(100, 200, 300)]).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.metadata.color_space {
            PngColorSpace::Gamma { gamma, .. } => {
                assert!(
                    (gamma - 1.0).abs() < 0.001,
                    "expected gamma ≈ 1.0, got {}",
                    gamma
                );
            }
            other => panic!("expected Gamma, got {:?}", other),
        }
        assert!(
            matches!(decoded.image, PngImage::Rgb16(_)),
            "expected Rgb16 variant, got {:?}",
            decoded.image
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Encoding — 16-bit endianness
    // ═══════════════════════════════════════════════════════════════════════

    /// Encode Rgb16 with known values, decode, verify exact values.
    /// Catches byte-swap bugs.
    #[test]
    fn encode_16bit_endianness_roundtrip() {
        let pixels = vec![Rgb16::new(0x0102, 0x0304, 0x0506)];
        let image = Image::from_vec(1, 1, pixels).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Rgb16(img) => {
                let p = img.as_slice()[0];
                assert_eq!(p.r.0, 0x0102, "red channel mismatch");
                assert_eq!(p.g.0, 0x0304, "green channel mismatch");
                assert_eq!(p.b.0, 0x0506, "blue channel mismatch");
            }
            other => panic!("expected Rgb16, got {:?}", other),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Encoding — indexed
    // ═══════════════════════════════════════════════════════════════════════

    /// Roundtrip indexed image without transparency.
    #[test]
    fn encode_indexed_roundtrip_no_transparency() {
        let pixels = vec![Indexed8(0), Indexed8(1), Indexed8(2)];
        let image = Image::from_vec(3, 1, pixels.clone()).unwrap();
        let palette = [
            Srgba8::new(255, 0, 0, 255),
            Srgba8::new(0, 255, 0, 255),
            Srgba8::new(0, 0, 255, 255),
        ];
        let bytes = encode_indexed(&image, &palette, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Indexed8 { data, palette: pal } => {
                assert_eq!(data.as_slice(), &pixels);
                // First 3 entries must match.
                assert_eq!(pal[0], Srgba8::new(255, 0, 0, 255));
                assert_eq!(pal[1], Srgba8::new(0, 255, 0, 255));
                assert_eq!(pal[2], Srgba8::new(0, 0, 255, 255));
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    /// Roundtrip indexed image with transparency (tRNS chunk).
    #[test]
    fn encode_indexed_roundtrip_with_transparency() {
        let pixels = vec![Indexed8(0), Indexed8(1)];
        let image = Image::from_vec(2, 1, pixels).unwrap();
        let palette = [
            Srgba8::new(255, 0, 0, 128), // semi-transparent
            Srgba8::new(0, 255, 0, 255), // opaque
        ];
        let bytes = encode_indexed(&image, &palette, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Indexed8 { palette: pal, .. } => {
                assert_eq!(pal[0].a.0, 128, "alpha of entry 0 not preserved");
                assert_eq!(pal[1].a.0, 255, "alpha of entry 1 not preserved");
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    /// Empty palette → EncodeFailed.
    #[test]
    fn encode_indexed_empty_palette() {
        let image = Image::from_vec(1, 1, vec![Indexed8(0)]).unwrap();
        let result = encode_indexed(&image, &[], &PngEncodeOptions::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, IoError::EncodeFailed { .. }),
            "expected EncodeFailed, got {:?}",
            err
        );
    }

    /// Palette with 257 entries → EncodeFailed.
    #[test]
    fn encode_indexed_palette_too_large() {
        let image = Image::from_vec(1, 1, vec![Indexed8(0)]).unwrap();
        let palette = vec![Srgba8::new(0, 0, 0, 255); 257];
        let result = encode_indexed(&image, &palette, &PngEncodeOptions::default());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), IoError::EncodeFailed { .. }));
    }

    /// Pixel index ≥ palette length → EncodeFailed.
    #[test]
    fn encode_indexed_index_out_of_range() {
        let image = Image::from_vec(1, 1, vec![Indexed8(5)]).unwrap();
        let palette = [Srgba8::new(0, 0, 0, 255); 3]; // max valid index = 2
        let result = encode_indexed(&image, &palette, &PngEncodeOptions::default());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), IoError::EncodeFailed { .. }));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Encoding — metadata roundtrip
    // ═══════════════════════════════════════════════════════════════════════

    /// Text chunks survive an encode→decode roundtrip.
    #[test]
    fn encode_text_chunk_roundtrip() {
        let image = Image::from_vec(1, 1, vec![Srgb8::new(0, 0, 0)]).unwrap();
        let mut opts = PngEncodeOptions::default();
        opts.text_chunks = vec![PngTextChunk {
            keyword: "Title".into(),
            text: "Test Image".into(),
            language: None,
            translated_keyword: None,
        }];
        let bytes = encode(&image, &opts).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert!(
            decoded
                .metadata
                .text_chunks
                .iter()
                .any(|c| c.keyword == "Title" && c.text == "Test Image"),
            "text chunk not found in decoded metadata: {:?}",
            decoded.metadata.text_chunks
        );
    }

    /// Timestamp survives an encode→decode roundtrip.
    /// Note: only `decode` (not `decode_reader`) populates `last_modified`.
    #[test]
    fn encode_timestamp_roundtrip() {
        let image = Image::from_vec(1, 1, vec![Srgb8::new(0, 0, 0)]).unwrap();
        let ts = PngTimestamp {
            year: 2025,
            month: 6,
            day: 25,
            hour: 14,
            minute: 30,
            second: 0,
        };
        let mut opts = PngEncodeOptions::default();
        opts.last_modified = Some(ts);
        let bytes = encode(&image, &opts).unwrap();
        // Must use `decode`, not `decode_reader`, because tIME parsing
        // requires raw byte access.
        let decoded = decode(&bytes).unwrap();
        assert_eq!(
            decoded.metadata.last_modified,
            Some(ts),
            "timestamp not preserved in roundtrip"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Encoding — encode_writer byte-identity with encode
    // ═══════════════════════════════════════════════════════════════════════

    /// `encode_writer` to a Vec<u8> must produce byte-identical output
    /// to `encode`.
    #[test]
    fn encode_writer_byte_identical_to_encode() {
        let image = Image::from_vec(
            2,
            2,
            vec![
                Srgb8::new(10, 20, 30),
                Srgb8::new(40, 50, 60),
                Srgb8::new(70, 80, 90),
                Srgb8::new(100, 110, 120),
            ],
        )
        .unwrap();
        let opts = PngEncodeOptions::default();

        let via_encode = encode(&image, &opts).unwrap();

        let mut via_writer = Vec::new();
        encode_writer(&image, &mut via_writer, &opts).unwrap();

        assert_eq!(via_encode, via_writer, "encode and encode_writer differ");
    }

    /// Same test for 16-bit to catch any endianness divergence.
    #[test]
    fn encode_writer_byte_identical_16bit() {
        let image = Image::from_vec(1, 1, vec![Rgb16::new(0x1234, 0x5678, 0x9ABC)]).unwrap();
        let opts = PngEncodeOptions::default();

        let via_encode = encode(&image, &opts).unwrap();

        let mut via_writer = Vec::new();
        encode_writer(&image, &mut via_writer, &opts).unwrap();

        assert_eq!(via_encode, via_writer);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Encoding — compression / filter options
    // ═══════════════════════════════════════════════════════════════════════

    /// Encoding with explicit compression level produces a valid PNG.
    #[test]
    fn encode_with_compression_level() {
        let image = Image::from_vec(
            2,
            2,
            vec![
                Srgb8::new(0, 0, 0),
                Srgb8::new(255, 255, 255),
                Srgb8::new(128, 128, 128),
                Srgb8::new(64, 64, 64),
            ],
        )
        .unwrap();
        let mut opts = PngEncodeOptions::default();
        opts.compression_level = Some(9);
        let bytes = encode(&image, &opts).unwrap();
        // Should still decode correctly.
        let decoded = decode(&bytes).unwrap();
        assert!(matches!(decoded.image, PngImage::Srgb8(_)));
    }

    /// Encoding with a fixed filter strategy produces a valid PNG.
    #[test]
    fn encode_with_filter_strategy() {
        let image = Image::from_vec(
            2,
            2,
            vec![
                Srgb8::new(0, 0, 0),
                Srgb8::new(255, 255, 255),
                Srgb8::new(128, 128, 128),
                Srgb8::new(64, 64, 64),
            ],
        )
        .unwrap();
        let mut opts = PngEncodeOptions::default();
        opts.filter = Some(PngFilterStrategy::Paeth);
        let bytes = encode(&image, &opts).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert!(matches!(decoded.image, PngImage::Srgb8(_)));
    }

    /// Encoding with adaptive filter produces a valid PNG.
    #[test]
    fn encode_with_adaptive_filter() {
        let image = Image::from_vec(
            2,
            2,
            vec![
                Srgb8::new(0, 0, 0),
                Srgb8::new(255, 255, 255),
                Srgb8::new(128, 128, 128),
                Srgb8::new(64, 64, 64),
            ],
        )
        .unwrap();
        let mut opts = PngEncodeOptions::default();
        opts.filter = Some(PngFilterStrategy::Adaptive);
        let bytes = encode(&image, &opts).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert!(matches!(decoded.image, PngImage::Srgb8(_)));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Encoding — multi-row images
    // ═══════════════════════════════════════════════════════════════════════

    /// Verify encoding preserves dimensions on a non-trivial image.
    #[test]
    fn encode_preserves_dimensions() {
        let pixels: Vec<Srgb8> = (0..17 * 31)
            .map(|i| {
                let v = (i % 256) as u8;
                Srgb8::new(v, v, v)
            })
            .collect();
        let image = Image::from_vec(17, 31, pixels.clone()).unwrap();
        let bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Srgb8(img) => {
                assert_eq!(img.width(), 17);
                assert_eq!(img.height(), 31);
                assert_eq!(img.as_slice(), &pixels);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // encode_png_image tests
    // ═══════════════════════════════════════════════════════════════════════

    /// Roundtrip `Srgb8` through `encode_png_image`.
    #[test]
    fn encode_png_image_roundtrip_srgb8() {
        let pixels = vec![Srgb8::new(10, 20, 30), Srgb8::new(40, 50, 60)];
        let image = Image::from_vec(2, 1, pixels.clone()).unwrap();
        let png_image = PngImage::Srgb8(image);
        let bytes = encode_png_image(&png_image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Srgb8(img) => {
                assert_eq!(img.as_slice(), &pixels);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    /// Roundtrip `Mono16` through `encode_png_image`.
    #[test]
    fn encode_png_image_roundtrip_mono16() {
        let pixels = vec![Mono16::new(1000), Mono16::new(2000), Mono16::new(3000)];
        let image = Image::from_vec(3, 1, pixels.clone()).unwrap();
        let png_image = PngImage::Mono16(image);
        let bytes = encode_png_image(&png_image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Mono16(img) => {
                assert_eq!(img.as_slice(), &pixels);
            }
            other => panic!("expected Mono16, got {:?}", other),
        }
    }

    /// Roundtrip `Srgba16` through `encode_png_image`.
    #[test]
    fn encode_png_image_roundtrip_srgba16() {
        let pixels = vec![Srgba16::new(100, 200, 300, 400)];
        let image = Image::from_vec(1, 1, pixels.clone()).unwrap();
        let png_image = PngImage::Srgba16(image);
        let bytes = encode_png_image(&png_image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Srgba16(img) => {
                assert_eq!(img.as_slice(), &pixels);
            }
            other => panic!("expected Srgba16, got {:?}", other),
        }
    }

    /// Roundtrip indexed image through `encode_png_image`.
    #[test]
    fn encode_png_image_roundtrip_indexed() {
        let palette_entries = [
            Srgba8::new(255, 0, 0, 255),
            Srgba8::new(0, 255, 0, 255),
            Srgba8::new(0, 0, 255, 128),
        ];
        let mut full_palette = [Srgba8::new(0, 0, 0, 255); 256];
        for (i, p) in palette_entries.iter().enumerate() {
            full_palette[i] = *p;
        }
        let pixels = vec![Indexed8(0), Indexed8(1), Indexed8(2), Indexed8(0)];
        let image = Image::from_vec(2, 2, pixels.clone()).unwrap();
        let png_image = PngImage::Indexed8 {
            data: image,
            palette: Box::new(full_palette),
        };
        let bytes = encode_png_image(&png_image, &PngEncodeOptions::default()).unwrap();
        let decoded = decode(&bytes).unwrap();
        match decoded.image {
            PngImage::Indexed8 { data, palette } => {
                assert_eq!(data.as_slice(), &pixels);
                // Verify the first 3 palette entries match
                for i in 0..3 {
                    assert_eq!(palette[i], palette_entries[i]);
                }
            }
            other => panic!("expected Indexed8, got {:?}", other),
        }
    }

    /// Roundtrip: decode → encode_png_image → decode, verify identical pixels.
    #[test]
    fn encode_png_image_decode_roundtrip() {
        // Create a known image, encode it, decode it, re-encode via
        // encode_png_image, decode again, compare.
        let pixels = vec![
            Srgb8::new(11, 22, 33),
            Srgb8::new(44, 55, 66),
            Srgb8::new(77, 88, 99),
            Srgb8::new(111, 122, 133),
        ];
        let image = Image::from_vec(2, 2, pixels.clone()).unwrap();
        let first_bytes = encode(&image, &PngEncodeOptions::default()).unwrap();
        let first_decoded = decode(&first_bytes).unwrap();

        // Re-encode from the PngImage
        let second_bytes =
            encode_png_image(&first_decoded.image, &PngEncodeOptions::default()).unwrap();
        let second_decoded = decode(&second_bytes).unwrap();

        match second_decoded.image {
            PngImage::Srgb8(img) => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.height(), 2);
                assert_eq!(img.as_slice(), &pixels);
            }
            other => panic!("expected Srgb8, got {:?}", other),
        }
    }

    /// All 16 non-indexed variants roundtrip through `encode_png_image`.
    #[test]
    fn encode_png_image_all_non_indexed_variants() {
        // Helper: build a 1x1 PngImage of each variant, roundtrip, check variant name
        macro_rules! check_variant {
            ($variant:ident, $pixel:expr) => {{
                let img = Image::fill(1, 1, $pixel);
                let png_img = PngImage::$variant(img);
                let bytes = encode_png_image(&png_img, &PngEncodeOptions::default()).unwrap();
                let decoded = decode(&bytes).unwrap();
                // Verify we get the same variant back
                let got = variant_name(&decoded.image);
                assert_eq!(
                    got,
                    stringify!($variant),
                    "variant mismatch for {}",
                    stringify!($variant)
                );
            }};
        }

        check_variant!(SrgbMono8, SrgbMono8::new(42));
        check_variant!(SrgbMonoA8, SrgbMonoA8::new(42, 200));
        check_variant!(Srgb8, Srgb8::new(10, 20, 30));
        check_variant!(Srgba8, Srgba8::new(10, 20, 30, 40));
        check_variant!(Mono8, Mono8::new(42));
        check_variant!(MonoA8, MonoA8::new(42, 200));
        check_variant!(Rgb8, Rgb8::new(10, 20, 30));
        check_variant!(Rgba8, Rgba8::new(10, 20, 30, 40));
        check_variant!(SrgbMono16, SrgbMono16::new(1000));
        check_variant!(SrgbMonoA16, SrgbMonoA16::new(1000, 2000));
        check_variant!(Srgb16, Srgb16::new(100, 200, 300));
        check_variant!(Srgba16, Srgba16::new(100, 200, 300, 400));
        check_variant!(Mono16, Mono16::new(1000));
        check_variant!(MonoA16, MonoA16::new(1000, 2000));
        check_variant!(Rgb16, Rgb16::new(100, 200, 300));
        check_variant!(Rgba16, Rgba16::new(100, 200, 300, 400));
    }

    /// `encode_png_image` preserves metadata options.
    #[test]
    fn encode_png_image_with_options() {
        let image = Image::fill(2, 2, Srgb8::new(10, 20, 30));
        let png_image = PngImage::Srgb8(image);
        let options = PngEncodeOptions {
            text_chunks: vec![PngTextChunk {
                keyword: "Comment".into(),
                text: "test comment".into(),
                language: None,
                translated_keyword: None,
            }],
            last_modified: Some(PngTimestamp {
                year: 2025,
                month: 6,
                day: 15,
                hour: 12,
                minute: 30,
                second: 45,
            }),
            ..PngEncodeOptions::default()
        };
        let bytes = encode_png_image(&png_image, &options).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.metadata.text_chunks.len(), 1);
        assert_eq!(decoded.metadata.text_chunks[0].keyword, "Comment");
        assert_eq!(decoded.metadata.text_chunks[0].text, "test comment");
        let ts = decoded.metadata.last_modified.unwrap();
        assert_eq!(ts.year, 2025);
        assert_eq!(ts.month, 6);
        assert_eq!(ts.day, 15);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test coverage gap: sub-byte indexed decode (1-bit, 4-bit)
    // ═══════════════════════════════════════════════════════════════════════

    /// Decode a 1-bit indexed PNG (8×1, 2-entry palette).
    /// Exercises the `(ColorType::Indexed, BitDepth::One)` arm and
    /// `unpack_sub_byte` with indexed data (no value-scaling).
    #[test]
    fn decode_indexed_1bit_sub_byte() {
        use irys_cv::image::ImageView;
        // 8 pixels packed into 1 byte: indices alternate 0 and 1.
        // 0b01010101 = pixel indices [0,1,0,1,0,1,0,1]
        let palette = [
            255u8, 0, 0, // entry 0: red
            0, 255, 0, // entry 1: green
        ];
        let data = [0x55u8]; // 0b01010101
        let png_data = encode_indexed_png(8, 1, png::BitDepth::One, &palette, None, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Indexed8 {
                data: img,
                palette: pal,
            } => {
                assert_eq!(img.width(), 8);
                assert_eq!(img.height(), 1);
                // Indices should be unpacked without scaling.
                assert_eq!(img.get(0, 0).unwrap().0, 0);
                assert_eq!(img.get(1, 0).unwrap().0, 1);
                assert_eq!(img.get(2, 0).unwrap().0, 0);
                assert_eq!(img.get(3, 0).unwrap().0, 1);
                assert_eq!(img.get(4, 0).unwrap().0, 0);
                assert_eq!(img.get(5, 0).unwrap().0, 1);
                assert_eq!(img.get(6, 0).unwrap().0, 0);
                assert_eq!(img.get(7, 0).unwrap().0, 1);
                // Palette entries
                assert_eq!(pal[0], Srgba8::new(255, 0, 0, 255));
                assert_eq!(pal[1], Srgba8::new(0, 255, 0, 255));
            }
            other => panic!("expected Indexed8, got {:?}", variant_name(other)),
        }
    }

    /// Decode a 4-bit indexed PNG (4×1, 16-entry palette).
    /// Exercises the `(ColorType::Indexed, BitDepth::Four)` arm.
    #[test]
    fn decode_indexed_4bit_sub_byte() {
        use irys_cv::image::ImageView;
        // 4 pixels at 4-bit each = 2 bytes.
        // byte 0: high nibble = index 3, low nibble = index 7  → 0x37
        // byte 1: high nibble = index 0, low nibble = index 15 → 0x0F
        let mut palette_rgb = [0u8; 16 * 3];
        for i in 0..16 {
            palette_rgb[i * 3] = (i * 17) as u8; // R
            palette_rgb[i * 3 + 1] = 0; // G
            palette_rgb[i * 3 + 2] = 0; // B
        }
        let data = [0x37u8, 0x0F];
        let png_data = encode_indexed_png(4, 1, png::BitDepth::Four, &palette_rgb, None, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Indexed8 {
                data: img,
                palette: pal,
            } => {
                assert_eq!(img.width(), 4);
                assert_eq!(img.height(), 1);
                assert_eq!(img.get(0, 0).unwrap().0, 3);
                assert_eq!(img.get(1, 0).unwrap().0, 7);
                assert_eq!(img.get(2, 0).unwrap().0, 0);
                assert_eq!(img.get(3, 0).unwrap().0, 15);
                // Spot-check palette
                assert_eq!(pal[3].r.0, 3 * 17);
                assert_eq!(pal[15].r.0, 15 * 17);
            }
            other => panic!("expected Indexed8, got {:?}", variant_name(other)),
        }
    }

    /// Decode a 2-bit indexed PNG (4×1, 4-entry palette).
    /// Exercises the `(ColorType::Indexed, BitDepth::Two)` arm.
    #[test]
    fn decode_indexed_2bit_sub_byte() {
        use irys_cv::image::ImageView;
        // 4 pixels at 2-bit each = 1 byte.
        // 0b_00_01_10_11 = 0x1B → indices [0, 1, 2, 3]
        let palette = [
            10u8, 20, 30, // entry 0
            40, 50, 60, // entry 1
            70, 80, 90, // entry 2
            100, 110, 120, // entry 3
        ];
        let data = [0x1Bu8];
        let png_data = encode_indexed_png(4, 1, png::BitDepth::Two, &palette, None, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::Indexed8 {
                data: img,
                palette: pal,
            } => {
                assert_eq!(img.width(), 4);
                assert_eq!(img.get(0, 0).unwrap().0, 0);
                assert_eq!(img.get(1, 0).unwrap().0, 1);
                assert_eq!(img.get(2, 0).unwrap().0, 2);
                assert_eq!(img.get(3, 0).unwrap().0, 3);
                assert_eq!(pal[0], Srgba8::new(10, 20, 30, 255));
                assert_eq!(pal[3], Srgba8::new(100, 110, 120, 255));
            }
            other => panic!("expected Indexed8, got {:?}", variant_name(other)),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test coverage gap: unpack_sub_byte with non-byte-aligned width
    // ═══════════════════════════════════════════════════════════════════════

    /// Decode a 1-bit grayscale PNG where width (3) is not a multiple of 8.
    /// Exercises padding-bit handling in `unpack_sub_byte`.
    #[test]
    fn decode_sub_byte_non_aligned_width() {
        use irys_cv::image::ImageView;
        // 3×1 at 1-bit: 3 pixels in 1 byte, 5 padding bits.
        // 0b_1_0_1_00000 = 0xA0 → pixel values [1, 0, 1]
        // After full-range scaling: [255, 0, 255]
        let data = [0xA0u8];
        let png_data = encode_png(3, 1, png::ColorType::Grayscale, png::BitDepth::One, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::SrgbMono8(img) => {
                assert_eq!(img.width(), 3);
                assert_eq!(img.height(), 1);
                assert_eq!(img.get(0, 0).unwrap(), SrgbMono8::new(255));
                assert_eq!(img.get(1, 0).unwrap(), SrgbMono8::new(0));
                assert_eq!(img.get(2, 0).unwrap(), SrgbMono8::new(255));
            }
            other => panic!("expected SrgbMono8, got {:?}", variant_name(other)),
        }
    }

    /// Decode a 1-bit grayscale PNG with non-aligned width and multiple rows.
    /// Each row is padded independently.
    #[test]
    fn decode_sub_byte_non_aligned_width_multirow() {
        use irys_cv::image::ImageView;
        // 3×2 at 1-bit: each row is 1 byte (3 pixels + 5 padding bits).
        // Row 0: 0b_1_1_0_00000 = 0xC0 → [255, 255, 0]
        // Row 1: 0b_0_1_1_00000 = 0x60 → [0, 255, 255]
        let data = [0xC0u8, 0x60];
        let png_data = encode_png(3, 2, png::ColorType::Grayscale, png::BitDepth::One, &data);
        let decoded = decode(&png_data).unwrap();

        match &decoded.image {
            PngImage::SrgbMono8(img) => {
                assert_eq!(img.width(), 3);
                assert_eq!(img.height(), 2);
                // Row 0
                assert_eq!(img.get(0, 0).unwrap(), SrgbMono8::new(255));
                assert_eq!(img.get(1, 0).unwrap(), SrgbMono8::new(255));
                assert_eq!(img.get(2, 0).unwrap(), SrgbMono8::new(0));
                // Row 1
                assert_eq!(img.get(0, 1).unwrap(), SrgbMono8::new(0));
                assert_eq!(img.get(1, 1).unwrap(), SrgbMono8::new(255));
                assert_eq!(img.get(2, 1).unwrap(), SrgbMono8::new(255));
            }
            other => panic!("expected SrgbMono8, got {:?}", variant_name(other)),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test coverage gap: extract_color_space — iCCP path
    // ═══════════════════════════════════════════════════════════════════════

    /// Inject a minimal iCCP chunk into a raw PNG byte stream.
    ///
    /// The profile bytes are zlib-compressed (stored, no compression).
    /// The chunk is inserted right after IHDR.
    fn inject_iccp_chunk(png_bytes: &[u8], profile_bytes: &[u8]) -> Vec<u8> {
        let ihdr_end = 8 + 4 + 4 + 13 + 4; // sig + len + type + data + crc
        assert!(png_bytes.len() > ihdr_end, "PNG too short for IHDR");

        // Build zlib-compressed profile (stored block, no compression).
        // Header: 0x78 0x01 (CMF=0x78, FLG=0x01 → (0x7801) % 31 == 0)
        // Stored deflate block: BFINAL=1 BTYPE=00
        let len = profile_bytes.len() as u16;
        let nlen = !len;
        let mut zlib = Vec::new();
        zlib.push(0x78);
        zlib.push(0x01);
        zlib.push(0x01); // BFINAL=1, BTYPE=00(stored)
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&nlen.to_le_bytes());
        zlib.extend_from_slice(profile_bytes);
        // Adler-32 of uncompressed data
        let mut s1: u32 = 1;
        let mut s2: u32 = 0;
        for &b in profile_bytes {
            s1 = (s1 + b as u32) % 65521;
            s2 = (s2 + s1) % 65521;
        }
        let adler32 = (s2 << 16) | s1;
        zlib.extend_from_slice(&adler32.to_be_bytes());

        // iCCP chunk data: keyword NUL compression_method(0) compressed_profile
        let keyword = b"TestProfile";
        let mut chunk_data = Vec::new();
        chunk_data.extend_from_slice(keyword);
        chunk_data.push(0); // NUL separator
        chunk_data.push(0); // compression method = zlib
        chunk_data.extend_from_slice(&zlib);

        let chunk_type = b"iCCP";
        let length_bytes = (chunk_data.len() as u32).to_be_bytes();

        let mut crc_input = Vec::new();
        crc_input.extend_from_slice(chunk_type);
        crc_input.extend_from_slice(&chunk_data);
        let crc_bytes = png_crc32(&crc_input).to_be_bytes();

        let mut out = Vec::with_capacity(png_bytes.len() + 12 + chunk_data.len());
        out.extend_from_slice(&png_bytes[..ihdr_end]);
        out.extend_from_slice(&length_bytes);
        out.extend_from_slice(chunk_type);
        out.extend_from_slice(&chunk_data);
        out.extend_from_slice(&crc_bytes);
        out.extend_from_slice(&png_bytes[ihdr_end..]);
        out
    }

    /// Decoding a PNG with an iCCP chunk produces `PngColorSpace::IccProfile`.
    #[test]
    fn decode_with_icc_profile() {
        let data = [128u8; 4];
        let base_png = encode_png(2, 2, png::ColorType::Grayscale, png::BitDepth::Eight, &data);
        let profile_bytes: Vec<u8> = (0..32).collect();
        let png_data = inject_iccp_chunk(&base_png, &profile_bytes);

        let decoded = decode(&png_data).unwrap();
        match &decoded.metadata.color_space {
            PngColorSpace::IccProfile { profile } => {
                assert_eq!(profile.as_ref(), &profile_bytes[..]);
            }
            other => panic!("expected IccProfile, got {:?}", other),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test coverage gap: extract_color_space — cHRM chromaticities
    // ═══════════════════════════════════════════════════════════════════════

    /// Inject a cHRM chunk into a raw PNG byte stream, right after IHDR.
    ///
    /// cHRM chunk data is 32 bytes: 8 × u32 big-endian, each scaled ×100000.
    /// Order: white_x, white_y, red_x, red_y, green_x, green_y, blue_x, blue_y.
    fn inject_chrm_chunk(
        png_bytes: &[u8],
        white: (u32, u32),
        red: (u32, u32),
        green: (u32, u32),
        blue: (u32, u32),
    ) -> Vec<u8> {
        let ihdr_end = 8 + 4 + 4 + 13 + 4;
        assert!(png_bytes.len() > ihdr_end, "PNG too short for IHDR");

        let mut chrm_data = [0u8; 32];
        chrm_data[0..4].copy_from_slice(&white.0.to_be_bytes());
        chrm_data[4..8].copy_from_slice(&white.1.to_be_bytes());
        chrm_data[8..12].copy_from_slice(&red.0.to_be_bytes());
        chrm_data[12..16].copy_from_slice(&red.1.to_be_bytes());
        chrm_data[16..20].copy_from_slice(&green.0.to_be_bytes());
        chrm_data[20..24].copy_from_slice(&green.1.to_be_bytes());
        chrm_data[24..28].copy_from_slice(&blue.0.to_be_bytes());
        chrm_data[28..32].copy_from_slice(&blue.1.to_be_bytes());

        let chunk_type = b"cHRM";
        let length_bytes = 32u32.to_be_bytes();

        let mut crc_input = Vec::with_capacity(36);
        crc_input.extend_from_slice(chunk_type);
        crc_input.extend_from_slice(&chrm_data);
        let crc_bytes = png_crc32(&crc_input).to_be_bytes();

        let mut out = Vec::with_capacity(png_bytes.len() + 44);
        out.extend_from_slice(&png_bytes[..ihdr_end]);
        out.extend_from_slice(&length_bytes);
        out.extend_from_slice(chunk_type);
        out.extend_from_slice(&chrm_data);
        out.extend_from_slice(&crc_bytes);
        out.extend_from_slice(&png_bytes[ihdr_end..]);
        out
    }

    /// Decoding a PNG with gAMA + cHRM produces `PngColorSpace::Gamma`
    /// with populated chromaticities.
    #[test]
    fn decode_with_chrm_chunk() {
        let data = [100u8; 4];
        // Encode with gAMA = 0.45455 (sRGB-ish gamma) so we hit the Gamma path.
        let base_png = encode_png_with_gamma(
            2,
            2,
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            1.0 / 2.2,
            &data,
        );
        // Inject cHRM with sRGB primaries (×100000 fixed-point).
        let png_data = inject_chrm_chunk(
            &base_png,
            (31270, 32900), // white
            (64000, 33000), // red
            (30000, 60000), // green
            (15000, 6000),  // blue
        );
        let decoded = decode(&png_data).unwrap();
        match &decoded.metadata.color_space {
            PngColorSpace::Gamma {
                gamma,
                chromaticities,
            } => {
                // Gamma should be approximately 1/2.2 ≈ 0.45455
                assert!((*gamma - (1.0 / 2.2)).abs() < 0.001, "gamma = {}", gamma);
                let c = chromaticities
                    .as_ref()
                    .expect("chromaticities should be Some");
                // White point: 0.3127, 0.3290
                assert!((c[0] - 0.3127).abs() < 0.001, "white_x = {}", c[0]);
                assert!((c[1] - 0.3290).abs() < 0.001, "white_y = {}", c[1]);
                // Red: 0.64, 0.33
                assert!((c[2] - 0.64).abs() < 0.001, "red_x = {}", c[2]);
                assert!((c[3] - 0.33).abs() < 0.001, "red_y = {}", c[3]);
            }
            other => panic!("expected Gamma with chromaticities, got {:?}", other),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test coverage gap: extract_text_chunks — zTXt
    // ═══════════════════════════════════════════════════════════════════════

    /// Decoding a PNG with a zTXt (compressed Latin-1) text chunk.
    #[test]
    fn decode_ztxt_text_chunk() {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, 1, 1);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .add_ztxt_chunk("Comment".into(), "compressed text data".into())
                .unwrap();
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[0u8]).unwrap();
        }

        let decoded = decode(&out).unwrap();
        let ztxt = decoded
            .metadata
            .text_chunks
            .iter()
            .find(|c| c.keyword == "Comment")
            .expect("zTXt chunk not found");
        assert_eq!(ztxt.text, "compressed text data");
        assert!(ztxt.language.is_none(), "zTXt should not have language");
        assert!(
            ztxt.translated_keyword.is_none(),
            "zTXt should not have translated_keyword"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test coverage gap: extract_text_chunks — iTXt with language tag
    // ═══════════════════════════════════════════════════════════════════════

    /// Decoding a PNG with an iTXt chunk that has language + translated keyword.
    #[test]
    fn decode_itxt_text_chunk() {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, 1, 1);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();

            // Write a fully-populated iTXt chunk via write_text_chunk.
            let mut itxt = png::text_metadata::ITXtChunk::new(
                "Description".to_string(),
                "hello world".to_string(),
            );
            itxt.language_tag = "en".to_string();
            itxt.translated_keyword = "Beschreibung".to_string();
            writer.write_text_chunk(&itxt).unwrap();

            writer.write_image_data(&[0u8]).unwrap();
        }

        let decoded = decode(&out).unwrap();
        let chunk = decoded
            .metadata
            .text_chunks
            .iter()
            .find(|c| c.keyword == "Description")
            .expect("iTXt chunk not found");
        assert_eq!(chunk.text, "hello world");
        assert_eq!(chunk.language.as_deref(), Some("en"));
        assert_eq!(chunk.translated_keyword.as_deref(), Some("Beschreibung"));
    }
}
