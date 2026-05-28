//! Image I/O for irys-cv — feature-gated codec support.
//!
//! # Architecture
//!
//! - **Per-codec exhaustive enums** — each codec defines its own output enum
//!   (e.g. [`png::PngImage`]) whose variants carry concrete `Image<P>`
//!   values.  These enums are the primary API for users who know the format.
//!
//! - **Per-codec decoded structs** — each codec returns a `#[non_exhaustive]`
//!   struct (e.g. [`png::PngDecoded`]) containing both the pixel data and
//!   ancillary metadata.  Metadata construction is essentially free because
//!   codec crates parse ancillary chunks during decoding regardless.
//!
//! - **Top-level convenience** — [`DecodedImage`] is a `#[non_exhaustive]`
//!   union of all per-codec decoded structs, returned by [`load`] for users
//!   who want format-agnostic decoding.
//!
//! - **Feature-gated codecs** — one Cargo feature per codec, none enabled by
//!   default.  The core crate stays dependency-free.
//!
//! - **Encoding is generic** — encode paths accept
//!   `&(impl ImageView<Pixel = P> + PlainImage)` where `P: PlainPixel`,
//!   so no variant dispatch is needed.  Only decoding returns variant enums.

mod error;

#[cfg(feature = "png")]
pub mod png;

#[cfg(feature = "jpeg")]
pub mod jpeg;

#[cfg(feature = "bmp")]
pub mod bmp;

pub use error::IoError;

// ═══════════════════════════════════════════════════════════════════════════════
// ImageFormat — format detection
// ═══════════════════════════════════════════════════════════════════════════════

/// Detected image format based on magic bytes.
///
/// Used by [`detect_format`] and internally by [`load`] to dispatch to the
/// correct codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    Bmp,
}

/// Detect the image format by inspecting the leading magic bytes.
///
/// Returns `None` if the signature doesn't match any known format.
/// This never reads more than the first 16 bytes.
///
/// | Format | Signature                            |
/// |--------|--------------------------------------|
/// | PNG    | `89 50 4E 47 0D 0A 1A 0A`           |
/// | JPEG   | `FF D8 FF`                           |
/// | BMP    | `42 4D`                              |
///
/// # Examples
///
/// ```
/// use irys_cv_io::{detect_format, ImageFormat};
///
/// let png_sig = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
/// assert_eq!(detect_format(&png_sig), Some(ImageFormat::Png));
///
/// assert_eq!(detect_format(&[0x00, 0x00]), None);
/// ```
pub fn detect_format(bytes: &[u8]) -> Option<ImageFormat> {
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        Some(ImageFormat::Png)
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(ImageFormat::Jpeg)
    } else if bytes.starts_with(&[0x42, 0x4D]) {
        Some(ImageFormat::Bmp)
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DecodedImage — top-level union of all codecs
// ═══════════════════════════════════════════════════════════════════════════════

/// Universal decoded image — union of all per-codec decoded structs.
///
/// This enum is `#[non_exhaustive]` because new codecs (e.g. WebP) may be
/// added over time.  Users should include a wildcard arm when matching.
///
/// Each variant wraps the per-codec decoded struct (e.g. [`png::PngDecoded`]),
/// which itself contains the pixel data and ancillary metadata.  This means
/// `load()` always gives you metadata for free — no separate "with metadata"
/// function needed.
///
/// Per-codec pixel enums (e.g. [`png::PngImage`]) are **not**
/// `#[non_exhaustive]` — they are exhaustive spec sheets.  If you know the
/// format, prefer decoding with the per-codec API directly; you get
/// exhaustive matching and avoid the wildcard arm.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv_io::{load, DecodedImage};
/// let bytes = std::fs::read("image.png").unwrap();
/// match load(&bytes).unwrap() {
///     #[cfg(feature = "png")]
///     DecodedImage::Png(decoded) => {
///         // `decoded.image` is a `PngImage` — match exhaustively
///         // `decoded.metadata` carries colour-space info, text chunks, etc.
///     }
///     _ => { /* unknown or unsupported codec */ }
/// }
/// ```
#[non_exhaustive]
pub enum DecodedImage {
    /// A decoded PNG file — pixel data + metadata.
    /// Only available with the `png` feature.
    #[cfg(feature = "png")]
    Png(png::PngDecoded),
    /// A decoded JPEG file — pixel data + metadata.
    /// Only available with the `jpeg` feature.
    #[cfg(feature = "jpeg")]
    Jpeg(jpeg::JpegDecoded),
    /// A decoded BMP file — pixel data + metadata.
    /// Only available with the `bmp` feature.
    #[cfg(feature = "bmp")]
    Bmp(bmp::BmpDecoded),
}

// ═══════════════════════════════════════════════════════════════════════════════
// load — format-agnostic convenience entry point
// ═══════════════════════════════════════════════════════════════════════════════

/// Load an image from an in-memory byte slice, auto-detecting the format.
///
/// Inspects the leading bytes with [`detect_format`], then dispatches to the
/// appropriate per-codec decoder.  Returns a [`DecodedImage`] wrapping the
/// per-codec result (pixels + metadata).
///
/// # Errors
///
/// - [`IoError::InvalidFormat`] if the magic bytes don't match any known
///   (and enabled) format.
/// - Any error the underlying codec decoder can produce.
///
/// # Examples
///
/// ```no_run
/// # use irys_cv_io::{load, DecodedImage};
/// let bytes = std::fs::read("photo.png").unwrap();
/// match load(&bytes).unwrap() {
///     #[cfg(feature = "png")]
///     DecodedImage::Png(decoded) => {
///         use irys_cv_io::png::PngImage;
///         match decoded.image {
///             PngImage::Srgb8(image) => { /* Image<Srgb8> */ }
///             _ => {}
///         }
///     }
///     _ => {}
/// }
/// ```
pub fn load(bytes: &[u8]) -> Result<DecodedImage, IoError> {
    match detect_format(bytes) {
        #[cfg(feature = "png")]
        Some(ImageFormat::Png) => Ok(DecodedImage::Png(png::decode(bytes)?)),
        #[cfg(feature = "jpeg")]
        Some(ImageFormat::Jpeg) => Ok(DecodedImage::Jpeg(jpeg::decode(bytes)?)),
        #[cfg(feature = "bmp")]
        Some(ImageFormat::Bmp) => Ok(DecodedImage::Bmp(bmp::decode(bytes)?)),
        // Reachable iff a feature is disabled: e.g. with `--features jpeg`,
        // a PNG signature reaches this arm because the PNG case is gated
        // out above. With `--features all-codecs` every codec is enabled
        // and the compiler reports this arm unreachable — the allow is
        // necessary for the feature-disabled cases.
        #[allow(unreachable_patterns)]
        Some(_) => Err(IoError::InvalidFormat {
            reason: "detected format is not supported (enable the corresponding feature)",
        }),
        None => Err(IoError::InvalidFormat {
            reason: "unrecognised image format (magic bytes don't match any known codec)",
        }),
    }
}

/// Load an image from a streaming reader, auto-detecting the format.
///
/// Reads enough leading bytes to detect the format, then dispatches to the
/// appropriate per-codec streaming decoder.
///
/// # Errors
///
/// Same as [`load`], plus [`IoError::Io`] for read failures.
pub fn load_reader(mut reader: impl std::io::Read) -> Result<DecodedImage, IoError> {
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    load(&buf)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use irys_cv::image::ImageView;

    // ── detect_format — positive tests per format ────────────────────────

    #[test]
    fn detect_format_png() {
        let sig = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(detect_format(&sig), Some(ImageFormat::Png));
    }

    #[test]
    fn detect_format_png_with_trailing_data() {
        let mut data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        data.extend_from_slice(&[0x00; 100]);
        assert_eq!(detect_format(&data), Some(ImageFormat::Png));
    }

    #[test]
    fn detect_format_jpeg() {
        let sig = [0xFF, 0xD8, 0xFF];
        assert_eq!(detect_format(&sig), Some(ImageFormat::Jpeg));
    }

    #[test]
    fn detect_format_jpeg_with_trailing_data() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        assert_eq!(detect_format(&data), Some(ImageFormat::Jpeg));
    }

    #[test]
    fn detect_format_tiff_le_is_no_longer_recognised() {
        // P1-8: TIFF support was removed. The signature must now return
        // None so callers don't see a phantom variant they can't decode.
        let sig = [0x49, 0x49, 0x2A, 0x00];
        assert_eq!(detect_format(&sig), None);
    }

    #[test]
    fn detect_format_tiff_be_is_no_longer_recognised() {
        let sig = [0x4D, 0x4D, 0x00, 0x2A];
        assert_eq!(detect_format(&sig), None);
    }

    #[test]
    fn detect_format_bmp() {
        let sig = [0x42, 0x4D];
        assert_eq!(detect_format(&sig), Some(ImageFormat::Bmp));
    }

    #[test]
    fn detect_format_bmp_with_trailing_data() {
        let data = [0x42, 0x4D, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(detect_format(&data), Some(ImageFormat::Bmp));
    }

    // ── detect_format — negative / edge cases ────────────────────────────

    #[test]
    fn detect_format_empty() {
        assert_eq!(detect_format(&[]), None);
    }

    #[test]
    fn detect_format_single_byte() {
        assert_eq!(detect_format(&[0x89]), None);
    }

    #[test]
    fn detect_format_unknown_signature() {
        assert_eq!(detect_format(&[0x00, 0x00, 0x00, 0x00]), None);
    }

    #[test]
    fn detect_format_short_for_png() {
        // First 7 of 8 PNG signature bytes — should not match.
        let short = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A];
        assert_eq!(detect_format(&short), None);
    }

    #[test]
    fn detect_format_near_miss_png() {
        // Correct first 4 bytes but wrong continuation.
        let near = [0x89, 0x50, 0x4E, 0x47, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(detect_format(&near), None);
    }

    #[test]
    fn detect_format_short_for_jpeg() {
        // Only 2 of 3 JPEG signature bytes.
        assert_eq!(detect_format(&[0xFF, 0xD8]), None);
    }

    #[test]
    fn detect_format_short_for_legacy_tiff_le_signature() {
        // Even the full 4-byte TIFF LE signature is no longer recognised
        // (P1-8). Shorter prefixes were never recognised.
        assert_eq!(detect_format(&[0x49, 0x49, 0x2A]), None);
    }

    // ── ImageFormat — trait coverage ─────────────────────────────────────

    #[test]
    fn image_format_debug_and_eq() {
        let fmt = ImageFormat::Png;
        let dbg = format!("{:?}", fmt);
        assert_eq!(dbg, "Png");
        assert_eq!(fmt, fmt.clone());
        assert_ne!(ImageFormat::Png, ImageFormat::Jpeg);

        // Cover all variants in Debug.
        let _ = format!("{:?}", ImageFormat::Jpeg);
        let _ = format!("{:?}", ImageFormat::Bmp);
    }

    // ── detect_format — priority: first matching format wins ─────────────

    #[test]
    fn detect_format_bmp_prefix_not_confused_with_longer() {
        // BMP signature is only 2 bytes — make sure it doesn't false-match
        // when followed by arbitrary trailing bytes (here, the bytes that
        // used to be a TIFF LE signature before P1-8).
        let data = [0x42, 0x4D, 0x49, 0x49, 0x2A, 0x00];
        assert_eq!(detect_format(&data), Some(ImageFormat::Bmp));
    }

    // ── load / load_reader ───────────────────────────────────────────────

    #[cfg(feature = "jpeg")]
    #[test]
    fn load_jpeg_dispatches_correctly() {
        // Build a minimal JPEG via the codec, then load() it.
        use irys_cv::image::Image;
        use irys_cv::pixel::Srgb8;
        let img = Image::fill(4, 4, Srgb8::new(100, 150, 200));
        let bytes = jpeg::encode(&img, &jpeg::JpegEncodeOptions::default()).unwrap();
        let decoded = load(&bytes).unwrap();
        match decoded {
            DecodedImage::Jpeg(d) => match &d.image {
                jpeg::JpegImage::Srgb8(img) => {
                    assert_eq!(img.width(), 4);
                    assert_eq!(img.height(), 4);
                }
                other => panic!("expected Srgb8, got {:?}", other),
            },
            #[allow(unreachable_patterns)]
            _ => panic!("expected DecodedImage::Jpeg"),
        }
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn load_reader_jpeg_dispatches_correctly() {
        use irys_cv::image::Image;
        use irys_cv::pixel::SrgbMono8;
        let img = Image::fill(8, 6, SrgbMono8::new(128));
        let bytes = jpeg::encode(&img, &jpeg::JpegEncodeOptions::default()).unwrap();
        let decoded = load_reader(std::io::Cursor::new(&bytes)).unwrap();
        match decoded {
            DecodedImage::Jpeg(d) => match &d.image {
                jpeg::JpegImage::SrgbMono8(img) => {
                    assert_eq!(img.width(), 8);
                    assert_eq!(img.height(), 6);
                }
                other => panic!("expected SrgbMono8, got {:?}", other),
            },
            #[allow(unreachable_patterns)]
            _ => panic!("expected DecodedImage::Jpeg"),
        }
    }

    #[test]
    fn load_unknown_format_returns_error() {
        let result = load(&[0x00, 0x00, 0x00, 0x00]);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn load_empty_returns_error() {
        let result = load(&[]);
        assert!(matches!(result, Err(IoError::InvalidFormat { .. })));
    }

    #[test]
    fn load_unsupported_format_returns_error() {
        // What used to be a TIFF magic prefix is no longer recognised at all
        // (P1-8). `load` must surface an `InvalidFormat` error rather than
        // silently picking the wrong codec.
        let tiff_le = [0x49, 0x49, 0x2A, 0x00];
        let result = load(&tiff_le);
        assert!(matches!(result, Err(crate::IoError::InvalidFormat { .. })));
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn load_jpeg_returns_metadata() {
        use irys_cv::image::Image;
        use irys_cv::pixel::Srgb8;
        let img = Image::fill(2, 2, Srgb8::new(0, 0, 0));
        let bytes = jpeg::encode(&img, &jpeg::JpegEncodeOptions::default()).unwrap();
        let decoded = load(&bytes).unwrap();
        match decoded {
            DecodedImage::Jpeg(d) => {
                assert_eq!(d.metadata.source_bit_depth, jpeg::JpegBitDepth::Eight);
                assert_eq!(d.metadata.color_space, jpeg::JpegColorSpace::Srgb);
            }
            #[allow(unreachable_patterns)]
            _ => panic!("expected DecodedImage::Jpeg"),
        }
    }

    #[cfg(feature = "bmp")]
    #[test]
    fn load_bmp_dispatches_correctly() {
        use irys_cv::image::Image;
        use irys_cv::pixel::Srgb8;
        let img = Image::fill(4, 4, Srgb8::new(100, 150, 200));
        let bytes = bmp::encode(&img, &bmp::BmpEncodeOptions::default()).unwrap();
        let decoded = load(&bytes).unwrap();
        match decoded {
            DecodedImage::Bmp(d) => match &d.image {
                bmp::BmpImage::Srgb8(img) => {
                    assert_eq!(img.width(), 4);
                    assert_eq!(img.height(), 4);
                }
                other => panic!("expected Srgb8, got {:?}", other),
            },
            #[allow(unreachable_patterns)]
            _ => panic!("expected DecodedImage::Bmp"),
        }
    }

    #[cfg(feature = "bmp")]
    #[test]
    fn load_reader_bmp_dispatches_correctly() {
        use irys_cv::image::Image;
        use irys_cv::pixel::Srgb8;
        let img = Image::fill(3, 2, Srgb8::new(42, 84, 126));
        let bytes = bmp::encode(&img, &bmp::BmpEncodeOptions::default()).unwrap();
        let decoded = load_reader(std::io::Cursor::new(&bytes)).unwrap();
        match decoded {
            DecodedImage::Bmp(d) => match &d.image {
                bmp::BmpImage::Srgb8(img) => {
                    assert_eq!(img.width(), 3);
                    assert_eq!(img.height(), 2);
                }
                other => panic!("expected Srgb8, got {:?}", other),
            },
            #[allow(unreachable_patterns)]
            _ => panic!("expected DecodedImage::Bmp"),
        }
    }

    #[cfg(feature = "bmp")]
    #[test]
    fn load_bmp_returns_metadata() {
        use irys_cv::image::Image;
        use irys_cv::pixel::Srgb8;
        let img = Image::fill(2, 2, Srgb8::new(0, 0, 0));
        let bytes = bmp::encode(&img, &bmp::BmpEncodeOptions::default()).unwrap();
        let decoded = load(&bytes).unwrap();
        match decoded {
            DecodedImage::Bmp(d) => {
                assert_eq!(d.metadata.source_bit_depth, bmp::BmpBitDepth::TwentyFour);
                assert_eq!(d.metadata.color_space, bmp::BmpColorSpace::Srgb);
                assert_eq!(d.metadata.header_version, bmp::BmpHeaderVersion::Info);
                assert_eq!(d.metadata.compression, bmp::BmpCompression::None);
            }
            #[allow(unreachable_patterns)]
            _ => panic!("expected DecodedImage::Bmp"),
        }
    }
}
