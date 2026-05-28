# irys-cv-io

Image I/O for [irys-cv](../irys-cv) — feature-gated codec support.

## Features

Each codec is behind its own Cargo feature flag. None are enabled by default,
keeping the core dependency-free.

| Feature      | Codecs enabled        | Dependencies                       |
|--------------|-----------------------|------------------------------------|
| `png`        | PNG decode + encode   | [`png`](https://crates.io/crates/png) |
| `jpeg`       | JPEG decode + encode  | [`jpeg-decoder`](https://crates.io/crates/jpeg-decoder), [`jpeg-encoder`](https://crates.io/crates/jpeg-encoder) |
| `all-codecs` | All of the above      | all of the above                   |

Enable features in your `Cargo.toml`:

```toml
[dependencies]
irys-cv-io = { version = "0.1", features = ["jpeg"] }
# or enable everything:
# irys-cv-io = { version = "0.1", features = ["all-codecs"] }
```

## Quick Start

### Format-agnostic loading

```rust,no_run
use irys_cv_io::{load, DecodedImage};

let bytes = std::fs::read("image.jpg").unwrap();
match load(&bytes).unwrap() {
    #[cfg(feature = "jpeg")]
    DecodedImage::Jpeg(decoded) => {
        println!("JPEG {}x{}", decoded.image.width(), decoded.image.height());
    }
    #[cfg(feature = "png")]
    DecodedImage::Png(_decoded) => {
        println!("PNG");
    }
    _ => {}
}
```

### JPEG decoding

```rust,no_run
use irys_cv_io::jpeg::{self, JpegImage};

let bytes = std::fs::read("photo.jpg").unwrap();
let decoded = jpeg::decode(&bytes).unwrap();

// Exhaustive match — all possible JPEG pixel formats:
match decoded.image {
    JpegImage::Srgb8(image) => {
        // image is Image<Srgb8> — the common case
    }
    JpegImage::SrgbMono8(image) => {
        // image is Image<SrgbMono8> — grayscale
    }
    JpegImage::SrgbMono16(image) => {
        // image is Image<SrgbMono16> — 12-bit extended JPEG
    }
}

// Metadata is always available:
if let Some(exif) = &decoded.metadata.exif {
    if let Some(orientation) = exif.orientation {
        println!("EXIF orientation: {orientation}");
    }
    if let Some(lat) = exif.gps_latitude {
        println!("GPS latitude: {lat}°");
    }
}
```

### JPEG encoding

```rust,no_run
use irys_cv::image::Image;
use irys_cv::pixel::Srgb8;
use irys_cv_io::jpeg::{self, JpegEncodeOptions, JpegSamplingFactor};

let image = Image::fill(320, 240, Srgb8::new(128, 64, 32));

// Encode with defaults (quality 85, baseline):
let bytes = jpeg::encode(&image, &JpegEncodeOptions::default()).unwrap();

// Encode with custom options:
let mut opts = JpegEncodeOptions::default();
opts.quality = 95;
opts.sampling_factor = Some(JpegSamplingFactor::F1x1); // 4:4:4
opts.progressive = true;
let bytes = jpeg::encode(&image, &opts).unwrap();

std::fs::write("output.jpg", bytes).unwrap();
```

### PNG decoding

```rust,no_run
use irys_cv_io::png::{self, PngImage};

let bytes = std::fs::read("image.png").unwrap();
let decoded = png::decode(&bytes).unwrap();

match decoded.image {
    PngImage::Srgb8(image) => { /* Image<Srgb8> */ }
    PngImage::Srgba8(image) => { /* Image<Srgba8> */ }
    _ => { /* other PNG pixel formats */ }
}
```

## Architecture

- **Per-codec exhaustive enums** — each codec defines its own output enum
  (e.g. `JpegImage`, `PngImage`) whose variants carry concrete `Image<P>`
  values. These enums are the primary API for users who know the format.

- **Per-codec decoded structs** — each codec returns a `#[non_exhaustive]`
  struct (e.g. `JpegDecoded`) containing both the pixel data and ancillary
  metadata. Metadata construction is essentially free because codec crates
  parse ancillary chunks during decoding regardless.

- **Top-level convenience** — `DecodedImage` is a `#[non_exhaustive]`
  union of all per-codec decoded structs, returned by `load()` for users
  who want format-agnostic decoding.

- **Sealed encode traits** — encode paths use sealed traits (`JpegPixel`,
  `PngPixel`) so only valid pixel types can be encoded. Attempting to
  encode an invalid type (e.g. `Srgba8` to JPEG) is a compile-time error.

## Design Notes

### JPEG is always sRGB

JPEG is defined as sRGB by JFIF 1.02. All decoded pixel types use sRGB
variants (`Srgb8`, `SrgbMono8`). Returning linear types would be a
type-level lie — the compiler would allow linear-math operations on
gamma-encoded data, producing silently wrong results.

### No CMYK support

CMYK → sRGB conversion requires an ICC profile and a colour management
engine. Naive conversion is incorrect for most real-world CMYK profiles.
CMYK JPEGs are rejected with `IoError::UnsupportedFeature`.

### EXIF orientation is not auto-applied

EXIF orientation is metadata, not pixel data. Auto-applying it would mean
the returned image silently differs from the raw JPEG samples — violating
the explicit data layout principle.

### No silent alpha stripping

`Srgba8` cannot be encoded to JPEG (compile-time error). JPEG has no alpha
channel, so discarding it must be an explicit user decision.

## License

`irys-cv-io` itself is licensed under **MIT** — see [LICENSE](../LICENSE) for details.

When the `jpeg` feature is enabled, this crate depends on
[`jpeg-encoder`](https://crates.io/crates/jpeg-encoder) which carries an
additional **IJG (Independent JPEG Group)** license requiring the following
acknowledgement:

> This software is based in part on the work of the Independent JPEG Group.

See [THIRD-PARTY-LICENSES.txt](./THIRD-PARTY-LICENSES.txt) for full license texts of
all dependencies and their attribution requirements.