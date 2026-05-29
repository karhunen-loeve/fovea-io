# fovea-io

[![Crates.io](https://img.shields.io/crates/v/fovea-io.svg)](https://crates.io/crates/fovea-io)
[![Documentation](https://docs.rs/fovea-io/badge.svg)](https://docs.rs/fovea-io)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`fovea-io` adds feature-gated PNG, JPEG, and BMP codecs for [`fovea`](https://github.com/karhunen-loeve/fovea) images.

```toml
[dependencies]
fovea = "0.1.0"
fovea-io = { version = "0.1.0", features = ["png"] }
```

## Features

Each codec is behind its own Cargo feature flag. No codecs are enabled by default.

| Feature | Codecs enabled | Dependencies |
|---|---|---|
| `png` | PNG decode + encode | [`png`](https://crates.io/crates/png) |
| `jpeg` | JPEG decode + encode | [`jpeg-decoder`](https://crates.io/crates/jpeg-decoder), [`jpeg-encoder`](https://crates.io/crates/jpeg-encoder) |
| `bmp` | BMP decode + encode | none beyond `fovea-io` |
| `all-codecs` | PNG, JPEG, and BMP | all optional codec dependencies |

Enable features in your `Cargo.toml`:

```toml
[dependencies]
fovea-io = { version = "0.1.0", features = ["jpeg"] }
# or enable everything:
# fovea-io = { version = "0.1.0", features = ["all-codecs"] }
```

## Quick start

Feature-free format detection works without enabling a codec:

```rust
use fovea_io::{detect_format, ImageFormat};

let png_sig = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
assert_eq!(detect_format(&png_sig), Some(ImageFormat::Png));
```

With codec features enabled, decode into per-format enums whose variants carry concrete typed images:

```rust,ignore
use fovea_io::{load, DecodedImage};

let bytes = std::fs::read("image.jpg").unwrap();
match load(&bytes).unwrap() {
    DecodedImage::Jpeg(decoded) => {
        println!("JPEG {}x{}", decoded.image.width(), decoded.image.height());
    }
    DecodedImage::Png(decoded) => {
        println!("PNG {}x{}", decoded.image.width(), decoded.image.height());
    }
    DecodedImage::Bmp(decoded) => {
        println!("BMP {}x{}", decoded.image.width(), decoded.image.height());
    }
    _ => {}
}
```

## Codec examples

### JPEG

```rust,ignore
use fovea::image::Image;
use fovea::pixel::Srgb8;
use fovea_io::jpeg::{self, JpegEncodeOptions, JpegImage};

let bytes = std::fs::read("photo.jpg").unwrap();
let decoded = jpeg::decode(&bytes).unwrap();

match decoded.image {
    JpegImage::Srgb8(image) => println!("RGB JPEG: {}x{}", image.width(), image.height()),
    JpegImage::SrgbMono8(image) => println!("grayscale JPEG: {}x{}", image.width(), image.height()),
    JpegImage::SrgbMono16(image) => println!("extended grayscale JPEG: {}x{}", image.width(), image.height()),
}

let image = Image::fill(320, 240, Srgb8::new(128, 64, 32));
let bytes = jpeg::encode(&image, &JpegEncodeOptions::default()).unwrap();
std::fs::write("output.jpg", bytes).unwrap();
```

### PNG

```rust,ignore
use fovea_io::png::{self, PngImage};

let bytes = std::fs::read("image.png").unwrap();
let decoded = png::decode(&bytes).unwrap();

match decoded.image {
    PngImage::Srgb8(image) => println!("8-bit sRGB PNG: {}x{}", image.width(), image.height()),
    PngImage::Srgba8(image) => println!("8-bit sRGBA PNG: {}x{}", image.width(), image.height()),
    _ => println!("another PNG pixel format"),
}
```

### BMP

```rust,ignore
use fovea::image::Image;
use fovea::pixel::Srgb8;
use fovea_io::bmp::{self, BmpEncodeOptions};

let image = Image::fill(320, 240, Srgb8::new(0, 0, 0));
let bytes = bmp::encode(&image, &BmpEncodeOptions::default()).unwrap();
std::fs::write("output.bmp", bytes).unwrap();
```

## Architecture

- **Per-codec exhaustive enums** — each codec defines its own output enum, such as `PngImage`, `JpegImage`, or `BmpImage`, whose variants carry concrete `Image<P>` values.
- **Metadata travels with pixels** — decoded structs return pixel data and format-specific metadata together.
- **Feature-gated codecs** — enabling one codec does not pull in the dependencies for another.
- **Sealed encode traits** — `PngPixel`, `JpegPixel`, and `BmpPixel` make unsupported output formats compile-time errors.

## Design principle references

`§N` references throughout this crate refer to the numbered design principles listed in the [`fovea` crate documentation](https://docs.rs/fovea).

## Part of the fovea project

- Core crate: [`fovea`](https://github.com/karhunen-loeve/fovea)
- Display support: [`fovea-display`](https://github.com/karhunen-loeve/fovea-display)
- End-to-end demos: [`fovea-examples`](https://github.com/karhunen-loeve/fovea-examples)

## License

`fovea-io` itself is licensed under the [MIT License](LICENSE).

When the `jpeg` feature is enabled, this crate depends on [`jpeg-encoder`](https://crates.io/crates/jpeg-encoder), which carries an additional **IJG (Independent JPEG Group)** license requiring this acknowledgement:

> This software is based in part on the work of the Independent JPEG Group.

See [THIRD-PARTY-LICENSES.txt](THIRD-PARTY-LICENSES.txt) for full license texts of dependencies and attribution requirements.
