# fovea-io

[![Crates.io](https://img.shields.io/crates/v/fovea-io.svg)](https://crates.io/crates/fovea-io)
[![Documentation](https://docs.rs/fovea-io/badge.svg)](https://docs.rs/fovea-io)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/karhunen-loeve/fovea-io/blob/main/LICENSE)

`fovea-io` is the file boundary for fovea: decode PNG, JPEG, and BMP bytes into typed `fovea::Image<P>` values, then let the core crate enforce pixel semantics from there.

The key idea is simple: I/O tells you what the file contains. It does not silently decide how to process it. Metadata travels with pixels, color conversions are explicit, and unsupported encodes are compile-time errors where possible.

## Install

No codecs are enabled by default. Enable only the formats you need:

```sh
cargo add fovea
cargo add fovea-io --features png
```

Enable every current codec with:

```sh
cargo add fovea-io --features all-codecs
```

## Features

| Feature | Enables | Pulls in |
|---|---|---|
| `png` | PNG decode + encode | `png` |
| `jpeg` | JPEG decode + encode | `jpeg-decoder`, `jpeg-encoder` |
| `bmp` | BMP decode + encode | no external codec crate |
| `all-codecs` | PNG, JPEG, and BMP | all optional codec dependencies |

## The fovea I/O model

```text
bytes → detect/decode → per-codec enum → concrete Image<P> → explicit transforms → encode
```

`load` is convenient when you do not know the file format. Per-codec APIs are better when you do know the format, because their image enums are exhaustive spec sheets.

```rust
use fovea_io::{ImageFormat, detect_format};

let png_sig = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
assert_eq!(detect_format(&png_sig), Some(ImageFormat::Png));
assert_eq!(detect_format(&[0x00, 0x00]), None);
```

## Decode once, then work with types

```rust,ignore
use fovea::image::ImageView;
use fovea_io::png::{self, PngImage};

let bytes = std::fs::read("input.png")?;
let decoded = png::decode(&bytes)?;

match decoded.image {
    PngImage::Srgb8(img) => {
        println!("8-bit sRGB PNG: {}x{}", img.width(), img.height());
        // img is Image<Srgb8>
    }
    PngImage::Srgba8(img) => {
        println!("8-bit sRGBA PNG: {}x{}", img.width(), img.height());
        // img is Image<Srgba8>
    }
    other => {
        println!("supported PNG, different pixel format: {other:?}");
    }
}
```

After the match, downstream code has concrete pixel types. A resize pipeline can reject gamma-incorrect interpolation; a display path can require display-ready pixels; a camera pipeline can keep raw linear data linear.

## Decode → transform → encode

This is the common shape for file tools:

```rust,ignore
use fovea::Size;
use fovea::image::Image;
use fovea::pixel::{RgbF32, Srgb8};
use fovea::transform::{Bilinear, SrgbGamma, convert_image, resize};
use fovea_io::png::{self, PngEncodeOptions, PngImage};

let bytes = std::fs::read("photo.png")?;
let decoded = png::decode(&bytes)?;

let srgb: Image<Srgb8> = match decoded.image {
    PngImage::Srgb8(img) => img,
    other => return Err(format!("expected Srgb8 PNG, got {other:?}").into()),
};

// Bilinear resize is done in linear light.
let linear: Image<RgbF32> = convert_image(&srgb, SrgbGamma);
let resized: Image<RgbF32> = resize(&linear, Size::new(800, 600), Bilinear);
let output: Image<Srgb8> = convert_image(&resized, SrgbGamma);

let out_bytes = png::encode(&output, &PngEncodeOptions::default())?;
std::fs::write("photo_800x600.png", out_bytes)?;
```

The I/O crate does not hide the `SrgbGamma` step. That is deliberate: decoding surfaces information; your pipeline chooses what to do with it.

## Which entry point?

| You know... | Use | Why |
|---|---|---|
| Nothing except "these are image bytes" | `load` / `load_reader` | Detects PNG/JPEG/BMP by magic bytes and returns `DecodedImage`. |
| The file is PNG | `png::decode` / `png::encode` | Exhaustive PNG pixel enum and PNG metadata. |
| The file is JPEG | `jpeg::decode` / `jpeg::encode` | JPEG-specific metadata and supported JPEG pixel shapes. |
| The file is BMP | `bmp::decode` / `bmp::encode` | BMP-specific representation without pulling in PNG/JPEG dependencies. |
| Only the signature matters | `detect_format` | Cheap format detection without codec features. |

## Metadata is not policy

Decoded values carry metadata alongside pixel data. fovea-io reports what the file said; it does not silently linearize, premultiply alpha, apply profiles, or drop ancillary data on your behalf.

That policy keeps I/O predictable:

- The decode boundary preserves file facts.
- The transform boundary names semantic changes.
- The encode boundary checks whether the target format can represent the requested pixel type.

## Crate ecosystem

| Crate | Purpose |
|---|---|
| `fovea` | Core image types, pixels, transforms, and analysis. |
| `fovea-io` | File codecs for typed fovea images. |
| `fovea-display` | Display strategies and debug windows. |
| `fovea-examples` | Repo-only end-to-end programs using all crates together. |

## License

`fovea-io` itself is licensed under the [MIT License](https://github.com/karhunen-loeve/fovea-io/blob/main/LICENSE).

When the `jpeg` feature is enabled, this crate depends on [`jpeg-encoder`](https://crates.io/crates/jpeg-encoder), which carries an additional **IJG (Independent JPEG Group)** license requiring this acknowledgement:

> This software is based in part on the work of the Independent JPEG Group.

See [THIRD-PARTY-LICENSES.txt](https://github.com/karhunen-loeve/fovea-io/blob/main/THIRD-PARTY-LICENSES.txt) for full license texts of dependencies and attribution requirements.
