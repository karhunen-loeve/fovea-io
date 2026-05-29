# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-05-29

First real public release. `0.1.0` was a name-reservation placeholder.

### Added

- Initial public release of the `fovea-io` crate.
- Feature-gated PNG (`png`), JPEG (`jpeg`), and BMP (`bmp`) codecs,
  plus an `all-codecs` umbrella feature.
- Feature-free format detection via `detect_format` and
  `ImageFormat`.
- `load` / `load_reader` convenience entry points that dispatch by
  detected format.
- Per-codec exhaustive output enums (`PngImage`, `JpegImage`,
  `BmpImage`) wrapping concrete `Image<P>` values together with
  format-specific metadata.
- Sealed `PngPixel` / `JpegPixel` / `BmpPixel` encode traits that
  turn unsupported output formats into compile-time errors.
- Three-tier error type `IoError` covering format detection failures,
  unsupported pixel formats, and per-codec decode/encode errors.

[0.1.1]: https://github.com/karhunen-loeve/fovea-io/releases/tag/v0.1.1
