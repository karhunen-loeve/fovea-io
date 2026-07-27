# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-07-27

### Changed

- Minimum required version of `fovea` bumped to `0.3.0`. No other changes:
  the codecs and the public API are unchanged from `0.2.0`, and the release
  exists so `fovea-io` remains co-installable with the `fovea 0.3.0`
  release.

## [0.2.0] — 2026-06-12

### Changed

- `#![warn(missing_docs)]` promoted to `#![deny(missing_docs)]`. All
  public API items are now documented; the deny lint enforces that new
  public items ship with docs.
- Minimum required version of `fovea` bumped to `0.2.0`.

### Fixed

- `fovea-io/src/png.rs`: fixed two broken intra-doc links that caused
  `cargo doc -p fovea-io` to fail with `#![deny(rustdoc::broken_intra_doc_links)]`:
  - The stale `[read_png](crate::read_png)` link on `PngImage` — there is
    no `read_png` at the crate root. Repointed to `crate::load` and
    `crate::png::decode`, which are the actual public entry points.
  - The `[TransferAssumption]` link in the `PngImage` colour-type table —
    `TransferAssumption` is a crate-private enum that rustdoc rejects as a
    link target from a public item. Converted to inline code with a
    "(crate-private)" note; the explanatory prose is unchanged.

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

[0.3.0]: https://github.com/karhunen-loeve/fovea-io/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/karhunen-loeve/fovea-io/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/karhunen-loeve/fovea-io/releases/tag/v0.1.1
