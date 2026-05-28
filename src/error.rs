// ═══════════════════════════════════════════════════════════════════════════════
// Crate-level I/O error
//
// Codec-agnostic: no feature-gated fields.  Underlying codec errors are
// type-erased via `Box<dyn Error>` so this type compiles identically
// regardless of which codecs are enabled.
//
// Trade-off: `Box<dyn Error>` prevents `Clone`.  We accept this because
// preserving error chains (`source()`) matters more for diagnostics than
// `Clone` does for error handling patterns.
// ═══════════════════════════════════════════════════════════════════════════════

/// Crate-level I/O error.
///
/// Each variant classifies *what went wrong*.  Variants that wrap a codec
/// error carry a type-erased source via `Box<dyn Error + Send + Sync>`,
/// keeping this type codec-agnostic (no feature-gated fields).
///
/// Match on the enum directly to branch on the error category.
///
/// # Examples
///
/// ```
/// use fovea_io::IoError;
///
/// let err = IoError::InvalidFormat { reason: "not a PNG file" };
/// assert_eq!(err.to_string(), "invalid format: not a PNG file");
///
/// match err {
///     IoError::InvalidFormat { reason } => assert_eq!(reason, "not a PNG file"),
///     _ => unreachable!(),
/// }
/// ```
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// The input bytes don't match the expected format signature.
    #[error("invalid format: {reason}")]
    InvalidFormat {
        /// A short, static description of why the format is invalid.
        reason: &'static str,
    },

    /// The file is structurally valid but contains data we can't decode
    /// (e.g. unsupported bit depth, compression method, colour type).
    #[error("unsupported feature: {reason}")]
    UnsupportedFeature {
        /// A short, static description of the unsupported feature.
        reason: &'static str,
    },

    /// The codec reported a corruption or constraint violation during
    /// decoding.  The wrapped source carries the codec-specific detail.
    #[error("decode failed")]
    DecodeFailed {
        /// The underlying codec error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// An encoding operation failed.  The wrapped source carries the
    /// codec-specific detail.
    #[error("encode failed")]
    EncodeFailed {
        /// The underlying codec error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A standard I/O error (read/write/seek failure).
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
