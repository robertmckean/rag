"""Storage package for normalized artifacts."""

# Storage helpers own file-format and write-order behavior for normalized artifacts.
# They intentionally avoid provider-specific knowledge so outputs stay reusable.
# Deterministic file writing is critical for repeatable tests and release diffs.
