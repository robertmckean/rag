"""Normalization package for canonical records."""

# Normalization code converts provider-specific exports into one shared schema.
# Provider extractors stay isolated here so cross-provider orchestration can stay minimal.
# Run writers live alongside extractors because they compose canonical records into artifacts.
