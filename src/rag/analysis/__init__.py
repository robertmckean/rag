"""Read-only analysis helpers for normalized outputs."""

# Analysis code lives separately from extraction so it cannot mutate canonical outputs.
# These helpers inspect normalized artifacts after a run has already completed.
# Keeping the package small makes post-normalization checks easier to reason about.
