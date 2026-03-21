"""Static alias normalization for recurring-entity extraction.

Maps known spelling variants to a canonical entity name.  The alias map is
explicit, version-controlled, and never inferred automatically.  A conflict
check prevents merging when two variants of the same canonical name appear
in the same narrative phase — evidence that they may be different entities.
"""

from __future__ import annotations


# Variant -> canonical form.  Extend as confirmed aliases are discovered.
ENTITY_ALIASES: dict[str, str] = {
    "Mark": "Marc",
}


def canonicalize_entity(name: str) -> str:
    """Return the canonical form of an entity name, or the name unchanged."""
    return ENTITY_ALIASES.get(name, name)


def alias_conflicts_in_phase(phase_entities: set[str]) -> set[str]:
    """Identify canonical names that have conflicting variants in a phase.

    Returns the set of canonical names for which two or more distinct variant
    spellings both appear in *phase_entities*.  These canonical names should
    NOT be merged — the co-occurrence suggests the variants are different
    entities despite the alias map entry.
    """
    # Build canonical -> set of variants present in this phase.
    canonical_variants: dict[str, set[str]] = {}
    for entity in phase_entities:
        canonical = ENTITY_ALIASES.get(entity)
        if canonical is None:
            continue
        # The variant mapped to a canonical form.  Track it.
        canonical_variants.setdefault(canonical, set()).add(entity)
        # Also check if the canonical form itself appears.
        if canonical in phase_entities:
            canonical_variants[canonical].add(canonical)

    # A canonical name with the canonical form itself in the phase also needs
    # to be tracked even if no variant triggered the above loop.
    for entity in phase_entities:
        if entity in ENTITY_ALIASES.values():
            # This entity IS a canonical form.  Check if any variant is present.
            for variant, canon in ENTITY_ALIASES.items():
                if canon == entity and variant in phase_entities:
                    canonical_variants.setdefault(entity, set()).add(entity)
                    canonical_variants[entity].add(variant)

    return {canon for canon, variants in canonical_variants.items() if len(variants) >= 2}
