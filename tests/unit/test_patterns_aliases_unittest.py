"""Unit tests for alias normalization (Phase 7)."""

from __future__ import annotations

import unittest

from rag.patterns.aliases import (
    ENTITY_ALIASES,
    alias_conflicts_in_phase,
    canonicalize_entity,
)


class CanonicalizeEntityTests(unittest.TestCase):
    """Tests for canonicalize_entity."""

    def test_alias_hit(self) -> None:
        self.assertEqual(canonicalize_entity("Mark"), "Marc")

    def test_alias_miss(self) -> None:
        self.assertEqual(canonicalize_entity("Craig"), "Craig")

    def test_canonical_form_passthrough(self) -> None:
        """The canonical name itself is returned unchanged."""
        self.assertEqual(canonicalize_entity("Marc"), "Marc")

    def test_unknown_entity_unchanged(self) -> None:
        self.assertEqual(canonicalize_entity("Xyzzy"), "Xyzzy")


class AliasConflictsInPhaseTests(unittest.TestCase):
    """Tests for alias_conflicts_in_phase."""

    def test_no_conflict_single_variant(self) -> None:
        """Only one variant present -> no conflict."""
        conflicts = alias_conflicts_in_phase({"Mark", "Craig"})
        self.assertEqual(conflicts, set())

    def test_no_conflict_canonical_only(self) -> None:
        """Only the canonical form present -> no conflict."""
        conflicts = alias_conflicts_in_phase({"Marc", "Craig"})
        self.assertEqual(conflicts, set())

    def test_conflict_both_variants_present(self) -> None:
        """Both 'Marc' and 'Mark' in the same phase -> conflict on 'Marc'."""
        conflicts = alias_conflicts_in_phase({"Marc", "Mark", "Craig"})
        self.assertEqual(conflicts, {"Marc"})

    def test_no_aliases_in_phase(self) -> None:
        """Phase with no aliased entities -> empty."""
        conflicts = alias_conflicts_in_phase({"Craig", "Benz", "Zeno"})
        self.assertEqual(conflicts, set())

    def test_empty_phase(self) -> None:
        conflicts = alias_conflicts_in_phase(set())
        self.assertEqual(conflicts, set())

    def test_seeded_alias_map_is_minimal(self) -> None:
        """The alias map should contain exactly the confirmed aliases."""
        self.assertEqual(ENTITY_ALIASES, {"Mark": "Marc"})


if __name__ == "__main__":
    unittest.main()
