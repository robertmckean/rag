"""Unit tests for assistant restatement detection and ranking downweight."""

from __future__ import annotations

import unittest

from rag.retrieval.types import (
    ASSISTANT_RESTATEMENT_FACTOR,
    RESTATEMENT_OVERLAP_THRESHOLD,
    RESTATEMENT_WINDOW_RADIUS,
    is_assistant_restatement,
    get_nearby_user_texts,
    _content_tokens,
)


class ContentTokensTests(unittest.TestCase):
    """Tests for tokenization helper."""

    def test_lowercase_and_strip_punctuation(self) -> None:
        tokens = _content_tokens("Hello, World! This is a TEST.")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("test", tokens)

    def test_stopwords_removed(self) -> None:
        tokens = _content_tokens("I was in the house and it was good")
        # stopwords: i, was, in, the, and, it
        self.assertNotIn("i", tokens)
        self.assertNotIn("was", tokens)
        self.assertNotIn("the", tokens)
        self.assertIn("house", tokens)
        self.assertIn("good", tokens)

    def test_empty_string(self) -> None:
        self.assertEqual(_content_tokens(""), set())


class RestatementDetectionTests(unittest.TestCase):
    """Tests for is_assistant_restatement."""

    def test_clear_restatement_detected(self) -> None:
        user_text = "I told Marc that I had no romantic interest. He said I have zero chance."
        assistant_text = "You told Marc you had no romantic interest and he told you that you have zero chance."
        self.assertTrue(is_assistant_restatement(assistant_text, [user_text]))

    def test_non_restatement_not_flagged(self) -> None:
        user_text = "I went to the villa and met everyone."
        assistant_text = (
            "The psychological dynamic here reveals a pattern of boundary testing. "
            "When someone introduces you to their social circle, they are signaling "
            "trust hierarchy and gauging your response to social pressure."
        )
        self.assertFalse(is_assistant_restatement(assistant_text, [user_text]))

    def test_threshold_boundary_below(self) -> None:
        """Overlap just below threshold should not flag."""
        # 10 content tokens in assistant, need <6 to overlap (threshold 0.60)
        user_text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        assistant_text = "alpha bravo charlie delta echo kilo lima mike november oscar"
        # 5 out of 10 overlap = 0.50, below 0.60
        self.assertFalse(is_assistant_restatement(assistant_text, [user_text]))

    def test_threshold_boundary_at(self) -> None:
        """Overlap exactly at threshold should flag."""
        user_text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        assistant_text = "alpha bravo charlie delta echo foxtrot kilo lima mike november"
        # 6 out of 10 overlap = 0.60, at threshold
        self.assertTrue(is_assistant_restatement(assistant_text, [user_text]))

    def test_punctuation_normalization(self) -> None:
        user_text = "Marc said: 'you have zero chance!'"
        assistant_text = "Marc said you have zero chance."
        self.assertTrue(is_assistant_restatement(assistant_text, [user_text]))

    def test_case_normalization(self) -> None:
        user_text = "MARC TOLD ME I HAVE ZERO CHANCE WITH PINN"
        assistant_text = "marc told you zero chance with pinn"
        self.assertTrue(is_assistant_restatement(assistant_text, [user_text]))

    def test_stopword_heavy_overlap_no_false_positive(self) -> None:
        """Two messages sharing only stopwords should not flag."""
        user_text = "I was going to the store and it was fine"
        assistant_text = "It was a beautiful day and I was happy to be there"
        # After removing stopwords, content tokens barely overlap
        self.assertFalse(is_assistant_restatement(assistant_text, [user_text]))

    def test_empty_assistant_text(self) -> None:
        self.assertFalse(is_assistant_restatement("", ["some user text"]))

    def test_empty_user_texts(self) -> None:
        self.assertFalse(is_assistant_restatement("some assistant text", []))

    def test_multiple_user_texts_any_match(self) -> None:
        """Restatement should trigger if any single user text matches."""
        user_texts = [
            "completely unrelated topic about cooking",
            "Marc told me I have zero chance with Pinn",
        ]
        assistant_text = "Marc told you that you have zero chance with Pinn"
        self.assertTrue(is_assistant_restatement(assistant_text, user_texts))

    def test_deterministic(self) -> None:
        user_text = "Marc said I have zero chance."
        assistant_text = "Marc said you have zero chance."
        results = [is_assistant_restatement(assistant_text, [user_text]) for _ in range(10)]
        self.assertTrue(all(r == results[0] for r in results))


class NearbyUserTextsTests(unittest.TestCase):
    """Tests for get_nearby_user_texts."""

    def _make_messages(self, roles_and_texts: list[tuple[str, str]]) -> tuple[dict[str, object], ...]:
        return tuple(
            {"message_id": f"m{i}", "author_role": role, "text": text}
            for i, (role, text) in enumerate(roles_and_texts)
        )

    def test_returns_user_texts_within_radius(self) -> None:
        msgs = self._make_messages([
            ("user", "first user message"),
            ("assistant", "assistant reply"),
            ("user", "second user message"),
        ])
        texts = get_nearby_user_texts(msgs, 1)  # focal is assistant at index 1
        self.assertEqual(len(texts), 2)
        self.assertIn("first user message", texts)
        self.assertIn("second user message", texts)

    def test_excludes_focal_message(self) -> None:
        msgs = self._make_messages([
            ("user", "user message"),
            ("user", "focal user message"),
            ("user", "another user message"),
        ])
        texts = get_nearby_user_texts(msgs, 1)
        self.assertNotIn("focal user message", texts)

    def test_excludes_assistant_messages(self) -> None:
        msgs = self._make_messages([
            ("assistant", "assistant before"),
            ("assistant", "focal assistant"),
            ("assistant", "assistant after"),
        ])
        texts = get_nearby_user_texts(msgs, 1)
        self.assertEqual(len(texts), 0)

    def test_respects_window_radius(self) -> None:
        # Build a long conversation with user messages beyond the radius.
        roles = [("user", f"msg {i}") for i in range(20)]
        roles[10] = ("assistant", "focal")
        msgs = self._make_messages(roles)
        texts = get_nearby_user_texts(msgs, 10)
        # Window: indices 5..15 (radius=5), excluding focal at 10
        self.assertEqual(len(texts), 10)  # 5 before + 5 after, all user


class RestatementFactorTests(unittest.TestCase):
    """Tests for ranking factor constants."""

    def test_factor_value(self) -> None:
        self.assertEqual(ASSISTANT_RESTATEMENT_FACTOR, 0.5)

    def test_threshold_value(self) -> None:
        self.assertEqual(RESTATEMENT_OVERLAP_THRESHOLD, 0.60)

    def test_window_radius_value(self) -> None:
        self.assertEqual(RESTATEMENT_WINDOW_RADIUS, 5)

    def test_user_message_not_affected(self) -> None:
        """is_assistant_restatement is only called for assistant messages in production.
        Verify the function itself doesn't break if called with user text."""
        # This is a logic test — user messages should never be passed to this function.
        # But the function should still return a sensible result.
        result = is_assistant_restatement("hello world", ["hello world"])
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
