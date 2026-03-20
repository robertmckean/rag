import unittest
from pathlib import Path
from unittest.mock import patch

from rag.retrieval.lexical import RetrievalFilters, parse_query, retrieve_message_timeline, retrieve_message_windows


# These tests cover the first message-first lexical retrieval slice and its result contract.
# The fixture run is designed to exercise ranking, window expansion, filtering, and tie-breaking.
# Assertions stay focused on observable behavior rather than internal helper implementation details.
class RetrievalLexicalTests(unittest.TestCase):
    # Point tests at the normalized run fixture used for all first-slice retrieval checks.
    def setUp(self) -> None:
        self.run_dir = Path("tests/fixtures/retrieval/sample_run")

    # Verify that lexical ranking returns contextual windows around the highest-scoring focal messages.
    def test_ranks_messages_and_returns_context_windows(self) -> None:
        results = retrieve_message_windows(self.run_dir, "resume", limit=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].provider, "chatgpt")
        self.assertEqual(results[0].conversation_id, "chatgpt:conversation:conv-resume")
        self.assertEqual(results[0].focal_message_id, "chatgpt:message:msg-user-1")
        self.assertEqual(results[0].window_start_sequence_index, 0)
        self.assertEqual(results[0].window_end_sequence_index, 3)
        self.assertEqual(
            [message["message_id"] for message in results[0].messages],
            [
                "chatgpt:message:msg-system-1",
                "chatgpt:message:msg-user-1",
                "chatgpt:message:msg-assistant-1",
                "chatgpt:message:msg-user-2",
            ],
        )
        self.assertEqual(results[0].match_basis["retrieval_mode"], "relevance")

    # Verify that multiple high-scoring focal hits dedupe down to one identical conversation window.
    def test_dedupes_identical_windows(self) -> None:
        results = retrieve_message_windows(self.run_dir, "resume", limit=10)

        self.assertEqual(len(results), 1)

    # Verify that provider and date filters narrow the candidate pool before ranking.
    def test_applies_provider_and_date_filters(self) -> None:
        results = retrieve_message_windows(
            self.run_dir,
            "burnout",
            filters=RetrievalFilters(
                provider="claude",
                date_from="2025-03-10",
                date_to="2025-03-20",
            ),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].provider, "claude")
        self.assertEqual(results[0].focal_message_id, "claude:message:msg-1")

    # Verify that author-role and conversation filters are honored.
    def test_applies_author_role_and_conversation_filters(self) -> None:
        results = retrieve_message_windows(
            self.run_dir,
            "leadership",
            filters=RetrievalFilters(
                conversation_id="chatgpt:conversation:conv-resume",
                author_role="assistant",
            ),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].focal_message_id, "chatgpt:message:msg-assistant-1")

    # Verify that deterministic tie-breaking prefers the earlier timestamp when scores are equal.
    def test_uses_deterministic_tie_breaking(self) -> None:
        results = retrieve_message_windows(self.run_dir, "resume", limit=5)

        self.assertEqual(results[0].focal_message_id, "chatgpt:message:msg-user-1")
        self.assertGreater(results[0].match_basis["scoring_features"]["bm25_score"], 0.0)
        self.assertEqual(results[0].match_basis["scoring_features"]["title_overlap"], 1)
        self.assertEqual(results[0].match_basis["scoring_features"]["chronological_rank_basis"], "relevance_score_desc")

    # Verify that BM25 scoring details are returned in a readable debug-friendly structure.
    def test_populates_bm25_scoring_details(self) -> None:
        results = retrieve_message_windows(self.run_dir, "resume leadership", limit=5)

        self.assertEqual(results[0].focal_message_id, "chatgpt:message:msg-assistant-1")
        self.assertEqual(results[0].match_basis["raw_query"], "resume leadership")
        self.assertEqual(results[0].match_basis["normalized_query_terms"], ["resume", "leadership"])
        self.assertEqual(results[0].match_basis["scoring_terms"], ["resume", "leadership"])
        self.assertEqual(results[0].match_basis["quoted_phrases"], [])
        self.assertEqual(results[0].match_basis["normalized_phrase_targets"], [])
        scoring_features = results[0].match_basis["scoring_features"]
        self.assertEqual(scoring_features["query_terms"], ["resume", "leadership"])
        self.assertGreater(scoring_features["document_length"], 0)
        self.assertGreater(scoring_features["average_document_length"], 0.0)
        self.assertEqual(
            [item["term"] for item in scoring_features["bm25_term_contributions"]],
            ["resume", "leadership"],
        )
        self.assertGreater(scoring_features["final_score"], scoring_features["bm25_score"])

    # Verify that the phrase and title boosts stay secondary to BM25 relevance.
    def test_phrase_and_title_boosts_do_not_override_base_relevance(self) -> None:
        results = retrieve_message_windows(self.run_dir, "resume leadership", limit=5)

        self.assertEqual(results[0].focal_message_id, "chatgpt:message:msg-assistant-1")
        self.assertGreater(results[0].score, results[-1].score)
        self.assertEqual(results[-1].focal_message_id, "chatgpt:message:msg-user-2")

    # Verify that newest mode reorders matching windows by descending focal timestamp.
    def test_newest_mode_orders_by_descending_timestamp(self) -> None:
        results = retrieve_message_windows(self.run_dir, "resume leadership", limit=5, mode="newest")

        self.assertEqual([result.focal_message_id for result in results], [
            "chatgpt:message:msg-user-2",
            "chatgpt:message:msg-assistant-1",
        ])
        self.assertEqual(results[0].match_basis["scoring_features"]["chronological_rank_basis"], "created_at_desc")
        self.assertEqual(results[0].match_basis["result_view"], "contextual_window")

    # Verify that oldest mode reorders matching windows by ascending focal timestamp.
    def test_oldest_mode_orders_by_ascending_timestamp(self) -> None:
        results = retrieve_message_windows(self.run_dir, "resume leadership", limit=5, mode="oldest")

        self.assertEqual([result.focal_message_id for result in results], [
            "chatgpt:message:msg-user-1",
            "chatgpt:message:msg-user-2",
        ])
        self.assertEqual(results[0].match_basis["scoring_features"]["chronological_rank_basis"], "created_at_asc")
        self.assertEqual(results[0].match_basis["result_view"], "contextual_window")

    # Verify that relevance_recency adds a visible but modest boost for newer matching messages.
    def test_relevance_recency_mode_adds_modest_recency_boost(self) -> None:
        results = retrieve_message_windows(self.run_dir, "resume", limit=5, mode="relevance_recency")

        self.assertEqual(results[0].focal_message_id, "chatgpt:message:msg-assistant-1")
        self.assertEqual(results[0].match_basis["scoring_features"]["retrieval_mode"], "relevance_recency")
        self.assertGreater(results[0].match_basis["scoring_features"]["recency_boost"], 0.0)
        self.assertEqual(
            results[0].match_basis["scoring_features"]["chronological_rank_basis"],
            "relevance_score_desc_plus_recency_boost",
        )

    # Verify that date filters and window expansion continue to work in chronological modes.
    def test_filters_and_windows_still_work_in_newest_mode(self) -> None:
        results = retrieve_message_windows(
            self.run_dir,
            "burnout rag",
            limit=5,
            mode="newest",
            filters=RetrievalFilters(provider="claude", date_from="2025-03-10", date_to="2025-03-20"),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].provider, "claude")
        self.assertEqual(results[0].window_start_sequence_index, 0)
        self.assertEqual(results[0].window_end_sequence_index, 2)

    # Verify that timeline exploration orders matching focal messages chronologically across conversations.
    def test_timeline_orders_matching_results_across_conversations(self) -> None:
        results = retrieve_message_timeline(self.run_dir, "project", limit=5)

        self.assertEqual([result.focal_message_id for result in results], [
            "chatgpt:message:msg-user-2",
            "claude:message:msg-3",
        ])
        self.assertEqual(
            [result.focal_created_at for result in results],
            ["2025-03-01T09:03:00Z", "2025-03-15T11:05:00Z"],
        )
        self.assertEqual(results[0].match_basis["result_view"], "timeline_compact")

    # Verify that timeline results expose the required traceability fields.
    def test_timeline_results_include_traceability_fields(self) -> None:
        results = retrieve_message_timeline(self.run_dir, "project", limit=5)

        self.assertEqual(results[0].run_id, "sample-run")
        self.assertEqual(results[0].provider, "chatgpt")
        self.assertEqual(results[0].conversation_id, "chatgpt:conversation:conv-resume")
        self.assertEqual(results[0].author_role, "user")
        self.assertIn("messages.jsonl", results[0].provenance["messages_jsonl_path"])
        self.assertEqual(
            results[0].match_basis["scoring_features"]["chronological_rank_basis"],
            "created_at_asc_across_conversations",
        )

    # Verify that timeline stays explicitly distinct from contextual newest/oldest retrieval.
    def test_timeline_is_distinct_from_contextual_chronological_modes(self) -> None:
        timeline_results = retrieve_message_timeline(self.run_dir, "project", limit=5)
        newest_results = retrieve_message_windows(self.run_dir, "project", limit=5, mode="newest")

        self.assertFalse(hasattr(timeline_results[0], "messages"))
        self.assertTrue(hasattr(newest_results[0], "messages"))
        self.assertEqual(timeline_results[0].match_basis["result_view"], "timeline_compact")
        self.assertEqual(newest_results[0].match_basis["result_view"], "contextual_window")

    # Verify that lexical matching still gates which records appear in the timeline.
    def test_timeline_requires_lexical_match(self) -> None:
        results = retrieve_message_timeline(self.run_dir, "project", filters=RetrievalFilters(provider="claude"))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].focal_message_id, "claude:message:msg-3")

    # Verify that timeline ordering remains deterministic for repeated calls.
    def test_timeline_ordering_is_deterministic(self) -> None:
        first = retrieve_message_timeline(self.run_dir, "project", limit=5)
        second = retrieve_message_timeline(self.run_dir, "project", limit=5)

        self.assertEqual(
            [result.focal_message_id for result in first],
            [result.focal_message_id for result in second],
        )

    # Verify that timeline deduplication is by focal message id rather than by window boundaries.
    def test_timeline_dedupes_by_focal_message_id(self) -> None:
        results = retrieve_message_timeline(self.run_dir, "resume leadership", limit=10)

        self.assertEqual(
            [result.focal_message_id for result in results],
            [
                "chatgpt:message:msg-user-1",
                "chatgpt:message:msg-assistant-1",
                "chatgpt:message:msg-user-2",
            ],
        )

    # Verify that query parsing deduplicates repeated terms while preserving order.
    def test_query_parsing_deduplicates_repeated_terms(self) -> None:
        parsed_query = parse_query("resume resume leadership resume")

        self.assertEqual(parsed_query.normalized_query_terms, ("resume", "leadership"))
        self.assertEqual(parsed_query.scoring_terms, ("resume", "leadership"))

    # Verify that quoted phrases are preserved for exact phrase matching while tokens still normalize.
    def test_query_parsing_extracts_quoted_phrases(self) -> None:
        parsed_query = parse_query('resume "leadership bullets" planning')

        self.assertEqual(parsed_query.quoted_phrases, ("leadership bullets",))
        self.assertEqual(parsed_query.normalized_phrase_targets, ("leadership bullets",))
        self.assertEqual(parsed_query.scoring_terms, ("resume", "leadership", "bullets", "planning"))

    # Verify that stopword filtering is default-off while preserving the filtered-term diagnostic.
    def test_query_parsing_keeps_stopwords_when_filtering_is_off(self) -> None:
        parsed_query = parse_query('find "where I talked about burnout" in the conversations')

        self.assertEqual(
            parsed_query.stopword_filtered_terms,
            ("find", "talked", "about", "burnout", "conversations"),
        )
        self.assertEqual(
            parsed_query.scoring_terms,
            ("find", "where", "i", "talked", "about", "burnout", "in", "the", "conversations"),
        )
        self.assertEqual(parsed_query.normalized_phrase_targets, ("where i talked about burnout",))

    # Verify that enabling stopword filtering still falls back safely if filtering would empty scoring terms.
    def test_query_parsing_stopword_filter_falls_back_when_terms_would_be_empty(self) -> None:
        with patch("rag.retrieval.lexical.STOPWORD_FILTER_ENABLED", True):
            parsed_query = parse_query("the and of")

        self.assertEqual(parsed_query.normalized_query_terms, ("the", "and", "of"))
        self.assertEqual(parsed_query.stopword_filtered_terms, ())
        self.assertEqual(parsed_query.scoring_terms, ("the", "and", "of"))

    # Verify that unquoted multi-word queries no longer create a fallback phrase target.
    def test_unquoted_multiword_query_does_not_create_phrase_target(self) -> None:
        parsed_query = parse_query("resume leadership")

        self.assertEqual(parsed_query.quoted_phrases, ())
        self.assertEqual(parsed_query.normalized_phrase_targets, ())

    # Verify that quoted phrase matching produces explicit phrase metadata in the result.
    def test_quoted_phrase_matching_surfaces_phrase_metadata(self) -> None:
        results = retrieve_message_windows(self.run_dir, '"resume summary"', limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].match_basis["quoted_phrases"], ["resume summary"])
        self.assertEqual(results[0].match_basis["normalized_phrase_targets"], ["resume summary"])
        self.assertEqual(results[0].match_basis["matched_phrase_targets"], ["resume summary"])


if __name__ == "__main__":
    unittest.main()
