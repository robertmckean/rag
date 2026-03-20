from src.rag.normalize.timestamps import normalize_timestamp


# Timestamp normalization is shared by every extractor, so regressions fan out quickly.
# These cases cover the main supported input shapes used by the current exports.
# Blank and missing values should collapse cleanly to None rather than inventing timestamps.

# Verify that numeric epoch seconds normalize into canonical UTC strings.
def test_normalize_epoch_seconds() -> None:
    assert normalize_timestamp(1674954271.549849) == "2023-01-29T03:04:31.549849Z"


# Verify that ISO timestamps already in canonical shape stay unchanged.
def test_normalize_iso_string() -> None:
    assert (
        normalize_timestamp("2025-02-27T13:53:05.581841Z")
        == "2025-02-27T13:53:05.581841Z"
    )


# Verify that blank timestamp strings collapse to None.
def test_normalize_blank_to_none() -> None:
    assert normalize_timestamp("   ") is None


# Verify that missing timestamps remain absent after normalization.
def test_normalize_none_to_none() -> None:
    assert normalize_timestamp(None) is None
