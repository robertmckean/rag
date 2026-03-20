from src.rag.normalize.timestamps import normalize_timestamp


def test_normalize_epoch_seconds() -> None:
    assert normalize_timestamp(1674954271.549849) == "2023-01-29T03:04:31.549849Z"


def test_normalize_iso_string() -> None:
    assert (
        normalize_timestamp("2025-02-27T13:53:05.581841Z")
        == "2025-02-27T13:53:05.581841Z"
    )


def test_normalize_blank_to_none() -> None:
    assert normalize_timestamp("   ") is None


def test_normalize_none_to_none() -> None:
    assert normalize_timestamp(None) is None
