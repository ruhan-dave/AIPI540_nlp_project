"""The CSV contract: what the service accepts, and what it writes back."""

from __future__ import annotations

import pytest

from service.batch_io import (
    CsvValidationError,
    chunked,
    iter_review_rows,
    read_result_csv,
    result_columns,
    validate_review_csv,
    write_result_csv,
)

VALID = b"content,score\nThe app is great,5\nThe app crashes,1\n"


def test_validate_accepts_a_valid_csv() -> None:
    result = validate_review_csv(VALID, max_bytes=1024, max_rows=10)
    assert result.row_count == 2
    assert result.columns == ("content", "score")


def test_validate_requires_a_content_column() -> None:
    with pytest.raises(CsvValidationError, match="content"):
        validate_review_csv(b"text,score\nhello,5\n", max_bytes=1024, max_rows=10)


def test_validate_rejects_a_header_with_no_rows() -> None:
    with pytest.raises(CsvValidationError, match="no review rows"):
        validate_review_csv(b"content,score\n", max_bytes=1024, max_rows=10)


def test_validate_rejects_an_empty_content_column() -> None:
    with pytest.raises(CsvValidationError, match="empty"):
        validate_review_csv(b"content,score\n,5\n,4\n", max_bytes=1024, max_rows=10)


def test_validate_rejects_an_empty_file() -> None:
    with pytest.raises(CsvValidationError, match="empty"):
        validate_review_csv(b"", max_bytes=1024, max_rows=10)


def test_validate_rejects_a_file_that_is_not_utf8() -> None:
    with pytest.raises(CsvValidationError, match="UTF-8"):
        validate_review_csv(b"\xff\xfe\x00binary", max_bytes=1024, max_rows=10)


def test_validate_enforces_the_row_limit() -> None:
    payload = b"content\n" + b"a review\n" * 5
    with pytest.raises(CsvValidationError, match="over the 3 review limit"):
        validate_review_csv(payload, max_bytes=1024, max_rows=3)


def test_validate_enforces_the_byte_limit() -> None:
    with pytest.raises(CsvValidationError, match="over the 10 byte limit"):
        validate_review_csv(VALID, max_bytes=10, max_rows=10)


def test_validate_ignores_blank_lines() -> None:
    result = validate_review_csv(
        b"content\nhello\n\n \nworld\n", max_bytes=1024, max_rows=10
    )
    assert result.row_count == 2


def test_iter_review_rows_adds_a_source_row_index() -> None:
    rows = list(iter_review_rows(VALID))
    assert [row["source_row_index"] for row in rows] == ["0", "1"]
    assert rows[0]["content"] == "The app is great"
    assert rows[0]["score"] == "5"


def test_iter_review_rows_keeps_an_explicit_source_row_index() -> None:
    rows = list(iter_review_rows(b"content,source_row_index\nhello,7\n"))
    assert rows[0]["source_row_index"] == "7"


def test_iter_review_rows_fills_a_blank_source_row_index() -> None:
    rows = list(iter_review_rows(b"content,source_row_index\nhello,\n"))
    assert rows[0]["source_row_index"] == "0"


def test_chunked_splits_into_fixed_size_chunks() -> None:
    rows = [{"content": str(index)} for index in range(5)]
    chunks = list(chunked(rows, 2))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]


def test_result_columns_preserve_the_input_and_add_the_service_columns() -> None:
    columns = result_columns(["content", "score", "source_row_index"])
    assert columns[:3] == ["content", "score", "source_row_index"]
    assert "bert_topic" in columns
    assert "split_fallback" in columns
    # A column the caller never supplied is not invented.
    assert "review_id" not in columns


def test_result_columns_keep_review_id_when_the_input_has_it() -> None:
    columns = result_columns(["content", "review_id"])
    assert columns.count("review_id") == 1


def test_write_result_csv_round_trips() -> None:
    rows = [{"content": "hello", "sub_comment": "hello", "bert_topic": "function"}]
    body = write_result_csv(rows, ["content", "sub_comment", "bert_topic"])
    parsed = read_result_csv(body)
    assert parsed[0]["bert_topic"] == "function"
    assert parsed[0]["content"] == "hello"


def test_write_result_csv_fills_missing_columns() -> None:
    body = write_result_csv([{"content": "hello"}], ["content", "bert_topic"])
    parsed = read_result_csv(body)
    assert parsed[0]["bert_topic"] == ""
