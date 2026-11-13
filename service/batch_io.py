"""CSV intake, chunking and result assembly for review jobs.

The input contract is one CSV per job with a non-empty ``content`` column.  Every
other column is optional and is carried through to the result, so a caller can
keep ``score``, ``date`` or ``review_id`` next to the review text.  The result is
one row per sub-comment.

Nothing here holds a whole job in a DataFrame: the input is read row by row and
the worker works in fixed-size chunks, so the memory cost is bounded by the
50 MiB upload limit rather than by the number of reviews.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

#: The one column a review CSV must have.
CONTENT_COLUMN = "content"

#: A stable index of the row in the uploaded file, so results can be combined
#: back in the caller's order.  Added when the input does not carry one.
SOURCE_ROW_INDEX = "source_row_index"

#: Only carried through when the input has it.
REVIEW_ID = "review_id"

#: The columns the service adds to every result row.
RESULT_EXTRA_COLUMNS = (
    SOURCE_ROW_INDEX,
    "sub_comment_index",
    "sub_comment",
    "bert_topic",
    "bert_topic_conf",
    "bert_sentiment",
    "bert_sentiment_raw",
    "split_fallback",
)

#: S3 objects are read in bounded chunks rather than one giant read.
READ_CHUNK_BYTES = 1024 * 1024


class CsvValidationError(ValueError):
    """The uploaded CSV does not satisfy the job contract."""


@dataclass(frozen=True)
class ValidatedCsv:
    """What the server learned about an uploaded CSV."""

    row_count: int
    columns: tuple[str, ...]


def validate_review_csv(
    payload: bytes, *, max_bytes: int, max_rows: int
) -> ValidatedCsv:
    """Check that ``payload`` is a review CSV this service will accept.

    Raises :class:`CsvValidationError` with a message meant for the person who
    uploaded the file.
    """
    if not payload:
        raise CsvValidationError("the uploaded file is empty")
    if len(payload) > max_bytes:
        raise CsvValidationError(
            f"the uploaded file is {len(payload)} bytes, over the "
            f"{max_bytes} byte limit"
        )

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CsvValidationError("the uploaded file is not valid UTF-8") from exc

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration:
        raise CsvValidationError("the uploaded file has no header row") from None

    columns = [column.strip() for column in header]
    if CONTENT_COLUMN not in columns:
        found = ", ".join(columns) if columns else "none"
        raise CsvValidationError(
            f"the CSV needs a `{CONTENT_COLUMN}` column with the review text "
            f"(columns found: {found})"
        )

    content_at = columns.index(CONTENT_COLUMN)
    rows = 0
    with_text = 0
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            # A blank line is not a review.
            continue
        rows += 1
        if len(row) > content_at and row[content_at].strip():
            with_text += 1

    if rows == 0:
        raise CsvValidationError("the CSV has a header but no review rows")
    if with_text == 0:
        raise CsvValidationError(f"the `{CONTENT_COLUMN}` column is empty")
    if rows > max_rows:
        raise CsvValidationError(
            f"the CSV has {rows} reviews, over the {max_rows} review limit for one job"
        )

    return ValidatedCsv(row_count=rows, columns=tuple(columns))


def iter_review_rows(payload: bytes) -> Iterator[dict[str, str]]:
    """Yield one dict per review, each with a stable ``source_row_index``."""
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    for index, row in enumerate(reader):
        if row is None:
            continue
        clean = {
            key: ("" if value is None else value)
            for key, value in row.items()
            if key is not None
        }
        if not any(value.strip() for value in clean.values()):
            continue
        if not clean.get(SOURCE_ROW_INDEX, "").strip():
            clean[SOURCE_ROW_INDEX] = str(index)
        yield clean


def chunked(
    rows: Iterable[dict[str, str]], size: int
) -> Iterator[list[dict[str, str]]]:
    """Group rows into lists of at most ``size`` rows."""
    chunk: list[dict[str, str]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def result_columns(input_columns: Sequence[str]) -> list[str]:
    """The result header: the input columns, then the columns the service adds.

    ``review_id`` is only added when the input carries it — the input columns are
    preserved as they are, and a column the caller never supplied is not invented.
    """
    columns = list(input_columns)
    for name in RESULT_EXTRA_COLUMNS:
        if name not in columns:
            columns.append(name)
    return columns


def write_result_csv(rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> bytes:
    """Render result rows as a CSV with exactly ``columns``."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue().encode("utf-8")


def read_result_csv(payload: bytes) -> list[dict[str, str]]:
    """Parse a result CSV back into rows (used by the tests and the assembler)."""
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    return [dict(row) for row in reader]


# --- S3 object helpers ---------------------------------------------------
#
# The service talks to S3 through the low-level client so the bucket and key are
# always explicit.  The body is read in bounded chunks: an upload is capped at
# 50 MiB, so one object is small enough to hold but not so small that a single
# unbounded read is acceptable.


def build_s3_client(region: str) -> Any:
    """An S3 client that signs with SigV4.

    SigV4 is the signature version every current S3 region requires and the one
    AWS recommends; the legacy v2 presigned URLs are on their way out.
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3", region_name=region, config=Config(signature_version="s3v4")
    )


def read_object_bytes(client: Any, bucket: str, key: str) -> bytes:
    """Read an S3 object into memory."""
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    buffer = bytearray()
    while True:
        chunk = body.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        buffer.extend(chunk)
    return bytes(buffer)


def write_object_bytes(
    client: Any, bucket: str, key: str, body: bytes, *, content_type: str = "text/csv"
) -> None:
    """Write an S3 object."""
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def presign_object_get(client: Any, bucket: str, key: str, expires_in: int) -> str:
    """Build a short-lived presigned GET URL for an S3 object."""
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )
