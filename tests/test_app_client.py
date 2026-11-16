"""The frontend's review-service client: the requests it makes, in order.

The batch flow is three steps — ask for a presigned URL, PUT the CSV to S3,
finalize the upload — and the order matters: a job that exists but was never
finalized is a job nobody works on.  These tests pin that sequence without a
server: the HTTP calls are replaced, so what is under test is the client's
contract with the service, not the service itself.
"""

from __future__ import annotations

import httpx
import pytest

import app as frontend

UPLOAD_URL = "https://reviewassist-jobs.s3.us-east-1.amazonaws.com/inputs/u1/reviews.csv"


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict]]:
    """Record every HTTP call the client makes and answer it plausibly."""
    recorded: list[tuple[str, str, dict]] = []

    def fake_post(url: str, **kwargs):
        recorded.append(("POST", url, kwargs))
        if url.endswith("/jobs/uploads"):
            return httpx.Response(
                201,
                json={
                    "upload_id": "u1",
                    "upload_url": UPLOAD_URL,
                    "input_key": "inputs/u1/reviews.csv",
                    "expires_in": 900,
                },
            )
        return httpx.Response(
            202, json={"job_id": "u1", "status": "PENDING", "total_reviews": 2}
        )

    def fake_put(url: str, **kwargs):
        recorded.append(("PUT", url, kwargs))
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "put", fake_put)
    monkeypatch.setattr(frontend, "API_TOKEN", "test-token")
    return recorded


def test_the_batch_csv_goes_straight_to_s3(calls) -> None:
    """The CSV is uploaded to S3, not posted through the API Gateway."""
    job_id = frontend.upload_batch_csv(b"content\nhello\n", "batch.csv")

    assert job_id == "u1"
    assert [method for method, _, _ in calls] == ["POST", "PUT", "POST"]
    assert calls[0][1].endswith("/jobs/uploads")
    assert calls[1][1] == UPLOAD_URL
    assert calls[1][2]["content"] == b"content\nhello\n"
    assert calls[2][1].endswith("/jobs/uploads/u1/finalize")


def test_every_call_carries_the_bearer_token(calls) -> None:
    frontend.upload_batch_csv(b"content\nhello\n", "batch.csv")

    for method, _, kwargs in calls:
        if method == "PUT":
            # The presigned URL carries its own authorization.
            continue
        assert kwargs["headers"]["Authorization"] == "Bearer test-token"


def test_a_refused_upload_url_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs):
        return httpx.Response(503, text="the service is saturated")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(frontend, "API_TOKEN", "test-token")

    with pytest.raises(RuntimeError, match="refused to issue an upload URL"):
        frontend.upload_batch_csv(b"content\nhello\n", "batch.csv")


def test_a_failed_upload_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs):
        return httpx.Response(
            201,
            json={
                "upload_id": "u1",
                "upload_url": UPLOAD_URL,
                "input_key": "inputs/u1/reviews.csv",
                "expires_in": 900,
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "put", lambda url, **kwargs: httpx.Response(403))
    monkeypatch.setattr(frontend, "API_TOKEN", "test-token")

    with pytest.raises(RuntimeError, match="could not be uploaded"):
        frontend.upload_batch_csv(b"content\nhello\n", "batch.csv")


def test_a_rejected_finalize_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs):
        if url.endswith("/jobs/uploads"):
            return httpx.Response(
                201,
                json={
                    "upload_id": "u1",
                    "upload_url": UPLOAD_URL,
                    "input_key": "inputs/u1/reviews.csv",
                    "expires_in": 900,
                },
            )
        return httpx.Response(422, text="the CSV needs a `content` column")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "put", lambda url, **kwargs: httpx.Response(200))
    monkeypatch.setattr(frontend, "API_TOKEN", "test-token")

    with pytest.raises(RuntimeError, match="rejected the uploaded batch"):
        frontend.upload_batch_csv(b"text\nhello\n", "batch.csv")
