"""The HTTP surface: auth, health, prediction and the job routes."""

from __future__ import annotations

import dataclasses
import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from service.inference import Splitter
from service.jobs import FAILED, PENDING
from service.server import ServiceContext

from .conftest import API_TOKEN, BUCKET

VALID_CSV = b"content,score\nThe app is great,5\nThe app crashes,1\n"


def auth(token: str = API_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def submit(client: TestClient, payload: bytes = VALID_CSV, **kwargs):
    return client.post(
        "/jobs",
        files={"file": ("reviews.csv", payload, "text/csv")},
        data={"options": json.dumps({"split": True})},
        headers=auth(),
        **kwargs,
    )


# --- health ---------------------------------------------------------------


def test_healthz_is_open(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_healthz_reports_503_while_the_model_is_loading(
    make_client,
) -> None:
    class LoadingRuntime:
        ready = False
        load_error = "checkpoint not found"

        def start(self) -> None:
            pass

        def predict(self, texts):  # pragma: no cover - never reached
            raise AssertionError("the model is not ready")

    with make_client(runtime=LoadingRuntime()) as client:
        response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["status"] == "loading"


# --- authentication -------------------------------------------------------


def test_predict_requires_a_token(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "great app"})
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_predict_rejects_a_wrong_token(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "great app"}, headers=auth("nope"))
    assert response.status_code == 401


def test_the_job_routes_require_a_token(client: TestClient) -> None:
    assert client.post("/jobs", files={"file": ("r.csv", VALID_CSV, "text/csv")}).status_code == 401
    assert client.get("/jobs/job-1").status_code == 401
    assert client.get("/jobs/job-1/result").status_code == 401


# --- prediction -----------------------------------------------------------


def test_predict_labels_the_review(client: TestClient) -> None:
    response = client.post(
        "/predict", json={"text": "great app", "split": True}, headers=auth()
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openrouter"
    assert body["split_fallback"] is False
    assert len(body["sub_comments"]) == 2
    first = body["sub_comments"][0]
    assert first["text"] == "great app (part 1)"
    assert first["topic"] == "function"
    assert -1.0 <= first["sentiment"] <= 1.0
    assert "sentiment_raw" in first


def test_predict_rejects_an_empty_review(client: TestClient) -> None:
    response = client.post("/predict", json={"text": ""}, headers=auth())
    assert response.status_code == 422


def test_predict_falls_back_to_the_whole_review_without_a_provider_key(
    make_client, aws_stack: ServiceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    # scripts/llm_api raises at import when the provider key is missing.  A
    # None entry in sys.modules makes that import fail the same way.
    monkeypatch.setitem(sys.modules, "scripts.llm_api", None)
    with make_client(splitter=Splitter(aws_stack.settings)) as client:
        response = client.post(
            "/predict", json={"text": "great app", "split": True}, headers=auth()
        )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "none"
    assert body["split_fallback"] is True
    assert len(body["sub_comments"]) == 1
    assert body["sub_comments"][0]["text"] == "great app"


def test_the_rate_limit_returns_429_with_a_retry_after(make_client) -> None:
    with make_client(settings={"rate_limit_per_minute": 1}) as client:
        first = client.post("/predict", json={"text": "one"}, headers=auth())
        second = client.post("/predict", json={"text": "two"}, headers=auth())
    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1


def test_predict_returns_503_while_the_model_is_loading(make_client) -> None:
    class LoadingRuntime:
        ready = False
        load_error = None

        def start(self) -> None:
            pass

        def predict(self, texts):  # pragma: no cover - never reached
            raise AssertionError("the model is not ready")

    with make_client(runtime=LoadingRuntime()) as client:
        response = client.post("/predict", json={"text": "one"}, headers=auth())
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"


# --- jobs -----------------------------------------------------------------


def test_create_job_accepts_a_csv_and_enqueues_only_a_reference(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    response = submit(client)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert response.json()["total_reviews"] == 2

    keys = [
        entry["Key"]
        for entry in aws_stack.s3.list_objects_v2(Bucket=BUCKET).get("Contents", [])
    ]
    assert f"inputs/{job_id}/reviews.csv" in keys

    messages = aws_stack.queue.receive(wait_seconds=0)
    assert len(messages) == 1
    body = json.loads(messages[0]["Body"])
    assert set(body) == {"job_id", "input_key", "options"}
    assert body["job_id"] == job_id
    # No review text and no CSV body travel through the queue.
    assert "great" not in messages[0]["Body"]


def test_create_job_records_a_pending_job(client: TestClient, aws_stack: ServiceContext) -> None:
    job_id = submit(client).json()["job_id"]
    record = aws_stack.jobs.get(job_id)
    assert record["status"] == PENDING
    assert record["total_reviews"] == 2
    assert record["options"] == {"split": True}


def test_create_job_rejects_a_csv_without_a_content_column(client: TestClient) -> None:
    response = submit(client, b"text,score\nhello,5\n")
    assert response.status_code == 422
    assert "content" in response.json()["detail"]


def test_create_job_rejects_a_csv_with_no_rows(client: TestClient) -> None:
    response = submit(client, b"content,score\n")
    assert response.status_code == 422
    assert "no review rows" in response.json()["detail"]


def test_create_job_rejects_more_rows_than_the_job_limit(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    payload = b"content\n" + b"a review\n" * 1001
    response = submit(client, payload)
    assert response.status_code == 422
    assert "over the 1000 review limit" in response.json()["detail"]


def test_create_job_rejects_an_upload_over_the_byte_limit(make_client) -> None:
    with make_client(settings={"max_multipart_upload_bytes": 1024}) as client:
        response = submit(client, b"content\n" + b"x" * 2000 + b"\n")
    assert response.status_code == 413


def test_the_default_limits_match_the_documented_contract(settings) -> None:
    assert settings.max_upload_bytes == 50 * 1024 * 1024
    # The multipart route carries the CSV through the API Gateway, whose 10 MB
    # payload cap cannot be raised.
    assert settings.max_multipart_upload_bytes == 8 * 1024 * 1024
    assert settings.max_reviews_per_job == 1000
    assert settings.worker_chunk_size == 25
    assert settings.bert_batch_size == 64


def test_a_failed_enqueue_is_reported_and_marks_the_job_failed(
    client: TestClient, aws_stack: ServiceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**_kwargs):
        raise RuntimeError("sqs is down")

    monkeypatch.setattr(aws_stack.queue, "send", boom)
    response = submit(client)
    assert response.status_code == 503

    # The job was recorded, and it is recorded as failed — never as success.
    keys = [
        entry["Key"]
        for entry in aws_stack.s3.list_objects_v2(Bucket=BUCKET).get("Contents", [])
    ]
    job_id = keys[0].split("/")[1]
    assert aws_stack.jobs.get(job_id)["status"] == FAILED


def test_job_status_reports_progress(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    job_id = submit(client).json()["job_id"]
    aws_stack.jobs.update_progress(
        job_id=job_id,
        completed_reviews=2,
        completed_chunks=[0],
        lease_expires_at=9999999999.0,
    )
    response = client.get(f"/jobs/{job_id}", headers=auth())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == PENDING
    assert body["completed_reviews"] == 2
    assert body["total_reviews"] == 2


def test_job_status_404s_for_an_unknown_job(client: TestClient) -> None:
    response = client.get("/jobs/does-not-exist", headers=auth())
    assert response.status_code == 404


def test_the_result_is_refused_before_the_job_completes(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    job_id = submit(client).json()["job_id"]
    response = client.get(f"/jobs/{job_id}/result", headers=auth())
    assert response.status_code == 409


def test_the_result_is_a_presigned_url_after_the_job_completes(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    job_id = submit(client).json()["job_id"]
    result_key = f"results/{job_id}/results.csv"
    aws_stack.s3.put_object(
        Bucket=BUCKET,
        Key=result_key,
        Body=b"content,bert_topic\nhello,function\n",
        ContentType="text/csv",
    )
    aws_stack.jobs.complete(
        job_id=job_id, result_key=result_key, completed_reviews=2
    )

    response = client.get(f"/jobs/{job_id}/result", headers=auth())
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["expires_in"] == 900
    assert "X-Amz-Signature" in body["url"]
    assert "results.csv" in body["url"]


def test_the_completed_result_is_readable_from_storage(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    job_id = submit(client).json()["job_id"]
    result_key = f"results/{job_id}/results.csv"
    aws_stack.s3.put_object(
        Bucket=BUCKET, Key=result_key, Body=b"content\nhello\n", ContentType="text/csv"
    )
    aws_stack.jobs.complete(
        job_id=job_id, result_key=result_key, completed_reviews=2
    )
    body = aws_stack.s3.get_object(Bucket=BUCKET, Key=result_key)["Body"].read()
    assert b"hello" in body


# --- the presigned upload flow ---------------------------------------------
#
# The API Gateway caps a request payload at 10 MB, so a large CSV goes straight
# to the private bucket through a presigned URL and never through the gateway.


def upload_ticket(client: TestClient) -> dict:
    response = client.post(
        "/jobs/uploads", json={"filename": "batch.csv"}, headers=auth()
    )
    assert response.status_code == 201
    return response.json()


def put_upload(aws_stack: ServiceContext, ticket: dict, payload: bytes = VALID_CSV) -> None:
    """Stand in for the client uploading the CSV to the presigned URL."""
    aws_stack.s3.put_object(
        Bucket=BUCKET, Key=ticket["input_key"], Body=payload, ContentType="text/csv"
    )


def test_the_upload_routes_require_a_token(client: TestClient) -> None:
    assert client.post("/jobs/uploads", json={}).status_code == 401
    assert (
        client.post("/jobs/uploads/abc123/finalize", json={}).status_code == 401
    )


def test_the_upload_ticket_is_a_presigned_url_for_a_server_chosen_key(
    client: TestClient,
) -> None:
    ticket = upload_ticket(client)
    assert ticket["input_key"] == f"inputs/{ticket['upload_id']}/reviews.csv"
    assert "X-Amz-Signature" in ticket["upload_url"]
    assert ticket["expires_in"] == 900
    # The key is the server's, and it is scoped to the input prefix.
    assert ticket["input_key"].startswith("inputs/")


def test_finalize_creates_the_job_from_the_uploaded_object(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    ticket = upload_ticket(client)
    put_upload(aws_stack, ticket)

    response = client.post(
        f"/jobs/uploads/{ticket['upload_id']}/finalize",
        json={"options": {"split": True}},
        headers=auth(),
    )
    assert response.status_code == 202
    body = response.json()
    assert body["total_reviews"] == 2
    assert body["status"] == "PENDING"

    # The upload id is the job id, so the same object cannot become two jobs.
    assert body["job_id"] == ticket["upload_id"]

    record = aws_stack.jobs.get(body["job_id"])
    assert record["input_key"] == ticket["input_key"]
    assert record["total_reviews"] == 2

    messages = aws_stack.queue.receive(wait_seconds=0)
    assert len(messages) == 1
    assert json.loads(messages[0]["Body"])["job_id"] == body["job_id"]


def test_finalize_without_an_uploaded_object_is_refused(client: TestClient) -> None:
    ticket = upload_ticket(client)
    response = client.post(
        f"/jobs/uploads/{ticket['upload_id']}/finalize", json={}, headers=auth()
    )
    assert response.status_code == 409
    assert "not received" in response.json()["detail"]


def test_finalize_rejects_a_csv_without_a_content_column(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    ticket = upload_ticket(client)
    put_upload(aws_stack, ticket, b"text,score\nhello,5\n")
    response = client.post(
        f"/jobs/uploads/{ticket['upload_id']}/finalize", json={}, headers=auth()
    )
    assert response.status_code == 422
    assert "content" in response.json()["detail"]


def test_finalize_rejects_an_upload_over_the_byte_limit(
    make_client, aws_stack: ServiceContext
) -> None:
    with make_client(settings={"max_upload_bytes": 1024}) as client:
        ticket = upload_ticket(client)
        put_upload(aws_stack, ticket, b"content\n" + b"x" * 2000 + b"\n")
        response = client.post(
            f"/jobs/uploads/{ticket['upload_id']}/finalize", json={}, headers=auth()
        )
    assert response.status_code == 413


def test_finalize_rejects_an_upload_id_that_is_not_one(
    client: TestClient,
) -> None:
    """The upload id becomes part of an S3 key, so it is not free text."""
    response = client.post(
        "/jobs/uploads/not-a-uuid/finalize", json={}, headers=auth()
    )
    assert response.status_code == 422
    assert "upload id" in response.json()["detail"]


def test_finalize_refuses_a_traversal_attempt(client: TestClient) -> None:
    """A crafted upload id must not be able to point at another object."""
    response = client.post(
        "/jobs/uploads/..%2F..%2Fresults%2Fx/finalize", json={}, headers=auth()
    )
    assert response.status_code in (404, 422)


def test_finalizing_the_same_upload_twice_is_refused(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    ticket = upload_ticket(client)
    put_upload(aws_stack, ticket)
    first = client.post(
        f"/jobs/uploads/{ticket['upload_id']}/finalize", json={}, headers=auth()
    )
    assert first.status_code == 202
    second = client.post(
        f"/jobs/uploads/{ticket['upload_id']}/finalize", json={}, headers=auth()
    )
    assert second.status_code == 409
    assert "already finalized" in second.json()["detail"]


def test_a_failed_enqueue_during_finalize_marks_the_job_failed(
    client: TestClient, aws_stack: ServiceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**_kwargs):
        raise RuntimeError("sqs is down")

    monkeypatch.setattr(aws_stack.queue, "send", boom)
    ticket = upload_ticket(client)
    put_upload(aws_stack, ticket)
    response = client.post(
        f"/jobs/uploads/{ticket['upload_id']}/finalize", json={}, headers=auth()
    )
    assert response.status_code == 503
    assert aws_stack.jobs.get(ticket["upload_id"])["status"] == FAILED


# --- the prediction deadline -----------------------------------------------


def test_predict_falls_back_to_the_whole_review_when_the_split_overruns_the_deadline(
    make_client, aws_stack: ServiceContext
) -> None:
    """The gateway answers 504 after 30 seconds; the API must answer first.

    The provider call is abandoned, not cancelled — it is already paid for —
    but the caller gets the whole review instead of waiting for it.
    """
    import time as time_module

    def slow_generate(text: str, timeout=None):
        time_module.sleep(0.5)
        return [SimpleNamespace(sub_comment=text, topic="other", sentiment=0.1)]

    with make_client(
        settings={"predict_deadline_seconds": 0.05},
        splitter=Splitter(aws_stack.settings, generate=slow_generate),
    ) as client:
        response = client.post(
            "/predict", json={"text": "great app", "split": True}, headers=auth()
        )

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "none"
    assert body["split_fallback"] is True
    assert len(body["sub_comments"]) == 1
    assert body["sub_comments"][0]["text"] == "great app"


def test_the_splitter_passes_its_budget_to_the_provider(
    make_client, aws_stack: ServiceContext
) -> None:
    """The provider call is given the configured budget, not an open-ended one."""
    seen: dict[str, object] = {}

    def recording_generate(text: str, timeout=None):
        seen["timeout"] = timeout
        return [SimpleNamespace(sub_comment=text, topic="other", sentiment=0.1)]

    settings = dataclasses.replace(aws_stack.settings, predict_timeout_seconds=7)
    with make_client(
        settings={"predict_timeout_seconds": 7},
        splitter=Splitter(settings, generate=recording_generate),
    ) as client:
        client.post("/predict", json={"text": "great app"}, headers=auth())

    assert seen["timeout"] == 7
