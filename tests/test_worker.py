"""The worker: chunk checkpoints, resume, duplicate handling and the DLQ path."""

from __future__ import annotations

import dataclasses
import json
import logging
import time

import pytest
from fastapi.testclient import TestClient

from service.batch_io import read_result_csv
from service.inference import Splitter
from service.jobs import COMPLETED, FAILED, RETRYABLE
from service.server import ServiceContext
from service.worker import Heartbeat, JobWorker

from .conftest import API_TOKEN, BUCKET

THREE_REVIEWS = b"content\nfirst review\nsecond review\nthird review\n"


def auth(token: str = API_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def submit(client: TestClient, payload: bytes = THREE_REVIEWS) -> str:
    response = client.post(
        "/jobs",
        files={"file": ("reviews.csv", payload, "text/csv")},
        data={"options": json.dumps({"split": True})},
        headers=auth(),
    )
    assert response.status_code == 202
    return response.json()["job_id"]


def build_worker(aws_stack: ServiceContext, **settings_overrides) -> JobWorker:
    settings = dataclasses.replace(aws_stack.settings, **settings_overrides)
    return JobWorker(
        settings=settings,
        jobs=aws_stack.jobs,
        queue=aws_stack.queue,
        s3=aws_stack.s3,
        runtime=aws_stack.runtime,
        splitter=aws_stack.splitter,
        worker_id="worker-test",
    )


class CountingSplitter:
    """Wraps a splitter and counts how many reviews it was asked to split.

    The count is what proves a resumed job did not pay for the same LLM calls
    twice.
    """

    def __init__(self, inner: Splitter) -> None:
        self.inner = inner
        self.calls = 0

    @property
    def provider(self) -> str:
        return self.inner.provider

    def split(self, text: str, *, enabled: bool = True):
        self.calls += 1
        return self.inner.split(text, enabled=enabled)


def test_the_worker_completes_a_job_and_writes_the_result(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    job_id = submit(client)
    worker = build_worker(aws_stack)
    message = aws_stack.queue.receive(wait_seconds=0)[0]

    worker.handle_message(message)

    record = aws_stack.jobs.get(job_id)
    assert record["status"] == COMPLETED
    assert record["completed_reviews"] == 3

    body = aws_stack.s3.get_object(Bucket=BUCKET, Key=record["result_key"])["Body"].read()
    rows = read_result_csv(body)
    # Three reviews, two sub-comments each.
    assert len(rows) == 6
    assert rows[0]["source_row_index"] == "0"
    assert rows[0]["sub_comment_index"] == "0"
    assert rows[0]["bert_topic"] == "function"
    assert rows[0]["split_fallback"] == "False"
    assert "bert_sentiment_raw" in rows[0]


def test_the_message_is_deleted_only_after_the_job_succeeds(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    submit(client)
    worker = build_worker(aws_stack)
    message = aws_stack.queue.receive(wait_seconds=0)[0]
    worker.handle_message(message)
    assert aws_stack.queue.receive(wait_seconds=0) == []


def test_the_input_columns_are_carried_into_the_result(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    submit(client, b"content,score,date\nhello,3,2025-01-15\n")
    worker = build_worker(aws_stack)
    message = aws_stack.queue.receive(wait_seconds=0)[0]
    worker.handle_message(message)

    record = aws_stack.jobs.get(json.loads(message["Body"])["job_id"])
    body = aws_stack.s3.get_object(Bucket=BUCKET, Key=record["result_key"])["Body"].read()
    rows = read_result_csv(body)
    assert rows[0]["score"] == "3"
    assert rows[0]["date"] == "2025-01-15"


def test_a_failed_chunk_is_not_repeated_on_the_next_attempt(
    client: TestClient, aws_stack: ServiceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    counting = CountingSplitter(aws_stack.splitter)
    settings = dataclasses.replace(aws_stack.settings, worker_chunk_size=1)
    worker = JobWorker(
        settings=settings,
        jobs=aws_stack.jobs,
        queue=aws_stack.queue,
        s3=aws_stack.s3,
        runtime=aws_stack.runtime,
        splitter=counting,
        worker_id="worker-test",
    )

    job_id = submit(client)
    message = aws_stack.queue.receive(wait_seconds=0)[0]

    original = worker._label_chunk  # noqa: SLF001 - the failure is injected here
    chunk_calls = {"n": 0}

    def flaky(chunk, split_enabled):
        chunk_calls["n"] += 1
        if chunk_calls["n"] == 2:
            raise RuntimeError("the second chunk blew up")
        return original(chunk, split_enabled)

    monkeypatch.setattr(worker, "_label_chunk", flaky)
    worker.handle_message(message)

    record = aws_stack.jobs.get(job_id)
    assert record["status"] == RETRYABLE
    assert record["completed_chunks"] == [0]
    # Only chunk 0 was split: the failure hit chunk 1 before its split ran.
    assert counting.calls == 1

    # The retry resumes from the last committed chunk: chunk 0 is not split
    # again, so the splitter is called for chunks 1 and 2 only.
    monkeypatch.setattr(worker, "_label_chunk", original)
    worker.handle_message(message)

    record = aws_stack.jobs.get(job_id)
    assert record["status"] == COMPLETED
    assert record["completed_chunks"] == [0, 1, 2]
    assert counting.calls == 3

    body = aws_stack.s3.get_object(Bucket=BUCKET, Key=record["result_key"])["Body"].read()
    assert len(read_result_csv(body)) == 6


def test_a_duplicate_of_a_completed_job_is_discarded(
    client: TestClient, aws_stack: ServiceContext
) -> None:
    job_id = submit(client)
    worker = build_worker(aws_stack)
    message = aws_stack.queue.receive(wait_seconds=0)[0]

    worker.handle_message(message)
    # The same message turns up again.
    worker.handle_message(message)

    record = aws_stack.jobs.get(job_id)
    assert record["status"] == COMPLETED
    # The duplicate did not claim the job, so no second attempt was counted.
    assert record["attempts"] == 1
    assert aws_stack.queue.receive(wait_seconds=0) == []


def test_a_job_that_keeps_failing_is_marked_failed_and_stays_on_the_queue(
    client: TestClient, aws_stack: ServiceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = submit(client)
    worker = build_worker(aws_stack)
    message = aws_stack.queue.receive(wait_seconds=0)[0]

    def boom(*_args, **_kwargs):
        raise RuntimeError("the worker keeps failing")

    monkeypatch.setattr(worker, "process_job", boom)

    for _ in range(3):
        worker.handle_message(message)

    record = aws_stack.jobs.get(job_id)
    assert record["status"] == FAILED
    assert record["attempts"] == 3

    # The message was never deleted, so SQS redelivers it and — past
    # maxReceiveCount — moves it to the DLQ.
    aws_stack.queue.set_visibility(message["ReceiptHandle"], 0)
    assert len(aws_stack.queue.receive(wait_seconds=0)) == 1


def test_the_heartbeat_renews_the_visibility_and_the_lease(
    aws_stack: ServiceContext, caplog: pytest.LogCaptureFixture
) -> None:
    job_id = "heartbeat-job"
    aws_stack.jobs.create(
        job_id=job_id,
        input_key=f"inputs/{job_id}/reviews.csv",
        total_reviews=1,
        options={"split": True},
        ttl_days=7,
    )
    aws_stack.jobs.claim(job_id=job_id, owner="worker-test", lease_seconds=1)
    before = aws_stack.jobs.get(job_id)["lease_expires_at"]

    settings = dataclasses.replace(
        aws_stack.settings, heartbeat_interval=1, job_lease_seconds=60
    )
    heartbeat = Heartbeat(
        settings=settings,
        queue=aws_stack.queue,
        jobs=aws_stack.jobs,
        receipt_handle="receipt-1",
        job_id=job_id,
        owner="worker-test",
    )
    with caplog.at_level(logging.INFO, logger="service.worker"):
        heartbeat.start()
        time.sleep(1.5)
        heartbeat.stop()

    assert aws_stack.jobs.get(job_id)["lease_expires_at"] > before
    assert any(record.message == "job_heartbeat" for record in caplog.records)
