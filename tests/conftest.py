"""Shared fixtures for the service tests.

The AWS-facing tests run against moto's in-memory S3, SQS and DynamoDB rather
than against hand-written fakes.  The behaviour under test — a conditional
update that must not fire twice, a lease that expires, a message that is only
deleted on success — is DynamoDB's and SQS's behaviour, and moto implements it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Iterator

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from service.batch_io import build_s3_client  # noqa: E402
from service.inference import ModelRuntime, Splitter  # noqa: E402
from service.jobs import JobQueue, JobStore  # noqa: E402
from service.server import ServiceContext, create_app  # noqa: E402
from service.settings import Settings  # noqa: E402

REGION = "us-east-1"
BUCKET = "reviewassist-jobs-test"
QUEUE_NAME = "reviewassist-jobs-test"
TABLE_NAME = "reviewassist-jobs-test"
API_TOKEN = "test-token"


def stub_predict(texts: list[str]) -> list[dict[str, Any]]:
    """A deterministic stand-in for the BERT head.

    The tests that use it are about the job pipeline — chunking, progress,
    leases, the result schema — not about BERT's accuracy, which is measured
    against the corpus elsewhere.
    """
    return [
        {
            "text": text,
            "topic": "function",
            "topic_conf": 0.9,
            "sentiment": 0.5,
            "sentiment_raw": 0.5,
        }
        for text in texts
    ]


def stub_generate(text: str) -> list[SimpleNamespace]:
    """A deterministic stand-in for the LLM splitter: two sub-comments."""
    return [
        SimpleNamespace(sub_comment=f"{text} (part 1)", topic="other", sentiment=0.4),
        SimpleNamespace(sub_comment=f"{text} (part 2)", topic="other", sentiment=-0.4),
    ]


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)


@pytest.fixture
def aws(aws_env: None) -> Iterator[None]:
    with mock_aws():
        yield


@pytest.fixture
def settings() -> Settings:
    return Settings(
        model_dir=ROOT_DIR / "models" / "saved_bert_model",
        model_device="cpu",
        aws_region=REGION,
        job_bucket=BUCKET,
        queue_url="",
        dlq_url="",
        jobs_table=TABLE_NAME,
        api_token=API_TOKEN,
        llm_provider="openrouter",
        max_upload_bytes=50 * 1024 * 1024,
        max_reviews_per_job=1000,
        worker_chunk_size=25,
        bert_batch_size=64,
        sqs_visibility_timeout=600,
        retry_visibility_seconds=30,
        heartbeat_interval=60,
        job_lease_seconds=600,
        job_ttl_days=7,
        result_url_ttl_seconds=900,
        max_inflight_predicts=4,
        rate_limit_per_minute=120,
        log_level="INFO",
    )


@pytest.fixture
def aws_stack(aws: None, settings: Settings) -> ServiceContext:
    """The service's collaborators, wired to moto's AWS."""
    s3 = build_s3_client(REGION)
    s3.create_bucket(Bucket=BUCKET)
    queue_url = boto3.client("sqs", region_name=REGION).create_queue(
        QueueName=QUEUE_NAME
    )["QueueUrl"]

    table = boto3.resource("dynamodb", region_name=REGION).create_table(
        TableName=TABLE_NAME,
        KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
        AttributeDefinitions=[
            {"AttributeName": "job_id", "AttributeType": "S"}
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()

    real_settings = dataclasses.replace(settings, queue_url=queue_url)
    runtime = ModelRuntime(real_settings, predict=stub_predict)
    runtime.start()
    runtime.wait_until_ready(timeout=10)

    return ServiceContext(
        settings=real_settings,
        runtime=runtime,
        splitter=Splitter(real_settings, generate=stub_generate),
        jobs=JobStore(TABLE_NAME, REGION),
        queue=JobQueue(queue_url, region=REGION),
        s3=s3,
    )


@pytest.fixture
def make_context(aws_stack: ServiceContext):
    """Build a variant of the context with different settings or collaborators."""

    def _make(**overrides: Any) -> ServiceContext:
        settings_overrides = overrides.pop("settings", None)
        if settings_overrides:
            overrides["settings"] = dataclasses.replace(
                aws_stack.settings, **settings_overrides
            )
        return dataclasses.replace(aws_stack, **overrides)

    return _make


@pytest.fixture
def make_client(aws_stack: ServiceContext):
    """Build a TestClient over a variant of the context.

    The client is a context manager so the ASGI app runs on one event loop for
    its whole lifetime.
    """

    @contextlib.contextmanager
    def _make(**overrides: Any) -> Iterator[TestClient]:
        settings_overrides = overrides.pop("settings", None)
        if settings_overrides:
            overrides["settings"] = dataclasses.replace(
                aws_stack.settings, **settings_overrides
            )
        context = dataclasses.replace(aws_stack, **overrides)
        with TestClient(create_app(context)) as client:
            yield client

    return _make


@pytest.fixture
def client(aws_stack: ServiceContext) -> Iterator[TestClient]:
    with TestClient(create_app(aws_stack)) as test_client:
        yield test_client
