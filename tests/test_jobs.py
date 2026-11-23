"""The DynamoDB job lifecycle: claims, leases, retries and the TTL."""

from __future__ import annotations

import time

import pytest

from service.jobs import COMPLETED, FAILED, PENDING, RETRYABLE, RUNNING, JobStore


@pytest.fixture
def store(aws_stack) -> JobStore:
    return aws_stack.jobs


def make_job(store: JobStore, job_id: str = "job-1") -> dict:
    return store.create(
        job_id=job_id,
        input_key=f"inputs/{job_id}/reviews.csv",
        total_reviews=3,
        options={"split": True},
        ttl_days=7,
    )


def test_create_starts_a_pending_job(store: JobStore) -> None:
    item = make_job(store)
    assert item["status"] == PENDING
    assert item["total_reviews"] == 3
    assert item["attempts"] == 0


def test_create_sets_a_ttl(store: JobStore) -> None:
    make_job(store)
    record = store.get("job-1")
    assert record["expires_at"] > time.time()


def test_claim_takes_ownership_and_counts_the_attempt(store: JobStore) -> None:
    make_job(store)
    claimed = store.claim(job_id="job-1", owner="worker-1")
    assert claimed is not None
    assert claimed["status"] == RUNNING
    assert claimed["attempts"] == 1
    assert claimed["lease_owner"] == "worker-1"


def test_a_live_lease_blocks_a_second_claim(store: JobStore) -> None:
    make_job(store)
    assert store.claim(job_id="job-1", owner="worker-1") is not None
    # A duplicate must not start a second worker.
    assert store.claim(job_id="job-1", owner="worker-2") is None


def test_an_expired_lease_can_be_claimed_again(store: JobStore) -> None:
    make_job(store)
    store.claim(job_id="job-1", owner="worker-1", now=1000.0, lease_seconds=10)
    reclaimed = store.claim(job_id="job-1", owner="worker-2", now=1100.0, lease_seconds=10)
    assert reclaimed is not None
    assert reclaimed["lease_owner"] == "worker-2"
    assert reclaimed["attempts"] == 2


def test_a_completed_job_is_never_claimed(store: JobStore) -> None:
    make_job(store)
    store.complete(job_id="job-1", result_key="results/job-1/results.csv", completed_reviews=3)
    assert store.claim(job_id="job-1", owner="worker-2") is None
    assert store.get("job-1")["status"] == COMPLETED


def test_a_failed_job_is_never_claimed(store: JobStore) -> None:
    make_job(store)
    store.fail(job_id="job-1", error="boom")
    assert store.claim(job_id="job-1", owner="worker-2") is None


def test_release_makes_the_job_retryable(store: JobStore) -> None:
    make_job(store)
    store.claim(job_id="job-1", owner="worker-1")
    assert store.release(job_id="job-1", error="boom", max_attempts=3) == RETRYABLE
    assert store.get("job-1")["error"] == "boom"
    # A released job can be picked up again.
    assert store.claim(job_id="job-1", owner="worker-2") is not None


def test_release_fails_the_job_once_the_attempt_budget_is_spent(store: JobStore) -> None:
    make_job(store)
    for attempt in range(3):
        store.claim(
            job_id="job-1",
            owner=f"worker-{attempt}",
            now=1000.0 + attempt * 100,
            lease_seconds=10,
        )
    assert store.release(job_id="job-1", error="boom", max_attempts=3) == FAILED
    assert store.get("job-1")["status"] == FAILED


def test_update_progress_records_the_chunk_checkpoints(store: JobStore) -> None:
    make_job(store)
    store.claim(job_id="job-1", owner="worker-1")
    store.update_progress(
        job_id="job-1",
        completed_reviews=25,
        completed_chunks=[0],
        lease_expires_at=time.time() + 600,
    )
    record = store.get("job-1")
    assert record["completed_reviews"] == 25
    assert record["completed_chunks"] == [0]


def test_renew_lease_only_works_for_the_owner(store: JobStore) -> None:
    make_job(store)
    store.claim(job_id="job-1", owner="worker-1")
    assert store.renew_lease(job_id="job-1", owner="worker-1", lease_expires_at=9999.0)
    assert not store.renew_lease(job_id="job-1", owner="worker-2", lease_expires_at=9999.0)


def test_complete_clears_the_lease(store: JobStore) -> None:
    make_job(store)
    store.claim(job_id="job-1", owner="worker-1")
    store.complete(job_id="job-1", result_key="results/job-1/results.csv", completed_reviews=3)
    record = store.get("job-1")
    assert record["status"] == COMPLETED
    assert record["result_key"] == "results/job-1/results.csv"
    assert "lease_owner" not in record
