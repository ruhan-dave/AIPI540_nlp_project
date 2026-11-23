"""The SQS worker: split and label review batches, chunk by chunk.

One SQS message is one job reference.  The worker claims the job in DynamoDB,
reads the input CSV from S3 in 25-review chunks, splits each review with the
LLM, labels the resulting sub-comments with BERT, writes a chunk result file to
S3 and records the progress.

Two things make a long job safe:

* The SQS visibility timeout (600 s) and the DynamoDB lease are renewed every
  60 seconds, so a job that takes 50 minutes is never handed to a second worker.
* Each chunk is committed to S3 before the progress is recorded, so a worker that
  dies resumes from the last committed chunk instead of repeating the LLM calls
  it already paid for.

A message is deleted only after the job has completed.  A failing job is left on
the queue so SQS redelivers it; after ``maxReceiveCount`` it reaches the DLQ and
the job record is marked ``FAILED``.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import signal
import threading
import time
import uuid
from typing import Any, Mapping, Sequence

from service.batch_io import (
    CONTENT_COLUMN,
    build_s3_client,
    chunked,
    iter_review_rows,
    read_object_bytes,
    result_columns,
    write_object_bytes,
    write_result_csv,
)
from service.inference import ModelRuntime, Splitter
from service.jobs import COMPLETED, FAILED, JobQueue, JobStore
from service.settings import Settings

logger = logging.getLogger(__name__)

#: Where a job's committed chunks live, and where its assembled result lives.
CHUNK_KEY_TEMPLATE = "chunks/{job_id}/{chunk_index:05d}.csv"
RESULT_KEY_TEMPLATE = "results/{job_id}/results.csv"

#: How many times a job may be attempted before it is marked failed.
MAX_ATTEMPTS = 3


class Heartbeat:
    """Renews the SQS visibility timeout and the DynamoDB lease while a job runs.

    The job takes far longer than the queue's visibility timeout — one LLM call
    per review — so the lease has to be renewed while the work is in flight, not
    only between chunks.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        queue: JobQueue,
        jobs: JobStore,
        receipt_handle: str,
        job_id: str,
        owner: str,
    ) -> None:
        self._settings = settings
        self._queue = queue
        self._jobs = jobs
        self._receipt_handle = receipt_handle
        self._job_id = job_id
        self._owner = owner
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"heartbeat-{self._job_id[:8]}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._settings.heartbeat_interval):
            visibility_ok = True
            try:
                self._queue.set_visibility(
                    self._receipt_handle, self._settings.sqs_visibility_timeout
                )
            except Exception as exc:
                visibility_ok = False
                logger.warning(
                    "heartbeat_visibility_failed",
                    extra={
                        "job_id": self._job_id,
                        "detail": f"{type(exc).__name__}: {exc}",
                    },
                )
            try:
                self._jobs.renew_lease(
                    job_id=self._job_id,
                    owner=self._owner,
                    lease_expires_at=time.time() + self._settings.job_lease_seconds,
                )
            except Exception as exc:
                logger.warning(
                    "heartbeat_lease_failed",
                    extra={
                        "job_id": self._job_id,
                        "detail": f"{type(exc).__name__}: {exc}",
                    },
                )
            logger.info(
                "job_heartbeat",
                extra={
                    "job_id": self._job_id,
                    "visibility_renewed": visibility_ok,
                    "visibility_timeout": self._settings.sqs_visibility_timeout,
                },
            )


class JobWorker:
    """Turns queued job references into completed, labelled result CSVs."""

    def __init__(
        self,
        *,
        settings: Settings,
        jobs: JobStore,
        queue: JobQueue,
        s3: Any,
        runtime: ModelRuntime,
        splitter: Splitter,
        worker_id: str | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self._settings = settings
        self._jobs = jobs
        self._queue = queue
        self._s3 = s3
        self._runtime = runtime
        self._splitter = splitter
        self._worker_id = worker_id or uuid.uuid4().hex[:12]
        self._max_attempts = max_attempts

    @property
    def runtime(self) -> ModelRuntime:
        """The model runtime this worker labels with."""
        return self._runtime

    # --- the loop ----------------------------------------------------------

    def run_forever(self, stop_event: threading.Event) -> None:
        """Long-poll the queue until ``stop_event`` is set."""
        logger.info(
            "worker_started",
            extra={"worker_id": self._worker_id, "queue_url": self._settings.queue_url},
        )
        while not stop_event.is_set():
            try:
                messages = self._queue.receive(
                    wait_seconds=20,
                    visibility_timeout=self._settings.sqs_visibility_timeout,
                )
            except Exception as exc:
                logger.error(
                    "receive_failed", extra={"detail": f"{type(exc).__name__}: {exc}"}
                )
                time.sleep(5)
                continue

            for message in messages:
                if stop_event.is_set():
                    break
                try:
                    self.handle_message(message)
                except Exception as exc:
                    # A malformed message must not take the worker down; it is
                    # logged and the message is left to expire into the DLQ.
                    logger.error(
                        "message_failed",
                        extra={"detail": f"{type(exc).__name__}: {exc}"},
                        exc_info=True,
                    )

    # --- one message -------------------------------------------------------

    def handle_message(self, message: Mapping[str, Any]) -> None:
        """Process one SQS message."""
        receipt_handle = message["ReceiptHandle"]
        body = json.loads(message["Body"])
        job_id = body["job_id"]
        input_key = body["input_key"]
        options = body.get("options") or {}
        split_enabled = bool(options.get("split", True))

        claimed = self._jobs.claim(
            job_id=job_id,
            owner=self._worker_id,
            lease_seconds=self._settings.job_lease_seconds,
        )
        if claimed is None:
            self._handle_unclaimed(job_id, receipt_handle)
            return

        started = time.time()
        heartbeat = Heartbeat(
            settings=self._settings,
            queue=self._queue,
            jobs=self._jobs,
            receipt_handle=receipt_handle,
            job_id=job_id,
            owner=self._worker_id,
        )
        heartbeat.start()
        try:
            self.process_job(claimed, input_key, split_enabled)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            status = self._jobs.release(
                job_id=job_id, error=detail, max_attempts=self._max_attempts
            )
            logger.error(
                "job_failed",
                extra={
                    "job_id": job_id,
                    "status": status,
                    "detail": detail,
                    "elapsed_seconds": round(time.time() - started, 3),
                },
            )
            # The message is deliberately NOT deleted: SQS redelivers it and,
            # once maxReceiveCount is reached, moves it to the DLQ.
            self._safe_set_visibility(receipt_handle)
            return
        finally:
            heartbeat.stop()

        self._queue.delete(receipt_handle)
        logger.info(
            "job_completed",
            extra={
                "job_id": job_id,
                "worker_id": self._worker_id,
                "elapsed_seconds": round(time.time() - started, 3),
            },
        )

    def _handle_unclaimed(self, job_id: str, receipt_handle: str) -> None:
        """A message for a job this worker may not claim."""
        current = self._jobs.get(job_id)
        status = (current or {}).get("status")
        if status in (COMPLETED, FAILED):
            # A duplicate of a finished job: drop it so it cannot be processed
            # a second time.
            self._queue.delete(receipt_handle)
            logger.info(
                "job_duplicate_discarded",
                extra={"job_id": job_id, "status": status},
            )
            return
        # Another worker holds a live lease.  Leave the message in flight so it
        # comes back later instead of starting a second worker now.
        logger.info(
            "job_duplicate_waiting",
            extra={"job_id": job_id, "status": status},
        )

    def _safe_set_visibility(self, receipt_handle: str) -> None:
        """Make a failed message visible again quickly, ignoring SQS errors."""
        try:
            self._queue.set_visibility(
                receipt_handle, self._settings.retry_visibility_seconds
            )
        except Exception as exc:
            logger.warning(
                "retry_visibility_failed",
                extra={"detail": f"{type(exc).__name__}: {exc}"},
            )

    # --- the job -----------------------------------------------------------

    def process_job(
        self, job: Mapping[str, Any], input_key: str, split_enabled: bool
    ) -> None:
        """Run a claimed job to completion."""
        job_id = job["job_id"]
        bucket = self._settings.job_bucket

        payload = read_object_bytes(self._s3, bucket, input_key)
        rows = list(iter_review_rows(payload))
        if not rows:
            raise ValueError("the job input has no review rows")
        columns = list(rows[0].keys())

        completed_chunks = {int(index) for index in job.get("completed_chunks", [])}
        completed_reviews = int(job.get("completed_reviews", 0))

        for chunk_index, chunk in enumerate(chunked(rows, self._settings.worker_chunk_size)):
            if chunk_index in completed_chunks:
                # Already committed by an earlier attempt — do not pay for the
                # LLM calls again.
                continue
            result_rows = self._label_chunk(chunk, split_enabled)
            body = write_result_csv(result_rows, result_columns(columns))
            write_object_bytes(
                self._s3,
                bucket,
                CHUNK_KEY_TEMPLATE.format(job_id=job_id, chunk_index=chunk_index),
                body,
            )
            completed_chunks.add(chunk_index)
            completed_reviews += len(chunk)
            self._jobs.update_progress(
                job_id=job_id,
                completed_reviews=completed_reviews,
                completed_chunks=sorted(completed_chunks),
                lease_expires_at=time.time() + self._settings.job_lease_seconds,
            )
            logger.info(
                "job_chunk_committed",
                extra={
                    "job_id": job_id,
                    "chunk": chunk_index,
                    "completed_reviews": completed_reviews,
                    "total_reviews": len(rows),
                    "provider": self._splitter.provider,
                },
            )

        result_key = RESULT_KEY_TEMPLATE.format(job_id=job_id)
        write_object_bytes(
            self._s3,
            bucket,
            result_key,
            self._assemble(job_id, columns, sorted(completed_chunks)),
        )
        self._jobs.complete(
            job_id=job_id,
            result_key=result_key,
            completed_reviews=completed_reviews,
        )

    def _label_chunk(
        self, chunk: Sequence[Mapping[str, Any]], split_enabled: bool
    ) -> list[dict[str, Any]]:
        """Split and label one chunk of reviews.

        The LLM is called once per review; the resulting sub-comments are then
        labelled by BERT in batches.
        """
        pending: list[tuple[int, str, bool]] = []
        for row_index, row in enumerate(chunk):
            text = row.get(CONTENT_COLUMN, "") or ""
            split = self._splitter.split(text, enabled=split_enabled)
            for sub_text in split.texts:
                pending.append((row_index, sub_text, split.fallback))

        predictions = self._runtime.predict([item[1] for item in pending])

        results: list[dict[str, Any]] = []
        counters: dict[int, int] = {}
        for (row_index, sub_text, fallback), prediction in zip(pending, predictions):
            row = dict(chunk[row_index])
            row.update(
                {
                    "sub_comment_index": counters.get(row_index, 0),
                    "sub_comment": sub_text,
                    "bert_topic": prediction["topic"],
                    "bert_topic_conf": prediction["topic_conf"],
                    "bert_sentiment": prediction["sentiment"],
                    "bert_sentiment_raw": prediction["sentiment_raw"],
                    "split_fallback": fallback,
                }
            )
            counters[row_index] = counters.get(row_index, 0) + 1
            results.append(row)
        return results

    def _assemble(
        self, job_id: str, columns: Sequence[str], chunk_indexes: Sequence[int]
    ) -> bytes:
        """Concatenate the committed chunk files into the final result CSV."""
        bucket = self._settings.job_bucket
        header = result_columns(columns)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer, fieldnames=header, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()

        for chunk_index in chunk_indexes:
            key = CHUNK_KEY_TEMPLATE.format(job_id=job_id, chunk_index=chunk_index)
            body = read_object_bytes(self._s3, bucket, key)
            reader = csv.DictReader(io.StringIO(body.decode("utf-8"), newline=""))
            for row in reader:
                writer.writerow({column: row.get(column, "") for column in header})
        return buffer.getvalue().encode("utf-8")


def build_worker(settings: Settings) -> JobWorker:
    """Assemble a worker from the settings."""
    runtime = ModelRuntime(settings)
    return JobWorker(
        settings=settings,
        jobs=JobStore(settings.jobs_table, settings.aws_region),
        queue=JobQueue(settings.queue_url, region=settings.aws_region),
        s3=build_s3_client(settings.aws_region),
        runtime=runtime,
        splitter=Splitter(settings),
    )


def main() -> None:
    """Run the worker until it is asked to stop."""
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    worker = build_worker(settings)
    runtime = worker.runtime
    runtime.start()
    if not runtime.wait_until_ready(timeout=600):
        logger.error("worker_model_not_ready", extra={"detail": runtime.load_error})
        raise SystemExit(1)

    stop = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("worker_stopping", extra={"signal": signum})
        stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    worker.run_forever(stop)


if __name__ == "__main__":
    main()
