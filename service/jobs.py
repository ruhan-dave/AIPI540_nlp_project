"""Job state in DynamoDB and the SQS queue that carries job references.

The DynamoDB item is the single source of truth for a job's lifecycle:
``PENDING`` → ``RUNNING`` → ``COMPLETED`` / ``FAILED``, with ``RETRYABLE`` for a
job whose worker died but which may be picked up again.  Only job ids, status,
counters, the lease and S3 keys live here — never review text and never the CSV
body.

The queue message is deliberately tiny: ``{job_id, input_key, options}``.  The
review text stays in S3, where it is private, encrypted and expired after seven
days.
"""

from __future__ import annotations

import json
import logging
import time
from decimal import Decimal
from typing import Any, Mapping

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

PENDING = "PENDING"
RUNNING = "RUNNING"
RETRYABLE = "RETRYABLE"
COMPLETED = "COMPLETED"
FAILED = "FAILED"

#: A job in one of these states is never picked up again.
TERMINAL_STATUSES = (COMPLETED, FAILED)

#: The statuses a worker may claim.
CLAIMABLE_STATUSES = (PENDING, RETRYABLE)

#: DynamoDB's name for a conditional-update failure.
_CONDITIONAL_FAILURE = "ConditionalCheckFailedException"


def _strip_none(values: Mapping[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values — the DynamoDB resource layer rejects them."""
    return {key: value for key, value in values.items() if value is not None}


def _ts(value: float | int) -> Decimal:
    """DynamoDB has no float type: timestamps go in as decimals."""
    return Decimal(str(value))


class JobStore:
    """DynamoDB-backed job records."""

    def __init__(self, table_name: str, region: str, *, table: Any = None) -> None:
        self.table_name = table_name
        self.region = region
        self._table = (
            table
            if table is not None
            else boto3.resource("dynamodb", region_name=region).Table(table_name)
        )

    # -- reads -------------------------------------------------------------

    def get(self, job_id: str) -> dict[str, Any] | None:
        """The job record, or ``None`` when the job is unknown."""
        response = self._table.get_item(Key={"job_id": job_id})
        return response.get("Item")

    # -- writes ------------------------------------------------------------

    def create(
        self,
        *,
        job_id: str,
        input_key: str,
        total_reviews: int,
        options: Mapping[str, Any],
        ttl_days: int,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Create the ``PENDING`` record for a newly accepted job."""
        now = time.time() if now is None else now
        item = _strip_none(
            {
                "job_id": job_id,
                "status": PENDING,
                "total_reviews": int(total_reviews),
                "completed_reviews": 0,
                "input_key": input_key,
                "options": {"split": bool(options.get("split", True))},
                "created_at": _ts(now),
                "updated_at": _ts(now),
                "expires_at": _ts(now + ttl_days * 86400),
                "completed_chunks": [],
                "attempts": 0,
            }
        )
        self._table.put_item(
            Item=item, ConditionExpression="attribute_not_exists(job_id)"
        )
        return item

    def claim(
        self,
        *,
        job_id: str,
        owner: str,
        now: float | None = None,
        lease_seconds: int = 600,
    ) -> dict[str, Any] | None:
        """Take ownership of a job.

        A job may be claimed when it is ``PENDING``, explicitly ``RETRYABLE``, or
        ``RUNNING`` with an expired lease.  A completed or failed job is never
        claimed, and a job with a live lease is left alone so a duplicate waits
        rather than starting a second worker.

        Returns the updated record, or ``None`` when the job is not claimable.
        """
        now = time.time() if now is None else now
        try:
            response = self._table.update_item(
                Key={"job_id": job_id},
                UpdateExpression=(
                    "SET #status = :running, lease_owner = :owner, "
                    "lease_expires_at = :lease, updated_at = :now, "
                    "attempts = if_not_exists(attempts, :zero) + :one"
                ),
                ConditionExpression=(
                    "#status = :pending OR #status = :retryable OR "
                    "(#status = :running AND (attribute_not_exists(lease_expires_at) "
                    "OR lease_expires_at < :now))"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":pending": PENDING,
                    ":retryable": RETRYABLE,
                    ":running": RUNNING,
                    ":owner": owner,
                    ":lease": _ts(now + lease_seconds),
                    ":now": _ts(now),
                    ":zero": 0,
                    ":one": 1,
                },
                ReturnValues="ALL_NEW",
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == _CONDITIONAL_FAILURE:
                return None
            raise
        return response.get("Attributes")

    def update_progress(
        self,
        *,
        job_id: str,
        completed_reviews: int,
        completed_chunks: list[int],
        lease_expires_at: float,
        now: float | None = None,
    ) -> None:
        """Record the progress of a running job and extend its lease."""
        now = time.time() if now is None else now
        self._table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=(
                "SET completed_reviews = :done, completed_chunks = :chunks, "
                "lease_expires_at = :lease, updated_at = :now"
            ),
            ConditionExpression="attribute_exists(job_id)",
            ExpressionAttributeValues={
                ":done": int(completed_reviews),
                ":chunks": [int(index) for index in completed_chunks],
                ":lease": _ts(lease_expires_at),
                ":now": _ts(now),
            },
        )

    def renew_lease(
        self,
        *,
        job_id: str,
        owner: str,
        lease_expires_at: float,
        now: float | None = None,
    ) -> bool:
        """Extend the lease of a job this worker owns.

        Returns ``False`` when the lease was taken over by somebody else.
        """
        now = time.time() if now is None else now
        try:
            self._table.update_item(
                Key={"job_id": job_id},
                UpdateExpression="SET lease_expires_at = :lease, updated_at = :now",
                ConditionExpression="lease_owner = :owner",
                ExpressionAttributeValues={
                    ":lease": _ts(lease_expires_at),
                    ":now": _ts(now),
                    ":owner": owner,
                },
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == _CONDITIONAL_FAILURE:
                return False
            raise
        return True

    def complete(
        self,
        *,
        job_id: str,
        result_key: str,
        completed_reviews: int,
        now: float | None = None,
    ) -> None:
        """Mark a job as completed and record where its result lives."""
        now = time.time() if now is None else now
        self._table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=(
                "SET #status = :completed, result_key = :result, "
                "completed_reviews = :done, updated_at = :now "
                "REMOVE lease_owner, lease_expires_at, #error"
            ),
            ConditionExpression="attribute_exists(job_id)",
            ExpressionAttributeNames={"#status": "status", "#error": "error"},
            ExpressionAttributeValues={
                ":completed": COMPLETED,
                ":result": result_key,
                ":done": int(completed_reviews),
                ":now": _ts(now),
            },
        )

    def release(
        self,
        *,
        job_id: str,
        error: str,
        max_attempts: int = 3,
        now: float | None = None,
    ) -> str:
        """Give a failed job back for another attempt, or mark it failed.

        Returns the new status: ``RETRYABLE`` while attempts remain, ``FAILED``
        once the attempt budget is spent.
        """
        now = time.time() if now is None else now
        item = self.get(job_id) or {}
        attempts = int(item.get("attempts", 0))
        status = FAILED if attempts >= max_attempts else RETRYABLE
        self._table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=(
                "SET #status = :status, #error = :error, updated_at = :now "
                "REMOVE lease_owner, lease_expires_at"
            ),
            ConditionExpression="attribute_exists(job_id)",
            ExpressionAttributeNames={"#status": "status", "#error": "error"},
            ExpressionAttributeValues={
                ":status": status,
                ":error": error or "unknown error",
                ":now": _ts(now),
            },
        )
        return status

    def fail(self, *, job_id: str, error: str, now: float | None = None) -> None:
        """Mark a job as failed for good."""
        now = time.time() if now is None else now
        self._table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=(
                "SET #status = :failed, #error = :error, updated_at = :now "
                "REMOVE lease_owner, lease_expires_at"
            ),
            ConditionExpression="attribute_exists(job_id)",
            ExpressionAttributeNames={"#status": "status", "#error": "error"},
            ExpressionAttributeValues={
                ":failed": FAILED,
                ":error": error or "unknown error",
                ":now": _ts(now),
            },
        )


class JobQueue:
    """The SQS queue carrying job references."""

    def __init__(self, queue_url: str, *, client: Any = None, region: str = "") -> None:
        self.queue_url = queue_url
        self._client = (
            client
            if client is not None
            else boto3.client("sqs", region_name=region or None)
        )

    def send(
        self, *, job_id: str, input_key: str, options: Mapping[str, Any]
    ) -> str:
        """Enqueue a job reference.

        The message carries no review text and no CSV body — only the pointer to
        the private S3 object and the options to use.
        """
        message = {
            "job_id": job_id,
            "input_key": input_key,
            "options": {"split": bool(options.get("split", True))},
        }
        response = self._client.send_message(
            QueueUrl=self.queue_url, MessageBody=json.dumps(message)
        )
        return response.get("MessageId", "")

    def receive(
        self, *, wait_seconds: int = 20, visibility_timeout: int = 600
    ) -> list[dict[str, Any]]:
        """Long-poll for one or more job references."""
        response = self._client.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_seconds,
            VisibilityTimeout=visibility_timeout,
        )
        return list(response.get("Messages", []))

    def delete(self, receipt_handle: str) -> None:
        """Delete a message — only ever called after the job has succeeded."""
        self._client.delete_message(
            QueueUrl=self.queue_url, ReceiptHandle=receipt_handle
        )

    def set_visibility(self, receipt_handle: str, timeout: int) -> None:
        """Change how long a message stays invisible."""
        self._client.change_message_visibility(
            QueueUrl=self.queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=timeout,
        )
