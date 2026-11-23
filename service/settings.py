"""Runtime configuration for the ReviewAssist backend.

Every value comes from the environment.  In AWS the plain values ride on the ECS
task definition and the two secret values ride on its ``secrets`` block; locally
they come from the shell.  Nothing here touches the filesystem or the network at
import time.

The settings object is frozen: a task that starts with the wrong configuration
should fail at start-up, not half-way through a job.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Mapping

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "saved_bert_model"

MIB = 1024 * 1024
SECONDS_PER_DAY = 86400

VALID_PROVIDERS = ("openrouter", "gemini")


class ConfigurationError(RuntimeError):
    """The process is missing configuration it cannot run without."""


def _env_str(env: Mapping[str, str], name: str, default: str = "") -> str:
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _env_str(env, name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _env_str(env, name).lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigurationError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    """Everything the API and the worker need to know about their environment."""

    # Model
    model_dir: pathlib.Path
    model_device: str

    # AWS
    aws_region: str
    job_bucket: str
    queue_url: str
    dlq_url: str
    jobs_table: str

    # Auth and provider
    api_token: str
    llm_provider: str

    # Limits
    max_upload_bytes: int
    max_reviews_per_job: int
    worker_chunk_size: int
    bert_batch_size: int

    # Timing
    sqs_visibility_timeout: int
    retry_visibility_seconds: int
    heartbeat_interval: int
    job_lease_seconds: int
    job_ttl_days: int
    result_url_ttl_seconds: int

    # Saturation
    max_inflight_predicts: int
    rate_limit_per_minute: int

    log_level: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build the settings from the environment.

        Raises :class:`ConfigurationError` when a required value is missing or
        malformed — a service that starts with a wrong configuration and only
        notices at request time is worse than one that refuses to start.
        """
        env = os.environ if env is None else env
        settings = cls(
            model_dir=pathlib.Path(
                _env_str(env, "MODEL_DIR", str(DEFAULT_MODEL_DIR))
            ),
            model_device=_env_str(env, "MODEL_DEVICE", "cpu"),
            aws_region=_env_str(env, "AWS_REGION", "us-east-1"),
            job_bucket=_env_str(env, "JOB_BUCKET"),
            queue_url=_env_str(env, "QUEUE_URL"),
            dlq_url=_env_str(env, "DLQ_URL"),
            jobs_table=_env_str(env, "JOBS_TABLE"),
            api_token=_env_str(env, "API_TOKEN"),
            llm_provider=_env_str(env, "LLM_PROVIDER", "openrouter").lower(),
            max_upload_bytes=_env_int(env, "MAX_UPLOAD_BYTES", 50 * MIB),
            max_reviews_per_job=_env_int(env, "MAX_REVIEWS_PER_JOB", 1000),
            worker_chunk_size=_env_int(env, "WORKER_CHUNK_SIZE", 25),
            bert_batch_size=_env_int(env, "BERT_BATCH_SIZE", 64),
            sqs_visibility_timeout=_env_int(env, "SQS_VISIBILITY_TIMEOUT", 600),
            retry_visibility_seconds=_env_int(env, "RETRY_VISIBILITY_SECONDS", 30),
            heartbeat_interval=_env_int(env, "HEARTBEAT_INTERVAL", 60),
            job_lease_seconds=_env_int(env, "JOB_LEASE_SECONDS", 600),
            job_ttl_days=_env_int(env, "JOB_TTL_DAYS", 7),
            result_url_ttl_seconds=_env_int(env, "RESULT_URL_TTL_SECONDS", 900),
            max_inflight_predicts=_env_int(env, "MAX_INFLIGHT_PREDICTS", 4),
            rate_limit_per_minute=_env_int(env, "RATE_LIMIT_PER_MINUTE", 120),
            log_level=_env_str(env, "LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Fail loudly on a configuration the service cannot honour."""
        missing = [
            name
            for name, value in (
                ("JOB_BUCKET", self.job_bucket),
                ("QUEUE_URL", self.queue_url),
                ("JOBS_TABLE", self.jobs_table),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                "Missing required environment variables: " + ", ".join(missing)
            )
        if self.llm_provider not in VALID_PROVIDERS:
            raise ConfigurationError(
                f"LLM_PROVIDER must be one of {VALID_PROVIDERS}, got "
                f"{self.llm_provider!r}"
            )
        if self.max_reviews_per_job < 1:
            raise ConfigurationError("MAX_REVIEWS_PER_JOB must be at least 1")
        if self.worker_chunk_size < 1:
            raise ConfigurationError("WORKER_CHUNK_SIZE must be at least 1")
        if self.bert_batch_size < 1:
            raise ConfigurationError("BERT_BATCH_SIZE must be at least 1")
        if self.max_upload_bytes < 1:
            raise ConfigurationError("MAX_UPLOAD_BYTES must be positive")
        if self.heartbeat_interval < 1:
            raise ConfigurationError("HEARTBEAT_INTERVAL must be at least 1")
        if self.job_lease_seconds <= self.heartbeat_interval:
            raise ConfigurationError(
                "JOB_LEASE_SECONDS must be longer than HEARTBEAT_INTERVAL, "
                "otherwise the lease expires before it can be renewed"
            )
