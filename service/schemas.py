"""Wire formats for the ReviewAssist HTTP API.

These are the shapes the Streamlit frontend (``app.py``) talks to.  They are
additive-only: a field may gain a new value, never lose one, so an older
frontend keeps working against a newer API.
"""

from __future__ import annotations

from enum import Enum

import pydantic


class JobStatus(str, Enum):
    """The lifecycle of a batch job.

    ``PENDING`` → ``RUNNING`` → ``COMPLETED`` / ``FAILED``.  ``RETRYABLE`` is the
    worker's own state for a job whose worker died but which may be picked up
    again; it is never returned to a client as a terminal state.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRYABLE = "RETRYABLE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PredictRequest(pydantic.BaseModel):
    """One review to split and label."""

    text: str = pydantic.Field(min_length=1, description="the raw review text")
    split: bool = pydantic.Field(
        default=True,
        description="split the review into sub-comments with the LLM first",
    )


class SubCommentOut(pydantic.BaseModel):
    """One labelled sub-comment.

    ``sentiment`` is a regression score in [-1, 1] — not a probability and not a
    1-5 star score.  ``sentiment_raw`` keeps the unclamped head output.
    """

    text: str
    topic: str
    topic_conf: float
    sentiment: float
    sentiment_raw: float


class PredictResponse(pydantic.BaseModel):
    """The result of ``POST /predict``.

    ``provider`` names the splitter that ran (``openrouter`` / ``gemini``), or
    ``none`` when the whole review was used as a single sub-comment.
    ``split_fallback`` says *why*: it is true only when a split was requested and
    failed, not when the caller asked for no split.
    """

    provider: str
    split_fallback: bool
    sub_comments: list[SubCommentOut]


class JobOptions(pydantic.BaseModel):
    """Per-job options, sent as a JSON string in the ``options`` form field."""

    split: bool = pydantic.Field(
        default=True,
        description="split each review into sub-comments with the LLM",
    )


class JobCreated(pydantic.BaseModel):
    """The response to ``POST /jobs``."""

    job_id: str
    status: JobStatus
    total_reviews: int


class JobState(pydantic.BaseModel):
    """The response to ``GET /jobs/{job_id}``."""

    job_id: str
    status: JobStatus
    completed_reviews: int
    total_reviews: int
    error: str | None = None


class JobResult(pydantic.BaseModel):
    """The response to ``GET /jobs/{job_id}/result``."""

    job_id: str
    url: str
    expires_in: int
