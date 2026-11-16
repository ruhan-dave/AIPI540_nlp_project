"""The ReviewAssist HTTP API.

Three surfaces:

* ``POST /predict`` — one review in, split and labelled sub-comments out.
* ``POST /jobs``, ``GET /jobs/{job_id}``, ``GET /jobs/{job_id}/result`` — the
  asynchronous batch API the Streamlit frontend drives.
* ``POST /jobs/uploads`` and ``POST /jobs/uploads/{upload_id}/finalize`` — the
  presigned upload flow.  A CSV uploaded this way goes straight to the private
  S3 bucket and never passes through the API Gateway, whose 10 MB payload cap
  cannot be raised.

``GET /healthz`` is unauthenticated so the load balancer can poll it: it answers
``503`` until the local BERT checkpoint is loaded and ``200`` afterwards.  Every
other route requires ``Authorization: Bearer $API_TOKEN``.

The API never puts AWS credentials in a response and never logs review text.
"""

from __future__ import annotations

import asyncio
import functools
import hmac
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from botocore.exceptions import ClientError
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from service.batch_io import (
    CsvValidationError,
    build_s3_client,
    object_size,
    presign_object_get,
    presign_object_put,
    read_object_bytes,
    validate_review_csv,
    write_object_bytes,
)
from service.inference import ModelRuntime, SplitResult, Splitter
from service.jobs import COMPLETED, JobQueue, JobStore
from service.schemas import (
    FinalizeRequest,
    JobCreated,
    JobOptions,
    JobResult,
    JobState,
    JobStatus,
    PredictRequest,
    PredictResponse,
    UploadRequest,
    UploadTicket,
)
from service.settings import MIB, ConfigurationError, Settings

logger = logging.getLogger(__name__)

#: Uploads are read from the client in bounded pieces.
UPLOAD_CHUNK_BYTES = 1024 * 1024

#: The default options field, matching `JobOptions`.
DEFAULT_OPTIONS = '{"split": true}'

#: An upload id is a UUID4 hex.  It is checked against this pattern because it
#: becomes part of an S3 key: a value like ``../../results/x`` must never be
#: able to point the finalize route at another object.
UPLOAD_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

#: The executor the deadline-bounded split runs on.
_split_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="split")


class SplitDeadlineExceeded(Exception):
    """The split did not finish inside the request's deadline."""


def _discard_result(future: Any) -> None:
    """Consume the outcome of an abandoned call so asyncio does not warn."""

    if not future.cancelled():
        future.exception()


async def split_within_deadline(
    splitter: Splitter,
    text: str,
    *,
    enabled: bool,
    deadline_seconds: float,
) -> SplitResult:
    """Run the split, giving up after ``deadline_seconds``.

    The provider call is **not** cancelled when the deadline passes — it has
    already been paid for — but the caller stops waiting for it and falls back
    to the whole review.  That is what keeps the API answering inside the API
    Gateway's integration timeout instead of letting the gateway answer 504
    while the service is still working.
    """
    loop = asyncio.get_running_loop()
    call = functools.partial(splitter.split, text, enabled=enabled)
    future = loop.run_in_executor(_split_executor, call)
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=deadline_seconds)
    except asyncio.TimeoutError as exc:
        future.add_done_callback(_discard_result)
        raise SplitDeadlineExceeded from exc


class RateLimiter:
    """A per-token sliding-window rate limiter.

    The window is per process, so with the API scaled to more than one task the
    effective limit is the limit times the number of tasks.  That is the right
    trade for a demo guard against a runaway frontend; a shared, exact limit
    belongs in a gateway or in DynamoDB, not in the application.
    """

    def __init__(self, limit_per_minute: int, window_seconds: int = 60) -> None:
        self._limit = limit_per_minute
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> int | None:
        """Record a call for ``key``.

        Returns the number of seconds to wait when the caller is over the limit,
        or ``None`` when the call is allowed.
        """
        if self._limit <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return max(1, int(self._window - (now - hits[0])) + 1)
            hits.append(now)
            return None


@dataclass
class ServiceContext:
    """Everything the API talks to, assembled once at start-up."""

    settings: Settings
    runtime: ModelRuntime
    splitter: Splitter
    jobs: JobStore
    queue: JobQueue
    s3: Any

    # Rebuilt from the settings in __post_init__ rather than passed in: the
    # capacity belongs to the settings, and a replaced context must not keep the
    # old one.
    limiter: RateLimiter = field(init=False, default=None)  # type: ignore[assignment]
    semaphore: threading.Semaphore = field(init=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.limiter = RateLimiter(self.settings.rate_limit_per_minute)
        # A threading semaphore, not an asyncio one: the guard covers work that
        # runs in a threadpool, and an asyncio primitive binds to the event loop
        # it first runs in.
        self.semaphore = threading.Semaphore(self.settings.max_inflight_predicts)


def build_context(settings: Settings) -> ServiceContext:
    """Assemble the service's collaborators from the settings."""
    return ServiceContext(
        settings=settings,
        runtime=ModelRuntime(settings),
        splitter=Splitter(settings),
        jobs=JobStore(settings.jobs_table, settings.aws_region),
        queue=JobQueue(settings.queue_url, region=settings.aws_region),
        s3=build_s3_client(settings.aws_region),
    )


async def read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, refusing anything over the limit."""
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    413,
                    detail=(
                        f"the upload is larger than the {max_bytes // MIB} MiB limit"
                    ),
                )
            chunks.append(chunk)
    finally:
        await upload.close()
    return b"".join(chunks)


def create_app(context: ServiceContext | None = None) -> FastAPI:
    """Build the FastAPI application.

    ``context`` is injected so the tests can supply their own AWS clients; the
    production path builds it from the environment with :func:`build_context`.
    """
    if context is None:
        context = build_context(Settings.from_env())
    if not context.settings.api_token:
        raise ConfigurationError(
            "API_TOKEN is required: without it the API would serve every request "
            "unauthenticated"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        context.runtime.start()
        yield

    app = FastAPI(
        title="ReviewAssist API",
        version="0.1.0",
        summary="Split and label Google Play reviews with BERT.",
        lifespan=lifespan,
    )
    app.state.context = context

    # --- authentication ----------------------------------------------------

    def require_token(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> str:
        """Reject anything but the configured bearer token."""
        expected = request.app.state.context.settings.api_token
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(
                401,
                detail="a bearer token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not hmac.compare_digest(token.strip().encode(), expected.encode()):
            raise HTTPException(
                401,
                detail="the bearer token is not valid",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token.strip()

    # --- routes ------------------------------------------------------------

    @app.get("/healthz")
    def healthz(request: Request) -> JSONResponse:
        """Unauthenticated readiness probe for the load balancer."""
        runtime = request.app.state.context.runtime
        if runtime.ready:
            return JSONResponse({"status": "ready"})
        payload: dict[str, Any] = {"status": "loading"}
        if runtime.load_error:
            payload["detail"] = runtime.load_error
        return JSONResponse(payload, status_code=503)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(
        payload: PredictRequest,
        request: Request,
        token: str = Depends(require_token),
    ) -> PredictResponse:
        """Split one review and label every sub-comment with BERT."""
        ctx = request.app.state.context
        if not ctx.runtime.ready:
            raise HTTPException(
                503,
                detail="the model is not loaded yet",
                headers={"Retry-After": "5"},
            )

        retry_after = ctx.limiter.check(token)
        if retry_after is not None:
            raise HTTPException(
                429,
                detail="too many requests",
                headers={"Retry-After": str(retry_after)},
            )

        if not ctx.semaphore.acquire(blocking=False):
            raise HTTPException(
                503,
                detail="the service is saturated",
                headers={"Retry-After": "1"},
            )

        try:
            try:
                split = await split_within_deadline(
                    ctx.splitter,
                    payload.text,
                    enabled=payload.split,
                    deadline_seconds=ctx.settings.predict_deadline_seconds,
                )
            except SplitDeadlineExceeded:
                # The provider call is still running and still paid for.  The
                # caller gets the whole review instead of waiting: the response
                # has to arrive before the API Gateway's integration timeout.
                logger.warning(
                    "predict_deadline_exceeded",
                    extra={
                        "provider": ctx.splitter.provider,
                        "deadline_seconds": ctx.settings.predict_deadline_seconds,
                    },
                )
                split = SplitResult(
                    texts=[payload.text], fallback=True, provider="none"
                )
            predictions = await run_in_threadpool(ctx.runtime.predict, split.texts)
        finally:
            ctx.semaphore.release()

        logger.info(
            "predict_completed",
            extra={
                "provider": split.provider,
                "split_fallback": split.fallback,
                "sub_comments": len(predictions),
            },
        )
        return PredictResponse(
            provider=split.provider,
            split_fallback=split.fallback,
            sub_comments=[
                {
                    "text": prediction["text"],
                    "topic": prediction["topic"],
                    "topic_conf": prediction["topic_conf"],
                    "sentiment": prediction["sentiment"],
                    "sentiment_raw": prediction["sentiment_raw"],
                }
                for prediction in predictions
            ],
        )

    @app.post("/jobs", status_code=202, response_model=JobCreated)
    async def create_job(
        request: Request,
        file: UploadFile = File(...),
        options: str = Form(default=DEFAULT_OPTIONS),
        token: str = Depends(require_token),
    ) -> JobCreated:
        """Accept one CSV as one batch job.

        This route carries the CSV through the API, so it is limited to
        ``MAX_MULTIPART_UPLOAD_BYTES`` — an API Gateway HTTP API caps a request
        payload at 10 MB and that cap cannot be raised.  A larger CSV goes
        through the presigned upload flow instead.
        """
        ctx = request.app.state.context
        settings = ctx.settings

        try:
            job_options = JobOptions.model_validate_json(options or DEFAULT_OPTIONS)
        except Exception as exc:
            raise HTTPException(
                422, detail=f"the options field is not valid JSON: {exc}"
            ) from exc

        payload = await read_upload(file, settings.max_multipart_upload_bytes)
        try:
            validated = validate_review_csv(
                payload,
                max_bytes=settings.max_multipart_upload_bytes,
                max_rows=settings.max_reviews_per_job,
            )
        except CsvValidationError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

        job_id = uuid.uuid4().hex
        input_key = f"inputs/{job_id}/reviews.csv"

        try:
            write_object_bytes(
                ctx.s3, settings.job_bucket, input_key, payload, content_type="text/csv"
            )
        except Exception as exc:
            logger.error(
                "job_input_upload_failed",
                extra={"job_id": job_id, "detail": f"{type(exc).__name__}: {exc}"},
            )
            raise HTTPException(
                503, detail="the job input could not be stored"
            ) from exc

        ctx.jobs.create(
            job_id=job_id,
            input_key=input_key,
            total_reviews=validated.row_count,
            options=job_options.model_dump(),
            ttl_days=settings.job_ttl_days,
        )

        try:
            ctx.queue.send(
                job_id=job_id,
                input_key=input_key,
                options=job_options.model_dump(),
            )
        except Exception as exc:
            # The job exists but nothing will process it.  Say so instead of
            # reporting a success that will never complete.
            logger.error(
                "job_enqueue_failed",
                extra={"job_id": job_id, "detail": f"{type(exc).__name__}: {exc}"},
            )
            ctx.jobs.fail(job_id=job_id, error="the job could not be queued")
            raise HTTPException(
                503, detail="the job could not be queued"
            ) from exc

        logger.info(
            "job_accepted",
            extra={
                "job_id": job_id,
                "total_reviews": validated.row_count,
                "split": job_options.split,
            },
        )
        return JobCreated(
            job_id=job_id, status=JobStatus.PENDING, total_reviews=validated.row_count
        )

    @app.post("/jobs/uploads", status_code=201, response_model=UploadTicket)
    async def create_upload(
        request: Request,
        body: UploadRequest | None = None,
        token: str = Depends(require_token),
    ) -> UploadTicket:
        """Issue a short-lived, key-scoped presigned URL for one CSV upload.

        The CSV is uploaded by the caller straight to the private S3 bucket, so
        it never passes through the API Gateway — whose 10 MB payload cap
        cannot be raised.  The key is chosen here, never by the caller, and the
        URL authorises one PUT to that key and nothing else.
        """
        ctx = request.app.state.context
        settings = ctx.settings

        upload_id = uuid.uuid4().hex
        input_key = f"inputs/{upload_id}/reviews.csv"
        upload_url = presign_object_put(
            ctx.s3, settings.job_bucket, input_key, settings.upload_url_ttl_seconds
        )

        logger.info(
            "upload_ticket_issued",
            extra={
                "upload_id": upload_id,
                "filename": body.filename if body else None,
                "expires_in": settings.upload_url_ttl_seconds,
            },
        )
        return UploadTicket(
            upload_id=upload_id,
            upload_url=upload_url,
            input_key=input_key,
            expires_in=settings.upload_url_ttl_seconds,
        )

    @app.post("/jobs/uploads/{upload_id}/finalize", status_code=202, response_model=JobCreated)
    async def finalize_upload(
        upload_id: str,
        request: Request,
        body: FinalizeRequest | None = None,
        token: str = Depends(require_token),
    ) -> JobCreated:
        """Verify an uploaded CSV, turn it into a job and enqueue that job.

        The object is read back from S3 and validated here — its size, its
        ``content`` column and its row count — so a job only ever enters the
        queue after its input has been checked.  The upload id becomes the job
        id, so finalizing the same upload twice is refused instead of
        creating a second job for the same object.
        """
        ctx = request.app.state.context
        settings = ctx.settings

        if not UPLOAD_ID_PATTERN.fullmatch(upload_id):
            raise HTTPException(422, detail="the upload id is not valid")

        job_options = body.options if body else JobOptions()
        input_key = f"inputs/{upload_id}/reviews.csv"

        try:
            size = object_size(ctx.s3, settings.job_bucket, input_key)
        except Exception as exc:
            logger.error(
                "upload_check_failed",
                extra={
                    "upload_id": upload_id,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
            raise HTTPException(
                503, detail="the uploaded file could not be checked"
            ) from exc

        if size is None:
            raise HTTPException(
                409,
                detail="the upload was not received: there is no object at the upload key",
            )
        if size > settings.max_upload_bytes:
            raise HTTPException(
                413,
                detail=(
                    f"the uploaded file is larger than the "
                    f"{settings.max_upload_bytes // MIB} MiB limit"
                ),
            )

        try:
            payload = read_object_bytes(ctx.s3, settings.job_bucket, input_key)
        except Exception as exc:
            logger.error(
                "upload_read_failed",
                extra={
                    "upload_id": upload_id,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
            raise HTTPException(
                503, detail="the uploaded file could not be read"
            ) from exc

        try:
            validated = validate_review_csv(
                payload,
                max_bytes=settings.max_upload_bytes,
                max_rows=settings.max_reviews_per_job,
            )
        except CsvValidationError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

        job_id = upload_id
        try:
            ctx.jobs.create(
                job_id=job_id,
                input_key=input_key,
                total_reviews=validated.row_count,
                options=job_options.model_dump(),
                ttl_days=settings.job_ttl_days,
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                raise HTTPException(
                    409, detail="this upload was already finalized"
                ) from exc
            logger.error(
                "job_record_failed",
                extra={
                    "job_id": job_id,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
            raise HTTPException(
                503, detail="the job could not be recorded"
            ) from exc

        try:
            ctx.queue.send(
                job_id=job_id,
                input_key=input_key,
                options=job_options.model_dump(),
            )
        except Exception as exc:
            # The job exists but nothing will process it.  Say so instead of
            # reporting a success that will never complete.
            logger.error(
                "job_enqueue_failed",
                extra={"job_id": job_id, "detail": f"{type(exc).__name__}: {exc}"},
            )
            ctx.jobs.fail(job_id=job_id, error="the job could not be queued")
            raise HTTPException(
                503, detail="the job could not be queued"
            ) from exc

        logger.info(
            "job_accepted",
            extra={
                "job_id": job_id,
                "total_reviews": validated.row_count,
                "split": job_options.split,
                "upload": True,
            },
        )
        return JobCreated(
            job_id=job_id, status=JobStatus.PENDING, total_reviews=validated.row_count
        )

    @app.get("/jobs/{job_id}", response_model=JobState)
    def job_status(
        job_id: str,
        request: Request,
        token: str = Depends(require_token),
    ) -> JobState:
        """The status and progress of one job."""
        ctx = request.app.state.context
        job = ctx.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, detail="unknown job")
        return JobState(
            job_id=job_id,
            status=_job_status(job),
            completed_reviews=int(job.get("completed_reviews", 0)),
            total_reviews=int(job.get("total_reviews", 0)),
            error=job.get("error") or None,
        )

    @app.get("/jobs/{job_id}/result", response_model=JobResult)
    def job_result(
        job_id: str,
        request: Request,
        token: str = Depends(require_token),
    ) -> JobResult:
        """A short-lived presigned URL for a completed job's result CSV."""
        ctx = request.app.state.context
        settings = ctx.settings
        job = ctx.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, detail="unknown job")
        if job.get("status") != COMPLETED:
            raise HTTPException(
                409,
                detail=f"the job is {job.get('status')}, not {COMPLETED}",
            )
        result_key = job.get("result_key")
        if not result_key:
            raise HTTPException(500, detail="the job has no result object")

        url = presign_object_get(
            ctx.s3, settings.job_bucket, result_key, settings.result_url_ttl_seconds
        )
        return JobResult(
            job_id=job_id,
            url=url,
            expires_in=settings.result_url_ttl_seconds,
        )

    return app


def _job_status(job: dict[str, Any]) -> JobStatus:
    """The job's status as a :class:`JobStatus`.

    A status this build does not know about means the record was written by a
    different version — that is a corrupt record, not something to paper over.
    """
    raw = str(job.get("status", "PENDING"))
    try:
        return JobStatus(raw)
    except ValueError as exc:
        raise HTTPException(
            500, detail=f"the job record has an unknown status {raw!r}"
        ) from exc


def main() -> None:
    """Run the API."""
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    context = build_context(settings)
    app = create_app(context)
    port = int(os.environ.get("PORT", "8000"))
    logger.info("api_starting", extra={"port": port, "region": settings.aws_region})

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
