"""The inference runtime shared by the API and the worker.

Two things live here:

* :class:`ModelRuntime` owns the local BERT checkpoint.  The load runs on a
  background thread so the API binds its port immediately and ``/healthz`` can
  answer ``503`` while the 438 MB checkpoint is still being read from disk.
* :class:`Splitter` owns the LLM split.  ``scripts.llm_api`` raises at *import*
  time when the active provider's key is missing, so the import is deferred and a
  missing key becomes a visible whole-review fallback rather than a crash.

Neither ever logs review text: the corpus is user content and the logs are
shipped to CloudWatch.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from service.settings import Settings

logger = logging.getLogger(__name__)

# The canonical shape of one prediction.  Both the API response and the result
# CSV are built from it, so there is exactly one place that knows what a
# prediction looks like.
Prediction = dict[str, Any]


class Predictor(Protocol):
    """What the API and the worker need from a model."""

    @property
    def ready(self) -> bool: ...

    @property
    def load_error(self) -> str | None: ...

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...


@dataclass
class SplitResult:
    """The outcome of one split request."""

    texts: list[str]
    fallback: bool
    provider: str


class ModelRuntime:
    """The local BERT checkpoint, loaded once per process."""

    def __init__(
        self,
        settings: Settings,
        *,
        predict: Any = None,
    ) -> None:
        self._settings = settings
        self._predict_impl = predict
        self._model = None
        self._tokenizer = None
        self._device = None
        self._state = "loading"
        self._error: str | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Begin loading the checkpoint in the background. Returns immediately."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._load, name="bert-loader", daemon=True
        )
        self._thread.start()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Block until the checkpoint is loaded (or has failed to load)."""
        if self._thread is None:
            self.start()
        self._thread.join(timeout)
        return self.ready

    def _load(self) -> None:
        if self._predict_impl is not None:
            # Injected predictor — the checkpoint is not read at all.
            self._state = "ready"
            return
        try:
            from models.bert_inference import load_bert_cached

            model, tokenizer, device = load_bert_cached(
                str(self._settings.model_dir), self._settings.model_device
            )
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._state = "failed"
            logger.error(
                "bert_load_failed",
                extra={"detail": self._error, "model_dir": str(self._settings.model_dir)},
            )
            return
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._state = "ready"
        logger.info("bert_load_ready", extra={"device": str(device)})

    # -- state -------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._state == "ready"

    @property
    def load_error(self) -> str | None:
        return self._error

    @property
    def device(self) -> str | None:
        return None if self._device is None else str(self._device)

    # -- inference ---------------------------------------------------------

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Label each text with a topic and a sentiment score.

        The returned dicts use the canonical :data:`Prediction` shape: ``text``,
        ``topic``, ``topic_conf``, ``sentiment`` and ``sentiment_raw``.
        """
        items = list(texts)
        if not items:
            return []
        if not self.ready:
            raise RuntimeError(
                f"the BERT checkpoint is not loaded ({self._state}): {self._error}"
            )

        if self._predict_impl is not None:
            raw = self._predict_impl(items)
        else:
            from models.bert_inference import predict_sub_comments

            raw = predict_sub_comments(
                items,
                self._model,
                self._tokenizer,
                batch_size=self._settings.bert_batch_size,
            )

        return [
            {
                "text": prediction.get("text")
                or prediction.get("sub_comment")
                or "",
                "topic": prediction["topic"],
                "topic_conf": float(prediction["topic_conf"]),
                "sentiment": float(prediction["sentiment"]),
                "sentiment_raw": float(prediction["sentiment_raw"]),
            }
            for prediction in raw
        ]


class Splitter:
    """The LLM split, with an explicit whole-review fallback.

    A missing provider key, a transport failure or an unusable response all end
    the same way: the whole review becomes one sub-comment and the caller is told
    the split fell back.  Nothing is silently swallowed.
    """

    def __init__(self, settings: Settings, *, generate: Any = None) -> None:
        self._settings = settings
        self._generate = generate
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def provider(self) -> str:
        """The configured provider id (``openrouter`` or ``gemini``)."""
        return self._settings.llm_provider

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._generate is not None:
            return self._generate
        with self._lock:
            if self._generate is not None:
                return self._generate
            if self._error is not None:
                return None
            try:
                from scripts.llm_api import llm_generate
            except Exception as exc:
                # scripts/llm_api raises ValueError when the active provider's key
                # is unset.  That is a configuration state, not a crash: the
                # service still labels the review, just without a split.
                self._error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "splitter_unavailable",
                    extra={"detail": self._error, "provider": self.provider},
                )
                return None
            self._generate = llm_generate
            return self._generate

    def split(self, text: str, *, enabled: bool = True) -> SplitResult:
        """Split ``text`` into sub-comment texts.

        With ``enabled=False`` the whole review is returned as a single
        sub-comment and no provider is reported — that is the fast test path,
        not a fallback.
        """
        if not enabled:
            return SplitResult(texts=[text], fallback=False, provider="none")

        generate = self._load()
        if generate is None:
            return SplitResult(texts=[text], fallback=True, provider="none")

        try:
            sub_comments = generate(text)
        except Exception as exc:
            logger.warning(
                "split_failed",
                extra={
                    "detail": f"{type(exc).__name__}: {exc}",
                    "provider": self.provider,
                },
            )
            return SplitResult(texts=[text], fallback=True, provider="none")

        texts = [sc.sub_comment for sc in sub_comments if sc.sub_comment]
        if not texts:
            logger.warning("split_empty", extra={"provider": self.provider})
            return SplitResult(texts=[text], fallback=True, provider="none")
        return SplitResult(texts=texts, fallback=False, provider=self.provider)
