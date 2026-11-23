"""Local-only BERT multi-task inference for ReviewAssist.

The checkpoint in ``models/saved_bert_model/`` is a *wrapper-keyed* artifact: 199
``bert.*`` encoder tensors plus ``topic_classifier.{weight,bias}`` and
``sentiment_regressor.{weight,bias}`` (203 keys total).  It therefore **cannot** be
loaded with ``AutoModel.from_pretrained(model_dir)`` — that would expect a bare
``BertModel`` with no ``bert.`` prefix and no heads.  The load below is a manual,
strict ``load_state_dict`` into :class:`BertMultiTask`.

Everything is read from disk.  There is no HuggingFace Hub lookup, no network call,
and no cache dependency: ``config.json`` is parsed directly into a ``BertConfig``,
the tokenizer comes from ``tokenizer.json`` via ``local_files_only=True``, and the
weights come from ``model.safetensors`` via ``safetensors.torch.load_file``.

Label contract (see ``models/saved_bert_model/head_config.json``):

* **8 topics**, in the data-derived order ``sorted(df['topic'].unique())`` over
  ``data/processed/processed_data.csv``.  This is *not* the 6-label vocabulary in the
  split prompt (``scripts/llm_api.py``) — the splitter keeps 6, the BERT head has 8.
* **Sentiment is a regression score in [-1, 1]**, regressing the LLM's per-sub-comment
  ``sentiment`` float — *not* the review ``score`` column (1-5).  Raw head outputs
  overshoot the range on ~18% of the corpus, so they are clamped to [-1, 1] and the
  unclamped value is preserved as ``sentiment_raw``.
"""

from __future__ import annotations

import functools
import json
import pathlib
from typing import Any, Iterable, Sequence

import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import BertConfig, BertModel, BertTokenizerFast

# ROOT_DIR pattern from scripts/preprocess_data.py:17 — path resolution that does not
# depend on the current working directory.
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "saved_bert_model"

# Label order is data-derived and MUST stay byte-identical to the checkpoint:
#   topic_to_id = {t: i for i, t in enumerate(sorted(df['topic'].unique()))}
# over data/processed/processed_data.csv, which yields 8 topics.
# This is NOT the 6-label list in scripts/llm_api.py.
TOPIC_LABELS = [
    "customer service",
    "design",
    "function",
    "other",
    "overall",
    "price",
    "quality",
    "security",
]
NUM_TOPICS = len(TOPIC_LABELS)  # 8
MAX_LENGTH = 64  # notebooks/bert_modeling.ipynb cell 13
SENTIMENT_RANGE = (-1.0, 1.0)  # the LLM's per-sub-comment sentiment float


class BertMultiTask(nn.Module):
    """BERT encoder with a topic-classification head and a sentiment-regression head.

    The class body is copied verbatim from ``notebooks/bert_modeling.ipynb`` cell 11.
    The only change is ``__init__``'s signature: the notebook took a
    ``base_model_name`` and called ``BertModel.from_pretrained('bert-base-uncased')``,
    which needs a Hub download that is forbidden here.  The encoder is instead
    constructed from the local ``config.json``.
    """

    def __init__(self, config: BertConfig, num_topics: int = NUM_TOPICS):
        super().__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.topic_classifier = nn.Linear(self.bert.config.hidden_size, num_topics)
        self.sentiment_regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        topic_logits = self.topic_classifier(pooled_output)
        sentiment_score = self.sentiment_regressor(pooled_output)
        return topic_logits, sentiment_score


def build_config(model_dir: pathlib.Path | str | None = None) -> BertConfig:
    """Build the encoder config from the local ``config.json``.

    Deliberately *not* ``BertConfig.from_pretrained(model_dir)``: that is a
    ``from_pretrained`` call in disguise, and its behaviour depends on the Hub cache
    being present or absent.  Reading the JSON and constructing ``BertConfig(**raw)``
    is one fewer moving part, has no cache semantics, and tolerates the
    ``architectures: ["BertMultiTask"]`` key that ``from_pretrained`` would try to
    resolve to an importable class.

    Falls back to constructing the ``bert-base-uncased`` shape in code only if
    ``config.json`` is absent.
    """
    model_dir = pathlib.Path(model_dir or DEFAULT_MODEL_DIR)
    config_path = model_dir / "config.json"

    if config_path.exists():
        raw = json.loads(config_path.read_text())
        raw.pop("architectures", None)  # names a custom class HF cannot import
        # "transformers_version" in the file records the version that wrote the
        # artifact, not the one running here; BertConfig ignores it.
        return BertConfig(**raw)

    return BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        hidden_act="gelu",
    )


def default_device() -> torch.device:
    """MPS when available, then CUDA, then CPU.

    The container has no MPS — it is CPU-only.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_bert(
    model_dir: pathlib.Path | str | None = None,
    device: torch.device | str | None = None,
):
    """Load the local BERT checkpoint. No network, no Hub, no cache.

    Returns ``(model, tokenizer, device)`` with the model in ``eval()`` mode and
    already moved onto ``device``.
    """
    model_dir = pathlib.Path(model_dir or DEFAULT_MODEL_DIR)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"BERT checkpoint directory not found: {model_dir}. "
            "model.safetensors is gitignored (438 MB exceeds GitHub's 100 MB per-file "
            "limit) and must be supplied out-of-band."
        )

    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"BERT weights not found: {weights_path}. "
            "The checkpoint is not tracked in git — copy it into "
            "models/saved_bert_model/ before loading."
        )

    # local_files_only=True is what makes the Hub lookup impossible rather than
    # merely unlikely. The fast tokenizer loads from tokenizer.json alone —
    # vocab.txt and special_tokens_map.json are not needed.
    tokenizer = BertTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)

    model = BertMultiTask(build_config(model_dir), NUM_TOPICS)

    # The checkpoint is safetensors, not pytorch_model.bin; there is no .bin file.
    state = load_file(str(weights_path))

    # STRICT load. The weights are keyed for this wrapper (bert.* + the two heads);
    # a lenient load would silently accept a mismatched label count and produce
    # garbage. strict=True raises on any missing or unexpected key.
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {weights_path} does not match the BertMultiTask architecture. "
            "The weights are wrapper-keyed (199 bert.* tensors + topic_classifier + "
            "sentiment_regressor) and must be loaded strictly into BertMultiTask. "
            f"Original error: {exc}"
        ) from exc

    if model.topic_classifier.weight.shape[0] != NUM_TOPICS:
        raise RuntimeError(
            f"topic_classifier head has {model.topic_classifier.weight.shape[0]} rows, "
            f"expected {NUM_TOPICS}. Check models/saved_bert_model/head_config.json "
            "for the label contract."
        )

    device = torch.device(device) if device is not None else default_device()
    # The 438 MB checkpoint loads onto CPU first; .to(device) happens *after* the
    # load, never straight onto MPS.
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_sub_comments(
    texts: Sequence[str],
    model: BertMultiTask,
    tokenizer: BertTokenizerFast,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Predict topic + sentiment for each text.

    Returns one dict per input, in input order::

        {"sub_comment", "topic", "topic_conf", "sentiment", "sentiment_raw", "topic_probs"}

    ``sentiment`` is clamped to ``[-1, 1]``; ``sentiment_raw`` keeps the unclamped
    regression output so the clamping rate stays observable.  ``topic_probs`` is the
    full 8-way ``{label: probability}`` mapping.
    """
    texts = list(texts)
    if not texts:
        return []

    device = next(model.parameters()).device
    results: list[dict[str, Any]] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoding = tokenizer(
            chunk,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.inference_mode():
            topic_logits, sentiment_scores = model(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            probs = torch.softmax(topic_logits, dim=-1)
            topic_idx = probs.argmax(dim=-1)
            sentiment_raw = sentiment_scores.squeeze(-1)

        probs_cpu = probs.cpu()
        idx_cpu = topic_idx.cpu()
        raw_cpu = sentiment_raw.cpu()
        sentiment_cpu = raw_cpu.clamp(SENTIMENT_RANGE[0], SENTIMENT_RANGE[1])

        for i, text in enumerate(chunk):
            row_probs = probs_cpu[i]
            results.append(
                {
                    "sub_comment": text,
                    "topic": TOPIC_LABELS[int(idx_cpu[i])],
                    "topic_conf": float(row_probs[int(idx_cpu[i])]),
                    "sentiment": float(sentiment_cpu[i]),
                    "sentiment_raw": float(raw_cpu[i]),
                    "topic_probs": {
                        label: float(row_probs[j]) for j, label in enumerate(TOPIC_LABELS)
                    },
                }
            )

    return results


def annotate_dataframe(
    df: pd.DataFrame,
    text_col: str = "sub_comment",
    model: BertMultiTask | None = None,
    tokenizer: BertTokenizerFast | None = None,
    batch_size: int = 32,
    device: torch.device | str | None = None,
) -> pd.DataFrame:
    """Add ``bert_topic`` / ``bert_topic_conf`` / ``bert_sentiment`` columns to ``df``.

    ``model`` and ``tokenizer`` may be passed in to reuse an already-loaded
    checkpoint; otherwise they are loaded with :func:`load_bert`.
    """
    if model is None or tokenizer is None:
        model, tokenizer, _ = load_bert(device=device)

    preds = predict_sub_comments(
        df[text_col].fillna("").astype(str).tolist(),
        model,
        tokenizer,
        batch_size=batch_size,
    )

    out = df.copy()
    out["bert_topic"] = [p["topic"] for p in preds]
    out["bert_topic_conf"] = [p["topic_conf"] for p in preds]
    out["bert_sentiment"] = [p["sentiment"] for p in preds]
    out["bert_sentiment_raw"] = [p["sentiment_raw"] for p in preds]
    return out


@functools.lru_cache(maxsize=1)
def load_bert_cached(
    model_dir: pathlib.Path | str | None = None,
    device: torch.device | str | None = None,
):
    """Process-wide singleton around :func:`load_bert`.

    The checkpoint is 438 MB; load it once per process.  Streamlit wraps this in
    ``@st.cache_resource``; the async service calls it directly.
    """
    return load_bert(model_dir=model_dir, device=device)
