#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Batch-annotate the existing corpus with the local BERT checkpoint.

Reads ``data/processed/processed_data.csv``, explodes the ``sub_comments`` JSON
column into one row per sub-comment, runs BERT over every sub-comment, and writes:

* ``data/processed/bert_sub_comments.csv`` — the flat per-sub-comment annotations.
* ``data/processed/bert_agreement_report.txt`` — topic agreement, macro-F1, sentiment
  RMSE/Pearson r, and the clamping / low-confidence rates.

This is the **local corpus-agreement CLI**. It is not the user-facing upload
pipeline — the app's URL/CSV batch mode submits jobs to the async service.

NOTE ON WHAT THE REPORT MEASURES: the corpus is the *training* corpus, and its
labels come from the LLM annotator that produced it. The numbers therefore measure
**agreement with that annotator** on data the model has already seen — they are
optimistic and are **not** accuracy against ground truth.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

# The repo root has to be importable for `models.bert_inference` to resolve to this
# repo's package.
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.bert_inference import (  # noqa: E402
    NUM_TOPICS,
    SENTIMENT_RANGE,
    TOPIC_LABELS,
    load_bert,
    predict_sub_comments,
)

DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "processed_data.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "bert_sub_comments.csv"
DEFAULT_REPORT = ROOT_DIR / "data" / "processed" / "bert_agreement_report.txt"

# How many sub-comments to hand to predict_sub_comments at a time. Keeps the
# per-row topic_probs mapping out of memory for the full 34k corpus.
CHUNK_SIZE = 2048


def explode_sub_comments(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sub-comment.

    ``sub_comments`` holds a JSON array per review, written by
    ``scripts/preprocess_data.py``. Parsed with ``json.loads`` — the same parser
    ``scripts/build_features.py`` uses. (The notebook's ``ast.literal_eval`` is the
    legacy path.)
    """
    rows = []
    for review_index, raw in enumerate(df["sub_comments"]):
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "review_index": review_index,
                    "sub_comment": item.get("sub_comment", ""),
                    "gemini_topic": item.get("topic"),
                    "gemini_sentiment": item.get("sentiment"),
                }
            )
    return pd.DataFrame(rows)


def annotate(
    sub_comments: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int = 64,
    device=None,
) -> pd.DataFrame:
    """Run BERT over every sub-comment and attach the predictions."""
    texts = sub_comments["sub_comment"].fillna("").astype(str).tolist()

    predictions = []
    for start in range(0, len(texts), CHUNK_SIZE):
        chunk = texts[start : start + CHUNK_SIZE]
        predictions.extend(predict_sub_comments(chunk, model, tokenizer, batch_size=batch_size))
        print(f"  annotated {min(start + CHUNK_SIZE, len(texts))}/{len(texts)} sub-comments")

    out = sub_comments.copy()
    out["bert_topic"] = [p["topic"] for p in predictions]
    out["bert_topic_conf"] = [p["topic_conf"] for p in predictions]
    out["bert_sentiment"] = [p["sentiment"] for p in predictions]
    out["bert_sentiment_raw"] = [p["sentiment_raw"] for p in predictions]
    out["bert_topic_probs"] = [json.dumps(p["topic_probs"]) for p in predictions]
    return out


def build_report(df: pd.DataFrame) -> str:
    """Topic agreement, macro-F1, sentiment error and calibration stats."""
    from sklearn.metrics import f1_score

    lines = []
    add = lines.append

    add("=" * 78)
    add("BERT vs. corpus annotator — agreement report")
    add("=" * 78)
    add("")
    add("WHAT THIS MEASURES")
    add("-------------------")
    add(
        "This corpus is the corpus BERT was TRAINED on, and its labels were produced"
    )
    add(
        "by the LLM annotator that created the sub-comments (Gemini). The numbers"
    )
    add(
        "below therefore measure AGREEMENT WITH THAT ANNOTATOR on data the model"
    )
    add(
        "has already seen. They are optimistic by construction and are NOT accuracy"
    )
    add("against human ground truth. No such ground truth exists in this repo.")
    add("")

    n = len(df)
    add(f"sub-comments annotated: {n}")

    # ---- topic agreement -------------------------------------------------
    agree = (df["gemini_topic"] == df["bert_topic"]).mean() if n else float("nan")
    add("")
    add("TOPIC AGREEMENT (BERT argmax vs. corpus topic)")
    add("-" * 78)
    add(f"overall accuracy: {agree:.4f}")

    truth = df["gemini_topic"].tolist()
    pred = df["bert_topic"].tolist()
    macro_f1 = f1_score(
        truth, pred, labels=TOPIC_LABELS, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        truth, pred, labels=TOPIC_LABELS, average="weighted", zero_division=0
    )
    add(f"macro-F1 over the {NUM_TOPICS} labels: {macro_f1:.4f}")
    add(f"weighted-F1 over the {NUM_TOPICS} labels: {weighted_f1:.4f}")

    add("")
    add(f"{'topic':<18}{'n':>8}{'predicted':>11}{'correct':>9}{'recall':>9}")
    for label in TOPIC_LABELS:
        mask = df["gemini_topic"] == label
        total = int(mask.sum())
        predicted = int((df["bert_topic"] == label).sum())
        correct = int((mask & (df["bert_topic"] == label)).sum())
        recall = correct / total if total else float("nan")
        add(f"{label:<18}{total:>8}{predicted:>11}{correct:>9}{recall:>9.4f}")

    add("")
    add("Confusion matrix (rows = corpus topic, columns = BERT topic)")
    add("-" * 78)
    header = "".join(f"{label[:6]:>8}" for label in TOPIC_LABELS)
    add(f"{'':<16}{header}")
    for true_label in TOPIC_LABELS:
        row = df[df["gemini_topic"] == true_label]["bert_topic"].value_counts()
        cells = "".join(f"{int(row.get(label, 0)):>8}" for label in TOPIC_LABELS)
        add(f"{true_label:<16}{cells}")

    # ---- sentiment -------------------------------------------------------
    true_sent = pd.to_numeric(df["gemini_sentiment"], errors="coerce")
    bert_sent = pd.to_numeric(df["bert_sentiment_raw"], errors="coerce")
    valid = true_sent.notna() & bert_sent.notna()

    add("")
    add("SENTIMENT (regression on the corpus sentiment float)")
    add("-" * 78)
    if valid.any():
        t = true_sent[valid].to_numpy(dtype=float)
        p = bert_sent[valid].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        mae = float(np.mean(np.abs(t - p)))
        if t.std() > 0 and p.std() > 0:
            pearson = float(np.corrcoef(t, p)[0, 1])
        else:
            pearson = float("nan")
        add(f"n compared:        {int(valid.sum())}")
        add(f"RMSE:              {rmse:.4f}")
        add(f"MAE:               {mae:.4f}")
        add(f"Pearson r:         {pearson:.4f}")
        add(
            f"corpus sentiment range: [{t.min():.4f}, {t.max():.4f}]"
        )
    else:
        add("no comparable sentiment values")

    # ---- calibration -----------------------------------------------------
    low_conf = float((df["bert_topic_conf"] < 0.5).mean()) if n else float("nan")
    clamped = (
        float((df["bert_sentiment_raw"].abs() > max(abs(SENTIMENT_RANGE[0]), abs(SENTIMENT_RANGE[1]))).mean())
        if n
        else float("nan")
    )
    add("")
    add("CALIBRATION")
    add("-" * 78)
    add(
        f"sentiment clamped to [{SENTIMENT_RANGE[0]}, {SENTIMENT_RANGE[1]}]: "
        f"{clamped:.4f} of predictions"
    )
    add(f"topic confidence < 0.5: {low_conf:.4f} of predictions")
    add(
        "the raw (unclamped) values are kept in bert_sentiment_raw so these rates "
        "stay observable"
    )
    add("")
    add("=" * 78)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate the corpus with the local BERT checkpoint."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="corpus CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="annotated CSV")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="agreement report")
    parser.add_argument("--batch-size", type=int, default=64, help="inference batch size")
    parser.add_argument(
        "--device", default=None, help="torch device (mps / cuda / cpu). Default: auto."
    )
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    report_path = pathlib.Path(args.report)

    print(f"Reading corpus from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  {len(df)} reviews")

    print("Exploding sub-comments...")
    sub_comments = explode_sub_comments(df)
    print(f"  {len(sub_comments)} sub-comments")

    print("Loading the local BERT checkpoint (no network)...")
    model, tokenizer, device = load_bert(device=args.device)
    print(f"  device: {device}")

    print("Annotating...")
    annotated = annotate(
        sub_comments, model, tokenizer, batch_size=args.batch_size, device=device
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_path, index=False)
    print(f"Wrote {len(annotated)} rows to {output_path}")

    report = build_report(annotated)
    report_path.write_text(report)
    print(f"Wrote the agreement report to {report_path}")


if __name__ == "__main__":
    main()
