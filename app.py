"""ReviewAssist — Streamlit UI.

Three input modes:

* **Manual Review Input** — one review, analysed synchronously in four comparison
  model choices (Naive Bayes | Logistic Regression | RAKE | BERT Multi-Task), with a
  visible multi-select and all four selected by default.
* **Google Play Store App URL** — fetch the newest English/US reviews for an app and
  submit them to the async review service, one BERT job per batch.
* **CSV Upload** — upload raw reviews with a required ``content`` column and submit
  them the same way. Batch mode uses BERT Multi-Task.

The batch modes talk to the service described in
``.claude/plans/aws-deployment.md``: ``POST /jobs`` with one CSV per job,
``GET /jobs/{job_id}`` for progress, and ``GET /jobs/{job_id}/result`` for the
finished CSV.
"""

import io
import json
import os
import pathlib
import pickle
import sys
import uuid

import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# The repo root has to be importable so `models` and `scripts` resolve to this repo.
ROOT_DIR = pathlib.Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.scoring import (  # noqa: E402
    WORD2VEC_DOWNLOAD_URL,
    WORD2VEC_RELATIVE_PATH,
    calculate_score_lr,
    calculate_score_nb,
    extract_keywords_rake,
    highlight_sentence_html,
    highlight_sentence_nb_html,
    lr_keywords_from_model,
    load_word2vec,
    review_sentiment_probabilities,
)

# --- run limits ----------------------------------------------------------
MAX_BATCHES_PER_RUN = 10
MIN_REVIEWS_PER_BATCH = 100
MAX_REVIEWS_PER_BATCH = 1000
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MiB
MANUAL_MODEL_OPTIONS = [
    "Naive Bayes",
    "Logistic Regression",
    "RAKE",
    "BERT Multi-Task",
]

# --- review service ------------------------------------------------------
# Never put AWS credentials in Streamlit or the browser — the app only knows the
# service URL and its bearer token.
REVIEW_SERVICE_URL = os.environ.get(
    "REVIEW_SERVICE_URL", "http://localhost:8000"
).rstrip("/")
API_TOKEN = os.environ.get("API_TOKEN", "")
BERT_INFERENCE_MODE = os.environ.get("BERT_INFERENCE_MODE", "local").strip().lower()

RESULT_PREVIEW_ROWS = 1000

TOPIC_LABEL_SUMMARY = (
    "8 topics (customer service, design, function, other, overall, price, quality, "
    "security)"
)


# =========================================================================
# Artifact loading
# =========================================================================


@st.cache_resource
def load_classical_artifacts():
    """Load the pickled LR/NB models and vectorizers.

    Cached: they are read once per process, not once per rerun.
    """
    paths = {
        "lr_model": ROOT_DIR / "models" / "lr_model.pkl",
        "tfidf_vectorizer": ROOT_DIR / "models" / "tfidf_vectorizer.pkl",
        "nb_model": ROOT_DIR / "models" / "nb_model.pkl",
        "count_vectorizer": ROOT_DIR / "models" / "count_vectorizer.pkl",
    }
    unusable = [
        f"{name} ({path})"
        for name, path in paths.items()
        if not path.exists() or path.stat().st_size == 0
    ]
    if unusable:
        raise FileNotFoundError(
            "These model artifacts are missing or empty: "
            + ", ".join(unusable)
            + ". Re-create them with: python scripts/build_features.py && "
            "python scripts/model.py"
        )

    loaded = {}
    for name, path in paths.items():
        with open(path, "rb") as fh:
            loaded[name] = pickle.load(fh)

    positive_words, negative_words = lr_keywords_from_model(
        loaded["lr_model"], loaded["tfidf_vectorizer"]
    )
    return (
        loaded["lr_model"],
        loaded["tfidf_vectorizer"],
        loaded["nb_model"],
        loaded["count_vectorizer"],
        positive_words,
        negative_words,
    )


@st.cache_resource
def load_bert_artifacts():
    """Load the 438 MB BERT checkpoint once per process."""
    from models.bert_inference import load_bert_cached

    return load_bert_cached()


@st.cache_resource
def get_word2vec():
    """Load the word2vec binary if it is on disk, else ``None``."""
    return load_word2vec()


# =========================================================================
# LLM split (lazy, cached, with a visible fallback)
# =========================================================================


def provider_label() -> str:
    provider = os.environ.get("LLM_PROVIDER", "openrouter").strip().lower()
    return "Gemini" if provider == "gemini" else "OpenRouter"


@st.cache_data(show_spinner=False)
def split_review(text: str):
    """Split a review into sub-comments with the configured LLM provider.

    Returns ``(texts, provider_label, fallback_reason)``. ``fallback_reason`` is
    ``None`` on success; otherwise it explains why the whole review is used as a
    single sub-comment.

    ``scripts.llm_api`` is imported *inside* the function: it raises ValueError at
    import when the active provider's key is missing, and a module-level import
    would take the whole app down.
    """
    try:
        from scripts.llm_api import llm_generate
    except Exception as exc:
        return (
            [text],
            provider_label(),
            f"the split provider is not configured ({type(exc).__name__}: {exc})",
        )

    try:
        sub_comments = llm_generate(text)
    except Exception as exc:
        return (
            [text],
            provider_label(),
            f"the split request failed ({type(exc).__name__}: {exc})",
        )

    texts = [sc.sub_comment for sc in sub_comments if sc.sub_comment]
    if not texts:
        return [text], provider_label(), "the split returned no sub-comments"
    return texts, provider_label(), None


# =========================================================================
# The four comparison columns
# =========================================================================


def render_nb_column(text: str, nb_model, count_vectorizer, word2vec_available: bool):
    st.subheader("Naive Bayes Analysis")
    positive, negative = review_sentiment_probabilities(
        text, nb_model, count_vectorizer
    )
    if word2vec_available:
        highlighted = (
            '<div style="background-color: white; padding: 10px; border-radius: 5px;">'
            f"{highlight_sentence_nb_html(text, nb_model, count_vectorizer)}</div>"
        )
        components.html(highlighted, height=300, scrolling=True)
    st.write(f"Positive: {positive:.2%}")
    st.write(f"Negative: {negative:.2%}")


def render_lr_column(
    text: str,
    lr_model,
    tfidf_vectorizer,
    positive_words,
    negative_words,
    word2vec_available: bool,
):
    st.subheader("Logistic Regression Analysis")
    positive, negative = review_sentiment_probabilities(
        text, lr_model, tfidf_vectorizer
    )
    if word2vec_available:
        scores = calculate_score_lr(text, positive_words, negative_words)
        highlighted = (
            '<div style="background-color: white; padding: 10px; border-radius: 5px;">'
            f"{highlight_sentence_html(text, scores)}</div>"
        )
        components.html(highlighted, height=300, scrolling=True)
    st.write(f"Positive: {positive:.2%}")
    st.write(f"Negative: {negative:.2%}")


def render_rake_column(text: str):
    st.subheader("RAKE Keyword Extraction")
    keywords = extract_keywords_rake(text, top_n=5)
    st.write("Top 5 Keywords:")
    if not keywords:
        st.caption("No keywords extracted.")
    for keyword in keywords:
        st.write(f"- {keyword}")


def render_bert_results(rows: list[dict], source_label: str):
    """Render local or service predictions with the same table and rollup."""
    if not rows:
        st.warning("BERT returned no sub-comment predictions.")
        return

    table_rows = []
    for row in rows:
        raw_sentiment = row.get("sentiment_raw")
        table_rows.append(
            {
                "sub_comment": row.get("sub_comment") or row.get("text", ""),
                "topic": row.get("topic", "other"),
                "topic_confidence": round(float(row.get("topic_conf", 0.0)), 4),
                # A regression score in [-1, 1] — not a probability.
                "sentiment": round(float(row.get("sentiment", 0.0)), 3),
                "sentiment_raw": (
                    round(float(raw_sentiment), 3)
                    if raw_sentiment is not None
                    else None
                ),
            }
        )

    table = pd.DataFrame(table_rows)
    st.dataframe(
        table,
        hide_index=True,
        alt="BERT topic and sentiment predictions for each review sub-comment",
    )
    per_topic = (
        table.groupby("topic")["topic_confidence"].mean().sort_values(ascending=False)
    )
    st.write(
        f"Dominant topic: {per_topic.index[0]} "
        f"(mean confidence {per_topic.iloc[0]:.3f})"
    )
    st.write(f"Mean sentiment: {table['sentiment'].mean():.2f} (-1.0 to 1.0)")
    st.caption(f"{TOPIC_LABEL_SUMMARY} · {source_label}")


def render_local_bert_column(text: str):
    st.subheader("BERT Multi-Task Analysis")
    from models.bert_inference import predict_sub_comments

    model, tokenizer, device = load_bert_artifacts()
    texts, label, fallback_reason = split_review(text)
    if fallback_reason:
        st.warning(
            f"LLM split unavailable — {fallback_reason}. "
            "Set OPENROUTER_API_KEY (or GEMINI_API_KEY) in .env for sub-comment "
            "granularity; showing BERT on the whole review instead."
        )
    else:
        st.caption(f"Split into {len(texts)} sub-comment(s) by {label}")

    rows = predict_sub_comments(texts, model, tokenizer)
    render_bert_results(rows, f"local device: {device}")


def render_remote_bert_column(text: str):
    """Use the deployed inference service so the frontend need not load BERT."""
    st.subheader("BERT Multi-Task Analysis")
    if not API_TOKEN:
        st.error("Remote BERT mode requires API_TOKEN in the frontend environment.")
        return

    try:
        response = httpx.post(
            f"{REVIEW_SERVICE_URL}/predict",
            headers=_auth_headers(),
            json={"text": text, "split": True},
            timeout=120.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        st.error(f"Could not reach the BERT service at {REVIEW_SERVICE_URL}: {exc}")
        return

    rows = payload.get("sub_comments", [])
    provider = payload.get("provider", "none")
    if provider == "none":
        st.warning("The service used the whole-review fallback because its LLM splitter is unavailable.")
    else:
        st.caption(f"Split and labeled by the remote service ({provider})")
    render_bert_results(rows, "remote service")


def render_bert_column(text: str):
    if BERT_INFERENCE_MODE == "remote":
        render_remote_bert_column(text)
    else:
        render_local_bert_column(text)


def get_artifacts():
    try:
        return load_classical_artifacts()
    except Exception as exc:
        st.error(f"Could not load the trained models: {exc}")
        st.stop()


def render_review(text: str, selected_models: list[str]) -> None:
    """Render only the selected model columns for a single review."""
    selected_models = [
        name for name in MANUAL_MODEL_OPTIONS if name in (selected_models or [])
    ]
    if not selected_models:
        st.info("Select at least one model to analyze this review.")
        return

    classical = None
    word2vec_available = False
    if {"Naive Bayes", "Logistic Regression"}.intersection(selected_models):
        classical = get_artifacts()
        word2vec_available = get_word2vec() is not None
        if not word2vec_available:
            st.warning(
                "The word2vec binary is missing. Naive Bayes and Logistic Regression "
                "still show review-level scores, but word-level colouring is unavailable. "
                f"Download the 1.53 GB Figshare file and place it at "
                f"{ROOT_DIR / WORD2VEC_RELATIVE_PATH}."
            )

    columns = st.columns(len(selected_models))
    for name, column in zip(selected_models, columns):
        with column:
            if name == "Naive Bayes":
                _, _, nb_model, count_vectorizer, _, _ = classical
                render_nb_column(text, nb_model, count_vectorizer, word2vec_available)
            elif name == "Logistic Regression":
                (
                    lr_model,
                    tfidf_vectorizer,
                    _,
                    _,
                    positive_words,
                    negative_words,
                ) = classical
                render_lr_column(
                    text,
                    lr_model,
                    tfidf_vectorizer,
                    positive_words,
                    negative_words,
                    word2vec_available,
                )
            elif name == "RAKE":
                render_rake_column(text)
            elif name == "BERT Multi-Task":
                render_bert_column(text)


# =========================================================================
# Batch mode — sources
# =========================================================================


def extract_app_id(url: str):
    import re

    match = re.search(r"id=([^&]+)", url)
    return match.group(1) if match else None


def fetch_play_store_reviews(app_id: str, count: int) -> pd.DataFrame:
    """Fetch the newest English/US reviews for an app, keeping their metadata."""
    from google_play_scraper import Sort, reviews

    result, _ = reviews(
        app_id,
        lang="en",
        country="us",
        sort=Sort.NEWEST,
        count=count,
    )

    rows = []
    for review in result:
        at = review.get("at")
        rows.append(
            {
                "content": review.get("content", "") or "",
                "score": review.get("score"),
                "date": at.isoformat() if hasattr(at, "isoformat") else at,
                "review_id": review.get("reviewId"),
            }
        )
    return pd.DataFrame(rows)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read and validate an uploaded review CSV."""
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"The file is {uploaded_file.size / (1024 * 1024):.1f} MiB. "
            f"The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB."
        )

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not parse the CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The CSV has no rows.")
    if "content" not in df.columns:
        raise ValueError(
            "The CSV needs a `content` column with the review text. "
            f"Columns found: {', '.join(map(str, df.columns))}."
        )
    if df["content"].isna().all():
        raise ValueError("The `content` column is empty.")

    return df


def batch_controls() -> tuple[int, int, int]:
    """The shared batch controls: number of batches and reviews per batch."""
    num_batches = st.number_input(
        "Number of batches",
        min_value=1,
        max_value=MAX_BATCHES_PER_RUN,
        value=1,
        step=1,
    )
    reviews_per_batch = st.slider(
        "Reviews per batch",
        min_value=MIN_REVIEWS_PER_BATCH,
        max_value=MAX_REVIEWS_PER_BATCH,
        value=100,
        step=100,
    )
    num_batches = int(num_batches)
    reviews_per_batch = int(reviews_per_batch)
    requested = num_batches * reviews_per_batch
    st.caption(
        f"{requested} reviews requested — {num_batches} job(s) of up to "
        f"{reviews_per_batch} reviews each."
    )
    return num_batches, reviews_per_batch, requested


def plan_batches(
    source: pd.DataFrame, num_batches: int, reviews_per_batch: int
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Select the requested prefix and split it into per-job chunks.

    Every chunk carries a stable ``source_row_index`` so results can be combined in
    the original order.
    """
    selected = source.head(num_batches * reviews_per_batch).copy()
    selected["source_row_index"] = range(len(selected))

    chunks = []
    for i in range(num_batches):
        start = i * reviews_per_batch
        chunk = selected.iloc[start : start + reviews_per_batch]
        if chunk.empty:
            break
        chunks.append(chunk)
    return selected, chunks


# =========================================================================
# Batch mode — review service client
# =========================================================================


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


def submit_batch_run(chunks: list[pd.DataFrame], run_id: str) -> list[dict]:
    """Submit one job per chunk. Returns the job records.

    A batch is only recorded once the service has returned a job id — Streamlit
    never reports a batch as accepted before that.
    """
    jobs = []
    for i, chunk in enumerate(chunks, start=1):
        payload = chunk.to_csv(index=False).encode("utf-8")
        response = httpx.post(
            f"{REVIEW_SERVICE_URL}/jobs",
            headers=_auth_headers(),
            files={"file": (f"{run_id}_batch_{i}.csv", payload, "text/csv")},
            data={"options": json.dumps({"split": True})},
            timeout=120.0,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Batch {i} was rejected by the service "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )
        body = response.json()
        job_id = body.get("job_id") or body.get("id")
        if not job_id:
            raise RuntimeError(f"The service did not return a job id: {body}")

        jobs.append(
            {
                "batch": i,
                "job_id": job_id,
                "rows": int(len(chunk)),
                "offset": int(chunk["source_row_index"].min()),
            }
        )
    return jobs


def fetch_job_status(job_id: str) -> dict:
    response = httpx.get(
        f"{REVIEW_SERVICE_URL}/jobs/{job_id}",
        headers=_auth_headers(),
        timeout=30.0,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Could not read the status of job {job_id} "
            f"(HTTP {response.status_code}): {response.text[:300]}"
        )
    return response.json()


def fetch_job_result(job_id: str) -> pd.DataFrame:
    """Fetch a completed job's result CSV through the presigned result URL."""
    response = httpx.get(
        f"{REVIEW_SERVICE_URL}/jobs/{job_id}/result",
        headers=_auth_headers(),
        timeout=60.0,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Could not fetch the result of job {job_id} "
            f"(HTTP {response.status_code}): {response.text[:300]}"
        )
    body = response.json()
    url = body.get("url") or body.get("result_url") or body.get("download_url")
    if not url:
        raise RuntimeError(f"The service did not return a result URL: {body}")

    csv_response = httpx.get(url, timeout=180.0)
    csv_response.raise_for_status()
    return pd.read_csv(io.BytesIO(csv_response.content))


RESULT_COLUMNS = [
    "source_row_index",
    "content",
    "sub_comment",
    "bert_topic",
    "bert_topic_conf",
    "bert_sentiment",
    "split_fallback",
]


def render_result_table(combined: pd.DataFrame) -> pd.DataFrame:
    """The on-screen result preview: the documented columns, capped at 1,000 rows."""
    columns = [c for c in RESULT_COLUMNS if c in combined.columns]
    preview = combined[columns].head(RESULT_PREVIEW_ROWS)
    if len(combined) > RESULT_PREVIEW_ROWS:
        st.caption(
            f"Showing the first {RESULT_PREVIEW_ROWS} of {len(combined)} rows — "
            "the download below has all of them."
        )
    return preview


def render_batch_results(run: dict) -> None:
    """Poll the run's jobs, show progress, and render the combined results."""
    jobs = run["jobs"]
    statuses = []
    failed = []

    for job in jobs:
        try:
            status = fetch_job_status(job["job_id"])
        except Exception as exc:
            status = {"status": "UNKNOWN", "error": str(exc)}
        job["status"] = status.get("status", "UNKNOWN")
        job["completed_reviews"] = status.get("completed_reviews", 0)
        job["total_reviews"] = status.get("total_reviews", job["rows"])
        statuses.append(
            {
                "batch": job["batch"],
                "job_id": job["job_id"],
                "status": job["status"],
                "completed": job["completed_reviews"],
                "total": job["total_reviews"],
            }
        )
        if job["status"] == "FAILED":
            failed.append(job)

    st.dataframe(
        pd.DataFrame(statuses),
        hide_index=True,
        alt="Progress for each submitted review batch",
    )

    total = sum(j["total_reviews"] for j in jobs) or 1
    done = sum(j["completed_reviews"] for j in jobs)
    st.progress(
        min(done / total, 1.0), text=f"Overall: {done}/{total} reviews processed"
    )

    if failed:
        st.error(
            "These batches failed: "
            + ", ".join(f"batch {j['batch']} ({j['job_id']})" for j in failed)
        )

    completed = [j for j in jobs if j["status"] == "COMPLETED"]
    if not completed:
        st.info("No batch has finished yet. Use **Refresh job status** to poll again.")
        return

    frames = []
    for job in completed:
        try:
            frames.append(fetch_job_result(job["job_id"]))
        except Exception as exc:
            st.error(f"Could not fetch the result of batch {job['batch']}: {exc}")

    if not frames:
        st.warning("No batch results could be downloaded.")
        return

    combined = pd.concat(frames, ignore_index=True)
    if "source_row_index" in combined.columns:
        combined = combined.sort_values("source_row_index", kind="stable").reset_index(
            drop=True
        )

    st.write(f"**{len(combined)} sub-comment rows** from {len(completed)} batch(es).")
    st.dataframe(
        render_result_table(combined),
        hide_index=True,
        alt="Combined BERT topic and sentiment results",
    )

    st.download_button(
        "Download the full annotated CSV",
        data=combined.to_csv(index=False).encode("utf-8"),
        file_name=f"reviewassist_{run['run_id']}.csv",
        mime="text/csv",
    )


def run_batch(source: pd.DataFrame, num_batches: int, reviews_per_batch: int) -> None:
    """The shared batch flow: preview, confirm, submit, poll, render."""
    available = len(source)
    selected, chunks = plan_batches(source, num_batches, reviews_per_batch)
    selected_count = len(selected)
    unprocessed = available - selected_count

    st.info(
        f"**{available}** rows available · **{selected_count}** selected · "
        f"**{len(chunks)}** job(s) to create · up to **{reviews_per_batch}** reviews "
        f"per job · about **{selected_count}** LLM split request(s)."
    )

    confirmed = True
    if unprocessed > 0:
        st.warning(
            f"Only the first {selected_count} of {available} rows will be processed — "
            f"{unprocessed} row(s) will be left unprocessed."
        )
        confirmed = st.checkbox(
            f"I understand that {unprocessed} row(s) will be left unprocessed"
        )

    if not confirmed:
        st.info("Tick the confirmation box to submit the selected rows.")
        return

    if st.button(f"Submit {len(chunks)} job(s)", type="primary"):
        run_id = uuid.uuid4().hex[:12]
        try:
            jobs = submit_batch_run(chunks, run_id)
        except Exception as exc:
            st.error(f"Could not submit the batch run: {exc}")
            return
        st.session_state["batch_runs"] = st.session_state.get("batch_runs", {})
        st.session_state["batch_runs"][run_id] = {"run_id": run_id, "jobs": jobs}
        st.success(f"Submitted {len(jobs)} job(s) under run {run_id}.")


def render_run_progress() -> None:
    """Show every submitted run and its progress; poll on demand."""
    runs = st.session_state.get("batch_runs", {})
    if not runs:
        return

    st.subheader("Batch runs")
    for run_id, run in runs.items():
        st.write(f"**Run {run_id}** — {len(run['jobs'])} job(s)")
        if st.button("Refresh job status", key=f"refresh_{run_id}"):
            render_batch_results(run)


# =========================================================================
# Page
# =========================================================================

st.set_page_config(page_title="ReviewAssist", layout="wide")
st.title("ReviewAssist: Google Play Store Review Analyzer")

st.sidebar.header("Input Options")
input_mode = st.sidebar.radio(
    "Choose Input Mode:",
    ["Manual Review Input", "Google Play Store App URL", "CSV Upload"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Model guide** — choose models in the main panel for manual reviews:
- 🤖 **Logistic Regression**:
  - Uses TF-IDF weighting
  - Considers feature importance
  - Better with imbalanced data

- 🧠 **Naive Bayes**:
  - Uses word counts
  - Assumes feature independence
  - Faster training time

- 🔑 **RAKE**:
  - Extracts keyword phrases
  - No training and no word2vec needed

- 🗣️ **BERT (multi-task)**:
  - Topic classification + sentiment regression
  - Labels each sub-comment separately
  - Runs locally in development or through the service when hosted
"""
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[AIPI540_nlp_project](https://github.com/ruhan-dave/AIPI540_nlp_project)"
)

if input_mode == "Manual Review Input":
    st.subheader("Manual review analysis")
    with st.container(border=True):
        st.markdown("### Choose models to run")
        st.caption(
            "All four are selected by default. Select one or more models to compare."
        )
        selected_models = st.pills(
            "Models to run",
            options=MANUAL_MODEL_OPTIONS,
            selection_mode="multi",
            default=MANUAL_MODEL_OPTIONS,
            key="manual_model_selection",
            help="Choose which model columns to show for this review.",
        )
    new_review = st.text_area(
        "Enter a review:", "This app is terrible and crashes constantly."
    )
    if new_review:
        render_review(new_review, selected_models or [])

elif input_mode == "Google Play Store App URL":
    st.subheader("Google Play Store batch run")
    st.info(
        "Batch jobs always use BERT Multi-Task. The model selector applies to manual reviews."
    )
    app_url = st.text_input("Enter the Google Play Store app URL:")
    num_batches, reviews_per_batch, requested_reviews = batch_controls()

    if app_url:
        app_id = extract_app_id(app_url)
        if not app_id:
            st.error(
                "Invalid Google Play Store URL — expected an `id=` parameter, "
                "e.g. https://play.google.com/store/apps/details?id=com.example.app"
            )
        else:
            st.write(
                f"Fetching up to {requested_reviews} reviews for app `{app_id}`..."
            )
            try:
                reviews_df = fetch_play_store_reviews(app_id, requested_reviews)
            except Exception as exc:
                st.error(f"Error fetching reviews: {exc}")
            else:
                if reviews_df.empty:
                    st.warning("The Play Store returned no reviews for that app.")
                else:
                    st.write(
                        f"Fetched {len(reviews_df)} review(s); "
                        f"{len(reviews_df)} are available for selection."
                    )
                    st.dataframe(
                        reviews_df.head(10),
                        hide_index=True,
                        alt="Preview of the newest Google Play reviews",
                    )
                    run_batch(reviews_df, num_batches, reviews_per_batch)

elif input_mode == "CSV Upload":
    st.subheader("CSV batch run")
    st.info(
        "Batch jobs always use BERT Multi-Task. The model selector applies to manual reviews."
    )
    st.markdown(
        "Upload a CSV of raw reviews. The **`content`** column is required; any other "
        "columns (for example `score` and `date`) are preserved in the result."
    )
    st.code(
        "content,score,date\n"
        '"The app is useful, but the subscription is too expensive.",2,2025-01-15',
        language="csv",
    )
    num_batches, reviews_per_batch, requested_reviews = batch_controls()

    uploaded = st.file_uploader("Upload a review CSV", type=["csv"], help="Max 50 MiB.")
    if uploaded is not None:
        try:
            source_df = read_uploaded_csv(uploaded)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.write(
                f"{len(source_df)} row(s) available, {requested_reviews} requested. "
                f"Up to {len(source_df)} can be processed."
            )
            st.dataframe(
                source_df.head(10),
                hide_index=True,
                alt="Preview of uploaded review rows",
            )
            run_batch(source_df, num_batches, reviews_per_batch)

render_run_progress()
