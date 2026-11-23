# ReviewAssist — AIPI540 A2

## What this is

**ReviewAssist** analyzes Google Play Store app reviews and splits each into sub-comments, each carrying a
**topic** (`price`, `quality`, `design`, `function`, `customer service`, `other`) and a **sentiment score** in
[-1, 1]. A single review often mixes sentiments — *"I like the app but the subscription costs too much"* — and
whole-sentence sentiment misses that nuance.

Duke AIPI540 A2. Python 3.12 · scikit-learn · PyTorch/transformers · OpenRouter (MiniMax M3.1 Flash) or
`google-genai` (Gemini) for annotation · Streamlit for the demo.

Two parallel tracks, one data contract: **`scripts/`** is the pipeline that runs the project; **`notebooks/`** is
where things get tried.

## Architecture map

```
AIPI540_nlp_project/
  scripts/                 # the pipeline — this is the code that runs
    llm_api.py             # the split prompt + the sub_comment/topic/sentiment contract; TOPIC_LIST (:38)
    llm_providers.py       # provider transport: OpenRouter (default) + Gemini
    preprocess_data.py     # raw CSV → sub_comments JSON per row; checkpointed, resumable, 20 threads
    check_label_quality.py # verifies sub_comments reconstruct the original review
    build_features.py      # clean text + derive features + sentiment labels
    model.py               # trains + pickles LR (tfidf) and NB (count); predict()
    eval_model.py          # multi-label topic P/R/F1 + sentiment MSE
    bert_inference.py      # annotates the corpus with the local BERT checkpoint + agreement report
    scraper.py             # pulls Play Store reviews for 40 apps
    make_dataset.py        # places the raw CSV in data/raw/
  notebooks/               # exploration — machine_learning, naive_approach, bert_modeling
  models/                  # trained artifacts (.pkl) + saved_bert_model/ + machine_learning.py (a Colab export)
    bert_inference.py      # Hub-free BERT loader + batched predictor + the 8-label contract
    scoring.py             # import-safe NB/LR/RAKE scoring helpers the app uses
  data/raw/                # source CSV — 16k reviews (tracked)
  data/processed/          # processed_data.csv → processed_reviews.csv (see the chain below)
  app.py                   # Streamlit UI: four columns — NB | LR | RAKE | BERT
  main.py                  # entry point; shells out to `streamlit run app.py`
  .claude/                 # AI layer: 37 skills, 6 agents, 8 hooks, 8 reference docs (untracked)
```

**The pipeline chain** — each stage's output is the next stage's input:

```
data/raw/sentiment-analysis-dataset-google-play-app-reviews.csv
  └─ preprocess_data.py ─→ data/processed/processed_data.csv      (+ sub_comments JSON per row)
       └─ build_features.py ─→ data/processed/processed_reviews.csv  (+ cleaned_content, sentiment, sub-comment counts)
            └─ model.py train_models() ─→ models/*.pkl
```

`processed_reviews.csv` is the file `model.py` wants — it carries the `content` + `sentiment` columns training
needs, which `processed_data.csv` does not.

## Ground rules (conventions)

- **The topic vocabulary lives in two places that must stay in sync** — `scripts/llm_api.py:38` (the label list
  inside the split prompt, as `TOPIC_LIST`) and `scripts/eval_model.py:5` (`TOPIC_VOCAB`, which builds the metric
  matrices). Change one, change the other. A label in one and not the other silently scores as a false negative.
  **The splitter uses 6 labels; the BERT head has 8.** The split prompt keeps the 6-label vocabulary so the
  sub-comment distribution stays matched to what BERT was trained on.
- **The BERT head's 8 labels are data-derived and order-sensitive** — `customer service, design, function, other,
  overall, price, quality, security`, i.e. `sorted(df['topic'].unique())` over the training corpus.
  `models/bert_inference.py:TOPIC_LABELS` is the single source of truth; `models/saved_bert_model/config.json`
  (`id2label`/`label2id`) and `head_config.json` (`topic_labels`) must agree with it. `overall` and `security` have
  2 examples each — their head rows are essentially untrained noise, but **do not drop them**: that would renumber
  every label after index 4 and silently corrupt the mapping.
- **BERT sentiment is a regression score in [-1, 1], not a 1–5 star score.** The head regresses the LLM's
  per-sub-comment `sentiment` float. Raw outputs overshoot the range on ~18% of the corpus, so they are clamped to
  [-1, 1] and the raw value is kept as `sentiment_raw`. Never render it as a percentage.
- **The BERT checkpoint loads from local files only.** `models/bert_inference.py` reads `config.json` directly into a
  `BertConfig`, loads the tokenizer with `local_files_only=True`, and loads `model.safetensors` with a **strict**
  `load_state_dict`. The weights are keyed for the `BertMultiTask` *wrapper* (199 `bert.*` tensors + two heads), so
  `AutoModel.from_pretrained(model_dir)` will not work and must not be "fixed" into the code.
- **Direction of travel: `notebooks/` → `scripts/`.** Notebooks are where ideas get tried; `scripts/` is where
  they live once they work. Neither the app nor the pipeline imports a notebook — a change that lives only in a
  notebook ships nothing.
- **Text cleaning has three copies that must stay identical** — `scripts/build_features.py:22`,
  `models/machine_learning.py:38`, and `notebooks/naive_approach.ipynb` (the third copy is
  `notebooks/machine_learning.ipynb`, which `models/machine_learning.py` was exported from). The contract:
  expand contractions → lowercase → strip non-alpha → `word_tokenize` → drop stopwords and tokens ≤ 2 chars →
  `WordNetLemmatizer`. Change one copy, change all three. **`models/scoring.py` imports the canonical one
  (`from scripts.build_features import clear_content`) — do not add a fourth copy.**
- **Model artifacts are named `<family>_model.pkl` + `<family>_vectorizer.pkl`.** `scripts/model.py:73-102`
  writes them; `app.py` loads them by those exact names. A new model family means new files there, a
  branch in `scripts/model.py:104` (`predict`), and a column in `app.py`.
- **The classical models train on `cleaned_content`, not `content`.** `scripts/build_features.py` keeps both, and the
  app scores cleaned tokens against the model's feature vocabulary — training on the raw text puts raw tokens in the
  vocabulary and every lookup in the app misses.
- **Model loading is cwd-independent in the new modules.** `models/scoring.py`, `models/bert_inference.py`,
  `scripts/llm_api.py` and `scripts/bert_inference.py` anchor on `ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent`.
  `scripts/model.py:83` still builds paths like `models/lr_model.pkl` against the cwd — run it from the repo root.
- **The sub-comment sentiment is a float in [-1, 1], not a label.** `SubComment.sentiment` is `float`
  (`scripts/llm_api.py:26`); "positive / negative / neutral" only exists at the *review* level, derived from
  `score` in `scripts/build_features.py:119`. Don't compare a sub-comment's sentiment to those strings.
- **Fail loud on missing config — for the active provider only.** `scripts/llm_api.py` raises `ValueError` at import
  when the key for the *active* provider (`LLM_PROVIDER`, default `openrouter`) is unset. The app imports it lazily
  and catches that to fall back to whole-review BERT; the pipeline has no other channel to report a missing key.
- **`scripts/` and `models/` are regular packages** (`__init__.py` present) so `import scripts` resolves to this repo
  and not the unrelated `scripts` package in site-packages. Run the scripts as `python -m scripts.<name>` from the repo
  root; `python scripts/<name>.py` puts `scripts/` on `sys.path` instead of the repo root.
- **Network retries belong to tenacity, not to try/except.** `scripts/preprocess_data.py:27-39`: 3 attempts,
  exponential backoff 2–10 s, `retry_if_exception_type((ConnectionError, TimeoutError))`, `reraise=True`.
- **The pipeline checkpoints; don't delete the checkpoint.** `preprocess_data.py:20` writes
  `data/processed/processing_temp.csv` every 5 records and resumes from it, removing it only on success
  (`:116-118`). That file is what makes a 20-thread LLM pass over 16k reviews survivable.
- **There is no test suite.** "Done" for a model change means re-running the training script and reading the
  printed `classification_report` (`scripts/model.py:58,67`) and metric dict (`scripts/eval_model.py`).

## Working principles (agent steering)

- **Plan before non-trivial changes.** The same logic exists in `scripts/` and `notebooks/`, so it is easy to
  edit the copy that is wired into nothing. Sketch the approach and confirm before writing.
- **When the repo disagrees with itself, ask — don't pick.** The processed-CSV paths still conflict (Gotcha 3).
  The topic-count conflict is *resolved*: the **splitter uses 6 labels** (`scripts/llm_api.py:TOPIC_LIST`, the
  corpus vocabulary) and the **BERT head uses 8** (`models/bert_inference.py:TOPIC_LABELS`, derived from the
  corpus). Those are two different contracts, not a disagreement.
- **Fail fast on bad input; no silent fallbacks.** `scripts/preprocess_data.py:47-49` currently swallows a
  per-record failure and returns `None`; prefer surfacing it. No bare `except:` — there are several in
  `models/machine_learning.py`.
- **Keep scope to the ask.** There are live breakages (Gotchas); flag them, don't fix them unasked.

## Commands

Run everything from the repo root.

| Command | What it does |
|---|---|
| `pip install -r requirements.txt` | Install dependencies |
| `python -m scripts.preprocess_data` | Raw CSV → `sub_comments` (LLM-annotated, resumable) |
| `python -m scripts.check_label_quality` | Verify sub_comments reconstruct the original review |
| `python scripts/build_features.py` | `processed_data.csv` → `processed_reviews.csv` (cleaned text + labels) |
| `python scripts/model.py` | Train + pickle the LR and NB models — **needs `build_features.py` to have run first** |
| `python scripts/eval_model.py` | Topic precision/recall/F1 + sentiment MSE |
| `python scripts/bert_inference.py --batch-size 64` | Annotate the corpus with the local BERT checkpoint + write the agreement report |
| `streamlit run app.py` | The demo UI (equivalent: `python main.py`) |

**Setup:** `OPENROUTER_API_KEY` (or `GEMINI_API_KEY`) in `.env` (gitignored) — required by `scripts/llm_api.py` for
the *active* provider (`LLM_PROVIDER`, default `openrouter`); `llm_api.py` raises at import without it, so
`preprocess_data.py` cannot even start. Separately, `models/trained_model/GoogleNews-vectors-negative300.bin.gz`
is gitignored and must be downloaded by hand (figshare link in `models/scoring.py`) — the app uses it for
word-level highlighting and degrades to review-level scores without it.

## Gotchas

Live, verified breakages. Full inventory with fixes: `.claude/references/known-issues.md`.

1. **`build_features.py`'s sub-comment counters are always zero.** It compares
   `comment.get('sentiment') == 'positive'` (`scripts/build_features.py:78-80`), but the LLM writes a float in
   [-1, 1] — 0 of 421 sampled sub-comments match. `positive_count` / `negative_count` / `neutral_count` need a
   threshold, not a string compare.
2. **The data was annotated with 8 topics; the code scores 6.** `data/processed/processed_data.csv` contains
   `security` and `overall` sub-comments, which `scripts/eval_model.py:5` excludes from the metric matrices
   entirely — those rows become invisible false negatives. (The BERT head *does* carry all 8.)
3. **The two "wrong path" references are `scripts/check_label_quality.py:9` and the README** — both point at
   `models/data/processed_data.csv`, which does not exist. The real file is
   `data/processed/processed_data.csv`. (`scripts/model.py:149` and `setup.py:45` are *correct* — they want
   `processed_reviews.csv`, the `build_features.py` output.)
4. **`scripts/model.py` exits 0 when `data/processed/processed_reviews.csv` is missing** — it prints "not found",
   trains nothing, and returns success. Check the file exists *and* the pickles' sizes before trusting it.
5. **Reviews whose text is only stopwords clean to an empty string**, which `to_csv` writes as an empty field and
   `read_csv` reads back as `NaN` — the vectorizers reject `NaN`. `scripts/model.py` fills and drops those rows.
6. **`gemini-2.0-flash` (the model the corpus was annotated with) has been retired by Google.** The Gemini provider
   now defaults to `gemini-2.5-flash`; override with `GEMINI_MODEL`.
7. **The word2vec binary is gitignored and absent**, so the app's NB and LR columns show review-level scores without
   word-level colouring until it is downloaded by hand.

## Where new code goes

- **New model family** → prototype in `notebooks/`, then `scripts/model.py` (`train_models` / `save_models` /
  `predict`) → `models/<family>_model.pkl` + `models/<family>_vectorizer.pkl` → a column in `app.py`.
- **New topic label** → `scripts/llm_api.py` prompt (`TOPIC_LIST`) **and** `scripts/eval_model.py:5`, then re-run
  `preprocess_data.py` (the existing rows carry the old labels). If the BERT head changes, update
  `models/bert_inference.py:TOPIC_LABELS` and `models/saved_bert_model/config.json` + `head_config.json` together.
- **New review source** → `scripts/scraper.py` (app list) → `scripts/make_dataset.py` → `data/raw/`.
- **New preprocessing step** → `scripts/build_features.py`, keeping `clear_content` in sync across all copies.
- **New UI panel** → the four-column layout in `app.py`; the scoring helpers live in `models/scoring.py`
  (not `models/machine_learning.py` — that module executes a training pipeline at import time).

## On-demand context

- `.claude/references/data-pipeline.md` — pipeline stages, the `sub_comments` contract, resuming, data flow.
- `.claude/references/models.md` — the model families, how to retrain and score each, and the BERT
  multi-task architecture + inference contract.
- `.claude/references/{conventions,engineering,direction}.md` — **stale course templates.** They describe a
  different project ("AI Tutor") and are not yet rewritten for this repo. Don't treat them as truth here.
