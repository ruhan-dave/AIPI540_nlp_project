# Implementation Report — integrate BERT into ReviewAssist

**Plan**: `.claude/plans/integrate-bert-model-into-app.md`
**Branch**: `feature/integrate-bert-model-into-app`
**Commit**: `1def4af` — *feat: wire the saved BERT checkpoint into the app as a fourth analysis column*
**Pull request**: https://github.com/ruhan-dave/AIPI540_nlp_project/pull/1 (base `main` ← head `feature/integrate-bert-model-into-app`)
**Status**: COMPLETE (one validation blocked by a credential — see *Issues encountered*)

## Summary

The saved multi-task BERT checkpoint is now wired into `app.py` as a fourth analysis column
(Naive Bayes | Logistic Regression | RAKE | BERT), and the app — which could not boot at all — now
does. `models/bert_inference.py` loads the 438 MB checkpoint from local files only (no HuggingFace
Hub, no network) with a strict state-dict load, and the checkpoint's label contract (8 topics,
sentiment in [-1, 1]) is resolved in `config.json` / `head_config.json`. Alongside that,
`models/scoring.py` replaces the non-importable `models/machine_learning.py` with an import-safe
scoring library, the split provider moved from the retired `gemini-2.0-flash` to
`minimax/minimax-m3.1-flash` via OpenRouter, and the Streamlit app grew a URL/CSV batch client
that submits one async job per batch to the service owned by `.claude/plans/aws-deployment.md`.

## Tasks completed

- **Task 1** — `scripts/__init__.py`, `models/__init__.py` (CREATE, empty). Both now resolve to this
  repo instead of the unrelated `scripts` package in site-packages. `contractions`, `rake-nltk`,
  `gensim`, `google-play-scraper` verified importable.
- **Task 2** — `models/bert_inference.py` (CREATE). `BertMultiTask` (class body copied verbatim from
  notebook cell 11; only `__init__` changed to take a `BertConfig`), `TOPIC_LABELS`, `build_config`,
  `load_bert`, `predict_sub_comments`, `annotate_dataframe`, `load_bert_cached`.
- **Task 3** — `models/saved_bert_model/config.json` + `head_config.json` (UPDATE). Real 8-label
  `id2label`/`label2id`; `topic_labels_resolved: true`; sentiment contract corrected to
  `target_range [-1.0, 1.0]` / `source_column "sub_comments[].sentiment"`; `source_checkpoint` added.
- **Task 4** — `scripts/llm_providers.py` (CREATE) + `scripts/llm_api.py` (UPDATE). OpenRouter
  (default) and Gemini transports, `LLM_PROVIDER` switch, `json_schema` → `json_object` fallback,
  `SubComment`/`llm_generate` signature unchanged.
- **Task 5** — `models/scoring.py` (CREATE). `clear_content` re-exported from the canonical copy,
  `load_word2vec`, `lr_keywords_from_model`, `calculate_score_lr`, `calculate_score_rake`,
  `calculate_score_nb`, `highlight_sentence_html`, `highlight_sentence_nb_html`,
  `extract_keywords_rake`, `review_sentiment_probabilities`.
- **Task 6** — `scripts/model.py` (UPDATE) + regenerated artifacts. Now trains on `cleaned_content`.
  All four pickles are non-empty and load; stale `models/nb_vectorizer.pkl` deleted.
- **Task 7** — `scripts/bert_inference.py` (CREATE). Corpus annotation CLI + agreement report.
- **Task 8** — `app.py` (UPDATE). Five boot breakages fixed, four columns, three input modes, batch
  controls, service client, progress, results table, CSV download.
- **Task 8b** — `README.md`, `CLAUDE.md`, `.claude/references/models.md` (UPDATE).

## Tests added

The repo has no test suite and the plan does not introduce pytest. Verification was done with
executable checks:

| Check | Result |
|---|---|
| `py_compile` over all 7 changed modules | pass |
| Offline BERT load — empty `HF_HOME` + `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` | pass, topics `function` / `design` / `customer service` — identical to the plan's expected values |
| Checkpoint metadata assertions (id2label / topic_labels / target_range) | pass |
| WordNet precondition | pass (after correcting the assertion — see *Deviations* 4) |
| `build_features.py` → `model.py` | pass; LR accuracy 0.87, NB 0.86 on the held-out split |
| Four pickles non-empty and loadable | pass |
| Corpus annotation CLI | 34,559 rows, non-null rate 1.0, topic agreement **0.8406**, clamped **0.1846** — both match the plan's expected figures |
| Agreement report carries the "agreement, not accuracy" caveat | pass |
| Provider logic vs. a mocked transport (7 cases: happy path, 400→`json_object` fallback, double-400 raises, schema drift rejected, error body surfaced, `LLM_FORCE_JSON_OBJECT` escape hatch, `llm_generate` dispatch) | pass |
| Gemini live round trip | pass — 2 sub-comments, 6-label topics, sentiments in [-1, 1] |
| OpenRouter live round trip | **blocked** — the `.env` key is rejected (see *Issues encountered*) |
| Streamlit app via `AppTest`: boots, 4 columns, BERT table columns/topics/confidences/sentiment bounds, word2vec warning visible, no `st.error` on boot, batch controls default 1 × 100 (range 1–10 / 100–1,000), CSV instructions + example shown | pass |
| App manual path with a working provider | pass — "I love this app, but the price is too high and customer service never replies." → 3 sub-comments labelled `other` / `price` / `customer service`, sentiment clamped (1.038 → 1.000) |
| Batch pipeline vs. a mocked service: CSV validation (valid / no `content` / empty / oversized), chunking + stable `source_row_index`, one `POST /jobs` per chunk with multipart CSV + `options`, service rejection surfaced, missing job id surfaced, status polling, presigned result fetch, preview capped at 1,000 rows | pass |

## Validation results

- **Level 1 (syntax/metadata)** — pass.
- **Level 2 (model and scoring)** — pass.
- **Level 3 (local data pipeline)** — pass.
- **Level 4 (Streamlit app)** — pass for everything that does not need the deployed service. The
  URL/CSV end-to-end gate is owned by `.claude/plans/aws-deployment.md` and was not run (no service
  is deployed).

## Deviations from the plan

1. **The plan's LR-polarity "fix" was itself inverted — not applied as written.** The plan states that
   sklearn's `coef_[0]` is the *negative* class coefficient and instructs swapping the two `head()`
   calls. That is wrong for sklearn's **binary** `LogisticRegression`, whose `coef_` has a single row
   (`shape == (1, n_features)`) whose decision function points at `classes_[1]` = `'positive'`
   (`classes_` is sorted). The `coef_[i] ↔ classes_[i]` rule applies to the *multinomial* case.
   Applying the plan's snippet would have put `useless / disappointed / waste` in the **positive**
   dictionary. Implemented per sklearn's actual semantics, which satisfies the plan's own validation
   criterion: the positive head contains `love, great, best, easy, amazing, helpful, excellent,
   perfect, thank, awesome` and the negative head contains `useless, complicated, deleted, uninstall,
   disappointed, waste, uninstalling, confusing, worst`.
2. **`gemini-2.0-flash` has been retired by Google** — the Gemini path was dead as written ("This
   model models/gemini-2.0-flash is no longer available"). The provider now defaults to
   `gemini-2.5-flash`, overridable with `GEMINI_MODEL`. The plan asked for the call to be moved
   verbatim; the model id had to change for the path to work at all.
3. **The WordNet precondition assertion in the plan is wrong.** `WordNetLemmatizer().lemmatize('crashing')`
   returns `'crashing'` regardless of whether WordNet is installed — the default POS is `'n'`. Used
   `lemmatize('mice') == 'mouse'` plus verb morphology instead; both pass, so the corpus is present
   and the lemmatizer is live.
4. **`scripts/model.py` also drops empty reviews.** 63 of 10,942 rows are nothing but stopwords and
   clean to `""`; `to_csv` writes that as an empty field and `read_csv` reads it back as `NaN`, which
   the vectorizers reject with `ValueError: np.nan is an invalid document`. The plan's edit did not
   account for this; `fillna('')` + dropping those rows was added.
5. **Added `models/scoring.py:review_sentiment_probabilities`.** The original app rendered
   `scores_lr['positive']` / `scores_nb['positive']`, but `calculate_score_lr` returns a *per-token*
   dict — that line was a `KeyError` waiting to happen (the app never got far enough to hit it). The
   new helper returns `{'positive': p, 'negative': 1-p}` from `predict_proba` on the cleaned text —
   the same shape as `scripts/model.py:predict()`, which the plan names as the app's intended contract.
6. **Added a `ROOT_DIR` `sys.path` bootstrap** to `scripts/llm_api.py`, `scripts/llm_providers.py`,
   `scripts/bert_inference.py` and `models/scoring.py`. `python scripts/x.py` puts `scripts/` on
   `sys.path`, not the repo root, so `import scripts` still resolves to site-packages — `__init__.py`
   alone does not fix the direct-script invocation. `python -m scripts.x` from the repo root does.
   CLAUDE.md now documents `-m`.
7. **The 438 MB `model.safetensors` is gitignored** (GitHub's limit is 100 MB per file). The rest of
   `models/saved_bert_model/` — `config.json`, `head_config.json`, `tokenizer.json`,
   `tokenizer_config.json` — is tracked, because that metadata *is* the label contract. This answers
   the plan's open question Q5; a fresh clone needs the weights supplied out of band.
8. **`head_config.json` had no `source_checkpoint` key** to "correct" — it was added.
9. **`notebooks/bert_modeling.ipynb` is now tracked** (it was untracked). `models/bert_inference.py`
   mirrors its `BertMultiTask` class, so the source of truth belongs in the repo.
10. **`TOPIC_LIST` was extracted as a named constant** in `scripts/llm_api.py` (the 6-label list was
    inline in the prompt string). The rendered prompt is **byte-identical** to the previous version
    (verified by diffing the rendered strings against `git show HEAD:scripts/llm_api.py`), and the
    vocabulary now has one source of truth shared with the OpenRouter JSON schema.
11. **`app.py`'s artifact paths are `ROOT_DIR`-anchored**, not cwd-relative — the app now works from
    any directory, matching the new modules' pattern.
12. **Manual Review Input is the first radio option** (so it is the default) rather than the URL mode.
13. **`pip install -r requirements.txt` was not run** (the plan explicitly forbids it: the pins
    differ from the working environment). `contractions`, `rake-nltk`, `gensim` and
    `google-play-scraper` were verified importable.
14. **Deleted the stale `models/nb_vectorizer.pkl`** once `count_vectorizer.pkl` was regenerated, as
    the plan instructed.

## Issues encountered

- **The `.env` `OPENROUTER_API_KEY` is invalid.** OpenRouter answers `401 {"error":{"message":"User
  not found."}}` for it. The MiniMax model id itself is correct — `minimax/minimax-m3.1-flash` is in
  the OpenRouter catalogue. The provider's transport, fallback and validation were therefore verified
  against a mocked transport (7 cases, all passing) rather than a live call. **Action for the owner:**
  replace the key in `.env`; no code change is needed.
- **The local NLTK corpora (`punkt`, `punkt_tab`, `wordnet`, `omw-1.4`) were missing** and were
  downloaded. `wordnet` is present only as a zip under `~/nltk_data/corpora`, which NLTK reads fine.
- **Pre-existing working-tree deletions** (`all_rag_techniques/reliable_rag.ipynb` and five
  non-English `models/tokenizers/punkt/*.pickle` files) were present before this work started. They
  were **restored** from HEAD so this PR's diff carries only feature work. They are still deleted in
  the working tree if that was deliberate — re-delete them in a separate commit.
- **The batch end-to-end path could not be exercised** against a real service: the companion
  deployment plan has not been run. The client's request/response handling was verified against a
  mocked service instead, and the deployed-service gate remains with `.claude/plans/aws-deployment.md`.
