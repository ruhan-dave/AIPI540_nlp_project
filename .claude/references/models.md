# Models — the four families, where each lives, how to retrain and score it

Read this when you're changing model code, adding a family, or touching `app.py`.

## Where each family lives

| Family | Training code | Artifact | Where it's used |
|---|---|---|---|
| Logistic Regression | `scripts/model.py` + `notebooks/machine_learning.ipynb` | `models/lr_model.pkl`, `models/tfidf_vectorizer.pkl` | `app.py` col 2; `models/scoring.py:calculate_score_lr` |
| Naive Bayes | `scripts/model.py` + `notebooks/naive_approach.ipynb` | `models/nb_model.pkl`, `models/count_vectorizer.pkl` | `app.py` col 1; `models/scoring.py:calculate_score_nb` |
| RAKE | `models/scoring.py:extract_keywords_rake` | none (stateless) | `app.py` col 3 |
| BERT (multi-task) | `notebooks/bert_modeling.ipynb` | `models/saved_bert_model/` (config + tokenizer + `model.safetensors`) | `app.py` col 4; `models/bert_inference.py` |

`models/machine_learning.py` is a Colab export of `notebooks/machine_learning.ipynb` (its docstring says so) —
it is a script that also acts as a module, and importing it executes the whole pipeline at import time. It is
**legacy**: the app uses `models/scoring.py` instead, which is import-safe and holds the NB/LR/RAKE helpers.

## How the classical models score a review

`models/scoring.py` holds the helpers the app uses. Two layers:

- **Review level** — `review_sentiment_probabilities(text, model, vectorizer)` returns
  `{'positive': p, 'negative': 1-p}` from `predict_proba` on the *cleaned* text. This is the headline number
  each column shows. It matches the shape `scripts/model.py:predict()` returns.
- **Word level** — per-token scores drive the colouring: clean the text, then compare each token against a
  set of sentiment-bearing keywords and score by **word2vec cosine similarity**.

- **Keyword sources** — LR coefficients (top 30 each way, normalized to z-scores) and RAKE-extracted phrases.
- **Threshold 0.5** — a token counts as matching a keyword only above this similarity.
- **LR score** — `similarity × keyword_weight`, averaged over matches, then mapped through
  `2·sigmoid(x) − 1` to land in [-1, 1] (`models/scoring.py:score_to_trans`).
- **NB score** — per-token class log-posteriors → `p(positive)` → `2·p − 1`. No word2vec needed.
- **RAKE score** — cosine similarity to the *centroid* of the positive-keyword set minus the same for negative,
  thresholded at 0.1 → -1 / 0 / +1.
- **word2vec** — `models/trained_model/GoogleNews-vectors-negative300.bin.gz`, gitignored, downloaded by hand.
  Needed for word-level highlighting; without it the app falls back to review-level scores.

### LR keyword polarity

sklearn's binary `LogisticRegression` has a **single** coefficient row (`coef_.shape == (1, n_features)`)
and its decision function points at `classes_[1]` — which is `'positive'`, because `classes_` is sorted.
So a **higher** coefficient means a word pushes the prediction towards *positive*:

```python
positive = coef_df.sort_values("coefficient", ascending=False).head(top_n)  # most positive-class words
negative = coef_df.sort_values("coefficient", ascending=True).head(top_n)   # most negative-class words
```

The "one coefficient row per class" rule (`coef_[i]` ↔ `classes_[i]`) applies to the **multinomial** case and
does *not* apply here. Verified on the current artifact: the positive head contains `love, great, best,
excellent, perfect` and the negative head contains `useless, disappointed, waste, uninstall, worst`.

Both models train on `cleaned_content` (the lemmatised, stop-worded text), not the raw `content` — the app
scores cleaned tokens, so the vocabulary has to be the cleaned one.

## BERT (multi-task) — `notebooks/bert_modeling.ipynb` → `models/saved_bert_model/`

One `bert-base-uncased` encoder with two heads:

- **Topic** — multi-class classification → `topic_logits` → softmax. Metric: accuracy.
- **Sentiment** — regression. Metric: RMSE.
- **Loss** — `λ_topic · CE(topic) + λ_sentiment · MSE(sentiment)`, λ = 1.0 each.
- **Config** — max_length 64, batch 16, lr 1e-4, 8 epochs, `random_split` into train/val/test.
- **Preprocessing** — extract `sub_comment` from the `sub_comments` JSON, lowercase, strip punctuation, drop
  stopwords, stem; `BertTokenizerFast` with attention masks, truncation and padding.

The notebook currently reads a **stale hardcoded absolute path** (cell 1) that points outside the repo.

### Inference contract (`models/bert_inference.py`)

The checkpoint is **local-only**: `models/bert_inference.py` reads `config.json` directly into a `BertConfig`,
loads the tokenizer with `local_files_only=True`, and loads `model.safetensors` with a strict `load_state_dict`.
No Hub, no network, no cache dependency.

- **Weights are wrapper-keyed** — 199 `bert.*` tensors plus `topic_classifier` and `sentiment_regressor` (203 keys).
  `AutoModel.from_pretrained(model_dir)` will **not** load them; the load must go into `BertMultiTask` with
  `strict=True`.
- **8 topics**, data-derived order: `customer service, design, function, other, overall, price, quality, security`
  (`sorted(df['topic'].unique())` over the corpus). `TOPIC_LABELS` in `models/bert_inference.py` is the single
  source of truth; `config.json` (`id2label`/`label2id`) and `head_config.json` (`topic_labels`) mirror it.
  `overall` and `security` have 2 training examples each — their rows are near-untrained noise, but dropping them
  would renumber every label after index 4.
- **Sentiment is a regression score in [-1, 1]** (the LLM's per-sub-comment `sentiment` float) — *not* the 1–5
  review `score`. ~18% of raw outputs fall outside [-1, 1] and are clamped; the raw value is kept as
  `sentiment_raw`.

### Corpus agreement

`python scripts/bert_inference.py --batch-size 64` annotates all 34,559 sub-comments and writes
`data/processed/bert_sub_comments.csv` + `data/processed/bert_agreement_report.txt`. The report measures
**agreement with the annotator that produced the corpus labels**, on the corpus BERT was trained on — optimistic
by construction, and not accuracy against human ground truth.

## Retraining

```bash
python scripts/build_features.py     # → data/processed/processed_reviews.csv
python scripts/model.py              # → models/*.pkl, prints classification_report for LR and NB
python scripts/eval_model.py         # topic P/R/F1 + sentiment MSE (runs on its own sample data)
python scripts/bert_inference.py --batch-size 64   # → bert_sub_comments.csv + agreement report
```

`scripts/model.py` has **no test split check and no persisted metrics** — it prints a `classification_report`
per model and nothing more. `scripts/eval_model.py` computes the real multi-label metrics but is currently
wired to its own hardcoded sample, not to trained-model output.

## Conventions to preserve

- Artifacts are `<family>_model.pkl` + `<family>_vectorizer.pkl` in `models/`, written by
  `scripts/model.py` and loaded by `app.py` by exact name.
- `scripts/model.py:104` (`predict`) branches on `model_type` (`'lr'` / `'nb'`) and returns
  `(prediction, scores)`.
- The app compares all families side by side in **four** columns rather than picking one — a new family should
  get a fifth column, not replace an existing one.
- `models/scoring.py` **imports** `clear_content` from `scripts/build_features.py`; it must never hold a copy.
- The BERT checkpoint is the one **directory** artifact, not a pickle pair. Its metadata
  (`config.json` + `head_config.json`) is the label/sentiment contract and is tracked; the 438 MB
  `model.safetensors` is gitignored and supplied out of band.
