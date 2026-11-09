"""Import-safe scoring helpers for the Streamlit app.

``models/machine_learning.py`` is a Colab export that *executes the whole training
pipeline at import time*: it reads a CSV path that does not exist and loads a word2vec
binary that is not on disk.  Importing it from ``app.py`` therefore raised
``FileNotFoundError`` at startup.  It stays where it is as a legacy export; this
module supersedes it for app purposes.

Nothing here does file I/O, training, or network access at import time — the word2vec
binary is loaded lazily and its absence is reported, not raised at import.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

# The repo root has to be importable so `scripts` resolves to *this* repo's package
# rather than the unrelated `scripts` package installed in site-packages.
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# The canonical text cleaner. Imported, NOT copied — CLAUDE.md requires exactly three
# copies of clear_content (scripts/build_features.py, models/machine_learning.py,
# notebooks/naive_approach.ipynb) and a fourth copy would silently drift.
from scripts.build_features import clear_content  # noqa: E402,F401  (re-exported)

# The word2vec binary is gitignored and must be downloaded by hand.
WORD2VEC_RELATIVE_PATH = "models/trained_model/GoogleNews-vectors-negative300.bin.gz"
WORD2VEC_DOWNLOAD_URL = (
    "https://figshare.com/articles/dataset/GoogleNews-vectors-negative300/23601195?file=41403483"
)

# Lazily-loaded module-level singleton; None means "not available".
_WORD2VEC: Any = None
_WORD2VEC_LOADED = False


def load_word2vec(model_path: str | pathlib.Path | None = None):
    """Load the GoogleNews word2vec binary, or return ``None`` if it is not on disk.

    The file is ~700 MB, gitignored, and downloaded by hand from figshare.  Returning
    ``None`` lets the caller degrade loudly (one warning) instead of crashing at
    import; the word-level scorers raise a clear RuntimeError if they are called
    without it.
    """
    global _WORD2VEC, _WORD2VEC_LOADED

    if model_path is None and _WORD2VEC_LOADED:
        return _WORD2VEC

    path = (
        pathlib.Path(model_path)
        if model_path is not None
        else ROOT_DIR / "models" / "trained_model" / "GoogleNews-vectors-negative300.bin.gz"
    )

    if not path.exists():
        if model_path is None:
            _WORD2VEC_LOADED = True
            _WORD2VEC = None
        return None

    from gensim.models import KeyedVectors

    _WORD2VEC = KeyedVectors.load_word2vec_format(str(path), binary=True)
    if model_path is None:
        _WORD2VEC_LOADED = True
    return _WORD2VEC


def _require_word2vec():
    """Return the word2vec model or raise a message that says how to get it."""
    model = load_word2vec()
    if model is None:
        raise RuntimeError(
            f"word2vec model not found at {ROOT_DIR / WORD2VEC_RELATIVE_PATH}. "
            f"Download it from {WORD2VEC_DOWNLOAD_URL} "
            "(it is gitignored because it is ~700 MB)."
        )
    return model


def score_to_trans(x: float) -> float:
    """Map a value into [-1, 1] via 2*sigmoid(x) - 1.

    Copied from models/machine_learning.py:179-181.
    """
    return 2 * 1 / (1 + np.exp(-x)) - 1


def lr_keywords_from_model(
    lr_model, vectorizer, top_n: int = 30
) -> tuple[dict[str, float], dict[str, float]]:
    """Derive the positive/negative keyword dictionaries from a fitted LR model.

    Reproduces ``models/machine_learning.py:84-98`` from the pickled model:
    normalise the coefficients to z-scores and keep the top and bottom ``top_n``.

    Polarity: sklearn's binary ``LogisticRegression`` has a single coefficient row
    (``coef_.shape == (1, n_features)``) and its decision function points at
    ``classes_[1]``, which is ``'positive'`` because ``classes_`` is sorted.  A
    *higher* coefficient therefore means a word pushes the prediction towards
    *positive*.  (The "coef_[i] corresponds to classes_[i]" rule applies to the
    multinomial case, where ``coef_`` has one row per class — not to the binary case
    used here.)

    Returns ``(positive_words, negative_words)``, each a ``{feature: z_score}`` dict.
    """
    feature_names = vectorizer.get_feature_names_out()
    coef_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": lr_model.coef_[0]}
    )

    # Normalise the coefficient for display (% of the color)
    mean_coef = coef_df["coefficient"].mean()
    std_coef = coef_df["coefficient"].std()
    coef_df["normalized_coefficient"] = (coef_df["coefficient"] - mean_coef) / std_coef

    positive = coef_df.sort_values("coefficient", ascending=False).head(top_n)
    negative = coef_df.sort_values("coefficient", ascending=True).head(top_n)

    positive_dict = positive.set_index("feature")["normalized_coefficient"].to_dict()
    negative_dict = negative.set_index("feature")["normalized_coefficient"].to_dict()
    return positive_dict, negative_dict


def calculate_score_lr(
    new_input: str,
    positive_words: dict,
    negative_words: dict,
    similarity_threshold: float = 0.5,
) -> dict[str, float]:
    """Assign an adjusted weight to each token of ``new_input``.

    Port of ``models/machine_learning.py:183-206``: for every cleaned token, average
    ``similarity * keyword_weight`` over the keywords it matches (cosine similarity
    above ``similarity_threshold``), then map through ``score_to_trans`` into [-1, 1].

    Requires the word2vec binary.
    """
    model = _require_word2vec()

    result: dict[str, float] = {}
    new_review_cleared = clear_content(new_input)
    new_review_list = new_review_cleared.split(" ")
    agg_words = positive_words | negative_words

    for frag in new_review_list:
        acc_score = 0.0
        matches = 0
        for k, v in agg_words.items():
            try:
                similarity = model.similarity(frag, k)
            except KeyError:
                continue  # the fragment is not in the word2vec vocabulary
            if similarity >= similarity_threshold:
                acc_score += similarity * v  # generate new score for color
                matches += 1

        result[frag] = 0.0 if matches == 0 else acc_score / matches

    # Generate the transparency based on normal cdf
    for w in result:
        if result[w] != 0:  # faster processing
            result[w] = score_to_trans(result[w])
    return result


def word_to_center_similarity(word: str, word_list: list, model=None) -> float:
    """Cosine similarity of ``word`` to the centroid of ``word_list``.

    Port of ``models/machine_learning.py:265-274``.
    """
    model = model if model is not None else _require_word2vec()

    vectors = [model[w] for w in word_list]
    centroid_vector = np.mean(vectors, axis=0)
    word_vector = model[word]
    similarity = np.dot(centroid_vector, word_vector) / (
        np.linalg.norm(centroid_vector) * np.linalg.norm(word_vector)
    )
    return similarity


def calculate_score_rake(
    new_input: str,
    positive_words: list,
    negative_words: list,
    threshold: float = 0.1,
) -> dict[str, float]:
    """Score each token by its similarity to the positive/negative keyword centroids.

    Port of ``models/machine_learning.py:279-299``.  Requires the word2vec binary.
    """
    model = _require_word2vec()

    result: dict[str, float] = {}
    new_review_cleared = clear_content(new_input)
    new_review_list = new_review_cleared.split(" ")

    for frag in new_review_list:  # compare similarity of the word to center of the list
        if frag not in model:
            result[frag] = 0.0
        else:
            pos_similarity = word_to_center_similarity(frag, positive_words, model)
            neg_similarity = word_to_center_similarity(frag, negative_words, model)
            score = pos_similarity - neg_similarity
            if score > threshold:
                result[frag] = 1.0
            elif score < -threshold:
                result[frag] = -1.0
            else:
                result[frag] = 0.0

    return result


def calculate_score_nb(new_input: str, nb_model, count_vectorizer) -> dict[str, float]:
    """Per-token Naive Bayes sentiment scores in [-1, 1].

    New — the app has always wanted this and it did not exist anywhere.

    For each token of the *cleaned* review, look up its count-vectorizer feature
    index (tokens outside the trained vocabulary score 0.0) and turn the class
    log-posteriors into a probability::

        log_pos = class_log_prior_[positive] + feature_log_prob_[positive][idx]
        log_neg = class_log_prior_[negative] + feature_log_prob_[negative][idx]
        p_pos   = softmax([log_neg, log_pos])[positive]
        score   = 2 * p_pos - 1

    The ``2 * p - 1`` mapping matches the ``2*sigmoid(x) - 1`` convention used
    elsewhere in the repo, so the score drops straight into the highlight colour
    scale.  No word2vec required.
    """
    classes = list(nb_model.classes_)
    try:
        positive_idx = classes.index("positive")
        negative_idx = classes.index("negative")
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Naive Bayes model was trained on {classes}, expected 'negative' and 'positive'."
        ) from exc

    vocabulary = count_vectorizer.vocabulary_
    class_log_prior = nb_model.class_log_prior_
    feature_log_prob = nb_model.feature_log_prob_

    result: dict[str, float] = {}
    for token in clear_content(new_input).split(" "):
        idx = vocabulary.get(token)
        if idx is None:
            result[token] = 0.0  # not in the trained vocabulary
            continue

        log_pos = class_log_prior[positive_idx] + feature_log_prob[positive_idx][idx]
        log_neg = class_log_prior[negative_idx] + feature_log_prob[negative_idx][idx]

        # softmax([log_neg, log_pos])[positive]
        max_log = max(log_neg, log_pos)
        exp_neg = np.exp(log_neg - max_log)
        exp_pos = np.exp(log_pos - max_log)
        p_pos = exp_pos / (exp_neg + exp_pos)

        result[token] = 2 * p_pos - 1
    return result


def review_sentiment_probabilities(
    text: str, model, vectorizer
) -> tuple[float, float]:
    """Review-level ``(positive, negative)`` probabilities from a fitted classifier.

    The per-token scorers below drive the word highlighting; this gives the single
    review-level number the columns headline. It is the same shape
    ``scripts/model.py``'s ``predict()`` returns (``{'positive': ..., 'negative':
    ...}``), which is what the app already renders.

    The text is cleaned first, because that is the token space the models were
    trained in.
    """
    classes = list(model.classes_)
    if "positive" not in classes or "negative" not in classes:
        raise ValueError(
            f"Model was trained on {classes}, expected 'negative' and 'positive'."
        )

    features = vectorizer.transform([clear_content(text)])
    proba = model.predict_proba(features)[0]
    positive = float(proba[classes.index("positive")])
    negative = float(proba[classes.index("negative")])
    return positive, negative


def highlight_sentence_html(
    sentence: str, sentiment_dict: dict, similarity_threshold: float = 0.5
) -> str:
    """Colour each word of ``sentence`` by its sentiment score.

    Port of ``models/machine_learning.py:220-253``: negative words get a red wash,
    positive words a green one, and the intensity scales with the score.

    Each raw word is mapped onto its nearest scored token through word2vec, so this
    requires the word2vec binary.
    """
    model = _require_word2vec()

    words = sentence.split()
    highlighted_words = []  # for display

    for word in words:
        clean_word = word.lower().strip('.,!?;:()[]{}""\'')

        # assign score to the raw text
        value = 0.0
        for k in sentiment_dict:
            try:  # secondary mapping
                similarity = model.similarity(clean_word, k)
            except KeyError:
                continue  # no match
            if similarity >= similarity_threshold:
                value = similarity * sentiment_dict[k]
                break  # break when find the first match

        # formatted output
        if value < 0:
            # negative word -> red
            brightness = 255 - int((abs(value) * 255))
            color = f"rgb(255, {brightness}, {brightness})"
            highlighted_words.append(
                f'<span style="background-color: {color};">{word}</span>'
            )
        elif value > 0:
            # positive words -> green
            brightness = 255 - int((value * 255))
            color = f"rgb({brightness}, 255, {brightness})"
            highlighted_words.append(
                f'<span style="background-color: {color};">{word}</span>'
            )
        else:  # below treshold or no match, directly append
            highlighted_words.append(word)

    return " ".join(highlighted_words)


def highlight_sentence_nb_html(sentence: str, nb_model, count_vectorizer) -> str:
    """Colour each word of ``sentence`` by its Naive Bayes score.

    New — the app's Naive Bayes column needs the same renderer as the others.
    """
    return highlight_sentence_html(
        sentence, calculate_score_nb(sentence, nb_model, count_vectorizer)
    )


def extract_keywords_rake(text: str, top_n: int = 5) -> list[str]:
    """Extract the top ``top_n`` RAKE keyword phrases from a single string.

    New — ``models/machine_learning.py:111`` takes a *list* of texts and returns a
    ``value_counts`` Series, which is a different API from the one the app wants.

    RAKE needs no word2vec, so this works even without the missing binary.
    """
    from rake_nltk import Rake

    r = Rake()
    r.extract_keywords_from_text(text)
    # get_ranked_phrases_with_scores() returns (score, phrase) tuples.
    ranked = r.get_ranked_phrases_with_scores()
    return [phrase for _score, phrase in ranked[:top_n]]
