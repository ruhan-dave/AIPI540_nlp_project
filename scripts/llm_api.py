import json
import os
import pathlib
import sys
from typing import List

import pydantic
from dotenv import load_dotenv

# The repo root has to be importable for the `scripts.*` package imports below.
# `python scripts/llm_api.py` puts scripts/ on sys.path, not the repo root, and the
# repo root is what makes `scripts` resolve to *this* package rather than the
# unrelated `scripts` package installed in site-packages.
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.llm_providers import gemini_generate, openrouter_generate

# Load environment variables from .env file
load_dotenv()


class SubComment(pydantic.BaseModel):
    sub_comment: str
    topic: str
    sentiment: float


# The split vocabulary: 6 labels.
#
# This is NOT the BERT head's 8-label vocabulary (models/bert_inference.py:TOPIC_LABELS).
# The splitter stays on 6 labels because the corpus was annotated with them and BERT
# was trained on those spans; the BERT head additionally carries two near-empty
# classes ("overall", "security") that exist in the training data.
#
# scripts/eval_model.py:5 (TOPIC_VOCAB) holds the same 6 labels — keep the two in sync.
TOPIC_LIST = [
    "price",
    "quality",
    "design",
    "function",
    "customer service",
    "other",
]

# Which provider splits the reviews. "openrouter" (default) or "gemini".
# Gemini is retained so the existing Gemini-annotated corpus stays reproducible.
PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter").strip().lower()

# Will raise error if not found to avoid silent failures.
# Only the *active* provider's key is required — the previous unconditional
# GEMINI_API_KEY check took the whole app down when only OpenRouter was configured.
if PROVIDER == "gemini":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not found. Please add it to your .env file."
        )
elif PROVIDER == "openrouter":
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not found. Please add it to your .env file."
        )
else:
    raise ValueError(
        f"Unknown LLM_PROVIDER {PROVIDER!r}. Expected 'openrouter' or 'gemini'."
    )


def build_split_prompt(user_comment: str) -> str:
    """The split prompt. Unchanged from the original Gemini-only version.

    The 6-label vocabulary is interpolated from :data:`TOPIC_LIST` so there is a
    single source of truth for it.
    """
    return f"""
    You are a helpful assistant that can analyze user's comment and split it into a list of sub-comments (size from 1 to 10, 1 means that the sub-comment is the same as the user's comment).
    Each sub-comment should keep the exactly same text in the user's comment. You just hard split the comment into several sub-comments.
    You are not allowed to change the text in the user's comment.
    For each sub-comment, you should also provide a topic (must in {json.dumps(TOPIC_LIST)}) related to the sub-comment that the user is talking about, and a sentiment score (from -1 to 1, -1 means that the sub-comment is negative, 1 means that the sub-comment is positive, 0 means that the sub-comment is neutral) to describe the user's attitude towards the sub-comment with the topic.
    Here is an example for user's comment:
    "I love this product, but the price is too high."
    The example output should be (in JSON format):
    [
        {{
            "sub_comment": "I love this product",
            "topic": "quality",
            "sentiment": 1
        }},
        {{
            "sub_comment": ", but the price is too high.",
            "topic": "price",
            "sentiment": -1
        }}
    ]

    Here is other example for user's comment:
    "I can't afford this product, it's too expensive."
    The example output should be:
    [
        {{
            "sub_comment": "I can't afford this product, it's too expensive.",
            "topic": "price",
            "sentiment": -1
        }}
    ]

    NOTE: sub-comment split is hard split, it should can be back to the original comment if we concatenate all sub-comments with empty string (not space). So you need to make sure include the punctuation and space in the sub-comment. As general rule, it is better to let following sub-comment carry the punctuation and space that the previous sub-comment ends with.

    NOTE: If previous or following sub-comment has the same topic, you should merge them into one sub-comment. Because in general, we think those sub-comments are talking about the same thing.
    
    Now, please analyze the user's comment:
    "{user_comment}"
    Please output the result:
    """


def llm_generate(user_comment: str) -> List[SubComment]:
    """Split a review into sub-comments.

    The signature and return type are unchanged, so
    ``scripts/preprocess_data.py``'s ``[user.model_dump() for user in sub_comments]``
    keeps working.
    """
    prompt = build_split_prompt(user_comment)

    if PROVIDER == "gemini":
        return gemini_generate(prompt, api_key=api_key)
    return openrouter_generate(prompt, api_key=api_key)


if __name__ == "__main__":
    llm_generate("The user experience is not good, I can't use it.")
