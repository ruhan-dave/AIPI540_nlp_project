"""Provider layer for the review-splitting LLM.

The split prompt and the ``SubComment`` schema live in ``scripts/llm_api.py`` — that
is the single source of truth for both.  This module supplies only the **transport**
and the **response parsing** for each provider:

* **OpenRouter** (default) — ``minimax/minimax-m3.1-flash`` over the OpenAI-compatible
  ``/chat/completions`` endpoint.
* **Gemini** — the original ``google-genai`` call, kept so the existing
  Gemini-annotated corpus stays reproducible.

Both providers return ``List[SubComment]``; ``llm_generate`` never has to care which
one ran.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

import httpx
from dotenv import load_dotenv

# The repo root has to be importable for the late `scripts.llm_api` import below.
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "minimax/minimax-m3.1-flash"

# The corpus was annotated with "gemini-2.0-flash", which Google has since retired
# ("This model models/gemini-2.0-flash is no longer available"). gemini-2.5-flash is
# the closest still-served model in the same line. Set GEMINI_MODEL to pin a
# different one.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# OpenRouter asks for these so the provider can attribute usage to this app.
HTTP_REFERER = "https://github.com/ruhan-dave/AIPI540_nlp_project"
APP_TITLE = "ReviewAssist"

# Appended when we have to fall back to plain JSON mode: OpenRouter's json_object
# mode does not constrain the shape, so we name it in the prompt instead.
JSON_OBJECT_INSTRUCTION = """

Respond with a JSON array only — no prose, no markdown fence. Each element must have
exactly this shape:
[{"sub_comment": "<verbatim text>", "topic": "<one of: price, quality, design, function, customer service, other>", "sentiment": <number between -1 and 1>}]
"""


def _schema_types():
    """Late import of the schema definitions.

    ``scripts/llm_api.py`` imports this module at module level; importing it back at
    module level here would be circular.  The import happens at call time instead, by
    which point ``scripts.llm_api`` is fully initialised.
    """
    from scripts.llm_api import TOPIC_LIST, SubComment

    return SubComment, TOPIC_LIST


def _sub_comment_schema(topic_list: list[str]) -> dict[str, Any]:
    """JSON schema for the sub-comment array, used as OpenRouter's ``response_format``."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "sub_comment": {"type": "string"},
                "topic": {"type": "string", "enum": topic_list},
                "sentiment": {"type": "number", "minimum": -1, "maximum": 1},
            },
            "required": ["sub_comment", "topic", "sentiment"],
            "additionalProperties": False,
        },
    }


def _post_openrouter(body: dict[str, Any], api_key: str, timeout: float) -> httpx.Response:
    return httpx.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": APP_TITLE,
        },
        json=body,
        timeout=timeout,
    )


def _raise_for_status(response: httpx.Response) -> None:
    """Raise with the response body attached.

    httpx's own error message omits the body, and OpenRouter puts the actionable
    part there ("User not found.", "This model is not available", ...).
    """
    if response.is_error:
        raise httpx.HTTPStatusError(
            f"{response.status_code} from OpenRouter: {response.text[:500]}",
            request=response.request,
            response=response,
        )


def openrouter_generate(
    prompt: str,
    api_key: str,
    model: str | None = None,
    timeout: float = 60.0,
) -> list[Any]:
    """Split a review into sub-comments through OpenRouter.

    Tries strict ``json_schema`` structured output first.  Support for
    ``response_format`` is provider-dependent and the MiniMax upstream may not honour
    strict schema enforcement, so a **400 mentioning ``response_format``** falls back
    **once** to ``{"type": "json_object"}`` with the shape named in the prompt.  If
    that also fails the error propagates — an unvalidated payload is never returned.
    """
    sub_comment_cls, topic_list = _schema_types()

    body: dict[str, Any] = {
        "model": model or DEFAULT_OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": 4096,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "sub_comments",
                "strict": True,
                "schema": _sub_comment_schema(topic_list),
            },
        },
    }

    # Escape hatch: force the plain JSON mode from the start. Useful for comparing the
    # two modes on a sample.
    force_json_object = os.environ.get("LLM_FORCE_JSON_OBJECT", "").strip() == "1"
    if force_json_object:
        body["response_format"] = {"type": "json_object"}
        body["messages"][0]["content"] = prompt + JSON_OBJECT_INSTRUCTION

    response = _post_openrouter(body, api_key, timeout)

    if response.status_code == 400 and "response_format" in response.text:
        # The provider rejected the structured-output request — retry once in plain
        # JSON mode, naming the shape in the prompt.
        body["response_format"] = {"type": "json_object"}
        body["messages"][0]["content"] = prompt + JSON_OBJECT_INSTRUCTION
        response = _post_openrouter(body, api_key, timeout)

    _raise_for_status(response)

    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    items = json.loads(content)
    if not isinstance(items, list):
        raise ValueError(
            f"Expected a JSON array of sub-comments, got {type(items).__name__}"
        )

    # Validate at the boundary so a schema drift fails loudly here rather than
    # corrupting the corpus further downstream.
    sub_comments = [sub_comment_cls(**item) for item in items]
    print(f"[openrouter] split into {len(sub_comments)} sub-comment(s)")
    return sub_comments


def gemini_generate(prompt: str, api_key: str | None = None) -> list[Any]:
    """Split a review into sub-comments through Gemini.

    The original ``google-genai`` call from ``scripts/llm_api.py``, moved verbatim.
    The SDK is imported lazily so the OpenRouter path does not depend on it.
    """
    from google import genai
    from google.genai import types

    sub_comment_cls, _ = _schema_types()

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not found. Please add it to your .env file."
        )

    client = genai.Client(api_key=api_key)
    model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=list[sub_comment_cls],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    print(response.text)

    return response.parsed
