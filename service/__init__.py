"""ReviewAssist backend service.

One image, two entry points:

* ``python -m service.server`` — the FastAPI API: a synchronous ``/predict`` route
  and the asynchronous ``/jobs`` batch API the Streamlit frontend drives.
* ``python -m service.worker`` — the SQS worker that splits and labels batches.

The service never downloads a model. ``models/saved_bert_model/`` is baked into the
image and loaded from local files only, so a cold start is a file read and nothing
else.
"""

from __future__ import annotations

import pathlib
import sys

# The repo root has to be importable so `models` and `scripts` resolve to this repo
# rather than to any same-named package installed in site-packages.  `python -m
# service.server` already puts the working directory on sys.path, but the service
# must not depend on the caller's working directory.
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

__all__ = ["ROOT_DIR", "__version__"]

__version__ = "0.1.0"
