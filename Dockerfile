# The ReviewAssist backend image: the FastAPI API and the SQS worker.
#
# One image, two commands:
#   python -m service.server
#   python -m service.worker
#
# The BERT checkpoint is baked in and read from local files only.  The image sets
# the HuggingFace offline flags and points HF_HOME at an empty directory, so a
# cold start can never reach the network for a model.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/tmp/hf_home \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1

WORKDIR /app

# Torch first and from the CPU-only index: the API and the worker both run on
# CPU, and the CUDA wheels are several gigabytes of dead weight.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.6.0"

COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

# The checkpoint, the splitter and the service.
#
# models/saved_bert_model/model.safetensors is gitignored but MUST ship in the
# image: .dockerignore deliberately does not exclude that directory.  A fresh
# clone has to supply the weights before this build will produce a usable image.
COPY models/saved_bert_model/ models/saved_bert_model/
COPY models/bert_inference.py models/__init__.py models/
COPY scripts/ scripts/
COPY service/ service/

# An empty HF_HOME, owned by the unprivileged user that runs the service.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /tmp/hf_home \
    && chown -R appuser:appuser /app /tmp/hf_home

USER appuser

EXPOSE 8000

CMD ["python", "-m", "service.server"]
