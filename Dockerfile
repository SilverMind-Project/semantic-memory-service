FROM python:3.14-slim

WORKDIR /app

# Layers are ordered so that everything above the source copy survives a code
# change. Previously the source was copied before both the dependency install
# and the 20MB tokenizer download, so editing one Python file reinstalled every
# dependency (including a git clone of triton-shared) and re-fetched the
# tokenizer over the network.

# 1. System packages. Changes only when this line does.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Embedding model tokenizer (~20MB, needed by triton-shared TextEmbedder).
#    Above the source copy so a code edit never re-downloads it, and so a
#    rebuild does not need to reach huggingface.co at all.
RUN mkdir -p /models/embeddinggemma-300m/1 && \
    python -c "from urllib.request import urlretrieve; urlretrieve('https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer.json', '/models/embeddinggemma-300m/1/tokenizer.json')"

# 3. Third-party dependencies, resolved from pyproject.toml alone.
#    hatchling needs the `app` package present to build a wheel, so a stub
#    stands in for it; the stub distribution is then uninstalled, leaving the
#    dependencies behind. This keeps the expensive install cached until
#    pyproject.toml itself changes.
COPY pyproject.toml README.md ./
RUN mkdir -p app && touch app/__init__.py \
    && pip install --no-cache-dir . \
    && pip uninstall -y semantic-memory-service \
    && rm -rf app

# 4. Application source. Only these layers rebuild on a code change.
#    The package is deliberately not pip-installed: app/db/migrate.py locates
#    alembic.ini relative to its own __file__ (three parents up), which resolves
#    to /app only when the code is imported from the source tree rather than
#    from site-packages.
COPY alembic.ini ./
COPY app ./app

EXPOSE 8400

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8400/health || exit 1

CMD ["python", "-m", "app.run"]
