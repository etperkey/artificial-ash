# Reproducible environment for the ASH abstract AI-text audit.
# Base image pinned by digest (python 3.12.10, the version the study ran on);
# every Python package pinned by version and hash in requirements.txt.
FROM python:3.12.10-slim@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 TZ=UTC MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /capsule
COPY requirements.txt .
RUN pip install --require-hashes --no-deps -r requirements.txt

COPY reproduce.py ./
COPY scripts/ scripts/
COPY manifests/ manifests/
COPY tests/ tests/

# Data, replay files and results are written under /capsule; mount folders there to keep them.
RUN useradd --create-home --uid 1000 capsule && chown -R capsule /capsule
USER capsule
ENTRYPOINT ["python", "-I", "reproduce.py"]
CMD ["selftest"]
