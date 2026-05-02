FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ffmpeg is required by librosa/soundfile to decode .m4a / .mp3 / .flac.
# python:3.11-slim is Debian-based and ships almost nothing, so we install it
# explicitly. --no-install-recommends keeps the layer small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch wheels first (matches the repo-root Dockerfile pattern),
# then the rest of the ASR deps. Splitting the layer means torch is cached even
# if requirements-asr.txt changes later.
COPY requirements-asr.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-asr.txt

COPY src/ ./src/

ENTRYPOINT ["python", "-m", "src.inference"]
CMD ["--help"]
