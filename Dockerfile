FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1

COPY src/ ./src/

ENTRYPOINT ["python", "-m", "src.inference"]
CMD ["--help"]
